"""MetaClaw-Bench rollout driver for OpenClaw-RL Hybrid RL training (migration).

Day-sequential (day01 -> day30, concurrency=1), reusing MetaClaw-official's
own round-loop machinery (workspace isolation, gateway, real `openclaw
agent` CLI, inline scoring/feedback) via direct import instead of
reimplementing it -- see docs/metaclaw_migration_plan.md ("已查证的官方机制"
and "查证记录（三）") for the full design rationale and the two server-side
patches this depends on.

Architecture summary (see the migration doc for the "why"):
  - Real `openclaw agent` CLI subprocess drives each round (same as
    MetaClaw's own benchmark harness), NOT a self-controlled generation
    loop -- preserves the real OpenClaw "coding" tool profile MetaClaw-Bench
    tasks are authored against.
  - Each ROUND is its own proxy session (2026-08-19c; was one session per
    day before this), session_id = f"metaclaw-{test_id}-{group_id}-{round_id}"
    (the "metaclaw-" prefix is load-bearing: prepare_patched_openclaw_opd.sh
    pattern-matches it via _METACLAW_SESSION_RE to flag every turn in the
    session as MetaClaw round mode -- intermediate tool-call turns cannot
    carry a custom body field of their own, since OpenClaw's internal HTTP
    client constructs those requests, not this driver). Deliberately
    diverges from MetaClaw-official's own eval harness (_run_group shares
    one session across a whole day) -- switched to per-round sessions
    because sharing one transcript let an early round's overlong response
    balloon the context for every later round that day, dragging down
    otherwise-fine rounds (including multi_choice) via context overflow.
    The day's WORKSPACE (files an agent actually writes) is untouched by
    this -- only the raw chat transcript is no longer shared across rounds;
    cross-round continuity is still carried explicitly via the
    [Previous Feedback] text. See docs/metaclaw_migration_plan.md for the
    full writeup.
  - After each round finishes, the round's deterministic checker/multi-choice
    result (via the official _compute_inline_score/_build_feedback_text) is
    submitted to the proxy as a synthetic "next turn" message containing a
    {"metaclaw_verdict": true, "eval_score": ..., "hint": ...} JSON payload --
    prepare_patched_openclaw_combine_select.sh's patched _opd_evaluate()
    recognizes this and skips the LLM judge entirely for the round's final
    turn. Intermediate tool-call turns within the round are judged
    independently by a new task-agnostic step judge (also added by that
    patch), not by this driver.
  - Acc./Compl. (paper Table 1's metrics) are computed LIVE as this run
    progresses, matching MetaClaw-official's own rl_run.py methodology
    (verified via direct read: it runs `metaclaw-bench run --scene-per-train
    N` as a single pass, training and scoring the same data together, no
    held-out set) -- NOT via a separate before/after clean-checkpoint eval.
    See docs/metaclaw_migration_plan.md "训练/评测数据重叠" for the full
    reasoning. This is why day-level resume (METACLAW_PROGRESS_DIR) exists:
    under this design a crash-restart-from-day01 would corrupt the
    aggregate, not just waste compute.

Requires:
  - METACLAW_ROOT: path to the MetaClaw-official checkout.
  - METACLAW_ALL_TESTS_JSON: path to all_tests.json (relative to
    METACLAW_ROOT, or absolute).
  - METACLAW_COMBINE_PROXY_URL: our patched combine_select proxy's
    /v1/chat/completions URL. Placeholder default only -- the real port is a
    modelfactory-side deployment detail, not assumed here (see
    docs/metaclaw_migration_plan.md "不在本轮做的").
  - openclaw_cfg/openclaw.json's BENCHMARK_BASE_URL must ALSO point at that
    same proxy (separate env var openclaw itself reads -- see that file),
    so the real `openclaw agent` subprocess's own model calls land on the
    proxy too, not on MetaClaw's own api_server.py.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Wire up MetaClaw-official's own round-loop machinery (import, don't
# reimplement -- see docs/metaclaw_migration_plan.md "已查证的官方机制").
# ---------------------------------------------------------------------------

_METACLAW_ROOT_RAW = os.environ.get("METACLAW_ROOT", "")
if not _METACLAW_ROOT_RAW:
    raise RuntimeError(
        "METACLAW_ROOT env var must point at the MetaClaw-official checkout "
        "(e.g. D:/MAO/Claude/MetaClaw-official on this project's dev machine, "
        "wherever it is cloned on modelfactory for the real run)."
    )
METACLAW_ROOT = Path(_METACLAW_ROOT_RAW)
if not METACLAW_ROOT.exists():
    raise RuntimeError(f"METACLAW_ROOT does not exist: {METACLAW_ROOT}")

_BENCHMARK_DIR = METACLAW_ROOT / "benchmark"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from src.infer.infer_cmd import (  # noqa: E402
    _build_feedback_text,
    _compute_inline_score,
    _copy_eval_scripts,
    _copy_workspace_for_test,
    _find_free_port,
    _patch_agent_workspace,
    _prepare_session,
    _prepare_work_copy,
    _start_work_gateway,
    _wait_for_gateway,
)
from src.infer.prompts import FORMAT_ERROR, with_feedback  # noqa: E402
from src.infer.query_reader import get_default_query_reader  # noqa: E402
from src.scoring.scoring_cmd import _score_file_check, _score_multi_choice  # noqa: E402
from src.utils import get_project_root, resolve_path  # noqa: E402


async def _run_openclaw_agent(
    session_id: str,
    message: str,
    openclaw_config_path: Path,
    openclaw_state_dir: Path,
    project_root: Path,
    agent_id: str,
    gateway_port: int | None = None,
    timeout: float | None = None,
) -> tuple[int, str, str]:
    """Local copy of MetaClaw-official's own infer_cmd.py::_run_openclaw_agent
    (2026-08-19d) -- NOT imported, because the official function silently
    drops its own `agent_id` parameter: it never appends `--agent <id>` to
    the `openclaw agent` subprocess argv, and its only caller in official
    code (`_run_question`) never passes `agent_id` either. Confirmed (CLI,
    cross-referenced against this machine's actual installed OpenClaw
    build, `session-key-*.js`/`session-*.js` in node_modules) this is a
    real gap in OpenClaw's own session-key resolution, not something
    MetaClaw's harness compensates for elsewhere: without --agent,
    `resolveSessionKeyForRequest` computes the CORRECT default agent id
    internally (works when only one agent is configured, ours -- `agents.
    list` has exactly one entry, `metaclaw_agent`) but then, when no
    existing session-store entry matches the (new) --session-id, its
    fallback path builds the session key from the RAW --agent value
    instead of that already-correct default -- and an absent --agent
    normalizes to the literal string "main", not our configured agent.
    Every round's files were actually being written successfully, just
    into `{state_dir}/workspace-main/` (OpenClaw's built-in default-agent
    workspace) instead of the per-day workspace_copy the checker reads
    from (`_patch_agent_workspace` only patches the `metaclaw_agent` entry
    in openclaw.json) -- explaining why Compl. has been ~0.0% across every
    MetaClaw migration run to date: the checker was never wrong about the
    files not being where it looked, the files just were never being
    written there in the first place. See docs/metaclaw_migration_plan.md
    for the full trace.

    Fix: pass --agent explicitly (agent_id has no default here, unlike the
    official signature's `agent_id: str | None = None` -- a training run
    silently falling back to no --agent, and therefore back to this exact
    bug, is worse than a hard failure). `run_day` already threads
    `test["agent"]` (== "metaclaw_agent" for every real day) through
    `_patch_agent_workspace`/`_prepare_session`; this is the one remaining
    call site that wasn't also given it.

    Deliberately NOT a prepare_patched_*.sh-style patched copy of the whole
    ~1400-line infer_cmd.py (this project's usual pattern for OpenClaw-RL-
    official files that get imported by the slime/proxy process) -- this
    driver only ever calls this one function from infer_cmd.py's training
    path (_run_group/_run_question, the official eval-only orchestration,
    are not used here at all), so a full patched-copy-plus-sys.path-
    rewrite would be reproducing ~1400 lines of unrelated code to change
    two argv entries. If MetaClaw-Bench's own OFFLINE eval path
    (`infer_cmd.py::_run_question`/`_run_group`, used for e.g. before/after
    Compl. comparisons outside this driver) is ever needed with a correct
    --agent too, that is a separate, still-unaddressed gap -- fix it there
    when/if that path is actually used, not preemptively here.
    """
    env = {
        **os.environ,
        "METACLAW_ROOT": str(project_root),
        "OPENCLAW_CONFIG_PATH": str(openclaw_config_path),
        "OPENCLAW_STATE_DIR": str(openclaw_state_dir),
    }
    if gateway_port is not None:
        env["OPENCLAW_GATEWAY_PORT"] = str(gateway_port)
    proc = await asyncio.create_subprocess_exec(
        "openclaw", "agent",
        "--session-id", session_id,
        "--agent", agent_id,
        "--message", message,
        cwd=str(project_root),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return -1, "", f"Timeout after {timeout}s"
    return proc.returncode, stdout_bytes.decode(), stderr_bytes.decode()


# ---------------------------------------------------------------------------
# Proxy wiring
# ---------------------------------------------------------------------------

# TODO(modelfactory): placeholder only -- fill in the real proxy port once
# assigned. See docs/metaclaw_migration_plan.md "不在本轮做的".
PROXY_URL = os.environ.get(
    "METACLAW_COMBINE_PROXY_URL", "http://127.0.0.1:30000/v1/chat/completions"
)
_MODEL_ID = os.environ.get("METACLAW_MODEL_ID", "qwen3-4b")

# Authorization header for our own direct httpx POSTs to the proxy
# (_send_verdict_turn/_send_session_close_only) -- the patched proxy's
# _check_auth (openclaw_opd_api_server.py) rejects any request without
# "Authorization: Bearer <SGLANG_API_KEY>" with 401 whenever SGLANG_API_KEY
# is set server-side (verified via direct source read, 2026-08-18 -- real
# training run metaclaw_migration_20260817_181404 hit this on every
# close/verdict submission). `openclaw agent`'s own real-turn requests never
# hit this because BENCHMARK_API_KEY=${SGLANG_API_KEY} is wired into that
# subprocess's openclaw.json provider config by the launch script; our
# synthetic verdict/close POSTs bypass that config entirely and need the
# same value passed in explicitly.
_API_KEY = os.environ.get("SGLANG_API_KEY", "")

# Load-bearing prefix -- see module docstring and
# prepare_patched_openclaw_opd.sh's _METACLAW_SESSION_RE.
_SESSION_ID_PREFIX = "metaclaw-"

# OpenClaw's own embedded-agent-runner fallback text (src/agents/embedded-
# agent-runner/run.ts and run/incomplete-turn.ts in the may_2026_5_11
# snapshot) for "the CLI process technically exited rc=0 but the agent
# itself never produced a usable turn". Detected here for TRANSCRIPT
# VISIBILITY ONLY (2026-08-19) -- deliberately does NOT change
# agent_succeeded, official_score, or verdict submission. An earlier
# version of this fix routed detected turns through the agent_succeeded=False
# infra-failure path, which was wrong (confirmed in review): that path
# excludes the round from the Acc. denominator and skips verdict submission
# entirely, letting the next round's step-judge score the leftover pending
# turn instead of the checker -- semantically wrong for a genuine (if
# unusable) task attempt. Once the verdict-signal-skip and verdict-early-
# return fixes land, rc==0 + this fallback text flows through the ALREADY-
# correct normal path on its own: _compute_inline_score naturally scores
# empty/fallback content as failed, eval_score=-1.0 gets submitted as a
# real verdict. This constant exists purely so a human scanning the
# transcript can immediately tell "the agent gave up" apart from "the agent
# tried and got the task wrong".
_GENERATE_FAIL_MARKERS = ("Agent couldn't generate a response",)

# Shared gateway auth token for MetaClaw's own _start_work_gateway /
# _run_openclaw_agent (imported verbatim above, not our code). Verified
# root cause of metaclaw_migration_20260817_181404's 100%-failure run
# (2026-08-18): neither function ever sets OPENCLAW_GATEWAY_TOKEN, so on
# an OpenClaw build that enforces gateway auth (GatewayCredentialsRequiredError
# exists in the may_2026_5_11 CLI snapshot, confirmed ABSENT in
# march_2026_3_8 via `git grep` on both tags -- a real May-only addition,
# not present when MetaClaw's own harness code was written), each day's
# gateway subprocess auto-generates a random runtime-only token that the
# separate `openclaw agent` client subprocess never learns, so every
# websocket connection is rejected before any round can run. Both
# functions build their subprocess env as `{**os.environ, ...}`, so setting
# this once here (before any day runs) is enough to share one token between
# every day's gateway and its own agent calls, with zero changes to
# MetaClaw-official's vendored code. Harmless to reuse across all 30 days --
# each day's gateway is a short-lived localhost-only process, not exposed
# externally.
if not os.environ.get("OPENCLAW_GATEWAY_TOKEN"):
    os.environ["OPENCLAW_GATEWAY_TOKEN"] = secrets.token_hex(16)

# Optional resilience knobs (2026-08-17) -- default to 0 (off), matching
# MetaClaw-official's own defaults exactly (infer_cmd.py's `retry: int = 0`;
# its HTTP-based _query_teacher_logprobs has no retry at all either -- see
# docs/metaclaw_migration_plan.md 查证记录四 for the verification). Kept as
# explicit opt-in env vars rather than turned on by default -- whether they
# help is a real-training-data question, not something to decide from code
# reading alone.
AGENT_RETRY = int(os.environ.get("METACLAW_AGENT_RETRY", "0"))
VERDICT_RETRY = int(os.environ.get("METACLAW_VERDICT_RETRY", "0"))

# Adjustable training window (2026-08-20) -- see docs/metaclaw_migration_plan.md
# "方案：可调 K 天训练窗口 + 冻结评测剩余天数". Unset (default, empty string) means
# disabled: train every day, exactly matching prior behavior -- this is a
# genuine absence of the feature (day_index > TRAIN_UNTIL_DAY can never be
# true when TRAIN_UNTIL_DAY is None), not "set K to a large number". Set to
# an integer K (1-based index into test_list; K=0 is valid -- freezes before
# day 1, i.e. a pure base-model pass through this same 30-day harness) to
# train through day K and freeze from day K+1 onward: no further samples
# reach the training queue (see _send_freeze_signal below and the
# openclaw-rl-metaclaw-train-until-day patches in prepare_patched_openclaw_
# opd.sh/prepare_patched_openclaw_combine.sh), but the remaining days still
# run for real (agent + checker), producing a complete Acc./Compl. report --
# just against a fixed checkpoint instead of a rolling one. A day already
# completed by a prior crashed run (METACLAW_RESUME) is bucketed by this
# same day_index regardless of whether IT was trained originally --
# resume/freeze are orthogonal.
_TRAIN_UNTIL_DAY_RAW = os.environ.get("METACLAW_TRAIN_UNTIL_DAY", "")
TRAIN_UNTIL_DAY: int | None = int(_TRAIN_UNTIL_DAY_RAW) if _TRAIN_UNTIL_DAY_RAW.strip() else None

# 503-gated pause-retry (2026-08-19) -- NOT the same mechanism as
# AGENT_RETRY/VERDICT_RETRY above, deliberately independent budget, always
# on (no env var to disable -- unlike AGENT_RETRY there's no "maybe it
# doesn't help" question here, a 503 from this proxy always specifically
# means submission_enabled.is_set() is False, i.e. genuinely temporary).
# Root cause (metaclaw_migration_20260819_132608, confirmed via direct
# code read of openclaw_opd_api_server.py's submission_enabled check):
# slime's own rollout loop calls pause_submission() the moment a training
# batch fills up, and doesn't resume until actor train + checkpoint save +
# update_weights finish (observed once: 4m20s, including a 66.5s
# save_model) -- every request hitting the proxy during that window,
# `openclaw agent`'s own included, gets an instant 503. With
# AGENT_RETRY's default of 0 (and even a nonzero value, since that loop
# has no sleep between attempts), a single pause window was silently
# eating every remaining round for the rest of that day and several days
# after it -- not a training-signal-corruption bug like the two fixed
# 2026-08-19 (rollout/training data stayed clean), but a real-data-loss
# one: the driver's own Acc./Compl. aggregate call these rounds "N/A
# infra failure" when the truth is "never got a chance to answer".
#
# Gated specifically on signals confirmed to mean "submission is paused for
# a weight update", NOT applied to every kind of failure. Originally (2026-
# 08-19) this was just the literal 503 -- timeouts were deliberately
# excluded, reasoning that a timeout means the request was already deep
# into a real generation attempt when it died, a genuinely different
# failure mode than a 503 (proxy rejects before doing any work at all), and
# that blindly waiting out a timeout the same way risks compounding a real
# GPU-contention/long-sequence problem into an even longer stall.
#
# Narrowed, not reversed (2026-08-21): real log review (3 lost rounds in
# the K=6 run -- day03/r11, day05/r13, day06/r4 -- plus a retroactive scan
# of every prior migration run, 28/28 real occurrences, 0 counterexamples)
# confirmed that OpenClaw's own gateway timeout message specifically
# ("GatewayClientRequestError: FailoverError: LLM request timed out") is,
# at least in every observed case, itself a symptom of the SAME pause
# window: SGLang's pause_generation aborts whatever was in-flight the
# moment submission_enabled clears, and OpenClaw's gateway reports that as
# a timeout rather than a clean 503 (the connection was live and got cut,
# not rejected up front) -- confirmed via _run_round's own is_aborted/
# degraded-turn-drop handling firing in the proxy log at the exact same
# second. This is NOT a blanket "treat all timeouts as safe to wait
# out" reversal -- only this specific, stable OpenClaw error string is
# added to the marker set; a driver-side asyncio timeout (round_timeout
# actually set to a real value, currently always None -- see
# _run_openclaw_agent's own "Timeout after {timeout}s" fallback, deliberately
# NOT added here) would still fall straight through to plain
# AGENT_RETRY/VERDICT_RETRY behavior, unchanged, since that failure mode
# has no comparable real-data evidence tying it to a pause window.
PAUSE_RETRY_INTERVAL_SECONDS = float(os.environ.get("METACLAW_PAUSE_RETRY_INTERVAL", "15"))
PAUSE_RETRY_MAX_WAIT_SECONDS = float(os.environ.get("METACLAW_PAUSE_RETRY_MAX_WAIT", "900"))

# OpenClaw's own FailoverError text for the two confirmed pause-window
# symptoms -- observed verbatim in real stderr:
#   "FailoverError: 503 status code (no body)" (rejected before any work)
#   "FailoverError: LLM request timed out" (in-flight generation aborted
#   mid-stream when the pause window opened -- added 2026-08-21, see above)
# This is what _run_round checks in the `openclaw agent` subprocess's
# stderr (we have no structured HTTP visibility into that subprocess's own
# internal requests, only its exit code/stdout/stderr). _post_with_retry
# does NOT use these strings -- it talks to the proxy directly via httpx
# and gets a real status code (`e.response.status_code == 503`), which is
# more precise than text-matching and catches a 503 even if httpx's own
# rendering of it differs from OpenClaw's ("503 Service Unavailable" vs
# OpenClaw's "503 status code" are not the same text); the timeout-abort
# symptom has no analogous httpx-visible signal on the verdict/close side,
# since that path was never the one whose in-flight generation got cut.
_AGENT_PAUSE_MARKERS = ("503 status code", "LLM request timed out")

# Day-level resume, take 2 (2026-08-18 -- supersedes the 2026-08-17 "no
# resume, full restart" decision recorded in metaclaw_migration_plan.md).
# The design changed because the SCORING design changed: this driver now
# reports its own Acc./Compl. aggregate computed LIVE from this same run's
# actual responses (matching MetaClaw-official's own rl_run.py methodology
# -- verified via direct read: it runs `metaclaw-bench run --scene-per-train
# N` as ONE pass, so a Table-1-style number is a running aggregate over
# responses generated at whatever training progress existed when each day
# ran, not a separate before/after clean-checkpoint eval). Under that
# design, "restart from day01 on crash" would corrupt the final aggregate
# (days already scored would be scored AGAIN, using different -- more
# trained -- weights, double-counted or inconsistent with what actually
# happened) rather than just wasting compute, so resume is no longer
# optional.
#
# Two of the three original objections to day-level resume no longer apply:
#   - Workspace consistency: confirmed via _copy_workspace_for_test
#     (infer_cmd.py:162-193) that every day's workspace is rebuilt from
#     workspace_src FRESH regardless of day -- days never inherit each
#     other's file state (matches the already-verified "no cross-day
#     persistence" finding). Skipping day N's re-execution on resume does
#     not leave any missing file-state behind for day N+1, because day N+1
#     never depended on day N's actual workspace effects in the first place.
#   - Checkpoint/day desync: training-side --load auto-resume already
#     exists (run_openclaw_topk_select_modelfactory.sh) and is unaffected
#     by this. The remaining risk is narrower than before: if the crash
#     happens between "day N's verdict was POSTed to the proxy" and "that
#     sample's gradient update got saved to a checkpoint", resuming from an
#     older checkpoint means day N's specific training contribution is
#     lost -- but day N's RECORDED SCORE (persisted below, independent of
#     training checkpoints) remains a truthful record of what the model
#     actually produced at that point, so the final aggregate stays
#     correct even though the weight trajectory has a small gap. This is
#     accepted, not solved -- same category of trade-off as the paper's own
#     live single-pass method, which never claims a clean weight/day
#     invariant either.
#
# Persistence and resume are two SEPARATE, independently-controlled
# switches (2026-08-18) -- deliberately not "one env var does both", so a
# normal training run can never accidentally skip a day just because it
# happens to reuse a directory that already has leftover files in it from
# an earlier run. Normal training always processes every day fresh,
# unconditionally, regardless of what METACLAW_PROGRESS_DIR contains.
#
#   METACLAW_PROGRESS_DIR: if set, every day's per-round scores are written
#     to <dir>/<test_id>.json after that day finishes -- pure logging,
#     never changes what this run does. Safe (and recommended) to always
#     set this so a future resume is *possible*, without opting into resume
#     now.
#   METACLAW_RESUME=1: the ONLY thing that makes startup actually check
#     <dir>/<test_id>.json and skip days that already have one (no openclaw
#     agent call, no verdict submission -- just reload the persisted scores
#     for the final aggregate). Requires METACLAW_PROGRESS_DIR to also be
#     set (pointing at the SAME directory the crashed run used). This is a
#     manual, deliberate action after a real crash -- not something that
#     happens implicitly.
PROGRESS_DIR_RAW = os.environ.get("METACLAW_PROGRESS_DIR", "")
PROGRESS_DIR = Path(PROGRESS_DIR_RAW) if PROGRESS_DIR_RAW else None
if PROGRESS_DIR is not None:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

RESUME = os.environ.get("METACLAW_RESUME", "0") == "1"
if RESUME and PROGRESS_DIR is None:
    raise RuntimeError(
        "METACLAW_RESUME=1 requires METACLAW_PROGRESS_DIR to also be set "
        "(pointing at the same directory the crashed run used)."
    )

# report.json/report.md are this run's actual DELIVERABLE (see
# _build_report/_render_report_markdown below) -- a separate concern from
# day-level resume progress above, so a normal run gets its results saved
# to a file without having to opt into resume-tracking just to get that.
# Defaults to PROGRESS_DIR if set (one less directory to configure when
# resume is already in use); falls back to print-only (module docstring's
# original behavior) only if NEITHER is set -- the launch script always
# sets METACLAW_REPORT_DIR (to <LOGS_DIR>/report), so in practice a real
# training run always gets report.json/report.md on disk without the user
# needing to set anything.
REPORT_DIR_RAW = os.environ.get("METACLAW_REPORT_DIR", "")
REPORT_DIR = Path(REPORT_DIR_RAW) if REPORT_DIR_RAW else PROGRESS_DIR
if REPORT_DIR is not None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _day_progress_path(test_id: str) -> Path:
    assert PROGRESS_DIR is not None
    return PROGRESS_DIR / f"{test_id}.json"


def _load_day_progress(test_id: str) -> list[dict[str, Any]] | None:
    """Return the day's persisted round scores (from the file, verbatim --
    may be `[]` if every round that day failed at the infra level) if
    RESUME is on and a progress file exists, else None. Always None when
    RESUME is off, even if a progress file happens to exist. Does NOT
    itself decide whether an empty list counts as "done" -- see the
    `if resumed:` (not `is not None`) check at the call site in main()."""
    if not RESUME or PROGRESS_DIR is None:
        return None
    path = _day_progress_path(test_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(
            "[MetaClawRollout] progress file %s unreadable (%s), re-running day %s",
            path, e, test_id,
        )
        return None


def _save_day_progress(test_id: str, round_scores: list[dict[str, Any]]) -> None:
    if PROGRESS_DIR is None:
        return
    _day_progress_path(test_id).write_text(
        json.dumps(round_scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _score_round_official(test_id: str, group_id: str, round_record: dict[str, Any],
                           answer_text: str, inline_score: dict[str, Any]) -> dict[str, Any]:
    """Compute the OFFICIAL scoring.json-equivalent record for one round.

    Reuses scoring_cmd.py's own scorers directly (same functions
    `metaclaw-bench scoring` calls) rather than approximating with the
    binary inline_score["passed"] used for training reward -- multi_choice
    gets real partial credit (1-(fp+fn)/n), matching what a real
    `metaclaw-bench run` on this same data would have produced, so the
    aggregate this driver reports is genuinely comparable to Table 1's
    Acc./Compl., not a simplified stand-in.

    Return shape deliberately mirrors scoring_cmd.py::_score_one's record
    (test_id/group_id/round_id/question_type/score/metrics) minus
    extracted_answer/correct_answer (report_cmd.py never reads those two) --
    see _build_report/_render_report_markdown below, which are report_cmd.py's
    aggregation and rendering logic ported to work off this list directly
    instead of scanning scoring.json files off disk.
    """
    question_type = round_record.get("type", "multi_choice")
    if question_type == "file_check":
        scored = _score_file_check({"inline_score": inline_score})
    else:
        eval_cfg = round_record.get("eval", {})
        scored = _score_multi_choice(
            answer_text,
            eval_cfg.get("answer", ""),
            len(eval_cfg.get("options", {})),
        )
    return {
        "test_id": test_id,
        "group_id": group_id,
        "round_id": round_record.get("id", "unknown"),
        "question_type": question_type,
        "score": scored["score"],
        "metrics": scored.get("metrics", {}),
    }


def _aggregate_acc_compl(all_round_scores: list[dict[str, Any]]) -> tuple[float, float | None]:
    """Acc. (mean score, all rounds) / Compl. (mean score, file_check rounds
    only) over every round score collected so far -- see paper Table 1's
    caption for the same definitions. This is the ONLY place these numbers
    are computed for this migration (see docs/metaclaw_migration_plan.md
    "训练/评测数据重叠" -- a separate before/after eval against the same
    all_tests.json used for training was tried and retracted: it does not
    avoid train/test overlap, it just moves when the overlap happens, so it
    added no rigor over this live aggregate).
    """
    if not all_round_scores:
        return 0.0, None
    acc = sum(r["score"] for r in all_round_scores) / len(all_round_scores)
    file_check_scores = [r["score"] for r in all_round_scores if r["question_type"] == "file_check"]
    compl = sum(file_check_scores) / len(file_check_scores) if file_check_scores else None
    return acc, compl


_EMPTY_TOKENS = {"input": 0, "output": 0, "cache_read": 0, "total_input": 0}


def _build_report(all_round_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a report.json-equivalent structure from this run's own round
    records -- same shape report_cmd.py::run_report produces (summary +
    by_task, with per-task accuracy and averaged metrics), so this driver's
    output is directly comparable to a `metaclaw-bench run` baseline's
    report.json, side by side.

    Token usage is always zero here (not a bug to "fix" -- see
    _render_report_markdown): this driver never captures OpenClaw's
    llm_log/usage data (_run_openclaw_agent only returns raw stdout text),
    unlike the official infer_result.json pipeline report_cmd.py reads from.
    Left in the same shape rather than omitted so the two reports line up
    field-for-field.
    """
    by_task: dict[str, dict[str, Any]] = {}
    for record in all_round_scores:
        test_id = record["test_id"]
        task = by_task.setdefault(test_id, {
            "accuracy": 0.0,
            "tokens": {"agent": dict(_EMPTY_TOKENS), "compaction": dict(_EMPTY_TOKENS)},
            "questions": [],
            "metrics": {},
        })
        task["questions"].append({
            "group_id": record["group_id"],
            "round_id": record["round_id"],
            "score": record["score"],
            "tokens": {"agent": dict(_EMPTY_TOKENS)},
            "metrics": record["metrics"],
        })
        for key, value in record["metrics"].items():
            task["metrics"][key] = task["metrics"].get(key, 0.0) + value

    summary_metrics: dict[str, float] = {}
    total_questions = 0
    total_correct = 0.0
    for task in by_task.values():
        questions = task["questions"]
        n = len(questions)
        correct = sum(q["score"] for q in questions)
        task["accuracy"] = correct / n if n else 0.0
        for key in task["metrics"]:
            task["metrics"][key] /= n
        for key, value in task["metrics"].items():
            summary_metrics[key] = summary_metrics.get(key, 0.0) + value * n
        total_questions += n
        total_correct += correct

    if total_questions > 0:
        for key in summary_metrics:
            summary_metrics[key] /= total_questions

    return {
        "summary": {
            "total_questions": total_questions,
            "correct": total_correct,
            "accuracy": total_correct / total_questions if total_questions else 0.0,
            "tokens": {"agent": dict(_EMPTY_TOKENS), "compaction": dict(_EMPTY_TOKENS)},
            "metrics": summary_metrics,
        },
        "by_task": by_task,
    }


def _render_report_markdown(report: dict[str, Any]) -> str:
    """Same table layout as report_cmd.py::_render_markdown, so a training
    run's report.md and a `metaclaw-bench run` baseline's report.md can be
    read side by side without translating formats.

    Token Usage section is kept (always 0) rather than dropped, for the same
    field-alignment reason as _build_report -- and per docs/metaclaw_migration_plan.md,
    the baseline's own 0/0 token usage turned out to be a metaclaw report_cmd.py
    log-format mismatch, not a signal either report actually tracks reliably.
    """
    lines = ["# Benchmark Report (metaclaw_rollout_driver.py live run)", ""]
    summary = report["summary"]
    tokens = summary.get("tokens", {})
    agent_tok = tokens.get("agent", {})
    comp_tok = tokens.get("compaction", {})
    lines += [
        "## Summary",
        "",
        f"- **Total questions**: {summary['total_questions']}",
        f"- **Correct**: {summary['correct']:.1f}",
        f"- **Accuracy**: {summary['accuracy']:.1%}",
        "",
        "### Token Usage",
        "",
        "(not tracked by this driver -- always 0, see function docstring)",
        "",
        "| Type | Total Input | Output |",
        "|------|-------------|--------|",
        f"| agent | {agent_tok.get('total_input', 0):,} | {agent_tok.get('output', 0):,} |",
        f"| compaction | {comp_tok.get('total_input', 0):,} | {comp_tok.get('output', 0):,} |",
    ]

    metrics = summary.get("metrics", {})
    if metrics:
        lines += ["", "### Metrics (Average)", ""]
        for key in sorted(metrics.keys()):
            lines.append(f"- **{key}**: {metrics[key]:.4f}")

    lines += ["", "## By Task", ""]

    all_metric_keys: set[str] = set()
    for task_data in report["by_task"].values():
        all_metric_keys.update(task_data.get("metrics", {}).keys())

    header = "| Task | Questions | Correct | Accuracy |"
    separator = "|------|-----------|---------|----------|"
    for key in sorted(all_metric_keys):
        header += f" {key} |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)

    for task_id, task_data in sorted(report["by_task"].items()):
        q_count = len(task_data.get("questions", []))
        correct = sum(q["score"] for q in task_data.get("questions", []))
        acc = task_data["accuracy"]
        row = f"| {task_id} | {q_count} | {correct:.1f} | {acc:.1%} |"
        task_metrics = task_data.get("metrics", {})
        for key in sorted(all_metric_keys):
            row += f" {task_metrics[key]:.4f} |" if key in task_metrics else " - |"
        lines.append(row)

    return "\n".join(lines) + "\n"


async def _post_with_retry(
    client: httpx.AsyncClient,
    session_id: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    retry: int,
    label: str,
) -> None:
    """POST with up to `retry` extra attempts on failure, log style matches
    MetaClaw-official's own `[retry N/M]` convention in infer_cmd.py.
    retry=0 (the default -- see AGENT_RETRY/VERDICT_RETRY) means exactly one
    attempt, no retry, matching MetaClaw's own default behavior and our own
    prior behavior before this was added.

    Attaches Authorization here (not at each call site) so there is one
    place that has to remember it -- see _API_KEY comment above. Also
    treats a non-2xx response as a failure: httpx does not raise on
    4xx/5xx by default, so without raise_for_status() a 401 from the
    proxy's _check_auth would previously look identical to a real success
    in this driver's own logs (the actual metaclaw_migration_20260817_181404
    401s were only found by reading the proxy's log directly, not this
    driver's).

    503 specifically (submission_enabled paused, see PAUSE_RETRY_* module
    comment) gets its own patient wait-and-retry, independent of `retry` --
    a verdict/close POST landing in a pause window is cheap to retry (no
    generation involved, just waiting for the pause to lift), and losing it
    silently would mean a round that genuinely got answered never actually
    reaches the training queue. This still never raises to the caller on
    final failure (existing behavior, log-and-move-on), it just tries
    harder specifically for this one well-understood, always-temporary
    failure mode before giving up.
    """
    headers = {**headers, "Authorization": f"Bearer {_API_KEY}"} if _API_KEY else headers
    attempt = 0
    pause_wait_start: float | None = None
    while True:
        try:
            response = await client.post(PROXY_URL, json=payload, headers=headers)
            response.raise_for_status()
            return
        except Exception as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 503:
                if pause_wait_start is None:
                    pause_wait_start = time.monotonic()
                elapsed = time.monotonic() - pause_wait_start
                if elapsed < PAUSE_RETRY_MAX_WAIT_SECONDS:
                    logger.warning(
                        "[MetaClawRollout] session=%s %s 503 pause-retry "
                        "elapsed=%.0fs/%.0fs, waiting %.0fs before retry",
                        session_id, label, elapsed, PAUSE_RETRY_MAX_WAIT_SECONDS,
                        PAUSE_RETRY_INTERVAL_SECONDS,
                    )
                    await asyncio.sleep(PAUSE_RETRY_INTERVAL_SECONDS)
                    continue
                logger.warning(
                    "[MetaClawRollout] session=%s %s 503 pause-retry exhausted "
                    "after %.0fs -- giving up (this is a pause-retry timeout, "
                    "not a generic submission failure)",
                    session_id, label, elapsed,
                )
                return

            if attempt < retry:
                logger.warning(
                    "[MetaClawRollout] session=%s %s submission failed (attempt %d/%d), "
                    "retrying: %s", session_id, label, attempt + 1, retry, e,
                )
                attempt += 1
                continue
            logger.warning(
                "[MetaClawRollout] session=%s %s submission failed (final attempt): %s",
                session_id, label, e,
            )
            return


async def _send_freeze_signal(client: httpx.AsyncClient, retry: int) -> None:
    """POST the training-freeze signal for METACLAW_TRAIN_UNTIL_DAY (see that
    module constant's comment).

    Recognized by prepare_patched_openclaw_opd.sh's chat_completions patch
    BEFORE the submission_enabled 503 gate -- unlike every other call this
    driver makes, this one is safe to send even during a slime pause window,
    since it is a control-plane message, not a training-data submission. The
    proxy sets a persistent flag and returns immediately without ever calling
    `await request.json()`, so the body here is a genuine placeholder (never
    parsed) rather than something that has to mimic a real chat-completions
    shape.

    Called once per frozen day (not just once when the threshold is first
    crossed, see main()) -- idempotent on the proxy side (re-setting an
    already-True flag is a no-op), but resending guards against this one
    call itself failing silently and leaving the rest of the run un-frozen
    with no other signal that it happened.
    """
    await _post_with_retry(
        client, "metaclaw-freeze-signal",
        payload={},
        headers={"X-Metaclaw-Freeze-Training": "true"},
        retry=retry,
        label="freeze-signal",
    )


async def _send_verdict_turn(
    client: httpx.AsyncClient,
    session_id: str,
    eval_score: float,
    hint: str,
    session_done: bool,
    retry: int = 0,
) -> None:
    """POST a synthetic next-turn carrying the round's deterministic verdict.

    Reuses the SAME reactive next-state mechanism Personal Agent Track
    already depends on (a new "main" turn's messages[-1] becomes the
    previous turn's next_state) -- no new admin endpoint needed. Our patched
    _opd_evaluate() (prepare_patched_openclaw_combine_select.sh) recognizes
    the metaclaw_verdict JSON and uses eval_score/hint directly instead of
    calling any LLM judge.

    max_tokens=0 (2026-08-19, was 8) is a dedicated signal, not just "keep it
    small" -- the proxy's openclaw-rl-metaclaw-verdict-signal-skip patch
    (prepare_patched_openclaw_opd.sh) checks for exactly this value and skips
    calling SGLang entirely for this request. This exists because
    X-Turn-Type: main has an unavoidable side effect beyond the one we want
    (using this call's own messages[-1] as next_state for the PREVIOUS
    pending turn): it ALSO unconditionally registers THIS call's own
    generated completion as a new pending turn awaiting its own future
    next_state. With any non-zero max_tokens the model still generates
    something (confirmed via real Qwen3-4B-Thinking-2507 tokenizer testing
    that even an empty assistant message produces non-empty response tokens
    after chat-template diffing -- the Thinking template's `</think>` closing
    tag and `<|im_end|>` turn marker are structural, not content-conditional,
    so the proxy's own "empty response" skip check can never catch this no
    matter how small max_tokens is), and that stray completion becomes a
    real training sample once the next round's turn supplies it a next_state
    -- confirmed as the actual mechanism behind
    metaclaw_migration_20260818_182736's contamination (69/234 submitted RL
    samples were response_len==13 stubs from exactly this path, scored +1 by
    the step-judge with no idea it was looking at a throwaway artifact, not a
    real intermediate tool-call turn). See docs/metaclaw_migration_plan.md
    for the full investigation.
    """
    verdict_payload = json.dumps(
        {"metaclaw_verdict": True, "eval_score": eval_score, "hint": hint},
        ensure_ascii=False,
    )
    await _post_with_retry(
        client, session_id,
        payload={
            "model": _MODEL_ID,
            "messages": [{"role": "user", "content": verdict_payload}],
            "temperature": 0.0,
            "max_tokens": 0,
        },
        headers={
            "X-Session-Id": session_id,
            "X-Turn-Type": "main",
            "X-Session-Done": "true" if session_done else "false",
        },
        retry=retry,
        label="verdict",
    )


async def _send_session_close_only(
    client: httpx.AsyncClient, session_id: str, retry: int = 0,
) -> None:
    """Close the round's session WITHOUT submitting a verdict for the pending turn.

    Used when the round's `openclaw agent` CLI call itself failed (rc != 0 --
    gateway hiccup, subprocess crash/timeout, not a genuine task attempt).
    Called unconditionally for every failed round (2026-08-19c, was only for
    the day's last round back when a whole day was one session) -- with one
    session per round, there is no longer a "next round in the same session"
    to leave the pending turn for; skipping this close would strand it in
    the proxy's per-session state forever instead of getting force-dropped.
    Mirrors OpenClaw-RL's own General Agent
    tracks (toolcall-rl/swe-rl): both explicitly set ``Sample.Status.ABORTED``
    and return BEFORE the sample reaches reward_func/normal submission when
    generation/execution infrastructure fails mid-attempt, so the failure
    never produces a false training signal -- see
    docs/metaclaw_migration_plan.md for the full comparison.

    X-Turn-Type is deliberately "side" (not "main"): this skips
    _handle_request's whole "main" branch (no attempt to fire evaluation for
    the previous turn using this request's content as next_state) and goes
    straight to the session_done cleanup path, which drops any turn that
    never got a real next_state -- exactly the outcome we want, achieved
    with existing proxy machinery, no new endpoint or turn-buffering change
    needed.
    """
    await _post_with_retry(
        client, session_id,
        payload={
            "model": _MODEL_ID,
            "messages": [{"role": "user", "content": "(session closed after agent failure)"}],
            "temperature": 0.0,
            "max_tokens": 8,
        },
        headers={
            "X-Session-Id": session_id,
            "X-Turn-Type": "side",
            "X-Session-Done": "true",
        },
        retry=retry,
        label="session-close",
    )


def _build_opd_hint(round_record: dict[str, Any], inline_score: dict[str, Any]) -> str:
    """Build the OPD hint text for a FAILED round, per round type.

    NOT the same as MetaClaw-Bench's own _build_feedback_text -- that
    function is designed for feedback shown to the NEXT round (a different
    purpose), and for file_check rounds it reads the question's static
    `feedback.incorrect` text. Verified (see docs/metaclaw_migration_plan.md
    查证记录一, via full reads of check_iso8601.py/check_metadata.py) that a
    single checker script commonly fails for several distinct reasons, but
    `feedback.incorrect` only describes ONE of them -- using it as an OPD
    hint when the actual failure differs would point the hint-conditioned
    distillation target at the wrong correction and actively corrupt
    training, exactly the risk this project flagged before writing any of
    this code. Use the checker's own actual stdout instead (already captured
    into inline_score by _compute_inline_score/_run_file_check -- dynamically
    accurate to the real failure, no extra checker execution needed).

    If the checker produced neither stdout nor stderr (a silent failure --
    e.g. a bare sys.exit(1) with no diagnostic print), there is no reliable
    signal to build a hint from. Return "" rather than falling back to the
    static feedback.incorrect text -- an empty hint correctly yields
    RL-only training (checker eval_score still applies) instead of risking
    an OPD hint that may not describe what actually happened. Confirmed
    (2026-08-19, metaclaw_migration_20260819_153518 log cross-check) this
    silent-failure case is rare: 1 of 55 failed file_check rounds.

    multi_choice has no such problem: _build_feedback_text's per-option text
    is already selected from the agent's actual wrong/missed options, so it
    is reused as-is here.
    """
    question_type = round_record.get("type", "multi_choice")
    if question_type == "file_check":
        stdout = (inline_score.get("stdout") or "").strip()
        stderr = (inline_score.get("stderr") or "").strip()
        if stdout:
            return stdout
        if stderr:
            return stderr
        return ""
    return _build_feedback_text(round_record, inline_score)


# Sanity cap so an otherwise-clean FAIL line doesn't turn the next-round
# feedback into an essay -- checker stdout is meant to add a short, concrete
# progress signal, not replace the static feedback text.
_FC_STDOUT_MAX_LEN = 200


def _filtered_checker_stdout(inline_score: dict[str, Any]) -> str:
    """Return a short, appendable line from a FAILED file_check round's real
    checker stdout -- "" if the stdout isn't in clean enough shape to append.

    Deliberately NOT "append raw stdout unconditionally" (2026-08-20, CLI
    real-data cross-check of metaclaw_migration_20260820_122808): most
    file_check checkers (check_iso8601.py and friends, used by P1/day01-05)
    print a clean, specific `FAIL: field: value` line on failure -- genuinely
    more concrete than the static feedback.incorrect text and worth adding.
    But ~1/4 of real P1 failures instead have the checker script itself
    crash (a bare `python -c` snippet with no exception handling), producing
    a raw Python Traceback -- appending THAT verbatim would make the
    next-round feedback less clear, not more. Skip entirely (fall back to
    the static-only feedback, unchanged from before this function existed)
    on anything containing "Traceback" or not starting with "FAIL" --
    intentionally the simpler of the two options CLI offered (the other
    being "salvage just the last exception line"), since a wrong guess at
    what counts as a usable salvage is worse than adding nothing this round.
    """
    stdout = (inline_score.get("stdout") or "").strip()
    if not stdout or "Traceback" in stdout or not stdout.startswith("FAIL"):
        return ""
    if len(stdout) > _FC_STDOUT_MAX_LEN:
        stdout = stdout[:_FC_STDOUT_MAX_LEN].rstrip() + "..."
    return stdout


# check_filename.py's lenient --dir mode only checks "some 8-digit date +
# snake_case" (any date passes), but the static feedback.incorrect text
# always shows the fictional scenario's specific date as an example -- which
# reads as "must match this exact date". Appending this note only when we
# can positively confirm --dir mode (see _is_dir_mode_filename_check) --
# from day11 onward the checker switches to an exact glob on the scenario
# date (e.g. glob('day11/20260330_*.md')), where that date genuinely DOES
# matter and generalizing it away would teach the wrong thing.
_FC_DIR_MODE_NOTE = (
    " (Note: any valid 8-digit date + snake_case filename satisfies this "
    "check -- the exact date shown above is only an example, not a literal "
    "requirement.)"
)


def _is_dir_mode_filename_check(round_record: dict[str, Any]) -> bool:
    """True iff this round's checker is check_filename.py's lenient --dir
    mode, as opposed to an exact-date glob check. There is no dedicated
    mode/checker-type field in the question data (confirmed 2026-08-20, CLI
    cross-check across all 30 days' questions.json) -- detected from
    round_record["eval"]["command"] instead, the only place this
    distinction is actually recoverable."""
    command = round_record.get("eval", {}).get("command", "")
    return "check_filename.py" in command and "--dir" in command


def _build_next_round_feedback(
    round_record: dict[str, Any], inline_score: dict[str, Any], answer_text: str,
) -> str:
    """Wraps MetaClaw-official's own _build_feedback_text with narrowly-scoped,
    additive-only augmentations -- never replaces or removes the official
    static text, each gated so it can only ever fire on an actual FAILURE (a
    round that already produced non-empty static feedback), matching the
    real-data investigation in docs/metaclaw_migration_plan.md ("方案：
    next-round 反馈 + FORMAT_ERROR + is_invalid_tool_use", 2026-08-20).

    Three independent pieces, safe to reason about separately:
    - multi_choice format failures (feedback text == FORMAT_ERROR, the
      literal constant from prompts.py) get a snippet of the model's own
      actual failed response appended, so 20+ consecutive format failures
      (confirmed in real day10-14 collapse data) no longer produce
      byte-identical feedback every single time.
    - file_check failures get the checker's real stdout appended, if (and
      only if) it looks clean (see _filtered_checker_stdout).
    - file_check failures ALSO get a date-genericization note appended, if
      (and only if) this round's checker is confirmed to be check_filename.py
      --dir mode (see _is_dir_mode_filename_check) -- never applied to the
      exact-date glob checks used from day11 onward.

    2026-08-25 (see docs/metaclaw_migration_plan.md "方案 v2: round 前后
    diff 判定训练奖励", CLI review): `_compute_training_verdict` may have
    already upgraded this round to `inline_score["training_passed"]=True`
    despite the official checker's raw `passed=False` -- a historical-deficit
    false negative (an earlier round in the same day fell behind a
    cumulative min-count/min-entries threshold), not a real mistake THIS
    round made. This function must treat that the same as a genuine pass --
    otherwise eval_score/OPD would say "fine" while the very next round's
    visible [Previous Feedback] still says "FAIL: expected >= N, found K",
    splitting the two signals apart (exactly the gap CLI's review caught).
    Building `feedback_score` as a shallow copy with `passed` forced True
    keeps this consistent with however a genuine pass is normally worded
    (`feedback.correct`, often empty) instead of hardcoding "" here.

    When training_passed is still False, the diff-based diagnostic
    (`inline_score["training_hint"]`, if any) takes priority over the
    official checker stdout -- CLI's explicit call: the round-local
    diagnosis must lead, the official aggregate "found K, need N" line is at
    most a secondary fallback now, not the primary failure text.
    """
    training_passed = inline_score.get("training_passed", inline_score.get("passed", False))
    feedback_score = inline_score
    if training_passed and not inline_score.get("passed", False):
        feedback_score = {**inline_score, "passed": True}

    text = _build_feedback_text(round_record, feedback_score)
    if not text:
        return text

    if text == FORMAT_ERROR and answer_text.strip():
        snippet = answer_text.strip()[:120]
        text = f"{text} (your previous response: {snippet!r})"

    if round_record.get("type") == "file_check" and not training_passed:
        training_hint = inline_score.get("training_hint", "")
        if training_hint:
            text = f"{text}\n{training_hint}"
        else:
            stdout_line = _filtered_checker_stdout(inline_score)
            if stdout_line:
                text = f"{text}\n{stdout_line}"
        if _is_dir_mode_filename_check(round_record):
            text = f"{text}{_FC_DIR_MODE_NOTE}"

    return text


# ---------------------------------------------------------------------------
# Phase 1 fix: round-local training verdict via before/after diff (2026-08-25)
# -- see docs/metaclaw_migration_plan.md "方案 v2: round 前后 diff 判定训练
# 奖励" for the full design + CLI's two rounds of real-data review.
#
# Problem this solves: several file_check checkers (check_filename.py --dir,
# a handful of hand-written `python -c` glob-count snippets, and
# check_done_log.py --min-entries) score CUMULATIVE state across the whole
# day/log, not just this round's own contribution. A round that does exactly
# what its own question asks can still get eval_score=-1 because an EARLIER
# round in the same day fell behind -- the deficit is structurally
# unrecoverable under normal one-artifact-per-round agent behavior (see
# docs/metaclaw_migration_plan.md's "重大发现" section for the original
# check_filename.py --dir discovery and its day01-30 generalization).
#
# Fix: for these specific checker shapes, additionally judge "did THIS round
# make its own correct incremental contribution" via a before/after diff, and
# combine it with the official verdict RELAX-ONLY (`official_pass or
# round_local_pass`) so a diff-parsing bug can only ever fail to fix the
# connat, never invent a new false negative. Official Acc./Compl.
# (_score_round_official) is completely unaffected -- this only changes what
# feeds eval_score / OPD hint / next-round feedback.
#
# Scope (Phase 1, per CLI's 30-day real-data classification):
#   A1     -- check_filename.py --dir [--min-count N]         (70 rounds)
#   A2/A3  -- hand-written `len(glob(...))>=N` / `if glob(...)` (48+27 rounds)
#   B      -- check_done_log.py --min-entries                  (46 rounds/75 segs)
# Explicitly NOT covered in Phase 1 (falls back to the segment's own official
# exit code, unchanged behavior) -- CLI confirmed these need different
# handling or are rare enough not to be worth it yet:
#   - glob + content/ISO validation in the same python -c snippet (2 rounds,
#     e.g. day09/r3, day10/r6)
#   - filtered glob, e.g. `[f for f in glob.glob(...) if 'adr' in f]`
#     (1 round, day11/r6)
#   - two independent globs with two thresholds (1 round, day26/r8 -- Phase 2)
#   - check_backup.py / check_metadata.py / check_iso8601.py segments
#     (~120 segments) -- these are already round-local by design, no fix
#     needed; still re-executed standalone (see _rerun_segment_official) so
#     `&&` short-circuiting from an earlier failed segment doesn't hide
#     their own real exit code.
# ---------------------------------------------------------------------------

# Mirrors check_filename.py::PATTERN (MetaClaw-Bench eval/scripts/
# check_filename.py) verbatim -- a stable single-line regex, not imported
# because that script is invoked as a subprocess via eval.command, never as
# a library. Same "local copy of a small stable regex" risk class already
# accepted for _run_openclaw_agent's local copy of infer_cmd.py's function.
_FC_PATTERN = re.compile(r'^\d{8}_[a-z][a-z0-9_]*\.[a-z0-9]+$')

# Mirrors check_done_log.py::LINE_PATTERN verbatim, same rationale as above.
_DL_LINE_PATTERN = re.compile(
    r'^\[DONE\] (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+08:00) \| ([^\|]+) \| (.+)$'
)

_GLOB_LITERAL_RE = re.compile(r"glob\.glob\('([^']+)'\)")
_GLOB_FILTER_RE = re.compile(r"\[[^\]]*for\s+\w+\s+in\s+glob\.glob\([^\]]*\bif\b")


def _split_command_segments(command: str) -> list[str]:
    """Split an eval.command string on ' && ' into independently-runnable
    segments. CLI confirmed (30-day real-data scan, 0 counterexamples) no
    embedded `python -c` string in this dataset contains a literal '&&' of
    its own, so this naive split is safe for the current data -- not a
    guarantee for hypothetical future question data."""
    return [seg.strip() for seg in command.split(" && ") if seg.strip()]


def _classify_segment(segment: str) -> tuple[str, dict[str, Any]]:
    """Classify one eval.command segment for the round-local diff fix.

    Returns (category, params). category is one of:
      "A1"       -- check_filename.py --dir mode
      "A2A3"     -- a single, unfiltered, count-or-existence-only glob check
      "B"        -- check_done_log.py --min-entries
      "OFFICIAL" -- everything else (backup/metadata/iso8601, multi-glob,
                    content-checking glob snippets, filtered glob, or
                    anything this parser can't confidently recognize) --
                    re-executed standalone, judged by its own exit code,
                    never diffed. Falling into this bucket is always safe
                    (relax-only), it just means this segment's cumulative-
                    count connat (if any) doesn't get resolved this round.
    """
    if "check_filename.py" in segment and "--dir" in segment:
        dir_m = re.search(r'--dir\s+(\S+)', segment)
        ext_m = re.search(r'--ext\s+(\S+)', segment)
        if dir_m and ext_m:
            return "A1", {"dir": dir_m.group(1).rstrip("/"), "ext": ext_m.group(1)}
        return "OFFICIAL", {}

    if "check_done_log.py" in segment:
        log_m = re.search(r'check_done_log\.py\s+(\S+\.log)', segment)
        if log_m:
            prefix_m = re.search(r'--task-prefix\s+(\S+)', segment)
            return "B", {
                "logfile": log_m.group(1),
                "task_prefix": prefix_m.group(1) if prefix_m else None,
            }
        return "OFFICIAL", {}

    # Must come BEFORE the glob check below: day11+ often invokes
    # check_metadata.py with a `$(python -c "...glob.glob(...)...")` subshell
    # argument selecting WHICH file to check -- that glob call is picking a
    # target file for a content check, not counting/existence-checking
    # anything itself, so this must classify as OFFICIAL (re-run whole,
    # judged by its own exit code) even though the substring "glob.glob("
    # is present somewhere in the segment. Caught by a real test case
    # (day11's `check_metadata.py $(python -c "...glob.glob(...)...")`
    # segment) that the naive glob-substring check below misclassified as
    # A2A3 before this guard was added.
    if "check_metadata.py" in segment or "check_backup.py" in segment or "check_iso8601.py" in segment:
        return "OFFICIAL", {}

    if "glob.glob(" in segment:
        glob_calls = _GLOB_LITERAL_RE.findall(segment)
        if len(glob_calls) != 1:
            return "OFFICIAL", {}  # 0 (unparseable) or 2+ (e.g. day26/r8) -- Phase 2/fallback
        if re.search(r'json\.load\(|\.read\(\)', segment):
            return "OFFICIAL", {}  # content check embedded (e.g. day09/r3, day10/r6) -- fallback
        if _GLOB_FILTER_RE.search(segment):
            return "OFFICIAL", {}  # filtered glob (e.g. day11/r6) -- fallback
        return "A2A3", {"glob_expr": glob_calls[0]}

    return "OFFICIAL", {}


def _list_matching_files(directory: Path, ext: str | None) -> set[str]:
    """Read-only directory scan mirroring check_filename.py::check_dir's
    matching logic. ext=None skips the extension filter."""
    if not directory.is_dir():
        return set()
    names = os.listdir(directory)
    if ext is not None:
        ext_lower = ext.lstrip(".").lower()
        return {
            f for f in names
            if _FC_PATTERN.match(f) and f.rsplit(".", 1)[-1].lower() == ext_lower
        }
    return {f for f in names if _FC_PATTERN.match(f)}


def _read_log_lines(path: Path) -> list[str]:
    """Mirrors check_done_log.py's own line-reading logic exactly (drop
    blank lines, strip trailing newline)."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def _snapshot_segment(category: str, params: dict[str, Any], workspace_path: Path) -> Any:
    """Read-only snapshot taken once before the round's agent call and once
    after -- the diff between the two is this round's own contribution.
    Only called for "A1"/"A2A3"/"B" (never "OFFICIAL", which has no diff)."""
    if category == "A1":
        directory = workspace_path / params["dir"]
        return {
            "compliant": _list_matching_files(directory, params["ext"]),
            "all": set(os.listdir(directory)) if directory.is_dir() else set(),
        }
    if category == "A2A3":
        glob_expr = params["glob_expr"]
        directory = workspace_path / (os.path.dirname(glob_expr) or ".")
        return {
            "compliant": set(glob.glob(str(workspace_path / glob_expr))),
            "all": set(os.listdir(directory)) if directory.is_dir() else set(),
        }
    if category == "B":
        return _read_log_lines(workspace_path / params["logfile"])
    return None


def _prepare_before_snapshots(
    round_record: dict[str, Any], workspace_path: Path,
) -> list[tuple[str, tuple[str, dict[str, Any]], Any]] | None:
    """Classify this round's eval.command into segments and snapshot
    whatever each diff-capable segment needs, BEFORE the round's agent call
    runs. Returns None for non-file_check rounds (multi_choice never uses
    this mechanism -- its official `passed` is used as-is, unconditionally).
    Always computed for file_check rounds regardless of how the round turns
    out -- cheap (os.listdir/glob/read lines, no subprocess), the expensive
    per-segment official reruns only happen later if the round actually
    needs them (see _compute_training_verdict)."""
    if round_record.get("type") != "file_check":
        return None
    command = round_record.get("eval", {}).get("command", "")
    if not command:
        return None
    result = []
    for segment_text in _split_command_segments(command):
        category, params = _classify_segment(segment_text)
        before = _snapshot_segment(category, params, workspace_path) if category != "OFFICIAL" else None
        result.append((segment_text, (category, params), before))
    return result


def _diagnose_file_segment(before_all: set[str], after_all: set[str], ext: str | None) -> str:
    """Best-effort, check_filename.py::check_file()-style specific diagnostic
    for why this round produced no new compliant file -- used instead of the
    aggregate "found K, need N" message, which never explains what was wrong
    with what the agent actually wrote this round."""
    new_any = after_all - before_all
    if not new_any:
        return "no new file was created in this round"
    candidate = sorted(new_any)[0]
    if not _FC_PATTERN.match(candidate):
        return f"'{candidate}' does not match YYYYMMDD_snake_case.ext pattern"
    if ext is not None and candidate.rsplit(".", 1)[-1].lower() != ext.lstrip(".").lower():
        got_ext = candidate.rsplit(".", 1)[-1] if "." in candidate else ""
        return (
            f"'{candidate}' matches the naming pattern but has extension "
            f".{got_ext}, expected .{ext.lstrip('.')}"
        )
    return f"'{candidate}' was created but was not counted (unexpected)"


def _diagnose_log_segment(new_lines: list[str], task_prefix: str | None) -> str:
    """Same idea as _diagnose_file_segment but for check_done_log.py --
    reports which of the newly-appended lines (not historical ones) failed
    and why, instead of the aggregate entry count."""
    if not new_lines:
        return "no new entry was appended to done.log this round"
    for line in new_lines:
        m = _DL_LINE_PATTERN.match(line)
        if not m:
            return f"your new done.log entry does not match format: {line!r}"
        if len(m.group(4).strip()) > 80:
            return f"your new done.log entry's summary exceeds 80 chars: {line!r}"
    if task_prefix:
        last_task_id = _DL_LINE_PATTERN.match(new_lines[-1]).group(3).strip()
        if not last_task_id.startswith(task_prefix):
            return (
                f"your new done.log entry's task id {last_task_id!r} does not "
                f"start with {task_prefix!r}"
            )
    return ""  # only reached if this is called when nothing was actually wrong


def _check_new_log_lines(new_lines: list[str], task_prefix: str | None) -> bool:
    if not new_lines:
        return False
    for line in new_lines:
        m = _DL_LINE_PATTERN.match(line)
        if not m or len(m.group(4).strip()) > 80:
            return False
    if task_prefix:
        last_task_id = _DL_LINE_PATTERN.match(new_lines[-1]).group(3).strip()
        if not last_task_id.startswith(task_prefix):
            return False
    return True


def _rerun_segment_official(segment_text: str, workspace_path: Path, timeout: float = 30.0) -> bool:
    """Re-execute one `&&`-segment in isolation (own subprocess, same
    shell=True/cwd semantics as MetaClaw-official's own _run_file_check) so
    its exit code reflects ITS OWN outcome, not whatever the original
    whole-command short-circuit happened to report. CLI confirmed (398-
    segment scan across all 30 days) no write-like calls in any observed
    segment (5 official check_*.py scripts are all read-only `open`; no
    embedded python -c snippet does write/a/w/unlink), so re-running a
    segment here has no side effects on top of what the original run
    already did."""
    try:
        proc = subprocess.run(
            segment_text, shell=True, cwd=str(workspace_path),
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _compute_training_verdict(
    round_record: dict[str, Any],
    inline_score: dict[str, Any],
    before_snapshots: list[tuple[str, tuple[str, dict[str, Any]], Any]] | None,
    workspace_path: Path,
) -> tuple[bool, str]:
    """Return (training_passed, diagnostic_hint) for one round.

    Relax-only: this can only turn an official FAIL into a training PASS,
    never the reverse. If the official chain already passed, or this round
    isn't classifiable file_check (multi_choice, or an empty/unparseable
    command), training_passed is just an alias of the official `passed` and
    no diff logic runs at all -- the (cheap) before_snapshots computed for a
    round that turns out to officially pass are simply unused.
    """
    official_passed = inline_score.get("passed", False)
    if official_passed or not before_snapshots:
        return official_passed, ""

    seg_passes: list[bool] = []
    diag_parts: list[str] = []
    for segment_text, seg_info, before in before_snapshots:
        category, params = seg_info
        seg_official_pass = _rerun_segment_official(segment_text, workspace_path)

        if category == "OFFICIAL" or before is None:
            seg_passes.append(seg_official_pass)
            continue

        seg_round_local_pass = False
        seg_diag = ""
        if category in ("A1", "A2A3"):
            after = _snapshot_segment(category, params, workspace_path)
            new_compliant = after["compliant"] - before["compliant"]
            seg_round_local_pass = bool(new_compliant)
            if not seg_round_local_pass:
                ext = params.get("ext")  # A1 has it, A2A3 doesn't
                seg_diag = _diagnose_file_segment(before["all"], after["all"], ext)
        elif category == "B":
            after = _read_log_lines(workspace_path / params["logfile"])
            if after[: len(before)] != before:
                # Non-append history -- CLI flagged this premise as
                # unverifiable from question data alone; degrade to the
                # official verdict for this segment and log for monitoring
                # (see docs/metaclaw_migration_plan.md "仍未决" item 1).
                logger.warning(
                    "[MetaClawRollout] round=%s done.log %s: history rewritten, "
                    "not pure-append -- degrading to official verdict for this segment",
                    round_record.get("id"), params["logfile"],
                )
            else:
                new_lines = after[len(before):]
                seg_round_local_pass = _check_new_log_lines(new_lines, params.get("task_prefix"))
                if not seg_round_local_pass:
                    seg_diag = _diagnose_log_segment(new_lines, params.get("task_prefix"))

        seg_pass = seg_official_pass or seg_round_local_pass
        seg_passes.append(seg_pass)
        if not seg_pass and seg_diag:
            diag_parts.append(seg_diag)

    training_passed = all(seg_passes) if seg_passes else official_passed
    if training_passed:
        return True, ""
    hint = "\n".join(diag_parts) if diag_parts else _build_opd_hint(round_record, inline_score)
    return False, hint


async def _run_round(
    session_id: str,
    test_id: str,
    group_id: str,
    agent_id: str,
    round_record: dict[str, Any],
    query: str,
    openclaw_config_path: Path,
    openclaw_state_dir: Path,
    project_root: Path,
    workspace_path: Path,
    gateway_port: int,
    round_timeout: float | None,
    retry: int = 0,
) -> tuple[dict[str, Any], bool, dict[str, Any] | None, str]:
    """Run one round via the real `openclaw agent` CLI.

    Mirrors src.infer.infer_cmd._run_group's per-round logic (agent
    invocation + inline scoring), simplified for live training rather than
    resumable batch evaluation -- no infer_result.json files, no
    resume-skip.

    retry mirrors infer_cmd.py's own `_run_question` retry loop exactly
    (same shape, same log style) -- see AGENT_RETRY module constant for why
    this defaults to 0 (matching MetaClaw-official's own default) rather
    than being enabled unconditionally. A 503 from the proxy (submission
    paused for a weight update -- see PAUSE_RETRY_* module comment) is
    handled by a SEPARATE, independent wait-and-retry loop that never
    consumes this `retry` budget -- only non-503 failures (timeout,
    crashes, other errors) count against it, unchanged from before.

    Returns (inline_score, agent_succeeded, official_score, answer_text).
    answer_text (2026-08-20) is this round's own raw model output ("" when
    agent_succeeded=False) -- the caller needs it for the NEXT round's
    feedback text (see _build_next_round_feedback's FORMAT_ERROR handling),
    not for anything about THIS round.
    agent_succeeded=False means the CLI call itself failed on every attempt
    (gateway hiccup, subprocess crash/timeout) -- NOT that the model
    attempted the task and got it wrong. The caller MUST NOT submit
    inline_score as a real training verdict in that case (see
    _send_session_close_only and its call site) -- _compute_inline_score is
    still called here (mirroring MetaClaw-official's own infer_cmd.py, which
    does the same on agent failure) purely for logging visibility, its
    result is not a trustworthy training signal when agent_succeeded=False.
    official_score is None in that same case (an infra failure is not a
    genuine task attempt, so it must not silently count as a 0 in the
    Acc./Compl. aggregate any more than it counts as a real training
    verdict) -- otherwise it's the scoring_cmd.py-equivalent continuous
    score used for that aggregate.

    2026-08-25: the returned inline_score dict also carries two additional
    keys, `training_passed` and `training_hint` (see _compute_training_verdict
    and docs/metaclaw_migration_plan.md "方案 v2") -- the round-local,
    diff-based verdict/diagnostic that the caller should use for
    eval_score/OPD hint/next-round feedback instead of the raw official
    `passed`, which can be a historical-deficit false negative for a handful
    of cumulative-count checker shapes. `_score_round_official`/
    official_score (Acc./Compl.) are computed from the ORIGINAL inline_score
    before these keys are added and are completely unaffected.
    """
    # Snapshot BEFORE the agent runs -- must happen here, not after
    # _compute_inline_score, since the whole point is to see what THIS
    # round's own agent call adds relative to what already existed. Cheap
    # (no subprocess) even when unused (see _compute_training_verdict:
    # skipped entirely when the round officially passes).
    before_snapshots = _prepare_before_snapshots(round_record, workspace_path)

    rc, stdout, stderr = -1, "", ""
    attempt = 0
    pause_wait_start: float | None = None
    while True:
        rc, stdout, stderr = await _run_openclaw_agent(
            session_id=session_id,
            message=query,
            openclaw_config_path=openclaw_config_path,
            openclaw_state_dir=openclaw_state_dir,
            project_root=project_root,
            agent_id=agent_id,
            gateway_port=gateway_port,
            timeout=round_timeout,
        )
        if rc == 0:
            break

        _matched_pause_marker = next(
            (m for m in _AGENT_PAUSE_MARKERS if m in stderr), None
        )
        if _matched_pause_marker is not None:
            if pause_wait_start is None:
                pause_wait_start = time.monotonic()
            elapsed = time.monotonic() - pause_wait_start
            if elapsed < PAUSE_RETRY_MAX_WAIT_SECONDS:
                logger.warning(
                    "[MetaClawRollout] session=%s round=%s pause-retry "
                    "(matched %r) elapsed=%.0fs/%.0fs, waiting %.0fs before retry",
                    session_id, round_record["id"], _matched_pause_marker, elapsed,
                    PAUSE_RETRY_MAX_WAIT_SECONDS, PAUSE_RETRY_INTERVAL_SECONDS,
                )
                await asyncio.sleep(PAUSE_RETRY_INTERVAL_SECONDS)
                continue
            logger.warning(
                "[MetaClawRollout] session=%s round=%s pause-retry (matched %r) "
                "exhausted after %.0fs -- treating as infrastructure failure "
                "(this is a pause-retry timeout, not a generic agent failure)",
                session_id, round_record["id"], _matched_pause_marker, elapsed,
            )
            break

        if attempt < retry:
            logger.warning(
                "[MetaClawRollout] session=%s round=%s openclaw agent failed "
                "(rc=%d), retry %d/%d: %s",
                session_id, round_record["id"], rc, attempt + 1, retry, stderr[:500],
            )
            attempt += 1
            continue
        break

    agent_succeeded = rc == 0
    if not agent_succeeded:
        # Attempt count deliberately not stated here -- could be `retry + 1`
        # (generic failure path) or many more (503 pause-retry loop, see the
        # dedicated log line already emitted above for that case); this
        # message is a summary, not a duplicate of that detail.
        logger.warning(
            "[MetaClawRollout] session=%s round=%s openclaw agent failed (rc=%d): %s "
            "-- treating as infrastructure failure, will NOT submit a training verdict",
            session_id, round_record["id"], rc, stderr[:500],
        )
        answer_text = ""
    else:
        answer_text = stdout

    inline_score = _compute_inline_score(round_record, answer_text, workspace_path)
    passed = inline_score.get("passed", False)
    official_score = (
        _score_round_official(test_id, group_id, round_record, answer_text, inline_score)
        if agent_succeeded else None
    )

    # Round-local diff-based verdict (2026-08-25, Phase 1 -- see docs/
    # metaclaw_migration_plan.md "方案 v2"). Computed AFTER official_score
    # (which must only ever see the pristine official inline_score) --
    # training_passed/training_hint are added to inline_score afterward,
    # additive keys only, nothing existing is overwritten or removed.
    training_passed, training_hint = (
        _compute_training_verdict(round_record, inline_score, before_snapshots, workspace_path)
        if agent_succeeded else (False, "")
    )
    inline_score["training_passed"] = training_passed
    inline_score["training_hint"] = training_hint

    # Human-readable transcript (2026-08-18) -- mirrors openclaw-test's
    # student_chat.py print style (`>>`/`<<` turn markers, full untruncated
    # text) so a person tailing metaclaw_rollout.log can eyeball actual
    # question/answer/verdict content directly, the same way Personal Agent
    # Track's simulation.log already works -- this is deliberately plain
    # print(), not logger, to match that existing convention and stay easy
    # to read without log-level noise. Full stdout is printed even though it
    # may include the model's raw tool-call trace, not just the final answer
    # -- that raw trace is often exactly where a human spots a pattern an
    # agent parsing structured fields alone would miss.
    print(f"\n  {'-' * 56}")
    print(f"  round={round_record['id']}  session={session_id}  agent={test_id}")
    print(f"  {'-' * 56}")
    print(f"  >> Query -> OpenClaw:\n{query}\n")
    if agent_succeeded:
        if any(marker in answer_text for marker in _GENERATE_FAIL_MARKERS):
            # Plain ASCII, deliberately no emoji/unicode decoration -- a
            # print() encoding crash here would take down the whole
            # multi-hour training run over a cosmetic transcript annotation,
            # not worth the risk on whatever locale/codepage the deployment
            # terminal or log capture ends up using.
            print(f"  [GENERATE-FAIL] fallback text detected (see module docstring "
                  f"-- scored normally as a real failed attempt, not an infra failure)")
        print(f"  << OpenClaw -> Query:\n{answer_text}\n")
    else:
        print(f"  << OpenClaw -> Query: AGENT FAILED (rc={rc})\n{stderr[:2000]}\n")
    score_str = f"{official_score['score']:.3f}" if official_score is not None else "N/A (infra failure)"
    print(
        f"  verdict: passed={passed}  training_passed={training_passed}  "
        f"agent_succeeded={agent_succeeded}  official_score={score_str}"
    )
    if training_passed and not passed:
        print(f"  (round-local diff upgraded this round -- see docs/metaclaw_migration_plan.md 方案 v2)")
    elif not training_passed and training_hint:
        print(f"  training hint (diff-based, goes into eval_score=-1 + next round's feedback):\n{training_hint}\n")

    logger.info(
        f"{_GREEN}[MetaClawRollout] session=%s round=%s passed=%s training_passed=%s "
        f"agent_succeeded=%s{_RESET}",
        session_id, round_record["id"], passed, training_passed, agent_succeeded,
    )
    return inline_score, agent_succeeded, official_score, answer_text


_REQUIRED_PLUGINS = ("rl-training-headers",)


def _ensure_plugins_allowlisted(openclaw_json_path: Path, plugin_ids: tuple[str, ...]) -> None:
    """Add *plugin_ids* to this work copy's openclaw.json plugins.allow list.

    Root cause (2026-08-18, verified via direct read of the may_2026_5_11
    OpenClaw snapshot, not the CLI's own vaguer guess): the real training run
    showed OpenClaw's own generated requests carrying NO session_id and NO
    [RL-TRAINING-META] marker at all -- not just intermediate tool-call
    turns, everything OpenClaw itself sends. `openclaw plugins enable
    rl-training-headers` only writes to the GLOBAL `~/.openclaw/openclaw.json`
    plugins.enabled list; it does not affect a specific config's own
    plugins.allow field. MetaClaw-official's own `openclaw_cfg/openclaw.json`
    /`metaclaw.json` templates (which `_prepare_work_copy` copies into every
    day's isolated work copy, completely separate from the global config)
    both hard-code `"allow": ["llm-prompt-logger"]` -- and
    `src/plugins/config-activation-shared.ts::resolvePluginActivationDecisionShared`
    has an explicit gate: `if (config.allow.length > 0 && !explicitlyAllowed)
    return {enabled: false, cause: "not-in-allowlist"}`. A non-empty
    plugins.allow list silently excludes anything not named in it,
    REGARDLESS of global enablement -- rl-training-headers was never
    loaded for a single MetaClaw session, so before_prompt_build never fired,
    so the marker was never injected, so every OpenClaw-generated request
    fell through the proxy's session/turn_type parsing to the
    session=unknown/turn_type=side default and got dropped as non-training
    data. This is why the third real training run (2026-08-18) showed the
    agent visibly working (day01 fully scored, GPU utilized) while
    `waiting for combine samples: 0/16` never moved -- nothing was ever
    reaching the training queue.

    Patches the WORK COPY (not MetaClaw-official's own template files,
    which stay untouched) after _prepare_work_copy/_patch_agent_workspace
    build it, before the gateway starts -- the same "patch the copy, not
    the official source" convention already used for driver-specific state.
    """
    try:
        config = json.loads(openclaw_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(
            "[MetaClawRollout] could not read %s to allowlist plugins (%s), "
            "leaving as-is -- rl-training-headers may silently fail to load",
            openclaw_json_path, e,
        )
        return
    plugins_cfg = config.setdefault("plugins", {})
    allow_list = plugins_cfg.setdefault("allow", [])
    changed = False
    for plugin_id in plugin_ids:
        if plugin_id not in allow_list:
            allow_list.append(plugin_id)
            changed = True
    if changed:
        openclaw_json_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )


async def run_day(
    test: dict[str, Any],
    all_tests: dict[str, Any],
    project_root: Path,
    retry: int = 0,
) -> list[dict[str, Any]]:
    """Run one day and return its round-level official scores (see
    _score_round_official) for the caller to fold into the final
    Acc./Compl. aggregate. Persists them to METACLAW_PROGRESS_DIR (if set)
    only after the whole day completes without raising -- a mid-day crash
    must not leave a partial/misleading progress file that a resumed run
    would treat as "day done".
    """
    test_id = test["id"]
    agent_id = test["agent"]
    eval_name = test["eval"]

    print(f"\n{'#' * 60}")
    print(f"# Day {test_id}")
    print(f"{'#' * 60}")

    workspace_src = resolve_path(
        all_tests["workspace_src"].replace("${METACLAW_ROOT}", str(project_root))
    )
    openclaw_state_dir = resolve_path(all_tests["openclaw_state_dir"])
    openclaw_config_src = None
    openclaw_config_file_raw = all_tests.get("openclaw_config_file")
    if openclaw_config_file_raw:
        openclaw_config_src = resolve_path(
            openclaw_config_file_raw.replace("${METACLAW_ROOT}", str(project_root))
        )
    eval_dir = resolve_path(all_tests["eval_dir"])

    # Isolated work copy -- same official functions MetaClaw-Bench's own
    # evaluation harness uses, so cross-day isolation semantics match
    # exactly (see docs/metaclaw_migration_plan.md 查证记录二 第 3 条:
    # no cross-day persistence, each day starts from a fresh copy).
    work_openclaw_state_dir = _prepare_work_copy(
        openclaw_state_dir, project_root, openclaw_config_src
    )
    work_dir = work_openclaw_state_dir.parent
    openclaw_json_path = work_openclaw_state_dir / "openclaw.json"

    workspace_copy = _copy_workspace_for_test(workspace_src, work_dir, test_id)
    _copy_eval_scripts(eval_dir, workspace_copy)
    _patch_agent_workspace(openclaw_json_path, agent_id, workspace_copy)
    _ensure_plugins_allowlisted(openclaw_json_path, _REQUIRED_PLUGINS)

    gateway_port = _find_free_port()
    gateway_proc, gateway_log = await _start_work_gateway(work_openclaw_state_dir, gateway_port)
    ready = await _wait_for_gateway(gateway_port)
    if not ready:
        if gateway_proc.returncode is None:
            gateway_proc.terminate()
            await gateway_proc.wait()
        raise RuntimeError(f"[{test_id}] gateway on port {gateway_port} never became ready")

    query_reader = get_default_query_reader()
    groups = query_reader.read_queries(eval_dir, eval_name)

    day_round_scores: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for group in groups:
                rounds = group.get("rounds", [])
                prev_inline_score: dict | None = None
                prev_round_record: dict | None = None
                prev_answer_text: str = ""

                for idx, round_record in enumerate(rounds):
                    # One session PER ROUND (2026-08-19c), not one session
                    # per day. Was: session_id = f"{_SESSION_ID_PREFIX}{test_id}"
                    # shared across every round in the day, matching
                    # MetaClaw-official's own _run_group. Changed because that
                    # sharing is exactly what let one file_check round's very
                    # long response (transcript growth confirmed via
                    # metaclaw_migration_20260819_153518/173654 logs -- CLI
                    # cross-check: day07 r5 alone ~17k chars) balloon the SAME
                    # session's context for every later round THAT DAY,
                    # including unrelated multi_choice rounds that were
                    # otherwise answered fine (day01-06 MC steady at
                    # 85-97%) -- once a day's transcript got long enough,
                    # everything after it, file_check and multi_choice alike,
                    # died to context overflow / empty responses (day07 r6
                    # onward, day08 r10 onward, day09 r8 onward, day10 r10
                    # onward, all confirmed via real log review). Per-round
                    # session_id gives each round a fresh, empty transcript
                    # (_prepare_session just touches a new .jsonl -- verified
                    # via direct read of MetaClaw-official's infer_cmd.py,
                    # no other persistence mechanism exists) while leaving the
                    # day's WORKSPACE untouched (same workspace_copy/
                    # gateway_port/openclaw_json_path for the whole day,
                    # unchanged below) -- a later round can still see files an
                    # earlier round actually wrote, it just no longer inherits
                    # the earlier round's raw chat transcript. Cross-round
                    # continuity for "what went wrong last time" is carried by
                    # the explicit [Previous Feedback] text (query/
                    # with_feedback below, unaffected by this change), not by
                    # shared conversation history. This does NOT fix a
                    # file_check round writing an overlong response or
                    # scoring 0 -- only stops that from dragging down every
                    # later round in the same day. A deliberate divergence
                    # from MetaClaw-official's own eval harness (_run_group
                    # shares one session across a day) -- acceptable because
                    # MetaClaw's own scorer never reads the transcript, and
                    # this migration was never aligned with MetaClaw's own
                    # training-mode code (openclaw_env_rollout.py) either,
                    # which uses a completely different one-session-per-task
                    # model with no day/round/feedback structure at all (see
                    # docs/metaclaw_migration_plan.md "三方对照"). The
                    # "{group_id}-" component is redundant in every real
                    # all_tests.json (group["id"] always == test_id for
                    # QuestionsJsonQueryReader-format data, confirmed via a
                    # full 346-round scan across all 30 days' questions.json,
                    # all round ids are plain r1..r15, alphanumeric only) but
                    # kept anyway as a defensive guard against
                    # EvalFlowQueryReader's legacy format, which CAN produce
                    # multiple groups per day -- costs nothing, protects
                    # against an assumption this code does not actually rely
                    # on holding forever. Also structurally closes the
                    # previously-deferred "跨 round 污染" risk (see
                    # docs/metaclaw_migration_plan.md 下一步工程任务 第 1 项):
                    # that bug required a crashed round's orphaned pending
                    # turn to be picked up by the SAME session's next message;
                    # with every round now its own session (and session_done
                    # sent unconditionally below, not just on the day's last
                    # round), there is no longer a "same session" for a later
                    # round to leak into.
                    round_session_id = (
                        f"{_SESSION_ID_PREFIX}{test_id}-{group.get('id', 'unknown')}-"
                        f"{round_record['id']}"
                    )
                    _prepare_session(work_openclaw_state_dir, agent_id, round_session_id)

                    question_text = round_record["question"]
                    feedback_text: str | None = None
                    if prev_inline_score is not None and prev_round_record is not None:
                        candidate = _build_next_round_feedback(
                            prev_round_record, prev_inline_score, prev_answer_text,
                        )
                        if candidate:
                            feedback_text = candidate
                    query = (
                        with_feedback(feedback_text, question_text)
                        if feedback_text else question_text
                    )

                    inline_score, agent_succeeded, official_score, answer_text = await _run_round(
                        session_id=round_session_id,
                        test_id=test_id,
                        group_id=group.get("id", "unknown"),
                        agent_id=agent_id,
                        round_record=round_record,
                        query=query,
                        openclaw_config_path=openclaw_json_path,
                        openclaw_state_dir=work_openclaw_state_dir,
                        project_root=project_root,
                        workspace_path=workspace_copy,
                        gateway_port=gateway_port,
                        round_timeout=None,
                        retry=retry,
                    )
                    if official_score is not None:
                        day_round_scores.append(official_score)

                    if agent_succeeded:
                        # 2026-08-25 (see docs/metaclaw_migration_plan.md "方案
                        # v2"): eval_score/OPD hint now key off training_passed
                        # (the round-local diff verdict), not the raw official
                        # `passed` -- relax-only, so this can only turn an
                        # official FAIL into a training PASS, never the
                        # reverse. training_hint (diff-based diagnosis) is
                        # preferred over _build_opd_hint's aggregate stdout;
                        # falls back to it only when training_hint is empty
                        # (e.g. an OFFICIAL-only segment failed, or the
                        # silent-failure case _build_opd_hint's own docstring
                        # already documents).
                        training_passed = inline_score.get(
                            "training_passed", inline_score.get("passed", False)
                        )
                        eval_score = 1.0 if training_passed else -1.0
                        hint = "" if training_passed else (
                            inline_score.get("training_hint") or _build_opd_hint(round_record, inline_score)
                        )
                        if hint:
                            print(f"  OPD hint (goes into next round's feedback):\n{hint}\n")
                        # session_done=True unconditionally (2026-08-19c) --
                        # was `session_done=is_last_round` back when a whole
                        # day was one session. Every round is now its own
                        # complete session, so every round's verdict must
                        # close it (same reasoning as _send_session_close_only
                        # below), not just the day's last round.
                        await _send_verdict_turn(
                            client, round_session_id, eval_score, hint,
                            session_done=True, retry=VERDICT_RETRY,
                        )
                    else:
                        # Infrastructure failure, not a real task attempt -- do NOT
                        # submit a verdict (would fabricate a false -1 training
                        # signal). Mirrors toolcall-rl/swe-rl's Sample.Status.ABORTED
                        # early-return; see _send_session_close_only's docstring.
                        # Unconditional (2026-08-19c), not `if is_last_round:` --
                        # under one-session-per-day, a non-last-round failure's
                        # orphaned pending turn was left to be picked up by the
                        # same session's next real message (the very "跨 round
                        # 污染" mechanism this change also closes off). Under
                        # one-session-per-round there is no longer a "next
                        # message in the same session" ever coming -- skipping
                        # this close would leave that pending turn stuck in the
                        # proxy's per-session state forever instead of being
                        # force-dropped, which is strictly worse than the old
                        # behavior, not just a no-op.
                        await _send_session_close_only(
                            client, round_session_id, retry=VERDICT_RETRY,
                        )

                    prev_inline_score = inline_score
                    prev_round_record = round_record
                    prev_answer_text = answer_text
    finally:
        if gateway_proc.returncode is None:
            gateway_proc.terminate()
            await gateway_proc.wait()

    # Only reached if the day completed without raising -- see docstring on
    # why this must not happen inside `finally`.
    _save_day_progress(test_id, day_round_scores)
    logger.info(f"{_GREEN}[MetaClawRollout] day=%s done{_RESET}", test_id)
    return day_round_scores


async def main() -> None:
    all_tests_raw = os.environ.get("METACLAW_ALL_TESTS_JSON")
    if not all_tests_raw:
        raise RuntimeError("METACLAW_ALL_TESTS_JSON env var must point at all_tests.json")
    all_tests_path = resolve_path(all_tests_raw)
    all_tests = json.loads(all_tests_path.read_text(encoding="utf-8"))
    test_list = all_tests.get("test", [])
    project_root = get_project_root()

    # Optional smoke-test knob (2026-08-17): only process the first N days
    # instead of the full all_tests.json list. For verifying the pipeline
    # end-to-end (real openclaw agent subprocess, checker execution, verdict
    # recognized by the proxy, session_id propagation) before committing to
    # a full day01->day30 run -- see docs/metaclaw_migration_plan.md
    # pre-training checklist. Unset (default) processes every day, unchanged
    # from prior behavior.
    max_days_raw = os.environ.get("METACLAW_MAX_DAYS", "")
    if max_days_raw:
        max_days = int(max_days_raw)
        test_list = test_list[:max_days]

    logger.info(
        f"{_YELLOW}[MetaClawRollout] %d day(s) loaded from %s, concurrency=1, "
        f"strict order, agent_retry=%d, verdict_retry=%d, progress_dir=%s, resume=%s, "
        f"report_dir=%s, train_until_day=%s{_RESET}",
        len(test_list), all_tests_path, AGENT_RETRY, VERDICT_RETRY, PROGRESS_DIR, RESUME,
        REPORT_DIR, TRAIN_UNTIL_DAY if TRAIN_UNTIL_DAY is not None else "disabled",
    )

    # concurrency=1, strict day01 -> day30 order. If METACLAW_RESUME=1 (see
    # module-level comment above), a day already completed by a prior
    # (crashed) run is skipped entirely -- no re-execution, no
    # re-submission to the training proxy -- and its persisted scores are
    # reused for the final aggregate as-is. Off (default): every day always
    # runs fresh, unconditionally, matching prior behavior exactly --
    # normal training is never at risk of accidentally skipping a day.
    all_round_scores: list[dict[str, Any]] = []
    # Only ever populated when TRAIN_UNTIL_DAY is set -- see the reporting
    # section below, which only reads/prints these (and only adds the extra
    # report.json keys) in that same case, so leaving them empty when the
    # feature is off has no observable effect.
    train_round_scores: list[dict[str, Any]] = []
    frozen_round_scores: list[dict[str, Any]] = []
    for day_index, test in enumerate(test_list, start=1):
        test_id = test["id"]
        is_frozen_day = TRAIN_UNTIL_DAY is not None and day_index > TRAIN_UNTIL_DAY
        bucket = frozen_round_scores if is_frozen_day else train_round_scores
        resumed = _load_day_progress(test_id)
        # Deliberately `if resumed:` not `if resumed is not None:` -- a day
        # where every round failed at the infrastructure level (e.g. the
        # 2026-08-18 context-overflow incident) still completes run_day()
        # without raising, so it gets persisted as test_id.json containing
        # `[]`. An empty list is a real, useful diagnostic record ("this day
        # was attempted and produced zero samples"), but it must NOT count
        # as "done" for resume purposes -- there is nothing to reuse, and
        # skipping it would silently give up retrying that day forever. `[]`
        # and `None` are both falsy, so this one check correctly retries
        # both "file doesn't exist" and "file exists but is empty".
        if resumed:
            logger.info(
                f"{_YELLOW}[MetaClawRollout] day=%s already completed (resume), "
                f"skipping -- reusing %d persisted round score(s){_RESET}",
                test_id, len(resumed),
            )
            all_round_scores.extend(resumed)
            bucket.extend(resumed)
            continue
        if is_frozen_day:
            # Sent every frozen day (not just once when the threshold is
            # first crossed) -- see _send_freeze_signal's docstring for why.
            async with httpx.AsyncClient(timeout=120.0) as freeze_client:
                await _send_freeze_signal(freeze_client, retry=VERDICT_RETRY)
        day_scores = await run_day(test, all_tests, project_root, retry=AGENT_RETRY)
        all_round_scores.extend(day_scores)
        bucket.extend(day_scores)

    acc, compl = _aggregate_acc_compl(all_round_scores)
    compl_str = f"{compl:.1%}" if compl is not None else "n/a (no file_check rounds)"
    logger.info(
        f"{_GREEN}[MetaClawRollout] run complete. %d day(s), %d scored round(s). "
        f"Acc.=%.1f%% Compl.=%s{_RESET}",
        len(test_list), len(all_round_scores), acc * 100, compl_str,
    )

    # report.json/report.md -- same filenames and shape as report_cmd.py's
    # own output (see _build_report/_render_report_markdown), so this run's
    # results can be read side by side with a `metaclaw-bench run` baseline
    # without translating formats. Compl. has no equivalent field in the
    # official report (it's specific to this migration's Table 1 mapping,
    # not something report_cmd.py itself computes -- see
    # docs/metaclaw_migration_plan.md), so it's logged above and appended as
    # an extra line in report.md rather than invented as a fake report.json
    # field that would not match the official schema.
    report = _build_report(all_round_scores)
    md = _render_report_markdown(report) + f"\n**Compl. (file_check only)**: {compl_str}\n"

    # Train/Frozen window split (2026-08-20) -- ONLY added when
    # METACLAW_TRAIN_UNTIL_DAY is set. When it is not, `report` keeps
    # exactly the same top-level keys as before this feature existed --
    # this `if` is the one place that has to hold for "unset = unchanged
    # report format" to actually be true, not just "unset = unchanged
    # training behavior". Namespaced under metaclaw_* keys rather than
    # touching report's existing fields, so a report.json from a run that
    # DOES use this feature still compares field-for-field against a
    # `metaclaw-bench run` baseline's report.json for everything except
    # these new keys (see _build_report's own docstring for why that
    # alignment matters).
    if TRAIN_UNTIL_DAY is not None:
        train_acc, train_compl = _aggregate_acc_compl(train_round_scores)
        frozen_acc, frozen_compl = _aggregate_acc_compl(frozen_round_scores)
        train_compl_str = f"{train_compl:.1%}" if train_compl is not None else "n/a"
        frozen_compl_str = f"{frozen_compl:.1%}" if frozen_compl is not None else "n/a"
        logger.info(
            f"{_GREEN}[MetaClawRollout] TRAIN_UNTIL_DAY=%d -- "
            f"Train window (day1-%d, %d round(s), rolling weights): "
            f"Acc.=%.1f%% Compl.=%s | "
            f"Frozen window (day%d-%d, %d round(s), fixed checkpoint): "
            f"Acc.=%.1f%% Compl.=%s{_RESET}",
            TRAIN_UNTIL_DAY, TRAIN_UNTIL_DAY, len(train_round_scores),
            train_acc * 100, train_compl_str,
            TRAIN_UNTIL_DAY + 1, len(test_list), len(frozen_round_scores),
            frozen_acc * 100, frozen_compl_str,
        )
        report["metaclaw_train_until_day"] = TRAIN_UNTIL_DAY
        report["metaclaw_train_window"] = _build_report(train_round_scores)
        report["metaclaw_frozen_window"] = _build_report(frozen_round_scores)
        md += (
            f"\n## Train / Frozen window split (METACLAW_TRAIN_UNTIL_DAY={TRAIN_UNTIL_DAY})\n\n"
            f"**Train window is process-monitoring only** (rolling weights while "
            f"training -- same caveat as the blended full-run number above). "
            f"**Frozen window is the number that answers \"did training help\"** "
            f"(fixed checkpoint from day {TRAIN_UNTIL_DAY}, no further weight "
            f"updates from day {TRAIN_UNTIL_DAY + 1} onward).\n\n"
            f"| Window | Days | Rounds | Acc. | Compl. |\n"
            f"|---|---|---|---|---|\n"
            f"| Train (rolling weights) | day1-{TRAIN_UNTIL_DAY} | "
            f"{len(train_round_scores)} | {train_acc * 100:.1f}% | {train_compl_str} |\n"
            f"| Frozen (fixed checkpoint) | day{TRAIN_UNTIL_DAY + 1}-{len(test_list)} | "
            f"{len(frozen_round_scores)} | {frozen_acc * 100:.1f}% | {frozen_compl_str} |\n"
        )

    print("\n" + md)
    if REPORT_DIR is not None:
        (REPORT_DIR / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (REPORT_DIR / "report.md").write_text(md, encoding="utf-8")
        logger.info(
            f"{_GREEN}[MetaClawRollout] report written: %s{_RESET}",
            REPORT_DIR / "report.json",
        )
    else:
        logger.warning(
            f"{_YELLOW}[MetaClawRollout] METACLAW_REPORT_DIR/METACLAW_PROGRESS_DIR "
            f"not set -- report.json/report.md NOT written to disk, only printed "
            f"above and to this log file{_RESET}"
        )


if __name__ == "__main__":
    # 2026-08-18: stdout switches from line-buffered to fully block-buffered
    # (4-8KB) the moment it's not a tty -- exactly what the launch script's
    # `> metaclaw_rollout.log 2>&1 &` redirect does. logging's default
    # handler writes to stderr, which stays unbuffered regardless, so only
    # the terse logger.info lines showed up in real time while the detailed
    # print() transcripts (added this same day for readability) sat in the
    # buffer, invisible to `tail -f` until it happened to fill or the
    # process exited. Force line buffering so both streams behave the same
    # way once redirected to a file.
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
