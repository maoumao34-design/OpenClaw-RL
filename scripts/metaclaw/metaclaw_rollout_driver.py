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
  - Each day is ONE proxy session, session_id = f"metaclaw-{test_id}"
    (the "metaclaw-" prefix is load-bearing: prepare_patched_openclaw_opd.sh
    pattern-matches it via _METACLAW_SESSION_RE to flag every turn in the
    session as MetaClaw round mode -- intermediate tool-call turns cannot
    carry a custom body field of their own, since OpenClaw's internal HTTP
    client constructs those requests, not this driver).
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
import json
import logging
import os
import secrets
import sys
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
    _run_openclaw_agent,
    _start_work_gateway,
    _wait_for_gateway,
)
from src.infer.prompts import with_feedback  # noqa: E402
from src.infer.query_reader import get_default_query_reader  # noqa: E402
from src.scoring.scoring_cmd import _score_file_check, _score_multi_choice  # noqa: E402
from src.utils import get_project_root, resolve_path  # noqa: E402

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


def _day_progress_path(test_id: str) -> Path:
    assert PROGRESS_DIR is not None
    return PROGRESS_DIR / f"{test_id}.json"


def _load_day_progress(test_id: str) -> list[dict[str, Any]] | None:
    """Return the day's persisted round scores if RESUME is on and the day
    was already completed by a prior run, else None (always None when
    RESUME is off, even if a progress file happens to exist)."""
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


def _score_round_official(round_record: dict[str, Any], answer_text: str,
                           inline_score: dict[str, Any]) -> dict[str, Any]:
    """Compute the OFFICIAL scoring.json-equivalent score for one round.

    Reuses scoring_cmd.py's own scorers directly (same functions
    `metaclaw-bench scoring` calls) rather than approximating with the
    binary inline_score["passed"] used for training reward -- multi_choice
    gets real partial credit (1-(fp+fn)/n), matching what a real
    `metaclaw-bench run` on this same data would have produced, so the
    aggregate this driver reports is genuinely comparable to Table 1's
    Acc./Compl., not a simplified stand-in.
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
    return {"question_type": question_type, "score": scored["score"]}


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
    """
    headers = {**headers, "Authorization": f"Bearer {_API_KEY}"} if _API_KEY else headers
    for attempt in range(retry + 1):
        try:
            response = await client.post(PROXY_URL, json=payload, headers=headers)
            response.raise_for_status()
            return
        except Exception as e:
            if attempt < retry:
                logger.warning(
                    "[MetaClawRollout] session=%s %s submission failed (attempt %d/%d), "
                    "retrying: %s", session_id, label, attempt + 1, retry, e,
                )
            else:
                logger.warning(
                    "[MetaClawRollout] session=%s %s submission failed (final attempt): %s",
                    session_id, label, e,
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
    calling any LLM judge. max_tokens is kept small since this call's own
    completion is never read -- only its message content (used as
    next_state) and headers matter.

    Only call this when the round's `openclaw agent` CLI call actually
    succeeded (rc == 0) -- see _send_session_close_only for the
    infrastructure-failure case, and the comment at its call site in
    run_day for why these must not be conflated.
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
            "max_tokens": 8,
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
    """Close the day's session WITHOUT submitting a verdict for the pending turn.

    Used when the round's `openclaw agent` CLI call itself failed (rc != 0 --
    gateway hiccup, subprocess crash/timeout, not a genuine task attempt) and
    this was the day's last round. Mirrors OpenClaw-RL's own General Agent
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
        return _build_feedback_text(round_record, inline_score)
    return _build_feedback_text(round_record, inline_score)


async def _run_round(
    session_id: str,
    round_record: dict[str, Any],
    query: str,
    openclaw_config_path: Path,
    openclaw_state_dir: Path,
    project_root: Path,
    workspace_path: Path,
    gateway_port: int,
    round_timeout: float | None,
    retry: int = 0,
) -> tuple[dict[str, Any], bool, dict[str, Any] | None]:
    """Run one round via the real `openclaw agent` CLI.

    Mirrors src.infer.infer_cmd._run_group's per-round logic (agent
    invocation + inline scoring), simplified for live training rather than
    resumable batch evaluation -- no infer_result.json files, no
    resume-skip.

    retry mirrors infer_cmd.py's own `_run_question` retry loop exactly
    (same `for attempt in range(retry + 1)` shape, same log style) -- see
    AGENT_RETRY module constant for why this defaults to 0 (matching
    MetaClaw-official's own default) rather than being enabled unconditionally.

    Returns (inline_score, agent_succeeded, official_score).
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
    """
    rc, stdout, stderr = -1, "", ""
    for attempt in range(retry + 1):
        rc, stdout, stderr = await _run_openclaw_agent(
            session_id=session_id,
            message=query,
            openclaw_config_path=openclaw_config_path,
            openclaw_state_dir=openclaw_state_dir,
            project_root=project_root,
            gateway_port=gateway_port,
            timeout=round_timeout,
        )
        if rc == 0:
            break
        if attempt < retry:
            logger.warning(
                "[MetaClawRollout] session=%s round=%s openclaw agent failed "
                "(rc=%d), retry %d/%d: %s",
                session_id, round_record["id"], rc, attempt + 1, retry, stderr[:500],
            )

    agent_succeeded = rc == 0
    if not agent_succeeded:
        logger.warning(
            "[MetaClawRollout] session=%s round=%s openclaw agent failed (rc=%d) "
            "after %d attempt(s): %s "
            "-- treating as infrastructure failure, will NOT submit a training verdict",
            session_id, round_record["id"], rc, retry + 1, stderr[:500],
        )
        answer_text = ""
    else:
        answer_text = stdout

    inline_score = _compute_inline_score(round_record, answer_text, workspace_path)
    passed = inline_score.get("passed", False)
    logger.info(
        f"{_GREEN}[MetaClawRollout] session=%s round=%s passed=%s agent_succeeded=%s{_RESET}",
        session_id, round_record["id"], passed, agent_succeeded,
    )
    official_score = (
        _score_round_official(round_record, answer_text, inline_score)
        if agent_succeeded else None
    )
    return inline_score, agent_succeeded, official_score


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
    # Load-bearing: NOT test["session"] verbatim -- must start with
    # _SESSION_ID_PREFIX so the proxy's _METACLAW_SESSION_RE recognizes
    # every turn in this day as MetaClaw round mode. One session per day,
    # matching MetaClaw-official's own _run_group (all of a day's rounds
    # share one session transcript).
    session_id = f"{_SESSION_ID_PREFIX}{test_id}"

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

    gateway_port = _find_free_port()
    gateway_proc, gateway_log = await _start_work_gateway(work_openclaw_state_dir, gateway_port)
    ready = await _wait_for_gateway(gateway_port)
    if not ready:
        if gateway_proc.returncode is None:
            gateway_proc.terminate()
            await gateway_proc.wait()
        raise RuntimeError(f"[{test_id}] gateway on port {gateway_port} never became ready")

    _prepare_session(work_openclaw_state_dir, agent_id, session_id)

    query_reader = get_default_query_reader()
    groups = query_reader.read_queries(eval_dir, eval_name)

    day_round_scores: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for group in groups:
                rounds = group.get("rounds", [])
                prev_inline_score: dict | None = None
                prev_round_record: dict | None = None

                for idx, round_record in enumerate(rounds):
                    question_text = round_record["question"]
                    feedback_text: str | None = None
                    if prev_inline_score is not None and prev_round_record is not None:
                        candidate = _build_feedback_text(prev_round_record, prev_inline_score)
                        if candidate:
                            feedback_text = candidate
                    query = (
                        with_feedback(feedback_text, question_text)
                        if feedback_text else question_text
                    )

                    inline_score, agent_succeeded, official_score = await _run_round(
                        session_id=session_id,
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

                    is_last_round = idx == len(rounds) - 1

                    if agent_succeeded:
                        passed = inline_score.get("passed", False)
                        eval_score = 1.0 if passed else -1.0
                        hint = "" if passed else _build_opd_hint(round_record, inline_score)
                        await _send_verdict_turn(
                            client, session_id, eval_score, hint,
                            session_done=is_last_round, retry=VERDICT_RETRY,
                        )
                    else:
                        # Infrastructure failure, not a real task attempt -- do NOT
                        # submit a verdict (would fabricate a false -1 training
                        # signal). Mirrors toolcall-rl/swe-rl's Sample.Status.ABORTED
                        # early-return; see _send_session_close_only's docstring.
                        # Only need to actually talk to the proxy if this was the
                        # day's last round (otherwise the round's turn, if any was
                        # even submitted, simply stays pending and gets picked up
                        # normally by whatever comes next).
                        if is_last_round:
                            await _send_session_close_only(
                                client, session_id, retry=VERDICT_RETRY,
                            )

                    prev_inline_score = inline_score
                    prev_round_record = round_record
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
        f"strict order, agent_retry=%d, verdict_retry=%d, progress_dir=%s, resume=%s{_RESET}",
        len(test_list), all_tests_path, AGENT_RETRY, VERDICT_RETRY, PROGRESS_DIR, RESUME,
    )

    # concurrency=1, strict day01 -> day30 order. If METACLAW_RESUME=1 (see
    # module-level comment above), a day already completed by a prior
    # (crashed) run is skipped entirely -- no re-execution, no
    # re-submission to the training proxy -- and its persisted scores are
    # reused for the final aggregate as-is. Off (default): every day always
    # runs fresh, unconditionally, matching prior behavior exactly --
    # normal training is never at risk of accidentally skipping a day.
    all_round_scores: list[dict[str, Any]] = []
    for test in test_list:
        test_id = test["id"]
        resumed = _load_day_progress(test_id)
        if resumed is not None:
            logger.info(
                f"{_YELLOW}[MetaClawRollout] day=%s already completed (resume), "
                f"skipping -- reusing %d persisted round score(s){_RESET}",
                test_id, len(resumed),
            )
            all_round_scores.extend(resumed)
            continue
        day_scores = await run_day(test, all_tests, project_root, retry=AGENT_RETRY)
        all_round_scores.extend(day_scores)

    acc, compl = _aggregate_acc_compl(all_round_scores)
    compl_str = f"{compl:.1%}" if compl is not None else "n/a (no file_check rounds)"
    logger.info(
        f"{_GREEN}[MetaClawRollout] run complete. %d day(s), %d scored round(s). "
        f"Acc.=%.1f%% Compl.=%s{_RESET}",
        len(test_list), len(all_round_scores), acc * 100, compl_str,
    )
    if PROGRESS_DIR is not None:
        (PROGRESS_DIR / "final_scores.json").write_text(
            json.dumps(
                {"acc": acc, "compl": compl, "num_rounds": len(all_round_scores)},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
