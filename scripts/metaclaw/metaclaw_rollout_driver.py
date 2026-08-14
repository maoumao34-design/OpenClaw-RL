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

# Load-bearing prefix -- see module docstring and
# prepare_patched_openclaw_opd.sh's _METACLAW_SESSION_RE.
_SESSION_ID_PREFIX = "metaclaw-"


async def _send_verdict_turn(
    client: httpx.AsyncClient,
    session_id: str,
    eval_score: float,
    hint: str,
    session_done: bool,
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
    """
    verdict_payload = json.dumps(
        {"metaclaw_verdict": True, "eval_score": eval_score, "hint": hint},
        ensure_ascii=False,
    )
    try:
        await client.post(
            PROXY_URL,
            json={
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
        )
    except Exception as e:
        logger.warning(
            "[MetaClawRollout] session=%s verdict submission failed: %s", session_id, e
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
) -> dict[str, Any]:
    """Run one round via the real `openclaw agent` CLI.

    Mirrors src.infer.infer_cmd._run_group's per-round logic (agent
    invocation + inline scoring), simplified for live training rather than
    resumable batch evaluation -- no infer_result.json files, no
    resume-skip.
    """
    rc, stdout, stderr = await _run_openclaw_agent(
        session_id=session_id,
        message=query,
        openclaw_config_path=openclaw_config_path,
        openclaw_state_dir=openclaw_state_dir,
        project_root=project_root,
        gateway_port=gateway_port,
        timeout=round_timeout,
    )
    if rc != 0:
        logger.warning(
            "[MetaClawRollout] session=%s round=%s openclaw agent failed (rc=%d): %s",
            session_id, round_record["id"], rc, stderr[:500],
        )
        answer_text = ""
    else:
        answer_text = stdout

    inline_score = _compute_inline_score(round_record, answer_text, workspace_path)
    passed = inline_score.get("passed", False)
    logger.info(
        f"{_GREEN}[MetaClawRollout] session=%s round=%s passed=%s{_RESET}",
        session_id, round_record["id"], passed,
    )
    return inline_score


async def run_day(
    test: dict[str, Any],
    all_tests: dict[str, Any],
    project_root: Path,
) -> None:
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

                    inline_score = await _run_round(
                        session_id=session_id,
                        round_record=round_record,
                        query=query,
                        openclaw_config_path=openclaw_json_path,
                        openclaw_state_dir=work_openclaw_state_dir,
                        project_root=project_root,
                        workspace_path=workspace_copy,
                        gateway_port=gateway_port,
                        round_timeout=None,
                    )

                    passed = inline_score.get("passed", False)
                    eval_score = 1.0 if passed else -1.0
                    hint = "" if passed else _build_opd_hint(round_record, inline_score)

                    is_last_round = idx == len(rounds) - 1
                    await _send_verdict_turn(
                        client, session_id, eval_score, hint, session_done=is_last_round,
                    )

                    prev_inline_score = inline_score
                    prev_round_record = round_record
    finally:
        if gateway_proc.returncode is None:
            gateway_proc.terminate()
            await gateway_proc.wait()

    logger.info(f"{_GREEN}[MetaClawRollout] day=%s done{_RESET}", test_id)


async def main() -> None:
    all_tests_raw = os.environ.get("METACLAW_ALL_TESTS_JSON")
    if not all_tests_raw:
        raise RuntimeError("METACLAW_ALL_TESTS_JSON env var must point at all_tests.json")
    all_tests_path = resolve_path(all_tests_raw)
    all_tests = json.loads(all_tests_path.read_text(encoding="utf-8"))
    test_list = all_tests.get("test", [])
    project_root = get_project_root()

    logger.info(
        f"{_YELLOW}[MetaClawRollout] %d day(s) loaded from %s, concurrency=1, "
        f"strict order{_RESET}",
        len(test_list), all_tests_path,
    )

    # concurrency=1, strict day01 -> day30 order -- see
    # docs/metaclaw_migration_plan.md "查证记录（二）" 第 2 条: this is the
    # property that preserves the "online update while running through the
    # day-stream" design, mirroring how MetaClaw's own harness forces
    # workers=1 whenever scene_per_train is set.
    for test in test_list:
        await run_day(test, all_tests, project_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
