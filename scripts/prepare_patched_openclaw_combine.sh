#!/bin/bash
# Patches openclaw-combine/openclaw_combine_api_server.py's
# `_maybe_submit_ready_samples()` -- the single dispatch point shared by ALL
# three submission branches (OPD+RL combined / OPD-only / RL-only, see the
# class docstring's truth table) -- to drop turns flagged by
# prepare_patched_openclaw_opd.sh's `is_aborted`/`generated_while_paused`/
# `is_duplicate_user_retry` markers before they reach ANY of the three
# branches.
#
# Why this file specifically (docs/issues_log.md 2026-08-13 entry): actual
# training imports `OpenClawCombineSelectAPIServer`, which subclasses
# `OpenClawCombineAPIServer` and overrides `_opd_evaluate()` but NOT
# `_maybe_submit_ready_samples()` -- that dispatch function lives entirely in
# this file (openclaw-combine/openclaw_combine_api_server.py), which
# previously had NO patch script at all. Patching only
# openclaw_opd_api_server.py or openclaw_combine_select_api_server.py (the
# two files that already had patch scripts) would have been silently
# invisible to real training -- the flags would be written into turn_data,
# but nothing would ever read them at the point that decides whether to
# submit to Megatron.
#
# Root cause being fixed: 408/503 timeouts on the Student <-> OpenClaw
# gateway link cause two categories of turns to still reach the training
# queue even though the environment that produced them was known-unreliable:
#   A. SGLang itself aborted the in-flight generation (pause_generation
#      cutting off a request mid-flight for a weight sync) -- finish_reason
#      == "abort", content is typically a truncated fragment.
#   B. The generation completed normally, but by the time its result was
#      written to turn_data, submission had already been paused for a
#      training step -- the environment that produced it was mid-pause,
#      even though nothing about the generation itself looks wrong.
# Confirmed via real run data (P11/P20: fr=abort turns submitted as OPD+RL
# with reward=+1.0; P17: 15 turns from a 408-retry-recovery window submitted
# as OPD+RL/RL with mixed +1/-1) that BOTH branches (_submit_turn_sample and
# _submit_rl_turn_sample) already receive this kind of polluted data --
# gating only the OPD hint-acceptance path (as an earlier draft of this fix
# considered) would leave the RL-only branch completely unprotected.
#
# D. Student mechanically re-POSTs the exact same instruction after a
#    408/503 (send_to_openclaw()'s retry loop does not reword the message),
#    causing that instruction to reappear as a turn's next_state later in
#    the same session. Real data (run 20260813_094000, "P17") corrected an
#    earlier, broader hypothesis here: it is NOT true that "the whole
#    embedded run becomes untrustworthy once a 408 happens somewhere in it"
#    -- P17's own successful write (+1) happened on a retry, and turns
#    before the 408 (read/rewrite/legitimately-failed edits, all with a
#    real next_state) are exactly as trustworthy as turns from a problem
#    that never saw a 408 at all. Only turns whose next_state duplicates an
#    earlier user message in the same session are actually degraded; see
#    prepare_patched_openclaw_opd.sh's `is_duplicate_user_retry` (detected
#    in `_fire_opd_task`, before the PRM judge even runs) for the detection
#    logic. Blacklisting an entire run_id would have silently discarded
#    P17's own successful write turn -- do not reintroduce that design.
#
# Explicitly NOT covered by this patch (see docs/issues_log.md 2026-08-13):
# "C" -- OpenClaw's gateway declaring a request aborted/timed out while
# SGLang's own generation for that exact call actually completed normally
# (finish_reason=stop) -- is a real, independent phenomenon (a race between
# the gateway-level and SGLang-level completion signals) but OPD has no
# local visibility into the gateway's own timeout decision, so catching it
# would require a new OpenClaw plugin hook (agent_end/model_call_ended)
# notifying OPD out-of-band. This run's only sampled instance of "C" was
# entangled with the tool-misuse episode below and is not clean evidence for
# it, so it is deliberately deferred rather than guessed at.
#
# Also explicitly NOT covered: a third, unrelated failure mode found in the
# same investigation -- the policy misusing `edit`'s exact-substring-match
# semantics (anchoring on a bare "\n" instead of real surrounding context),
# repeatedly resubmitting near-identical failing edits, and OpenClaw's own
# reply still showing a stale "Edit failed" warning after a later write
# succeeds. This is a tool-competence problem, not a timeout/abort/retry
# problem -- it needs its own rule (a "wave 3", alongside existing rules
# 1-5), not a slot in this timeout-focused patch.
#
# Official openclaw-combine/ directory is left untouched; this writes a
# patched copy to DEST_DIR, and the caller must prepend DEST_DIR to
# PYTHONPATH ahead of openclaw-combine/ so `import openclaw_combine_api_server`
# resolves to the patched copy (see run_openclaw_topk_select_modelfactory.sh's
# PATCHED_COMBINE_DIR handling -- must be wired in alongside the existing
# PATCHED_OPD_DIR/PATCHED_COMBINE_SELECT_DIR, both in the launcher's
# PYTHONPATH line AND in every train_*.sh caller that sets these env vars).
set -euo pipefail

REPO_ROOT=${1:?usage: prepare_patched_openclaw_combine.sh <repo_root> <dest_dir>}
DEST_DIR=${2:?usage: prepare_patched_openclaw_combine.sh <repo_root> <dest_dir>}
SRC="${REPO_ROOT}/openclaw-combine/openclaw_combine_api_server.py"
DEST="${DEST_DIR}/openclaw_combine_api_server.py"

if [ ! -f "${SRC}" ]; then
    echo "错误：找不到官方文件 ${SRC}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

python3 - "${SRC}" "${DEST}" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(src_path, encoding="utf-8").read()

old_loop_head = (
    '        for turn_num in sorted(list(pending.keys())):\n'
    '            td = pending[turn_num]\n'
    '            task = prm_tasks.get(turn_num)\n'
    '\n'
    '            if task is None:\n'
)
if text.count(old_loop_head) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the "
        f"_maybe_submit_ready_samples loop head in {src_path}, found "
        f"{text.count(old_loop_head)} (official file may have changed "
        "upstream -- update this patch)"
    )
new_loop_head = (
    '        for turn_num in sorted(list(pending.keys())):\n'
    '            td = pending[turn_num]\n'
    '            task = prm_tasks.get(turn_num)\n'
    '\n'
    '            # --- openclaw-rl-degraded-turn-drop (temporary, safe to remove) ---\n'
    '            # See prepare_patched_openclaw_opd.sh for where these three flags\n'
    '            # are set. Checked before the task-readiness logic below so a\n'
    '            # degraded turn is dropped as soon as it is seen, whether or not\n'
    '            # its PRM task has finished yet -- and BEFORE the\n'
    '            # opd_accepted/has_valid_rl dispatch below, so it cannot reach\n'
    '            # _submit_turn_sample (OPD+RL or OPD-only) OR _submit_rl_turn_sample\n'
    '            # (RL-only) -- both are gated by this single shared loop, which is\n'
    '            # the point of patching this file specifically (see this script\'s\n'
    '            # top-of-file comment). Note is_duplicate_user_retry turns normally\n'
    '            # never reach this loop at all (_fire_opd_task returns before\n'
    '            # creating a PRM task, so `task` stays None and nothing gets\n'
    '            # added to pending for them) -- this branch is a backstop in case\n'
    '            # the flag ever ends up on a turn that did get a task.\n'
    '            if td.get("is_aborted") or td.get("generated_while_paused") or td.get("is_duplicate_user_retry"):\n'
    '                pending.pop(turn_num, None)\n'
    '                prm_tasks.pop(turn_num, None)\n'
    '                if task is not None:\n'
    '                    task.cancel()\n'
    '                if td.get("is_aborted"):\n'
    '                    reason = (\n'
    '                        "is_aborted (SGLang finish_reason=abort, generation was "\n'
    '                        "cut off mid-flight)"\n'
    '                    )\n'
    '                elif td.get("generated_while_paused"):\n'
    '                    reason = (\n'
    '                        "generated_while_paused (submission was disabled when "\n'
    '                        "this turn finished generating)"\n'
    '                    )\n'
    '                else:\n'
    '                    reason = (\n'
    '                        "is_duplicate_user_retry (next_state repeats an earlier "\n'
    '                        "user message in this session)"\n'
    '                    )\n'
    '                logger.info(\n'
    '                    "[openclaw-rl-degraded-turn-drop] session=%s turn=%d dropped "\n'
    '                    "(%s) -- not submitted to OPD or RL",\n'
    '                    session_id, turn_num, reason,\n'
    '                )\n'
    '                continue\n'
    '\n'
    '            if task is None:\n'
)
text = text.replace(old_loop_head, new_loop_head, 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched (dispatch-time drop of is_aborted/generated_while_paused/is_duplicate_user_retry turns, gates both OPD and RL submission paths) -> {dest_path}")
PY

echo "已生成 openclaw_combine_api_server.py 补丁: ${DEST_DIR}/openclaw_combine_api_server.py（_maybe_submit_ready_samples 拦截 is_aborted/generated_while_paused/is_duplicate_user_retry，OPD+RL 两条提交路径一起挡住，见 docs/issues_log.md 2026-08-13 条目）"
