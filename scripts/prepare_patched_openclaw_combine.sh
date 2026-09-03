#!/bin/bash
# Patches openclaw-combine/openclaw_combine_api_server.py's
# `_maybe_submit_ready_samples()` -- the single dispatch point shared by ALL
# three submission branches (OPD+RL combined / OPD-only / RL-only, see the
# class docstring's truth table) -- to drop turns flagged by
# prepare_patched_openclaw_opd.sh's `is_aborted`/`generated_while_paused`/
# `is_duplicate_user_retry` markers before they reach ANY of the three
# branches, AND (temporary diagnostic experiment, 2026-08-13, see
# docs/issues_log.md) turns flagged `skip_forced_negative_override` by
# prepare_patched_openclaw_combine_select.sh's `_opd_evaluate()` -- turns
# where the PRM originally scored +1 but one of the existing eval_score
# overrides (invalid-tool-use-penalty/truncation-penalty) forced it to -1.
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

# ---------------------------------------------------------------------
# openclaw-rl-skip-forced-negative-override (2026-08-13, temporary
# diagnostic experiment, see docs/issues_log.md 2026-08-13 entry). This is
# a SECOND, independent check point in the same function -- deliberately
# NOT merged into the is_aborted/generated_while_paused/is_duplicate_user_retry
# check above, because that one reads `td` (turn_data, known before the PRM
# judge ever runs); this one reads `opd_result` (only known after
# `task.result()` succeeds -- see prepare_patched_openclaw_combine_select.sh
# for where `skip_forced_negative_override` is computed inside
# _opd_evaluate()). Placed BEFORE `eval_score = opd_result.get("eval_score")`
# and the `_eval_scores.append(eval_score)` call that follows it -- if this
# ran after eval_scores.append, the -1 would still get recorded (polluting
# eval-mode/wandb bookkeeping) even though the sample itself gets skipped,
# making it impossible to tell from that bookkeeping whether the skip
# actually took effect.
# ---------------------------------------------------------------------
skip_forced_neg_old = (
    '                if self._eval_mode:\n'
    '                    with self._eval_scores_lock:\n'
    '                        self._eval_scores.append(0.0)\n'
    '                continue\n'
    '\n'
    '            eval_score = opd_result.get("eval_score")\n'
)
if text.count(skip_forced_neg_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the task.result() "
        f"failure-handling + eval_score assignment block in {src_path}, "
        f"found {text.count(skip_forced_neg_old)} (official file may have "
        "changed upstream -- update this patch)"
    )
skip_forced_neg_new = (
    '                if self._eval_mode:\n'
    '                    with self._eval_scores_lock:\n'
    '                        self._eval_scores.append(0.0)\n'
    '                continue\n'
    '\n'
    '            # --- openclaw-rl-skip-forced-negative-override (temporary, safe to remove) ---\n'
    '            if opd_result.get("skip_forced_negative_override"):\n'
    '                continue\n'
    '\n'
    '            eval_score = opd_result.get("eval_score")\n'
)
text = text.replace(skip_forced_neg_old, skip_forced_neg_new, 1)

# ---------------------------------------------------------------------
# openclaw-rl-metaclaw-train-until-day (temporary, safe to remove) -- see
# docs/metaclaw_migration_plan.md "方案：可调 K 天训练窗口 + 冻结评测剩余天数".
# This is the ONE enforcement point for the freeze flag set by
# prepare_patched_openclaw_opd.sh's chat_completions patch
# (self._metaclaw_training_frozen). Placed here, not in openclaw_opd_api_
# server.py or openclaw_combine_select_api_server.py, for the same reason
# this whole file exists (see this script's top-of-file comment): actual
# training imports OpenClawCombineSelectAPIServer, which subclasses this
# class and overrides _opd_evaluate() but NOT _maybe_submit_ready_samples()
# -- this is the ONLY place that actually calls _submit_turn_sample/
# _submit_rl_turn_sample, for BOTH the OPD and RL-only paths. Setting the
# flag anywhere else without also gating here would repeat the exact
# "flag written, nobody reads it" mistake this file was created to avoid.
#
# Deliberately a SEPARATE check from skip_forced_negative_override above
# (not merged into it) -- this one is a global switch unrelated to any
# per-turn eval_score override, and unlike the is_aborted/generated_while_
# paused/is_duplicate_user_retry check further up (which reads per-turn
# `td` flags), this reads process-wide server state, so it belongs at this
# late, unconditional gate rather than the earlier per-turn one.
#
# NOT retroactive: turns already in `pending`/`prm_tasks` for THIS session
# when the flag flips mid-day are not force-dropped by this check alone --
# they still get evaluated and hit this same gate when their own task
# completes, at which point they ARE dropped (frozen is checked before
# either submit call, not just for turns arriving after the flag flips).
# The remaining edge case (a handful of dayK's own trailing async step-
# judge tasks resolving after the freeze signal for dayK+1 has already
# been sent) is a known, accepted small-scale race -- see the migration
# doc's "dayK 尾部竞态" section: it can only ever DROP a few of dayK's own
# tail samples, never misattribute or corrupt data across the day
# boundary, and is documented rather than engineered away.
# ---------------------------------------------------------------------
train_until_day_gate_old = (
    '            # --- openclaw-rl-skip-forced-negative-override (temporary, safe to remove) ---\n'
    '            if opd_result.get("skip_forced_negative_override"):\n'
    '                continue\n'
    '\n'
    '            eval_score = opd_result.get("eval_score")\n'
)
if text.count(train_until_day_gate_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the "
        f"skip_forced_negative_override + eval_score assignment block in "
        f"{src_path}, found {text.count(train_until_day_gate_old)} "
        "(official file may have changed upstream -- update this patch)"
    )
train_until_day_gate_new = (
    '            # --- openclaw-rl-skip-forced-negative-override (temporary, safe to remove) ---\n'
    '            if opd_result.get("skip_forced_negative_override"):\n'
    '                continue\n'
    '\n'
    '            # --- openclaw-rl-metaclaw-train-until-day (temporary, safe to remove) ---\n'
    '            if getattr(self, "_metaclaw_training_frozen", False):\n'
    '                logger.info(\n'
    '                    "[metaclaw-freeze] session=%s turn=%d dropped (training frozen) "\n'
    '                    "-- not submitted to OPD or RL",\n'
    '                    session_id, turn_num,\n'
    '                )\n'
    '                continue\n'
    '\n'
    '            eval_score = opd_result.get("eval_score")\n'
)
text = text.replace(train_until_day_gate_old, train_until_day_gate_new, 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched (dispatch-time drop of is_aborted/generated_while_paused/is_duplicate_user_retry/skip_forced_negative_override/metaclaw_training_frozen turns, gates both OPD and RL submission paths) -> {dest_path}")
PY

echo "已生成 openclaw_combine_api_server.py 补丁: ${DEST_DIR}/openclaw_combine_api_server.py（_maybe_submit_ready_samples 拦截 is_aborted/generated_while_paused/is_duplicate_user_retry/skip_forced_negative_override，OPD+RL 两条提交路径一起挡住，见 docs/issues_log.md 2026-08-13 条目）"
