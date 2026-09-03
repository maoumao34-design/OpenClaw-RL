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

# ---------------------------------------------------------------------
# Source of the 1/N advantage scaler, written out verbatim below. Kept as a
# literal here (rather than a checked-in .py under scripts/) so the whole
# proxy-side toolchain stays in one place and the file lands in DEST_DIR,
# which the launcher already prepends to the training process's PYTHONPATH.
# ---------------------------------------------------------------------
ROUND_SCALE_SRC = '''"""Per-round advantage scaling for the MetaClaw migration.

Registered via slime's --custom-reward-post-process-path, which short-circuits
at the top of RolloutManager._post_process_rewards. Returns (raw_rewards,
rewards); `rewards` is what becomes the advantage, `raw_rewards` is only logged.

What it does
------------
Divides each sample's advantage by the number of turns in the round it came
from (openclaw_combine_api_server.py stamps that onto sample.metadata as
`metaclaw_round_turns`). Samples without that key are passed through unchanged,
so the Personal Agent Track behaves exactly as it does today.

Why
---
slime's sum_of_sample_mean (backends/megatron_utils/cp_utils.py:70) averages
within a sample and then sums ACROSS samples, so every sample weighs the same
regardless of length. A MetaClaw round emits one sample per turn, so a round
that took 20 turns contributes twenty times the gradient of a round answered in
one -- verbosity earns a 20x bonus that has nothing to do with being right.
That weighting is what let day06-r7 (186 turns) flush 186 negatives into a
single batch in metaclaw_migration_20260902_094458 and dominate it.

Dividing by N makes each round contribute exactly one round's worth. Combined
with emitting a round as one group (so --rollout-batch-size counts complete
rounds), a batch is now a clean average over that many rounds, and no single
round can fill one on its own.

What it does NOT do
-------------------
It scales magnitude, not sign. With --disable-rewards-normalization and
n_samples_per_prompt=1 there is still no within-group comparison, so a batch in
which every round failed is still uniformly negative -- just at 1/N strength.
Fixing that needs n_samples_per_prompt > 1, not this hook.
"""

import logging

logger = logging.getLogger(__name__)


def _is_dummy(sample):
    """slime injects placeholder samples when a batch is smaller than dp_size.

    They carry reward 0.0 and loss_mask [0], so they never contribute gradient,
    but giving them an explicit zero keeps the logged counters honest.
    _drop_removed_samples has already run by this point, so remove_sample can
    only be true for these.
    """
    meta = getattr(sample, "metadata", None) or {}
    return bool(meta.get("dummy_removed_sample")) or bool(
        getattr(sample, "remove_sample", False)
    )


def metaclaw_round_scale(args, samples, **kwargs):
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    advantages = []
    scaled = 0
    passthrough = 0
    turn_counts = []

    for sample, reward in zip(samples, raw_rewards):
        if _is_dummy(sample):
            advantages.append(0.0)
            continue
        meta = getattr(sample, "metadata", None) or {}
        n_turns = meta.get("metaclaw_round_turns")
        if isinstance(n_turns, int) and n_turns > 0:
            advantages.append(reward / n_turns)
            scaled += 1
            turn_counts.append(n_turns)
        else:
            # Not a MetaClaw round sample. Pass through unchanged: this hook
            # short-circuits _post_process_rewards entirely, so the fallback
            # has to reproduce the default behaviour under
            # --disable-rewards-normalization, which is advantage = reward.
            advantages.append(reward)
            passthrough += 1

    if turn_counts:
        distinct_rounds = len(set(turn_counts))
        logger.info(
            "[metaclaw-round-scale] %d/%d sample(s) scaled by 1/turns "
            "(turns min=%d max=%d), %d passed through; "
            "advantage min=%.4f max=%.4f",
            scaled,
            len(samples),
            min(turn_counts),
            max(turn_counts),
            passthrough,
            min(advantages),
            max(advantages),
        )
        if distinct_rounds == 1 and scaled == len(turn_counts) and passthrough == 0:
            logger.warning(
                "[metaclaw-round-scale] every sample in this batch reports the "
                "same turn count (%d) -- if that is one round filling the whole "
                "batch, the round-as-one-group emission is not working",
                turn_counts[0],
            )
    else:
        logger.info(
            "[metaclaw-round-scale] no MetaClaw round samples in this batch of "
            "%d; all rewards passed through unchanged",
            len(samples),
        )

    return raw_rewards, advantages
'''

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

# ---------------------------------------------------------------------
# openclaw-rl-metaclaw-round-group (2026-09-03): a MetaClaw round is emitted
# as ONE group of N per-turn samples, all carrying the round's deterministic
# checker verdict, and metadata_ saying how many turns the round had so the
# advantage hook can divide by N.
#
# What this buys, and why it is not just "outcome mode again":
#   - Every round contributes exactly one round's worth of gradient. Without
#     the 1/N scaling, slime's sum_of_sample_mean weighs every SAMPLE equally,
#     so a 20-turn round outweighs a 1-turn round twentyfold -- verbosity earns
#     a 20x bonus unrelated to being right. That weighting is what let day06-r7
#     (186 turns) flush 186 negatives into one batch in 20260902_094458.
#   - Because _drain_output_queue counts GROUPS, a batch is now always exactly
#     `rollout_batch_size` COMPLETE rounds. A single failed round can no longer
#     fill a batch on its own: an all-negative batch requires every round in it
#     to have failed.
#   - Samples stay per-turn, so each one is exactly the prompt the model saw
#     and the response it produced. Nothing is reconstructed. See the
#     openclaw-rl-metaclaw-round-group note in prepare_patched_openclaw_opd.sh
#     for why a single flat trajectory sample is not constructible at all under
#     OpenClaw's dropReasoningFromHistory behaviour.
#
# The group is queued once, when the verdict resolves, from
# _metaclaw_submit_round below -- never incrementally, because
# _drain_output_queue does `completed_groups[group_id] = group` (an overwrite,
# not an append), so a group put twice would lose its earlier members.
# ---------------------------------------------------------------------
submit_collect_old = (
    '        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))\n'
)
if text.count(submit_collect_old) != 2:
    raise SystemExit(
        f"patch failed: expected exactly 2 output_queue.put calls in {src_path} "
        f"(_submit_turn_sample and _submit_rl_turn_sample), found "
        f"{text.count(submit_collect_old)} (official file may have changed "
        "upstream -- update this patch)"
    )
submit_collect_new = (
    '        # --- openclaw-rl-metaclaw-round-group ---\n'
    '        # When the caller is assembling a whole round, hand the sample back\n'
    '        # instead of queueing it: the round is queued once, as one group.\n'
    '        _mc_collect = turn_data.get("metaclaw_round_collect")\n'
    '        if _mc_collect is not None:\n'
    '            sample.group_index = turn_data["metaclaw_round_group_index"]\n'
    '            sample.metadata = {\n'
    '                **(getattr(sample, "metadata", None) or {}),\n'
    '                "metaclaw_round_id": session_id,\n'
    '                "metaclaw_round_turns": turn_data["metaclaw_round_turns"],\n'
    '            }\n'
    '            _mc_collect.append(sample)\n'
    '            return\n'
    '        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))\n'
)
text = text.replace(submit_collect_old, submit_collect_new, 2)

# The round assembler itself. Inserted ahead of _maybe_submit_ready_samples so
# it reads in dispatch order.
round_submit_anchor = (
    '    def _maybe_submit_ready_samples(\n'
)
if text.count(round_submit_anchor) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 _maybe_submit_ready_samples "
        f"definition in {src_path}, found {text.count(round_submit_anchor)} "
        "(official file may have changed upstream -- update this patch)"
    )
round_submit_new = (
    '    async def _metaclaw_submit_round(\n'
    '        self, session_id: str, turns: list, verdict_td: dict,\n'
    '        opd_result: dict, reward: float,\n'
    '    ):\n'
    '        """Submit one MetaClaw round as a single group of per-turn samples.\n'
    '\n'
    '        `turns` are the round\'s held intermediate turns (RL-only: they never\n'
    '        got a judge, by design -- see the fire gate in\n'
    '        prepare_patched_openclaw_opd.sh). `verdict_td` is the final turn,\n'
    '        which additionally carries the OPD hint when one was accepted.\n'
    '        Every sample gets the same reward: the round\'s checker verdict.\n'
    '        """\n'
    '        collect: list = []\n'
    '        group_index = next(self._group_counter)\n'
    '        all_tds = list(turns) + [verdict_td]\n'
    '        n_turns = len(all_tds)\n'
    '        for _td in all_tds:\n'
    '            _td["metaclaw_round_collect"] = collect\n'
    '            _td["metaclaw_round_group_index"] = group_index\n'
    '            _td["metaclaw_round_turns"] = n_turns\n'
    '\n'
    '        for _td in turns:\n'
    '            await self._submit_rl_turn_sample(_td, session_id, reward)\n'
    '        if opd_result.get("accepted"):\n'
    '            await self._submit_turn_sample(\n'
    '                verdict_td, session_id, opd_result, reward=reward,\n'
    '            )\n'
    '        else:\n'
    '            await self._submit_rl_turn_sample(verdict_td, session_id, reward)\n'
    '\n'
    '        if not collect:\n'
    '            logger.warning(\n'
    '                "[openclaw-rl-metaclaw-round-group] session=%s produced no "\n'
    '                "samples from %d turn(s) -- nothing queued",\n'
    '                session_id, n_turns,\n'
    '            )\n'
    '            return\n'
    '        logger.info(\n'
    '            "[openclaw-rl-metaclaw-round-group] session=%s queued group=%d "\n'
    '            "with %d sample(s) from %d turn(s), reward=%.1f "\n'
    '            "(advantage will be scaled by 1/%d downstream)",\n'
    '            session_id, group_index, len(collect), n_turns, reward, n_turns,\n'
    '        )\n'
    '        await asyncio.to_thread(self.output_queue.put, (group_index, collect))\n'
    '\n'
    '    def _maybe_submit_ready_samples(\n'
)
text = text.replace(round_submit_anchor, round_submit_new, 1)

# Dispatch: when the verdict resolves, take over the whole round instead of
# letting the per-turn branches below run. Placed immediately after
# `has_valid_rl` is computed so it can reuse that check, and before the three
# official submission branches.
round_dispatch_old = (
    '            opd_accepted = opd_result.get("accepted")\n'
    '            has_valid_rl = self._is_valid_rl_score(eval_score)\n'
    '\n'
)
if text.count(round_dispatch_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 opd_accepted/has_valid_rl block in "
        f"{src_path}, found {text.count(round_dispatch_old)} "
        "(official file may have changed upstream -- update this patch)"
    )
round_dispatch_new = (
    '            opd_accepted = opd_result.get("accepted")\n'
    '            has_valid_rl = self._is_valid_rl_score(eval_score)\n'
    '\n'
    '            # --- openclaw-rl-metaclaw-round-group (2026-09-03) ---\n'
    '            if opd_result.get("metaclaw_verdict"):\n'
    '                # The held intermediate turns are still in `pending`: they\n'
    '                # have no PRM task, so the `task is None` branch above skips\n'
    '                # them without popping (force_drop is False on this path).\n'
    '                _mc_turns = [\n'
    '                    pending[t] for t in sorted(pending.keys()) if t < turn_num\n'
    '                ]\n'
    '                for _t in [t for t in list(pending.keys()) if t < turn_num]:\n'
    '                    pending.pop(_t, None)\n'
    '                if not has_valid_rl:\n'
    '                    # No usable outcome for this round, so there is nothing\n'
    '                    # for the held turns to inherit. Dropping is the only\n'
    '                    # honest option -- the alternative would be inventing a\n'
    '                    # reward for turns whose round was never scored.\n'
    '                    logger.warning(\n'
    '                        "[openclaw-rl-metaclaw-round-group] session=%s verdict "\n'
    '                        "turn=%d carries no valid outcome (%r) -- dropping the "\n'
    '                        "round and its %d held turn(s)",\n'
    '                        session_id, turn_num, eval_score, len(_mc_turns),\n'
    '                    )\n'
    '                    continue\n'
    '                self._safe_create_task(\n'
    '                    self._metaclaw_submit_round(\n'
    '                        session_id, _mc_turns, td, opd_result,\n'
    '                        reward=float(eval_score),\n'
    '                    )\n'
    '                )\n'
    '                continue\n'
    '\n'
)
text = text.replace(round_dispatch_old, round_dispatch_new, 1)

import os as _emit_os
round_scale_path = _emit_os.path.join(_emit_os.path.dirname(dest_path), "metaclaw_round_scale.py")
with open(round_scale_path, "w", encoding="utf-8") as f:
    f.write(ROUND_SCALE_SRC)
print(f"wrote per-round advantage scaler -> {round_scale_path}")

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched (dispatch-time drop of is_aborted/generated_while_paused/is_duplicate_user_retry/skip_forced_negative_override/metaclaw_training_frozen turns, gates both OPD and RL submission paths) -> {dest_path}")
PY

echo "已生成 openclaw_combine_api_server.py 补丁: ${DEST_DIR}/openclaw_combine_api_server.py（_maybe_submit_ready_samples 拦截 is_aborted/generated_while_paused/is_duplicate_user_retry/skip_forced_negative_override，OPD+RL 两条提交路径一起挡住，见 docs/issues_log.md 2026-08-13 条目）"
