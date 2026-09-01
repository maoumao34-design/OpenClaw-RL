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
import os
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(src_path, encoding="utf-8").read()

# ---------------------------------------------------------------------
# Source of the batch-baseline reward post-processor, written out verbatim
# below. Kept as a literal here (rather than a checked-in .py under
# scripts/) so the whole proxy-side toolchain stays in one place and the
# file lands in DEST_DIR, which is already on the training process's
# PYTHONPATH.
# ---------------------------------------------------------------------
BATCH_BASELINE_SRC = '''"""Batch-level advantage baseline for the MetaClaw migration.

Registered via slime's --custom-reward-post-process-path, which short-circuits
at the top of RolloutManager._post_process_rewards. Returns (raw_rewards,
rewards); `rewards` is what becomes the advantage, `raw_rewards` is only
logged.

Why this exists
---------------
slime's get_grpo_returns broadcasts a sample's scalar reward across all of its
tokens, and GRPO's own normalization works per group_index. Our proxy assigns
every sample its own group_index and the training script sets
--n-samples-per-prompt 1, so each group holds exactly one sample and
normalizing it is identically zero -- which is why
--disable-rewards-normalization is mandatory here rather than a tunable. The
consequence is that the advantage degenerates into the raw reward: a batch
where every sample is -1 pushes down every token the model just produced with
nothing pushed up, which is how metaclaw_migration_20260831_154301 diverged
(grad_norm 2543.9, policy drift 21.94).

Raising --n-samples-per-prompt the way toolcall-rl does (8 attempts at one
math problem, normalized within the group) is not available to us: one sample
here means running the real agent against a real workspace that later rounds
read from, in a strict day01->day30 order, so eight samples would need eight
workspaces with no principled way to choose which one the day continues from.

MetaClaw itself faces exactly this task shape and answers it with a batch
baseline instead of within-group comparison -- metaclaw/data_formatter.py::
compute_advantages centres and scales over the whole batch. This module is
that, applied through slime's hook. It touches neither group_index nor the
output queue.

What it does and does not fix
-----------------------------
It removes the "all-negative batch means uniform suppression" failure mode: an
all-identical batch now yields zero advantage (that batch simply does not
update) instead of pushing everything down. It does NOT discriminate within a
sign when rewards are binary -- centring +-1 moves the zero point and rescales,
but every +1 still receives the same positive advantage. Rewards would have to
be continuous for that.
"""

import logging

logger = logging.getLogger(__name__)


def _is_dummy(sample):
    """slime injects placeholder samples when a batch is smaller than dp_size.

    They carry reward 0.0 and loss_mask [0], so they never contribute gradient
    themselves -- but left in the statistics their zeros drag the batch mean
    and distort every real sample's advantage. _drop_removed_samples has
    already run by this point, so remove_sample can only be true for these.
    """
    meta = getattr(sample, "metadata", None) or {}
    return bool(meta.get("dummy_removed_sample")) or bool(
        getattr(sample, "remove_sample", False)
    )


def metaclaw_batch_baseline(args, samples, **kwargs):
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    real_indices = [i for i, s in enumerate(samples) if not _is_dummy(s)]

    if len(real_indices) < 2:
        logger.warning(
            "[metaclaw-batch-baseline] only %d real sample(s) in a batch of %d -- "
            "no baseline can be formed, emitting zero advantage (this batch will "
            "not update the policy)",
            len(real_indices),
            len(samples),
        )
        return raw_rewards, [0.0] * len(raw_rewards)

    values = [raw_rewards[i] for i in real_indices]
    mean_r = sum(values) / len(values)
    std_r = (sum((v - mean_r) ** 2 for v in values) / len(values)) ** 0.5

    advantages = [0.0] * len(raw_rewards)
    for i in real_indices:
        advantages[i] = (raw_rewards[i] - mean_r) / (std_r + 1e-8)

    # Observability is not optional here: the baseline changes the advantage
    # only, never the reward, so the reward-side counters this run is compared
    # against (batch reward, +/- sample counts) look identical whether or not
    # this function is doing anything. These lines are the only evidence that
    # it ran and had an effect.
    real_advs = [advantages[i] for i in real_indices]
    n_pos = sum(1 for a in real_advs if a > 0)
    n_neg = sum(1 for a in real_advs if a < 0)
    if std_r <= 1e-8:
        logger.warning(
            "[metaclaw-batch-baseline] batch of %d real sample(s) has zero reward "
            "variance (all %.3f) -- advantage is all zero, this batch will not "
            "update the policy. That is the intended degenerate case: previously "
            "such a batch suppressed every token uniformly.",
            len(real_indices),
            mean_r,
        )
    else:
        logger.info(
            "[metaclaw-batch-baseline] n_real=%d/%d reward_mean=%.4f reward_std=%.4f "
            "-> advantage min=%.4f max=%.4f pos=%d neg=%d",
            len(real_indices),
            len(samples),
            mean_r,
            std_r,
            min(real_advs),
            max(real_advs),
            n_pos,
            n_neg,
        )

    return raw_rewards, advantages
'''

# ---------------------------------------------------------------------
# openclaw-rl-metaclaw-midround-reward (2026-08-31, temporary, safe to
# remove) -- see docs/metaclaw_migration_plan.md "方案：中间轮次改吃本轮
# 最终 checker 结果的消融实验".
#
# Ablation switch. Default "judge" keeps every existing behavior; only
# METACLAW_MIDROUND_REWARD=outcome activates any of the code below.
#
# Purpose: test whether the independent step judge's positive rewards on
# intermediate FC turns are the upstream cause of the thinking inflation
# that starts at day17 (see the diagnosis section in the migration doc).
# Under "outcome", intermediate turns are STILL submitted -- sample count
# is deliberately unchanged, because dropping them outright would drop
# ~72% of the training mix and confound with "less training is better",
# which the K=6 frozen run already demonstrated -- but their reward comes
# from the round's deterministic checker verdict instead of the judge.
# ---------------------------------------------------------------------
midround_import_old = "logger = logging.getLogger(__name__)\n"
if text.count(midround_import_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 'logger = logging.getLogger(__name__)' "
        f"in {src_path} to anchor the midround-reward env read, found "
        f"{text.count(midround_import_old)} (official file may have changed upstream)"
    )
midround_import_new = (
    "logger = logging.getLogger(__name__)\n"
    "\n"
    "# --- openclaw-rl-metaclaw-midround-reward (temporary, safe to remove) ---\n"
    "import os as _mr_os\n"
    "\n"
    "# judge   -- current behavior, intermediate turns keep the step judge's own +-1\n"
    "# outcome -- intermediate turns inherit the round's deterministic checker result\n"
    "#            verbatim. Tried in metaclaw_migration_20260831_154301 and it\n"
    "#            diverged: round pass rate is ~17% while the judge was giving ~69%\n"
    "#            +1, so replacing the judge wholesale collapsed the positive signal\n"
    "#            and batches went 0/16 from step 6 on. Kept selectable for\n"
    "#            comparison, but do not expect it to train.\n"
    "# blend   -- route B (2026-08-31): outcome supplies the direction, the judge\n"
    "#            score only nudges it. Mirrors toolcall-rl's\n"
    "#            `final_score = base_score + prm_step_coef * prm_step_mean`, where a\n"
    "#            misjudged step is diluted rather than becoming the entire reward.\n"
    "#            The point is that rewards stop being a flat +-1: the batch gets a\n"
    "#            spread of magnitudes instead of sixteen identical -1s. That matters\n"
    "#            because reward normalization cannot help here -- every sample is\n"
    "#            its own group of one (see openclaw_combine_api_server's\n"
    "#            group_index assignment), so normalizing is identically zero and\n"
    "#            --disable-rewards-normalization is mandatory, not a tunable.\n"
    "_METACLAW_MIDROUND_REWARD = _mr_os.environ.get(\n"
    "    \"METACLAW_MIDROUND_REWARD\", \"judge\"\n"
    ").strip().lower()\n"
    "\n"
    "# Weight on the judge's nudge under blend mode. toolcall-rl defaults its\n"
    "# equivalent (prm_step_coef) to 1.0, but its base_score and step scores live on\n"
    "# comparable scales there; here the outcome is a hard +-1, so a full-weight\n"
    "# judge score could flip the sign and undo the point of the mode. Default 0.3\n"
    "# keeps every blended reward strictly on the outcome's side of zero\n"
    "# (|0.3 * judge| < 1), making the judge a magnitude adjustment only.\n"
    "_METACLAW_MIDROUND_JUDGE_COEF = float(\n"
    "    _mr_os.environ.get(\"METACLAW_MIDROUND_JUDGE_COEF\", \"0.3\")\n"
    ")\n"
    "\n"
    "# Ceiling applied to a FAILED round's intermediate turns, mirroring\n"
    "# toolcall-rl's `min(-0.6, score + tool_call_reward)`. Note that is a ceiling,\n"
    "# not a floor: a failed sample can never score better than this. At the default\n"
    "# coefficient it is INERT -- -1 + 0.3*1 = -0.7 is already below -0.6, so failed\n"
    "# rewards land in [-1.3, -0.7] on their own. It exists as the sign guard for\n"
    "# larger coefficients: at coef=1.0 (toolcall-rl's own default) an unclamped\n"
    "# failed turn with a +1 judge would reach 0.0 and lose its sign entirely.\n"
    "# Set to 0 to disable.\n"
    "_METACLAW_MIDROUND_FAIL_CEILING = float(\n"
    "    _mr_os.environ.get(\"METACLAW_MIDROUND_FAIL_CEILING\", \"-0.6\")\n"
    ")\n"
    "\n"
    "\n"
    "def _metaclaw_blend_reward(outcome, judge_score, hard_negative):\n"
    "    \"\"\"Route B reward composition (2026-08-31).\n"
    "\n"
    "    outcome supplies the sign, the judge score only adjusts magnitude. A turn\n"
    "    condemned by a forced-(-1) rule short-circuits to -1.0 regardless -- the\n"
    "    same hard-negative precedence the other modes honor, since blending a\n"
    "    known-invalid tool call back toward zero would defeat the override.\n"
    "\n"
    "    On a FAILED round the result is additionally capped at\n"
    "    _METACLAW_MIDROUND_FAIL_CEILING so a nudge can never make a failure look\n"
    "    near-neutral; on a SUCCEEDED round no cap applies, since there the judge\n"
    "    nudge is what distinguishes a clean trajectory from a wasteful one.\n"
    "    \"\"\"\n"
    "    if hard_negative:\n"
    "        return -1.0\n"
    "    _base = float(outcome)\n"
    "    _judge = 0.0 if judge_score is None else float(judge_score)\n"
    "    _blended = _base + _METACLAW_MIDROUND_JUDGE_COEF * _judge\n"
    "    if _base < 0 and _METACLAW_MIDROUND_FAIL_CEILING:\n"
    "        _blended = min(_METACLAW_MIDROUND_FAIL_CEILING, _blended)\n"
    "    return _blended\n"
)
text = text.replace(midround_import_old, midround_import_new, 1)

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
# openclaw-rl-metaclaw-midround-reward: verdict task FAILURE terminal state
# (2026-08-31, CLI review requirement).
#
# "verdict fired" must not be treated as "an outcome will definitely
# arrive". If the verdict's _opd_evaluate task raises, the official
# handler below logs and `continue`s -- the result dict never exists, so
# the metaclaw_verdict marker is never seen, no outcome is ever recorded,
# and force_drop refuses to discard held turns (because verdict_turn is
# set). Held samples would then sit in memory forever: never submitted,
# never cleaned up. Comparing turn_num against the recorded verdict_turn
# is the only way to recognize this case, since the exception carries no
# branch information.
# ---------------------------------------------------------------------
verdict_fail_old = (
    '            except Exception as e:\n'
    '                logger.warning(\n'
    '                    "[OpenClaw-Combine] evaluation task failed session=%s turn=%d: %s",\n'
    '                    session_id,\n'
    '                    turn_num,\n'
    '                    e,\n'
    '                )\n'
)
if text.count(verdict_fail_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 evaluation-task-failed except block "
        f"in {src_path}, found {text.count(verdict_fail_old)} "
        "(official file may have changed upstream -- update this patch)"
    )
verdict_fail_new = (
    '            except Exception as e:\n'
    '                logger.warning(\n'
    '                    "[OpenClaw-Combine] evaluation task failed session=%s turn=%d: %s",\n'
    '                    session_id,\n'
    '                    turn_num,\n'
    '                    e,\n'
    '                )\n'
    '                # --- openclaw-rl-metaclaw-midround-reward (temporary) ---\n'
    '                if _METACLAW_MIDROUND_REWARD in ("outcome", "blend"):\n'
    '                    _mr = self._metaclaw_round.get(session_id)\n'
    '                    if _mr is not None and _mr.get("verdict_turn") == turn_num:\n'
    '                        _dropped = len(_mr.get("held", {}))\n'
    '                        logger.warning(\n'
    '                            "[openclaw-rl-metaclaw-midround-reward] session=%s "\n'
    '                            "verdict task (turn=%d) FAILED -- no round outcome will "\n'
    '                            "ever arrive, discarding %d held intermediate turn(s) "\n'
    '                            "(terminal state: failed)",\n'
    '                            session_id, turn_num, _dropped,\n'
    '                        )\n'
    '                        # Tombstone, NOT pop (2026-08-31b fix, CLI review): the\n'
    '                        # verdict branch of _opd_evaluate returns without calling\n'
    '                        # any LLM, while a step-judge task makes prm_m LLM calls,\n'
    '                        # so the verdict routinely resolves BEFORE the round\'s own\n'
    '                        # intermediate judge tasks. Popping here would let those\n'
    '                        # late tasks re-create a fresh "pending" entry via\n'
    '                        # setdefault below and get held against an outcome that can\n'
    '                        # never arrive -- re-leaking the very turn_data (prompt/\n'
    '                        # response token lists) this branch is trying to release.\n'
    '                        # Keeping a terminal marker lets them be discarded on sight;\n'
    '                        # the entry itself is removed by the post-loop cleanup once\n'
    '                        # pending and prm_tasks have both drained.\n'
    '                        _mr["held"] = {}\n'
    '                        _mr["state"] = "failed"\n'
)
text = text.replace(verdict_fail_old, verdict_fail_new, 1)

# ---------------------------------------------------------------------
# openclaw-rl-metaclaw-midround-reward: hold / inherit / flush dispatch.
#
# Placed AFTER skip_forced_negative_override and the train-until-day
# freeze gate, and after the degraded-turn-drop check further up, so every
# existing drop rule (is_aborted / generated_while_paused /
# is_duplicate_user_retry / skip_forced_negative_override /
# metaclaw_training_frozen) still fires FIRST -- holding must never
# resurrect a sample the current rules would have discarded (CLI review
# requirement). Placed BEFORE the `eval_score = opd_result.get(...)` line
# and its self._eval_scores append, so an intermediate turn's judge score
# can never be recorded as the sample's actual training reward.
# ---------------------------------------------------------------------
midround_dispatch_old = (
    '            eval_score = opd_result.get("eval_score")\n'
    '            if self._eval_mode and eval_score is not None:\n'
)
if text.count(midround_dispatch_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 eval_score/_eval_mode block in "
        f"{src_path}, found {text.count(midround_dispatch_old)} "
        "(official file may have changed upstream -- update this patch)"
    )
midround_dispatch_new = (
    '            # --- openclaw-rl-metaclaw-midround-reward (temporary, safe to remove) ---\n'
    '            if _METACLAW_MIDROUND_REWARD in ("outcome", "blend") and (\n'
    '                opd_result.get("metaclaw_verdict") or opd_result.get("metaclaw_round_step")\n'
    '            ):\n'
    '                _mr = self._metaclaw_round.setdefault(\n'
    '                    session_id,\n'
    '                    {"held": {}, "outcome": None, "verdict_turn": None, "state": "pending"},\n'
    '                )\n'
    '                if opd_result.get("metaclaw_verdict"):\n'
    '                    _verdict_score = opd_result.get("eval_score")\n'
    '                    if _verdict_score is None or not self._is_valid_rl_score(_verdict_score):\n'
    '                        # Verdict arrived but carries no usable outcome -- same\n'
    '                        # terminal state as the task raising: nothing to inherit,\n'
    '                        # so discard rather than leak. Tombstone rather than pop,\n'
    '                        # for the same late-task reason documented in the except\n'
    '                        # branch above.\n'
    '                        logger.warning(\n'
    '                            "[openclaw-rl-metaclaw-midround-reward] session=%s "\n'
    '                            "verdict turn=%d produced no valid outcome (%r) -- "\n'
    '                            "discarding %d held intermediate turn(s) "\n'
    '                            "(terminal state: failed)",\n'
    '                            session_id, turn_num, _verdict_score, len(_mr["held"]),\n'
    '                        )\n'
    '                        _mr["held"] = {}\n'
    '                        _mr["state"] = "failed"\n'
    '                    else:\n'
    '                        _mr["outcome"] = float(_verdict_score)\n'
    '                        _mr["state"] = "succeeded"\n'
    '                        # Flush BEFORE this verdict turn itself is submitted below,\n'
    '                        # and before any session cleanup can run.\n'
    '                        for _h_turn in sorted(_mr["held"].keys()):\n'
    '                            _h_td, _h_res = _mr["held"][_h_turn]\n'
    '                            # --- openclaw-rl-metaclaw-hard-negative-precedence\n'
    '                            # (2026-08-31) --- A turn that one of the forced-(-1)\n'
    '                            # rules already condemned (truncated generation, known\n'
    '                            # invalid tool use) must NOT be rewarded just because\n'
    '                            # the round happened to succeed. Real case from\n'
    '                            # metaclaw_migration_20260831_154301, day02/r2/turn2:\n'
    '                            # invalid-tool-use forced 1.0 -> -1.0, then the outcome\n'
    '                            # flush submitted it at +1.0 -- a repeated/degenerate\n'
    '                            # tool call getting positively reinforced. Outcome\n'
    '                            # inheritance replaces the JUDGE score, never a hard\n'
    '                            # rule\'s verdict.\n'
    '                            if _METACLAW_MIDROUND_REWARD == "blend":\n'
    '                                _h_reward = _metaclaw_blend_reward(\n'
    '                                    _mr["outcome"],\n'
    '                                    _h_res.get("judge_raw_score"),\n'
    '                                    _h_res.get("hard_negative"),\n'
    '                                )\n'
    '                            else:\n'
    '                                _h_reward = (\n'
    '                                    -1.0 if _h_res.get("hard_negative") else float(_mr["outcome"])\n'
    '                                )\n'
    '                            logger.info(\n'
    '                                "[openclaw-rl-metaclaw-midround-reward] session=%s "\n'
    '                                "turn=%d mode=%s flushed with reward=%.3f "\n'
    '                                "(inherited_outcome=%.1f hard_negative=%s "\n'
    '                                "judge_raw_score=%s legacy_reward_would_have_been=%s)",\n'
    '                                session_id, _h_turn, _METACLAW_MIDROUND_REWARD,\n'
    '                                _h_reward, _mr["outcome"],\n'
    '                                _h_res.get("hard_negative"),\n'
    '                                _h_res.get("judge_raw_score"),\n'
    '                                _h_res.get("legacy_reward_would_have_been"),\n'
    '                            )\n'
    '                            self._safe_create_task(\n'
    '                                self._submit_rl_turn_sample(\n'
    '                                    _h_td, session_id, _h_reward,\n'
    '                                )\n'
    '                            )\n'
    '                        _mr["held"] = {}\n'
    '                    # Fall through: the verdict turn itself keeps its normal\n'
    '                    # reward/hint handling, unchanged from judge mode.\n'
    '                else:\n'
    '                    if _mr.get("state") in ("failed", "no_verdict"):\n'
    '                        # This round already reached a terminal failure state, so no\n'
    '                        # outcome is ever coming. Discard on sight instead of holding\n'
    '                        # -- holding here is what would re-leak turn_data after the\n'
    '                        # failure branches released it (2026-08-31b fix, CLI review).\n'
    '                        logger.info(\n'
    '                            "[openclaw-rl-metaclaw-midround-reward] session=%s "\n'
    '                            "turn=%d discarded -- round already in terminal state %r, "\n'
    '                            "no outcome to inherit",\n'
    '                            session_id, turn_num, _mr.get("state"),\n'
    '                        )\n'
    '                        continue\n'
    '                    _known = _mr.get("outcome")\n'
    '                    if _known is None:\n'
    '                        _mr["held"][turn_num] = (td, opd_result)\n'
    '                        logger.info(\n'
    '                            "[openclaw-rl-metaclaw-midround-reward] session=%s "\n'
    '                            "turn=%d held awaiting round outcome "\n'
    '                            "(judge_raw_score=%s legacy_reward_would_have_been=%s)",\n'
    '                            session_id, turn_num,\n'
    '                            opd_result.get("judge_raw_score"),\n'
    '                            opd_result.get("legacy_reward_would_have_been"),\n'
    '                        )\n'
    '                        continue\n'
    '                    # Outcome already known: this judge task finished AFTER the\n'
    '                    # verdict flush (real race -- the verdict path runs\n'
    '                    # force_drop before intermediate tasks are necessarily done,\n'
    '                    # and each task re-enters this method via its own done\n'
    '                    # callback). Submit straight away with the stored outcome\n'
    '                    # instead of dropping it.\n'
    '                    # Same composition and hard-negative precedence as the\n'
    '                    # flush loop above.\n'
    '                    if _METACLAW_MIDROUND_REWARD == "blend":\n'
    '                        _late_reward = _metaclaw_blend_reward(\n'
    '                            _known,\n'
    '                            opd_result.get("judge_raw_score"),\n'
    '                            opd_result.get("hard_negative"),\n'
    '                        )\n'
    '                    else:\n'
    '                        _late_reward = (\n'
    '                            -1.0 if opd_result.get("hard_negative") else float(_known)\n'
    '                        )\n'
    '                    logger.info(\n'
    '                        "[openclaw-rl-metaclaw-midround-reward] session=%s turn=%d "\n'
    '                        "mode=%s late judge task, submitted with reward=%.3f "\n'
    '                        "(inherited_outcome=%.1f hard_negative=%s "\n'
    '                        "judge_raw_score=%s legacy_reward_would_have_been=%s)",\n'
    '                        session_id, turn_num, _METACLAW_MIDROUND_REWARD,\n'
    '                        _late_reward, _known,\n'
    '                        opd_result.get("hard_negative"),\n'
    '                        opd_result.get("judge_raw_score"),\n'
    '                        opd_result.get("legacy_reward_would_have_been"),\n'
    '                    )\n'
    '                    self._safe_create_task(\n'
    '                        self._submit_rl_turn_sample(td, session_id, _late_reward)\n'
    '                    )\n'
    '                    continue\n'
    '\n'
    '            eval_score = opd_result.get("eval_score")\n'
    '            if self._eval_mode and eval_score is not None:\n'
)
text = text.replace(midround_dispatch_old, midround_dispatch_new, 1)

# ---------------------------------------------------------------------
# openclaw-rl-metaclaw-midround-reward: post-loop resolution of held turns
# on the infrastructure-failure path.
#
# force_drop_without_next_state=True reaches here from two very different
# places and they must NOT be treated the same:
#   - the verdict path (openclaw_opd_api_server.py's verdict-signal-skip
#     branch) runs it right after firing the verdict task, i.e. while an
#     outcome is genuinely in flight -> held turns must survive;
#   - _send_session_close_only (agent infrastructure failure) runs it
#     without ever firing a verdict -> nothing will ever arrive, so held
#     turns must be discarded here or they leak forever, since they are no
#     longer in `pending` for the loop above to drop.
# verdict_turn is None in exactly the second case.
# ---------------------------------------------------------------------
midround_postloop_old = (
    '            elif has_valid_rl:\n'
    '                self._safe_create_task(\n'
    '                    self._submit_rl_turn_sample(td, session_id, float(eval_score))\n'
    '                )\n'
)
if text.count(midround_postloop_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 has_valid_rl dispatch tail in "
        f"{src_path}, found {text.count(midround_postloop_old)} "
        "(official file may have changed upstream -- update this patch)"
    )
midround_postloop_new = (
    '            elif has_valid_rl:\n'
    '                self._safe_create_task(\n'
    '                    self._submit_rl_turn_sample(td, session_id, float(eval_score))\n'
    '                )\n'
    '\n'
    '        # --- openclaw-rl-metaclaw-midround-reward (temporary, safe to remove) ---\n'
    '        if _METACLAW_MIDROUND_REWARD in ("outcome", "blend"):\n'
    '            _mr = self._metaclaw_round.get(session_id)\n'
    '            if (\n'
    '                _mr is not None\n'
    '                and force_drop_without_next_state\n'
    '                and _mr.get("verdict_turn") is None\n'
    '                and _mr.get("state") == "pending"\n'
    '            ):\n'
    '                # Infrastructure-failure path: the session was closed via\n'
    '                # _send_session_close_only, which never fires a verdict, so no\n'
    '                # outcome can ever arrive. Held turns are no longer in `pending`\n'
    '                # for the loop above to drop, so they must be released here.\n'
    '                # Tombstone rather than pop -- intermediate judge tasks can still\n'
    '                # be in flight at this point, and popping would let them re-create\n'
    '                # a fresh pending entry and be held forever.\n'
    '                if _mr["held"]:\n'
    '                    logger.info(\n'
    '                        "[openclaw-rl-metaclaw-midround-reward] session=%s closed "\n'
    '                        "without a verdict ever being fired (agent infrastructure "\n'
    '                        "failure) -- discarding %d held intermediate turn(s) "\n'
    '                        "(terminal state: no_verdict)",\n'
    '                        session_id, len(_mr["held"]),\n'
    '                    )\n'
    '                _mr["held"] = {}\n'
    '                _mr["state"] = "no_verdict"\n'
    '            # Single cleanup point for every terminal state (2026-08-31/08-31b,\n'
    '            # CLI review). Deliberately NOT gated on force_drop_without_next_state:\n'
    '            # the verdict flush happens on the task-completion callback, which calls\n'
    '            # this method WITHOUT force_drop, so a force_drop-gated cleanup could\n'
    '            # never run on the normal successful path. The pending/prm_tasks\n'
    '            # emptiness checks are what make that safe: a judge task finishing after\n'
    '            # the verdict still needs _mr["outcome"] (success) or _mr["state"]\n'
    '            # (failure) to decide what to do with itself, and while such a task is\n'
    '            # outstanding its turn is still in both dicts. Only once both have\n'
    '            # drained can nothing further arrive for this session.\n'
    '            if (\n'
    '                _mr is not None\n'
    '                and _mr.get("state") in ("succeeded", "failed", "no_verdict")\n'
    '                and not _mr["held"]\n'
    '                and not pending\n'
    '                and not prm_tasks\n'
    '            ):\n'
    '                self._metaclaw_round.pop(session_id, None)\n'
)
text = text.replace(midround_postloop_old, midround_postloop_new, 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched (dispatch-time drop of is_aborted/generated_while_paused/is_duplicate_user_retry/skip_forced_negative_override/metaclaw_training_frozen turns, gates both OPD and RL submission paths) -> {dest_path}")

# ---------------------------------------------------------------------
# openclaw-rl-metaclaw-batch-baseline (2026-09-01, temporary, safe to
# remove) -- see docs/metaclaw_migration_plan.md "方案：批级基线，对齐
# MetaClaw 的 compute_advantages".
#
# Emitted as a SEPARATE module (not a patch into the official file) because
# slime loads it by import path via --custom-reward-post-process-path, and
# because it replaces a whole function rather than editing one. Written into
# DEST_DIR, which the training launcher already prepends to PYTHONPATH
# (run_openclaw_topk_select_modelfactory.sh's PATCHED_COMBINE_DIR), so the
# import path is simply `metaclaw_batch_baseline.metaclaw_batch_baseline`.
# ---------------------------------------------------------------------
baseline_path = os.path.join(os.path.dirname(dest_path), "metaclaw_batch_baseline.py")
with open(baseline_path, "w", encoding="utf-8") as f:
    f.write(BATCH_BASELINE_SRC)
print(f"wrote batch-baseline reward post-processor -> {baseline_path}")
PY

echo "已生成 openclaw_combine_api_server.py 补丁: ${DEST_DIR}/openclaw_combine_api_server.py（_maybe_submit_ready_samples 拦截 is_aborted/generated_while_paused/is_duplicate_user_retry/skip_forced_negative_override，OPD+RL 两条提交路径一起挡住，见 docs/issues_log.md 2026-08-13 条目）"
