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

# The round-group membership filter below needs _flatten_message_content,
# which the combine server does not import (the OPD server defines it).
import_old = (
    "from openclaw_opd_api_server import OpenClawOPDAPIServer, generate, reward_func  # noqa: F401\n"
)
import_new = (
    "from openclaw_opd_api_server import (  # noqa: F401\n"
    "    OpenClawOPDAPIServer,\n"
    "    _flatten_message_content,\n"
    "    generate,\n"
    "    reward_func,\n"
    ")\n"
    "\n"
    "# --- openclaw-rl-metaclaw-trajectory ---\n"
    "# Matches what OpenClaw strips out of an assistant turn before replaying\n"
    "# it, so an earlier turn can be located inside a later prompt by the part\n"
    "# of it that survived.\n"
    "import re as _mc_re  # noqa: E402\n"
    '_MC_THINK_RE = _mc_re.compile(r"<think>.*?</think>", _mc_re.DOTALL)\n'
)
if text.count(import_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the openclaw_opd_api_server "
        f"import line in {src_path}, found {text.count(import_old)} "
        "(official file may have changed upstream -- update this patch)"
    )
text = text.replace(import_old, import_new, 1)

# ---------------------------------------------------------------------
# openclaw-rl-metaclaw-trajectory (2026-09-14)
#
# A round is ONE sample, not one per turn. The round's +/-1 and the round's
# hint both act on the whole thing, which is what they are about: the hint
# describes the artifact the round was supposed to produce, not any single
# step.
#
# How the sequence is built, and why it is not built by concatenation:
# turn N's prompt is NOT turn N-1's prompt plus its response (measured 0/127
# on production records) -- OpenClaw strips the thinking out of an assistant
# turn before replaying it, so the chain does not nest and there is nothing to
# concatenate. Instead the LAST turn's prompt is used as the backbone, since
# it already holds the task, every earlier action and every tool result in the
# exact form the model read them, and each earlier turn's visible output is
# swapped back out for its full response so the thinking is trained on too.
#
# A turn whose visible output cannot be located in the backbone is left as-is
# and simply not masked: its tokens stay in the sequence (they are what the
# model actually read) but contribute no gradient. That degrades one turn
# rather than discarding the round, which is what the 2026-09-03 attempt did
# when its prefix assertion failed -- it dropped 96 of 120 rounds.
#
# The shape -- one sample, loss_mask 1 on generated spans and 0 on tool
# results, rollout_log_probs zero-filled on the unmasked spans -- is slime's
# own multi-turn form; see slime/examples/search-r1/generate_with_search.py.
# ---------------------------------------------------------------------
loss_mask_old = (
    '        sample.loss_mask = [1] * len(response_ids)\n'
)
if text.count(loss_mask_old) != 2:
    raise SystemExit(
        f"patch failed: expected exactly 2 loss_mask assignments in {src_path} "
        f"(_submit_turn_sample and _submit_rl_turn_sample), found "
        f"{text.count(loss_mask_old)} (official file may have changed "
        "upstream -- update this patch)"
    )
loss_mask_new = (
    '        # --- openclaw-rl-metaclaw-trajectory ---\n'
    '        # A trajectory sample masks only the generated spans; tool results\n'
    '        # sit in the same response segment and must earn no gradient.\n'
    '        _mc_mask = turn_data.get("metaclaw_loss_mask")\n'
    '        if _mc_mask is not None and len(_mc_mask) == len(response_ids):\n'
    '            sample.loss_mask = list(_mc_mask)\n'
    '        else:\n'
    '            if _mc_mask is not None:\n'
    '                logger.error(\n'
    '                    "[openclaw-rl-metaclaw-trajectory] loss_mask length %d "\n'
    '                    "!= response length %d -- falling back to all-ones, so "\n'
    '                    "this sample would train on tool results",\n'
    '                    len(_mc_mask), len(response_ids),\n'
    '                )\n'
    '            sample.loss_mask = [1] * len(response_ids)\n'
)
text = text.replace(loss_mask_old, loss_mask_new, 2)

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
    '    @staticmethod\n'
    '    def _metaclaw_visible(text_in):\n'
    '        """The part of a response that survives into the next prompt.\n'
    '\n'
    '        OpenClaw removes <think> blocks from an assistant turn before\n'
    '        replaying it, so this is the form an earlier turn takes inside\n'
    '        a later prompt -- and therefore what has to be searched for\n'
    '        when putting the thinking back.\n'
    '        """\n'
    '        return _MC_THINK_RE.sub("", text_in or "").strip()\n'
    '\n'
    '    def _metaclaw_build_trajectory(self, session_id, turns):\n'
    '        """Assemble the round into one sample: ids, mask, logprobs.\n'
    '\n'
    '        Returns a turn_data-shaped dict so the existing submit paths\n'
    '        can build the Sample, or None when it cannot be assembled.\n'
    '        """\n'
    '        final = turns[-1]\n'
    '        backbone = final.get("prompt_text") or ""\n'
    '        if not backbone:\n'
    '            return None\n'
    '\n'
    '        pieces = []\n'
    '        cursor = 0\n'
    '        missing = 0\n'
    '        for _td in turns[:-1]:\n'
    '            _vis = self._metaclaw_visible(_td.get("response_text"))\n'
    '            if not _vis:\n'
    '                missing += 1\n'
    '                continue\n'
    '            _at = backbone.find(_vis, cursor)\n'
    '            if _at < 0:\n'
    '                # Left in the backbone as OpenClaw rendered it, but not\n'
    '                # masked: those tokens are real, they just earn no\n'
    '                # gradient. Degrading one turn beats discarding the\n'
    '                # round, which is what the 2026-09-03 attempt did.\n'
    '                missing += 1\n'
    '                logger.warning(\n'
    '                    "[openclaw-rl-metaclaw-trajectory] session=%s a turn "\n'
    '                    "was not found in the final prompt -- left unmasked",\n'
    '                    session_id,\n'
    '                )\n'
    '                continue\n'
    '            pieces.append((False, backbone[cursor:_at]))\n'
    '            pieces.append((True, _td))\n'
    '            cursor = _at + len(_vis)\n'
    '        pieces.append((False, backbone[cursor:]))\n'
    '        pieces.append((True, final))\n'
    '\n'
    '        # Everything before the first generated span is the prompt.\n'
    '        first_gen = next(i for i, (g, _) in enumerate(pieces) if g)\n'
    '        prompt_text = "".join(x for g, x in pieces[:first_gen] if not g)\n'
    '        prompt_ids = self.tokenizer(\n'
    '            prompt_text, add_special_tokens=False,\n'
    '        )["input_ids"]\n'
    '\n'
    '        response_ids, loss_mask, logprobs = [], [], []\n'
    '        text_parts = []\n'
    '        for is_gen, item in pieces[first_gen:]:\n'
    '            if is_gen:\n'
    '                # Tokens the model produced, verbatim -- never re-tokenised.\n'
    '                _ids = list(item["response_ids"])\n'
    '                _lp = list(item.get("response_logprobs") or [])\n'
    '                if len(_lp) > len(_ids):\n'
    '                    _lp = _lp[: len(_ids)]\n'
    '                elif len(_lp) < len(_ids):\n'
    '                    _lp = _lp + [0.0] * (len(_ids) - len(_lp))\n'
    '                response_ids += _ids\n'
    '                loss_mask += [1] * len(_ids)\n'
    '                logprobs += _lp\n'
    '                text_parts.append(item.get("response_text") or "")\n'
    '            else:\n'
    '                if not item:\n'
    '                    continue\n'
    '                _ids = self.tokenizer(\n'
    '                    item, add_special_tokens=False,\n'
    '                )["input_ids"]\n'
    '                response_ids += _ids\n'
    '                loss_mask += [0] * len(_ids)\n'
    '                logprobs += [0.0] * len(_ids)\n'
    '                text_parts.append(item)\n'
    '\n'
    '        if not any(loss_mask):\n'
    '            logger.warning(\n'
    '                "[openclaw-rl-metaclaw-trajectory] session=%s assembled a "\n'
    '                "trajectory with nothing masked -- dropping",\n'
    '                session_id,\n'
    '            )\n'
    '            return None\n'
    '\n'
    '        return {\n'
    '            "prompt_ids": prompt_ids,\n'
    '            "response_ids": response_ids,\n'
    '            "response_logprobs": logprobs,\n'
    '            "prompt_text": prompt_text,\n'
    '            "response_text": "".join(text_parts),\n'
    '            "messages": final.get("messages"),\n'
    '            "tools": final.get("tools"),\n'
    '            "metaclaw_loss_mask": loss_mask,\n'
    '            "metaclaw_trajectory": True,\n'
    '            "metaclaw_missing_turns": missing,\n'
    '        }\n'
    '\n'
    '    async def _metaclaw_submit_round(\n'
    '        self, session_id: str, turns: list, verdict_td: dict,\n'
    '        opd_result: dict, reward: float,\n'
    '    ):\n'
    '        """Submit one MetaClaw round as ONE trajectory sample.\n'
    '\n'
    '        `turns` are the held intermediate turns of the round, and\n'
    '        `verdict_td` is the last real generating turn. The round-level\n'
    '        +/-1 acts on the whole trajectory, and so does the OPD teacher\n'
    '        signal when a hint was accepted -- not on any single step.\n'
    '        """\n'
    '        all_tds = list(turns) + [verdict_td]\n'
    '        n_turns = len(all_tds)\n'
    '        traj = self._metaclaw_build_trajectory(session_id, all_tds)\n'
    '        if traj is None:\n'
    '            logger.warning(\n'
    '                "[openclaw-rl-metaclaw-trajectory] session=%s could not "\n'
    '                "assemble a trajectory from %d turn(s) -- nothing queued",\n'
    '                session_id, n_turns,\n'
    '            )\n'
    '            return\n'
    '\n'
    '        collect: list = []\n'
    '        group_index = next(self._group_counter)\n'
    '        traj["metaclaw_round_collect"] = collect\n'
    '        traj["metaclaw_round_group_index"] = group_index\n'
    '        traj["metaclaw_round_turns"] = n_turns\n'
    '\n'
    '        if opd_result.get("accepted"):\n'
    '            await self._submit_turn_sample(\n'
    '                traj, session_id, opd_result, reward=reward,\n'
    '            )\n'
    '        else:\n'
    '            await self._submit_rl_turn_sample(traj, session_id, reward)\n'
    '\n'
    '        if not collect:\n'
    '            logger.warning(\n'
    '                "[openclaw-rl-metaclaw-trajectory] session=%s produced no "\n'
    '                "sample from %d turn(s) -- nothing queued",\n'
    '                session_id, n_turns,\n'
    '            )\n'
    '            return\n'
    '        _masked = sum(traj["metaclaw_loss_mask"])\n'
    '        logger.info(\n'
    '            "[openclaw-rl-metaclaw-trajectory] session=%s queued group=%d "\n'
    '            "trajectory from %d turn(s), reward=%.1f, %d/%d tokens masked, "\n'
    '            "%d turn(s) not located, opd=%s",\n'
    '            session_id, group_index, n_turns, reward, _masked,\n'
    '            len(traj["metaclaw_loss_mask"]), traj["metaclaw_missing_turns"],\n'
    '            bool(opd_result.get("accepted")),\n'
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
    '                # --- round boundary (2026-09-07) ---\n'
    '                # `t < turn_num` alone silently assumes `pending` can only\n'
    '                # ever hold THIS round\'s turns. That held for free while\n'
    '                # every round had its own session; with one session per day\n'
    '                # it holds only as long as every verdict dispatches. If one\n'
    '                # does not (duplicate-retry drop, a network failure, the\n'
    '                # has_valid_rl `continue` below), that round\'s turns stay\n'
    '                # pending and the NEXT round would sweep them up, giving\n'
    '                # them the next round\'s reward and inflating its 1/N\n'
    '                # denominator. Bounding below by the previous verdict makes\n'
    '                # the round boundary explicit instead of implied.\n'
    '                _mc_bounds = getattr(self, "_metaclaw_last_verdict_turn", None)\n'
    '                if _mc_bounds is None:\n'
    '                    _mc_bounds = self._metaclaw_last_verdict_turn = {}\n'
    '                _mc_lo = _mc_bounds.get(session_id, 0)\n'
    '                _mc_orphans = [t for t in sorted(pending.keys()) if t <= _mc_lo]\n'
    '                if _mc_orphans:\n'
    '                    # Left behind by an earlier round whose verdict never\n'
    '                    # dispatched. Dropping them is the honest option -- they\n'
    '                    # belong to a round this verdict did not score.\n'
    '                    logger.warning(\n'
    '                        "[openclaw-rl-metaclaw-round-group] session=%s verdict "\n'
    '                        "turn=%d found %d orphaned turn(s) %r from a round whose "\n'
    '                        "verdict never dispatched -- dropping, NOT merging into "\n'
    '                        "this round",\n'
    '                        session_id, turn_num, len(_mc_orphans), _mc_orphans,\n'
    '                    )\n'
    '                    for _t in _mc_orphans:\n'
    '                        pending.pop(_t, None)\n'
    '                _mc_cand = [t for t in sorted(pending.keys()) if _mc_lo < t < turn_num]\n'
    '\n'
    '                # The turn-number bound alone is not enough. It only moves\n'
    '                # forward when a verdict DISPATCHES, so a round whose verdict\n'
    '                # was lost outright (the POST failed, and VERDICT_RETRY is 0\n'
    '                # by default) leaves its turns inside the next round\'s window.\n'
    '                # Membership is decided instead by the round\'s own task text:\n'
    '                # a turn from round r carries tasks 1..r in its prompt, so it\n'
    '                # contains round r\'s task while an earlier round\'s turn does\n'
    '                # not. That needs no verdict bookkeeping to be correct.\n'
    '                _mc_prefix = (opd_result.get("metaclaw_task_prefix") or "")[:120]\n'
    '                if _mc_prefix:\n'
    '                    _mc_mine, _mc_foreign = [], []\n'
    '                    for _t in _mc_cand:\n'
    '                        _hit = any(\n'
    '                            isinstance(_m, dict) and _m.get("role") == "user"\n'
    '                            and _mc_prefix in _flatten_message_content(_m.get("content"))\n'
    '                            for _m in (pending[_t].get("messages") or [])\n'
    '                        )\n'
    '                        (_mc_mine if _hit else _mc_foreign).append(_t)\n'
    '                    if _mc_foreign:\n'
    '                        logger.warning(\n'
    '                            "[openclaw-rl-metaclaw-round-group] session=%s verdict "\n'
    '                            "turn=%d: %d turn(s) %r in this window do not carry "\n'
    '                            "this round\'s task -- they belong to a round whose "\n'
    '                            "verdict was lost; dropping, NOT merging",\n'
    '                            session_id, turn_num, len(_mc_foreign), _mc_foreign,\n'
    '                        )\n'
    '                        for _t in _mc_foreign:\n'
    '                            pending.pop(_t, None)\n'
    '                    _mc_cand = _mc_mine\n'
    '\n'
    '                _mc_turns = [pending[t] for t in _mc_cand]\n'
    '                for _t in _mc_cand:\n'
    '                    pending.pop(_t, None)\n'
    '                _mc_bounds[session_id] = turn_num\n'
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

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched (dispatch-time drop of is_aborted/generated_while_paused/is_duplicate_user_retry/skip_forced_negative_override/metaclaw_training_frozen turns, gates both OPD and RL submission paths) -> {dest_path}")
PY

echo "已生成 openclaw_combine_api_server.py 补丁: ${DEST_DIR}/openclaw_combine_api_server.py（_maybe_submit_ready_samples 拦截 is_aborted/generated_while_paused/is_duplicate_user_retry/skip_forced_negative_override，OPD+RL 两条提交路径一起挡住，见 docs/issues_log.md 2026-08-13 条目）"
