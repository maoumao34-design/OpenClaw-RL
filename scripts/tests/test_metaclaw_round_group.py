"""Wiring assertions for round-as-one-trajectory (2026-09-14).

Checked against the code the patch scripts actually emit from the real official
source, never against a copy pasted here.

History worth keeping in view: from 2026-09-03 to 2026-09-14 a round was N
per-turn samples whose advantages were divided by N by a
`--custom-reward-post-process-path` hook (`metaclaw_round_scale`). That hook is
gone: a round is now ONE sample, so N is 1 and there is nothing left to divide.
Its own regression coverage was deleted with it rather than left passing
against a file nobody loads.

Behaviour of the trajectory builder itself lives in
test_metaclaw_trajectory.py. This file covers the wiring around it, which
needs torch/slime to import and so can only be checked at the source level.

Usage (needs the official repo checked out):
    python scripts/tests/test_metaclaw_round_group.py [OFFICIAL_REPO_ROOT]
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS_DIR)


def main():
    official = (
        sys.argv[1] if len(sys.argv) > 1
        else os.environ.get(
            "OPENCLAW_RL_OFFICIAL",
            os.path.join(os.path.dirname(ROOT), "OpenClaw-RL-official"),
        )
    )
    if not os.path.isdir(official):
        print(f"  -- skipped: {official} not present (set OPENCLAW_RL_OFFICIAL)")
        return

    tmp = tempfile.mkdtemp(prefix="mc_group_test_")
    for script in ("prepare_patched_openclaw_opd.sh",
                   "prepare_patched_openclaw_combine.sh",
                   "prepare_patched_openclaw_combine_select.sh"):
        subprocess.run(["bash", os.path.join(SCRIPTS_DIR, script), official, tmp],
                       check=True, stdout=subprocess.DEVNULL)

    n = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    combine = open(os.path.join(tmp, "openclaw_combine_api_server.py"),
                   encoding="utf-8").read()
    opd = open(os.path.join(tmp, "openclaw_opd_api_server.py"),
               encoding="utf-8").read()
    select = open(os.path.join(tmp, "openclaw_combine_select_api_server.py"),
                  encoding="utf-8").read()

    print("[the 1/N scaler is gone, not merely unused]")
    ck(not os.path.exists(os.path.join(tmp, "metaclaw_round_scale.py")),
       "no metaclaw_round_scale.py is emitted any more")
    profile = open(os.path.join(SCRIPTS_DIR,
                                "run_openclaw_topk_select_modelfactory.sh"),
                   encoding="utf-8").read()
    ck("sed -i -e 's|--disable-rewards-normalization|" not in profile,
       "the profile no longer injects --custom-reward-post-process-path")

    print("\n[a round is assembled into one trajectory sample]")
    ck("async def _metaclaw_submit_round(" in combine, "the round assembler exists")
    ck("def _metaclaw_build_trajectory(" in combine, "the trajectory builder exists")
    ck("await asyncio.to_thread(self.output_queue.put, (group_index, collect))" in combine,
       "the round is queued exactly once -- _drain_output_queue overwrites on "
       "repeated group ids, so an incremental put would lose members")
    ck(combine.count("await self._submit_turn_sample(") == 1
       and combine.count("await self._submit_rl_turn_sample(") == 1,
       "exactly one sample is submitted per round, on one of the two paths")
    ck("for _td in turns:" not in combine,
       "the per-turn submission loop is gone, not left beside the new path")

    print("\n[the mask reaches BOTH submit paths]")
    # 2026-09-08 cost a whole run to this exact class of mistake: the hand-back
    # went into the parent while the Select subclass overrode both methods.
    for name, src in (("parent", combine), ("Select subclass", select)):
        ck(src.count('_mc_mask = turn_data.get("metaclaw_loss_mask")') == 2,
           f"{name}: both submit paths honour a caller-supplied loss_mask")
        ck(src.count("sample.loss_mask = list(_mc_mask)") == 2,
           f"{name}: both paths use it when its length matches")
        ck(src.count("sample.loss_mask = [1] * len(response_ids)") == 2,
           f"{name}: both paths still fall back to all-ones otherwise")
    ck(combine.count('_mc_collect = turn_data.get("metaclaw_round_collect")') == 2
       and select.count('_mc_collect = turn_data.get("metaclaw_round_collect")') == 2,
       "the hand-back covers both submit paths in parent and subclass alike")

    print("\n[teacher tokens stay aligned to the trajectory]")
    ck('if turn_data.get("metaclaw_trajectory"):' in select,
       "the OPD branch knows when it is looking at a trajectory")
    ck("_enhanced_prompt_text, add_special_tokens=False," in select
       and ')["input_ids"] + list(_mc_resp_ids)' in select,
       "a trajectory's response ids are appended verbatim rather than "
       "re-tokenised -- re-tokenising would misalign the teacher log-probs")

    print("\n[proxy still holds intermediate turns]")
    ck("held (intermediate turn, no judge, no sample of its own)" in opd,
       "intermediate MetaClaw turns fire no judge of their own")
    ck("Most likely cause is OpenClaw compacting" not in opd,
       "the wrong compaction diagnosis is gone")

    print("\n[OPD hint still anchors on the round task]")
    ck("_mc_task_idx" in select,
       "the verdict branch locates THIS round's task via task_prefix")
    ck("_mc_msgs[: _mc_task_idx + 1], _metaclaw_hint," in select,
       "the hint is appended to that task, not to the last tool result")
    ck("+ _mc_msgs[_mc_task_idx + 1 :]" in select,
       "the rest of the conversation is spliced back unchanged")
    ck("_mc_first_user" not in select,
       "the day-unsafe first-user form is gone")

    print("\n[round membership is unchanged]")
    ck("if not has_valid_rl:" in combine and "carries no valid outcome" in combine,
       "a verdict with no usable outcome drops the round instead of inventing one")
    ck('if opd_result.get("metaclaw_verdict"):' in combine,
       "the verdict result takes over dispatch for the whole round")

    print(f"\nall {n} assertions passed")


if __name__ == "__main__":
    main()
