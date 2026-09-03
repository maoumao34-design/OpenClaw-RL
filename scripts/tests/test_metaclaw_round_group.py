"""Regression assertions for round-as-one-group + 1/N advantage scaling (2026-09-03).

Two pieces, both exercised against the code the patch scripts actually emit
from the real official source rather than a copy pasted here:

  1. `metaclaw_round_scale` -- the --custom-reward-post-process-path hook that
     divides each sample's advantage by the number of turns in its round.
  2. The proxy-side wiring that makes a round arrive as ONE group of per-turn
     samples, checked at the source level (it needs torch/slime to run).

Usage (needs the official repo checked out):
    python scripts/tests/test_metaclaw_round_group.py [OFFICIAL_REPO_ROOT]
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
DEFAULT_OFFICIAL = os.path.normpath(
    os.path.join(SCRIPTS_DIR, "..", "..", "OpenClaw-RL-official")
)


class _S:
    """Minimal stand-in for slime's Sample as the reward hook sees it."""

    def __init__(self, score, turns=None, dummy=False, remove_sample=False):
        self._score = score
        self.remove_sample = remove_sample or dummy
        self.metadata = {}
        if dummy:
            self.metadata["dummy_removed_sample"] = True
        if turns is not None:
            self.metadata["metaclaw_round_turns"] = turns
            self.metadata["metaclaw_round_id"] = "metaclaw-day01-day01-r1"

    def get_reward_value(self, args):
        return self._score


def _round(reward, turns):
    """One round: `turns` per-turn samples all carrying the round's verdict."""
    return [_S(reward, turns=turns) for _ in range(turns)]


def main():
    official = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OFFICIAL
    if not os.path.isdir(official):
        raise SystemExit(f"official repo not found: {official}")

    tmp = tempfile.mkdtemp(prefix="mc_group_test_")
    for script in ("prepare_patched_openclaw_opd.sh",
                   "prepare_patched_openclaw_combine.sh",
                   "prepare_patched_openclaw_combine_select.sh"):
        subprocess.run(["bash", os.path.join(SCRIPTS_DIR, script), official, tmp],
                       check=True, stdout=subprocess.DEVNULL)

    spec = importlib.util.spec_from_file_location(
        "metaclaw_round_scale", os.path.join(tmp, "metaclaw_round_scale.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scale = mod.metaclaw_round_scale
    args = None

    n = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    # -- the scaling itself ---------------------------------------------------
    print("[1/N scaling]")
    raw, adv = scale(args, _round(1.0, 4))
    ck(raw == [1.0] * 4, "raw rewards pass through untouched")
    ck(adv == [0.25] * 4, "a 4-turn round gives every one of its samples 1/4")
    ck(abs(sum(adv) - 1.0) < 1e-9, "the round totals exactly one round's worth")

    raw, adv = scale(args, _round(-1.0, 20))
    ck(all(a == -0.05 for a in adv), "a 20-turn failed round gives each sample -1/20")
    ck(abs(sum(adv) - (-1.0)) < 1e-9, "...and still totals exactly one round")

    # The whole point: round total is independent of turn count.
    print("\n[round weight is independent of how many turns it took]")
    totals = []
    for t in (1, 2, 5, 20, 186):
        _, a = scale(args, _round(1.0, t))
        totals.append(round(sum(a), 9))
    ck(totals == [1.0] * 5,
       "1, 2, 5, 20 and 186-turn rounds all contribute exactly 1.0 "
       "(186 is the real day06-r7 round that dominated a batch in 20260902_094458)")

    print("\n[a batch of several rounds]")
    batch = _round(1.0, 2) + _round(-1.0, 10) + _round(1.0, 6)
    raw, adv = scale(args, batch)
    ck(len(adv) == 18, "advantage is returned for every sample")
    ck(abs(sum(adv[:2]) - 1.0) < 1e-9, "round 1 (2 turns, passed) totals +1")
    ck(abs(sum(adv[2:12]) - (-1.0)) < 1e-9, "round 2 (10 turns, failed) totals -1")
    ck(abs(sum(adv[12:]) - 1.0) < 1e-9, "round 3 (6 turns, passed) totals +1")
    ck(abs(sum(adv) - 1.0) < 1e-9,
       "the batch is the sum over rounds (2 passed, 1 failed), not over turns -- "
       "under the old per-sample weighting the 10-turn failure would have "
       "outweighed both successes")
    ck(max(abs(a) for a in adv) <= 1.0,
       "|advantage| never exceeds the raw reward magnitude")

    # -- pass-through and degenerate cases -----------------------------------
    print("\n[non-MetaClaw samples and degenerate input]")
    raw, adv = scale(args, [_S(1.0), _S(-1.0)])
    ck(adv == [1.0, -1.0],
       "samples without round metadata pass through unchanged -- this hook "
       "short-circuits _post_process_rewards, so it must reproduce the default "
       "--disable-rewards-normalization behaviour for the Personal Agent Track")

    raw, adv = scale(args, _round(1.0, 3) + [_S(-1.0)])
    ck(adv[:3] == [1.0 / 3] * 3 and adv[3] == -1.0,
       "scaled and pass-through samples coexist in one batch")

    raw, adv = scale(args, _round(1.0, 2) + [_S(0.0, dummy=True)])
    ck(adv[2] == 0.0, "dummy samples get zero")
    ck(adv[:2] == [0.5, 0.5], "dummies do not disturb the real samples")

    raw, adv = scale(args, [_S(0.0, remove_sample=True)])
    ck(adv == [0.0], "remove_sample without the metadata marker is also excluded")

    raw, adv = scale(args, [])
    ck(raw == [] and adv == [], "an empty batch does not crash")

    for bad in (0, -3, "4", None, 2.5):
        s = _S(1.0)
        s.metadata = {"metaclaw_round_turns": bad}
        _, a = scale(args, [s])
        ck(a == [1.0], f"a malformed turn count ({bad!r}) falls back to no scaling")

    for k in (1, 3, 18):
        raw, adv = scale(args, _round(1.0, k))
        ck(len(raw) == k and len(adv) == k,
           f"both lists come back at full input length ({k}) -- slime asserts on this")

    # -- proxy-side wiring (source-level: needs torch/slime to run) -----------
    print("\n[proxy emits a round as one group]")
    combine = open(os.path.join(tmp, "openclaw_combine_api_server.py"),
                   encoding="utf-8").read()
    ck("async def _metaclaw_submit_round(" in combine, "the round assembler exists")
    ck(combine.count('_mc_collect = turn_data.get("metaclaw_round_collect")') == 2,
       "both submission paths can hand a sample back instead of queueing it")
    ck(combine.count(
        "await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))") == 2,
       "both paths still queue individually when no round is being assembled "
       "(the Personal Agent Track path)")
    ck("await asyncio.to_thread(self.output_queue.put, (group_index, collect))" in combine,
       "the round is queued exactly once, as one group -- _drain_output_queue "
       "overwrites on repeated group ids, so an incremental put would lose members")
    ck('"metaclaw_round_turns": turn_data["metaclaw_round_turns"]' in combine,
       "every sample carries its round's turn count for the hook to divide by")
    ck('if opd_result.get("metaclaw_verdict"):' in combine,
       "the verdict result takes over dispatch for the whole round")
    ck("if not has_valid_rl:" in combine and "carries no valid outcome" in combine,
       "a verdict with no usable outcome drops the round instead of inventing one")
    ck(combine.index("if not has_valid_rl:") < combine.index("if opd_accepted and has_valid_rl:"),
       "that drop is checked before the official per-turn dispatch branches")

    print("\n[proxy holds intermediate turns]")
    opd = open(os.path.join(tmp, "openclaw_opd_api_server.py"), encoding="utf-8").read()
    ck("held (intermediate turn, no judge, no sample of its own)" in opd,
       "intermediate MetaClaw turns fire no judge")
    ck("_metaclaw_build_trajectory" not in opd,
       "the reverted flat-trajectory assembler is gone")
    ck("Most likely cause is OpenClaw compacting" not in opd,
       "the wrong compaction diagnosis is gone (real cause: dropReasoningFromHistory)")

    print("\n[OPD hint anchors on the round task]")
    select = open(os.path.join(tmp, "openclaw_combine_select_api_server.py"),
                  encoding="utf-8").read()
    ck("_mc_first_user = next(" in select,
       "the verdict branch locates the round's first user message")
    ck("_mc_msgs[: _mc_first_user + 1], _metaclaw_hint," in select,
       "the hint is appended to the task, not to the last tool result")
    ck("+ _mc_msgs[_mc_first_user + 1 :]" in select,
       "the rest of the conversation is spliced back unchanged")

    print(f"\nall {n} assertions passed")


if __name__ == "__main__":
    main()
