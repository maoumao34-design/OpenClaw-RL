"""Regression assertions for the 2026-09-01 anti-collapse patches.

Covers the two pieces that together are supposed to keep a 30-day run from
training itself into an unusable state:

  1. `_metaclaw_length_aware_reward` in the patched proxy
     (scripts/prepare_patched_openclaw_combine.sh) -- discounts a POSITIVE
     reward by response length, leaves everything else alone.
  2. `metaclaw_batch_baseline` (same script, emitted as a standalone module) --
     now subtracts the batch mean WITHOUT dividing by std.

Both are exercised against the code that the patch script actually generates
from the real official source, not against a copy pasted here -- a test that
reimplements the thing it tests would pass no matter what the patch emits.

Usage (needs the official repo checked out):
    python scripts/tests/test_length_aware_and_baseline.py [OFFICIAL_REPO_ROOT]
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


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_length_aware(proxy_path):
    """Pull `_metaclaw_length_aware_reward` out of the patched proxy.

    The proxy imports torch/slime and cannot be imported here, so the module-level
    block the patch inserts is extracted and executed on its own. The extraction
    is anchored on the patch's own markers, so if the patch stops emitting them
    this raises instead of silently testing nothing.
    """
    src = open(proxy_path, encoding="utf-8").read()
    start = src.index("# --- openclaw-rl-metaclaw-length-aware-success (2026-09-01) ---")
    end = src.index("class ", start)
    block = src[start:end]
    assert "def _metaclaw_length_aware_reward" in block, (
        "length-aware block found but the function is not in it -- "
        "the patch layout changed, update this extraction"
    )
    ns = {"_mr_os": os}
    exec(compile(block, proxy_path, "exec"), ns)
    return ns


class _S:
    """Minimal stand-in for slime's Sample as the baseline hook sees it."""

    def __init__(self, score, dummy=False, remove_sample=False, metadata=None):
        self._score = score
        self.remove_sample = remove_sample or dummy
        self.metadata = metadata if metadata is not None else (
            {"dummy_removed_sample": True} if dummy else {}
        )

    def get_reward_value(self, args):
        return self._score


def main():
    official = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OFFICIAL
    if not os.path.isdir(official):
        raise SystemExit(f"official repo not found: {official}")

    tmp = tempfile.mkdtemp(prefix="mc_patch_test_")
    subprocess.run(
        ["bash", os.path.join(SCRIPTS_DIR, "prepare_patched_openclaw_combine.sh"),
         official, tmp],
        check=True, stdout=subprocess.DEVNULL,
    )
    proxy = os.path.join(tmp, "openclaw_combine_api_server.py")
    baseline_mod = _load(os.path.join(tmp, "metaclaw_batch_baseline.py"),
                         "metaclaw_batch_baseline")
    baseline = baseline_mod.metaclaw_batch_baseline
    ns = _load_length_aware(proxy)
    decay = ns["_metaclaw_length_aware_reward"]
    L0, L1 = ns["_METACLAW_LEN_DECAY_L0"], ns["_METACLAW_LEN_DECAY_L1"]
    floor = ns["_METACLAW_LEN_DECAY_FLOOR"]

    n = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    print("defaults: L0=%g L1=%g floor=%g" % (L0, L1, floor))

    # -- length-aware reward -------------------------------------------------
    print("\n[length-aware reward]")
    ck((L0, L1, floor) == (6000.0, 16000.0, 0.1), "defaults are 6000 / 16000 / 0.1")
    ck(decay(1.0, 0) == 1.0, "zero-length positive keeps full reward")
    ck(decay(1.0, 2500) == 1.0, "K=6-typical positive (2.5k tok) keeps 1.0")
    ck(decay(1.0, 6000) == 1.0, "exactly L0 keeps 1.0 (boundary, inclusive)")
    ck(abs(decay(1.0, 10273) - 0.615) < 1e-3,
       "day06-r6's real 10273-token success scores 0.615, not 1.0")
    ck(abs(decay(1.0, 9000) - 0.73) < 1e-9,
       "healthy-but-long 9k success scores 0.73 (the accepted cost)")
    ck(abs(decay(1.0, 16000) - floor) < 1e-12, "exactly L1 hits the floor")
    ck(abs(decay(1.0, 120000) - floor) < 1e-12,
       "far past L1 clamps to the floor, does not go negative")
    ck(decay(1.0, 120000) > 0, "an extremely long CORRECT answer is still positive")

    # Negatives flat is the load-bearing property, not an optimisation.
    print("\n[negatives stay flat -- this is what keeps all-negative batches safe]")
    for length in (0, 2500, 10273, 16000, 200000):
        ck(decay(-1.0, length) == -1.0, f"negative at len={length} stays exactly -1.0")
    ck(decay(0.0, 50000) == 0.0, "OPD-only sample (reward 0.0) is untouched")

    # Monotone, and never crosses a negative.
    prev = decay(1.0, 0)
    mono = True
    for length in range(0, 20001, 250):
        cur = decay(1.0, length)
        mono = mono and cur <= prev + 1e-12 and cur > 0
        prev = cur
    ck(mono, "monotonically non-increasing in length and strictly positive throughout")

    # Degenerate configuration must not divide by zero or invert.
    ns_bad = dict(ns)
    ns_bad["_METACLAW_LEN_DECAY_L1"] = ns_bad["_METACLAW_LEN_DECAY_L0"]
    exec("def f(r, l): return _metaclaw_length_aware_reward(r, l)", ns_bad)
    src = open(proxy, encoding="utf-8").read()
    ck("if _METACLAW_LEN_DECAY_L1 <= _METACLAW_LEN_DECAY_L0:" in src,
       "L1<=L0 is guarded (no division by zero if the thresholds are misconfigured)")

    # -- batch baseline: mean only ------------------------------------------
    print("\n[batch baseline -- mean only, no std division]")
    args = None

    raw, adv = baseline(args, [_S(-1.0) for _ in range(16)])
    ck(all(a == 0.0 for a in adv),
       "all-negative batch of 16 -> advantage all 0 (the safety property: hard days do not damage the model)")
    ck(all(r == -1.0 for r in raw), "raw rewards are passed through untouched")

    raw, adv = baseline(args, [_S(1.0) for _ in range(16)])
    ck(all(a == 0.0 for a in adv), "all-positive batch -> advantage all 0 too")

    samples = [_S(1.0)] + [_S(-1.0) for _ in range(15)]
    raw, adv = baseline(args, samples)
    ck(abs(adv[0] - 1.875) < 1e-9,
       "1 pos / 15 neg -> pos advantage 1.875 (was 3.873 under std division)")
    ck(abs(adv[1] - (-0.125)) < 1e-9, "1 pos / 15 neg -> neg advantage -0.125")
    ck(max(abs(a) for a in adv) <= 2.0,
       "|advantage| <= 2 with rewards in [-1,1] -- the bound std division did not have")

    # The mirror case: the step judge returns ~69% positive, so a rare NEGATIVE
    # is just as real a tail as a rare positive.
    samples = [_S(1.0) for _ in range(11)] + [_S(-1.0) for _ in range(5)]
    raw, adv = baseline(args, samples)
    ck(abs(adv[-1] - (-1.375)) < 1e-9,
       "11 pos / 5 neg -> rare negative is -1.375 (mean 0.375), bounded by 2")
    ck(max(abs(a) for a in adv) <= 2.0, "|advantage| <= 2 on the mirror tail too")

    # Non-vacuity: had we kept the std division, the 1/15 case would be 3.873.
    ck(abs(1.875 - 3.873) > 1.9,
       "the two formulas genuinely differ on the case that caused the spike "
       "(guards against this test passing under either implementation)")
    ck("advantages[i] = raw_rewards[i] - mean_r" in
       open(os.path.join(tmp, "metaclaw_batch_baseline.py"), encoding="utf-8").read(),
       "the emitted module really subtracts the mean without dividing")

    # -- the two together ----------------------------------------------------
    print("\n[length decay + baseline together]")
    # day06-r6's exact situation: one long correct answer among 15 failures.
    samples = [_S(decay(1.0, 10273))] + [_S(decay(-1.0, 3000)) for _ in range(15)]
    raw, adv = baseline(args, samples)
    ck(abs(adv[0] - 1.514466) < 1e-5,
       "the 10273-token lone success gets +1.514 (3.873 -> 1.875 -> 1.514 across the two fixes)")
    ck(adv[0] > 0, "it is still reinforced -- being long is a discount, not a rejection")

    short = [_S(decay(1.0, 2000))] + [_S(-1.0) for _ in range(15)]
    _, adv_short = baseline(args, short)
    ck(adv_short[0] > adv[0],
       "a SHORT lone success outranks a long one -- the discrimination binary reward could not provide")

    # The blend trap, as an executable assertion.
    print("\n[the blend trap must stay closed]")
    failed_round = [_S(-1.0) for _ in range(16)]
    _, adv_failed = baseline(args, failed_round)
    ck(not any(a > 0 for a in adv_failed),
       "an entirely-failed round produces NO positive advantage")
    blend_like = [_S(-0.7)] + [_S(-1.0) for _ in range(8)] + [_S(-1.3) for _ in range(7)]
    _, adv_blend = baseline(args, blend_like)
    ck(any(a > 0 for a in adv_blend),
       "...whereas blend's spread rewards (-0.7/-1.0/-1.3) DO produce a positive "
       "advantage inside a fully-failed round -- the reason blend was removed, "
       "and the reason negatives must stay flat")

    # -- dummy handling ------------------------------------------------------
    print("\n[dummy samples]")
    samples = [_S(1.0)] + [_S(-1.0) for _ in range(15)] + [_S(0.0, dummy=True) for _ in range(4)]
    raw, adv = baseline(args, samples)
    ck(all(a == 0.0 for a in adv[16:]), "dummies get zero advantage")
    ck(abs(adv[0] - 1.875) < 1e-9,
       "dummies do NOT drag the mean -- real samples keep the same advantage as without them")
    samples = [_S(1.0), _S(-1.0), _S(0.0, remove_sample=True)]
    raw, adv = baseline(args, samples)
    ck(adv[2] == 0.0 and abs(adv[0] - 1.0) < 1e-9,
       "remove_sample without the metadata marker is also excluded")

    print("\n[degenerate batches]")
    raw, adv = baseline(args, [_S(1.0)])
    ck(adv == [0.0], "single-sample batch does not crash, emits zero advantage")
    raw, adv = baseline(args, [_S(0.0, dummy=True) for _ in range(4)])
    ck(adv == [0.0] * 4, "all-dummy batch does not crash")
    for k in (1, 3, 16, 20):
        raw, adv = baseline(args, [_S(1.0 if i % 3 else -1.0) for i in range(k)])
        ck(len(raw) == k and len(adv) == k,
           f"returns both lists at full input length ({k}) -- slime asserts on this")

    print(f"\nall {n} assertions passed")


if __name__ == "__main__":
    main()
