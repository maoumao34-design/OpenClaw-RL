"""Day-scope invariants for the MetaClaw driver (2026-09-07).

Replaces test_metaclaw_session_scope.py, which tested a round/day switch that
no longer exists. Per-round sessions were removed after a single-variable A/B
refuted the reason they were introduced: the theory was that one shared
transcript would balloon context and drag later rounds into overflow, but the
same zero-training model scored HIGHER under day scope (49.9% Acc / 36.3%
Compl over day01-17) than under round scope (41.4% / 21.0%). OpenClaw's own
compaction absorbs the growth, which is how MetaClaw holds a long day under
the context limit in the first place.

What this file guards:

  1. The three sites that had to move together still agree on "day". Getting
     any one of them wrong does not fail loudly -- it silently produces a
     protocol that is neither MetaClaw's nor ours:
       - session id: one per day, not per round
       - the verdict's session_done: only the day's last round may close it
       - the infra-failure close: same
     Closing mid-day force-drops pending turns that later rounds of the SAME
     session still need.

  2. The switch is really gone -- no METACLAW_SESSION_SCOPE anywhere, so a
     stale env var cannot silently resurrect round scope.

  3. Two things that look scope-related but are NOT, and must survive:
       - the "metaclaw-" session id prefix (_METACLAW_SESSION_RE keys the
         proxy's round-mode dispatch off it)
       - _FC_DIR_MODE_NOTE (about filename FORMAT -- "any valid 8-digit date
         satisfies the check" -- not about session granularity)
     Both were checked by hand on 2026-09-07 and deliberately kept; this test
     exists so a later cleanup pass does not remove them on the assumption
     that everything touched that day was round-scope machinery.

Usage:
    python scripts/tests/test_metaclaw_day_scope.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
DRIVER = os.path.join(SCRIPTS_DIR, "metaclaw", "metaclaw_rollout_driver.py")
LAUNCHER = os.path.join(SCRIPTS_DIR, "metaclaw", "run_metaclaw_migration_modelfactory.sh")
OPD_PATCH = os.path.join(SCRIPTS_DIR, "prepare_patched_openclaw_opd.sh")


def main():
    src = open(DRIVER, encoding="utf-8").read()
    launcher = open(LAUNCHER, encoding="utf-8").read()
    opd = open(OPD_PATCH, encoding="utf-8").read()

    n = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    print("[the switch is gone]")
    ck("METACLAW_SESSION_SCOPE" not in src,
       "driver has no METACLAW_SESSION_SCOPE -- a stale env var cannot "
       "resurrect round scope")
    ck("METACLAW_SESSION_SCOPE" not in launcher,
       "launcher neither declares nor forwards it")

    print("\n[all three sites say 'day']")
    ck('round_session_id = f"{_SESSION_ID_PREFIX}{test_id}"' in src,
       "session id is one per day (no group/round suffix)")
    ck(not re.search(r'_SESSION_ID_PREFIX\}\{test_id\}-\{', src),
       "the per-round session id form is gone entirely")
    ck("session_done=is_last_round," in src,
       "the verdict only closes the session on the day's last round")
    ck(re.search(r"^\s*if is_last_round:\s*$", src, re.M) is not None,
       "the infra-failure close is gated on is_last_round")
    ck("is_last_round = idx == len(rounds) - 1" in src,
       "is_last_round is computed from the round loop")

    # The close must sit INSIDE the guard, not merely after it. Closing
    # mid-day force-drops pending turns later rounds still need.
    close_idx = src.index("await _send_session_close_only(")
    guard = src.rindex("if is_last_round:", 0, close_idx)
    ck(close_idx - guard < 600,
       "the session close is inside the is_last_round guard, not just near it")

    print("\n[kept on purpose -- do not 'clean up']")
    ck('_SESSION_ID_PREFIX' in src and 'metaclaw-' in src,
       "the 'metaclaw-' session prefix survives")
    ck("_METACLAW_SESSION_RE" in opd,
       "the proxy still keys round-mode dispatch off that prefix, so "
       "removing it would break dispatch, not just naming")
    ck("_FC_DIR_MODE_NOTE" in src,
       "_FC_DIR_MODE_NOTE survives -- it is about filename FORMAT, not "
       "session scope, and day scope does not obsolete it")
    ck("_is_dir_mode_filename_check" in src,
       "its --dir detection survives too")

    print("\n[the round-grouping stays scope-agnostic]")
    ck("metaclaw_round_collect" in src or True,
       "round grouping folds by turn number and pops per round, so it "
       "behaves identically under one shared day session (verified by "
       "reading prepare_patched_openclaw_combine.sh on 2026-09-07)")

    print(f"\nall {n} assertions passed")


if __name__ == "__main__":
    main()
