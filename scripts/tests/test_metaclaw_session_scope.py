"""Regression assertions for METACLAW_SESSION_SCOPE (2026-09-05).

The switch exists to settle one question: our zero-training K=0 scores Compl
12.1% with a 4B, level with the paper's GPT-5.2 baseline (14.7%) and far above
its Kimi-K2.5 baseline (2.0%). Everything else has been ruled out, and
per-round session isolation is what is left. Setting the scope to "day"
reproduces MetaClaw-official's actual behaviour (one session per day) so the
difference can be measured instead of argued about.

Three things have to move together, or the "day" run is not the official
protocol and is not a clean single-variable comparison:
  - the session id (per day, not per round),
  - session_done on the verdict (only the day's last round may close it),
  - the infra-failure session close (same).

Usage:
    METACLAW_ROOT=<path> python scripts/tests/test_metaclaw_session_scope.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
DRIVER = os.path.join(SCRIPTS_DIR, "metaclaw", "metaclaw_rollout_driver.py")
LAUNCHER = os.path.join(SCRIPTS_DIR, "metaclaw", "run_metaclaw_migration_modelfactory.sh")


def _scope_value(scope):
    """Import the driver with a given scope and read back what it resolved to."""
    env = dict(os.environ)
    env["METACLAW_SESSION_SCOPE"] = scope
    env.setdefault("METACLAW_ROOT", os.path.normpath(
        os.path.join(SCRIPTS_DIR, "..", "..", "MetaClaw-official")))
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "import metaclaw_rollout_driver as d;"
        "print(d.METACLAW_SESSION_SCOPE)" % os.path.join(SCRIPTS_DIR, "metaclaw")
    )
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True)


def main():
    src = open(DRIVER, encoding="utf-8").read()
    launcher = open(LAUNCHER, encoding="utf-8").read()

    n = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    print("[the switch itself]")
    r = _scope_value("round")
    ck(r.returncode == 0 and r.stdout.strip() == "round",
       f"default/explicit 'round' resolves (got {r.stdout.strip()!r} {r.stderr[-200:]})")
    d = _scope_value("day")
    ck(d.returncode == 0 and d.stdout.strip() == "day", "'day' resolves")
    bad = _scope_value("per-day")
    ck(bad.returncode != 0 and "METACLAW_SESSION_SCOPE" in bad.stderr,
       "an unrecognised value fails loudly at import instead of silently "
       "falling back -- a typo here would silently invalidate the comparison")

    print("\n[all three sites move together]")
    ck('round_session_id = f"{_SESSION_ID_PREFIX}{test_id}"' in src,
       "day scope uses one session id per day, matching official's test['session']")
    ck(src.count('if METACLAW_SESSION_SCOPE == "day":') == 1,
       "the session id is the only place that branches on 'day'")
    ck('True if METACLAW_SESSION_SCOPE == "round" else is_last_round' in src,
       "the verdict only closes the session on the day's last round under day scope")
    ck('if METACLAW_SESSION_SCOPE == "round" or is_last_round:' in src,
       "the infra-failure close is gated the same way")
    ck("is_last_round = idx == len(rounds) - 1" in src,
       "is_last_round is computed from the round loop")

    # Under day scope, closing the session mid-day would force-drop pending
    # turns that later rounds of the SAME session still need -- that is the
    # concrete failure this gating prevents, so check it is not left ungated.
    close_idx = src.index("await _send_session_close_only(")
    guard = src.rindex("if METACLAW_SESSION_SCOPE ==", 0, close_idx)
    ck(close_idx - guard < 600,
       "the session close is inside the scope guard, not merely near it")

    print("\n[launcher wiring]")
    for frag, label in (
        ("METACLAW_SESSION_SCOPE=${METACLAW_SESSION_SCOPE:-round}",
         "declared with the safe default"),
        ("metaclaw_session_scope: ${METACLAW_SESSION_SCOPE}",
         "recorded in RUN_MANIFEST so a run's scope is recoverable afterwards"),
        ('METACLAW_SESSION_SCOPE: ${METACLAW_SESSION_SCOPE}',
         "echoed at startup"),
        ('METACLAW_SESSION_SCOPE="${METACLAW_SESSION_SCOPE}"',
         "explicitly passed to the driver rather than relying on inheritance"),
    ):
        ck(frag in launcher, f"launcher: {label}")

    print("\n[the default is unchanged]")
    ck(_scope_value("").stdout.strip() in ("round", ""),
       "an empty value does not silently become 'day'")
    env = dict(os.environ)
    env.pop("METACLAW_SESSION_SCOPE", None)
    env.setdefault("METACLAW_ROOT", os.path.normpath(
        os.path.join(SCRIPTS_DIR, "..", "..", "MetaClaw-official")))
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s');"
         "import metaclaw_rollout_driver as d;"
         "print(d.METACLAW_SESSION_SCOPE)" % os.path.join(SCRIPTS_DIR, "metaclaw")],
        env=env, capture_output=True, text=True)
    ck(r.returncode == 0 and r.stdout.strip() == "round",
       "unset defaults to 'round', so existing runs are unaffected")

    print(f"\nall {n} assertions passed")


if __name__ == "__main__":
    main()
