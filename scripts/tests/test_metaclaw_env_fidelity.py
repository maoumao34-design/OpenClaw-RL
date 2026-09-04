"""Regression assertions for environment fidelity (2026-09-04).

Two boundaries this project got wrong and has now fixed:

  1. `[Previous Feedback]` goes into the next round's QUERY -- the model reads
     it -- so it must be byte-identical to MetaClaw-official's own
     `_build_feedback_text`. Anything extra makes the benchmark easier and
     inflates our own Acc./Compl. The extra material now lives only in the OPD
     hint, which goes into teacher_tokens and the model never sees.
  2. A round the agent crashed on is scored 0 and kept in the denominator,
     the way `metaclaw-bench scoring` does it -- not dropped, which used to
     remove it from both numerator and denominator.

Runs the driver's real functions (needs MetaClaw-official checked out for the
official `_build_feedback_text` these are compared against).

Usage:
    python scripts/tests/test_metaclaw_env_fidelity.py [METACLAW_ROOT]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
DEFAULT_METACLAW = os.path.normpath(os.path.join(REPO_ROOT, "..", "MetaClaw-official"))


def main():
    metaclaw = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_METACLAW
    bench = os.path.join(metaclaw, "benchmark")
    if not os.path.isdir(bench):
        raise SystemExit(f"MetaClaw-official/benchmark not found: {bench}")
    sys.path.insert(0, bench)
    sys.path.insert(0, os.path.join(SCRIPTS_DIR, "metaclaw"))

    from src.infer.prompts import FORMAT_ERROR
    from src.infer.infer_cmd import _build_feedback_text

    import metaclaw_rollout_driver as drv

    n = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    # A real --dir mode round (the checker whose criterion we used to leak).
    dir_round = {
        "id": "r3", "type": "file_check",
        "eval": {"command": "python scripts/check_filename.py --dir day07 --pattern x"},
        "feedback": {"correct": "", "incorrect": "The filename is not correct."},
    }
    plain_round = {
        "id": "r4", "type": "file_check",
        "eval": {"command": "python scripts/check_iso8601.py day01/standup.json"},
        "feedback": {"correct": "", "incorrect": "Timestamps must be ISO8601."},
    }
    mc_round = {
        "id": "r5", "type": "multi_choice",
        "eval": {"answer": "A,C", "options": {"A": "a", "B": "b", "C": "c"}},
        "feedback": {},
    }

    failed = {"passed": False, "stdout": "FAIL: day07/report.md: bad name", "stderr": ""}
    failed_dirmode = dict(failed)
    upgraded = {**failed, "training_passed": True, "training_hint": "wrote 1 new file"}
    mc_format_fail = {"passed": False, "format_valid": False, "selected": []}

    # -- 1. the environment must be exactly official ------------------------
    print("[[Previous Feedback]] must be byte-identical to official]")
    for label, rr, sc in (
        ("failed --dir round", dir_round, failed_dirmode),
        ("failed plain file_check", plain_round, failed),
        ("passed round", plain_round, {"passed": True}),
        ("MC format failure", mc_round, mc_format_fail),
    ):
        ours = drv._build_next_round_feedback(rr, sc, "some model answer")
        official = _build_feedback_text(rr, sc)
        ck(ours == official, f"{label}: ours == official _build_feedback_text")

    ck(drv._FC_DIR_MODE_NOTE not in drv._build_next_round_feedback(
        dir_round, failed_dirmode, "x"),
       "the checker's acceptance criterion never reaches the agent")
    ck("FAIL: day07/report.md" not in drv._build_next_round_feedback(
        dir_round, failed_dirmode, "x"),
       "the checker's real stdout never reaches the agent")
    ck("some model answer" not in drv._build_next_round_feedback(
        mc_round, mc_format_fail, "some model answer"),
       "the model's own failed response is not echoed back to it")

    # The training_passed override used to make the environment disagree with
    # the official checker. It must not any more.
    ck(drv._build_next_round_feedback(dir_round, upgraded, "x")
       == _build_feedback_text(dir_round, upgraded),
       "a Phase 1 training_passed upgrade does not change what the agent sees")
    ck(drv._build_next_round_feedback(dir_round, upgraded, "x")
       == drv._build_next_round_feedback(dir_round, failed_dirmode, "x"),
       "...specifically, an upgraded round still reads as a failure to the agent, "
       "because the official checker failed it")

    # -- 2. the training side keeps everything ------------------------------
    print("\n[the OPD hint keeps the extra material]")
    hint_dir = drv._build_opd_hint(dir_round, failed_dirmode, "x")
    ck("FAIL: day07/report.md" in hint_dir, "hint carries the checker's real stdout")
    ck(drv._FC_DIR_MODE_NOTE in hint_dir, "hint carries the --dir criterion note")

    hint_plain = drv._build_opd_hint(plain_round, failed, "x")
    ck("FAIL: day07/report.md" in hint_plain, "non---dir round still gets the stdout")
    ck(drv._FC_DIR_MODE_NOTE not in hint_plain,
       "the --dir note is NOT appended to exact-date glob checks")

    hint_mc = drv._build_opd_hint(mc_round, mc_format_fail, "my bad answer")
    ck(FORMAT_ERROR in hint_mc, "MC hint starts from the official format-error text")
    ck("my bad answer" in hint_mc,
       "MC hint carries the model's own failed response, so a run of format "
       "failures no longer produces byte-identical distillation targets")
    ck(drv._build_opd_hint(mc_round, mc_format_fail, "") == FORMAT_ERROR,
       "with no answer text the MC hint is just the official text")

    silent = {"passed": False, "stdout": "", "stderr": ""}
    ck(drv._build_opd_hint(plain_round, silent, "x") == "",
       "a silent checker failure still yields an empty hint (RL-only), not the "
       "static feedback text -- that fallback was rejected on 2026-08-19")
    ck(drv._build_opd_hint(dir_round, silent, "x") == "",
       "...and the --dir note is not appended to an empty hint either")

    # -- 3. the denominator --------------------------------------------------
    print("\n[infra-failed rounds are scored 0 and kept]")
    src = open(os.path.join(SCRIPTS_DIR, "metaclaw", "metaclaw_rollout_driver.py"),
               encoding="utf-8").read()
    ck('"metrics": {"infra_failure": True},' in src,
       "an infra failure produces a real score record")
    ck(src.count("day_round_scores.append(") == 2,
       "both the scored and the infra-failure paths append")
    # Structural, not just "the text exists somewhere": the zero-score append
    # must be the `else` of the `official_score is not None` guard, so it
    # actually runs. A version that merely defines the record without wiring
    # it to that else would pass a substring check and still drop the round.
    infra = src.index('"metrics": {"infra_failure": True},')
    # rindex so the guard found is the code line, not the comment above it
    # that quotes the old version of this very line.
    guard = src.rindex("\n                    if official_score is not None:\n", 0, infra)
    code = [ln for ln in src[guard:infra].split("\n")
            if ln.strip() and not ln.strip().startswith("#")]
    ck(code[0].strip() == "if official_score is not None:"
       and code[1].strip() == "day_round_scores.append(official_score)"
       and code[2].strip() == "else:"
       and code[3].strip().startswith("day_round_scores.append("),
       "the zero-score record is the else-branch of the official_score guard, "
       f"with nothing else gating it (got: {[c.strip()[:40] for c in code[:4]]})")

    agg = drv._aggregate_acc_compl
    fc = lambda s: {"question_type": "file_check", "score": s}
    ck(agg([fc(1.0), fc(1.0)]) == (1.0, 1.0), "two passes -> 100%")
    acc, compl = agg([fc(1.0), fc(1.0), fc(0.0)])
    ck(abs(compl - 2 / 3) < 1e-9,
       "an infra-failed round scored 0 lands in the denominator and lowers Compl")
    ck(abs(agg([fc(1.0), fc(1.0)])[1] - 1.0) < 1e-9
       and abs(acc - 2 / 3) < 1e-9,
       "dropping it instead would have reported 100% -- the old inflation")

    print(f"\nall {n} assertions passed")


if __name__ == "__main__":
    main()
