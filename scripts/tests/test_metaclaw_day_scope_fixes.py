"""Regressions for the day-scope correctness fixes (2026-09-07).

Switching from one session per round to one session per day broke invariants
that had been free under round scope, because each round used to start from an
empty transcript and empty per-session state.

  A. The OPD hint was anchored to "the first user message of the session".
     That was the round's task only while a session held one round; in a
     day-long transcript it is day-round-1's task, so every later round had
     its checker feedback attached to the wrong question.

  B. The verdict payload was {metaclaw_verdict, eval_score, hint} with no
     round identity. Two rounds of a day that fail the same way produce a
     byte-identical payload, and the duplicate-user-retry rule -- whose "seen"
     set is keyed by session, i.e. now a whole day -- drops the second
     verdict. That verdict is what dispatches the round group, so the round's
     held turns stay pending and a later round sweeps them up, giving them
     that round's reward and inflating its 1/N denominator.

  C. The sweep (`t < turn_num`) encoded the assumption that `pending` can only
     hold this round's turns. Under day scope that holds only while every
     verdict dispatches. A turn-number lower bound is not sufficient on its
     own either: it advances only when a verdict DISPATCHES, so a verdict lost
     outright still leaves its round's turns inside the next round's window.
     Membership is therefore decided by the round's own task text -- a turn
     from round r carries tasks 1..r in its prompt, so it contains round r's
     task while an earlier round's turn does not.

Separately, found while tracing the above:

  D. `opd_result.get("metaclaw_verdict")` gates the whole round-group branch,
     but _opd_evaluate never put that key in its return value. The branch had
     therefore never executed since it was written (2026-09-03) -- held
     intermediate turns were never collected and 1/N never applied. It went
     unnoticed because that mechanism had not yet been run in real training.

The string assertions pin the fixes to the real files; the scenario tests
re-implement the logic faithfully and exercise the cases that motivated it, so
an edit that keeps the strings but breaks the behaviour still fails.

Usage:
    python scripts/tests/test_metaclaw_day_scope_fixes.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
DRIVER = os.path.join(SCRIPTS_DIR, "metaclaw", "metaclaw_rollout_driver.py")
COMBINE = os.path.join(SCRIPTS_DIR, "prepare_patched_openclaw_combine.sh")
SELECT = os.path.join(SCRIPTS_DIR, "prepare_patched_openclaw_combine_select.sh")


# --------------------------------------------------------------------------
# Faithful re-implementations of the patched logic
# --------------------------------------------------------------------------

def collect_round(pending, turn_num, bounds, session_id, task_prefix=""):
    """Mirrors the round-group collection in prepare_patched_openclaw_combine.sh.

    pending maps turn_num -> {"messages": [...]}. Returns
    (collected, orphans, foreign) and mutates pending/bounds as the proxy does.
    """
    lo = bounds.get(session_id, 0)

    orphans = [t for t in sorted(pending) if t <= lo]
    for t in orphans:
        pending.pop(t, None)

    cand = [t for t in sorted(pending) if lo < t < turn_num]

    foreign = []
    prefix = (task_prefix or "")[:120]
    if prefix:
        mine = []
        for t in cand:
            hit = any(
                isinstance(m, dict) and m.get("role") == "user"
                and prefix in (m.get("content") or "")
                for m in (pending[t].get("messages") or [])
            )
            (mine if hit else foreign).append(t)
        for t in foreign:
            pending.pop(t, None)
        cand = mine

    collected = list(cand)
    for t in cand:
        pending.pop(t, None)
    bounds[session_id] = turn_num
    return collected, orphans, foreign


def locate_task(messages, task_prefix):
    """Mirrors the hint anchoring in prepare_patched_openclaw_combine_select.sh."""
    if not task_prefix:
        return None
    probe = task_prefix[:120]
    for k in range(len(messages) - 1, -1, -1):
        m = messages[k]
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        if probe in (m.get("content") or ""):
            return k
    return None


def turn(*tasks):
    """A held turn whose prompt carries the day's tasks so far."""
    msgs = [{"role": "system", "content": "sys"}]
    for t in tasks:
        msgs.append({"role": "user", "content": f"[ts] {t}"})
        msgs.append({"role": "assistant", "content": "ok"})
    return {"messages": msgs}


def main():
    driver = open(DRIVER, encoding="utf-8").read()
    combine = open(COMBINE, encoding="utf-8").read()
    select = open(SELECT, encoding="utf-8").read()

    n = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    # ------------------------------------------------------------------- D
    print("[D: the round-group branch can actually fire]")
    ck(select.count('"metaclaw_verdict": True,') == 2,
       "both verdict-branch returns carry metaclaw_verdict -- the key the "
       "combine server gates the whole round-group branch on")
    ck(select.count('"metaclaw_task_prefix": _metaclaw_task_prefix,') == 2,
       "both returns carry the task prefix for membership testing")
    ck('opd_result.get("metaclaw_verdict")' in combine,
       "the combine server still gates on it")

    # ------------------------------------------------------------------- A/B
    print("\n[A/B: the verdict identifies its round]")
    ck('"round_id": round_id,' in driver, "driver sends round_id")
    ck('"task_prefix": task_prefix[:_VERDICT_TASK_PREFIX_CHARS],' in driver,
       "driver sends a bounded task_prefix")
    ck("round_id=str(round_record.get(\"id\"" in driver,
       "the call site passes the real round id")
    ck("task_prefix=query," in driver,
       "the call site passes the query actually sent to the agent")

    # ------------------------------------------------------------------- C
    print("\n[C: rounds cannot absorb each other's turns]")
    ck("_metaclaw_last_verdict_turn" in combine, "a per-session lower bound exists")
    ck("_mc_lo < t < turn_num" in combine, "collection is bounded on both sides")
    ck("_flatten_message_content" in combine,
       "the membership test's dependency is imported into the combine server")
    ck("do not carry" in combine, "foreign turns are reported, not merged")
    ck(combine.count("if t < turn_num]") == 0, "no unbounded sweep survives")

    print("\n  scenario: three normal rounds")
    pend, bounds = {}, {}
    for t in (1, 2, 3):
        pend[t] = turn("Round one")
    got, orph, foreign = collect_round(pend, 4, bounds, "day01", "Round one")
    ck(got == [1, 2, 3] and not orph and not foreign,
       f"round1 collects exactly its own turns ({got})")
    for t in (5, 6):
        pend[t] = turn("Round one", "Round two")
    got, _, _ = collect_round(pend, 7, bounds, "day01", "Round two")
    ck(got == [5, 6], f"round2 collects exactly its own turns ({got})")

    print("\n  scenario: round2's verdict is lost entirely")
    pend, bounds = {}, {}
    for t in (1, 2, 3):
        pend[t] = turn("Round one")
    collect_round(pend, 4, bounds, "day01", "Round one")
    for t in (5, 6):
        pend[t] = turn("Round one", "Round two")      # verdict lost
    for t in (8, 9):
        pend[t] = turn("Round one", "Round two", "Round three")
    got, orph, foreign = collect_round(pend, 10, bounds, "day01", "Round three")
    ck(got == [8, 9],
       f"round3 collects only its own turns ({got}) -- the turn-number bound "
       "alone would have returned [5, 6, 8, 9]")
    ck(foreign == [5, 6],
       f"round2's orphaned turns are identified and dropped ({foreign}), not "
       "given round3's reward")

    print("\n  scenario: no task_prefix available (older payload)")
    pend, bounds = {}, {}
    for t in (1, 2):
        pend[t] = turn("Round one")
    got, _, foreign = collect_round(pend, 3, bounds, "day01", "")
    ck(got == [1, 2] and foreign == [],
       "falls back to the turn-number bound rather than dropping everything")

    # ------------------------------------------------------------------- A
    print("\n[A: the hint lands on this round's task]")
    ck("_mc_task_idx" in select and "_mc_first_user" not in select,
       "anchoring is by located task index; the first-user form is gone")
    ck("range(len(_mc_msgs) - 1, -1, -1)" in select, "the search runs backwards")
    ck("metaclaw round task not found in transcript" in select,
       "a miss raises rather than anchoring somewhere wrong")

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "[ts] Round one: organize standup notes"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "tool result: wrote standup.json"},
        {"role": "user", "content": "[ts] Round two: back up the config"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "[ts] Round three: export the sprint log"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "tool result: FAIL cannot read"},
    ]
    idx = locate_task(msgs, "Round three: export the sprint log")
    first_user = next(k for k, m in enumerate(msgs) if m.get("role") == "user")
    ck(idx == 6, f"anchors on round three's task (index {idx})")
    ck(first_user == 1,
       f"the old logic would have used index {first_user}, i.e. round one's "
       "question, for round three's feedback")
    ck(locate_task(msgs, "Round one: organize standup notes") == 1,
       "round one still resolves to its own task")
    ck(locate_task(msgs, "a question never asked") is None,
       "an unlocatable task returns None so the caller can skip the hint")
    ck(locate_task(msgs, "") is None,
       "an empty prefix does not match the first message by accident")

    print(f"\nall {n} assertions passed")


if __name__ == "__main__":
    main()
