"""The H-e probe must read the hint the proxy got and put it where training put it.

scripts/probe_opd_commit_direction.py measures which way the hint-conditioned
teacher pushes each response token. Its number is only meaningful if three
plumbing steps are exactly right, and each can go wrong without any error:

1. The hint text. It exists only in metaclaw_rollout.log, printed next to a
   "training hint" that is often identical, with httpx logger lines merged in
   from stderr right after it and, for the last round, the end-of-run report.
   Reading the wrong block gives a plausible-looking but wrong teacher.
2. The hint's position. Training appends it to the round's TASK message via
   _append_hint_to_messages and re-renders; the probe splices it into the
   recorded prompt_text instead. The two must produce the same bytes,
   including when a later tool result quotes the task text, and including
   .strip() at the ends. This is checked against the official function
   itself, extracted from openclaw_opd_api_server.py.
3. Which round a recorded turn belongs to, in a day-long transcript that
   carries every earlier round's task, and where a later day may repeat a
   question word for word.

The GPU part (two forward passes) is not exercised here; everything feeding it is.

Usage:
    OPENCLAW_RL_OFFICIAL=<path> python scripts/tests/test_probe_opd_commit_direction.py
"""

import ast
import copy
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS_DIR)
OFFICIAL = os.environ.get(
    "OPENCLAW_RL_OFFICIAL", os.path.join(os.path.dirname(ROOT), "OpenClaw-RL-official")
)
sys.path.insert(0, SCRIPTS_DIR)

import probe_opd_commit_direction as P  # noqa: E402


# --------------------------------------------------------------------------
# A rollout log written with the driver's own print formats
# (metaclaw_rollout_driver.py:1641-1665, 1750-1752, 1919).
# --------------------------------------------------------------------------

def day_header(test_id):
    return f"\n{'#' * 60}\n# Day {test_id}\n{'#' * 60}\n"


def round_block(round_id, session_id, test_id, query, answer, passed, training_passed,
                training_hint=None, opd_hint=None, logger_in_query=False):
    out = [f"\n  {'-' * 56}\n",
           f"  round={round_id}  session={session_id}  agent={test_id}\n",
           f"  {'-' * 56}\n"]
    q = query
    if logger_in_query:
        # stderr is unbuffered while stdout is block-buffered into the pipe,
        # so a logger line can land between two printed lines of the query.
        assert "\n" in q, "logger_in_query needs a multi-line query to land inside"
        first, rest = q.split("\n", 1)
        q = first + "\nINFO:httpx:HTTP Request: GET http://127.0.0.1:30000/health \"HTTP/1.1 200 OK\"\n" + rest
    out.append(f"  >> Query -> OpenClaw:\n{q}\n\n")
    out.append(f"  << OpenClaw -> Query:\n{answer}\n\n")
    out.append(f"  verdict: passed={passed}  training_passed={training_passed}  "
               f"agent_succeeded=True  official_score={1.0 if passed else 0.0:.3f}\n")
    if training_hint:
        out.append(f"  training hint (diff-based, goes into eval_score=-1 + next round's feedback):\n"
                   f"{training_hint}\n\n")
    out.append(f"INFO:__main__:\x1b[32m[MetaClawRollout] session={session_id} round={round_id} "
               f"passed={passed} training_passed={training_passed}\x1b[0m\n")
    if opd_hint:
        out.append(f"  OPD hint (training-side only, the agent never sees it):\n{opd_hint}\n\n")
        out.append("INFO:httpx:HTTP Request: POST http://127.0.0.1:30000/v1/chat/completions "
                   "\"HTTP/1.1 200 OK\"\n")
    return "".join(out)


Q1 = "Create a file named notes_2026-01-01.md in the day01 folder\nwith a one-line summary."
Q2 = ("Save the meeting metadata as JSON under day01/, one file per meeting, using the "
      "ISO-8601 date in the filename. " + "Each file must contain title, date and attendees. " * 8)
Q5 = Q2  # day05 repeats day01's question word for word

OPD_HINT_2 = ("FAIL: expected day01/2026-01-01_standup.json\n"
              "\n"
              "found: day01/standup.json (missing ISO-8601 date)")
TRAINING_HINT_2 = "round-local diff: nothing written under day01/ this round"
OPD_HINT_LAST = "FAIL: attendees must be a list, got str"

LOG = (
    day_header("day01")
    + round_block("r1", "sess-d01", "day01", Q1, "Done.", True, True, logger_in_query=True)
    + round_block("r2", "sess-d01", "day01", Q2, "Saved standup.json", False, False,
                  training_hint=TRAINING_HINT_2, opd_hint="  " + OPD_HINT_2 + "  ")
    + day_header("day05")
    + round_block("r9", "sess-d05", "day05", Q5, "Saved.", False, False, opd_hint=OPD_HINT_LAST)
    + "\n## MetaClaw report\n| Split | Days | Acc |\n|---|---|---|\n| Train | day1-30 | 41.1% |\n"
)


# --------------------------------------------------------------------------
# A Qwen3-style renderer: user/system/assistant content goes verbatim between
# "<|im_start|>{role}\n" and "<|im_end|>", consecutive tool messages share one
# user block and are wrapped in <tool_response>. That is what the probe's
# splice relies on.
# --------------------------------------------------------------------------

def render(messages):
    out, i = [], 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "tool":
            out.append("<|im_start|>user")
            while i < len(messages) and messages[i]["role"] == "tool":
                out.append("\n<tool_response>\n" + messages[i]["content"] + "\n</tool_response>")
                i += 1
            out.append("<|im_end|>\n")
            continue
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        i += 1
    out.append("<|im_start|>assistant\n<think>\n")
    return "".join(out)


def extract_official(names):
    path = os.path.join(OFFICIAL, "openclaw-opd", "openclaw_opd_api_server.py")
    if not os.path.exists(path):
        return None
    tree = ast.parse(open(path, encoding="utf-8").read())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"copy": copy}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), ns)
    return ns


def training_teacher_prompt(official, messages, probe, hint):
    """What the combine_select patch does: locate the task (last user message
    containing the probe), append the hint to it with the official helper,
    splice the rest back, re-render."""
    idx = None
    for k in range(len(messages) - 1, -1, -1):
        m = messages[k]
        if m["role"] == "user" and probe in official["_flatten_message_content"](m["content"]):
            idx = k
            break
    if idx is None:
        return None
    enhanced = official["_append_hint_to_messages"](messages[: idx + 1], hint) + messages[idx + 1:]
    return render(enhanced)


def main():
    n = 0
    skipped = 0

    def ck(cond, label):
        nonlocal n
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        n += 1
        print(f"  ok  {label}")

    # ---------------- 1. the hint text ----------------
    print("[rollout log parsing]")
    rounds = P.parse_rollout_log(LOG)
    by_id = {r.round_id: r for r in rounds}
    ck([r.round_id for r in rounds] == ["r1", "r2", "r9"], "three rounds, in order")
    ck(by_id["r1"].training_passed is True and by_id["r1"].hint == "" and not by_id["r1"].has_opd,
       "a passed round has no hint, and therefore no OPD")
    ck(by_id["r2"].training_passed is False, "the verdict line's training_passed is read")
    ck(by_id["r2"].hint == OPD_HINT_2,
       "the OPD hint is read exactly: inner blank line kept, ends stripped as the proxy does")
    ck(TRAINING_HINT_2 not in by_id["r2"].hint,
       "the 'training hint' printed just before is NOT taken as the OPD hint")
    ck("httpx" not in by_id["r2"].hint and "MetaClawRollout" not in by_id["r2"].hint,
       "logger lines merged in from stderr are dropped from the hint")
    ck(by_id["r1"].query == Q1,
       "a multi-line query is kept whole, and a logger line landing inside it is dropped")
    ck(by_id["r2"].query == Q2.strip(), "a long single-line query is exact")
    ck(by_id["r9"].hint == OPD_HINT_LAST and "report" not in by_id["r9"].hint.lower(),
       "the last round's hint stops before the end-of-run report")
    ck(by_id["r2"].probe == Q2[:300].strip()[:120] and len(by_id["r2"].probe) == 120,
       "probe = query[:300].strip()[:120], as the proxy computes it")

    print("\n[hint parse cross-check against the proxy's hint_len]")
    tlog = ("\x1b[36m[openclaw-rl-metaclaw-verdict-opd-hint] session=a turn=3 accepted K_i=1 "
            f"hint_len={len(OPD_HINT_2)}\x1b[0m\n"
            "[OpenClaw-Combine-Select] session=b turn=1 accepted K_i=2 hint_lens=[31, 40]\n"
            "\x1b[36m[openclaw-rl-metaclaw-verdict-opd-hint] session=c turn=7 accepted K_i=1 "
            f"hint_len={len(OPD_HINT_LAST)}\x1b[0m\n")
    lens = P.training_log_hint_lens(tlog)
    ck(lens == [len(OPD_HINT_2), len(OPD_HINT_LAST)],
       "only MetaClaw verdict hint_len lines are read, not the official multi-cand line")
    ck(P.hint_parse_agreement(rounds, lens) == 1.0, "a correct parse agrees 100%")
    ck(P.hint_parse_agreement(rounds, [len(OPD_HINT_2) + 1, len(OPD_HINT_LAST)]) == 0.5,
       "an off-by-one hint is caught as disagreement")
    short = [P.RoundInfo(0, "day01", "x", "s", hint="too short")]   # 9 chars
    ck(not short[0].has_opd and P.hint_parse_agreement(short, [9]) is None,
       "hints of 10 chars or fewer are excluded, as the proxy never distils them")

    # ---------------- 2. the hint's position ----------------
    print("\n[hint splice == training's _append_hint_to_messages]")
    official = extract_official({"_append_hint_to_messages", "_flatten_message_content"})
    probe2 = by_id["r2"].probe
    msgs = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": Q1},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "  " + Q2 + "  \n"},      # whitespace at both ends
        {"role": "assistant", "content": "Reading the folder first."},
        {"role": "tool", "content": "cat: " + Q2},              # a tool result quoting the task
        {"role": "assistant", "content": "Writing the file."},
        {"role": "tool", "content": "ok"},
    ]
    student = render(msgs)
    spliced = P.splice_hint(student, probe2, OPD_HINT_2)
    ck(spliced is not None and P.HINT_HEADER in spliced, "the probe finds the task and splices the hint in")
    ck(spliced.index(P.HINT_HEADER) < spliced.index("<tool_response>\ncat: "),
       "the hint goes on the task message, not on the later tool result that quotes it")
    if official is not None:
        want = training_teacher_prompt(official, msgs, probe2, OPD_HINT_2)
        ck(spliced == want, "byte-identical to the official helper applied the way training applies it")
        want_n = training_teacher_prompt(official, msgs, probe2, P.NEUTRAL_HINT)
        ck(P.splice_hint(student, probe2, P.NEUTRAL_HINT) == want_n, "same for the neutral control hint")
    else:
        skipped += 2
        print(f"  -- skipped official comparison: {OFFICIAL} not present (set OPENCLAW_RL_OFFICIAL)")
    ck(P.splice_hint(render(msgs[:3]), probe2, OPD_HINT_2) is None,
       "no task in the transcript -> None, where training would also skip the hint")
    tail = student[student.index("<|im_start|>assistant\nReading"):]
    ck(spliced.endswith(tail),
       "everything after the task message is byte-identical, so every response token has the same prefix")

    # ---------------- 3. which round owns a turn ----------------
    print("\n[round assignment]")
    r1, r2 = by_id["r1"], by_id["r2"]
    ck(P.assign_round(render(msgs[:3]), [r1, r2]) is r1, "a transcript holding only round 1's task -> r1")
    ck(P.assign_round(student, [r1, r2]) is r2, "once round 2's task is in, later turns belong to r2")
    both = render([{"role": "user", "content": "Last time: " + Q1 + "\n\nNow: " + Q2}])
    ck(P.assign_round(both, [r1, r2]) is r2,
       "feedback quoting the previous question inside the next task -> the later round wins the tie")

    def rec(session, turn, messages, response="thinking\n</think>\n\nok<|im_end|>\n"):
        return {"session_id": session, "turn": turn, "prompt_text": render(messages), "response_text": response}

    day01_upto_r1 = msgs[:3]
    day01_upto_r2 = msgs[:5]
    records = [
        rec("sess-d01", 1, day01_upto_r1),                    # r1, passed -> no OPD
        rec("sess-d01", 2, day01_upto_r2),                    # r2
        rec("sess-d01", 3, msgs),                             # r2, last turn
        rec("sess-d05", 1, [{"role": "user", "content": Q5}]),  # day05 repeat of r2's question
        rec("proxy-only-id", 1, day01_upto_r2),               # session id the driver never printed
        rec("sess-d01", 4, msgs, response="   "),             # empty reply
    ]
    records.append(dict(records[1]))                          # exact duplicate (live + archive)
    counts = Counter()
    turns = P.collect_turns(rounds, P.dedupe_records(records), ["01"], counts)
    got = sorted((t.session, t.turn, t.rnd.round_id, t.is_last) for t in turns)
    ck(got == [("proxy-only-id", 1, "r2", True), ("sess-d01", 2, "r2", False), ("sess-d01", 3, "r2", True)],
       "failed-round turns kept, last turn of the round marked, duplicate dropped")
    ck(counts["turn_round_has_no_opd_hint"] == 1, "the passed round's turn is counted and dropped")
    ck(counts["record_not_in_selected_days"] == 1,
       "a day05 turn repeating a day01 question is NOT pulled into day01")
    ck(counts["record_session_unmatched"] == 1, "an unknown session falls back to question matching, and is counted")
    ck(counts["record_empty"] == 1, "an empty reply is counted and skipped")

    # ---------------- token categories ----------------
    print("\n[token categories]")
    CLOSE, END, WS, NL = 99, 77, 5, 6
    resp = [10, 11, 12, CLOSE, WS, 20, WS, 21, 22, 23, 24, 25, 26, 27, 28, 29, END, NL]
    is_ws = [x in (WS, NL) for x in resp]
    c = P.categorize(resp, CLOSE, END, is_ws)
    ck(c["think"] == [0, 1, 2] and c["think_tail"] == [0, 1, 2] and c["close"] == [3],
       "thinking tokens and the </think> token")
    ck(c["act"] == list(range(4, 16)), "action = everything after </think> and before <|im_end|>")
    ck(c["act_head"] == [5, 7, 8, 9, 10, 11, 12, 13],
       "action head = first 8 non-whitespace tokens (the template's blank line is skipped)")
    ck(c["end"] == [16] and c["all"] == list(range(0, 17)),
       "<|im_end|> is its own category; the trailing newline is excluded from all")
    ck(P.categorize([10, 11, END], CLOSE, END, [False] * 3) is None, "no </think> -> not measurable")
    c2 = P.categorize([10, CLOSE, 20, 21], CLOSE, END, [False] * 4)
    ck(c2["end"] == [] and c2["act"] == [2, 3], "a cut-off reply with no <|im_end|> still has its action")
    long_think = list(range(1000, 1100)) + [CLOSE, END]
    ck(P.categorize(long_think, CLOSE, END, [False] * 102)["think_tail"] == list(range(36, 100)),
       "think_tail is the last 64 thinking tokens")

    d = [-0.1, -0.1, -0.1, -2.0] + [-0.5] * 12 + [-1.0, 0.0]
    m = P.cat_means(d, c)
    k = P.contrasts(m)
    ck(abs(k["close_minus_think"] - (-1.9)) < 1e-9 and abs(k["act_head_minus_think"] - (-0.4)) < 1e-9,
       "contrasts are category mean minus thinking mean")
    ck(math.isnan(P.contrasts(P.cat_means([0.0] * 4, c2))["end_minus_think"]),
       "a missing category gives NaN, not a fake zero")

    # ---------------- statistics ----------------
    print("\n[summary statistics]")
    s = P.summarize([-1.0, -2.0, -3.0, 1.0, float("nan")])
    ck(s["n"] == 4 and abs(s["mean"] + 1.25) < 1e-12 and s["frac_neg"] == 0.75, "NaN dropped; mean and sign share")
    ck(s["ci95"][0] <= s["mean"] <= s["ci95"][1], "the bootstrap interval brackets the mean")
    ck(P.summarize([float("nan")]) == {"n": 0}, "nothing measurable -> n=0")

    print(f"\nall {n} assertions passed" + (f" ({skipped} skipped)" if skipped else ""))


if __name__ == "__main__":
    main()
