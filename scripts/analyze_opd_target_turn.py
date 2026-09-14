#!/usr/bin/env python3
"""Does the OPD teacher forward land on the turn that produced the artifact?

OPD's semantics: the hint goes at the start of the question, and what gets
trained is the final artifact -- the written file. The code implements that by
firing on the LAST REAL GENERATING TURN of the round (the one whose next_state
is the verdict), splicing the hint at the round's task, and running the teacher
forward over that turn's own response.

That is only meaningful if the write the checker judges actually happens on
that turn. If the file was written two turns earlier and the last turn merely
says "done, I've written it", the hint is being used to correct a sentence of
acknowledgement, and OPD teaches nothing.

This measures it on the production record. A round's final turn is identified
exactly, not heuristically: `_flush_pending_record` stores the next request's
last message as `next_state`, so the final turn of a round is the one whose
next_state parses as JSON with metaclaw_verdict true.

The tool-name histogram is printed BEFORE any classification, because the
write-detecting name set is a guess about this agent's tools and the histogram
is what shows whether the guess missed something. Read it before trusting the
verdict.

Usage:
    python3 analyze_opd_target_turn.py <record.jsonl>
"""
import collections
import json
import re
import sys

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Tools that produce or modify a file directly.
WRITE_TOOLS = {
    "write", "edit", "create", "str_replace", "str_replace_editor",
    "apply_patch", "multi_edit", "multiedit", "notebook_edit", "write_file",
    "create_file", "patch",
}
# A shell tool can also write. These are the shapes that do.
SHELL_TOOLS = {"exec", "bash", "shell", "run_command", "run", "terminal"}
SHELL_WRITE_RE = re.compile(
    r">>?\s*\S|\btee\b|\bcp\b|\bmv\b|\btouch\b|\bmkdir\b|\bsed\s+-i\b|"
    # Destructive ops count as writes as well. A round can fail because a later
    # turn deleted or clobbered what an earlier one produced, and then THAT
    # turn is what the hint is about -- targeting the last write has to mean
    # the last turn that CHANGED the workspace, not just the last one to create.
    r"\brm\b|\brmdir\b|\bunlink\b|\btruncate\b|\bshred\b|"
    r"\bopen\s*\([^)]*['\"][wa]|\bjson\.dump|\bwriteFile|\bprintf\b.*>",
)


def tool_calls_of(rec):
    """Every tool call in one turn, as (name, argstring).

    Reads both the structured field and the inline <tool_call> blocks, because
    the record carries `tool_calls` only when the proxy parsed it out, while
    response_text always has the raw form.
    """
    out = []
    for tc in rec.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = fn.get("name")
        if name:
            out.append((str(name), json.dumps(fn.get("arguments"), ensure_ascii=False)))
    if out:
        return out
    for blob in TOOL_CALL_RE.findall(rec.get("response_text") or ""):
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        name = payload.get("name") or (payload.get("function") or {}).get("name")
        if name:
            out.append((str(name),
                        json.dumps(payload.get("arguments"), ensure_ascii=False)))
    return out


def is_write(name, args):
    n = name.strip().lower()
    if n in WRITE_TOOLS:
        return True
    if n in SHELL_TOOLS and SHELL_WRITE_RE.search(args or ""):
        return True
    return False


def verdict_of(rec):
    """The verdict payload this turn's next_state carries, or None."""
    ns = rec.get("next_state")
    if not isinstance(ns, dict) or ns.get("role") != "user":
        return None
    content = ns.get("content")
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return None
    if isinstance(parsed, dict) and parsed.get("metaclaw_verdict") is True:
        return parsed
    return None


def build_rounds(path):
    sessions = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("session_id") is not None:
                sessions[rec["session_id"]].append(rec)

    rounds = []
    for sid, recs in sessions.items():
        recs.sort(key=lambda r: r.get("turn", 0))
        cur = []
        for rec in recs:
            cur.append(rec)
            v = verdict_of(rec)
            if v is not None:
                rounds.append({"session": sid, "turns": cur, "verdict": v})
                cur = []
        # a trailing group with no verdict is an unfinished round: not a round
    return sessions, rounds


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    path = sys.argv[1]
    sessions, rounds = build_rounds(path)

    print("=" * 72)
    print(f"file: {path}")
    print(f"sessions: {len(sessions)}   rounds with a verdict: {len(rounds)}")

    if not rounds:
        print("\nNo round in this file has a verdict in next_state, so the OPD")
        print("target turn cannot be identified. Either this is an eval run, or")
        print("the last turn of each round is still pending and was never")
        print("flushed. Nothing measured -- do not read this as a negative.")
        return

    print("=" * 72)
    print("\n[tool names seen] -- read this BEFORE the verdict below")
    hist = collections.Counter()
    wr_hist = collections.Counter()
    for rd in rounds:
        for t in rd["turns"]:
            for name, args in tool_calls_of(t):
                hist[name] += 1
                if is_write(name, args):
                    wr_hist[name] += 1
    if not hist:
        print("  NONE -- no tool call parsed from any turn. The classifier")
        print("  cannot work; check the record's response_text shape first.")
        return
    for name, c in hist.most_common():
        mark = f"  <- counted as write ({wr_hist[name]}/{c})" if wr_hist[name] else ""
        print(f"  {c:5d}  {name}{mark}")

    print("\n[where is the last write, relative to the OPD target turn?]")
    print("  0 = on the target turn itself (OPD lands on the artifact)")
    dist = collections.Counter()
    no_write = 0
    hinted = 0
    hinted_dist = collections.Counter()
    for rd in rounds:
        turns = rd["turns"]
        last_write = None
        for i, t in enumerate(turns):
            if any(is_write(n, a) for n, a in tool_calls_of(t)):
                last_write = i
        has_hint = len(str(rd["verdict"].get("hint") or "")) > 10
        if has_hint:
            hinted += 1
        if last_write is None:
            no_write += 1
            if has_hint:
                hinted_dist["no write at all"] += 1
            continue
        offset = (len(turns) - 1) - last_write
        dist[offset] += 1
        if has_hint:
            hinted_dist[offset] += 1

    total_w = sum(dist.values())
    for off in sorted(dist):
        print(f"  offset {off:2d} : {dist[off]:5d}  ({100.0*dist[off]/max(total_w,1):.1f}%)")
    print(f"  no write found in the whole round: {no_write}")

    print(f"\n[rounds carrying a usable hint] {hinted}/{len(rounds)} "
          f"({100.0*hinted/len(rounds):.1f}%)")
    if hinted:
        print("  their last-write offsets:")
        for k in sorted(hinted_dist, key=lambda x: (isinstance(x, str), x)):
            print(f"    {k}: {hinted_dist[k]}")

    on_target = hinted_dist.get(0, 0)
    print("\n" + "=" * 72)
    if not hinted:
        print("VERDICT: no round here carries a hint, so nothing can be said about")
        print("         whether OPD lands on the artifact. Need a run with failures.")
    elif on_target / hinted >= 0.8:
        print(f"VERDICT: {on_target}/{hinted} hinted rounds wrote on the target turn.")
        print("         OPD lands on the artifact. The current design is sound and")
        print("         needs no change.")
    elif on_target / hinted <= 0.3:
        print(f"VERDICT: only {on_target}/{hinted} hinted rounds wrote on the target")
        print("         turn. The teacher forward is mostly correcting a turn that")
        print("         did NOT produce the file -- OPD is aimed at the wrong turn,")
        print("         and it should follow the last WRITE instead of the last turn.")
    else:
        print(f"VERDICT: mixed -- {on_target}/{hinted} hinted rounds wrote on the")
        print("         target turn. Aiming OPD at the last write rather than the")
        print("         last turn would strictly improve it, without changing the")
        print("         rollout side.")
    print("=" * 72)


if __name__ == "__main__":
    main()
