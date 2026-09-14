#!/usr/bin/env python3
"""What shape is a MetaClaw round, really?

Three things came out of the OPD-target measurement that need explaining before
anything is changed, and all three are about the shape of a round rather than
about OPD:

  1. The last write sat one turn before the OPD target in 41 of 42 rounds.
     The likely reason is structural, not degradation: an agent loop only ends
     when the model stops calling tools, because a turn WITH a tool call gets a
     tool result and therefore another turn. If that is right, the final turn
     can never hold the write and the OPD target is wrong by construction.
     But one round did score offset 0, so "never" is not established. This
     counts how many rounds end on a turn that called any tool, and dumps the
     ones that do.

  2. 27 of 69 rounds contained no write at all. For a file-production
     benchmark that is a large fraction failing before producing anything.
     Are those truncated (hit the generation cap), or short prose rounds?

  3. 69 rounds over 133 turns is about 1.9 turns per round -- far below the
     3 to 6 this project has been assuming when reasoning about the 1/N
     advantage scaling and about per-token pressure being proportional to
     A/(N*T). If N is really about 2, the N part of that story is weak and the
     length part carries it. This reports the real distribution.

Usage:
    python3 analyze_round_shape.py <record.jsonl> [--dump N]

--dump N prints up to N full round summaries for the anomalous cases (a round
ending on a tool call, and a round with no write), because those are the ones
where a count alone does not say what happened.
"""
import collections
import json
import re
import sys

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

WRITE_TOOLS = {
    "write", "edit", "create", "str_replace", "str_replace_editor",
    "apply_patch", "multi_edit", "multiedit", "notebook_edit", "write_file",
    "create_file", "patch",
}
SHELL_TOOLS = {"exec", "bash", "shell", "run_command", "run", "terminal"}
SHELL_WRITE_RE = re.compile(
    r">>?\s*\S|\btee\b|\bcp\b|\bmv\b|\btouch\b|\bmkdir\b|\bsed\s+-i\b|"
    # Destructive ops count as writes as well -- see analyze_opd_target_turn.py.
    r"\brm\b|\brmdir\b|\bunlink\b|\btruncate\b|\bshred\b|"
    r"\bopen\s*\([^)]*['\"][wa]|\bjson\.dump|\bwriteFile|\bprintf\b.*>",
)


def tool_calls_of(rec):
    out = []
    for tc in rec.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        if fn.get("name"):
            out.append((str(fn["name"]),
                        json.dumps(fn.get("arguments"), ensure_ascii=False)))
    if out:
        return out
    for blob in TOOL_CALL_RE.findall(rec.get("response_text") or ""):
        try:
            p = json.loads(blob)
        except json.JSONDecodeError:
            continue
        name = p.get("name") or (p.get("function") or {}).get("name")
        if name:
            out.append((str(name), json.dumps(p.get("arguments"), ensure_ascii=False)))
    return out


def is_write(name, args):
    n = name.strip().lower()
    return n in WRITE_TOOLS or (n in SHELL_TOOLS and bool(SHELL_WRITE_RE.search(args or "")))


def verdict_of(rec):
    ns = rec.get("next_state")
    if not isinstance(ns, dict) or ns.get("role") != "user":
        return None
    c = ns.get("content")
    if isinstance(c, list):
        c = "".join(b.get("text", "") for b in c if isinstance(b, dict))
    if not isinstance(c, str):
        return None
    try:
        p = json.loads(c)
    except (TypeError, ValueError):
        return None
    return p if isinstance(p, dict) and p.get("metaclaw_verdict") is True else None


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
    rounds, trailing = [], 0
    for sid, recs in sessions.items():
        recs.sort(key=lambda r: r.get("turn", 0))
        cur = []
        for rec in recs:
            cur.append(rec)
            v = verdict_of(rec)
            if v is not None:
                rounds.append({"session": sid, "turns": cur, "verdict": v})
                cur = []
        if cur:
            trailing += len(cur)
    return rounds, trailing


def describe_turn(rec):
    resp = rec.get("response_text") or ""
    visible = THINK_RE.sub("", resp).strip()
    tcs = tool_calls_of(rec)
    return {
        "turn": rec.get("turn"),
        "resp_chars": len(resp),
        "think_chars": sum(len(x) for x in THINK_RE.findall(resp)),
        "visible_chars": len(visible),
        "tools": [n for n, _ in tcs],
        "has_write": any(is_write(n, a) for n, a in tcs),
        "visible_head": visible[:110].replace("\n", " "),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    path = sys.argv[1]
    dump_n = 3
    if "--dump" in sys.argv:
        i = sys.argv.index("--dump")
        if i + 1 < len(sys.argv):
            dump_n = int(sys.argv[i + 1])

    rounds, trailing = build_rounds(path)
    print("=" * 72)
    print(f"file: {path}")
    print(f"rounds with a verdict: {len(rounds)}   "
          f"turns in an unfinished trailing group: {trailing}")
    if not rounds:
        print("nothing to measure")
        return
    print("=" * 72)

    # -- 3. turns per round
    tpr = collections.Counter(len(r["turns"]) for r in rounds)
    total_turns = sum(len(r["turns"]) for r in rounds)
    print(f"\n[turns per round]  mean = {total_turns / len(rounds):.2f}")
    for k in sorted(tpr):
        print(f"  {k:2d} turn(s): {tpr[k]:4d}  ({100.0*tpr[k]/len(rounds):.1f}%)")
    print("  -> the 1/N advantage scaling divides by this. If it is near 2, the")
    print("     N term in A/(N*T) is small and length is what dominates.")

    # -- 1. does any round end on a tool call?
    ends_with_tool = [r for r in rounds if tool_calls_of(r["turns"][-1])]
    print(f"\n[rounds whose FINAL turn called any tool] "
          f"{len(ends_with_tool)}/{len(rounds)} "
          f"({100.0*len(ends_with_tool)/len(rounds):.1f}%)")
    if not ends_with_tool:
        print("  -> ZERO. The final turn never calls a tool, so it can never hold")
        print("     the write: the OPD target is wrong BY CONSTRUCTION, not")
        print("     because this run was degraded.")
    else:
        print("  -> not zero, so the 'structural' explanation is incomplete.")
        print("     Dumping these rounds -- read them before accepting any fix:")
        for r in ends_with_tool[:dump_n]:
            print(f"\n  --- session={r['session']} "
                  f"eval_score={r['verdict'].get('eval_score')} ---")
            for t in r["turns"]:
                print(f"      {describe_turn(t)}")

    # -- 2. rounds with no write
    no_write = [r for r in rounds
                if not any(any(is_write(n, a) for n, a in tool_calls_of(t))
                           for t in r["turns"])]
    print(f"\n[rounds with NO write anywhere] {len(no_write)}/{len(rounds)} "
          f"({100.0*len(no_write)/len(rounds):.1f}%)")
    if no_write:
        nw_tpr = collections.Counter(len(r["turns"]) for r in no_write)
        print(f"  their turn counts: {dict(sorted(nw_tpr.items()))}")
        resp_lens = sorted(len(r["turns"][-1].get("response_text") or "")
                           for r in no_write)
        print(f"  final-turn response chars: min={resp_lens[0]} "
              f"p50={resp_lens[len(resp_lens)//2]} max={resp_lens[-1]}")
        print("  -> a long final response with no tool call is the truncation /")
        print("     thinking-loop signature; a short one is the model giving up.")
        print("  Dumping:")
        for r in no_write[:dump_n]:
            print(f"\n  --- session={r['session']} "
                  f"eval_score={r['verdict'].get('eval_score')} "
                  f"hint={str(r['verdict'].get('hint'))[:70]!r} ---")
            for t in r["turns"]:
                print(f"      {describe_turn(t)}")

    # -- where writes sit, among rounds that have one
    with_write = [r for r in rounds if r not in no_write]
    offs = collections.Counter()
    for r in with_write:
        last = max(i for i, t in enumerate(r["turns"])
                   if any(is_write(n, a) for n, a in tool_calls_of(t)))
        offs[(len(r["turns"]) - 1) - last] += 1
    print(f"\n[last-write offset from the final turn] over {len(with_write)} rounds")
    for k in sorted(offs):
        print(f"  offset {k:2d}: {offs[k]:4d}")
    print("=" * 72)


if __name__ == "__main__":
    main()
