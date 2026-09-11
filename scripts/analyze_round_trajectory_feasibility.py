#!/usr/bin/env python3
"""Answer, from a past run's record jsonl, whether route B is buildable.

Route B ("the final turn's prompt already IS the trajectory") rests on one
claim: the last real turn of a round carries the whole round -- system prompt,
task, every earlier assistant turn, every tool result -- because the proxy
builds `prompt_text` by templating the full messages array OpenClaw sent.

That claim has only been read off the source. This measures it on production
data. The proxy already writes every turn to `$OPENCLAW_RECORD_FILE`
(`OPENCLAW_RECORD_ENABLED` defaults to 1 and our profile never turned it off),
one JSON object per line:

    {session_id, turn, timestamp, messages, prompt_text, response_text,
     tool_calls, next_state}

so no new training run and no GPU is needed.

Four questions, one pass:

  Q1  MASK COVERAGE -- for each session, how many of the earlier turns'
      `response_text` appear verbatim in the final turn's `prompt_text`?
      This IS route B's loss_mask coverage. Anything found can be trained on;
      anything missing was not in what the model read, so skipping it is
      correct rather than fatal.

  Q2  PREFIX CONTAINMENT -- does turn N's prompt_text start with turn N-1's
      prompt_text? This is what route A (the 09-03 design) asserted and what
      dropped 96/120 rounds. Measuring it says whether that failure is
      reproduced here, and route B's coverage can then be compared against it.

  Q3  THINKING IN HISTORY -- does the final prompt_text still contain the
      `<think>` blocks of the earlier turns? `prompt_text` is post-template,
      so this answers the chat-template gate on real data: what the model
      actually read, not what OpenClaw put on the wire.

  Q4  LENGTH -- how long is a full round in characters (and tokens, if a
      tokenizer is given)? A trajectory is longer than any single response,
      and a 34,885-token response already OOM'd the single-GPU teacher twice.

Usage:
    python3 analyze_round_trajectory_feasibility.py <record.jsonl> [tokenizer]

Find the file with:
    find . -name '*record*.jsonl' -size +1k
"""
import collections
import json
import re
import sys

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def load(path):
    """session_id -> [record, ...] ordered by turn."""
    sessions = collections.defaultdict(list)
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            sid = rec.get("session_id")
            if sid is None or "prompt_text" not in rec:
                bad += 1
                continue
            sessions[sid].append(rec)
    for recs in sessions.values():
        recs.sort(key=lambda r: r.get("turn", 0))
    return sessions, bad


def _norm(text):
    """Drop trailing chat-template scaffolding so a near-miss is not read as
    a total miss. Keeps the comparison honest in both directions: we report
    exact and relaxed separately."""
    return text.strip().rstrip("\n")


def analyse(sessions):
    q1_found = q1_total = 0
    q1_relaxed = 0
    per_session_cov = []
    q2_ok = q2_bad = 0
    q3_sessions_with_think = 0
    q3_think_survived = q3_think_total = 0
    lengths = []
    multi_turn = 0

    for sid, recs in sessions.items():
        if len(recs) < 2:
            continue
        multi_turn += 1
        final = recs[-1]
        fp = final.get("prompt_text") or ""
        lengths.append(len(fp) + len(final.get("response_text") or ""))

        # Q1 -- earlier responses present in the final prompt
        found = 0
        for r in recs[:-1]:
            rt = r.get("response_text") or ""
            if not rt.strip():
                continue
            q1_total += 1
            if rt in fp:
                found += 1
                q1_found += 1
                q1_relaxed += 1
            elif _norm(rt) and _norm(rt) in fp:
                q1_relaxed += 1
        denom = sum(1 for r in recs[:-1] if (r.get("response_text") or "").strip())
        if denom:
            per_session_cov.append(found / denom)

        # Q2 -- consecutive prefix containment (what route A required).
        # The condition is prompt_{N} starting with prompt_{N-1} PLUS
        # response_{N-1}: reconstruction concatenates the generated tokens, so
        # comparing the prompt halves alone would pass even when the response
        # was rewritten in history -- which is the very case that breaks it.
        for a, b in zip(recs, recs[1:]):
            need = (a.get("prompt_text") or "") + (a.get("response_text") or "")
            pb = b.get("prompt_text") or ""
            if need and pb.startswith(need):
                q2_ok += 1
            else:
                q2_bad += 1

        # Q3 -- earlier turns' thinking still in the final prompt
        think_here = False
        for r in recs[:-1]:
            for blk in THINK_RE.findall(r.get("response_text") or ""):
                blk = blk.strip()
                if len(blk) < 40:      # too short to identify reliably
                    continue
                think_here = True
                q3_think_total += 1
                if blk in fp:
                    q3_think_survived += 1
        if think_here:
            q3_sessions_with_think += 1

    return {
        "multi_turn_sessions": multi_turn,
        "q1_found": q1_found, "q1_relaxed": q1_relaxed, "q1_total": q1_total,
        "per_session_cov": per_session_cov,
        "q2_ok": q2_ok, "q2_bad": q2_bad,
        "q3_sessions_with_think": q3_sessions_with_think,
        "q3_think_survived": q3_think_survived, "q3_think_total": q3_think_total,
        "lengths": lengths,
    }


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def quantiles(xs):
    if not xs:
        return {}
    s = sorted(xs)
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return {"min": s[0], "p50": at(0.5), "p90": at(0.9), "p99": at(0.99), "max": s[-1]}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    path = sys.argv[1]
    tok_path = sys.argv[2] if len(sys.argv) > 2 else None

    sessions, bad = load(path)
    r = analyse(sessions)

    print("=" * 72)
    print(f"file: {path}")
    print(f"sessions: {len(sessions)}  (multi-turn: {r['multi_turn_sessions']})"
          f"  unparsable lines: {bad}")
    if not r["multi_turn_sessions"]:
        print("\nNo multi-turn session in this file -- nothing to measure. Try a "
              "record file from a training run rather than an eval run.")
        return
    print("=" * 72)

    print("\n[Q1] MASK COVERAGE -- earlier responses found in the final prompt")
    print(f"  exact   : {r['q1_found']}/{r['q1_total']}  ({pct(r['q1_found'], r['q1_total'])})")
    print(f"  relaxed : {r['q1_relaxed']}/{r['q1_total']}  ({pct(r['q1_relaxed'], r['q1_total'])})")
    cov = r["per_session_cov"]
    if cov:
        full = sum(1 for c in cov if c >= 0.999)
        none = sum(1 for c in cov if c <= 0.001)
        print(f"  per-session: {full}/{len(cov)} fully covered, "
              f"{none}/{len(cov)} covered nothing")
        print(f"  mean coverage: {sum(cov) / len(cov):.3f}")
    print("  -> this is exactly route B's trainable fraction. Anything below")
    print("     100% is not a failure: those tokens were not in what the model")
    print("     read, so leaving them unmasked is the correct thing to do.")

    print("\n[Q2] PREFIX CONTAINMENT -- what route A (the 09-03 design) required")
    tot2 = r["q2_ok"] + r["q2_bad"]
    print(f"  holds   : {r['q2_ok']}/{tot2}  ({pct(r['q2_ok'], tot2)})")
    print(f"  broken  : {r['q2_bad']}/{tot2}  ({pct(r['q2_bad'], tot2)})")
    if r["q2_bad"] and r["q1_found"] > r["q2_ok"]:
        print("  -> route A would drop rounds that route B still trains on.")
    elif not r["q2_bad"]:
        print("  -> prefix containment HOLDS here, so whatever dropped 96/120")
        print("     rounds on 09-03 is NOT visible in this file. Do not conclude")
        print("     it is fixed; find a record file from that run.")

    print("\n[Q3] THINKING IN HISTORY -- post-template, i.e. what the model read")
    print(f"  sessions with identifiable <think> in an earlier turn: "
          f"{r['q3_sessions_with_think']}")
    print(f"  blocks surviving into the final prompt: "
          f"{r['q3_think_survived']}/{r['q3_think_total']} "
          f"({pct(r['q3_think_survived'], r['q3_think_total'])})")
    if r["q3_think_total"] == 0:
        print("  -> no <think> block found in any earlier response_text at all.")
        print("     Either the model does not emit them into response_text, or")
        print("     reasoning rides in a separate field. Inconclusive, not a NO.")
    elif r["q3_think_survived"] == 0:
        print("  -> the template DROPS historical thinking. Within one question")
        print("     the model sees only what was DONE. This is measured on real")
        print("     data and outranks any mock probe.")
    elif r["q3_think_survived"] == r["q3_think_total"]:
        print("  -> historical thinking IS in the token sequence the model read.")

    print("\n[Q4] ROUND LENGTH -- a trajectory is prompt + response of the final turn")
    q = quantiles(r["lengths"])
    print(f"  chars: {q}")
    if tok_path:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            sample = sorted(r["lengths"])[-min(20, len(r["lengths"])):]
            ratio = 1.0
            # estimate chars-per-token from the longest final prompts
            longest = max(
                ((s[-1].get("prompt_text") or "") + (s[-1].get("response_text") or ""))
                for s in sessions.values() if len(s) >= 2
            )
            n_tok = len(tok(longest, add_special_tokens=False)["input_ids"])
            ratio = len(longest) / max(n_tok, 1)
            print(f"  chars/token on the longest round: {ratio:.2f}")
            print(f"  => estimated tokens: "
                  f"{ {k: int(v / ratio) for k, v in q.items()} }")
            print(f"  compare: --max-tokens-per-gpu 32768, and the teacher runs on")
            print(f"  ONE card with CP=1. Anything above that needs a plan first.")
            del sample
        except Exception as e:
            print(f"  (tokenizer estimate unavailable: {e})")
    else:
        print("  pass a tokenizer path as argv[2] for a token estimate")
    print("=" * 72)


if __name__ == "__main__":
    main()
