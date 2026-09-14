#!/usr/bin/env python3
"""Does OPD's directive signal attenuate on tokens the hint has nothing to say about?

Why this decides something
--------------------------
OPD's per-token advantage is built from ``ell_T(v) - ell_old(v)``, where both
sides come from the SAME weights and differ only in whether the round's hint
was spliced into the prompt (the teacher load is the policy load -- see the
launcher's PRM_TEACHER_LOAD="${POLICY_TORCH_DIST}"). So that difference is
purely "what does adding the hint do to this token's log-prob".

The open question is whether it self-attenuates. If the hint says "the filename
must carry an ISO 8601 date", it should move the tokens of a write call and
leave the tokens of a directory listing alone. If it moves everything equally,
then letting a round-level hint condition a whole trajectory would spread
mis-teaching onto every turn -- which is the objection to doing exactly that.

  attenuates      -> a round-level hint can safely condition the whole round
  does not        -> it must stay aimed at the turn that produced the artifact

The readout is INTERNAL: write-turn tokens versus non-write-turn tokens, from
the same rounds and the same model. No absolute threshold has to be guessed.

The mismatched-hint control is not optional
-------------------------------------------
A third condition re-runs each turn with a hint taken from a DIFFERENT round.
Without it, "the difference is small on non-write turns" cannot be told apart
from "any extra text perturbs the logits by about this much, everywhere". If
the correct hint and a wrong hint move the tokens equally, then ``ell_T -
ell_old`` is not carrying hint-specific information at all, and that is a
finding about OPD itself rather than about trajectories.

Costs one GPU. No training, no optimizer -- three forward passes per turn.

Usage:
    python3 measure_opd_hint_attenuation.py <record.jsonl> <model_path> \
        [--max-rounds 20] [--max-tokens 24000]
"""
import argparse
import collections
import copy
import json
import math
import random
import re
import sys

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
IM_END = "<|im_end|>"

WRITE_TOOLS = {
    "write", "edit", "create", "str_replace", "str_replace_editor",
    "apply_patch", "multi_edit", "multiedit", "notebook_edit", "write_file",
    "create_file", "patch",
}
SHELL_TOOLS = {"exec", "bash", "shell", "run_command", "run", "terminal"}
SHELL_WRITE_RE = re.compile(
    r">>?\s*\S|\btee\b|\bcp\b|\bmv\b|\btouch\b|\bmkdir\b|\bsed\s+-i\b|"
    r"\brm\b|\brmdir\b|\bunlink\b|\btruncate\b|\bshred\b|"
    r"\bopen\s*\([^)]*['\"][wa]|\bjson\.dump|\bwriteFile|\bprintf\b.*>",
)


def tool_calls_of(rec):
    out = []
    for tc in rec.get("tool_calls") or []:
        if isinstance(tc, dict):
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


def is_write_turn(rec):
    for n, a in tool_calls_of(rec):
        nl = n.strip().lower()
        if nl in WRITE_TOOLS or (nl in SHELL_TOOLS and SHELL_WRITE_RE.search(a or "")):
            return True
    return False


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
    return rounds


def splice_hint(prompt_text, task_prefix, hint):
    """Insert the hint where the proxy inserts it: appended to the round's task.

    The proxy does this on the message list (_append_hint_to_messages, with the
    task located by task_prefix). Here it is done on the rendered prompt so the
    no-hint condition can be the REAL prompt_text -- which already carries the
    tool schemas the record does not store. Splicing on text keeps both
    conditions differing by exactly the hint and nothing else.

    Returns None when the task cannot be located, which is the same thing the
    proxy does rather than anchoring the hint to the wrong round.
    """
    probe = (task_prefix or "")[:120]
    if not probe:
        return None
    at = prompt_text.rfind(probe)
    if at < 0:
        return None
    end = prompt_text.find(IM_END, at)
    if end < 0:
        return None
    suffix = f"\n\n[user's hint / instruction]\n{hint.strip()}"
    return prompt_text[:end] + suffix + prompt_text[end:]


class Scorer:
    """Per-token log-probs of a response, given a prompt, without ever
    materialising [T, vocab] logits.

    A 24k-token sequence against a 152k vocab would be about 7 GiB in bf16 and
    twice that once softmaxed -- on one card, on top of the weights, that does
    not fit. So the base model runs once for hidden states, then the LM head is
    applied in slices and only the gathered target log-probs are kept.
    """

    def __init__(self, model_path, chunk=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).eval().cuda()
        self.chunk = chunk

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @property
    def _base(self):
        # The transformer trunk without the LM head, whatever it is called.
        for name in ("model", "transformer"):
            if hasattr(self.model, name):
                return getattr(self.model, name)
        raise RuntimeError("cannot find the model trunk")

    def token_logprobs(self, full_ids, response_len):
        """log p(token_i | prefix) for the last `response_len` tokens."""
        torch = self.torch
        with torch.no_grad():
            inp = torch.tensor([full_ids], device="cuda")
            h = self._base(input_ids=inp, use_cache=False).last_hidden_state[0]
            head = self.model.get_output_embeddings()
            T = len(full_ids)
            start = T - response_len
            out = []
            # position i predicts token i+1, so the log-prob of response token
            # at index j comes from hidden state j-1.
            for lo in range(start, T, self.chunk):
                hi = min(T, lo + self.chunk)
                hs = h[lo - 1:hi - 1]
                logits = head(hs).float()
                lp = torch.log_softmax(logits, dim=-1)
                tgt = inp[0, lo:hi]
                out.append(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu())
                del logits, lp, hs
            return torch.cat(out).tolist()


def summarise(vals):
    if not vals:
        return {}
    s = sorted(vals)
    n = len(s)
    at = lambda q: s[min(n - 1, int(q * n))]
    return {
        "n": n,
        "mean_abs": round(sum(abs(x) for x in s) / n, 5),
        "p50_abs": round(sorted(abs(x) for x in s)[n // 2], 5),
        "p90_abs": round(sorted(abs(x) for x in s)[int(0.9 * n)], 5),
        "mean": round(sum(s) / n, 5),
        "min": round(s[0], 5),
        "max": round(s[-1], 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("model")
    ap.add_argument("--max-rounds", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    rounds = build_rounds(args.record)
    hinted = [r for r in rounds
              if len(str(r["verdict"].get("hint") or "").strip()) > 10]
    print("=" * 72)
    print(f"rounds: {len(rounds)}   with a usable hint: {len(hinted)}")
    if len(hinted) < 2:
        print("Fewer than two hinted rounds, so the mismatched-hint control "
              "cannot be built. Nothing measured.")
        return
    random.shuffle(hinted)
    hinted = hinted[: args.max_rounds]
    print(f"measuring {len(hinted)} of them (--max-rounds)")
    print("=" * 72)

    sc = Scorer(args.model)
    buckets = collections.defaultdict(list)
    counted = skipped = 0

    for ri, rd in enumerate(hinted):
        hint = str(rd["verdict"]["hint"]).strip()
        task_prefix = str(rd["verdict"].get("task_prefix") or "")
        # a hint from another round, for the control
        other = random.choice([r for r in hinted if r is not rd])
        wrong_hint = str(other["verdict"]["hint"]).strip()

        for rec in rd["turns"]:
            prompt = rec.get("prompt_text") or ""
            resp = rec.get("response_text") or ""
            if not prompt or not resp.strip():
                skipped += 1
                continue
            p_ok = splice_hint(prompt, task_prefix, hint)
            p_bad = splice_hint(prompt, task_prefix, wrong_hint)
            if p_ok is None or p_bad is None:
                skipped += 1
                continue

            r_len = len(sc.ids(resp))
            if r_len < 1:
                skipped += 1
                continue
            base_ids = sc.ids(prompt + resp)
            ok_ids = sc.ids(p_ok + resp)
            bad_ids = sc.ids(p_bad + resp)
            if max(len(base_ids), len(ok_ids), len(bad_ids)) > args.max_tokens:
                skipped += 1
                continue

            try:
                lp_base = sc.token_logprobs(base_ids, r_len)
                lp_ok = sc.token_logprobs(ok_ids, r_len)
                lp_bad = sc.token_logprobs(bad_ids, r_len)
            except RuntimeError as e:      # OOM on a long one: report, continue
                print(f"  [skip] turn {rec.get('turn')}: {e}")
                skipped += 1
                continue

            group = "WRITE" if is_write_turn(rec) else "non-write"
            buckets[(group, "correct_hint")] += [
                a - b for a, b in zip(lp_ok, lp_base)
            ]
            buckets[(group, "wrong_hint")] += [
                a - b for a, b in zip(lp_bad, lp_base)
            ]
            counted += 1
        print(f"  round {ri+1}/{len(hinted)} done "
              f"(turns measured so far: {counted}, skipped: {skipped})")

    print("\n" + "=" * 72)
    print("ell_T - ell_old  per response token,  by turn type and hint condition")
    print("=" * 72)
    for group in ("WRITE", "non-write"):
        for cond in ("correct_hint", "wrong_hint"):
            v = buckets.get((group, cond))
            print(f"  {group:10s} {cond:13s} {summarise(v)}")

    w_ok = summarise(buckets.get(("WRITE", "correct_hint")))
    w_bad = summarise(buckets.get(("WRITE", "wrong_hint")))
    n_ok = summarise(buckets.get(("non-write", "correct_hint")))
    n_bad = summarise(buckets.get(("non-write", "wrong_hint")))

    print("\n" + "=" * 72)
    if not (w_ok and n_ok):
        print("VERDICT: one of the two turn groups has no tokens, so the internal")
        print("         comparison is unavailable. Re-run with more rounds.")
        print("=" * 72)
        return

    # Control first: if a wrong hint moves tokens as much as the right one,
    # nothing downstream means what it is supposed to mean.
    if w_bad and w_ok["mean_abs"] <= 1.25 * w_bad["mean_abs"]:
        print("VERDICT: CONTROL FAILED. A hint from another round moves the")
        print(f"         write-turn tokens about as much ({w_bad['mean_abs']}) as the")
        print(f"         correct hint does ({w_ok['mean_abs']}). ell_T - ell_old is")
        print("         then mostly 'extra text perturbs the logits', not hint-")
        print("         specific information -- which is a finding about OPD's")
        print("         directive signal itself, and has to be resolved before")
        print("         any conclusion about trajectories.")
    elif n_ok["mean_abs"] <= 0.5 * w_ok["mean_abs"]:
        print(f"VERDICT: ATTENUATES. Non-write turns move {n_ok['mean_abs']} per")
        print(f"         token against {w_ok['mean_abs']} on write turns")
        print(f"         ({n_ok['mean_abs']/w_ok['mean_abs']:.2f}x).")
        print("         A round-level hint conditioning the whole round would put")
        print("         little pressure on turns it has nothing to say about, so")
        print("         it does not need to be aimed at a single turn.")
    else:
        print(f"VERDICT: DOES NOT ATTENUATE. Non-write turns move {n_ok['mean_abs']}")
        print(f"         against {w_ok['mean_abs']} on write turns")
        print(f"         ({n_ok['mean_abs']/w_ok['mean_abs']:.2f}x).")
        print("         Letting a round-level hint condition a whole trajectory")
        print("         would push on turns the hint is not about. Keep OPD aimed")
        print("         at the turn that produced the artifact.")
    print("=" * 72)
    print(f"\nturns measured: {counted}   skipped: {skipped}")
    print("Skips are mostly length (--max-tokens) and task-prefix misses; a high")
    print("skip count means this measured a short-round subset, not the whole set.")


if __name__ == "__main__":
    main()
