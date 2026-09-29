#!/usr/bin/env python3
"""Does the OPD teacher push DOWN the student's commit tokens?  (H-e, 2026-09-28)

Background (docs/work_log.md, 2026-09-28):
  * Repeated thinking appears in every run that has OPD switched on, and in
    none without it -- pure RL's thinking actually shrinks from ~4k to ~2.4k.
  * Only FAILED rounds carry a hint. metaclaw_rollout_driver.py builds
    `hint = "" if training_passed else (...)`. Passed rounds still get OPD,
    but their teacher is the un-hinted sequence, i.e. the initial model
    itself, so for them OPD only pulls back toward the initial model
    (openclaw_topk_select_loss.py computes OPD for every sample).
  * H-e: a teacher that has been told "you got this wrong" is less willing
    than the student to stop thinking and act. Failed rounds push that
    propensity down, passed rounds only pull it back to the initial level,
    and nothing pushes it above -- so OPD teaches hesitation -> longer
    thinking -> loops.

What this measures
------------------
At training step 0 the student IS the initial model, and the teacher is the
initial model with the hint appended to this round's task message. So what
OPD does at step 0 is computable with the initial checkpoint alone.

PRIMARY -- the stop decision. OPD acts on the student's top-4 at every
position (S_t in openclaw_topk_select_loss.py). Wherever </think> is among
those four, OPD directly moves the probability of stopping right there,
toward the teacher's. So at each thinking position:

    d_stop(t) = log P_T(</think> | ...) - log P_S(</think> | ...)

    stop_cand  mean d_stop over positions where </think> is in the student's
               top-4 -- where OPD actually targets it
    a_stop     the same positions, clamp(d_stop, +-1) * w, where w is
               </think>'s weight within the student's top-4: the advantage
               OPD applies to the stop token (1.0 = the official
               OPENCLAW_TOPK_ADV_DIFF_CLIP)
    stop_all   mean d_stop over every thinking position, for context

SECONDARY -- the realised token at each position,

    d(t) = log P_T(tok_t | ...) - log P_S(tok_t | ...)

compared across the categories below.

All log-probabilities are taken at temperature 0.6 by default: training
divides logits by --rollout-temperature (0.6 in the official script) for the
student and, through the same get_responses, for the teacher
(slime/backends/megatron_utils/loss.py). Generation also runs at 0.6 here:
the requests carry no temperature and sglang (sampling_defaults=model) takes
it from the model's generation_config.json.

Token categories in each response (response = thinking + </think> + action + <|im_end|>)
    think      every thinking token, the reference level
    think_tail the last 64 thinking tokens, i.e. the approach to closing
    close      the </think> token itself
    act_head   the first 8 non-whitespace tokens after </think>
    act        every token after </think>, before <|im_end|>
    end        the <|im_end|> that ends the turn
    all        the whole response (its mean is the per-turn analogue of the
               negative k1 seen in the pure-OPD run)

Pre-registered verdict (docs/work_log.md 2026-09-29, fixed before running;
supersedes the 09-28 version, which only had the realised-token view)
    PRIMARY, over all turns of failed rounds (every turn has a stop decision,
    and in training every turn of the round carries the same hint):
        mechanism present  <=>  the 95% CI of stop_cand lies entirely below 0
                                AND the 95% CI of (fail - neutral) stop_cand
                                lies entirely below 0 (--neutral-control).
        A CI that spans 0, or lies above 0, means it is not present.
    SECONDARY: on the last turn of each failed round, (close - think) and
    (act_head - think) of the realised-token d, as before.
    This shows what OPD does to stopping at step 0. Whether that is what
    causes the loops is the separate causal test (METACLAW_OPD_MASK_COMMIT).

Log parsing caveat: the driver's stdout is block-buffered into the pipe
while its stderr is not, so a logger line can land inside a printed line at
a flush boundary. That loses the round (its marker no longer matches) rather
than corrupting it silently; --training-log checks the parsed hint lengths
against what the proxy actually received.

Faithfulness to training (each point checked against the producing code)
    * Hint text: taken from metaclaw_rollout.log, the only place it is kept.
      The proxy never records it; the verdict is posted by the driver
      directly and never enters OpenClaw's transcript.
    * Hint placement: training appends "\\n\\n[user's hint / instruction]\\n"
      + hint.strip() to the round's TASK message and .strip()s the result
      (_append_hint_to_messages, openclaw_opd_api_server.py). The task is
      the last user message containing task_prefix[:120], where task_prefix
      is query[:300].strip() (prepare_patched_openclaw_combine_select.sh).
      Splicing into the recorded prompt_text reproduces that byte for byte
      and keeps the tool definitions, which the record does not store.
    * Response tokens: training builds response_ids by re-tokenising the
      rendered response_text (openclaw_opd_api_server.py), which is exactly
      what the record stores, so the tokens here are the training tokens.

Known approximation
    Turn by turn, with the recorded prompt_text, whose history has thinking
    stripped. Since 09-14 training uses whole-round trajectory samples whose
    earlier turns keep their thinking. That changes the context, not the
    question of which way the hint pushes a closing token.

Usage (modelfactory, one GPU; the system python3 has no transformers):
    /dfs/data/envs/openclaw-rl/bin/python scripts/probe_opd_commit_direction.py \\
        --rollout-log  <LOGS_DIR>/metaclaw_rollout.log \\
        --training-log <LOGS_DIR>/training.log \\
        --records <OFFICIAL>/openclaw-combine/results/record_<RUN_ID>.jsonl \\
                  <OFFICIAL>/openclaw-combine/results/record_<RUN_ID>_archive.jsonl \\
        --model <Qwen3-4B-Thinking-2507 HF dir> \\
        --days 01-16 --neutral-control --out probe_he_<RUN_ID>.json

    The records live in the official checkout's openclaw-combine/results/,
    not in the run's log directory. Take only days before the run's repeated
    thinking set in (a looping turn has no meaningful stop decision).
    Add --dry-run first: it parses, assigns turns to rounds, splices the hint
    and categorises tokens without running the model, and with --model it
    also prints the </think> and <|im_end|> ids the loss patch relies on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HINT_HEADER = "[user's hint / instruction]"
NEUTRAL_HINT = "(no additional information)"
TASK_PREFIX_CHARS = 300   # _VERDICT_TASK_PREFIX_CHARS in metaclaw_rollout_driver.py
TASK_PROBE_CHARS = 120    # _mc_probe = _metaclaw_task_prefix[:120] in the combine_select patch
MIN_HINT_CHARS = 10       # the proxy only distils when len(hint) > 10
THINK_TAIL = 64
ACT_HEAD = 8
STUDENT_TOPK = 4          # OPENCLAW_TOPK_K: the S_t that OPD acts on
OPD_DIFF_CLIP = 1.0       # OPENCLAW_TOPK_ADV_DIFF_CLIP in the official script
TEMPERATURE = 0.6         # --rollout-temperature in the official script

# ---------------------------------------------------------------------------
# metaclaw_rollout.log parsing
# ---------------------------------------------------------------------------

_SEP_RE = re.compile(r"^\s*-{56}\s*$")
_HASH_RE = re.compile(r"^#{60}\s*$")
_HDR_RE = re.compile(r"^\s*round=(?P<round>\S+)\s+session=(?P<session>\S+)\s+agent=(?P<agent>\S+)\s*$")
_VERDICT_RE = re.compile(r"^\s*verdict: passed=(?P<p>\w+)\s+training_passed=(?P<tp>\w+)")
# The driver logs with logging.basicConfig's default "LEVEL:name:message" and
# its stderr is merged into the same file (2>&1), so httpx's request lines
# land right after the OPD hint when the verdict is posted.
_LOGGER_RE = re.compile(r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL):[\w.\-]+:")
# The end-of-run report is printed after the last round.
_REPORT_RE = re.compile(r"^(#{1,3} |\|.*\|\s*$)")

QUERY_MARK = ">> Query -> OpenClaw:"
ANSWER_PREFIX = "<< OpenClaw -> Query:"
GENFAIL_PREFIX = "[GENERATE-FAIL]"
TRAINING_HINT_PREFIX = "training hint (diff-based"
OPD_HINT_MARK = "OPD hint (training-side only, the agent never sees it):"


@dataclass
class RoundInfo:
    order: int
    day: str
    round_id: str
    session: str
    query: str = ""
    training_passed: bool | None = None
    hint: str = ""              # stripped, exactly as the proxy uses it

    @property
    def has_opd(self) -> bool:
        return len(self.hint) > MIN_HINT_CHARS

    @property
    def probe(self) -> str:
        return self.query[:TASK_PREFIX_CHARS].strip()[:TASK_PROBE_CHARS]


def parse_rollout_log(text: str) -> list[RoundInfo]:
    rounds: list[RoundInfo] = []
    cur: RoundInfo | None = None
    mode = None                 # None | "query" | "hint"
    buf: list[str] = []

    def close_section():
        nonlocal mode, buf
        kept = [l for l in buf if not _LOGGER_RE.match(l)]
        if cur is not None and mode == "query":
            cur.query = "\n".join(kept).strip()
        elif cur is not None and mode == "hint":
            cur.hint = "\n".join(kept).strip()
        mode, buf = None, []

    # split("\n"), not splitlines(): the latter also breaks on U+2028 and
    # friends inside a question, and the probe would then never match.
    for line in text.split("\n"):
        line = line.rstrip("\r")
        s = line.strip()
        m = _HDR_RE.match(line)
        if m:
            close_section()
            cur = RoundInfo(order=len(rounds), day=m["agent"], round_id=m["round"], session=m["session"])
            rounds.append(cur)
            continue
        if _SEP_RE.match(line) or _HASH_RE.match(line):
            close_section()
            continue
        if cur is None:
            continue
        if mode == "hint" and _REPORT_RE.match(line):
            close_section()
            cur = None
            continue
        if s == QUERY_MARK:
            close_section()
            mode = "query"
            continue
        if mode == "query" and (s.startswith(ANSWER_PREFIX) or s.startswith(GENFAIL_PREFIX)):
            close_section()
            continue
        if s == OPD_HINT_MARK:
            close_section()
            mode = "hint"
            continue
        if s.startswith(TRAINING_HINT_PREFIX):
            # Printed before the OPD hint and often identical to it; it is
            # not what the proxy received, so it must never be read as one.
            close_section()
            continue
        v = _VERDICT_RE.match(line)
        if v:
            close_section()
            cur.training_passed = v["tp"] == "True"
            continue
        if mode in ("query", "hint"):
            buf.append(line)
    close_section()
    return rounds


def expand_days(spec: str) -> list[str]:
    """"01-16,20" -> ["01", ..., "16", "20"]. Anything that is not a numeric
    range is kept as given, so an exact test_id still works."""
    out: list[str] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                raise ValueError(f"empty day range: {part}")
            width = len(m.group(1))
            out.extend(str(d).zfill(width) for d in range(lo, hi + 1))
        else:
            out.append(part)
    return out


def day_matches(day: str, wanted: Iterable[str]) -> bool:
    m = re.search(r"(\d+)$", day)
    for w in wanted:
        if day == w:
            return True
        if m and w.isdigit() and int(m.group(1)) == int(w):
            return True
    return False


def training_log_hint_lens(text: str) -> list[int]:
    # One line per accepted MetaClaw verdict hint; the value is
    # len(_metaclaw_hint), and _metaclaw_hint is the payload hint .strip()ed.
    return [int(m.group(1)) for m in re.finditer(
        r"metaclaw-verdict-opd-hint\].*?accepted K_i=1 hint_len=(\d+)", text)]


def hint_parse_agreement(rounds: list[RoundInfo], hint_lens: list[int]) -> float | None:
    parsed = Counter(len(r.hint) for r in rounds if r.has_opd)
    logged = Counter(hint_lens)
    if not parsed or not logged:
        return None
    matched = sum((parsed & logged).values())
    return matched / min(sum(parsed.values()), sum(logged.values()))


# ---------------------------------------------------------------------------
# prompt_text surgery
# ---------------------------------------------------------------------------

_USER_SEG_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def task_segments(prompt_text: str) -> list[tuple[int, int, str]]:
    """(content_start, content_end, content) for every rendered user message
    that is not a tool response. Qwen3 renders tool results inside a user
    block that begins with <tool_response>; training only ever looks at
    role == "user" messages, so those are skipped here too."""
    out = []
    for m in _USER_SEG_RE.finditer(prompt_text):
        content = m.group(1)
        if content.lstrip().startswith("<tool_response>"):
            continue
        out.append((m.start(1), m.end(1), content))
    return out


def locate_task(segments: list[tuple[int, int, str]], probe: str) -> tuple[int, int, str] | None:
    if not probe:
        return None
    for seg in reversed(segments):          # backwards, like the training patch
        if probe in seg[2]:
            return seg
    return None


def splice_hint(prompt_text: str, probe: str, hint: str) -> str | None:
    """prompt_text with the hint appended to the round's task message exactly
    as _append_hint_to_messages does it, or None where training would also
    have found no task (and skipped the hint)."""
    seg = locate_task(task_segments(prompt_text), probe)
    if seg is None:
        return None
    start, end, content = seg
    new = (content + f"\n\n{HINT_HEADER}\n{hint.strip()}").strip()
    return prompt_text[:start] + new + prompt_text[end:]


def assign_round(prompt_text: str, candidates: list[RoundInfo]) -> RoundInfo | None:
    """The round a turn belongs to: the one whose task appears LATEST in the
    transcript. Ties go to the later round, which covers a feedback message
    that quotes the previous question inside the next round's task."""
    segs = task_segments(prompt_text)
    best, best_key = None, None
    for r in candidates:
        seg = locate_task(segs, r.probe)
        if seg is None:
            continue
        key = (seg[0], r.order)
        if best_key is None or key > best_key:
            best, best_key = r, key
    return best


# ---------------------------------------------------------------------------
# token categories
# ---------------------------------------------------------------------------

def categorize(resp_ids: list[int], close_id: int, end_id: int, is_ws: list[bool]) -> dict[str, list[int]] | None:
    try:
        c = resp_ids.index(close_id)
    except ValueError:
        return None
    e = next((i for i in range(len(resp_ids) - 1, c, -1) if resp_ids[i] == end_id), None)
    stop = e if e is not None else len(resp_ids)
    act = list(range(c + 1, stop))
    return {
        "think": list(range(0, c)),
        "think_tail": list(range(max(0, c - THINK_TAIL), c)),
        "close": [c],
        "act_head": [i for i in act if not is_ws[i]][:ACT_HEAD],
        "act": act,
        "end": [e] if e is not None else [],
        "all": list(range(0, stop + (1 if e is not None else 0))),
    }


def cat_means(d: list[float], cats: dict[str, list[int]]) -> dict[str, float]:
    return {k: (sum(d[i] for i in idx) / len(idx) if idx else float("nan")) for k, idx in cats.items()}


def stop_metrics(close_s: list[float], close_t: list[float], in_topk: list[bool],
                 w_close: list[float], c: int, clip: float = OPD_DIFF_CLIP) -> dict[str, float]:
    """What OPD does to "stop thinking now" over one turn's thinking.

    Positions 0..c inclusive, c being where </think> was written -- that
    position is itself a stop decision. close_s / close_t are log P(</think>)
    under student and teacher; in_topk and w_close come from the STUDENT,
    because S_t is the student's top-K and w is softmax(ell_old | S_t).
    """
    nan = float("nan")
    pos = range(0, c + 1)
    d = [close_t[i] - close_s[i] for i in pos]
    cand = [i for i in pos if in_topk[i]]
    dc = [close_t[i] - close_s[i] for i in cand]
    return {
        "stop_all": sum(d) / len(d) if d else nan,
        "stop_cand": sum(dc) / len(dc) if dc else nan,
        "a_stop": (sum(max(-clip, min(clip, close_t[i] - close_s[i])) * w_close[i] for i in cand)
                   / len(cand)) if cand else nan,
        "n_cand": float(len(cand)),
    }


def contrasts(m: dict[str, float]) -> dict[str, float]:
    return {
        "close_minus_think": m["close"] - m["think"],
        "act_head_minus_think": m["act_head"] - m["think"],
        "end_minus_think": m["end"] - m["think"],
        "close_minus_think_tail": m["close"] - m["think_tail"],
        "all_mean": m["all"],
    }


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def summarize(values: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    v = [x for x in values if x == x]       # drop NaN
    if not v:
        return {"n": 0}
    v.sort()
    n = len(v)
    mean = sum(v) / n
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(v) for _ in range(n)) / n for _ in range(n_boot))
    return {
        "n": n,
        "mean": mean,
        "median": v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2,
        "frac_neg": sum(x < 0 for x in v) / n,
        "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
    }


PRIMARY_KEYS = ["stop_cand", "a_stop", "stop_all", "n_cand"]
SECONDARY_KEYS = ["close_minus_think", "act_head_minus_think", "end_minus_think",
                  "close_minus_think_tail", "all_mean"]


def summarize_rows(rows: list[dict], neutral: bool) -> dict:
    keys = PRIMARY_KEYS + SECONDARY_KEYS
    out = {}
    for subset, sel in (("all_turns", lambda r: True), ("last_turn_of_round", lambda r: r["last"])):
        rs = [r for r in rows if sel(r)]
        s = {"fail": {k: summarize([r["fail"][k] for r in rs]) for k in keys}}
        if neutral:
            s["neutral"] = {k: summarize([r["neutral"][k] for r in rs]) for k in keys}
            s["fail_minus_neutral"] = {k: summarize([r["fail"][k] - r["neutral"][k] for r in rs])
                                       for k in keys if k != "n_cand"}
        out[subset] = s
    return out


def primary_verdict(summary: dict, neutral: bool) -> dict:
    """The pre-registered primary test on stop_cand over all failed-round
    turns. below_neutral is None when no neutral control was run, and then
    `present` cannot be True: without the control a negative stop_cand could
    just be "any appended text makes the model less ready to stop"."""
    a = summary["all_turns"]
    f = a["fail"]["stop_cand"]
    below_zero = bool(f.get("n")) and f["ci95"][1] < 0
    below_neutral = None
    if neutral:
        g = a["fail_minus_neutral"]["stop_cand"]
        below_neutral = bool(g.get("n")) and g["ci95"][1] < 0
    return {"below_zero": below_zero, "below_neutral": below_neutral,
            "present": below_zero and below_neutral is True}


# ---------------------------------------------------------------------------
# model side (imported lazily so --dry-run and the tests need no torch)
# ---------------------------------------------------------------------------

def load_model(path: str, dtype: str, attn: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=getattr(torch, dtype), attn_implementation=attn,
    ).cuda().eval()
    return tok, model


def response_logprobs(model, prompt_ids: list[int], resp_ids: list[int], close_id: int | None = None,
                      temperature: float = 1.0, chunk: int = 1024) -> dict[str, list]:
    """Per response position, at the given temperature (training divides
    logits by --rollout-temperature before log_softmax):
        real     log P(resp_t | prompt, resp_<t)
        close    log P(</think>)                          (if close_id given)
        in_topk  </think> is in the top-STUDENT_TOPK      (if close_id given)
        w_close  its softmax weight within that top-K     (if close_id given)
    Runs the decoder once, then the LM head only over the response span, in
    chunks -- full-vocabulary logits over a 30k-token prompt would not fit."""
    import torch
    with torch.no_grad():
        dev = next(model.parameters()).device
        ids = torch.tensor([prompt_ids + resp_ids], device=dev)
        hidden = model.get_decoder()(input_ids=ids, use_cache=False).last_hidden_state
        p, r = len(prompt_ids), len(resp_ids)
        h = hidden[0, p - 1:p + r - 1]
        tgt = ids[0, p:p + r]
        head = model.get_output_embeddings()
        real = torch.empty(r, dtype=torch.float32, device=dev)
        close = torch.empty(r, dtype=torch.float32, device=dev)
        in_topk = torch.zeros(r, dtype=torch.bool, device=dev)
        w_close = torch.zeros(r, dtype=torch.float32, device=dev)
        for s in range(0, r, chunk):
            lp = (head(h[s:s + chunk]).float() / temperature).log_softmax(-1)
            real[s:s + chunk] = lp.gather(-1, tgt[s:s + chunk, None]).squeeze(-1)
            if close_id is not None:
                close[s:s + chunk] = lp[:, close_id]
                top_lp, top_idx = lp.topk(STUDENT_TOPK, dim=-1)
                hit = top_idx == close_id
                in_topk[s:s + chunk] = hit.any(-1)
                w_close[s:s + chunk] = (top_lp.softmax(-1) * hit).sum(-1)
        out = {"real": real.cpu().tolist()}
        if close_id is not None:
            out.update(close=close.cpu().tolist(), in_topk=in_topk.cpu().tolist(),
                       w_close=w_close.cpu().tolist())
        return out


def alignment_check(model, prompt_ids: list[int], resp_ids: list[int], tol: float = 2e-3):
    """Guard against an off-by-one in response_logprobs, which would give
    every token another token's probability and still look plausible.

    The reference is the model's own loss with labels masked to the
    response: HF shifts logits and labels internally, independently of the
    slicing above. Mean NLL over the response must agree."""
    import torch
    lp = response_logprobs(model, prompt_ids, resp_ids, temperature=1.0)["real"]
    mine = -sum(lp) / len(lp)
    with torch.no_grad():
        dev = next(model.parameters()).device
        ids = torch.tensor([prompt_ids + resp_ids], device=dev)
        labels = ids.clone()
        labels[0, : len(prompt_ids)] = -100
        ref = model(input_ids=ids, labels=labels, use_cache=False).loss.float().item()
    return mine, ref, abs(mine - ref) <= tol * max(1.0, abs(ref))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    rnd: RoundInfo
    session: str
    turn: int
    prompt_text: str
    response_text: str
    is_last: bool = False


def dedupe_records(recs: Iterable[dict]) -> list[dict]:
    """The live record and its archive overlap whenever a snapshot was taken
    mid-run; keep one copy of each turn."""
    seen, out = set(), []
    for rec in recs:
        pt, rt = rec.get("prompt_text") or "", rec.get("response_text") or ""
        key = (rec.get("session_id"), rec.get("turn"),
               hashlib.sha1((pt + "\x00" + rt).encode("utf-8")).hexdigest())
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def load_records(paths: list[str]) -> list[dict]:
    def read():
        for p in paths:
            for line in Path(p).read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    return dedupe_records(read())


def collect_turns(rounds: list[RoundInfo], records: list[dict], days: list[str], counts: Counter) -> list[Turn]:
    day_rounds = [r for r in rounds if day_matches(r.day, days)]
    by_session: dict[str, list[RoundInfo]] = {}
    for r in day_rounds:
        by_session.setdefault(r.session, []).append(r)
    all_sessions = {r.session for r in rounds}
    turns: list[Turn] = []
    for rec in records:
        pt, rt = rec.get("prompt_text") or "", rec.get("response_text") or ""
        if not pt or not rt.strip():
            counts["record_empty"] += 1
            continue
        sid = rec.get("session_id")
        if sid in all_sessions:
            # The record's session is one the driver printed: only that
            # session's rounds can own it, and if that session is not in the
            # selected days the turn is out of scope. Matching by question
            # text alone could pull a later day's repeat of a day-01 question.
            cands = by_session.get(sid, [])
        else:
            # Session ids not comparable between proxy and driver: fall back
            # to question matching over the selected days.
            counts["record_session_unmatched"] += 1
            cands = day_rounds
        rnd = assign_round(pt, cands) if cands else None
        if rnd is None:
            counts["record_not_in_selected_days"] += 1
            continue
        counts["turn_assigned"] += 1
        if not rnd.has_opd:
            counts["turn_round_has_no_opd_hint"] += 1
            continue
        turns.append(Turn(rnd, rec.get("session_id") or "", int(rec.get("turn") or 0), pt, rt))
    last: dict[tuple, int] = {}
    for t in turns:
        k = (t.session, t.rnd.order)
        last[k] = max(last.get(k, -1), t.turn)
    for t in turns:
        t.is_last = t.turn == last[(t.session, t.rnd.order)]
    return turns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rollout-log", required=True)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--training-log")
    ap.add_argument("--model")
    ap.add_argument("--days", default="01-03", help='e.g. "01-16" or "01,02,05-08"')
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--neutral-control", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    days = expand_days(args.days)
    counts: Counter = Counter()

    rounds = parse_rollout_log(Path(args.rollout_log).read_text(encoding="utf-8", errors="replace"))
    with_hint = [r for r in rounds if r.has_opd]
    print(f"[log] rounds parsed={len(rounds)}  with OPD hint={len(with_hint)}")
    # A round with an OPD hint must have been a training failure; anything
    # else means the parser attached a hint to the wrong round.
    bad = [r for r in with_hint if r.training_passed is True]
    print(f"[log] hint on a training_passed=True round: {len(bad)} (must be 0)")
    agreement = None
    if args.training_log:
        lens = training_log_hint_lens(Path(args.training_log).read_text(encoding="utf-8", errors="replace"))
        agreement = hint_parse_agreement(rounds, lens)
        print(f"[log] proxy logged {len(lens)} hint_len values; parsed-hint length agreement = "
              f"{'n/a' if agreement is None else f'{agreement:.1%}'}")
        if agreement is not None and agreement < 0.9:
            print("[log] WARNING: under 90% -- the hint parser is likely wrong for this log; "
                  "check before trusting any number below.")

    records = load_records(args.records)
    turns = collect_turns(rounds, records, days, counts)
    print(f"[records] {len(records)} unique turns; {dict(counts)}")
    print(f"[turns] failed-round turns with a hint in days {days}: {len(turns)} "
          f"(last-of-round: {sum(t.is_last for t in turns)})")
    if len(turns) > args.max_turns:
        rng = random.Random(0)
        lasts = [t for t in turns if t.is_last]
        rest = [t for t in turns if not t.is_last]
        rng.shuffle(rest)
        turns = (lasts + rest)[: args.max_turns]
        print(f"[turns] capped to {len(turns)} (all last-of-round turns kept first)")

    if args.dry_run:
        tok = None
        if args.model:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model)
            # The loss patch (METACLAW_OPD_MASK_COMMIT) defaults to these two
            # ids; if they differ, pass METACLAW_THINK_CLOSE_ID / _IM_END_ID.
            print(f"[ids] </think>={tok.convert_tokens_to_ids('</think>')} "
                  f"<|im_end|>={tok.convert_tokens_to_ids('<|im_end|>')} "
                  f"(loss patch defaults: 151668 / 151645)")
        for t in turns[:3]:
            sp = splice_hint(t.prompt_text, t.rnd.probe, t.rnd.hint)
            print("-" * 72)
            print(f"day={t.rnd.day} round={t.rnd.round_id} turn={t.turn} last={t.is_last}")
            print(f"hint[:200]={t.rnd.hint[:200]!r}")
            if sp is None:
                print("task not located (training would also have skipped this hint)")
                continue
            i = sp.find(HINT_HEADER)
            print(f"spliced context: ...{sp[max(0, i - 120):i + 160]!r}...")
            if tok is not None:
                ids = tok(t.response_text, add_special_tokens=False)["input_ids"]
                ws = [tok.decode([x]).strip() == "" for x in ids]
                cats = categorize(ids, tok.convert_tokens_to_ids("</think>"),
                                  tok.convert_tokens_to_ids("<|im_end|>"), ws)
                print("categories:", None if cats is None else {k: len(v) for k, v in cats.items()})
        return 0

    if not args.model:
        ap.error("--model is required unless --dry-run")
    tok, model = load_model(args.model, args.dtype, args.attn)
    close_id = tok.convert_tokens_to_ids("</think>")
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if not turns:
        print("no turns to measure")
        return 1

    # Alignment check on a short slice of a real turn (full-vocabulary logits
    # are only affordable on a short sequence); semantics do not matter here.
    t0 = min(turns, key=lambda t: len(t.prompt_text) + len(t.response_text))
    p0 = tok(t0.prompt_text, add_special_tokens=False)["input_ids"][-2048:]
    r0 = tok(t0.response_text, add_special_tokens=False)["input_ids"][:512]
    mine, ref, ok = alignment_check(model, p0, r0)
    print(f"[check] response NLL: sliced LM head {mine:.5f} vs model loss {ref:.5f} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("[check] response_logprobs is misaligned; refusing to report numbers.")
        return 2

    rows = []
    for n, t in enumerate(turns, 1):
        teacher_pt = splice_hint(t.prompt_text, t.rnd.probe, t.rnd.hint)
        if teacher_pt is None:
            counts["skip_task_not_located"] += 1
            continue
        resp = tok(t.response_text, add_special_tokens=False)["input_ids"]
        ws = [tok.decode([x]).strip() == "" for x in resp]
        cats = categorize(resp, close_id, end_id, ws)
        if cats is None:
            counts["skip_no_think_close"] += 1
            continue
        s_ids = tok(t.prompt_text, add_special_tokens=False)["input_ids"]
        t_ids = tok(teacher_pt, add_special_tokens=False)["input_ids"]
        if max(len(s_ids), len(t_ids)) + len(resp) > args.max_tokens:
            counts["skip_too_long"] += 1
            continue
        T = args.temperature
        out_s = response_logprobs(model, s_ids, resp, close_id, T)
        out_t = response_logprobs(model, t_ids, resp, close_id, T)
        c = cats["close"][0]

        def measure(out_x):
            d = [a - b for a, b in zip(out_x["real"], out_s["real"])]
            m = contrasts(cat_means(d, cats))
            # in_topk / w_close from the STUDENT pass: S_t is the student's top-K.
            m.update(stop_metrics(out_s["close"], out_x["close"], out_s["in_topk"], out_s["w_close"], c))
            return m

        row = {"day": t.rnd.day, "round": t.rnd.round_id, "turn": t.turn, "last": t.is_last,
               "n_think": len(cats["think"]), "n_resp": len(resp), "fail": measure(out_t)}
        if args.neutral_control:
            n_pt = splice_hint(t.prompt_text, t.rnd.probe, NEUTRAL_HINT)
            n_ids = tok(n_pt, add_special_tokens=False)["input_ids"]
            row["neutral"] = measure(response_logprobs(model, n_ids, resp, close_id, T))
        rows.append(row)
        if n % 20 == 0:
            print(f"[run] {n}/{len(turns)} done")

    summary = summarize_rows(rows, args.neutral_control)
    print("=" * 72)
    for subset, s in summary.items():
        print(f"[{subset}]")
        for block, stats in s.items():
            for k, st in stats.items():
                if st.get("n"):
                    print(f"  {block:18s} {k:24s} n={st['n']:4d} mean={st['mean']:+.4f} "
                          f"median={st['median']:+.4f} neg={st['frac_neg']:.0%} "
                          f"ci95=[{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}]")
    v = primary_verdict(summary, args.neutral_control)
    print("-" * 72)
    print(f"PRIMARY (all failed-round turns, stop_cand): CI entirely below 0 = {v['below_zero']}; "
          f"more negative than neutral = {v['below_neutral']}")
    print(f"  => OPD suppresses stopping at step 0: {v['present']}")
    lt = summary["last_turn_of_round"]["fail"]
    below = [k for k in ("close_minus_think", "act_head_minus_think")
             if lt[k].get("n") and lt[k]["ci95"][1] < 0]
    print(f"SECONDARY (last turn, realised token): CI entirely below 0 for {below or 'neither'}")
    print(f"counts: {dict(counts)}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "args": vars(args), "counts": dict(counts), "hint_len_agreement": agreement,
            "summary": summary, "rows": rows,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
