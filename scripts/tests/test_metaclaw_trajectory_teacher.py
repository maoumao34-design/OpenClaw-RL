"""The OPD teacher sequence of a trajectory sample must end in that trajectory.

Runs 20261015 and 20261016_135901 both died at day04-05 with the teacher
forward reporting "chunk has 4079 tokens but payload has 19055 index rows":
the round's verdict is evaluated on its LAST turn alone, before the round is
assembled, so the teacher ids it produced were that one turn's prompt + hint +
re-tokenised response (14919 tokens), while the student's response segment was
the whole 6-turn round (19055 tokens). _metaclaw_submit_round then shipped the
stale ids with the trajectory, and slicing the last 19055 tokens of a
14919-token sequence started below zero.

These tests run the code the patch script GENERATES (pulled out of the patched
server and exec'd onto a stub), on the production response shape used by
test_metaclaw_trajectory.py, and pin down:

  - the submitted teacher candidate ends in exactly the trajectory's
    response ids, so student and teacher share one response;
  - the hint lands in this round's task message of the round's first prompt;
  - the stale per-turn candidate from opd_result is never shipped;
  - when the hint cannot be placed, the round goes in RL-only instead of
    carrying a mismatched teacher.

Usage:
    OPENCLAW_RL_OFFICIAL=<path> python scripts/tests/test_metaclaw_trajectory_teacher.py
"""

import asyncio
import ast
import copy
import itertools
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_metaclaw_trajectory as base  # noqa: E402

IM_END = base.IM_END
ASSISTANT_HDR = base.ASSISTANT_HDR
GEN_TAIL = base.GEN_TAIL


class TemplateTokenizer(base.FakeTokenizer):
    """One id per character, plus a chat template in the Qwen3 shape."""

    def apply_chat_template(self, messages, tools=None, tokenize=False,
                            add_generation_prompt=False):
        out = "".join(
            f"<|im_start|>{m['role']}\n{m.get('content') or ''}{IM_END}\n"
            for m in messages
        )
        if add_generation_prompt:
            out += ASSISTANT_HDR + GEN_TAIL
        return out


def opd_helpers():
    """The real hint / template helpers, taken from the official OPD server."""
    path = os.path.join(base.OFFICIAL, "openclaw-opd", "openclaw_opd_api_server.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    wanted = {"_flatten_message_content", "_normalize_messages_for_template",
              "_normalize_tool_call", "_append_hint_to_messages"}
    src = "\n\n".join(ast.unparse(n) for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name in wanted)
    ns = {"copy": copy, "json": json}
    exec(compile(src, path, "exec"), ns)
    return ns


def extract(source, names, ns):
    parts = []
    for name in names:
        m = re.search(
            r"^(    (?:@staticmethod\n    )?(?:async )?def " + name + r"\(.*?)(?=^    (?:@|async |def ))",
            source, re.S | re.M,
        )
        if not m:
            raise AssertionError(f"could not find {name} in the generated server")
        parts.append(m.group(1))
    exec(compile("class _S:\n" + "\n".join(parts), "<generated>", "exec"), ns)
    return ns["_S"]


class Recorder:
    def __init__(self):
        self.calls = []

    def put(self, item):
        self.calls.append(item)


def main():
    source = base.build_patched_server()
    if source is None:
        return
    log = base.StubLogger()
    ns = {
        **opd_helpers(),
        "_MC_THINK_CLOSE": base.THINK_CLOSE,
        "_MC_ASSISTANT_HDR": ASSISTANT_HDR,
        "_MC_MAX_TRAJ_TOKENS": 0,
        "logger": log,
        "asyncio": asyncio,
    }
    Cls = extract(source, ("_metaclaw_visible", "_metaclaw_generation_tail",
                           "_metaclaw_build_trajectory",
                           "_metaclaw_trajectory_teacher_ids",
                           "_metaclaw_submit_round"), ns)

    n = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    def make_server():
        srv = base.StubServer()
        srv.tokenizer = TemplateTokenizer()
        srv._group_counter = itertools.count(7)
        srv.output_queue = Recorder()
        srv.submitted = []
        srv._metaclaw_visible = Cls._metaclaw_visible
        srv._metaclaw_generation_tail = Cls._metaclaw_generation_tail
        for name in ("_metaclaw_build_trajectory", "_metaclaw_trajectory_teacher_ids",
                     "_metaclaw_submit_round"):
            setattr(srv, name, getattr(Cls, name).__get__(srv))

        async def submit_opd(td, sid, opd, reward=0.0):
            srv.submitted.append(("opd", td, opd))
            td["metaclaw_round_collect"].append(object())

        async def submit_rl(td, sid, reward):
            srv.submitted.append(("rl", td, None))
            td["metaclaw_round_collect"].append(object())

        srv._submit_turn_sample = submit_opd
        srv._submit_rl_turn_sample = submit_rl
        return srv

    # A two-turn round in the production shape: the second turn's prompt
    # replays the first turn without its thinking.
    task = "Day 4 task 3: write day04/20260326_summary.json from the notes."
    sysmsg = {"role": "system", "content": "You are an agent."}
    first_msgs = [sysmsg, {"role": "user", "content": task}]
    tpl = TemplateTokenizer()
    p1 = tpl.apply_chat_template(first_msgs, add_generation_prompt=True)
    A1 = '<tool_call>\n{"name": "read", "arguments": {"path": "notes.md"}}\n</tool_call>'
    r1 = base.response("Let me read the notes first. " * 40, A1)
    t1 = base.turn(r1, prompt_text=p1)
    t1["messages"] = first_msgs
    obs = "<|im_start|>user\n<tool_response>\nnote body\n</tool_response>" + IM_END + "\n"
    p2 = p1[: -len(GEN_TAIL)] + "\n" + A1 + IM_END + "\n" + obs + ASSISTANT_HDR + GEN_TAIL
    A2 = '<tool_call>\n{"name": "write", "arguments": {"path": "day04/x.json"}}\n</tool_call>'
    r2 = base.response("Now I write it.", A2)
    t2 = base.turn(r2, prompt_text=p2)
    t2["messages"] = first_msgs + [{"role": "assistant", "content": A1},
                                   {"role": "user", "content": "note body"}]
    base.assert_fixture_is_realistic(t1, t2)

    hint = "The file name must follow YYYYMMDD_name; day04/x.json does not."
    # What _opd_evaluate hands over: built from the verdict (last) turn alone.
    stale = tpl(tpl.apply_chat_template(t2["messages"], add_generation_prompt=True)
                + hint + r2)["input_ids"]
    opd = {"accepted": True, "hint": hint, "teacher_tokens_candidates": [stale],
           "metaclaw_verdict": True, "metaclaw_task_prefix": task[:300]}

    print("[the teacher sequence is rebuilt for the assembled trajectory]")
    srv = make_server()
    asyncio.run(srv._metaclaw_submit_round("s1", [t1], t2, opd, reward=-1.0))
    ck(len(srv.submitted) == 1 and srv.submitted[0][0] == "opd",
       "an accepted hint still goes in as an OPD+RL sample")
    _, td, sent = srv.submitted[0]
    resp = list(td["response_ids"])
    cand = sent["teacher_tokens_candidates"][0]
    ck(len(resp) > len(r2), "fixture is meaningful: the trajectory response is longer "
       "than the verdict turn's own response")
    ck(len(stale) < len(resp),
       "fixture reproduces the crash shape: the stale per-turn teacher is SHORTER "
       "than the student's response (19055 vs 14919 in production)")
    ck(cand is not stale and cand != stale, "the stale per-turn candidate is not shipped")
    ck(len(cand) > len(resp) and cand[-len(resp):] == resp,
       "the teacher candidate ends in exactly the trajectory's response ids")
    prompt_part = "".join(chr(c) for c in cand[: len(cand) - len(resp)])
    ck(hint in prompt_part and task in prompt_part,
       "the hint sits in the teacher prompt alongside this round's task")
    ck(prompt_part.index(task) < prompt_part.index(hint),
       "the hint is appended to the task message, not ahead of it")
    ck(prompt_part.endswith(ASSISTANT_HDR + GEN_TAIL),
       "the teacher prompt ends with the generation prompt, as the student's does")
    ck(opd["teacher_tokens_candidates"][0] is stale,
       "the caller's opd_result is not mutated")
    ck(len(srv.output_queue.calls) == 1, "the round is queued once")

    print("\n[a hint that cannot be placed sends the round in RL-only]")
    srv = make_server()
    lost = dict(opd, metaclaw_task_prefix="a task that is not in this transcript")
    asyncio.run(srv._metaclaw_submit_round("s2", [t1], t2, lost, reward=-1.0))
    ck(len(srv.submitted) == 1 and srv.submitted[0][0] == "rl",
       "no mismatched teacher is shipped: the round is submitted RL-only")
    ck(any("could not place the hint" in m for m in log.messages),
       "the fallback is logged")

    print("\n[no hint accepted: unchanged RL-only path]")
    srv = make_server()
    asyncio.run(srv._metaclaw_submit_round(
        "s3", [t1], t2, {"accepted": False, "metaclaw_verdict": True}, reward=1.0))
    ck(len(srv.submitted) == 1 and srv.submitted[0][0] == "rl",
       "a round without an accepted hint stays RL-only")

    print(f"\nall {n} checks passed")


if __name__ == "__main__":
    main()
