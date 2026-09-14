"""Behavioural tests for the trajectory sample builder.

These run against the code the patch script actually GENERATES, not against a
copy of it: the methods are pulled out of the patched server source and exec'd
onto a stub.

THE FIXTURES ARE THE OTHER HALF, AND THEY ARE WHERE THIS FILE FAILED ONCE.
The first version of this test passed all 19 of its assertions while the real
thing scored zero splices in production, because the fixtures were written from
an assumption instead of from a record: they wrapped thinking in
`<think>...</think>`, whereas Qwen3-Thinking's template puts the opening tag in
the GENERATION PROMPT, so a real response carries only the closer. Testing the
generated code against an invented input shape is the same class of mistake as
testing a copy of the code.

So the shape below is taken from run 20260914_154812's record, and
`assert_fixture_is_realistic` refuses any response that contains an opening
`<think>` -- a fixture that drifts back toward the convenient shape fails
before it can make anything else look green.

    prompt_text ends: <|im_end|>\\n<|im_start|>assistant\\n<think>\\n
    response_text:    <thinking prose></think>\\n<tool_call>\\n{...}\\n</tool_call><|im_end|>
    backbone renders: <|im_start|>assistant\\n<tool_call>\\n{...}\\n</tool_call><|im_end|>

What each case pins down:

  - the mask lands on generated spans and nowhere else. A trajectory carries
    tool results inside its response segment, so an off-by-anything mask
    trains the model on text it never wrote.
  - a spliced turn gets its opening <think> back, unmasked. The backbone has
    no opener, so without this the sequence carries a closing tag with nothing
    to close -- a shape the model neither read nor produced.
  - a turn that cannot be located degrades to "not masked" rather than taking
    the round down with it. The 2026-09-03 attempt asserted prefix containment
    and dropped 96 of 120 rounds when the assertion failed.

Usage:
    OPENCLAW_RL_OFFICIAL=<path> python scripts/tests/test_metaclaw_trajectory.py
"""

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS_DIR)
OFFICIAL = os.environ.get(
    "OPENCLAW_RL_OFFICIAL", os.path.join(os.path.dirname(ROOT), "OpenClaw-RL-official")
)

THINK_CLOSE = "</think>"
ASSISTANT_HDR = "<|im_start|>assistant"
GEN_TAIL = "\n<think>\n"          # what the generation prompt appends
IM_END = "<|im_end|>"


class FakeTokenizer:
    """One id per character, so spans can be checked by length and position."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class StubServer:
    def __init__(self):
        self.tokenizer = FakeTokenizer()


class StubLogger:
    def __init__(self):
        self.messages = []

    def warning(self, fmt, *a):
        self.messages.append(fmt % a if a else fmt)

    error = warning

    def info(self, *a, **k):
        pass


def build_patched_server():
    src = os.path.join(OFFICIAL, "openclaw-combine", "openclaw_combine_api_server.py")
    if not os.path.exists(src):
        print(f"  -- skipped: {src} not present (set OPENCLAW_RL_OFFICIAL)")
        return None
    dest = tempfile.mkdtemp(prefix="mctraj")
    subprocess.run(
        ["bash", os.path.join(SCRIPTS_DIR, "prepare_patched_openclaw_combine.sh"),
         OFFICIAL, dest],
        check=True, capture_output=True,
    )
    with open(os.path.join(dest, "openclaw_combine_api_server.py"), encoding="utf-8") as f:
        return f.read()


def extract_methods(source):
    wanted = ("_metaclaw_visible", "_metaclaw_generation_tail",
              "_metaclaw_build_trajectory")
    parts = []
    for name in wanted:
        m = re.search(
            r"^(    (?:@staticmethod\n    )?def " + name + r"\(.*?)(?=^    (?:@|async |def ))",
            source, re.S | re.M,
        )
        if not m:
            raise AssertionError(f"could not find {name} in the generated server")
        parts.append(m.group(1))
    ns = {
        "_MC_THINK_CLOSE": THINK_CLOSE,
        "_MC_ASSISTANT_HDR": ASSISTANT_HDR,
        # Read the real default out of the generated file rather than
        # restating it, so a change to the cap cannot leave this test
        # asserting against a number the server no longer uses.
        "_MC_MAX_TRAJ_TOKENS": int(
            re.search(r"_MC_MAX_TRAJ_TOKENS = int\(_mc_os\.getenv\("
                      r"\"METACLAW_MAX_TRAJECTORY_TOKENS\", \"(\d+)\"\)\)",
                      source).group(1)
        ),
        "logger": StubLogger(),
    }
    exec(compile("class _S:\n" + "\n".join(parts), "<generated>", "exec"), ns)
    return ns["_S"], ns["logger"]


# --------------------------------------------------------------------------
# Fixtures in the production shape
# --------------------------------------------------------------------------

def response(thinking, action):
    """A real response: thinking with no opener, then the closer, then the call."""
    return f"{thinking}{THINK_CLOSE}\n{action}{IM_END}"


def rendered(action):
    """How the backbone renders that turn on replay: no thinking at all."""
    return f"{ASSISTANT_HDR}\n{action}{IM_END}"


def turn(resp, prompt_text=None, logprobs=None):
    ids = [ord(c) for c in resp]
    return {
        "response_text": resp,
        "response_ids": ids,
        "response_logprobs": logprobs if logprobs is not None else [-0.5] * len(ids),
        "prompt_text": prompt_text if prompt_text is not None
        else f"TASK{IM_END}{ASSISTANT_HDR}{GEN_TAIL}",
        "messages": [{"role": "user", "content": "task"}],
    }


def assert_fixture_is_realistic(*turns):
    for t in turns:
        r = t["response_text"]
        assert "<think>" not in r, (
            "fixture drifted back to an opening <think>, which real responses "
            "never carry -- that is exactly what hid this bug the first time"
        )
        if r:
            assert THINK_CLOSE in r or "tool_call" not in r, (
                "a fixture response with a tool call should carry the closing tag"
            )


def main():
    source = build_patched_server()
    if source is None:
        return
    Cls, log = extract_methods(source)

    srv = StubServer()
    srv._metaclaw_visible = Cls._metaclaw_visible
    srv._metaclaw_generation_tail = Cls._metaclaw_generation_tail
    srv._metaclaw_build_trajectory = Cls._metaclaw_build_trajectory.__get__(srv)

    n = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    print("[the fixtures match production, or nothing below means anything]")
    A1 = '<tool_call>\n{"name": "write", "arguments": {"path": "day02/a.json"}}\n</tool_call>'
    A2 = '<tool_call>\n{"name": "write", "arguments": {"path": "day02/b.json"}}\n</tool_call>'
    r1 = response("Okay, let's tackle this. I will write the file.", A1)
    rN = response("Now I confirm it landed.", "All done.")
    assert_fixture_is_realistic(turn(r1), turn(rN))
    ck("<think>" not in r1 and THINK_CLOSE in r1,
       "a fixture response carries the closing tag and no opener, as real ones do")

    print("\n[_metaclaw_visible cuts at the closing tag]")
    ck(srv._metaclaw_visible(r1) == A1 + IM_END,
       "thinking is stripped even though there is no opening tag to pair with")
    ck(srv._metaclaw_visible("no tags here") == "no tags here",
       "a response with no thinking at all passes through")
    ck(srv._metaclaw_visible(None) == "", "None is handled")

    print("\n[_metaclaw_generation_tail reads the opener off the prompt]")
    ck(srv._metaclaw_generation_tail(f"X{IM_END}{ASSISTANT_HDR}{GEN_TAIL}") == GEN_TAIL,
       "the tail after the assistant header is returned verbatim")
    ck(srv._metaclaw_generation_tail("no header at all") == "",
       "a prompt with no assistant header yields nothing rather than guessing")

    print("\n[a two-turn round splices, and restores the opener]")
    TASK = "SYSTEM+TASK"
    TOOL = f"{IM_END}\n<|im_start|>user\nTOOLRESULT{IM_END}"
    backbone = TASK + IM_END + rendered(A1) + TOOL + ASSISTANT_HDR + GEN_TAIL
    t1 = turn(r1)
    tN = turn(rN, prompt_text=backbone)
    assert_fixture_is_realistic(t1, tN)
    traj = srv._metaclaw_build_trajectory("s", [t1, tN])
    ck(traj is not None, "the round assembles")
    ck(traj["metaclaw_missing_turns"] == 0,
       "the earlier turn IS located now -- before the fix this was 0 hits in "
       "every multi-turn round of run 20260914_154812")

    masked = "".join(chr(i) for i, m in
                     zip(traj["response_ids"], traj["metaclaw_loss_mask"]) if m)
    unmasked = "".join(chr(i) for i, m in
                       zip(traj["response_ids"], traj["metaclaw_loss_mask"]) if not m)
    ck(masked == r1 + rN,
       "masked tokens are exactly the two full responses, thinking included")
    ck(TOOL in unmasked, "the tool result is present but unmasked")
    ck(masked != traj["response_text"],
       "masked is a strict subset -- if it equalled the whole segment the "
       "splice silently did nothing, which is how the first bug looked")

    print("\n[the spliced thinking gets its opener back, unmasked]")
    full = traj["prompt_text"] + traj["response_text"]
    ck(f"{ASSISTANT_HDR}{GEN_TAIL}{r1}" in full,
       "the reconstruction reads exactly as the turn's own prompt+response did")
    ck(GEN_TAIL in unmasked,
       "the opener is in the sequence but carries no gradient -- it was prompt")
    # Counted over the whole sequence, not the response segment: the first
    # turn's opener sits in the prompt segment, which is where the generation
    # prompt put it.
    ck(full.count(THINK_CLOSE) == full.count("<think>"),
       "every closing tag in the sequence has an opener")

    print("\n[alignment invariants]")
    ck(len(traj["metaclaw_loss_mask"]) == len(traj["response_ids"]),
       "mask length equals response length")
    ck(len(traj["response_logprobs"]) == len(traj["response_ids"]),
       "logprobs length equals response length")
    ck(all(lp == 0.0 for lp, m in zip(traj["response_logprobs"],
                                      traj["metaclaw_loss_mask"]) if not m),
       "logprobs are zero wherever the mask is zero")
    ck(all(lp != 0.0 for lp, m in zip(traj["response_logprobs"],
                                      traj["metaclaw_loss_mask"]) if m),
       "logprobs are the real ones wherever the mask is one")

    print("\n[three turns, all locatable]")
    r2 = response("Second step, writing b.", A2)
    backbone3 = (TASK + IM_END + rendered(A1) + TOOL + rendered(A2) + TOOL
                 + ASSISTANT_HDR + GEN_TAIL)
    tt = [turn(r1), turn(r2), turn(rN, prompt_text=backbone3)]
    assert_fixture_is_realistic(*tt)
    tr = srv._metaclaw_build_trajectory("s", tt)
    ck(tr["metaclaw_missing_turns"] == 0, "both earlier turns are located")
    m3 = "".join(chr(i) for i, m in zip(tr["response_ids"], tr["metaclaw_loss_mask"]) if m)
    ck(m3 == r1 + r2 + rN, "all three responses are masked, in order")
    ck(tr["response_text"].count(GEN_TAIL) == 2,
       "an opener is restored for each spliced turn, and only for those")

    print("\n[a turn that cannot be located degrades, it does not drop the round]")
    lost = response("I did something not in the backbone.", "<tool_call>\nGHOST\n</tool_call>")
    before = len(log.messages)
    tb = [turn(lost), turn(rN, prompt_text=TASK + IM_END + TOOL + ASSISTANT_HDR + GEN_TAIL)]
    assert_fixture_is_realistic(*tb)
    trb = srv._metaclaw_build_trajectory("s", tb)
    ck(trb is not None, "the round still assembles")
    ck(trb["metaclaw_missing_turns"] == 1, "the unlocatable turn is counted")
    ck(len(log.messages) > before, "and warned about, not silently dropped")
    mb = "".join(chr(i) for i, m in zip(trb["response_ids"], trb["metaclaw_loss_mask"]) if m)
    ck(mb == rN, "only the final response is masked")

    print("\n[degenerate rounds]")
    ck(srv._metaclaw_build_trajectory(
        "s", [turn("", prompt_text=TASK)]) is None,
       "a round whose only turn generated nothing returns None")
    ck(srv._metaclaw_build_trajectory("s", [turn(rN, prompt_text="")]) is None,
       "a round whose final turn has no prompt_text returns None")

    print("\n[an oversized round is dropped, not allowed to kill the run]")
    # 20260914_171937 died when gather_log_probs_at_indices was handed 71668
    # index rows for a 21495-token chunk: one round, 17 turns. Losing that
    # round is the cheap failure; losing the run is not.
    big_action = "<tool_call>\n" + ("X" * 40000) + "\n</tool_call>"
    big = response("thinking", big_action)
    bb = TASK + IM_END + rendered(big_action) + TOOL + ASSISTANT_HDR + GEN_TAIL
    tbig = [turn(big), turn(rN, prompt_text=bb)]
    assert_fixture_is_realistic(*tbig)
    before = len(log.messages)
    ck(srv._metaclaw_build_trajectory("s", tbig) is None,
       "a round past the cap returns None instead of producing the sample")
    ck(any("DROPPED an oversized round" in m for m in log.messages[before:]),
       "and says so loudly, with the numbers, so the run can be judged")
    ck(any("round boundary" in m for m in log.messages[before:]),
       "and points at the round boundary, since that is what a round this "
       "large usually means")

    print("\n[non-vacuity: the cap does not fire on a normal round]")
    # Without this, a cap set absurdly low would pass every assertion above
    # while silently discarding every round in production.
    ck(srv._metaclaw_build_trajectory("s", [t1, tN]) is not None,
       "the ordinary two-turn round from earlier still assembles")
    ck(srv._metaclaw_build_trajectory("s", tt) is not None,
       "so does the three-turn round")

    print(f"\nall {n} assertions passed")


if __name__ == "__main__":
    main()
