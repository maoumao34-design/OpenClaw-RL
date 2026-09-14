"""Behavioural tests for the trajectory sample builder.

These run against the code the patch script actually GENERATES, not against a
copy of it: the two methods are pulled out of the patched server source and
exec'd onto a stub. A test that re-implemented the algorithm would keep passing
after the patch drifted, which is the failure mode this project has already
paid for twice (a comment left describing per-round sessions after the code
moved to per-day; a hand-back patched onto a parent whose subclass overrides
the method).

What matters here, and what each case pins down:

  - the mask lands on generated spans and nowhere else. A trajectory carries
    tool results inside its response segment, so an off-by-anything mask
    trains the model on text it never wrote.
  - a turn that cannot be located degrades to "not masked" rather than taking
    the round down with it. The 2026-09-03 attempt asserted prefix containment
    and dropped 96 of 120 rounds when the assertion failed.
  - logprobs stay aligned to ids, with zeros on the unmasked spans, which is
    the shape slime's own multi-turn example produces.

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

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class FakeTokenizer:
    """One id per character, so spans can be checked by length and position."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class StubServer:
    def __init__(self):
        self.tokenizer = FakeTokenizer()


class StubLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, fmt, *a):
        self.warnings.append(fmt % a if a else fmt)

    def error(self, fmt, *a):
        self.warnings.append(fmt % a if a else fmt)

    def info(self, *a, **k):
        pass


def build_patched_server():
    """Generate the patched server and return its source."""
    src = os.path.join(
        OFFICIAL, "openclaw-combine", "openclaw_combine_api_server.py"
    )
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
    """Pull the two trajectory methods out of the generated file and bind them."""
    out = {}
    for name in ("_metaclaw_visible", "_metaclaw_build_trajectory"):
        m = re.search(
            r"^(    (?:@staticmethod\n    )?def " + name + r"\(.*?)(?=^    (?:@|async |def ))",
            source, re.S | re.M,
        )
        if not m:
            raise AssertionError(f"could not find {name} in the generated server")
        out[name] = m.group(1)

    body = "class _S:\n" + out["_metaclaw_visible"] + "\n" + out["_metaclaw_build_trajectory"]
    ns = {"_MC_THINK_RE": THINK_RE, "logger": StubLogger()}
    exec(compile(body, "<generated>", "exec"), ns)
    return ns["_S"], ns["logger"]


def turn(response, prompt_text=None, logprobs=None):
    ids = [ord(c) for c in response]
    return {
        "response_text": response,
        "response_ids": ids,
        "response_logprobs": logprobs if logprobs is not None else [-0.5] * len(ids),
        "prompt_text": prompt_text or "",
        "messages": [{"role": "user", "content": "task"}],
    }


def visible(text):
    return THINK_RE.sub("", text).strip()


def main():
    source = build_patched_server()
    if source is None:
        return
    Cls, log = extract_methods(source)

    srv = StubServer()
    srv._metaclaw_visible = Cls._metaclaw_visible
    srv._metaclaw_build_trajectory = Cls._metaclaw_build_trajectory.__get__(srv)

    n = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    TASK = "SYSTEM+TASK|"
    TOOL = "|TOOLRESULT|"

    print("[a normal two-turn round]")
    r1 = "<think>thinking one</think>WRITE_CALL_1"
    r2 = "<think>thinking two</think>DONE"
    t1 = turn(r1)
    t2 = turn(r2, prompt_text=TASK + visible(r1) + TOOL)
    traj = srv._metaclaw_build_trajectory("s", [t1, t2])
    ck(traj is not None, "a locatable round assembles")
    ck(traj["prompt_text"] == TASK, "prompt is everything before the first generation")
    ck(len(traj["metaclaw_loss_mask"]) == len(traj["response_ids"]),
       "mask length equals response length")
    ck(len(traj["response_logprobs"]) == len(traj["response_ids"]),
       "logprobs length equals response length")
    ck(traj["metaclaw_missing_turns"] == 0, "no turn reported missing")

    # The masked ids must be exactly the two responses, in order.
    masked = [i for i, m in zip(traj["response_ids"], traj["metaclaw_loss_mask"]) if m]
    ck(masked == [ord(c) for c in r1 + r2],
       "masked tokens are exactly the two full responses, thinking included")
    unmasked = [i for i, m in zip(traj["response_ids"], traj["metaclaw_loss_mask"]) if not m]
    ck(unmasked == [ord(c) for c in TOOL],
       "unmasked tokens are exactly the tool result")
    ck(all(lp == 0.0 for lp, m in zip(traj["response_logprobs"],
                                      traj["metaclaw_loss_mask"]) if not m),
       "logprobs are zero wherever the mask is zero")
    ck(all(lp != 0.0 for lp, m in zip(traj["response_logprobs"],
                                      traj["metaclaw_loss_mask"]) if m),
       "logprobs are the real ones wherever the mask is one")

    print("\n[a turn that cannot be located degrades, it does not drop the round]")
    t1b = turn("<think>x</think>NOT_IN_THE_BACKBONE")
    t2b = turn(r2, prompt_text=TASK + TOOL)
    before = len(log.warnings)
    traj_b = srv._metaclaw_build_trajectory("s", [t1b, t2b])
    ck(traj_b is not None, "the round still assembles")
    ck(traj_b["metaclaw_missing_turns"] == 1, "the unlocatable turn is counted")
    ck(len(log.warnings) > before, "and it is warned about, not silently dropped")
    masked_b = [i for i, m in zip(traj_b["response_ids"], traj_b["metaclaw_loss_mask"]) if m]
    ck(masked_b == [ord(c) for c in r2],
       "only the final response is masked; the missing turn contributes no gradient")

    print("\n[non-vacuity: the mask really can be wrong]")
    # If the builder ignored the tool result, masked would equal the whole
    # response segment. Prove the two differ for this input.
    ck(len(traj["response_ids"]) > len(masked),
       "the response segment is strictly larger than the masked part")
    ck(visible(r1) in t2["prompt_text"] and r1 not in t2["prompt_text"],
       "the fixture really does strip thinking from the backbone, as production does")

    print("\n[a round with nothing to mask is dropped]")
    t_empty = turn("", prompt_text=TASK)
    ck(srv._metaclaw_build_trajectory("s", [t_empty]) is None,
       "a round whose only turn generated nothing returns None")

    print("\n[no backbone at all]")
    ck(srv._metaclaw_build_trajectory("s", [turn("abc", prompt_text="")]) is None,
       "a round whose final turn has no prompt_text returns None")

    print("\n[three turns, middle one unlocatable]")
    m1 = "<think>a</think>ACT_ONE"
    m2 = "<think>b</think>ACT_TWO_MISSING"
    m3 = "<think>c</think>FINAL"
    tt = [turn(m1), turn(m2),
          turn(m3, prompt_text=TASK + visible(m1) + TOOL + TOOL)]
    tr = srv._metaclaw_build_trajectory("s", tt)
    ck(tr["metaclaw_missing_turns"] == 1, "exactly one turn reported missing")
    masked_c = [i for i, m in zip(tr["response_ids"], tr["metaclaw_loss_mask"]) if m]
    ck(masked_c == [ord(c) for c in m1 + m3],
       "the located turns are masked and the missing one is not")

    print(f"\nall {n} assertions passed")


if __name__ == "__main__":
    main()
