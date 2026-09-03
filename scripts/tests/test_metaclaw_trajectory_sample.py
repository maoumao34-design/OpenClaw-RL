"""Regression assertions for the trajectory-level sample assembly (2026-09-03).

Covers `_metaclaw_build_trajectory` in the patched proxy -- the function that
folds one MetaClaw round's N turns into a single training sample, with the
model's own output masked 1 and the tool observations it conditioned on masked
0 (toolcall-rl's shape, generate_with_retool.py:709-765).

Exercised against the code the patch script actually emits from the real
official source, not a copy pasted here: a test that reimplements the thing it
tests passes no matter what the patch produces.

Usage (needs the official repo checked out):
    python scripts/tests/test_metaclaw_trajectory_sample.py [OFFICIAL_REPO_ROOT]
"""

import logging
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
DEFAULT_OFFICIAL = os.path.normpath(
    os.path.join(SCRIPTS_DIR, "..", "..", "OpenClaw-RL-official")
)

START = "# --- openclaw-rl-metaclaw-trajectory-sample (2026-09-03) ---"
END = "# openclaw-rl-metaclaw (temporary, safe to remove) -- see"


class FakeTokenizer:
    """One token per character, so every length in the test is hand-checkable."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _load_builder(proxy_path, max_tokens=None):
    src = open(proxy_path, encoding="utf-8").read()
    i = src.index(START)
    j = src.index(END, i)
    block = src[i:j]
    assert "def _metaclaw_build_trajectory" in block, (
        "marker found but the builder is not inside it -- patch layout changed"
    )
    env = dict(os.environ)
    if max_tokens is not None:
        env["METACLAW_TRAJ_MAX_TOKENS"] = str(max_tokens)
    ns = {"os": type("_O", (), {"environ": env})(), "logger": logging.getLogger("t")}
    exec(compile(block, proxy_path, "exec"), ns)
    return ns


def _turn(prompt_text, response_text):
    """Build a turn_data the way the official main-turn handler would.

    prompt_ids / response_ids are the char codes of the respective strings, and
    response_logprobs is length-matched to response_ids -- both invariants the
    real handler guarantees before the builder ever sees the turn.
    """
    tok = FakeTokenizer()
    rid = tok(response_text)["input_ids"]
    return {
        "prompt_text": prompt_text,
        "prompt_ids": tok(prompt_text)["input_ids"],
        "response_text": response_text,
        "response_ids": rid,
        "response_logprobs": [-0.5] * len(rid),
        "messages": [{"role": "user", "content": "the round's task"}],
        "tools": [{"name": "write"}],
    }


def _round(*pairs):
    """A well-formed round: every turn's prompt extends the previous full text."""
    pending = {}
    prompt = "P0:"
    for n, (resp, obs) in enumerate(pairs, start=1):
        pending[n] = _turn(prompt, resp)
        prompt = prompt + resp + obs
    return pending


def main():
    official = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OFFICIAL
    if not os.path.isdir(official):
        raise SystemExit(f"official repo not found: {official}")

    tmp = tempfile.mkdtemp(prefix="mc_traj_test_")
    subprocess.run(
        ["bash", os.path.join(SCRIPTS_DIR, "prepare_patched_openclaw_opd.sh"),
         official, tmp],
        check=True, stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["bash", os.path.join(SCRIPTS_DIR, "prepare_patched_openclaw_combine.sh"),
         official, tmp],
        check=True, stdout=subprocess.DEVNULL,
    )
    proxy = os.path.join(tmp, "openclaw_opd_api_server.py")
    ns = _load_builder(proxy)
    build = ns["_metaclaw_build_trajectory"]
    tok = FakeTokenizer()

    n = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    print("default limit:", ns["_METACLAW_TRAJ_MAX_TOKENS"])
    ck(ns["_METACLAW_TRAJ_MAX_TOKENS"] == 31000,
       "default METACLAW_TRAJ_MAX_TOKENS is 31000 (just under max-tokens-per-gpu 32768)")

    # -- happy path ----------------------------------------------------------
    print("\n[three-turn round, prefix intact]")
    pending = _round(("act1", "<obs1>"), ("act2", "<observation2>"), ("final answer", ""))
    r = build(pending, 3, tok)
    ck(r is not None, "a well-formed round assembles")

    expect_resp = "act1" + "<obs1>" + "act2" + "<observation2>" + "final answer"
    ck(r["response_text"] == expect_resp,
       "response text is action/observation/action/observation/action in order")
    ck(r["prompt_text"] == "P0:", "prompt is turn 1's prompt, not the last turn's")
    ck(len(r["response_ids"]) == len(expect_resp), "response_ids covers the whole trajectory")
    ck(len(r["metaclaw_loss_mask"]) == len(r["response_ids"]), "mask is token-aligned")
    ck(len(r["response_logprobs"]) == len(r["response_ids"]), "logprobs are token-aligned")

    mask = r["metaclaw_loss_mask"]
    ck(sum(mask) == len("act1") + len("act2") + len("final answer"),
       "exactly the model's own tokens are trainable")
    ck(len(mask) - sum(mask) == len("<obs1>") + len("<observation2>"),
       "exactly the observation tokens are masked out")

    # Observations must be masked at the right POSITIONS, not merely in the
    # right quantity -- an off-by-one here trains on tool output.
    pos = 0
    ok = True
    for seg, trainable in (("act1", 1), ("<obs1>", 0), ("act2", 1),
                           ("<observation2>", 0), ("final answer", 1)):
        ok = ok and all(m == trainable for m in mask[pos:pos + len(seg)])
        pos += len(seg)
    ck(ok, "every segment is masked at its own position, not just in aggregate")

    ck(all(lp == 0.0 for lp, m in zip(r["response_logprobs"], mask) if m == 0),
       "observation spans carry 0.0 log probs (toolcall-rl's padding)")
    ck(all(lp == -0.5 for lp, m in zip(r["response_logprobs"], mask) if m == 1),
       "action spans keep the log probs SGLang actually returned")

    # The invariant that makes the whole reconstruction checkable.
    ck(r["prompt_text"] + r["response_text"] == pending[3]["prompt_text"] + "final answer",
       "assembled text equals the last turn's own prompt+response (prefix invariant)")

    ck(r["messages"] == pending[1]["messages"],
       "messages come from turn 1 -- its last user message is the round's task, "
       "which is where the OPD hint gets appended")
    ck(r["metaclaw_traj_turns"] == 3, "turn count is reported for the logs")

    # -- degenerate but valid shapes ----------------------------------------
    print("\n[single-turn round]")
    pending = _round(("answered immediately", ""))
    r = build(pending, 1, tok)
    ck(r is not None and set(r["metaclaw_loss_mask"]) == {1},
       "a round answered in one turn has no observations and is fully trainable")

    print("\n[turns beyond the verdict turn are not folded in]")
    pending = _round(("act1", "<obs1>"), ("act2", "<obs2>"), ("stray", ""))
    r = build(pending, 2, tok)
    ck(r is not None and "stray" not in r["response_text"],
       "only turns <= last_turn_num are folded")

    # -- failure paths -------------------------------------------------------
    print("\n[prefix reconstruction failures are detected, not papered over]")
    pending = _round(("act1", "<obs1>"), ("act2", "<obs2>"))
    pending[2]["prompt_text"] = "COMPACTED HISTORY:"          # what compaction looks like
    pending[2]["prompt_ids"] = tok(pending[2]["prompt_text"])["input_ids"]
    ck(build(pending, 2, tok) is None,
       "a rewritten/compacted turn-2 prompt drops the round instead of splicing")

    pending = _round(("act1", "<obs1>"), ("act2", "<obs2>"))
    pending[2]["prompt_text"] = "P0:act"       # history truncated below turn 1's output
    pending[2]["prompt_ids"] = tok(pending[2]["prompt_text"])["input_ids"]
    ck(build(pending, 2, tok) is None,
       "a turn-2 prompt shorter than turn 1's prompt+response is rejected too")

    # Deliberately NOT an error: a next prompt that still starts with
    # prompt+response but adds only a little just means a short observation.
    # There is no way to tell "short tool result" from "truncated" here, and
    # treating the short case as corruption would drop legitimate rounds.
    pending = _round(("act1", "<obs1>"), ("act2", ""))
    pending[2]["prompt_text"] = "P0:act1<"
    pending[2]["prompt_ids"] = tok(pending[2]["prompt_text"])["input_ids"]
    r_short = build(pending, 2, tok)
    ck(r_short is not None and r_short["response_text"] == "act1" + "<" + "act2",
       "a short-but-still-prefixed next prompt is a short observation, not corruption")

    print("\n[other guards]")
    pending = _round(("act1", "<obs1>"), ("act2", ""))
    pending[1]["response_logprobs"] = [-0.5]                    # deliberately wrong length
    ck(build(pending, 2, tok) is None,
       "a logprob/token length mismatch drops the round rather than misaligning it")

    ck(build({}, 3, tok) is None, "an empty pending map returns None, does not crash")
    ck(build(_round(("a", "")), 0, tok) is None, "no turns at or below the verdict turn")

    print("\n[length ceiling]")
    ns_small = _load_builder(proxy, max_tokens=20)
    build_small = ns_small["_metaclaw_build_trajectory"]
    ck(ns_small["_METACLAW_TRAJ_MAX_TOKENS"] == 20, "the ceiling is env-tunable")
    pending = _round(("x" * 50, ""))
    ck(build_small(pending, 1, tok) is None,
       "a trajectory over the ceiling is dropped -- it could never be packed into "
       "a micro-batch, so submitting it would be worse than dropping it")
    pending = _round(("xx", ""))
    ck(build_small(pending, 1, tok) is not None, "a short trajectory still passes")

    # The ceiling counts prompt + response, not response alone: a huge prompt
    # with a tiny response is just as unpackable.
    ns_mid = _load_builder(proxy, max_tokens=30)
    pending = {1: _turn("P" * 40, "hi")}
    ck(ns_mid["_metaclaw_build_trajectory"](pending, 1, tok) is None,
       "the ceiling counts prompt+response, not response alone")

    # -- the consuming side --------------------------------------------------
    print("\n[combine-server side reads the mask]")
    combine = open(os.path.join(tmp, "openclaw_combine_api_server.py"),
                   encoding="utf-8").read()
    ck(combine.count('_mc_loss_mask = turn_data.get("metaclaw_loss_mask")') == 2,
       "both submission paths (OPD+RL and RL-only) read the explicit mask")
    ck(combine.count("sample.loss_mask = list(_mc_loss_mask)") == 2,
       "both paths use it instead of all-ones when present")
    ck(combine.count("sample.loss_mask = [1] * len(response_ids)") == 2,
       "both paths still fall back to all-ones for non-trajectory samples")
    ck(combine.count(
        '"[openclaw-rl-metaclaw-trajectory-sample] session=%s mask/token "') == 2,
       "a mask/token length mismatch is refused rather than silently guessed")

    print("\n[fire gate]")
    opd = open(proxy, encoding="utf-8").read()
    ck('if turn_data.get("metaclaw_round_mode"):' in opd
       and "held (intermediate turn, no judge, no sample)" in opd,
       "intermediate MetaClaw turns fire no judge and produce no sample")
    ck("_traj = _metaclaw_build_trajectory(_pending, turn_num, self.tokenizer)" in opd,
       "the verdict turn assembles the trajectory")
    ck("_pending.pop(_t, None)" in opd,
       "folded turns are removed from pending so the session can drain")

    print(f"\nall {n} assertions passed")


if __name__ == "__main__":
    main()
