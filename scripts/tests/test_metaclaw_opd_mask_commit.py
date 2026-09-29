"""The H-e causal test must mask exactly the stop decision, and nothing when off.

prepare_patched_openclaw_combine_select.sh writes a patched copy of the
official openclaw_topk_select_loss.py (openclaw-rl-metaclaw-opd-mask-commit,
TEMPORARY DIAGNOSTIC, 2026-09-29). With METACLAW_OPD_MASK_COMMIT=1 it zeroes
OPD at two kinds of response position: from each </think> through the
<|im_end|> that ends that turn, and wherever </think> is in the student's
top-K. The experiment only means something if:

1. With the variable unset the loss is the official one. Checked here by
   diffing the generated file against the official: no official line may be
   removed or changed, and every added line is a definition or sits behind
   `if _MC_MASK_COMMIT`.
2. The variable actually reaches the Ray actor. The launcher's Python patch
   is run against the official launcher and the resulting RUNTIME_ENV_JSON
   is expanded in bash and parsed as JSON, both with the variable unset and set.
3. The mask is right position by position, inside a multi-turn trajectory
   sample with a tool result between turns. Checked on a hand-built sample,
   and the test proves it can tell: two plausible wrong versions of the
   helper must give a DIFFERENT mask. This part needs torch, so it runs on
   modelfactory and is skipped (and says so) elsewhere.

Usage:
    OPENCLAW_RL_OFFICIAL=<path> python scripts/tests/test_metaclaw_opd_mask_commit.py
    (needs a python3 on PATH for the prepare script; on Windows point one at
    the real interpreter, since python3 there is the Store alias)
"""

import difflib
import json
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
LOSS_REL = os.path.join("openclaw-combine", "openclaw_topk_select_loss.py")
LAUNCHER_REL = os.path.join("openclaw-combine", "run_qwen3_4b_openclaw_topk_select.sh")
PROFILE = os.path.join(SCRIPTS_DIR, "run_openclaw_topk_select_modelfactory.sh")

CLOSE, END, OPEN, IM_START = 151668, 151645, 151667, 151644


def generate(dest):
    subprocess.run(
        ["bash", os.path.join(SCRIPTS_DIR, "prepare_patched_openclaw_combine_select.sh"), OFFICIAL, dest],
        check=True, capture_output=True,
    )
    with open(os.path.join(dest, "openclaw_topk_select_loss.py"), encoding="utf-8") as f:
        return f.read()


def extract_helper(source):
    # From the flag definition to the helper's final return, nothing else.
    m = re.search(r"^(_MC_MASK_COMMIT = .*?^    return ~\(acting \| stop_candidate\)\n)",
                  source, re.S | re.M)
    if not m:
        raise AssertionError("could not find the mask helper in the generated loss")
    return m.group(1)


def main():
    n = 0
    skipped = []

    def ck(cond, label):
        nonlocal n
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        n += 1
        print(f"  ok  {label}")

    official_loss = os.path.join(OFFICIAL, LOSS_REL)
    if not os.path.exists(official_loss):
        print(f"  -- skipped everything: {official_loss} not present (set OPENCLAW_RL_OFFICIAL)")
        return

    # ---------------- 1. off means official ----------------
    print("[with the variable off, the loss is the official one]")
    patched = generate(tempfile.mkdtemp(prefix="mcmask"))
    compile(patched, "patched_loss", "exec")
    ck(True, "the generated loss compiles")
    with open(official_loss, encoding="utf-8") as f:
        official = f.read()
    diff = list(difflib.unified_diff(official.split("\n"), patched.split("\n"), lineterm="", n=0))
    removed = [l for l in diff if l.startswith("-") and not l.startswith("---")]
    added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
    ck(not removed, "no official line is removed or changed -- the patch only adds")
    ck(patched.count("def _metaclaw_commit_keep_mask(") == 1, "the helper is defined once")
    ck(patched.count("if _MC_MASK_COMMIT:") == 2,
       "two gated blocks: the startup notice and the mask application")
    i_gate = patched.index("            if _MC_MASK_COMMIT:\n")
    ck(patched.index("                pg_t = pg_t * _mc_keep.to(pg_t.dtype)\n", i_gate)
       < patched.index("            all_pg.append(pg_t)\n", i_gate),
       "the OPD term is masked before it is collected, inside the gate")
    ck('reported["opd_commit_keep_frac"]' in patched and "if _mc_keep_frac is not None:" in patched,
       "keep_frac is reported only when something was masked")
    ck('os.getenv("METACLAW_OPD_MASK_COMMIT", "0") == "1"' in patched, "the default is off")
    stray = [l for l in added if re.match(r"\s*(pg_t|all_pg|loss|opd_loss)\s*=", l)
             and "_mc_keep" not in l]
    ck(not stray, "no added line reassigns the loss outside the mask")

    # ---------------- 2. the variable reaches the Ray actor ----------------
    print("\n[the variable travels in RUNTIME_ENV_JSON]")
    with open(PROFILE, encoding="utf-8") as f:
        profile = f.read()
    m = re.search(r'^python3 - "\$\{PATCHED\}" "\$\{REPO_ROOT\}" <<\'PY\'\n(.*?)\nPY\n', profile, re.S | re.M)
    ck(m is not None, "found the launcher patch in the profile")
    work = tempfile.mkdtemp(prefix="mcrt")
    launcher = os.path.join(work, "launcher.sh")
    with open(os.path.join(OFFICIAL, LAUNCHER_REL), encoding="utf-8") as f:
        open(launcher, "w", encoding="utf-8", newline="\n").write(f.read())
    patch_py = os.path.join(work, "patch.py")
    open(patch_py, "w", encoding="utf-8", newline="\n").write(m.group(1))
    env = dict(os.environ, PYTHONUTF8="1")
    subprocess.run([sys.executable, patch_py, launcher, "/repo"], check=True, capture_output=True, env=env)
    text = open(launcher, encoding="utf-8").read()
    block = re.search(r'^RUNTIME_ENV_JSON="\{\n.*?\n\}"$', text, re.S | re.M).group(0)
    base_env = ("REPO_ROOT=/repo SCRIPT_DIR=/s SLIME_ROOT=/sl OPENCLAW_EVAL_MODE=1 "
                "OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron OPENCLAW_TOPK_W_RL=0 OPENCLAW_TOPK_W_OPD=1 "
                "OPENCLAW_TOPK_ADV_DIFF_CLIP=1 OPENCLAW_TOPK_MAX_CAND=3 TRAIN_EPOCHS=1 WANDB_API_KEY=k")
    for setting, want in (("", "0"), ("METACLAW_OPD_MASK_COMMIT=1", "1")):
        script = f"{base_env} {setting}\n{block}\nprintf '%s' \"$RUNTIME_ENV_JSON\"\n"
        out = subprocess.run(["bash", "-c", "set -a\n" + script], check=True, capture_output=True, text=True)
        ev = json.loads(out.stdout)["env_vars"]
        ck(ev["METACLAW_OPD_MASK_COMMIT"] == want and ev["METACLAW_THINK_CLOSE_ID"] == str(CLOSE)
           and ev["METACLAW_IM_END_ID"] == str(END) and ev["OPENCLAW_TOPK_W_RL"] == "0",
           f"valid JSON; mask={want} when {'set' if setting else 'unset'}, ids and W_RL still carried")

    # ---------------- 3. the mask, position by position ----------------
    print("\n[the mask on a two-turn trajectory sample]")
    try:
        import torch
    except ImportError:
        skipped.append("mask behaviour (needs torch; run on modelfactory)")
        print("  -- skipped: torch not installed here; run this test on modelfactory")
        torch = None
    if torch is not None:
        helper = extract_helper(patched)

        def build(src, on="1"):
            os.environ["METACLAW_OPD_MASK_COMMIT"] = on
            ns = {"os": os, "torch": torch, "print": lambda *a, **k: None}
            exec(compile(src, "helper", "exec"), ns)
            return ns

        ck(build(helper, on="0")["_MC_MASK_COMMIT"] is False and build(helper)["_MC_MASK_COMMIT"] is True,
           "the flag follows the environment variable")
        # turn 1: think(0-2) </think>(3) act(4-5) <|im_end|>(6)
        # tool result and generation tail (7-11), loss_mask 0 in training
        # turn 2: think(12-13) </think>(14) act(15) <|im_end|>(16), trailing newline (17)
        toks = [11, 12, 13, CLOSE, 20, 21, END, 30, 31, IM_START, 77, OPEN,
                14, 15, CLOSE, 22, END, 198]
        R = len(toks)
        s_idx = torch.tensor([[1000 + i, 2000 + i, 3000 + i, 4000 + i] for i in range(R)])
        s_idx[1, 2] = CLOSE      # </think> among the student's top-4 while still thinking
        s_idx[12, 0] = CLOSE
        want = [True, False, True, False, False, False, False, True, True, True, True, True,
                False, True, False, False, False, True]
        f = build(helper)["_metaclaw_commit_keep_mask"]
        got = f(torch.tensor(toks), s_idx).tolist()
        ck(got == want, "masked: each </think> through its <|im_end|>, and </think>-in-top-K positions; "
                        "kept: other thinking, the tool result, the generation tail")
        ck(f(torch.tensor([], dtype=torch.long), torch.zeros(0, 4, dtype=torch.long)).tolist() == [],
           "an empty response gives an empty mask")
        wrong_end = helper.replace("prev_end = torch.cat([neg[:1], last_end[:-1]])", "prev_end = last_end")
        ck(wrong_end != helper and build(wrong_end)["_metaclaw_commit_keep_mask"](
            torch.tensor(toks), s_idx).tolist() != want,
           "control: counting a turn's own <|im_end|> as already closed gives a different mask")
        no_cand = helper.replace("return ~(acting | stop_candidate)", "return ~acting")
        ck(no_cand != helper and build(no_cand)["_metaclaw_commit_keep_mask"](
            torch.tensor(toks), s_idx).tolist() != want,
           "control: ignoring </think>-in-top-K gives a different mask")
        os.environ.pop("METACLAW_OPD_MASK_COMMIT", None)

    print(f"\nall {n} assertions passed" + (f"; skipped: {', '.join(skipped)}" if skipped else ""))


if __name__ == "__main__":
    main()
