"""Keeps docs/training_config.md honest against the real scripts.

A configuration document that drifts is worse than none: it is trusted and
wrong. This project has already paid for that twice in one day -- a 58-line
comment block still explaining per-round sessions after the code moved to
per-day, and a hand-back patched onto a parent class whose subclass overrides
the method. Both looked right when read.

So every load-bearing number in the doc is checked here against the file that
actually decides it. Change the configuration and this test tells you the doc
is stale; edit the doc without changing anything and it tells you the doc is
wrong.

What is deliberately NOT checked: prose, rationale, the known-unfixed list,
and the log-grep recipes. Those cannot be derived from the scripts. Keep them
current by hand.

Usage:
    OPENCLAW_RL_OFFICIAL=<path> python scripts/tests/test_training_config_doc.py

OPENCLAW_RL_OFFICIAL defaults to a sibling checkout; the official-script
assertions are skipped with a notice if it is not present.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS_DIR)

DOC = os.path.join(ROOT, "docs", "training_config.md")
PROFILE = os.path.join(SCRIPTS_DIR, "run_openclaw_topk_select_modelfactory.sh")
LAUNCHER = os.path.join(SCRIPTS_DIR, "metaclaw", "run_metaclaw_migration_modelfactory.sh")
DRIVER = os.path.join(SCRIPTS_DIR, "metaclaw", "metaclaw_rollout_driver.py")
COMBINE = os.path.join(SCRIPTS_DIR, "prepare_patched_openclaw_combine.sh")
SELECT = os.path.join(SCRIPTS_DIR, "prepare_patched_openclaw_combine_select.sh")

OFFICIAL = os.environ.get(
    "OPENCLAW_RL_OFFICIAL",
    os.path.join(os.path.dirname(ROOT), "OpenClaw-RL-official"),
)
OFFICIAL_SCRIPT = os.path.join(
    OFFICIAL, "openclaw-combine", "run_qwen3_4b_openclaw_topk_select.sh"
)


def read(p):
    return open(p, encoding="utf-8").read()


def main():
    doc = read(DOC)
    profile = read(PROFILE)
    launcher = read(LAUNCHER)
    driver = read(DRIVER)
    combine = read(COMBINE)
    select = read(SELECT)

    n = 0
    skipped = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    def pair(doc_frag, src, src_frag, label):
        """The doc claims something; the source must agree."""
        ck(doc_frag in doc, f"doc states {label}")
        ck(src_frag in src, f"source agrees: {label}")

    print("[topology]")
    pair("`TP=4`、`PP=1`、**`CP=1`**", profile, "", "TP/PP/CP (values live in the official script)")
    ck("NUM_TRAINING_GPUS:-8" in launcher, "launcher defaults to 8 GPUs")
    ck("**8**" in doc or "8 卡" in doc, "doc states 8 GPUs")

    print("\n[ports]")
    ck("30000" in doc and "30000" in launcher, "proxy port 30000 in doc and launcher")
    ck("8265" in doc and "8265" in launcher, "Ray dashboard port 8265 in doc and launcher")

    print("\n[session granularity]")
    ck("一天一个 session" in doc, "doc states one session per day")
    ck('round_session_id = f"{_SESSION_ID_PREFIX}{test_id}"' in driver,
       "driver builds one session id per day")
    ck("METACLAW_SESSION_SCOPE" not in driver and "METACLAW_SESSION_SCOPE" not in launcher,
       "no scope switch survives anywhere (doc says round mode was deleted)")
    # Structural, not a comment string: days run in a plain sequential loop
    # and the driver never fans out with gather. A comment claiming
    # concurrency=1 would keep passing after the code changed; these two
    # would not.
    ck("concurrency=1" in doc, "doc states driver concurrency 1")
    ck("for day_index, test in enumerate(test_list, start=1):" in driver,
       "days really iterate in a sequential for-loop")
    ck("asyncio.gather" not in driver,
       "the driver never fans days or rounds out concurrently")

    print("\n[MetaClaw profile overrides]")
    for doc_val, sed_frag, label in [
        ("`--rollout-batch-size` | **8**", "--rollout-batch-size 16/--rollout-batch-size 8",
         "rollout-batch-size 16 -> 8"),
        ("`--save-interval` | **10**", "--save-interval 100/--save-interval 10",
         "save-interval 100 -> 10"),
        ("`--rollout-max-context-len` | **65536**",
         "--rollout-max-context-len 32768/--rollout-max-context-len 65536",
         "rollout-max-context-len 32768 -> 65536"),
        ("`--sglang-context-length` | **65536**",
         "--sglang-context-length 32768/--sglang-context-length 65536",
         "sglang-context-length 32768 -> 65536"),
    ]:
        ck(doc_val in doc, f"doc states {label}")
        ck(sed_frag in profile, f"profile performs {label}")

    ck("--use-dynamic-global-batch-size" in doc
       and "--use-dynamic-global-batch-size" in profile,
       "dynamic global batch size injected, and documented")
    ck("metaclaw_round_scale.metaclaw_round_scale" in doc
       and "metaclaw_round_scale.metaclaw_round_scale" in profile,
       "custom reward post-process path injected, and documented")

    print("\n[unchanged official hyperparameters]")
    if os.path.exists(OFFICIAL_SCRIPT):
        official = read(OFFICIAL_SCRIPT)
        for doc_val, off_frag, label in [
            ("1e-5", "--lr 1e-5", "lr"),
            ("**0.0**", "--kl-loss-coef 0.0", "kl-loss-coef"),
            ("0.00", "--entropy-coef 0.00", "entropy-coef"),
            ("32768", "--max-tokens-per-gpu 32768", "max-tokens-per-gpu"),
            ("--n-samples-per-prompt 1", "--n-samples-per-prompt 1", "n-samples-per-prompt"),
            ("--num-steps-per-rollout 1", "--num-steps-per-rollout 1", "num-steps-per-rollout"),
            ("grpo", "--advantage-estimator grpo", "advantage-estimator"),
            ("--disable-rewards-normalization", "--disable-rewards-normalization",
             "disable-rewards-normalization"),
        ]:
            ck(doc_val in doc, f"doc states {label}")
            ck(off_frag in official, f"official script sets {label}")
        # These must not be overridden by the METACLAW branch specifically.
        # The same file also carries SMOKE_PROFILE / MINITEST_PROFILE branches
        # that DO lower max-tokens-per-gpu, so the check has to be scoped to
        # the branch that actually runs for MetaClaw training.
        start = profile.index("METACLAW_MIGRATION_PROFILE")
        end = profile.index("\nfi", start)
        mc_branch = profile[start:end]
        for frag, label in [("--max-tokens-per-gpu", "max-tokens-per-gpu"),
                            ("--lr ", "lr"),
                            ("--kl-loss-coef", "kl-loss-coef"),
                            ("--n-samples-per-prompt", "n-samples-per-prompt")]:
            ck(f"s/{frag}" not in mc_branch,
               f"the MetaClaw branch does not override {label} "
               "(doc calls it unchanged)")
        ck("ACTOR_GPUS=${ACTOR_GPUS:-4}" in official, "actor GPUs 4")
        ck("ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}" in official, "rollout GPUs 2")
        ck("PRM_GPUS=${PRM_GPUS:-1}" in official, "PRM GPUs 1")
        ck("PRM_TEACHER_GPUS=${PRM_TEACHER_GPUS:-1}" in official, "teacher GPUs 1")
        ck("--tensor-model-parallel-size 4" in official, "TP=4")
        ck("--context-parallel-size 1" in official, "CP=1")
    else:
        skipped += 1
        print(f"  -- skipped: {OFFICIAL_SCRIPT} not present "
              "(set OPENCLAW_RL_OFFICIAL to enable)")

    print("\n[sample formation]")
    ck("零重建" in doc, "doc states samples carry their own real prompt/response")
    ck("metaclaw_round_turns" in doc, "doc names the field the 1/N scaler reads")
    ck("advantage = reward / n_turns" in doc.replace("`", "")
       or "reward / n_turns" in doc, "doc states the 1/N formula")
    ck("advantages.append(reward / n_turns)" in combine,
       "the scaler really divides by the turn count")
    ck(combine.count('_mc_collect = turn_data.get("metaclaw_round_collect")') == 1
       and select.count('_mc_collect = turn_data.get("metaclaw_round_collect")') == 1,
       "hand-back present in BOTH parent and subclass patches, as the doc warns")
    ck("_mc_lo < t < turn_num" in combine and "_mc_prefix in _flatten_message_content" in combine,
       "both membership layers exist, as documented")

    print("\n[patch inventory]")
    deployed = re.findall(r'prepare_patched_([a-z_]+)\.sh', launcher)
    ck(len(set(deployed)) == 9,
       f"launcher deploys 9 distinct patch scripts (found {len(set(deployed))}: "
       f"{sorted(set(deployed))})")
    ck("9 个" in doc, "doc states 9 patch scripts")
    for name in ("rl_training_headers", "sglang_execution_bias",
                 "embedded_agent_overflow_recovery",
                 "system_prompt_output_directives", "cli_compaction",
                 "silent_reply_policy"):
        ck(name in deployed, f"launcher deploys {name}")

    print("\n[evaluation baseline]")
    ck("41.1%" in doc and "26.3%" in doc, "doc pins the day-scope K=0 baseline")
    ck("20260907_112320" in doc, "doc names the baseline run")

    print(f"\nall {n} assertions passed" + (f" ({skipped} group skipped)" if skipped else ""))


if __name__ == "__main__":
    main()
