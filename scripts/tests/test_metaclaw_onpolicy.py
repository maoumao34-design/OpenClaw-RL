"""H-f diagnostic: train only on trajectories from the weights being trained.

openclaw-rl-metaclaw-onpolicy (TEMPORARY DIAGNOSTIC, 2026-10-08) spans four
generated files and the launcher. It is only a valid test of H-f if:

1. Every turn is tagged with the weight version sglang reported for it, and
   the tags reach the round's sample -- or, with the switch off, nothing at all
   is added (proxy: prepare_patched_openclaw_opd.sh; hand-off:
   prepare_patched_openclaw_combine.sh).
2. The rollout keeps exactly the trajectories whose every turn carries the
   target version, drops stale and untagged ones and counts them, does not
   pause the proxy, and still archives records -- and with the switch off it
   behaves as the official rollout. The patched rollout is EXECUTED here, with
   slime replaced by stubs, not just read.
3. The train loop starts collecting the next batch after the weight update,
   not before training, and nowhere else.
4. The switch reaches Ray, and only when it is on does the launcher run the
   patched loop.

Usage:
    OPENCLAW_RL_OFFICIAL=<path> python scripts/tests/test_metaclaw_onpolicy.py
    (needs a python3 on PATH for the prepare scripts; on Windows point one at
    the real interpreter, since python3 there is the Store alias)
"""

import asyncio
import contextlib
import enum
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.environ.get("ONPOLICY_SCRIPTS_DIR") or os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))
OFFICIAL = os.environ.get(
    "OPENCLAW_RL_OFFICIAL", os.path.join(os.path.dirname(ROOT), "OpenClaw-RL-official")
)
LAUNCHER_REL = os.path.join("openclaw-combine", "run_qwen3_4b_openclaw_topk_select.sh")


def prepare(script):
    dest = tempfile.mkdtemp(prefix="mconp")
    subprocess.run(["bash", os.path.join(SCRIPTS_DIR, script), OFFICIAL, dest],
                   check=True, capture_output=True)
    return dest


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def extract_def(source, name):
    m = re.search(rf"^(def {name}\(.*?)(?=^\S|\Z)", source, re.S | re.M)
    if not m:
        raise AssertionError(f"could not find {name}")
    return m.group(1)


# ---------------------------------------------------------------------------
# stubs for executing the patched rollout without slime
# ---------------------------------------------------------------------------

class StubSample:
    class Status(enum.Enum):
        COMPLETED = "completed"
        ABORTED = "aborted"

    def __init__(self, versions=None, length=10, status=None, index=0):
        self.metadata = {} if versions is None else {"metaclaw_weight_versions": versions}
        self.response_length = length
        self.status = status or StubSample.Status.COMPLETED
        self.index = index


class StubOutput:
    def __init__(self, samples, metrics=None):
        self.samples, self.metrics = samples, metrics


def load_rollout(source):
    stubs = {
        "openclaw_combine_select_api_server": types.SimpleNamespace(OpenClawCombineSelectAPIServer=object),
        "slime": types.ModuleType("slime"),
        "slime.rollout": types.ModuleType("slime.rollout"),
        "slime.rollout.base_types": types.SimpleNamespace(RolloutFnTrainOutput=StubOutput),
        "slime.rollout.sglang_rollout": types.SimpleNamespace(eval_rollout=None),
        "slime.utils": types.ModuleType("slime.utils"),
        "slime.utils.async_utils": types.SimpleNamespace(run=asyncio.run),
        "slime.utils.types": types.SimpleNamespace(Sample=StubSample),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        mod = types.ModuleType("patched_rollout")
        exec(compile(source, "patched_rollout", "exec"), mod.__dict__)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


class FakeServer:
    def __init__(self):
        self.calls = []

    def reset_eval_scores(self):
        self.calls.append("reset")

    def drain_eval_scores(self):
        return []

    def purge_record_files(self):
        self.calls.append("purge")


class FakeWorker:
    def __init__(self, groups):
        self._groups = list(groups)
        self._server = FakeServer()
        self.calls = []

    def get_completed_groups(self):
        out, self._groups = self._groups, []
        return out

    def get_queue_size(self):
        return len(self._groups)

    def resume_submission(self):
        self.calls.append("resume")

    def pause_submission(self):
        self.calls.append("pause")
        self._server.purge_record_files()


def main():
    n = 0
    skipped = []

    def ck(cond, label):
        nonlocal n
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        n += 1
        print(f"  ok  {label}")

    if not os.path.exists(os.path.join(OFFICIAL, LAUNCHER_REL)):
        print(f"  -- skipped everything: {OFFICIAL} not present (set OPENCLAW_RL_OFFICIAL)")
        return

    # ---------------- 1a. the proxy tags turns ----------------
    print("[proxy: tag each turn with its weight version]")
    opd = read(os.path.join(prepare("prepare_patched_openclaw_opd.sh"), "openclaw_opd_api_server.py"))
    ck('_MC_ONPOLICY = _mc_os_onpolicy.getenv("METACLAW_ONPOLICY", "0") == "1"' in opd, "the switch defaults to off")
    ck(opd.count("turn_data[\"weight_version\"] = _mc_weight_version_of(output)") == 1
       and "            if _MC_ONPOLICY:\n                turn_data[\"weight_version\"]" in opd,
       "the tag is added once, and only behind the switch")
    i_out = opd.index("            output = sglang_resp.json()")
    i_tag = opd.index("turn_data[\"weight_version\"]")
    i_def = opd.rindex("    async def _handle_request(", 0, i_tag)
    ck(i_def < i_out < i_tag, "the tag is set inside the request handler, after sglang's response is read")
    ns = {}
    exec(extract_def(opd, "_mc_weight_version_of"), ns)
    f = ns["_mc_weight_version_of"]
    ck(f({"metadata": {"weight_version": "3"}}) == "3", "reads metadata.weight_version (the chat endpoint)")
    ck(f({"meta_info": {"weight_version": 2}}) == "2", "also reads meta_info.weight_version, as a string")
    ck(f({"choices": []}) is None and f(None) is None, "absent -> None, never a guess")

    # ---------------- 1b. the tags reach the round's sample ----------------
    print("\n[round hand-off: tags reach the sample]")
    comb = read(os.path.join(prepare("prepare_patched_openclaw_combine.sh"), "openclaw_combine_api_server.py"))
    put = "        await asyncio.to_thread(self.output_queue.put, (group_index, collect))\n"
    marker = "        # --- openclaw-rl-metaclaw-onpolicy (2026-10-08) -- TEMPORARY DIAGNOSTIC (H-f) ---\n"
    ck(comb.count(marker) == 1 and comb.index(marker) < comb.index(put),
       "the hand-off sits right before the round's queue put")
    block = comb[comb.index(marker):comb.index(put)]
    body = "def handoff(all_tds, collect):\n" + "".join("    " + l[8:] + "\n" for l in block.rstrip("\n").split("\n"))
    ns = {}
    exec(body, ns)
    s = types.SimpleNamespace(metadata={"metaclaw_round_id": "r1"})
    ns["handoff"]([{"weight_version": "4"}, {"weight_version": "5"}], [s])
    ck(s.metadata == {"metaclaw_round_id": "r1", "metaclaw_weight_versions": ["4", "5"]},
       "every turn's version is carried, existing metadata kept")
    s2 = types.SimpleNamespace(metadata={"metaclaw_round_id": "r1"})
    ns["handoff"]([{}, {}], [s2])
    ck(s2.metadata == {"metaclaw_round_id": "r1"}, "untagged turns (switch off) -> nothing added")

    # ---------------- 2. the rollout, executed ----------------
    print("\n[rollout: keep only the target version, executed with stubs]")
    sel_dir = prepare("prepare_patched_openclaw_combine_select.sh")
    rollout_src = read(os.path.join(sel_dir, "openclaw_combine_select_rollout.py"))
    R = load_rollout(rollout_src)
    args = types.SimpleNamespace(rollout_batch_size=3, sglang_router_ip="127.0.0.1", sglang_router_port=1)

    ck(R._mc_group_versions([StubSample(["5", "5"]), StubSample(["5"])]) == {"5"}, "one version across a group")
    ck(R._mc_group_versions([StubSample(["4", "5"])]) == {"4", "5"}, "a trajectory spanning an update has two")
    ck(R._mc_group_versions([StubSample()]) is None and R._mc_group_versions([StubSample(["5", None])]) is None,
       "any untagged turn -> None")

    def groups():
        return [(0, [StubSample(["5", "5"], 10)]), (1, [StubSample(["4", "5"], 90)]),
                (2, [StubSample()]), (3, [StubSample(["5"], 20)]), (4, [StubSample(["4"], 70)]),
                (5, [StubSample(["5"], 30)])]

    stats = {"stale": 0, "untagged": 0, "stale_lens": [], "kept_lens": []}
    with contextlib.redirect_stdout(io.StringIO()):
        kept = asyncio.run(R._drain_output_queue(args, FakeWorker(groups()), mc_target="5", mc_stats=stats))
    ck([g[0].response_length for g in kept] == [10, 20, 30], "kept: exactly the three version-5 groups")
    ck(stats["stale"] == 2 and stats["untagged"] == 1, "dropped and counted: 2 stale, 1 untagged")
    ck(sorted(stats["stale_lens"]) == [70, 90] and stats["kept_lens"] == [10, 20, 30],
       "lengths of kept and dropped are recorded for the confound check")
    with contextlib.redirect_stdout(io.StringIO()):
        off = asyncio.run(R._drain_output_queue(args, FakeWorker(groups())))
    ck([g[0].response_length for g in off] == [10, 90, 10], "switch off: the official drain, no filtering")

    real_version = R._mc_current_weight_version
    R.get_global_worker = lambda a, b: w
    R._mc_current_weight_version = lambda a: "5"
    w = FakeWorker(groups())
    R._MC_ONPOLICY = True
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        res = R.generate_rollout_openclaw_combine_select(args, 7, None)
    line = [l for l in out.getvalue().splitlines() if "openclaw-rl-metaclaw-onpolicy" in l]
    ck("pause" not in w.calls and "purge" in w._server.calls,
       "on: the proxy is not paused, and the record archive still happens")
    ck(len(res.samples) == 3 and line and "rollout=7 target=5 kept=3 stale=2 untagged=1" in line[0],
       "on: one summary line per step with target, kept, stale and untagged")
    w = FakeWorker(groups())
    R._MC_ONPOLICY = False
    with contextlib.redirect_stdout(io.StringIO()):
        res = R.generate_rollout_openclaw_combine_select(args, 7, None)
    ck(w.calls == ["resume", "pause"] and len(res.samples) == 3, "off: resume, drain, pause -- as official")
    R._mc_current_weight_version = real_version

    print("\n[rollout: reading the version]")

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(table):
        def opener(url, timeout=None):
            for suffix, payload in table.items():
                if url.endswith(suffix):
                    if isinstance(payload, Exception):
                        raise payload
                    return Resp(json.dumps(payload).encode())
            raise OSError("404")
        return opener

    real = R._mc_urlreq.urlopen
    try:
        R._mc_urlreq.urlopen = fake_urlopen({"/model_info": OSError("404"), "/get_model_info": {"weight_version": "7"}})
        ck(R._mc_current_weight_version(args) == "7", "tries the known paths in turn until one has the field")
        R._mc_urlreq.urlopen = fake_urlopen({"/model_info": {"server_args": {"weight_version": "3"}}})
        ck(R._mc_current_weight_version(args) == "3", "finds it one level down")
        R._mc_urlreq.urlopen = fake_urlopen({"/model_info": {"model_path": "x"}})
        try:
            R._mc_current_weight_version(args)
            raised = False
        except RuntimeError:
            raised = True
        ck(raised, "no version anywhere -> raises instead of guessing")
        os.environ["METACLAW_WEIGHT_VERSION_URL"] = "http://engine:30001/custom"
        R._mc_urlreq.urlopen = fake_urlopen({"/custom": {"weight_version": "9"}, "/model_info": {"weight_version": "1"}})
        ck(R._mc_current_weight_version(args) == "9", "METACLAW_WEIGHT_VERSION_URL overrides the router")
    finally:
        os.environ.pop("METACLAW_WEIGHT_VERSION_URL", None)
        R._mc_urlreq.urlopen = real

    # ---------------- 3. the train loop ----------------
    print("\n[train loop: collect after the update]")
    loop = read(os.path.join(sel_dir, "train_async_onpolicy.py"))
    compile(loop, "train_async_onpolicy", "exec")
    official_loop = read(os.path.join(OFFICIAL, "slime", "train_async.py"))
    call = "rollout_manager.generate.remote(rollout_id + 1)"
    ck(official_loop.count(call) == 1 and loop.count(call) == 1, "the next-batch start exists exactly once")
    i_update = loop.index("            actor_model.update_weights()\n")
    i_train = loop.index("actor_model.async_train(rollout_id, rollout_data_curr_ref)")
    i_call = loop.index(call)
    ck(i_train < i_update < i_call, "it now comes after training and after the weight update")
    ck(official_loop.index(call) < official_loop.index("actor_model.async_train(rollout_id, rollout_data_curr_ref)"),
       "(control) in the official loop it came before training")
    ck("assert args.update_weights_interval == 1" in loop, "guards the one-update-per-step assumption")
    ck(loop.count("rollout_manager.generate.remote(args.start_rollout_id)") == 1, "the first batch still starts before the loop")

    # ---------------- 4. the launcher ----------------
    print("\n[launcher: the switch reaches Ray; the loop swaps only when on]")
    profile = read(os.path.join(SCRIPTS_DIR, "run_openclaw_topk_select_modelfactory.sh"))
    m = re.search(r'^python3 - "\$\{PATCHED\}" "\$\{REPO_ROOT\}" <<\'PY\'\n(.*?)\nPY\n', profile, re.S | re.M)
    ck(m is not None, "found the launcher patch")
    work = tempfile.mkdtemp(prefix="mconrt")
    patch_py = os.path.join(work, "patch.py")
    open(patch_py, "w", encoding="utf-8", newline="\n").write(m.group(1))
    base_env = ("REPO_ROOT=/repo SCRIPT_DIR=/s SLIME_ROOT=/sl OPENCLAW_EVAL_MODE=1 "
                "OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron OPENCLAW_TOPK_W_RL=0 OPENCLAW_TOPK_W_OPD=1 "
                "OPENCLAW_TOPK_ADV_DIFF_CLIP=1 OPENCLAW_TOPK_MAX_CAND=3 TRAIN_EPOCHS=1 WANDB_API_KEY=k")
    for on in (False, True):
        launcher = os.path.join(work, f"launcher_{on}.sh")
        open(launcher, "w", encoding="utf-8", newline="\n").write(read(os.path.join(OFFICIAL, LAUNCHER_REL)))
        env = dict(os.environ, PYTHONUTF8="1")
        env.pop("METACLAW_ONPOLICY", None)
        if on:
            env["METACLAW_ONPOLICY"] = "1"
        subprocess.run([sys.executable, patch_py, launcher, "/repo"], check=True, capture_output=True, env=env)
        text = read(launcher)
        block = re.search(r'^RUNTIME_ENV_JSON="\{\n.*?\n\}"$', text, re.S | re.M).group(0)
        setting = "METACLAW_ONPOLICY=1" if on else ""
        script = f"set -a\n{base_env} {setting}\n{block}\nprintf '%s' \"$RUNTIME_ENV_JSON\"\n"
        ev = json.loads(subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True).stdout)["env_vars"]
        swapped = '"${PATCHED_COMBINE_SELECT_DIR}/train_async_onpolicy.py"' in text
        official = '"${SLIME_ROOT}/train_async.py"' in text
        ck(ev["METACLAW_ONPOLICY"] == ("1" if on else "0") and "METACLAW_WEIGHT_VERSION_URL" in ev
           and ev["OPENCLAW_TOPK_W_RL"] == "0" and ev["METACLAW_OPD_MASK_COMMIT"] == "0",
           f"valid JSON, METACLAW_ONPOLICY={'1' if on else '0'}, the other switches still carried")
        ck(swapped == on and official == (not on),
           f"entry point: {'patched loop' if on else 'official train_async.py'}")
    ck('if [ ! -f "${PATCHED_COMBINE_SELECT_DIR:-}/train_async_onpolicy.py" ]' in profile,
       "the profile refuses to start when the patched loop is missing")

    print(f"\nall {n} assertions passed" + (f"; skipped: {', '.join(skipped)}" if skipped else ""))


if __name__ == "__main__":
    main()
