#!/usr/bin/env bash
# Patch OpenClaw-RL-official combine launcher for modelfactory Ray jobs:
#   - Fix REPO_ROOT when patched script lives outside openclaw-combine/
#   - Ray job uses SLIME_ROOT/train_async.py (not /workspace/train_async.py)
#   - Skip aggressive pkill python
#
# SMOKE_PROFILE=1 applies 3-GPU smoke sed overrides (see smoke_run_qwen3_4b_openclaw_combine.sh).

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/dfs/data/openclaw-rl-project/OpenClaw-RL-official}
OFFICIAL="${REPO_ROOT}/openclaw-combine/run_qwen3_4b_openclaw_combine.sh"
PATCHED="${OPENCLAW_COMBINE_SCRIPT:-${TMPDIR:-/tmp}/run_qwen3_4b_openclaw_combine_modelfactory.sh}"
SMOKE_PROFILE=${SMOKE_PROFILE:-0}

if [ ! -f "${OFFICIAL}" ]; then
    echo "错误：找不到官方 combine 脚本: ${OFFICIAL}" >&2
    exit 1
fi

cp "${OFFICIAL}" "${PATCHED}"

if [ "${SMOKE_PROFILE}" = "1" ]; then
    sed -i \
        -e 's/--tensor-model-parallel-size 4/--tensor-model-parallel-size 1/' \
        -e 's/--rollout-num-gpus-per-engine 2/--rollout-num-gpus-per-engine 1/' \
        -e 's/--rollout-batch-size 16/--rollout-batch-size 4/' \
        -e 's/--max-tokens-per-gpu 32768/--max-tokens-per-gpu 8192/' \
        -e 's/--rollout-max-context-len 32768/--rollout-max-context-len 8192/' \
        -e 's/--sglang-context-length 32768/--sglang-context-length 8192/' \
        -e 's/^export TP="2"/export TP="1"/' \
        -e 's/^export CONTEXT_LENGTH="32768"/export CONTEXT_LENGTH="8192"/' \
        -e 's/PRM_GPUS=${PRM_GPUS:-2}/PRM_GPUS=${PRM_GPUS:-1}/' \
        -e 's/PRM_NUM_GPUS_PER_ENGINE=${PRM_NUM_GPUS_PER_ENGINE:-2}/PRM_NUM_GPUS_PER_ENGINE=${PRM_NUM_GPUS_PER_ENGINE:-1}/' \
        "${PATCHED}"
fi

python3 - "${PATCHED}" "${REPO_ROOT}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
repo_root = sys.argv[2]
text = path.read_text()
text = text.replace(
    "pkill -9 sglang\nsleep 3\nray stop --force\npkill -9 ray\npkill -9 python\nsleep 3\npkill -9 ray\npkill -9 python",
    'echo "[modelfactory] skip aggressive pkill; stopping Ray only"\nray stop --force 2>/dev/null || true',
    1,
)
text = text.replace(
    'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"\nREPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"',
    f'SCRIPT_DIR="{repo_root}/openclaw-combine"\nREPO_ROOT="{repo_root}"',
    1,
)
old_ray = (
    'ray job submit --address="http://127.0.0.1:8265" \\\n'
    '   --runtime-env-json="${RUNTIME_ENV_JSON}" \\\n'
    '   -- python3 train_async.py \\'
)
new_ray = (
    'ray job submit --address="http://127.0.0.1:8265" \\\n'
    '   --working-dir="${SLIME_ROOT}" \\\n'
    '   --runtime-env-json="${RUNTIME_ENV_JSON}" \\\n'
    '   -- python3 "${SLIME_ROOT}/train_async.py" \\'
)
if old_ray not in text:
    raise SystemExit("patch failed: ray job submit block not found in combine launcher")
text = text.replace(old_ray, new_ray, 1)
path.write_text(text)
PY

chmod +x "${PATCHED}"
exec bash "${PATCHED}"
