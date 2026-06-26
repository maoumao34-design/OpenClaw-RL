#!/usr/bin/env bash
# =============================================================================
# SMOKE TEST ONLY — 不要用于 Table 3 正式训练
#
# 从 OpenClaw-RL-official 的官方 combine 脚本生成临时缩配版（3 GPU）：
#   Actor×1 (TP=1) + Rollout×1 + PRM×1，OPD teacher 走 inference（不占 Megatron 卡）
#
# 正式 8 GPU 论文布局请用：
#   openclaw-combine/run_qwen3_4b_openclaw_combine.sh
#   （由 scripts/train_with_services.sh 调用）
# =============================================================================

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/dfs/data/openclaw-rl-project/OpenClaw-RL-official}
OFFICIAL="${REPO_ROOT}/openclaw-combine/run_qwen3_4b_openclaw_combine.sh"
PATCHED="${SMOKE_COMBINE_SCRIPT:-${TMPDIR:-/tmp}/smoke_run_qwen3_4b_openclaw_combine.sh}"

if [ ! -f "${OFFICIAL}" ]; then
    echo "错误：找不到官方 combine 脚本: ${OFFICIAL}" >&2
    echo "请设置 REPO_ROOT 指向 OpenClaw-RL-official 克隆目录。" >&2
    exit 1
fi

export NUM_GPUS=${NUM_GPUS:-3}
export ACTOR_GPUS=${ACTOR_GPUS:-1}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
export OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=${OPENCLAW_COMBINE_OPD_TEACHER_SOURCE:-inference}
export PRM_GPUS=${PRM_GPUS:-1}
export PRM_NUM_GPUS_PER_ENGINE=${PRM_NUM_GPUS_PER_ENGINE:-1}
export PRM_TEACHER_GPUS=${PRM_TEACHER_GPUS:-0}
export USE_WANDB=${USE_WANDB:-0}

echo "=============================================="
echo "  SMOKE combine launcher (NOT production)"
echo "  NUM_GPUS=${NUM_GPUS}  ACTOR=${ACTOR_GPUS}  ROLLOUT=${ROLLOUT_GPUS}  PRM=${PRM_GPUS}"
echo "  OPD teacher: ${OPENCLAW_COMBINE_OPD_TEACHER_SOURCE}"
echo "  Patched script: ${PATCHED}"
echo "=============================================="

# Patched script is written outside openclaw-combine/ (e.g. logs/). The official script
# derives REPO_ROOT from SCRIPT_DIR/.. which breaks slime/Megatron paths — override below.
sed \
    -e 's/--tensor-model-parallel-size 4/--tensor-model-parallel-size 1/' \
    -e 's/--rollout-num-gpus-per-engine 2/--rollout-num-gpus-per-engine 1/' \
    -e 's/--rollout-batch-size 16/--rollout-batch-size 4/' \
    -e 's/--max-tokens-per-gpu 32768/--max-tokens-per-gpu 8192/' \
    -e 's/--rollout-max-context-len 32768/--rollout-max-context-len 8192/' \
    -e 's/--sglang-context-length 32768/--sglang-context-length 8192/' \
    -e 's/^export TP="2"/export TP="1"/' \
    -e 's/^export CONTEXT_LENGTH="32768"/export CONTEXT_LENGTH="8192"/' \
    "${OFFICIAL}" > "${PATCHED}"

# Official launcher kills all python on the node; that breaks modelfactory job runners.
# Smoke only needs a clean Ray session.
python3 - "${PATCHED}" "${REPO_ROOT}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
repo_root = sys.argv[2]
text = path.read_text()
text = text.replace(
    "pkill -9 sglang\nsleep 3\nray stop --force\npkill -9 ray\npkill -9 python\nsleep 3\npkill -9 ray\npkill -9 python",
    'echo "[smoke] skip aggressive pkill; stopping Ray only"\nray stop --force 2>/dev/null || true',
    1,
)
text = text.replace(
    'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"\nREPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"',
    f'SCRIPT_DIR="{repo_root}/openclaw-combine"\nREPO_ROOT="{repo_root}"',
    1,
)
path.write_text(text)
PY

chmod +x "${PATCHED}"

exec bash "${PATCHED}"
