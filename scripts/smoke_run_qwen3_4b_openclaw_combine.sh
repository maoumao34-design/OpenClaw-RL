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

sed \
    -e 's/--tensor-model-parallel-size 4/--tensor-model-parallel-size 1/' \
    -e 's/--rollout-num-gpus-per-engine 2/--rollout-num-gpus-per-engine 1/' \
    -e 's/--rollout-batch-size 16/--rollout-batch-size 4/' \
    -e 's/^export TP="2"/export TP="1"/' \
    "${OFFICIAL}" > "${PATCHED}"
chmod +x "${PATCHED}"

exec bash "${PATCHED}"
