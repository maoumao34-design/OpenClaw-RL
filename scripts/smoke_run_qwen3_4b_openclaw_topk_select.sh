#!/usr/bin/env bash
# SMOKE TEST ONLY — 4 GPU 缩配 topk-select（modelfactory patch + smoke sed）
#
# topk-select 强制 OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron，
# 必须为 PRM Teacher 分配独立 GPU，最少需要 4 张：
#   Actor×1(TP=1) + Rollout×1 + PRM SGLang×1 + PRM Teacher×1
set -euo pipefail

SCRIPTS_DIR=$(dirname "$(realpath "$0")")
export SMOKE_PROFILE=1
export OPENCLAW_TOPK_SELECT_SCRIPT="${SMOKE_TOPK_SELECT_SCRIPT:-${TMPDIR:-/tmp}/smoke_run_qwen3_4b_openclaw_topk_select.sh}"

export NUM_GPUS=${NUM_GPUS:-4}
export ACTOR_GPUS=${ACTOR_GPUS:-1}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
export PRM_GPUS=${PRM_GPUS:-1}
export PRM_NUM_GPUS_PER_ENGINE=${PRM_NUM_GPUS_PER_ENGINE:-1}
export PRM_TEACHER_GPUS=${PRM_TEACHER_GPUS:-1}
export USE_WANDB=${USE_WANDB:-0}

echo "=============================================="
echo "  SMOKE topk-select launcher (NOT production)"
echo "  NUM_GPUS=${NUM_GPUS}  ACTOR=${ACTOR_GPUS}  ROLLOUT=${ROLLOUT_GPUS}  PRM=${PRM_GPUS}  TEACHER=${PRM_TEACHER_GPUS}"
echo "  OPD teacher: megatron (topk-select forced)"
echo "  k=4 m=1(smoke) seq-optimal"
echo "  Patched script: ${OPENCLAW_TOPK_SELECT_SCRIPT}"
echo "=============================================="

exec bash "${SCRIPTS_DIR}/run_openclaw_topk_select_modelfactory.sh"
