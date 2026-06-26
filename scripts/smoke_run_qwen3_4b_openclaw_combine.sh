#!/usr/bin/env bash
# SMOKE TEST ONLY — 3 GPU 缩配 combine（modelfactory patch + smoke sed）
set -euo pipefail

SCRIPTS_DIR=$(dirname "$(realpath "$0")")
export SMOKE_PROFILE=1
export OPENCLAW_COMBINE_SCRIPT="${SMOKE_COMBINE_SCRIPT:-${TMPDIR:-/tmp}/smoke_run_qwen3_4b_openclaw_combine.sh}"

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
echo "  NUM_GPUS=${NUM_GPUS}  ACTOR=${ACTOR_GPUS}  ROLLOUT=${ROLLOUT_GPUS}  PRM=${PRM_GPUS} (TP=${PRM_NUM_GPUS_PER_ENGINE})"
echo "  OPD teacher: ${OPENCLAW_COMBINE_OPD_TEACHER_SOURCE}"
echo "  Patched script: ${OPENCLAW_COMBINE_SCRIPT}"
echo "=============================================="

exec bash "${SCRIPTS_DIR}/run_openclaw_combine_modelfactory.sh"
