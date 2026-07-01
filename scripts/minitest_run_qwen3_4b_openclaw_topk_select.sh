#!/usr/bin/env bash
# 5-GPU pre-test launcher for Paper Table 3 Hybrid RL main method.
#
# 目的：在申请 8 GPU 正式 job 前，以 5 GPU（2+1+1+1）验证完整训练流水线。
#
# GPU 布局（论文接近配置，仅减少并行度）：
#   Actor×2  (TP=2)  + Rollout×1 + PRM SGLang×1 + PRM Teacher×1 = 5 GPU
#
# 与 8GPU 正式配置的唯一差异（不影响流程正确性）：
#   --tensor-model-parallel-size  4 → 2   (Actor TP，纯吞吐差异)
#   --rollout-num-gpus-per-engine 2 → 1   (SGLang rollout GPU)
#   export TP="2" → "1"                   (api server SGLang TP)
#   --num-rollout 100000000 → 300         (~18 步，仅验证流水线，不做完整训练)
#
# 其他配置与 8GPU 正式版完全一致：
#   context=32768, batch=16, m=3, k=4, sequence_optimal, optimizer args 等
#
# 用 minitest_train_with_services.sh 启动完整流水线（含 Simulator + 收敛检测）。
set -euo pipefail
SCRIPTS_DIR=$(dirname "$(realpath "$0")")
export MINITEST_PROFILE=1
export NUM_GPUS=${NUM_GPUS:-5}
export ACTOR_GPUS=${ACTOR_GPUS:-2}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
export PRM_GPUS=${PRM_GPUS:-1}
export PRM_NUM_GPUS_PER_ENGINE=${PRM_NUM_GPUS_PER_ENGINE:-1}
export PRM_TEACHER_GPUS=${PRM_TEACHER_GPUS:-1}
export USE_WANDB=${USE_WANDB:-0}
exec bash "${SCRIPTS_DIR}/run_openclaw_topk_select_modelfactory.sh"
