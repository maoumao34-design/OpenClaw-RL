#!/bin/bash
# train_with_services.sh
#
# 完整的 OpenClaw-RL Hybrid RL 训练启动脚本。
#
# GPU 分配（默认 9×H20）：
#   GPU 0-7  → 训练 (Ray + Megatron + SGLang)
#   GPU 8    → Qwen3-32B Simulator (SGLang, port 30001)
#
# 端口说明：
#   30000  → RL training proxy（训练进程自动启动）
#   30001  → Simulator / External LLM
#   18789  → OpenClaw gateway
#
# 使用前设置（必填）：
#   export POLICY_MODEL_PATH=/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507
#   export SIMULATOR_MODEL_PATH=/dfs/data/models/Qwen/Qwen3-32B
#   export OPENCLAW_GATEWAY_TOKEN=<token>   # 从 openclaw.json 读取
#
# 获取 OPENCLAW_GATEWAY_TOKEN：
#   cat ~/.openclaw/openclaw.json | grep -i token
#
# 可选覆盖：
#   SAVE_CKPT, REPO_ROOT, NUM_TRAINING_GPUS, SIMULATOR_GPU
#   NUM_PROBLEMS_PER_ROUND, CONDA_ENV, WANDB_API_KEY

set -euo pipefail

# =====================================================================
# 配置
# =====================================================================
POLICY_MODEL_PATH=${POLICY_MODEL_PATH:?"必须设置 POLICY_MODEL_PATH"}
SIMULATOR_MODEL_PATH=${SIMULATOR_MODEL_PATH:?"必须设置 SIMULATOR_MODEL_PATH"}
OPENCLAW_GATEWAY_TOKEN=${OPENCLAW_GATEWAY_TOKEN:?"必须设置 OPENCLAW_GATEWAY_TOKEN"}

SAVE_CKPT=${SAVE_CKPT:-/dfs/data/openclaw-rl-project/checkpoints/qwen3-4b-openclaw-combine}
REPO_ROOT=${REPO_ROOT:-/dfs/data/openclaw-rl-project/OpenClaw-RL-official}
NUM_TRAINING_GPUS=${NUM_TRAINING_GPUS:-8}
SIMULATOR_GPU=${SIMULATOR_GPU:-8}
NUM_PROBLEMS_PER_ROUND=${NUM_PROBLEMS_PER_ROUND:-6}
DATASET=${DATASET:-${REPO_ROOT}/openclaw-test/GSM8K.json}
CONDA_ENV=${CONDA_ENV:-}

OPENCLAW_DIR=${REPO_ROOT}/openclaw-test
LOGS_DIR=${LOGS_DIR:-/dfs/data/openclaw-rl-project/logs/$(date +%Y%m%d_%H%M%S)}
WORKSPACE=${HOME}/.openclaw/workspace

mkdir -p "${LOGS_DIR}"
echo "日志目录: ${LOGS_DIR}"

# =====================================================================
# conda 环境
# =====================================================================
if [ -n "${CONDA_ENV}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    echo "已激活 conda: ${CONDA_ENV}"
fi

# =====================================================================
# 工具函数
# =====================================================================
wait_for_port() {
    local name=$1
    local port=$2
    local max_wait=${3:-600}
    local waited=0
    echo "等待 ${name} (port ${port})..."
    while ! nc -z localhost "${port}" 2>/dev/null; do
        sleep 10
        waited=$((waited + 10))
        if [ ${waited} -ge ${max_wait} ]; then
            echo "超时：${name} 在 ${max_wait}s 内未启动" >&2
            return 1
        fi
        echo "  已等待 ${waited}s..."
    done
    echo "${name} 已就绪 (port ${port})"
}

# =====================================================================
# 退出清理
# =====================================================================
SIMULATOR_PID=""
OPENCLAW_PID=""
SIM_LOOP_PID=""

cleanup() {
    echo ""
    echo "清理中..."
    [ -n "${SIM_LOOP_PID}" ] && kill "${SIM_LOOP_PID}" 2>/dev/null || true
    [ -n "${OPENCLAW_PID}" ]  && kill "${OPENCLAW_PID}"  2>/dev/null || true
    [ -n "${SIMULATOR_PID}" ] && kill "${SIMULATOR_PID}" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    wait 2>/dev/null || true
    echo "清理完成。"
}
trap cleanup EXIT INT TERM

# =====================================================================
# 第1步：启动训练
# 训练脚本开头会 pkill sglang/python/ray，必须先启动训练，
# 等 Ray head 就绪后再启动 Simulator，避免被 pkill 杀掉。
# =====================================================================
echo ""
echo "=== [1/4] 启动训练（GPU 0-$((NUM_TRAINING_GPUS-1))，通过 Ray）==="

# 限制训练只使用 GPU 0..N-1，留出 GPU ${SIMULATOR_GPU} 给 Simulator
TRAINING_CUDA_DEVICES=$(seq -s, 0 $((NUM_TRAINING_GPUS - 1)))

CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_DEVICES}" \
  NUM_GPUS="${NUM_TRAINING_GPUS}" \
  HF_CKPT="${POLICY_MODEL_PATH}" \
  REF_LOAD="${POLICY_MODEL_PATH}" \
  SAVE_CKPT="${SAVE_CKPT}" \
  PRM_MODEL_PATH="${POLICY_MODEL_PATH}" \
  PRM_TEACHER_LOAD="${POLICY_MODEL_PATH}" \
  bash "${REPO_ROOT}/openclaw-combine/run_qwen3_4b_openclaw_combine.sh" \
  > "${LOGS_DIR}/training.log" 2>&1 &
TRAINING_PID=$!

echo "训练 PID: ${TRAINING_PID}，等待 Ray head (port 8265)..."
until curl -sf http://127.0.0.1:8265/api/version > /dev/null 2>&1; do
    sleep 5
    # 检查训练进程是否已意外退出
    if ! kill -0 "${TRAINING_PID}" 2>/dev/null; then
        echo "错误：训练进程意外退出，查看 ${LOGS_DIR}/training.log" >&2
        exit 1
    fi
done
echo "Ray head 已就绪（pkill 阶段已结束）"

# =====================================================================
# 第2步：启动 Qwen3-32B Simulator（GPU ${SIMULATOR_GPU}，port 30001）
# =====================================================================
echo ""
echo "=== [2/4] 启动 Qwen3-32B Simulator（GPU ${SIMULATOR_GPU}，port 30001）==="

CUDA_VISIBLE_DEVICES="${SIMULATOR_GPU}" \
  MODEL_PATH="${SIMULATOR_MODEL_PATH}" \
  PORT=30001 \
  TP_SIZE=1 \
  MODEL_NAME=qwen3-32b \
  bash "${OPENCLAW_DIR}/launch_user_llm.sh" \
  > "${LOGS_DIR}/simulator.log" 2>&1 &
SIMULATOR_PID=$!

# =====================================================================
# 第3步：启动 OpenClaw gateway（port 18789，指向 port 30000）
# =====================================================================
echo ""
echo "=== [3/4] 启动 OpenClaw gateway（port 18789）==="

# OPENAI_BASE_URL 指向 RL training proxy (port 30000)，不是 OpenAI
OPENAI_BASE_URL=http://localhost:30000/v1 \
  OPENAI_API_KEY=EMPTY \
  openclaw start \
  > "${LOGS_DIR}/openclaw.log" 2>&1 &
OPENCLAW_PID=$!

# 等待 Simulator 和 OpenClaw 就绪
wait_for_port "Simulator"         30001 600
wait_for_port "OpenClaw gateway"  18789 120

# port 30000 由训练进程在第一次 rollout 时自动启动，等待它就绪
wait_for_port "RL training proxy" 30000 600

# =====================================================================
# 第4步：模拟循环（持续为训练提供 rollout 数据）
# =====================================================================
echo ""
echo "=== [4/4] 启动模拟循环（Student → TA → Teacher）==="

run_simulation_round() {
    local round=$1
    echo "--- 模拟 round ${round} 开始 ---"

    # 每轮清理 workspace，避免上轮文件干扰
    rm -rf "${WORKSPACE}/homework" "${WORKSPACE}/homework1" "${WORKSPACE}/homework2"

    # Student：让 OpenClaw 解 GSM8K 作业（要求非 AI 风格输出）
    OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
    OPENAI_API_KEY=EMPTY \
    OPENAI_BASE_URL=http://localhost:30001/v1 \
    EXTERNAL_MODEL=qwen3-32b \
    OPENCLAW_GATEWAY_URL=http://localhost:18789 \
    python "${OPENCLAW_DIR}/student_chat.py" \
        --dataset "${DATASET}" \
        --num-problems "${NUM_PROBLEMS_PER_ROUND}"

    # TA：让 OpenClaw 批改作业（要求详细批改意见）
    OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
    OPENAI_API_KEY=EMPTY \
    OPENAI_BASE_URL=http://localhost:30001/v1 \
    EXTERNAL_MODEL=qwen3-32b \
    OPENCLAW_GATEWAY_URL=http://localhost:18789 \
    python "${OPENCLAW_DIR}/TA_chat.py" \
        --dataset "${DATASET}" \
        --num-problems "${NUM_PROBLEMS_PER_ROUND}"

    # Teacher：让 OpenClaw 写温暖的教师评语（要求包含 well done / excellent）
    OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
    OPENAI_API_KEY=EMPTY \
    OPENAI_BASE_URL=http://localhost:30001/v1 \
    EXTERNAL_MODEL=qwen3-32b \
    OPENCLAW_GATEWAY_URL=http://localhost:18789 \
    python "${OPENCLAW_DIR}/teacher_chat.py" \
        --dataset "${DATASET}" \
        --num-problems "${NUM_PROBLEMS_PER_ROUND}"

    echo "--- 模拟 round ${round} 完成（${NUM_PROBLEMS_PER_ROUND} 个问题 × 3 persona = $((NUM_PROBLEMS_PER_ROUND * 3)) 个 session）---"
}

simulation_loop() {
    local round=0
    while kill -0 "${TRAINING_PID}" 2>/dev/null; do
        round=$((round + 1))
        run_simulation_round ${round} \
            2>&1 | tee -a "${LOGS_DIR}/simulation.log" || true
    done
    echo "训练进程已结束，模拟循环退出。"
}

simulation_loop &
SIM_LOOP_PID=$!

# =====================================================================
# 状态输出
# =====================================================================
echo ""
echo "所有服务已启动，训练进行中..."
echo "  日志目录:        ${LOGS_DIR}/"
echo "  训练日志:        tail -f ${LOGS_DIR}/training.log"
echo "  Simulator 日志:  tail -f ${LOGS_DIR}/simulator.log"
echo "  OpenClaw 日志:   tail -f ${LOGS_DIR}/openclaw.log"
echo "  模拟循环日志:    tail -f ${LOGS_DIR}/simulation.log"
echo "  Ray dashboard:   http://127.0.0.1:8265"
echo ""
echo "按 Ctrl+C 停止所有服务。"

wait "${TRAINING_PID}"
echo "训练完成！检查点: ${SAVE_CKPT}"
