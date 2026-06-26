#!/bin/bash
# train_with_services.sh
#
# OpenClaw-RL Hybrid RL 训练编排（Table 3 Phase 1: Joint Hybrid RL）
# 基于 commit 0f4582c5 的 9 GPU 论文布局，Simulator 改为外部 API 服务。
#
# modelfactory job 提交：
#   代码解释器: bash
#   代码路径:   /dfs/data/openclaw-rl-project/openclaw-rl/scripts/train_with_services.sh
#   GPU 数量:   8
#
# GPU 分配（论文 faithful，8×H20 全用于训练）：
#   GPU 0-7  → Actor×4 + Rollout×2 + PRM×1 + PRM Teacher×1（megatron 默认）
#   Simulator → 外部机器（不在本 job 占 GPU）
#
# 提交 job 前必须 export（或在平台环境变量里配置）：
#   SIMULATOR_BASE_URL=http://<simulator-host>:30001/v1
#   SIMULATOR_API_KEY=<与 SGLang --api-key 一致；无 auth 则 EMPTY>
#
# 端口（训练机本机）：
#   30000  → RL training proxy
#   18789  → OpenClaw gateway

set -euo pipefail

# =====================================================================
# 配置
# =====================================================================
POLICY_MODEL_PATH=${POLICY_MODEL_PATH:-/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507}
POLICY_TORCH_DIST=${POLICY_TORCH_DIST:-/dfs/data/models/torch_dist/qwen3-4b-thinking-2507}

# 外部 Simulator（必填）
SIMULATOR_BASE_URL=${SIMULATOR_BASE_URL:-}
SIMULATOR_API_KEY=${SIMULATOR_API_KEY:-EMPTY}
EXTERNAL_MODEL=${EXTERNAL_MODEL:-qwen3-32b}

SGLANG_API_KEY=${SGLANG_API_KEY:-openclaw-rl-key}

# 训练占满 8 张 GPU（论文 4+2+1+1）
NUM_TRAINING_GPUS=${NUM_TRAINING_GPUS:-8}
TRAINING_CUDA_DEVICES=${TRAINING_CUDA_DEVICES:-$(seq -s, 0 $((NUM_TRAINING_GPUS - 1)))}

if [ -z "${SIMULATOR_BASE_URL}" ]; then
    echo "错误：必须设置 SIMULATOR_BASE_URL（外部 Simulator OpenAI 兼容地址）" >&2
    echo "示例：export SIMULATOR_BASE_URL=http://10.x.x.x:30001/v1" >&2
    exit 1
fi

# OPENCLAW_GATEWAY_TOKEN
if [ -z "${OPENCLAW_GATEWAY_TOKEN:-}" ]; then
    OPENCLAW_GATEWAY_TOKEN=$(python3 -c "
import json, pathlib, sys
cfg = pathlib.Path.home() / '.openclaw/openclaw.json'
if not cfg.exists(): sys.exit(1)
d = json.loads(cfg.read_text())
v = (d.get('gateway') or {}).get('auth', {}).get('token', '')
if v: print(v); sys.exit(0)
v = (d.get('gateway') or {}).get('token', '') or d.get('token', '')
if v: print(v); sys.exit(0)
sys.exit(1)
" 2>/dev/null) || true
fi

if [ -z "${OPENCLAW_GATEWAY_TOKEN:-}" ]; then
    echo "错误：无法读取 OPENCLAW_GATEWAY_TOKEN" >&2
    exit 1
fi

SAVE_CKPT=${SAVE_CKPT:-/dfs/data/openclaw-rl-project/checkpoints/qwen3-4b-openclaw-combine}
REPO_ROOT=${REPO_ROOT:-/dfs/data/openclaw-rl-project/OpenClaw-RL-official}
NUM_PROBLEMS_PER_ROUND=${NUM_PROBLEMS_PER_ROUND:-6}
DATASET=${DATASET:-${REPO_ROOT}/openclaw-test/GSM8K.json}
SESSION_LIMIT=${SESSION_LIMIT:-72}
CONDA_ENV=${CONDA_ENV:-/dfs/data/envs/openclaw-rl}
CONDA_BASE=${CONDA_BASE:-/dfs/data/miniconda3}

OPENCLAW_DIR=${REPO_ROOT}/openclaw-test
LOGS_DIR=${LOGS_DIR:-/dfs/data/openclaw-rl-project/logs/$(date +%Y%m%d_%H%M%S)}
WORKSPACE=${HOME}/.openclaw/workspace
SCRIPTS_DIR=$(dirname "$(realpath "$0")")

mkdir -p "${LOGS_DIR}"
echo "日志目录: ${LOGS_DIR}"
echo "外部 Simulator: ${SIMULATOR_BASE_URL} (model=${EXTERNAL_MODEL})"

# =====================================================================
# conda
# =====================================================================
if [ -n "${CONDA_ENV}" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    echo "已激活 conda: ${CONDA_ENV}"
fi

# =====================================================================
# 工具函数
# =====================================================================
dump_log_tail() {
    local logfile=$1
    local lines=${2:-80}
    if [ -f "${logfile}" ]; then
        echo "--- ${logfile} (last ${lines} lines) ---" >&2
        tail -n "${lines}" "${logfile}" >&2
        echo "--- end ---" >&2
    fi
}

wait_for_port() {
    local name=$1
    local port=$2
    local max_wait=${3:-600}
    local pid=${4:-}
    local logfile=${5:-}
    local waited=0
    echo "等待 ${name} (port ${port})..."
    while ! nc -z localhost "${port}" 2>/dev/null; do
        sleep 10
        waited=$((waited + 10))
        if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
            echo "错误：${name} 进程已退出" >&2
            dump_log_tail "${logfile}"
            return 1
        fi
        if [ ${waited} -ge ${max_wait} ]; then
            echo "超时：${name} 在 ${max_wait}s 内未启动" >&2
            dump_log_tail "${logfile}"
            return 1
        fi
        echo "  已等待 ${waited}s..."
    done
    echo "${name} 已就绪 (port ${port})"
}

wait_for_external_simulator() {
    local max_wait=${1:-300}
    local waited=0
    # SIMULATOR_BASE_URL=http://host:30001/v1 → health 在 http://host:30001/health
    local health_url="${SIMULATOR_BASE_URL%/v1}/health"
    echo "检查外部 Simulator: ${health_url}"
    while ! curl -sf "${health_url}" > /dev/null 2>&1; do
        sleep 10
        waited=$((waited + 10))
        if [ ${waited} -ge ${max_wait} ]; then
            echo "超时：外部 Simulator 在 ${max_wait}s 内不可达" >&2
            echo "请确认已在另一台机器启动 scripts/launch_simulator.sh，且训练机网络可达" >&2
            return 1
        fi
        echo "  已等待 ${waited}s..."
    done
    echo "外部 Simulator 已就绪 (${health_url})"
}

# =====================================================================
# 清理
# =====================================================================
OPENCLAW_PID=""
SIM_LOOP_PID=""

cleanup() {
    echo ""
    echo "清理中..."
    [ -n "${SIM_LOOP_PID}" ] && kill "${SIM_LOOP_PID}" 2>/dev/null || true
    [ -n "${OPENCLAW_PID}" ]  && kill "${OPENCLAW_PID}"  2>/dev/null || true
    ray stop --force 2>/dev/null || true
    wait 2>/dev/null || true
    echo "清理完成。"
}
trap cleanup EXIT INT TERM

# =====================================================================
# 第1步：启动训练（GPU 0-7 全部用于论文 megatron 布局）
# =====================================================================
echo ""
echo "=== [1/3] 启动训练（GPU ${TRAINING_CUDA_DEVICES}，NUM_GPUS=${NUM_TRAINING_GPUS}）==="

CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_DEVICES}" \
  NUM_GPUS="${NUM_TRAINING_GPUS}" \
  HF_CKPT="${POLICY_MODEL_PATH}" \
  REF_LOAD="${POLICY_TORCH_DIST}" \
  SAVE_CKPT="${SAVE_CKPT}" \
  PRM_MODEL_PATH="${POLICY_MODEL_PATH}" \
  PRM_TEACHER_LOAD="${POLICY_TORCH_DIST}" \
  SGLANG_API_KEY="${SGLANG_API_KEY}" \
  bash "${REPO_ROOT}/openclaw-combine/run_qwen3_4b_openclaw_combine.sh" \
  > "${LOGS_DIR}/training.log" 2>&1 &
TRAINING_PID=$!

echo "训练 PID: ${TRAINING_PID}，等待 Ray head (port 8265)..."
until curl -sf http://127.0.0.1:8265/api/version > /dev/null 2>&1; do
    sleep 5
    if ! kill -0 "${TRAINING_PID}" 2>/dev/null; then
        echo "错误：训练进程意外退出" >&2
        dump_log_tail "${LOGS_DIR}/training.log"
        exit 1
    fi
done
echo "Ray head 已就绪"

# =====================================================================
# 第2步：确认外部 Simulator + 启动 OpenClaw gateway
# =====================================================================
echo ""
echo "=== [2/3] 外部 Simulator 连通性 + OpenClaw gateway ==="

wait_for_external_simulator 300

echo ""
echo "启动 OpenClaw gateway（port 18789）..."
OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
  openclaw gateway run --allow-unconfigured --force \
  > "${LOGS_DIR}/openclaw.log" 2>&1 &
OPENCLAW_PID=$!

wait_for_port "OpenClaw gateway"  18789 300 "${OPENCLAW_PID}" "${LOGS_DIR}/openclaw.log"
wait_for_port "RL training proxy" 30000 900 "" "${LOGS_DIR}/training.log"

# =====================================================================
# 第3步：模拟循环
# =====================================================================
echo ""
echo "=== [3/3] 模拟循环（Student → TA → Teacher）==="

STUDENT_ALL="${LOGS_DIR}/results_student_all.txt"
TA_ALL="${LOGS_DIR}/results_TA_all.txt"
TEACHER_ALL="${LOGS_DIR}/results_teacher_all.txt"

run_simulation_round() {
    local round=$1
    echo "--- 模拟 round ${round} 开始 ---"

    local round_student="${LOGS_DIR}/results_student_round${round}.txt"
    local round_ta="${LOGS_DIR}/results_TA_round${round}.txt"
    local round_teacher="${LOGS_DIR}/results_teacher_round${round}.txt"

    rm -rf "${WORKSPACE}/homework" "${WORKSPACE}/homework1" "${WORKSPACE}/homework2"

    OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
    OPENAI_API_KEY="${SIMULATOR_API_KEY}" \
    OPENAI_BASE_URL="${SIMULATOR_BASE_URL}" \
    EXTERNAL_MODEL="${EXTERNAL_MODEL}" \
    OPENCLAW_GATEWAY_URL=http://localhost:18789 \
    python "${OPENCLAW_DIR}/student_chat.py" \
        --dataset "${DATASET}" \
        --num-problems "${NUM_PROBLEMS_PER_ROUND}" \
        --output "${round_student}"
    cat "${round_student}" >> "${STUDENT_ALL}"

    OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
    OPENAI_API_KEY="${SIMULATOR_API_KEY}" \
    OPENAI_BASE_URL="${SIMULATOR_BASE_URL}" \
    EXTERNAL_MODEL="${EXTERNAL_MODEL}" \
    OPENCLAW_GATEWAY_URL=http://localhost:18789 \
    python "${OPENCLAW_DIR}/TA_chat.py" \
        --dataset "${DATASET}" \
        --num-problems "${NUM_PROBLEMS_PER_ROUND}" \
        --output "${round_ta}"
    cat "${round_ta}" >> "${TA_ALL}"

    OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
    OPENAI_API_KEY="${SIMULATOR_API_KEY}" \
    OPENAI_BASE_URL="${SIMULATOR_BASE_URL}" \
    EXTERNAL_MODEL="${EXTERNAL_MODEL}" \
    OPENCLAW_GATEWAY_URL=http://localhost:18789 \
    python "${OPENCLAW_DIR}/teacher_chat.py" \
        --dataset "${DATASET}" \
        --num-problems "${NUM_PROBLEMS_PER_ROUND}" \
        --output "${round_teacher}"
    cat "${round_teacher}" >> "${TEACHER_ALL}"

    echo "--- 模拟 round ${round} 完成 ---"
}

simulation_loop() {
    local round=0
    local total_sessions=0
    local max_rounds=$(( (SESSION_LIMIT + NUM_PROBLEMS_PER_ROUND - 1) / NUM_PROBLEMS_PER_ROUND ))

    while kill -0 "${TRAINING_PID}" 2>/dev/null && [ ${round} -lt ${max_rounds} ]; do
        round=$((round + 1))
        total_sessions=$((round * NUM_PROBLEMS_PER_ROUND))
        echo "模拟 round ${round}/${max_rounds}（累计 session：${total_sessions}/${SESSION_LIMIT}）"
        run_simulation_round ${round} \
            2>&1 | tee -a "${LOGS_DIR}/simulation.log" || true
    done

    echo "模拟循环结束（共 ${round} 轮）"
    echo ""
    echo "=== 收敛检测（Table 3 指标）==="
    python "${SCRIPTS_DIR}/check_convergence.py" \
        --student "${STUDENT_ALL}" \
        --ta      "${TA_ALL}" \
        --teacher "${TEACHER_ALL}" \
        2>&1 | tee "${LOGS_DIR}/convergence_result.txt"
}

simulation_loop &
SIM_LOOP_PID=$!

echo ""
echo "所有服务已启动，训练进行中..."
echo "  日志目录:       ${LOGS_DIR}/"
echo "  外部 Simulator: ${SIMULATOR_BASE_URL}"
echo "  训练日志:       tail -f ${LOGS_DIR}/training.log"
echo "  OpenClaw 日志:  tail -f ${LOGS_DIR}/openclaw.log"
echo "  Ray dashboard:  http://127.0.0.1:8265"

wait "${TRAINING_PID}"
echo "训练完成！检查点: ${SAVE_CKPT}"
