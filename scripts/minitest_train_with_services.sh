#!/bin/bash
# =============================================================================
# PRE-TEST（8GPU 正式训练前置验证）— 5 GPU Hybrid RL 完整流水线
#
# 目的：在申请 8 GPU 提交正式 Table 3 论文训练之前，用 5 GPU 跑通完整训练
#       流水线，确认 actor、rollout、PRM、Simulator 端到端均无问题。
#
# ⚠️  仅用于流水线验证，num-rollout=300（~18 步），不做完整训练，结果不计入论文。
# ⚠️  正式训练请用 scripts/train_with_services.sh（8 GPU）。
#
# modelfactory job 提交：
#   代码解释器: bash
#   代码路径:   .../openclaw-rl/scripts/minitest_train_with_services.sh
#   GPU 数量:   5
#
# GPU 分配（5×H20）：
#   Actor×2 (TP=2) + Rollout×1 + PRM SGLang×1 + PRM Teacher×1
#
# 与 8GPU 正式配置的差异（minitest_run_qwen3_4b_openclaw_topk_select.sh 通过
# MINITEST_PROFILE=1 自动 sed 打补丁，无需手动修改）：
#   tensor-model-parallel-size  4 → 2
#   rollout-num-gpus-per-engine 2 → 1
#   export TP="2" → "1"
#   num-rollout 100000000 → 300
#
# Simulator 地址：编辑 scripts/simulator.env（见 simulator.env.example）
# =============================================================================

set -euo pipefail

SCRIPTS_DIR=$(dirname "$(realpath "$0")")
SIMULATOR_ENV="${SCRIPTS_DIR}/simulator.env"
if [ ! -f "${SIMULATOR_ENV}" ]; then
    if [ -f "${SCRIPTS_DIR}/simulator.env.example" ]; then
        cp "${SCRIPTS_DIR}/simulator.env.example" "${SIMULATOR_ENV}"
        echo "已创建 ${SIMULATOR_ENV} — 请填写 Simulator 地址和 key 后重新提交 job"
    fi
    exit 1
fi
set -a
# shellcheck disable=SC1091
source "${SIMULATOR_ENV}"
set +a
echo "已加载: ${SIMULATOR_ENV}"

# =====================================================================
# 配置
# =====================================================================
POLICY_MODEL_PATH=${POLICY_MODEL_PATH:-/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507}
POLICY_TORCH_DIST=${POLICY_TORCH_DIST:-/dfs/data/models/Qwen3-4B-Thinking-2507-torch-dist}

SIMULATOR_BASE_URL=${SIMULATOR_BASE_URL:-}
SIMULATOR_API_KEY=${SIMULATOR_API_KEY:-EMPTY}
EXTERNAL_MODEL=${EXTERNAL_MODEL:-qwen3-32b}

SGLANG_API_KEY=${SGLANG_API_KEY:-openclaw-rl-key}

# 5 GPU pre-test（论文 8GPU 正式用 train_with_services.sh）
NUM_TRAINING_GPUS=${NUM_TRAINING_GPUS:-5}
TRAINING_CUDA_DEVICES=${TRAINING_CUDA_DEVICES:-$(seq -s, 0 $((NUM_TRAINING_GPUS - 1)))}

if [ -z "${SIMULATOR_BASE_URL}" ] || [[ "${SIMULATOR_BASE_URL}" == *"<"* ]]; then
    echo "错误：请在 ${SIMULATOR_ENV} 中填写 SIMULATOR_BASE_URL" >&2
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

SAVE_CKPT=${SAVE_CKPT:-/dfs/data/openclaw-rl-project/checkpoints/minitest-qwen3-4b-openclaw-topk-select}
REPO_ROOT=${REPO_ROOT:-/dfs/data/openclaw-rl-project/OpenClaw-RL-official}
NUM_PROBLEMS_PER_ROUND=${NUM_PROBLEMS_PER_ROUND:-6}
DATASET=${DATASET:-${REPO_ROOT}/openclaw-test/GSM8K.json}
SESSION_LIMIT=${SESSION_LIMIT:-72}
CONDA_ENV=${CONDA_ENV:-/dfs/data/envs/openclaw-rl}
CONDA_BASE=${CONDA_BASE:-/dfs/data/miniconda3}

OPENCLAW_DIR=${REPO_ROOT}/openclaw-test
LOGS_DIR=${LOGS_DIR:-/dfs/data/openclaw-rl-project/logs/minitest_$(date +%Y%m%d_%H%M%S)}
WORKSPACE=${HOME}/.openclaw/workspace
MINITEST_TOPK_SELECT_LAUNCHER="${SCRIPTS_DIR}/minitest_run_qwen3_4b_openclaw_topk_select.sh"

if [ ! -f "${MINITEST_TOPK_SELECT_LAUNCHER}" ]; then
    echo "错误：找不到 ${MINITEST_TOPK_SELECT_LAUNCHER}" >&2
    exit 1
fi

mkdir -p "${LOGS_DIR}"
echo ""
echo "============================================================"
echo "  OpenClaw-RL PRE-TEST (5 GPU) — 8GPU 正式训练前置验证"
echo "============================================================"
echo "日志目录:       ${LOGS_DIR}"
echo "外部 Simulator: ${SIMULATOR_BASE_URL} (model=${EXTERNAL_MODEL})"
echo "训练 GPU:       ${TRAINING_CUDA_DEVICES} (NUM_GPUS=${NUM_TRAINING_GPUS})"
echo "num-rollout:    300 (~18 步，验证流水线；正式 8GPU 用 train_with_services.sh)"
echo ""

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
        local total
        total=$(wc -l < "${logfile}" | tr -d ' ')
        echo "--- ${logfile} (${total} lines total; showing last ${lines}) ---" >&2
        if [ "${total}" -le 200 ]; then
            cat "${logfile}" >&2
        else
            tail -n "${lines}" "${logfile}" >&2
        fi
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
    while ! curl -s --max-time 5 "http://localhost:${port}/" > /dev/null 2>&1; do
        sleep 10
        waited=$((waited + 10))
        if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
            echo "错误：${name} 进程已退出" >&2
            dump_log_tail "${logfile}"
            return 1
        fi
        if [ -n "${logfile}" ] && [ -f "${logfile}" ]; then
            if grep -qE "Job 'raysubmit_.*' failed|can't open file '/workspace/train_async.py'|patch failed|Traceback|CUDA out of memory" "${logfile}" 2>/dev/null; then
                echo "错误：Ray 训练 job 已失败（见 ${logfile}）" >&2
                dump_log_tail "${logfile}" 120
                return 1
            fi
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
# 第1步：启动训练（MINITEST_PROFILE=1，5 GPU）
# =====================================================================
echo ""
echo "=== [1/3] 启动 PRE-TEST 训练（GPU ${TRAINING_CUDA_DEVICES}，NUM_GPUS=${NUM_TRAINING_GPUS}）==="

export MINITEST_TOPK_SELECT_SCRIPT="${LOGS_DIR}/.minitest_run_qwen3_4b_openclaw_topk_select.sh"

CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_DEVICES}" \
  REPO_ROOT="${REPO_ROOT}" \
  NUM_GPUS="${NUM_TRAINING_GPUS}" \
  HF_CKPT="${POLICY_MODEL_PATH}" \
  REF_LOAD="${POLICY_TORCH_DIST}" \
  SAVE_CKPT="${SAVE_CKPT}" \
  PRM_MODEL_PATH="${POLICY_MODEL_PATH}" \
  PRM_TEACHER_LOAD="${POLICY_TORCH_DIST}" \
  SGLANG_API_KEY="${SGLANG_API_KEY}" \
  USE_WANDB=0 \
  OPENCLAW_TOPK_SELECT_SCRIPT="${MINITEST_TOPK_SELECT_SCRIPT}" \
  bash "${MINITEST_TOPK_SELECT_LAUNCHER}" \
  > "${LOGS_DIR}/training.log" 2>&1 &
TRAINING_PID=$!

echo "训练 PID: ${TRAINING_PID}，等待 Ray head (port 8265)..."
until curl -sf http://127.0.0.1:8265/api/version > /dev/null 2>&1; do
    sleep 5
    if ! kill -0 "${TRAINING_PID}" 2>/dev/null; then
        echo "错误：PRE-TEST 训练进程意外退出" >&2
        dump_log_tail "${LOGS_DIR}/training.log" 120
        exit 1
    fi
done
echo "Ray head 已就绪"

# =====================================================================
# 第2步：确认外部 Simulator + 启动 OpenClaw gateway
# =====================================================================
echo ""
echo "=== [2/3] 外部 Simulator + OpenClaw gateway + RL proxy :30000 ==="

wait_for_external_simulator 300

echo "等待 RL training proxy (port 30000)..."
wait_for_port "RL training proxy" 30000 900 "" "${LOGS_DIR}/training.log"

echo "启动 OpenClaw gateway（port 18789）..."
OPENCLAW_SKIP_BROWSER_CONTROL_SERVER=1 \
OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
openclaw gateway run --allow-unconfigured --force \
  > "${LOGS_DIR}/openclaw.log" 2>&1 &
OPENCLAW_PID=$!

wait_for_port "OpenClaw gateway" 18789 900 "${OPENCLAW_PID}" "${LOGS_DIR}/openclaw.log"

# =====================================================================
# 第3步：模拟循环（与正式训练相同的 Joint 结构，训练完 300 rollout 后自然结束）
# =====================================================================
echo ""
echo "=== [3/3] 模拟循环（INIT 顺序 → Joint 三角色并行，训练结束后自动停止）==="

STUDENT_ALL="${LOGS_DIR}/results_student_all.txt"
TA_ALL="${LOGS_DIR}/results_TA_all.txt"
TEACHER_ALL="${LOGS_DIR}/results_teacher_all.txt"

run_one_persona() {
    local name=$1 script=$2 num=$3 output=$4
    OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
    OPENAI_API_KEY="${SIMULATOR_API_KEY}" \
    OPENAI_BASE_URL="${SIMULATOR_BASE_URL}" \
    EXTERNAL_MODEL="${EXTERNAL_MODEL}" \
    OPENCLAW_GATEWAY_URL=http://localhost:18789 \
    python "${OPENCLAW_DIR}/${script}" \
        --dataset "${DATASET}" \
        --num-problems "${num}" \
        --output "${output}" \
    || echo "警告：${name} 模拟未完全完成，继续训练"
}

run_init_phase() {
    echo ""
    echo "=== INIT：建立 homework1/ homework2/（SESSION_LIMIT=${SESSION_LIMIT} 题）==="

    rm -rf "${WORKSPACE}/homework" "${WORKSPACE}/homework1" "${WORKSPACE}/homework2"

    local init_s="${LOGS_DIR}/results_student_init.txt"
    local init_ta="${LOGS_DIR}/results_TA_init.txt"
    local init_t="${LOGS_DIR}/results_teacher_init.txt"

    echo "  [INIT 1/3] Student → homework/..."
    run_one_persona "Student" "student_chat.py" "${SESSION_LIMIT}" "${init_s}"
    cat "${init_s}" >> "${STUDENT_ALL}"

    echo "  [INIT 2/3] TA（ensure_homework_dir: homework→homework1/）..."
    run_one_persona "TA" "TA_chat.py" "${SESSION_LIMIT}" "${init_ta}"
    cat "${init_ta}" >> "${TA_ALL}"

    echo "  [INIT 3/3] Teacher（ensure_homework_dir: homework1→homework2/）..."
    run_one_persona "Teacher" "teacher_chat.py" "${SESSION_LIMIT}" "${init_t}"
    cat "${init_t}" >> "${TEACHER_ALL}"

    echo "=== INIT 完成：homework1/ homework2/ 已固定，进入 Joint 阶段 ==="
}

run_joint_round() {
    local round=$1
    echo "--- Joint round ${round} 开始（${NUM_PROBLEMS_PER_ROUND} 题 × 三角色并行）---"

    local round_s="${LOGS_DIR}/results_student_round${round}.txt"
    local round_ta="${LOGS_DIR}/results_TA_round${round}.txt"
    local round_t="${LOGS_DIR}/results_teacher_round${round}.txt"

    run_one_persona "Student" "student_chat.py" "${NUM_PROBLEMS_PER_ROUND}" "${round_s}" \
        >> "${LOGS_DIR}/sim_student.log" 2>&1 &
    local pid_s=$!

    run_one_persona "TA" "TA_chat.py" "${NUM_PROBLEMS_PER_ROUND}" "${round_ta}" \
        >> "${LOGS_DIR}/sim_ta.log" 2>&1 &
    local pid_ta=$!

    run_one_persona "Teacher" "teacher_chat.py" "${NUM_PROBLEMS_PER_ROUND}" "${round_t}" \
        >> "${LOGS_DIR}/sim_teacher.log" 2>&1 &
    local pid_t=$!

    wait "${pid_s}" "${pid_ta}" "${pid_t}"

    cat "${round_s}"  >> "${STUDENT_ALL}"
    cat "${round_ta}" >> "${TA_ALL}"
    cat "${round_t}"  >> "${TEACHER_ALL}"

    echo "--- Joint round ${round} 完成 ---"
}

simulation_loop() {
    run_init_phase 2>&1 | tee -a "${LOGS_DIR}/simulation.log"

    local round=0

    while kill -0 "${TRAINING_PID}" 2>/dev/null; do
        round=$((round + 1))
        echo "Joint round ${round}"
        run_joint_round ${round} \
            2>&1 | tee -a "${LOGS_DIR}/simulation.log" || true
    done

    echo "模拟循环结束（INIT 1 次 + Joint ${round} 轮）"
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
echo "所有服务已启动，PRE-TEST 训练进行中（300 rollout 后自动结束）..."
echo "  日志目录:       ${LOGS_DIR}/"
echo "  外部 Simulator: ${SIMULATOR_BASE_URL}"
echo "  训练日志:       tail -f ${LOGS_DIR}/training.log"
echo "  OpenClaw 日志:  tail -f ${LOGS_DIR}/openclaw.log"
echo "  Ray dashboard:  http://127.0.0.1:8265"
echo ""
echo "PRE-TEST 通过后，用 scripts/train_with_services.sh（8 GPU）提交正式论文训练。"

wait "${TRAINING_PID}"
echo "PRE-TEST 训练完成！检查点: ${SAVE_CKPT}"
