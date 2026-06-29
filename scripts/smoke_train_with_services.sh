#!/bin/bash
echo "=== SCRIPT STARTED: CWD=$(pwd) SELF=$0 ===" && touch /dfs/data/openclaw-rl-project/logs/smoke_debug_started
# =============================================================================
# SMOKE TEST ONLY — Step B：3 GPU 端到端冒烟（Hybrid RL 缩配）
#
# 目的：在申请 8 GPU 正式 job 前，验证依赖、Ray、RL proxy :30000、
#       OpenClaw gateway、外部 Simulator、student/TA/teacher 模拟不会立刻报错。
#
# ⚠️  不是论文 Table 3 配置，结果不可用于论文复现对比。
# ⚠️  正式训练请用 scripts/train_with_services.sh（8 GPU）。
#
# modelfactory job 提交示例：
#   代码解释器: bash
#   代码路径:   .../OpenClaw-RL/scripts/smoke_train_with_services.sh
#   GPU 数量:   3（或 4，脚本默认只用 0-2）
#
# Simulator 地址：编辑 scripts/simulator.env（见 simulator.env.example）
#
# openclaw.json 需已配置 primary=sglang/qwen3-4b, baseUrl=http://127.0.0.1:30000/v1
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

POLICY_MODEL_PATH=${POLICY_MODEL_PATH:-/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507}
POLICY_TORCH_DIST=${POLICY_TORCH_DIST:-/dfs/data/models/torch_dist/qwen3-4b-thinking-2507}

SIMULATOR_BASE_URL=${SIMULATOR_BASE_URL:-}
SIMULATOR_API_KEY=${SIMULATOR_API_KEY:-EMPTY}
EXTERNAL_MODEL=${EXTERNAL_MODEL:-qwen3-32b}
SGLANG_API_KEY=${SGLANG_API_KEY:-openclaw-rl-key}

# 3 GPU 缩配（第 4 张卡若有也不使用，除非改 TRAINING_CUDA_DEVICES）
NUM_TRAINING_GPUS=${NUM_TRAINING_GPUS:-3}
TRAINING_CUDA_DEVICES=${TRAINING_CUDA_DEVICES:-$(seq -s, 0 $((NUM_TRAINING_GPUS - 1)))}

SAVE_CKPT=${SAVE_CKPT:-/dfs/data/openclaw-rl-project/checkpoints/smoke-qwen3-4b-openclaw-combine}
REPO_ROOT=${REPO_ROOT:-/dfs/data/openclaw-rl-project/OpenClaw-RL-official}
NUM_PROBLEMS_PER_ROUND=${NUM_PROBLEMS_PER_ROUND:-1}
SESSION_LIMIT=${SESSION_LIMIT:-1}
DATASET=${DATASET:-${REPO_ROOT}/openclaw-test/GSM8K.json}
CONDA_ENV=${CONDA_ENV:-/dfs/data/envs/openclaw-rl}
CONDA_BASE=${CONDA_BASE:-/dfs/data/miniconda3}

OPENCLAW_DIR=${REPO_ROOT}/openclaw-test
LOGS_DIR=${LOGS_DIR:-/dfs/data/openclaw-rl-project/logs/smoke_$(date +%Y%m%d_%H%M%S)}
WORKSPACE=${HOME}/.openclaw/workspace
SMOKE_COMBINE_LAUNCHER="${SCRIPTS_DIR}/smoke_run_qwen3_4b_openclaw_combine.sh"

if [ -z "${SIMULATOR_BASE_URL}" ] || [[ "${SIMULATOR_BASE_URL}" == *"<"* ]]; then
    echo "错误：请在 ${SIMULATOR_ENV} 中填写 SIMULATOR_BASE_URL" >&2
    exit 1
fi

if [ ! -x "${SMOKE_COMBINE_LAUNCHER}" ] && [ ! -f "${SMOKE_COMBINE_LAUNCHER}" ]; then
    echo "错误：找不到 ${SMOKE_COMBINE_LAUNCHER}" >&2
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
    echo "错误：无法读取 OPENCLAW_GATEWAY_TOKEN（检查 ~/.openclaw/openclaw.json）" >&2
    exit 1
fi

mkdir -p "${LOGS_DIR}"
echo ""
echo "============================================================"
echo "  OpenClaw-RL SMOKE TEST (3 GPU) — NOT production training"
echo "============================================================"
echo "日志目录:       ${LOGS_DIR}"
echo "外部 Simulator: ${SIMULATOR_BASE_URL} (model=${EXTERNAL_MODEL})"
echo "训练 GPU:       ${TRAINING_CUDA_DEVICES} (NUM_GPUS=${NUM_TRAINING_GPUS})"
echo "模拟:           ${NUM_PROBLEMS_PER_ROUND} 题 × 1 轮 (Student→TA→Teacher)"
echo "正式训练脚本:   scripts/train_with_services.sh (8 GPU)"
echo ""

if [ -n "${CONDA_ENV}" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    echo "已激活 conda: ${CONDA_ENV}"
fi

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
    while ! nc -z localhost "${port}" 2>/dev/null; do
        sleep 10
        waited=$((waited + 10))
        if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
            echo "错误：${name} 依赖进程已退出" >&2
            dump_log_tail "${logfile}"
            return 1
        fi
        if [ -n "${logfile}" ] && [ -f "${logfile}" ]; then
        if grep -qE "Job 'raysubmit_.*' failed|can't open file '/workspace/train_async.py'|patch failed|Traceback|CUDA out of memory" "${logfile}" 2>/dev/null; then
            echo "错误：Ray 训练 job 已失败（见 ${logfile}）" >&2
            grep -E "failed|Error|error|Traceback|can't open|patch failed|OOM|CUDA" "${logfile}" 2>/dev/null | tail -20 >&2 || true
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
            echo "超时：外部 Simulator 不可达" >&2
            return 1
        fi
        echo "  已等待 ${waited}s..."
    done
    echo "外部 Simulator 已就绪"
}

launch_openclaw_gateway() {
    echo "启动 OpenClaw gateway（port 18789，headless）..."
    local openclaw_cmd=(openclaw gateway run --allow-unconfigured --force --bind loopback --verbose
        --token "${OPENCLAW_GATEWAY_TOKEN}")
    if command -v stdbuf >/dev/null 2>&1; then
        OPENCLAW_SKIP_CHANNELS=1 \
        OPENCLAW_SKIP_PROVIDERS=1 \
        OPENCLAW_SKIP_BROWSER_CONTROL_SERVER=1 \
        OPENCLAW_GATEWAY_STARTUP_TRACE=1 \
        OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
        stdbuf -oL -eL "${openclaw_cmd[@]}" >> "${LOGS_DIR}/openclaw.log" 2>&1 &
    else
        OPENCLAW_SKIP_CHANNELS=1 \
        OPENCLAW_SKIP_PROVIDERS=1 \
        OPENCLAW_SKIP_BROWSER_CONTROL_SERVER=1 \
        OPENCLAW_GATEWAY_STARTUP_TRACE=1 \
        OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
        "${openclaw_cmd[@]}" >> "${LOGS_DIR}/openclaw.log" 2>&1 &
    fi
    OPENCLAW_PID=$!
    echo "OpenClaw PID: ${OPENCLAW_PID}"
}

wait_for_openclaw_gateway() {
    local max_wait=${1:-900}
    local waited=0
    local logfile="${LOGS_DIR}/openclaw.log"
    local last_log_size=0
    echo "等待 OpenClaw gateway (http://127.0.0.1:18789/healthz)..."
    while ! curl -sf "http://127.0.0.1:18789/healthz" > /dev/null 2>&1; do
        sleep 10
        waited=$((waited + 10))
        if [ -n "${OPENCLAW_PID}" ] && ! kill -0 "${OPENCLAW_PID}" 2>/dev/null; then
            echo "错误：OpenClaw gateway 进程已退出" >&2
            dump_log_tail "${logfile}"
            return 1
        fi
        if [ -f "${logfile}" ]; then
            local cur_size
            cur_size=$(wc -c < "${logfile}" | tr -d ' ')
            if [ "${cur_size}" -eq "${last_log_size}" ] && [ $((waited % 60)) -eq 0 ] && [ "${waited}" -ge 60 ]; then
                echo "  提示：openclaw.log ${waited}s 无新输出；确认 ~/.openclaw/openclaw.json 含 gateway.mode=local 且 controlUi.enabled=false" >&2
            fi
            last_log_size="${cur_size}"
        fi
        if [ ${waited} -ge ${max_wait} ]; then
            echo "超时：OpenClaw gateway 在 ${max_wait}s 内未启动" >&2
            dump_log_tail "${logfile}"
            return 1
        fi
        echo "  已等待 ${waited}s..."
    done
    echo "OpenClaw gateway 已就绪 (port 18789)"
}

OPENCLAW_PID=""
TRAINING_PID=""

cleanup() {
    echo ""
    echo "清理中..."
    [ -n "${OPENCLAW_PID}" ] && kill "${OPENCLAW_PID}" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    wait 2>/dev/null || true
    echo "清理完成。"
}
trap cleanup EXIT INT TERM

# --- [1/3] 启动缩配训练 ---
echo ""
echo "=== [1/3] 启动 SMOKE 训练（GPU ${TRAINING_CUDA_DEVICES}）==="

export SMOKE_COMBINE_SCRIPT="${LOGS_DIR}/.smoke_run_qwen3_4b_openclaw_combine.sh"

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
  bash "${SMOKE_COMBINE_LAUNCHER}" \
  > "${LOGS_DIR}/training.log" 2>&1 &
TRAINING_PID=$!

echo "训练 PID: ${TRAINING_PID}，等待 Ray head (port 8265)..."
until curl -sf http://127.0.0.1:8265/api/version > /dev/null 2>&1; do
    sleep 5
    if ! kill -0 "${TRAINING_PID}" 2>/dev/null; then
        echo "错误：SMOKE 训练进程意外退出" >&2
        dump_log_tail "${LOGS_DIR}/training.log" 120
        exit 1
    fi
done
echo "Ray head 已就绪"

# --- [2/3] Simulator + OpenClaw + RL proxy ---
echo ""
echo "=== [2/3] 外部 Simulator + OpenClaw gateway + RL proxy :30000 ==="

wait_for_external_simulator 300

# RL proxy :30000 必须先起来；OpenClaw primary 指向 127.0.0.1:30000，且 SGLang 加载较慢。
echo "等待 RL training proxy (port 30000)..."
wait_for_port "RL training proxy" 30000 900 "" "${LOGS_DIR}/training.log"

launch_openclaw_gateway
wait_for_openclaw_gateway 900

# --- [3/3] 一轮模拟（foreground，失败即退出）---
echo ""
echo "=== [3/3] SMOKE 模拟 1 轮（Student → TA → Teacher，各 ${NUM_PROBLEMS_PER_ROUND} 题）==="

STUDENT_OUT="${LOGS_DIR}/results_student_smoke.txt"
TA_OUT="${LOGS_DIR}/results_TA_smoke.txt"
TEACHER_OUT="${LOGS_DIR}/results_teacher_smoke.txt"

rm -rf "${WORKSPACE}/homework" "${WORKSPACE}/homework1" "${WORKSPACE}/homework2"

run_smoke_chat() {
    local script_name=$1
    local output_path=$2
    OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
    OPENAI_API_KEY="${SIMULATOR_API_KEY}" \
    OPENAI_BASE_URL="${SIMULATOR_BASE_URL}" \
    EXTERNAL_MODEL="${EXTERNAL_MODEL}" \
    OPENCLAW_GATEWAY_URL=http://localhost:18789 \
    python "${OPENCLAW_DIR}/${script_name}" \
        --dataset "${DATASET}" \
        --num-problems "${NUM_PROBLEMS_PER_ROUND}" \
        --max-turns 4 \
        --output "${output_path}"
}

run_smoke_chat "student_chat.py" "${STUDENT_OUT}" 2>&1 | tee "${LOGS_DIR}/simulation.log"
run_smoke_chat "TA_chat.py" "${TA_OUT}" 2>&1 | tee -a "${LOGS_DIR}/simulation.log"
run_smoke_chat "teacher_chat.py" "${TEACHER_OUT}" 2>&1 | tee -a "${LOGS_DIR}/simulation.log"

if ! kill -0 "${TRAINING_PID}" 2>/dev/null; then
    echo ""
    echo "❌ SMOKE FAILED：模拟结束后训练进程已退出" >&2
    dump_log_tail "${LOGS_DIR}/training.log" 120
    exit 1
fi

echo ""
echo "============================================================"
echo "  ✅ SMOKE PASSED"
echo "============================================================"
echo "  Ray dashboard:     http://127.0.0.1:8265"
echo "  RL proxy:          http://127.0.0.1:30000/v1"
echo "  OpenClaw gateway:  http://127.0.0.1:18789"
echo "  日志目录:          ${LOGS_DIR}/"
echo ""
echo "下一步：用 scripts/train_with_services.sh 提交 8 GPU 正式训练。"
echo "（本 smoke job 将在清理后退出，不会长时间占用 GPU。）"
echo "============================================================"

# 结束训练进程，释放 GPU（smoke 不要求跑完 num-rollout）
kill "${TRAINING_PID}" 2>/dev/null || true
wait "${TRAINING_PID}" 2>/dev/null || true

exit 0
