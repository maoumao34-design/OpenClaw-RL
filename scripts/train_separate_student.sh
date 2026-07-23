#!/bin/bash
# train_separate_student.sh
#
# OpenClaw-RL Hybrid RL 训练编排（Table 3 Phase 3a: Separate-Student）
#
# 2026-07-23：架构重新核实后确认，Joint 需要复用 Separate 各阶段真实产出的
# homework 内容才能正确建立 homework1/homework2（不能自己顺序拼凑），而
# Separate 本身也是 Table 3 要报告的一列数字，无论如何都要先跑。这个脚本是
# 四阶段方案（Separate-Student → Separate-TA → Separate-Teacher → Joint）
# 的第一步。完整推导过程见 docs/issues_log.md 2026-07-23 条目、
# docs/paper_reproduction_scope.md "Joint vs Separate 区别"。
#
# 跟 train_with_services.sh（Joint）的区别：
#   - 只跑 Student 一个角色，不涉及 TA/Teacher，不需要 INIT 阶段
#   - 题数上限用论文 Appendix A.1 的 SESSION_LIMIT=72（"at most 72 tasks are
#     used for evaluation"），不是 Joint 那种覆盖全量数据集的 JOINT_NUM_PROBLEMS
#   - Student 的 72 题跑完后主动停止训练（官方 --num-rollout 是个很大的数，
#     不会因为模拟对话跑完就自动停，这个"停止"逻辑是我们自己设计的编排，
#     没有官方参考）
#   - workspace 直接落在 /dfs/data 下的永久路径（不是 runtime/<run_id>/ 这种
#     每次训练都换时间戳的临时目录），跑出来的 homework/ 就是 Phase B（Separate-TA）
#     和 Phase D（Joint）要用的真实产物，不需要额外拷贝一步
#
# 共用不变的部分（跟 train_with_services.sh 完全一致，直接复用）：GPU 启动、
# 外部 Simulator 连通性检查、OpenClaw gateway 启动、5 个 OpenClaw 版本漂移
# 补丁、verification-gate 补丁、PRM turn 内容调试补丁、conda 环境。
#
# modelfactory job 提交：
#   代码解释器: /bin/bash -i /dfs/data/start_tools.sh && /bin/bash -i
#   代码路径:   /dfs/data/openclaw-rl-project/openclaw-rl/scripts/train_separate_student.sh
#   GPU 数量:   8（跟 Joint 用同一套训练基础设施配置，论文训练超参数不分 Joint/Separate）
#
# Simulator 地址：编辑 scripts/simulator.env（见 simulator.env.example）
#
# 端口（训练机本机）：
#   30000  → RL training proxy
#   18789  → OpenClaw gateway

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

# 外部 Simulator（必填）
SIMULATOR_BASE_URL=${SIMULATOR_BASE_URL:-}
SIMULATOR_API_KEY=${SIMULATOR_API_KEY:-EMPTY}
EXTERNAL_MODEL=${EXTERNAL_MODEL:-qwen3-32b}

SGLANG_API_KEY=${SGLANG_API_KEY:-openclaw-rl-key}

# 训练占满 8 张 GPU（论文 4+2+1+1，Separate 和 Joint 用同一套训练超参数配置）
NUM_TRAINING_GPUS=${NUM_TRAINING_GPUS:-8}
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

SAVE_CKPT=${SAVE_CKPT:-/dfs/data/openclaw-rl-project/checkpoints/qwen3-4b-openclaw-separate-student}
REPO_ROOT=${REPO_ROOT:-/dfs/data/openclaw-rl-project/OpenClaw-RL-official}
DATASET=${DATASET:-${REPO_ROOT}/openclaw-test/GSM8K.json}
# 论文 Appendix A.1："By default, we set the conversation-session limit to
# 72, meaning that at most 72 tasks are used for evaluation." Separate-Student
# 只跑这一个角色，题数上限直接用这个值，不需要 Joint 那种覆盖全量数据集的
# JOINT_NUM_PROBLEMS。
SESSION_LIMIT=${SESSION_LIMIT:-72}
CONDA_ENV=${CONDA_ENV:-/dfs/data/envs/openclaw-rl}
CONDA_BASE=${CONDA_BASE:-/dfs/data/miniconda3}

LOGS_DIR=${LOGS_DIR:-/dfs/data/openclaw-rl-project/logs/separate_student_$(date +%Y%m%d_%H%M%S)}
# 2026-07-23：workspace 直接落在 /dfs/data 下的永久路径，不用 runtime/<run_id>/
# 这种每次训练都换时间戳的临时目录——这里跑出来的 homework/ 是 Phase B
# （Separate-TA）和 Phase D（Joint 的 homework1 种子）要复用的真实产物，
# 必须能跨训练任务稳定找到，不能用会变的路径。见 docs/paper_reproduction_scope.md
# "产物永久存放路径"。
WORKSPACE="/dfs/data/openclaw-rl-project/table3-artifacts/separate-student"
OPENCLAW_DIR="${LOGS_DIR}/openclaw-test-patched"

mkdir -p "${LOGS_DIR}" "${WORKSPACE}"
echo "日志目录: ${LOGS_DIR}"
echo "workspace（永久）: ${WORKSPACE}"
echo "外部 Simulator: ${SIMULATOR_BASE_URL} (model=${EXTERNAL_MODEL})"

# openclaw-test/*.py 硬编码 "model": "default"，当前 OpenClaw CLI（2026.6.9）的
# /v1/chat/completions 只认 openclaw/openclaw-<agentId> 这套 agent-target 格式，
# 会直接 400。生成一份改了这一个字段的补丁副本，官方目录本身不动。这份补丁同时
# 包含 Student/TA/Teacher 会话级文件核验（verification-gate），虽然这次只跑
# Student，但补丁脚本三个文件一起生成，不影响 Separate-Student 本身。
bash "${SCRIPTS_DIR}/prepare_openclaw_test_scripts.sh" "${REPO_ROOT}" "${OPENCLAW_DIR}"

# header/dispatcher 注入均已确认在这个 OpenClaw 版本上结构性失效，改用
# appendSystemContext 往 system prompt 正文里塞标记、服务端解析后清理掉。
# 详见 scripts/prepare_patched_rl_training_headers.sh 和
# scripts/prepare_patched_openclaw_opd.sh 顶部注释。
PATCHED_OPD_DIR="${LOGS_DIR}/patched-openclaw-opd"
bash "${SCRIPTS_DIR}/prepare_patched_openclaw_opd.sh" "${REPO_ROOT}" "${PATCHED_OPD_DIR}"

# 临时诊断补丁：PRM eval 日志加一行调试信息，打出每个 turn 实际打分用的
# response_text/next_state_text，方便确认"哪个 turn 编号对应哪个真实动作"。
# 见 docs/issues_log.md 2026-07-22 条目。
PATCHED_COMBINE_SELECT_DIR="${LOGS_DIR}/patched-openclaw-combine-select"
bash "${SCRIPTS_DIR}/prepare_patched_openclaw_combine_select.sh" "${REPO_ROOT}" "${PATCHED_COMBINE_SELECT_DIR}"

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
    # 2026-07-23 修复：原来拼 <base>/health 是沿用自建 sglang 服务器（launch_simulator.sh）
    # 自带的标准探活路径。换成走内部平台网关（如 Lenovo modelfactory service-large）时，
    # 这类网关往往只实现标准 OpenAI 兼容路径（/v1/models、/v1/chat/completions 等），
    # 没有 /health，探活请求会一直连不上/超时，跟 Simulator 服务本身是否就绪无关。改用
    # GET /v1/models（两种后端都支持的标准路径）+ 可选 Authorization 头（自建 sglang 未
    # 设 key 时这个头会被忽略，无副作用）；curl 显式加 --max-time，避免网络不可达时一直
    # 挂起、把每轮循环拖到远超预期的 10s。见 docs/issues_log.md 2026-07-23 条目。
    local models_url="${SIMULATOR_BASE_URL%/}/models"
    local auth_args=()
    if [ -n "${SIMULATOR_API_KEY:-}" ] && [ "${SIMULATOR_API_KEY}" != "EMPTY" ]; then
        auth_args=(-H "Authorization: Bearer ${SIMULATOR_API_KEY}")
    fi
    echo "检查外部 Simulator: ${models_url}"
    while ! curl -sf --max-time 10 "${models_url}" "${auth_args[@]}" > /dev/null 2>&1; do
        sleep 10
        waited=$((waited + 10))
        if [ ${waited} -ge ${max_wait} ]; then
            echo "超时：外部 Simulator 在 ${max_wait}s 内不可达" >&2
            echo "请确认 Simulator 服务已启动、网络策略已开通、SIMULATOR_BASE_URL 配置正确" >&2
            return 1
        fi
        echo "  已等待 ${waited}s..."
    done
    echo "外部 Simulator 已就绪 (${models_url})"
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
# 第1步：启动训练（GPU 0-7，跟 Joint 用同一套 Hybrid RL 训练后端）
# =====================================================================
echo ""
echo "=== [1/3] 启动训练（GPU ${TRAINING_CUDA_DEVICES}，NUM_GPUS=${NUM_TRAINING_GPUS}）==="

CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_DEVICES}" \
  REPO_ROOT="${REPO_ROOT}" \
  NUM_GPUS="${NUM_TRAINING_GPUS}" \
  HF_CKPT="${POLICY_MODEL_PATH}" \
  REF_LOAD="${POLICY_TORCH_DIST}" \
  SAVE_CKPT="${SAVE_CKPT}" \
  PRM_MODEL_PATH="${POLICY_MODEL_PATH}" \
  PRM_TEACHER_LOAD="${POLICY_TORCH_DIST}" \
  SGLANG_API_KEY="${SGLANG_API_KEY}" \
  PATCHED_OPD_DIR="${PATCHED_OPD_DIR}" \
  PATCHED_COMBINE_SELECT_DIR="${PATCHED_COMBINE_SELECT_DIR}" \
  bash "${SCRIPTS_DIR}/run_openclaw_topk_select_modelfactory.sh" \
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

# RL proxy :30000 必须先起来；OpenClaw primary 指向 127.0.0.1:30000，且 SGLang 加载较慢。
echo "等待 RL training proxy (port 30000)..."
wait_for_port "RL training proxy" 30000 900 "" "${LOGS_DIR}/training.log"

echo ""
echo "确保 chatCompletions 端点已启用..." | tee -a "${LOGS_DIR}/openclaw.log"
openclaw config set gateway.http.endpoints.chatCompletions.enabled true \
    >> "${LOGS_DIR}/openclaw.log" 2>&1
echo "[verify] gateway.http.endpoints.chatCompletions.enabled = $(openclaw config get gateway.http.endpoints.chatCompletions.enabled 2>&1 | tail -1)" \
    | tee -a "${LOGS_DIR}/openclaw.log"

echo "确保 compaction.reserveTokens 为官方默认值 16384..." | tee -a "${LOGS_DIR}/openclaw.log"
openclaw config set agents.defaults.compaction.reserveTokens 16384 \
    >> "${LOGS_DIR}/openclaw.log" 2>&1
echo "[verify] agents.defaults.compaction.reserveTokens = $(openclaw config get agents.defaults.compaction.reserveTokens 2>&1 | tail -1)" \
    | tee -a "${LOGS_DIR}/openclaw.log"

echo "确保 compaction.reserveTokensFloor 为 16384（reserveTokens 单独设置对 precheck 阈值计算无效，见 issues_log.md 2026-07-15）..." \
    | tee -a "${LOGS_DIR}/openclaw.log"
openclaw config set agents.defaults.compaction.reserveTokensFloor 16384 \
    >> "${LOGS_DIR}/openclaw.log" 2>&1
echo "[verify] agents.defaults.compaction.reserveTokensFloor = $(openclaw config get agents.defaults.compaction.reserveTokensFloor 2>&1 | tail -1)" \
    | tee -a "${LOGS_DIR}/openclaw.log"

# agents.defaults.workspace 优先级高于 OPENCLAW_WORKSPACE_DIR 环境变量
# （agent-scope-config.ts 先查 config 再退回环境变量），每次启动前强制设为
# 本次的永久 workspace 路径。
echo "确保 agents.defaults.workspace 指向永久 workspace 路径..." \
    | tee -a "${LOGS_DIR}/openclaw.log"
openclaw config set agents.defaults.workspace "${WORKSPACE}" \
    >> "${LOGS_DIR}/openclaw.log" 2>&1
echo "[verify] agents.defaults.workspace = $(openclaw config get agents.defaults.workspace 2>&1 | tail -1)" \
    | tee -a "${LOGS_DIR}/openclaw.log"

# 部署 rl-training-headers 插件（appendSystemContext 版本）。
echo "生成并部署 rl-training-headers 插件（appendSystemContext 版本）..." \
    | tee -a "${LOGS_DIR}/openclaw.log"
PATCHED_PLUGIN_DIR="${LOGS_DIR}/patched-rl-training-headers"
bash "${SCRIPTS_DIR}/prepare_patched_rl_training_headers.sh" "${REPO_ROOT}" "${PATCHED_PLUGIN_DIR}"
SYSTEM_PLUGIN_DIR="/usr/lib/node_modules/openclaw/dist/extensions/rl-training-headers"
mkdir -p "${SYSTEM_PLUGIN_DIR}"
cp "${PATCHED_PLUGIN_DIR}/index.js" "${SYSTEM_PLUGIN_DIR}/index.js"
cp "${PATCHED_PLUGIN_DIR}/openclaw.plugin.json" "${SYSTEM_PLUGIN_DIR}/openclaw.plugin.json"
cp "${PATCHED_PLUGIN_DIR}/package.json" "${SYSTEM_PLUGIN_DIR}/package.json"
openclaw plugins enable rl-training-headers >> "${LOGS_DIR}/openclaw.log" 2>&1 || true

# 5 个 OpenClaw 版本漂移补丁（Execution Bias / overflow-recovery /
# Assistant Output Directives / cli-compaction / Silent Reply Policy），
# 跟 Separate/Joint 无关，Student/TA/Teacher 都需要，原样复用。详细说明见
# train_with_services.sh 同名补丁块的注释，这里不重复。
echo "生成并部署 sglang execution-bias 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
SGLANG_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/extensions/sglang/index.js"
PATCHED_SGLANG_DIR="${LOGS_DIR}/patched-sglang"
bash "${SCRIPTS_DIR}/prepare_patched_sglang_execution_bias.sh" "${SGLANG_LIVE_FILE}" "${PATCHED_SGLANG_DIR}"
cp "${PATCHED_SGLANG_DIR}/index.js" "${SGLANG_LIVE_FILE}"

echo "生成并部署 embedded-agent overflow-recovery 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
EMBEDDED_AGENT_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/embedded-agent-Cv16r2d1.js"
PATCHED_EMBEDDED_AGENT_DIR="${LOGS_DIR}/patched-embedded-agent"
bash "${SCRIPTS_DIR}/prepare_patched_embedded_agent_overflow_recovery.sh" "${EMBEDDED_AGENT_LIVE_FILE}" "${PATCHED_EMBEDDED_AGENT_DIR}"
cp "${PATCHED_EMBEDDED_AGENT_DIR}/embedded-agent-Cv16r2d1.js" "${EMBEDDED_AGENT_LIVE_FILE}"

echo "生成并部署 system-prompt output-directives 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
SYSTEM_PROMPT_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/system-prompt-config-CLAPATdy.js"
PATCHED_SYSTEM_PROMPT_DIR="${LOGS_DIR}/patched-system-prompt"
bash "${SCRIPTS_DIR}/prepare_patched_system_prompt_output_directives.sh" "${SYSTEM_PROMPT_LIVE_FILE}" "${PATCHED_SYSTEM_PROMPT_DIR}"
cp "${PATCHED_SYSTEM_PROMPT_DIR}/system-prompt-config-CLAPATdy.js" "${SYSTEM_PROMPT_LIVE_FILE}"

echo "生成并部署 cli-compaction 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
CLI_COMPACTION_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/cli-compaction-B6C2IDnn.js"
PATCHED_CLI_COMPACTION_DIR="${LOGS_DIR}/patched-cli-compaction"
bash "${SCRIPTS_DIR}/prepare_patched_cli_compaction.sh" "${CLI_COMPACTION_LIVE_FILE}" "${PATCHED_CLI_COMPACTION_DIR}"
cp "${PATCHED_CLI_COMPACTION_DIR}/cli-compaction-B6C2IDnn.js" "${CLI_COMPACTION_LIVE_FILE}"

echo "生成并部署 silent-reply-policy 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
SILENT_REPLY_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/effective-reply-route-BnYlac-J.js"
PATCHED_SILENT_REPLY_DIR="${LOGS_DIR}/patched-silent-reply-policy"
bash "${SCRIPTS_DIR}/prepare_patched_silent_reply_policy.sh" "${SILENT_REPLY_LIVE_FILE}" "${PATCHED_SILENT_REPLY_DIR}"
cp "${PATCHED_SILENT_REPLY_DIR}/effective-reply-route-BnYlac-J.js" "${SILENT_REPLY_LIVE_FILE}"

echo "声明 sglang/qwen3-4b 的 contextWindow/maxTokens..." | tee -a "${LOGS_DIR}/openclaw.log"
python3 -c "
import json, pathlib
cfg = pathlib.Path.home() / '.openclaw/openclaw.json'
d = json.loads(cfg.read_text())
sg = d.setdefault('models', {}).setdefault('providers', {}).setdefault('sglang', {})
sg['api'] = 'openai-completions'
sg.pop('headers', None)
sg['models'] = [{
    'id': 'qwen3-4b',
    'name': 'Qwen3-4B-Thinking (RL policy)',
    'reasoning': True,
    'input': ['text'],
    'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0},
    'contextWindow': 32768,
    'maxTokens': 8192,
}]
cfg.write_text(json.dumps(d, indent=2, ensure_ascii=False))
print('patched models.providers.sglang.models')
" 2>&1 | tee -a "${LOGS_DIR}/openclaw.log"

echo "启动 OpenClaw gateway（port 18789）..."
OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
  OPENCLAW_WORKSPACE_DIR="${WORKSPACE}" \
  openclaw gateway run --allow-unconfigured --force \
  >> "${LOGS_DIR}/openclaw.log" 2>&1 &
OPENCLAW_PID=$!

wait_for_port "OpenClaw gateway" 18789 300 "${OPENCLAW_PID}" "${LOGS_DIR}/openclaw.log"

# =====================================================================
# 第3步：Student-only 模拟（Table 3 Phase 3a: Separate-Student）
# 论文 Appendix A.1："In the student evaluation setting, the script
# automatically creates a working directory... By default, we set the
# conversation-session limit to 72." 只跑 Student 一个角色，不需要 INIT，
# 不涉及 TA/Teacher/homework1/homework2。
# =====================================================================
echo ""
echo "=== [3/3] Student-only 模拟（最多 ${SESSION_LIMIT} 题）==="

STUDENT_ALL="${LOGS_DIR}/results_student_all.txt"

run_one_persona() {
    local name=$1 script=$2 num=$3 output=$4
    wait_for_port "OpenClaw gateway" 18789 120 "${OPENCLAW_PID:-}" "${LOGS_DIR}/openclaw.log" || {
        echo "错误：${name} 模拟开始前网关不可达" >&2
    }
    if OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
       OPENAI_API_KEY="${SIMULATOR_API_KEY}" \
       OPENAI_BASE_URL="${SIMULATOR_BASE_URL}" \
       EXTERNAL_MODEL="${EXTERNAL_MODEL}" \
       OPENCLAW_GATEWAY_URL=http://localhost:18789 \
       OPENCLAW_WORKSPACE="${WORKSPACE}" \
       python "${OPENCLAW_DIR}/${script}" \
           --dataset "${DATASET}" \
           --num-problems "${num}" \
           --output "${output}" \
           --max-retries 8; then
        return 0
    fi
    echo "警告：${name} 模拟未完全完成，继续（数据可能不完整，见 issues_log.md）" >&2
}

run_student_only_phase() {
    echo "--- Separate-Student 阶段开始（最多 ${SESSION_LIMIT} 题）---"
    local results="${LOGS_DIR}/results_student_separate.txt"
    run_one_persona "Student" "student_chat.py" "${SESSION_LIMIT}" "${results}"
    cat "${results}" >> "${STUDENT_ALL}" 2>/dev/null || true
    echo "--- Separate-Student 阶段结束 ---"
}

simulation_loop() {
    run_student_only_phase 2>&1 | tee -a "${LOGS_DIR}/simulation.log"

    # 2026-07-23：官方 --num-rollout 是个很大的数，不会因为模拟对话跑完就
    # 自动停止训练——这个"跑完就主动停止训练"的逻辑是我们自己设计的编排，
    # 没有官方参考（同 Joint 阶段的收尾逻辑，这次改成真正主动 kill，不是
    # 单纯跳出等待循环）。
    echo "Student 模拟已跑完，主动停止训练..." | tee -a "${LOGS_DIR}/simulation.log"
    kill "${TRAINING_PID}" 2>/dev/null || true
    ray stop --force 2>/dev/null || true

    echo "" | tee -a "${LOGS_DIR}/simulation.log"
    echo "=== 收敛检测（Table 3 指标，仅 Student）===" | tee -a "${LOGS_DIR}/simulation.log"
    # check_convergence.py 的 --ta/--teacher 是必填参数，这次没有 TA/Teacher
    # 数据，传两个不存在的占位路径——脚本本身对"文件不存在"有容错（当 0 个
    # session 处理），会打印"NOT converged (checked 0 sessions)"，不影响
    # Student 那一行的真实结果。
    python "${SCRIPTS_DIR}/check_convergence.py" \
        --student "${STUDENT_ALL}" \
        --ta      "${LOGS_DIR}/results_TA_all.txt.does-not-exist" \
        --teacher "${LOGS_DIR}/results_teacher_all.txt.does-not-exist" \
        2>&1 | tee "${LOGS_DIR}/convergence_result.txt"
}

simulation_loop &
SIM_LOOP_PID=$!

echo ""
echo "所有服务已启动，训练进行中..."
echo "  日志目录:       ${LOGS_DIR}/"
echo "  workspace（永久）: ${WORKSPACE}"
echo "  外部 Simulator: ${SIMULATOR_BASE_URL}"
echo "  训练日志:       tail -f ${LOGS_DIR}/training.log"
echo "  OpenClaw 日志:  tail -f ${LOGS_DIR}/openclaw.log"
echo "  Ray dashboard:  http://127.0.0.1:8265"

wait "${TRAINING_PID}" 2>/dev/null || true
echo "训练完成（Separate-Student 阶段跑完后主动停止）！检查点: ${SAVE_CKPT}"
echo "homework/ 永久产物: ${WORKSPACE}/homework/"
