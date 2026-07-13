#!/bin/bash
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
#   GPU 数量:   4（topk-select 需要 PRM Teacher，最少 Actor×1+Rollout×1+PRM×1+Teacher×1=4 GPU）
#
# Simulator 地址：编辑 scripts/simulator.env（见 simulator.env.example）
#
# openclaw.json 需已配置 primary=sglang/qwen3-4b, baseUrl=http://127.0.0.1:30000/v1
# =============================================================================

set -euo pipefail

# 2026-07-13：wandb.ai 在这个环境里直连不通，需要先起代理才能上报。这几行
# 不应该让整个训练 job 失败（代理偶尔起不来不该阻塞训练本身），所以都不用
# set -e 的严格模式处理，失败只打日志、不中断。
SINGBOX_OUT=$(bash /dfs/share-groups/foundationmodelgroup/LRM/proxy/sing-box.sh start 2>&1) \
    && echo "${SINGBOX_OUT}" \
    || echo "警告：代理启动失败，继续（wandb 可能上报不了）：${SINGBOX_OUT}"
set +u; source ~/.bashrc || true; set -u  # 拿 ~/.bashrc 里的 WANDB_API_KEY 等；.bashrc 引用未设置的 $PS1，set -u 下会报错，临时关闭
# 2026-07-13：pon/poff 在提交的 job 容器里查不到（跟交互式 workspace 终端不是
# 同一个环境，job 每次都是全新安装 sing-box），放弃依赖它，改成从 sing-box.sh
# 自己的启动输出里动态解析实际监听端口，不硬编码。
SINGBOX_PORT=$(echo "${SINGBOX_OUT:-}" | grep -oP 'Proxy listening on port:\s*\K[0-9]+' | head -1)
if [ -n "${SINGBOX_PORT:-}" ]; then
    export http_proxy="http://127.0.0.1:${SINGBOX_PORT}"
    export https_proxy="http://127.0.0.1:${SINGBOX_PORT}"
    echo "[proxy] 检测到端口 ${SINGBOX_PORT}，http_proxy=${http_proxy}"
else
    echo "警告：未能从 sing-box 输出解析出端口，wandb 可能上报不了"
fi
curl -I https://www.google.com || echo "警告：代理连通性检查失败，继续"

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
POLICY_TORCH_DIST=${POLICY_TORCH_DIST:-/dfs/data/models/Qwen3-4B-Thinking-2507-torch-dist}

SIMULATOR_BASE_URL=${SIMULATOR_BASE_URL:-}
SIMULATOR_API_KEY=${SIMULATOR_API_KEY:-EMPTY}
EXTERNAL_MODEL=${EXTERNAL_MODEL:-qwen3-32b}
SGLANG_API_KEY=${SGLANG_API_KEY:-openclaw-rl-key}

# 3 GPU 缩配（第 4 张卡若有也不使用，除非改 TRAINING_CUDA_DEVICES）
NUM_TRAINING_GPUS=${NUM_TRAINING_GPUS:-4}
TRAINING_CUDA_DEVICES=${TRAINING_CUDA_DEVICES:-$(seq -s, 0 $((NUM_TRAINING_GPUS - 1)))}

SAVE_CKPT=${SAVE_CKPT:-/dfs/data/openclaw-rl-project/checkpoints/smoke-qwen3-4b-openclaw-topk-select}
REPO_ROOT=${REPO_ROOT:-/dfs/data/openclaw-rl-project/OpenClaw-RL-official}
NUM_PROBLEMS_PER_ROUND=${NUM_PROBLEMS_PER_ROUND:-1}
SESSION_LIMIT=${SESSION_LIMIT:-1}
DATASET=${DATASET:-${REPO_ROOT}/openclaw-test/GSM8K.json}
CONDA_ENV=${CONDA_ENV:-/dfs/data/envs/openclaw-rl}
CONDA_BASE=${CONDA_BASE:-/dfs/data/miniconda3}

LOGS_DIR=${LOGS_DIR:-/dfs/data/openclaw-rl-project/logs/smoke_$(date +%Y%m%d_%H%M%S)}
WORKSPACE=${HOME}/.openclaw/workspace
OPENCLAW_DIR="${LOGS_DIR}/openclaw-test-patched"
SMOKE_TOPK_SELECT_LAUNCHER="${SCRIPTS_DIR}/smoke_run_qwen3_4b_openclaw_topk_select.sh"

if [ -z "${SIMULATOR_BASE_URL}" ] || [[ "${SIMULATOR_BASE_URL}" == *"<"* ]]; then
    echo "错误：请在 ${SIMULATOR_ENV} 中填写 SIMULATOR_BASE_URL" >&2
    exit 1
fi

if [ ! -x "${SMOKE_TOPK_SELECT_LAUNCHER}" ] && [ ! -f "${SMOKE_TOPK_SELECT_LAUNCHER}" ]; then
    echo "错误：找不到 ${SMOKE_TOPK_SELECT_LAUNCHER}" >&2
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

# openclaw-test/*.py 硬编码 "model": "default"，当前 OpenClaw CLI（2026.6.9）的
# /v1/chat/completions 只认 openclaw/openclaw-<agentId> 这套 agent-target 格式，
# 会直接 400。生成一份改了这一个字段的补丁副本，官方目录本身不动。
bash "${SCRIPTS_DIR}/prepare_openclaw_test_scripts.sh" "${REPO_ROOT}" "${OPENCLAW_DIR}"

# header/dispatcher 注入均已确认在这个 OpenClaw 版本上结构性失效——OpenClaw 加了
# 一层 SSRF 安全机制，绕开所有外部注入的 fetch/dispatcher，无配置开关（详见
# docs/issues_log.md 2026-07-09 第四/五部分）。改用 appendSystemContext 往
# system prompt 正文里塞 "[RL-TRAINING-META] session_id=... turn_type=..." 标记，
# 服务端解析后在转发给 sglang / 计算训练样本之前清理掉，模型和训练数据都看不到这段
# 标记。详见 scripts/prepare_patched_rl_training_headers.sh 和
# scripts/prepare_patched_openclaw_opd.sh 顶部注释。
PATCHED_OPD_DIR="${LOGS_DIR}/patched-openclaw-opd"
bash "${SCRIPTS_DIR}/prepare_patched_openclaw_opd.sh" "${REPO_ROOT}" "${PATCHED_OPD_DIR}"

echo ""
echo "============================================================"
echo "  OpenClaw-RL SMOKE TEST (4 GPU) — NOT production training"
echo "============================================================"
echo "日志目录:       ${LOGS_DIR}"
echo "外部 Simulator: ${SIMULATOR_BASE_URL} (model=${EXTERNAL_MODEL})"
echo "训练 GPU:       ${TRAINING_CUDA_DEVICES} (NUM_GPUS=${NUM_TRAINING_GPUS})"
echo "模拟:           INIT ${NUM_PROBLEMS_PER_ROUND} 题×3 顺序 + 1 轮 Joint 并行（验证并发）topk-select k=4 m=1"
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
    while ! curl -s --max-time 5 "http://localhost:${port}/" > /dev/null 2>&1; do
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
    # 这个 job 容器的 ~/.openclaw/openclaw.json 可能是从早于本次修复的镜像/模板
    # 生成的（openclaw 本体在 /usr/lib/node_modules/ 是系统级安装，job 容器都有；
    # 但 ~/.openclaw/openclaw.json 是用户配置，不保证和交互式 shell 同步）。
    # 每次起 gateway 前强制确保这个开关是开着的，不依赖持久化是否跨环境生效。
    echo "确保 chatCompletions 端点已启用..." | tee -a "${LOGS_DIR}/openclaw.log"
    openclaw config set gateway.http.endpoints.chatCompletions.enabled true \
        >> "${LOGS_DIR}/openclaw.log" 2>&1
    echo "[verify] gateway.http.endpoints.chatCompletions.enabled = $(openclaw config get gateway.http.endpoints.chatCompletions.enabled 2>&1 | tail -1)" \
        | tee -a "${LOGS_DIR}/openclaw.log"

    # 2026-07-13 修复：这个环境里 compaction.reserveTokens 实际生效值是 20000（来源
    # 不明，不是 openclaw.json 里显式设置的，也不是官方代码默认值），导致给模型输出
    # 预留的 token 太多、留给 prompt 的预算被压缩，TA 批改任务的 prompt 稳定超预算
    # 造成 context overflow（issues_log.md 2026-07-13 条目）。显式设回官方默认值 16384。
    echo "确保 compaction.reserveTokens 为官方默认值 16384..." | tee -a "${LOGS_DIR}/openclaw.log"
    openclaw config set agents.defaults.compaction.reserveTokens 16384 \
        >> "${LOGS_DIR}/openclaw.log" 2>&1
    echo "[verify] agents.defaults.compaction.reserveTokens = $(openclaw config get agents.defaults.compaction.reserveTokens 2>&1 | tail -1)" \
        | tee -a "${LOGS_DIR}/openclaw.log"

    # 部署 rl-training-headers 插件（appendSystemContext 版本，见上方 PATCHED_OPD_DIR
    # 注释）。写入 OpenClaw 自己的系统安装目录（openclaw plugins list --verbose 确认的
    # source 路径），不是插件扩展开发目录——这个 OpenClaw 版本的插件加载器只扫描这里，
    # 没有指向任意目录加载插件的机制。这台机器上这个目录跟其他服务（code-server 等）
    # 共享，每次 smoke 都会覆盖，只影响这一个插件自己的文件。
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

    # models.providers.sglang 未显式声明 models[] 时 OpenClaw 走自动发现，不知道
    # 真实的 contextWindow/maxTokens，会用过大的默认值请求 max_completion_tokens
    # （实测 178210），被 sglang 400 拒绝。显式声明，maxTokens 明显小于
    # contextWindow 留出 prompt 空间（同 PRM_MAX_NEW_TOKENS 的道理）。
    #
    # contextWindow 改回 32768（与 run_openclaw_topk_select_modelfactory.sh 里
    # 官方 sglang-context-length 一致，不再缩到 8192）：2026-07-08 smoke 用
    # 8192 时 student/TA/teacher 100% 命中 "Context overflow: prompt too large
    # for the model"——这句提示语是 OpenClaw 自己的措辞，很可能是 OpenClaw
    # 客户端拿这里配置的 contextWindow 做请求前检查、在打到 sglang 之前就直接
    # 拒绝了，不是 sglang 400。8192 对真实 agent 系统提示词（工具 schema +
    # 作业内容）明显不够，contextWindow 和 sglang 实际 context_length 必须一致
    # 且足够大，否则两边有一个偏小都会复现这个问题。
    echo "声明 sglang/qwen3-4b 的 contextWindow/maxTokens..." | tee -a "${LOGS_DIR}/openclaw.log"
    python3 -c "
import json, pathlib
cfg = pathlib.Path.home() / '.openclaw/openclaw.json'
d = json.loads(cfg.read_text())
sg = d.setdefault('models', {}).setdefault('providers', {}).setdefault('sglang', {})
sg['api'] = 'openai-completions'
# 不再用静态 X-Turn-Type header（写死 'main' 分不清真实轮次和 OpenClaw 内部的
# context-summarization 兜底调用）。改用插件的 appendSystemContext 动态标记，
# 见上方 PATCHED_OPD_DIR/插件部署那两处注释。
sg['models'] = [{
    'id': 'qwen3-4b',
    'name': 'Qwen3-4B-Thinking (RL policy, smoke)',
    'reasoning': True,
    'input': ['text'],
    'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0},
    'contextWindow': 32768,
    'maxTokens': 4096,
}]
cfg.write_text(json.dumps(d, indent=2, ensure_ascii=False))
print('patched models.providers.sglang.models')
" 2>&1 | tee -a "${LOGS_DIR}/openclaw.log"

    echo "启动 OpenClaw gateway（port 18789）..."
    OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
    openclaw gateway run --allow-unconfigured --force \
        >> "${LOGS_DIR}/openclaw.log" 2>&1 &
    OPENCLAW_PID=$!
    echo "Gateway PID: ${OPENCLAW_PID}"
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

export SMOKE_TOPK_SELECT_SCRIPT="${LOGS_DIR}/.smoke_run_qwen3_4b_openclaw_topk_select.sh"

CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_DEVICES}" \
  REPO_ROOT="${REPO_ROOT}" \
  NUM_GPUS="${NUM_TRAINING_GPUS}" \
  HF_CKPT="${POLICY_MODEL_PATH}" \
  REF_LOAD="${POLICY_TORCH_DIST}" \
  SAVE_CKPT="${SAVE_CKPT}" \
  PRM_MODEL_PATH="${POLICY_MODEL_PATH}" \
  PRM_TEACHER_LOAD="${POLICY_TORCH_DIST}" \
  SGLANG_API_KEY="${SGLANG_API_KEY}" \
  USE_WANDB="${USE_WANDB:-0}" \
  PATCHED_OPD_DIR="${PATCHED_OPD_DIR}" \
  bash "${SMOKE_TOPK_SELECT_LAUNCHER}" \
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

# --- [3/3] SMOKE 模拟（INIT 顺序建立 + 1 轮 Joint 并行，foreground）---
echo ""
echo "=== [3/3] SMOKE 模拟：INIT 顺序（${NUM_PROBLEMS_PER_ROUND} 题×3）+ 1 轮 Joint 并行 ==="

STUDENT_OUT="${LOGS_DIR}/results_student_smoke.txt"
TA_OUT="${LOGS_DIR}/results_TA_smoke.txt"
TEACHER_OUT="${LOGS_DIR}/results_teacher_smoke.txt"

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

# INIT 阶段：顺序建立 homework1/ homework2/（与正式训练架构一致）
rm -rf "${WORKSPACE}/homework" "${WORKSPACE}/homework1" "${WORKSPACE}/homework2"

echo "[DEBUG] === OpenClaw API probe ===" | tee -a "${LOGS_DIR}/openclaw_debug.log"
echo "[DEBUG] 1) GET /v1/models" | tee -a "${LOGS_DIR}/openclaw_debug.log"
curl -sv -X GET http://localhost:18789/v1/models \
    -H "Authorization: Bearer ${OPENCLAW_GATEWAY_TOKEN}" \
    >> "${LOGS_DIR}/openclaw_debug.log" 2>&1 || true
echo "[DEBUG] 2) POST /v1/chat/completions model=openclaw/default" | tee -a "${LOGS_DIR}/openclaw_debug.log"
curl -sv -X POST http://localhost:18789/v1/chat/completions \
    -H "Authorization: Bearer ${OPENCLAW_GATEWAY_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"model":"openclaw/default","user":"smoke-probe","messages":[{"role":"user","content":"hello"}],"stream":false}' \
    >> "${LOGS_DIR}/openclaw_debug.log" 2>&1 || true
echo "--- openclaw_debug.log ---" && cat "${LOGS_DIR}/openclaw_debug.log"

echo "  [SMOKE INIT 1/3] Student → homework/..."
run_smoke_chat "student_chat.py" "${STUDENT_OUT}" 2>&1 | tee "${LOGS_DIR}/simulation.log"

echo "  [SMOKE INIT 2/3] TA（ensure_homework_dir: homework→homework1/）..."
run_smoke_chat "TA_chat.py" "${TA_OUT}" 2>&1 | tee -a "${LOGS_DIR}/simulation.log"

echo "  [SMOKE INIT 3/3] Teacher（ensure_homework_dir: homework1→homework2/）..."
run_smoke_chat "teacher_chat.py" "${TEACHER_OUT}" 2>&1 | tee -a "${LOGS_DIR}/simulation.log"

# Joint 阶段：三角色并行（验证并发无文件冲突）
echo ""
echo "  [SMOKE Joint] 三角色并行（各 ${NUM_PROBLEMS_PER_ROUND} 题）..."

JOINT_S="${LOGS_DIR}/results_student_smoke_joint.txt"
JOINT_TA="${LOGS_DIR}/results_TA_smoke_joint.txt"
JOINT_T="${LOGS_DIR}/results_teacher_smoke_joint.txt"

run_smoke_chat "student_chat.py" "${JOINT_S}" >> "${LOGS_DIR}/sim_student.log" 2>&1 &
pid_s=$!
run_smoke_chat "TA_chat.py" "${JOINT_TA}" >> "${LOGS_DIR}/sim_ta.log" 2>&1 &
pid_ta=$!
run_smoke_chat "teacher_chat.py" "${JOINT_T}" >> "${LOGS_DIR}/sim_teacher.log" 2>&1 &
pid_t=$!
wait "${pid_s}" "${pid_ta}" "${pid_t}"

echo "  [SMOKE Joint] 完成"
echo "  并行日志：${LOGS_DIR}/sim_student.log / sim_ta.log / sim_teacher.log"
echo ""

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
