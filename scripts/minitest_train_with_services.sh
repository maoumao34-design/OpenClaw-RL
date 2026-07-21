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
#   代码解释器: /bin/bash -i /dfs/data/start_tools.sh && /bin/bash -i
#              （2026-07-13 起：wandb.ai 需要走代理，start_tools.sh 负责
#              起代理，-i 让 pon 这个 alias 能展开；纯 `bash` 提交代理会
#              连不上，wandb 上报不了）
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

# wandb.ai 代理改由提交时的 start_tools.sh（bash -i 链式提交）负责起，见
# work_log.md 2026-07-13 条目；脚本内不再重复处理，避免和 start_tools.sh
# 冲突/重复启动。

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
# 见 train_with_services.sh 同日注释：官方脚本一次性设计，无分轮次编排依据
JOINT_NUM_PROBLEMS=${JOINT_NUM_PROBLEMS:-1319}
DATASET=${DATASET:-${REPO_ROOT}/openclaw-test/GSM8K.json}
SESSION_LIMIT=${SESSION_LIMIT:-72}
CONDA_ENV=${CONDA_ENV:-/dfs/data/envs/openclaw-rl}
CONDA_BASE=${CONDA_BASE:-/dfs/data/miniconda3}

LOGS_DIR=${LOGS_DIR:-/dfs/data/openclaw-rl-project/logs/minitest_$(date +%Y%m%d_%H%M%S)}
WORKSPACE=${HOME}/.openclaw/workspace
OPENCLAW_DIR="${LOGS_DIR}/openclaw-test-patched"
MINITEST_TOPK_SELECT_LAUNCHER="${SCRIPTS_DIR}/minitest_run_qwen3_4b_openclaw_topk_select.sh"

if [ ! -f "${MINITEST_TOPK_SELECT_LAUNCHER}" ]; then
    echo "错误：找不到 ${MINITEST_TOPK_SELECT_LAUNCHER}" >&2
    exit 1
fi

mkdir -p "${LOGS_DIR}"

# openclaw-test/*.py 硬编码 "model": "default"，当前 OpenClaw CLI（2026.6.9）的
# /v1/chat/completions 只认 openclaw/openclaw-<agentId> 这套 agent-target 格式，
# 会直接 400。生成一份改了这一个字段的补丁副本，官方目录本身不动。
bash "${SCRIPTS_DIR}/prepare_openclaw_test_scripts.sh" "${REPO_ROOT}" "${OPENCLAW_DIR}"

# header/dispatcher 注入均已确认在这个 OpenClaw 版本上结构性失效（详见
# docs/issues_log.md 2026-07-09 第四/五部分）。改用 appendSystemContext 往
# system prompt 正文里塞 "[RL-TRAINING-META] session_id=... turn_type=..." 标记，
# 服务端解析后在转发给 sglang / 计算训练样本之前清理掉。之前只加在 smoke 里测试，
# 现在已用真实数据验证过，同步到这里。详见 scripts/prepare_patched_rl_training_headers.sh
# 和 scripts/prepare_patched_openclaw_opd.sh 顶部注释。
PATCHED_OPD_DIR="${LOGS_DIR}/patched-openclaw-opd"
bash "${SCRIPTS_DIR}/prepare_patched_openclaw_opd.sh" "${REPO_ROOT}" "${PATCHED_OPD_DIR}"

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
  USE_WANDB="${USE_WANDB:-1}" \
  OPENCLAW_TOPK_SELECT_SCRIPT="${MINITEST_TOPK_SELECT_SCRIPT}" \
  PATCHED_OPD_DIR="${PATCHED_OPD_DIR}" \
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

# 每次起 gateway 前强制确保这个开关是开着的，不依赖持久化是否跨环境生效（同
# smoke_train_with_services.sh 的 launch_openclaw_gateway() 逻辑）。
echo "确保 chatCompletions 端点已启用..." | tee -a "${LOGS_DIR}/openclaw.log"
openclaw config set gateway.http.endpoints.chatCompletions.enabled true \
    >> "${LOGS_DIR}/openclaw.log" 2>&1
echo "[verify] gateway.http.endpoints.chatCompletions.enabled = $(openclaw config get gateway.http.endpoints.chatCompletions.enabled 2>&1 | tail -1)" \
    | tee -a "${LOGS_DIR}/openclaw.log"

# 2026-07-13 修复：这个环境里 compaction.reserveTokens 实际生效值是 20000（来源不明，
# 不是 openclaw.json 里显式设置的，也不是官方代码默认值），导致给模型输出预留的
# token 太多、留给 prompt 的预算被压缩到 32768-20000=12768，TA 批改任务的 prompt
# 一般在 13.6K 左右，稳定超预算约 843 token，造成 TA 每次都 context overflow /
# 生成不了回复（issues_log.md 2026-07-13 条目）。显式设回官方默认值 16384。
echo "确保 compaction.reserveTokens 为官方默认值 16384..." | tee -a "${LOGS_DIR}/openclaw.log"
openclaw config set agents.defaults.compaction.reserveTokens 16384 \
    >> "${LOGS_DIR}/openclaw.log" 2>&1
echo "[verify] agents.defaults.compaction.reserveTokens = $(openclaw config get agents.defaults.compaction.reserveTokens 2>&1 | tail -1)" \
    | tee -a "${LOGS_DIR}/openclaw.log"

# 2026-07-15 修复：reserveTokens 单独设置对实际 context-overflow precheck 阈值
# 计算无效（官方 GitHub issue #66830，memoryFlush/preflight 读的是 reserveTokensFloor
# 不是 reserveTokens，不分 provider/模型都会复现），同样显式设置。见 train_with_services.sh
# 同日注释和 docs/issues_log.md 2026-07-15 条目。
echo "确保 compaction.reserveTokensFloor 为 16384..." | tee -a "${LOGS_DIR}/openclaw.log"
openclaw config set agents.defaults.compaction.reserveTokensFloor 16384 \
    >> "${LOGS_DIR}/openclaw.log" 2>&1
echo "[verify] agents.defaults.compaction.reserveTokensFloor = $(openclaw config get agents.defaults.compaction.reserveTokensFloor 2>&1 | tail -1)" \
    | tee -a "${LOGS_DIR}/openclaw.log"

# 部署 rl-training-headers 插件（appendSystemContext 版本）。写入 OpenClaw 自己
# 的系统安装目录（openclaw plugins list --verbose 确认的 source 路径），不是插件
# 扩展开发目录——这个 OpenClaw 版本的插件加载器只扫描这里。
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

# Patch 内置 sglang 扩展，移除 Execution Bias 章节里导致决策犹豫循环的那一行
# （见 scripts/prepare_patched_sglang_execution_bias.sh 顶部完整说明、
# docs/issues_log.md 2026-07-16/17 条目）。第一次运行会自动备份未修改原文件。
echo "生成并部署 sglang execution-bias 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
SGLANG_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/extensions/sglang/index.js"
PATCHED_SGLANG_DIR="${LOGS_DIR}/patched-sglang"
bash "${SCRIPTS_DIR}/prepare_patched_sglang_execution_bias.sh" "${SGLANG_LIVE_FILE}" "${PATCHED_SGLANG_DIR}"
cp "${PATCHED_SGLANG_DIR}/index.js" "${SGLANG_LIVE_FILE}"

# Patch 内置 embedded-agent-runner 的 context overflow 恢复逻辑，理由同
# train_with_services.sh 同名代码块的完整说明。
echo "生成并部署 embedded-agent overflow-recovery 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
EMBEDDED_AGENT_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/embedded-agent-Cv16r2d1.js"
PATCHED_EMBEDDED_AGENT_DIR="${LOGS_DIR}/patched-embedded-agent"
bash "${SCRIPTS_DIR}/prepare_patched_embedded_agent_overflow_recovery.sh" "${EMBEDDED_AGENT_LIVE_FILE}" "${PATCHED_EMBEDDED_AGENT_DIR}"
cp "${PATCHED_EMBEDDED_AGENT_DIR}/embedded-agent-Cv16r2d1.js" "${EMBEDDED_AGENT_LIVE_FILE}"

# Patch 内置 system-prompt 里的 "## Assistant Output Directives" 章节，理由同
# train_with_services.sh 同名代码块的完整说明。
echo "生成并部署 system-prompt output-directives 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
SYSTEM_PROMPT_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/system-prompt-config-CLAPATdy.js"
PATCHED_SYSTEM_PROMPT_DIR="${LOGS_DIR}/patched-system-prompt"
bash "${SCRIPTS_DIR}/prepare_patched_system_prompt_output_directives.sh" "${SYSTEM_PROMPT_LIVE_FILE}" "${PATCHED_SYSTEM_PROMPT_DIR}"
cp "${PATCHED_SYSTEM_PROMPT_DIR}/system-prompt-config-CLAPATdy.js" "${SYSTEM_PROMPT_LIVE_FILE}"

# Patch write/edit 工具选择指引，理由同 train_with_services.sh 同名代码块的
# 完整说明。
echo "生成并部署 write/edit 工具选择指引补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
PATCHED_WRITE_EDIT_DIR="${LOGS_DIR}/patched-write-edit-guidance"
bash "${SCRIPTS_DIR}/prepare_patched_write_edit_guidance.sh" "${SYSTEM_PROMPT_LIVE_FILE}" "${PATCHED_WRITE_EDIT_DIR}"
cp "${PATCHED_WRITE_EDIT_DIR}/system-prompt-config-CLAPATdy.js" "${SYSTEM_PROMPT_LIVE_FILE}"

# Patch 内置 cli-compaction 的 "cli_budget" 预压缩检查，理由同
# train_with_services.sh 同名代码块的完整说明。
echo "生成并部署 cli-compaction 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
CLI_COMPACTION_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/cli-compaction-B6C2IDnn.js"
PATCHED_CLI_COMPACTION_DIR="${LOGS_DIR}/patched-cli-compaction"
bash "${SCRIPTS_DIR}/prepare_patched_cli_compaction.sh" "${CLI_COMPACTION_LIVE_FILE}" "${PATCHED_CLI_COMPACTION_DIR}"
cp "${PATCHED_CLI_COMPACTION_DIR}/cli-compaction-B6C2IDnn.js" "${CLI_COMPACTION_LIVE_FILE}"

# models.providers.sglang 未显式声明 models[] 时 OpenClaw 走自动发现，会用过大的
# 默认值请求 max_completion_tokens，被 sglang 400 拒绝（同 smoke 的问题）。
#
# 2026-07-15 修复：maxTokens 原为 4096（跟 train_with_services.sh 同源的历史
# 遗留值，见该文件同日注释和 docs/issues_log.md 2026-07-15 条目），官方 README.md
# 示例配置 contextWindow=32768 对应 maxTokens=8192，改成跟官方一致。
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
    'name': 'Qwen3-4B-Thinking (RL policy, minitest)',
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
  openclaw gateway run --allow-unconfigured --force \
  >> "${LOGS_DIR}/openclaw.log" 2>&1 &
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
    # 2026-07-14 修复：之前"失败整个重跑"的外层重试（07-10 加的）会让
    # student_chat.py/TA_chat.py/teacher_chat.py 每次重新调用时清空自己的
    # 输出文件重新开始——如果第一次已经做完一部分真实题目才失败，重跑不仅
    # 浪费时间、还会把这部分已经拿到手的真实数据一起清空覆盖掉，比"只跑
    # 一次不重试"保留的数据更少。官方 README 参考流程也是只跑一次，不做
    # 整体重来。改成只跑之前确认一次网关可达（07-10 那次真实问题的修复，
    # 保留），脚本本身只调用一次，失败就是失败，交给脚本自己内部的重试
    # （send_to_openclaw 的几次重试）处理瞬时失败，不再由编排层重来。
    #
    # 2026-07-17 修复：加大 --max-retries 到 8，理由同 train_with_services.sh
    # 同名函数的完整注释（openclaw_combine_select_rollout.py 的正常训练步
    # 暂停机制实测可达 38~115 秒，远超官方默认 7 秒重试预算，见
    # docs/issues_log.md 2026-07-17 条目）。
    wait_for_port "OpenClaw gateway" 18789 120 "${OPENCLAW_PID:-}" "${LOGS_DIR}/openclaw.log" || {
        echo "错误：${name} 模拟开始前网关不可达" >&2
    }
    if OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
       OPENAI_API_KEY="${SIMULATOR_API_KEY}" \
       OPENAI_BASE_URL="${SIMULATOR_BASE_URL}" \
       EXTERNAL_MODEL="${EXTERNAL_MODEL}" \
       OPENCLAW_GATEWAY_URL=http://localhost:18789 \
       python "${OPENCLAW_DIR}/${script}" \
           --dataset "${DATASET}" \
           --num-problems "${num}" \
           --output "${output}" \
           --max-retries 8; then
        return 0
    fi
    echo "警告：${name} 模拟未完全完成，继续训练（数据可能不完整，见 issues_log.md）" >&2
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

# 2026-07-14：Joint 阶段改为跟 train_with_services.sh 一致的做法，见该文件
# 同日注释——官方 student_chat.py/TA_chat.py/teacher_chat.py 是一次性脚本，
# 官方仓库没有"分轮次反复调用"的编排参考，之前的循环设计没有官方依据。
run_joint_phase() {
    echo "--- Joint 阶段开始（${JOINT_NUM_PROBLEMS} 题 × 三角色并发，持续到训练结束）---"

    local joint_s="${LOGS_DIR}/results_student_joint.txt"
    local joint_ta="${LOGS_DIR}/results_TA_joint.txt"
    local joint_t="${LOGS_DIR}/results_teacher_joint.txt"

    run_one_persona "Student" "student_chat.py" "${JOINT_NUM_PROBLEMS}" "${joint_s}" \
        >> "${LOGS_DIR}/sim_student.log" 2>&1 &
    local pid_s=$!

    run_one_persona "TA" "TA_chat.py" "${JOINT_NUM_PROBLEMS}" "${joint_ta}" \
        >> "${LOGS_DIR}/sim_ta.log" 2>&1 &
    local pid_ta=$!

    run_one_persona "Teacher" "teacher_chat.py" "${JOINT_NUM_PROBLEMS}" "${joint_t}" \
        >> "${LOGS_DIR}/sim_teacher.log" 2>&1 &
    local pid_t=$!

    while kill -0 "${TRAINING_PID}" 2>/dev/null; do
        if ! kill -0 "${pid_s}" 2>/dev/null \
            && ! kill -0 "${pid_ta}" 2>/dev/null \
            && ! kill -0 "${pid_t}" 2>/dev/null; then
            break
        fi
        sleep 5
    done
    kill "${pid_s}" "${pid_ta}" "${pid_t}" 2>/dev/null || true
    wait "${pid_s}" "${pid_ta}" "${pid_t}" 2>/dev/null || true

    cat "${joint_s}"  >> "${STUDENT_ALL}" 2>/dev/null || true
    cat "${joint_ta}" >> "${TA_ALL}" 2>/dev/null || true
    cat "${joint_t}"  >> "${TEACHER_ALL}" 2>/dev/null || true

    echo "--- Joint 阶段结束 ---"
}

simulation_loop() {
    run_init_phase 2>&1 | tee -a "${LOGS_DIR}/simulation.log"

    run_joint_phase 2>&1 | tee -a "${LOGS_DIR}/simulation.log"

    echo "模拟循环结束（INIT 1 次 + Joint 阶段 1 次）"
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
