#!/bin/bash
# train_with_services.sh
#
# OpenClaw-RL Hybrid RL 训练编排（Table 3 Phase 1: Joint Hybrid RL）
# 基于 commit 0f4582c5 的 9 GPU 论文布局，Simulator 改为外部 API 服务。
#
# 正式训练前冒烟（3 GPU，非论文配置）请用 scripts/smoke_train_with_services.sh
# 说明见 docs/smoke_test.md
#
# modelfactory job 提交：
#   代码解释器: /bin/bash -i /dfs/data/start_tools.sh && /bin/bash -i
#              （2026-07-13 起：wandb.ai 需要走代理，start_tools.sh 负责
#              起代理，-i 让 pon 这个 alias 能展开；纯 `bash` 提交代理会
#              连不上，wandb 上报不了）
#   代码路径:   /dfs/data/openclaw-rl-project/openclaw-rl/scripts/train_with_services.sh
#   GPU 数量:   8
#
# GPU 分配（论文 faithful，8×H20 全用于训练）：
#   GPU 0-7  → Actor×4 + Rollout×2 + PRM×1 + PRM Teacher×1（megatron 默认）
#   Simulator → 外部机器（不在本 job 占 GPU）
#
# Simulator 地址：编辑 scripts/simulator.env（见 simulator.env.example）
#
# 端口（训练机本机）：
#   30000  → RL training proxy
#   18789  → OpenClaw gateway

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

# 外部 Simulator（必填）
SIMULATOR_BASE_URL=${SIMULATOR_BASE_URL:-}
SIMULATOR_API_KEY=${SIMULATOR_API_KEY:-EMPTY}
EXTERNAL_MODEL=${EXTERNAL_MODEL:-qwen3-32b}

SGLANG_API_KEY=${SGLANG_API_KEY:-openclaw-rl-key}

# 训练占满 8 张 GPU（论文 4+2+1+1）
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

SAVE_CKPT=${SAVE_CKPT:-/dfs/data/openclaw-rl-project/checkpoints/qwen3-4b-openclaw-topk-select}
REPO_ROOT=${REPO_ROOT:-/dfs/data/openclaw-rl-project/OpenClaw-RL-official}
# 2026-07-14：Joint 阶段题目数量。官方 student_chat.py/TA_chat.py/teacher_chat.py
# 本身是一次性脚本（--num-problems 道题跑完就退出，无 --loop/--continuous 参数，
# 见 openclaw-test/student_chat.py:104,238），官方仓库里也找不到"分轮次反复调用"
# 这种编排的参考实现（openclaw-combine/openclaw-opd 均无调用这三个脚本的代码）。
# 之前"每轮 6 题、循环直到训练结束"的设计是我们自己发明的，没有官方依据。改成
# 匹配官方 README "run them together" 的字面做法：Joint 阶段三角色各自传一个
# 足够大的题目数、同时后台启动一次，让训练循环自然消耗真实样本直到结束，不用
# 分批反复调用。GSM8K.json 全量 1319 题，取这个值保证不会在训练结束前提前耗尽。
JOINT_NUM_PROBLEMS=${JOINT_NUM_PROBLEMS:-1319}
DATASET=${DATASET:-${REPO_ROOT}/openclaw-test/GSM8K.json}
SESSION_LIMIT=${SESSION_LIMIT:-72}
CONDA_ENV=${CONDA_ENV:-/dfs/data/envs/openclaw-rl}
CONDA_BASE=${CONDA_BASE:-/dfs/data/miniconda3}

LOGS_DIR=${LOGS_DIR:-/dfs/data/openclaw-rl-project/logs/$(date +%Y%m%d_%H%M%S)}
# 2026-07-22：workspace（homework/homework1/homework2 等运行时文件）从
# ${HOME}/.openclaw/workspace（/root 下，配额小，且 GPU 空闲被平台回收重启后
# 会静默回滚到上次快照，见 issues_log.md）迁到 /dfs/data 下的 runtime/ 目录，
# 跟 logs/ 用同一个时间戳配对，方便按 run 对照查找。
RUNTIME_DIR=${RUNTIME_DIR:-/dfs/data/openclaw-rl-project/runtime/$(basename "${LOGS_DIR}")}
WORKSPACE="${RUNTIME_DIR}/workspace"
OPENCLAW_DIR="${LOGS_DIR}/openclaw-test-patched"

mkdir -p "${LOGS_DIR}" "${WORKSPACE}"
echo "日志目录: ${LOGS_DIR}"
echo "外部 Simulator: ${SIMULATOR_BASE_URL} (model=${EXTERNAL_MODEL})"

# openclaw-test/*.py 硬编码 "model": "default"，当前 OpenClaw CLI（2026.6.9）的
# /v1/chat/completions 只认 openclaw/openclaw-<agentId> 这套 agent-target 格式，
# 会直接 400。生成一份改了这一个字段的补丁副本，官方目录本身不动。
bash "${SCRIPTS_DIR}/prepare_openclaw_test_scripts.sh" "${REPO_ROOT}" "${OPENCLAW_DIR}"

# header/dispatcher 注入均已确认在这个 OpenClaw 版本上结构性失效——OpenClaw 加了
# 一层 SSRF 安全机制，绕开所有外部注入的 fetch/dispatcher，无配置开关（详见
# docs/issues_log.md 2026-07-09 第四/五部分）。改用 appendSystemContext 往
# system prompt 正文里塞 "[RL-TRAINING-META] session_id=... turn_type=..." 标记，
# 服务端解析后在转发给 sglang / 计算训练样本之前清理掉，模型和训练数据都看不到这段
# 标记。已在 smoke/minitest 用真实数据验证过，同步到 8GPU 正式脚本。详见
# scripts/prepare_patched_rl_training_headers.sh 和 scripts/prepare_patched_openclaw_opd.sh
# 顶部注释。
PATCHED_OPD_DIR="${LOGS_DIR}/patched-openclaw-opd"
bash "${SCRIPTS_DIR}/prepare_patched_openclaw_opd.sh" "${REPO_ROOT}" "${PATCHED_OPD_DIR}"

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
  REPO_ROOT="${REPO_ROOT}" \
  NUM_GPUS="${NUM_TRAINING_GPUS}" \
  HF_CKPT="${POLICY_MODEL_PATH}" \
  REF_LOAD="${POLICY_TORCH_DIST}" \
  SAVE_CKPT="${SAVE_CKPT}" \
  PRM_MODEL_PATH="${POLICY_MODEL_PATH}" \
  PRM_TEACHER_LOAD="${POLICY_TORCH_DIST}" \
  SGLANG_API_KEY="${SGLANG_API_KEY}" \
  PATCHED_OPD_DIR="${PATCHED_OPD_DIR}" \
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
# 每次起 gateway 前强制确保这个开关是开着的，不依赖持久化是否跨环境生效（同
# smoke/minitest 的 launch_openclaw_gateway() 逻辑）。
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

# 2026-07-15 修复：上面这个 reserveTokens 从 07-13 设到现在，实际运行时的
# context-overflow precheck 日志（[agent/embedded][context-overflow-precheck]）
# 一直显示 effectiveReserveTokens=20000，跟设置的 16384 对不上——查了官方
# GitHub issue #66830（"reserveTokens vs reserveTokensFloor asymmetry"），
# 确认这是 OpenClaw 自身的已知问题：memoryFlush/preflight 这两条阈值计算路径
# 根本不读 reserveTokens 字段，读的是另一个从未配置过的 reserveTokensFloor
# （不读的话就用它自己的内部默认值，找到的证据显示落在 20000），不分 provider/
# 模型都会复现。同样显式设置 reserveTokensFloor=16384，跟 reserveTokens 保持
# 一致。见 docs/issues_log.md 2026-07-15 条目。
echo "确保 compaction.reserveTokensFloor 为 16384（reserveTokens 单独设置对 precheck 阈值计算无效，见 issues_log.md 2026-07-15）..." \
    | tee -a "${LOGS_DIR}/openclaw.log"
openclaw config set agents.defaults.compaction.reserveTokensFloor 16384 \
    >> "${LOGS_DIR}/openclaw.log" 2>&1
echo "[verify] agents.defaults.compaction.reserveTokensFloor = $(openclaw config get agents.defaults.compaction.reserveTokensFloor 2>&1 | tail -1)" \
    | tee -a "${LOGS_DIR}/openclaw.log"

# 2026-07-22：openclaw.json 里 agents.defaults.workspace 优先级高于
# OPENCLAW_WORKSPACE_DIR 环境变量（agent-scope-config.ts 先查 config 再退回
# 环境变量），光设环境变量不够、之前设置过的值会一直覆盖，每次启动前强制设为
# 本次 run 的 runtime 目录。
echo "确保 agents.defaults.workspace 指向本次 run 的 runtime 目录..." \
    | tee -a "${LOGS_DIR}/openclaw.log"
openclaw config set agents.defaults.workspace "${WORKSPACE}" \
    >> "${LOGS_DIR}/openclaw.log" 2>&1
echo "[verify] agents.defaults.workspace = $(openclaw config get agents.defaults.workspace 2>&1 | tail -1)" \
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

# Patch 内置 embedded-agent-runner 的 context overflow 恢复逻辑，让它在遇到
# "Already compacted"（2026年5月中~6月才加入 OpenClaw，论文提交时不存在）
# 时不再直接放弃、而是像另一条本来就有的优雅路径一样重试原提示词（见
# scripts/prepare_patched_embedded_agent_overflow_recovery.sh 顶部完整说明、
# docs/issues_log.md 2026-07-17/20 条目）。这个 bundle 文件名是内容哈希命名
# 的，OpenClaw 升级后会变化，补丁脚本找不到锚点会明确报错退出，不会静默失败。
echo "生成并部署 embedded-agent overflow-recovery 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
EMBEDDED_AGENT_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/embedded-agent-Cv16r2d1.js"
PATCHED_EMBEDDED_AGENT_DIR="${LOGS_DIR}/patched-embedded-agent"
bash "${SCRIPTS_DIR}/prepare_patched_embedded_agent_overflow_recovery.sh" "${EMBEDDED_AGENT_LIVE_FILE}" "${PATCHED_EMBEDDED_AGENT_DIR}"
cp "${PATCHED_EMBEDDED_AGENT_DIR}/embedded-agent-Cv16r2d1.js" "${EMBEDDED_AGENT_LIVE_FILE}"

# Patch 内置 system-prompt 里的 "## Assistant Output Directives" 章节，给五条
# MEDIA:/audio_as_voice/reply_to_current 指令加一句"仅在适用时才需要遵守"的
# 显式条件说明（论文提交时的 2026.3.8 版本对应章节"## Reply Tags"本就是这种
# 显式条件框架，2026-04 之后加入的这版丢了这层框架）。同一类决策犹豫循环的
# 第三个独立触发源，见 scripts/prepare_patched_system_prompt_output_directives.sh
# 顶部完整说明、docs/issues_log.md 2026-07-20 条目。这个 bundle 文件名是内容
# 哈希命名的，OpenClaw 升级后会变化，补丁脚本找不到锚点会明确报错退出。
echo "生成并部署 system-prompt output-directives 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
SYSTEM_PROMPT_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/system-prompt-config-CLAPATdy.js"
PATCHED_SYSTEM_PROMPT_DIR="${LOGS_DIR}/patched-system-prompt"
bash "${SCRIPTS_DIR}/prepare_patched_system_prompt_output_directives.sh" "${SYSTEM_PROMPT_LIVE_FILE}" "${PATCHED_SYSTEM_PROMPT_DIR}"
cp "${PATCHED_SYSTEM_PROMPT_DIR}/system-prompt-config-CLAPATdy.js" "${SYSTEM_PROMPT_LIVE_FILE}"

# （曾经在这里部署过 write/edit 工具选择指引补丁，2026-07-21 当天已撤销：
# 论文提交时锁定的上游包版本（0.57.1）本来就没有任何 write/edit 工具选择
# 指引，模型当时是在完全没有提示的情况下自己做决定的。往系统提示词里加
# 这句指引虽然确认能改变模型行为（debug 诊断验证过），但这是论文原始环境
# 里从未存在过的额外帮助，不是"恢复论文条件"，会实质影响复现忠实度。改为
# 完全通过训练奖励信号纠正这类错误——模型输入保持跟论文原始环境一样干净，
# 只改训练信号本身。见 docs/issues_log.md 2026-07-21 条目。）

# Patch 内置 cli-compaction 的 "cli_budget" 预压缩检查，让它在遇到
# "Already compacted" 时不再无条件抛错、污染一个本已成功完成的回合。这套
# cli_budget 压缩机制论文提交时完全不存在（2026-03-08 快照里搜不到任何
# "CLI transcript compaction"/"cli_budget" 相关代码，2026-06-09 已存在），
# 命中同一个已知的 "Already compacted" 压缩冷却期报错时会在真实工具调用
# 已经成功执行之后直接抛错，把回合的 HTTP 响应污染成 internal error——
# 客户端据此重试，重试收到的"已经写好了"类回复其实是真话（真实工作已经在
# 报错的那次尝试里完成了），不是模型撒谎。这是训练数据里出现"假完成声明"
# 现象的真正根因，PRM 打分只看对话对象满不满意、不核实真实文件状态，把这种
# 短促的"已完成"话术当正样本训练进去、逐渐过度泛化到真正没做的场景。见
# scripts/prepare_patched_cli_compaction.sh 顶部完整说明、
# docs/issues_log.md 2026-07-21 条目（含真实 debug 级别诊断证据）。这个
# bundle 文件名是内容哈希命名的，OpenClaw 升级后会变化，补丁脚本找不到
# 锚点会明确报错退出。
echo "生成并部署 cli-compaction 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
CLI_COMPACTION_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/cli-compaction-B6C2IDnn.js"
PATCHED_CLI_COMPACTION_DIR="${LOGS_DIR}/patched-cli-compaction"
bash "${SCRIPTS_DIR}/prepare_patched_cli_compaction.sh" "${CLI_COMPACTION_LIVE_FILE}" "${PATCHED_CLI_COMPACTION_DIR}"
cp "${PATCHED_CLI_COMPACTION_DIR}/cli-compaction-B6C2IDnn.js" "${CLI_COMPACTION_LIVE_FILE}"

# Patch 内置的 Silent Reply Policy，强制 resolveSilentReplyPolicyFromPolicies
# 恒定返回 "disallow"。这套策略（src/shared/silent-reply-policy.ts）论文提交时
# 完全不存在（2026-03-08 快照搜不到任何 silent-reply 相关文件，2026-05-11
# 已作为一整套新功能出现），默认策略是 direct=disallow / group=allow /
# internal=allow，而我们训练用的 session key（agent:main:openai-user:...）
# 匹配不到任何已知会话类型，会落到默认的 internal 分类——意外命中"允许静默
# 回复"这个论文原始环境本不存在的新分支。真实训练里观察到模型在这个分支下
# 幻觉出一整套不存在的"silent reply protocol"、反复自我重复退化（run
# 20260721_152519，Problem 24 起）。这是第 5 个确认"论文提交时不存在、之后
# 才加入"的 OpenClaw 行为（前 4 个是 Execution Bias / Assistant Output
# Directives / overflow-recovery / cli-compaction），累计证据支持"论文作者
# 实际用的是 3 月或更早版本"这个判断。见 scripts/prepare_patched_silent_reply_policy.sh
# 顶部完整说明、docs/issues_log.md 2026-07-22 条目（含证据边界说明：这次未做到
# debug 级别实锤，只有"版本缺失+会话分类命中"两条强关联证据）。这个 bundle
# 文件名是内容哈希命名的，OpenClaw 升级后会变化，补丁脚本找不到锚点会明确
# 报错退出。
echo "生成并部署 silent-reply-policy 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
SILENT_REPLY_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/effective-reply-route-BnYlac-J.js"
PATCHED_SILENT_REPLY_DIR="${LOGS_DIR}/patched-silent-reply-policy"
bash "${SCRIPTS_DIR}/prepare_patched_silent_reply_policy.sh" "${SILENT_REPLY_LIVE_FILE}" "${PATCHED_SILENT_REPLY_DIR}"
cp "${PATCHED_SILENT_REPLY_DIR}/effective-reply-route-BnYlac-J.js" "${SILENT_REPLY_LIVE_FILE}"

# models.providers.sglang 未显式声明 models[] 时 OpenClaw 走自动发现，会用过大的
# 默认值请求 max_completion_tokens，被 sglang 400 拒绝（同 smoke/minitest 的问题）。
#
# 2026-07-15 修复：maxTokens 原为 4096，是 smoke 早期 contextWindow 还缩在 8192
# 时（07-07）"maxTokens = contextWindow 一半留 prompt 空间"这个逻辑下的产物；
# 后来 contextWindow 改回官方 32768，maxTokens 没有跟着重新计算，就这样被复制
# 到 minitest/8GPU 正式训练脚本里，形成了 contextWindow=32768 配 maxTokens=4096
# 这个不匹配组合。官方 README.md（Slime-based RL server 配置示例）contextWindow
# =32768 对应 maxTokens=8192，我们这里明显偏小。TA 批改任务比 Student 复杂得多
# （先读文件，再要求生成结构化多点评语，Qwen3-Thinking 还要先输出一大段 <think>
# 推理），系统性地更容易撞到这个偏小的输出预算——8GPU 正式训练（run 8yn4i8ml）
# TA 从 INIT 第 0 题到崩溃前的第 37 题，100% 命中 stopReason=length 截断失败，
# 一次没成功过，Student 同期基本正常，符合这个不对称的解释。改成跟官方一致的
# 8192，见 docs/issues_log.md 2026-07-15 条目。
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
# 第3步：模拟循环（论文 Joint：INIT 顺序建立 + Joint 三角色并行）
# 论文 Appendix A.1：先顺序完成 Student→TA→Teacher 建立 homework1/ homework2/，
# 之后三 Simulator 同时运行（homework1/ homework2/ 固定不清空）。
# =====================================================================
echo ""
echo "=== [3/3] 模拟循环（INIT 顺序 → Joint 三角色并行）==="

STUDENT_ALL="${LOGS_DIR}/results_student_all.txt"
TA_ALL="${LOGS_DIR}/results_TA_all.txt"
TEACHER_ALL="${LOGS_DIR}/results_teacher_all.txt"

# 单角色调用封装（$1=名称 $2=脚本 $3=题数 $4=输出文件）
run_one_persona() {
    local name=$1 script=$2 num=$3 output=$4
    # 2026-07-14 修复：之前"失败整个重跑"的外层重试（07-10 加的）会让
    # student_chat.py/TA_chat.py/teacher_chat.py 每次重新调用时清空自己的
    # 输出文件重新开始——如果第一次已经做完一部分真实题目才失败，重跑不仅
    # 浪费时间、还会把这部分已经拿到手的真实数据一起清空覆盖掉，比"只跑
    # 一次不重试"保留的数据更少。官方 README 参考流程也是只跑一次，不做
    # 整体重来。改成只跑之前确认一次网关可达（09-10 那次真实问题的修复，
    # 保留），脚本本身只调用一次，失败就是失败，交给脚本自己内部的重试
    # （send_to_openclaw 的几次重试）处理瞬时失败，不再由编排层重来。
    #
    # 2026-07-17 修复：官方 --max-retries 默认 3（1s+2s+4s=7 秒总重试预算），
    # 查明我们自己训练代码里 openclaw_combine_select_rollout.py 的
    # pause_submission()/resume_submission() 每一步训练（rollout 攒够样本后
    # 到权重同步完成前）都会主动暂停接受新的 /v1/chat/completions 请求、直接
    # 返回 503（openclaw_opd_api_server.py:344-345），这是正常设计、不是 bug。
    # 平时暂停只有几秒钟，7 秒预算够扛；但这次实测到暂停时长可达 38~115 秒
    # （见 docs/issues_log.md 2026-07-17 条目），远超预算，导致 Student/TA/
    # Teacher 未捕获异常直接崩溃、这个角色接下来几小时都没有任何数据。
    # 加大 --max-retries 到 8（1+2+4+8+16+32+64+128=255 秒总预算，覆盖实测
    # 最长 115 秒暂停约 2.2 倍余量），不改官方重试逻辑本身（退避算法不变），
    # 只是给正常的暂停窗口留够扛住的时间。
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
    echo "警告：${name} 模拟未完全完成，继续训练（数据可能不完整，见 issues_log.md）" >&2
}

# ─────────────────────────────────────────────────────────────────
# INIT 阶段（论文 Appendix A.1：顺序执行 Student → TA → Teacher）
# 处理全量 SESSION_LIMIT 道题，一次性建立 homework1/ homework2/：
#   Student  写解答  → homework/
#   TA       批改    → ensure_homework_dir 复制 homework→homework1/，追加批注
#   Teacher  点评    → ensure_homework_dir 复制 homework1→homework2/，追加点评
# Joint 阶段中 homework1/ homework2/ 不再清空（固定内容来自本阶段）
# ─────────────────────────────────────────────────────────────────
run_init_phase() {
    echo ""
    echo "=== INIT：建立 homework1/ homework2/（SESSION_LIMIT=${SESSION_LIMIT} 题，一次性）==="

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

# ─────────────────────────────────────────────────────────────────
# Joint 阶段（论文 Appendix A.1 原文 "run them together"；官方 student_chat.py/
# TA_chat.py/teacher_chat.py 本身是一次性脚本，跑完 --num-problems 道题就退出，
# 官方仓库没有"分轮次反复调用"的编排代码，2026-07-14 issues_log.md 条目已确认）。
# 三角色各自传 JOINT_NUM_PROBLEMS（覆盖全量数据集）、同时后台启动一次，训练
# 循环持续消耗真实样本直到自己结束（num-rollout 跑完/被手动停止），模拟进程
# 跟着训练进程一起收尾，不用我们自己发明分批次循环。
# homework1/ homework2/ 由 INIT 建立后不再清空；ensure_homework_dir 检测到目录
# 已存在会跳过复制，三角色各自操作独立目录无文件冲突：
#   Student  → homework/（prepare_homework_files 覆写题目文件，不影响 homework1/）
#   TA       → homework1/（追加批注，不影响 homework/）
#   Teacher  → homework2/（追加点评，不影响 homework1/）
# ─────────────────────────────────────────────────────────────────
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

    # 训练进程结束，或者三个模拟进程都已经自己跑完（1319 题全部处理完），
    # 谁先发生就收尾，避免模拟进程在训练已经结束后还傻跑，或者训练还没
    # 结束却干等已经退出的模拟进程。
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
echo "所有服务已启动，训练进行中..."
echo "  日志目录:       ${LOGS_DIR}/"
echo "  外部 Simulator: ${SIMULATOR_BASE_URL}"
echo "  训练日志:       tail -f ${LOGS_DIR}/training.log"
echo "  OpenClaw 日志:  tail -f ${LOGS_DIR}/openclaw.log"
echo "  Ray dashboard:  http://127.0.0.1:8265"

wait "${TRAINING_PID}"
echo "训练完成！检查点: ${SAVE_CKPT}"
