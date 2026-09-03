#!/bin/bash
# run_metaclaw_migration_modelfactory.sh
#
# MetaClaw 迁移训练编排——把已复现校准的 Hybrid RL 方法（GRPO+OPD
# topk-select）应用到 MetaClaw-Bench 上做泛化性实验。详细设计见
# docs/metaclaw_migration_plan.md（目标/方法映射/查证记录/方案 B 设计）。
#
# 跟 train_separate_student.sh 的区别：
#   - 训练起点是干净 base Qwen3-4B（不是 Personal Agent Track 训完的
#     checkpoint），SAVE_CKPT 用独立路径，不覆盖/污染已有 checkpoint
#   - 没有外部 Simulator：MetaClaw-Bench 的"提问方"是静态题目文本 + 确定性
#     checker 脚本，不需要另一个 LLM 扮演 Student/TA/Teacher；PRM 步骤判官
#     用的是训练本身自带的 PRM SGLang 引擎（topk-select 8GPU 拓扑里的
#     PRM SGLang×1），不是外部服务，不需要 wait_for_external_simulator
#   - 用 scripts/metaclaw/metaclaw_rollout_driver.py（day01→day30 严格顺序，
#     concurrency=1）代替 student_chat.py，"跑完就主动停止训练"这条编排逻辑
#     保留（官方 --num-rollout 不会自己停）
#   - 额外需要 MetaClaw-Bench 自己的 openclaw.json（openclaw_cfg/openclaw.json）
#     模板，通过 BENCHMARK_BASE_URL/BENCHMARK_API_KEY/BENCHMARK_MODEL 三个
#     环境变量把它的 model provider 指向本次训练起的代理（30000 端口）——
#     这个配置文件本身不用 openclaw config set 改全局配置，rollout driver
#     内部对每一天都会调用官方 _prepare_work_copy/_patch_agent_workspace
#     生成隔离的工作副本，替换的是那份副本，不是 ~/.openclaw/openclaw.json
#
# 共用不变的部分（跟 train_separate_student.sh 完全一致，直接复用）：
#   GPU 训练启动（run_openclaw_topk_select_modelfactory.sh 扩展了
#   METACLAW_MIGRATION_PROFILE=1 只调 --save-interval，其余参数不变）、
#   RL training proxy 三个补丁脚本（PATCHED_OPD_DIR/PATCHED_COMBINE_DIR/
#   PATCHED_COMBINE_SELECT_DIR，本轮已扩展了 MetaClaw 三路分派逻辑但脚本
#   本身位置不变）、6 个 OpenClaw 系统级版本漂移/行为补丁（rl-training-headers
#   等，全局部署，跟具体训练场景无关，必须部署——否则 session_id 传不到
#   代理，见 metaclaw_migration_plan.md"启动脚本必须复用的现有依赖"一节）。
#
# modelfactory job 提交：
#   代码解释器: /bin/bash -i /dfs/data/start_tools.sh && /bin/bash -i
#   代码路径:   /dfs/data/openclaw-rl-project/OpenClaw-RL/scripts/metaclaw/run_metaclaw_migration_modelfactory.sh
#   GPU 数量:   8（跟 Personal Agent Track 用同一套 topk-select 训练拓扑）
#
# 需要额外准备：MetaClaw-official 检出到 modelfactory（METACLAW_ROOT 指向的路径）
#
# 端口（训练机本机）：
#   30000  → RL training proxy（同时服务 openclaw agent 的模型请求和 driver 的 verdict 提交）
#   18789  → 未使用（MetaClaw 场景每天用独立临时网关端口，见 _start_work_gateway）

set -euo pipefail

SCRIPTS_DIR=$(dirname "$(realpath "$0")")
OPENCLAW_RL_ROOT=$(cd "${SCRIPTS_DIR}/../.." && pwd)
OPENCLAW_RL_GIT_SHA=$(cd "${OPENCLAW_RL_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
export OPENCLAW_RL_GIT_SHA

# 本次提交的统一时间戳，SAVE_CKPT 和 LOGS_DIR 默认值共用同一个，方便按
# 时间戳对上是同一次训练；见下面 SAVE_CKPT 定义处的说明。
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =====================================================================
# 配置
# =====================================================================
# 训练起点：干净 base Qwen3-4B（已确认，2026-08-17，见
# docs/metaclaw_migration_plan.md"训练起点"一节）——不接 Personal Agent
# Track 训完的 checkpoint，避免"方法本身行不行"和"旧 checkpoint 是否已
# 定型"两个因素纠缠在一起。
POLICY_MODEL_PATH=${POLICY_MODEL_PATH:-/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507}
POLICY_TORCH_DIST=${POLICY_TORCH_DIST:-/dfs/data/models/Qwen3-4B-Thinking-2507-torch-dist}

SGLANG_API_KEY=${SGLANG_API_KEY:-openclaw-rl-key}

NUM_TRAINING_GPUS=${NUM_TRAINING_GPUS:-8}
TRAINING_CUDA_DEVICES=${TRAINING_CUDA_DEVICES:-$(seq -s, 0 $((NUM_TRAINING_GPUS - 1)))}

# OPENCLAW_GATEWAY_TOKEN（跟 train_separate_student.sh 同一读取逻辑，仅用于
# 系统级 `openclaw config`/`openclaw plugins` 命令的鉴权，不是 MetaClaw
# 每天临时网关的 token）
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

# 独立 checkpoint 路径——不跟 Personal Agent Track 的 checkpoint 混用。
#
# 2026-08-19 修复（真实发现，不是预防性改动）：CLI 核对 run_openclaw_
# topk_select_modelfactory.sh 发现 --load "${SAVE_CKPT}" 是无条件加的
# （Megatron 在这个目录不存在/没有 latest_checkpointed_iteration.txt 时
# 才会回退到 --ref-load 用干净预训练权重，见该脚本顶部注释）——这跟"按
# 天续跑"（METACLAW_PROGRESS_DIR/METACLAW_RESUME）完全是两回事、两个
# 独立开关：不设 METACLAW_RESUME 只保证不会跳过已经打过分的天，**不保证
# 训练权重是干净的**。之前这里用固定路径（不带时间戳）作为默认值，
# 意味着只要上一次训练在这个目录存过 checkpoint（哪怕是训坏的），下一次
# 提交训练——即使完全不碰任何续跑相关的环境变量——也会静默地把那份
# 权重通过 --load 加载进来继续训，不是从干净的 base Qwen3-4B 开始，跟
# 用户"实验阶段每次都要重新开始跑"这个要求直接冲突。
#
# 改成默认按时间戳生成唯一路径（跟 LOGS_DIR 共用同一个 RUN_TIMESTAMP，
# 方便按时间戳对应同一次训练）：每次不显式设置 SAVE_CKPT 提交训练，
# 目录天然不存在，--load 自动回退到 --ref-load 干净权重，不需要手动
# 删除旧目录或记住换路径。旧 checkpoint 不会被清掉，各自留在自己的
# 时间戳目录下，需要的话仍可以手动 SAVE_CKPT=<旧路径> 显式接着训。
SAVE_CKPT=${SAVE_CKPT:-/dfs/data/openclaw-rl-project/checkpoints/qwen3-4b-openclaw-metaclaw-migration_${RUN_TIMESTAMP}}
REPO_ROOT=${REPO_ROOT:-/dfs/data/openclaw-rl-project/OpenClaw-RL-official}

# MetaClaw-official 检出路径（需要额外准备，不是 openclaw-rl 仓库自带的）
METACLAW_ROOT=${METACLAW_ROOT:-/dfs/data/openclaw-rl-project/MetaClaw-official}
if [ ! -d "${METACLAW_ROOT}" ]; then
    echo "错误：METACLAW_ROOT 不存在: ${METACLAW_ROOT}（需要先 git clone MetaClaw-official 到这个路径）" >&2
    exit 1
fi
METACLAW_ALL_TESTS_JSON=${METACLAW_ALL_TESTS_JSON:-${METACLAW_ROOT}/benchmark/data/metaclaw-bench/all_tests.json}
BENCHMARK_MODEL=${BENCHMARK_MODEL:-qwen3-4b}
BENCHMARK_WORKSPACE_DIR=${BENCHMARK_WORKSPACE_DIR:-${METACLAW_ROOT}/benchmark/data/metaclaw-bench/workspaces/shared}

# 断点续跑（2026-08-18 决定，取代 08-17"不做断点续跑"——设计改成跟
# MetaClaw 官方 rl_run.py 一样边训练边算分之后，Table 1 式的 Acc./Compl.
# 数字变成了这一趟运行本身的实时累积结果，"崩溃后从 day01 重跑"会导致
# 已经算过分的天被用不同（更新过的）权重重新生成一次答案，污染最终聚合
# 分数——不再只是浪费算力，必须要有断点续跑。训练侧 checkpoint 本来就有
# --load 自动续训（见 run_openclaw_topk_select_modelfactory.sh），这里
# 只需要让"这天已经打过分"这件事本身也能跨进程重启保留下来。
#
# 有意分成两个独立开关（2026-08-18，用户明确要求不能做成自动续跑）：
#   METACLAW_PROGRESS_DIR：设了就会把每天的分数落盘，纯记录，不改变这次
#     跑的行为——正常训练也建议一直设着，这样万一真的崩溃了，之后才有
#     数据可以续跑；不设就跟以前一样完全不记录。
#   METACLAW_RESUME=1：唯一真正触发"跳过已完成的天"的开关，必须手动
#     显式设置，且必须搭配同一个 METACLAW_PROGRESS_DIR 目录才生效。
#     正常训练默认不设（=0），永远从 day01 完整跑——不会因为凑巧复用了
#     一个已经有旧文件的目录就意外跳过某些天。只有确认要续跑一次真实的
#     崩溃时，才手动加上 METACLAW_RESUME=1 重新提交这个脚本。
# 注意这两个只管"按天续跑"，report.json/report.md 落盘是另一个独立开关
# METACLAW_REPORT_DIR（见下面 LOGS_DIR 定义之后），不设 METACLAW_PROGRESS_DIR
# 也照样会有 report 文件，不用为了拿到报告文件被迫开启续跑功能。
# 详见 docs/metaclaw_migration_plan.md。
METACLAW_PROGRESS_DIR=${METACLAW_PROGRESS_DIR:-}
METACLAW_RESUME=${METACLAW_RESUME:-0}

# 可选的鲁棒性开关，默认 0（不重试，跟 MetaClaw 官方 infer_cmd.py 的
# retry=0 默认值一致，见 docs/metaclaw_migration_plan.md 查证记录四）。
# 要不要开、开到多大，等真实训练观察效果后再决定，不预设。
METACLAW_AGENT_RETRY=${METACLAW_AGENT_RETRY:-0}
METACLAW_VERDICT_RETRY=${METACLAW_VERDICT_RETRY:-0}


# 训练前冒烟测试用：只跑前 N 天（默认空 = 跑全部 30 天）。第一次跑强烈
# 建议先设 METACLAW_MAX_DAYS=1，确认整条链路（真实 openclaw agent 子
# 进程、checker 执行、verdict 被代理正确识别、session_id 正确传递）
# 走通了，再不设这个变量、正式跑全部 30 天。见 docs/metaclaw_migration_plan.md
# 训练前清单。
METACLAW_MAX_DAYS=${METACLAW_MAX_DAYS:-}

CONDA_ENV=${CONDA_ENV:-/dfs/data/envs/openclaw-rl}
CONDA_BASE=${CONDA_BASE:-/dfs/data/miniconda3}

LOGS_DIR=${LOGS_DIR:-/dfs/data/openclaw-rl-project/logs/metaclaw_migration_${RUN_TIMESTAMP}}
mkdir -p "${LOGS_DIR}"

# report.json/report.md 落盘目录，跟上面断点续跑的 METACLAW_PROGRESS_DIR
# 是两回事——这个是这次跑的实际交付结果，不应该要求用户手动设置才能拿到
# 文件（否则默认情况下分数只会 print 进 metaclaw_rollout.log，没有独立
# 文件，真实训练里发现的问题）。这里给一个总是有值的默认路径
# （LOGS_DIR 下面的 report/ 子目录，每次跑都是新的时间戳目录，不会跟别的
# 跑混在一起），除非显式指定 METACLAW_PROGRESS_DIR，report 也会跟着落到
# 那个目录（driver 内部逻辑：METACLAW_REPORT_DIR 优先，没设就退回
# METACLAW_PROGRESS_DIR）。
METACLAW_REPORT_DIR=${METACLAW_REPORT_DIR:-${LOGS_DIR}/report}

cat > "${LOGS_DIR}/RUN_MANIFEST.txt" <<EOF
openclaw-rl commit: ${OPENCLAW_RL_GIT_SHA}
started: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
command: $0 $*
metaclaw_root: ${METACLAW_ROOT}
checkpoint start: base ${POLICY_TORCH_DIST}（非 Personal Agent Track checkpoint）
sample_unit: per-turn samples, one group per round (1/N advantage scaling)
EOF

echo "日志目录: ${LOGS_DIR}"
echo "METACLAW_ROOT: ${METACLAW_ROOT}"
echo "SAVE_CKPT（独立于 Personal Agent Track）: ${SAVE_CKPT}"
echo "样本单位: 每 turn 一个样本，一个 round 一个 group（advantage 按 1/N 缩放）"

# =====================================================================
# 生成三个补丁代理目录（脚本本身跟 Personal Agent Track 共用，本轮已
# 扩展了 MetaClaw 三路分派逻辑——verdict / 步骤判官 / 原逻辑不变，见
# docs/metaclaw_migration_plan.md"已实现"一节）
# =====================================================================
PATCHED_OPD_DIR="${LOGS_DIR}/patched-openclaw-opd"
bash "${SCRIPTS_DIR}/../prepare_patched_openclaw_opd.sh" "${REPO_ROOT}" "${PATCHED_OPD_DIR}"

PATCHED_COMBINE_SELECT_DIR="${LOGS_DIR}/patched-openclaw-combine-select"
bash "${SCRIPTS_DIR}/../prepare_patched_openclaw_combine_select.sh" "${REPO_ROOT}" "${PATCHED_COMBINE_SELECT_DIR}"

PATCHED_COMBINE_DIR="${LOGS_DIR}/patched-openclaw-combine"
bash "${SCRIPTS_DIR}/../prepare_patched_openclaw_combine.sh" "${REPO_ROOT}" "${PATCHED_COMBINE_DIR}"

# =====================================================================
# conda
# =====================================================================
if [ -n "${CONDA_ENV}" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    echo "已激活 conda: ${CONDA_ENV}"
fi

# =====================================================================
# 工具函数（同 train_separate_student.sh）
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

# =====================================================================
# 清理
# =====================================================================
DRIVER_PID=""

cleanup() {
    echo ""
    echo "清理中..."
    [ -n "${DRIVER_PID}" ] && kill "${DRIVER_PID}" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    wait 2>/dev/null || true
    echo "清理完成。"
}
trap cleanup EXIT INT TERM

# =====================================================================
# 第1步：启动训练（GPU 0-7，跟 Personal Agent Track 用同一套 topk-select
# 训练后端，未修改——PRM SGLang 引擎本身就在这套拓扑里，MetaClaw 步骤
# 判官直接复用，不需要额外的外部服务）
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
  PATCHED_COMBINE_DIR="${PATCHED_COMBINE_DIR}" \
  PATCHED_COMBINE_SELECT_DIR="${PATCHED_COMBINE_SELECT_DIR}" \
  OPENCLAW_RL_GIT_SHA="${OPENCLAW_RL_GIT_SHA}" \
  METACLAW_MIGRATION_PROFILE="1" \
  bash "${SCRIPTS_DIR}/../run_openclaw_topk_select_modelfactory.sh" \
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
# 第2步：RL proxy 就绪 + OpenClaw 系统级补丁部署（rl-training-headers 等
# 六项，必须做——见 docs/metaclaw_migration_plan.md"启动脚本必须复用的
# 现有依赖"一节，不做的话 session_id 传不到代理，MetaClaw 三路分派全部
# 静默失效）
# =====================================================================
echo ""
echo "=== [2/3] RL training proxy + OpenClaw 系统级补丁 ==="

echo "等待 RL training proxy (port 30000)..."
wait_for_port "RL training proxy" 30000 900 "" "${LOGS_DIR}/training.log"

echo "部署 rl-training-headers 插件（appendSystemContext 版本，session_id 传递依赖此项）..." \
    | tee -a "${LOGS_DIR}/openclaw.log"
PATCHED_PLUGIN_DIR="${LOGS_DIR}/patched-rl-training-headers"
bash "${SCRIPTS_DIR}/../prepare_patched_rl_training_headers.sh" "${REPO_ROOT}" "${PATCHED_PLUGIN_DIR}"
SYSTEM_PLUGIN_DIR="/usr/lib/node_modules/openclaw/dist/extensions/rl-training-headers"
mkdir -p "${SYSTEM_PLUGIN_DIR}"
cp "${PATCHED_PLUGIN_DIR}/index.js" "${SYSTEM_PLUGIN_DIR}/index.js"
cp "${PATCHED_PLUGIN_DIR}/openclaw.plugin.json" "${SYSTEM_PLUGIN_DIR}/openclaw.plugin.json"
cp "${PATCHED_PLUGIN_DIR}/package.json" "${SYSTEM_PLUGIN_DIR}/package.json"
openclaw plugins enable rl-training-headers >> "${LOGS_DIR}/openclaw.log" 2>&1 || true

echo "部署 sglang execution-bias 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
SGLANG_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/extensions/sglang/index.js"
PATCHED_SGLANG_DIR="${LOGS_DIR}/patched-sglang"
bash "${SCRIPTS_DIR}/../prepare_patched_sglang_execution_bias.sh" "${SGLANG_LIVE_FILE}" "${PATCHED_SGLANG_DIR}"
cp "${PATCHED_SGLANG_DIR}/index.js" "${SGLANG_LIVE_FILE}"

echo "部署 embedded-agent overflow-recovery 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
EMBEDDED_AGENT_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/embedded-agent-Cv16r2d1.js"
PATCHED_EMBEDDED_AGENT_DIR="${LOGS_DIR}/patched-embedded-agent"
bash "${SCRIPTS_DIR}/../prepare_patched_embedded_agent_overflow_recovery.sh" "${EMBEDDED_AGENT_LIVE_FILE}" "${PATCHED_EMBEDDED_AGENT_DIR}"
cp "${PATCHED_EMBEDDED_AGENT_DIR}/embedded-agent-Cv16r2d1.js" "${EMBEDDED_AGENT_LIVE_FILE}"

echo "部署 system-prompt output-directives 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
SYSTEM_PROMPT_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/system-prompt-config-CLAPATdy.js"
PATCHED_SYSTEM_PROMPT_DIR="${LOGS_DIR}/patched-system-prompt"
bash "${SCRIPTS_DIR}/../prepare_patched_system_prompt_output_directives.sh" "${SYSTEM_PROMPT_LIVE_FILE}" "${PATCHED_SYSTEM_PROMPT_DIR}"
cp "${PATCHED_SYSTEM_PROMPT_DIR}/system-prompt-config-CLAPATdy.js" "${SYSTEM_PROMPT_LIVE_FILE}"

echo "部署 cli-compaction 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
CLI_COMPACTION_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/cli-compaction-B6C2IDnn.js"
PATCHED_CLI_COMPACTION_DIR="${LOGS_DIR}/patched-cli-compaction"
bash "${SCRIPTS_DIR}/../prepare_patched_cli_compaction.sh" "${CLI_COMPACTION_LIVE_FILE}" "${PATCHED_CLI_COMPACTION_DIR}"
cp "${PATCHED_CLI_COMPACTION_DIR}/cli-compaction-B6C2IDnn.js" "${CLI_COMPACTION_LIVE_FILE}"

echo "部署 silent-reply-policy 补丁..." | tee -a "${LOGS_DIR}/openclaw.log"
SILENT_REPLY_LIVE_FILE="/usr/lib/node_modules/openclaw/dist/effective-reply-route-BnYlac-J.js"
PATCHED_SILENT_REPLY_DIR="${LOGS_DIR}/patched-silent-reply-policy"
bash "${SCRIPTS_DIR}/../prepare_patched_silent_reply_policy.sh" "${SILENT_REPLY_LIVE_FILE}" "${PATCHED_SILENT_REPLY_DIR}"
cp "${PATCHED_SILENT_REPLY_DIR}/effective-reply-route-BnYlac-J.js" "${SILENT_REPLY_LIVE_FILE}"

# =====================================================================
# 第3步：MetaClaw rollout driver（day01→day30，concurrency=1，见
# scripts/metaclaw/metaclaw_rollout_driver.py 顶部文档字符串）
#
# BENCHMARK_BASE_URL/BENCHMARK_API_KEY/BENCHMARK_MODEL 是
# openclaw_cfg/openclaw.json 里 ${...} 占位符对应的环境变量——driver
# 内部对每一天都会先用官方 _prepare_work_copy 生成隔离配置副本，这里设的
# 环境变量在副本被 OpenClaw 自己加载时完成替换，不需要手工改 JSON 文件。
# =====================================================================
echo ""
echo "=== [3/3] MetaClaw rollout driver（day01→day30）==="

# BENCHMARK_BASE_URL 没有 trailing /chat/completions，假设 OpenClaw 的
# "openai-completions" provider 客户端会自己拼接标准 OpenAI 路径（同
# models.providers.sglang 在其他训练脚本里的用法）——这条假设未在真实
# MetaClaw agent 请求链路上验证过，如果联调时发现 404，先查这里。
METACLAW_ALL_TESTS_JSON="${METACLAW_ALL_TESTS_JSON}" \
  METACLAW_ROOT="${METACLAW_ROOT}" \
  METACLAW_COMBINE_PROXY_URL="http://127.0.0.1:30000/v1/chat/completions" \
  METACLAW_MODEL_ID="${BENCHMARK_MODEL}" \
  METACLAW_MAX_DAYS="${METACLAW_MAX_DAYS}" \
  METACLAW_AGENT_RETRY="${METACLAW_AGENT_RETRY}" \
  METACLAW_VERDICT_RETRY="${METACLAW_VERDICT_RETRY}" \
  METACLAW_PROGRESS_DIR="${METACLAW_PROGRESS_DIR}" \
  METACLAW_REPORT_DIR="${METACLAW_REPORT_DIR}" \
  METACLAW_RESUME="${METACLAW_RESUME}" \
  BENCHMARK_BASE_URL="http://127.0.0.1:30000/v1" \
  BENCHMARK_API_KEY="${SGLANG_API_KEY}" \
  BENCHMARK_MODEL="${BENCHMARK_MODEL}" \
  BENCHMARK_WORKSPACE_DIR="${BENCHMARK_WORKSPACE_DIR}" \
  SGLANG_API_KEY="${SGLANG_API_KEY}" \
  python "${SCRIPTS_DIR}/metaclaw_rollout_driver.py" \
  > >(tee -a "${LOGS_DIR}/metaclaw_rollout.log") 2>&1 &
DRIVER_PID=$!

echo ""
echo "所有服务已启动，训练进行中..."
echo "  日志目录:         ${LOGS_DIR}/"
echo "  METACLAW_ROOT:    ${METACLAW_ROOT}"
echo "  训练日志:         tail -f ${LOGS_DIR}/training.log"
echo "  rollout driver:   ${LOGS_DIR}/metaclaw_rollout.log（同时也直接打在本终端/job 输出里，跟 Personal Agent Track 的 simulation.log 一样用 tee，不用单独开一个终端 tail -f 才能看）"
echo "  Ray dashboard:    http://127.0.0.1:8265"

wait "${DRIVER_PID}" 2>/dev/null || true
echo "day01→day30 全部跑完，主动停止训练..." | tee -a "${LOGS_DIR}/metaclaw_rollout.log"
kill "${TRAINING_PID}" 2>/dev/null || true
ray stop --force 2>/dev/null || true

echo "训练完成（MetaClaw 迁移，day01→day30 跑完后主动停止）！检查点: ${SAVE_CKPT}"
