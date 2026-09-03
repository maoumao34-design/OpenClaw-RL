#!/usr/bin/env bash
# Patch OpenClaw-RL-official topk-select launcher for modelfactory Ray jobs:
#   - Fix REPO_ROOT when patched script lives outside openclaw-combine/
#   - Ray job uses SLIME_ROOT/train_async.py (not /workspace/train_async.py)
#   - Skip aggressive pkill python
#
# This wraps run_qwen3_4b_openclaw_topk_select.sh — the paper's main Hybrid RL
# method (Table 3 avg 10.3, confirmed by Table 5 k=4) with k=4, m=3,
# seq-optimal hint selection and Megatron PRM Teacher.
# 8 GPU layout: Actor×4 (TP=4) + Rollout×2 + PRM SGLang×1 + PRM Teacher×1.
#
# SMOKE_PROFILE=1    applies 4-GPU smoke sed overrides (see smoke_run_qwen3_4b_openclaw_topk_select.sh).
# MINITEST_PROFILE=1 applies 5-GPU pre-test sed overrides (see minitest_run_qwen3_4b_openclaw_topk_select.sh).
# METACLAW_MIGRATION_PROFILE=1 lowers --save-interval only, for the MetaClaw
# migration's much smaller per-pass data volume (see scripts/metaclaw/, and
# docs/metaclaw_migration_plan.md for the sizing rationale).

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/dfs/data/openclaw-rl-project/OpenClaw-RL-official}
OFFICIAL="${REPO_ROOT}/openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh"
PATCHED="${OPENCLAW_TOPK_SELECT_SCRIPT:-${TMPDIR:-/tmp}/run_qwen3_4b_openclaw_topk_select_modelfactory.sh}"
SMOKE_PROFILE=${SMOKE_PROFILE:-0}
MINITEST_PROFILE=${MINITEST_PROFILE:-0}
METACLAW_MIGRATION_PROFILE=${METACLAW_MIGRATION_PROFILE:-0}

if [ ! -f "${OFFICIAL}" ]; then
    echo "错误：找不到官方 topk-select 脚本: ${OFFICIAL}" >&2
    exit 1
fi

cp "${OFFICIAL}" "${PATCHED}"

# 断点续训：Megatron 在 --load 无效（不存在 / 没有 latest_checkpointed_iteration.txt）时
# 会自动回退到 --ref-load 从预训练权重重新开始（args.finetune=True, start_rollout_id=0，
# 见 slime/utils/arguments.py:1814-1831）。官方脚本从不传 --load，导致任何一次重启
# 都会丢弃已训练进度。这里让 --load 指向 SAVE_CKPT：首次运行时该目录还不存在，会
# 自动回退到 ref_load（行为不变）；一旦存过 checkpoint，重启即可自动续训。
sed -i \
    -e 's/--save "${SAVE_CKPT}"/--save "${SAVE_CKPT}"\n   --load "${SAVE_CKPT}"/' \
    "${PATCHED}"

if [ "${SMOKE_PROFILE}" = "1" ]; then
    # topk-select 已经默认 PRM_GPUS=1 PRM_NUM_GPUS_PER_ENGINE=1，无需 sed
    # 额外缩配：PRM_M 3→1、OPENCLAW_TOPK_MAX_CAND 3→1（smoke 只需验证流通）
    #
    # sglang/rollout context 不再缩到 8192：2026-07-08 smoke 实测 8192 下
    # student/TA/teacher 全部 100% context overflow（0/1 完成），training.log
    # 里唯一能跑通、被当成 MAIN 样本提交的是 OpenClaw 内部的 275-token
    # "context summarization" 兜底调用，session_id 也因此全是 unknown——
    # 不是 header workaround 的 bug，是 smoke 本身没有一次真实对话跑得完，
    # 无法验证 session_id 解析逻辑。改回官方值 32768（与 minitest/8GPU 一致，
    # 论文原作者验证过对真实 agent 对话够用）。PRM_MAX_NEW_TOKENS 保留官方
    # 默认 8192（当初缩到 4096 是为了避开 8192 context 下的 400，context
    # 改回 32768 后不再需要）。
    #
    # --max-tokens-per-gpu 仍保留 8192（TP=1 单卡训练侧的显存预算，跟
    # sglang context 无关，是否需要一起调大待用真实 run 的报错验证，
    # 不提前假设）。
    sed -i \
        -e 's/--tensor-model-parallel-size 4/--tensor-model-parallel-size 1/' \
        -e 's/--rollout-num-gpus-per-engine 2/--rollout-num-gpus-per-engine 1/' \
        -e 's/--rollout-batch-size 16/--rollout-batch-size 4/' \
        -e 's/--max-tokens-per-gpu 32768/--max-tokens-per-gpu 8192/' \
        -e 's/^export TP="2"/export TP="1"/' \
        -e 's/export PRM_M="${PRM_M:-3}"/export PRM_M="${PRM_M:-1}"/' \
        -e 's/export OPENCLAW_TOPK_MAX_CAND="${OPENCLAW_TOPK_MAX_CAND:-3}"/export OPENCLAW_TOPK_MAX_CAND="${OPENCLAW_TOPK_MAX_CAND:-1}"/' \
        "${PATCHED}"
elif [ "${MINITEST_PROFILE}" = "1" ]; then
    # 5-GPU pre-test：仅减少并行度和数据量，其他与 8GPU 正式完全一致。
    # Actor TP 4→2, Rollout 2→1, SGLang TP "2"→"1"（均为并行度调整，不影响流程）
    # num-rollout 100000000→300（~18 训练步，验证流水线，不做完整训练）
    # 不变：context=32768, batch=16, m=3, k=4, sequence_optimal
    sed -i \
        -e 's/--tensor-model-parallel-size 4/--tensor-model-parallel-size 2/' \
        -e 's/--rollout-num-gpus-per-engine 2/--rollout-num-gpus-per-engine 1/' \
        -e 's/^export TP="2"/export TP="1"/' \
        -e 's/--num-rollout 100000000/--num-rollout 300/' \
        -e 's/--save-interval 100/--save-interval 5/' \
        "${PATCHED}"
elif [ "${METACLAW_MIGRATION_PROFILE}" = "1" ]; then
    # 2026-08-17：MetaClaw 迁移一遍 30 天数据量远小于 Personal Agent Track
    # 一次训练的量级，官方默认 --save-interval 100 大概率整遍跑完都不到
    # 100 步、一个 checkpoint 都存不下来。粗略估算：30 天 × 约 10
    # round/天 = 约 300 round，每个 round 大约 1 条最终轮次样本 + 数条
    # 中间轮次步骤判官样本（保守估计均值约 3 条/round）≈ 900 条样本；
    # --rollout-batch-size 16（官方默认，未改动）下，约 900/16 ≈ 56
    # 训练步——这个换算比例参照上面 MINITEST_PROFILE 注释"num-rollout
    # 300 → 约 18 训练步"（300/18≈16.7，与 batch-size=16 基本吻合）。
    # 目标一遍存下约 5 个 checkpoint，56/5≈11.2，向下取整到 10（宁可多存
    # 几个也不要少于 5 个，估算本身不准，实际数字要等真实训练跑一次才知
    # 道，见 docs/metaclaw_migration_plan.md）。只改这一处，并行度/
    # batch-size/上下文窗口等其余参数不动，跟 8GPU 正式训练配置一致。
    # 2026-08-18：真实训练 metaclaw_migration_20260818_* 在 day01 就大面积
    # context overflow——openclaw agent 请求 16661 输入 + 30313 max_tokens =
    # 46974 token，超过官方默认 32768，代理转 500，driver 记
    # agent_succeeded=False，连续失败触发 router 熔断，后续全 503，全程零
    # 训练样本。根因：官方默认 32768 是 Personal Agent Track（GSM8K 风格，
    # 短对话）专属的调优值（对比 OpenClaw-RL 自己的 toolcall-rl 4B 脚本用
    # 的还更小，--rollout-max-context-len 16384，说明 32768 在 OpenClaw-RL
    # 自己的赛道里已经算大的），MetaClaw 的系统提示词（工具 schema + skills
    # + memory + 当天任务文件）比这重得多——MetaClaw 官方 openclaw_cfg/
    # openclaw.json 自己就声明 "contextWindow": 50000, "maxTokens": 50000，
    # 不是我们瞎猜的数字。改到 65536（用户决定，训练过程中还有额外累积的
    # 会话内容，比 50000 再多留一点余量）。CONTEXT_LENGTH（喂给 sglang
    # 启动本身）、--rollout-max-context-len（slime 侧 rollout 有效性判断）、
    # --sglang-context-length（sglang rollout 引擎实际配置）三处是同一个
    # "这个引擎能吃多少 token" 概念的三处体现，必须一起改，改漏一处就会
    # 出现"sglang 能接但 slime 认为超限"或反过来的新不一致。
    # --max-tokens-per-gpu 32768 不动——这是训练侧单 GPU 显存预算，跟
    # sglang context 是否溢出无关（同上面 SMOKE_PROFILE 分支的既有结论：
    # 是否需要一起调大待真实训练报错验证，不提前假设）。
    #
    # --rollout-batch-size 16 -> 8 + --use-dynamic-global-batch-size
    #（2026-09-03，按题成组）：代理侧现在把一个 round 的全部 turn 样本
    # 用同一个 group_index 一次性入队，而 _drain_output_queue 数的是**组**
    #（openclaw_combine_select_rollout.py:110 的 target_data_size =
    # args.rollout_batch_size），所以 8 就是**8 道完整的题**。
    # 再开 --use-dynamic-global-batch-size：slime 不再用
    # rollout_batch_size × n_samples // num_steps 这个公式，而是从**实际收到
    # 的样本数**反推 global_batch_size（rollout.py:287，没有 dynamic_history
    # 时 desired_steps=1），于是「收满 8 道题的全部 turn → 训练一次」。
    # 两个必须一起改：只改 batch-size 不开 dynamic，公式会把 gbs 定成 8，
    # 而一批实际有约 35 个样本（健康段 4.44 turn/题），会被切成 4 步并且
    # 把一道题劈到不同 step 里，按题成组的意义就没了。
    #
    # 为什么是 8 不是 4：346 道题 ÷ 8 = 43 次更新（day22 那次约 96 次），
    # 每次吃约 35 个样本而不是 16 个——总数据量一样，是「43 次大步」对
    # 「96 次小步」。目标是「跑满 30 天且不训坏」，少而大的步更稳。取 4 能
    # 追平到 86 次，但全负批概率从 0.83^8≈23% 涨到 0.83^4≈47%，不划算。
    sed -i \
        -e 's/--save-interval 100/--save-interval 10/' \
        -e 's/export CONTEXT_LENGTH="32768"/export CONTEXT_LENGTH="65536"/' \
        -e 's/--rollout-max-context-len 32768/--rollout-max-context-len 65536/' \
        -e 's/--sglang-context-length 32768/--sglang-context-length 65536/' \
        -e 's/--rollout-batch-size 16/--rollout-batch-size 8/' \
        "${PATCHED}"
    if ! grep -q -- "--rollout-batch-size 8" "${PATCHED}"; then
        echo "错误：--rollout-batch-size 未能改成 8（官方脚本可能已改动）" >&2
        exit 1
    fi
    # 幂等注入：按题成组必须配 dynamic gbs 和 1/N advantage 缩放钩子，
    # 三者少一个这套设计就不成立，所以放在同一个分支里、各自带失败即报错。
    if ! grep -q -- "--use-dynamic-global-batch-size" "${PATCHED}"; then
        sed -i -e 's|--rollout-batch-size 8|--rollout-batch-size 8\n   --use-dynamic-global-batch-size|' "${PATCHED}"
    fi
    if ! grep -q -- "--use-dynamic-global-batch-size" "${PATCHED}"; then
        echo "错误：--use-dynamic-global-batch-size 注入失败" >&2
        exit 1
    fi
    if ! grep -q -- "--custom-reward-post-process-path" "${PATCHED}"; then
        sed -i -e 's|--disable-rewards-normalization|--disable-rewards-normalization\n   --custom-reward-post-process-path metaclaw_round_scale.metaclaw_round_scale|' "${PATCHED}"
    fi
    if ! grep -q -- "--custom-reward-post-process-path" "${PATCHED}"; then
        echo "错误：--custom-reward-post-process-path 注入失败" >&2
        exit 1
    fi
fi

python3 - "${PATCHED}" "${REPO_ROOT}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
repo_root = sys.argv[2]
text = path.read_text()
text = text.replace(
    "pkill -9 sglang\nsleep 3\nray stop --force\npkill -9 ray\npkill -9 python\nsleep 3\npkill -9 ray\npkill -9 python",
    'echo "[modelfactory] skip aggressive pkill; stopping Ray only"\nray stop --force 2>/dev/null || true',
    1,
)
text = text.replace(
    'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"\nREPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"',
    f'SCRIPT_DIR="{repo_root}/openclaw-combine"\nREPO_ROOT="{repo_root}"',
    1,
)
old_ray = (
    'ray job submit --address="http://127.0.0.1:8265" \\\n'
    '   --runtime-env-json="${RUNTIME_ENV_JSON}" \\\n'
    '   -- python3 train_async.py \\'
)
new_ray = (
    'ray job submit --address="http://127.0.0.1:8265" \\\n'
    '   --working-dir="${SLIME_ROOT}" \\\n'
    '   --runtime-env-json="${RUNTIME_ENV_JSON}" \\\n'
    '   -- python3 "${SLIME_ROOT}/train_async.py" \\'
)
if old_ray not in text:
    raise SystemExit("patch failed: ray job submit block not found in topk-select launcher")
text = text.replace(old_ray, new_ray, 1)

# PATCHED_OPD_DIR (optional, set by caller): prepend a directory containing a
# patched openclaw_opd_api_server.py ahead of the official openclaw-opd/ so
# `import openclaw_opd_api_server` resolves to the patched copy. See
# scripts/prepare_patched_openclaw_opd.sh for why (rl-training-headers
# X-Session-Id fallback). No-op when PATCHED_OPD_DIR is unset.
#
# PATCHED_COMBINE_SELECT_DIR (optional, set by caller, temporary diagnostic):
# same mechanism, for a patched openclaw_combine_select_api_server.py ahead
# of the official openclaw-combine/ -- adds one debug log line per PRM eval
# turn showing the actual response_text/next_state_text used to compute that
# turn's score, so turn->content mapping can be read directly instead of
# guessed. See scripts/prepare_patched_openclaw_combine_select.sh. No-op
# when PATCHED_COMBINE_SELECT_DIR is unset; safe to stop wiring this in once
# no longer needed for debugging.
#
# PATCHED_COMBINE_DIR (optional, set by caller): same mechanism, for a
# patched openclaw_combine_api_server.py ahead of the official
# openclaw-combine/ -- this is where _maybe_submit_ready_samples() (the
# single dispatch point shared by both the OPD and RL submission paths)
# lives. Real training imports OpenClawCombineSelectAPIServer, which
# subclasses this file's OpenClawCombineAPIServer WITHOUT overriding
# _maybe_submit_ready_samples() -- patching only openclaw_opd_api_server.py
# or openclaw_combine_select_api_server.py leaves this dispatch function
# unpatched and the fix silently inert. See
# scripts/prepare_patched_openclaw_combine.sh (docs/issues_log.md 2026-08-13
# entry). Only needs to be ahead of the official openclaw-combine/ path
# below (its order relative to PATCHED_COMBINE_SELECT_DIR does not matter --
# they patch different files). No-op when PATCHED_COMBINE_DIR is unset.
old_pythonpath = (
    '\\"PYTHONPATH\\": \\"${REPO_ROOT}/Megatron-LM:${SCRIPT_DIR}:${REPO_ROOT}/openclaw-opd:${SLIME_ROOT}\\",'
)
new_pythonpath = (
    '\\"PYTHONPATH\\": \\"${PATCHED_OPD_DIR:+${PATCHED_OPD_DIR}:}'
    '${PATCHED_COMBINE_DIR:+${PATCHED_COMBINE_DIR}:}'
    '${PATCHED_COMBINE_SELECT_DIR:+${PATCHED_COMBINE_SELECT_DIR}:}'
    '${REPO_ROOT}/Megatron-LM:${SCRIPT_DIR}:${REPO_ROOT}/openclaw-opd:${SLIME_ROOT}\\",'
)
if old_pythonpath not in text:
    raise SystemExit("patch failed: PYTHONPATH line not found in topk-select launcher")
text = text.replace(old_pythonpath, new_pythonpath, 1)

# NCCL_DEBUG=INFO: diagnostic for the recurring silent mid-training hang
# (2026-07-09, docs/issues_log.md) -- training goes completely silent a few
# minutes in, no Python traceback, no OOM killer signature, reproduced 4x
# across smoke/minitest (TP=1 and TP=2). Suspected NCCL-level collective
# hang during the first real distributed gradient sync. This surfaces NCCL's
# own diagnostic output on the next run so we have real evidence instead of
# guessing. Not a permanent config -- meant to be reverted once the actual
# cause is identified (NCCL_DEBUG=INFO is very verbose).
old_cuda_conn = '\\"CUDA_DEVICE_MAX_CONNECTIONS\\": \\"1\\",'
new_cuda_conn = '\\"CUDA_DEVICE_MAX_CONNECTIONS\\": \\"1\\",\n    \\"NCCL_DEBUG\\": \\"INFO\\",'
if old_cuda_conn not in text:
    raise SystemExit("patch failed: CUDA_DEVICE_MAX_CONNECTIONS line not found in topk-select launcher")
text = text.replace(old_cuda_conn, new_cuda_conn, 1)

# 2026-07-13: 官方脚本把 --wandb-key ${WANDB_KEY_VALUE} 当成命令行参数传给
# train_async.py，wandb 会把完整启动命令记到 run 的 "Command" 字段里，项目一旦
# 公开这个 key 就跟着明文暴露（真实发生过，见 issues_log.md 2026-07-13 条目）。
# wandb SDK 在没有显式 key 时会自动读 WANDB_API_KEY 环境变量完成登录（见
# slime/utils/wandb_utils.py:40，args.wandb_key is None 时跳过显式 login，
# 后续 wandb.init() 自己兜底），所以只需要把 key 从 CLI 参数改成走 Ray runtime
# env 的环境变量，不影响功能。
old_nccl_env = '\\"NCCL_DEBUG\\": \\"INFO\\",'
new_nccl_env = '\\"NCCL_DEBUG\\": \\"INFO\\",\n    \\"WANDB_API_KEY\\": \\"${WANDB_API_KEY:-}\\",'
if old_nccl_env not in text:
    raise SystemExit("patch failed: NCCL_DEBUG line not found in topk-select launcher")
text = text.replace(old_nccl_env, new_nccl_env, 1)

old_wandb_args = (
    "  WANDB_ARGS=(\n"
    "    --use-wandb\n"
    "    --wandb-project ${WANDB_PROJECT}\n"
    "    --wandb-group qwen3-4b-openclaw-topk-select\n"
    "    --wandb-key ${WANDB_KEY_VALUE}\n"
    "  )"
)
if old_wandb_args not in text:
    raise SystemExit("patch failed: WANDB_ARGS block not found in topk-select launcher")

# 2026-07-27: 把这次训练用的 openclaw-rl commit（我们自己的补丁/脚本仓库，
# 不是 OpenClaw-RL-official）拼进 wandb group 名字里——训练次数一多，光
# 靠打开 wandb 看 run 名字就能知道这次用的是哪个版本的补丁，不用再跑去
# 日志目录翻 RUN_MANIFEST.txt（train_separate_student.sh 里同步写了一份，
# 两边都能查）。取不到时用 "unknown" 兜底，不阻断训练。
new_wandb_args = (
    "  WANDB_ARGS=(\n"
    "    --use-wandb\n"
    "    --wandb-project ${WANDB_PROJECT}\n"
    '    --wandb-group qwen3-4b-openclaw-topk-select-${OPENCLAW_RL_GIT_SHA:-unknown}\n'
    "  )"
)
text = text.replace(old_wandb_args, new_wandb_args, 1)

path.write_text(text)
PY

chmod +x "${PATCHED}"
exec bash "${PATCHED}"
