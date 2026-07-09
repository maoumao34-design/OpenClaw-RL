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

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/dfs/data/openclaw-rl-project/OpenClaw-RL-official}
OFFICIAL="${REPO_ROOT}/openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh"
PATCHED="${OPENCLAW_TOPK_SELECT_SCRIPT:-${TMPDIR:-/tmp}/run_qwen3_4b_openclaw_topk_select_modelfactory.sh}"
SMOKE_PROFILE=${SMOKE_PROFILE:-0}
MINITEST_PROFILE=${MINITEST_PROFILE:-0}

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
old_pythonpath = (
    '\\"PYTHONPATH\\": \\"${REPO_ROOT}/Megatron-LM:${SCRIPT_DIR}:${REPO_ROOT}/openclaw-opd:${SLIME_ROOT}\\",'
)
new_pythonpath = (
    '\\"PYTHONPATH\\": \\"${PATCHED_OPD_DIR:+${PATCHED_OPD_DIR}:}'
    '${REPO_ROOT}/Megatron-LM:${SCRIPT_DIR}:${REPO_ROOT}/openclaw-opd:${SLIME_ROOT}\\",'
)
if old_pythonpath not in text:
    raise SystemExit("patch failed: PYTHONPATH line not found in topk-select launcher")
text = text.replace(old_pythonpath, new_pythonpath, 1)

path.write_text(text)
PY

chmod +x "${PATCHED}"
exec bash "${PATCHED}"
