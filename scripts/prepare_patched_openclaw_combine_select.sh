#!/bin/bash
# TEMPORARY DIAGNOSTIC PATCH -- openclaw-rl-debug-turn-content
#
# Patches openclaw-combine/openclaw_combine_select_api_server.py's
# `_opd_evaluate()` (the function that actually decides each turn's PRM
# eval_score for the topk-select Hybrid RL method) to also log a truncated
# snippet of the response_text/next_state_text that produced a given turn's
# score, right next to the existing
#   "PRM eval session=... turn=N eval_votes=[...] -> eval_score=X"
# line.
#
# Why: that existing log line has no way to be mapped back to a specific
# real conversation action without guessing -- turn numbers assigned by this
# pipeline do not correspond 1:1 with "conversation turn N" as counted by
# student_chat.py/TA_chat.py/teacher_chat.py's own turn printouts (a single
# user-facing message can spawn more than one policy-model completion, e.g.
# one to decide to call a tool and a separate one to generate the
# confirmation reply after the tool result comes back), and PRM eval log
# lines can appear out of order (evaluated by a concurrent worker pool).
# Confirmed (2026-07-22) via real data that a naive "turn N = conversation
# cycle N" assumption produces contradictory conclusions when cross-checked
# against the known "next_state = a correction request -> score should be
# -1" rule. See docs/issues_log.md 2026-07-22 for the full investigation.
#
# This is a pure observability addition -- it does not change any reward,
# training, or data path, only adds one more log line. Safe to remove
# entirely (or just stop generating/wiring it in) once no longer needed for
# debugging -- unlike the other patches in this directory, this one is not
# fixing anything, just adding visibility.
#
# Official openclaw-combine/ directory is left untouched; this writes a
# patched copy to DEST_DIR, and the caller must prepend DEST_DIR to
# PYTHONPATH ahead of openclaw-combine/ so
# `import openclaw_combine_select_api_server` resolves to the patched copy
# (see run_openclaw_topk_select_modelfactory.sh's PATCHED_COMBINE_SELECT_DIR
# handling).
set -euo pipefail

REPO_ROOT=${1:?usage: prepare_patched_openclaw_combine_select.sh <repo_root> <dest_dir>}
DEST_DIR=${2:?usage: prepare_patched_openclaw_combine_select.sh <repo_root> <dest_dir>}
SRC="${REPO_ROOT}/openclaw-combine/openclaw_combine_select_api_server.py"
DEST="${DEST_DIR}/openclaw_combine_select_api_server.py"

if [ ! -f "${SRC}" ]; then
    echo "错误：找不到官方文件 ${SRC}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

python3 - "${SRC}" "${DEST}" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(src_path, encoding="utf-8").read()

marker = "openclaw-rl-debug-turn-content"
if marker in text:
    raise SystemExit(
        f"patch failed: marker already present in {src_path} -- "
        "the source may already be patched. Investigate before proceeding."
    )

old_block = (
    '            eval_score = _prm_eval_majority_vote(eval_raw)\n'
    '            logger.info(\n'
    '                "%s[OpenClaw-Combine-Select] PRM eval session=%s turn=%d "\n'
    '                "eval_votes=%s -> eval_score=%.1f%s",\n'
    '                _CYAN, session_id, turn_num,\n'
    '                [s if s is not None else "fail" for s in eval_raw],\n'
    '                eval_score, _RESET,\n'
    '            )\n'
)
if text.count(old_block) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the PRM eval logger.info "
        f"block in {src_path}, found {text.count(old_block)} (official file may "
        "have changed upstream -- re-verify this patch)"
    )

new_block = old_block + (
    f'            # --- {marker} (temporary, safe to remove) ---\n'
    '            logger.info(\n'
    f'                "%s[{marker}] session=%s turn=%d response_text=%r "\n'
    '                "next_state_role=%s next_state_text=%r%s",\n'
    '                _CYAN, session_id, turn_num,\n'
    '                turn_data["response_text"][:120],\n'
    '                next_state_role,\n'
    '                next_state_text[:120],\n'
    '                _RESET,\n'
    '            )\n'
)
text = text.replace(old_block, new_block, 1)

# ---------------------------------------------------------------------
# 2026-07-27 补丁：openclaw-rl-invalid-tool-use-penalty
#
# 背景（docs/issues_log.md 2026-07-27 条目）：_build_prm_eval_prompt()
# （openclaw-opd/openclaw_opd_api_server.py）的判分规则写死"工具调用只要
# 没报错就该打正分"，完全不检测这次调用是不是在原地打转。真实训练日志
# 证实 Problem 42 那种连续 20+ 轮的循环——模型反复用 sessions_send（目标
# 是自己当前 session，自问自答）、sessions_yield（从未真正派生过子
# agent 却在等结果）这两个工具，每一次都因为"技术上没报错"被判正分。
#
# openclaw_opd_api_server.py 那边的补丁（prepare_patched_openclaw_opd.sh）
# 已经在 turn_data 里加了 "is_invalid_tool_use" 标记，检测三条逻辑上
# 必然成立、不依赖具体任务的无效模式（read 类查询工具紧邻重复、
# sessions_send 自问自答、sessions_yield 没有对应的 sessions_spawn）。
# 这里读取这个标记，命中时直接把 eval_score 强制覆盖成 -1.0，不再指望
# LLM 判官自己发现这个盲区。
#
# 覆盖点选在 eval_score 刚算出来之后：_submit_turn_sample（hint 被
# 采纳时）和 _submit_rl_turn_sample（没有 hint 时）用的 reward 都直接
# 来自这同一个 eval_score 变量（见 openclaw_combine_api_server.py 里
# _maybe_submit_ready_samples 的调度逻辑），所以只需要改这一处，两条
# 路径都会生效。
# ---------------------------------------------------------------------
eval_score_old = (
    '            eval_score = _prm_eval_majority_vote(eval_raw)\n'
)
if text.count(eval_score_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the eval_score assignment "
        f"in {src_path}, found {text.count(eval_score_old)} (official file may have "
        "changed upstream -- re-verify this patch)"
    )
eval_score_new = eval_score_old + (
    '            # --- openclaw-rl-invalid-tool-use-penalty (temporary, safe to remove) ---\n'
    '            if turn_data.get("is_invalid_tool_use"):\n'
    '                logger.info(\n'
    '                    "%s[openclaw-rl-invalid-tool-use-penalty] session=%s turn=%d "\n'
    '                    "known-invalid tool use -- overriding eval_score %.1f -> -1.0%s",\n'
    '                    _CYAN, session_id, turn_num, eval_score, _RESET,\n'
    '                )\n'
    '                eval_score = -1.0\n'
)
text = text.replace(eval_score_old, eval_score_new, 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched -> {dest_path}")
PY

python3 -m py_compile "${DEST}"
echo "已生成 openclaw_combine_select_api_server.py 调试补丁（openclaw-rl-debug-turn-content）: ${DEST}"
