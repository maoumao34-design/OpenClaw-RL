#!/bin/bash
# Patch openclaw-test/{student,TA,teacher}_chat.py:
#
#   Rewrite the literal `"model": "default"` field to `"model": "openclaw/default"`,
#   the agent-target format OpenClaw 2026.6.9's /v1/chat/completions endpoint
#   actually expects. This is a pure API-compatibility shim (this OpenClaw CLI
#   version's routing format, unrelated to the paper's method) -- without it
#   every request 400s immediately, nothing can run at all.
#
# 2026-07-23: the homework-verification-gate patch (deterministic file-write
# check + 32B recheck before honoring DONE_SENTINEL) that used to live here
# has been removed for this experiment, per explicit user request, to isolate
# whether the write/overwrite-compliance problems we kept hitting are actually
# caused by the external Simulator model (Qwen3-32B) itself being too weak/
# non-compliant to reliably notice its own mistakes, rather than a gap in the
# harness. This run therefore uses the *unmodified* official DONE_SENTINEL
# logic (no session-continuation gate at all) -- see docs/issues_log.md
# 2026-07-23 entry for the removal rationale and the full prior design
# (several real-data-driven revisions) if it ever needs to be restored.
#
# Reproduction-fidelity note: swapping the Simulator model away from Qwen3-32B
# (paper Section 4.1) makes this run NOT a valid Table 3 data point -- it is a
# diagnostic-only experiment to isolate the 32B model's contribution to the
# write-compliance failures, not a replacement for the official Table 3
# reproduction runs.
#
# This only rewrites a known, literal source line; no training logic, reward,
# or data path is touched. The official openclaw-test/ directory is left
# untouched -- this writes patched copies to DEST_DIR instead.
set -euo pipefail

REPO_ROOT=${1:?usage: prepare_openclaw_test_scripts.sh <repo_root> <dest_dir>}
DEST_DIR=${2:?usage: prepare_openclaw_test_scripts.sh <repo_root> <dest_dir>}
SRC_DIR="${REPO_ROOT}/openclaw-test"

mkdir -p "${DEST_DIR}"

if [ ! -e "${DEST_DIR}/GSM8K.json" ]; then
    ln -sf "${SRC_DIR}/GSM8K.json" "${DEST_DIR}/GSM8K.json"
fi

for filename in student_chat.py TA_chat.py teacher_chat.py; do
    src_path="${SRC_DIR}/${filename}"
    dest_path="${DEST_DIR}/${filename}"
    old_model='"model": "default"'
    count=$(grep -o "${old_model}" "${src_path}" | wc -l | tr -d ' ')
    if [ "${count}" != "1" ]; then
        echo "patch failed: expected exactly 1 occurrence of ${old_model} in ${filename}, found ${count}" >&2
        exit 1
    fi
    sed "s/${old_model}/\"model\": \"openclaw\/default\"/" "${src_path}" > "${dest_path}"
    echo "patched -> ${dest_path}"
done

# ---------------------------------------------------------------------
# 2026-07-29 补丁：去掉 STUDENT_SYSTEM_PROMPT 里"or anything too AI-like" /
# "If it looks too 'AI-like'" 这两处开放式兜底判断，只保留三个具体格式特征
# （bold / numbered lists / "**Final answer**:"）作为要求重写的触发条件。
#
# 背景（docs/issues_log.md 2026-07-29 条目）：实测比对 Problem 20/21 两个真实
# session 发现，Simulator（当前用 DeepSeek V4 顶替论文原定的 Qwen3-32B）会
# 用这条开放式兜底去挑一些不在三个具体格式特征之列的"AI 感"（比如 emoji、
# "像喝咖啡一样聊"这种场景化措辞），而模型应对"听起来更自然"这个要求的方式
# 是往回复里加东西（emoji、场景化描述）而不是做减法，导致重写次数越多、
# 内容越长，且跟 Table 3 实际拿来算收敛的正则（只看 bold/numbered-list/
# boxed 三项）完全脱节——训练时的奖励信号和最终评判标准不是一回事。
# 去掉这两处兜底后，Simulator 要求重写的条件收窄成跟收敛正则一致的三个
# 具体特征，不再因为主观"感觉AI味"而触发无休止的重写-加内容循环。
#
# 用户对此提出的假设：可能是当前顶替 Qwen3-32B 的 DeepSeek V4 本身能力更强、
# 对"AI 感"的判断比论文原定模型更敏感/更严格，导致 4B policy 跟不上这个
# 标准——这个假设未经证实，此改动是否能验证/缓解这一点也待观察。
#
# 复现忠实性说明：这是主动偏离论文自己写定的 Simulator 提示词，不是修复
# 一个"跟论文原意不符"的 bug，而是为了让训练时的奖励信号跟 Table 3 汇报
# 用的评判标准保持一致而做的改动。这会让 Simulator 不再纠正三个具体格式
# 特征之外的"AI 感"内容（比如 emoji、场景化措辞），这些内容如果模型本身
# 倾向于生成，可能会正大光明地留在最终答案里而不再被要求修改——收敛数字
# 可能因此变好看，但不代表模型真的学会了"完全自然地表达"，需要在结果里
# 明确说明这一点，不能当成默认修复处理。
# ---------------------------------------------------------------------
python3 - "${SRC_DIR}/student_chat.py" "${DEST_DIR}/student_chat.py" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(dest_path, encoding="utf-8").read()

old_criteria = (
    'lists, "**Final answer**:", or anything too AI-like, tell it to \\\n'
    'rewrite in a more natural way but keep all the steps.'
)
if text.count(old_criteria) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the AI-like criteria "
        f"sentence in student_chat.py, found {text.count(old_criteria)} "
        "(official file may have changed upstream -- update this patch)"
    )
new_criteria = (
    'lists, or "**Final answer**:", tell it to \\\n'
    'rewrite in a more natural way but keep all the steps.'
)
text = text.replace(old_criteria, new_criteria, 1)

old_step1 = (
    '1. Look at what the AI gives you. If it looks too "AI-like", tell it to redo it. '
    'If not, no need to redo. \\\n'
)
if text.count(old_step1) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of Steps item 1 in "
        f"student_chat.py, found {text.count(old_step1)} (official file may "
        "have changed upstream -- update this patch)"
    )
new_step1 = (
    '1. Look at what the AI gives you. If it has bold text, numbered lists, or '
    '"**Final answer**:", tell it to redo it. If not, no need to redo. \\\n'
)
text = text.replace(old_step1, new_step1, 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched (AI-like catch-all removed) -> {dest_path}")
PY

echo "已生成 openclaw-test 补丁: ${DEST_DIR}（model 字段兼容修复 + student_chat.py 去掉开放式 AI-like 兜底，homework-verification-gate 已移除，见 docs/issues_log.md 2026-07-23 / 2026-07-29）"
