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
    'is the WRITING STYLE. If the AI\'s answer has stuff like bold text, numbered \\\n'
    'lists, "**Final answer**:", or anything too AI-like, tell it to \\\n'
    'rewrite in a more natural way but keep all the steps.'
)
if text.count(old_criteria) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the AI-like criteria "
        f"sentence in student_chat.py, found {text.count(old_criteria)} "
        "(official file may have changed upstream -- update this patch)"
    )
# Narrow WHAT counts as AI-like to the three concrete markers the Table 3
# regex actually checks, but keep "AI-like" itself as the framing concept --
# not a bare checklist disconnected from it (see docs/issues_log.md 2026-07-29).
new_criteria = (
    'is the WRITING STYLE. If the AI\'s answer looks AI-like -- stuff like bold \\\n'
    'text, numbered lists, or "**Final answer**:" -- tell it to \\\n'
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
    '1. Look at what the AI gives you. If it looks AI-like -- bold text, numbered '
    'lists, or "**Final answer**:" -- tell it to redo it. If not, no need to redo. \\\n'
)
text = text.replace(old_step1, new_step1, 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched (AI-like catch-all removed) -> {dest_path}")
PY

# ---------------------------------------------------------------------
# 2026-08-03 补丁：student_chat.py 的 main() 对每道题的调用外层加 try/except，
# 单题崩溃不再拖死整场训练。
#
# 背景（docs/issues_log.md 2026-08-03 条目）：官方源码 main() 里 `for i in
# range(count): completed = run_one_problem(...)` 外层没有任何异常捕获——
# 这学期反复观察到的"某道题连续 408/超时、run_one_problem() 内部
# resp.raise_for_status() 抛出未捕获的 HTTPError、整个 for 循环直接终止、
# 后续所有题目一个都不再尝试"，根因就在这里，不是训练侧的问题。
#
# 这是纯编排加固，不改训练算法/奖励逻辑：某道题崩溃时只把它记成
# incomplete（跟原本 max_turns 用完时的"incomplete"语义一致），继续跑
# 下一题，最大化这次训练实际能收集到的有效 session 数。
# ---------------------------------------------------------------------
python3 - "${SRC_DIR}/student_chat.py" "${DEST_DIR}/student_chat.py" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(dest_path, encoding="utf-8").read()

old_loop = (
    '    results = []\n'
    '    for i in range(count):\n'
    '        completed = run_one_problem(\n'
    '            problem_index=i,\n'
    '            gateway_url=gateway_url,\n'
    '            gateway_token=gateway_token,\n'
    '            external_client=external_client,\n'
    '            model=model,\n'
    '            max_turns=args.max_turns,\n'
    '            max_retries=args.max_retries,\n'
    '            output_file=args.output,\n'
    '        )\n'
    '        results.append(completed)\n'
)
if text.count(old_loop) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the main() problem "
        f"loop in student_chat.py, found {text.count(old_loop)} (official "
        "file may have changed upstream -- update this patch)"
    )
new_loop = (
    '    results = []\n'
    '    for i in range(count):\n'
    '        try:\n'
    '            completed = run_one_problem(\n'
    '                problem_index=i,\n'
    '                gateway_url=gateway_url,\n'
    '                gateway_token=gateway_token,\n'
    '                external_client=external_client,\n'
    '                model=model,\n'
    '                max_turns=args.max_turns,\n'
    '                max_retries=args.max_retries,\n'
    '                output_file=args.output,\n'
    '            )\n'
    '        except Exception as e:\n'
    '            print(f"\\n  [ERROR] Problem {i} crashed: {e}. Marking incomplete, continuing to next problem.")\n'
    '            completed = False\n'
    '        results.append(completed)\n'
)
text = text.replace(old_loop, new_loop, 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched (per-problem exception handling added) -> {dest_path}")
PY

# ---------------------------------------------------------------------
# 2026-08-06 补丁：student_chat.py 的 FIRST_MESSAGE_TEMPLATE 里
# "Show me the answer first" 改成 "Show me your full solution with all
# the steps first"。
#
# 背景（docs/issues_log.md 2026-08-06 条目）：这句话是直接发给 policy
# （4B 模型本身）的第一条消息，不经过 Student LLM 的系统提示词过滤。
# "the answer" 对训练中的 4B 模型来说容易被理解成"只给最终答案"，而不是
# "完整解出来、给我看解法"。这个偏差一旦发生，STUDENT_SYSTEM_PROMPT 里
# Student 判断要不要打回重写只检查三个具体格式特征（bold / 编号列表 /
# "**Final answer**:"），一个不带任何格式标记的简短答案不会触发重写，
# 这条"短答"坏样本会被当满足要求直接推进到写文件那一步——这是"短答 /
# 只给 answer"这类坏行为的一个可能诱因（基于逻辑推理提出，尚未有真实
# 训练数据验证这个改动能降低短答现象出现的频率）。
#
# 复现忠实性说明：这是主动偏离论文原始 student_chat.py 的 prompt 设计，
# 跟 07-29 那次去掉"AI-like"开放式兜底判断是同一类性质。
# ---------------------------------------------------------------------
python3 - "${SRC_DIR}/student_chat.py" "${DEST_DIR}/student_chat.py" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(dest_path, encoding="utf-8").read()

old_first_message = (
    'FIRST_MESSAGE_TEMPLATE = (\n'
    '    "Hey, I have my homework in the file homework/{index}.txt in your workspace. "\n'
    '    "Can you read it and help me solve it? "\n'
    '    "Show me the answer first — don\'t write to the file until I tell you to."\n'
    ')\n'
)
if text.count(old_first_message) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of FIRST_MESSAGE_TEMPLATE in "
        f"student_chat.py, found {text.count(old_first_message)} (official file may "
        "have changed upstream -- update this patch)"
    )
new_first_message = (
    'FIRST_MESSAGE_TEMPLATE = (\n'
    '    "Hey, I have my homework in the file homework/{index}.txt in your workspace. "\n'
    '    "Can you read it and help me solve it? "\n'
    '    "Show me your full solution with all the steps first — don\'t write to the file until I tell you to."\n'
    ')\n'
)
text = text.replace(old_first_message, new_first_message, 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched (FIRST_MESSAGE_TEMPLATE de-biased away from bare-answer framing) -> {dest_path}")
PY

echo "已生成 openclaw-test 补丁: ${DEST_DIR}（model 字段兼容修复 + student_chat.py 去掉开放式 AI-like 兜底 + 单题异常容错 + FIRST_MESSAGE_TEMPLATE 去 bare-answer 歧义，homework-verification-gate 已移除，见 docs/issues_log.md 2026-07-23 / 2026-07-29 / 2026-08-03 / 2026-08-06）"
