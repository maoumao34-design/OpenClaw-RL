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
    'rewrite in a more natural way but keep all the steps. Just tell it to fix \\\n'
    'the style — don\'t fix it yourself.'
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
#
# 08-07 二次修订（docs/issues_log.md 2026-08-07 条目，跟下面 new_steps 的
# 四次修订是同一批改动）：去掉 "but keep all the steps" -- 这句话跟 Step 1
# "otherwise" 分支里同一句话（见下面 new_steps）是同一个诱因的两处来源，
# 真实数据显示 Simulator 会机械念这句话，逼着已经有紧凑步骤的正确回答
# 继续"加内容"，是"越改越长"两拍循环的燃料之一。改成明确"别要求加
# 内容"，因为走到这个分支说明内容本身已经通过前置的 Tier-0 判断（见
# new_steps），这里只该管格式/语气，不该再管内容完整性。old_criteria 的
# 匹配范围顺带往后扩到 "don't fix it yourself." 这句，是为了把新加的
# "别要求加内容"这半句干净地插进去，避免跟后面未改动的原文重复。
new_criteria = (
    'is the WRITING STYLE. If the AI\'s answer looks AI-like -- stuff like bold \\\n'
    'text, numbered lists, or "**Final answer**:" -- tell it to \\\n'
    'rewrite in a more natural way. Just tell it to fix the style -- don\'t ask \\\n'
    'it to add more or redo the math, and don\'t fix it yourself.'
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
# [已撤销] 2026-08-06 曾对 FIRST_MESSAGE_TEMPLATE 打过补丁（"Show me the
# answer first" 依次改成方案 B"...full solution with all the steps..."、
# 08-07 改窄为方案 A"...full worked answer..."），2026-08-07 当天晚些
# 时候确认撤销、完全恢复官方原始措辞，不再对这一行打补丁。
#
# 撤销原因（docs/issues_log.md 2026-08-07 条目）：这次改动的初衷是防止
# 4B 模型把"the answer"理解成"只给裸答案"。但同一天新增的 Steps
# 重构（见下）已经把"是否给出了带过程的真答案"做成 Student 每一轮持续
# 生效的检查（Tier-0，见下方补丁），不再是开场白一次性的措辞能单独覆盖
# 的职责。既然这个要求已经有了贯穿全程的机制兜底，FIRST_MESSAGE_TEMPLATE
# 就不需要再额外强调"full worked"——继续保留反而是重复限定，且 08-07
# 上午的真实数据已经证明"完整步骤"这类强调在开场白里加码是"格式癫痫+
# 拒写"这类新失效模式的诱因之一（见同日更早的 issues_log 条目）。撤销后
# 完全恢复论文原始的 "Show me the answer first — don't write to the file
# until I tell you to."，不再对这一行做任何改动。
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2026-08-07 补丁：STUDENT_SYSTEM_PROMPT 的 Steps 部分重构，把"这算不算
# 一个真答案"和"格式像不像 AI"拆成两层独立判断，并给 Step 3（写入确认）
# 单独加一条"核实是否真的写了"的要求。
#
# 背景（docs/issues_log.md 2026-08-07 条目）：定位到"格式癫痫+拒写"这类
# 新失效模式的两个必要因子之一是 FIRST_MESSAGE_TEMPLATE 被迫独自扛起
# "既要防裸答、又不能诱发过度表演"这个两难。讨论后决定把"必须有完整步骤
# 的解答"这个要求从开场白一次性的措辞，改成 Student 每一轮都会做的
# 持续检查，这样 FIRST_MESSAGE_TEMPLATE 不用再兼顾这个职责。
#
# 版本对照（相对官方原始未改动版本，不是相对 07-29 已部署版本）：
#
#   官方原始 Step 1（未改动）：
#     "1. Look at what the AI gives you. If it looks too "AI-like", tell
#     it to redo it. If not, no need to redo."
#     —— 只有一条笼统的"AI 味"判断，没有列具体特征，也完全没有"这压根
#     不是答案"这层判断。
#
#   07-29 已部署改动（现状，本补丁的基准）：把"太 AI 味"这个开放式判断
#   收窄成三个具体特征（bold / 编号列表 / "**Final answer**:"），Step
#   2/3 未改动。
#
#   本次（08-07）在 07-29 基础上新增：
#     1. Step 1 新增前置判断层——先判断"这到底算不算一个答案"（没有真实
#        回应 / 只有裸答案没解释 / 像原始 tool-call 或 JSON），命中就
#        直接要求"给我真正的答案"，不提写作风格；只有确认是真答案之后
#        才轮到原有的 AI 味格式判断（降级成第二层、有条件触发）。官方
#        原始版本和 07-29 版本都没有这层区分。
#     2. Step 2 基本不变，只加一个澄清性括号。
#     3. Step 3 新增"核实是否真的写了文件"这个要求——官方原始版本假设
#        AI 会诚实报告"已保存"，完全没有覆盖"AI 拒绝写入/空谈拖延"这
#        种情况（这正是 08-07 新发现的失效模式）；新版本要求 Student 在
#        看到拖延/拒绝时明确要求"actually do it"。
#     4. 顶栏新增一句"不要假装自己已经读过文件/解出了什么"，防止 Student
#        角色串戏（真实数据里见过泄文场景下 Student 说出"Got it, I
#        read..."这种不该由"什么都不懂的学生"说出的话）。
#
# 08-07 二次修订（同日，用户让 CLI 复核这版草稿后的反馈）：
#   - Step 3 原稿"it confirms saving, or you can tell it wrote/edited the
#     file"这个措辞太松——泄文场景下如果 AI 输出了一段看起来像 tool_call
#     的原始 JSON/代码片段（但实际没有真正执行），可能被误认成"已经写
#     了"，甚至被 Student 顺势催 HOMEWORK_DONE。收紧成：只有 AI 用大白话
#     明确确认已保存/追加内容才算数，看到 tool-call 样式的原始文本/JSON
#     不能当作证据。
#   - 新增顶栏"不要假装自己已读过文件/解出了什么"这一条（即上面第 4
#     点），堵角色串戏这个口子。
#   - CLI 同时指出这段 prompt 改动是"软刹车"，不是格式通道的硬约束——
#     KL≈0 时模型仍可能在其他地方漂出类似的假 tool 格式，真正稳住还得
#     配合打分层硬规则或采样侧控制，这条本身不是承诺"彻底不再泄文"，
#     只是明显降低"泄文被 Student 帮腔、进而拿到误 +1"这条主要燃料，
#     这个限定在最终决定要不要保留/回退时需要一并考虑，不能只看这段
#     prompt 有没有生效。
#
# 08-07 三次修订（用真实训练数据验证后发现 Tier-0 判断太严格）：真实训练
# 里观察到即使 policy 已经给出了带过程的解答，Tier-0 判断仍偶尔误判成
# "没有真正回答"。收紧判断门槛的初衷是防裸答/tool-call 空壳，但措辞没有
# 明确"只要有一定过程就不该触发"，导致 Simulator 对"过程够不够"这件事
# 判断过严。改成显式声明：只要有哪怕简短、不完整的推理或步骤就应该算作
# 已回答，只有真的完全没有任何过程展示时才触发这条规则；不能因为"解释
# 感觉短了点/本可以更详细"就触发。
#
# 这是主动设计的新机制，不是修复已知 bug，效果需要真实训练数据验证——
# 如果这次改动没有改善"格式癫痫+拒写"现象，或者引入了新的问题（比如
# Student 因为多了一层判断而变得更啰嗦、或者第一层判断本身出现新的误判
# 模式），需要回退到 07-29 版本（只有 AI 味判断，没有这层前置检查），
# 决策依据是下一轮训练的真实数据，不是理论推演。
#
# 08-07 四次修订（新训练 run 20260807_183828 的完整原因排查，见
# docs/issues_log.md 2026-08-07 条目）：定位到"超长 thinking 空转顶垮训练"
# 这条主线问题的上游诱因，不在生成侧，而在 Student 这层——
#   1. Tier-0 判断对"紧凑算式"（如 "5x5=25, 4x10=40, 40-25=15"，没有连接
#      散文句子）经常误判成"没有真正回答"（真实数据约 35% 的"要
#      step-by-step"请求打在已经有算式的回复上，P16/P17/P20 等题实锤）；
#   2. AI 味重写分支里写死的 "but keep all the steps"（跟上面 new_criteria
#      里同一句话是同一个诱因），逼着已经通过 Tier-0、内容本身没问题的
#      回答在"改自然语气"的同时被迫继续"加步骤"。
# 这两条叠加形成"加长→重写→再加长"的两拍循环，policy 学会用更长、更
# 口语化的方式"表演"完整解答，thinking 里反复打磨措辞草稿（真实数据里
# 见过同一句"Let me try to break it down..."类型的草稿句重复 16-25 次），
# 最终烧光 maxTokens 预算、产生空回复，形成本轮训练从 P28 起系统性崩溃
# 的因果链上游。这次改动清的是这条链条最上游的燃料，不是给 length 加
# 物理硬顶（那是另一条独立方向，讨论后本轮未做，见 issues_log），也不
# 直接解决失败后期的 tool_call XML 元循环二次崩溃（那是这条链条下游的
# 独立症状，本身应该会因为上游触发频率下降而减少，但不是这次改动直接
# 针对的目标）。
#
# 08-10 五次修订（新一轮训练仍见误判，跟用户+CLI 逐轮核实后重新设计
# Step 1，见 docs/issues_log.md 2026-08-10 另一条条目）：
#   背景——四次修订版本只有两分支（没答 / otherwise 直接查 AI 味），完全
#   没有"只有最终答案、没有过程"这类检测；用户澄清真实观察到的问题方向
#   是过度触发（把有过程的正确答案误判成没答/只有答案），不是漏判（裸
#   答案被放过）。一开始按"代码判、Simulator 只管措辞"设计了方案，用户
#   要求仍然全程交给 Simulator 判断，只是把判断标准写得更precise；随后
#   发现"只有答案"最初设计漏掉了②分支，加回来时又发现"判断有没有过程"
#   如果只看有没有运算符号（+ - × / =），会把用大白话叙述计算过程（比如
#   "4 guppies, plus 2 he bought, so that's 6 guppies... 16 minus 11
#   equals 5"，全程没有一个符号）的正确回答误判成"只有答案"。
#
#   最终方案：Step 1 从两分支改成三分支（①有没有真答案 → ②是不是只有
#   答案没过程 → ③才查 AI 味）：
#   - ①从"找不到理由才算没答"（负向排除）改写成"主动列举证据"（正向
#     枚举：孤立数字/Answer 模板/收尾句数字/紧凑算式），减少 Simulator
#     凭感觉判断的空间。
#   - ②新增，判断依据不是"有没有符号"，而是"整段回复里数字个数是不是
#     只有一个"——不管过程是用符号（"5x5=25"）还是大白话（"4 plus 2 is
#     6"）表达，只要出现第二个数字就说明有推导，不算"只有答案"；只有
#     从头到尾真的只提过一个数字（比如"The answer is $93."）才会触发这
#     条，专门堵之前"催过程"完全没有对应检测的缺口。
#   - ③（AI 味重写）文字基本不变，只是判断前提改成"确认有真实过程之后"
#     才检查。
#   - Step 2/3（写入请求 + 写入核实）完全不动——用户明确要求这部分维持
#     现有判断，这次训练的真实数据显示这部分工作得很好，不需要跟着改。
#   顶部 criteria 段落里"must still include the full solution process
#   with all steps shown"这句是论文原文（核实过官方源码从未被任何补丁
#   动过），保留②之后不再跟这句话冲突，不需要软化。
# ---------------------------------------------------------------------
python3 - "${SRC_DIR}/student_chat.py" "${DEST_DIR}/student_chat.py" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(dest_path, encoding="utf-8").read()

old_steps = (
    'Steps:\n'
    '1. Look at what the AI gives you. If it looks AI-like -- bold text, numbered lists, or "**Final answer**:" -- tell it to redo it. If not, no need to redo. \\\n'
    'Do NOT mention writing to the file in the same message. Only ask for a rewrite.\n'
    '2. After the AI shows you the satisfactory version and it looks good, THEN in a \\\n'
    'separate message ask it to append the answers to the end of the homework file \\\n'
    '(not overwrite it). Do NOT combine a rewrite request and a write request.\n'
    '3. After the AI says it saved the file, say exactly: HOMEWORK_DONE\n'
)
if text.count(old_steps) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the Steps block in "
        f"student_chat.py, found {text.count(old_steps)} (official file may "
        "have changed upstream, or an earlier patch in this script changed -- "
        "update this patch)"
    )
new_steps = (
    'Steps:\n'
    '1. Look at what the AI gives you in response to your solve request. Check in this order:\n'
    '   - Is there any real, identifiable final answer or worked calculation anywhere in the reply? This includes: a number standing on its own (possibly with $, a unit, or bold, like "14" or "$93" or "**6**"), a template like "Answer: X" / "Final answer: X" / "The answer is X" / boxed{X}, a closing sentence stating a concluding number (like "...so the total is $93."), OR any calculation shown anywhere, even compact and without full prose (like "5x5=25, 4x10=40, 40-25=15"). If you cannot find ANY of this -- it\'s empty, off-topic, confused, just restates the problem, just chats, just asks you something, or is a raw tool-call/JSON/code block with no real answer sentence -- tell it plainly that it did not really answer and you need to see the actual worked-out solution. Do NOT mention writing to the file or style in this message -- just ask for the real answer.\n'
    '   - If you DID find something like that: check if the ENTIRE reply mentions only ONE number in total (the final answer itself), with no other numbers, quantities, or intermediate values mentioned anywhere -- e.g. it\'s ONLY something like "90 liters." or "The answer is $93." or just "6" and nothing else. If there is more than one number anywhere in the reply -- whether the calculation is shown with symbols (like "5x5=25, 4x10=40") or described in plain words (like "4 plus 2 is 6" or "16 minus 11 equals 5"), that already counts as showing work -- do NOT treat it as "only the answer" just because it\'s brief, phrased in words instead of symbols, or doesn\'t use "=" signs. Only use this check when there is truly a single number and nothing else in the whole reply. If it\'s truly bare like that, tell it you need to see the actual steps/calculation, not just the final number -- do NOT mention writing to the file or style in this message either.\n'
    '   - Otherwise (it has real work/calculation shown, in any form): if it looks AI-like -- bold text, numbered lists, or "**Final answer**:" -- tell it to redo it in a more natural, casual way. Just ask it to fix the tone/formatting -- don\'t ask for more steps or more detail, the content is already fine. Do NOT mention writing to the file in the same message. Only ask for a rewrite. If it\'s already natural-sounding with no AI-like formatting, no need to redo.\n'
    '2. After the AI shows you a satisfactory version (a real explanation, not AI-like), THEN in a separate message ask it to append the answers to the end of the homework file (not overwrite it). Do NOT combine a rewrite request and a write request.\n'
    '3. Once you have asked it to write the file, check whether it actually did. Only count it as done if it plainly confirms in normal words that it saved or appended the content -- a raw tool-call-looking snippet or JSON blob is NOT proof it actually wrote anything. If it stalls, refuses, only shows you that kind of raw snippet without a real confirmation, or just talks without actually writing, tell it to actually do it. If it plainly confirms it wrote it, say exactly: HOMEWORK_DONE\n'
)
text = text.replace(old_steps, new_steps, 1)

old_no_solve_line = (
    'You CANNOT solve, rewrite, rephrase, or produce any answer yourself. \\\n'
    'You can ONLY tell the AI what to do. Never use academic or technical language.\n'
)
if text.count(old_no_solve_line) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the 'CANNOT solve' "
        f"line in student_chat.py, found {text.count(old_no_solve_line)} "
        "(official file may have changed upstream -- update this patch)"
    )
new_no_solve_line = (
    'You CANNOT solve, rewrite, rephrase, or produce any answer yourself. \\\n'
    'You can ONLY tell the AI what to do. Never use academic or technical language. \\\n'
    'Never pretend you have already read the file, solved something, or figured \\\n'
    'anything out yourself -- you have no idea what is in it. Only give instructions.\n'
)
text = text.replace(old_no_solve_line, new_no_solve_line, 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched (Steps: real-answer check split from AI-like check + write-verification tightened + role-bleed guard added) -> {dest_path}")
PY

echo "已生成 openclaw-test 补丁: ${DEST_DIR}（model 字段兼容修复 + student_chat.py 去掉开放式 AI-like 兜底 + 单题异常容错 + Steps 真答案/AI味两层判断+写入核实（严格版）+ 角色串戏防护，FIRST_MESSAGE_TEMPLATE 已撤销恢复官方原始措辞，homework-verification-gate 已移除，见 docs/issues_log.md 2026-07-23 / 2026-07-29 / 2026-08-03 / 2026-08-06 / 2026-08-07）"
