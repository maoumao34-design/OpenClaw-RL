#!/bin/bash
# Patch openclaw_opd_api_server.py's session_id/turn_type derivation, plus
# (2026-07-16) a degenerate-response filter and truncation-diagnostic logging
# -- see that patch block near the bottom of this file's Python heredoc for
# details, and docs/issues_log.md 2026-07-16 for the investigation behind it.
#
# Why (session_id/turn_type patch): the paper's intended mechanism for X-Session-Id/X-Turn-Type is the
# official `rl-training-headers` OpenClaw plugin, which injects HTTP headers
# via a before_prompt_build hook. Confirmed via extensive source-level
# investigation (2026-07-09, docs/issues_log.md) that on this OpenClaw build
# (2026.6.9), NO header/dispatcher-based injection mechanism can reach a real
# outbound request -- OpenClaw added a deliberate SSRF-safety layer
# (fetchWithSsrFGuardInternal / fetchWithRuntimeDispatcher) that bypasses any
# externally-patched fetch/dispatcher for all non-Vitest-mock requests, with
# no config escape hatch. This is a structural, intentional security boundary
# in the current OpenClaw version, not present when the paper's plugin was
# written (~2026.3-4), confirmed by reading OpenClaw's actual source from
# that era.
#
# Working alternative: the patched plugin (scripts/prepare_patched_rl_training_headers.sh)
# uses before_prompt_build's `appendSystemContext` return field instead --
# this modifies request CONTENT, not transport, so it is unaffected by the
# SSRF-safety bypass. It encodes ctx.trigger/ctx.sessionId as a
# "[RL-TRAINING-META] session_id=... turn_type=..." marker appended to the
# system prompt. This file's patch:
#   1. Parses that marker out of `messages` as a fallback for session_id/turn_type
#      (still lower priority than explicit X-Session-Id/X-Turn-Type headers or
#      body fields, in case those ever start working again on a future OpenClaw
#      version).
#   2. Strips the marker text out of `messages` BEFORE forwarding to sglang and
#      BEFORE computing training-sample prompt_ids -- so neither the policy
#      model nor the training data ever sees the marker. Verified this keeps
#      model-facing and training-facing content identical to what the paper's
#      (never-working-here) header mechanism would have produced.
#
# The old Runtime-line session_id fallback (from OpenClaw's own "Runtime:"
# line, unrelated to the plugin) is kept as a lower-priority fallback below
# the marker -- harmless to keep, doesn't hurt if the marker is ever absent.
#
# This is a deviation from the paper's plugin-based design -- documented in
# docs/issues_log.md. Official openclaw-opd/ directory is left untouched; this
# writes a patched copy to DEST_DIR and the caller must prepend DEST_DIR to
# PYTHONPATH ahead of openclaw-opd/ so `import openclaw_opd_api_server`
# resolves to the patched copy.
set -euo pipefail

REPO_ROOT=${1:?usage: prepare_patched_openclaw_opd.sh <repo_root> <dest_dir>}
DEST_DIR=${2:?usage: prepare_patched_openclaw_opd.sh <repo_root> <dest_dir>}
SRC="${REPO_ROOT}/openclaw-opd/openclaw_opd_api_server.py"

if [ ! -f "${SRC}" ]; then
    echo "错误：找不到官方文件 ${SRC}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

python3 - "${SRC}" "${DEST_DIR}/openclaw_opd_api_server.py" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(src_path, encoding="utf-8").read()

old_block = (
    '            body = await request.json()\n'
    '            session_id = x_session_id or body.get("session_id") or "unknown"\n'
    '            turn_type = (x_turn_type or body.get("turn_type") or "side").strip().lower()\n'
)
if old_block not in text:
    raise SystemExit(
        "patch failed: expected chat_completions body/session_id/turn_type block not found "
        "in openclaw_opd_api_server.py (official file may have changed upstream -- update this patch)"
    )

class_marker = "class OpenClawOPDAPIServer"
if class_marker not in text:
    raise SystemExit("patch failed: OpenClawOPDAPIServer class not found")

helper = '''
_RUNTIME_SESSION_RE = re.compile(r"session=agent:[^:]+:openai-user:(\\S+)")
_RL_META_RE = re.compile(r"\\[RL-TRAINING-META\\] session_id=(\\S*) turn_type=(\\S*)")
# openclaw-rl-invalid-tool-use-penalty 规则 3（2026-08-06）：见该规则定义处的
# 注释。匹配官方 OpenClaw 自己 isSilentReplyText() 的语义（token-only，大小写
# 不敏感，允许以空白分隔的重复 token）。
_NO_REPLY_ONLY_RE = re.compile(r"^\\s*NO_REPLY(?:\\s+NO_REPLY)*\\s*$", re.IGNORECASE)

# openclaw-rl-shadow-sentence-repeat（2026-08-06 引入，2026-08-07 起同时是
# 规则 5 的输入，见该规则定义处）：句子切分正则，跟 docs/issues_log.md
# 2026-08-06 条目里人工统计"同句原样重复几次"用的口径一致（按句末标点/
# 换行切分，只统计长度 >= min_len 的句子）。
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\\s+|\\n+")

# 规则 5（2026-08-07）判负分用的阈值，由真实 shadow 数据校准（run
# 20260807_183828）：干净对照组 stop/+1（n=34, max copies=9）零误伤；已知
# 的 3 条近顶格 tool_calls 泄漏样本（copies=12/21/30）全部命中；核实过
# tool_calls/+1 里不存在"低 copies 高 thinking"的语义换皮漏网形态，不需要
# 额外规则兜底。见规则 5 定义处的完整说明。
_SENTENCE_REPEAT_INVALID_THRESHOLD = 12


def _max_sentence_copies(text, min_len=40):
    """Max number of times any single normalized sentence (len >= min_len)
    repeats verbatim within `text`. Mirrors the manual analysis methodology in
    docs/issues_log.md 2026-08-06 so results are directly comparable. Pure
    text statistic on its own -- but its result also feeds openclaw-rl-
    invalid-tool-use-penalty 规则 5 (2026-08-07), which can force eval_score
    to -1.0 when the count reaches _SENTENCE_REPEAT_INVALID_THRESHOLD.
    """
    if not text:
        return 0
    counts = {}
    for raw in _SENTENCE_SPLIT_RE.split(text):
        sentence = raw.strip()
        if len(sentence) < min_len:
            continue
        counts[sentence] = counts.get(sentence, 0) + 1
    return max(counts.values()) if counts else 0


# openclaw-rl-invalid-tool-use-penalty 规则 1a 通用化（2026-08-06）：见该规则
# 定义处的注释。已核实这次作业环境实际会用到的工具集合（read/write/edit/
# exec/message/web_search/sessions_*）里唯一真正有"轮询"语义的是 OpenClaw
# 自带的 process 工具（查询后台 exec 状态），跟 loopDetection 自己对轮询类
# 工具单独放宽阈值是同一个顾虑。这里豁免 process 工具本身，以及任何工具
# 调用里 args.action == "poll" 的情况（防御性地覆盖其他可能设计成轮询语义
# 的工具），其余工具一律不再区分名字。
_POLL_EXEMPT_TOOL_NAMES = {"process"}


def _is_poll_style_call(tool_name, args_raw):
    if tool_name in _POLL_EXEMPT_TOOL_NAMES:
        return True
    try:
        args = json.loads(args_raw) if args_raw else {}
    except (TypeError, ValueError):
        return False
    action = args.get("action") if isinstance(args, dict) else None
    return isinstance(action, str) and action.strip().lower() == "poll"


# openclaw-rl-invalid-tool-use-penalty 规则 4（2026-08-07，08-07 二次修订
# 见下）：见该规则定义处的注释。student_chat.py 把 session_user 设成
# f"student-hw-{problem_index}-{os.getpid()}"，这个值本来会作为 Runtime
# 行的 openai-user:<user> 字段出现在 system prompt 里，可以用已有的
# _extract_session_id_from_system_prompt() 解析出来。
#
# 二次修订原因：最初实现误以为这个值经由既有机制最终会成为这里看到的
# session_id 参数本身，实测证伪——RL-TRAINING-META 标记里的
# ctx.sessionId（OpenClaw 自己内部的 UUID）在 session_id 派生优先级链条
# 里排在 Runtime-line fallback 前面（见 chat_completions() 里 session_id
# 的 or 链），只要这个标记正常工作，session_id 参数拿到的永远是 UUID，
# Runtime-line fallback 根本轮不到执行，规则 4 对着 UUID 跑
# student-hw-(\\d+)- 永远匹配不上，之前是死代码。
#
# 修复：不动全局 session_id 优先级（其他规则依赖 session_id 是稳定 key，
# UUID 完全够用，不应该为了这一条规则改全局行为）。改成规则 4 检查点
# 自己单独调用 _extract_session_id_from_system_prompt(messages)，从
# Runtime 行独立解析出 student-hw-{index}-{pid}，不经过 session_id 变量。
# 只匹配 student_chat.py 的这个命名惯例，TA_chat.py/teacher_chat.py 等
# 其他 session 类型解析不出来，返回 None，规则 4 对它们自动不生效；
# side 请求（如 context summarization）没有 Runtime 行，但这些请求本来
# 就不会有 write/edit 调用，规则 4 的工具名判断已经天然把它们挡在外面。
_HOMEWORK_SESSION_RE = re.compile(r"student-hw-(\\d+)-")


def _expected_homework_path(session_id_or_runtime_user):
    match = _HOMEWORK_SESSION_RE.search(session_id_or_runtime_user or "")
    if not match:
        return None
    return f"homework/{match.group(1)}.txt"


def _extract_session_id_from_system_prompt(messages):
    """Fallback session id when X-Session-Id/body.session_id/RL-TRAINING-META
    marker are all absent.

    OpenClaw's own Runtime line (embedded in the first system message) carries
    "session=agent:<agentId>:openai-user:<user>", where <user> is exactly the
    OpenAI `user` field the caller passed (student_chat.py etc. set this to a
    stable per-conversation id). Independent of the rl-training-headers plugin.
    """
    if not isinstance(messages, list):
        logger.warning("[SESSION-ID-DEBUG] messages is not a list: %r", type(messages))
        return None
    system_contents = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            system_contents.append(f"<non-str:{type(content)!r}>")
            continue
        system_contents.append(content)
        match = _RUNTIME_SESSION_RE.search(content)
        if match:
            return match.group(1)
    logger.warning(
        "[SESSION-ID-DEBUG] no Runtime session match; %d system message(s), tail=%r",
        len(system_contents),
        system_contents[-1][-500:] if system_contents else None,
    )
    return None


def _extract_rl_meta_from_messages(messages):
    """Read the (session_id, turn_type) pair the patched rl-training-headers
    plugin appends to the system prompt via before_prompt_build's
    appendSystemContext (header injection is confirmed structurally
    impossible on this OpenClaw build -- see docs/issues_log.md). Either
    element may be None if the marker is absent or a field is empty.
    """
    if not isinstance(messages, list):
        return None, None
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        match = _RL_META_RE.search(content)
        if match:
            session_id = match.group(1) or None
            turn_type = match.group(2) or None
            return session_id, turn_type
    return None, None


def _strip_rl_meta_from_messages(messages):
    """Remove the RL-TRAINING-META marker from system message content.

    Must run before the messages are forwarded to sglang and before training
    prompt_ids are computed, so neither the policy model nor the training
    data ever sees this plugin-internal marker text.
    """
    if not isinstance(messages, list):
        return messages
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system" and isinstance(msg.get("content"), str):
            new_content = _RL_META_RE.sub("", msg["content"]).rstrip()
            if new_content != msg["content"]:
                msg = {**msg, "content": new_content}
        cleaned.append(msg)
    return cleaned


# openclaw-rl-duplicate-user-retry-penalty (2026-08-13): see the full design
# note near _fire_opd_task below. OpenClaw prefixes every user message with
# a "[Day YYYY-MM-DD HH:MM GMT+N]" timestamp before it reaches the model --
# Student's mechanical retries (send_to_openclaw() re-POSTs the exact same
# message text on 408/503) resend byte-identical content, but the timestamp
# differs across retries when they land in different minutes (confirmed via
# real run 20260813_094000: P7/P17's retries only match after stripping this
# prefix; P11's retry happened to land in the same minute and would have
# matched even without stripping, which is why it looked deceptively simple
# before this was checked against more than one sample). Comparing raw
# next_state_text without stripping this prefix would silently miss most
# real retries.
_OPENCLAW_TIMESTAMP_PREFIX_RE = re.compile(
    r"^\\[[A-Za-z]{3} \\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2} GMT[+-]\\d+\\]\\s*"
)


def _strip_openclaw_timestamp_prefix(text):
    return _OPENCLAW_TIMESTAMP_PREFIX_RE.sub("", text or "", count=1)


# openclaw-rl-metaclaw (temporary, safe to remove): a MetaClaw round\'s
# intermediate tool-call turns are generated INSIDE the real `openclaw agent`
# CLI subprocess\'s own internal loop -- the rollout driver never constructs
# those individual HTTP requests itself, so it cannot attach a custom JSON
# body field to them (this is the exact same constraint that made
# X-Session-Id/X-Turn-Type headers unusable and forced the RL-TRAINING-META
# system-prompt-marker workaround above). session_id, however, IS reliably
# propagated into every one of those requests via that same already-proven
# marker mechanism (it is what session_id derivation depends on in the first
# place) -- so "is this session a MetaClaw round" is derived from session_id
# itself, not from a body field. The rollout driver is responsible for using
# this exact prefix when it calls `openclaw agent --session-id`.
_METACLAW_SESSION_RE = re.compile(r"^metaclaw-")


# openclaw-rl-metaclaw (temporary, safe to remove) -- see
# docs/metaclaw_migration_plan.md "查证记录（三）" for the full design
# rationale. Judges an INTERMEDIATE tool-call turn inside a MetaClaw-Bench
# round (a turn that is not the round\'s final, checker-scored answer).
# Modeled directly on OpenClaw-RL\'s own toolcall-rl track
# (generate_with_retool.py::_build_prm_step_messages) rather than on this
# file\'s _build_prm_eval_prompt (that one is Personal Agent Track dialogue-
# correction criteria -- bold text, redo requests, etc. -- which do not apply
# to a tool-call action and would inject mismatched signal if reused here).
# Deliberately task-agnostic and strictly \\boxed{1}/\\boxed{-1} (no neutral
# option), matching toolcall-rl\'s own step judge exactly -- reuses the
# existing _query_prm_eval_once/_prm_eval_majority_vote machinery unchanged,
# only the prompt differs.
#
# Takes `history` only (no separate `problem` argument): the same body-field
# constraint explained above for _METACLAW_SESSION_RE also rules out passing
# a separate task-description field for this prompt -- `history` is the
# turn\'s own fully-rendered chat-template prompt_text, which already starts
# with this round\'s question as the first user message, so the task is
# implicitly present without needing a second channel.
def _build_metaclaw_step_judge_messages(history, action, observation):
    system = (
        "You are a process reward model (PRM).\\n"
        "Judge whether the current step is helpful and correct for completing the task "
        "described at the start of the conversation below.\\n"
        "You may think first, but your final output MUST be a strict decision format.\\n"
        "Valid decision is exactly one of: \\\\boxed{1} or \\\\boxed{-1}."
    )
    user = (
        f"Conversation so far (task + trajectory):\\n{history}\\n\\n"
        f"Current action:\\n{action}\\n\\n"
        f"Next state / observation:\\n{observation}\\n\\n"
        "Now output your evaluation on the quality of the current action, "
        "then output your final decision, \\\\boxed{1} or \\\\boxed{-1}\\n"
        "Do NOT continue the task. Your job is to judge the quality of the current "
        "action, not to continue it."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

'''

text = text.replace(class_marker, helper + "\n" + class_marker, 1)

new_block = (
    '            body = await request.json()\n'
    '            _rl_meta_session_id, _rl_meta_turn_type = _extract_rl_meta_from_messages(body.get("messages"))\n'
    '            if isinstance(body.get("messages"), list):\n'
    '                body["messages"] = _strip_rl_meta_from_messages(body["messages"])\n'
    '            session_id = (\n'
    '                x_session_id\n'
    '                or body.get("session_id")\n'
    '                or _rl_meta_session_id\n'
    '                or _extract_session_id_from_system_prompt(body.get("messages"))\n'
    '                or "unknown"\n'
    '            )\n'
    '            turn_type = (x_turn_type or body.get("turn_type") or _rl_meta_turn_type or "side").strip().lower()\n'
)
text = text.replace(old_block, new_block, 1)

if "\nimport re\n" not in text:
    text = text.replace("import json\n", "import json\nimport re\n", 1)

# ---------------------------------------------------------------------
# 2026-07-16 补丁：拦截退化生成 + 记录顶格截断时的推理原文
#
# 背景（docs/issues_log.md 2026-07-16 条目）：run 20260715_180549 的
# train/grad_norm 爆炸根因确认是模型退化输出（同一乱码字符 "𬣳"，token
# id=122362，跨两次独立训练复现；以及顶格跑满 max_tokens 却从未闭合
# <think> 的情况）被当正常样本喂回了训练队列。
#
# 这次做三件事：
#   1. 生成时用 logit_bias 直接屏蔽已知乱码 token（id=122362），让对话
#      本身就不会再收到这条坏回复——不是等生成完再丢弃样本，那样对话
#      历史里还是会留着这条坏回复，只是不计入训练数据（2026-07-16 第
#      一版实现搞错成了事后过滤，已修正为生成时屏蔽，见 issues_log.md）。
#   2. 已知乱码 token 的兜底检查（SGLang logit_bias 并非 100% 可靠，见
#      issues_log.md 记录的已知 issue）——不再按 content 长度过滤，短数字
#      回复被判 -1 是正常有效的 RL 训练信号，不是需要剔除的坏样本（2026-07-16
#      同一天内二次更新，见 issues_log.md）。
#   3. 顶格截断（finish_reason=="length"）*不*拦截——是否是真实的"卡死
#      循环"bug 还没查清（此前的日志只记录了 thinking 的字符数，从没
#      记录过原文，没法判断内容是循环还是没写完的正常推理），先把
#      reasoning 原文完整记下来，供下次复现时诊断，再决定要不要处理。
# ---------------------------------------------------------------------

forward_body_old = (
    '        tools = body.get("tools")\n'
    '        forward_body = {k: v for k, v in body.items() if k not in _NON_STANDARD_BODY_KEYS}\n'
    '        forward_body["stream"] = False\n'
    '        forward_body.pop("stream_options", None)\n'
    '        forward_body["logprobs"] = True\n'
    '        forward_body["top_logprobs"] = 1\n'
    '        if "model" not in forward_body:\n'
    '            forward_body["model"] = self.served_model_name\n'
)
if forward_body_old not in text:
    raise SystemExit(
        "patch failed: expected forward_body construction block not found "
        "in openclaw_opd_api_server.py (official file may have changed upstream -- update this patch)"
    )
forward_body_new = forward_body_old + (
    '        # 2026-07-16: 生成时直接屏蔽已知乱码 token（id=122362，"𬣳"），\n'
    '        # 不是等生成完再丢弃样本——那样对话本身还是会看到这条坏回复，\n'
    '        # 只是不计入训练数据。见 issues_log.md 2026-07-16。\n'
    '        forward_body.setdefault("logit_bias", {})\n'
    '        forward_body["logit_bias"]["122362"] = -100\n'
)
text = text.replace(forward_body_old, forward_body_new, 1)

thinking_log_old = (
    '        logger.info(\n'
    '            "%s[OpenClaw-OPD] [%s] session=%s thinking=%d chars, response:\\n%s%s",\n'
    '            _RED,\n'
    '            turn_type,\n'
    '            session_id,\n'
    '            len(reasoning),\n'
    '            content,\n'
    '            _RESET,\n'
    '        )\n'
    '        if tool_calls:\n'
)
if thinking_log_old not in text:
    raise SystemExit(
        "patch failed: expected thinking=%d chars logging block not found "
        "in openclaw_opd_api_server.py (official file may have changed upstream -- update this patch)"
    )
thinking_log_new = (
    '        logger.info(\n'
    '            "%s[OpenClaw-OPD] [%s] session=%s thinking=%d chars, response:\\n%s%s",\n'
    '            _RED,\n'
    '            turn_type,\n'
    '            session_id,\n'
    '            len(reasoning),\n'
    '            content,\n'
    '            _RESET,\n'
    '        )\n'
    '        _finish_reason = choice.get("finish_reason", "stop")\n'
    '        if _finish_reason == "length":\n'
    '            logger.info(\n'
    '                "[OpenClaw-OPD] [%s] session=%s TRUNCATED (finish_reason=length) reasoning_text:\\n%s",\n'
    '                turn_type, session_id, reasoning,\n'
    '            )\n'
    '        if tool_calls:\n'
)
text = text.replace(thinking_log_old, thinking_log_new, 1)

# 2026-07-16: tool_calls 日志原本不带 session_id，没法跟同一请求的其他日志行
# 关联（并发场景下日志行天然交错，事后分析时无法可靠配对）。加上 session_id。
tool_calls_log_old = (
    '        if tool_calls:\n'
    '            logger.info("[OpenClaw-OPD] tool_calls: %s", str(tool_calls)[:500])\n'
)
if tool_calls_log_old not in text:
    raise SystemExit(
        "patch failed: expected tool_calls logging block not found "
        "in openclaw_opd_api_server.py (official file may have changed upstream -- update this patch)"
    )
tool_calls_log_new = (
    '        # openclaw-rl-invalid-tool-use-penalty 规则 1b (2026-07-28): must run\n'
    '        # BEFORE this turn\'s own tool_calls are inspected below. messages[-1]\n'
    '        # here is the real result of the PREVIOUS main turn\'s action (OpenClaw\n'
    '        # always sends the full conversation history, so its last element is\n'
    '        # whatever came back after the previous turn). If that previous turn\n'
    '        # was itself a read, self._pending_read[session_id] holds its (path,\n'
    '        # limit); resolve it into self._read_coverage now, before the block\n'
    '        # below potentially overwrites self._pending_read[session_id] with\n'
    '        # THIS turn\'s own (not-yet-executed) read call. Getting this ordering\n'
    '        # wrong would silently lose the previous turn\'s pending read every\n'
    '        # time the current turn is also a read.\n'
    '        if turn_type == "main" and self._turn_counts.get(session_id, 0) > 0 and messages:\n'
    '            _pending_read_info = self._pending_read.pop(session_id, None)\n'
    '            if _pending_read_info is not None:\n'
    '                _pr_path, _pr_limit = _pending_read_info\n'
    '                _pr_result = (\n'
    '                    _flatten_message_content(messages[-1].get("content"))\n'
    '                    if isinstance(messages[-1], dict) else ""\n'
    '                )\n'
    '                _pr_result_len = len(_pr_result)\n'
    '                _pr_cov = self._read_coverage.setdefault(session_id, {}).setdefault(\n'
    '                    _pr_path, {"eof": False, "max_limit": None}\n'
    '                )\n'
    '                # sticky/monotonic on purpose: a read sequence can go\n'
    '                # big-limit -> small-limit -> big-limit (seen in real data), and\n'
    '                # a later small-limit read genuinely IS truncated for THAT call\n'
    '                # even though an earlier bigger read already proved EOF. Once\n'
    '                # EOF is confirmed for a path it must stay confirmed regardless\n'
    '                # of what any later read (with a smaller limit) reports; the\n'
    '                # max_limit ever tried must only grow, never shrink either.\n'
    '                if _pr_limit is None or _pr_result_len < _pr_limit:\n'
    '                    _pr_cov["eof"] = True\n'
    '                else:\n'
    '                    _pr_cov["max_limit"] = max(_pr_cov["max_limit"] or 0, _pr_limit)\n'
    '\n'
    '        # openclaw-rl-invalid-tool-use-penalty: default: this turn is not a\n'
    '        # known-invalid action unless proven otherwise below. Computed\n'
    '        # unconditionally (not just inside "if tool_calls:") so it is always\n'
    '        # defined by the time turn_data gets built further down, including on\n'
    '        # turns with no tool call at all (plain text turns), where it must be\n'
    '        # False, not stale from a previous turn.\n'
    '        _is_invalid_tool_use = False\n'
    '\n'
    '        # --- openclaw-rl-invalid-tool-use-penalty 规则 3（NO_REPLY 误用检测，\n'
    '        # 2026-08-06，temporary, safe to remove）---\n'
    '        # 背景（docs/issues_log.md 2026-08-05 条目，NO_REPLY 分析）：NO_REPLY\n'
    '        # 是 OpenClaw 自带的真实 token（SILENT_REPLY_TOKEN，见\n'
    '        # openclaw/src/auto-reply/tokens.ts），但只在 group chat 场景才会被\n'
    '        # 写进 system prompt 教给模型（buildGroupChatContext()，且要求\n'
    '        # silentReplyPolicy != "disallow"）；这次训练用的是 direct/webchat\n'
    '        # 单会话对话，system prompt 里从没出现过这个约定（buildDirectChatContext()\n'
    '        # 通篇没有任何 NO_REPLY 相关字样）。真实训练日志里反复观察到模型在\n'
    '        # 成功 write 之后自己吐出字面文本 "NO_REPLY" 作为回复正文——这是模型\n'
    '        # 自发套用了一个这次训练里没人教过的惯例，不是遵循 OpenClaw 给的\n'
    '        # 指示。下游表现是 Student 收到 "No response from OpenClaw."（网关把\n'
    '        # 这当 token-only 静默回复处理，见 normalizeReplyPayload() 里\n'
    '        # isSilentReplyPayloadText 命中的分支），教学任务实际上被跳过了，跟\n'
    '        # 现有的 silent-reply-policy 补丁（只管"真·空回复"）是两码事，那个\n'
    '        # 补丁管不到这里。这不是"工具调用"，是纯文本 turn，检测放在\n'
    '        # tool_calls 判断之外；跟规则 1a/1b/2a/2b 共用同一个\n'
    '        # _is_invalid_tool_use 标记——下游 openclaw_combine_select_api_server.py\n'
    '        # 只认这一个字段名，语义上已经是"已知无效动作"而不严格限定于工具\n'
    '        # 调用（见规则 1a 定义处的注释）。\n'
    '        if _NO_REPLY_ONLY_RE.match(content):\n'
    '            _is_invalid_tool_use = True\n'
    '\n'
    '        if tool_calls:\n'
    '            logger.info("[OpenClaw-OPD] session=%s tool_calls: %s", session_id, str(tool_calls)[:500])\n'
    '            _tool_name = tool_calls[0].get("function", {}).get("name")\n'
    '            _tool_args_raw = tool_calls[0].get("function", {}).get("arguments")\n'
    '            _cur_call = (_tool_name, _tool_args_raw)\n'
    '\n'
    '            # --- openclaw-rl-debug-repeat-thinking (temporary, safe to remove) ---\n'
    '            # 2026-07-24: log full reasoning text whenever a tool_call exactly\n'
    '            # repeats the previous one -- observational only, does not feed\n'
    '            # into the scoring logic below.\n'
    '            _prev_call = self._last_tool_call.get(session_id)\n'
    '            if _prev_call is not None and _prev_call == _cur_call:\n'
    '                logger.info(\n'
    '                    "[openclaw-rl-debug-repeat-thinking] session=%s repeated tool_call=%s reasoning:\\n%s",\n'
    '                    session_id, _cur_call, reasoning,\n'
    '                )\n'
    '\n'
    '            # --- openclaw-rl-invalid-tool-use-penalty (2026-07-27, temporary, safe to remove) ---\n'
    '            # docs/issues_log.md 2026-07-27 条目。三条规则都是"不管换到什么\n'
    '            # 任务/工具集都必然成立"的逻辑判断，尽量降低误伤合理工具用法的风险：\n'
    '            #\n'
    '            # 规则 1a（2026-08-06 通用化，见同日 issues_log.md 条目）：紧邻上一次\n'
    '            # 调用（工具名 + 参数）完全相同 -> 没有新信息或没有新效果，不再限定\n'
    '            # read/write 这两个工具名。原始动机（read 的道理：中间没有\n'
    '            # write/edit，文件内容不可能变化，重复读取确定没有新信息；write 的\n'
    '            # 道理对称：紧邻上一次把同样内容写到同一个 path，文件已经是这个内容\n'
    '            # 了，再写一次没有新效果）对任何工具都成立，不是 read/write 特有的\n'
    '            # 性质——真正需要排除的只有结果会随外部状态自行变化、重复调用本身\n'
    '            # 就有意义的轮询类工具。已核实这次作业环境实际会用到的工具集合\n'
    '            # （read/write/edit/exec/message/web_search/sessions_*）里唯一有\n'
    '            # 这种语义的是 OpenClaw 自带的 process（查询后台 exec 状态），用\n'
    '            # _is_poll_style_call() 豁免它（以及任何 args.action=="poll" 的\n'
    '            # 调用，防御性覆盖其他可能设计成轮询语义的工具），其余工具一律\n'
    '            # 不再区分名字。之前只限定 read/write 时，edit/sessions_*/message\n'
    '            # 这类工具的紧邻精确重复完全没有规则覆盖；真实训练数据里见过反复\n'
    '            # 重复同一个 write、每次都被判官打 +1、session 上下文持续膨胀直至\n'
    '            # 拖垮 SGLang/网关的案例，同样的机制对其他工具一样成立。\n'
    '            if (\n'
    '                _prev_call is not None\n'
    '                and _prev_call == _cur_call\n'
    '                and not _is_poll_style_call(_tool_name, _tool_args_raw)\n'
    '            ):\n'
    '                _is_invalid_tool_use = True\n'
    '\n'
    '            # 规则 1b（2026-07-28）：同一个 path 换 limit 参数重复读取，会\n'
    '            # 逃过规则 1a 的逐字匹配（docs/issues_log.md 2026-07-28：真实\n'
    '            # 数据里对同一个 ~250 字节文件连续 read 了 48 次，limit 从 50\n'
    '            # 试到 100000，每次参数字符串都不同，全部逃过了规则 1a）。\n'
    '            # self._read_coverage[session_id][path] 累积记录这个 path 目前\n'
    '            # 为止已知的最大覆盖范围（写入时机见上面 turn_type=="main" 那段\n'
    '            # 的注释）：一旦任何一次 read 被证实读到了 EOF（结果长度 <\n'
    '            # 那次的 limit，或那次根本没传 limit），"eof" 就永久置 True，\n'
    '            # 不会因为后面又用小 limit 读了一次（那次确实会被截断，但不\n'
    '            # 代表文件变大了）而被撤销；否则记录"max_limit"为迄今试过的\n'
    '            # 最大 limit。一旦 eof 为 True，后续任何 limit 的重复读取都\n'
    '            # 判定为无效；否则只有这次 limit 明确大于 max_limit 才可能读到\n'
    '            # 新内容，不大于则判定为无效。\n'
    '            if _tool_name == "read":\n'
    '                try:\n'
    '                    _read_args = json.loads(_tool_args_raw) if _tool_args_raw else {}\n'
    '                except (TypeError, ValueError):\n'
    '                    _read_args = {}\n'
    '                _read_path = _read_args.get("path")\n'
    '                _read_limit = _read_args.get("limit")\n'
    '                _prior_cov = self._read_coverage.get(session_id, {}).get(_read_path)\n'
    '                if _prior_cov is not None:\n'
    '                    if _prior_cov["eof"]:\n'
    '                        _is_invalid_tool_use = True\n'
    '                    elif (\n'
    '                        _prior_cov["max_limit"] is not None\n'
    '                        and _read_limit is not None\n'
    '                        and _read_limit <= _prior_cov["max_limit"]\n'
    '                    ):\n'
    '                        _is_invalid_tool_use = True\n'
    '                self._pending_read[session_id] = (_read_path, _read_limit)\n'
    '\n'
    '            # 规则 2a：sessions_send 的目标是自己当前这个 session（自问\n'
    '            # 自答）-- 这种消息不可能真正送达任何外部对象，这是逻辑上的\n'
    '            # 必然结论，不依赖具体任务。发给真正不同的、合法派生的子\n'
    '            # session 不受影响。\n'
    '            if _tool_name == "sessions_send":\n'
    '                try:\n'
    '                    _send_args = json.loads(_tool_args_raw) if _tool_args_raw else {}\n'
    '                except (TypeError, ValueError):\n'
    '                    _send_args = {}\n'
    '                if _send_args.get("sessionKey") == "current":\n'
    '                    _is_invalid_tool_use = True\n'
    '\n'
    '            # 规则 2b：sessions_yield（结束当前轮次、等子 agent 结果回来）\n'
    '            # 在这个对话里从未真正调用过 sessions_spawn 派生过子 agent 的\n'
    '            # 情况下毫无意义 -- 没有派发过任务，逻辑上不可能有结果可等。\n'
    '            if _tool_name == "sessions_spawn":\n'
    '                self._has_spawned_subagent[session_id] = True\n'
    '            if _tool_name == "sessions_yield" and not self._has_spawned_subagent.get(session_id, False):\n'
    '                _is_invalid_tool_use = True\n'
    '\n'
    '            # 规则 4（2026-08-07，08-07 二次修订见 _expected_homework_path\n'
    '            # 定义处的注释）：write/edit 的目标 path 跟这个 session 对应的\n'
    '            # 正确 homework 文件不符 -- 判定为绕开任务要求的旁路写入。跟\n'
    '            # "有没有新信息"无关，是任务设计层面的必然错误：\n'
    '            # FIRST_MESSAGE_TEMPLATE 明确只要求把答案写到\n'
    '            # homework/{index}.txt，写别的路径没有任何合法理由。真实数据\n'
    '            # 实锤（docs/issues_log.md 2026-08-07 条目，Problem 17、session\n'
    '            # a0b1e908）：模型反复写入/读取旁路文件 17_solution.md 而不是\n'
    '            # homework/17.txt，因为跟上一拍工具名不同（read/write 交替），\n'
    '            # 逃过了规则 1a 的紧邻精确匹配检测，连续拿到 +1 进入训练。\n'
    '            # 注意：这里传的是 _extract_session_id_from_system_prompt(messages)\n'
    '            # 从 Runtime 行独立解析出来的值，不是 session_id 参数本身\n'
    '            # （session_id 参数拿到的是 RL-TRAINING-META 里的 UUID，优先级\n'
    '            # 高于 Runtime-line fallback，对它跑 student-hw-(\\d+)- 永远\n'
    '            # 匹配不上）。\n'
    '            if _tool_name in ("write", "edit"):\n'
    '                _expected_hw_path = _expected_homework_path(\n'
    '                    _extract_session_id_from_system_prompt(messages)\n'
    '                )\n'
    '                if _expected_hw_path is not None:\n'
    '                    try:\n'
    '                        _wr_args = json.loads(_tool_args_raw) if _tool_args_raw else {}\n'
    '                    except (TypeError, ValueError):\n'
    '                        _wr_args = {}\n'
    '                    _wr_path = _wr_args.get("path") if isinstance(_wr_args, dict) else None\n'
    '                    if _wr_path is not None and _wr_path != _expected_hw_path:\n'
    '                        _is_invalid_tool_use = True\n'
    '\n'
    '            self._last_tool_call[session_id] = _cur_call\n'
    '\n'
    '        # --- openclaw-rl-invalid-tool-use-penalty 规则 5 (2026-08-07) ---\n'
    '        # docs/issues_log.md 2026-08-07 条目：句子原样重复 >=\n'
    '        # _SENTENCE_REPEAT_INVALID_THRESHOLD 次 -> 判定为已知无效动作，强制\n'
    '        # -1。阈值由真实 shadow 数据校准（run 20260807_183828）：干净对照组\n'
    '        # stop/+1（n=34, max copies=9）零误伤；已知的 3 条近顶格 tool_calls\n'
    '        # 泄漏样本（P16/19/23, copies=12/21/30）全部命中；核实过\n'
    '        # tool_calls/+1 里不存在"低 copies 高 thinking"的语义换皮漏网形态，\n'
    '        # 不需要额外规则兜底。跟有没有调用工具无关（复读发生在 reasoning\n'
    '        # 本身，不依赖有没有调用工具），位置放在 tool_calls 判断之外，纯\n'
    '        # 文本 turn（比如规则 3 的 NO_REPLY 情形）也会被检查。\n'
    '        _max_sentence_copies_count = _max_sentence_copies(reasoning)\n'
    '        if _max_sentence_copies_count >= _SENTENCE_REPEAT_INVALID_THRESHOLD:\n'
    '            _is_invalid_tool_use = True\n'
    '\n'
    '        # --- openclaw-rl-shadow-sentence-repeat (2026-08-06, temporary, safe to\n'
    '        # remove) --- docs/issues_log.md 2026-08-06 条目：最初只是为"同句原样\n'
    '        # 重复 >= N 次强制判负分"这条候选规则收集校准数据，之前只有\n'
    '        # repeat-thinking/TRUNCATED 两条路径能拿到 reasoning 全文，是有偏子集。\n'
    '        # 2026-08-07 起数据已经用于校准并落地成上面的规则 5，这里的日志继续\n'
    '        # 保留，是规则 5 判定结果的可观测记录，不再是纯 observability——\n'
    '        # is_invalid_tool_use 字段现在会反映规则 5（以及其他规则）的效果。\n'
    '        logger.info(\n'
    '            "[openclaw-rl-shadow-sentence-repeat] session=%s turn_type=%s "\n'
    '            "thinking_chars=%d max_sentence_copies=%d finish_reason=%s "\n'
    '            "is_invalid_tool_use=%s",\n'
    '            session_id, turn_type, len(reasoning), _max_sentence_copies_count,\n'
    '            _finish_reason, _is_invalid_tool_use,\n'
    '        )\n'
)
text = text.replace(tool_calls_log_old, tool_calls_log_new, 1)

init_dict_old = (
    '        self._pending_records: dict[str, dict[str, Any]] = {}\n'
)
if init_dict_old not in text:
    raise SystemExit(
        "patch failed: expected _pending_records init line not found "
        "in openclaw_opd_api_server.py (official file may have changed upstream -- update this patch)"
    )
init_dict_new = init_dict_old + (
    '        # openclaw-rl-debug-repeat-thinking: per-session (tool_name, arguments)\n'
    '        # of the most recent tool call, used to detect exact-repeat tool calls.\n'
    '        self._last_tool_call: dict[str, tuple] = {}\n'
    '        # openclaw-rl-invalid-tool-use-penalty: per-session flag, True once\n'
    '        # sessions_spawn has actually been called (used by the sessions_yield\n'
    '        # check -- see docs/issues_log.md 2026-07-27).\n'
    '        self._has_spawned_subagent: dict[str, bool] = {}\n'
    '        # openclaw-rl-invalid-tool-use-penalty 规则 1b（2026-07-28）：per-\n'
    '        # session, per-path 累积的已知覆盖范围：{"eof": bool, "max_limit":\n'
    '        # int|None}。eof 一旦为 True 就不会再被撤销（sticky/monotonic，\n'
    '        # 详见写入处的注释——避免大 limit 读完之后又用小 limit 读一次，\n'
    '        # 把"已确认读到 EOF"这个状态错误地覆盖掉）。\n'
    '        self._read_coverage: dict[str, dict[str, dict]] = {}\n'
    '        # 同上：per-session，记录"刚发起、结果还没回来"的 read 请求\n'
    '        # (path, limit)，在下一轮看到它的真实结果时回填进\n'
    '        # self._read_coverage（见 turn_type=="main" 那段的注释）。\n'
    '        self._pending_read: dict[str, tuple] = {}\n'
    '        # openclaw-rl-duplicate-user-retry-penalty (2026-08-13): per-session\n'
    '        # set of user-message texts (timestamp-prefix stripped) already seen\n'
    '        # either as an existing message in a turn\'s prompt or as a previously\n'
    '        # accepted next_state -- see _fire_opd_task for how this is populated\n'
    '        # and checked. Cleared at session_done alongside self._turn_counts.\n'
    '        self._seen_user_messages: dict[str, set] = {}\n'
)
text = text.replace(init_dict_old, init_dict_new, 1)

# ---------------------------------------------------------------------
# 2026-07-27 补丁：把"这次工具调用是否属于三条已确认无效模式之一"这个
# 标记带进 turn_data，供 openclaw-combine/openclaw_combine_select_api_server.py
# 的 PRM 打分逻辑读取，强制把这类 turn 的 eval_score 改成 -1（见该文件
# 补丁脚本 prepare_patched_openclaw_combine_select.sh 里的对应改动）。
#
# 根因（docs/issues_log.md 2026-07-27 条目）：_build_prm_eval_prompt() 的
# 判分规则写死"工具调用只要没报错就该打正分"，完全不检测这次调用是不是
# 在原地打转（重复读取、自己给自己发消息、无意义地等一个从未存在过的
# 子任务结果）。这里不指望 LLM 判官自己纠正这个盲区，改成代码层面直接
# 检测三条逻辑上必然成立的无效模式、直接覆盖分数。
# ---------------------------------------------------------------------
turn_data_old = (
    '            turn_data = {\n'
    '                "prompt_ids": prompt_ids,\n'
    '                "response_ids": response_ids,\n'
    '                "response_logprobs": response_logprobs,\n'
    '                "prompt_text": prompt_text,\n'
    '                "response_text": response_text,\n'
    '                "messages": messages,\n'
    '                "tools": tools,\n'
    '                "has_next_state": False,\n'
    '            }\n'
)
if turn_data_old not in text:
    raise SystemExit(
        "patch failed: expected turn_data construction block not found "
        "in openclaw_opd_api_server.py (official file may have changed upstream -- update this patch)"
    )
turn_data_new = (
    '            turn_data = {\n'
    '                "prompt_ids": prompt_ids,\n'
    '                "response_ids": response_ids,\n'
    '                "response_logprobs": response_logprobs,\n'
    '                "prompt_text": prompt_text,\n'
    '                "response_text": response_text,\n'
    '                "messages": messages,\n'
    '                "tools": tools,\n'
    '                "has_next_state": False,\n'
    '                # openclaw-rl-invalid-tool-use-penalty (2026-07-27): see comment\n'
    '                # near _is_invalid_tool_use above.\n'
    '                "is_invalid_tool_use": _is_invalid_tool_use,\n'
    '                # openclaw-rl-truncation-penalty (2026-08-06, temporary, safe to\n'
    '                # remove): finish_reason == "length" 说明模型还没说完就被切断，\n'
    '                # 见 docs/issues_log.md 2026-08-06 条目。\n'
    '                "is_truncated": _finish_reason == "length",\n'
    '                # openclaw-rl-repeat-thinking-hint (2026-08-11): 跟 is_invalid_tool_use\n'
    '                # 分开的单独标记——is_invalid_tool_use 是规则 1-5 的 OR，无法区分这次\n'
    '                # 具体是哪条触发；openclaw_combine_select_api_server.py 那边要挂"别\n'
    '                # 复读"这句针对性提醒，只能认这条单独的标记，不能认 is_invalid_tool_use，\n'
    '                # 否则会在 Rule 4 等其他规则触发时也错误地挂上这句文不对题的提醒。\n'
    '                # 复用规则 5 已经算出的 _max_sentence_copies_count/\n'
    '                # _SENTENCE_REPEAT_INVALID_THRESHOLD，不单独再算一遍、不用第二个阈值，\n'
    '                # 避免两处判断口径不一致。见 prepare_patched_openclaw_combine_select.sh\n'
    '                # 里的对应改动。\n'
    '                "is_repeat_thinking_violation": _max_sentence_copies_count >= _SENTENCE_REPEAT_INVALID_THRESHOLD,\n'
    '                # openclaw-rl-degraded-turn-drop (2026-08-13): 408/503 重试链污染\n'
    '                # 训练信号的 A/B 两类——见 docs/issues_log.md 2026-08-13 条目。\n'
    '                # A：这次生成本身被 SGLang 中断（pause_generation 打断 in-flight\n'
    '                # 生成），finish_reason 是 "abort"，不是正常的 stop/tool_calls/\n'
    '                # length，内容通常是半截，不该进训练。\n'
    '                "is_aborted": _finish_reason == "abort",\n'
    '                # B：这次生成吐出结果的那一刻，submission 已经被训练步暂停\n'
    '                # （weight sync 期间的正常暂停窗口）——不管内容本身好不好，\n'
    '                # 产生它的环境当时是不可信的，同一批 in-flight 请求会在暂停期间\n'
    '                # 陆续吐出结果。判断点选在生成结束、写入 turn_data 这一刻（不是\n'
    '                # 请求刚进来的时候），因为暂停可能发生在生成过程中途。\n'
    '                "generated_while_paused": not self.submission_enabled.is_set(),\n'
    '                # openclaw-rl-metaclaw (temporary, safe to remove): lets\n'
    '                # openclaw_combine_select_api_server.py tell an intermediate\n'
    '                # tool-call turn inside a MetaClaw round apart from an unrelated\n'
    '                # Personal Agent Track turn. Derived from session_id (see\n'
    '                # _METACLAW_SESSION_RE above), not a body field -- see that\n'
    '                # comment for why. False for every non-MetaClaw session, so this\n'
    '                # has no effect on existing training paths.\n'
    '                "metaclaw_round_mode": bool(_METACLAW_SESSION_RE.match(session_id or "")),\n'
    '            }\n'
)
text = text.replace(turn_data_old, turn_data_new, 1)

empty_response_old = (
    '            if not response_ids and not response_text.strip():\n'
    '                logger.info("[OpenClaw-OPD] MAIN session=%s -> empty response, skipping", session_id)\n'
    '                output["session_id"] = session_id\n'
    '                return {"response": output}\n'
)
if empty_response_old not in text:
    raise SystemExit(
        "patch failed: expected empty-response-skip block not found "
        "in openclaw_opd_api_server.py (official file may have changed upstream -- update this patch)"
    )
empty_response_new = empty_response_old + (
    '\n'
    '            # 2026-07-16: 只兜底拦截已知乱码 token（id=122362，"𬣳"）——主要\n'
    '            # 靠上面 forward_body 的 logit_bias 在生成阶段就屏蔽掉，这里只是\n'
    '            # 防 SGLang logit_bias 在个别版本上失效（见 issues_log.md 2026-07-16\n'
    '            # WebSearch 记录的已知 issue），正常情况下不应该再命中。\n'
    '            # 不再按 content 长度过滤——像"25"这种被 TA 判 -1 的短数字回复\n'
    '            # 是正常、有效的 RL 训练信号（教模型"这样答不满足要求"），不是\n'
    '            # 需要剔除的坏样本；只有真正空的内容才需要拦（跟官方一致，见\n'
    '            # 上面的 empty_response 检查），见 issues_log.md 2026-07-16 更新。\n'
    '            _KNOWN_GLITCH_TOKEN_IDS = {122362}  # "𬣳"\n'
    '            if any(_tid in _KNOWN_GLITCH_TOKEN_IDS for _tid in response_ids):\n'
    '                logger.info(\n'
    '                    "[OpenClaw-OPD] MAIN session=%s -> degenerate response (content=%r), skipping",\n'
    '                    session_id, content[:50],\n'
    '                )\n'
    '                output["session_id"] = session_id\n'
    '                return {"response": output}\n'
)
text = text.replace(empty_response_old, empty_response_new, 1)

# ---------------------------------------------------------------------
# openclaw-rl-duplicate-user-retry-penalty (2026-08-13): D 的最终方案，见
# docs/issues_log.md 2026-08-13 条目完整讨论过程。
#
# 背景：Student 撞 408/503 后机械重发同一句指令（send_to_openclaw() 的
# 重试循环里 message 变量不变，不会让 Simulator 重新措辞），导致这句指令
# 在同一 session 的对话历史里重复出现。真实数据（run 20260813_094000）
# 证实：一开始怀疑的"P17 整个 run 都因为 408 而不可信"是错的——P17 里
# 真正把题做完的那次 write（+1）就发生在重试之后，同一个 run_id 下既有
# 该丢的重复指令 turn，也有该留的正常完成 turn，按 run_id 整体拉黑会
# 连坐好样本。真正该丢的只是"next_state 是一句已经在这个 session 里
# 出现过的 user 消息"这一类具体特征的 turn，不是整个重试 run。
#
# 检测点选在 _fire_opd_task（PRM 判分之前），不在 _opd_evaluate：省一次
# judge 调用，而且命中时可以直接不 create_task，turn 天然落入
# openclaw_combine_api_server.py 的 `if task is None` 分支之前那道拦截
# （跟 is_aborted/generated_while_paused 共用同一处，见
# prepare_patched_openclaw_combine.sh），不会滞留在 _pending_turn_data
# 里等到 session_done 才被动清理。
#
# 集合怎么攒：只靠"历史上出现过的 next_state"不够——对话第一条 user
# 消息（题面本身）从来不会是某个 turn 的 next_state，如果题面本身因为
# 超时被重发，纯粹这样攒集合会漏判。所以 fire 时先把 turn_data["messages"]
# 里已经存在的所有 user 正文（剥完时间戳前缀）灌进这个 session 的已见
# 集合，再拿 next_state 去查，两类来源都覆盖。
#
# 处置方式是丢弃（不提交），不是强制 eval_score=-1：这条 turn 本身的
# 内容可能完全正常（P11 T4 就是开始写入的正常内容，只是配对的
# next_state 是重试进来的重复指令）——错的是"这次评价用的 next_state
# 配对坏了"，不是"这个动作本身该罚"，跟 tool-error-penalty 那类"动作
# 本身有问题"的规则性质不同，不应该用同一种"强制 -1 仍然提交"的处理。
# ---------------------------------------------------------------------
fire_opd_task_old = (
    '    def _fire_opd_task(self, session_id: str, turn_num: int, turn_data: dict[str, Any], next_state: dict[str, Any]):\n'
    '        if not self._prm_enabled or not next_state:\n'
    '            return\n'
    '        task = asyncio.create_task(self._opd_evaluate(session_id, turn_num, turn_data, next_state))\n'
    '        task.add_done_callback(self._task_done_cb)\n'
    '        task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))\n'
    '        self._prm_tasks.setdefault(session_id, {})[turn_num] = task\n'
    '        turn_data["has_next_state"] = True\n'
)
if fire_opd_task_old not in text:
    raise SystemExit(
        "patch failed: expected _fire_opd_task body not found "
        "in openclaw_opd_api_server.py (official file may have changed upstream -- update this patch)"
    )
fire_opd_task_new = (
    '    def _fire_opd_task(self, session_id: str, turn_num: int, turn_data: dict[str, Any], next_state: dict[str, Any]):\n'
    '        if not self._prm_enabled or not next_state:\n'
    '            return\n'
    '\n'
    '        # --- openclaw-rl-duplicate-user-retry-penalty (temporary, safe to remove) ---\n'
    '        _seen = self._seen_user_messages.setdefault(session_id, set())\n'
    '        for _msg in turn_data.get("messages") or []:\n'
    '            if isinstance(_msg, dict) and _msg.get("role") == "user":\n'
    '                _text = _strip_openclaw_timestamp_prefix(_flatten_message_content(_msg.get("content")))\n'
    '                if _text:\n'
    '                    _seen.add(_text)\n'
    '        if next_state.get("role") == "user":\n'
    '            _next_text = _strip_openclaw_timestamp_prefix(_flatten_message_content(next_state.get("content")))\n'
    '            if _next_text and _next_text in _seen:\n'
    '                turn_data["is_duplicate_user_retry"] = True\n'
    '                logger.info(\n'
    '                    "[openclaw-rl-duplicate-user-retry] session=%s turn=%d dropped "\n'
    '                    "(next_state repeats an earlier user message in this session) -- "\n'
    '                    "not submitted to OPD or RL",\n'
    '                    session_id, turn_num,\n'
    '                )\n'
    '                self._maybe_submit_ready_samples(session_id)\n'
    '                return\n'
    '            if _next_text:\n'
    '                _seen.add(_next_text)\n'
    '\n'
    '        task = asyncio.create_task(self._opd_evaluate(session_id, turn_num, turn_data, next_state))\n'
    '        task.add_done_callback(self._task_done_cb)\n'
    '        task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))\n'
    '        self._prm_tasks.setdefault(session_id, {})[turn_num] = task\n'
    '        turn_data["has_next_state"] = True\n'
)
text = text.replace(fire_opd_task_old, fire_opd_task_new, 1)

session_done_cleanup_old = (
    '            self._turn_counts.pop(session_id, None)\n'
)
if text.count(session_done_cleanup_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the session_done "
        f"_turn_counts.pop cleanup line in openclaw_opd_api_server.py, found "
        f"{text.count(session_done_cleanup_old)} (official file may have "
        "changed upstream -- update this patch)"
    )
session_done_cleanup_new = (
    '            self._turn_counts.pop(session_id, None)\n'
    '            self._seen_user_messages.pop(session_id, None)\n'
)
text = text.replace(session_done_cleanup_old, session_done_cleanup_new, 1)

# ---------------------------------------------------------------------
# 2026-08-19 补丁：openclaw-rl-metaclaw-verdict-signal-skip
#
# 背景：MetaClaw 迁移的 driver 用一个 X-Turn-Type: main 的合成 POST 把
# checker 的确定性 verdict 挂给上一轮当 next_state（这一步的唯一目的，
# 见 metaclaw_rollout_driver.py::_send_verdict_turn 的文档字符串）。但
# turn_type=="main" 在这个函数里天然还有第二个、不需要的副作用：这次
# POST 自己的生成结果也会被注册成一条新的待评估 pending turn（下面
# `if turn_type == "main":` 分支的原有逻辑）。原来想靠"把 max_tokens
# 调小/调到 0，让生成结果为空，指望现有的空响应检查接住"来避免这个
# 副作用——用真实 Qwen3-4B-Thinking-2507 tokenizer 实测过，这个假设不
# 成立：即使 assistant 消息 content/reasoning_content 都是空字符串，
# Thinking 模板本身仍然会补上结构性的 `</think>` 闭合标签和 `<|im_end|>`
# 轮次结束标记，这几个 token 不是空白，现有 `response_text.strip()`
# 检查过滤不掉，那条早退分支永远不会命中（历史证据：2026-08-18
# metaclaw_migration_20260818_182736 那次 max_tokens=8 时留下了
# 13-token 的 thinking 残片；改小到 0 只会留下约 5-token 的模板闭合
# 残片，Bug 换个形态但没堵住）。
#
# 真正的修法是控制流层面直接短路，不依赖生成结果长什么样：driver 把
# `max_tokens: 0` 当一个专用信号（不会跟 OpenClaw 真实请求撞车——它
# 自己配置的 maxTokens 是 50000），代理识别到这个信号后完全不调用
# SGLang（连"0 会不会被 SGLang 拒绝"这个问题都不存在），只做这次调用
# 唯一该做的事——给上一轮挂 next_state、触发它的评估——然后照抄当前
# （含 openclaw-rl-duplicate-user-retry-penalty 补丁新增的
# `_seen_user_messages.pop`）session_done 清理逻辑直接返回，全程不碰
# apply_chat_template / _pending_turn_data / _turn_counts，第二个副
# 作用从源头上就不会发生。
#
# 必须放在 messages 校验之后、forward_body 构造/SGLang POST 之前——
# 再往前 messages 还没定义，没法执行"挂 next_state"这一步；这个信号
# 分支本身要抢在真正调用 SGLang 之前拦下来，不能等生成完再补救。
#
# 这个补丁本身放在脚本末尾（`_seen_user_messages.pop` 那个 session_done
# 清理补丁之后）执行，不是因为代码位置要在后面——它改的是函数最前面
# messages 校验之后那一段。放在这里纯粹是为了避开一个字符串匹配冲突：
# 这段新代码自己内部也要写一遍"照抄"的 session_done 清理逻辑（含
# `_turn_counts.pop`），如果这个 text.replace() 在 session_done_cleanup_old
# 那个 patch 之前执行，会让 `_turn_counts.pop(session_id, None)\n` 这行在
# 全文里变成 2 处，导致后面那个 patch 的"必须只有 1 处"校验失败——放在
# 它后面执行就不会有这个问题，纯粹是脚本内 patch 顺序的技术细节，不影响
# 最终生成文件里两段代码的相对位置。
# ---------------------------------------------------------------------
metaclaw_verdict_skip_old = (
    '        messages = body.get("messages")\n'
    '        if not isinstance(messages, list) or not messages:\n'
    '            raise HTTPException(status_code=400, detail="messages must be a non-empty list")\n'
    '\n'
    '        tools = body.get("tools")\n'
)
if metaclaw_verdict_skip_old not in text:
    raise SystemExit(
        "patch failed: expected messages-validation block not found "
        "in openclaw_opd_api_server.py (official file may have changed upstream -- update this patch)"
    )
metaclaw_verdict_skip_new = (
    '        messages = body.get("messages")\n'
    '        if not isinstance(messages, list) or not messages:\n'
    '            raise HTTPException(status_code=400, detail="messages must be a non-empty list")\n'
    '\n'
    '        # --- openclaw-rl-metaclaw-verdict-signal-skip (temporary, safe to remove) ---\n'
    '        if turn_type == "main" and body.get("max_tokens") == 0:\n'
    '            logger.info(\n'
    '                "[openclaw-rl-metaclaw-verdict-signal-skip] session=%s -> "\n'
    '                "max_tokens=0 signal, skipping SGLang call entirely "\n'
    '                "(no pending turn registered)",\n'
    '                session_id,\n'
    '            )\n'
    '            _prev_turn_num = self._turn_counts.get(session_id, 0)\n'
    '            if _prev_turn_num > 0 and messages:\n'
    '                self._flush_pending_record(session_id, messages[-1])\n'
    '                _prev_turn_data = self._pending_turn_data.get(session_id, {}).get(_prev_turn_num)\n'
    '                if _prev_turn_data is not None:\n'
    '                    self._fire_opd_task(session_id, _prev_turn_num, _prev_turn_data, messages[-1])\n'
    '            if session_done:\n'
    '                self._flush_pending_record(session_id, None)\n'
    '                self._maybe_submit_ready_samples(session_id, force_drop_without_next_state=True)\n'
    '                self._turn_counts.pop(session_id, None)\n'
    '                self._seen_user_messages.pop(session_id, None)\n'
    '                logger.info("[OpenClaw-OPD] session=%s done -> cleaned up", session_id)\n'
    '            return {"response": {"choices": [], "session_id": session_id}}\n'
    '\n'
    '        tools = body.get("tools")\n'
)
text = text.replace(metaclaw_verdict_skip_old, metaclaw_verdict_skip_new, 1)

# ---------------------------------------------------------------------------
# openclaw-rl-metaclaw-train-until-day (temporary, safe to remove) -- see
# docs/metaclaw_migration_plan.md "方案：可调 K 天训练窗口 + 冻结评测剩余天数"
# for the full design. Two independent patches:
#   1. __init__: self._metaclaw_training_frozen = False -- the flag itself,
#      defaulting to False so a process that never receives the freeze
#      signal below behaves identically to before this patch existed.
#   2. chat_completions route: recognize a dedicated X-Metaclaw-Freeze-
#      Training header BEFORE the submission_enabled 503 gate (not inside
#      _handle_request, which the route only reaches AFTER that gate) --
#      this is a control-plane signal from the rollout driver, not a
#      training-data submission, and must not be blockable by slime's
#      pause-for-weight-update window. Sets the flag and returns
#      immediately, never touching body/session_id/turn_type machinery.
# The actual enforcement (dropping samples instead of submitting them once
# frozen) lives in openclaw_combine_api_server.py's _maybe_submit_ready_samples
# override (prepare_patched_openclaw_combine.sh), not here -- this file only
# owns the flag and how it gets set.
# ---------------------------------------------------------------------------
train_until_day_init_old = (
    '        self._server: uvicorn.Server | None = None\n'
    '        self._thread: threading.Thread | None = None\n'
    '        self.app = self._build_app()\n'
)
if train_until_day_init_old not in text:
    raise SystemExit(
        "patch failed: expected __init__ tail block not found "
        "in openclaw_opd_api_server.py (official file may have changed upstream -- update this patch)"
    )
train_until_day_init_new = (
    '        self._server: uvicorn.Server | None = None\n'
    '        self._thread: threading.Thread | None = None\n'
    '\n'
    '        # --- openclaw-rl-metaclaw-train-until-day (temporary, safe to remove) ---\n'
    '        # Default False: unless the rollout driver explicitly sends the freeze\n'
    '        # signal below (only happens when METACLAW_TRAIN_UNTIL_DAY is set), this\n'
    '        # is never touched and behaves exactly as if this patch did not exist.\n'
    '        self._metaclaw_training_frozen = False\n'
    '\n'
    '        self.app = self._build_app()\n'
)
text = text.replace(train_until_day_init_old, train_until_day_init_new, 1)

chat_completions_freeze_old = (
    '        @app.post("/v1/chat/completions")\n'
    '        async def chat_completions(\n'
    '            request: Request,\n'
    '            authorization: str | None = Header(default=None),\n'
    '            x_session_id: str | None = Header(default=None),\n'
    '            x_turn_type: str | None = Header(default=None),\n'
    '            x_session_done: str | None = Header(default=None),\n'
    '        ):\n'
    '            owner: OpenClawOPDAPIServer = request.app.state.owner\n'
    '            await owner._check_auth(authorization)\n'
    '            if not owner.submission_enabled.is_set():\n'
    '                raise HTTPException(status_code=503, detail="submission paused for weight update")\n'
)
if text.count(chat_completions_freeze_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the chat_completions "
        f"auth/submission_enabled block in {src_path}, found "
        f"{text.count(chat_completions_freeze_old)} (official file may have "
        "changed upstream -- update this patch)"
    )
chat_completions_freeze_new = (
    '        @app.post("/v1/chat/completions")\n'
    '        async def chat_completions(\n'
    '            request: Request,\n'
    '            authorization: str | None = Header(default=None),\n'
    '            x_session_id: str | None = Header(default=None),\n'
    '            x_turn_type: str | None = Header(default=None),\n'
    '            x_session_done: str | None = Header(default=None),\n'
    '            x_metaclaw_freeze_training: str | None = Header(default=None),\n'
    '        ):\n'
    '            owner: OpenClawOPDAPIServer = request.app.state.owner\n'
    '            await owner._check_auth(authorization)\n'
    '\n'
    '            # --- openclaw-rl-metaclaw-train-until-day (temporary, safe to\n'
    '            # remove) --- Deliberately BEFORE the submission_enabled 503 gate:\n'
    '            # this is a control-plane message from the rollout driver (see\n'
    '            # metaclaw_rollout_driver.py::_send_freeze_signal), not a training\n'
    '            # sample, and must not be blockable by slime pausing submission for\n'
    '            # a weight-update window -- a driver that only ever hits 503 here\n'
    '            # would never manage to actually freeze training. Never parses\n'
    '            # `body` (frozen and unfrozen requests alike can send an empty\n'
    '            # placeholder payload; the header alone decides).\n'
    '            if (\n'
    '                x_metaclaw_freeze_training\n'
    '                and x_metaclaw_freeze_training.strip().lower() in ("1", "true", "yes", "on")\n'
    '            ):\n'
    '                owner._metaclaw_training_frozen = True\n'
    '                logger.info("[metaclaw-freeze] training_frozen=True (signal received)")\n'
    '                return JSONResponse(content={"status": "frozen"})\n'
    '\n'
    '            if not owner.submission_enabled.is_set():\n'
    '                raise HTTPException(status_code=503, detail="submission paused for weight update")\n'
)
text = text.replace(chat_completions_freeze_old, chat_completions_freeze_new, 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched -> {dest_path}")
PY

echo "已生成 session_id 兼容补丁: ${DEST_DIR}/openclaw_opd_api_server.py"
