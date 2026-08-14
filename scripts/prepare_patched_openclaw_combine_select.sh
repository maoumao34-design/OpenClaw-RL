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

# ---------------------------------------------------------------------
# openclaw-rl-metaclaw (temporary, safe to remove): see
# docs/metaclaw_migration_plan.md "查证记录（三）" for the full design.
#
# Two MetaClaw-specific cases short-circuit the normal PRM judge path in
# _opd_evaluate() entirely; every other session/turn (all of Personal Agent
# Track, plus MetaClaw sessions before this patch existed) falls through to
# the original, unmodified logic unchanged:
#
#   1. next_state_text parses as {"metaclaw_verdict": true, "eval_score":
#      ..., "hint": ...} -- the rollout driver already ran MetaClaw-Bench's
#      deterministic checker for this round's FINAL turn and is handing us
#      the result directly. No LLM judge call at all.
#   2. turn_data["metaclaw_round_mode"] is true but next_state_text is NOT a
#      verdict -- an INTERMEDIATE tool-call turn inside a MetaClaw round.
#      Personal Agent Track's judge prompts (bold text / numbered list /
#      redo-request criteria) do not apply to a tool-call action and would
#      inject mismatched signal if reused here; use a task-agnostic step
#      judge instead (_build_metaclaw_step_judge_messages, added to
#      openclaw_opd_api_server.py by prepare_patched_openclaw_opd.sh,
#      modeled on OpenClaw-RL's own toolcall-rl track). RL-only, independent
#      of the round's eventual checker outcome, no OPD hint.
# ---------------------------------------------------------------------
opd_evaluate_head_old = (
    '        next_state_text = (\n'
    '            _flatten_message_content(next_state.get("content")) if next_state else ""\n'
    '        )\n'
    '        next_state_role = next_state.get("role", "user") if next_state else "user"\n'
    '        judge_msgs = _build_hint_judge_messages(\n'
    '            turn_data["response_text"], next_state_text, next_state_role,\n'
    '        )\n'
    '        if self._prm_tokenizer:\n'
    '            judge_prompt = self._prm_tokenizer.apply_chat_template(\n'
    '                judge_msgs, tokenize=False, add_generation_prompt=True,\n'
    '            )\n'
    '        else:\n'
    '            judge_prompt = "\\n".join(m["content"] for m in judge_msgs)\n'
    '\n'
    '        votes = await asyncio.gather(\n'
    '            *[self._query_judge_once(judge_prompt, i) for i in range(self._prm_m)]\n'
    '        )\n'
    '\n'
    '        # PRM eval branch (unchanged from parent).\n'
    '        if self._eval_mode:\n'
)
if text.count(opd_evaluate_head_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the _opd_evaluate head "
        f"block in {src_path}, found {text.count(opd_evaluate_head_old)} "
        "(official file may have changed upstream -- update this patch)"
    )
opd_evaluate_head_new = (
    '        next_state_text = (\n'
    '            _flatten_message_content(next_state.get("content")) if next_state else ""\n'
    '        )\n'
    '        next_state_role = next_state.get("role", "user") if next_state else "user"\n'
    '\n'
    '        # --- openclaw-rl-metaclaw (temporary, safe to remove) ---\n'
    '        _metaclaw_verdict = None\n'
    '        try:\n'
    '            _metaclaw_parsed = json.loads(next_state_text) if next_state_text else None\n'
    '        except (TypeError, ValueError):\n'
    '            _metaclaw_parsed = None\n'
    '        if isinstance(_metaclaw_parsed, dict) and _metaclaw_parsed.get("metaclaw_verdict") is True:\n'
    '            _metaclaw_verdict = _metaclaw_parsed\n'
    '\n'
    '        if _metaclaw_verdict is not None:\n'
    '            eval_score = float(_metaclaw_verdict.get("eval_score", 0.0))\n'
    '            _metaclaw_hint = (_metaclaw_verdict.get("hint") or "").strip()\n'
    '            logger.info(\n'
    '                "%s[openclaw-rl-metaclaw-deterministic-reward] session=%s turn=%d "\n'
    '                "checker eval_score=%.1f hint_len=%d%s",\n'
    '                _CYAN, session_id, turn_num, eval_score, len(_metaclaw_hint), _RESET,\n'
    '            )\n'
    '            votes = (\n'
    '                [{"vote_id": 0, "score": 1, "hint": _metaclaw_hint, "raw": ""}]\n'
    '                if len(_metaclaw_hint) > 10 else []\n'
    '            )\n'
    '        elif turn_data.get("metaclaw_round_mode"):\n'
    '            _step_judge_msgs = _build_metaclaw_step_judge_messages(\n'
    '                turn_data["prompt_text"],\n'
    '                turn_data["response_text"],\n'
    '                next_state_text,\n'
    '            )\n'
    '            if self._prm_tokenizer:\n'
    '                _step_judge_prompt = self._prm_tokenizer.apply_chat_template(\n'
    '                    _step_judge_msgs, tokenize=False, add_generation_prompt=True,\n'
    '                )\n'
    '            else:\n'
    '                _step_judge_prompt = "\\n".join(m["content"] for m in _step_judge_msgs)\n'
    '            async with self._teacher_lp_semaphore:\n'
    '                _step_raw = await asyncio.gather(\n'
    '                    *[self._query_prm_eval_once(_step_judge_prompt, i) for i in range(self._prm_m)]\n'
    '                )\n'
    '            eval_score = _prm_eval_majority_vote(_step_raw)\n'
    '            logger.info(\n'
    '                "%s[openclaw-rl-metaclaw-step-judge] session=%s turn=%d "\n'
    '                "intermediate step votes=%s -> eval_score=%.1f%s",\n'
    '                _CYAN, session_id, turn_num,\n'
    '                [s if s is not None else "fail" for s in _step_raw],\n'
    '                eval_score, _RESET,\n'
    '            )\n'
    '            return {\n'
    '                "accepted": False,\n'
    '                "teacher_tokens_candidates": None,\n'
    '                "hint": "",\n'
    '                "hints": [],\n'
    '                "votes": [],\n'
    '                "eval_score": eval_score,\n'
    '            }\n'
    '        else:\n'
    '            judge_msgs = _build_hint_judge_messages(\n'
    '                turn_data["response_text"], next_state_text, next_state_role,\n'
    '            )\n'
    '            if self._prm_tokenizer:\n'
    '                judge_prompt = self._prm_tokenizer.apply_chat_template(\n'
    '                    judge_msgs, tokenize=False, add_generation_prompt=True,\n'
    '                )\n'
    '            else:\n'
    '                judge_prompt = "\\n".join(m["content"] for m in judge_msgs)\n'
    '\n'
    '            votes = await asyncio.gather(\n'
    '                *[self._query_judge_once(judge_prompt, i) for i in range(self._prm_m)]\n'
    '            )\n'
    '\n'
    '        # PRM eval branch (unchanged from parent, only runs for non-MetaClaw\n'
    '        # turns -- the metaclaw_verdict branch above already assigned eval_score).\n'
    '        if self._eval_mode and _metaclaw_verdict is None:\n'
)
text = text.replace(opd_evaluate_head_old, opd_evaluate_head_new, 1)

opd_evaluate_else_old = (
    '        else:\n'
    '            eval_score = None\n'
)
if text.count(opd_evaluate_else_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the eval_score=None "
        f"else block in {src_path}, found {text.count(opd_evaluate_else_old)} "
        "(official file may have changed upstream -- update this patch)"
    )
opd_evaluate_else_new = (
    '        elif _metaclaw_verdict is None:\n'
    '            eval_score = None\n'
)
text = text.replace(opd_evaluate_else_old, opd_evaluate_else_new, 1)

if "\nimport json\n" not in text:
    text = text.replace("import logging\n", "import json\nimport logging\n", 1)

import_block_old = (
    'from openclaw_opd_api_server import (\n'
    '    _append_hint_to_messages,\n'
    '    _build_hint_judge_messages,\n'
    '    _build_prm_eval_prompt,\n'
    '    _flatten_message_content,\n'
    '    _normalize_messages_for_template,\n'
    '    _prm_eval_majority_vote,\n'
    ')\n'
)
if import_block_old not in text:
    raise SystemExit(
        "patch failed: expected openclaw_opd_api_server import block not found "
        "in openclaw_combine_select_api_server.py (official file may have changed "
        "upstream -- update this patch)"
    )
import_block_new = (
    'from openclaw_opd_api_server import (\n'
    '    _append_hint_to_messages,\n'
    '    _build_hint_judge_messages,\n'
    '    _build_metaclaw_step_judge_messages,\n'
    '    _build_prm_eval_prompt,\n'
    '    _flatten_message_content,\n'
    '    _normalize_messages_for_template,\n'
    '    _prm_eval_majority_vote,\n'
    ')\n'
)
text = text.replace(import_block_old, import_block_new, 1)

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
    '            # --- openclaw-rl-skip-forced-negative-override (2026-08-13, temporary\n'
    '            # diagnostic experiment, see docs/issues_log.md 2026-08-13 entry) ---\n'
    '            # Captured BEFORE any of the three override blocks below run, so it\n'
    '            # reflects the PRM\'s own original judgment, not an already-overridden\n'
    '            # value (matters if more than one override condition fires on the same\n'
    '            # turn -- want "was this originally +1" not "was it +1 a moment ago").\n'
    '            _original_eval_score = eval_score\n'
    '            # --- openclaw-rl-invalid-tool-use-penalty (temporary, safe to remove) ---\n'
    '            if turn_data.get("is_invalid_tool_use"):\n'
    '                logger.info(\n'
    '                    "%s[openclaw-rl-invalid-tool-use-penalty] session=%s turn=%d "\n'
    '                    "known-invalid tool use -- overriding eval_score %.1f -> -1.0%s",\n'
    '                    _CYAN, session_id, turn_num, eval_score, _RESET,\n'
    '                )\n'
    '                eval_score = -1.0\n'
    '\n'
    '            # --- openclaw-rl-tool-error-penalty (2026-07-28, temporary, safe to remove) ---\n'
    '            # docs/issues_log.md 2026-07-28 条目。_build_prm_eval_prompt() 自己写的\n'
    '            # 规则里本来就包含"环境返回 error/failure -> 该打 -1"，但真实数据证实\n'
    '            # LLM 判官不总是照着自己这条规则执行（比如一次连续 3 次 edit 失败、\n'
    '            # 每次工具原样返回 {"status": "error", ...} 的场景里，判官投票并未\n'
    '            # 稳定打出 -1）。这里不再依赖判官，直接从 next_state 本身检测：\n'
    '            # 只要这一步的环境反馈是一个 status=="error" 的工具结果，不管是\n'
    '            # 哪个工具（edit/write/message/...)产生的，直接强制 -1，通用、不挑\n'
    '            # 工具，是对判官自己规则的代码层兜底，不是新增规则。\n'
    '            if next_state_role == "tool":\n'
    '                try:\n'
    '                    _next_state_parsed = json.loads(next_state_text)\n'
    '                except (TypeError, ValueError):\n'
    '                    _next_state_parsed = None\n'
    '                if isinstance(_next_state_parsed, dict) and _next_state_parsed.get("status") == "error":\n'
    '                    logger.info(\n'
    '                        "%s[openclaw-rl-tool-error-penalty] session=%s turn=%d "\n'
    '                        "tool result status=error -- overriding eval_score %.1f -> -1.0%s",\n'
    '                        _CYAN, session_id, turn_num, eval_score, _RESET,\n'
    '                    )\n'
    '                    eval_score = -1.0\n'
    '\n'
    '            # --- openclaw-rl-truncation-penalty (2026-08-06, temporary, safe to remove) ---\n'
    '            # docs/issues_log.md 2026-08-06 条目。顶格截断（finish_reason==length）时\n'
    '            # 模型还没说完就被生成上限切断，不能代表这是一次完整、正确的回答——即使\n'
    '            # 判官因为看到的内容凑巧"看起来还行"给了正分也不该采信。已用真实数据实锤\n'
    '            # 过至少 2 条顶格样本被误判 +1（rtok=8197，同一句话原样重复 12/45 次，\n'
    '            # 明显是空转被切断，不是正常完成）。这条独立于"同句原样重复 >= N 次"这条\n'
    '            # 候选规则（后者的安全阈值还没标定，尚未启用，见同一条 issues_log 记录）。\n'
    '            if turn_data.get("is_truncated"):\n'
    '                logger.info(\n'
    '                    "%s[openclaw-rl-truncation-penalty] session=%s turn=%d "\n'
    '                    "truncated (finish_reason=length) -- overriding eval_score %.1f -> -1.0%s",\n'
    '                    _CYAN, session_id, turn_num, eval_score, _RESET,\n'
    '                )\n'
    '                eval_score = -1.0\n'
    '\n'
    '            # --- openclaw-rl-skip-forced-negative-override (2026-08-13, temporary\n'
    '            # diagnostic experiment) --- see docs/issues_log.md 2026-08-13 entry for\n'
    '            # the full investigation (170852 vs 160713/143003 batch-composition\n'
    '            # comparison). Real data shows the decisive factor in whether a run\n'
    '            # "unlocks" clean turn-1s is whether these specific samples -- PRM\n'
    '            # originally scored the turn +1 (the file usually did get written\n'
    '            # correctly), but one of the three overrides above forced it to -1\n'
    '            # because of tool-decision spinning (repeat/truncation), not because\n'
    '            # the content was actually bad -- happen to pile into the same async\n'
    '            # training batch. This is a plain predicate on the before/after values,\n'
    '            # not a rule-name allowlist: deliberately does NOT special-case which of\n'
    '            # the three overrides fired, and deliberately does NOT touch a genuine\n'
    '            # PRM-native -1 (real style-rewrite feedback -- Table 3\'s actual signal,\n'
    '            # must stay) or a 0.0 -> -1.0 transition (not "was +1", seen in 160713).\n'
    '            _skip_forced_negative_override = (\n'
    '                _original_eval_score in (1.0, 1) and eval_score == -1.0\n'
    '            )\n'
    '            if _skip_forced_negative_override:\n'
    '                logger.info(\n'
    '                    "%s[openclaw-rl-skip-forced-negative-override] session=%s "\n'
    '                    "turn=%d original=+1 final=-1 -- will not be submitted to "\n'
    '                    "OPD or RL%s",\n'
    '                    _CYAN, session_id, turn_num, _RESET,\n'
    '                )\n'
)
text = text.replace(eval_score_old, eval_score_new, 1)

# ---------------------------------------------------------------------
# 2026-08-11 补丁：openclaw-rl-repeat-thinking-hint
#
# 背景（docs/issues_log.md 2026-08-11 条目）：规则 5（同句原样重复 >=
# _SENTENCE_REPEAT_INVALID_THRESHOLD 次强制判 -1）本身阈值已经用真实数据
# 校准过、没有误伤好样本，但连续两轮训练发现它仍然高频触发——诊断结论是
# 阈值没问题，是"信用分配"太糊：负分打在整个 turn 上（哪怕 write 已经
# 成功、PRM 也投了 +1），模型学不到"具体是因为哪句话复读了才被打分"，
# 更容易学成"这种写文件上下文倒霉"而不是"别在 thinking 里复读"。
#
# 这里用 OPD 现成的 hint 机制补一条更贴原因的信号：_append_hint_to_messages()
# 只是把一段文字拼进最后一条 user 消息、再对同一段 response_text 重新算一遍
# teacher 分布，不要求这段 hint 来自 PRM 投票。命中 Rule 5 时，直接把这次
# turn 的 accepted 列表整体替换成一条写死的"别复读"提醒，而不是继续用
# PRM 自己投的（很可能文不对题，因为判官提示词根本不知道要查复读）hint——
# 这正好堵上"PRM 不投复读 -> accepted 为空 -> OPD 完全没有该 turn 样本、
# 只剩 GRPO 的 -1"这个缺口。
#
# 替换必须放在 accepted = accepted[: _max_cand()] 截断之后、
# if not accepted: 判定之前，否则命中 Rule 5 但 PRM 恰好也没投出 hint 的
# turn 仍然会被当成"no valid hint"整条丢弃，起不到补信号的作用。
#
# 只针对 is_repeat_thinking_violation（Rule 5 专属标记，跟 1-5 通用的
# is_invalid_tool_use 分开，见 prepare_patched_openclaw_opd.sh 里的对应
# 改动）生效，Rule 1-4 命中的 turn 不受影响，仍然只有 eval_score 强制
# -1（上面那段补丁），没有强制 hint。
#
# 这是主动设计的新机制，"罚 + 教"逻辑上说得通（hint 条件化的 teacher 在
# 复读发生的具体 token 位置概率会明显被压低，天然是逐 token 定位的，不需要
# 额外去找复读句子的 token 区间），但是否真的让模型学会不复读，需要下一轮
# 训练之后用真实 shadow 数据验证，不是这次改动就能保证的。
# ---------------------------------------------------------------------
accepted_cap_old = (
    '        accepted.sort(key=lambda v: len(v["hint"]))\n'
    '        accepted = accepted[: _max_cand()]\n'
)
if text.count(accepted_cap_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the accepted-cap block "
        f"in {src_path}, found {text.count(accepted_cap_old)} (official file may "
        "have changed upstream -- re-verify this patch)"
    )
accepted_cap_new = accepted_cap_old + (
    '\n'
    '        # --- openclaw-rl-repeat-thinking-hint (temporary, safe to remove) ---\n'
    '        if turn_data.get("is_repeat_thinking_violation"):\n'
    '            logger.info(\n'
    '                "%s[openclaw-rl-repeat-thinking-hint] session=%s turn=%d "\n'
    '                "sentence-repeat violation -- overriding accepted hint(s) with "\n'
    '                "fixed repetition-reminder hint%s",\n'
    '                _CYAN, session_id, turn_num, _RESET,\n'
    '            )\n'
    '            accepted = [{\n'
    '                "score": 1,\n'
    '                "hint": (\n'
    '                    "You repeated the exact same sentence in your reasoning many "\n'
    '                    "times instead of making progress. Do not restate the same "\n'
    '                    "sentence or idea again -- after thinking something once, move "\n'
    '                    "on to the next step or give your final answer."\n'
    '                ),\n'
    '            }]\n'
)
text = text.replace(accepted_cap_old, accepted_cap_new, 1)

# ---------------------------------------------------------------------
# openclaw-rl-skip-forced-negative-override (2026-08-13, temporary
# diagnostic experiment): carry the flag computed above out of
# _opd_evaluate() via its return dict, since the dispatch point that
# decides whether to submit (_maybe_submit_ready_samples, in the parent
# class's file openclaw_combine_api_server.py) only sees this function's
# return value (`opd_result`), not its local variables. Both return points
# (accepted-hint path and no-valid-hint path) need it -- either one could
# be reached by a turn where the eval_score override fired.
# ---------------------------------------------------------------------
return_no_hint_old = (
    '            return {\n'
    '                "accepted": False,\n'
    '                "teacher_tokens_candidates": None,\n'
    '                "hint": "",\n'
    '                "hints": [],\n'
    '                "votes": votes,\n'
    '                "eval_score": eval_score,\n'
    '            }\n'
)
if text.count(return_no_hint_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the no-valid-hint "
        f"return dict in {src_path}, found {text.count(return_no_hint_old)} "
        "(official file may have changed upstream -- re-verify this patch)"
    )
return_no_hint_new = (
    '            return {\n'
    '                "accepted": False,\n'
    '                "teacher_tokens_candidates": None,\n'
    '                "hint": "",\n'
    '                "hints": [],\n'
    '                "votes": votes,\n'
    '                "eval_score": eval_score,\n'
    '                "skip_forced_negative_override": _skip_forced_negative_override,\n'
    '            }\n'
)
text = text.replace(return_no_hint_old, return_no_hint_new, 1)

return_accepted_old = (
    '        return {\n'
    '            "accepted": True,\n'
    '            "teacher_tokens_candidates": candidates,\n'
    '            "hint": hints[0],   # for log-line back-compat with parent\n'
    '            "hints": hints,\n'
    '            "votes": votes,\n'
    '            "eval_score": eval_score,\n'
    '        }\n'
)
if text.count(return_accepted_old) != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of the accepted return "
        f"dict in {src_path}, found {text.count(return_accepted_old)} "
        "(official file may have changed upstream -- re-verify this patch)"
    )
return_accepted_new = (
    '        return {\n'
    '            "accepted": True,\n'
    '            "teacher_tokens_candidates": candidates,\n'
    '            "hint": hints[0],   # for log-line back-compat with parent\n'
    '            "hints": hints,\n'
    '            "votes": votes,\n'
    '            "eval_score": eval_score,\n'
    '            "skip_forced_negative_override": _skip_forced_negative_override,\n'
    '        }\n'
)
text = text.replace(return_accepted_old, return_accepted_new, 1)

if "\nimport json\n" not in text:
    text = text.replace("import logging\n", "import json\nimport logging\n", 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched -> {dest_path}")
PY

python3 -m py_compile "${DEST}"
echo "已生成 openclaw_combine_select_api_server.py 调试补丁（openclaw-rl-debug-turn-content）: ${DEST}"
