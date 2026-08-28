# -*- coding: utf-8 -*-
"""解析 Separate-Student 训练日志，按论文 Table 3 口径统计收敛情况。

Table 3 指标（见 CLAUDE.md）：
  - Student 规则：回复中不含 bold(**)、编号列表(行首 "N.")、\\boxed{}
  - 收敛条件：连续 3 个 session 中，policy 对**第一条消息**的回复均满足该规则
  - 报告数字：达到收敛所需的最少 session 数（越小越好）

用法：python analyze_log.py <日志路径>
"""

import re
import sys

PROBLEM_RE = re.compile(r"^# Problem (\d+) \(session: (\S+)\)", re.M)
TURN_RE = re.compile(r"^\s*Turn (\d+)/\d+\s*$", re.M)
REPLY_RE = re.compile(r"^\s*<< OpenClaw -> Student:\s*$", re.M)
DONE_RE = re.compile(r"^\s*Turn (\d+): Student confirmed problem \d+ is done!", re.M)

# Student 偏好规则：不要 bold / 编号列表 / \boxed{}
VIOLATION_RE = re.compile(r"\*\*|^\s*\d+\.\s|\\boxed\{", re.M)


def split_sessions(text):
    """按 Problem 切分，返回 [(编号, session_id, 正文), ...]。"""
    marks = list(PROBLEM_RE.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((int(m.group(1)), m.group(2), text[m.end():end]))
    return out


def first_turn_reply(body):
    """取 Turn 1 里 OpenClaw 的回复正文。"""
    turns = list(TURN_RE.finditer(body))
    if not turns:
        return None
    turn1 = body[turns[0].end(): turns[1].start() if len(turns) > 1 else len(body)]
    reply = REPLY_RE.search(turn1)
    return turn1[reply.end():].strip() if reply else None


def analyze(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    rows = []
    for num, sid, body in split_sessions(text):
        reply = first_turn_reply(body)
        done = DONE_RE.search(body)
        rows.append({
            "problem": num,
            "session": sid,
            "turns": int(done.group(1)) if done else None,
            "ok": bool(reply) and not VIOLATION_RE.search(reply),
            "reply": reply or "",
        })
    return rows


def converge_at(rows, window=3):
    """返回首次出现连续 window 个满足规则的 session 时，已用的 session 数。"""
    streak = 0
    for i, r in enumerate(rows, 1):
        streak = streak + 1 if r["ok"] else 0
        if streak == window:
            return i
    return None


def main():
    path = sys.argv[1]
    rows = analyze(path)

    print(f"共解析 {len(rows)} 个 session\n")
    print("session | 总turn数 | Turn1 是否满足偏好")
    print("-" * 42)
    for r in rows:
        print(f"{r['problem']:>7} | {str(r['turns'] or '-'):>8} | "
              f"{'满足' if r['ok'] else '不满足'}")

    n = converge_at(rows)
    ok_count = sum(r["ok"] for r in rows)
    turns = [r["turns"] for r in rows if r["turns"]]

    print("\n" + "=" * 42)
    print(f"Turn1 满足偏好的 session：{ok_count} / {len(rows)}")
    if turns:
        print(f"平均 turn 数：{sum(turns) / len(turns):.2f}"
              f"（最少 {min(turns)}，最多 {max(turns)}）")
    print(f"连续 3 个 session 收敛于：第 {n} 个 session" if n else "未收敛")


if __name__ == "__main__":
    main()
