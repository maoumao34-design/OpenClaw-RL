#!/usr/bin/env python3
"""
Table 3 rule-based evaluation for OpenClaw-RL Personal Agent track.

Measures minimum sessions for 3 consecutive policy first-responses
to satisfy the user preference rule (paper Section 4.1).

Input file format (produced by student_chat.py / TA_chat.py / teacher_chat.py):
    [session: <session_id>]
    <first OpenClaw response (may be multi-line)>

    [session: <session_id>]
    <next response>
    ...

Usage:
    python evaluate_table3.py results_student.txt --scenario student
    python evaluate_table3.py results_TA.txt      --scenario ta
    python evaluate_table3.py results_teacher.txt --scenario teacher --verbose
"""

import re
import argparse


# ── Rule definitions (paper Section 4.1 / CLAUDE.md) ─────────────────────────

def satisfies_student(r: str) -> bool:
    """No bold / numbered list / \\boxed{} — response looks natural, not AI-formatted."""
    return not re.search(r'\*\*|^\d+\.|\\boxed\{', r, re.MULTILINE)

def satisfies_ta(r: str) -> bool:
    """Response > 100 words — sufficiently detailed grading comment."""
    return len(r.split()) > 100

def satisfies_teacher(r: str) -> bool:
    """Response contains warm words — friendly tone."""
    return any(w in r.lower() for w in ['well done', 'excellent', 'great job'])

RULES = {
    'student': satisfies_student,
    'ta':      satisfies_ta,
    'teacher': satisfies_teacher,
}


# ── Parser ────────────────────────────────────────────────────────────────────

def load_responses(path: str) -> list:
    """
    Parse results file. Each entry:
        [session: <id>]\\n<response text>\\n\\n
    Returns list of response strings in chronological order.
    """
    with open(path, encoding='utf-8') as f:
        content = f.read()

    responses = []
    for block in re.split(r'\n\n+', content.strip()):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if lines[0].startswith('[session:'):
            response = '\n'.join(lines[1:]).strip()
        else:
            response = block
        if response:
            responses.append(response)
    return responses


# ── Convergence ───────────────────────────────────────────────────────────────

def find_convergence(responses: list, rule, consecutive_needed: int = 3) -> int:
    """
    Return 1-indexed session number at which convergence is first reached
    (end of first streak of `consecutive_needed` consecutive passes).
    Returns -1 if not reached within the provided sessions.
    """
    streak = 0
    for i, r in enumerate(responses, start=1):
        if rule(r):
            streak += 1
            if streak >= consecutive_needed:
                return i
        else:
            streak = 0
    return -1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Table 3 rule-based evaluation (OpenClaw-RL Personal Agent).'
    )
    parser.add_argument('results_file',
                        help='Path to results file (results_student.txt / results_TA.txt / results_teacher.txt)')
    parser.add_argument('--scenario', choices=['student', 'ta', 'teacher'], required=True,
                        help='Evaluation scenario')
    parser.add_argument('--consecutive', type=int, default=3,
                        help='Consecutive passes needed for convergence (default: 3, per paper)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-session pass/fail detail')
    args = parser.parse_args()

    responses = load_responses(args.results_file)
    rule = RULES[args.scenario]

    print(f"Scenario    : {args.scenario}")
    print(f"Sessions    : {len(responses)}")
    print(f"Consecutive : {args.consecutive}")
    print()

    result = find_convergence(responses, rule, args.consecutive)
    if result == -1:
        print(f"Result      : NOT converged within {len(responses)} sessions")
    else:
        print(f"Result      : Converged at session {result}  ← Table 3 value")

    if args.verbose:
        print('\nPer-session detail:')
        streak = 0
        for i, r in enumerate(responses, start=1):
            passed = rule(r)
            streak = streak + 1 if passed else 0
            preview = r[:80].replace('\n', ' ')
            print(f"  [{i:3d}] {'PASS' if passed else 'FAIL'}  streak={streak}  {preview!r}")


if __name__ == '__main__':
    main()
