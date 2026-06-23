#!/usr/bin/env python3
"""
check_convergence.py

Post-processes openclaw-test output files to measure the Table 3 convergence metric:
the minimum session number where the policy's first-message response satisfies the
user preference rule for 3 consecutive sessions.

Rules (paper Section 4.1 p.10, verbatim from PDF):
  Student:  no bold (**), no numbered list (^\d+.), no \boxed{}
  TA:       word count > 100
  Teacher:  contains 'well done', 'excellent', or 'great job' (case-insensitive)

The output files are written by student_chat.py / TA_chat.py / teacher_chat.py via their
--output flag. Each script appends one block per session:

    [session: student-hw-0-12345]
    <first-message response text>

    [session: student-hw-1-12345]
    ...

Since each script call CLEARS its output file on start, train_with_services.sh accumulates
responses across rounds by appending each round's output to a master file
(results_student_all.txt etc.) before the next round begins.

Usage:
  python check_convergence.py \\
      --student  /path/to/results_student_all.txt \\
      --ta       /path/to/results_TA_all.txt \\
      --teacher  /path/to/results_teacher_all.txt

Returns:
  Exit code 0 on success, 1 if any required file is missing.
"""

import argparse
import re
import sys

CONSECUTIVE_NEEDED = 3
SESSION_LIMIT = 72


# ---------------------------------------------------------------------------
# Convergence rules (paper Section 4.1, p.10)
# ---------------------------------------------------------------------------

def satisfies_student(response: str) -> bool:
    """No bold, no numbered list, no \\boxed{}."""
    return not re.search(r'\*\*|^\d+\.|\bboxed\{', response, re.MULTILINE)


def satisfies_ta(response: str) -> bool:
    """Response word count > 100."""
    return len(response.split()) > 100


def satisfies_teacher(response: str) -> bool:
    """Contains warm phrase."""
    lower = response.lower()
    return any(phrase in lower for phrase in ['well done', 'excellent', 'great job'])


RULES = {
    'student': satisfies_student,
    'ta':      satisfies_ta,
    'teacher': satisfies_teacher,
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_output_file(path: str) -> list[str]:
    """
    Parse a results_*_all.txt file into an ordered list of responses.
    Returns empty list if file not found.
    """
    try:
        with open(path, encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: file not found: {path}", file=sys.stderr)
        return []

    # Split on [session: ...] markers; first block before first marker is empty
    blocks = re.split(r'\[session:[^\]]+\]\n', content)
    return [b.strip() for b in blocks if b.strip()]


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------

def find_convergence(responses: list[str], rule_fn) -> int | None:
    """
    Scan responses in order. Return the 1-indexed session number at which
    3 consecutive sessions pass the rule. Return None if never converges.
    """
    consecutive = 0
    for i, response in enumerate(responses):
        if rule_fn(response):
            consecutive += 1
            if consecutive >= CONSECUTIVE_NEEDED:
                return i + 1   # 1-indexed session number
        else:
            consecutive = 0
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure Table 3 convergence from openclaw-test output files'
    )
    parser.add_argument('--student', required=True, metavar='FILE',
                        help='Accumulated student first-response file')
    parser.add_argument('--ta',      required=True, metavar='FILE',
                        help='Accumulated TA first-response file')
    parser.add_argument('--teacher', required=True, metavar='FILE',
                        help='Accumulated teacher first-response file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-session pass/fail for each persona')
    args = parser.parse_args()

    missing = [p for p in [args.student, args.ta, args.teacher]
               if not __import__('os').path.exists(p)]
    if missing:
        for p in missing:
            print(f"Error: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Table 3 Convergence Results")
    print("=" * 60)

    conv_values: list[float] = []

    for persona, path in [('Student', args.student), ('TA', args.ta), ('Teacher', args.teacher)]:
        key = persona.lower()
        responses = parse_output_file(path)
        n = len(responses)
        rule_fn = RULES[key]
        conv = find_convergence(responses, rule_fn)

        if args.verbose:
            print(f"\n  [{persona}] per-session results ({n} sessions):")
            consec = 0
            for i, r in enumerate(responses):
                passed = rule_fn(r)
                consec = consec + 1 if passed else 0
                marker = " ← CONVERGED" if consec >= CONSECUTIVE_NEEDED and conv == i + 1 else ""
                print(f"    session {i+1:3d}: {'PASS' if passed else 'fail'} (consec={consec}){marker}")

        if conv is not None:
            conv_values.append(conv)
            print(f"  {persona:8s}: converged at session {conv:3d}  (checked {n} sessions)")
        else:
            print(f"  {persona:8s}: NOT converged  (checked {n}/{SESSION_LIMIT} sessions)")

    print()
    if conv_values:
        avg = sum(conv_values) / len(conv_values)
        print(f"  Average : {avg:.1f}  ({len(conv_values)}/3 personas converged)")
    else:
        print("  Average : N/A  (no persona converged)")
    print("=" * 60)


if __name__ == '__main__':
    main()
