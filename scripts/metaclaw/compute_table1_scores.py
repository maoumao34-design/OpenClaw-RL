"""Compute Table 1-style Acc./Compl. scores from a `metaclaw-bench run` output directory.

MetaClaw paper (arXiv:2603.17187) Table 1 caption: "Acc.: mean per-question
accuracy. Compl.: file-check completion rate." (fulltext.txt line 333-334).

  Acc.    = mean `score` across ALL scoring.json files (both multi_choice and
            file_check question types combined) -- same definition
            benchmark/src/report/report_cmd.py's own "Accuracy" field already
            uses; this script just also surfaces it standalone without
            needing to parse reports.md.
  Compl.  = mean `score` restricted to question_type == "file_check" only
            (equivalently: fraction of file_check rounds with
            metrics["passed"] == true).

NOT computed anywhere in the official benchmark/src/ code -- verified by
grepping the whole tree for "completion", zero matches (see
docs/metaclaw_migration_plan.md "如何给任意一个 checkpoint 打分"). This
script does not reimplement any scoring logic -- it only aggregates the
`score`/`question_type` fields that the official
benchmark/src/scoring/scoring_cmd.py already computed and wrote into each
round's scoring.json.

Usage:
    python compute_table1_scores.py <metaclaw-bench run 的输出目录>

Requires scoring.json files to already exist under the given directory --
i.e. run `metaclaw-bench run` (or at least `metaclaw-bench scoring`) first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print(f"用法: python {sys.argv[0]} <metaclaw-bench run 输出目录>", file=sys.stderr)
        sys.exit(1)

    result_root = Path(sys.argv[1]).resolve()
    if not result_root.exists():
        print(f"错误：目录不存在: {result_root}", file=sys.stderr)
        sys.exit(1)

    scoring_files = sorted(result_root.rglob("scoring.json"))
    if not scoring_files:
        print(
            f"错误：{result_root} 下没有找到任何 scoring.json"
            "（先跑 `metaclaw-bench run`，或至少跑完 scoring 这一步）",
            file=sys.stderr,
        )
        sys.exit(1)

    all_scores: list[float] = []
    file_check_scores: list[float] = []
    skipped = 0

    for path in scoring_files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] 读取失败 {path}: {e}", file=sys.stderr)
            skipped += 1
            continue

        score = record.get("score")
        if not isinstance(score, (int, float)):
            print(f"[warn] {path} 没有有效的 score 字段，跳过", file=sys.stderr)
            skipped += 1
            continue

        all_scores.append(float(score))
        if record.get("question_type") == "file_check":
            file_check_scores.append(float(score))

    if not all_scores:
        print("错误：没有任何有效样本", file=sys.stderr)
        sys.exit(1)

    acc = sum(all_scores) / len(all_scores)
    print(f"输出目录: {result_root}")
    print(f"有效样本: {len(all_scores)}（跳过 {skipped} 个）")
    print()
    print(f"Acc.   (全部 {len(all_scores)} 题平均分)              = {acc:.1%}")

    if file_check_scores:
        compl = sum(file_check_scores) / len(file_check_scores)
        print(f"Compl. (仅 file_check {len(file_check_scores)} 题通过率) = {compl:.1%}")
    else:
        print("Compl.: 这批题目里没有 file_check 类型的题，无法计算")


if __name__ == "__main__":
    main()
