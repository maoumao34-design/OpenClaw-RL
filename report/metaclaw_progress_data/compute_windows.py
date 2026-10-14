# -*- coding: utf-8 -*-
"""MetaClaw 进展页的窗口汇总（按题数加总，不是日均）。

规则见 ../metaclaw进展_新增四页方案.md「计分、窗口与口径规则」：
  - 没跑到的天不在 CSV 里；最后一天没跑完（partial=1）整天排除
  - 某趟没完整覆盖的窗口返回 None（表里留"—"）
  - 起病日 = 当天 copies>=8 的 main turn 占比 >= 25% 的第一天
  - Compl 用 passed_all = fc_pass / n_run（与已汇报 PPT 第 5 页同口径）

用法：python compute_windows.py [CSV 目录，默认本脚本所在目录]
"""
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["B0", "B0'", "H0", "OPD", "RL", "ENT", "KL", "MASK", "ONP"]
ONSET_THR = 0.25


def load(d=HERE):
    runs = {}
    for f in glob.glob(os.path.join(d, "*_daily.csv")):
        tag = os.path.basename(f).split("_")[0]
        with open(f, encoding="utf-8") as fh:
            runs[tag] = {int(r["day"]): r for r in csv.DictReader(fh)}
    return runs


def last_full(runs, tag):
    return max(d for d, r in runs[tag].items() if r["partial"] == "0")


def onset(runs, tag, thr=ONSET_THR):
    for d in sorted(runs[tag]):
        r = runs[tag][d]
        if r["partial"] == "0" and r["copies_ge8_frac"] and float(r["copies_ge8_frac"]) >= thr:
            return d
    return None


def agg(runs, tag, a, b):
    """day a..b 闭区间汇总；该趟没完整覆盖则返回 None。"""
    rows = [runs[tag].get(d) for d in range(a, b + 1)]
    if any(r is None or r["partial"] == "1" for r in rows):
        return None
    s = lambda k: sum(float(r[k]) for r in rows)
    turns = s("main_turns")
    return dict(acc=100 * s("acc_correct") / s("n_run"),
                pa=100 * s("fc_pass") / s("n_run"),
                cf=100 * s("fc_pass") / s("n_fc"),
                rep=100 * s("copies_ge8") / turns if turns else float("nan"),
                drop=int(s("dropped")), to=int(s("idle_timeout")), n=int(s("n_run")))


GROUPS = {
    "消融": (["B0", "B0'", "H0", "OPD", "RL"],
             [(1, 3), (4, 4), (5, 8), (9, 11), (12, 20), (21, 27)]),
    "正则": (["B0", "B0'", "H0", "ENT", "KL"],
             [(1, 5), (6, 8), (9, 17), (18, 20)]),
    "机制·遮罩": (["B0", "B0'", "OPD", "MASK"], [(1, 3), (4, 4), (5, 18)]),
    "机制·同策略": (["B0", "B0'", "H0", "ONP"], [(1, 7), (8, 8), (9, 9)]),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = sys.argv[1] if len(sys.argv) > 1 else HERE
    runs = load(d)

    print("== 每趟：最后完整天 / 成批复读首日(copies≥8 占比≥25%) / 首个DROPPED / 首个 Acc=0 天")
    for t in ORDER:
        fd = next((x for x in sorted(runs[t]) if int(runs[t][x]["dropped"]) > 0), None)
        z = next((x for x in sorted(runs[t])
                  if runs[t][x]["partial"] == "0" and float(runs[t][x]["acc"]) == 0), None)
        print(f"{t:5s} last_full=day{last_full(runs, t):02d}  onset={onset(runs, t)}  "
              f"first_dropped={fd}  first_acc0={z}")

    print("\n== day01（各趟 step 0 之前，权重都是初始模型）Acc / passed_all")
    for t in ORDER:
        r = runs[t][1]
        print(f"{t:5s} acc={100*float(r['acc']):5.1f}  passed_all={100*float(r['passed_all']):5.1f}  "
              f"train_step_end='{r['train_step_end']}'")

    for title, (tags, wins) in GROUPS.items():
        print(f"\n== {title}  (Acc / passed_all / compl_fc ; 复读% ; DROPPED ; idle_TO)")
        for a, b in wins:
            print(f"  day{a:02d}-{b:02d}")
            for t in tags:
                x = agg(runs, t, a, b)
                if x is None:
                    print(f"     {t:5s}   —（未覆盖）")
                else:
                    print(f"     {t:5s} {x['acc']:5.1f} / {x['pa']:5.1f} / {x['cf']:5.1f} ; "
                          f"{x['rep']:5.1f}% ; {x['drop']:3d} ; {x['to']:3d}  (n={x['n']})")


if __name__ == "__main__":
    main()
