# -*- coding: utf-8 -*-
"""Window aggregates for MetaClaw progress slides (question-sum, not daily mean)."""
import csv, glob, os, sys

sys.stdout.reconfigure(encoding="utf-8")
D = sys.argv[1]
runs = {}
for f in glob.glob(os.path.join(D, "*_daily.csv")):
    tag = os.path.basename(f).split("_")[0]
    with open(f, encoding="utf-8") as fh:
        runs[tag] = {int(r["day"]): r for r in csv.DictReader(fh)}

def last_full(tag):
    return max(d for d, r in runs[tag].items() if r["partial"] == "0")

def onset(tag, thr=0.25):
    for d in sorted(runs[tag]):
        r = runs[tag][d]
        if r["partial"] == "0" and r["copies_ge8_frac"] and float(r["copies_ge8_frac"]) >= thr:
            return d
    return None

def agg(tag, a, b):
    rows = [runs[tag].get(d) for d in range(a, b + 1)]
    if any(r is None or r["partial"] == "1" for r in rows):
        return None  # run didn't fully cover this window
    s = lambda k: sum(float(r[k]) for r in rows)
    return dict(acc=100 * s("acc_correct") / s("n_run"),
                pa=100 * s("fc_pass") / s("n_run"),
                cf=100 * s("fc_pass") / s("n_fc"),
                rep=100 * s("copies_ge8") / s("main_turns") if s("main_turns") else float("nan"),
                drop=int(s("dropped")), to=int(s("idle_timeout")), n=int(s("n_run")))

print("== 每趟：最后完整天 / 成批复读首日(copies≥8 占比≥25%) / 首个DROPPED / 首个 Acc=0 天")
for t in ["B0", "B0'", "H0", "OPD", "RL", "ENT", "KL", "MASK", "ONP"]:
    fd = next((d for d in sorted(runs[t]) if int(runs[t][d]["dropped"]) > 0), None)
    z = next((d for d in sorted(runs[t]) if runs[t][d]["partial"] == "0" and float(runs[t][d]["acc"]) == 0), None)
    print(f"{t:5s} last_full=day{last_full(t):02d}  onset={onset(t)}  first_dropped={fd}  first_acc0={z}")

print("\n== day01（各趟 step 0 之前，权重都是初始模型）Acc / passed_all")
for t in ["B0", "B0'", "H0", "OPD", "RL", "ENT", "KL", "MASK", "ONP"]:
    r = runs[t][1]
    print(f"{t:5s} acc={100*float(r['acc']):5.1f}  passed_all={100*float(r['passed_all']):5.1f}  train_step_end='{r['train_step_end']}'")

def table(title, tags, wins):
    print(f"\n== {title}  (Acc / passed_all / compl_fc ; 复读% ; DROPPED ; idle_TO)")
    for a, b in wins:
        print(f"  day{a:02d}-{b:02d}")
        for t in tags:
            x = agg(t, a, b)
            if x is None:
                print(f"     {t:5s}   —（未覆盖）")
            else:
                print(f"     {t:5s} {x['acc']:5.1f} / {x['pa']:5.1f} / {x['cf']:5.1f} ; {x['rep']:5.1f}% ; {x['drop']:3d} ; {x['to']:3d}  (n={x['n']})")

table("消融", ["B0", "B0'", "H0", "OPD", "RL"],
      [(1, 3), (4, 4), (5, 8), (9, 11), (12, 20), (21, 27), (1, 4), (1, 20)])
table("正则", ["B0", "B0'", "H0", "ENT", "KL"],
      [(1, 5), (6, 8), (9, 17), (18, 20), (1, 20)])
table("机制·遮罩", ["B0", "B0'", "OPD", "MASK"], [(1, 3), (4, 4), (5, 18), (1, 4)])
table("机制·同策略", ["B0", "B0'", "H0", "ONP"], [(1, 7), (8, 8), (9, 9), (1, 9)])
