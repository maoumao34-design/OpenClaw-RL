# CLI 数据查询：MetaClaw 迁移实验逐日指标（9 趟）

> 用途：给 mentor 的进展 PPT（接在《Openclaw-RL论文复现及迁移内容分享》第 5 页之后，新增 3 页对比实验）。页面方案见 [`metaclaw进展_新增四页方案.md`](metaclaw进展_新增四页方案.md)。
>
> **只读任务**：不改代码、不重跑任何训练，只从现有日志里取数。

---

## 1. 要查的 9 趟

| 代号 | Run ID | 角色 | 配置 |
|---|---|---|---|
| B0 | `20260907_112320` | 零训练基线（定版）| K=0，不训练 |
| B0' | `20260909_094645` | 零训练基线（复核）| 同上 |
| H0 | `20260914_181842` | 混合训练（**三组对比的公共对照**）| `W_RL=1`、`W_OPD=1`、熵 0、KL 0 |
| OPD | `20260917_163256` | 纯 OPD | `W_RL=0` |
| RL | `20260917_180600` | 纯 RL | `W_OPD=0` |
| ENT | `20260918_154523` | 混合 + 熵 | `METACLAW_ENTROPY_COEF=0.1` |
| KL | `20260927_040725` | 混合 + KL | `METACLAW_KL_LOSS_COEF=0.1` |
| MASK | `20260930_212143` | 纯 OPD + 遮罩 | `METACLAW_OPD_MASK_COMMIT=1`、`W_RL=0` |
| ONP | `20261012_072251` | 混合 + 同策略 | `METACLAW_ONPOLICY=1` |

训练趟的日志一般在 `/dfs/data/openclaw-rl-project/logs/metaclaw_migration_<RunID>/`（`training.log`、`metaclaw_rollout.log`、`report/`、`record_*_archive.jsonl`）。两趟基线若不在这个目录下，按实际位置找，**并在结果里写明实际路径**。

---

## 2. 输出：每趟一个逐日 CSV

文件名 `<代号>_<RunID>_daily.csv`，每个**出现过的 day** 一行（包括没跑完的最后一天，标注见 `partial` 列）。

| 列 | 定义 |
|---|---|
| `day` | 01–30 |
| `n_q` | 当天题数（按 `all_tests.json`；没跑完的天写实际跑到的轮数，并在 `partial` 标 1）|
| `partial` | 当天没跑完为 1，否则 0 |
| `acc` | **含部分分**的 Acc，与 `report.md` 同口径（例：B0 day02 = 8.8/11 = 80.3%）。**不要用 rollout 的 `passed=True` 离散计数** |
| `acc_correct` | Acc 的分子（含部分分的 Correct 合计）|
| `n_fc` | 当天 file_check 题数 |
| `fc_pass` | 当天 file_check 通过数（官方 `_score_file_check` 二值判分）|
| `compl_fc` | `fc_pass / n_fc` ——**正式 Compl 口径**（只看 file_check 子集，全 30 天共 224 题）|
| `passed_all` | `report.json` / `scoring.json` 里 `metrics.passed` 的全题平均（**不是 Compl**，只用于下面第 4 节的核对）|
| `main_turns` | 当天 main turn 数 |
| `copies_ge8` | 当天 `max_sentence_copies ≥ 8` 的 main turn 数（`[openclaw-rl-shadow-sentence-repeat]` 日志）|
| `copies_ge8_frac` | `copies_ge8 / main_turns` |
| `copies_max` | 当天 `max_sentence_copies` 最大值 |
| `idle_timeout` | **按本轮最终回复**是 idle timeout 的轮数（不要按"块内出现过 `timed out`"计，那样会把邻题算进来）|
| `compaction_to` | `Compaction timed out` 的轮数（主要是 RL 趟）|
| `infra_fail` | infra failure 轮数（`official_score=N/A`）|
| `dropped` | 当天 `DROPPED an oversized round` 次数 |
| `train_step_end` | 当天最后一轮结束时，已完成的最大训练 step（按时间戳对齐 `training.log`）。基线填空 |

### 没有 `report.json` 的趟

已知 H0 没有 `report/report.json`。这类趟按官方分从 rollout 重建 `acc` / `compl_fc`。**对有 report 的趟也用同一方法重建一遍**，和 report 逐日比对，差异写进汇总。

---

## 3. 输出：一份汇总 `summary.md`

1. **每趟最后跑到的 day / round，以及停止原因**（driver 异常、被信号终止、正常跑完等；已知 ONP 在 day10 中途 driver 死掉、启动脚本仍打印"全部跑完"，请顺带查死因）
2. **每趟训练总步数**和 **DROPPED 总数**
3. 第 4 节每一项核对的结果（对得上 / 对不上 + 差多少）
4. 重建方法与 report 的逐日比对结果
5. 任何定义上拿不准、做了取舍的地方，写明怎么取的

---

## 4. 核对项（用已知数字验证口径，对不上先报告、不要硬凑）

| # | 核对 | 期望值 | 来源 |
|---|---|---|---|
| ① | B0、B0' 30 天总 Acc / Compl | 41.1 / 26.3；44.7 / 25.9 | `training_config.md` |
| ② | ENT 30 天总 Acc / Compl / 正确数 / FC 通过数 | 16.5 / 14.7 / 57/346 / 33/224 | work_log 09-20 |
| ③ | H0 30 天总 Acc / Compl | 22.8 / 18.3（重建值）| work_log 09-20 |
| ④ | KL 与 H0、B0 的窗口 Acc / Compl | day01–13：B0 52.3/41.5、H0 53.2/43.6、KL 61.7/45.7；day14–20：B0 53.5/36.4、H0 1.6/0.0、KL 16.9/1.8 | work_log 09-28 |
| ⑤ | ONP day01–09 逐日 Acc / Compl / copies≥8 | 70.0/50.0、71.2/57.1、72.2/62.5、46.7/16.7、74.4/75.0、43.6/16.7、58.5/50.0、0.0/0.0、23.6/0.0；copies≥8：0/19、1/21、0/25、0/22、1/28、0/20、0/21、11/98、0/17 | work_log 10-13 |
| ⑥ | MASK day04、day05 的 copies≥8 | 5/19、21/22 | work_log 10-08 |
| ⑦ | DROPPED 总数 | OPD 9、RL 16、H0 93、ENT 227、B0 0 | work_log 09-18、09-20 |
| ⑧ | **已汇报 PPT 第 5 页的 Compl 用的是哪个口径**（见下）| — | — |

### ⑧ 需要特别查清：第 5 页的 Compl 和正式口径对不上

已汇报的 PPT 第 5 页（H0 vs B0，day01–08 逐日表）里，**B0 day01 的 Compl 是 30.0**；而 work_log 09-20 按"file_check 224 题通过率"列的 **B0 day01 Compl 是 50**。Acc 两边一致（68.3）。

第 5 页的逐日 Compl（基线 30.0 / 45.5 / 33.3 / 30.0 / 53.8 / 10.0 / 36.4 / 25.0，训练 20.0 / 36.4 / 33.3 / 20.0 / 53.8 / 10.0 / 63.6 / 25.0；窗口 day01–08 = 33.7 / 33.7、day09–20 = 21.0 / 8.0）分母看起来像当天全部题数而不是 file_check 题数，**怀疑用的是 `passed_all` 而不是 `compl_fc`**。

请用 B0、H0 两趟的 `compl_fc` 和 `passed_all` 分别算 day01–08 逐日值及两个窗口值，确认第 5 页对应的是哪一列。**新增页要和第 5 页能接上，所以这条决定新表用哪个口径**。

---

## 5. 结果放哪

CSV 和 `summary.md` 放进仓库 `openclaw-rl/report/metaclaw_progress_data/`，commit 并 push。本地拉下来后据此切窗口、填表。

不要提交任何原始日志，只提交上面这些汇总文件。
