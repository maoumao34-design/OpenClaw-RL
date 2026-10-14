# MetaClaw 迁移实验逐日指标

取数日期：2026-10-14。只读现有日志，没有改训练代码，也没有重跑。

九趟日志都在 `/dfs/data/openclaw-rl-project/logs/metaclaw_migration_<RunID>/`（`training.log`、`metaclaw_rollout.log`；跑完的三趟另有 `report/report.json`）。`n_q` 来自 `all_tests.json` 指向的 `eval/dayNN/questions.json`。

没跑到的天没有补行。没跑到的轮不进任何分母。`acc` 的分母是 `n_run`，`compl_fc` 的分母是当天实际跑过的 file_check 题数。

## 1. 停在哪、为什么停

启动脚本在 `wait` driver 之后**无条件**打印 `day01→day30 全部跑完，主动停止训练...`，然后 `kill` 训练进程（`scripts/metaclaw/run_metaclaw_migration_modelfactory.sh`）。这条字样不能当作跑完。

没跑完的六趟是同一种退出形态：rollout 里没有 Python traceback；下一行就是上面那句提示；`training.log` 最后一行仍是 Ray `Job is currently running`。脚本把 `wait` 的退出码丢掉了（`wait || true`），日志里没有信号名、没有 OOM。进程退出本身归为**外部原因**。模型在退出前是否已经复读崩掉，另记在下表。

| 代号 | 最后位置 | 最后完整天 | 停止 | 依据 |
|---|---|---|---|---|
| B0 | day30 r15 / 15 | day30 | 正常跑完 | 写出 `report.json`，汇总 Acc 41.1 / Compl 26.3，然后才是那句提示 |
| B0' | day30 r15 / 15 | day30 | 正常跑完 | 同上，44.7 / 25.9 |
| ENT | day30 r15 / 15 | day30 | 正常跑完 | 写出 report，Compl 14.7 |
| H0 | day21 r6 / 10 | day20 | 外部原因 | r6 判分后还有一次 chat `HTTP 200`，随后进程消失。退出前已是复读：r6 的 `max_sentence_copies=2954`，当天 6 轮全是 idle timeout，step 停在 15 |
| OPD | day05 r8 / 13 | day04 | 外部原因 | r8 判分（idle timeout，score 0）后还有 `HTTP 200`，随后进程消失。day05 已跑的 8 轮全是 idle timeout，15/15 个 main turn copies≥8 |
| RL | day28 r11 / 13 | day27 | 外部原因 | 最后一条 verdict 是 `official_score=N/A (infra failure)`，原文 `Compaction timed out`，然后进程消失。压缩超时从 day18 起成片，是模型把 transcript 撑爆之后的结果；进程退出本身没有异常栈 |
| KL | day22 r2 / 11 | day21 | 外部原因 | r2 判分后还有 `HTTP 200`，随后进程消失。day19 起每轮 idle timeout，step 停在 23 |
| MASK | day19 r2 / 11 | day18 | 外部原因 | r2 判分后是 verdict 的 `HTTP 200`，没有 Python 异常。work_log 2026-10-08 记为 driver 被信号终止。step 6 之后已经没有新 step |
| ONP | day10 r12 / 13 | day09 | 外部原因 | r12 判分 `official_score=0.800` 之后，SGLang 仍在为下一轮做 prefill（09:24:44），09:24:54 Ray 仍是 RUNNING，然后 driver 没了。不是复读把进程打死。缺的证据是平台侧的信号记录 |

## 2. 训练步数与 DROPPED

步数是 `training.log` 里 `step N:` 的个数，编号从 0 连续到最大步。两趟基线没有训练 step。

| 代号 | step 0 … | DROPPED |
|---|---|---|
| B0 | （无） | 0 |
| B0' | （无） | 0 |
| H0 | 15（16 步） | 93 |
| OPD | 4（5 步） | 9 |
| RL | 18（19 步） | 16 |
| ENT | 10（11 步） | 227 |
| KL | 23（24 步） | 38 |
| MASK | 6（7 步） | 105 |
| ONP | 8（9 步） | 3 |

## 3. 第 4 节核对

① 对得上。B0 41.1 / 26.3（142.036/346，file_check 59/224）。B0' 44.7 / 25.9（154.729/346，58/224）。

② 对得上。ENT 16.5 / 14.7，57.014/346，33/224。

③ 旧的 22.8 / 18.3 **对得上，但分母是全部 30 天**。跑完的 day01–day20 分子是 79.035 和 41；除以 346 和 224 就是 22.8 / 18.3。day21 只跑了 6/10，这 6 轮分数都是 0，day22–30 没跑。按这次的规则（只算实际跑过的题，含这 6 个 0 分）：**33.9 / 26.8**（79.035/233，41/153）。只算完整天、不含 day21：34.8 / 27.5（79.035/227，41/149）。

④ 对得上。窗口是题数加总，不是日均。

| 窗口 | B0 | H0 | KL |
|---|---|---|---|
| day01–13 | 52.3 / 41.5 | 53.2 / 43.6 | 61.7 / 45.7 |
| day14–20 | 53.5 / 36.4 | 1.6 / 0.0 | 16.9 / 1.8 |

⑤ Acc / Compl 九天都对得上。copies≥8 除 day08 外都对得上。**day08 对不上，CSV 没有改成核对表上的数。** 核对表是 11/98，按「只计 main turn」是 **10/95**。多出来的 3 条是 `session=unknown turn_type=side`，其中一条 copies=13；它们不是 `session=metaclaw-day08` 的 main turn，是旧计数沿用了上一个 session。

⑥ 对得上。MASK day04 为 5/19，day05 为 21/22。

⑦ 对得上。OPD 9、RL 16、H0 93、ENT 227、B0 0。

⑧ 第 5 页的 Compl 是 **`passed_all`**（当天 file_check 通过数 / 当天全部题数），不是 `compl_fc`。B0 day01：`passed_all` 30.0，`compl_fc` 50.0，Acc 两边都是 68.3。

B0 day01–08 的 `passed_all`：30.0、45.5、33.3、30.0、53.8、10.0、36.4、25.0。H0：20.0、36.4、33.3、20.0、53.8、10.0、63.6、25.0。窗口 day01–08 为 33.7 / 33.7，day09–20 为 21.0 / 8.0。正式 Compl 对不上这页（B0 day01–08 的 `compl_fc` 是 50.0、71.4、50.0、50.0、87.5、16.7、50.0、37.5；两个窗口是 52.6 / 52.6 和 31.5 / 12.0）。

要和第 5 页接上，新表用 `passed_all`。正式 file_check 通过率仍用 `compl_fc`。两列都留在 CSV 里。

## 4. 重建和 report

有 `report.json` 的是 B0、B0'、ENT。这三趟 CSV 用 report 里的全精度分数。用 rollout 上印出的三位小数字再加总，逐日和 report 比，Correct 差超过 0.05 或 file_check 通过数不一致的天数是 **0**。

没有 report 的是 H0、OPD、RL、KL、MASK、ONP。这六趟用 rollout 的 `official_score`（印到三位小数）重建，所以日 Acc 比全精度略有舍入。窗口核对仍落在 0.1 个百分点以内。

## 5. 取舍

- `partial = 1` 当且仅当 `n_run < n_q`。Acc、Compl、`passed_all` 的分母都不含没跑到的轮。`passed_all = fc_pass / n_run`。完整天里 `n_run = n_q`，因此和第 5 页的「全题平均」是同一个数。
- file_check 通过：分数 ≥ 0.999。有 report 的天，这个计数和 `metrics.passed is True` 的个数一致。
- idle timeout 只看该轮最终回复（`<< OpenClaw -> Query:` 到 `verdict:`）里有没有 `model idle timeout` 或 `LLM request timed out`。题面和 Previous Feedback 里的 “timed out” 不算。
- `compaction_to` 只看同一段最终回复里的 `Compaction timed out`。
- `infra_fail` 是 `official_score=N/A`。这些轮 Acc 记 0，并留在 `n_run` 里。
- `copies_*` 只计 `[openclaw-rl-shadow-sentence-repeat]` 且 `turn_type=main`、`session=metaclaw-dayNN` 的行。Ray 的 `[repeated Nx across cluster]` 按 N 倍计入。
- `train_step_end`：下一 day 在 `training.log` 里第一次出现的时间之前，已经打印出来的最大 `step N`。基线这一列留空。step 行如果落在下一天已经开始之后，就算到下一天，所以上一天可能是空的（ONP / OPD / KL / MASK / ENT 的 day01 就是这样：step 0 的时间戳比 day02 的第一行晚几秒到几十秒）。
- 部分最后一天如果两轮都是 idle timeout，shadow 日志里可以没有 main turn（MASK day19、KL day22 的 `main_turns=0`）。
