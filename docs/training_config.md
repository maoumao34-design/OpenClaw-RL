[← 工作记录](work_log.md)

# MetaClaw 迁移：当前训练配置（唯一现状来源）

> **这份文档描述"现在跑起来会发生什么"，不记录历史推理。**
> 为什么这样设计、走过哪些弯路，去看 [`metaclaw_migration_plan.md`](metaclaw_migration_plan.md)。
>
> ## 维护规则：这是**快照**，不是实时镜像
>
> **更新时机：一次实验出结果、配置稳定下来之后**，把这一轮真正改动过的部分同步进来。
> **迭代过程中允许滞后**——为每次中间改动同步一遍是浪费，也会把校验测试养成"经常挂、习惯性忽略"。
>
> 更新时跑一次 `scripts/tests/test_training_config_doc.py`，它把本文档的关键数值逐条对着真实脚本核，
> **文档写错、或脚本改了没同步，都会挂**。这不是形式主义：本项目已经因为"注释停在旧版"
> 和"补丁打错层"各栽过一次。
>
> 所以：**测试挂 ≠ 出错**。如果你正在迭代中间，挂是预期的；只有在快照时刻它必须是绿的。

**快照对应：2026-09-08，commit `e7979e1`**（此后若有未同步的改动，以代码为准）

---

## 一、硬件拓扑（8 卡）

| 组件 | 卡数 | 并行 | 备注 |
|---|---|---|---|
| 训练 actor | **4** | `TP=4`、`PP=1`、**`CP=1`** | |
| SGLang rollout | 2 | 1 个 engine（2 卡/engine）| |
| PRM 判官 | 1 | 无 | |
| **OPD teacher** | **1** | **无 TP、无 CP** | ⚠️ 2026-09-08 OOM 发生在这张卡 |

**`CP=1` 意味着序列不分片**，长回复的全词表 logits 必须在单卡上放下。

---

## 二、服务与端口

```
训练进程（Ray head, dashboard :8265）
   └── RL training proxy :30000     ← 跑在训练进程里，两者同生共死
driver（day01 → day30，concurrency=1 严格串行）
每天一个临时 gateway 端口（_start_work_gateway 分配，非固定 18789）
```

**proxy 与训练同进程**——训练崩则 proxy 一起挂，driver 随后全是 network error。

---

## 三、一道题的完整流程

```
driver 拼 query = 本轮题目 + [Previous Feedback]（官方原文，无加料）
   ↓
openclaw agent 子进程（--agent metaclaw_agent，profile: coding 全套工具）
   ↓
agent 每次调模型 → :30000 proxy → SGLang
      每次请求 = 一个 turn，proxy 记 turn_data
      （prompt_ids / response_ids / prompt_text / response_text / messages）
   ↓
agent 结束 → driver 跑官方 checker → eval_score = ±1
   ↓
driver POST verdict（max_tokens=0，不生成）
      {metaclaw_verdict, eval_score, hint, round_id, task_prefix}
      session_done = 仅当天最后一轮
   ↓
proxy 认出 verdict → 整轮打包
```

**会话粒度：一天一个 session**，id = `metaclaw-{test_id}`，全天共用 transcript，上下文靠 OpenClaw 压缩兜住。**按题隔离已于 2026-09-07 整体删除。**

---

## 四、样本怎么形成

```
中间 tool-call turn ──→ HELD（门在 _fire_opd_task 最顶端 return）
                        不打分、不单独成样本
                        ⇒ 步骤判官因此不可达（有意为之）

verdict turn ─────────→ _metaclaw_submit_round
                          ├ 归属判定（两层，缺一不可）
                          │   ① _mc_lo < t < turn_num
                          │   ② turn 的 prompt 含本轮 task_prefix
                          │   不属于本轮 → 显式丢弃 + 告警，不并入
                          ├ 全部 turn 共用一个 group_index
                          ├ 每个 turn stamp metaclaw_round_turns = N
                          └ 一次性 output_queue.put((group_index, collect))
```

**每个 turn 一个样本，带自己那一刻真实的 prompt/response，零重建。**
（轨迹级方案已于 2026-09-03 被真实训练证伪并删除：`dropReasoningFromHistory` 让扁平轨迹无法忠实构造。）

**⚠️ 交还逻辑必须同时存在于父类和 Select 子类**——服务器实例化的是 `OpenClawCombineSelectAPIServer`，它覆写了两个 `_submit_*`。只打父类等于没打（2026-09-08 实测：held 98 次，`queued group=` 0 次）。

---

## 五、奖励与 advantage

| 层 | 内容 |
|---|---|
| reward | **本轮 checker 判定 ±1**，整轮所有 turn 共享 |
| 官方基线 | `--advantage-estimator grpo` + `--disable-rewards-normalization`（advantage = reward）|
| **本项目钩子** | `metaclaw_round_scale` 接管 `_post_process_rewards`：**`advantage = reward / n_turns`** |
| 非 MetaClaw 样本 | 原样透传（advantage = reward）|
| OPD | hint 贴在**本轮题面**上 → teacher forward → `teacher_log_probs` 进样本 |

**1/N 直到 2026-09-08 的 `e7979e1` 才真正接通。**此前所有训练（含全部塌陷事件）都是每个 turn 拿完整 ±1。

---

## 六、更新节奏

```
--rollout-batch-size 8            ← 8 个 group = 8 道完整的题
--use-dynamic-global-batch-size   ← 由实收样本数推导 gbs
--num-steps-per-rollout 1
--n-samples-per-prompt 1          ← 每题只跑一次，无组内比较
```

**收满 8 道题 → 更新一次权重。**

---

## 七、超参

| 参数 | 值 | 来源 |
|---|---|---|
| `--lr` | 1e-5，constant | 官方，未改 |
| `--kl-loss-coef` | **0.0**（等于不加 KL）| 官方，未改 |
| `--entropy-coef` | 0.00 | 官方，未改 |
| `--max-tokens-per-gpu` | 32768 | 官方，未改 |
| `--rollout-max-context-len` | **65536** | MetaClaw profile 从 32768 提高 |
| `--sglang-context-length` | **65536** | 同上 |
| `--rollout-batch-size` | **8** | MetaClaw profile 从 16 降低 |
| `--save-interval` | **10** | MetaClaw profile 从 100 降低 |
| `--use-dynamic-batch-size` | 开 | 官方 |
| `--use-dynamic-global-batch-size` | 开 | MetaClaw profile 注入 |
| `--custom-reward-post-process-path` | `metaclaw_round_scale.metaclaw_round_scale` | MetaClaw profile 注入 |

生成上限由 MetaClaw benchmark 自己的配置决定：`openclaw_cfg/openclaw.json` 的 `contextWindow: 50000` / `maxTokens: 50000`。**官方 `--rollout-max-response-len 8192` 对本路径不生效**（它管 slime 自己的 rollout 引擎，而我们的生成由 openclaw agent 发起）。

---

## 八、补丁清单（9 个，启动脚本按序部署）

**代理侧 3 个**（生成打过补丁的 server 副本）：

| 脚本 | 作用 |
|---|---|
| `prepare_patched_openclaw_opd.sh` | round-mode 门控、held、duplicate-user-retry 等 |
| `prepare_patched_openclaw_combine.sh` | 父类 submit 交还、round-group 收集与归属判定、1/N 缩放器 |
| `prepare_patched_openclaw_combine_select.sh` | **子类 submit 交还**、verdict 解析、OPD hint 定位 |

**OpenClaw 系统侧 6 个**（改本机安装的 dist）：
`rl-training-headers`、`sglang-execution-bias`、`embedded-agent-overflow-recovery`、`system-prompt-output-directives`、`cli-compaction`、`silent-reply-policy`

**训练侧规则**：truncation-penalty（截断强制 -1、不丢样本）、tool-error-penalty、duplicate-user-retry（丢弃）、degraded-turn-drop、invalid-tool-use-penalty、skip-forced-negative-override

**环境侧（2026-09-04 起与官方一致）**：三项反馈加料全部移入 OPD hint（**模型看不见**）；`[Previous Feedback]` 用官方原文；infra 失败轮记 0 分**进分母**。

---

## 九、已知未修

| | 影响 |
|---|---|
| **`--dir --min-count` 累计阶梯** | **31% 的 FC 轮次**（70/224）：早轮欠账 → 本轮做对也判 -1。**奖励本身错，不是归属错** |
| 空回复 turn 变 2~3 token 样本 | 官方守卫因 `response_text` 恒含 `<\|im_end\|>` 形同虚设 |
| `sum_of_sample_mean` 每样本等权 | 2 token 与 5000 token 同权重 |
| 批级基线放大稀有正样本 | |
| 全负批 | 1/N 只压幅度不改符号；要 `n_samples>1` 才有组内比较 |
| **`CP=1` + teacher 单卡** | 长回复必 OOM。2026-09-08：34,885 token 的回复 → 34885×151936×4B = **19.75 GiB** 单次分配 |

---

## 十、每趟跑必看的日志判据

```bash
# 1/N 与成组是否真生效（e7979e1 之前恒为 0）
grep -c "scaled by 1/turns" <LOGS_DIR>/training.log
grep -c "queued group="     <LOGS_DIR>/training.log

# 归属过滤与 hint 定位（正常应为 0）
grep -c "do not carry"                       <LOGS_DIR>/training.log
grep -c "could not locate this round's task" <LOGS_DIR>/training.log

# 显存/长度（>32768 说明有超长单条样本）
grep "max_tokens_per_microbatch" <LOGS_DIR>/training.log | tail -20

# 链路健康
grep -c "infra failure -- scored 0"          <LOGS_DIR>/metaclaw_rollout.log
grep -ci "out of memory"                     <LOGS_DIR>/training.log
```

**前两条都非零，才说明这一趟是在设计意图下训练。**

---

## 十一、评测口径

| | |
|---|---|
| 定版基线 | **按天 K=0：Acc 41.1% / Compl 26.3%**（`20260907_112320`）|
| **噪声下限（2026-09-09 加）** | 第二趟 K=0（`20260909_094645`，同配置零训练）跑出 **44.7% / 25.9%**。→ **Acc 跨趟差 3.6pt，Compl 跨趟差 0.4pt**。**Acc 提升低于约 4pt 与噪声不可区分；Compl 则相当稳，小幅变化就有意义** |
| 对照方式 | 一律与该基线比，**不与论文 Table 1 比**（两列都不可比，见计划文档「🔴 当前结论」）|
| **训练后必须拆两栏报 Acc** | ① MCQ 抽取成功率 ② 抽取成功者中的正确率。格式合规是表层行为，RL 几步就能学会，只报总 Acc 会把"学会写反斜杠"误当成"方法有效" |
| Compl 定义 | 仅 file_check 的 224 题；`report.json` 的 `metrics.passed` 是全题平均，**不是 Compl** |
