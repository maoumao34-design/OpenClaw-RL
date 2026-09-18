[← 工作记录](work_log.md) · [改动验证台账 →](change_ledger.md)

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

**快照对应：2026-09-18**（轨迹级样本 + 超长轮次熔断 + 熵系数可调）（此后若有未同步的改动，以代码为准）

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
                          ├ _metaclaw_build_trajectory：整轮拼成**一条**序列
                          └ 一次性 output_queue.put((group_index, [一个样本]))
```

**一个 round = 一个样本（2026-09-14）。**轨迹以**终轮 prompt 为骨架**构造——它已经按模型真实读到的样子含有题面、全部历史动作和全部工具结果——再把每个更早 turn 的可见输出**换回它的完整 response（含思考）**：

```
序列      [题面][think₁+action₁][tool₁][think₂+action₂][tool₂][think_N+action_N]
loss_mask        1111111111111   0000   1111111111111   0000   1111111111111
```

**不是前后拼接**：`prompt_N ≠ prompt_{N-1} + response_{N-1}`（生产数据实测 0/127，OpenClaw 回放时会剥掉 assistant 的思考），链条不嵌套，没有可拼的东西。

**某个 turn 在骨架里找不到 → 原样留着但不 mask**：它的 token 是模型真实读到的，只是不产生梯度。**降级一个 turn，而不是丢掉整轮**——2026-09-03 那次断言前缀包含、断言失败就整轮丢，代价是 120 轮里丢了 96 轮。

样本形态（一个样本、生成段 mask=1、工具结果段 mask=0、`rollout_log_probs` 在未 mask 段补 0）**是 slime 自己的多轮写法**，见 `slime/examples/search-r1/generate_with_search.py`。


（轨迹级方案已于 2026-09-03 删除实现。**⚠️ 当时给的理由「`dropReasoningFromHistory` 让扁平轨迹无法忠实构造」已于 2026-09-10 实测推翻——那个开关是关的，reasoning 会原样回放。**真实成因仍未查明，头号嫌疑是同一段 policy 里对我们这种 provider 同样开着的 `sanitizeToolCallIds`/`toolCallIdMode: strict`。详见 [`metaclaw_migration_plan.md`](metaclaw_migration_plan.md) 第九节。）

**⚠️ 交还逻辑必须同时存在于父类和 Select 子类**——服务器实例化的是 `OpenClawCombineSelectAPIServer`，它覆写了两个 `_submit_*`。只打父类等于没打（2026-09-08 实测：held 98 次，`queued group=` 0 次）。

---

## 五、奖励与 advantage

| 层 | 内容 |
|---|---|
| reward | **本轮 checker 判定 ±1**，整轮所有 turn 共享 |
| 官方基线 | `--advantage-estimator grpo` + `--disable-rewards-normalization`（advantage = reward）|
| **本项目钩子** | **已删除**（2026-09-14）。一个 round 是一个样本，N=1，没有可除的东西 |
| 非 MetaClaw 样本 | 原样透传（advantage = reward）|
| OPD | hint 贴在**本轮题面**上 → teacher 在**整条轨迹**上 forward → `teacher_log_probs` 进样本。2026-09-14 前只覆盖最后一个 turn |

**1/N 从未被一趟跑完的训练验证过**（09-03 实现、09-09 才接通，而那趟 OOM），随轨迹级样本一起删除。见 [`change_ledger.md`](change_ledger.md)。

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
| `--entropy-coef` | **默认 0.00（官方值），可由 `METACLAW_ENTROPY_COEF` 覆盖** | 2026-09-18 加覆盖口。非零时 `openclaw_topk_select_loss.py` 走 `need_entropy_for_loss` 分支自动开 `with_entropy`，**不需要改 loss 代码**，副作用是真实熵终于进日志（`train/entropy_loss`）|
| `--max-tokens-per-gpu` | 32768 | 官方，未改 |
| `--rollout-max-context-len` | **65536** | MetaClaw profile 从 32768 提高 |
| `--sglang-context-length` | **65536** | 同上 |
| `--rollout-batch-size` | **8** | MetaClaw profile 从 16 降低 |
| `--save-interval` | **10** | MetaClaw profile 从 100 降低 |
| `--use-dynamic-batch-size` | 开 | 官方 |
| `--use-dynamic-global-batch-size` | 开 | MetaClaw profile 注入 |


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

**训练侧规则**：tool-error-penalty、duplicate-user-retry（丢弃）、degraded-turn-drop、invalid-tool-use-penalty、skip-forced-negative-override

> ⚠️ **`truncation-penalty` 在 MetaClaw 这条路上不生效**（2026-09-15 查明，此前本表把它列为生效规则是错的）。
> `is_truncated → -1` 只存在于 **step-judge 分支**和 **PRM 分支**，两者都被 `_metaclaw_verdict is None` 挡着，
> 而 MetaClaw 走的是 verdict 分支。**截断目前不受任何惩罚。**
>
> 另：`invalid-tool-use-penalty` 虽然接在 verdict 分支上，但 **(a) 之下它只检查本轮最后一个真实 turn**。
> 实测 108 个含超长 turn 的 round 里约 **18 个**是更早 turn 已崩、最后一 turn `invalid=False`。

**环境侧（2026-09-04 起与官方一致）**：三项反馈加料全部移入 OPD hint（**模型看不见**）；`[Previous Feedback]` 用官方原文；infra 失败轮记 0 分**进分母**。

---

## 九、已知未修

| | 影响 |
|---|---|
| **`--dir --min-count` 累计阶梯（两个方向都错）** | **负向**：早轮欠账 → 本轮做对也判 −1（31% 的 FC 轮次，70/224）。**正向（2026-09-17 新记）**：checker 只数目录里已有的合规文件、**完全不看最终回复**，所以**一个零写入的超时轮可以靠更早轮次的残留文件白拿 +1**（实证：`20260914_181842` day09 r8，本轮写入前目录已有 4 个合规 json，`--min-count 4` 直接过线）。**这个方向更有害——它把错误的 +1 喂进优化器，强化的正是退化行为**，且污染不对称（超时轮越多的跑次白拿越多）。**奖励本身错，不是归属错** |
| 空回复 turn 变 2~3 token 样本 | 官方守卫因 `response_text` 恒含 `<\|im_end\|>` 形同虚设 |
| `sum_of_sample_mean` 每样本等权 | 2 token 与 5000 token 同权重。**⚠️ 2026-09-15：「这导致模型倾向长回复」已被推翻**——退化是 88–96% 字面重复的循环、且在 step10 骤跳，不是稀释压力推出来的缓慢爬升。它至多是循环发生后的**维持**机制 |
| 批级基线放大稀有正样本 | |
| 全负批 | 1/N 只压幅度不改符号；要 `n_samples>1` 才有组内比较 |
| **没有任何正则把策略拉住** | `--entropy-coef 0.00` **且** `--kl-loss-coef 0.0` —— 既不拉向高熵也不拉向 base，跑的是纯奖励最大化。**2026-09-18 三臂消融**：零训练基线跑完 30 天且 `DROPPED=0`，而纯 RL / 纯 OPD / 混合**三个训练臂全部退化成复读、无一跑完**。复读按定义即低熵行为。熵系数已可调（见第七节），KL 仍为 0 |
| **`CP=1` + teacher 单卡** | 长回复必 OOM。2026-09-08：34,885 token 的回复 → 34885×151936×4B = **19.75 GiB** 单次分配 |
| **超长样本会杀掉整趟训练** | 已死过三趟：09-08 / 09-09 teacher OOM；`20260914_171937` 是 `gather_log_probs_at_indices` 拿到 71668 行 index 对 21495 token 的 chunk。**2026-09-14 加了熔断**（`METACLAW_MAX_TRAJECTORY_TOKENS`，默认 32768）：超限的轮次告警并丢弃，训练继续。**熔断一旦触发就要先查轮边界**——一个 round 只有几个 turn，样本这么大通常是别的轮次的 turn 混进来了 |

---

## 十、每趟跑必看的日志判据

```bash
# 轨迹级样本是否真生效
grep -c "queued group="                     <LOGS_DIR>/training.log   # ≈ 完成的轮数
grep -c "scaled by 1/turns"                 <LOGS_DIR>/training.log   # 应为 0（1/N 已删）
grep    "tokens masked"                     <LOGS_DIR>/training.log | tail -20
grep -c "not found in the final prompt"     <LOGS_DIR>/training.log   # 降级计数，应很低
grep -c "DROPPED an oversized round"        <LOGS_DIR>/training.log   # 熔断，应为 0

# 归属过滤与 hint 定位（正常应为 0）
grep -c "do not carry"                       <LOGS_DIR>/training.log
grep -c "could not locate this round's task" <LOGS_DIR>/training.log

# 显存/长度（>32768 说明有超长单条样本）
grep "max_tokens_per_microbatch" <LOGS_DIR>/training.log | tail -20

# 链路健康
grep -c "infra failure -- scored 0"          <LOGS_DIR>/metaclaw_rollout.log
grep -ci "out of memory"                     <LOGS_DIR>/training.log
```

**`queued group=` 非零、且 `tokens masked` 的比值合理（工具结果不该占绝大多数），才说明这一趟是在设计意图下训练。**

---

## 十一、评测口径

| | |
|---|---|
| 定版基线 | **按天 K=0：Acc 41.1% / Compl 26.3%**（`20260907_112320`）|
| **噪声下限（2026-09-09 加）** | 第二趟 K=0（`20260909_094645`，同配置零训练）跑出 **44.7% / 25.9%**。→ **Acc 跨趟差 3.6pt，Compl 跨趟差 0.4pt**。**Acc 提升低于约 4pt 与噪声不可区分；Compl 则相当稳，小幅变化就有意义** |
| 对照方式 | 一律与该基线比，**不与论文 Table 1 比**（两列都不可比，见计划文档「🔴 当前结论」）|
| **训练后必须拆两栏报 Acc** | ① MCQ 抽取成功率 ② 抽取成功者中的正确率。格式合规是表层行为，RL 几步就能学会，只报总 Acc 会把"学会写反斜杠"误当成"方法有效" |
| Compl 定义 | 仅 file_check 的 224 题；`report.json` 的 `metrics.passed` 是全题平均，**不是 Compl** |
| **Acc 口径：用 report Acc，不要用 rollout 的 `passed=True` 计数** | `report.md` 的 Acc **含部分分**（基线 day02 = `Correct 8.8 / 11` → 80.3%），而 rollout 里 `passed=True` 的离散数是 8/11 ≈ 72.7%。**两者差 7.6pt**，混用会得出相反结论。2026-09-18 险些据此误判一趟标定跑 |
