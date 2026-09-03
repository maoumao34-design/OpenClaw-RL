# 阶段状态归档

[← 工作记录](work_log.md)

本文件存放**已被后续结果取代**的阶段状态快照，从 `work_log.md` 拆出（2026-08-28）——原先 37 个历史块散落在 `work_log.md` 各处、合计 1024 行占全文 39%，把最新条目和当前状态埋在了文件底部。
**当前状态不在这里**，在 [`work_log.md`](work_log.md) 顶部。

按日期**倒序**排列（最新的历史快照在前）。各块内容原样迁移，未作任何改写。

---

## 历史状态（2026-09-02，已被 9/3 全面回退取代）

### 已就绪
**OpenClaw-RL Separate/Personal Agent Track**（同 08-13，未变）。
**MetaClaw 迁移**：同 09-01（历史状态），另加：[x] **目标已重设为"跑满 30 天、总分高于只训到 day6"**，判据随之改为"整体持续变好、绝不训坏"，而不是每一批信号都更对；[x] **两次失败已归结为同一个病**——二值 reward 区分不了"答对且简洁"和"答对但啰嗦"，而 `loss_mask` 覆盖整段 response（含 `<think>`）；[x] **`blend` 已删除**（实测它会让全失败 round 里被判官认可的那一步拿到 +1.121 的正 advantage，摧毁"全负批不更新"这条安全性质）；[x] **长度感知的正奖励已实现**（只对 reward>0 生效、按响应 token、覆盖所有模式的所有正样本、负样本一律平坦 -1）；[x] **批级基线已改成只减均值不除 std**（`|adv| ≤ 2`，1 正 15 负从 3.873 降到 1.875）；[x] **41 项断言全部通过、双向非空洞性已验证**。

### 已知限制 / 未解决
同 09-01（历史状态），另加：**`20260902_094458` 的 checkpoint 已污染**（day06-r8 thinking 已到 120945），跟 `20260831_154301` 一样不能作为起点，下一次必须从干净 base 起步。**round 轮数没有上限**——day06-r7 空转 186 轮是那次崩溃的起点，杠杆（driver 的 `round_timeout` / 代理侧按 `_turn_counts` 拒绝）副作用未查清，**本轮刻意没动**。**退化熔断尚无指标与阈值依据，同样没做**。`L0=6000` 的代价是明确接受的：一次健康但偏长的 9k 成功会拿 0.73 而不是 1.0。

### 下一步
1. **OpenClaw-RL 复现**：同 08-17
2. **MetaClaw 迁移**：**跑一轮"复现 `20260827_163030` 的 judge 配置 + 批级基线 + 两项防训坏补丁"**，不用 outcome、不用 blend，**从干净 base 起步**。判据：① day17 不再出现 18k→115k 的 thinking 断崖；② day16-22 的 MC 格式失败率明显低于 17/26；③ Acc. 可以低于只训到 day6 那次，但不能出现 day20-22 归零式崩塌；④ 全负批 advantage 全 0、不再出现约 3.87 的稀有正样本放大；⑤ `[metaclaw-batch-baseline]` 与 `[openclaw-rl-metaclaw-length-aware-success]` 两类日志行必须真的出现（**reward 侧计数不会变，别盯它**）
3. 跑完之后再谈轮数上限和退化熔断——两者都需要先拿到这一轮的日志才有依据
4. 其余同 08-17

### 未验证
- [ ] **长度感知正奖励在真实训练里的效果**——41 项本地断言只证明算术和边界正确，"它能否真的挡住 thinking 膨胀"要真实训练回答
- [ ] **`L0=6000`/`L1=16000` 这两个阈值是否合适**——取自 K=6 与 `20260827` 的正样本长度分布（p90 5.3k vs 6.9k），是有依据的估计，不是实测最优
- [ ] **只减均值的批级基线能否真的阻止发散**——去掉了放大器，但"够不够稳"未知
- [ ] **两项改动叠加后会不会把正信号压得太弱**——长度打折 + 批中心化都在缩小正 advantage，有没有过度未知
- [ ] **中间步骤判官正奖励是不是 thinking 膨胀的上游原因**——至今没有被干净地回答过（承接 09-01）
- [ ] **round 轮数上限的两个候选杠杆各自的副作用**——`round_timeout` 与代理侧按 `_turn_counts` 拒绝，都没查
- [ ] **路线 A 的 token 序列重建与 logprob 拼接是否可行**——原料齐全但未实测（承接 09-01）
- [ ] **Phase 1 在真实训练环境下的实际效果**——打分正确性已核实，训练效果层面未回答
- [ ] **Traceback 泄漏修复在真实训练中是否生效**——合成测试通过，需确认真实 `[Previous Feedback]` 里 Traceback 归零
- [ ] **`done.log` 非追加场景真实触发率**——监控已埋点，等真实训练观察
- [ ] **`_AGENT_PAUSE_MARKERS` 扩展在真实暂停窗口下是否真的挽回了原本会丢的样本**
- [ ] **K=6 冻结实验的结果用官方独立 `metaclaw-bench run` 重新核实**——目前走的是训练自己的 harness
- [ ] "对齐/不对齐基线 Acc. 差异" vs "`plugins.allow` 无条件排除插件"这两个结论之间的矛盾（承接 08-18，仍未解开）
- [ ] 官方 MetaClaw Compl. 非零的真实原因——开放问题，不阻塞
- 其余同 09-01（历史状态，见 [`status_history.md`](status_history.md)）

---

## 历史状态（2026-09-01，已被 9/2 结果取代）

### 已就绪
**OpenClaw-RL Separate/Personal Agent Track**（同 08-13，未变）。
**MetaClaw 迁移**：同 08-26（历史状态），另加：[x] **Phase 1 打分逻辑已用真实训练数据核实正确**（4 个升级案例全部无误、反向 0 次、诊断文案具体化生效）；[x] **Traceback 泄漏进 agent 可见反馈的回归已修复**（`_compute_training_verdict` 去掉多余 fallback，合成测试已确认非空转）；[x] **`20260827_163030` 训练的退化机制已定位到 day17 thinking 断崖，链条有三项数据支撑**（格式失败 17/26 vs K=6 的 0/27、thinking 18k→115k 的时间断点、FC 派生样本占权重约 90%）；[x] **消融方案已通过 CLI 查验并实现**（中间轮次改吃本轮最终 checker 结果，`METACLAW_MIDROUND_REWARD` opt-in，默认 `judge` 行为不变）；[x] **硬负分优先级 + 结构性无效工具调用检测（规则 6）已修复**（commit `2944f87`）；[x] **outcome 消融训崩的根因已查清并在源码层面确认**（全负 batch + advantage 退化成原始 reward）；[x] **真正的根因已定位到"用着 GRPO 估计器却一个真正的组都没有"**，`toolcall-rl`/MetaClaw 的分组方式与任务形态已逐字查证；[x] **批级基线方案已出，可行性已在源码层面查证**（slime 自带 `--custom-reward-post-process-path` 钩子，不用碰 `group_index`/队列），**待 CLI 在真实环境查证 5 个点后实现**。

### 已知限制 / 未解决
同 08-26（历史状态），另加：**`20260831_154301` 的 checkpoint 已污染，不能作为后续训练起点**。**批级基线方案尚未实现，也未经真实环境查证**——5 个待查点见迁移文档。**`blend` 模式可能是多余的**：它照搬 toolcall-rl 的奖励合成，但 toolcall-rl 能靠它工作的前提是同时有 8 样本组内归一化，我们只搬了一半；默认关闭、回退成本低，视批级基线结果决定去留。**路线 A（轨迹级样本）仍未实现**，且它也只降低负样本数量、不产生正样本。**day11-15 的 FC 全 0 是能力墙**（K=6 冻结模型同样 0），跟训练信号无关。

### 下一步
1. **OpenClaw-RL 复现**：同 08-17
2. **MetaClaw 迁移**：**等 CLI 在真实环境查证批级基线方案的 5 个点**，通过后实现并跑一轮。**先单独跑批级基线、不叠加 `blend`**（`METACLAW_MIDROUND_REWARD=judge` + 自定义钩子）——这是最干净的对照，只改 advantage 算法、reward 形态完全不变，跟 K=6 那次 judge 模式唯一差别就是基线。主判据是 `batch reward` 不再出现 `0/16`、`grad_norm`/`policy drift` 保持在 K=6 量级，**不是 Acc**。**必须从干净 base 起步。**
3. 其余同 08-17

### 未验证
- [ ] **批级基线方案的 5 个真实环境问题**——`load_function` 的路径格式与 modelfactory 上的放置位置、dummy 样本会不会污染批均值、真实到达钩子的样本数、std≈0 时全 0 advantage 是否可接受、与 `step_wise` estimator 是否冲突
- [ ] **批级基线能否真的阻止发散**——源码层面确认了 advantage 会被中心化（必有正负），但真实训练里够不够稳定未知
- [ ] **中间步骤判官正奖励是不是 thinking 膨胀的上游原因**——原来的消融设计已被证明会因全负 batch 而发散，**这个问题至今没有被干净地回答过**；要等训练能稳定跑起来之后才谈得上重新验证
- [ ] **路线 A 的 token 序列重建与 logprob 拼接是否可行**——原料齐全（逐轮 `response_logprobs` 都存着），但未实测；`rollout_log_probs`/`teacher_log_probs` 都按 `response_length` 严格切片，对不齐就是硬错误
- [ ] **Phase 1 在真实训练环境下的实际效果**——打分正确性已核实，但训练效果层面未回答
- [ ] **Traceback 泄漏修复在真实训练中是否生效**——合成测试通过，需确认真实 `[Previous Feedback]` 里 Traceback 归零
- [ ] **`METACLAW_TRAIN_UNTIL_DAY` 默认关闭时是否真的与当前 `day12` 训练行为完全一致**——用户即将验证，这次改动能不能信任的前提
- [ ] **`done.log` 非追加场景真实触发率**——监控日志已埋点（`_compute_training_verdict` 里 `logger.warning`），只能等真实训练跑起来后观察
- [ ] **`_rerun_segment_official` 额外 subprocess 调用在真实训练节奏下的耗时影响**——本地未测过实际耗时，CLI 判断"可忽略"是基于全量 398 段无写操作迹象的静态扫描，不是真实计时
- [ ] **`_AGENT_PAUSE_MARKERS` 扩展在真实暂停窗口下是否真的挽回了原本会丢的样本**——下一轮训练需要确认日志里出现"pause-retry (matched 'LLM request timed out')"且题目最终计分成功
- [ ] **K=6 冻结实验的结果用官方独立 `metaclaw-bench run` 重新核实**——目前的 Frozen 窗口评测走的是训练自己的 harness，跟官方 bench 不完全同构
- [ ] `metaclaw_migration_20260820_*`（六处修复已合入）完整 30 天跑完后，`Compl.` 是否脱离 0.0%、Acc. 相对**新基线（17.8%）**有没有提升——`--agent` 修复的核心验证点，注意不要再拿旧的 8.1% 做对比
- [ ] `METACLAW_TRAIN_UNTIL_DAY` 设置为具体 K 值时，冻结是否真的生效（`[metaclaw-freeze]` 日志、样本提交数骤降为 0）、dayK 尾部竞态实际丢弃规模
- [ ] "对齐/不对齐基线 Acc. 差异" vs "`plugins.allow` 无条件排除插件"这两个结论之间的矛盾，具体机制是什么（承接 08-18，仍未解开）
- [ ] 官方 MetaClaw Compl. 非零的真实原因（OpenClaw CLI 版本差异 or 官方外层脚本另有处理）——开放问题，不阻塞
- 其余同 08-19（历史状态，见上）

---

## 历史状态（2026-08-26，已被 8/27 结果取代）

### 已就绪
**OpenClaw-RL Separate/Personal Agent Track**（同 08-13，未变）。
**MetaClaw 迁移**：同 08-25（历史状态，暂停窗口修复 + thinking 空转真正机制定位均已就绪），另加：[x] **累计跨轮次计数缺陷的 diff-based 修复（Phase 1）已实现、本地验证通过、已提交推送**（`py_compile`+合成测试+全量 30 天真实数据分类扫描三重验证，分类数字跟 CLI 报告完全一致）。真实训练环境完全未验证。

### 已知限制 / 未解决
同 08-25（历史状态），未变，另加：**diff-based 修复（Phase 1）完全没有在真实训练环境跑过**——`day12`+ 是否真的不再复现 `094611` 的 thinking 空转崩溃模式，仍需下一轮训练验证，本地验证只能确认代码逻辑本身没错，不能确认训练效果。

### 下一步
1. **OpenClaw-RL 复现**：同 08-17
2. **MetaClaw 迁移**：**diff-based 修复（Phase 1）已实现并推送，下一步是提交一轮真实训练验证**
3. 其余同 08-17

### 未验证（截至 08-26 末，见 08-27 条目最新结果）
- [ ] diff-based 修复（Phase 1）在真实训练环境下的实际效果
- 其余同 08-25（历史状态，见上）

---

## 历史状态（2026-08-25，已被 8/26 结果取代）

### 已就绪
**OpenClaw-RL Separate/Personal Agent Track**（同 08-13，未变）。
**MetaClaw 迁移**：同 08-21（历史状态，六处修复 + `METACLAW_TRAIN_UNTIL_DAY` + thinking 空转三处修复 + 新基线 + K=6 里程碑均已就绪），另加：[x] **暂停窗口砍断在飞生成、误判成 timeout 而不是 503 的问题已修复**（`_AGENT_PAUSE_MARKERS` 扩展，28/28 历史数据零反例支持），预期能挽回 K=6 实验里丢掉的那类样本；[x] **"超长 thinking 空转"复现的真正机制已定位（经过一次自我纠正）**——最初以为是"中段 step-judge 跟最终结果脱钩"，已撤回；真正原因是 `check_filename.py --dir --min-count` 累计计数缺陷：这一轮即使做对了，也可能因为当天更早题目的欠账被 checker 判 -1，持续的"看似纠偏、实际拧不动"负反馈才是把模型逼向空转的机制；[x] **diff-based 修复方案已定稿为 v2，经 CLI 两轮真实数据核对确认 Phase 1 范围可以进入实现，但代码完全未动**——设计文档已写完，尚未开始写任何代码。`py_compile`+合成测试验证通过（pause-marker），真实训练完全未验证，不影响正在跑的 `day12` 这条训练。

### 已知限制 / 未解决
同 08-21（历史状态），未变，另加：**`_AGENT_PAUSE_MARKERS` 扩展完全未在真实训练中验证**——合成测试只能确认代码逻辑正确，不能确认真实暂停窗口下这条新 marker 真的按预期挽回样本。**累计跨轮次计数缺陷的 diff-based 修复设计（v2）已通过 CLI 两轮核对，但完全没有代码实现**——覆盖范围比最初发现的 `check_filename.py --dir --min-count`（day06-10）更大：day11-15/16-23 的 glob 计数是同类问题，day21-30 的 `check_done_log.py --min-entries` 是更严重的一种（历史一行写错则永久 FAIL，不只是门槛追不上）。`455a54f` 那三处修复（next-round 反馈+`is_invalid_tool_use` 接线）不解决这类问题，下一轮训练如果还是崩，符合预期，不代表之前的诊断错了。

### 下一步
1. **OpenClaw-RL 复现**：同 08-17
2. **MetaClaw 迁移**：**diff-based 修复设计（v2）已通过 CLI 两轮核对确认可以进入实现，下一步是照设计开始写代码**——`_run_round`/`_compute_training_verdict`/driver 主循环三处接线。用户另开默认配置训练对比 `TRAIN_UNTIL_DAY` 行为一致性的任务仍然有效，可以并行推进
3. 其余同 08-17

### 未验证
- [ ] **`METACLAW_TRAIN_UNTIL_DAY` 默认关闭时是否真的与当前 `day12` 训练行为完全一致**——用户即将验证，这次改动能不能信任的前提
- [ ] **累计跨轮次计数缺陷 diff-based 修复（v2，Phase 1）尚未开始实现**——设计已定稿并通过 CLI 两轮核对，代码完全未动
- [ ] **`_AGENT_PAUSE_MARKERS` 扩展在真实暂停窗口下是否真的挽回了原本会丢的样本**——下一轮训练需要确认日志里出现"pause-retry (matched 'LLM request timed out')"且题目最终计分成功
- [ ] **K=6 冻结实验的结果用官方独立 `metaclaw-bench run` 重新核实**——目前的 Frozen 窗口评测走的是训练自己的 harness，跟官方 bench 不完全同构
- [ ] `metaclaw_migration_20260820_*`（六处修复已合入）完整 30 天跑完后，`Compl.` 是否脱离 0.0%、Acc. 相对**新基线（17.8%）**有没有提升——`--agent` 修复的核心验证点，注意不要再拿旧的 8.1% 做对比
- [ ] `METACLAW_TRAIN_UNTIL_DAY` 设置为具体 K 值时，冻结是否真的生效（`[metaclaw-freeze]` 日志、样本提交数骤降为 0）、dayK 尾部竞态实际丢弃规模
- [ ] "对齐/不对齐基线 Acc. 差异" vs "`plugins.allow` 无条件排除插件"这两个结论之间的矛盾，具体机制是什么（承接 08-18，仍未解开）
- [ ] 官方 MetaClaw Compl. 非零的真实原因（OpenClaw CLI 版本差异 or 官方外层脚本另有处理）——开放问题，不阻塞
- 其余同 08-19（历史状态，见上）

---

## 历史状态（2026-08-21，已被 8/25 结果取代）

### 已就绪
**OpenClaw-RL Separate/Personal Agent Track**（同 08-13，未变）。
**MetaClaw 迁移**：同 08-20（历史状态，六处修复 + `METACLAW_TRAIN_UNTIL_DAY` + thinking 空转三处修复均已实现，已合入 `metaclaw_migration_20260820_*` 这条正在跑的训练），另加：[x] **训练前基线已用 `--agent` 修复后的 harness 重新定版**（Acc.=17.8%/Compl.=0.0%，seed 465485731，取代旧的 8.1% 版本）；[x] **里程碑：`K=6` 冻结实验第一次拿到可信的正面训练效果**（主结果=全程 live 聚合 Acc.=37.3%/Compl.=13.9%，相对基线 +19.5pt/+13.9pt，`Compl.` 首次非零；用的是 `0882f69`，不含 thinking 空转修复）；[x] `METACLAW_TRAIN_UNTIL_DAY` 的设计动机表述两次被用户纠正后已改正，明确其"临时工具，非永久方法学"的定位。基线和 K=6 实验都是真实跑通的结果，不是待验证项；文档表述纠正不涉及代码改动。

### 已知限制 / 未解决
同 08-20（历史状态），未变，另加：**K=6 实验的正面结果不能代表 thinking 空转问题已解决**——冻结窗口没崩，结构性原因是 day6 后没再继续训练，不是修复生效；也没有用官方独立 bench 重新核实过这个冻结 checkpoint，严格 apples-to-apples 对比还没做。新基线确认了 file_check（`Compl.=0.0%`）是真实能力上限，不是链路问题——thinking 空转三处修复即使完全生效，也只是"不再空转"，不代表 file_check 通过率会提高，这是两件不同的事。

### 下一步
1. **OpenClaw-RL 复现**：同 08-17
2. **MetaClaw 迁移**：用户先另开一次默认配置（不设置 `METACLAW_TRAIN_UNTIL_DAY`）的训练，跟当前 `day12` 这条正在跑的训练直接对比，确认这两批改动在默认/未触发状态下真的没有引入任何行为差异
3. 其余同 08-17

### 未验证（截至 08-21 末，见 08-25 条目最新结果）
- [ ] `METACLAW_TRAIN_UNTIL_DAY` 默认关闭时是否真的与当前 `day12` 训练行为完全一致
- [ ] thinking 空转三处修复上线后，训练继续超过 day6/day12 是否还会复现"越写越长/thinking 循环"
- [ ] K=6 冻结实验的结果用官方独立 `metaclaw-bench run` 重新核实
- 其余同 08-20（历史状态，见上）

---

## 历史状态（2026-08-20，已被 8/21 结果取代）

### 已就绪
**OpenClaw-RL Separate/Personal Agent Track**（同 08-13，未变）。
**MetaClaw 迁移**：同 08-19（六处修复：训练信号 + checkpoint 默认路径 + 503 暂停重试 + OPD hint 接线 + session 拆分 + `--agent` workspace 修复，已合入 `metaclaw_migration_20260820_*` 这条正在跑的训练，目前到 day12 效果不错），另加：[x] **`METACLAW_TRAIN_UNTIL_DAY` 可调训练窗口 + 冻结评测已实现**（driver + 2 处 proxy 补丁），[x] **day12-14 超长 thinking 空转的三处根因修复已实现**（next-round 反馈追加真实 stdout + 日期说明 + FORMAT_ERROR 原文片段 + `is_invalid_tool_use` 接线到 MetaClaw 两条打分分支）。两项功能均 `py_compile`/合成测试/完整补丁链验证通过，真实训练完全未验证，且明确不影响正在跑的 `day12` 这条训练。

### 已知限制 / 未解决
同 08-19（历史状态），未变，另加：**`METACLAW_TRAIN_UNTIL_DAY` 默认关闭是否真的行为一致 + thinking 空转三处修复均完全未在真实训练中验证**——尤其是"未设置=行为完全一致"这个最关键的前提，只能靠代码结构上的门控条件保证；dayK 尾部竞态（少量样本被连带丢弃）是已知、接受的设计，尚未在真实数据上观察过实际丢弃规模；thinking 空转修复只覆盖了这次诊断出的三个根因，没有解决"P2 阶段真实通过率能不能提高"这个更根本的能力问题。

### 下一步
1. **OpenClaw-RL 复现**：同 08-17
2. **MetaClaw 迁移**：用户先另开一次默认配置（不设置 `METACLAW_TRAIN_UNTIL_DAY`）的训练，跟当前 `day12` 这条正在跑的训练直接对比，确认这两批改动在默认/未触发状态下真的没有引入任何行为差异
3. 其余同 08-17

### 未验证（截至 08-20 末，见 08-21 条目最新结果）
- [ ] `METACLAW_TRAIN_UNTIL_DAY` 默认关闭时是否真的与当前 `day12` 训练行为完全一致
- [ ] thinking 空转三处修复上线后，训练继续超过 day6/day12 是否还会复现"越写越长/thinking 循环"
- 其余同 08-19（历史状态，见上）

---

## 历史状态（2026-08-19，已被 8/20 结果取代）

### 已就绪
**OpenClaw-RL Separate/Personal Agent Track**（同 08-13，未变）。
**MetaClaw 迁移**：同 08-18，另加：[x] `metaclaw_migration_20260818_182736`"先好后差"塌陷模式的根因（checker 分数丢失 + verdict 残片污染 GRPO）已定位并修复；[x] 默认提交训练会静默加载上次可能训坏的权重这个问题已修复（`SAVE_CKPT` 默认带时间戳）；[x] 训练暂停期间 503 把整段天数空转吃掉的问题已修复（503 专属耐心等待重试环，跟 timeout 等其它失败类型分开处理）；[x] checker 算出的 OPD hint 被无条件丢弃的问题已修复（Layer 1 收窄 file_check 静态文案退路 + Layer 2 真正接回 `accepted=True` 材料化）；[x] session 从"一天一个"拆成"每题一个"，切断了同一天题目间的 context overflow 连坐，顺带结构性解决了 08-17 的"跨 round 污染"老问题；[x] **`openclaw agent` 补上 `--agent`，`write` 不再写进 checker 看不到的默认 workspace（很可能是 Compl. 至今为 0 的主因）**。所有改动均合成数据/官方源文件/代码核实/bash 逻辑验证，真实训练未验证。

### 已知限制 / 未解决
同 08-18，未变，另加：**六处修复（训练信号 + checkpoint 默认路径 + 503 暂停重试 + OPD hint 接线 + session 拆分 + `--agent` workspace 修复）均完全未在真实训练中验证**（`py_compile`/回归测试/代码核实通过不代表真实代理/SGLang/Megatron/OpenClaw CLI 链路行为符合预期）；`metaclaw_migration_20260818_182736` 的 checkpoint 因为训练信号 bug 不能代表训练方法真实效果，之前基于它做的"数字接近基线"讨论仍然是真实观察到的现象，但不能归因于"方法本身效果不明显"；`metaclaw_migration_20260819_132608` 的 Acc./Compl. 曲线因为 503 把大段天数空转吃掉，同样不能当成 Table 1 式的完整 30 天真实效果看；`metaclaw_migration_20260819_153518`/`173654` 已确认存在"越写越长、write 调用消失 + 同一天后续题目被拖垮 + write 写错目录"三层模式，前两者剩下的成因（下一轮静态反馈文案文不对题、step-judge 对空谈长分析打 +1）尚未设计修复方案，下一轮训练大概率仍会复现这部分症状，不是回归；MetaClaw-Bench 自己的离线评测路径（`_run_question`/`_run_group`）仍未修 `--agent` 缺口，这次没用到那条路径。

### 下一步（8/19 视角，已被 8/20 更新）
1. **OpenClaw-RL 复现**：同 08-17
2. **MetaClaw 迁移**：**重新提交训练的时机由用户判断**（CLI 明确要求不要热补正在跑的 `153518`/`173654`）。提交时带上今天全部六处修复，不显式设置 `SAVE_CKPT`，按迁移文档的冒烟清单逐项确认：`openclaw-rl-metaclaw-verdict-signal-skip` 专用日志出现、verdict 之后没有新的短 response 残片、`deterministic-reward` 后面是长 prompt 的正常提交、`UnboundLocalError` 次数为 0、暂停窗口期间日志出现 503 pause-retry 等待记录、失败题日志里出现 `[openclaw-rl-metaclaw-verdict-opd-hint] ... accepted K_i=1`、`day07-10` 这类"从某题起到收工全是 Context overflow"的模式基本消失、**真实 session key 是 `agent:metaclaw_agent:explicit:...`、写入文件出现在正确的 `workspace_{test_id}_*` 目录、checker stdout 不再清一色 `FAIL: cannot read`**。缺任何一条就停，不要继续训练；全部确认后再看这次训练的逐日 Acc./Compl. 曲线（这次 `Compl.` 是否终于脱离 0.0% 是最值得看的信号），同时用干净数据验证 CLI 提出的"静态反馈文案文不对题"/"step-judge 奖励空谈"这两个新假设，为下一步单独设计方案做准备。如果 Acc. 有提升，要分开归因"选择题不再被 overflow 拖累"和"OPD 是否真的让 file_check 变好"
3. 其余同 08-17

（事后来看，08-19 提到的这六处修复到 08-20 已经真的提交跑了一轮——`metaclaw_migration_20260820_*`，跑到 day12 效果不错，本文档下一节记录后续。）

---

## 历史状态（2026-08-18，已被 8/19 结果取代）

### 已就绪
**OpenClaw-RL Separate/Personal Agent Track**（同 08-13，未变——见下方"历史状态（2026-08-13）"完整列表）。
**MetaClaw 迁移**：同 08-17，另加：[x] 网关鉴权、context overflow、`rl-training-headers` 插件白名单排除，三个依次挡路的基础设施根因均已定位并修复；[x] Acc./Compl. 改成边训练边算分（跟论文 Full 档方法学对齐，唯一产出方式）；[x] 按天断点续跑，手动触发，修复空天误判"已完成"的漏洞；[x] 训练 report 格式对齐官方 `report.json`/`report.md`，默认自动落盘；[x] 训练日志补上人类可读转录，修复了两处可见性问题（缓冲/终端不显示）；[x] **训练前基线已定版**（对齐基线，Acc.=8.1%/Compl.=0.0%，`run_20260818_141305`）。**[x] 第一次真正产生训练样本的训练已跑完**（`metaclaw_migration_20260818_182736`，234 样本、13 步、checkpoint `iter_0000009`）——Acc.=8.3%/Compl.=0.0%，数字接近基线，逐日模式"先好后差"（day01 38%→day11 后塌到 0），需要进一步分析转录判断下一步方向。

### 已知限制 / 未解决
同 08-17，未变，另加：**第一次真正训练的结果不理想**（跟基线几乎打平，`Compl.` 全程 0，`day11` 后 Acc. 塌陷）——需要看逐日转录判断根因（训练不稳定/reward 设计问题/OPD hint 质量/别的原因），还没做这个分析；"对齐/不对齐基线 Acc. 差异"跟"`plugins.allow` 无条件排除"这两个结论之间的矛盾仍未解开；转录终端可见性修复（`tee` 进程替换）尚未在真实训练中验证。

### 下一步
1. **OpenClaw-RL 复现**：同 08-17
2. **MetaClaw 迁移**：**分析这次训练的逐日转录和 `report.md`**，判断"先好后差"的具体原因（哪一天/哪一轮开始表现变差、模型具体在做什么导致塌陷）——这是决定下一步要调什么（学习率/reward/OPD hint/别的）的关键输入；下次重新提交训练时确认 `tee` 修复生效（终端能直接看到转录，不用单独 `tail -f`）
3. 其余同 08-17

### 未验证
- [ ] **"先好后差"塌陷模式的具体原因**——本次迁移当前最需要弄清楚的问题
- [ ] "对齐/不对齐基线 Acc. 差异" vs "`plugins.allow` 无条件排除插件"这两个结论之间的矛盾，具体机制是什么
- [ ] `tee` 进程替换修复在真实训练里是否真的解决了终端可见性问题
- 其余同 08-17（见上）

---

## 历史状态（2026-08-17，已被 8/18 结果取代）

### 已就绪
**OpenClaw-RL Separate/Personal Agent Track**（同 08-13，未变——见下方"历史状态（2026-08-13）"完整列表）。
**MetaClaw 迁移**：[x] 启动脚本 + `METACLAW_MAX_DAYS` 冒烟测试开关；[x] 训练信号安全性核查（基础设施故障保护已修复，A/B/D 确认不需调整）；[x] 训练起点/checkpoint 策略确定；[x] modelfactory 两项训练前准备（MetaClaw-official 克隆、仓库同步）已完成；[x] 三方对照表、完整设计文档。**第一次真实训练已提交，等待次日结果**。

### 已知限制 / 未解决
**OpenClaw-RL Separate/Personal Agent Track**（同 08-13，未变，见下方历史状态）。
**MetaClaw 迁移**：新步骤判官 prompt 零历史数据验证；中间轮次缺确定性锚点这个风险差异未处理（暂缓，见查证记录）；跨 round 污染 bug 未修（需要真实数据判断是否值得修）；`BENCHMARK_BASE_URL` 等 URL 形状假设未在真实链路验证——这次真实训练能直接检验这几条。

### 下一步
1. **OpenClaw-RL 复现**：提交新一轮训练，验证 Step 1 八/九次修订 + skip-forced-negative-override 诊断实验的真实效果（用户自行提交）
2. **MetaClaw 迁移**：**查看这次已提交训练的结果**——重点确认 `session_id` 是否正确以 `metaclaw-` 开头传递、`[openclaw-rl-metaclaw-deterministic-reward]`/`[openclaw-rl-metaclaw-step-judge]` 两个标记有没有出现、checker 是否真实执行、`agent_succeeded=False` 的触发频率和时机
3. 其余同 08-13（见下方历史状态）

### 未验证
- [ ] **本次已提交训练的真实结果**（session_id 传递、verdict/步骤判官分派是否生效、checker 真实执行情况）
- [ ] Step 1 八/九次修订 + skip-forced-negative-override 诊断实验的真实训练效果
- [ ] MetaClaw 新步骤判官 prompt 对 `run_command` 调用的判断质量
- [ ] 两个 round 之间会不会有杂音请求落进代理（决定要不要给中间轮次补确定性锚点）
- [ ] `agent_succeeded=False` 的真实触发频率/时机（决定跨 round 污染 bug 值不值得精确修）
- [ ] Simulator 拒绝合格 turn-1 是否在污染 PRM 打分（承接 08-11，仍未验证）

---

## 历史状态（2026-08-14，已被 8/17 结果取代）

- [x] MetaClaw 论文理解 + 官方代码克隆；迁移方案确认（目标/范围/方法映射/验收标准）；round 内多步 tool-call 训练信号设计（方案 B）；第一批实验文件（rollout driver + 代理侧 checker 奖励接入）本地验证通过

---

## 历史状态（2026-08-13，已被 8/14 结果取代）

### 已就绪
（同 08-11，未变——见下方"历史状态（2026-08-11）"完整列表。另加：Step 1 八次修订已实现并本地验证；408/503 污染信号 Wave 1（A+B）+ Wave 2（D）均已实现、本地验证、并用真实 094000 数据核实过对训练标签的实际影响，尚未用于新一轮实际训练验证线上效果）

### 已知限制 / 未解决
（同 08-11，未变——见下方"历史状态（2026-08-11）"完整列表。另加：C（网关/SGLang 时序错位）和 P17 `edit` 误用问题（潜在 Wave 3）设计已明确但未实现，均判定为非紧急；OpenClaw 展示层在成功 write 之后仍挂旧的 `lastToolError` 警告是独立的产品/展示层 bug，不在本项目补丁范围）

### 下一步
1. 提交新一轮训练，验证：Step 1 八次修订能否降低 Simulator 对合格 turn-1 的误判率；Wave 1+2 补丁上线后 A 类错 `+1`/乱码蒸馏是否真的消失
2. 视效果决定要不要投入 C（插件 `agent_end` 通知）或 Wave 3（`edit` 误用/`lastToolError` 展示 bug）
3. 其余同 08-11（见下方历史状态）

### 未验证
- [ ] Step 1 八次修订对 Simulator 误判率的真实线上影响
- [ ] Wave 1（A+B）+ Wave 2（D）上线后，A 类错 `+1` 是否真的消失，B/D 的实际触发频率
- [ ] Simulator 拒绝合格 turn-1 是否在污染 PRM 打分（承接 08-11，仍未验证）

---

## 历史状态（2026-08-11，已被 8/13 结果取代）

### 已就绪
（同 08-10，未变——见下方"历史状态（2026-08-10）"完整列表。规则 5 补充 OPD hint 纠正信号——`is_repeat_thinking_violation` 标记 + `accepted` 强制替换，已实现并本地验证。**STUDENT_SYSTEM_PROMPT 七次修订已撤销**（`git revert`，非破坏性，六次修订保留），当前仓库状态跟本次训练 170852 实际运行代码对齐，已用 `md5sum` 核对一致）

### 已知限制 / 未解决
（同 08-10，未变——见下方"历史状态（2026-08-10）"完整列表。另加：**170852 实锤 turn-1 不收敛的真实瓶颈**——policy 侧已学会给出合格 Turn1，Simulator 几乎 100% 误判打回（10/11），③型误判比②更多，还出现规则外的"编造"反对理由，判定为 Simulator 本身对"turn 1 是否够好"存在系统性偏严倾向，不是某条具体规则文字不精确；`separate_student_20260811_141207` 当时"训练量不足"的猜测已被这批更新的数据推翻）

### 下一步
1. **讨论如何解决 Simulator 系统性拒绝合格 turn-1 答案**——当前最优先，方向待定（候选：全局"默认接受、除非能具体举证违反哪条规则"的框架调整 / 重新评估 DeepSeek V4 替代 Qwen3-32B 的校准问题 / 核实误判是否在直接污染 PRM 打分）
2. 单独处理 P18 型"合格后跳过写入直接收工"——跟①②③过严方向相反，不要混在一起解决
3. 其余同 08-10（见下方历史状态）

### 未验证
- [ ] Simulator 拒绝合格 turn-1 答案是否真的在通过 `_build_prm_eval_prompt` 直接污染训练奖励信号（待 CLI 核实具体 turn 的 eval_score）
- [ ] 重新设计的下一版修复能否降低 Simulator 对合格 turn-1 的误判率
- [ ] P18 型"跳过写入直接收工"的发生率和根因

---

## 历史状态（2026-08-10，已被 8/11 结果取代）

### 已就绪
（同 08-07，未变——见上方"历史状态（2026-08-07）"完整列表。另加：`STUDENT_SYSTEM_PROMPT` 五次修订——Step 1 从两分支重构成三分支（真答案正向枚举 / 只有答案没过程新增检测 / AI 味重写），均已实现并本地验证；训练循环图已重排，homework 三文件对齐三角色位置）

### 已知限制 / 未解决
（同 08-07，未变，见上方"历史状态（2026-08-07）"完整列表。另加：本轮 Student 侧修复解决的是"复读为什么会发生"这条上游链路 + 补上"催过程"检测缺口，明确不解决 tool_call XML 元循环二次崩溃、也不是给 length 加生成侧物理硬顶——这两个方向仍是待办，视这轮效果再评估。**新发现：P54 型偶发死锁**——催写入后模型反复 read 不写，context 撑爆后同 session 内无法恢复；污染路径是"首次空转 read 仍拿 +1"+"死会话拖累同批次邻题"，不是重复 read 大量拿正分（Rule 1a 已覆盖）。四个候选对策（熔断/降阈值/补打分漏洞/调 reserveTokens）+ Step 3 具体纠偏话术，均已设计但**均未实施**，等这轮"重装环境重新训练"的结果再决定）

### 下一步
1. **等 CLI 把 OpenClaw + 4B 模型重装回干净状态、重新提交训练后的结果**——这是当前最优先的下一步
2. 根据这轮结果决定：Step 1 五次修订效果如何；P54 型死锁是不是环境本身的问题（重装后可能消失）还是训练特性导致的，需不需要落地方案 A/B/C 或 Step 3 纠偏话术
3. 视效果决定是否需要生成侧物理限位（frequency_penalty 等）作为补充
4. 其余同 08-07/08-06/07-29（见上方历史状态）

### 产出
- `scripts/prepare_openclaw_test_scripts.sh`：`STUDENT_SYSTEM_PROMPT` 五次修订（Step 1 三分支重构）
- `docs/personal_agent_dialogue_vs_training_loop.svg`/`.png`：训练顺序+作业文件传递移到顶部，与三角色位置对齐
- 详细核实过程见 [`issues_log.md`](issues_log.md) 2026-08-10 三条条目

### 未验证
- [ ] Step 1 五次修订能否同时降低假 Tier-0 触发频率、正确拦截裸答案、不误伤有过程的紧凑/大白话回答（已实现，待真实训练验证）
- [ ] 规则 5、规则 4（修复后）等 08-07 全部改动的真实训练效果——仍待验证，本轮训练是首次真正验证这批改动的机会
- [ ] P54 型死锁在重装环境后是否还会出现（尚未验证是不是环境残留问题）

---

## 历史状态（2026-08-07，已被 8/10 结果取代）

### 已就绪
（同 08-06，未变——见上方"历史状态（2026-08-06）"完整列表。另加：`STUDENT_SYSTEM_PROMPT` Steps 重构（真答案/AI 味两层判断 + 写入核实严格版 + 角色串戏防护）+ 规则 4（旁路写入判负分，**已修复"取值是 UUID 不是 student-hw-N-pid"这个致命 bug**）+ 规则 5（句子原样重复 ≥12 次判负分，阈值经真实 shadow 数据校准）均已实现并本地验证；`FIRST_MESSAGE_TEMPLATE` 补丁已完全撤销，逐字恢复论文原始措辞——这几处合起来是本轮训练要验证的核心改动组合，本轮训练已提交，**结果下周查看**）

### 已知限制 / 未解决
（同 08-06，未变，见上方"历史状态（2026-08-06）"完整列表。另加：
- 08-07 新训练暴露"格式癫痫+拒绝写文件"新失效模式，精确定位真正起始点是 P28（两个必要因子叠加：`FIRST_MESSAGE_TEMPLATE` 诱导的行为种子——已修（Steps 重构接手 + 开场白恢复原文）；PRM 判官把"催解题/假 approve"系统性判成 +1 的打分漏洞——**尚未修复**，CLI 认为这条比软化措辞更接近根因）
- P17 实锤：read/write 交替逃过 Rule 1a 检测，模型反复写入旁路文件 `17_solution.md` 被误判 +1——已用规则 4 修复"写错路径"这个具体子问题，但"交替读写、路径都对、只是没有实质进展"这个更通用的问题仍是待办
- Student 重写检测触发率随对话推进大幅下滑（P0-20 的 100% → P35-45 的 26%），原因未查
- "拒绝写文件"类失效目前没有专门检测规则，发生在对话末尾时会完全逃出 PRM 打分
- （暂不修）PRM 判分对"催写入=隐式认可"覆盖不精确（P4 实锤：`votes=[0,0,1]→eval_score=0.0`，应为 +1），从项目一开始就存在，留作候选修复项
）

### 下一步
1. **训练已提交，下周查看结果**——重点验证：`STUDENT_SYSTEM_PROMPT` Steps 重构 + `FIRST_MESSAGE_TEMPLATE` 完全恢复原文这个组合能否解决"格式癫痫+拒写"；规则 4 能否正确拦截 P17 型旁路写入、有没有误伤合法操作；shadow 句重复统计、顶格截断规则、PRM 规则 3、Rule 1a 通用化、单题异常容错是否都按预期生效
2. 根据下周结果决定：Steps 重构要不要回退（若引入新误判或没解决问题，见 issues_log 明确的回退标准）；要不要修"demand_solve/假 approve 不该判 +1"这条打分漏洞（若 Steps 修复不够彻底）；要不要给"响应文本格式违规"和"拒绝写入"加确定性检测规则
3. 其余同 08-06/07-29（见上方历史状态）

### 产出
- `scripts/prepare_patched_openclaw_opd.sh`：规则 3（NO_REPLY）+ 规则 4（旁路写入路径校验）+ 顶格截断标记 + shadow 句重复统计日志 + Rule 1a 通用化
- `scripts/prepare_patched_openclaw_combine_select.sh`：truncation-penalty 覆盖块
- `scripts/prepare_openclaw_test_scripts.sh`：`STUDENT_SYSTEM_PROMPT` Steps 重构（真答案/AI 味两层判断 + 写入核实 + 角色串戏防护）+ 单题异常容错；`FIRST_MESSAGE_TEMPLATE` 补丁已完全撤销
- 详细核实过程见 [`issues_log.md`](issues_log.md) 2026-08-07 全部条目（loopDetection 根因确认、格式癫痫诊断、P28 精确定位、PRM 隐式认可缺口、Steps 重构与两轮复核、规则 4、FIRST_MESSAGE_TEMPLATE 撤销）

### 未验证
- [ ] 本轮训练（已提交，**结果下周查看**）：Steps 重构 + `FIRST_MESSAGE_TEMPLATE` 恢复原文能否解决"格式癫痫+拒写"
- [ ] 规则 4 修复取值 bug 后能否真正触发（之前部署即死代码，从未真正生效过）、能否正确拦截旁路写入、有没有误伤合法的多文件操作
- [ ] 规则 5（句子重复 ≥12 判负分）能否解决"超长 thinking 空转顶垮训练"这条主线问题，thinking 增长趋势和 P28 式整段死亡是否消失
- [ ] Rule 1a 通用化、shadow 句重复统计、顶格截断规则、PRM 规则 3、单题异常容错——均延续自 08-06，仍待真实训练验证

---

## 历史状态（2026-08-06，已被 8/7 结果取代）

### 已就绪
（同 07-29，未变——见上方"历史状态（2026-07-29）"完整列表。另加：答辩用训练循环图两处表述问题已根据反馈修正；`tools.loopDetection` 开启 + `student_chat.py` 单题异常容错 + PRM 规则 3（NO_REPLY 误用判负分）+ `FIRST_MESSAGE_TEMPLATE` 去 bare-answer 歧义 + 顶格截断强制判负分 + shadow 句重复统计日志 + Rule 1a 通用化（去 read/write 白名单，豁免轮询）均已实现并本地验证）

### 已知限制 / 未解决
（同 07-29，未变，见上方"历史状态（2026-07-29）"完整列表。另加：`tools.loopDetection` 确认零触发根因是"本轮失效形态不在其设计检测范围内"（实测同 session 最大 exact 重复仅 9 次，远低于阈值 20），不是配置/时序问题，但也意味着它对本轮实际问题基本无效；唯一目前完全没有机制覆盖的一类失效是超长 thinking 原地复读（无固定工具调用，loopDetection 和精确匹配的 PRM 规则都覆盖不到）；`postCompactionGuard` 确认不适用于 07-29 的压缩死循环，那个问题仍无对策；"verbose CoT 自我强化"假说已被 n=342 全量数据推翻，真实模式是"失败轨迹在训练后期变长"这一更窄的现象，尚无对应修复；PRM 规则 3 已实现但尚未用真实训练验证效果）

### 下一步
1. 提交新训练，验证 PRM 规则 3（NO_REPLY 判负分）的真实生效情况
2. **新方向**：给"超长 thinking 原地复读"想对策——loopDetection 和现有精确匹配的 PRM 规则都管不到这类问题
3. 其余同 07-29（见上方"历史状态（2026-07-29）"完整列表）

### 产出
- `scripts/prepare_patched_openclaw_opd.sh`：新增规则 3（NO_REPLY 误用判负分）+ 顶格截断标记（`is_truncated`）+ shadow 句重复统计日志 + Rule 1a 通用化（去白名单、豁免轮询）
- `scripts/prepare_openclaw_test_scripts.sh`：`FIRST_MESSAGE_TEMPLATE` 去 bare-answer 歧义
- `scripts/prepare_patched_openclaw_combine_select.sh`：新增 `openclaw-rl-truncation-penalty` 覆盖块
- 详细核实过程见 [`issues_log.md`](issues_log.md) 2026-08-06 四条条目

### 未验证（08-07 已出结果，见下）
- `FIRST_MESSAGE_TEMPLATE` 方案 B 改动：**08-07 新训练证实这个改动是"格式癫痫+拒写"新失效模式的诱因，已改窄为方案 A**（详见下方 2026-08-07 记录）
- 其余（PRM 规则 3、顶格截断、shadow 句重复统计、Rule 1a 通用化、单题异常容错）仍待真实训练验证，未变

### 已确认结论
- `tools.loopDetection` 确认配置生效、非死代码；零触发是因为本轮失效形态（超长 thinking 复读、短答、NO_REPLY、顶格/空生成）本身不涉及"同参数工具调用重复≥20次"这个它唯一能检测的模式，不是配置/时序 bug

---

## 历史状态（2026-08-05，已被 8/6 结果取代）

### 已就绪
（同 07-29，未变——见上方"历史状态（2026-07-29）"完整列表。另加：答辩用训练循环图两处表述问题已根据反馈修正；`tools.loopDetection` 开启 + `student_chat.py` 单题异常容错均已实现并本地验证）

### 已知限制 / 未解决
（同 07-29，未变，见上方"历史状态（2026-07-29）"完整列表——07-30 提交的验证训练结果尚未汇报回来，本次训练是否解决了 Problem 19 型崩溃、两条 07-28 规则的真实触发效果均待确认。另加：`tools.loopDetection` 开启 + `student_chat.py` 单题容错均已实现，但尚未用真实训练验证效果；`postCompactionGuard` 确认不适用于 07-29 的压缩死循环，那个问题仍无对策）

### 下一步
1. 提交新训练，验证 `tools.loopDetection` 是否真实触发、能否提前拦截 Problem 17/19 型死循环、单题崩溃后是否能继续跑完剩余题目
2. 其余同 07-29（见上方"历史状态（2026-07-29）"完整列表）

### 产出
- `scripts/train_separate_student.sh`：新增 `tools.loopDetection.enabled true` 配置
- `scripts/prepare_openclaw_test_scripts.sh`：新增 `student_chat.py` 单题异常容错补丁
- 详细核实过程见 [`issues_log.md`](issues_log.md) 2026-08-05 条目

### 未验证
- [ ] `tools.loopDetection` 开启后能否有效阻止 Problem 17/19 型死循环（已实现，待真实训练验证）
- [ ] `student_chat.py` 单题异常容错后，训练能否在某题崩溃后继续跑完剩余题目（已实现，待真实训练验证）
- [ ] `loopDetection` 有没有误伤合理的重复调用（如轮询类工具）
（其余同 07-29，未变，见上方"历史状态（2026-07-29）"完整列表）

---

## 历史状态（2026-08-03，已被 8/5 结果取代）

### 已就绪
（同 07-29，未变——见上方"历史状态（2026-07-29）"完整列表。另加：答辩用训练循环图两处表述问题已根据反馈修正）

### 已知限制 / 未解决
（同 07-29，未变，见上方"历史状态（2026-07-29）"完整列表——07-30 提交的验证训练结果尚未汇报回来，本次训练是否解决了 Problem 19 型崩溃、两条 07-28 规则的真实触发效果均待确认）

### 下一步
（同 07-29，未变，见上方"历史状态（2026-07-29）"完整列表）

### 产出
- `docs/personal_agent_dialogue_vs_training_loop.svg`/`.png`：两处表述简化（07-31 工程细节、08-03 判官 M 票歧义）

### 未验证
（同 07-29，未变，见上方"历史状态（2026-07-29）"完整列表）

---

## 历史状态（2026-07-29，已被 8/3 结果取代）

### 已就绪
（同 07-28，未变——见上方"历史状态（2026-07-28）"完整列表。另加：Context overflow 死循环根因定位到 OpenClaw 自身压缩节流机制；长度膨胀问题定位到"重写循环主动加内容"这一具体机制；Simulator 提示词收窄改动已实现并本地验证）

### 已知限制 / 未解决
- **长度膨胀的具体成因已定位（重写循环导致模型主动加内容），已实施 Simulator 提示词收窄改动，但尚未用真实训练验证效果**——需要观察长度是否不再持续膨胀、真实 Turn1-干净率是否提升
- **这一改动是主动偏离论文原始 Simulator 提示词，不是复现 bug 修复**——去掉开放式"AI 味"判断后，emoji/场景化措辞等格式外内容可能不再被纠正、正大光明留在最终答案里，收敛数字可能变好看但不代表真正学会了论文期望的自然表达，结果汇报需明确说明
- **"Student 自己代答"（编题目/编解法/自己写完整步骤）的真正成因仍未查清**——已尝试的"Steps 第 0 条前置检查"方案经真实训练验证无效（Problem 21 仍然自己代答），已撤销，是当前最大的开放问题
- **两个独立问题待排查**：Problem 27 型"OpenClaw 无响应却被 Student 宣布已完成"（可能导致 `homework/` 产物混入假完成数据，影响 Phase B/D 复用）；Problem 25 型"Student 编造的问题描述跟 Policy 原始答案对不上，最终写入内容可能有误"（正确性问题）
- **"精确重复调用判负分"规则已扩展到 `write`（已实现，未用真实训练验证）**——Problem 19 崩溃根因是 `write` 精确重复调用因为每次都成功而持续被判官打正分，上下文无限膨胀直至拖垮网关；规则扩展后需要观察真实训练里是否还会出现同类崩溃
- **edit 死循环新增第三种独立根因，用户已决定暂不处理**：`oldText` 是纯空白/换行符时，OpenClaw 自身校验（`oldText.trim().length > 0`）必然拒绝，报错"Missing required parameter: edits"措辞误导，模型无法从报错反推真正问题，导致反复重试同一注定失败的调用（Problem 17，43 turn/24 分钟）——梳理过的训练端熔断/改 OpenClaw 报错信息两个方向均有明显顾虑，用户选择维持现状，接受为已知限制
- **收敛判定的正则本身分不清"干净的好回复"和"生成失败的错误提示"**——两者都可能因为不含 bold/numbered-list/boxed 而被误判为满足条件（07-29 实测发现 2 个假阳性案例），`check_convergence.py` 是否有同样漏洞尚未核查（用户明确表示这个不是当前优先级）
- OpenClaw 自身的压缩节流机制（`already_compacted_recently`）在超预算幅度很小时会造成永久性死循环，`/reset`/`/new` 均救不回来——这是 OpenClaw CLI 自身行为，非本项目补丁导致，暂无绕过方案
- 其余已知限制同 07-28（见上方历史状态）

### 下一步
1. 排查"Student 自己代答"为什么 Steps 第 0 条没能生效（是没被读到、还是被别的因素覆盖），再决定下一步方案
2. **已完成**：把"精确重复调用判负分"规则从只覆盖 `read` 扩展到 `write`（Problem 19 型崩溃的直接对策），已提交新训练等待验证；edit 死循环用户已决定暂不处理
3. 排查 Problem 27 型"假完成"的影响范围（`homework/` 产物是否需要清理重跑）、Problem 25 型答案正确性问题
4. 评估 Problem 42 型退化与"NO_REPLY/silent reply 幻觉"是否同一机制
5. 如果 GPU 有空余，测试 Qwen3-32B vs DeepSeek V4 的"AI 感判断"差异（也可对比换回 Qwen3-32B 后长度膨胀是否依然存在）
6. 下周导师会议后确定：TA/Teacher/Joint 是否彻底搁置、下一阶段方向选 General Agent（`toolcall-rl`）还是 SEA-Eval
7. 其余下一步同 07-28（见上方历史状态）

### 产出
- `scripts/prepare_openclaw_test_scripts.sh`：新增 Simulator 提示词 AI-like 开放式兜底移除补丁（Steps 第 0 条已实现但验证无效并撤销，不在当前代码里）
- 详细排查过程、Problem 20-30 逐题通读记录、Steps 第 0 条完整实现与撤销过程、Problem 19/17 崩溃根因见 [`issues_log.md`](issues_log.md) 2026-07-29 条目（含后续追加部分）

### 未验证
- [ ] Simulator 提示词收窄后，回复长度是否不再持续膨胀
- [ ] 真实 Turn1-干净率是否明显提升、能否凑出连续 3 个 session
- [ ] 去掉开放式兜底后，是否有格式外"AI 感"内容开始稳定留在最终答案里
- [ ] "Student 自己代答"的真正成因（开放问题，Steps 第 0 条已证实无效）
- [ ] Problem 27 型"假完成"的影响范围、Problem 25 型答案正确性问题
- [ ] "精确重复调用判负分"规则扩展到 `write` 后能否阻止 Problem 19 型崩溃（已实现，待真实训练验证）
- [ ] 两条 07-28 新规则在真实训练里的触发频率和效果（本次训练又被 Problem 19 崩溃打断，未跑完）
- [ ] Problem 33 型 edit 死循环是否能被新规则及时打断
- [ ] Problem 42 与 NO_REPLY/silent reply 幻觉是否同一机制
- [ ] "AI 感判断错位"现象是否是 DeepSeek V4 特有

---

## 历史状态（2026-07-28，已被 7/29 结果取代）

### 已就绪
（同 07-27，未变——见上方"历史状态（2026-07-27）"完整列表。另加：`edit` 反复失败根因已精确定位到 JSON 转义 bug；两条新 PRM 打分规则已实现并本地验证）

### 已知限制 / 未解决
- **两条新规则（工具错误结果通用判负分 + read 覆盖范围累积追踪）已实现、本地验证通过，但尚未用真实训练验证效果**——需要观察：(1) 两条规则各自的真实触发频率；(2) Problem 33 型 edit 死循环是否能在演变成 40-50+ 轮之前被打断；(3) 有没有误伤"合理的大文件分页读取"场景（设计上不应该，需要真实数据确认）
- **write/edit 内容语义正确性仍是明确留白**：新规则只覆盖"工具本身报错"和"重复读取无新信息"这类可以脱离任务语义判断的情况；"写/改的内容语义对不对"依然只能依赖 LLM 判官自己判断，没有确定性代码规则能覆盖
- 其余已知限制同 07-27（虚构文件名循环、`/reset`/`/new` 无法恢复 context overflow 等，见上方历史状态）

### 下一步
1. **已推送（commit `52e9c75`）、已 `git pull`、已重新提交训练**，等待跑起来后观察两条新规则的真实效果
2. 评估 Problem 42 型退化与"NO_REPLY/silent reply 幻觉"是否同一机制
3. 如果 GPU 有空余，测试 Qwen3-32B vs DeepSeek V4 的"AI 感判断"差异
4. 下周导师会议后确定：TA/Teacher/Joint 是否彻底搁置、下一阶段方向选 General Agent（`toolcall-rl`）还是 SEA-Eval
5. 其余下一步同 07-27（见上方历史状态）

### 产出
- `scripts/prepare_patched_openclaw_combine_select.sh`：新增 `openclaw-rl-tool-error-penalty` 规则
- `scripts/prepare_patched_openclaw_opd.sh`：新增规则 1b（read 覆盖范围累积追踪）
- 详细设计过程、bug 修正记录、验证方法见 [`issues_log.md`](issues_log.md) 2026-07-28 条目

### 未验证
- [ ] 两条新规则在真实训练里的触发频率和效果
- [ ] Problem 33 型 edit 死循环是否能被新规则及时打断
- [ ] read 覆盖范围追踪有没有误伤合理的大文件分页读取场景
- [ ] Problem 42 与 NO_REPLY/silent reply 幻觉是否同一机制
- [ ] "AI 感判断错位"现象是否是 DeepSeek V4 特有
- [ ] 其余同 07-27（见上方历史状态）

---

## 历史状态（2026-07-27，已被 7/28 结果取代）

### 已就绪
（同 2026-07-24，未变——见上方"历史状态（2026-07-24）"完整列表）

### 已知限制 / 未解决
- **Problem 42 型"永久退化"根因已查清（模型误用 sessions_send/sessions_yield 这类多 agent 工具，非训练权重瞬间损坏）；PRM 打分修正已实现并推送，已用真实训练验证补丁真实触发**（同一晚新训练里补丁触发 31 次，均正确覆盖为负分）——见下方"产出"和 [`issues_log.md`](issues_log.md) 相关条目
- **新发现（同一晚，补丁上线后）：`edit` 工具反复因"找不到精确匹配文本"而失败、PRM 判官大多仍打正分，导致某个 session 卡进 50+ 内部 turn 的死循环，进而拖垮 `submission` 流水线、最终整个训练任务被外部杀死**——这是今天补丁没有覆盖到的另一类退化模式（write/edit 内容语义盲区的具体实例），比"自问自答"问题后果更严重（不只是单题失败，是整个任务被杀）→ [`issues_log.md`](issues_log.md) 2026-07-27 最新条目
- **虚构文件名循环**：模型会把答案反复写去一个从未被提及的虚构文件（如 `N_answer.txt`/`N_answer.md`），不是单纯重复读同一个真实文件——这是比"重复调用同一工具"更具体的一种表现形式
- **write/edit 内容语义正确性仍是已知局限、这次没有解决，且已经实锤造成过一次任务被杀**：PRM 打分修正这次只覆盖了三条"逻辑上必然成立"的无效工具用法，"写入/编辑内容语义对不对"仍然只能依赖 LLM 判官自己读内容判断，没有确定性代码规则能覆盖——需要评估要不要针对"edit 明确返回找不到匹配文本"这种有明确错误标记、可以确定性判断的子情况单独加一条规则
- `/reset`/`/new` 在 context overflow 状态下并不能真正恢复 session（Problem 26 实测 5 次尝试全部失败）——官方错误提示的建议在这个状态下失效
- 其余已知限制同 07-24（见上方历史状态）

### 下一步
1. 评估要不要针对"`edit` 工具明确返回找不到匹配文本"这种情况单独加一条确定性判负分规则（思路类似"消息类工具需要真正送达"那条，靠 next_state 里的明确错误标记判断，不需要理解语义）
2. 评估 Problem 42 型退化与"NO_REPLY/silent reply 幻觉"是否同一机制，决定要不要合并处理
3. 如果 GPU 有空余，考虑用 `scripts/launch_simulator.sh` 起一次 Qwen3-32B 服务，测试它对已收集到的真实"正则通过但被要求重写"文本的判断，跟 DeepSeek V4 的结果对比，确认"AI 感判断错位"这个现象是不是 DeepSeek V4 特有的
4. 下周导师会议后确定：TA/Teacher/Joint 是否彻底搁置、下一阶段方向选 General Agent（`toolcall-rl`）还是 SEA-Eval
5. 其余下一步同 07-24（见上方历史状态）

### 产出
- `scripts/prepare_patched_openclaw_opd.sh`：新增 `is_invalid_tool_use` 判定逻辑（三条规则）
- `scripts/prepare_patched_openclaw_combine_select.sh`：新增读取该标记、覆盖 `eval_score` 的逻辑
- `scripts/train_separate_student.sh` + `scripts/run_openclaw_topk_select_modelfactory.sh`：新增训练可追溯性（`RUN_MANIFEST.txt` + wandb run 名字拼 git commit），已完成
- 详细设计过程和取舍见 [`issues_log.md`](issues_log.md) 2026-07-27 相关条目

### 未验证
- [ ] `edit` 反复失败判负分规则（如果决定做）的实际效果
- [ ] Problem 42 与 NO_REPLY/silent reply 幻觉是否同一机制
- [ ] "AI 感判断错位"现象是否是 DeepSeek V4 特有、换 Qwen3-32B 会不会有同样表现
- [ ] 其余同 07-24（见上方历史状态）

---

## 历史状态（2026-07-24，已被 7/27 结果取代——原文误标为「当前状态」，2026-08-28 迁移时更正）

### 已就绪
（同 2026-07-23，未变——见上方"历史状态（2026-07-23）"完整列表。另加：Execution Bias 全清空补丁已确认解决了当初瞄准的问题——Problem 31 那种"41 轮不收尾"的死循环没有复现；Separate-Student 收敛数字 22 已对齐论文 19.2；答辩用实验流程图已产出）

### 已知限制 / 未解决
- **训崩机制：** 模型存在"单 turn 内重复调用同一工具"的习惯性行为，通常无害、有界（能自行恢复），但个别情况会突变成无界循环（连续 32+ 次、从未自行恢复），导致 context 撑爆——真正的异常判断标准是"会不会自己终止"，不是重复次数多少。已定位很可能的根因（PRM 打分规则对"成功但重复"的工具调用一律打正分，训练信号里没有惩罚重复的机制），**但具体怎么修还没有最终确定**（本次一度实施又被撤回，见下）→ [`issues_log.md`](issues_log.md) 2026-07-24 条目
- Problem 36 无界循环的具体触发原因（是否对应训练 step 9 某个具体异常样本）尚未查清
- 其余已知限制（write/overwrite 核验失败是否 Qwen3-32B 特有、"先 Separate 后 Joint"结论未经官方参考验证等）状态不变，见上方"历史状态（2026-07-23）"

### 下一步
1. **本次已提交的诊断训练，结果下周再看**（用户明确要求，本周不再跟进）
2. 下周导师会议后确定：TA/Teacher/Joint 是否彻底搁置、下一阶段方向选 General Agent（`toolcall-rl`，已核实基础设施最轻、官方代码完整）还是 SEA-Eval（复用现有训练基础设施，但任务/reward 设计需从零开始）
3. PRM 打分规则的"重复调用惩罚"方案继续讨论定案后再实施——**当前代码库不含此项改动**，只有纯观测性质的重复检测日志
4. 排查 Problem 36 无界循环的具体触发原因：核对训练 step 9 对应训练 batch 里的样本/打分数据
5. 其余下一步同 07-23（见上方"历史状态（2026-07-23）"完整列表）

### 未验证
- [ ] Problem 36 无界循环的根本原因（训练 step 9 相关性 vs. 具体因果）
- [ ] PRM 打分"重复调用惩罚"方案定案后的实际效果（尚未实施）
- [ ] Execution Bias 补丁后的真实收敛 session 数（待训练结束后用修复过的脚本跑出）
- [ ] 其余同 07-23（见上方历史状态）

---

## 历史状态（2026-07-23，已被 7/24 结果取代——原文误标为「当前状态」，2026-08-28 迁移时更正）

### 已就绪
（同 2026-07-22，未变——见下方"历史状态（2026-07-22）"完整列表）

### 已知限制 / 未解决
- **架构层面确认：`train_with_services.sh` 现有的 Joint INIT 阶段不是论文真实做法**——完整四阶段方案（Separate-Student→Separate-TA→Separate-Teacher→Joint，错位复用产物）已定案，见 `docs/paper_reproduction_scope.md`，当前从 Phase A（`train_separate_student.sh`）开始实现
- `train_separate_student.sh` 已写完、本地语法检查通过，尚未推送/未在服务器实测
- **诊断实验：homework-verification-gate 补丁已移除，Simulator 临时换成 DeepSeek V4**（`prepare_openclaw_test_scripts.sh` 简化为只保留 model 字段兼容修复），用来判断此前反复出现的 write/overwrite 核验失败问题是 Qwen3-32B 这个具体模型能力不够，还是更深层不挑模型的设计问题——**这次实验的产出不算 Table 3 有效数据点**（论文 Section 4.1 明确用的是 Qwen3-32B），见 `docs/issues_log.md` 2026-07-23 条目
- **诊断实验初步结果非常正面**：72 题跑到第 25 题才第一次出现"异常"，且排查后确认第 25 题实际写入数据是干净的（`homework/25.txt` 内容完整正确），异常只是 OpenClaw 生成确认话术时撞上了训练步骤造成的长时间 503、Student 没识别出确认话术本身有问题，不影响真实文件内容——对比之前 Qwen3-32B 频繁出现的 write 覆盖/未写入问题，效果差异显著
- **CLAUDE.md 记录修正**：07-21 那条"OpenClaw 产品版本考古基准改成 05-12"的更正只反映了第一次尝试（错误地把该基准套到 OpenClaw CLI 工具本身），没有同步用户当场的纠正和后续查证结论——已直接修正 CLAUDE.md：05-12 基准只适用于 `OpenClaw-RL-official`（论文训练代码）的 v1/v2 判断，OpenClaw CLI 工具本身版本归属另算。**当前工作假设：基本可以确定论文作者用的是三月版本的 OpenClaw CLI**（从未锁定版本号、多个新增行为零 workaround、bring-your-own-openclaw 设计，综合判断）。
- **"session 持久化架构五月才有"这个结论已收回**：路径 `src/agents/sessions/agent-session.ts` 在三月确实搜不到，但三月版本在完全不同路径（`src/agents/pi-embedded-runner/`，包一层外部依赖 `pi-coding-agent` 自带的 SessionManager）下有等价机制，五月只是"内部化重写+改名"，不是凭空新增能力——之前判断是搜错路径导致的假阴性，这个方向的可修复性重新变回未知。
- **重大突破：坐实 Execution Bias 章节第 2 条有害指令是"No response"和"context overflow"两个问题的共同根因，已打补丁，待验证**（见 `docs/issues_log.md` 2026-07-23 最新条目）：
  - 用 Problem 31（session `86d5e924-...`）完整未截断的原始记录 + `training.log` 交叉核对，确认"180 秒等待过久"根本不是排队/单次超长生成，而是**模型每隔 ~20 秒做一次决策，反复调用 `read`/`memory_search`（工具本身完全正常、结果正确）而不给出答案，一个 session 内部连续跑了 41 轮、持续 12 分钟自己停不下来**——Problem 1（No response）和 Problem 2（context overflow）确认是这同一个模型行为的两个表现，不是两个独立机制
  - 根因定位到 `## Execution Bias` 系统提示词章节的"Weak/empty tool result: vary query, path, command, or source before concluding."这一句——`memory_search` 两次返回空结果，这句指令明确要求换源再试、不要下结论，跟观察到的行为精确吻合（结构性强关联证据，非模型原始 `thinking` 文字逐字实锤——`training.log` 调试日志拿不到 `thinking` 原文，只有字符数）
  - **用户决策：不再逐条排查其余 5 行是否也有害，直接把整个 Execution Bias 章节（7 行全部）清空，只留标题**——4B 模型能力不足以正确判断"什么时候算够"，这类优化指令对弱模型是负面影响，且整章节本来就是三月版本没有的东西，没必要保留任何一条
  - **已实现**：`scripts/prepare_patched_sglang_execution_bias.sh` 从"删 1 行留 6 行"改为"全清空只留标题"，处理了一个技术细节（不能用空字符串覆盖，会被 OpenClaw 自己的 trim+falsy 判断打回默认值，改用纯标题字符串），本地模拟跑过完整补丁逻辑验证通过
  - 已排除的方向：sglang 请求优先级调度（暂停发生在到达 sglang 之前，无关）；`--sglang-max-running-requests`（实测 4096，远超实际并发量，不是瓶颈）
- **同时坐实 Problem 27 的 reward blindness 猜测**：那次有缺陷的 write（用 write 整体覆盖、丢了展示过的完整步骤）被 PRM 打了满分 `eval_score=1.0`，真实提交进了训练队列——首次用真实数据证实此前 07-22 记录的"write 覆盖动作本身是否被 PRM 正确扣分尚未坐实"这个猜测，答案是没有被扣分。
- 07-22 晚提交的训练又在几小时内失败——具体原因还没来得及排查，优先级低于当前诊断实验
- 其余已知限制（"NO_REPLY"/"who am I"幻觉、silent reply 退化等）状态不变，见 07-22 历史状态

### 下一步
1. **推送这次的 Execution Bias 全清空补丁，重新跑一次训练验证 Problem 31 这类"反复工具调用不收尾"行为是否消失**（当前任务，下次会话优先做）
2. 如果这次补丁有效，诊断实验的原始目的（write-compliance 问题是否 Qwen3-32B 特有）才能真正得出可信结论——之前的数据被这两个更根本的问题干扰了
3. "OpenClaw 把重试请求当新一轮完整处理"这个方向的可修复性重新变回未知，暂不深挖（外部依赖包无源码可查），优先级低于验证 Execution Bias 补丁效果
4. Separate-Student（Qwen3-32B 正式版）真实数据跑出后，验证"Joint 应该复用 Separate 产物"这个结论、以及"16 条样本触发一步训练"频率是否对得上收敛速度（这次诊断实验里已经首次实测到"凑够 16 条样本触发训练"这个机制真实生效，可以作为频率参考）
5. 排查 07-22 晚训练为什么又是几小时内失败
6. 待 Separate-Student/TA/Teacher 全部完成后，重新设计 `train_with_services.sh` 的 Joint 阶段（去掉 INIT，直接消费 Separate 产物，三角色从一开始就同时启动）——**注意：Joint 阶段三角色并发会让 No response/context overflow 这类问题明显加重（并发请求量翻倍以上），设计时要考虑这一点**
7. 其余待办（PRM turn 内容调试补丁验证、not_requested 效果验证等）延后到架构问题理清之后

### 未验证
- [ ] **Execution Bias 全清空补丁是否真的解决了"反复工具调用不收尾"的问题**——刚打完补丁，还没有真实训练数据验证效果
- [ ] **write/overwrite 核验失败问题是否是 Qwen3-32B 特有**——本次 DeepSeek V4 诊断实验被更根本的上游问题干扰，原始诊断目的还没有可信结论
- [ ] "先跑完 Separate 再复用产物启动 Joint"这个结论——目前是逻辑推导出的最自洽版本，没有直接的官方参考实现证实
- [ ] Table 3 收敛 session 数很短（TA 最快不到 10 题）跟"16 条样本触发一步训练"这个频率对不对得上——等 Separate-Student（Qwen3-32B 正式版）真实数据核对每题实际产生多少条可训练样本
- [ ] 其余同 07-22（见下方历史状态）

---

## 历史状态（2026-07-22，已被 7/23 架构重新理解取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测）
- [x] `maxTokens=8192`、`reserveTokensFloor=16384`、`logit_bias` 屏蔽已知乱码 token、`memory-core` 插件禁用、退化样本过滤规则：均已用真实 GPU 数据验证生效
- [x] **5 个 OpenClaw 版本漂移补丁确认保留**：Execution Bias、context-overflow overflow-recovery、Assistant Output Directives、cli-compaction（均已用真实训练数据验证生效）+ Silent Reply Policy（本地测试通过、已接入三个训练脚本，服务器真实部署待验证）
- [x] write 覆盖导致 PRM 误判正分：已用真实数据实锤证实（Problem 11 两次独立训练命中同一模式）
- [x] **Student/TA/Teacher 会话级文件核验（v4：诊断分支跳过 32B、直接给出具体缺失内容）**：**已用真实数据验证生效**——Problem 0-4 全部在给出具体缺失内容后下一轮一次性正确修复（5/5），对比此前 Problem 8/9 的 10/10 失败
- [x] **workspace 从 `/root` 迁到 `/dfs/data/openclaw-rl-project/runtime/<run_id>/workspace`**：`agents.defaults.workspace`（openclaw.json，优先级高于环境变量）每次启动前强制覆盖 + `OPENCLAW_WORKSPACE_DIR`/`OPENCLAW_WORKSPACE` 双环境变量同步，三训练脚本统一改，本地语法检查通过，服务器真实部署待验证
- [x] **PRM turn 内容调试补丁**（`prepare_patched_openclaw_combine_select.sh`，临时诊断）：给 `openclaw_combine_select_api_server.py` 加一行日志，打出每个 turn 实际打分用的 response_text/next_state_text，本地测试通过，服务器真实部署待验证
- [x] **not_written 纠正消息新增 `not_requested` 变体**：`_write_was_requested()` 判断此前有没有真的要求过写入，没要求过就不再暗示"之前写过"，本地测试通过（含真实 Problem 10/11 数据 + 两次假阳性修正），服务器真实部署待验证

### 已知限制 / 未解决
- **新发现：策略模型全程只用 `write` 整体覆盖，从未用过 `edit`**（真实 session trajectory 工具调用数据证实）——不打算给策略模型加工具选择指引（开外挂），依赖奖励信号本身引导，但奖励信号是否真的在起作用还未坐实（见下一条）
- **新发现：write 覆盖动作本身是否被 PRM 正确扣分尚未坐实**——间接证据（turn 数量、投票特征）支持"reward blindness 依然存在、真正做错事的 write 动作没被罚、罚的是隔壁那轮确认回复"这个推测，但无法从现有日志 100% 确认，已加调试补丁（见上）等下次训练数据验证
- **新发现：模型偶发陷入"NO_REPLY 幻觉"/"who am I 幻觉"退化生成状态**（真实字面输出 "NO_REPLY" 文本或"Hey. I just came online. Who am I?"这类脱离上下文的回复，非系统层面的真正静默、也不是 session/prompt 污染导致）——跟已部署的 Silent Reply Policy 补丁是两回事，那个补丁修不了这个；**已确认不是本项目任何一次补丁（含今天新加的、含更早的 5 个 OpenClaw 版本漂移补丁）引入的新问题，"NO_REPLY" 早在 07-21 就已存在**；用户认可会污染训练数据，但仍属于已多次要求延后处理的范畴，本次决定先记录、继续观察
- **新发现：Problem 36 起 max-turns 激增 + "silent reply protocol"幻觉退化**，疑似与早期坏样本（Problem 4/11）训练带偏有关，但未做到 step 级别实锤，需要更精确的 `weight_version` 对照才能确认；结合本次"NO_REPLY 幻觉"发现，这两者可能是同一类退化生成现象的不同表现
- 批次污染→自我强化这个机制本身未被拦截（用户明确要求延后）
- 数学应用题反复自我重述 / 纯 token 退化，仍认为更可能是模型固有倾向，未做版本考古严格验证
- **8GPU 正式训练从未真正保存过 checkpoint**，每次任务提交都是从 base 模型重新开始
- `run_init_phase()`/`run_one_persona()` 缺乏阻塞机制的设计缺陷仍未修

### 下一步
1. **（已完成）** 提交（git commit + push）PRM turn 内容调试补丁 + not_requested 纠正消息修复；服务器已 `git pull`、清理残留 GPU 进程、重新提交训练（`bash train_with_services.sh`）——**明天查看这次训练结果**
2. 确认新的 `[openclaw-rl-debug-turn-content]` 日志真实出现，用它精确核对"write 覆盖动作本身有没有被正确扣分"这个悬而未决的问题
3. 观察类似 Problem 10 这种"Student 违反协议提前说完成"场景下，`not_requested` 纠正消息是否让对话更顺畅
4. 观察 write 覆盖/未完成却判定成功/"NO_REPLY 幻觉"/"who am I 幻觉"这几类问题的发生率变化
5. 视需要，按真实 `weight_version` 精确核对这几类幻觉退化与 Problem 4/11 坏样本训练 step 的先后关系；如需进一步确认是否与更早的 5 个 OpenClaw 版本漂移补丁有关，需要对比这些补丁部署前的更早期训练数据
6. **（用户明确要求延后）** 训练数据批次污染拦截；调小 `--save-interval`

### 未验证
- [ ] Silent Reply Policy 补丁在服务器真实部署文件上的锚点命中与实际效果（已确认这个补丁不能修"NO_REPLY 幻觉"，但仍可能对真正的系统级静默有效）
- [ ] PRM turn 内容调试补丁在服务器真实部署上的实际效果——能否精确定位"write 覆盖动作本身的分数"
- [ ] not_written 的 `not_requested` 纠正消息变体在服务器真实部署上的实际效果
- [ ] "NO_REPLY 幻觉"/"silent reply protocol"退化与早期坏样本训练的因果关系（目前只有时间线支持，未到 step 级别实锤）
- [ ] workspace 迁到 `/dfs/data` 后 `agents.defaults.workspace` 真的按新路径生效（服务器日志核实）；此前记录的"GPU 空闲回收重启后 workspace 静默回滚到快照"这个风险在新路径下是否真的不再出现（需要一次真实的空闲/重启周期才能验证）
- [ ] 8 GPU 正式 Table 3 训练完整跑通

---

## 历史状态（2026-07-21 上午，已被下方结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测）
- [x] `maxTokens=8192`、`reserveTokensFloor=16384`：已验证生效（07-15）
- [x] `logit_bias` 屏蔽已知乱码 token（id=122362）：**已用真实 GPU 数据验证 0 次复现**
- [x] `memory-core` 插件禁用：**已用真实 GPU 数据验证 0 次复现**
- [x] 退化样本过滤规则（只拦真正空内容 + 已知乱码 token 兜底）：**已验证生效**
- [x] "决策犹豫循环"诱因一：Execution Bias 章节 + 修复（`prepare_patched_sglang_execution_bias.sh`）：**已用真实训练数据验证钩子真实生效**
- [x] "503 崩溃"根因（`submission_enabled` 暂停窗口 + 重试预算过短）+ 修复（`--max-retries 8`）：**已用真实训练数据验证生效**
- [x] "决策犹豫循环"诱因二：context overflow 死锁（`agent-session.ts` "Already compacted"，`run.ts` 入口）+ 修复（`prepare_patched_embedded_agent_overflow_recovery.sh`）：**已用真实训练数据验证 0 次复现**
- [x] "决策犹豫循环"诱因三：Assistant Output Directives 章节 + 修复（`prepare_patched_system_prompt_output_directives.sh`）：本地测试通过，已接入三训练脚本，**真实训练效果待验证**
- [x] 假完成声明根因一：同一个"Already compacted"报错的第二入口（`cli-compaction.ts` 的 `cli_budget` 预压缩检查）+ 修复（`prepare_patched_cli_compaction.sh`）：**已用真实训练数据验证生效**（run `20260721_122947`，internal error 不再出现）
- [x] 假完成声明根因二：`write` 工具误用（模型用整体覆盖代替追加）+ 修复（`prepare_patched_write_edit_guidance.sh`）：**已接入正式训练（run `20260721_152519`），Problem 0/1 真实数据确认追加成功、文件结构完整保留**

### 已知限制 / 未解决
- 批次污染→自我强化这个机制本身未被拦截（用户明确要求延后），任何未来新出现的、能让某个 session 连续卡住的诱因都可能再次触发同样的训练级联损害
- 数学应用题反复自我重述 / 纯 token 退化这两类模式，目前认为更可能是 Qwen3-Thinking 自身固有倾向而非 OpenClaw 版本问题，未做版本考古严格验证
- **8GPU 正式训练从未真正保存过 checkpoint**（`--save-interval 100`，历次任务在崩溃前都没跑到过第 100 步），每次任务提交都是从 base 模型重新开始
- workspace 的 2GB 配额区在"GPU 空闲→平台自动回收→重启"后会静默回滚到上次保存的快照
- `run_init_phase()`/`run_one_persona()` 缺乏阻塞机制的设计缺陷仍未修
- 两个新补丁目前只用少量样本（Problem 0/1 等）确认"没复发已知问题"，还没做过大规模统计（类似之前 0-25 题扫描）来量化实际改善幅度

### 下一步
1. run `20260721_152519` 跑一段时间后，统计假完成声明 / 文件结构丢失的发生率，跟修复前基线对比（14/26=53.8% 结构丢失、14.3%~40.5% 假声明，取决于统计口径）
2. 确认已知诱因是否已覆盖"决策犹豫循环"/"假完成声明"两大类问题的全部来源，还是仍有其他未发现的诱因
3. 视情况重新评估"训练数据批次污染拦截"这个通用兜底方案的优先级
4. **（用户明确要求延后）** 调小 `--save-interval`；workspace 迁移到 `/dfs/data`；去掉 `run_init_phase()` 里无条件的 `rm -rf`

### 未验证
- [ ] 两个新补丁叠加后，假完成声明/结构丢失发生率的实际改善幅度（大规模统计）
- [ ] Assistant Output Directives 修复的实际效果
- [ ] 是否还有其他未发现的决策犹豫循环/假完成声明诱因
- [ ] 8 GPU 正式 Table 3 训练完整跑通

---

## 历史状态（2026-07-21 晚间，已被下方结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测）
- [x] `maxTokens=8192`、`reserveTokensFloor=16384`、`logit_bias` 屏蔽已知乱码 token、`memory-core` 插件禁用、退化样本过滤规则：均已用真实 GPU 数据验证生效
- [x] 假完成声明根因一：`cli-compaction.ts` 的 `cli_budget` 预压缩检查 + 修复（`prepare_patched_cli_compaction.sh`）：已用真实训练数据验证生效
- [x] write 覆盖导致 PRM 误判正分：已用真实数据实锤证实（Problem 11 两次独立训练命中同一模式）

### 重大未决问题（已解决，见下方 2026-07-22）
- 4 个已部署补丁（Execution Bias / context-overflow overflow-recovery / Assistant Output Directives / cli-compaction）的合法性——已解决

---

## 历史状态（2026-07-20，已被 7/21 结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测）
- [x] `maxTokens=8192`、`reserveTokensFloor=16384`：已验证生效（07-15）
- [x] `logit_bias` 屏蔽已知乱码 token（id=122362）：**已用真实 GPU 数据验证 0 次复现**
- [x] `memory-core` 插件禁用：**已用真实 GPU 数据验证 0 次复现**
- [x] 退化样本过滤规则（只拦真正空内容 + 已知乱码 token 兜底）：**已验证生效**
- [x] "决策犹豫循环"诱因一：Execution Bias 章节 + 修复（`prepare_patched_sglang_execution_bias.sh`）：**已用真实训练数据验证钩子真实生效**
- [x] "503 崩溃"根因（`submission_enabled` 暂停窗口 + 重试预算过短）+ 修复（`--max-retries 8`）：**已用真实训练数据验证生效**
- [x] "决策犹豫循环"诱因二：context overflow 死锁（`agent-session.ts` "Already compacted"）+ 修复（`prepare_patched_embedded_agent_overflow_recovery.sh`）：**已用真实训练数据验证 0 次复现**（run `20260720_112802`）
- [x] "决策犹豫循环"诱因三：Assistant Output Directives 章节 + 修复（`prepare_patched_system_prompt_output_directives.sh`）：本地测试通过，已接入三训练脚本，**真实训练效果待验证**

### 已知限制 / 未解决
- **新发现（07-20 晚，未解决）：假完成声明**——模型声称已写入文件但实际未写入，基线概率约 2%（两次对照训练均各 1 例），本次训练被放大到 14.3%；根因未定位，怀疑与三个已修复的"绕圈子"诱因被堵住后模型转向的替代退化路径有关（未证实）；关联官方 `DONE_SENTINEL`/PRM 打分不核实工具调用真实性这个设计空白
- 批次污染→自我强化这个机制本身未被拦截（用户明确要求延后），任何未来新出现的、能让某个 session 连续卡住的诱因都可能再次触发同样的训练级联损害
- 数学应用题反复自我重述 / 纯 token 退化这两类模式，目前认为更可能是 Qwen3-Thinking 自身固有倾向而非 OpenClaw 版本问题，未做版本考古严格验证
- **8GPU 正式训练从未真正保存过 checkpoint**（`--save-interval 100`，历次任务在崩溃前都没跑到过第 100 步），每次任务提交都是从 base 模型重新开始，此前训练进展不会累积，也意味着不同任务提交之间不存在权重层面的因果关系
- workspace 的 2GB 配额区在"GPU 空闲→平台自动回收→重启"后会静默回滚到上次保存的快照，训练**结束后**查看 workspace 文件状态需注意
- 官方 `DONE_SENTINEL` 完成判定不校验工具调用是否真的成功，是官方设计本身的空白
- `run_init_phase()`/`run_one_persona()` 缺乏阻塞机制的设计缺陷仍未修

### 下一步
1. 定位假完成声明第一次出现对应的具体训练 step 和批次样本构成
2. 确认 Assistant Output Directives 修复的实际效果
3. 确认已知诱因是否已覆盖"决策犹豫循环"大类问题的全部来源
4. 视情况重新评估"训练数据批次污染拦截"这个通用兜底方案的优先级

### 未验证
- [ ] Assistant Output Directives 修复的实际效果
- [ ] 是否还有其他未发现的决策犹豫循环诱因
- [ ] 8 GPU 正式 Table 3 训练完整跑通

---

## 历史状态（2026-07-17，已被 7/20 结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测）
- [x] `maxTokens=8192`、`reserveTokensFloor=16384`：已验证生效（07-15）
- [x] `logit_bias` 屏蔽已知乱码 token（id=122362）：**已用真实 GPU 数据验证 0 次复现**（run `20260716_182012` 全程 0 次）
- [x] `memory-core` 插件禁用：**已用真实 GPU 数据验证 0 次复现**（run `20260716_182012` 全程 0 次 `memory_get`）
- [x] 退化样本过滤规则（只拦真正空内容 + 已知乱码 token 兜底）：**已验证生效**，全程 0 次误触发
- [x] `tool_calls` 日志补 `session_id`：支持事后按 session 可靠关联分析
- [x] Git 远程地址改回直连 GitHub
- [x] "决策犹豫循环"根因定位（Execution Bias 章节，论文提交后新加）+ 修复（`prepare_patched_sglang_execution_bias.sh` 内容层 patch）：**已用真实训练数据验证钩子真实生效**（run `20260717_133740`/`20260717_171106`）
- [x] "503 崩溃"根因定位（`submission_enabled` 正常暂停窗口 + 重试预算过短）+ 修复（`--max-retries 8`）：**已用真实训练数据验证生效**（run `20260717_171106`）

### 已知限制 / 未解决
- 两个修复（execution-bias-fix、max-retries 8）都已确认真实触发，但**完整训练结果仍需要观察**——顶格截断次数会不会显著下降、类似量级的暂停窗口下重试会不会真的扛住，需要 run `20260717_171106` 跑完整个流程才能判断
- 新发现两个比 Execution Bias 更新的机制（PR #92191/#93073，2026-06-14/15 合并）：June 版本会自动重试"只有 thinking 没输出"的轮次，March 没有，可能复合放大循环伤害的影响尚未验证
- workspace 的 2GB 配额区（`~/.openclaw/workspace/`）在"GPU 空闲→平台自动回收→重启"后会静默回滚到上次保存的快照，训练**结束后**查看 workspace 文件状态时需要注意，不能直接当作训练进行中的真实状态
- 官方 `DONE_SENTINEL` 完成判定不校验工具调用是否真的成功，是官方设计本身的空白（假设待验证：是否为循环问题的上游诱因之一）
- `run_init_phase()`/`run_one_persona()` 缺乏阻塞机制的设计缺陷仍未修（07-14 起多次提及）

### 下一步
1. 观察 run `20260717_171106` 完整训练结果，确认顶格截断次数是否显著下降、503 崩溃是否不再出现
2. 如果仍有问题，评估是否需要处理 PR #92191/#93073 这类更新机制，或重新评估版本回退
3. 交叉比对"Student 假成功"题目与"TA 决策犹豫循环"session，验证因果关系假设（次要优先级）

### 未验证
- [ ] "决策犹豫循环"修复的实际效果（钩子已确认触发，结果待观察）
- [ ] "503 崩溃"修复的实际效果（重试预算已确认按 8 次生效，结果待观察）
- [ ] "Student 假成功"与"TA 决策犹豫循环"的因果关系
- [ ] 8 GPU 正式 Table 3 训练完整跑通

---

## 历史状态（2026-07-16，已被 7/17 结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测）
- [x] `maxTokens=8192`、`reserveTokensFloor=16384`：已验证生效（07-15）
- [x] wandb 集成：确认今天的连不上是 workspace 代理服务故障，不是训练本身问题，`start_tools.sh` 修复后正常
- [x] `logit_bias` 屏蔽已知乱码 token（id=122362）：生成阶段直接屏蔽，**已用真实 GPU 数据验证 0 次复现**（run `20260716_143407`）
- [x] 退化样本过滤规则：简化为只拦真正空内容（跟官方一致）+ 已知乱码 token 兜底，不再按 `content` 长度过滤
- [x] `memory-core` 插件禁用：解决 `memory_get` 工具干扰，**尚未验证新 run 实际效果**
- [x] `tool_calls` 日志补 `session_id`：支持事后按 session 可靠关联分析
- [x] Git 远程地址改回直连 GitHub（不再依赖易失效的 `ghproxy.net` 镜像）

### 已知限制 / 未解决
- `HEARTBEAT.md`/`AGENTS.md` 等 OpenClaw 核心自带"agent 身份文件"无法禁用（不是插件），只能靠退化过滤兜底，模型仍可能偶尔读到但预期影响远小于 `memory_get` 的卡循环模式
- grad_norm 缓慢爬升的**最初触发点**仍未 100% 定位——乱码 token、`memory_get` 卡循环只是两个已确认会放大问题的"下游因素"，是否还有更早的触发原因尚不清楚
- `run_init_phase()`/`run_one_persona()` 缺乏阻塞机制的设计缺陷仍未修（07-14 起多次提及）
- workspace 的 `sing-box` 代理服务偶发失效，需要手动 `start_tools.sh` 重启，暂无自动恢复机制

### 下一步
1. 查看 `20260716_182012` 训练结果，确认这批修复是否让训练稳定跑过之前失控的窗口
2. 如果 grad_norm 依然爬升，需要继续往前找最初触发点（不只是已知的两个放大器）
3. INIT+Joint 全部跑通后，观察 wandb 曲线 + `check_convergence.py` 结果

### 未验证
- [ ] `20260716_182012` 训练能否稳定跑完 INIT+Joint，不再复现 grad_norm 失控
- [ ] `memory-core` 禁用后 TA 撞轮次上限失败率是否显著下降
- [ ] 8 GPU 正式 Table 3 训练完整跑通

---

## 历史状态（2026-07-15，已被 7/16 结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测）
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次起 gateway 前强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`，`maxTokens` 已改为官方值 8192（原 4096 是历史遗留）
- [x] `rl-training-headers` 插件 + `openclaw_opd_api_server.py` 标记解析：**已用真实 GPU 数据验证生效**
- [x] `agents.defaults.compaction.reserveTokens=16384` + **新增 `reserveTokensFloor=16384`**：**已用诊断探测请求验证运行时真正生效**（此前 `reserveTokens` 单独设置对实际 precheck 无效，是 OpenClaw 官方已知 bug #66830）
- [x] wandb 集成：**已验证成功**，key 已改走环境变量不再暴露在 Command 字段
- [x] 系统内存 OOM 修复：**已验证**，A800/H20 上多次 `update_weights()` 无 OOM
- [x] GPU calloc / workspace 残留进程问题：机制已确认（Ctrl-C/中断后必须手动清理 `sglang::scheduler` 残留，07-15 又复现两次）
- [x] `run_one_persona()` 单次调用 + Joint 阶段一次性并发：设计已确认，但截至目前还没有一次 run 完整跑完 72 题验证到 Joint 阶段
- [x] `train/opd_loss` 常数 -1.0 现象排查完毕：确认真实教师信号占比 67%、`rho_v=1` 是架构必然，判定为早期训练正常现象，非 bug
- [x] wandb "important" 图表分组（10 张核心图）+ 个人工作区模板，后续新 run 自动套用
- [x] `scripts/check_convergence.py`

### 已知限制 / 未解决
- **8GPU 正式训练的 workspace 内存额度，之前一直是按 256GB 申请的，实测跑到 INIT 中途就用掉 78%、仍在爬升**——已改成 64CPU/1024GB 重新申请，但还没有一次完整 run 验证这个新额度是否真的够用
- `update_weights()` 触发的 pause/KV 缓存清空偶尔会打断"正在处理中"的对话请求，之后同一批 session 可能陷入持续性"生成结束但内容为空"的故障，不会自己恢复——怀疑跟内存压力有关联但未最终证实，是目前"跑到一半开始连续失败"的头号嫌疑
- workspace 模式下 GPU 长时间空闲（比如 rollout 饥饿）会触发 modelfactory 平台自动回收整个 workspace，训练进程和日志会毫无征兆地一起消失，本地日志查不出任何报错——`reserveTokensFloor` 修复降低了 context overflow 这个已知饥饿诱因，但内存压力也可能导致同样的饥饿链条，暂无系统性预防手段（讨论过 GPU keepalive 兜底，暂未实施）
- `run_init_phase()`/`run_one_persona()` 目前对"某个角色没跑完 72 题就崩溃"没有阻塞机制，只警告后放行，导致下一阶段可能建立在不完整数据上——07-15 三次 run 都因此产生了不完整/需要丢弃的数据，这个设计缺陷本身还没有修
- Joint 阶段三角色并发"超车"读空文件的风险仍未获得真实数据验证（至今没有一次 run 完整走到 Joint 阶段）
- 训练一结束就杀网关不等模拟循环跑完（已记录暂不修）
- `appendSystemContext` 标记多轮对话下的稳定性、context-summarization 是否触发 `before_prompt_build`，仍待验证
- workspace 自己的持久化存储配额只有 2GB（跟训练用的系统内存是完全独立的两个概念），关闭 workspace 前如果快超了要记得清理 `~/.openclaw/agents/main/sessions/`（历史对话转录）和 `~/.npm`（缓存），不要动 `/usr/lib/node_modules/openclaw`（本体）

### 下一步
1. 查看 `20260715_180549` 这次训练结果（64CPU/1024GB 新 workspace）：INIT 能否完整跑完 72 题、Joint 阶段能否正常推进，判断内存是否是根本瓶颈
2. 如果新内存额度下仍有角色跑不完 72 题，说明内存不是唯一原因，需要考虑给 `run_init_phase()` 加阻塞/重试机制（目前是已知设计缺陷，暂未修）
3. INIT+Joint 全部跑通后，观察 wandb 曲线 + 最终 `check_convergence.py` 结果
4. 8GPU 正式训练固定用同一种 GPU 架构和内存额度，不与其他方法/基线混用硬件配置

### 未验证
- [ ] 64CPU/1024GB 新 workspace 能否让 INIT 阶段三个角色都完整跑完 72 题（这是今天第三次尝试，前两次分别卡在 maxTokens 和 reserveTokensFloor，这次换了内存额度）
- [ ] `update_weights()` 打断在途请求导致的"持续性空回复"故障是否会因为内存余量变大而消失，还是内存无关、需要另外处理
- [ ] Joint 阶段"超车"现象是否显著影响数据质量（至今没有真实数据）
- [ ] 8 GPU 正式 Table 3 训练完整跑通

---

## 历史状态（2026-07-14，已被 7/15 结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测）
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次起 gateway 前强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`
- [x] `rl-training-headers` 插件 + `openclaw_opd_api_server.py` 标记解析：**已用真实 GPU 数据验证生效**
- [x] `agents.defaults.compaction.reserveTokens=16384` 强制设置：**已验证生效**（TA 产生真实回复）
- [x] wandb 集成：**已验证成功**，key 已改走环境变量不再暴露在 Command 字段
- [x] 系统内存 OOM 修复：**已验证**，A800/H20 上多次 `update_weights()` 无 OOM
- [x] GPU calloc / workspace 残留进程问题：**已验证解决**，`update_weights()` 已成功执行 30+ 次无崩溃（run `8yn4i8ml`）
- [x] `run_one_persona()` 改回单次调用（不再整体重跑丢数据）→ commit `0b25005`，**已用真实数据验证生效**（Student INIT 正常推进到 43/72 题，产出真实完整对话）
- [x] Joint 阶段改为一次性并发启动（不再是无官方依据的分轮次循环）→ commit `4be24ab`，**Joint 阶段本身尚未开始，仍待验证**（截至记录时 Student INIT 还在第 43/72 题）
- [x] `train/opd_loss` 常数 -1.0 现象排查完毕：确认真实教师信号占比 67%、`rho_v=1` 是架构必然，判定为早期训练正常现象，非 bug
- [x] wandb "important" 图表分组（10 张核心图）+ 个人工作区模板，后续新 run 自动套用
- [x] `scripts/check_convergence.py`

### 已知限制 / 未解决
- INIT 阶段 503 风暴根因未 100% 精确定位（已排除四个假设，见 [`issues_log.md`](issues_log.md)）
- Joint 阶段三角色并发可能"超车"读到空文件，官方无防护，已知不改，靠实测观察，Joint 阶段尚未开始所以还没有真实数据
- workspace 模式下手动中断（Ctrl-C）清理不彻底会残留 GPU 进程，每次重新提交前必须手动 `nvidia-smi` + `ps aux | grep "openclaw gateway"` 确认干净
- 训练一结束就杀网关不等模拟循环跑完（已记录暂不修）
- `appendSystemContext` 标记多轮对话下的稳定性、context-summarization 是否触发 `before_prompt_build`，仍待验证

### 下一步
1. 用今天改完的新版本（单次调用 + 一次性并发 Joint 阶段）重新提交 8GPU 训练
2. 确认 INIT 数据完整、Joint 阶段持续产出、训练正常推进后，观察 wandb 曲线 + 最终 `check_convergence.py` 结果
3. 8GPU 正式训练固定用同一种 GPU 架构，不与其他方法/基线混用硬件

### 未验证
- [ ] 新版 `run_one_persona()` + `run_joint_phase()` 在真实 8GPU 训练上的完整效果
- [ ] Joint 阶段"超车"现象是否显著影响数据质量
- [ ] 8 GPU 正式 Table 3 训练完整跑通

---

## 历史状态（2026-07-13，已被 7/14 结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测，flash-attn/APEX/TE/flashinfer 非 H20 专属编译）
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次起 gateway 前强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`（`contextWindow=32768`/`maxTokens=4096`），不再用静态 header
- [x] `scripts/prepare_patched_rl_training_headers.sh`：`rl-training-headers` 插件 `appendSystemContext` 版本，**已用真实 GPU 数据验证生效**（真实动态 `session_id`/`turn_type` 标记到达请求正文）
- [x] `scripts/prepare_patched_openclaw_opd.sh`：解析标记 + 转发前清理，**已用真实 GPU 数据验证生效**
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本已统一接入上述所有 workaround
- [x] `run_one_persona()` 网关断连重试修复，代码已就绪，**尚未被干净验证**（07-11 那次被 reserveTokens 问题盖住，见下）
- [x] `agents.defaults.compaction.reserveTokens=16384` 强制设置修复（TA/Teacher context overflow 根因），**已在 smoke 上验证生效**（TA 产生真实回复，不再是错误占位文本）
- [x] wandb 集成**已实测验证成功**（新提交方式 `代码解释器=/bin/bash -i /dfs/data/start_tools.sh && /bin/bash -i`，minitest/smoke 默认开启 `USE_WANDB=1`），wandb key 已改走环境变量不再暴露在 run 的 Command 字段里
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load` + `PATCHED_OPD_DIR` PYTHONPATH 注入
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）
- [x] 系统内存 OOM 修复：提高任务提交时申请的系统内存，A800 minitest 实测连续跑过 10 次 `update_weights()` 无 OOM

### 已知限制 / 未解决
- 训练一结束就立刻杀网关，不等模拟循环跑完（smoke、minitest 都复现过，已记录暂不修，见 [`issues_log.md`](issues_log.md) 2026-07-13 条目）
- minitest 共用同一个 checkpoint 路径，`--load` 自动续训会导致后续跑"续训到快完成直接结束"、验证结果失效——已清空 checkpoint，需要注意以后再犯（见 [`issues_log.md`](issues_log.md) 2026-07-13 第三条）
- 网关断连重试修复尚未被干净验证（偶发问题，需要真的撞上才能确认）
- ⚠️ **用户待办**：`openclaw_rl` wandb 项目当前是 Public，需要手动改回 Private；已暴露两次的 WANDB_API_KEY 需要去 wandb 网站撤销重新生成
- `appendSystemContext` 标记是否会污染 OpenClaw 自己持久化的对话历史，待更长多轮对话验证
- context-summarization 内部调用是否触发 `before_prompt_build`，待验证（决定是否顺带解决 main turn 误标问题）

### 下一步
1. 用户手动处理 wandb 项目权限 + 撤销重新生成 key（见上）
2. 清空 checkpoint 后重新提交 minitest，确认真正从头跑的完整流水线（INIT 数据完整 + 无 OOM），顺带确认新 run 的 Command 字段不再有 key
3. 提交 8 GPU 正式 Table 3 训练（`train_with_services.sh` 已就绪，需申请更高系统内存，wandb 集成已验证可用）
4. 8GPU 正式训练建议固定用同一种 GPU 架构（H20 或 A800 二选一，不要跟其他方法/基线的对比数字混用不同硬件）

### 未验证
- [ ] `run_one_persona()` 网关断连重试修复（偶发问题，待真实撞上验证）
- [ ] minitest 5 GPU 完整跑通（checkpoint 已清空，需重新提交，确认 INIT 数据完整 + 无 OOM）
- [ ] 新 run 的 Command 字段确认不再包含 `--wandb-key`
- [ ] `appendSystemContext` 标记多轮对话下的稳定性
- [ ] 8 GPU 正式 Table 3 训练

---

## 历史状态（2026-07-13，已被 7/14 结果取代——原文误标为「当前状态」，2026-08-28 迁移时更正）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测，flash-attn/APEX/TE/flashinfer 非 H20 专属编译）
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次起 gateway 前强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`（`contextWindow=32768`/`maxTokens=4096`），不再用静态 header
- [x] `scripts/prepare_patched_rl_training_headers.sh`：`rl-training-headers` 插件 `appendSystemContext` 版本，**已用真实 GPU 数据验证生效**（真实动态 `session_id`/`turn_type` 标记到达请求正文）
- [x] `scripts/prepare_patched_openclaw_opd.sh`：解析标记 + 转发前清理，**已用真实 GPU 数据验证生效**
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本已统一接入上述所有 workaround
- [x] `run_one_persona()` 网关断连重试修复，代码已就绪，**尚未被干净验证**（07-11 那次被 reserveTokens 问题盖住，见下）
- [x] `agents.defaults.compaction.reserveTokens=16384` 强制设置修复（TA/Teacher context overflow 根因），**已在 smoke 上验证生效**（TA 产生真实回复，不再是错误占位文本）
- [x] wandb 集成**已实测验证成功**（新提交方式 `代码解释器=/bin/bash -i /dfs/data/start_tools.sh && /bin/bash -i`，minitest/smoke 默认开启 `USE_WANDB=1`），wandb key 已改走环境变量不再暴露在 run 的 Command 字段里
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load` + `PATCHED_OPD_DIR` PYTHONPATH 注入
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）
- [x] 系统内存 OOM 修复：提高任务提交时申请的系统内存，A800 minitest 实测连续跑过 10 次 `update_weights()` 无 OOM

### 已知限制 / 未解决
- 训练一结束就立刻杀网关，不等模拟循环跑完（smoke、minitest 都复现过，已记录暂不修，见 [`issues_log.md`](issues_log.md) 2026-07-13 条目）
- minitest 共用同一个 checkpoint 路径，`--load` 自动续训会导致后续跑"续训到快完成直接结束"、验证结果失效——已清空 checkpoint，需要注意以后再犯（见 [`issues_log.md`](issues_log.md) 2026-07-13 第三条）
- 网关断连重试修复尚未被干净验证（偶发问题，需要真的撞上才能确认）
- ⚠️ **用户待办**：`openclaw_rl` wandb 项目当前是 Public，需要手动改回 Private；已暴露两次的 WANDB_API_KEY 需要去 wandb 网站撤销重新生成
- `appendSystemContext` 标记是否会污染 OpenClaw 自己持久化的对话历史，待更长多轮对话验证
- context-summarization 内部调用是否触发 `before_prompt_build`，待验证（决定是否顺带解决 main turn 误标问题）

### 下一步
1. 用户手动处理 wandb 项目权限 + 撤销重新生成 key（见上）
2. 清空 checkpoint 后重新提交 minitest，确认真正从头跑的完整流水线（INIT 数据完整 + 无 OOM），顺带确认新 run 的 Command 字段不再有 key
3. 提交 8 GPU 正式 Table 3 训练（`train_with_services.sh` 已就绪，需申请更高系统内存，wandb 集成已验证可用）
4. 8GPU 正式训练建议固定用同一种 GPU 架构（H20 或 A800 二选一，不要跟其他方法/基线的对比数字混用不同硬件）

### 未验证
- [ ] `run_one_persona()` 网关断连重试修复（偶发问题，待真实撞上验证）
- [ ] minitest 5 GPU 完整跑通（checkpoint 已清空，需重新提交，确认 INIT 数据完整 + 无 OOM）
- [ ] 新 run 的 Command 字段确认不再包含 `--wandb-key`
- [ ] `appendSystemContext` 标记多轮对话下的稳定性
- [ ] 8 GPU 正式 Table 3 训练

---

## 历史状态（2026-07-10，已被 7/13 结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测，flash-attn/APEX/TE/flashinfer 非 H20 专属编译）
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次起 gateway 前强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`（`contextWindow=32768`/`maxTokens=4096`），不再用静态 header
- [x] `scripts/prepare_patched_rl_training_headers.sh`：`rl-training-headers` 插件 `appendSystemContext` 版本，**已用真实 GPU 数据验证生效**（真实动态 `session_id`/`turn_type` 标记到达请求正文）
- [x] `scripts/prepare_patched_openclaw_opd.sh`：解析标记 + 转发前清理，**已用真实 GPU 数据验证生效**
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本已统一接入上述所有 workaround
- [x] `run_one_persona()` 网关断连重试修复（`minitest_train_with_services.sh` / `train_with_services.sh`），代码已就绪，待真实 job 验证
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load` + `PATCHED_OPD_DIR` PYTHONPATH 注入
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）
- [x] 系统内存 OOM 修复：提高任务提交时申请的系统内存，A800 minitest 实测连续跑过 10 次 `update_weights()` 无 OOM

### 已知限制 / 未解决
- INIT 阶段网关断连重试修复尚未在真实 job 上验证
- `appendSystemContext` 标记是否会污染 OpenClaw 自己持久化的对话历史，待更长多轮对话验证
- context-summarization 内部调用是否触发 `before_prompt_build`，待验证（决定是否顺带解决 main turn 误标问题）

### 下一步
1. 停掉当前 A800 minitest（已达成验证目的），拉最新代码重新提交，验证网关断连重试修复
2. minitest 完整跑通（INIT 数据完整 + 无 OOM）后提交 8 GPU 正式 Table 3 训练（`train_with_services.sh` 已就绪，需申请更高系统内存）
3. 8GPU 正式训练建议固定用同一种 GPU 架构（H20 或 A800 二选一，不要跟其他方法/基线的对比数字混用不同硬件）

### 未验证
- [ ] `run_one_persona()` 网关断连重试修复
- [ ] minitest 5 GPU 完整跑通
- [ ] `appendSystemContext` 标记多轮对话下的稳定性
- [ ] 8 GPU 正式 Table 3 训练

---

## 历史状态（2026-07-09，已被 7/10 A800 实测结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次起 gateway 前强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`（`contextWindow=32768`/`maxTokens=4096`），不再用静态 header
- [x] `scripts/prepare_patched_rl_training_headers.sh`：`rl-training-headers` 插件 `appendSystemContext` 版本，**已用真实 GPU 数据验证生效**（真实动态 `session_id`/`turn_type` 标记到达请求正文）
- [x] `scripts/prepare_patched_openclaw_opd.sh`：解析标记 + 转发前清理，**已用真实 GPU 数据验证生效**
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本已统一接入上述所有 workaround
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load` + `PATCHED_OPD_DIR` PYTHONPATH 注入
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）

### 已知限制 / 未解决
- 训练进行到中途崩溃：节点系统内存 OOM，发生在 `update_weights()` 权重同步阶段（7/10 已解决，见上）
- `appendSystemContext` 标记是否会污染 OpenClaw 自己持久化的对话历史，待更长多轮对话验证
- context-summarization 内部调用是否触发 `before_prompt_build`，待验证（决定是否顺带解决 main turn 误标问题）

### 下一步
1. minitest 提交时系统内存申请提高到 192GB（不够再升 256GB），重新提交验证 OOM 是否解决
2. 若解决，8GPU 正式提交同步申请更高系统内存（+ 视情况提高 CPU 核数）
3. minitest 完整跑通后提交 8 GPU 正式 Table 3 训练（`train_with_services.sh` 已就绪）

### 未验证
- [ ] minitest 5 GPU 完整跑通
- [ ] `appendSystemContext` 标记多轮对话下的稳定性
- [ ] 8 GPU 正式 Table 3 训练

---

## 历史状态（2026-07-08，已被 7/9 实测结果和机制切换取代）

### 已就绪
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次 `launch_openclaw_gateway()` 强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`（`contextWindow`/`maxTokens`）+ 静态 `headers.X-Turn-Type=main`——**已实测确认生效**（`[main]` 出现在 training.log 里，训练队列真实累积样本）
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本统一用真实 `openclaw gateway run`
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load` + smoke `PRM_MAX_NEW_TOKENS`/`PATCHED_OPD_DIR` PYTHONPATH 注入
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）

### 已知限制 / 未解决
- `scripts/prepare_patched_openclaw_opd.sh` 的 `X-Session-Id` Runtime 行解析**实测未生效**（一直是 `unknown`），已加调试日志（`[SESSION-ID-DEBUG]`），原因待明天的 smoke 结果确认
- `rl-training-headers` 插件在当前 OpenClaw（2026.6.9）里端到端不生效，已放弃依赖
- smoke（context=8192）下真实 agent 多轮对话可能撞 context overflow；minitest/8GPU（context=32768）预期不受影响
- 提交 job 前务必确认没有残留的手动测试进程（`openclaw gateway run`、mock server 等）占用端口/内存，否则可能触发 cgroup OOM 级联杀掉整个 job

### 下一步
1. 明天查 smoke 的 `[SESSION-ID-DEBUG]` 输出，定位 `X-Session-Id` 解析失败的真实原因并修复
2. 确认训练队列能稳定累积样本（`X-Turn-Type` 这部分已验证，只差 session_id）
3. header workaround 全部验证通过后传播到 minitest/train_with_services.sh（目前只在 smoke 里生效）
4. 提交 8 GPU 正式 Table 3 训练

### 未验证
- [ ] `X-Session-Id` 解析为什么在真实链路里不匹配（已加调试日志）
- [ ] minitest 5 GPU 完整跑通
- [ ] 8 GPU 正式 Table 3 训练

---

## 历史状态（2026-07-07，已被 7/8 实测结果部分取代）

### 已就绪（7/7 时点）
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次 `launch_openclaw_gateway()` 强制设置，不依赖跨环境持久化）
- [x] `models.providers.sglang`：显式声明 `models[]`（`contextWindow`/`maxTokens`）+ 静态 `headers.X-Turn-Type=main`（`launch_openclaw_gateway()` 里生成）
- [x] `scripts/prepare_patched_openclaw_opd.sh`：`X-Session-Id` 从 Runtime 行解析的兜底补丁，官方 `openclaw-opd/` 不动，`PATCHED_OPD_DIR` 接入训练 job `PYTHONPATH`
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本统一用真实 `openclaw gateway run`
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load` + smoke `PRM_MAX_NEW_TOKENS`/`PATCHED_OPD_DIR` PYTHONPATH 注入
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）

> 7/7 这些改动当时都还没实测；7/8 实测确认 `X-Turn-Type` 部分生效，`X-Session-Id` 部分不生效，需要继续排查。

---

## 历史状态（2026-07-06，已被 7/7 header workaround 取代）

### 已就绪（7/6 时点）
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（本次新增，是 18789 端点的真正开关）
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁（`"default"` → `"openclaw/default"`），官方目录不动
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本统一用真实 `openclaw gateway run`（commit `ea19053`，`rl_gateway_proxy.py` 已删除）
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load "${SAVE_CKPT}"` + smoke `PRM_MAX_NEW_TOKENS` 修复
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）

> 7/6 认为"插件+header"是正确方向，7/7 实测证实这套机制在当前 OpenClaw 版本里端到端不生效，已改用 headers 静态配置 + Runtime 行解析。

---

## 历史状态（2026-07-03，已被 7/6 gateway 架构修正取代）

### 已就绪
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist（`/dfs/data/models/Qwen3-4B-Thinking-2507-torch-dist`）
- [x] `openclaw.json` 配置正确（`rl-training-headers` enabled、sglang provider 已验证）
- [x] `scripts/rl_gateway_proxy.py`：替代 `openclaw gateway run`，注入 `X-Session-Id`/`X-Turn-Type: main`（commit `eafd060`）
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh`（已换用 proxy，commit `eafd060`）
- [x] `scripts/train_with_services.sh`（18789 URL 正确，启动顺序 bug 待修）
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh` / smoke / minitest launcher 脚本
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768，2026-07-03 修复）
- [x] conda env `/dfs/data/envs/openclaw-rl` 已有 fastapi / uvicorn / httpx

### 下一步
1. **重启 Simulator**（如 context=32768 版本尚未生效）：`launch_simulator.sh`
2. **重提交 smoke（4 GPU）**：`scripts/smoke_train_with_services.sh`，观察 `training.log` 出现 `combine samples: 16/16` → iter 1 即通过
3. **smoke 通过后**：修复 `train_with_services.sh` 启动顺序（先等 30000 再起 proxy）→ 提交 8 GPU 正式训练

### 未验证
- [ ] smoke 重跑（proxy 替换后，`X-Session-Id`/`X-Turn-Type: main` 注入是否使训练队列正常累积）
- [ ] 8 GPU 正式 Table 3 训练

---

## 历史状态（2026-06-23，已被 6/26 架构更新取代）

### 已就绪（6/23 时点）
- [x] 环境 + 所有 GPU 编译依赖
- [x] Qwen3-4B-Thinking-2507（`/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507`）
- [x] Qwen3-32B（`/dfs/data/models/Qwen/Qwen3-32B`，17 shards）
- [x] OpenClaw 安装完成（Node 22.23、rl-training-headers 插件）
- [x] `check_convergence.py`
- [x] Table 3 完整复现路线 → [`paper_reproduction_scope.md`](paper_reproduction_scope.md)

> 6/24 8 GPU job 因 Simulator 与 GPU 7 共用失败；6/26 已改为**外部 Simulator + 8 GPU 全用于训练**。
