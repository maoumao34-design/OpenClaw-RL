# OpenClaw-RL → MetaClaw 方法迁移记录

[← 工作记录](work_log.md)

本文档记录：把本项目复现的 OpenClaw-RL（论文 arXiv:2603.10165 v2，Personal Agent Track / Hybrid RL 方法）迁移到 MetaClaw（论文 arXiv:2603.17187，`D:\MAO\Paper\MetaClaw`）这一新阶段的方案讨论、决策和实现记录。

---

## 背景

OpenClaw-RL 的 Separate（Student 单角色）Personal Agent Track 复现已大体完成（截至 2026-08-13，仍有一些工程问题待收尾，详见 [`work_log.md`](work_log.md) 08-13 条目及之前）。下一阶段计划：把复现出来的训练方法（Hybrid RL = GRPO + OPD topk-select，含本项目这几周校准出来的一整套奖励信号修正规则）迁移应用到 MetaClaw 这篇论文对应的场景/数据库上。

## 参考资料

| 内容 | 位置 |
|------|------|
| MetaClaw 论文 PDF | `D:\MAO\Paper\MetaClaw\MetaClaw_2603.17187.pdf` |
| 论文全文提取（用于快速检索） | `D:\MAO\Paper\MetaClaw\fulltext.txt` |
| 官方代码 + 数据（已 `git clone`） | `D:\MAO\Claude\MetaClaw-official\`（`benchmark/data/metaclaw-bench` 是论文用的 934 题评测集，随仓库一起下载，无需单独获取） |
| 官方仓库 | https://github.com/aiming-lab/MetaClaw |
| OpenClaw-RL 复现产出（迁移的方法来源） | `D:\MAO\Claude\openclaw-rl\`（脚本）+ `D:\MAO\Claude\OpenClaw-RL-official\`（官方源码） |

---

## MetaClaw 论文核心机制（阅读笔记）

**Meta-model = (θ, S)**：θ 是基座 LLM 权重，S 是自然语言技能库，注入 system prompt。两条不同时间尺度的进化通路：

1. **Skill-driven fast adaptation**（快通路，无梯度）：失败轨迹 → LLM 技能工程师（`skill_evolver.py`）分析 → 合成新技能 → 立即注入 prompt，零权重更新、零服务中断。
2. **Opportunistic policy optimization**（慢通路，梯度）：适应后（技能生效之后）收集的轨迹 → PRM 打分（`prm_scorer.py`）→ 云端 LoRA 微调（`trainer.py`，GRPO + 可选 OPD）→ 权重热更新，由 OMLS 调度到用户空闲窗口触发。

**核心正确性机制——Skill generation versioning**：每条轨迹打上"技能代次"g；技能库从 Sg 演化到 Sg+1 时，驱动这次演化的失败轨迹（support data）被清出 RL buffer；只有 Sg+1 生效之后收集的轨迹（query data）才用于训练 θ——防止"旧技能版本下产生的失败"污染新版本的梯度，这是论文强调的关键设计。

**Opportunistic Meta-Learning Scheduler (OMLS)**：监控睡眠时段 / 系统输入空闲 / Google Calendar 三种信号，任一触发即打开训练窗口，任一恢复即暂停，支持跨碎片窗口断点续训。

**评测**：MetaClaw-Bench（934 题，44 模拟工作日，Part I 30 天 file-check+multi-choice、Part II 14 天含 5 条渐进式隐含偏好规则）+ AutoResearchClaw（23 阶段科研流水线，验证跨领域泛化）。

## 官方代码结构速览

```
MetaClaw-official/
├── metaclaw/                  # 核心 Python 包
│   ├── api_server.py          # 代理服务器，拦截 OpenClaw/CoPaw/IronClaw 等 agent 请求
│   ├── rollout.py / openclaw_env_rollout.py   # 轨迹收集
│   ├── prm_scorer.py          # PRM 判分
│   ├── trainer.py             # RL 训练循环（GRPO + 可选 OPD），云端 Tinker/MinT/Weaver 后端
│   ├── skill_evolver.py / skill_manager.py    # 技能库演化 + 检索
│   ├── scheduler.py / idle_detector.py / calendar_client.py  # OMLS
│   └── memory/                # 长期记忆层（v0.3.2 之后新增，跟本文核心方法关系较松）
├── benchmark/                 # MetaClaw-Bench 数据 + 评测脚本
├── extensions/metaclaw-openclaw/   # OpenClaw 一键接入插件
└── examples/, scripts/run_openclaw_tinker_opd.sh   # OPD 示例
```

## 与 OpenClaw-RL 的架构对照（初步，待讨论细化）

| 维度 | OpenClaw-RL（已复现） | MetaClaw |
|------|------|------|
| 训练后端 | 本地 Megatron + slime（8×GPU） | 云端 Tinker/MinT/Weaver（LoRA，无需本地 GPU） |
| RL 算法 | GRPO + OPD topk-select（Hybrid RL） | GRPO + 可选 OPD（蒸馏教师 log-prob） |
| 判分机制 | PRM/Simulator（Qwen3-32B 或替代模型）+ 本项目校准的多条代码层规则（Rule 1-5、A/B/D 等） | PRM（`prm_scorer.py`，判官模型待查具体配置） |
| 技能库/S | 无对应机制 | 核心组件，本项目复现范围完全没有 |
| 调度 | 无空闲窗口概念，训练和 rollout 同步进行 | OMLS，训练延迟到空闲窗口 |
| 数据有效性保护 | 本项目本轮加的 A/B/D（丢弃 abort/暂停期间生成/重复 user 重试的样本）+ 诊断实验（丢弃 PRM 原判 +1 被强制改判 -1 的样本） | Skill generation versioning（support/query 分离，防止旧技能版本失败污染新梯度）——概念上和本项目的"环境降级样本不该进训练"是同一类问题，但触发条件不同（技能版本切换 vs 基础设施故障/重试） |
| 评测 | Table 3 session 收敛计数（rule-based 判定） | MetaClaw-Bench 逐日准确率 + file-check 完成率 |

## 待讨论的开放问题

- [ ] 迁移的目标是什么：复现 MetaClaw 论文本身的结果，还是把 OpenClaw-RL 这套已校准的 Hybrid RL 训练方法应用到 MetaClaw 的数据/场景上做新的实验？
- [ ] 迁移范围：只迁移 RL/OPD 训练方法本身，还是也要接入技能库这条通路？
- [ ] 复用还是替换：MetaClaw 自带的 `trainer.py`/`prm_scorer.py` 已经是 GRPO+OPD 实现，是在这个基础上迁移改造，还是把 OpenClaw-RL 这边的 Megatron/slime 训练栈接到 MetaClaw 的数据流上？
- [ ] 计算资源：MetaClaw 原生设计是云端 LoRA、无本地 GPU；本项目目前用的是 modelfactory 8×GPU 全参数/本地训练路线，两者资源假设不同，需要先定下迁移后用哪种。

---

## 迁移目标（已确认，2026-08-14）

- **目的**：不是复现 MetaClaw 论文本身的结果，是把 OpenClaw-RL 这边已经校准好的 Hybrid RL 训练方法（Megatron+slime、GRPO+OPD topk-select、本项目这几周调出来的一整套奖励信号修正规则）应用到 MetaClaw-Bench 这个新场景/新数据上做新实验，**用来证明我们复现的方法有普适性**。
- **范围**：只迁移 RL/OPD 训练方法本身，不迁移 MetaClaw 的技能库（skill library）机制。
- **复用还是重建**：需要先摸清楚两边接口细节再决定，不预设结论。

## 查证记录：MetaClaw 官方 RL rollout 架构（2026-08-14）

读了 `metaclaw/openclaw_env_rollout.py`、`metaclaw/prm_scorer.py`、`metaclaw/trainer.py`、`benchmark/data/metaclaw-bench/` 实际数据、`benchmark/scripts/config/rl.yaml`，几个关键发现：

1. **MetaClaw 官方自己的 RL rollout 架构就是照着 OpenClaw-RL 设计的**——`openclaw_env_rollout.py` 文件头文档字符串原话：
   > Architecture (mirrors OpenClaw-RL, passive proxy + external task driver)
   > ...
   > Data format ... This format is consistent with slime's Dataset (used in OpenClaw-RL).

   请求协议（`X-Session-Id`/`X-Turn-Type: main`/`X-Session-Done: true` header）跟本项目早期折腾很久的那套几乎一致。区别：MetaClaw 的 rollout 直接打给自己的 proxy（`metaclaw/api_server.py`），不经过 OpenClaw 网关转发，不会撞上本项目当初发现的 SSRF 拦截问题（那个问题是"外部补丁过的 fetch/dispatcher 被 OpenClaw 自己的安全层绕开"，只在"改 OpenClaw 自己发出去的请求"这个场景下成立，MetaClaw 这里是全新发起请求，不受影响）。

2. **训练后端完全不同**：MetaClaw 的 `trainer.py`（GRPO + 可选 OPD）跑在 **Tinker 云端 LoRA** 上（`benchmark/scripts/config/rl.yaml` 里 `rl.model`/`rl.tinker_api_key` 指向 Tinker），不是本地 Megatron/slime。这是两边唯一真正不兼容的部分。

3. **MetaClaw-Bench 实际数据格式**（`benchmark/data/metaclaw-bench/eval/day01/questions.json`）：`all_tests.json` 列出 30 天（Part I），每天指向 `eval/dayNN/questions.json`，内含 ~10 条 "round"，分两类：
   - `file_check`：靠**自动化 checker 脚本**判定（如 `python scripts/check_iso8601.py day01/standup.json ...`，`expect_exit: 0`）
   - `multi_choice`：`\bbox{X,Y}` 格式作答，跟 `eval.answer` 精确匹配评分

   **两类打分都是完全确定性的（脚本 exit code / 精确匹配），不需要任何 LLM 判官**——跟本项目这学期反复处理的"Simulator 主观判断噪声大、系统性偏严"完全不是一回事，是更干净的实验场。

4. 另有一层 `benchmark/scripts/rl_run.py` + `benchmark/src/`，负责把"按天推进模拟" + proxy + 自动判分串起来，产出论文 Table 1 的数字，是比 `openclaw_env_rollout.py`+`trainer.py` 更上层的编排代码，细节尚未深入。

**初步倾向（待跟 OpenClaw-RL 论文 General Agent / tool-call track 的做法对照后确认）**：复用 MetaClaw-Bench 的任务数据+checker 脚本（新的训练环境），复用本项目已校准的 reward-server 逻辑（协议本来就设计成兼容），**不换训练后端**——继续用本地 Megatron+slime，不接 Tinker，因为目标是验证"我们的方法泛化"，不是"迁到 Tinker 平台"。

## 下一步：先看 OpenClaw-RL 论文自己怎么处理"tool-call 类型场景"

MetaClaw 本质上也是一种 tool-call 场景（agent 靠 `run_command` 工具跟 CLI 环境交互），只是任务数据不同。OpenClaw-RL 论文 Figure 5（General Agent track）里已经有一个专门对应"tool-call agent"的复现方向——`OpenClaw-RL-official/toolcall-rl/`（见项目 CLAUDE.md 对照表）。在决定具体怎么迁移之前，先去看这部分论文原始方法是怎么设计 rollout / 奖励 / 数据格式的，可能比照抄 Personal Agent Track（GSM8K + Student/TA/Teacher）的做法更贴合 MetaClaw 这种场景。

---

## 完整方案（2026-08-14 确认，可展示版本）

### 一句话概述

把 OpenClaw-RL 论文 Hybrid RL（GRPO + OPD topk-select）方法本身——已经在 Personal Agent Track 上复现、校准过一整套奖励信号修正规则——原样迁移到 MetaClaw-Bench 这个全新的任务场景/数据上，验证该方法的**普适性**：不是复现 MetaClaw 论文的结果，是用我们自己的方法去训一个新场景，看能不能训出提升。

### 目标与成功标准

- **目标**：Qwen3-4B policy 模型经过我们的 Hybrid RL 方法在 MetaClaw-Bench 上训练后，相对**它自己不训练的基线**（同一模型、同一 prompt、无任何适应机制）有明显提升。
- **不作为目标**：不要求接近或达到论文报告的 GPT-5.2（41.1%）/Kimi-K2.5 Full（40.6%）绝对分数——那些模型量级远大于 4B，绝对分数主要反映模型底子而非训练方法优劣，不是一个公平的对照。
- **补充验收信号**：逐日准确率曲线是否出现类似论文 Figure 2 的"训练信号攒够后明显提升"的拐点趋势（结构上可对照，不要求幅度一致）——用来证明"是这套在线训练机制在起作用"，不只是"最终分数好看"。

### 核心方法映射

| 环节 | Personal Agent Track（已复现） | MetaClaw 迁移版本 |
|------|------|------|
| 训练后端 | 本地 Megatron + slime | **不变**，继续用本地 Megatron + slime，不接 MetaClaw 自带的 Tinker 云端 LoRA |
| RL 主算法 | GRPO + OPD topk-select（Hybrid RL） | **不变**，同一套训练循环、同一套 loss 组合（`w_opd*L_opd + w_rl*L_rl`） |
| RL 标量奖励来源 | Simulator/PRM 主观判断（`_build_prm_eval_prompt`）+ 本项目校准的多条代码层规则 | **换成 MetaClaw-Bench 自带的确定性 checker**：`file_check` 题看 checker 脚本 exit code，`multi_choice` 题按 `max(0,1-(FP+FN)/n_options)` 精确匹配打分，不需要 LLM 判官 |
| OPD hint 来源 | PRM 投票生成候选 hint（多个模型调用） | **分题型处理，不能直接照抄 `feedback` 静态文字**（见下方"查证记录（二）"第 1 条的详细论证）：`multi_choice` 可以直接用 `feedback.options` 里对应错误选项的说明（本来就是按实际错选项动态挑选的，天然准确）；`file_check` 不能直接用 `feedback.incorrect`（这个字段只描述作者预设的**一种**失败原因，但同一个 checker 脚本经常有好几种不同的失败分支，用错了会把错误的纠正方向喂给蒸馏目标，等于主动训坏模型）——**改用 checker 实际执行产生的 stdout 诊断文本**（`infer_cmd.py::_run_file_check` 已经在 `inline_score["stdout"]` 里捕获了，只是官方自己的 `_build_feedback_text` 目前没接这段，我们接入 OPD hint 时要直接用这个原始 stdout，不要用静态 `feedback.incorrect`） |
| 训练环境/数据流 | Simulator 扮演 Student/TA/Teacher，多轮对话，GSM8K 题库 | Qwen3-4B 扮演 CLI agent（`run_command` 工具），跟真实 OpenClaw 环境交互，MetaClaw-Bench 的 30 天任务流 |
| 在线更新特性 | rollout 持续生成 + 训练持续进行 + 权重热更新，同一长任务内交替 | **保持不变，且比 MetaClaw 官方自己的机制更强**（见"查证记录（二）"第 2 条）——官方评测用的在线更新其实是"跑完 N 天 → 暂停接收请求 → 同步跑一次 train_step → 恢复"这种离散节流模式，不是真正并发；我们现有的连续异步 Megatron/slime 管线本身已经满足更强的"训练与生成并发、权重热更新不中断服务"这条性质，不需要新造机制。真正需要新建的是**按 day01→day30 顺序、concurrency=1 完全串行喂数据的 rollout driver**（模仿官方 harness 在开启 `scene_per_train` 时强制把 `workers` 压到 1 的做法），避免并发乱序导致后面天数的 rollout 用不上前面天数刚训完的权重 |
| 数据有效性保护 | 本项目校准的 A（abort）/B（暂停期间生成）/D（重复 user 重试） | **沿用同一套机制**，检测逻辑不变（对应的降级场景在新环境里应该同样存在：网关/生成中断、训练暂停窗口内生成、Student 侧因超时机械重发指令） |

### 不迁移的部分

- **技能库（skill library）机制**：MetaClaw 特有的 gradient-free 快通路，本项目复现范围不含这条通路，不迁移。
- **OMLS 空闲窗口调度**：MetaClaw 用来避免打断真实用户的机制；我们是做训练实验，不涉及真实部署场景，不需要。
- **Skill generation versioning（support/query 分离）**：机制上类似我们的"环境降级样本不该进训练"，但触发条件是"技能库版本切换"，我们没有技能库这条通路，不适用；我们自己的 A/B/D 系列规则已经覆盖了对应的数据有效性问题。

### 验收方案

1. **主指标**：Qwen3-4B 在 MetaClaw-Bench Part I（30 天，file-check 完成率 + 整体准确率）训练前 vs 训练后的对比。
2. **过程指标**：逐日准确率曲线（3 日滚动平均，对照论文 Figure 2 的画法），看有没有出现"前几天攒信号、之后明显提升"的结构性拐点。
3. **训练健康度指标**：沿用这次会话验证过的一套（A/B/D 触发频率、`+1`/`-1` 分布、batch 组成、是否出现类似"170852 vs 160713"那种概率性成功/失败的现象）。

### 已知风险 / 限制（如实列出，展示时需要一并说明）

- Qwen3-4B 在文件操作/JSON 结构化/shell 脚本这类任务上的底子未知，跟 GSM8K 数学题是完全不同的能力域，训练效果存在不确定性。
- MetaClaw-Bench 是作者自己编写的模拟基准，不是真实用户会话采集，论文原文也提醒"绝对数值可能不直接迁移到生产场景"，我们的结果同样适用这条限制。
- `file_check` 题的 OPD hint 改用 checker stdout 而不是静态 `feedback.incorrect`（见下方查证记录第 1 条）——这条修正逻辑已经想清楚，但**实际接入代码、实测蒸馏效果是否真的比静态文字更好，还没做**。
- 按天顺序、concurrency=1 串行喂数据这个设计，跟现有 Megatron/slime 的 batch 收集逻辑（`_drain_output_queue` 等）配合是否顺畅、吞吐是否够用，还没有实测验证（架构上确认可行，性能上未知）。
- 跨天没有任何文件/session 状态持久化（见下方查证记录第 3 条）——每天的"记忆"完全依赖模型权重本身的更新，如果某天的训练没有真正让权重产生可观测变化，后续天数就学不到前面天数的教训，这是一个比"batch 组成随机性影响训练成功率"（本项目在 separate 阶段反复验证过的现象）更敏感的失败模式，需要在正式跑之前想清楚怎么监控。

### 查证记录（二）：2026-08-14 续，三项此前标记"待验证"的假设逐一核查

用户明确要求"不要默认是对的"，继续查证前一版方案里几处未经验证就写下的假设，结果如下：

1. **OPD hint 不能直接复用 `feedback.incorrect` 静态文字（原方案的这一条是错的，已在上表改正）**。直接读了两个 checker 脚本全文（`check_iso8601.py` 57 行、`check_metadata.py` ~110 行）：单个 checker 经常有好几种互相独立的失败分支（比如 `check_metadata.py` 有"顶层对象缺失/必填字段缺失或为空/ISO8601 格式不对/status 枚举值不对/YAML frontmatter 解析失败"五种不同的 `fail()` 出口），但一道题的 `feedback.incorrect` 只是作者针对**其中一种**预设失败原因写的固定文字。如果模型这次实际触发的是另一种失败原因，拿这段固定文字当 hint 喂给蒸馏目标，方向就是错的，会主动把模型往错的方向训——这正是用户最初质疑的风险，查证后确认是真实存在的。**修正方案**：`infer_cmd.py::_run_file_check`（约 599 行起）已经把 checker 的实际 stdout 捕获进 `inline_score["stdout"]`，只是官方自己现在的 `_build_feedback_text`（685-746 行）没有接这段、只读静态 `feedback.incorrect`——我们接入时直接用这段**已经捕获好的实际 stdout** 做 hint 来源，不用改 checker 脚本本身。`multi_choice` 题没有这个问题：它的 per-option 说明本来就是按实际选错的选项动态挑选的（`missed_option`/`wrong_option`），天然准确，可以直接复用。

2. **"在线更新特性"的对照关系需要修正**。原方案笼统写"保持不变"，实际读了 `metaclaw/api_server.py`（`/v1/admin/train_step`，697-736 行）和 `metaclaw/cli.py`（`train-step` 命令，467-519 行）之后发现：MetaClaw 官方自己评测跑分用的"在线更新"，机制上是**离散的**——`benchmark/src/infer/infer_cmd.py::_run_one_all_tests` 里 `scene_per_train` 参数（"每跑完 N 个 scene 触发一次 `metaclaw train-step`"）生效时会强制把并发 `workers` 压到 1（1305-1319 行），也就是**严格按 test_list 顺序单线程跑完 N 天 → 同步调用一次 `/v1/admin/train_step` 完整跑完一步训练 → 再继续下 N 天**；`train_step` 执行期间，代理服务器通过 `submission_enabled` 门控把新的推理请求挂起排队（`api_server.py` 631-639 行），不是真并发。这跟我们现有的 Megatron/slime 管线（生成和训练常驻并发、`update_weights()` 热更新不中断服务）不是同一种机制，我们的更强——不需要模仿官方这套"暂停-同步训练-恢复"的离散节流，直接沿用现有连续异步管线即可。**需要迁移过来的不是这套离散机制本身，而是它背后确保的那条不变量：数据必须按 day01→day30 严格顺序处理，不能打乱/并发乱序**，所以我们的新 rollout driver 要仿照官方在 `scene_per_train` 生效时"把 workers 压到 1"的做法，把 concurrency 设成 1（现有 `openclaw_env_rollout.py` 的通用连续 rollout 循环是 `random.choices(tasks, k=concurrency)` 随机重复采样，那是给 AutoResearchClaw 之类的自由数据生成场景用的，不适用于按天顺序推进的 Bench 场景，不能照搬）。

3. **跨天没有工作区/状态持久化，此前的判断是错的，已改正**。前一轮只看了 `workspaces/shared/` 目录里同时摆着 `day01/`～`day04/` 等子目录，就推断"可能是持久化的共享工作区"——这是没有查证代码就下的错误结论。实际读了 `benchmark/src/infer/infer_cmd.py::_copy_workspace_for_test`（162-193 行）和 `_prepare_work_copy`（86-126 行）：**每一天开跑前都会从最原始的 `workspace_src`/`openclaw_state_dir` 重新复制一份全新、隔离的工作区和 agent 状态**，`_copy_workspace_for_test` 的文档字符串原话是"Other dayXX directories are excluded so the agent cannot accidentally see content from other test days"，明确禁止模型看到其他天的内容或前一天写过的文件。也就是说：**MetaClaw-Bench 评测协议里，唯一能把"前一天学到的教训"带到"第二天"的载体是模型权重本身（θ）**，不是任何文件/会话状态——这跟论文"meta-model = (θ, S)"的核心主张是一致的（技能库 S 我们不迁移，那么在我们的迁移版本里能带教训跨天走的就只剩 θ）。这个发现直接把上面第二条风险（如果某天训练没有产生可观测的权重变化，后续天数学不到前面的教训）坐实成一个需要认真监控的真实风险，不是理论假设。

### 下一步工程任务（待实现，未开始）

- [ ] 写一个新的 rollout driver（对标 `student_chat.py`），让 Qwen3-4B 通过 `run_command` 工具跑 MetaClaw-Bench 的任务，**concurrency=1 严格按 day01→day30 顺序处理**（不能照搬 `openclaw_env_rollout.py` 现成的 `random.choices` 随机连续采样循环），接入现有 OPD/Combine 服务器
- [ ] 在 OPD/Combine 服务器侧接入 checker 脚本执行 + 结果读取，替换 PRM 判分路径；`file_check` 的 OPD hint 用 checker 实际 stdout（不是静态 `feedback.incorrect`），`multi_choice` 的 OPD hint 可以直接用 `feedback.options` 里对应错误选项的说明
- [ ] 确认 A/B/D 系列规则在新环境下的检测逻辑是否需要调整（比如"重复 user 重试"的判定，MetaClaw 场景下 Student 角色不存在，需要重新定义"重复指令"从哪来）
- [ ] 验证 concurrency=1 严格串行的 rollout driver 跟现有 Megatron/slime 的 batch 收集逻辑（`_drain_output_queue` 等）配合的吞吐和正确性
- [ ] 设计一种手段，用来监控"某天的训练是否真的让权重产生了可观测变化"（呼应查证记录第 3 条的风险），否则没法判断某天没提升是模型能力上限还是训练没生效
- [ ] 跑通训练前基线评测（不训练，直接跑 MetaClaw-Bench 拿一个基线分数）

---

*后续讨论和实现记录追加在本文档下方，或视规模拆分到独立文档（参照 openclaw-rl 项目 `docs/` 的组织方式）。*
