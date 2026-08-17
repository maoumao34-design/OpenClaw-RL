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

### 查证记录（三）：2026-08-14 续，round 内多轮 tool-call 怎么给训练信号（第一批实验文件设计过程中发现的核心问题）

开始写第一批实验文件（rollout driver + 代理侧 checker 奖励接入）时，发现"一道题内模型连续调用好几次 `run_command` 才给最终答案，中间轮次怎么给奖励"这个问题两篇论文都没有现成答案，来回查证了好几轮，记录完整过程和最终结论：

1. **先查两边论文各自的多步 tool-call 处理方式**：`toolcall-rl/generate_with_retool.py`（OpenClaw-RL 论文 Figure 5 的 tool-call 方法）和 `swe-rl/generate_with_swe_remote.py`（同样是 Figure 5 的方法，真实 bash 操作 Docker 容器，跟 MetaClaw 场景最像）**做法一致**：RL 训练进程自己直接控制生成循环（自己调 sglang、自己调 tool 执行后端），整条多步轨迹拼成一个 `Sample(tokens, loss_mask, reward)`，一个 outcome-level reward 共享给整条轨迹。这是 OpenClaw-RL 论文 General Agent 赛道（gui-rl/swe-rl/terminal-rl/toolcall-rl，注意"General Agent"是 OpenClaw-RL 论文自己的术语，MetaClaw 论文没有这个概念）里贯穿始终的架构，不是 toolcall-rl 一家的特例。

2. **一度误判为"没法迁移"，原因是想在这套架构和"真实 openclaw agent CLI 子进程"之间选一个**：OpenClaw-RL 这套"自己控制生成循环"的架构，前提是绕开真实 `openclaw` CLI、自己实现 tool 执行；但读了 MetaClaw-Bench 真实用的 `benchmark/data/metaclaw-bench/openclaw_cfg/openclaw.json` 后确认，任务是按 OpenClaw 内置的 **`"tools": {"profile": "coding"}`**（真实的、结构化的 read/write/edit/bash 等工具画像）设计的，不是笼统的单个 shell 命令——换成自己实现的工具集会有真实的保真度损失（这正是本项目"复现保真度"红线要卡的东西）。一度把这个当成"要不要放弃真实 CLI 子进程换取完整训练信号"的两难。

3. **用户追问"MetaClaw 论文自己怎么拿到中间态反馈来生成 skill"，查出了真正的技术点**：`metaclaw/skill_evolver.py::evolve()` 操作的是 `ConversationSample` 列表，每个 sample = 代理（`api_server.py`）自己视角里的**一次独立 LLM 调用**（`api_server.py` 1340-1436 行：`evolution_every_n_turns`，代理自己按 session 攒够 N 个轮次的 `ConversationSample` 就触发一次技能演化）。这说明：**中间轮次的可见性问题，根子不在"编排层看不看得到"**（`infer_cmd.py` 确实看不到，这个之前判断对了）**，而在"代理自己的评估时机"**——不管上层是真实 `openclaw agent` 子进程还是别的驱动方式，子进程内部每一次单独的 LLM 调用最终都要真的发请求出去，打的就是代理，代理天然能看到每一个中间轮次，不需要编排层告诉它是第几轮。之前"要跨进程追踪轮次编号才能做"的说法是错的，是把"编排层看不见"和"代理看不见"混为一谈了。
   真正的问题是**代理现有的评估时机太快**：`openclaw_opd_api_server.py` 的机制是"下一轮请求一到，立刻用它的内容评估上一轮"（`_fire_opd_task`，同一 session 任何时刻只有一个轮次处于"待评估"状态）——round 还没跑完、checker 还没判，中间轮次早就已经被"下一次工具调用的内容"当 next_state 评估掉、提交或丢弃了，根本等不到 round 结束时我们注入的 checker verdict。

4. **MetaClaw 自己怎么解决"中间轮次没有确定性 ground truth"的**：不解决——中间轮次和最终轮次一视同仁，都扔给 `prm_scorer.py` 的通用主观判官打分（"这个回复对完成任务有没有帮助"，¬感知具体任务/checker），不做任何按 round 聚合。这正是本项目这次迁移想避开的东西（迁移的核心卖点就是用确定性 checker 替掉主观判官噪声）。

5. **最终采纳的方案（方案 B，用户已确认，已实现，见下方"已实现"小节）**：round 最终轮次（拿到最终答案、能跑 checker 的那一轮）用**确定性 checker 结果**评估；round 内其余中间轮次（工具调用探索步骤）不再套用 Personal Agent Track 那套跟 tool-call 完全不匹配的判分标准，而是仿照 **`toolcall-rl::_judge_step_with_prm`/`_build_prm_step_messages`**（444-530 行，通用、跟具体任务无关的"给定 history/action/observation，判断这一步是否有帮助"±1 PRM 步骤判官，OpenClaw-RL 论文自己的现成设计，不是 MetaClaw 的 `prm_scorer.py`，也不是 Personal Agent Track 的 Student/TA/Teacher 判分标准）新写一个 MetaClaw 专用的步骤判官 prompt，独立打分、不聚合进 round 的最终 reward。
   这样：真实 `openclaw agent` CLI 子进程和真实 `"coding"` 工具画像都保留（不放弃保真度）；round 最终结果用确定性 checker（迁移的核心卖点保留）；中间步骤有一个跟任务/工具调用场景匹配的独立信号来源（参照 OpenClaw-RL 自己的方法，不是硬套不匹配的判分标准，也不是完全没有信号）。
   **实现时发现比预想的更简单**：不需要改 `_handle_request` 里 main 轮次的缓冲/触发时机——中间轮次继续用现有的"下一次工具调用一到就立刻评估上一轮"机制，只是把评估内容从 Personal Agent Track 判官换成新的步骤判官；round 的最终轮次天然会在 round 结束后保持"待评估"状态（因为它的下一条真实消息要等到下一个 round 才会出现，或者当天已经是最后一个 round），driver 主动发一条合成的"下一轮"消息（带 `{"metaclaw_verdict": true, ...}` JSON）去触发它评估，复用的是 Personal Agent Track 早就验证过的"下一条消息内容 = 上一轮 next_state"这个反应式机制，不用新开 admin 端点，也不用改缓冲时机。区分"这个 session 是不是 MetaClaw round"靠 `session_id` 前缀模式匹配（`^metaclaw-`），不是靠请求体自定义字段——因为 round 内部中间轮次是真实 `openclaw agent` 子进程内部自己发出的请求，driver 根本没机会往请求体里塞自定义字段（跟当年 SSRF-guard 那次踩过的坑是同一个约束，body 字段这条路走不通，只有 `session_id` 这种已经被验证能可靠透传的东西才行）。

### 已实现（2026-08-14）

- [x] `openclaw-rl/scripts/metaclaw/metaclaw_rollout_driver.py`（新文件）：day01→day30 严格顺序、concurrency=1，直接 `import` 官方 `_copy_workspace_for_test`/`_copy_eval_scripts`/`_prepare_work_copy`/`_start_work_gateway`/`_run_openclaw_agent`/`_compute_inline_score`/`_build_feedback_text` 等函数（不重新实现），真实驱动 `openclaw agent` CLI。每个 round 跑完后把 `{"metaclaw_verdict": true, "eval_score", "hint"}` 通过一条合成的"下一轮"消息发给代理。`file_check` 的 hint 改用 checker 实际 stdout/stderr（不是静态 `feedback.incorrect`，对应查证记录一），`multi_choice` 沿用官方 `_build_feedback_text`（天然准确）。本地对着真实 `MetaClaw-official` 克隆验证过：真实 import 解析、真实读取 day01 的 10 道题、真实跑 checker 脚本（含 `_copy_eval_scripts` 这一步，验证过程中发现漏掉这步会导致 checker 恒定"文件不存在"失败，已修复）。
- [x] `openclaw-rl/scripts/prepare_patched_openclaw_opd.sh`（扩展）：新增 `_METACLAW_SESSION_RE`（`^metaclaw-`，判定 session 是否属于 MetaClaw round mode）、`_build_metaclaw_step_judge_messages`（步骤判官 prompt，仿 toolcall-rl）、`turn_data["metaclaw_round_mode"]` 字段。
- [x] `openclaw-rl/scripts/prepare_patched_openclaw_combine_select.sh`（扩展）：`_opd_evaluate` 新增三路分派——`next_state_text` 能解析出 verdict JSON → 用确定性 `eval_score`/`hint`，跳过所有 LLM 判官调用；`turn_data["metaclaw_round_mode"]` 为真但不是 verdict → 用新步骤判官（复用现成的 `_query_prm_eval_once`/`_prm_eval_majority_vote`，只换了 prompt 来源），RL-only 独立提交，不进 OPD；两者都不是 → 原 Personal Agent Track 逻辑完全不变。两个脚本都已对着本地真实 `OpenClaw-RL-official` 克隆跑过 `py_compile` 验证。

### 训练起点（已确认，2026-08-17）

从**干净的 base Qwen3-4B-Thinking-2507**（`torch_dist` 转换后的 checkpoint，不是 Personal Agent Track 训练完的 checkpoint）开始训练，不接着 separate_student 训完的 checkpoint 继续训。理由：迁移文档的目标是"验证 Hybrid RL 这套训练方法本身能不能在新场景生效"，如果接着旧 checkpoint 训，结果会跟"旧 checkpoint 是否已经过拟合/定型到 GSM8K 风格"这个因素纠缠在一起，说不清楚是方法不行还是起点不好；从 base 开始，结果只反映方法本身。

### 启动脚本必须复用的现有依赖（写启动脚本前先确认，容易漏）

- **`rl-training-headers` 插件必须部署+全局启用**：`_METACLAW_SESSION_RE` 整套按 `session_id` 前缀（`metaclaw-`）分派的机制，依赖代理能拿到真实 `session_id`——而这条链路（`--session-id` CLI 参数 → OpenClaw 内部 `ctx.sessionId` → `appendSystemContext` 写进 system prompt 的 `[RL-TRAINING-META]` 标记 → 代理侧正则解析）本身需要 `rl-training-headers` 插件生效才能工作。这个插件是**系统级全局部署**的（写到 `/usr/lib/node_modules/openclaw/dist/extensions/rl-training-headers/` + `openclaw plugins enable rl-training-headers`，不挂在某个具体 `openclaw.json` 里），`scripts/train_with_services.sh`/`scripts/train_separate_student.sh` 里都有这一步。MetaClaw 迁移的启动脚本必须原样复制这一步（`bash scripts/prepare_patched_rl_training_headers.sh ...` + 部署到系统目录 + `openclaw plugins enable`），否则 `session_id` 传不到代理，所有 `metaclaw-` 前缀判定会静默失效，全部回退到 Personal Agent Track 原逻辑。
- 同理，`sglang execution-bias`/`embedded-agent overflow-recovery` 这两个系统级 OpenClaw 行为补丁（`scripts/prepare_patched_sglang_execution_bias.sh`/`scripts/prepare_patched_embedded_agent_overflow_recovery.sh`）修的是 OpenClaw 本身的通用 bug，跟具体训练场景无关，MetaClaw 启动脚本也应该原样部署一遍（如果同一台机器上已经因为跑过 Personal Agent Track 训练而全局生效，重复部署是幂等的，无副作用）。
- PRM/judge 模型端点（`self._prm_url`）：MetaClaw 的步骤判官（`_build_metaclaw_step_judge_messages`）复用现成的 `_query_prm_eval_once`，走的是跟 Personal Agent Track 完全相同的 judge 模型服务，需要同样启动。

### 已实现（续，2026-08-17）

- [x] `openclaw-rl/scripts/metaclaw/run_metaclaw_migration_modelfactory.sh`（新建）：对标 `run_openclaw_topk_select_modelfactory.sh`/`train_separate_student.sh` 的启动编排——生成三个补丁代理目录 → 起训练后端（`--load` 指向 base Qwen3-4B `torch_dist`，`SAVE_CKPT` 用独立于 Personal Agent Track 的新路径）→ 等 RL proxy 30000 就绪 → 部署六项 OpenClaw 系统级补丁（`rl-training-headers` 等，"启动脚本必须复用的现有依赖"一节列的三点全部覆盖）→ 用 `BENCHMARK_BASE_URL`/`BENCHMARK_API_KEY`/`BENCHMARK_MODEL` 环境变量把 `openclaw_cfg/openclaw.json` 的 model provider 指向这次训练起的代理 → 启动 `metaclaw_rollout_driver.py` → driver 跑完（day01→day30 全部处理完）后主动停止训练。跟 `train_separate_student.sh` 的一处关键简化：**MetaClaw 不需要外部 Simulator**——`wait_for_external_simulator`/`SIMULATOR_ENV` 整段没有对应物，因为 MetaClaw-Bench 的"提问方"是静态题目文本+确定性 checker，不需要另一个 LLM 扮演角色；步骤判官用的 PRM SGLang 引擎已经是 topk-select 8GPU 拓扑自带的一部分，不需要额外服务。`bash -n` 语法检查通过（本地无法做超出语法检查的验证，六项系统级补丁部署+真实 `BENCHMARK_BASE_URL` 路径拼接是否正确仍需 modelfactory 真实环境验证）。

- [x] **`metaclaw_rollout_driver.py` 补上"基础设施故障不能被当训练信号"这道防护（2026-08-17）**。审查发现：`_run_round` 里 `openclaw agent` 真实 CLI 子进程失败（`rc != 0`，网关故障/子进程崩溃/超时）时，原实现照样往下跑 checker、照样提交这个 round 的 verdict——一次纯环境性故障会被当成"模型任务失败"提交 `eval_score=-1` 进训练，是真实漏洞，不是理论风险。参照 OpenClaw-RL 自己的 General Agent 赛道怎么处理同类问题：`toolcall-rl`/`swe-rl` 都是在生成/执行基础设施失败时显式设置 `sample.status = Sample.Status.ABORTED` 并**在 Sample 到达 reward_func/正常提交路径之前提前返回**（`generate_with_retool.py` 686-687 行；`generate_with_swe_trajectory.py` 三处）——这是 slime 框架级别的标准做法，不是本项目发明的新概念。
  把这个原则翻译到我们的架构：`agent_succeeded=False`（`rc != 0`）时，driver **不发送这个 round 的 verdict 消息**——round 的最终轮次会保持"待评估"直到 `session_done` 才被代理侧现成的 `force_drop_without_next_state` 机制丢弃，效果上等价于 `Sample.Status.ABORTED` 的"提前返回、不进训练"，不需要新写任何服务器侧代码。如果失败发生在当天最后一个 round，新增 `_send_session_close_only`——用 `X-Turn-Type: side`（不是 `main`）+ `X-Session-Done: true` 触发 session 收尾清理，跳过整个"main 轮次"分支，不携带任何 verdict 内容。
  **对照 MetaClaw 自己的 `infer_cmd.py`**：它没有这层保护——`_run_question` 重试几次仍失败后，直接把空字符串答案传给 `_compute_inline_score`，正常算出一个 "failed" 结果，不区分"环境故障"和"模型真的没做对"。这是可以接受的，因为 `infer_cmd.py` 只用于离线评测，误判只会给评测分数增加一点噪声，不会把错误信号焼进权重——**但我们直接复用的 `_compute_inline_score`/`_run_file_check` 是这段"只在低风险评测场景验证过"的代码，塞进了高风险的训练奖励场景，之前没有意识到需要单独补一道基础设施故障过滤**，这次算是把这道防护补齐。
  仍未覆盖的残留风险（如实记录，不夸大这次修复的完整性）：如果子进程在 round 跑到一半才崩溃（已经有几个中间 tool-call 轮次被步骤判官独立评估、提交过了），这些已提交的中间样本目前没有办法追溯撤回——跟 toolcall-rl"一个 episode 一个 Sample、崩溃时整个 Sample 还没构造完成"的情况不同，我们的架构是逐轮次提交，撞上这个时间窗口的中间样本已经进了训练队列，这是"每轮次一个 Sample"架构本身的固有限制，不是这次没修干净。`py_compile` + 真实 import 验证通过，真实 `rc != 0` 场景（网关故障等）本地无法复现，仍需 modelfactory 真实环境验证。

- **发现但未修复：跨 round 污染 bug（2026-08-17，需要真实数据才能判断值不值得精确修）**。`agent_succeeded=False` 时，driver 目前只处理了"这是当天最后一个 round"（发 `_send_session_close_only`）；如果失败发生在**不是最后一个 round 的中途**，且这个 round 崩溃前已经真实产生过至少一个轮次，代码现在什么都不做——那个轮次会一直"待评估"挂在代理里，下一个 round 开始后，它的第一个真实轮次一到，会被反应式机制误当成上一个失败 round 那个挂起轮次的 `next_state` 去评估，等于用完全不相关的下一个 round 内容判断上一个失败 round 的最后一步。不好简单补：直接照搬 `_send_session_close_only` 会带上 `X-Session-Done: true` 的副作用（`_turn_counts.pop(session_id, None)`），语义上是"这一整天结束了"，但这天后面还有别的 round 要继续跑，现在没有干净的区分手段。
  **要不要精确修，取决于这个失败模式在真实场景下有多常见**——只有 round "跑到一半才失败"（已经成功过几个工具调用之后才崩）才会触发，一上来就失败（比如网关连不上）不会留下挂起轮次。这个频率纯粹是实证问题，本地无法判断，需要看真实训练日志里 `agent_succeeded=False` 的那些 round 具体是在第几个工具调用之后失败的。**留到真实训练跑起来、观察这类日志之后再决定要不要修、怎么修**。

### 三方对照：迁移方法各环节分别用的哪种方式，跟 MetaClaw 自己、OpenClaw-RL tool-call 的差异（2026-08-17）

前面几轮查证分散在"查证记录（三）"里，这里按"环节"重新整理成一张对照表，一次看全——**没有一个环节是三方完全一致的，每个环节都是单独核实、单独决定的，不是整体套用某一方的方案**：

| 环节 | 我们的迁移方法 | MetaClaw 自己 | OpenClaw-RL tool-call（`toolcall-rl`/`swe-rl`） | 跟谁一致 / 跟谁不同 |
|------|------|------|------|------|
| **Agent 怎么跑多步 tool-call** | 真实 `openclaw agent` CLI 子进程，真实 `"coding"` 工具画像 | 训练用的 `openclaw_env_rollout.py`：自己直接控制生成循环，自己执行单个 `run_command`；评测用的 `infer_cmd.py`：真实 CLI 子进程 | 自己直接控制生成循环（直连 sglang），自己执行工具（本地沙箱/远程 exec server） | 跟 MetaClaw **评测**模式一致（保真度优先），跟 MetaClaw **训练**模式和 OpenClaw-RL tool-call 都不同（它们都放弃真实 CLI 换取生成循环的完全可控） |
| **Round 最终结果的奖励来源** | 确定性 checker（`file_check` exit code / `multi_choice` 精确匹配） | 从不用 checker 做训练奖励，只有 `prm_scorer.py` 通用主观判官 | 确定性 outcome check（`\boxed{}` 答案匹配 / test suite 通过） | 跟 OpenClaw-RL tool-call 一致（都是确定性 outcome 信号），跟 MetaClaw 自己完全不同（MetaClaw 的确定性 checker 只用于离线算 Table 1，从没进过训练奖励） |
| **Round 内中间 tool-call 轮次的信号** | 新写的任务无关步骤判官（仿 `_judge_step_with_prm`/`_build_prm_step_messages` 的 prompt 结构改写），独立打分，RL-only，不聚合进 round reward | 每个中间轮次也是独立 `ConversationSample`，用同一个 `prm_scorer.py` 通用判官打分，跟最终轮次一视同仁，没有"中间/最终"这个区分概念 | 不单独打分——整条轨迹一个 Sample，中间轮次跟最终轮次通过 `loss_mask` 共享同一个 outcome reward | **三方都不一样，是这次迁移自己合成的设计**——借了 toolcall-rl 判官 prompt 的写法，但没有借它"共享一个 reward"的机制（那个机制要求放弃真实 CLI，见上一行）；也没有照抄 MetaClaw"一视同仁"的处理方式（那样会引入跟任务不匹配的判分标准） |
| **Sample 提交粒度** | 沿用 combine_select 现有的"每轮次一个 Sample"（dynamic-history paradigm） | 每次 LLM 调用一个 `ConversationSample` | 整条轨迹一个 Sample（`tokens` 拼接 + `loss_mask` 区分模型生成段/观察段） | 跟 MetaClaw 自己一致（逐轮提交），跟 OpenClaw-RL tool-call 不同（整条轨迹一个 Sample，我们的架构做不到这个，见 Agent 执行方式那一行） |
| **OPD hint（蒸馏教师信号）** | round 最终轮：`file_check` 用 checker 实际 stdout，`multi_choice` 用 `feedback.options` 动态按错选项选；中间轮次没有 hint（RL-only） | 完全没有这个机制——MetaClaw 的训练通路是纯 GRPO + `prm_scorer` 判分，没有 Hybrid RL 那种 GRPO+OPD 组合损失 | 也没有——`toolcall-rl` 是纯 outcome reward（+可选 PRM 逐步分数直接相加），不是 loss 层面的 GRPO+OPD 组合 | **OPD 蒸馏是 Personal Agent Track Hybrid RL 独有机制，MetaClaw 和 OpenClaw-RL tool-call 都没有**——这次迁移的核心目的就是测试这个机制本身能不能搬到新场景，所以特意保留，只换了 hint 的来源（主观判官 → 确定性文本） |
| **训练循环的顺序约束** | day01→day30 concurrency=1 严格顺序；本地 Megatron+slime 连续异步管线（生成/训练同时进行，权重热更新） | 评测/`scene_per_train` 训练：`workers=1` 严格顺序，但触发机制是离散的"跑完 N 天→暂停接收请求→同步一步 `train_step`→恢复"，不是真并发 | 没有"天/日序"这个概念，标准 slime 异步 rollout，多条轨迹并发生成，顺序无关紧要 | 顺序约束这一点跟 MetaClaw 一致（都要保证"权重更新影响后续天数"这条在线学习假设），但触发机制比 MetaClaw 自己更强（连续异步 vs 离散暂停-恢复，不需要模仿它的暂停机制）；跟 OpenClaw-RL tool-call 完全不同（tool-call 的训练样本互相独立同分布，不存在"日序"或"跨天教训传递"的设计） |

**读这张表的方式**：每一行代表一次独立的技术判断（保真度 vs 可控性、复用 vs 新造、跟哪边一致跟哪边不一致），不是"选定一个阵营整体照搬"。这也是"查证记录（三）"里反复强调的——两篇论文对"round 内多步 tool-call 怎么给训练信号"这个具体问题都没有直接答案，逼着这次迁移在每个环节上分别做取舍，而不是找一个现成模板整体套用。

### 查证记录（四）：2026-08-17，training-signal-safety 审查逐项对照两篇论文/官方代码有没有处理过

上一轮（"已实现（续，2026-08-17）"）审查自己的实现有没有让误判/故障混进训练信号，一共发现四类问题，这里逐项查两篇论文/官方代码分别有没有先例。

1. **基础设施故障不能被当训练信号（已在上方"已实现（续）"修复）**：参照对象是 `toolcall-rl`/`swe-rl` 的 `Sample.Status.ABORTED` 早退机制——两条 GA 赛道都在生成/执行基础设施失败时提前 return，Sample 从不到达 `reward_func`，这是 slime 框架级别的标准做法。我们的修复（`agent_succeeded=False` 时不发 verdict，交给代理侧现成的 `force_drop_without_next_state` 丢弃）是这个原则在跨进程 HTTP 架构下的等价翻译，细节见上方条目。

2. **步骤判官 prompt 质量校验**：全仓库搜 `judge_accuracy`/`prm_accuracy`/`calibrat`/`ground_truth.*judge` 等关键词，OpenClaw-RL 和 MetaClaw 两边命中的全部是假阳性（loss 权重注释、内存重要性打分），**都没有内置的判官准确率校验/校准机制**。Personal Agent Track 判官系统性偏严是靠本项目自己反复训练+人工核对样本发现的，不是论文/官方代码自带的验证工序——我们的步骤判官"零验证直接上线"跟两边默认状态其实一致，只是我们还没走完这道人工核实的路。**这条待用户进一步指示后再处理，暂不下结论**。

3. **A/B/D 系列环境降级样本过滤，四条 GA 赛道有没有对应物**：搜 `duplicate`/`retry.*drop`/`dedup`，四条赛道（`gui-rl`/`swe-rl`/`terminal-rl`/`toolcall-rl`）命中的两处全是假阳性（PRM prompt 文本去重、GUI 任务表格查重指标）。**四条赛道完全没有"重复 user 重试"检测，原因是架构上不存在这个失败模式**——它们不是"Simulator/Student 走 HTTP 重试循环"的架构，是自己在 Python 里直接控制生成循环，遇到基础设施失败走的是第 1 条的 `Sample.Status.ABORTED`，不需要检测重复指令。
   **这个发现直接给"D 规则在 MetaClaw 场景下要不要调整"这条待办一个明确结论**：`metaclaw_rollout_driver.py` 现在也没有重试循环（`_run_round` 只调用一次 `_run_openclaw_agent`，不像 `student_chat.py` 那样 408/503 会机械重发），架构上跟四条 GA 赛道是同一类——**D 规则（`is_duplicate_user_retry`）在 MetaClaw 场景下是空转（不会触发，但也没有害处），不是缺口**，跟四条 GA 赛道不需要它的原因完全一样：没有对应的重试架构去触发它。
   顺带确认 A（`is_aborted`）/B（`generated_while_paused`）**不需要任何调整**——这两个标记在 `_handle_request` 里对所有生成通用计算（基于 `finish_reason`/`submission_enabled`，不区分 Personal Agent Track 还是 MetaClaw），配套的丢弃拦截点也是通用的，MetaClaw 的步骤判官轮次和最终 verdict 轮次自动受到同样保护，不需要专门适配。
   **结论：A/B/D 系列规则在 MetaClaw 场景下不需要调整**，原有的"下一步工程任务"里这一条可以关闭。

4. **提交失败默认丢弃这条设计是否干净——没有直接可比对象**：查 `toolcall-rl` 发现它的架构里根本没有"跨进程 HTTP 提交 Sample"这一步——`generate()` 直接把 Sample 对象 return 给 slime 框架自己的代码消费，是同进程函数返回，没有网络提交、也就没有"提交失败"这个故障面。这是我们（和 Personal Agent Track）"代理跟生成进程分离、靠 HTTP 传数据"架构特有的问题，两篇论文其他部分都没有直接可比对象——不是漏查，是确实不存在对应场景。`_send_verdict_turn` 提交失败时静默丢弃（不重试）这条设计本身是安全的（回合最终轮次会在 `session_done` 时因缺少 `next_state` 被丢弃，是数据丢失不是数据污染），只是没有先例可以对照，纯靠我们自己的架构分析确认。

**追加核实（2026-08-17）——上面 2/3 项分别跟 MetaClaw 自己的默认行为对照**：
- 第 2 项（agent 调用重试）：`infer_cmd.py` 765/1380 行、`run_cmd.py` 82 行，`retry: int = 0`——**MetaClaw 官方默认也不重试**，是个可选参数，需要调用方显式传更大的值。我们没接这个参数，效果上跟它的默认行为一致，只是它多留了一个开关。
- 第 3 项（HTTP 提交重试）：搜了 `api_server.py` 里 `_query_teacher_logprobs`（自己那条走 HTTP 的教师 logprob 查询）相关的 retry/except 逻辑，零命中——**MetaClaw 自己的 HTTP 提交也没有重试**，跟我们是同一个宽松程度。
- 第 4 项（断点续跑）：`_run_question`/`_run_group` 里 `result_path.exists()`/`existing_inline_score` 这类跳过已完成项的机制，MetaClaw **确实有、我们没有**，是真实缺口。
- 第 1 项（跨 round 污染）：MetaClaw 自己的架构完全不会撞上这个问题——它的训练奖励从不按 round 聚合，每个 LLM 调用独立当场打分，没有"round 边界"这个概念，也就不需要处理"round 之间的残留轮次"。这个 bug 是我们自己发明"round 级确定性 verdict 注入"机制才带出来的，两篇论文都没有先例可循。

**已实现（2026-08-17，对应第 2/3 项；第 4 项做了又撤，原因见下）**：
- [x] **第 2/3 项，可选重试，默认关闭**：新增 `METACLAW_AGENT_RETRY`（`_run_round` 内 `openclaw agent` CLI 调用失败时的重试次数，逻辑/日志风格照抄 `infer_cmd.py::_run_question` 的 `for attempt in range(retry + 1)`）、`METACLAW_VERDICT_RETRY`（`_send_verdict_turn`/`_send_session_close_only` 提交失败时的重试次数，新增 `_post_with_retry` 统一处理）两个环境变量，**默认都是 0（不重试），跟 MetaClaw 官方默认行为一致**，要不要开、开到多大留给真实训练观察效果后再决定。
- **第 4 项（断点续跑）：先按天粒度实现过一版，随后主动撤回，不做断点续跑**。撤回过程：
  1. 最初按天粒度实现（`METACLAW_PROGRESS_DIR` + `<test_id>.done` 标记），理由是 MetaClaw 官方的 round 粒度 `existing_inline_score` 跳过检查依赖"同一个 test 全程共用一份还在磁盘上的 workspace"，而我们的 `workspace_copy` 每次 `run_day()` 都全新重建，照搬 round 粒度会导致跳过的 round 的真实文件效果在新 workspace 里不存在。
  2. 追问"MetaClaw 自己是不是也这样"后，直接读了 `_run_one_test`（999-1042 行）——它**每次被调用都无条件重新调用 `_prepare_work_copy`/`_copy_workspace_for_test`**，没有任何"test 已完成就跳过建 workspace"的检查，`_run_one_all_tests` 的串行/并发两条路径调用 `_run_one_test` 时也没有异常处理或跳过逻辑。这说明 **MetaClaw 自己的 round 级"断点续传"，如果真的用来在进程重启、部分 round 已完成的场景下恢复，会撞上跟我们一模一样的 workspace 不一致问题**——不是它解决了我们没解决的问题，是这个功能在它自己的代码里可能也没有被这样验证过。
  3. 但这个不一致对 MetaClaw 自己的**评测**用途影响很轻（最多让崩溃点之后紧邻的一个 round 评分多一点噪声，摊到整体平均准确率里可忽略），对我们的**训练奖励**用途则完全不是一个量级（一条基于错位上下文的样本会被当真实梯度信号写进权重）——这正是这次会话反复出现的同一个模式：MetaClaw 很多代码是按"评测容错"标准写的，直接复用到训练场景需要重新按更高标准检查。
  4. 最终结论（用户拍板）：checkpoint 存盘节奏（按训练步数）本来就跟 rollout driver 的天数进度不同步，"标记这天完成"不代表这天的样本已经真的进了某个存盘的 checkpoint，进程崩溃重启后按天粒度续跑一样有可能静默丢失一段训练贡献且无法发现。而一遍 30 天训练总耗时有限（数量级是小时而不是天），**崩溃后直接从 day01 用干净的 base checkpoint 完整重跑，比维护任何一种不完全可靠的断点续跑机制更简单也更安全**——没有部分完成状态，也就没有任何"状态对不对得上"的风险。已把当时加的 `METACLAW_PROGRESS_DIR`/`_day_marker_path`/`_day_already_done`/`_mark_day_done` 全部移除。
- **checkpoint 存盘频率单独调整（2026-08-17，另一个决定）**：确认 checkpoint 本身就是完整、独立可用的模型快照，跟最终训练完的模型能力一样可以直接拿去跑 MetaClaw-Bench 评测，只是训练程度不同——于是把"记录训练中途进度供观察"和"断点续跑"这两件事分开处理，后者不做（见上），前者保留：`scripts/run_openclaw_topk_select_modelfactory.sh` 新增 `METACLAW_MIGRATION_PROFILE=1` 分支，把官方默认 `--save-interval 100` 改成 `--save-interval 10`。粗略推算：一遍 30 天约 300 个 round，每个 round 约 1 条最终轮次样本 + 数条中间轮次步骤判官样本（保守估计均值约 3 条/round）≈ 900 条样本，`--rollout-batch-size 16`（未改动）下约 56 训练步（换算比例参照 `MINITEST_PROFILE` 注释"num-rollout 300 → 约 18 训练步"，300/18≈16.7，与 batch-size=16 基本吻合），目标一遍存下约 5 个 checkpoint，56/5≈11.2，向下取整到 10（宁可多存不要少于 5 个）——**这是一个粗估计，实际每个 round 平均产生多少样本要等真实训练跑一次才知道，届时按真实数据重新校准这个数字**。`run_metaclaw_migration_modelfactory.sh` 调用训练后端时传 `METACLAW_MIGRATION_PROFILE="1"`。
- 落地文件：`scripts/metaclaw/metaclaw_rollout_driver.py`（`_post_with_retry`，`_run_round`/`_send_verdict_turn`/`_send_session_close_only` 加 `retry` 参数，断点续跑相关代码已移除干净）、`scripts/run_openclaw_topk_select_modelfactory.sh`（新增 `METACLAW_MIGRATION_PROFILE` 分支）、`scripts/metaclaw/run_metaclaw_migration_modelfactory.sh`（重试环境变量、`METACLAW_MIGRATION_PROFILE="1"`，移除 `METACLAW_PROGRESS_DIR`）。`py_compile` + 真实 import + sed 补丁对真实官方源码验证通过。

### 待验证：中间轮次缺确定性锚点这个风险差异，要不要照抄 toolcall-rl 补上（2026-08-17，暂缓，等真实环境验证后再定）

**问题**：`toolcall-rl` 的 `reward_func`（907-919 行）里 `final_score = base_score + prm_step_coef * prm_step_mean`——`base_score` 是确定性的 `\boxed{}` 对错判断，**永远**跟判官打的 `prm_step_mean` 加在一起，构成同一个 Sample 的最终 reward。哪怕判官这次判错了，`base_score` 这个确定性锚点还在，误差不会让整条训练信号完全脱离真实情况。
我们现在的设计里，round 内中间轮次是**独立提交的 RL-only 样本**，reward 就是步骤判官的分数本身，没有任何确定性分量兜底——判官用的机制（多票投票、同尺寸判官模型）跟 toolcall-rl 完全一致，但如果判官判错了，这次错误是未经稀释地直接进训练，风险等级比 toolcall-rl 高。

**能不能补上确定性锚点**：技术上可以——把中间轮次的评估从"下一条消息一到就立刻触发"改成"攒住整个 round，等 driver 发来 checker 的确定性结果后，一次性把这个 round 攒住的所有轮次都取出来，每个都用 `最终分数 = 步骤判官分数 + checker确定性结果` 提交"，照抄 toolcall-rl 的组合公式。副作用是会顺带修掉"round 跑到一半崩溃、已提交的中间样本没法追溯撤回"这个之前记录过的残留风险（因为改完之后中间轮次不会提前提交，round 崩溃时是整批一起被丢弃）。

**为什么暂缓，没有直接实现**：这个改动不碰任何 MetaClaw 官方代码（改动全部在我们自己的 `openclaw_opd_api_server.py`/`openclaw_combine_select_api_server.py` 里），但会让代理第一次需要依赖一个**MetaClaw/OpenClaw 真实运行时行为的假设**——代理本身完全没有"round"这个概念，只能靠"driver 严格串行调用（concurrency=1，一个 round 的 `_run_openclaw_agent` 完全跑完才发下一个）"这个前提，把"上一次 verdict 之后新攒下来的所有轮次"当成"这个 round 的全部轮次"。这个推断依赖两个前提：
1. **每个 round 是单独一次 `_run_openclaw_agent` 子进程调用**，不是一个长进程内部处理多个 round——已核实 `_run_group`（864/922 行）代码层面确实如此，`_prepare_session` 在 round 循环开始前只调一次，同一个 `session_id` 靠 OpenClaw 自己持久化的 session transcript 文件串起跨子进程的对话连续性。
2. **两个 round 之间不会有"杂音请求"落进代理**——比如 OpenClaw 自己的 context 压缩/内部重试机制，会不会在两次 `_run_openclaw_agent` 调用的间隙里偷偷再打一次请求过来。这一点**只核实了代码层面，没有真实跑过验证**——Personal Agent Track 训练过程中已知 OpenClaw 确实存在"context summarization"这类内部兜底调用（历史上多次造成 `session_id` 误标为 `unknown` 之类的问题），不能默认 MetaClaw 场景下不会有同类行为。

现在的"立刻触发"方案完全不依赖这两条假设——每个轮次独立处理，代理不需要知道 round 是什么，对 MetaClaw/OpenClaw 内部任何没预料到的行为都更健壮，这大概率是当初选这个方向时隐约在意但没有明说的顾虑，不只是"改动量小"。

**需要真实验证的具体问题**：modelfactory 上真实跑起来后，观察同一个 session 里两个连续 round（对应两次 `_run_openclaw_agent` 调用）之间，代理侧的 turn 计数/请求日志有没有出现不属于任一 round 真实工作内容的额外请求（比如 context 压缩触发的调用）。如果确认没有杂音、"round 边界=子进程边界"这个假设站得住，再回来实现"攒住+组合确定性锚点"这个改动；如果发现有杂音，说明"立刻触发"这个更简单、更健壮的方案就是对的选择，不用再改。

### 下一步工程任务（待实现，未开始）

- [ ] **verify** 上面"待验证"一节的具体问题：真实环境下两个连续 round 之间会不会有杂音请求落进代理——决定要不要给中间轮次补确定性锚点
- [ ] modelfactory 侧真实联调：真实 `openclaw agent` CLI 子进程 + 真实代理端口打通、合成 verdict 请求能否正确触发、步骤判官 prompt 对 `run_command` 调用的判断质量（新写的 prompt，没有历史数据验证过，两篇论文都没有内置判官校准机制可以参考，见查证记录四第 2 条，处理方式待定）、`BENCHMARK_BASE_URL="http://127.0.0.1:30000/v1"` 这个假设的 URL 形状（没有 trailing `/chat/completions`）在真实 OpenClaw `openai-completions` provider 客户端下是否正确
- [ ] 验证 concurrency=1 严格串行的 rollout driver 跟现有 Megatron/slime 的 batch 收集逻辑（`_drain_output_queue` 等）配合的吞吐和正确性
- [ ] 设计一种手段，用来监控"某天的训练是否真的让权重产生了可观测变化"（呼应查证记录第 3 条的风险），否则没法判断某天没提升是模型能力上限还是训练没生效
- [ ] 跑通训练前基线评测（不训练，直接跑 MetaClaw-Bench 拿一个基线分数）

---

*后续讨论和实现记录追加在本文档下方，或视规模拆分到独立文档（参照 openclaw-rl 项目 `docs/` 的组织方式）。*
