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

1. **主指标**：Acc./Compl.，跟论文 Table 1 方法学对齐——**训练本身这一趟运行的实时聚合分数**，不是训练前/训练后两次独立评测的对比（原因和设计见下方"训练/评测数据重叠"一节）。跟训练前基线（见下方"基线结果"一节，Acc.=8.1%/Compl.=0.0%）对比，判断训练有没有让这两个数字比基线好。
2. **过程指标**：逐日准确率曲线（3 日滚动平均，对照论文 Figure 2 的画法），看有没有出现"前几天攒信号、之后明显提升"的结构性拐点。
3. **训练健康度指标**：沿用这次会话验证过的一套（A/B/D 触发频率、`+1`/`-1` 分布、batch 组成、是否出现类似"170852 vs 160713"那种概率性成功/失败的现象）。

### 训练/评测数据重叠：论文自己怎么做的，我们的设计改动（2026-08-18）

**发现的问题**：训练用的 30 天数据（`all_tests.json`）跟"如何给任意一个 checkpoint 打分"一节里打分用的是同一份数据——训练前后分别独立评测这套方法，如果训练和评测数据完全重叠，训练后分数的提升说不清是真的学会了泛化能力还是记住了这些具体题目。

**查证论文自己怎么处理**：直接读了 MetaClaw 官方 `benchmark/scripts/rl_run.py`（Table 1 里 "MetaClaw (Full)" 这一档——skills + RL——的真实产出脚本），结论：**论文自己是边训练边算分，同一趟跑完，没有单独的 held-out 测试集，也完全没讨论这个 leakage 问题**（论文原文只提了"这是作者编写的模拟基准，绝对数值可能不直接迁移到生产场景"，这是在说仿真真实性，不是在说训练测试集重叠）。

关键代码（`rl_run.py` 259-268 行）：
```python
run_cmd = [
    cfg.BENCH_BIN, "run",
    "-i", cfg.BENCH_INPUT,      # all_tests_metaclaw.json，跟评测同一份 30 天数据
    "-o", cfg.BENCH_OUTPUT,
    "-w", "1", "-n", str(cfg.BENCH_COUNT),
    "--scene-per-train", str(cfg.SCENE_PER_TRAIN),   # =5，每完成 5 天触发一次 RL 训练
]
```
`metaclaw-bench run` 本身就是"推理→打分→出报告"一条龙命令；`rl_run.py` 只是在外面套了一层代理，代理每完成 5 个 scene 触发一次训练权重更新——**代理返回给 agent 的每个真实回复，就是当时那一刻的权重生成的，`metaclaw-bench run` 自己的推理→打分流程照常跑，Table 1 的 Full 这行数字就是这一趟 30 天全部打分的聚合**，前几天用没怎么训练过的权重、后面天数用训练过几轮的权重，混在一起算平均分。论文原文（Section 4.1.1）另有一句可查证的话："MetaClaw (Full): the full pipeline combining skill-driven fast adaptation with opportunistic policy optimization via RL **(5-day training run)**"——RL 训练只在 30 天里的 5 天内触发，但具体是哪 5 天、跟评测窗口的精确关系，论文原文没有交代清楚。

**设计改动（这次落地）**：把我们的迁移训练也改成跟论文一样——`metaclaw_rollout_driver.py` 现在在训练过程中直接用官方 `scoring_cmd.py` 的打分函数（`_score_multi_choice`/`_score_file_check`，跟 `metaclaw-bench scoring` 用的是同一段代码，不是简化版，multi_choice 有真实的部分正确分）给每一轮实时算分，全程跑完后聚合成 Acc./Compl. 输出——这是跟论文 Table 1 方法学真正对得上的数字，也是这次迁移**唯一**的 Acc./Compl. 产出方式（曾经想过额外保留一个"独立 SGLang + `metaclaw-bench run`"的补充打分法，用户指出那个方法一样用同一份 30 天数据、并不比这个更干净，已撤回，见上方"已废弃"一节）。

**连带的架构改动：checkpoint 的角色变了**。训练和打分现在共用同一趟运行，"崩溃后从 day01 完整重跑"会导致已经算过分的天用不同（更新过的）权重重新生成一次答案，污染最终聚合分数——不再只是浪费算力，是会算出错误结果。所以撤回了 08-17"不做断点续跑"的决定，重新加回按天粒度的进度持久化，**并且用户明确要求不能做成自动续跑，必须手动触发**：
- `metaclaw_rollout_driver.py` 把"落盘进度"和"读进度并跳过"拆成两个独立开关：
  - `METACLAW_PROGRESS_DIR`：设置后，每天跑完（且没有中途异常）就把这天的逐轮 official score 写入 `<dir>/<test_id>.json`——纯记录动作，不改变这次跑的任何行为，正常训练也建议一直设着，方便万一真崩溃了有数据可续。
  - `METACLAW_RESUME=1`：**唯一**真正触发"跳过已完成的天"的开关，必须手动显式设置，且必须同时设了 `METACLAW_PROGRESS_DIR`（否则直接报错拒绝启动）。不设（默认）＝无论 `METACLAW_PROGRESS_DIR` 里有没有旧文件，永远从 day01 完整重新跑，正常训练不会因为凑巧复用了一个已有旧文件的目录就意外跳过某些天。
  - 全部跑完后生成 `<dir>/report.json`/`<dir>/report.md`（见下方"跟官方 report.md 对齐"一节）。
- 重新评估了之前否决按天续跑的两条理由，发现**都不再成立**：
  1. **workspace 一致性**：直接读 `_copy_workspace_for_test`（`infer_cmd.py:162-193`）确认每天的 workspace 都是从 `workspace_src` 全新拷贝，从不继承前一天的实际文件效果（跟已查证的"跨天无状态持久化"结论一致）——跳过某天的重新执行，不会让后面天数缺少任何文件状态，因为后面天数本来就不依赖前一天的真实文件效果。
  2. **checkpoint/天数进度不同步**：训练侧 `--load` 自动续训已经存在（`run_openclaw_topk_select_modelfactory.sh` 的既有修复），不受这次改动影响。剩下的风险变窄了：如果崩溃发生在"某天的 verdict 已经 POST 给代理"和"这个样本的梯度更新真正存进某个 checkpoint"之间，重启后从更早的 checkpoint 继续训练，那天的训练贡献会丢——但那天的**打分记录**（独立于训练 checkpoint 持久化）依然是模型当时真实产出的如实记录，最终聚合分数不会因此出错，只是权重轨迹有一小段缺口。这是被接受、不是被解决的权衡，跟论文自己的单趟边训边评方法本来就不承诺"最终模型跟每一天严格对应"是同一类取舍。
- 启动脚本新增 `METACLAW_PROGRESS_DIR`/`METACLAW_RESUME`（均默认空/0＝不开启，行为不变）；要续跑，第一次提交训练时就固定一个 `METACLAW_PROGRESS_DIR`（不要用默认按时间戳生成的 `LOGS_DIR`），崩溃后手动加上 `METACLAW_RESUME=1`、指向同一个目录重新提交这个脚本，才会跳过已完成的天。
- 三处改动均用合成数据做过功能测试：`_score_round_official` 对 multi_choice 部分正确的题目算出了正确的部分分（不是二元 0/1）；`main()` 级别的集成测试模拟"day01 已完成"场景，确认 `METACLAW_RESUME=1` 时 day01 被正确跳过、`METACLAW_RESUME` 不设时即使进度文件存在也正确忽略、`METACLAW_RESUME=1` 但没设 `METACLAW_PROGRESS_DIR` 时正确报错拒绝启动。真实训练环境（真实崩溃/重启场景）尚未验证。

### 训练 report 跟官方 `report.md`/`report.json` 对齐（2026-08-18）

**为什么之前不一样**：官方训练脚本 `rl_run.py` 内部就是调 `metaclaw-bench run --scene-per-train N`——跟纯基线评测走的是同一个 `metaclaw-bench run` 命令，只是多一个触发训练的 flag，所以官方训练和基线评测产出的 report 格式本来就是一致的（`infer_cmd.py`→`scoring_cmd.py`→`report_cmd.py` 一条龙）。**我们自己的 `metaclaw_rollout_driver.py` 没走这条流水线**——用的是自己的 Hybrid RL（GRPO+OPD topk-select，跑在 slime+Megatron 上），架构上没法塞进"一个同步 CLI 命令里顺便调训练"这个模型，所以之前只输出一个简化的 `final_scores.json`，没有跟官方对齐格式。用户指出这个不一致后，把 `report_cmd.py::run_report`/`_render_markdown` 的聚合与渲染逻辑原样搬过来（新增 `_build_report`/`_render_report_markdown`），改成读这次跑的内存里的 round 记录而不是扫描 `scoring.json` 文件，输出**同样文件名、同样结构**的 `report.json`/`report.md`——训练完和基线评测的报告可以直接放在一起对照看，不用换算格式。

`_score_round_official` 的返回结构也同步改成对齐官方 `scoring_cmd.py::_score_one` 的字段（`test_id`/`group_id`/`round_id`/`question_type`/`score`/`metrics`），不再只留 `question_type`/`score` 两个字段——`metrics`（`exact_match`/`f1`/`iou`/`precision`/`recall`/`passed` 等）本来就是 `_score_multi_choice`/`_score_file_check` 算出来的，之前只是没保留。

两点跟官方 report 的已知差异，如实标注在输出里，不是缺陷：
- **Token Usage 恒为 0**：官方 `report_cmd.py` 读的是 `infer_result.json` 里 `llm_log.messages[].usage`，我们的 driver 从来不落这种结构化文件（`_run_openclaw_agent` 只拿到原始 stdout 文本），所以这块数据我们确实没有——保留这个字段位置（都填 0）是为了跟官方 report 字段对齐、方便并排对比，不是伪造数据。（顺带一提，训练前基线那份报告的 Token Usage 也是 0/0——查过 `_extract_agent_tokens` 代码，那是 MetaClaw 自己的日志格式没读到 usage 字段导致的统计口径问题，不代表推理没有真的发生。）
- **`report.json` 没有 Compl. 字段**：Compl.（仅 file_check 通过率）是本项目为了对应论文 Table 1 才额外算的量，不是 `report_cmd.py` 本身的概念，所以没有塞进 `report.json`（保持跟官方 schema 完全一致），而是在 `report.md` 末尾单独加一行、日志里也打出来。

用合成数据验证过：两天的 multi_choice + file_check 混合记录，`_build_report` 算出的 summary/by_task 聚合数字和官方 `report_cmd.py::run_report` 手算结果一致；`_render_report_markdown` 渲染出的表格格式（列名、顺序、`-`占位）跟真实基线报告贴出来的样例逐列对得上。真实训练环境的 report 内容尚未验证。

**补充修复（同日）：report 默认不落盘的问题**。CLI 核实发现：启动脚本默认 `METACLAW_PROGRESS_DIR` 是空的，而 report 生成之前挂在 `if PROGRESS_DIR is not None` 这个判断下——不设 `METACLAW_PROGRESS_DIR` 就意味着 `report.json`/`report.md`/按天分数文件全都不落盘，Acc./Compl. 只会 print 进 `metaclaw_rollout.log`，没有独立文件。这是把"按天续跑要不要开"和"report 要不要存文件"这两个本该独立的问题耦合到了同一个开关上——续跑确实该默认关（上面已经决定手动开），但 report 作为这次跑的实际交付结果，不应该要求用户额外设置才能拿到文件。

**修复**：新增独立的 `METACLAW_REPORT_DIR`，专门管 report 落盘，不影响按天续跑：
- driver 里 `REPORT_DIR` 默认取 `METACLAW_REPORT_DIR`，没设就退回 `METACLAW_PROGRESS_DIR`（省得两个都要配置），两个都没设才是"只 print 不落盘"（同时会打一条 WARNING 日志提醒）。
- 启动脚本给 `METACLAW_REPORT_DIR` 一个**始终有值**的默认路径：`${LOGS_DIR}/report`（`LOGS_DIR` 本来就是每次跑都会生成的带时间戳目录）——不需要用户做任何配置，正常提交训练就会自动拿到 report 文件。
- 用合成数据验证过三种组合：只设 `METACLAW_REPORT_DIR`（report 落盘，没有按天续跑文件）；只设 `METACLAW_PROGRESS_DIR`（report 退回落到这个目录，同时按天续跑文件也在）；两个都不设（不落盘，但会打 WARNING，不再是静默行为）。

### 已废弃：独立 SGLang + `metaclaw-bench run` 打分法（2026-08-18 提出，同日撤回）

曾经想过"起独立 SGLang 服务 + 跑官方 `metaclaw-bench run`"给任意 checkpoint（包括训练前 base、训练后 checkpoint）单独打分，作为跟下面"边训练边算分"方法并存的补充手段。**用户指出这是错误做法后撤回**：这个方法打分用的 `all_tests.json` 跟训练用的是同一份 30 天数据，训练后的 checkpoint 拿这份数据打分，分数提升说不清是泛化能力还是记住了具体题目——挪到"训练完再单独测"并不能解决训练测试集重叠问题，只是把重叠发生的时间往后挪了一步，并没有比下面的实时聚合方法更干净，之前"更干净的补充手段"这个说法是错误判断。相应的 `scripts/metaclaw/compute_table1_scores.py` 已删除（`scripts/launch_simulator.sh` 本身是 Personal Agent Track 外部 Simulator 用的通用脚本，不受影响，未删除）。

Acc./Compl. 现在**唯一**的、跟论文 Table 1（Full 档）方法学对齐的产生方式见下方"训练/评测数据重叠"一节——`metaclaw_rollout_driver.py` 自己在训练过程中实时算分聚合。**训练跑完保存的最终 checkpoint 仍然保留，作为最终交付结果的一部分**（跟 Acc./Compl. 数字本身是否"干净"无关，checkpoint 本身没有被拿去跟同一份数据重新对比评分的问题）——这部分不需要额外代码，Megatron `--save`/`--load` 机制本来就会持续存盘。

**这条"撤回"不等于禁止再用 `metaclaw-bench run` 打训练前基线分**——被撤回的具体做法是"训练后再用同一套方法给 checkpoint 单独打分、拿来跟基线对比"，那才是训练测试集重叠的地方（checkpoint 已经在这份数据上训练过）。**训练前**（模型完全没见过这份数据）用这个方法打一次性的基线分，不存在这个问题，是干净的 zero-shot 测量，跟论文自己 Table 1 的"Baseline"那一行是同一类东西——下面"基线结果"一节记录的就是这样打出来的一次性基线，仅此一次，不会在训练后重复用同一方法再打一次跟它对比。

### 历史基线（2026-08-18 版，已被 2026-08-20 版取代——`--agent` 缺口修复前的结果，不再作为主基线）

**这份基线是在"`openclaw agent` 从未传 `--agent`、写入落到 checker 看不到的默认 workspace"这个缺口（2026-08-19d 修复）之前跑的**——大量本该正常完成的题目因为这个缺口（间接导致的 context overflow，见下方新基线一节的解释）被记成 `format_valid=False`/未完成，**已确认不适合再当主基线用**，仅作历史记录保留，不要拿来跟新训练结果比。

打分条件（"对齐基线"，第二次跑，取代第一次不对齐的版本）：
- 6 个系统级补丁全部按训练时的实际状态部署：`sglang execution-bias`/`embedded-agent overflow-recovery`/`system-prompt output-directives`/`cli-compaction`/`silent-reply-policy` 五个版本漂移补丁**开启**；`rl-training-headers` 插件**关闭**（原因见"已知风险/限制"——这个插件注入的 `[RL-TRAINING-META]` 标记只有训练代理会剥除，基线直连 SGLang 没有剥除环节，开着会让模型看到训练时看不到的内容）。
- 独立 SGLang 服务（4B，端口 30002）+ 官方 `metaclaw-bench run` 走完整 `infer→scoring→report` 流水线，30 天全量、346 题。**没有 `--agent metaclaw_agent`**（这个修复是 2026-08-19d 才做的，这次基线跑在那之前）。

结果（目录 `/dfs/data/openclaw-rl-project/metaclaw-baseline-eval-aligned/run_20260818_141305/`，seed `589953305`）：

| 指标 | 对齐基线（旧，已作废） | 不对齐（更早，仅供参考） |
|------|:---:|:---:|
| Acc. | 8.1%（Correct=28.0/346） | 5.7% |
| Compl.（224 道 file_check） | 0.0%（0/224，全部 `score=0`/`passed=0`） | 0.0% |
| Context overflow | 49.1%（全题） | — |

Acc. 从 5.7%→8.1% 的提升全部来自 multi_choice（满分题数 12→18，部分正确题数 19→23）——这部分是被 `rl-training-headers` 标记污染拖累过的，`--agent` 修复后的新基线证实**这两次结果都被大量 Context overflow（49.1%）严重拉低**，不是"4B 模型在 file_check 上一题都做不对"这么简单的能力上限结论——旧结论里"关键结论"那部分已被推翻，不再采信。

### 基线结果（用于后续对比，2026-08-20 定版，`--agent` 修复后的新基线）

**这是本次 MetaClaw 迁移当前唯一采信的训练前基线，取代 2026-08-18 版**。选用理由：同一套"agentfix"harness（已部署 `--agent metaclaw_agent` 修复）下跑了三个不同 SGLang seed 的独立结果——`247444587`（Acc=17.7%，overflow 17.9%）、`589953305`（Acc=15.3%，overflow 5.5%）、`465485731`（Acc=17.8%，overflow 4.6%）——选 Acc 最高且 overflow 最低的 `465485731` 这次作为正式基线，其余两次仅作对照记录（不是丢弃，三次结果都指向同一个结论，互相印证）。

**身份与路径**：

| 项 | 值 |
|------|------|
| 名称 | agentfix baseline seed 465485731 |
| 结果目录 | `/dfs/data/openclaw-rl-project/metaclaw-baseline-eval-aligned/run_20260820_192625_agentfix_seed465485731/run_20260820_192725/` |
| report.json/report.md | 上述目录下 |
| Manifest | `.../run_20260820_192625_agentfix_seed465485731/SEED_RERUN_MANIFEST.txt` |
| Eval log | `/dfs/data/openclaw-rl-project/logs/metaclaw_baseline_agentfix_seed465485731_eval.log` |
| SGLang log | `/dfs/data/openclaw-rl-project/logs/metaclaw_baseline_agentfix_seed465485731_sglang_30002.log` |
| 时间 | Manifest started 2026-08-20T19:26:25+08:00，eval 实际启动 2026-08-20T19:27:25+08:00（pid 82876），完成约 2026-08-20T23:27+08:00 |

**打分条件**（跟 2026-08-18 版的关键差异是有 `--agent`，其余延续"对齐基线"设计）：
- 模型：`/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507`（served `qwen3-4b`）；独立 SGLang，`127.0.0.1:30002`，`context_length=65536`，`--random-seed 465485731`。
- Harness 命令：`python -m src.cli run -i .../all_tests.json -o <OUT> -w 1 -n 0`（`MetaClaw-official/benchmark` 目录下），**`--agent metaclaw_agent`**（`infer_cmd.py` 本地补丁，旧基线没有这个参数——正是 2026-08-19d 那次修复本身）。
- OpenClaw 系统级补丁 2-6（`sglang`/`embedded-agent`/`system-prompt`/`cli-compaction`/`silent-reply`）全部部署，跟训练时状态一致。
- `rl-training-headers` 插件关闭（直连 SGLang，避免注入未剥离的 `[RL-TRAINING-META]`，原因跟旧基线一致）。
- 题集：MetaClaw-Bench Part I，`all_tests.json`，346 题/30 天，全量。

**主指标（官方 report）**：

| 指标 | 值 |
|------|------|
| Acc. | **17.8%**（accuracy=0.17825488576933662，Correct=61.676/346） |
| Compl.（224 道 file_check 均分） | **0.0%**（0/224） |
| MC 均分（122 题） | 50.6% |
| MC format_valid | 89/122（73.0%） |
| Context overflow（全题） | 16/346（4.6%）；MC 10 道、FC 6 道 |

官方 `metrics`（平均）：`passed=0`，`exact_match=0.1127`，`f1=0.1727`，`iou=0.1616`，`precision=0.1900`，`recall=0.1666`。

**跟旧基线/同 harness 其它 seed 的对照**：

| 跑 | seed | Acc. | Context overflow（全题） |
|------|------|:---:|:---:|
| 旧对齐（已作废）`run_20260818_141305` | 589953305（无 `--agent`） | 8.1% | 49.1% |
| agentfix | 247444587 | 17.7% | 17.9% |
| agentfix | 589953305 | 15.3% | 5.5% |
| **本基线（采信）** | **465485731** | **17.8%** | **4.6%** |

**关键结论**：旧基线 8.1% 偏低的主因确认是**大量 Context overflow（49.1%）被记成未完成/`format_valid=False`**，不是"4B 模型在这类任务上能力真的这么差"——`--agent` 修复后 overflow 从 49.1% 骤降到 4.6%，Acc. 提升几乎全部来自 multi_choice（真正生成出答案时，两版的 `\boxed{}` 合规率都在 98% 左右，说明模型本身的格式遵循能力没有变化，变化的是"能不能把题真正答完"）。**`Compl.`（file_check）在新旧基线里都是 0.0%，这一点没有变**——file_check 类任务确实是真实的能力上限，不是链路/overflow 问题，这条结论继续保留，只是不再跟一个被 overflow 严重拉低的 Acc. 数字捆在一起。

按天 Acc.（本跑，供参考）：day01 38.3 | 02 19.7 | 03 -1.4 | 04 36.7 | 05 29.5 | 06 29.5 | 07 27.3 | 08 15.9 | 09 23.9 | 10 11.5 | 11 0.0 | 12 24.2 | 13 4.2 | 14 12.7 | 15 20.5 | 16 30.0 | 17 25.5 | 18 -1.4 | 19 13.0 | 20 23.1 | 21 0.0 | 22 23.5 | 23 2.8 | 24 6.1 | 25 16.7 | 26 16.7 | 27 33.6 | 28 35.9 | 29 1.5 | 30 18.4（负分是官方计分口径允许的，不是记录错误）。

**记法（可直接引用）**：
> Training-prior baseline (canonical): MetaClaw-Bench Part I, base Qwen3-4B-Thinking-2507, agentfix harness (--agent metaclaw_agent, patches 2-6 on, rl-training-headers off), SGLang seed 465485731, Acc 17.8%, Compl 0%, path metaclaw-baseline-eval-aligned/run_20260820_192625_agentfix_seed465485731/run_20260820_192725. Do not use old aligned 8.1% (run_20260818_141305) as primary baseline (overflow-dominated).

**当前机器状态提醒（操作性，非分析结论）**：这次基线打分用的独立 SGLang（4B，:30002）跟旧基线是同一台机器上的同一类临时服务，**下一次提交训练前需要确认这个独立评测用的 SGLang 服务已经停掉、`rl-training-headers` 插件重新 `enable`**（训练走 30000 端口代理，依赖这个插件的标记做 session 分派，关着会静默回退，重蹈这次迁移最早发现的问题）——具体清理动作和下一趟训练什么时候提交，由用户决定。

### 已知风险 / 限制（如实列出，展示时需要一并说明）

- Qwen3-4B 在文件操作/JSON 结构化/shell 脚本这类任务上的底子未知，跟 GSM8K 数学题是完全不同的能力域，训练效果存在不确定性。
- MetaClaw-Bench 是作者自己编写的模拟基准，不是真实用户会话采集，论文原文也提醒"绝对数值可能不直接迁移到生产场景"，我们的结果同样适用这条限制。
- `file_check` 题的 OPD hint 改用 checker stdout 而不是静态 `feedback.incorrect`（见下方查证记录第 1 条）——这条修正逻辑已经想清楚，但**实际接入代码、实测蒸馏效果是否真的比静态文字更好，还没做**。
- 按天顺序、concurrency=1 串行喂数据这个设计，跟现有 Megatron/slime 的 batch 收集逻辑（`_drain_output_queue` 等）配合是否顺畅、吞吐是否够用，还没有实测验证（架构上确认可行，性能上未知）。
- 跨天没有任何文件/session 状态持久化（见下方查证记录第 3 条）——每天的"记忆"完全依赖模型权重本身的更新，如果某天的训练没有真正让权重产生可观测变化，后续天数就学不到前面天数的教训，这是一个比"batch 组成随机性影响训练成功率"（本项目在 separate 阶段反复验证过的现象）更敏感的失败模式，需要在正式跑之前想清楚怎么监控。
- **`rl-training-headers` 插件对训练/基线两条链路的实际效果不对称**（2026-08-18 发现，同日用对齐基线验证过实际影响范围）：这个插件的注入是无条件的——只要启用，每次 `before_prompt_build` 都往系统提示词末尾追加 `[RL-TRAINING-META] session_id=... turn_type=...`（见上"启动脚本必须复用的现有依赖"一节）。这条标记只有**训练代理**（30000 端口，`openclaw_opd_api_server.py::_strip_rl_meta_from_messages`）才会在转发给 SGLang 之前剥掉——训练时模型看到的是干净提示词。基线评测走的是"直连 SGLang"，不经过训练代理，这条标记不会被剥，模型会看到训练时从没见过的这行后缀。**用关掉插件的"对齐基线"重跑验证过实际影响范围**：Acc. 从 5.7%→8.1%，提升全部来自 multi_choice（满分 12→18、部分正确 19→23）——说明这条标记确实污染过 multi_choice 的输出；但 **`Compl.=0.0%` 两次结果完全一样，没有被这条标记影响，是文件操作类任务的真实基线能力上限，不是链路污染**（完整数据见下方"基线结果"一节）。结论：这个不对称是真实存在的，但只影响 multi_choice 类型的评测/训练，不影响 file_check——重跑基线或做任何直连 SGLang 的评测前仍然应该 `openclaw plugins disable rl-training-headers`，但不要因为这条风险去怀疑 file_check 相关的数字。

  **补充说明（2026-08-20）**：这条实验用的 5.7%/8.1% 两个数字来自 `--agent` 修复**之前**的旧基线（那两次跑都受同一个 Context overflow 问题影响），绝对值已经过时（见下方新基线一节）。但这是一次自身闭环的相对比较（同一套有缺陷的 harness，只切换 `rl-training-headers` 开关），"标记只污染 multi_choice、不影响 file_check"这条结论作为相对关系没有必要重新验证——`Compl.=0.0%` 在新旧基线里都不变，进一步印证了这条结论没有被推翻，只是不要再拿 5.7%/8.1% 这两个绝对数字当参考。

### 训练故障复盘与修复：metaclaw_migration_20260817_181404（2026-08-18）

第一次真实训练（08-17 18:14→18:42）表面上"正常关闭"（driver 把 day01→day30 扫完后按设计主动 kill 训练），**但整次训练零有效样本**——从 day01 r1 起，346 次 `openclaw agent` 全部失败，`agent_succeeded=True` 0 次，训练侧一直卡在 `waiting for combine samples: 0/16`，checkpoint 目录从未创建。CLI 提供了详细日志诊断，以下是逐条代码级复核（不直接采信，按项目规则验证）后的结论：

**根因 1（主因，100% 训练样本丢失）：网关鉴权 token 未在两个子进程间共享**

日志报错：`GatewayCredentialsRequiredError: gateway agent requires credentials before opening a websocket`；网关日志：`Gateway auth token was missing. Generated a runtime token for this startup...`。

- `metaclaw_rollout_driver.py` 的 `_start_work_gateway`（起网关）和 `_run_openclaw_agent`（起 agent 客户端）**都是直接从 MetaClaw 官方 `infer_cmd.py` 原样 import 的**（见文件头 import 列表），不是我们自己写的代码。两个函数各自单独 `env = {**os.environ, ...}` 起子进程，从不设置 `OPENCLAW_GATEWAY_TOKEN`，也没有任何机制在它们之间传递 token。
- 核实这**不是我们移植时漏做的事**：直接读 MetaClaw 自己的 `_run_one_test`（`infer_cmd.py:999-1042`）调用序列，跟我们 driver 里的序列逐行一致——MetaClaw 自己的官方评测主线代码存在一模一样的 gap，只是从没暴露过。
- 用 `git grep` 在本地 `openclaw` 仓库两个版本快照（`march_2026_3_8` / `may_2026_5_11` 标签）分别搜 `GatewayCredentialsRequiredError`：**march 零命中，may 大量命中**（`src/commands/agent-via-gateway.ts` 等）——确认这是三月之后才加入的强制网关鉴权机制，不是一直存在的行为。命中 CLAUDE.md 的判断框架："只在 may 才出现、march 没有的行为，默认按版本漂移处理"。
- 顺带确认了 May 版自己的修复设计：`agent-via-gateway.ts` 的 `shouldRetryGatewayDispatchWithShellEnvFallback` 在遇到这个错误时会重试一次、改用 `OPENCLAW_GATEWAY_TOKEN` 环境变量兜底；`gateway/server.impl.ts` 也确认——网关启动时如果 `OPENCLAW_GATEWAY_TOKEN` 已经在环境变量里，就直接用这个值，不会生成随机 runtime token（`authBootstrap.generatedToken` 才会触发警告日志）。也就是说 May 版本身已经预留了"用环境变量共享同一个 token"这条路，只是 MetaClaw 官方代码没接上。
- **修复**（`metaclaw_rollout_driver.py`）：driver 进程启动时，若 `OPENCLAW_GATEWAY_TOKEN` 未设置，用 `secrets.token_hex(16)` 生成一个并写回 `os.environ`。因为 `_start_work_gateway`/`_run_openclaw_agent` 都用 `{**os.environ, ...}` 构建子进程环境，设一次即可让每天的网关和当天所有 `openclaw agent` 调用自动共享同一个 token，不改 MetaClaw 官方代码。每天复用同一个 token 没有风险——网关是纯本地短生命周期进程，不对外暴露。

**根因 2（次因，即使修好根因 1 也会持续存在）：driver 自己提交 verdict/close 时没带鉴权头**

CLI 观察到每天结束时 30000 端口的 close/verdict POST 返回 401。核实：训练代理（`openclaw_opd_api_server.py:365-372` 的 `_check_auth`）只要 `SGLANG_API_KEY` 在服务端环境变量里设了值，就要求所有请求带 `Authorization: Bearer <SGLANG_API_KEY>`，没带或不对就 401。`openclaw agent` 自己的真实请求不受影响，是因为启动脚本把 `BENCHMARK_API_KEY=${SGLANG_API_KEY}` 接进了它的 `openclaw.json` provider 配置；但 driver 自己直接用 httpx POST 提交 verdict/close（`_send_verdict_turn`/`_send_session_close_only`）完全绕过了这层配置，从来没带过这个头。

还发现一个连带问题：`_post_with_retry` 里 `client.post(...)` 从来没检查响应状态码（httpx 默认不对 4xx/5xx 抛异常），所以这些 401 会被当成"请求成功"，driver 自己的日志完全看不出异常——这也是为什么之前只能靠直接翻代理日志才发现，driver 日志本身没有任何警告。

**修复**（`metaclaw_rollout_driver.py`）：
- 新增 `_API_KEY = os.environ.get("SGLANG_API_KEY", "")`，`_post_with_retry` 内部统一在 headers 里补上 `Authorization: Bearer {_API_KEY}`（只改一处，两个调用方不用各自记得加）。
- `_post_with_retry` 补上 `response.raise_for_status()`，401/5xx 现在会走已有的重试/日志分支，不再被静默吞掉。
- 启动脚本 `run_metaclaw_migration_modelfactory.sh` 新增把 `SGLANG_API_KEY="${SGLANG_API_KEY}"` 传给 driver 进程（此前只传了语义相同但改了名字的 `BENCHMARK_API_KEY` 给 `openclaw agent` 用，driver 自己没拿到）。

**验证方式**：两处修复均用合成数据做过功能测试（伪造 httpx client 直接检查生成的 `os.environ["OPENCLAW_GATEWAY_TOKEN"]`、检查 `_post_with_retry` 实际发出的 headers 里 `Authorization` 值正确、伪造一个真实 401 `httpx.Response` 确认现在会触发警告日志而不是静默"成功"），未在真实网关/代理上跑过——下一次训练提交后才能确认这两个根因是否真的被修复到位（比如 `agent_succeeded=True` 是否开始出现、checkpoint 目录是否开始创建）。

### 训练故障复盘与修复（二）：metaclaw_migration_20260818_*，context overflow（2026-08-18）

上一轮网关鉴权修复后重新提交的训练，卡在了另一个真实问题上：**day01 起就大面积 context overflow，全程零训练样本。**

**现象**：`openclaw agent` 请求 16661 输入 + 30313 `max_tokens` = 46974 token，超过训练 SGLang 引擎的官方默认 `--sglang-context-length 32768`。SGLang 回 400 context overflow，代理转成 500，driver 记 `agent_succeeded=False`（正确地没有提交假训练信号，这部分行为符合预期）；连续失败触发 router 熔断，之后全部变成 503 no_available_workers。`agent_succeeded=True` 恒为 0，`day01.json`～`day04.json` 全是空 `[]`（`METACLAW_PROGRESS_DIR` 正确记录了实际发生的情况——空文件本身不是 bug，见下面 resume 修复）。

**根因**：32768 是官方 `run_qwen3_4b_openclaw_topk_select.sh` 里 `CONTEXT_LENGTH`/`--rollout-max-context-len`/`--sglang-context-length` 三处硬编码的值——这是论文针对 **Personal Agent Track**（GSM8K 风格、对话短）调优的数字，不是通用默认值（对比确认：OpenClaw-RL 自己的 `toolcall-rl` 4B 训练脚本用的是 `--rollout-max-context-len 16384`，比 32768 还小，说明 32768 在 OpenClaw-RL 自己的赛道里已经算大配置，不是任何"安全余量"）。MetaClaw 的系统提示词（工具 schema + skills + memory + 当天任务文件）比 GSM8K 对话重得多——**查了 MetaClaw 自己的官方配置模板 `openclaw_cfg/openclaw.json`/`metaclaw.json`（我们直接复用、没改过的那份），明确声明 `"contextWindow": 50000, "maxTokens": 50000`**，不是我们瞎猜的数字，是论文作者自己给这套基准配的值。之前跑通的基线评测用的是临时起的 65536 上下文 SGLang，凑巧覆盖了 50000，才没暴露这个问题。

**修复**（`scripts/run_openclaw_topk_select_modelfactory.sh`，`METACLAW_MIGRATION_PROFILE=1` 分支新增三条 sed）：把 `CONTEXT_LENGTH`/`--rollout-max-context-len`/`--sglang-context-length` 从 32768 统一改到 **65536**（用户决定：MetaClaw 官方声明的下限是 50000，但训练过程中会话内容会累积增长，比声明值再多留一点余量，65536 也是恰好跑通过的基线用过的值）。`--max-tokens-per-gpu 32768`（训练侧单 GPU 显存预算）不动——跟 sglang context 是否溢出无关，沿用同一脚本 `SMOKE_PROFILE` 分支的既有结论（是否需要一起调大待真实训练报错验证，不提前假设）。用 sed 直接在真实官方脚本上跑过一遍确认三处都改对、`--max-tokens-per-gpu` 确认没被误改。

**连带修复：resume 会把"零样本天"误判成"已完成"**。设计 `METACLAW_PROGRESS_DIR`/`METACLAW_RESUME` 时没考虑到"一天里所有轮次都因基础设施故障失败"这种情况——`run_day` 不会因此抛异常，所以空列表 `[]` 照样会被持久化成"这天跑完了"。如果之后误开 `METACLAW_RESUME=1` 指向这次的进度目录，day01～day04 会被当成已完成直接跳过，永远没机会用修好的 context 重跑。**修复**：`main()` 里判断"要不要跳过"的条件从 `if resumed is not None` 改成 `if resumed:`——`None`（文件不存在）和 `[]`（文件存在但是空）都是 falsy，同一个判断就能把两种"这天其实没有真正完成"的情况都正确导向重跑；同时空列表文件本身依然保留（诊断价值：能看出"这天真的跑过但一个样本都没产生"，不是"从没跑到过"），只是不再被当作"已完成"处理。

**明确建议**：**这一轮训练（`metaclaw_migration_20260818_*`）直接停掉，不要用它产生的 `METACLAW_PROGRESS_DIR` 做 resume**——里面全是空天，没有任何真实进度可续，重开一个新的 `METACLAW_PROGRESS_DIR`、用修好 context 的脚本从 day01 完整重跑。

**尚未验证/暂缓的问题**：CLI 同时发现这次失败的请求里没有出现 `[RL-TRAINING-META]` 标记（rl-training-headers 插件本该无条件注入的那个）。这条先不查——现在所有请求都还没走到"模型真正生成"这一步就先因为溢出失败了，没法判断是插件真的没生效、还是取样/日志位置没截到这段内容。等 context 修好、真的有请求能跑通之后，这是下一个要核实的问题：如果连这个标记都没有，`_METACLAW_SESSION_RE` 整套确定性 reward/步骤判官分派机制可能从一开始就没生效，训练可能一直在往 Personal Agent Track 的原逻辑上回退而不自知。

### 训练过程可读性：对齐 Personal Agent Track 的 simulation.log（2026-08-18）

用户反馈：之前 driver 日志只有 `passed=%s agent_succeeded=%s` 这类结构化一行摘要，看不到实际问了什么、模型答了什么、判官给了什么反馈——Personal Agent Track 的 `train_separate_student.sh` 有专门的 `simulation.log`（`student_chat.py` 直接 `print()` 每一轮 `>> Student -> OpenClaw:`/`<< OpenClaw -> Student:` 完整对话文本），用户需要能手动通读这种原始转录，才能看出 agent 容易漏掉的规律（不是结构化字段能覆盖的那种模式）。

**实现**：在 `_run_round`（生成每一轮实际内容的地方）里加了跟 `student_chat.py` 同样风格的 `print()`（不是 `logger`，保持跟已有 `simulation.log` 一致的朴素输出、不带日志级别噪音）：
- `run_day` 开头打印天级别的分隔标题（`# Day dayXX (session: metaclaw-dayXX)`）。
- 每轮打印完整的 `>> Query -> OpenClaw`（含拼接进去的上一轮反馈文字）和 `<< OpenClaw -> Query`（`openclaw agent` 的完整原始 stdout，不截断——即使这段包含模型的工具调用轨迹而不只是最终答案，这段原始轨迹往往正是人能看出问题、结构化字段看不出来的地方）。
- 每轮打印 verdict（`passed`/`agent_succeeded`/官方连续分数）。
- 如果失败且生成了 OPD hint，打印这段 hint 文本（会被喂进下一轮反馈）。

不需要额外的日志文件或环境变量——这些 `print()` 输出的去向和 Personal Agent Track 的 `simulation.log` 走的是同一个机制：启动脚本本来就把 driver 整个进程的 stdout/stderr 重定向进 `metaclaw_rollout.log`（`run_metaclaw_migration_modelfactory.sh` 第 3 步），只是之前 driver 自己没打印过这些内容。`tail -f metaclaw_rollout.log` 现在就能看到跟 `simulation.log` 同等详细程度的转录。

用合成数据验证过完整一轮的打印格式（mock `_run_openclaw_agent` 返回固定内容，跑真实的 `_run_round`），确认 query/answer/verdict 都正确显示、格式跟预期一致。真实训练环境的日志量/可读性尚未验证。

**补充修复：`print()` 内容在真实训练里没有实时出现（2026-08-18 当天发现）**。第三次真实训练提交后，CLI 核实 `metaclaw_rollout.log` 里只看到 `logger.info` 那一行摘要，没有 `>> Query -> OpenClaw`——诊断：Python 的 `stdout` 一旦不连终端（`> metaclaw_rollout.log 2>&1 &` 这种重定向）就会从行缓冲切换成全块缓冲（4-8KB 才刷新一次），而 `logging` 默认走 `stderr`，`stderr` 不受这个影响，所以只有 `logger.info` 那行实时可见，`print()` 的内容已经执行了，只是还堵在缓冲区里，要攒够量或者进程退出才会真正写进文件。**修复**：`if __name__ == "__main__":` 里加一行 `sys.stdout.reconfigure(line_buffering=True)`，强制 stdout 也变成行缓冲，跟 `stderr` 行为一致。用真实子进程重定向到文件的方式验证过（跟启动脚本的重定向写法完全一致）：不加这行时，子进程还在跑（没退出）的时候文件里读不到任何内容；加了这行后，子进程还没退出，文件里已经能读到之前打印的内容——修复前后各测了一遍，确认问题真实存在、修复真的有效。这次训练已经在跑，改动要下次重新提交训练才会生效；这次先靠现有的 `logger.info` 摘要行判断训练是否正常（`day01` 已出现 `agent_succeeded=True`/`passed=True`，训练链路本身是通的，只是详细转录这次看不到实时更新，得等进程退出或缓冲区攒满才能看到）。

**第三个补充修复：文件里有内容，但终端还是看不到（2026-08-18，第四次训练结果确认前一个修复真的生效了）**。第四次训练（带 `plugins.allow` 修复那次）跑完后确认 `metaclaw_rollout.log` 里已经有完整的 30 个 `# Day` 标题和 346 条 `>> Query -> OpenClaw`——`sys.stdout.reconfigure(line_buffering=True)` 那个修复确实有效，内容已经完整落盘。但直接盯着训练脚本自己的终端/job 输出还是看不到——因为启动脚本对 driver 用的是纯重定向 `python ... > metaclaw_rollout.log 2>&1 &`，只往文件写，不会同时出现在脚本自己的 stdout 里；对照 Personal Agent Track 的 `simulation_loop`，那边用的是 `... 2>&1 | tee -a simulation.log`——`tee` 会同时写文件和自己的 stdout，所以脚本本身的终端/job 输出里也能看到。**修复**：把 driver 的重定向从 `> "${LOGS_DIR}/metaclaw_rollout.log" 2>&1` 改成 `> >(tee -a "${LOGS_DIR}/metaclaw_rollout.log") 2>&1`（进程替换语法，不是直接 `| tee`——直接管道会导致 `$!` 拿到的是 `tee` 的 PID 而不是 python 自己的 PID，后面 `kill "${DRIVER_PID}"` 那段清理逻辑会杀错进程；进程替换保留 `$!` 仍然是 python 自己的 PID，`tee` 作为旁路子进程独立运行，两头都写）。验证过三点：`$!` 确实还是拿到 driver 自己的 PID（不是 `tee` 的）；`kill "$DRIVER_PID"` 确实能正确杀掉真正的 python 进程（用一个 30 秒 sleep 的测试进程验证过，kill 后进程真的消失，日志里没有出现"不该打印"的后续内容）；日志文件内容完整、没有因为改成 `tee` 少写东西。这个修复要下次训练才会生效，这次的 30 天转录内容仍然完整地在 `metaclaw_rollout.log` 文件里，只是没有同时出现在终端。

### 训练故障复盘与修复（三）：metaclaw_migration_20260818_175145，rl-training-headers 从未真正加载（2026-08-18）

第三次真实训练（context overflow 修好之后提交的那次）表面上一切正常——agent 在正常答题，GPU 4/5 利用率 80%+，`day01` 10 题全部有分（3 道 multi_choice 满分 + 1 道部分分，file_check 仍全 0，这部分符合基线预期），`day02` 也在往前走。**但训练队列一直是 `waiting for combine samples: 0/16`，`submitted OPD/RL = 0`——权重完全没有在学**。

**CLI 的现场诊断**：OpenClaw 自己发出的请求（包括真正干活的 read/write 工具调用轨迹）完全没有 `session_id`、也没有 `[RL-TRAINING-META]` 标记，代理只能记成 `session=unknown`/`turn_type=side`，直接当非训练数据丢弃；driver 自己直接 POST 的 checker verdict 能对上 `metaclaw-day01`（这条不依赖插件，是 driver 自己手动设置的 HTTP header），但 OpenClaw 自己产生的 MAIN 请求全部丢失，没有一条真正进入训练队列。CLI 把根因归到"work-copy 里 rl-training-headers 没把 session_id 打进 OpenClaw 内部请求"，方向是对的，但没有找到具体机制——**直接读了 OpenClaw 官方源码（`may_2026_5_11` 快照）确认了精确根因**：

1. `openclaw plugins enable rl-training-headers` 只会写入**全局** `~/.openclaw/openclaw.json` 的 `plugins.enabled`，这个全局状态跟某个具体 config 自己的 `plugins.allow` 字段是两回事。
2. MetaClaw-official 自己的 `openclaw_cfg/openclaw.json` **和** `metaclaw.json`（`_prepare_work_copy` 复制进每天隔离工作副本的那两份模板，两份都查证过）**都硬编码 `"plugins": {"allow": ["llm-prompt-logger"]}`**——完全没有 `rl-training-headers`。
3. `src/plugins/config-activation-shared.ts::resolvePluginActivationDecisionShared` 有一条明确的判断：`if (config.allow.length > 0 && !explicitlyAllowed) return {enabled: false, cause: "not-in-allowlist"}`——**非空的 `plugins.allow` 是一个真正的白名单闸门，不在名单里的插件会被直接排除，不管全局有没有 enable**。

也就是说：**`rl-training-headers` 从这次迁移一开始，就从没在任何一个 MetaClaw session 里真正加载过**——`before_prompt_build` 钩子从没触发过，标记从没注入过，OpenClaw 自己发出的每一条请求都因为拿不到 session_id 落进代理的默认分支被丢弃。这不是"某些中间轮次"的局部问题，是全局性的——之前几轮训练一直没能提交样本，除了网关鉴权和 context overflow 这两个已修的基础设施问题外，这条才是真正决定"能不能学到东西"的根因，而且此前一直被基础设施问题挡在前面，没机会暴露出来。

**修复**（`metaclaw_rollout_driver.py`）：新增 `_ensure_plugins_allowlisted()`，在每天的工作副本 `openclaw.json` 生成后（`_patch_agent_workspace` 之后、起网关之前）把 `"rl-training-headers"` 追加进这份文件自己的 `plugins.allow` 列表——只改工作副本，不碰 MetaClaw-official 的模板源文件（跟"改副本不改官方源文件"的既有约定一致）。用真实的 MetaClaw 官方 `openclaw_cfg/openclaw.json` 模板验证过：改之前 `allow=['llm-prompt-logger']`，改之后变成 `allow=['llm-prompt-logger', 'rl-training-headers']`（保留原有条目，不影响 `llm-prompt-logger` 自己的配置）；也验证过幂等性（重复调用不会重复追加）和防御性场景（配置文件完全没有 `plugins` 字段时能正确从零构建）。

**这次训练（`metaclaw_migration_20260818_175145`）已经在跑，且证实完全没有产生任何训练样本，不会有任何权重更新**——用户已知情，可以让它继续跑完只拿 Acc./Compl.（不影响这条数据的有效性，评测链路本身没问题），但这次的 checkpoint 学不到任何东西。这个修复要等下一次重新提交训练才会生效——**下次训练是第一次真正有条件验证"确定性 reward/步骤判官分派机制"是否按设计工作**，之前所有验证都因为更上游的基础设施问题（网关鉴权→ context overflow→ 插件白名单）依次卡住，从没真正走到这一步。

**一个尚未解开的疑点，如实记录**：如果 `plugins.allow` 真的从一开始就无条件排除了 `rl-training-headers`（不受全局 enable/disable 影响，代码逻辑上确认是硬性早退），那"对齐基线"（关插件）和"不对齐基线"（开插件）两次评测理论上不应该有任何差异——但实测 Acc. 从 5.7% 变到了 8.1%。这两个结论字面上互相矛盾，还没有找到能同时解释两者的机制（比如全局配置和工作副本配置之间是否存在某种合并/继承关系，本次没有查证）。当前的判断是：**不管这个矛盾怎么解释，把 `rl-training-headers` 加进工作副本的 `plugins.allow` 都是必须且无害的修复**——这是训练场景下让确定性 reward 机制工作的必要条件，不依赖这个疑点的答案。但那两次基线 Acc. 差异的真正原因，眼下不能 100%确定就是插件标记污染，这条待下次训练验证（如果标记确实开始出现在真实请求里，能间接印证是这个机制在起作用；如果修完之后基线/训练的差异模式还是解释不通，需要重新查这个疑点）。

### 第一次真正产生训练样本的训练结果：metaclaw_migration_20260818_182736（2026-08-18）

带 `plugins.allow` 修复重新提交后（commit `b18c791`，18:27→19:36），**这是本次迁移第一次真正走通训练样本链路的一次运行**：`[RL-TRAINING-META]` 标记确认出现，SIDE/skipped 请求数 = 0，累计提交 234 条 RL 样本，训练走完 step 0～13，`checkpoint` 存到 `iter_0000009`。day01→day30 全部跑完，`report.md` 已生成。

**结果对比**：

| 指标 | 这次训练中的评测（234 样本进了训练） | 对齐基线（`run_20260818_141305`，未训练） |
|------|:---:|:---:|
| Acc. | 8.3%（338 题，另 8 题基础设施失败未计入） | 8.1%（346 题） |
| Compl.（file_check） | 0.0% | 0.0% |

**如实评估，不回避**：数字跟冻结基线几乎一样，**不像训练方法已经拉开差距**。更值得注意的是逐日模式：`day01` 准确率还有 38%，`day11` 之后几乎全是 0——不是"训练完全没用、从头到尾都跟基线一样低"，而是**前几天表现明显更好、之后迅速塌陷到接近 0**，`Compl.` 则自始至终是 0，`file_check` 类任务完全没有被训出来。这个"先好后差"的模式本身就是一个需要解释的现象（可能是训练不稳定/灾难性遗忘，也可能是别的原因），不是简单的"方法无效"就能盖棺定论的，需要看训练过程中的具体样本（`metaclaw_rollout.log` 里的转录、`report.md` 的逐日明细）才能判断下一步该往哪个方向调整——这正是"人类可读转录"这个功能存在的意义，具体分析留给用户看原始转录后决定。

**更正（同日，下一节）**：上面这句判断当时是错的——这次训练看起来没被"基础设施问题"污染，但被两个更隐蔽的训练信号 bug 污染了，"先好后差"这个模式本身就是被污染训坏的直接后果，不是需要另外解释的独立现象。详见下一节的根因分析和修复。

### 训练信号根因分析与修复：checker 分数丢失 + verdict 残片污染 GRPO（2026-08-19）

CLI 深挖 `metaclaw_migration_20260818_182736` 的训练日志和 wandb 曲线，定位到"先好后差"塌陷模式的真正根因——**不是环境突然出问题，是从 step 0 起训练信号就在灌毒，大约 9 步后策略学会了空回复/纯 thinking，`day11` 只是权重已经被拉歪之后的必然结果**。经逐条代码核实（不直接采信 CLI 的结论），确认是两个真实 bug 叠加：

**Bug A：checker 分数从没打到真实最后一轮**——`_send_verdict_turn` 把 checker 判决当 `X-Turn-Type: main` 的合成轮次 POST 给代理，`_opd_evaluate()` 认出了 checker 分（日志里 265 次 `deterministic-reward`），但代码会继续往下掉进 PRM 分支共用的 `return` 语句，那两处 `return` 都引用 `_skip_forced_negative_override`——这个变量只在 `_metaclaw_verdict is None` 的 PRM 分支里才会被赋值，MetaClaw verdict 分支从来没赋值过就被读取，Python 必然 `UnboundLocalError`。265 次 checker 评估全部在这一步炸掉，真正该吃 ±1 的最终轮被静默丢弃。

**Bug B：verdict 自己的生成残片反而进了 GRPO**——`X-Turn-Type: main` 除了"把这次消息内容挂给上一轮当 next_state"这个需要的效果之外，还有一个无法避免的副作用：这次调用自己的生成结果也会被无条件注册成一条新的待评估轮次。`max_tokens=8` 时 Qwen3-4B-Thinking 把预算花在 thinking 上，留下 13-token 残片；这条残片后续被下一题的内容挂上 next_state、被 step-judge 打分、当 RL-only 提交——`metaclaw_migration_20260818_182736` 里 234 条提交样本中 113 条（48%）是 `response_len==13` 且 `prompt_len<200` 的残片，其中 69 条还拿到了 +1。真实 checker 信号整晚没进优化器，batch 里从第一步起就混着这类被误标的残片，GRPO 很快学会"少说话/空回复"，这才是 `day11` 后大面积 `generate-fail`/Acc. 塌陷的真正原因。

**方案制定过程（三轮 CLI 交叉核实，纠正了两版方案）**：
1. 第一版方案想用"把 `max_tokens` 调到 0、指望现有空响应检查接住"来堵 Bug B——CLI 用真实 Qwen3-4B-Thinking-2507 tokenizer 实测证伪：即使 assistant 消息完全为空，Thinking 模板本身仍会补上结构性的 `</think>` 闭合标签和 `<|im_end|>` 轮次结束标记（约 5 个 token），现有 `response_text.strip()` 检查过滤不掉这几个 token，这条早退分支永远不会命中——换个更短的残片，问题没堵住。
2. 改成代理侧控制流直接短路：driver 把 `max_tokens: 0` 当专用信号，代理识别到这个信号后**完全不调用 SGLang**，只做该做的事（挂 next_state、触发上一轮评估），然后照抄 `session_done` 清理逻辑直接返回——不依赖生成结果长什么样，也不依赖 SGLang 怎么解释 `max_tokens=0`。

**最终实现（四处改动，均已用合成数据/真实模板文件验证，未在真实训练中跑过）**：
1. **`scripts/prepare_patched_openclaw_opd.sh`** 新增 `openclaw-rl-metaclaw-verdict-signal-skip` 补丁：`_handle_request` 在 messages 校验之后、构造 `forward_body`/POST SGLang 之前，检查 `turn_type=="main" and body.get("max_tokens")==0`，命中就跳过 SGLang 调用，只执行"挂 next_state"逻辑 + 完整的 `session_done` 清理（含 `_seen_user_messages.pop`，跟当前实际清理逻辑逐行对齐）后直接返回。用真实官方文件跑过一遍完整补丁链，`py_compile` 通过，生成代码位置正确（插在 messages 校验和 `forward_body` 构造之间）。
2. **`scripts/prepare_patched_openclaw_combine_select.sh`** 新增 `openclaw-rl-metaclaw-verdict-early-return` 补丁：`_metaclaw_verdict is not None` 分支末尾加显式 `return`（结构照抄旁边 `metaclaw_round_mode`/step-judge 分支：`accepted: False`，`eval_score` 直接用 checker 分），不再往下掉进会崩溃的共用 PRM 出口。**代价（如实记录）**：原设计里"长 hint 走 OPD+RL"这条路径（`votes` 列表构造那段）因为这次改动被放弃，改成明确的 RL-only——但那条路径从 UnboundLocalError 出现以来就没有成功执行过一次，这不是削弱已经生效的功能，是把"从没跑通的死代码"正式确认为"这轮不做"。以后要恢复 OPD 处理，应显式设 `_skip_forced_negative_override = False` 再继续往下走，不能直接撤销这个 `return`。
3. 同一个补丁脚本，`metaclaw_round_mode`（step-judge）分支新增 `openclaw-rl-metaclaw-step-judge-truncation-penalty`：`eval_score = _prm_eval_majority_vote(_step_raw)` 之后检查 `turn_data.get("is_truncated")`，命中强制 `eval_score = -1.0`——跟现有 PRM 分支的 `openclaw-rl-truncation-penalty`（2026-08-06 已确定的"强制 -1、不丢样本"策略）保持一致，不是另选一个更弱的方案（core 补丁 1 从源头消灭 verdict 残片后，这条主要用于兜住其他真实中间轮次的截断，是加固不是主防线）。
4. **`scripts/metaclaw/metaclaw_rollout_driver.py`**：`_send_verdict_turn` 的 `max_tokens` 从 8 改成 0（`_send_session_close_only` 保持 8 不变，本来就是 `X-Turn-Type: side`，走不到新补丁分支）；`_run_round` 新增 `_GENERATE_FAIL_MARKERS` 检测 OpenClaw 自己的"⚠️ Agent couldn't generate a response"兜底文案，**只用于转录标注可见性**，不改变 `agent_succeeded`/打分/verdict 提交的任何逻辑——第 1、2 条修好后，`rc==0` + 兜底文案会自然地被 `_compute_inline_score`/`_score_round_official` 打出真实的低分/失败结果，走正常的 `eval_score=-1.0` verdict 提交通道，不需要新的分类逻辑（早期草稿曾想复用 `agent_succeeded=False` 的基础设施失败通道处理这种情况，被指出是错的：那个通道会把这道题从 Acc. 分母里排除、且不提交 verdict 改由下一轮 step-judge 打分，语义上不对——generate-fail 是"模型没给出可用回复"，不是"网关挂了"）。

**验证方式**：四处改动都用官方源文件跑过完整补丁链验证（`py_compile` 通过、生成代码位置正确），driver 侧新增了一次端到端回归测试（mock `_run_openclaw_agent`，确认普通轮次打分不受影响、verdict payload 正确变成 `max_tokens=0`、close payload 保持 `max_tokens=8` 不变、generate-fail 检测只加转录标注不改变 `agent_succeeded`/`official_score`）。**真实训练环境完全未验证**——下一次训练需要按冒烟清单逐项确认：

1. 代理日志里出现 `openclaw-rl-metaclaw-verdict-signal-skip` 专用日志；verdict 之后**没有**新的 `MAIN ... response_tokens=5`（或 13）
2. `deterministic-reward` 后面是长 `prompt_len` 的正常 RL 提交，不再是 `response_len=13` 这种残片
3. `UnboundLocalError` 次数为 0
4. `_send_session_close_only` 对应的请求仍然是 `X-Turn-Type: side`

缺任何一条就应该停下来，不要继续训练。

### 修复：默认提交训练会静默加载上一次的（可能训坏的）权重（2026-08-19）

CLI 核对 `run_openclaw_topk_select_modelfactory.sh` 发现一个跟今天训练信号污染修复完全独立、但同样会破坏"重新提交训练=真的从头开始"这个前提的问题：**权重续训和按天续跑是两套完全不相关的机制，之前只处理了后者。**

`run_openclaw_topk_select_modelfactory.sh` 给 Megatron 加的 `--load "${SAVE_CKPT}"` 是**无条件**的（这是更早为了让 Personal Agent Track 能在真正崩溃时自动续训加的，见该脚本自己的注释）——目录不存在、或没有 `latest_checkpointed_iteration.txt` 时才会回退到 `--ref-load` 用干净预训练权重。`run_metaclaw_migration_modelfactory.sh` 里 `SAVE_CKPT` 默认值原来是一个**固定路径**（不带时间戳），意味着：只要上一次训练在这个目录存过 checkpoint（哪怕是被训练信号污染训坏的），**下一次提交训练——即使完全不碰 `METACLAW_PROGRESS_DIR`/`METACLAW_RESUME` 这两个按天续跑相关的变量——也会静默把那份权重通过 `--load` 加载进来继续训**，不是从干净的 base Qwen3-4B 开始。这跟用户"实验阶段每次都要重新开始跑"这个明确要求直接冲突，此前"训练侧 checkpoint 本来就有 `--load` 自动续训，不受这次改动影响"这个判断（写在"训练/评测数据重叠"一节的按天续跑设计里）本身没错，但没有意识到这个"不受影响"恰恰是问题所在——权重续训完全没有对齐"每次重新开始"这个更早、更根本的决定。

**修复**：`SAVE_CKPT` 默认值改成带时间戳（跟 `LOGS_DIR` 共用同一个 `RUN_TIMESTAMP` 变量，脚本开头统一生成一次，方便按时间戳对应同一次训练）——不显式设置 `SAVE_CKPT` 提交训练，目录天然不存在，`--load` 自动回退到 `--ref-load` 干净权重，不需要手动删除旧目录或记住换路径。旧 checkpoint 不会被清掉，各自留在自己的时间戳目录下；需要的话仍然可以手动 `SAVE_CKPT=<旧路径>` 显式指定去接着训某一次跑出来的权重，只是不再是"什么都不设"时的默认行为。用 bash 单独验证过：默认解析出的 `SAVE_CKPT`/`LOGS_DIR` 用的是同一个时间戳；显式设置 `SAVE_CKPT` 时覆盖仍然生效，不受这次改动影响。

### 修复：训练暂停期间的 503 被当成基础设施失败，整段天数被空转吃掉（2026-08-19）

**现象**：`metaclaw_migration_20260819_132608`（换了新 `SAVE_CKPT`、修好训练信号污染之后提交的这次）从 `day06` 起，`Acc.` 不再是"答错了"的低分，而是大段"没答题"——CLI 沿时间线比对 `submission paused`、权重同步和 rollout 失败，定位到这不是训练信号问题（跟前一天那两个 bug 不是同一类），而是 **MetaClaw driver 和 slime 换权重时的暂停窗口在抢跑**。

**根因（代码+真实日志核实）**：`openclaw_opd_api_server.py` 的 `submission_enabled` 检查在最前面（鉴权之后、真正处理请求之前）：

```python
if not owner.submission_enabled.is_set():
    raise HTTPException(status_code=503, detail="submission paused for weight update")
```

slime 自己的 rollout 循环攒满一个训练 batch（16 条）就会 `pause_submission()`，直到这一步的 actor train + 存 checkpoint + `update_weights` 全部跑完才 resume——真实观测一次完整暂停窗口是 **4 分 20 秒**（含一次 66.5 秒的 `save_model`），不是瞬间的事。这个窗口内所有打到 30000 端口的请求，包括 `openclaw agent` 自己发的，全部立即收到 503。OpenClaw 自己对此没有退避重试，几乎立刻以 `FailoverError: 503 status code (no body)` 失败退出（`rc=1`）。driver 原来的 `AGENT_RETRY` 循环两次尝试之间**没有任何等待**，就算调大这个值也扛不住几分钟的暂停——一次暂停窗口就能把后面好几天的题目连续判成"基础设施失败"，日志显示 `day06` 整天、`day07` 全天、`day08` 前几题都被这样吃掉。**训练样本本身没有被污染**（暂停期间还在飞的那一轮会被 slime 自己标 `generated_while_paused` 丢弃，不会进 GRPO；503 也不会被当成 -1 提交）——纯粹是"天数被空转，Table 1 式的 30 天 Acc 不能用，前几天的信号还能看"。

**跟前一天训练信号污染的区别**：那次是权重真的被 13-token 残片污染训坏；这次权重是干净的，只是"没来得及答题"。之所以这次才暴露：昨天的 stub 让训练很快出结果，暂停窗口短，driver 来不及扫过整天；这次 stub 修好之后暂停变长，这个原本就存在的设计漏洞才显现出来。

**日志里其实有三类失败，不能一概而论**（这点由 CLI 用真实日志核实指出，避免"一律等待"误伤）：

| 类型 | 例子 | 特征 | 能不能靠"等暂停结束"救 |
|------|------|------|------|
| 503 | `day06` 整天、`day07`、`day08` 开头 | 代理还没开始处理就直接拒绝，约 1-2 秒/题 | 能——这正是吃掉天数的原因 |
| timeout | `day05 r4-r7/r13`："LLM request timed out" | 已经真实生成了很久才死 | 不一定——用同一套长等待去救，风险是把 GPU 争用/超长序列这类真实问题也一起拖成更长的空转 |
| generate-fail | `day08 r11` 等 | `rc==0`，OpenClaw 自己的兜底文案 | 不该走这条——已经在前一次修复里确认走正常打分通道，不受这次改动影响 |

**修复**：只针对 503 这一种失败模式加一个独立、耐心的等待重试环，其余失败模式（timeout、崩溃、别的 `FailoverError`）维持现在"立刻放弃、尊重 `AGENT_RETRY`/`VERDICT_RETRY`（默认 0，无等待）"的行为不变：

- **`_run_round`**（`openclaw agent` 子进程调用）：每次失败先检查 `stderr` 里有没有 `"503 status code"` 这个特征串（OpenClaw 自己 `FailoverError` 的确切文案）。命中就进入独立等待环：`sleep METACLAW_PAUSE_RETRY_INTERVAL`（默认 15 秒）后重试同一轮，不计入 `AGENT_RETRY` 的次数预算；累计等待时间（用 `time.monotonic()` 测的墙钟时间，从第一次命中 503 算起，不是单纯把 sleep 加总——每次失败的 agent 启动本身也要 1-2 秒，这部分也要算进预算）超过 `METACLAW_PAUSE_RETRY_MAX_WAIT`（默认 900 秒＝15 分钟，比观测到的 4 分 20 秒留了将近 4 倍余量，考虑到序列更长、又可能撞上存盘会更久）才真正判定为基础设施失败。没命中 503 特征串的失败（timeout 等）完全不进这个等待环，直接走原来的 `AGENT_RETRY` 逻辑，不受影响。
- **`_post_with_retry`**（verdict/close 提交）：同样的独立等待环，但检测方式更精确——这条路径直接用 `httpx`，503 会变成 `HTTPStatusError`，直接读 `e.response.status_code == 503` 判断，不用像 agent 侧那样匹配文本（那边文案是 `"503 Service Unavailable"`，跟 OpenClaw 自己的 `"503 status code"` 不是同一段文字，两边分开检测是对的，不能共用一个字符串常量）。等满预算仍然只是打 log、不向上抛异常（维持原有"记录后放行"的设计），但日志明确写成"pause-retry exhausted"，跟普通的"submission failed"分开，事后翻日志能一眼分清是哪种失败。close 提交如果撞上 503 也一样会进等待环——close 本身很便宜（不用重新生成任何内容），等的只是暂停结束，这么处理没问题。

暂时不做的：给 `/healthz` 加字段暴露 `submission_enabled` 状态、让 driver 精确轮询"是否已恢复"（比字符串/状态码匹配更精确，但要多改一处代理代码）。这次 503 的成因单一（就是 `submission_enabled` 暂停），现有检测已经足够可靠，没有需要区分的"第二种 503"。如果以后真出现"等满 15 分钟仍然 503"的情况，再考虑加这一层。

**验证方式**：新增 5 项合成数据回归测试，覆盖：`_run_round` 遇 503 重试后成功、遇 503 耗尽预算后正确放弃（用极短的等待参数验证墙钟计时逻辑本身是对的，不是靠真的等 15 分钟）、timeout 类失败完全不进等待环（零等待、只尝试一次，验证跟 503 处理是互斥的两条路径）；`_post_with_retry` 遇 503 重试后成功、遇 503 耗尽预算后正确放弃且不抛异常。全部通过，真实训练环境（真实的 4 分钟量级暂停窗口）尚未验证。

### 修复：checker 算出的 OPD hint 被无条件丢弃，file_check/多选题 verdict 轮次从未真正走过 OPD 蒸馏（2026-08-19b）

**背景**：CLI 对着 `metaclaw_migration_20260819_153518` 的日志排查"越写越长、`write` 调用消失"这个模式，核对 driver 里 `file_check` 的判分和 OPD hint 构造，发现上一条（"训练信号根因分析与修复"）里加的 `openclaw-rl-metaclaw-verdict-early-return` 有一个没写完的地方：那次修复的目的是堵住 `UnboundLocalError` 崩溃，返回值写成固定的 `accepted: False, hint: ""`，但 `_metaclaw_hint`（`_send_verdict_turn` 携带过来的、由 driver 侧 `_build_opd_hint` 算出的真实失败原因）在这之前已经算出来了，只打进日志的 `hint_len` 字段，从没真正被使用——多选题和 `file_check` 的 verdict 轮次自此只有 checker ±1 的 GRPO 信号，OPD hint 蒸馏这条路径实际上从未生效过。

**两处独立问题，分两层修，且顺序不能反**（Layer 1 必须先于 Layer 2，否则会把不可信的 hint 也喂给 OPD，比现在什么都不做还糟）：

**Layer 1（driver 侧，`_build_opd_hint`，只影响 `file_check`）**：原来 checker exit 1 且 stdout/stderr 都是空字符串时，退回题面写死的静态 `feedback.incorrect` 文案——函数自己的 docstring 早就写明白这条退路危险（"用静态文案当 hint，如果真实失败原因不是它描述的那种，会把蒸馏目标指向错误的纠正方向，主动污染训练"），但这条退路当时仍然保留着。`metaclaw_migration_20260819_153518` 的 day01 r5 就是实锤：`python -c` 找不到 T-405 静默失败（stdout/stderr 皆空），hint 变成"due_date 格式应该是 2026-03-18T18:00:00+08:00"，模型可能根本没碰这个 task。

**修复**：stdout/stderr 都空时直接返回 `""`，不再退回静态文案。CLI 用 `20260819_153518` 这趟真实数据核对过覆盖率影响：约 55 道失败的 `file_check` 题里，48 道 checker 有 `FAIL: ...` 这类 stdout、6 道是 `python -c` 的 traceback（留在 stderr，改动前后都会被保留，不受影响）、只有 1 道（day01 r5）落进"两边都空"这个分支——改完之后"猜错"的口子基本堵上，代价很小（1/55 变成无 hint，GRPO 的 -1 照常打，只是不再有 OPD）。`_build_opd_hint` 的 `len(hint) <= 10` 材料化门槛（Layer 2 复用父类同一标准）也核对过不会误杀——这趟数据里最短的真实 hint 是 28 字符。

**Layer 2（proxy 侧，`prepare_patched_openclaw_combine_select.sh`，`_metaclaw_verdict is not None` 分支）**：`_metaclaw_hint` 非空且长度 > 10 时，不再走固定的 `accepted: False` 返回，而是照抄父类 `OpenClaw-Combine-Select` 的 `accepted=True` 候选材料化代码——`_append_hint_to_messages(turn_data["messages"], _metaclaw_hint)` → `_normalize_messages_for_template` → `self.tokenizer.apply_chat_template(..., tools=turn_data.get("tools"))` → 拼接 `response_text` → tokenizer 编码成 `enhanced_ids`——返回 `accepted=True, teacher_tokens_candidates=[enhanced_ids], hints=[_metaclaw_hint]`，`eval_score` 仍然是 checker 的 ±1（不受影响，走的是 `OPD+RL` 而不是纯 `OPD`）。`_metaclaw_hint` 为空时（passed=True，或者 Layer 1 主动压掉了不可信的静态文案）维持原有 `accepted: False` 不变。中间轮次的 step-judge 分支（`metaclaw_round_mode`）不动——那条设计上没有具体纠正对象，继续 RL-only 是对的。

这段材料化代码用 `try/except` 包住：`apply_chat_template`/tokenizer 这条路径此前从没有在 MetaClaw 的 tool 消息结构上真正跑出 `accepted=True` 过（一直是先崩溃、后来是固定 `accepted=False`），是没有历史数据验证过的新代码路径，一旦真的因为消息结构不兼容而抛异常，退回现有的安全 `accepted=False` 返回，不让整条样本因为模板报错而丢失（不是必须项，但成本低、能避免"模板一炸整条样本消失"这个新引入的风险）。

**调度逻辑核实（CLI 提出，已核实不需要改动）**：官方原始 `openclaw_opd_api_server.py` 的 `_maybe_submit_ready_samples` 确实是"`accepted=False` 直接 `continue`，样本整条丢弃"（第 863-865 行），但 MetaClaw 走的是 `Combine → Combine-Select` 继承链，`openclaw_combine_api_server.py`（本项目早先给 Personal Agent Track 打的补丁，非本次新增）的调度是三路：`accepted 且有效 RL` 走 `OPD+RL`、`仅 accepted` 走纯 `OPD`、`仅有效 RL`（`accepted=False` 但 `eval_score` 非空）走 `_submit_rl_turn_sample`——这条 RL-only 路径本来就存在且一直在用（Personal Agent Track 的普通 PRM ±1 走的就是它），这次改动只碰 `_opd_evaluate` 的返回值，不涉及、也不需要碰这段调度代码。唯一真会导致整条样本丢失的地方是 `Combine` 里 `task.result()` 抛未捕获异常时的 `continue`（跟 `accepted` 是否为 `True/False` 无关）——这正是上面 try/except 防的那个场景。

**验证方式**：`metaclaw_rollout_driver.py` 的 `_build_opd_hint` 改动通过 `py_compile` 验证。`prepare_patched_openclaw_combine_select.sh` 的改动跑了完整补丁链对真实官方 `openclaw_combine_select_api_server.py` 生成输出，`py_compile` 通过，人工读取生成代码确认缩进、`try/except/else`、`if _metaclaw_hint and len(...) > 10:` 分支结构、两个 `return` 的位置都正确。**真实训练环境完全未验证**——下一轮训练需要在日志里找一条 `FAIL: cannot read` 类和一条选择题错题，确认后面跟着 `[openclaw-rl-metaclaw-verdict-opd-hint] session=... accepted K_i=1 hint_len=...`（不能只有 `submitted RL sample` 那行），且没有因为 `apply_chat_template` 在 MetaClaw 的 tool 消息结构上出错而卡住；stdout 全空的 `file_check` 失败题应仍然只是 RL-only。

**明确没有一起做的**：CLI 同一轮诊断还指出"越写越长"这个模式另有两个更致命的独立成因——(1) 下一轮 `_build_feedback_text` 返回的静态反馈文案本身可能文不对题（跟这次改的 verdict hint 是两套完全不同的代码路径，`_build_feedback_text` 影响的是下一题看到的 `[Previous Feedback]`，不是训练用的 teacher 信号）；(2) 中间轮次的 step-judge 对"没有实际调用 write/edit 的长分析"经常打 +1（`20260819_153518` 数据：45 次里 39 次，约 75-100%），相当于在奖励"堆字数而不落盘"这个行为。这两处都还没有设计成方案，只有诊断，跟这次两层改动不是同一严谨程度，决定不捆在一起改——先看这次 OPD 接线修复单独生效的效果，同时用下一轮训练的干净数据验证 CLI 这两个新假设，再分别单独设计方案。

### 改动：session 拆分——从"一天一个 session"改成"每题一个 session"，切断把整天拖垮的上下文堆积（2026-08-19c）

**背景**：CLI 排查"OPD 接线修复也救不了"的那部分越写越长问题时（见上一节"明确没有一起做的"第 (1)(2) 点），继续深挖出第三个独立成因，且是三个里对 Acc. 影响最直接的一个：`metaclaw_migration_20260819_153518`/`173654` 两次训练日志显示，`day01-06` 选择题正确率一直稳定在 85-97%（跟基线同一档，权重没训坏），但从 `day07` 起，一旦某道 `file_check` 题写出一篇几千到近两万字的长文（`day07 r5` 约 1.7 万字），**同一天后面所有题目**（不管是 `file_check` 还是本来答得好好的选择题）就会一起开始 Context overflow/空回复——`day07 r6` 起、`day08 r10` 起、`day09 r8` 起、`day10 r10` 起，均由 CLI 核对真实日志确认是同一个模式：选择题不是训坏了，是被同一天前面题目的长文本拖进了 overflow。

**根因**：这次迁移的设计一直是"一天一个 proxy session"（`session_id = f"metaclaw-{test_id}"`，`run_day` 里对全天所有 round 只调用一次 `_prepare_session`），刻意对齐 MetaClaw 官方自己的评测代码 `infer_cmd.py::_run_group`（`original_session_id` 对一天里所有 round 复用同一个值）。这意味着当天所有 round 的完整对话历史（包括某道题写崩的一万多字）会原样累积进同一份 transcript，作为下一题 `openclaw agent --session-id` 调用的上下文——一天前几题写崩，后面所有题目（不管类型）陪葬。

**读代码确认的关键事实**（不是假设，逐条核实过）：
1. `_prepare_session`（`MetaClaw-official/benchmark/src/infer/infer_cmd.py:423`）只是在 `work_openclaw_state_dir/agents/{agent_id}/sessions/{session_id}.jsonl` touch 一个文件（不存在才建）——换一个新 `session_id` 就是换一份全新空 transcript，纯文件级操作，跟 workspace/gateway 完全无关。
2. `run_day` 里 workspace（`_prepare_work_copy`/`_copy_workspace_for_test`/`_patch_agent_workspace`/`_ensure_plugins_allowlisted`）和 gateway（`_start_work_gateway`/`gateway_port`）全部是 day 级资源，一个参数都不依赖 session_id，天然可以在"当天 workspace 不变"的前提下单独拆分 session。
3. `openclaw_opd_api_server.py::_maybe_submit_ready_samples` 里 `_pending_turn_data`/`_prm_tasks` 是按 `session_id` 分桶的字典——新 session_id 对代理来说天然是一张白纸，proxy 侧不需要任何改动。
4. `[Previous Feedback]`（`_build_feedback_text`/`with_feedback`）是纯文本拼接，靠 `prev_inline_score`/`prev_round_record` 传递，不依赖共享对话历史——"跨题知道上一题哪里错了"这条路径本来就不靠 session 共享维持，不受这次改动影响。
5. MetaClaw 官方"训练模式"代码（`metaclaw/openclaw_env_rollout.py::run_task_episode`）用的是完全不同的模型：`session_id = f"env-{task_id}-{uuid4}"`，每个 task 独立 session，自己直接控制生成循环、走 shell 命令，根本没有 `day/group/round/[Previous Feedback]/checker` 这套结构——这次迁移本来就没有对齐这份代码（"三方对照"表已记录），所以"要不要按题拆 session"这件事上不存在 MetaClaw 训练侧的先例可以参照，是这次迁移自己的架构决定，不是照抄谁。

**CLI 额外核实的三件事**（读真实数据/官方源码，不是猜）：
1. **全部 30 天 346 道题的 `round_record["id"]` 扫描**：清一色 `r1`...`r15`，纯字母数字，没有空格/斜杠这类不安全字符；`preference_tags` 只在 `all_tests.json` 的 day 级出现，round 级没有；每天恰好 1 个 group，且 `group["id"] == test_id`（跟 `day07/questions.json` 单独核实的结果一致）。
2. **`openclaw agent --session-id` 是不是真隔离**：OpenClaw 官方文档确认换 id 就是换会话；CLI 把 id 解析成 `agent:{agentId}:explicit:{sessionId}`，transcript 路径正是 `_prepare_session` touch 的那个 `.jsonl`；`20260819_153518` 训练本身已经反向证明了这条 jsonl 就是在涨的那份上下文（同一天越问越长直到 overflow）。残留风险很小且不是 blocker：OpenClaw 的 `tools.sessions.visibility` 理论上允许 `sessions_list` 读到同 agent 目录下其它 session 的 jsonl，但 `153518` 这次训练实测工具只有 `read`/`write`/`edit`，`sessions_*` 调用 0 次；训练侧 plugins allowlist 只有 `rl-training-headers`，没有任何记忆类插件。
3. **overflow 真实模式不只 day07 一个案例**：`day07`（r5 写崩→r6 起全 overflow，含 2 道选择题）、`day08`（r1-7 已经很长，r8-9 generate-fail，r10 起 overflow，含末尾选择题）、`day09`（r1-3 超长→r5 generate-fail→r8 起 overflow，含 r10 选择题）、`day10`（r4-5 约 2 万字→r6 generate-fail→r10 起 overflow，含 2 道选择题）——模式一致：前面几道写崩的 `file_check` 把 transcript 撑满，后面不论题型一起死。

**修复（只改 `metaclaw_rollout_driver.py`，不碰 proxy 补丁/workspace/gateway 逻辑）**：
- day 级的 `session_id = f"{_SESSION_ID_PREFIX}{test_id}"` 和它对应的单次 `_prepare_session(...)` 调用整个删除，改成 round 循环内部为每道题现算：`round_session_id = f"{_SESSION_ID_PREFIX}{test_id}-{group.get('id','unknown')}-{round_record['id']}"`，紧接着调用 `_prepare_session(work_openclaw_state_dir, agent_id, round_session_id)`。`{group_id}-` 这段在真实数据里是冗余的（`group["id"]` 恒等于 `test_id`），但保留作为对 `EvalFlowQueryReader` legacy 格式（理论上一天可以有多个 group）的防御，成本为零。
- `_run_round`/`_send_verdict_turn`/`_send_session_close_only` 三处全部改用同一个 `round_session_id`，不再是共享的 day 级 `session_id`。
- `_send_verdict_turn` 的 `session_done` 从 `is_last_round`（只有当天最后一题才收尾）改成**每题无条件 `True`**——每题现在都是完整独立的 session，必须每次收尾，不能只等到当天最后一题。
- `agent_succeeded=False` 分支的 `_send_session_close_only` 同理，从 `if is_last_round:` 改成**每次失败都无条件调用**——不这样做的话，非当天最后一题的失败会把挂起轮次留在一个再也不会收到任何后续消息的 session 里，永远等不到清理，比原来的行为更糟（原来好歹能被当天下一题的 next_state 误判掉，现在的 session 拆分下这条误判路径也不存在了）。
- `is_last_round` 这个局部变量随之整个删除（原来只在这两处用到）。

**连带效果**：这个改动顺手把 08-17 记录的"跨 round 污染"这个此前搁置的老问题也一并结构性解决了（`docs/metaclaw_migration_plan.md`"下一步工程任务"第 1 项）——那个 bug 的前提是"崩溃 round 的挂起轮次被同一 session 的下一条消息误判成它的 next_state"，现在每个 round 都是独立 session、且每次都无条件发送 `session_done`，这个前提不再成立，不需要再单独设计修复。

**预期效果（CLI 给出，如实记录，不夸大）**：
- 会：`day07-10` 那种"选择题被同一天前面的长文拖垮"的模式应基本消失，Acc. 里的选择题部分能重新反映模型真实能力。
- 不会：`Compl.` 不会因此变好——单题依然可能写出一万字、依然可能打 0 分，这个改动只是让它不再拖累同一天后面所有题目，"file_check 学不会写文件"这件事仍然要靠 OPD 蒸馏/奖励设计解决，是另一件事。

**跟 MetaClaw 官方评测代码的分歧，明确记账**：MetaClaw 自己的 `_run_group` 是"一天一个 session"，这次改成"每题一个 session"是主动偏离——可以接受的理由是：MetaClaw 自己的确定性打分（`_compute_inline_score`）从不读 transcript，只读 workspace 文件/agent 最终回答，这次改动没有改变 checker 打分口径；而 MetaClaw 自己的训练侧代码（`openclaw_env_rollout.py`）本来就没有跟这次迁移的 day/round/feedback 结构对齐过，不存在"这次改动破坏了跟官方训练方法一致性"这层顾虑。这是训练 driver 自己的架构选择，不是无意识偏离。

**验证方式**：`py_compile` 通过。**真实训练环境完全未验证**——CLI 明确要求不要热补正在跑的 `153518`/`173654`，这次改动落地后由用户决定何时提交新训练。下一轮训练需要确认：`day07-10` 这类"从某题起到收工全是 Context overflow"的模式基本消失（单题仍可能很长/仍可能 0 分，但不该再连坐后面的题）；如果这次跟 Layer 1/2（OPD hint 接线修复）同一轮训练一起验证，Acc. 提升需要分开看归因——"选择题不再被 overflow 拖累"和"OPD 让 file_check 真的变好"是两件不同的事，不能笼统算作同一个改动的效果。

### 修复：`openclaw agent` 从未传 `--agent`，`write` 实际写进了 checker 看不到的默认 agent workspace（2026-08-19d）

**背景**：CLI 排查 `Compl.` 为什么一直是 0 时发现，模型经常真的在写文件（session transcript 里能看到 `Successfully wrote N bytes to day05/xxx.json` 这类真实成功回执），但 checker 评测的目录里那些文件从来不存在，stdout 永远是 `FAIL: cannot read day05/xxx.json`。核对真实文件位置：写入的文件在 `openclaw_state_.../workspace-main/day05/`，checker 读的是 `work/workspace_day05_.../day05/`——两个完全不同的目录。`openclaw.json` 里 `metaclaw_agent.workspace` 确实已经被 `_patch_agent_workspace` 改成了后者，但 `openclaw agent` 子进程实际用的 session key 是 `agent:main:explicit:metaclaw-day05`——**跑的根本不是 `metaclaw_agent` 这个被 patch 过的 agent，是 OpenClaw 自带的默认 `main` agent**，它的 workspace 是一个完全独立的、写死在 OpenClaw CLI 自己代码里的默认目录，跟 `openclaw.json` 里配的任何东西无关。

**根因（沿着 OpenClaw CLI 自己的源码逐层追出来的，不是猜）**：`MetaClaw-official/benchmark/src/infer/infer_cmd.py::_run_openclaw_agent` 构造 `openclaw agent --session-id <id> --message <text>` 子进程调用时，函数签名里有 `agent_id: str | None = None` 这个参数，但函数体从未把它拼进命令行；它唯一的官方调用方 `_run_question` 也从未把 `agent_id` 传给它——**这个缺口就存在于 MetaClaw 官方代码本身，我们的 driver 是直接 import 这个函数复用的，缺口原样带了进来**。

不传 `--agent` 时，OpenClaw CLI 自己的 session-key 解析（`agentViaGatewayCommand` → `resolveSessionKeyForRequest`，`src/agents/command/session.ts`）内部其实**先算对了一次**：`defaultAgentId = normalizeAgentId(resolveDefaultAgentId(cfg))`——因为我们的 `openclaw.json` 只配了一个 agent（`metaclaw_agent`），`resolveDefaultAgentId` 会正确返回 `"metaclaw_agent"`。但这个正确结果没有被用上：新的 `--session-id` 第一次出现时，在所有 agent 的 session store 里都找不到匹配，代码落到一个兜底分支——

```js
if (requestedSessionId && !sessionKey) sessionKey = buildExplicitSessionIdSessionKey({
    sessionId: requestedSessionId,
    agentId: opts.agentId,   // 用的是原始参数（undefined），不是上面已经算对的 defaultAgentId
});
```

`opts.agentId` 是 `undefined`（没传 `--agent`），`normalizeAgentId(undefined)` 硬编码返回字面量 `"main"`（`routing/session-key.ts`）——`defaultAgentId` 那次正确的计算结果被完全绕过。这是 OpenClaw 自己 session-key 解析代码里的一处真实不一致，不是 MetaClaw 或我们哪里少配置了什么。CLI 用本机实际安装的 OpenClaw 编译产物（`node_modules/openclaw/dist/session-*.js`）核对过这段逻辑，跟源码仓库对得上。

**"官方 Compl. 不为 0，应该是做过调整的吧"——查证结果：查不到确凿的规避机制，如实记录，不卡修复**：搜了整个仓库唯一显式传 `--agent` 的地方是 `metaclaw/utils.py::run_turn(..., "--agent", "main")`，那是另一条独立的调用路径（跟 `infer_cmd.py` 这套 bench 评测/训练脚本无关），解释不了 bench 的 `Compl.`；`_register_session_in_json` 只在题目自带的 `"update"` 字段（`type: session, action: new`）下才会触发，不是每题都走，也不构成通用的规避机制。真正的原因目前不确定，候选是：官方论文用的 OpenClaw CLI 版本可能是更老的版本、这段兜底当年行为不同（跟 `CLAUDE.md` 记录的三月/五月版本不确定性是同一类问题，但这次可能是 OpenClaw 自身代码的真实版本差异，不是新功能有无的问题）；或者官方真正跑评测的外层脚本（不在这个文件里）在别处传了等价参数。**这两条都没有确凿证据，留作开放问题，不阻塞这次修复**——不管官方当年为什么侥幸没事，显式传 `--agent` 都是绕开这整条不确定性、直接给出正确结果的做法，不依赖猜中官方的具体机制。

**影响范围**：这个缺口不是这次迁移新引入的，从 driver 第一次跑起来那天就存在，意味着**至今为止每一次 MetaClaw 迁移训练/基线跑的 `Compl.`＝0.0% 都可能主要是这个原因造成的，不是 checker 真的没找到文件，是文件从一开始就没写到 checker 会看的地方**。修完也不会让 `Compl.` 突然变成论文级别的数字——checker 能看见文件了，但文件内容本身对不对（比如日期是不是训练当天而不是三月的题面日期）、`generate-fail`、同一天 session 污染这些此前记录过的问题都还在，只是不再被"写到隔壁目录"这一条锁死成 0。

**修复**：给 `openclaw agent` 子进程调用显式加 `--agent {agent_id}`。**没有采用"整份拷贝 `infer_cmd.py` 再打补丁"（`prepare_patched_*.sh` 那套给 `OpenClaw-RL-official` 文件用的模式）**——CLI 指出这次不适用同一模式：那套是给会被 slime/proxy 直接 import 的模块用的，只能靠拷贝改 `PYTHONPATH`；而这里训练路径只通过 driver 调用 `_run_openclaw_agent` 这一个函数，`_run_group`/`_run_question` 那套官方评测编排代码根本没被用到，为了两个 argv 参数去拷贝一份 1400 行的文件、还要伪造 `src/infer/` 包结构、改 `sys.path`，过重也容易和真正的官方模块混着加载。改法是**在 `metaclaw_rollout_driver.py` 里本地复制这一个函数**（~40 行，跟官方版本逐行一致，只加了一行 `"--agent", agent_id,`），不 import 官方版本，不碰 `MetaClaw-official/` 任何文件——这也是这次迁移一直以来对 MetaClaw 官方代码缺口的处理惯例（比如 `OPENCLAW_GATEWAY_TOKEN` 也是写进 driver 自己的 `os.environ`，没有改官方源码）。

`agent_id` 在本地版本里是**必填参数**（官方签名是 `agent_id: str | None = None`，这里故意不给默认值）——训练路径上如果哪天不小心没传 `agent_id`，应该直接报错，而不是静默退回"不传 `--agent`"、重新掉进这个 bug。`run_day` 里 `agent_id = test["agent"]`（每天都是 `"metaclaw_agent"`）本来就已经喂给 `_patch_agent_workspace`/`_prepare_session`，这次只是把 `_run_round`/`_run_openclaw_agent` 也接上同一个变量，不是引入新概念。

**明确没有做的**：MetaClaw-Bench 自己的离线评测路径（`infer_cmd.py::_run_question`/`_run_group`，用于比如训练前后单独跑一次 `metaclaw-bench run` 拿 Compl. 对比，不经过这个 driver）目前仍然没有这个修复——那条路径这次完全没用到，等真的需要用它的时候再单独处理，不在这次范围内提前修。

**验证方式**：`py_compile` 通过。新增一个合成测试（mock `asyncio.create_subprocess_exec`，断言最终 argv 精确等于 `("openclaw", "agent", "--session-id", <id>, "--agent", "metaclaw_agent", "--message", <msg>)`）确认参数顺序和内容都对。**真实训练环境完全未验证**——下一轮训练需要确认：真实 session key 变成 `agent:metaclaw_agent:explicit:...`（不再是 `agent:main:explicit:...`）、`write` 之后的文件确实出现在 `workspace_{test_id}_*` 而不是 `workspace-main/`、checker `stdout` 不再是清一色的 `FAIL: cannot read ...`。这轮同样不热补正在跑的 job，何时提交新训练由用户决定。

### 方案：可调 K 天训练窗口 + 冻结评测剩余天数（2026-08-20）

**背景与定位——这是一个额外的、纯附加的能力，不是对现有训练逻辑的修改**。用户提出这个需求时，`metaclaw_migration_20260820_*` 这轮训练已经跑到 day12、效果不错（Acc./Compl. 趋势正常），这次改动**完全不能影响这条正在跑的训练**——默认（不设置开关）情况下必须和现在这条正在跑的训练行为逐字节一致，这是本次改动能不能合入的硬性前提，不是"尽量做到"。用户明确会在改完之后另开一次默认配置的训练，跟当前这条正在跑的训练做直接对比，验证"改了代码但没打开开关"这件事本身没有引入任何行为差异。

**动机**：论文 Full 档"边训边考"方法学下，Acc./Compl. 是一个混合了训练全程不同权重状态的滚动平均数——前几天接近 base，后几天用的是训到一半的权重，一个数字里混了多种能力水平，没法直接回答"训练到底有没有提高能力"。CLI 提出的方案：把 30 天拆成两段，前 K 天正常训练+边跑边考，dayK+1 起权重冻结、但继续用同一套 harness 真实跑完剩余天数——冻结段的 Acc./Compl. 才是能跟"训练前"直接对比的干净数字。

**核心设计前提（已用真实代码核实，不是假设）**：这次迁移的 Acc./Compl. 计算完全在 `_run_round` 本地完成（`_compute_inline_score`/`_score_round_official`），跟这个回合的对话样本有没有被提交去训练完全无关。这意味着"冻结训练但继续跑完 30 天"不需要另起一套评测哈内斯——`run_day`/`_run_round`/`_send_verdict_turn` 全部保持不变，唯一需要变化的是**代理侧收到 verdict 之后要不要把这个样本放进训练队列**。

**开关**：`METACLAW_TRAIN_UNTIL_DAY`。**未设置（默认）＝完全不启用**——`TRAIN_UNTIL_DAY: int | None = int(...) if 非空 else None`，用"环境变量是否非空"本身做开关，不是拿一个很大的数字当默认值假装禁用。天数用 `test_list` 的 1-based 下标（`day_index = idx + 1`），不解析 `test_id` 字符串，跟已有的 `METACLAW_MAX_DAYS` 是同一个"数第几个 test"的思路，两者正交。K=0 合法（day1 起就冻结，等价于用这套 harness 跑一次纯 base 模型的 30 天评测）。

**driver 侧改动**：`main()` 的天数循环里，`day_index > TRAIN_UNTIL_DAY` 的每一天，在调用 `run_day(...)` 之前都发一次冻结信号（新增 `_send_freeze_signal`，复用 `_post_with_retry`）——不是只在跨过阈值那一刻发一次，是每个冻结天都发，防止某一次网络调用失败导致后面整段静默没冻结成功（幂等，代理侧重复设置 `True` 没有副作用）。`run_day`/`_run_round`/`_send_verdict_turn`/`_send_session_close_only` 一行都没有改。

**代理侧改动，两处，位置是这次实现的关键**：

1. **冻结信号识别位置：`prepare_patched_openclaw_opd.sh` 打在 `chat_completions` 这个 FastAPI 路由函数本身，不是 `_handle_request`**——第一版方案写的是"`_handle_request` 里：鉴权 → `submission_enabled` → …"，跟真实代码顺序不符，被 CLI 用真实日志（`logs/metaclaw_migration_20260820_094611/patched-openclaw-opd/openclaw_opd_api_server.py` 601-609 行）指出来：`submission_enabled` 的 503 检查发生在 `chat_completions` 路由函数里，在调用 `_handle_request` **之前**，如果把冻结识别塞进 `_handle_request`，冻结信号本身在 dayK 末尾撞上权重同步暂停窗口时仍然会先吃一次 503，等于这处修正没真正落地。正确位置是 `chat_completions` 里 `await owner._check_auth(authorization)` 之后、`if not owner.submission_enabled.is_set():` 之前——识别到专用 header `X-Metaclaw-Freeze-Training: true` 就置位 `owner._metaclaw_training_frozen = True` 并立刻返回，完全绕开这道 503 闸（这是一条控制面消息，不是训练数据提交，不该被"训练暂停中"卡住）。新增的路由参数 `x_metaclaw_freeze_training: str | None = Header(default=None)` 跟现有 `x_session_id`/`x_turn_type`/`x_session_done` 是同一种 FastAPI 参数写法。这个早退发生在 `await request.json()` 之前，driver 侧发送的 body 可以是任意占位内容（实现用的是 `{}`），不需要伪装成真实 chat-completions 形状。`self._metaclaw_training_frozen` 在 `__init__` 里显式初始化成 `False`（挂在 `self._thread`/`self.app = self._build_app()` 之间）。

2. **实际拦截点：`prepare_patched_openclaw_combine.sh` 的 `_maybe_submit_ready_samples`，不是 OPD 自己、也不是 combine-select**——这是这个文件本来就存在的理由（见该文件顶部注释）：真实训练 import 的是 `OpenClawCombineSelectAPIServer`，继承自 `OpenClawCombineAPIServer`，只覆写 `_opd_evaluate()`，不覆写 `_maybe_submit_ready_samples()`——这个函数才是唯一真正调用 `_submit_turn_sample`/`_submit_rl_turn_sample` 的地方，同时覆盖 OPD 和 RL-only 两条提交路径。标志位加在 `openclaw-rl-skip-forced-negative-override` 那个既有拦截点之后、`eval_score = opd_result.get("eval_score")` 之前，跟现有 `is_aborted`/`generated_while_paused`/`is_duplicate_user_retry`/`skip_forced_negative_override` 是同一个拦截模式的延伸：`if getattr(self, "_metaclaw_training_frozen", False): ...continue`（`getattr` 兜底是双保险，即使哪天 `__init__` 的初始化因为某种原因没跑到，也不会因为属性不存在而报错）。`openclaw_combine_select_api_server.py` **不需要改动**——冻结检查在比 `_opd_evaluate` 更下游的分发层，`_opd_evaluate` 算出什么结果都会被同一道闸拦住。

**dayK 尾部竞态：接受，不做 drain，写清楚**。`run_day(dayK)` 返回时，代理里可能还有几个还没 `done()` 的 step-judge 异步任务；dayK+1 一发冻结信号，这几个本该算进 Train window 的样本可能被连带丢弃。跟 CLI 讨论后采纳"接受，文档写清楚"这个选项，不做等待清空、也不做按 session_id 解析"这题属于哪天"这类更精确但更重的方案——失败模式是"dayK 尾部少丢几个样本"，不是"训错"，影响面有限且早就在设计阶段就明确过。

**报告拆分，仅在设置了 `METACLAW_TRAIN_UNTIL_DAY` 时才出现**：`main()` 循环时把每天的 `official_score` 同时归入 `train_round_scores`/`frozen_round_scores` 两个桶（按 `is_frozen_day` 分流，`resume` 命中的天也一样归桶，résumé 和冻结是两个独立维度）。`report.json` 只有在 `TRAIN_UNTIL_DAY is not None` 时才新增 `metaclaw_train_until_day`/`metaclaw_train_window`/`metaclaw_frozen_window` 三个字段（各自复用 `_build_report`），**未设置时 `report.json`/`report.md` 的既有字段一个都不变**——这条本身也是"未设置=完全一致"这个前提的一部分，不能只验证训练行为一致、报告格式却悄悄多了字段。`report.md` 额外追加一段 Train/Frozen 对照表，明确标注"Train window 只是过程监控，Frozen window 才是回答'训练有没有用'的数字"。

**验证方式**：`py_compile` 通过；三个代理补丁脚本（`prepare_patched_openclaw_opd.sh`/`prepare_patched_openclaw_combine.sh`/`prepare_patched_openclaw_combine_select.sh`）依次跑完整补丁链，对真实官方源文件生成输出，全部 `py_compile` 通过，人工核对冻结检查在 `chat_completions` 里的位置确实在 `submission_enabled` 判断之前。新增合成测试覆盖：环境变量未设置/设为 `"0"`/设为 `"5"` 时 `TRAIN_UNTIL_DAY` 解析结果正确（`"0"` 不会被误判成"禁用"）；`is_frozen_day` 判断公式在多组 `(K, day_index)` 组合下结果正确；`_send_freeze_signal` 发出的请求 header/body 形状正确。**没有做端到端的"默认配置逐字节对比"合成测试**（`main()` 依赖真实 `openclaw agent`/代理/checker，本地无法完整跑通）——这个验证交给用户接下来另开的一次默认配置训练去跟当前 `day12` 这条正在跑的训练直接对比，代码层面能提供的保证是：新增的所有分支都由 `TRAIN_UNTIL_DAY is not None`（driver 侧）或 `owner._metaclaw_training_frozen`（代理侧，只有收到过冻结 header 才会变 `True`）这两个条件严格把门，未触发这两个条件时执行路径与改动前完全相同。**真实训练环境完全未验证**——不热改正在跑的 `day12` 这条训练，这次改动只影响未来新提交的训练任务。

### 方案：next-round 反馈 + FORMAT_ERROR + is_invalid_tool_use 三处修复（2026-08-20b）

**背景**：CLI 排查 `metaclaw_migration_20260820_094611`（全量 30 天训练）时发现"超长 thinking 空转"这个老问题在 day12-14 再次出现，且是**难度阶梯**触发的——day01-09 正常（thinking 5-9k 字），day10-11 先出现"只说不做、不调工具"，day12-14 才真正陷入 thinking 循环（7 万→22 万字，`finish_reason=length` 过半，字面上把"每个部分都满足用户要求"这类话复读几百到近两千次）。这条链路跟 Personal Agent Track 08-07~08-10 那次"超长 thinking 空转"是同一个机制（同一套刻板反馈反复灌 → 模型放弃用工具 → thinking 里空转），只是换了个触发场景：P2 阶段引入的文件命名规范题反复失败，同一份高度雷同的静态反馈（21 种骨架，18/39 同句式）逐天再灌一遍。

这次讨论经过多轮 CLI 用真实数据核实（含专门确认"改动不能影响 P1 阶段现在跑得不错的部分"这个约束），最终定稿三处独立修复，均已实现：

**修复 1：next-round 反馈追加真实 checker stdout（file_check 失败）**——`_build_feedback_text`（MetaClaw 官方函数，next-round 反馈和 `_build_opd_hint` 的训练 hint 都调用它）对 file_check 只读题面写死的 `feedback.incorrect`，从不读 checker 真实 stdout。新增 `metaclaw_rollout_driver.py::_build_next_round_feedback`，在官方函数返回值基础上追加真实 stdout（复用 `inline_score`，不用额外跑 checker）。**不是无条件追加**——CLI 用真实数据核实过：P1 用的 `check_iso8601.py`/内联 ISO 检查大多数失败时 stdout 很干净（`FAIL: field: value`），值得追加；但约 1/4 是检查脚本自己崩了产出 Python Traceback，原样拼进去只会更糊。`_filtered_checker_stdout` 只在 stdout 以 `FAIL` 开头、不含 `Traceback` 时才追加，否则整段跳过、退回纯静态反馈——采用 CLI 提供的两个选项里更简单的那个（跳过，不做"salvage 最后一行异常"）。

**修复 2：`check_filename.py --dir` 模式追加"日期不用精确匹配"的说明**——CLI 发现一个独立的真实缺口：`check_filename.py` 的 `--dir` 模式（P2，day06-10 用）只校验"任意 8 位日期+snake_case"，不要求日期等于场景虚构日期，但静态反馈的例子文件名（如 `20260327_...`）会让模型误以为必须精确匹配。**这条不能全局套用**——day11 起换成 `glob('dayXX/20260330_*.md')` 这类精确日期匹配，例子日期在那边确实必须精确，生搬硬套会教错。题面数据没有专门字段区分"宽松/精确"两种模式，只能靠解析 `round_record["eval"]["command"]`（`_is_dir_mode_filename_check`：含 `check_filename.py` 且含 `--dir` 才判定为宽松模式），检测不到就完全不加这条说明，不猜。CLI 确认 day01-05（P1）的 35 道 file_check 全部是 `check_iso8601`/内联检查，`check_filename.py`/`--dir` 零命中——这条改动对 P1 结构性零影响。

**修复 3：选择题 `FORMAT_ERROR` 追加原文片段**——CLI 发现 `_build_multi_choice_feedback` 格式解析失败（抽不到 `\bbox`）时，逐字返回 `prompts.py` 里的常量 `FORMAT_ERROR`，`094611` 训练里这条反馈在 day10-14 崩盘段连续出现 22 次，且 `_build_opd_hint` 对选择题也复用同一个函数，训练侧 hint 里同样重复。**这不是"无条件拼接的 bug"，是"格式失败这个条件反复触发导致的真实逐字复读"**——修法不是改官方函数（`_build_feedback_text`/`_build_multi_choice_feedback` 是 MetaClaw 官方代码，逻辑本身没错，只是文案固定），是在 driver 侧包一层：拿到返回值后判断是否等于 `prompts.py` 里 import 进来的 `FORMAT_ERROR` 常量（不在自己代码里抄字符串，避免以后官方改文案对不上），命中就追加模型这次实际输出的一小段原文，让反馈不再字节级相同。**一处改动同时覆盖 next-round 反馈和 OPD hint 两个使用点**，因为两边调用的是同一个官方函数。

**修复 4：`is_invalid_tool_use` 接线到 MetaClaw 的两条打分路径**——`_max_sentence_copies(reasoning) >= 12` 这套复读检测（规则 5，2026-08-07 为 Personal Agent Track 加的）已经在跑，`turn_data["is_invalid_tool_use"]` 对每个真实生成轮次都会算，但**强制 `eval_score` 覆盖成 -1 这个动作，只挂在 Personal Agent Track 的 PRM 分支**（`eval_score = _prm_eval_majority_vote(eval_raw)` 那一行之后），MetaClaw 的 `_metaclaw_verdict is not None`（checker ±1）和 `metaclaw_round_mode`（step-judge）两条路径都提前 `return`，从没读过这个标记。CLI 用 `094611` 真实 shadow 日志确认：52 次 `is_invalid_tool_use=True`，`[openclaw-rl-invalid-tool-use-penalty]` 强制 -1 触发 0 次——不是"规则 5 已经处理过 MetaClaw 的样本"，是"判了但没接到 MetaClaw 的 reward 管线上"。**修复不是重做检测**，是在 `prepare_patched_openclaw_combine_select.sh` 的两条分支里各加一次读取：`_metaclaw_verdict` 分支里，`eval_score = float(...)` 之后、OPD hint 材料化之前加 `if turn_data.get("is_invalid_tool_use"): eval_score = -1.0`（覆盖后的分数会同时用于 `accepted=True`/`accepted=False` 两种返回）；`metaclaw_round_mode` 分支里，跟现有的 `is_truncated` 强制 -1 并列加一条独立判断（两者都命中也没关系，各自打日志，最终都是 -1）。**明确不做**：`is_repeat_thinking_violation` 那条给 OPD hint 追加"别复读"提示的机制——那是 Personal Agent Track 判官投票路径专属的，MetaClaw 两条路径都没有对应的 LLM 判官投票环节，没有插入点，这次只做核心的强制 -1 接线。

**关于"改动不能影响 P1"这个约束，逐条核实过（不是假设）**：
- 修复 1：会碰到 P1（P1 的 file_check 通过率只有 45.7%，超过一半的失败题会触发这条追加），但改动性质是纯信息追加、不碰 `eval_score`/训练信号，风险可控——已确认 P1 用的 ISO checker 大多数失败 stdout 干净，追加是净收益，含 Traceback 的约 1/4 已被过滤规则挡住。
- 修复 2：结构性零命中 P1（P1 没有 `check_filename.py --dir` 题）。
- 修复 3：`094611` 这趟数据里 P1 阶段 MC 格式失败 0 次，几乎碰不到；即使碰到也只是让文字不再字节级重复，不改变判分。
- 修复 4：不是新增检测，是复用已经在跑、已经用真实数据两面校准过的规则 5（P1 良好样本最高复读 7 次，day12-14 坏样本最低 351 次，`N=12` 阈值两边都有数量级余量，直接沿用，不用重新调参）——P1 现在良好的样本本来就远低于这个阈值，不会被误伤。

**验证方式**：`py_compile` 通过（driver + 三个代理补丁脚本对真实官方源文件跑完整补丁链，全部编译通过）。新增合成测试覆盖：`_filtered_checker_stdout` 对干净/Traceback/非 FAIL 前缀/超长四种情况分别处理正确；`_is_dir_mode_filename_check` 对 `--dir`/精确 glob/缺失字段三种情况判断正确；`_build_next_round_feedback` 对"file_check 失败+宽松模式"“file_check 失败+精确模式（不加日期说明）"“file_check 失败+Traceback（跳过 stdout）"“file_check 通过（完全不增强）"“MC FORMAT_ERROR（追加原文片段）"五种场景验证输出内容符合预期；`is_invalid_tool_use` 接线用真实源码人工核对了插入位置（`_metaclaw_verdict` 分支：`eval_score` 赋值之后、hint 材料化之前；`metaclaw_round_mode` 分支：跟 `is_truncated` 并列，日志格式对齐 PA/truncation 同款 `overriding X -> -1.0`）。**真实训练环境完全未验证**——不热改正在跑的训练，下一轮训练需要确认：next-round 反馈里出现真实 checker stdout（干净情况）、`--dir` 模式题面出现日期说明、精确 glob 模式题面不出现、连续 MC 格式失败反馈不再字节级相同、`[openclaw-rl-metaclaw-invalid-tool-use-penalty]` 日志在命中复读等无效模式时正确触发。

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

  **已结构性解决，不再需要单独判断/修复（2026-08-19c）**：见"session 拆分——从'一天一个 session'改成'每题一个 session'"一节。那次改动是为了解决另一个问题（同一天后面题目被前面题目的长文本拖进 context overflow）而做的，但它顺带拆掉了这个 bug 的前提——每个 round 现在是独立的 proxy session，且 `_send_session_close_only`/`_send_verdict_turn` 的 `session_done` 都改成每题无条件发送，不再有"下一个 round 在同一个 session 里"这回事，"挂起轮次被下一个 round 的内容误当 next_state"这条路径不再存在，不需要再单独判断触发频率或设计修复。

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
