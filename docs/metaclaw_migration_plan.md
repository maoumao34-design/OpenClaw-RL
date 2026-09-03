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

### 基线结果（用于后续对比，2026-08-21 定版，`--agent` 修复后的新基线，评测本身跑于 2026-08-20）

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

**动机（2026-08-20 写的时候漏了一半理由、且把"临时权宜之计"错写成了"永久方法学结论"，两处都在 2026-08-21 被用户指出并改正）**：

**迁移完成的最终目标，是训练能稳定跑满完整 30 天不崩，直接拿"训练前全程 30 天基线混合数字"vs"训练后全程 30 天混合数字"这两个数字比——这就是论文自己验证 Hybrid RL 有没有用的办法，不需要任何冻结/分段技巧。** 这套"边训边考、全程混合数字直接比"的方法学本身没有问题，之前写"混合数字没法直接回答训练有没有提高能力"这句话说得太绝对，容易被理解成"这套方法学有缺陷、需要另找方法"——不是的。

**真正的问题是：现在还没有能力产出一个"训练全程 30 天不崩"的完整结果去跟基线比**——`094611` 那次全程训练已经证明，继续训练到 day12+ 会撞上 thinking 空转塌陷，Acc. 掉下去；这种情况下拿"全程 30 天混合数字"去跟基线比，比出来的是"训练中途崩了"而不是"方法本身有没有用"，这不是"边训边考"这套方法学的锅，是训练稳定性还没解决。

`METACLAW_TRAIN_UNTIL_DAY`/冻结窗口，是**这个现状下的一个临时权宜工具**——在训练还没法稳定跑满 30 天的这段时间里，先看看已经真实训出来的这部分（day1-K）有没有效果，不是要取代"训满 30 天、跟基线直接比"这个最终目标，更不是要长期维持一套"冻结窗口 vs 混合窗口"并存的双轨方法学。等 thinking 空转那三处修复验证有效、训练能稳定跑完整 30 天，就应该直接切回"训练前 30 天基线 vs 训练后 30 天混合数字"这套论文原生比法，冻结窗口这个工具到时候就不再需要了。

CLI 当时提出的具体方案（把 30 天拆成两段，前 K 天正常训练、dayK+1 起权重冻结继续跑完剩余天数）里，还带了一层"风险止损"的考虑——K 本身也是一个可以主动调小、观察到训练信号开始变差就提前停止更新权重的旋钮（"若 K 小冻结后仍明显高于 base → 早期正信号有用；若 K 大冻结后反而更差 → 支持继续吃 −1 会伤模型"），这条在 2026-08-20 写文档时被漏掉了，这次一并记录，但这条"风险止损"考虑也是服务于"训练稳定性还没达标"这个临时现状，不是这个工具存在的永久理由。

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

## 里程碑：K=6 冻结实验——首次观测到训练带来的真实、可信的能力提升（实验跑于 2026-08-20，2026-08-21 记录 + 更正出分口径）

**这是本次迁移第一个可信的"训练确实有用"的正面结果**，用的是当天早些时候实现的 `METACLAW_TRAIN_UNTIL_DAY` 功能（`commit 0882f69`）——训练 day1-6、day7 起冻结权重继续跑完剩余天数。

**出分口径更正（2026-08-21，重要）**：初版记录把"Frozen 窗口"当成了这次实验的主结果，这是错的，已改正。**主结果必须是本趟 `metaclaw_rollout_driver` 全程 live 聚合的 Acc./Compl.**——这跟论文 MetaClaw Full/Table 1 的出分方式完全一致（同一趟训练运行实时聚合，不是训后另开 held-out 评测），是这次迁移从 08-14 起就确认的方法学主线，`094611` 那种全程 30 天的混合数字也是同一套方法学的产物，不是"不够干净"的东西。Train/Frozen 窗口的拆分**只是同一次 run 的分段诊断**，供分析"训练效果有没有被拆分开来看"用，**不能拿 Frozen 窗口替代全程聚合去当这次实验的成绩**——这两者回答的是不同问题：全程聚合回答"这趟训练按论文方法学算出来的 Table 1 式数字是多少"，分段诊断回答"排除掉'混合了多种权重状态'这个因素之后，训练效果能不能单独看出来"。

**关键前提，必须记清楚（不受这次口径更正影响）**：**这次实验跑的是 `commit 0882f69`，不包含同一天稍晚才诊断/实现的"day12-14 超长 thinking 空转"三处修复（`commit 455a54f`）**——那三处修复是针对另一次全量 30 天训练（`094611`）的诊断结果，这次 K=6 实验在设计和实现上都早于那次诊断。这次实验里"冻结窗没有出现 `094611` 那种后期 Acc→0 塌陷"，**不能理解成"thinking 空转问题已经被验证解决了"**——这次没有崩，结构性原因是训练在 day6 就冻结了，day7 之后模型不再继续在（可能已经退化的）训练信号上更新权重，天然绕开了"继续训练→继续吃刻板反馈→陷入空转"这条链条，跟 `455a54f` 那三处"修反馈内容/接通已有复读检测"的修复是两件不同的事——那三处修复要在一次**继续训练到 day12 之后**的跑里才谈得上验证。

**身份与路径**：

| 项 | 值 |
|------|------|
| Run | `metaclaw_migration_20260820_122808` |
| 日志根目录 | `/dfs/data/openclaw-rl-project/logs/metaclaw_migration_20260820_122808/` |
| report.json / report.md | `.../report/report.json` · `.../report/report.md` |
| RUN_MANIFEST.txt / metaclaw_rollout.log / training.log | 同目录 |
| openclaw-rl commit | `0882f69` |
| 启动 / 报告写出 | 2026-08-20T04:28:08Z / 2026-08-20 20:54:18 +08:00 |
| 脚本 | `OpenClaw-RL/scripts/metaclaw/run_metaclaw_migration_modelfactory.sh` |
| 起点权重 | base `/dfs/data/models/Qwen3-4B-Thinking-2507-torch-dist`（非 Personal Agent Track ckpt） |
| 设置 | `METACLAW_TRAIN_UNTIL_DAY=6`（day1-6 更新权重；day7-30 冻结） |
| 题集 | MetaClaw-Bench Part I，30 天/期望 346 题；实际计分 343（3 题未进分，见下） |

**主结果（全程 live 聚合，本次实验对外主报这一行）**：

| 指标 | 值 |
|------|------|
| Acc. | **37.3%**（accuracy=0.3733791475773984，Correct=128.069/343） |
| Compl.（file_check only） | **13.9%** |
| vs 训练前基线（seed 465485731：Acc.=17.8%/Compl.=0%） | **Acc. +19.5pt，Compl. +13.9pt** |

官方 `metrics`（全程平均）：`passed=0.0904`，`exact_match=0.1487`，`f1=0.2789`，`iou=0.2530`，`precision=0.3281`，`recall=0.2630`。

未计分 3 题（`agent_succeeded=False` 或未写入分）：`day03/r11`（MC）、`day05/r13`（MC）、`day06/r4`（FC）。

**分段（同一次 run 的诊断细分，不是另一套评测）**：

| 窗口 | Days | Rounds | Acc. | Compl. | 含义 |
|------|------|:---:|:---:|:---:|------|
| Train（滚动权重） | day1-6 | 63 | 61.4% | 45.0% | 正在 GRPO/OPD 更新时的 live 分 |
| Frozen（固定 ckpt） | day7-30 | 280 | 31.9% | 7.1% | day6 后不再更新；诊断泛化/是否崩，不是论文主口径 |
| Overall（主结果） | day1-30 | 343 | 37.3% | 13.9% | 上两段混合；本实验对外主报此行 |

**按天明细**（Acc.=`official_score` 均值，Compl.=当日 file_check 均分，FC/MC 过关是分子/分母）：

| day | 窗口 | n | Acc% | Compl% | FC 过关 | MC 过关 |
|---|---|---|---|---|---|---|
| 01 | TRAIN | 10/10 | 40.0 | 16.7 | 1/6 | 1/4 |
| 02 | TRAIN | 11/11 | 53.0 | 28.6 | 2/7 | 3/4 |
| 03 | TRAIN | 11/12 | 71.2 | 62.5 | 5/8 | 2/3 |
| 04 | TRAIN | 10/10 | 53.3 | 33.3 | 2/6 | 1/4 |
| 05 | TRAIN | 12/13 | 83.3 | 75.0 | 6/8 | 4/4 |
| 06 | TRAIN | 9/10 | 63.2 | 40.0 | 2/5 | 2/4 |
| 07 | freeze | 11/11 | 21.2 | 12.5 | 1/8 | 0/3 |
| 08 | freeze | 12/12 | 37.9 | 12.5 | 1/8 | 2/4 |
| 09 | freeze | 11/11 | 49.4 | 37.5 | 3/8 | 1/3 |
| 10 | freeze | 13/13 | 46.4 | 33.3 | 3/9 | 2/4 |
| 11 | freeze | 10/10 | 40.0 | 0 | 0/6 | 4/4 |
| 12 | freeze | 11/11 | 31.5 | 0 | 0/7 | 2/4 |
| 13 | freeze | 12/12 | 37.5 | 0 | 0/7 | 2/5 |
| 14 | freeze | 11/11 | 25.8 | 0 | 0/8 | 2/3 |
| 15 | freeze | 13/13 | 22.3 | 0 | 0/9 | 1/4 |
| 16 | freeze | 10/10 | 46.7 | 16.7 | 1/6 | 2/4 |
| 17 | freeze | 11/11 | 27.0 | 0 | 0/7 | 2/4 |
| 18 | freeze | 12/12 | 36.1 | 12.5 | 1/8 | 1/4 |
| 19 | freeze | 11/11 | 35.3 | 14.3 | 1/7 | 1/4 |
| 20 | freeze | 13/13 | 35.6 | 20.0 | 2/10 | 1/3 |
| 21 | freeze | 10/10 | 36.7 | 0 | 0/6 | 3/4 |
| 22 | freeze | 11/11 | 29.5 | 0 | 0/7 | 1/4 |
| 23 | freeze | 12/12 | 21.5 | 0 | 0/8 | 0/4 |
| 24 | freeze | 11/11 | 25.8 | 0 | 0/7 | 1/4 |
| 25 | freeze | 13/13 | 20.5 | 0 | 0/9 | 2/4 |
| 26 | freeze | 12/12 | 33.3 | 0 | 0/7 | 3/5 |
| 27 | freeze | 11/11 | 36.4 | 0 | 0/6 | 2/5 |
| 28 | freeze | 13/13 | 30.8 | 0 | 0/8 | 2/5 |
| 29 | freeze | 11/11 | 22.0 | 0 | 0/7 | 0/4 |
| 30 | freeze | 15/15 | 22.9 | 0 | 0/10 | 1/5 |

**对照**：

| 参照 | Acc. | Compl. | 说明 |
|------|:---:|:---:|------|
| 训练前基线（选用）seed 465485731 | 17.8% | 0% | agentfix harness，`run_20260820_192625_.../run_20260820_192725` |
| 本次 K=6 主结果（全程） | **37.3%** | **13.9%** | live 聚合 |
| 本次训练窗 only | 61.4% | 45.0% | day1-6 滚动权重 |
| 本次冻结窗 only | 31.9% | 7.1% | day7-30 固定权重 |
| 论文 Kimi-K2.5 Baseline / Full | 21.4% / 40.6% | 2.0% / 16.5% | 不同模型与 Skills+RL 设定，仅数量级参照，不是同条件对比 |
| 早前全量训 `094611` | 后期崩塌 | — | 本 K=6 无 Acc→0 崩塌（原因见上"关键前提"，不是修复生效） |

**如实记录的局限（不夸大这次结果）**：
- 训练路径（走 proxy/`rl-training-headers`）和官方 `metaclaw-bench` 独立评测路径不完全同构——数量级对比站得住，但严格意义上的 apples-to-apples 对比，应该再单独用官方 bench 跑一次这个冻结后的 checkpoint 核实。
- 这次结果证明的是"`METACLAW_TRAIN_UNTIL_DAY` + 六处此前修复（尤其是 `--agent` workspace 修复、session 拆分、训练信号修复）组合起来，短训练窗口确实能带来干净可信的能力提升"——不代表 thinking 空转问题已经解决（见上面的前提说明），也不代表 30 天全程训练不会再出现 day12+ 那种塌陷（下一次验证需要用带上 `455a54f` 那三处修复的版本、训练超过 day6 继续跑，才能回答这个问题）。
- 跟论文 Kimi-K2.5 的数字只能当数量级参照——模型（Qwen3-4B vs Kimi-K2.5）、方法（我们的 Hybrid RL topk-select vs 论文的 Skills+RL）都不同，不是同条件对比。

**记法（可直接引用）**：
> K=6 freeze run 122808（commit 0882f69）：主报全程 live Acc.=37.3%/Compl.=13.9%（343/346 题），相对 agentfix 基线 seed465 的 17.8%/0% 提升明显；训练窗 day1-6 为 61.4%/45.0%，冻结窗 day7-30 为 31.9%/7.1%。主口径为全程聚合（对齐论文边训边分），冻结窗仅作分段诊断。

**意义**：这是这次 MetaClaw 迁移从"改 bug、修链路"阶段第一次拿到的正面训练效果证据——用**跟论文一致的全程 live 聚合口径**衡量，Acc. 从基线 17.8% 提升到 37.3%，`Compl.` 从 0% 提升到 13.9%（这次迁移第一次观测到非零的 `Compl.`）；Train/Frozen 分段拆分作为补充诊断，进一步确认这个提升不完全是"混合了多种权重状态"的假象——冻结窗单独看依然比基线同范围（约 16.2%）高出一截。

**这次实验本身是一个阶段性的中间结果，不是最终形态**：只训了 6 天就冻结，是因为训练还没法稳定跑满完整 30 天不崩（`094611` 那次全量训练在 day12+ 就撞上了 thinking 空转塌陷）——**这次迁移最终要拿到的，是训练稳定跑完整 30 天之后，"训练前全程混合基线"vs"训练后全程混合数字"这一组直接对比**，不需要任何冻结/分段技巧，这才是跟论文方法学完全一致的最终验证方式。当 thinking 空转三处修复（`455a54f`）验证生效、训练能稳定撑过 day12 及以后，下一步就该直接跑一次真正完整的 30 天训练，用这组"全程 vs 全程"的对比作为最终结论，而不是继续依赖 `METACLAW_TRAIN_UNTIL_DAY` 这个临时工具。

### 修复：暂停窗口把正在飞的生成砍断后，OpenClaw 报的是 timeout 不是 503，漏出 08-19 那次修复的覆盖范围（2026-08-25）

**背景**：K=6 冻结实验里未计分的 3 题（`day03/r11`/`day05/r13`/`day06/r4`）查出了同一个根因，但表现形态是新的——不是"新请求撞上已经暂停的 503"，是**暂停发生的那一刻，这道题的生成正在飞，被 SGLang 的 `pause_generation` 中途砍断**。以 `day05/r13` 为例，同一秒内依次发生：`drained 16 groups`→`submission paused`→这道题的 MAIN 轮次 `finish_reason=abort`→代理侧 `[openclaw-rl-degraded-turn-drop] ... is_aborted` 正确丢弃这个残缺样本（训练信号没被污染，这部分工作正常）→`Timer update_weights start`。但 OpenClaw 网关把这次中断报给 driver 的方式是 `GatewayClientRequestError: FailoverError: LLM request timed out`，不是 `"503 status code"`——`_run_round` 的 `_AGENT_PAUSE_MARKER` 只认后者，这次匹配不上，直接落进"`AGENT_RETRY=0`、立刻判 infra failure"这条路径，题目从 346 里被剔除，不进 Acc./Compl. 聚合。

**跟 08-19 那次 503 修复的关系，不是新问题、是同一诱因换了个表现形态**：08-19 那次修复时，CLI 用真实日志分析后明确把 timeout 排除在耐心等待之外——理由是"timeout 通常发生在已经深度生成中才挂，跟 503（一上来就被拒）性质不同，盲目等可能把真实的 GPU 争用问题拖得更长"。这次的数据核实了一件事：**至少这一句特定的 OpenClaw 网关文案（`"LLM request timed out"`），在已核实的样本里全部是暂停窗口导致的，不是"模型真的卡住了"**——`_run_round` 传的 `round_timeout=None`，driver 自己的 `asyncio.wait_for` 永不超时，日志里这句话 100% 是 `openclaw agent` 子进程自己打到 stderr、然后以非零码退出的，机制上跟 `"503 status code"` 完全一样（都是读子进程 stderr 里 OpenClaw 自己的固定错误文案）。

**真实数据核实（CLI，两轮）**：
1. K=6 这次：3/3 与暂停窗口在同一秒对齐。
2. 全部历史 migration run 回溯扫描：28 次真正的 `openclaw agent failed` + `LLM request timed out`（排除 transcript 里偶然出现同文案但不是失败原因的噪声），**28/28 都能对上 `pause`/`update_weights`/`finish_reason=abort`**，零反例；`094611` day12-14 那批裸的 `LLM request timed out` 字符串没有伴随 `openclaw agent failed`，是转录噪声，不会误触发这条新规则（按"只在 agent 真失败的 stderr 上匹配"这条既有原则，这些噪声天然被排除）。

**修复**：`_AGENT_PAUSE_MARKER`（单个字符串）扩成 `_AGENT_PAUSE_MARKERS = ("503 status code", "LLM request timed out")`（元组），`_run_round` 里的判断从 `if _AGENT_PAUSE_MARKER in stderr:` 改成 `any(marker in stderr for marker in _AGENT_PAUSE_MARKERS)` 命中任意一个都进现有那套耐心等待重试环——**完全复用现有的 `PAUSE_RETRY_INTERVAL_SECONDS`/`PAUSE_RETRY_MAX_WAIT_SECONDS` 预算，不新增参数**。日志里把"503 pause-retry"这个措辞改成"pause-retry (matched %r)"，带上具体命中的是哪个 marker，避免以后翻日志时被"503"这个字面词误导。

**08-19 的顾虑是收窄，不是整段收回**：只加了这一句特定、稳定的 OpenClaw 错误文案进 marker 集合，不是"以后所有 timeout 都当 503 处理"——如果 `round_timeout` 将来真的被设成一个具体数值、driver 自己的 `asyncio.wait_for` 触发了它自己的 `"Timeout after {timeout}s"` 兜底，那种超时跟暂停窗口无关，仍然应该立刻判失败，不进这套等待环，这个边界这次没有改变。

**Plan B（proxy `/healthz` 暴露 `submission_enabled` 状态、driver 精确轮询）暂不做**：08-19 设计 503 处理时就提过这个更精确但更重的方案，当时因为"没有需要区分的第二种情况"搁置；这次虽然确认了第二种触发文案，但 28/28 历史数据零反例，说明现有"文案匹配"这个粗粒度手段已经足够可靠，不需要马上上更重的精确判断机制——如果以后真出现"timeout 明显跟暂停无关却被这条新规则误当成安全等待"的反例，再考虑把 Plan B 捡回来。

**验证方式**：`py_compile` 通过。新增合成测试（mock `_run_openclaw_agent` 返回序列）覆盖三种场景：`"LLM request timed out"` 一次后成功→正确走耐心等待环后重试成功；持续 `"LLM request timed out"`→正确耗尽预算后判 `agent_succeeded=False`；跟暂停无关的普通失败文本→完全不进等待环、立刻按 `AGENT_RETRY=0` 判失败（零等待，确认没有被新 marker 误伤）。三个场景全部符合预期。**真实训练环境完全未验证**——不热改正在跑的训练，下一轮训练需要确认日志里出现"pause-retry (matched 'LLM request timed out')"这行，且这类原本会丢的题目能重试成功进入 Acc./Compl. 聚合。

## 查证记录：`094611`"超长 thinking 空转"复现的时间线——"中段 step-judge 脱钩是主根因"这个结论已撤回，真正机制见下一节

**背景**：`455a54f` 那三处修复（追加真实反馈、`FORMAT_ERROR` 去重、`is_invalid_tool_use` 接线）针对的是 day12-14 的表面症状；CLI 对同一批（`metaclaw_migration_20260820_100115`）日志做了更深的时间线分析，先划掉了几个不该当根因的表象（`day15+ Acc=0` 不是新现象，`day6`/`day9` 早就有过整天 `Acc=0`；thinking 爆炸/idle timeout 文案是 `day17` 起策略已经坏了之后的形态；`degraded-turn-drop` 是锁死阶段的放大器，不是最早点火的东西），给出了真实的三阶段时间线：

1. **阶段 A（约 day1-10）**：最终失败的 round 里，中段 step-judge 仍有 71% 是 +1；57/83 道最终失败题的**全部**中段步骤都是 +1。到 `day7-10`，任务 Acc 已经很差，但中段 RL+ 仍约 90-100%。
2. **阶段 B（day11-16）**：中段 RL+ 比例从 `day10` 约 91% 一路掉到 `day14` 41%、`day15` 15%；`day14-15` 的 `invalid-tool` 罚分暴涨（13→31），多数是"低复读次数 + 反复乱调工具"，不是规则 5 那种逐字复读。
3. **阶段 C（day17-22）**：几乎只剩 `finish_reason=length`、`tool_calls=0`，`max_sentence_copies` p90 到 130+；`agent_succeeded=True`（rc=0，内容是兜底文案）仍按 0 分进 Acc.，超时文案进下一题的 `[Previous Feedback]`；这个阶段几乎全部样本被 `is_duplicate_user_retry` 判定成 `degraded-turn-drop`，OPD/RL 提交量变成 0。

**这段真实时间线数据本身没有问题，保留**。**但当时基于这段数据得出的"主根因：中段 step-judge 打分跟最终任务成败解耦（信用分配错误），在最终会失败的轨迹上系统性给中段 +1 是训练动力学根因"这个结论，已被用户指出、CLI 重新核对后撤回**——阶段 A 里中段 +1（"写了文件、已经很接近正确答案"）本身是合理的过程 shaping，不是训练信号被污染；真正需要解释的问题不是"中段为什么给 +1"，是"持续的最终 -1/OPD hint，为什么没能把已经很接近的近似解（比如文件名 `2026_08_24_xxx.json` 只差把日期里的下划线去掉）拧成精确解，反而滑向了空转"——这个问题的答案是下一节记录的 `check_filename.py --dir --min-count` 累计计数缺陷：**这一轮即使写对了，也可能因为当天更早的题目已经欠账，被 checker 判 -1**，持续的"看似纠偏、实际拧不动"的负反馈，正是这个计数缺陷造成的，不需要另外假设"中段奖励结构错了"。之前"CLI 给出的对症方向：①重做中段奖励，失败最终局的中段不能默认 +1"这条也随之撤回，不再是待讨论的修复方向。

**保留、未被推翻的部分**：阶段 C"生成死亡→`is_duplicate_user_retry` 把残局样本整段丢掉→没有 -1/OPD 能拉回来"这条"崩溃为什么不可逆"的机制描述本身是真实观察，跟"中段脱钩是不是主根因"这个已撤回的判断相互独立，仍然成立，值得记住——但它现在应该理解成"下一节的累计计数缺陷持续制造错误 -1、模型学不会→最终触发 thinking 空转→空转样本又被 drop 掉、学不回来"这条链路的下游环节，不是一个独立的、需要单独设计中段奖励重构的根因。

**跟 MetaClaw 官方设计的关系（用户追问，已用真实源码核实，这部分判断不受上面撤回影响）**：直接读了 `metaclaw/prm_scorer.py`。MetaClaw 自己对**每一轮**（不区分中间/最终）都用同一个 `PRMScorer` 打分，判官 prompt 原文明确写"Do NOT compare against any follow-up turn. Only evaluate whether the response addresses the given instruction."——是刻意设计成纯局部、逐轮独立的判断，从不看这一步之后发生了什么、更不看整条任务最后有没有成功。MetaClaw 自己的训练奖励（不管中间还是最终）**从来没有任何确定性/最终结果信号参与**——他们的确定性 checker 只用来离线算 Table 1，从不进 RL 训练循环。这意味着"中段奖励跟最终结果脱钩"这个问题在 MetaClaw 自己的架构里根本不会显形，不是他们解决了这个问题，是这个矛盾在他们那边不存在——不过这一点现在更多是背景知识，不是"要不要照抄 MetaClaw 处理方式"这个决策的依据了，因为决策本身（重做中段奖励）已经撤回。

**pause-retry 扩展（本文档上一节）跟这条链路无关**：那次修复只影响暂停期间被误杀的计分（infra failure 层面），完全不碰奖励结构本身。

## 重大发现：`check_filename.py --dir --min-count` 是累计跨轮次计数，一旦落后就结构性补不回来——正确的做法也会被判负分

**已用真实代码 + 真实题面数据核实（不是推测）**：直接读了 `eval/scripts/check_filename.py` 全文（`benchmark/data/metaclaw-bench/eval/scripts/check_filename.py`）和 `eval/day06/questions.json` 的完整 `eval.command` 序列，结论如下。

**题目本身要求什么**：每道 file_check 题的题干只要求写/生成**这一轮的这一个**新文件，从不要求"一次写满 N 个文件"。

**checker 怎么判**：`check_dir(directory, ext, min_count)` 每次调用都是 `os.listdir(directory)` 现场扫描整个当天共享目录，数有多少个文件名匹配 `^\d{8}_[a-z][a-z0-9_]*\.[a-z0-9]+$` 这个正则且扩展名对得上，跟 `min_count` 比大小——**这是对"当天目录里现在实际有多少个合规文件"的累计计数，不是只看这一轮新写的那个文件对不对**。真实 `day06/questions.json` 里 `.json` 类型的 file_check 题的 `eval.command`：

```
r1 -> check_filename.py --dir day06/ --ext json                  （默认 min-count=1）
r2 -> check_filename.py --dir day06/ --ext json --min-count 2
r4 -> check_filename.py --dir day06/ --ext json --min-count 3
r7 -> check_filename.py --dir day06/ --ext json --min-count 4
r9 -> check_filename.py --dir day06/ --ext json --min-count 5
```

`min-count` 逐题递增，每次只比上一道同类型 file_check 题多 1——这个设计隐含的假设是"每道题都写对，累计数正好跟上"，但完全没有为"某一题写错了"这种情况留余地。

**后果：一旦某一题落后，后面每一题只让你新写 1 个文件，永远追不平**。举例：`r1` 要求 ≥1，如果这题写的文件名不合规（`found 0`）——`r2` 要求 ≥2，就算 `r2` 这题自己写得完全正确，目录里也只有 1 个合规文件（`r2` 自己写的这个），仍然 `FAIL`；`r4` 要求 ≥3，同样的情况会继续 `FAIL`……**这个缺口在正常"每题只写 1 个"的行为下是结构性、永久性追不平的**，除非某道题额外主动补写多个文件（题干从不会这样要求）。

**这意味着什么**：对于 `--dir --min-count` 这类 file_check 题，**checker 给某一轮打的 ±1，不是这一轮 agent 自己表现的干净函数——它被这一天之前所有同类型题的历史表现"传染"了**。一道题这一轮把文件名写得完全正确，如果前面已经落后过，checker 依然会给 -1。**这不是"agent 能力不够"，是训练信号本身在这类题上结构性地不公平**——我们一直假设"checker ±1 是可靠的最终事实、只有中间 step-judge 才可能判错"，这个假设对 `--dir --min-count` 这类题不成立。

**跟上一节"中段 step-judge 脱钩是主根因"的关系（更新）**：上一节那个判断已经撤回，**这一节的发现现在是"为什么持续的最终 -1/OPD 纠偏不动近似解、最终滑向 thinking 空转"这整条链路的真正解释，不是一个平行的、需要分别处理的独立根因**。中段 +1（"写了文件、已经很接近正确答案"）本身是合理的过程 shaping；模型确实在往正确方向靠近（比如把文件名从完全不合规写到只差日期格式这一步），但因为当天更早的题目已经欠账，这一轮即使写对也会被 checker 判 -1——持续收到"看似在纠偏、实际怎么改都还是 -1"的负反馈，这才是真正让训练信号失真、最终把模型逼向 thinking 空转的机制，不需要"中段奖励结构本身错了"这个额外假设。

**待讨论，不在这次记录里下结论**：怎么修——是改变 checker 使用方式（比如把"这一轮自己有没有正确完成"和"累计进度"拆成两个独立信号）、还是改这次迁移自己的判分口径（比如不直接用 checker 原始 ±1，而是判断"这一轮相对上一轮，合规文件数有没有净增加"）、还是这本来就是 MetaClaw 官方题面设计本身的特性（可能官方自己的 Simulator/Skills+RL 训练方法压根不会在这种"半路失败再也追不上"的场景下卡住，因为他们从不用这套确定性 checker 做训练奖励，参见"三方对照"）——这些都还没讨论，留到下一轮跟用户和 CLI 一起决定。

### 补充查证：day01-30 全部检查方式分阶段核查，发现第二类更严重的累计问题（`check_done_log.py`）+ 修复方向讨论（2026-08-25）

**背景**：上一节的发现只核实了 day06-10。用户要求核查 day15 及之后每 5 天一个阶段是否用同一种检查方式——查完发现不是，而且 day21 起引入的 `check_done_log.py` 是一种结构上更严重的累计问题，不是"追不上门槛"，是"一旦历史里有一行写错，之后永远过不了"。

**分阶段核查结果**（读了全部 30 天 `questions.json` 的 `eval.command` + `eval/scripts/` 下全部 5 个 checker 脚本源码）：

| 阶段 | 主要检查脚本 | 是否累计跨轮次污染 |
|------|------|------|
| day01-05 | `check_iso8601.py`（+ 少量 `python -c` 内联正则）| 否——都是查"某个题面自带的已存在文件"的具体字段格式，不是新建文件计数，天然只反映这一轮自己的表现 |
| day06-10 | `check_filename.py --dir --min-count` | 是——上一节已记录的累计文件数缺陷 |
| day11-15 | 题面自带 `python -c` 里手写 `sorted(glob('dayXX/YYYYMMDD_*.ext'))`，`len(files)>=N` 判断，`&&` 接一个 `check_metadata.py` 校验其中被选中的一个文件 | 是，结构上跟 day06-10 同类——只是不经过 `check_filename.py` 的 `--dir` 参数，改成题面自己手写的 glob 计数 |
| day16-20 | `check_backup.py`（新引入）与 `check_filename.py --dir --min-count` 混用 | `check_backup.py` 本身**否**（见下）；但同阶段仍混了几道 `--dir --min-count` 题，那几道继续是累计问题 |
| day21-25 | `check_done_log.py --min-entries N`（新引入）与 `--dir`/`check_backup` 混用 | **是，且是更严重的一种**（见下） |
| day26-30 | `check_backup` + `check_iso8601` + `check_metadata` + `check_done_log` 全部混用，偶尔仍带 glob 计数 | 视具体子检查而定，多种问题可能叠加在同一题里 |

**逐个脚本核实**（读了 `check_metadata.py`/`check_backup.py`/`check_done_log.py` 全文，此前只读过 `check_filename.py`）：

- `check_backup.py`：只查一个指定的具体文件名（如 `day20/weekly_status.json`）有没有对应 `.bak` 且内容跟原文件不同。一题绑定一个具体文件名，不同题目查不同文件，**没有跨轮次累计的结构**，本身没有问题。
- `check_metadata.py`：只查传进来的那一个具体文件路径的 metadata 字段是否齐全合规（JSON 的 `meta.*`、MD 的 YAML frontmatter、PY 的 docstring `Meta:` 段、CSV 首行 `# meta:` 注释），同样是单文件、round-local，没有问题。
- `check_done_log.py --min-entries N`：**比 `--dir --min-count` 更严重**。不是只比较条目数量，而是**每次调用都把 `done.log` 从第一行到最后一行全部重新按 `LINE_PATTERN`（`^\[DONE\] <ISO8601> \| <task_id> \| <summary，≤80字符>$`）校验一遍**：
  ```python
  for i, line in enumerate(lines):
      m = LINE_PATTERN.match(line)
      if not m:
          fail(f"line {i+1} does not match format: {line!r}")
  ```
  **只要曾经有一轮往 `done.log` 写错过一行格式（哪怕是很多轮之前），之后所有轮次不管新写的行多规范，判定永远是 FAIL**——`--dir --min-count` 至少理论上"门槛不再涨了就有机会追平"，这个是只要历史里有一行格式不对，就**永久性**过不了，除非有人手动把那行坏日志改掉，训练过程中不会有这种介入。`--task-prefix`（查最后一行 task_id 前缀）的校验同样被挡在这个全量逐行校验之前，历史行一坏就永远走不到。

**复合命令的连带问题**：day11 起大量题目是 `A && B` 形式（比如 `glob 计数 && check_metadata.py ...`，或 `check_filename.py --dir ... && check_done_log.py --min-entries ...`）。`check_backup.py`/`check_metadata.py`/`check_iso8601.py` 三个本身没有累计问题，但一旦排在它们前面的累计部分（glob 计数/`--dir --min-count`/`check_done_log`）因为历史欠账 FAIL，`&&` 短路，后面这几个原本没问题的检查根本没机会跑——即使这一轮的 metadata/backup 部分其实是对的，也拿不到任何信号。

**修复方向讨论（延续上一节，本次讨论后收敛的思路，仍未实现）**：

不能只改 checker 判定"数量"这一个维度了，要用**"round 开始前 / round 结束后"的 diff**——这个原理对文件计数和日志行计数是同一套：
- **文件累计类**（day06-10 的 `--dir`，day11-15/16-23 里仍出现的 glob 计数）：round 开始前 snapshot 目录里已有哪些文件，round 结束后再 snapshot 一次，取差集找出"这一轮新写的文件"，对新文件单独跑一次跟 `check_filename.py::check_file()`（day01-05 单文件模式）同样口径的判定（命名正则 + 扩展名），而不是比较目录里的绝对总数。day06-10 能直接复用 `--dir` 的 argparse 参数拿到目录/扩展名；day11-15 起没有干净的 CLI flag，需要从题面自带的 `python -c` 文本里解析出实际用的 glob 表达式。
- **日志行累计类**（day21-30 的 `check_done_log`）：同样的 diff 原理，换成"行"——round 开始前记 `done.log` 有几行、内容是什么，round 结束后只对**新追加的那几行**单独跑 `LINE_PATTERN` + 长度 + （若有 `--task-prefix`）前缀校验，完全不看历史行是否合规。这是"done.log 一旦写坏一行就永久卡死"这个问题唯一说得通的修法。
- **复合命令**：`&&` 拼接的题目需要拆开分别判断——累计部分（文件计数/日志行）用上面的新逻辑判，`check_backup`/`check_metadata`/`check_iso8601` 部分照官方原始逻辑独立跑，两边分别判完再合并结果，不能再依赖 `&&` 的短路。
- **只改训练奖励，不改官方 Acc./Compl. 口径**：官方 Table 1 的 Acc./Compl. 继续用 checker 的原始 pass/fail（绝对门槛/全量校验），一个字不改，保住跟论文的可比性——本来就是这套确定性 checker 从没被 MetaClaw 官方用作训练奖励，"用它做训练奖励"是这次迁移自己的设计，所以只在"checker 结果 → 训练用 `eval_score`"这一步的转换里做 diff-based 改动，不碰官方 checker 脚本或题面数据。这样同一道题会产生两个故意不同的判断：官方 Acc./Compl.（按官方口径可能仍是 FAIL）与训练奖励（按这一轮自身表现可能是 +1）。
- **反馈文字也要一起改，不能只改分数**：现有 `_build_opd_hint`（file_check 分支只给 checker 原始 stdout）和 `_build_next_round_feedback`（叠加静态文案 + 过滤后的 stdout）目前对 `--dir`/glob 累计类检查，失败反馈只有聚合数字（`found 3, need 5`），从不指出"这一轮自己新写的文件具体哪里错"（对比 day01-05 单文件模式的 `check_file()`，失败信息精确到具体文件名+原因）。用上面同一次 diff 拿到的"这一轮新增的文件/日志行"，可以顺带给出跟单文件模式同款的具体诊断（没写新文件/写了但命名不对/命名对但扩展名不对），而不是回退到没有信息量的聚合计数——奖励判断和反馈文案共用同一次 diff，不用分两次实现。若训练侧已经因为净增判 +1，但反馈文字仍然是官方那句"绝对门槛失败"，两个信号会打架，需要同步调整措辞。
- **两个被明确否决的替代方案**：(a) 让 agent 回头去补写早前欠下的文件——题面从不会要求这么做，要让 agent 这样做等于往题面塞了一条原本不存在的指令，改变了这道题在测什么；(b) 判负后由 driver 自己往目录里手动塞一个"正确"的文件把计数补平——这是伪造 workspace 状态骗过 checker，不是修复计分逻辑，会让后续题目看到的目录状态失真，比现在这个"不公平但真实"的状况更偏离复现真实 agent 行为。两者均未采纳。

**尚未确定/未实现**：day11-15 起 glob 表达式怎么从 `python -c` 文本里稳定解析出来（还没设计具体的解析方式）；这一整套 diff-based 改动尚未开始写代码，也还没交给 CLI review。

### 方案 v2：round 前后 diff 判定训练奖励（2026-08-25，CLI 用 30 天真实数据核对两轮后确认，Phase 1 可进入实现）

**status：v1（"只是设计，代码完全未动"）已被本节取代。** v1 提出后交给 CLI 用真实 30 天 `questions.json` + 现有 driver 代码做只读核对，发现 4 类边界后修订为 v2；v2 再交 CLI 二次核对，**结论是"relax-only + 统一 training_passed + 分类 fallback 构成完整安全网，Phase 1 范围可以进入实现"**。本节是定稿版本，代码仍然完全未动，下一步是实现。

**不变的两条原则**（v1 起就有，未变）：
1. 只改"checker 结果 → 训练用 `eval_score`"这一步转换，不碰官方 checker 脚本、题面数据、`--min-count`/`--min-entries`/`--task-prefix` 的数值本身；官方 Acc./Compl.（`_score_round_official`/`_aggregate_acc_compl`）继续吃 `_compute_inline_score` 的原始 `passed`，一个字不改。
2. 奖励判断和反馈文案共用同一次 diff，不分两次实现。

**核心安全约束（v2 新增，v1 没有）——relax-only**：

```python
seg_training_pass = seg_official_pass or seg_round_local_pass
```

**diff 只能把官方判负翻成正，永远不能把官方判正翻成负。** 动机：v1 的"用 diff 结果替换官方判定"在存在性检查类（`sys.exit(0 if files else 1)`，27 题）上会制造一类新的假阴性——官方只要目录里已有任一合规文件（可能是更早轮次留下的）就 PASS，但如果这一轮任务是"修改已有文件"而非"新建文件"，diff 会判"本轮无新增"从而给 -1，比官方更严，方向刚好跟 v1 想解决的问题相反。加上这条约束后：diff 逻辑本身出任何错（解析偏了、快照时机不对），最坏结果是"没解开连坐"、退回现状，不会引入新的错误负信号——CLI 核对确认这条约束"正好堵住 A3 假阴性"，且隐含的"官方因历史配额已达标、本轮零产出仍判 +1"是跟官方一致的合理行为，不是 bug。

**分类表（按 `eval.command` 特征分类，不按 day 段划分——CLI 核对：`--dir` 实际有 70 题，day16-23 还有 33 题，v1 按 day06-10 划分写窄了）**：

| 类别 | 特征 | 题量（CLI 实测） | 处理 | Phase |
|------|------|------|------|------|
| A1 | `check_filename.py --dir [--min-count N]` | 70 | 目录 diff + PATTERN/ext | 1 |
| A2 | glob 累计 `len(files)>=N` | 48 | glob 表达式 diff | 1 |
| A3 | glob 存在性 `0 if files else 1` | 27 | 同 A2（受 relax-only 保护） | 1 |
| B | `check_done_log.py --min-entries` | 46 题 / 75 段 | 日志行 diff | 1 |
| C1 | glob + 内容/ISO 校验（day09/r3、day10/r6） | 2 | **fallback 官方**（判定语义是内容不是计数） | — |
| C2 | 过滤式 glob（day11/r6，`[f for f in glob(...) if 'adr' in f]`） | 1 | **fallback 官方**（Phase 2 待定是否解析 filter predicate） | 2 |
| C3 | 双 glob 双阈值（day26/r8，`len(py)>=2 and len(js)>=2`） | 1 | 两组独立 diff，`bool(new_py) and bool(new_js)` | 2 |
| D | `check_backup`/`check_metadata`/`check_iso8601` | ~120 段 | 原样单独重跑，按 exit code，不改逻辑 | 1 |

C1/C2 的识别方式：段内出现 `json.load`/`re.match`/`open(` 等内容访问，或 glob 结果进了列表推导过滤 → 不进 diff 路径。**识别不出来就 fallback，不猜**——配合 relax-only 约束，误 fallback 的代价只是这题没解开连坐，不会判错。CLI 全量 398 个 `&&` 段扫描确认这个分类边界干净，没有 count 和 content 混在同一段的情况。

**机制 A：文件类 diff**（A1/A2/A3 共用）

```python
def _list_matching_files(directory: Path, pattern: re.Pattern, ext: str | None) -> set[str]:
    """只读扫描，不修改任何状态。pattern 是 check_filename.py::PATTERN 的字面量复制
    （单行稳定正则，风险等级同 _run_openclaw_agent 现有的"local copy"先例）。
    ext=None 时不过滤扩展名（用于 glob 模式）。"""
    if not directory.is_dir():
        return set()
    names = os.listdir(directory)
    if ext is not None:
        ext_lower = ext.lstrip(".").lower()
        return {f for f in names if pattern.match(f) and f.rsplit(".", 1)[-1].lower() == ext_lower}
    return {f for f in names if pattern.match(f)}
```

- **A1**：`directory`/`ext` 从 `round_record["eval"]["command"]` 解析 `--dir`/`--ext`（`_is_dir_mode_filename_check` 从布尔扩展成解析器）。
- **A2/A3**：正则抠出 `glob.glob('...')` 的字面量表达式（CLI 实测 79/79 命中，0 例外），round 前后对**同一个表达式**调用 `glob.glob()`，不自己拼目录/正则/扩展名。
- **C3（Phase 2）**：同一段抠出两个表达式，各自独立 diff。

```python
seg_round_local_pass = bool(after_set - before_set)
seg_training_pass    = seg_official_pass or seg_round_local_pass   # relax-only
```

**同一次 diff 顺带给出具体反馈**：额外做一次不限制 `ext`/`pattern` 的"任意新文件" diff，三种情况对应三种具体提示——`new_compliant` 非空（过了，不加料）/ 空但任意新文件非空（挑最接近的一个，仿 `check_filename.py::check_file()` 措辞给出"命名对但扩展名不对"等具体诊断）/ 两者都空（"这一轮没有新增任何文件"）。

**机制 B：done.log 行 diff**

```python
before_lines = _read_log_lines(logfile)   # round 开始前
after_lines  = _read_log_lines(logfile)   # round 结束后
```

**前提假设**：纯追加（`after[:len(before)] == before`）。CLI 确认题面数据无法证实这一点——**实现里必须打监控日志记录该假设被违反的次数，违反时立刻退化到官方原始判定**，不按新逻辑判。触发率只能等 Phase 1 上线后靠监控日志观察，属于仍未决问题。

```python
new_lines = after_lines[len(before_lines):]
seg_round_local_pass = (
    bool(new_lines)
    and all(_LINE_PATTERN.match(l) and len(_LINE_PATTERN.match(l).group(4)) <= 80 for l in new_lines)
    and (_task_prefix_ok(new_lines[-1], task_prefix) if task_prefix else True)
)
seg_training_pass = seg_official_pass or seg_round_local_pass   # relax-only
```

`_LINE_PATTERN` 是 `check_done_log.py::LINE_PATTERN` 的字面量复制。带 `--task-prefix` 时查 `new_lines[-1]`——CLI 确认纯追加前提下与官方查 `lines[-1]` 等价。

**复合命令（`&&` 拼接）**：按 ` && ` 切分（CLI 实测 0 反例）。逐段分类判定，**训练 `eval_score` = 各段 `seg_training_pass` 的 AND**——累计段用 diff 解连坐，D 类段仍按官方 exit code 严判，杜绝"文件数够了但 metadata 缺字段也给 +1"这种过度奖励；只要有一段 D 类官方 FAIL，即使累计段 diff 过，整体仍是 -1，这是防过度奖励的主闸，必须保持。**重跑 D 类段时整段原样交给 shell，不拆内部 `$(python -c ...)` 子壳**（CLI 明确点出的坑）。

**relax-only 要求累计段也单独跑一次官方命令拿 `seg_official_pass`**（多 1 次 subprocess/段）：CLI 对全量 398 个 `&&` 段做启发式扫描，未发现任何写操作迹象（5 个官方 `check_*.py` 均为只读 `open`，题面自带 `python -c` 段未见 `write`/`a`/`w`/`unlink` 等），确认这个额外调用安全、开销可忽略。**可选优化**（非必须）：整链 `_compute_inline_score` 已 PASS 时，各段 `seg_official_pass=True` 可直接认定、跳过分段重跑，只在整链 FAIL 时才分段。

**统一 `training_passed`，三处共用**（v1 的真实缺口，CLI 核对确认）：v1 只改了 `eval_score` 和 `_build_opd_hint`，漏了 `_build_next_round_feedback`——它的失败分支看的是 `inline_score.get("passed")`，结果会是训练判 +1、OPD 无 hint，但下一轮 `[Previous Feedback]` 里照样贴官方 `FAIL: expected >= N, found K` 静态聚合文案，信号只修好一半。改法：`training_passed`（仅 A/B 类生效，其余类 alias 官方 `passed`）同时驱动 `eval_score`、`_build_opd_hint` 是否产出 hint、`_build_next_round_feedback` 是否走失败分支。

**失败文案的优先级（CLI 二次核对时补的一句实现约定，之前模糊）**：`training_passed=False` 时，失败文案**以 diff 诊断为主**（命名/扩展名/无新文件/新 log 行格式）；官方静态 `feedback.incorrect` 只能作次要补充，**不能盖过 diff 具体诊断**——不能再优先贴官方聚合 `FAIL: expected >= N, found K` 那一句。

**接线点**（CLI 确认比预想简单）：`prepare_patched_openclaw_combine_select.sh` **不用改**（只消费 `_send_verdict_turn` 发来的 `eval_score`/`hint` JSON，格式不变）。改动全在 driver：
1. `_run_round` 里、`_run_openclaw_agent` 调用**之前**：按分类结果做 before snapshot（目录 listing / glob 求值 / 日志行）——workspace 此时已由 `_prepare_work_copy` 建好且跨 round 持久，是最自然的插入点。
2. agent 跑完：`_compute_inline_score` 照旧，供 `_score_round_official` → 官方 Acc./Compl. 完全不变。
3. 新增 `_compute_training_verdict(...) -> (training_passed, hint_text)`：分段、after snapshot、逐段判定、AND、生成具体 hint。非累计类直接 alias 官方 `passed`。
4. [metaclaw_rollout_driver.py:1420-1423](openclaw-rl/scripts/metaclaw/metaclaw_rollout_driver.py:1420) 那三行（`passed = inline_score.get("passed", False)` / `eval_score = 1.0 if passed else -1.0` / `hint = "" if passed else _build_opd_hint(...)`）改成消费 `training_passed` + diff hint。

**Phase 划分**：
- **Phase 1**：A1(70) + A2(48) + A3(27) + B(46题/75段) + D 段重跑 + relax-only + `training_passed` 三处接线。C1/C2/C3 全部 fallback 官方。
- **Phase 2**：C3 双 glob 双阈值；C2 过滤式 glob（或维持 fallback——1 题，解析 filter predicate 的实现成本可能不划算，CLI 认为维持 fallback 更划算）。

**仍未决 / 需实测**：
1. done.log 非追加场景真实触发率——题面数据无法预判，只能等 Phase 1 上线后靠监控日志观察。
2. 官方 Acc./Compl. 与训练信号会在"历史连坐但本轮净增"的题上分叉（报告里 Acc./Compl. 会看起来比训练信号差）——这是预期现象，不是新 bug，报告/文档里需要写一句避免误读。

**当前状态**：设计已经过 CLI 两轮真实数据核对确认，**Phase 1 范围（A1/A2/A3/B/D + relax-only + 三处接线）可以进入实现**；代码仍然完全未动。

### Phase 1 已实现（2026-08-26 提交，本地验证通过；首次真实训练结果见下一节）

代码全部落在 `scripts/metaclaw/metaclaw_rollout_driver.py`：

- **新增**：`_split_command_segments`/`_classify_segment`（8 类分类，A1/A2A3/B/OFFICIAL 四种可操作结果，C1/C2/C3 全部落进 OFFICIAL）、`_list_matching_files`/`_read_log_lines`/`_snapshot_segment`/`_prepare_before_snapshots`（before/after 快照）、`_diagnose_file_segment`/`_diagnose_log_segment`（具体反馈诊断）、`_check_new_log_lines`、`_rerun_segment_official`（segment 级独立重跑）、`_compute_training_verdict`（relax-only 汇总，`training_passed`/`training_hint` 的唯一产出点）。
- **`_run_round` 改动**：函数入口新增 `before_snapshots = _prepare_before_snapshots(...)`（在 `_run_openclaw_agent` 调用之前，纯读操作，官方 Acc./Compl. 走的 `official_score` 完全不受影响）；`_compute_inline_score`/`official_score` 计算之后，新增 `_compute_training_verdict(...)` 调用，结果以**新增键**注入 `inline_score["training_passed"]`/`inline_score["training_hint"]`（不覆盖/删除任何已有字段）。
- **`_build_next_round_feedback` 改动**：读取 `training_passed`；为 True 时构造一份 `passed` 强制为 True 的浅拷贝喂给官方 `_build_feedback_text`（跟真通过同样措辞，不是硬编码空字符串）；为 False 时优先用 `training_hint`，`training_hint` 为空才退回原有的 `_filtered_checker_stdout`（对应 CLI 二次核对补的那句"diff 诊断为主"实现约定）。
- **`run_day` 主循环改动**（原 ~1420-1423 行）：`eval_score`/`hint` 改成消费 `training_passed`/`training_hint`（`inline_score.get("training_hint") or _build_opd_hint(...)` 的优先级顺序），不再直接用官方 `passed`。
- **`prepare_patched_openclaw_combine_select.sh` 未改动**——按 CLI 确认，它只消费 `_send_verdict_turn` 发出的 `eval_score`/`hint` JSON，接口不变。

**分类识别修正（实现中发现，v2 设计文本没预料到）**：`_classify_segment` 最初对含 `glob.glob(` 子串的任意段一律考虑 A2A3，day11+ 常见的 `check_metadata.py $(python -c "...glob.glob(...)...")` 这种"用 glob 选目标文件、本身是内容检查"的写法会被误分类——已加一条前置守卫：段内出现 `check_metadata.py`/`check_backup.py`/`check_iso8601.py` 时直接判 OFFICIAL，不再往下走 glob 分支。

**验证**：
1. `py_compile` 通过。
2. 单元级合成测试（tmp workspace，覆盖：A1 净增文件把官方 FAIL 升级为 training PASS；本轮无新文件/新文件命名错误分别给出具体诊断而非聚合数字；官方已 PASS 时不触碰 diff 逻辑；复合命令里一段净增另一段没有，整体仍判负；两段都净增才整体判正；`done.log` 历史被改写时正确退化到官方判定并打日志）——全部通过。
3. **全量 30 天真实数据分类扫描**（遍历全部 `questions.json` 的 `eval.command`，逐段分类计数）：A1=70、A2A3=75（=CLI 报告的 A2 48 + A3 27）、B=75 段——**与 CLI 两轮真实数据核对报告的数字完全一致**，是比单个样例测试更有说服力的验证。

**仍未验证**：真实训练环境下这套逻辑的实际效果（`day12`+ 是否还会复现 `094611` 的 thinking 空转崩溃模式）、`done.log` 非追加场景的真实触发率（监控日志已埋点，等下一轮真实训练观察）、`_rerun_segment_official` 对每个 segment 额外一次 subprocess 调用在真实 GPU 训练节奏下的实际耗时影响。

### 首次真实训练核实：打分改对了，但暴露一处我自己引入的反馈质量回归（2026-08-27）

**CLI 用真实训练数据核实打分侧，结论是正确的**：

- **4 个升级案例全部核实无误**（`day06/r2`、`day08/r5`、`day08/r11`、`day09/r2`），四个都是标准场景——本轮按要求写对了一个合规文件（如 `20260827_test_results_summary.json`），只因前面轮次欠账、累计总数没到 `--min-count` 阈值而被官方判 FAIL，现在全部拿到训练 +1，正是这次修复的目标场景。
- **`passed=True` 而 `training_passed=False` 的反向情况 0 次**——relax-only 约束在真实数据上成立，不只是合成测试里成立。
- **诊断文案确实具体化了**：`'ci_build_report.json' does not match YYYYMMDD_snake_case.ext pattern`，而不是原来的 `expected >= 2, found 1`；53 次 `no new file was created in this round` 对应工具调用坍缩的那些轮，诊断准确。
- 顺带确认了一个设计预期：升级案例里 agent 用的是真实日期 `20260827` 而非题面场景日期 `20260326`——`--dir` 模式下任何 8 位日期都算合规，判 +1 正确；day11+ 切成精确日期 glob 后同样的写法不会被升级，符合 `_is_dir_mode_filename_check` 那条区分的设计意图。

**但暴露一处回归：14 处 Python Traceback 泄漏进了 agent 可见的 `[Previous Feedback]`**（CLI 发现，判分不受影响，只影响反馈质量）。

**根因比"少了一层过滤"更具体——是我在 Phase 1 里多加了一层不该加的 fallback**。这个 hint 有两个消费者，它们需要的兜底策略本来就不同：

| 消费者 | 兜底策略 | 为什么不同 |
|------|------|------|
| OPD hint（`run_day`）| `_build_opd_hint` 的原始 checker stdout，不过滤 | 这是本机制存在之前就有的行为，蒸馏目标要的是最原始的失败事实 |
| agent 可见的 next-round 反馈（`_build_next_round_feedback`）| `_filtered_checker_stdout`，**丢弃含 Traceback 的 stdout** | day01-05 题面自带的 `python -c` 没有异常处理，失败时会吐一整段调用栈，原样贴给模型只会让反馈更糊（这正是 2026-08-20 加 `_filtered_checker_stdout` 时的原始动机）|

两个调用点各自本来就写好了自己的 fallback。但我在 `_compute_training_verdict` 内部又补了一次 `hint = "\n".join(diag_parts) if diag_parts else _build_opd_hint(...)`——**这等于把两个调用点各自不同的兜底策略，在上游强行统一成了 OPD 那一套**：`training_hint` 不再为空，agent 可见路径就走进了 `if training_hint:` 分支，把自己的 Traceback 过滤整个跳过了。

**修复（一行）**：`_compute_training_verdict` 只返回 diff 推导出的诊断，没有就返回 `""`，不在函数内部做任何 fallback：

```python
return False, "\n".join(diag_parts)
```

两个调用点的 fallback 自动各自恢复——`run_day` 的 `training_hint or _build_opd_hint(...)` 拿到原始 stdout（跟改动前完全一致），`_build_next_round_feedback` 的 `else` 分支重新走 `_filtered_checker_stdout`（Traceback 被拦掉）。两处 docstring 都补了说明，讲清楚"这两个 fallback 故意不同、不能在上游合并"，避免以后又被当成重复代码合掉。

**验证**：`py_compile` 通过；新增合成测试复现 CLI 报的真实场景（checker 崩溃吐 Traceback 的 OFFICIAL 类轮次）——确认 agent 可见反馈不含 Traceback、官方静态文案仍在、**OPD hint 仍然拿到原始 stdout（行为未变）**、diff 诊断存在时仍然优先于聚合计数行。**并且确认了这个测试不是空转的**：手动模拟修复前的行为（把 `training_hint` 填成 `_build_opd_hint` 的原始输出）后重跑，Traceback 确实泄漏——说明测试真的能抓住这个 bug，不是碰巧通过。

**这次回归的教训**：Phase 1 的合成测试只覆盖了"diff 能产出诊断"的路径，没覆盖"diff 产不出诊断、走 fallback"的路径——而 bug 恰恰在后者。全量 30 天分类扫描也帮不上忙，因为它只验证分类计数，不验证反馈文本内容。**验证覆盖到了主路径不等于覆盖到了兜底路径**，这是本地验证通过、真实训练仍暴露问题的直接原因。

### 诊断：day17 thinking 断崖 → MC 格式失败 → Acc 崩塌，上游嫌疑是 FC 中间步骤的判官正奖励（2026-08-28）

**这一节的结论经历过两次自我纠正**，两次都被 CLI 用真实数据否掉，过程一并记下来，避免以后重复同样的推断方式。

**第一次推断（错，已否）**：拿 `day04/r5/turn3` 当"任务已完成后继续修改无关文件、仍拿 +1"的证据。CLI 查 `metaclaw_rollout.log:1339-1377` 后否掉：**该轮官方 checker 是 `passed=False`**（`check_iso8601.py` 抛 `AttributeError: 'list' object has no attribute 'get'`），任务压根没完成。这个样本只能证明"判官不惩罚修改本任务未要求的文件"，证明不了"完成之后才漂移"——它把"任务未完成"和"额外改无关文件"两个因素混在了一起。

**第二次推断（错，已否）**：据此提出"完成后额外操作比例应随天数单调上升"的统计口径。CLI 指出两点：(a) 该口径**在现有日志里算不出来**——没有逐 turn 的 checker 状态、没有 `completion_turn`，最终 checker 只在 round 结束时跑一次；(b) 实际统计**否证**了这个预期——中间步骤 +1 的高峰在 day07-12（348 步 / 239 个 +1），day13 之后反而下降，day19-22 的下降还是幸存者偏差（generation 已 length/abort，形不成可判定的中间 turn）。

**第三次推断（当前结论，有数据支撑）**：用户指出前两轮框架本身就有问题——(a) 拿 K=6 当参照系不对，K=6 不是目标，它存在的唯一原因就是全程训练会训坏；(b) **MC 退化更可能是 FC 训练的连带损害，不是独立问题**。第二点直接推翻了此前"差距在 MC 则判官消融无关"的说法：样本的**来源**是 FC，不代表它的**影响**只限于 FC，改的是同一份权重。

按样本构成算，这个反驳是有力的：

| 样本来源 | 数量（day01-22） | 占比 |
|------|------|------|
| FC 中间步骤（判官打分） | 628 | ~72% |
| FC 最终轮（checker ±1） | ~161 | ~18% |
| **FC 合计** | **~789** | **~90%** |
| MC 最终轮 | ~85 | ~10% |

**权重 90% 由 FC 派生信号塑造**，MC 只占 10%——MC 崩溃几乎不可能是那 10% 自己造成的。

**CLI 随后补的两组数据让链条闭合**（`metaclaw_migration_20260827_163030` vs K=6 的 `20260820_122808`）：

day16-22 的 MC 失败性质（按 `_compute_inline_score` 同款正则从实际回答重判，`report.json` 未保留 `format_valid`）：

| | 格式失败 | 格式正确但答错 | 正确 |
|---|---|---|---|
| 本次 day16-22 | **17/26** | 6/26 | 3/26 |
| K=6 day16-22 | **0/27** | 16/27 | 11/27 |

按天的 thinking 长度与 `finish_reason`（本次 run 的 MC 轮次）：

| Day | finish_reason | 平均 thinking | 平均 response tokens |
|---|---|---|---|
| 16 | 4 stop | 18k | 4.4k |
| **17** | **3 length / 1 stop** | **115k** | **27.3k** |
| 18 | 2 abort / 2 length | 133k | 30.7k |
| 19 | 4 length | 143k | 34.5k |
| 21 | 4 length | 145k | 34.4k |

K=6 在 day17-22 的 MC **全部是 stop**，平均 thinking 约 4k-20k，无 length/abort。

**所以真正的转折点是 day17（不是此前一直以为的 day12-14）**，链条是：

```
FC 中间步骤独立判官给 +1（其中 64 个样本 thinking ≥20k 且拿到 +1）
  ↓  loss_mask = [1]*len(response_ids)，整段含 <think> 都进 loss
长 thinking 被正强化，在持续训练中膨胀
  ↓  day16 还是 18k，day17 跳到 115k
生成撞 length 上限 / abort
  ↓  MC 需要在末尾输出 \bbox{}，撞上限的生成永远输不出来
MC 格式失败 17/26（K=6 为 0/27）→ MC 平均分 26.8% vs K=6 的 83.0%
```

**仍不能说已经证明"判官是唯一原因"**，但已经不是单样本推断——有明确的时间断点 + 格式失败性质对比 + 样本构成占比三项支撑。

**顺带澄清 day11-15 的 FC 全 0**：两边（本次与 K=6 冻结）都是 0/6..0/9，**这是共同的能力墙，不是本次训练特有的退化**——day11 起题目从 `check_filename --dir`（只查命名）换成 `glob 精确日期 && check_metadata.py`，后者硬性要求 `meta.created_at`/`author`/`status` 三字段齐全且格式合规。这条线跟训练信号无关，不要跟上面的链条混在一起。

**K=6 的正确定位**（此前用错了）：它不是目标，是**控制组**——同一 harness、同一数据、同一起点权重，唯一差别是"day6 之后有没有继续训练"。所以差值衡量的是"day7-22 这段训练造成了多大损害"（MC -20.9pt，FC -5.6pt），不是"离目标还差多远"。

### 方案（待 CLI 查验，未实现）：中间轮次改吃本轮最终 checker 结果的消融实验（2026-08-28）

**目的**：验证上一节链条的上游假设——**FC 中间步骤的独立判官正奖励，是不是超长 thinking 逐步膨胀的原因**。这是一次因果消融，不是性能优化。

**为什么不是"直接删掉中间轮次"**（用户最初的设想，讨论后调整）：
- 训练样本会从约 874 掉到约 246（约 -72%），训练步数从约 54 步掉到约 15 步——**跟"训得少所以好"混在一起**，而 K=6 已经演示过训得少更好，结论无法归因
- file_check 的文件是在中间 tool-call 轮次写的，**最终轮往往只是"我已创建该文件"这类总结文本**；只留最终轮等于训总结、丢动作

**采用的形态**：中间轮次**仍然提交为样本，样本量不变**，只把 reward 来源从"步骤判官分"换成"本轮最终 checker 的确定性 ±1"。判官仍然照常调用并记日志（用于事后对照），但分数不进训练。这样只替换一个变量。

**开关**：`METACLAW_MIDROUND_REWARD`，默认 `judge`（现有行为，一字不改），设为 `outcome` 启用。沿用 `METACLAW_TRAIN_UNTIL_DAY` 的 opt-in 惯例。

#### 已核实的机制前提（不是推断）

1. **08-17 暂缓该设计的理由已自动失效**。当时写"代理没有 round 概念，只能靠 driver 串行调用这个前提，且两轮之间可能有杂音请求"——那是"一天一个 session"时期。**08-19c 改成每轮一个 session 之后，`session_id` 本身就是 round 边界**（`metaclaw-{test_id}-{group_id}-{round_id}`），`_pending_turn_data[session_id]` 天然覆盖整轮，不需要任何跨 round 推断；即使 OpenClaw 内部真有额外调用，它带的也是同一个 `session_id`，本来就属于这一轮。
2. **继承链**：`OpenClawCombineSelectAPIServer` → `OpenClawCombineAPIServer` → `OpenClawOPDAPIServer`。`_opd_evaluate` 在 Select 层；`_maybe_submit_ready_samples` 在 Combine 层（覆盖了 OPD 层的同名方法）；verdict 分支在 OPD 层的 `_handle_request`。三层都要改。
3. **verdict 不是一条新的训练 turn**（CLI 指出，已核实 `prepare_patched_openclaw_opd.sh` 的 `openclaw-rl-metaclaw-verdict-signal-skip` 补丁）：`max_tokens=0` 的请求**不创建新 pending turn**，而是对**本 session 最后一个真实 turn** 调 `_fire_opd_task(...)`，用 verdict JSON 当 next_state；最终 checker 分数出现在那个 turn 的 `opd_result` 里。所以不能理解为"verdict turn 到达即提交"。
4. **异步补提交机制已存在**：`_fire_opd_task` 挂了 `task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))`——**任务完成后会再次触发提交，且不带 `force_drop`**。verdict 路径下 `force_drop` 先跑、任务后完成，最终轮样本正是靠这个回调补交的。

#### 设计

**层 1 — `prepare_patched_openclaw_combine_select.sh`（`_opd_evaluate`）**

只加显式标记，不改判分逻辑：
- verdict 分支的两个 return 都加 `"metaclaw_verdict": True`
- step-judge 分支的 return 加 `"metaclaw_round_step": True`、`"judge_raw_score": <majority vote 原始值>`、`"legacy_reward_would_have_been": <经 truncation/invalid-tool-use 覆盖后的分数>`

**两个分数都要留**：`judge_raw_score` 是判官原始投票结果，`legacy_reward_would_have_been` 才是"不开这个开关时实际会用的 reward"——事后对照旧行为应当用后者。按 CLI 要求，**不靠 `turn_num`/`accepted`/`hint` 是否为空去猜测分支**。

**层 2 — `prepare_patched_openclaw_combine.sh`（`_maybe_submit_ready_samples`）**

真正决定"何时提交、用什么 reward"的位置。新增三个按 session 的状态：

```
self._metaclaw_held[session_id][turn_num] = (turn_data, opd_result)  # 滞留的中间轮次，含完整 opd_result
self._metaclaw_outcome[session_id] = float                            # 本轮最终 checker 分，verdict 到达后写入
self._metaclaw_verdict_fired[session_id] = True                       # 由层 3 设置，见下
```

`outcome` 模式下的分发规则：

| 情况 | 处理 |
|---|---|
| `metaclaw_round_step` 且本轮 outcome **未知** | 移出 `pending`，**连同完整 `opd_result` 一起**存入 `_metaclaw_held`，不提交 |
| `metaclaw_round_step` 且本轮 outcome **已知**（verdict 先完成的竞态） | 直接用已存 outcome 提交 |
| `metaclaw_verdict` | 先记录 outcome → **flush 所有滞留轮次（用 outcome 当 reward）** → 再照原逻辑提交这一轮自己 |
| `force_drop_without_next_state=True` 且 `_metaclaw_verdict_fired` **未置位** | 丢弃滞留轮次（基础设施失败路径，永远等不到 outcome） |
| `force_drop_without_next_state=True` 且 `_metaclaw_verdict_fired` **已置位** | **不丢**（verdict 任务在飞，稍后回调会补 flush） |

中间轮次继续走**现有的提交路径**（`_submit_rl_turn_sample`），不受判官 `accepted` 过滤影响——只替换 reward 来源。最终 verdict 轮次的 hint / reward 逻辑一字不改。

**层 3 — `prepare_patched_openclaw_opd.sh`**

- `__init__` 初始化上面三个 dict
- `openclaw-rl-metaclaw-verdict-signal-skip` 分支里，`_fire_opd_task` 调用成功后**置位 `_metaclaw_verdict_fired[session_id]`**——必须显式标记，靠"是否还有未完成任务"推断不可靠：基础设施失败路径下中间轮次的任务同样可能未完成，两者靠任务状态无法区分
- session 彻底结束（flush 完或丢弃完）后清理这三个 dict，避免无界增长

#### 这个设计能消除什么、不能消除什么

- **能**：失败 round 中的长 thinking 拿 +1 —— 这正是链条的上游
- **不能**（CLI 指出，接受为已知残留）：**最终成功的 round 里，无效的额外操作仍会拿到正奖励**。`day04/r5` 那类"改了本任务未要求的文件"，只要该轮最终通过，仍是 +1。判官缺范围判据是独立成立的另一个缺陷（此前讨论的 A/B 方向），**不在这次消融范围内**，留待后续。

#### 验收条件（CLI 提出的三条 + 补充）

1. `_opd_evaluate` **显式返回** `metaclaw_verdict` / `metaclaw_round_step` 标记，不靠其它字段推断
2. 滞留时**缓存完整 `opd_result`**，不只是 `turn_data`；同时记录 `judge_raw_score` 与 `legacy_reward_would_have_been` 两个分数
3. **outcome 赋值与所有样本提交必须发生在 cleanup 之前**；且 verdict 之后才完成的 judge task 要能用已存的 session outcome 补提交（不能因为第一次 cleanup 时任务未完成就永久丢弃）
4. 每个中间样本提交时打日志，记 `inherited_outcome` 与 `legacy_reward_would_have_been`，用于确认确实没有混入旧判官分
5. **开关关闭时行为逐字节不变**，合成测试须覆盖

#### 跑完之后看什么（主判据不是 Acc）

1. **day17 那个断崖有没有消失**——按天平均 thinking 长度、`finish_reason=length` 占比
2. **MC 格式失败率**——本次 day16-22 是 17/26，K=6 是 0/27，看能否压回去
3. Acc./Compl. 只作次要参考——奖励源换了，绝对分数不直接可比

#### CLI 查验补的一条边界（已采纳）：verdict task 自身失败

原设计只区分了四种情况（verdict 未触发 / 已触发但 task 在跑 / 已成功拿到 outcome / 基础设施失败），**漏了"verdict 已触发但 `_opd_evaluate` task 抛异常或返回无效结果"**。这种情况下：`force_drop` 因为 `verdict_turn` 已置位而不丢弃滞留样本，异常分支又永远产不出 outcome——**滞留样本既不提交也不清理，永久留在内存里**。

已补成明确终态：`pending` / `succeeded` / `failed` / `no_verdict`。task 抛异常或 outcome 无效时一律 **discard-and-cleanup**——丢弃本 session 滞留样本、清理状态、记 `failed`。**"verdict fired" 不等于"最终一定拿得到 outcome"**。

实现上有个细节：task 抛异常时 `opd_result` 根本不存在，`metaclaw_verdict` 标记也就无从读取，所以**唯一能识别"失败的是 verdict task"的办法是拿 `turn_num` 跟记录的 `verdict_turn` 比对**——这也是 `verdict_turn` 存的是轮次号而不是布尔值的原因。

CLI 另外三条验收点也已采纳：
1. **现有丢弃门控必须在 outcome 缓存之前生效**——`is_aborted`/`generated_while_paused`/`is_duplicate_user_retry`/`skip_forced_negative_override`/`_metaclaw_training_frozen` 全部排在滞留逻辑之前，滞留不能复活本该丢弃的样本
2. **`_eval_scores` 不能误记判官分**——滞留逻辑插在 `eval_score = opd_result.get(...)` 及其 `_eval_scores.append` **之前**，中间轮次的判官分不会被记成实际训练 reward
3. **"默认行为逐字节不变"改成语义不变**——新增字段和日志后逐字节相同不可能；验收改为：默认 judge 下提交数量相同、每个样本 reward 相同、现有丢弃规则相同、verdict hint 行为相同、非 MetaClaw session 不受影响

### 已实现（2026-08-31，本地验证通过，真实训练未验证）

三层改动，全部在本项目自己的补丁脚本里，官方源码零改动：

- **`prepare_patched_openclaw_combine_select.sh`（`_opd_evaluate`）**：verdict 分支两个 return 都加 `metaclaw_verdict: True`；step-judge 分支加 `metaclaw_round_step: True` + `judge_raw_score`（多数投票原始值，在 truncation/invalid-tool-use 覆盖**之前**捕获）+ `legacy_reward_would_have_been`（覆盖之后的值，**对照旧行为要用这个**）。判分逻辑一行未改。
- **`prepare_patched_openclaw_combine.sh`（`_maybe_submit_ready_samples`）**：模块级读 `METACLAW_MIDROUND_REWARD`（默认 `judge`）；异常分支加 verdict-task-failed 的 discard-and-cleanup；主分发点加滞留/继承/flush 逻辑；循环之后加基础设施失败路径的滞留样本清理。
- **`prepare_patched_openclaw_opd.sh`**：`__init__` 初始化 `self._metaclaw_round`（单个 dict，含 `held`/`outcome`/`verdict_turn`/`state` 四个字段，比三个独立 dict 的生命周期管理更简单）；verdict-signal-skip 分支在 `_fire_opd_task` 实际执行后记录 `verdict_turn`。
- **`run_metaclaw_migration_modelfactory.sh`**：新增 `METACLAW_MIDROUND_REWARD` 声明并传给**训练后端进程**——读这个变量的是被训练进程 import 的代理侧代码，不是 driver，只传给 driver 会静默失效。

**验证**：
1. 三个补丁脚本对**真实官方源码**跑完整补丁链全部成功（所有锚点匹配，其中 combine 四处、combine_select 三处、opd 两处），`py_compile` 通过。
2. 行为测试：从**补丁生成的真实代码**里抽出 `_maybe_submit_ready_samples` 执行，22 项断言覆盖八个场景——judge 模式三项（提交数量/reward/不建状态）、outcome 正常路径四项、判官 task 晚于 verdict 完成的竞态、verdict task 抛异常、verdict 返回无效 outcome、基础设施失败、verdict 在飞时 `force_drop` 不得丢弃、`is_aborted` 与 `_metaclaw_training_frozen` 两个既有门控仍先于滞留逻辑生效。全部通过。
3. **测试非空转**：场景 6（无 verdict 必须丢）与场景 7（verdict 在飞不能丢）**互为反例**——判定条件写死成任一方向都会有一项失败，说明 `verdict_turn is None` 这个判据真的在起作用。

**真实训练完全未验证**——下一步是跑一轮 `METACLAW_MIDROUND_REWARD=outcome` 的消融，主判据见上面"跑完之后看什么"。

### 查证记录（五）：MetaClaw 自己对 min-count 这类"累计欠账"任务的解法路径——我们的迁移缺了一整条腿（2026-08-31）

**起因**：讨论 min-count 累计计数缺陷时，曾否决过"让 agent 回头补写早前欠下的文件"这个方向，理由是"题面从不要求这么做，让 agent 这样做等于往题面塞了一条原本不存在的指令"。用户追问 MetaClaw 自己的 skill/内化机制能不能看到当天历史状态、有没有可能做到回头改旧文件——查证后发现**当初那个否决理由是不完整的**。

**逐条查证结果（读源码，不是推测）**：

1. **agent 物理上完全有能力改早前的文件**。workspace 是**按天**复制的（`_copy_workspace_for_test` 每个 test 调一次），同一天所有 round 共用同一份、跨 round 持久（跨天才隔离，见查证记录二第 3 条）。而 agent 跑的是真实 `openclaw agent`、带完整 `"coding"` 工具画像（read/write/edit/bash）。**所以它随时可以列目录、看到早前那个命名不合规的文件、把它改名或补一个合规的**——min-count 的"欠账"在 MetaClaw 自己的设定里**并非不可恢复**，恢复路径一直存在，只是 agent 得自己想到要走。

2. **skill/memory 机制看不到 workspace，但能间接看到对话里的痕迹**。`skill_evolver.py` 与 `memory/manager.py` 全文 grep 无 `listdir`/`glob`/workspace 路径（唯一的 `os.path` 命中是 SkillEvolver prompt 里的一句示例文本），它们拿到的只有 `{prompt_text, response_text}` 对话文本。但 `prompt_text` 是完整渲染的对话，**包含前几轮的工具调用与工具返回结果**——agent 如果执行过 `ls`、或 checker 反馈进了下一轮的 `[Previous Feedback]`，这些内容就在 evolver 视野里。

3. **存在一条会促使它这么做的通路，而且能跨天累积**。`add_skills` → `_write_skill_md` 把技能写成 `SKILL.md` 落盘，**skill 在整个 run（30 天）内持续累积**，后续每轮通过 `_inject_skills` 按任务描述检索、注入 system message。所以 MetaClaw 有这样一条闭环：

```
day06 反复失败
  → 攒够 N 轮（skill_evolution_every_n_turns）触发 skill 演化
  → 另一个 LLM 读失败对话，总结出一条 skill
     （例如"保存文件前先列目录、确认已有文件命名是否合规"）
  → 落盘、后续轮次自动注入 system message
  → 之后遇到同类任务，模型被提示去检查/修复历史文件
```

**这条通路我们的迁移完全没有**——本项目只搬了 RL 那条线，skill 库明确不迁（见"完整方案"的"不迁移的部分"）。所以在我们的设定里，模型只能靠权重里学到的东西，**没有任何机制能把"回头检查旧文件"这个策略沉淀下来并在后续复用**。

**对已有判断的影响**：

- **不改变 Phase 1 修复的正确性**。Phase 1 改的是"训练奖励怎么算"，让本轮做对的行为不被历史欠账连坐——这在任何解法路径下都成立。
- **改变的是对标关系的理解**：论文 Full 档是 Skills+RL 双通路，我们只有 RL。而 min-count 这类**需要跨轮次策略**的任务，恰恰是 skill 通路擅长、纯 RL 很难学会的——要在权重里学会"先 ls 再决定写什么"，比在 prompt 里读到一条 skill 难得多。**我们可能一直在用一个缺了半条腿的配置去对标 MetaClaw 的结果。**
- **顺带解释了一个此前未解开的观察**：day11-15 的 FC 在训练版和 K=6 冻结版**都是 0**（见 2026-08-28 诊断一节）。此前归因为"`check_metadata.py` 三字段硬要求造成的共同能力墙"，这个说法不错但不完整——那些任务需要的正是"检查已有状态再行动"这类策略性行为，**纯 RL 在这类任务上可能存在真实的能力上限，不只是难度问题**。
- **当初否决"回头修复"的理由需要修正**：MetaClaw 官方设定里 agent 本来就有这个能力，它的 skill 机制也天然会朝这个方向演化——所以"回头修复"**不是外加的作弊，而是 MetaClaw 原本预期的一种解法路径**。当时的否决理由（"等于改了题目在测什么"）说错了；真正站得住的理由只有一条：我们不该由 driver 代替 agent 去做这件事（那才是伪造 workspace 状态）。

**仍未决**：既然纯 RL 对这类任务可能有能力上限，那"完整训满 30 天并超过 K=6"这个目标本身是否现实、还是说必须补上 skill 通路才谈得上对标 Full 档——这个问题此前没有被正面考虑过，需要单独讨论。

### `metaclaw_migration_20260831_154301` 复盘：outcome 消融训崩，根因是全负 batch（2026-08-31）

首次 `METACLAW_MIDROUND_REWARD=outcome` 真实训练**发散**，checkpoint 已污染、不能作为后续起点。CLI 用 `training.log` 逐 step 核实的证据链：

| step | batch reward | 正/负样本 | grad_norm | policy drift |
|---|---|---|---|---|
| 0 | -0.625 | 3 / 13 | 2.03 | 0.02 |
| 3 | +0.125 | 9 / 7 | 4.43 | 0.10 |
| 4 | -0.25 | 6 / 10 | 16.10 | 0.86 |
| 6 起 | **-1.0** | **0 / 16** | 23.84→**2543.9** | 0.31→**21.94** |

同期 K=6 的 judge 模式 batch reward 全为正、`grad_norm` 约 1.4-3.3、policy drift 稳定在 0.05-0.1。行为退化的时间点也对得上：step 3/4 之后首次出现 fresh session 的 `NO_REPLY`，step 6 起全 -1，之后响应长度从约 1370 掉到 160。

**机制在源码层面确认（不是推测）**：

```python
# slime/utils/ppo_utils.py::get_grpo_returns
returns.append(torch.ones_like(kl[i]) * rewards[i])   # 原始 reward 直接广播到每个 token
```

配上官方脚本的 `--n-samples-per-prompt 1` + `--disable-rewards-normalization`（我们的启动脚本原样继承），**advantage 就等于原始 reward，没有任何组内归一化**。一批 16 个全 -1 时，每个 token 的 advantage 都是 -1——只把模型刚产出的一切往下压、没有任何东西被往上推。

**为什么 outcome 模式必然导致全负 batch**：round 真实通过率只有约 17%，而 judge 模式下中间步骤约 69% 是 +1。**step judge 一直是这套训练里唯一稳定的正信号来源**，换成忠实反映成败的信号后正样本就塌了。而且失败 round 往往轮次更多（模型在乱试），所以**样本层面的负样本占比比 round 通过率还要低**——这是个放大效应。

**一个重要推论**：用户最初设想的"中间步骤完全不进训练、只留最终轮"（V1）**有同样的问题**，因为最终轮的通过率同样是 17%，而且每批只有 16 个样本、更容易凑出全负批。**不是思路不对，是它和 outcome 版共享同一个致命前提。**

**同时查实的两个独立缺陷（已修复，commit `2944f87`）**：
1. **硬负分被覆盖**：`day02/r2/turn2` 日志链完整——`invalid-tool-use-penalty ... eval_score 1.0 -> -1.0`，随后 `midround-reward ... inherited_outcome=1.0 ... legacy_reward_would_have_been=-1.0`，最终 `submitted ... index=24 score=1.0`。重复/退化的工具调用被正向强化。
2. **结构性无效工具调用首发漏检**：`day05/r10` 嵌套 JSON 转义出错 → SGLang 报 `Failed to parse JSON part` → validation error 进上下文 → 模型开始反复输出 `{"name": "write", "arguments": {}}`；`day08/r1` 连续 17 次、单个 round 产出 23 个训练样本跨多个 batch。首发那一次 `is_invalid_tool_use=False`（规则 1a 只能拦第二发起的逐字重复），而规则 4 整段只对 Personal Agent Track 生效，MetaClaw 这边一条规则都没覆盖。

**注意**：这两个修复本身正确且必须做，但**都会让负样本变多**，对全负 batch 是雪上加霜。**在 reward 方差问题解决之前，它们不构成"可以重新提交训练"的条件。**

### 查证记录（六）：`toolcall-rl` 与 MetaClaw 各自怎么处理中间步骤——我们两边的关键细节都没照搬（2026-08-31）

此前"三方对照"那张表把我们的步骤判官记成"刻意对齐 toolcall-rl"，逐字读两边源码后发现**这个说法不准确，而且掩盖了真正的风险差异**。

**`toolcall-rl`（`generate_with_retool.py`）：整条轨迹一个 sample，中间步骤分是"加"在 outcome 上的**

```python
loss_masks += [1] * len(cur_response_token_ids)          # 模型生成 → 训练
loss_masks += [0] * len(obs_tokens_ids)                  # 工具返回 → 不训练
sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)  # 观察段补零占位
assert len(response_token_ids) == len(sample.rollout_log_probs)   # 显式断言长度对齐
sample.tokens = prompt_tokens_ids + response_token_ids
sample.response_length = len(response_token_ids)         # response 段横跨整条轨迹
```

奖励合成（`reward_func`，856-893 行）：

```python
final_score = base_score + prm_step_coef * prm_step_mean
```

**关键差异：中间步骤分不替代 outcome，而是整条轨迹取平均后加一个标量微调**，`base_score`（确定性 `\boxed{}` 对错）永远在场当锚点。**单个步骤判错会被平均稀释**。

另有两处我们完全没有的机制：
- `if result["score"] < 0: tool_call_reward = (num_turns - 2)/2 * 0.1; score = min(-0.6, score + tool_call_reward)` —— **失败时按工具调用次数给补偿，且负分下限钳在 -0.6**，不是干脆的 -1
- `if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]: break` —— **硬性轮次上限**，坏轨迹不可能无限膨胀

**MetaClaw（`api_server.py::_submit_turn_sample`）：每次 LLM 调用一个 sample，靠 `loss_mask` 整段开关**

```python
exclude = not has_next_state or score == 0.0
loss_mask = [0] * len(response_ids) if exclude else [1] * len(response_ids)
```

没有轨迹概念、不分中间/最终。但有一个我们没有的设计：**`score == 0.0` 的样本整段 `loss_mask=0`、不参与训练**——`prm_scorer` 是 `+1/-1/0` 三档，平票或全部解析失败即 0，**"判官拿不准"的样本被自动排除，不会硬塞一个方向进训练**。配套还有 `at-least-one guarantee`：整个 session 一个有效样本都没有时破例保留一个。

**对我们的三点启示**：

1. **"刻意对齐 toolcall-rl"这个说法要修正**。同样是二档 ±1，toolcall-rl 的步骤分是**平均后加权加到 outcome 上**（判错被稀释），我们是**每个中间步骤直接拿自己那份 ±1 当完整 reward**（判错原封不动进训练）。**风险等级完全不同。**
2. **两边都不会让一个 round 产出十几二十个独立样本**：toolcall-rl 结构上不可能（一条轨迹一个 sample）+ `max_tool_calls` 上限；MetaClaw 靠 `score==0` 大量排除。**我们两条都没有**——这正是全负 batch 的直接成因。
3. **toolcall-rl 的负分有下限且失败时给补偿**，显然预料到了"失败轨迹占多数"；我们是干脆的 -1，没有缓冲。

### 方案：两条候选路线，分别实验（2026-08-31，待实现）

两条都值得试，**分别跑、不要合在一起**，否则出了结果无法归因。

#### 路线 A：轨迹级样本（照搬 toolcall-rl 结构）

一个 round 只提交一个 sample，最终 checker 结果作为整条轨迹的 reward。

**能解决**：中间"只调工具不产生结果"的步骤不再单独拿 +1；一个失败 round 从产出十几二十个 -1 压缩成 1 个；样本数不再被"模型乱试了多少轮"加权。全负 batch 的概率从必然降到约 `0.83^16 ≈ 5%`。

**必须完成的六件事**（CLI 核实，`rollout_log_probs`/`teacher_log_probs` 在 `megatron_utils/actor.py:437-455` 都按 `response_length` 切片，长度必须严格对齐）：
1. 保存每一轮模型生成片段及其 token span
2. **重新定义 prompt/response 切分**——prompt 只留第一条 user 消息，其后全部算 response（当前 `loss_mask` 长度天生等于最后一轮 response，**物理上无法覆盖更早片段**，不是"调 loss_mask 就够了"）
3. `loss_mask` 只在 assistant 生成片段置 1，工具返回/用户消息/系统提示置 0
4. 拼接逐轮 `rollout_log_probs`，非生成段补零（照 toolcall-rl 加长度断言）
5. **OPD/top-k 路径要重新处理整条序列的 teacher token/logprob**——这是最大的岔路，见下
6. 每个 round 最多一个 sample，重复 invalid call 不能继续产出几十个

**最大的未决点：OPD 怎么办**
- 轨迹样本做成 **RL-only** → 实现最简单，但**等于把 Hybrid RL 的 OPD 那一半在 MetaClaw 上整个砍掉**。而 OPD 恰恰是本次迁移要验证的核心机制（三方对照表：MetaClaw 和 toolcall-rl 都没有 OPD，是 Personal Agent Track 独有、我们特意保留的）。砍掉它，实验就不再是"迁移 Hybrid RL"。
- 轨迹样本**走 OPD** → 保住方法完整性，但要对整条轨迹重新做 hint 注入 + tokenize + teacher logprob 计算，工作量和出错面明显更大。

**已知局限**（CLI 指出，接受为残留）：能解决"失败 round 的中间工具调用拿正奖励"，但**成功轨迹里的无关工具调用和长 thinking 仍会整体拿 +1**；若 checker 因历史文件残留误判成功，整条坏轨迹仍被正向训练。所以硬负分优先级、malformed/重复调用上限这些仍要保留。

**实现前必须先做可行性验证**：拿真实日志的一个多轮 round，实测能否正确重建 token 序列、拼出长度对齐的 logprobs。验证不通过就不要动手。

#### 路线 B：toolcall-rl 式奖励合成（不动样本结构）

保持现有逐轮次样本结构，只改奖励合成方式：

- 中间步骤 reward 从"各拿一份完整 ±1"改为 **`outcome + coef × 该轮判官分`**——outcome 提供方向，判官分只作微调不作主导（对齐 toolcall-rl 的 `base_score + prm_step_coef * prm_step_mean`）
- 或更接近 MetaClaw：**判官分为 0（拿不准）的中间样本直接 `loss_mask=0` 排除**（需要先把判官从二档改成三档）

**优点**：完全不碰 token 切分、logprobs、OPD 任何一处，代价远小于路线 A，同时缓解"判官判错直接进训练"和"全负 batch"两个问题（outcome 给方向、判官分给方差）。

**缺点**：不解决"一个失败 round 产出十几二十个样本"——需要配合独立的样本数上限/重复调用去重。

#### 两条路线共同的前置项

无论走哪条，这些都要有（部分已完成）：
- [x] 硬负分优先级（commit `2944f87`）
- [x] 结构性无效工具调用检测，规则 6（commit `2944f87`）
- [ ] 单个 round 的样本数上限 / 重复无效调用去重
- [ ] 轨迹超 context 上限的处理
- [ ] 每次实验都从**干净 base** 起步（`20260831_154301` 的 checkpoint 已污染）

#### 查证结果（2026-08-31）：靠打开归一化来救全负 batch 这条路走不通，且它改变了两条路线的优先级

原本记为"仍未决"的那个问题（`--disable-rewards-normalization` 去掉会怎样）已经查完，答案是**否定的，而且开了会让训练彻底停摆**。

**决定性事实：我们每个样本都是独立的一组。**

```python
# openclaw_combine_api_server.py:88-89 / 133-134（opd 版同构）
sample.index = next(self._index_counter)
sample.group_index = next(self._group_counter)      # 每个样本一个全新 group_index
await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))   # 组里只有它自己
```

归一化对单元素组是恒等于零的：

```python
# slime/ray/rollout.py::normalize_vals
vals = vals - vals.mean()              # 长度 1 -> 恒为 0
if std_normalization:
    if len(vals) > 1: vals = vals / (vals.std() + 1e-6)
    else: vals = torch.zeros_like(vals)   # 长度 1 -> 直接置 0
```

`dynamic_history` 与非 `dynamic_history` 两条分支都走同一个 `normalize_vals`，结果一致——**开启归一化会把每个 reward 都变成 0，训练信号整个消失**。

还有第二道（同样只在归一化开启时才生效，目前是惰性的）：

```python
# slime/ray/rollout.py::_drop_constant_reward_groups
if max(vals) - min(vals) <= 1e-12:
    constant_groups.append(group_idx)
```

单元素组的 `max-min` 恒为 0，**每一组都会被判成"常数组"、整批丢弃只保留一组**。

**结论**：`--disable-rewards-normalization` 不是可调选项，是这套架构下的必需项。官方脚本这么设，是因为 `--n-samples-per-prompt 1` + 一样本一组的结构下，组内根本没有方差可归一。**全负 batch 无法靠配置解决，只能靠改变 reward 分布本身。**

**顺带修正上一节对 toolcall-rl 的一处误读**：`min(-0.6, score + tool_call_reward)` 此前被记成"负分下限钳在 -0.6"，**方向说反了**。`min` 取小值，实际是**封顶**——失败样本最好也只能到 -0.6，不会更高。真实效果是"按工具调用次数把 -1 抬高到最多 -0.6，但绝不让它变正"：`num_turns=2` 时补偿为 0、得分 -1；`num_turns>=10` 时补偿把 -1 抬到 -0.6 后封顶。**这个设计的作用恰恰是让失败样本的幅度产生差异，不再是清一色的 -1。**

**因此两条路线的优先级要反过来**（此前把 A 当主方案、B 当便宜替代，这个排序错了）：

- **路线 B 才是对症的那个**：`outcome + coef × 判官分` 天然产生连续分布，再配上 toolcall-rl 那套"按工具调用次数抬高失败分 + 封顶"，负样本的幅度就有了差异，不再是 16 个一模一样的 -1。
- **路线 A 单独做解决不了这个问题**：它把一个失败 round 从 20 个 -1 压成 1 个 -1，降低的是负样本**数量**，但每个样本仍是干脆的 ±1、通过率仍是 17%。全负批概率从必然降到约 5%，是改善不是解决。

**执行顺序改为：先做路线 B，再视结果决定要不要做路线 A。**

### 查证记录（七）：GRPO 的"组"到底是什么——我们用着 GRPO 估计器却一个组都没有（2026-09-01）

**这一节推翻了上一节"路线 B 才是对症的那个"这个判断。** 路线 B 已实现（`blend` 模式，commit `fa23472`），但查清 GRPO 分组机制后确认：**它只提供幅度差异，不产生正样本，对全负 batch 是缓解不是解药。**

**slime 标准数据路径里，"组"是同一 prompt 的多次采样**（`slime/rollout/data_source.py:102-111`）：

```python
for prompt_sample in prompt_samples:
    group = []
    for _ in range(self.args.n_samples_per_prompt):
        sample = copy.deepcopy(prompt_sample)          # 同一个 prompt
        sample.group_index = self.sample_group_index   # 同一个 group_index
        group.append(sample)
    self.sample_group_index += 1                       # 换下一道题才换组
```

**三方的分组方式对照（全部读源码确认）**：

| | 组是什么 | 配置 | 全负 batch 会怎样 |
|---|---|---|---|
| **toolcall-rl** | 同一道题独立采样 8 次 | `--n-samples-per-prompt 8`，`--rollout-batch-size 32`，`grpo` | 组内 8 次有对有错 → 减组均值后必有正负；全同的组被 `_drop_constant_reward_groups` 整组丢弃 |
| **MetaClaw** | **没有组**，整批一起归一化 | 自己的 `compute_advantages` | **减批均值后必有正负，问题结构性不存在** |
| **我们** | **每个样本自成一组** | `--n-samples-per-prompt 1` + `--disable-rewards-normalization` | **advantage = 原始 reward，全负 = 所有 token 一起被压，无任何强化** |

**MetaClaw 的实现**（`metaclaw/data_formatter.py:217-230`）：

```python
def compute_advantages(batch: list[ConversationSample]) -> list[float]:
    """Centre-and-scale rewards within the batch (GRPO style: (r - mean) / (std + eps))."""
    rewards = [s.reward for s in batch]
    mean_r = sum(rewards) / len(rewards)
    std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
    return [(r - mean_r) / (std_r + 1e-8) for r in rewards]
```

注释自称 "GRPO style"，但严格说这是**批级基线**（REINFORCE with baseline）——GRPO 要求组内是同一 prompt 的多次采样，这里是不同任务的样本混在一起归一化。**但正因如此，它对"整批都是负 reward"免疫**：减去批均值后，比均值好的样本必然拿到正 advantage。

**结论：我们是三者里唯一会崩的配置——用着 GRPO 的估计器，却一个真正的组都没有。** `--n-samples-per-prompt 1` 不是随便设的：我们的"一次采样"是真实 agent 跑一遍、会改真实 workspace，要采 8 次就得开 8 份独立 workspace，而且 8 次跑完状态各不相同，当天后续轮次不知道该接哪一份。这是真实的架构障碍。

#### 两条候选路径（待查清后选择，均未实现）

**路径 ①：学 toolcall-rl —— `n_samples_per_prompt > 1`**

要解决"8 份 workspace 并行 + 之后接哪一份"的架构问题。**初步判断代价过大、可能不现实**，但尚未正式评估过，先记下不排除。

**路径 ②：学 MetaClaw —— 批级基线**

让同一批样本共享 `group_index`，再打开归一化，等价于减批均值。**对我们架构改动最小，且有官方先例**（MetaClaw 自己就这么做，而我们本来就是在迁移 MetaClaw 的场景）。

已知的一个坑（**尚未查完**）：`_drain_output_queue` 是 `completed_groups[group_id] = group`，**共享 group_id 会让后来的样本覆盖先来的**，得先改成累加；还没查有没有别的地方假设了 `group_index` 唯一。

**两条路径都要先查清可行性再选，不要先动手。**

#### 任务形态对照：为什么 `n_samples_per_prompt=8` 在 toolcall-rl 可行、在我们这里不可行（2026-09-01）

查了 `toolcall-rl/README.md`、`tool_sandbox.py::TOOL_CONFIGS`、`generate_with_retool.py::execute_predictions` 之后确认，**差别是结构性的，不是工程量问题**。

toolcall-rl 的任务：**数学题 + Python 代码解释器**（基于 ReTool）。数据集 `DAPO-Math-17k`，一道题一个 prompt，标准答案是 `\boxed{}` 里的字符串；模型写代码 → 沙箱执行 → `<interpreter>结果</interpreter>` 回填上下文 → 继续推理 → 给答案。`max_turns: 16`、`max_tool_calls: 16`、工具只有一个 `code_interpreter`。

| | toolcall-rl | 我们（MetaClaw-Bench） |
|---|---|---|
| 一次采样是什么 | 跑一遍代码解释器沙箱 | 真实 `openclaw agent` 跑一遍、**读写真实 workspace 文件** |
| 采样之间的关系 | **完全独立**，沙箱用完即弃 | **同一天的 round 共享 workspace**，前一轮写的文件后一轮看得见 |
| 重复采样 8 次 | 开 8 个沙箱并发，互不干扰 | 要开 8 份独立 workspace；**跑完状态各不相同，当天后续轮次不知道该接哪一份** |
| 任务之间 | 每道数学题彼此独立 | **day01→day30 严格顺序**，后面的天依赖前面训出的权重（论文的在线学习假设） |
| 正确答案 | 唯一的 `\boxed{}` 字符串 | checker 判 workspace 状态，且部分是累计跨轮次计数 |

**核心矛盾**：GRPO 要求"同一 prompt 多次采样做相对比较"，这个前提要求**多次采样必须可独立、可丢弃**。数学题天然满足——8 次尝试互不影响，选谁都行。我们的任务**结构上违反这个前提**：round 不是"一道独立的题"，而是"30 天连续任务流里的一步"，它的产出（文件）就是下一步的输入。采样 8 次要么让 8 条时间线全部并行下去（成本 ×8 且发散成 8 个不同的"当天历史"），要么选一条丢掉 7 条——但被丢掉的 7 条正是 GRPO 需要的对照组，而选中的那条会把 workspace 定死。

**结论：路径 ① 基本可以排除**——不是工程量大，是任务形态不支持。这也解释了官方 `openclaw-combine` 脚本本来就设成 1，不是我们改小的。

**反过来，这让路径 ② 更有说服力**：MetaClaw 自己面对的就是同一种任务形态（30 天连续、有持久 workspace、无法重复采样），它给出的解法正是"放弃组内比较、改用批级基线"。**这不是权宜之计，是同类任务下的合理设计。**

顺带补上"我们是不是在用缺半条腿的配置对标 Full 档"那个问题的一块：**我们不只缺 skill 通路，连 advantage 的算法都跟 MetaClaw 不一样**——它有批级基线，我们退化成了原始 reward。

**关于已实现的 `blend`（路线 B）**：它照搬的是 **toolcall-rl** 的奖励合成方式（`base_score + coef × step_mean`）。但 toolcall-rl 之所以能靠这个工作，前提是它**同时**有 8 样本组内归一化；把它的奖励形态单独搬过来、却没有它的分组机制，等于只搬了一半。**若路径 ② 落地后证明 `blend` 是多余的或有干扰，应当回退**——它默认关闭（`METACLAW_MIDROUND_REWARD=judge`），回退成本只是删掉一个分支。

### 方案（待 CLI 在真实环境查证，未实现）：批级基线，对齐 MetaClaw 的 `compute_advantages`（2026-09-01）

**目标只有一个**：让 advantage 不再等于原始 reward，从而消除"全负 batch = 所有 token 一起被压、没有任何动作被强化"这个必然崩溃的结构。这是路径 ②。

#### 为什么这个方案比预想的干净得多

最初以为要"让同一批样本共享 `group_index` + 打开归一化"，并担心 `_drain_output_queue` 的 `completed_groups[group_id] = group` 会让共享 group_id 的样本互相覆盖。**查证后发现这条路根本不用走**——slime 自带一个正好合用的钩子：

```python
# slime/ray/rollout.py:339-341
def _post_process_rewards(self, samples):
    if self.custom_reward_post_process_func is not None:
        return self.custom_reward_post_process_func(self.args, samples)   # 最顶部短路
    ...默认的按 group_index 归一化...
```

由 `--custom-reward-post-process-path` 注册（`arguments.py:1342`）。调用点（`rollout.py:659`）：

```python
raw_rewards, rewards = self._post_process_rewards(samples)
assert len(raw_rewards) == len(samples)
train_data["rewards"] = rewards          # advantage 的来源
train_data["raw_reward"] = raw_rewards   # 仅用于日志/指标
```

**所以只要挂一个自定义函数，就能直接实现批级基线——不碰 `group_index`、不碰队列、不碰任何现有补丁。** 之前担心的覆盖问题不会遇到。

#### 三个前提已逐条核实

1. **钩子拿到的 `samples` 是完整一批的扁平列表** —— `_get_rollout_data`（`rollout.py:253-256`）里 `while isinstance(data[0], list): data = list(itertools.chain.from_iterable(data))`，分组结构在进入转换前就被拍平，正是批级基线需要的粒度。
2. **可以保留 `--disable-rewards-normalization`，两者不冲突** —— 自定义钩子在 `_post_process_rewards` 最顶部短路，根本不看 `rewards_normalization`；而 `_drop_constant_reward_groups` 的早退条件是 `if advantage_estimator not in ["grpo","gspo"] or not rewards_normalization: return samples`。**保持归一化关闭 → 不触发"每个单元素组都被判成常数组、整批被丢"的陷阱，同时仍能通过钩子拿到基线。**
3. **reward 取值链路对得上** —— `Sample.get_reward_value` 是 `self.reward[args.reward_key]`，官方脚本设了 `--reward-key score`，我们提交时写的是 `sample.reward = {"score": float(eval_score)}`。

#### 实现形态

新增一个函数（放在 `scripts/metaclaw/` 下的新文件，不改任何现有文件）：

```python
def metaclaw_batch_baseline(args, samples):
    """对齐 MetaClaw 自己的 metaclaw/data_formatter.py::compute_advantages。

    返回 (raw_rewards, rewards)：raw 用于日志，rewards 用于算 advantage。
    """
    raw = [s.get_reward_value(args) for s in samples]
    mean_r = sum(raw) / len(raw)
    std_r = (sum((r - mean_r) ** 2 for r in raw) / len(raw)) ** 0.5
    return raw, [(r - mean_r) / (std_r + 1e-8) for r in raw]
```

训练脚本加一行 `--custom-reward-post-process-path <该函数的导入路径>`。**改动量：一个新文件 + 一行参数。**

#### CLI 真实环境查证结果 + 本地复核（2026-09-01）

五点全部有结论，其中三点我在本地源码复核过：

| # | 问题 | 结论 | 谁验证的 |
|---|---|---|---|
| 1 | `load_function` 路径格式 | **点分路径**（`path.rpartition(".")` + `importlib.import_module`），不是 `module:func` | **本地复核确认**；同脚本里 `--custom-loss-function-path openclaw_topk_select_loss.openclaw_topk_select_loss_function` 就是同款先例 |
| 2 | dummy 是否污染均值 | **会**。`_make_dummy_samples` 在 `_post_process_rewards` 之前注入（`rollout.py:648-659`），reward=0.0 | **本地复核确认**。补充：dummy 自己 `loss_mask=[0]` 不产生梯度，危害是**它的 0.0 把批均值拉偏、扭曲真实样本的 advantage**；又因 `_drop_removed_samples` 先跑，此刻 `remove_sample=True` 的只剩 dummy |
| 3 | 真实 batch 大小 | 稳定 16，噪声可接受；`ACTOR_GPUS=4 + TP=4 → dp_size=1`，真实 run 几乎不会出现 dummy（但排除逻辑仍要写） | CLI（需日志） |
| 4 | std≈0 | 可接受且优于全压：advantage 全 0 = 这批不更新。MetaClaw 同款。**建议打 warning 而不是另开分支** | CLI 判断，采纳 |
| 5 | 与 `step_wise` 冲突 | 无冲突，训练用 `--advantage-estimator grpo` | **本地复核确认** |

**CLI 的主判据更正是对的，而且我原来那条是明确的错误**：我写的"batch reward 不再出现 0/16"站不住——**批级基线根本不改 reward，只改 advantage**，日志里的 `0/16` 照旧。

**但我不同意 CLI 的实验设计推荐**（它建议先跑 `judge` + 基线作为"干净对照"），有两条理由：

1. **judge 模式本来就没发散**。K=6 那次 `grad_norm` 1.4-3.3、drift 0.05-0.1 一切正常，发散只发生在 outcome 模式。而基线的核心机制恰恰是抢救全负 batch，在几乎不会出现全负 batch 的模式里跑它，**这个机制大概率一次都不触发**。
2. **二值奖励下，批中心化在同号样本之间不产生任何区分度**——这一条比 CLI 讲的"测不到 thinking 膨胀"更根本。中心化只移动零点、缩放幅度，**所有 +1 拿到同一个正 advantage**。举例 69% 为 +1 时：mean=0.38、std≈0.925 → `+1 → +0.67`、`-1 → -1.49`，效果是**稀有的负样本被放大到正样本的 2.2 倍**（标准的基线行为，强调意外结果），但不会区分 +1 里哪个是长 thinking。

**因此改为直接跑 `outcome` + 批级基线**：那才是真正发散过的配置，基线的抢救机制会被真实触发，而且能干净回答"advantage 退化是不是 outcome 崩盘的主因"。

#### 已实现（2026-09-01，本地验证通过，真实训练未验证）

- **`prepare_patched_openclaw_combine.sh`**：额外生成一个独立模块 `metaclaw_batch_baseline.py` 到 `DEST_DIR`。之所以是独立模块而不是打补丁进官方文件——slime 是按 import 路径加载它的，而且它替换的是整个函数而非编辑某一行。`DEST_DIR` 就是 `PATCHED_COMBINE_DIR`，已被训练启动脚本前置到 `PYTHONPATH`，所以导入路径就是 `metaclaw_batch_baseline.metaclaw_batch_baseline`。
- **`run_openclaw_topk_select_modelfactory.sh`**：`METACLAW_MIGRATION_PROFILE=1` 分支里注入 `--custom-reward-post-process-path`，带幂等判断（`grep -q` 避免重复注入）和失败即报错。**只在 MetaClaw 迁移场景加，Personal Agent Track 不受影响**——那边的 reward 分布没有"通过率 17%、整批同号"这个问题。
- **`--disable-rewards-normalization` 保持不变，不要去掉**（注释里写明了原因，避免以后被人"顺手清理"）。

**验证**：两个脚本 `bash -n` 通过；补丁链对真实官方源码跑通、模块正确生成；注入后的脚本语法正确且位置落在 `TOPK_SELECT_ARGS` 数组内、紧邻 `--advantage-estimator grpo`。18 项断言覆盖：全 -1 批 → advantage 全 0（不再是全压）、3/13 混合批产生正负两号、多数为正时稀有负样本幅度更大、dummy 拿零 advantage 且**不影响真实样本的 advantage**、`remove_sample` 无 metadata 标记时也被排除、单样本/全 dummy 批不崩、**连续 reward（blend）能在同号内产生区分度而二值不能**（把上面那条限制变成可执行的断言）、以及 slime 会 assert 的返回长度契约。

**`blend` 的去留**：暂时保留，理由不是"留着以防万一"，而是上面那条实测——**批级基线只有在 reward 连续时才能在同号样本间产生区分度，而 `blend` 是唯一的连续来源**。两者是互补的，但**不要同时跑**（无法归因）。等批级基线单独跑出结果，再决定要不要叠加或删除。

> **2026-09-02 更正：这一段的结论已作废，`blend` 已删除。** "reward 连续才能在同号内产生区分度"这条观察本身是对的，但当时只看到它的好处、没算它的代价：连续化**同时**作用在负样本上，于是一个**整轮全失败**的 round 也有了 reward 方差（-1.3/-1.0/-0.7），批级基线随即为它算出真实 advantage，其中被判官认可的那一步拿到 **+1.121**——从一条彻头彻尾失败的轨迹里正向强化了一步。这直接摧毁了基线最值钱的那条安全性质（全负批 → advantage 全 0 → 这批不更新）。区分度要，但只能加在**正样本**上，见下一节。



#### 原始待查证清单（已全部有结论，保留供追溯）

1. **`--custom-reward-post-process-path` 的路径格式** —— `load_function` 期望什么形式（`module.path:func` 还是 `module.path.func`？文件路径还是可导入模块名？），以及这个文件在 modelfactory 上要放在哪、`PYTHONPATH` 是否能解析到。
2. **`samples` 里是否可能混入 dummy/removed 样本** —— 调用点之前有 `_drop_removed_samples`、`_make_dummy_samples`（`rollout.py:648-657`，当样本数不足 `dp_size` 时会注入 dummy）。**dummy 样本的 reward 会不会污染批均值**，需要确认；若会，函数里要排除 `metadata.get("dummy_removed_sample")`。
3. **批大小是否足以支撑基线** —— `--rollout-batch-size 16`，但实际到达钩子的样本数会被 `disable_rollout_trim_samples`/`global_batch_size` 逻辑调整，真实每批多少个样本需要实测。**样本太少时批均值噪声会很大**。
4. **std 接近 0 的退化情况** —— 若一批 reward 恰好全相同（例如全 -1），`std_r ≈ 0`，`(r-mean)/(0+1e-8)` 会得到 0/1e-8 = 0，结果是全 0 advantage（等于这批不训练）。**这是否是可接受的行为**，还是应该像 MetaClaw 那样保留、或加一个"全同批直接跳过"的显式分支，需要判断。
5. **跟 `step_wise` advantage estimator 是否冲突** —— 官方脚本用的是 `--advantage-estimator grpo`，但 `_post_process_step_wise_rewards` 是另一条独立路径（`rollout.py:676`），确认我们不会同时踩到。

#### 实验设计

**跑 `METACLAW_MIDROUND_REWARD=outcome` + 批级基线，不叠加 `blend`。** 选 `outcome` 而不是 `judge` 的理由见上面对 CLI 实验设计的异议：只有 `outcome` 是真正发散过的配置，基线的抢救机制才会被触发。

**主判据（reward 侧的计数不会变，别盯它）**：
1. **`[metaclaw-batch-baseline]` 日志行**——这是基线是否生效的唯一直接证据：`reward_mean`/`reward_std`、advantage 的 `min/max/pos/neg`。全 0 方差时会打 warning
2. **`grad_norm` 与 `train_rollout_logprob_abs_diff`**——对照 K=6 的 1.4-3.3 / 0.05-0.1，而不是 outcome 那次的 2543.9 / 21.94
3. Acc./Compl. 是次要参考

**必须从干净 base 起步**（`20260831_154301` 的 checkpoint 已污染）。

#### 这一轮走过的弯路（留作记录，避免重复）

按时间顺序，这个问题被诊断了四次，前三次都不对或不完整：

1. **"中段 step-judge 跟最终结果脱钩"** → 已撤回（2026-08-25），中段 +1 本身是合理的过程 shaping
2. **"累计计数缺陷导致正确做法被判负"** → 是真实缺陷、已修（Phase 1），但**不是训崩的原因**
3. **"中间步骤判官奖励长 thinking"→ 做 outcome 消融** → 实现了，但**一跑就发散**，因为它把唯一的正信号来源拿掉了
4. **"改奖励合成形态"→ 做 `blend`** → 实现了，但查清 GRPO 分组后发现**它只给幅度差异、产生不了正样本**，对全负 batch 是缓解不是解药

**真正的根因直到第五次才找到**：我们用着 GRPO 估计器却一个真正的组都没有，advantage 退化成原始 reward。**前四次都在改 reward 的值，而问题出在 reward 到 advantage 的那一步。**

已实现但可能多余的东西：`blend` 模式（默认关闭，回退成本低）。Phase 1 的 diff 判定、硬负分优先级、规则 6 都是独立成立的真实修复，与本方案无关，保留。

#### 三个方案的重新排序

| 方案 | 能否产生正样本 | 状态 |
|---|---|---|
| 路径 ②（批级基线） | **能**（减批均值后必有正负） | 待查证可行性 |
| 路径 ①（多次采样） | 能（组内有对有错） | 架构障碍大，待评估 |
| 路线 A（轨迹级样本） | 不能，只降低负样本数量 | 未实现 |
| 路线 B（`blend`，已实现） | **不能**，只提供幅度差异 | 已实现，单独跑大概率仍会崩 |

**`blend` 保留价值**：它做的是相对塑形（让模型知道哪些步骤"没那么糟"），跟基线方案正交，可以叠加使用；但**不应指望它单独解决全负 batch**。

> **2026-09-02 更正：`blend` 已删除**，理由见上一条更正框（它给全失败 round 制造方差，反而让失败轨迹里的某一步被正向强化）。

### 2026-09-02：目标重定为"跑满 30 天且不训坏"之后的三项改动

#### 目标的变化（这一节是后面所有取舍的前提）

在此之前，每一轮改动的隐含目标都是"让这一批的训练信号更对"。用户明确重设了目标：**跑满 30 天，拿到比"只训到 day6"更高的总分**。随之而来的两条判据跟之前完全不同：

- **"有效"是相对于"只训到 day6"说的**，不是绝对值。后面的天数客观上更难，训练效果一定会变差，这本身可以接受。
- **真正的要求是不能因此把模型训坏**——整体持续变好，哪怕只好一点点也行；绝不能因为题目太难就崩到无法作答。

这直接推翻了我在上一轮写下的一条判断。当时我算出"44% 的批次 advantage 全 0、不产生更新"并把它当缺陷报了出来；按新目标看，**全负批 → advantage 全 0 恰恰是想要的安全性质**：难题不会伤害模型，只是不推进。这条性质从此是硬约束，后面每一项改动都要先证明自己不破坏它。

参照点是"只训到 day6"那次（judge 模式，`grad_norm` 1.4-3.3、drift 0.05-0.1，全程健康）和 `20260827_163030`（同样 judge 模式，跑到 day22，**没有崩，但中途结果不升反降**）。用户对后者的评价是"其实已经很接近成功了"——所以方向不是推倒重来，而是把它中途变差的那个具体机制堵上。

#### 诊断：两次失败是同一个病，形态不同

`20260827_163030`（judge 模式）：判官持续认可越来越长的 thinking → day17 thinking 从 18k 冲到 115k → 生成到长度上限也吐不出收尾的 `\boxed{}` → MC 题 day16-22 的格式失败 17/26（对照 day1-6 那次是 0/27）。**模型没有变笨，是变啰嗦到答不完。**

`20260902_094458`（outcome 模式 + 批级基线）：day06-r7 空转 186 轮（没有轮数上限）→ outcome flush 一次性倒出 186 个 `-1` → 该批变成 1 正 15 负 → 基线（当时还在除 std）给了那唯一的正样本 **3.873** 的 advantage —— 而那个正样本恰好是 day06-r6，一次 `response_len=10273` 的**啰嗦的成功** → step 8 `grad_norm=3.69` → day06-r8 thinking 直接到 120945。

两次的共同点：**二值 reward 区分不了"答对且简洁"和"答对但啰嗦"**，而 `loss_mask` 覆盖整段 response（含 `<think>`）。只要有任何机制放大了一次啰嗦的成功，模型就会滑进啰嗦的状态回不来。

#### 改动一：删除 `blend`

见上面两条更正框。它的实测行为破坏了"全负批不更新"这条安全性质，与新目标直接冲突。

#### 改动二：长度感知的正奖励（`_metaclaw_length_aware_reward`）

`score = reward × (1 - 0.9 × clip((L - L0)/(L1 - L0), 0, 1))`，默认 `L0=6000`、`L1=16000`、地板 0.1。

几个刻意的设计取舍：

- **只对 `reward > 0` 生效，负样本一律保持平坦的 -1。这一条是承重的，不是省事。** 让失败样本也随长度变化，就等于重新造了一个 `blend`——全失败 round 重新有了方差，最不糟的那一步会被正向强化。测试里把这一点写成了可执行断言（`[the blend trap must stay closed]`）。
- **地板取 0.1 而不是 0**：答对但很长仍然是答对，只是不如简洁的答对；不能让它跌到零，更不能过零。
- **单位是响应 token 数，不是 thinking 字符数**。token 才是 `loss_mask` 和 `response_length` 实际覆盖的东西；两种单位混用，日志侧的数字会悄悄跟训练侧对不上。
- **阈值取自真实分布**：只训到 day6 那次（健康）正样本 `response_len` 中位数约 2.5k、p90 约 5.3k、最长约 9k；`20260827`（漂移中）中位数 3.3k、p90 6.9k、最长 13k。`L0=6000` 刻意卡在两条 p90 之间——健康区间原样不动，只惩罚漂移区间。**代价是明确接受的**：一次健康但偏长的 9k 成功会拿 0.73 而不是 1.0。
- **落点在两个 `_submit_*_turn_sample` 里，而不是 outcome 模式的分派逻辑里**。分派逻辑只在 `METACLAW_MIDROUND_REWARD=outcome` 下运行，而下一次跑的是 judge 配置；挂在提交函数上则覆盖所有模式的所有正样本（中间步骤、outcome 继承、最终 verdict 轮一视同仁），且不需要维护任何按模式分支的逻辑。那里也是唯一能拿到真实 `response_length` 的地方。
- **不做开关**，只暴露 `METACLAW_LEN_DECAY_L0`/`L1` 两个阈值。它是防训坏补丁，不是可选对照。
- **官方的 Acc./Compl. 判分完全不动**，这只塑形训练侧 reward。

#### 改动三：批级基线改成只减均值、不除 std

`advantage = r - mean`，去掉 `/ (std + 1e-8)`。

MetaClaw 自己的 `compute_advantages` 是除 std 的，我们上一版照搬了。对我们的批形态这是错的：reward 在 ±1 附近时，批越偏斜、除数越小，**稀有的那一号被无界放大**：

| 批内 正/负（共 16） | 正 advantage | 负 advantage |
|---|---|---|
| 1 / 15 | **3.873** | -0.258 |
| 4 / 12 | 1.732 | -0.577 |
| 8 / 8 | 1.000 | -1.000 |
| 11 / 5 | 0.674 | -1.483 |

**两端对我们都是活的**：round 通过率约 17%，1 正批是常态（`20260902_094458` 的崩溃就是这么来的）；而步骤判官约 69% 给正分，所以稀有负样本会到 -1.483。**我上一轮只算了后一半就下了"稀有负样本被放大"的结论，没算前一半——而实际炸掉训练的恰恰是前一半。**

只减均值保留了全部想要的性质（全同批仍然精确居中到 0），同时把结果**有界**：reward ∈ [-1,1] 时 `|advantage| ≤ 2`，1 正 15 负给 **+1.875 / -0.125** 而不是 +3.873。稀有事件仍然主导它那一批（这是对的），但再也放大不成梯度尖峰。丢掉的那个尺度是个常数因子，被学习率吸收。

（顺带更正一条 CLI 的算术：它给的 1 正 15 负"+0.94/-0.06"是错的，那是把 reward 当成 {1,0} 算的；±1 下正确值是 +1.875/-0.125，它据此提的验收断言 `|adv|<1` 会直接挂掉，正确的界是 `|adv| ≤ 2`。）

#### 明确不做的事

- **不回退批级基线**（只改掉除 std 那一步）。
- **不重新启用 `blend`**。
- **不动官方 Acc./Compl. 判分**。
- **不加"advantage 为 0 时跳过 OPD"**。OPD 的 advantage 来自 teacher-vs-rollout 的 logprob 差（`hint_opd_loss.py:337-341`），跟 reward 无关，所以 GRPO advantage 为 0 时 OPD 并不停；而 `PRM_TEACHER_LOAD = POLICY_TORCH_DIST` 就是 base 模型，这条 OPD 实际是一个"拉回 base"的正则项——在难题批次上大概率是**稳定力量**，不是泄漏。

#### 验证（2026-09-02，本地，真实训练未验证）

三个补丁脚本 `bash -n` 通过；补丁链对真实官方源码跑通、`py_compile` 通过；`_metaclaw_length_aware_reward` 的两个落点确认落在两个提交函数里 `sample.reward` 赋值之前。

新增 `scripts/tests/test_length_aware_and_baseline.py`，**41 项断言，直接跑补丁脚本真实生成的代码**（不是在测试里重写一遍被测逻辑）。覆盖：长度打折的边界与单调性、极长的正确答案仍然为正、负样本在五个长度下都精确等于 -1.0、OPD-only 的 0.0 不受影响、全负/全正批 advantage 全 0、1 正 15 负 = +1.875/-0.125、11 正 5 负 = -1.375、`|adv| ≤ 2`、两项改动叠加后 day06-r6 那个真实样本从 3.873 降到 **1.514**、短的孤立成功排名高于长的、**blend 式散开的负 reward 确实会在全失败 round 里产生正 advantage**（把删除理由变成可执行断言）、dummy 不拉偏均值、`remove_sample` 无 metadata 标记时也被排除、单样本/全 dummy 批不崩、slime 会 assert 的返回长度契约。

**非空洞性双向验证**：把 `advantage = r - mean` 改回除 std → 断言按预期挂在 "1 pos / 15 neg -> pos advantage 1.875"；把 `if reward <= 0` 守卫拿掉 → 挂在 "negative at len=10273 stays exactly -1.0"。两次都已还原。

#### 下一次跑的配置与判据

**目标是复现 `20260827_163030` 的 judge 配置，再叠加批级基线 + 这两项防训坏补丁；不用 outcome、不用 blend；必须从干净 base 起步。**

选 judge 而不是 outcome，跟上一轮的选择相反，理由也变了：outcome 是为了触发基线的抢救机制才选的，但 `20260902_094458` 已经证明 outcome 会一次性倒出上百个负样本、自己制造极端偏斜批；而 judge 是那次"跑到 day22、没崩、只是中途变差"的配置——新目标下要修的正是它。

验收判据：

1. **day17 不再出现 18k → 115k 的 thinking 断崖**（对照只训到 day6 那次的 day16-18）。
2. **day16-22 的 MC 格式失败率明显低于 17/26。**
3. **Acc. 可以低于只训到 day6 那次，但不能出现 day20-22 归零式崩塌。**
4. 全负批 → advantage 全 0；日志里不再出现约 3.87 那种稀有正样本放大。
5. `[metaclaw-batch-baseline]` 与 `[openclaw-rl-metaclaw-length-aware-success]` 两类日志行必须真的出现——它们是这两项改动生效的唯一直接证据（reward 侧计数不会变，别盯它）。

#### 尚未做、需要先查清杠杆的两件事

1. **round 轮数上限**。`20260902_094458` 里 day06-r7 空转 186 轮是灾难的起点。现有唯一的杠杆是 driver 里 `_run_round` 那个调用点的 `round_timeout`（当前是 `None`），另一个候选是代理侧在 `_turn_counts` 超阈值后直接拒绝。两者的副作用还没查清，**先不动**。
2. **退化熔断**（按某个运行时指标自动冻结训练）。指标选什么、阈值多少，都还没有依据，同样先不动。

### 查证记录（八）：`toolcall-rl` 到底怎么处理"调用次数不确定的中间步骤"（2026-09-03）

09-01 的查证记录（六）只看到了奖励合成那一层，漏了最关键的一层：**样本粒度**。重查 `toolcall-rl/generate_with_retool.py` 全文，结论如下。

**一条轨迹 = 一个样本，不是一个 turn 一个样本。** 多轮拼成一条 token 序列，工具返回用 loss_mask 挖掉（`generate_with_retool.py:709-765`）：

```python
response_token_ids += cur_response_token_ids      # 模型输出
loss_masks       += [1] * len(cur_response_token_ids)
...
response_token_ids += obs_tokens_ids              # 工具返回
loss_masks       += [0] * len(obs_tokens_ids)
sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)
assert len(response_token_ids) == len(sample.rollout_log_probs)
```

**"这一轮调了几次工具"因此根本不是一个问题**——调得多序列长一点，样本数永远是 1。上限只有 `max_turns=16` / `max_tool_calls=16` 和 context 长度。这正是我们一直没解决的那个问题（一个 round 产出几个样本、怎么分组）在它那边压根不存在的原因。

**它同时记录每一段模型输出的 token 区间**（`step_action_spans`，`{step_index, token_start, token_end}`），落进 `sample.metadata["step_wise"]["step_token_spans"]`。PRM 对每一步异步打分，但**这些分不会变成独立样本**，只进 metadata。

**奖励合成**（`reward_func:856`）：`outcome_reward = math_dapo_compute_score(...)`，然后 `step_scores_with_outcome = [step + outcome for step in step_scores]`——**整条轨迹的最终结果加到每一步上，PRM 只负责在轨迹内部做区分**。

**`--advantage-estimator step_wise`**（PRM 版用它，`retool_qwen3_4b_prm_rl.sh:108`）：归一化在 `slime/ray/rollout.py:530` 做，按 **`(group_index, step_index)`** 分桶——第 3 步跟同一道题另外 7 条轨迹的第 3 步比；桶内方差 ≤1e-12 **整桶丢弃**，样本所有 step 都被丢就 `remove_sample = True`；最后 `loss.py:645` 把每步归一化后的标量**广播到它自己的 token 区间**（`full_adv[start:end] = step_reward`，再乘 `loss_mask`）。

**推论一：09-01 设想的"在 reward 钩子里做轨迹级 advantage 再广播"，slime 里已经有了，就是 `step_wise`。** 我们自己写的批级基线是在重造一个更差的轮子。

**推论二：把 per-step PRM 关掉（`prm_step_coef=0`）后，`step_scores_with_outcome` 退化成"每一步都等于该轮 checker 判定"**，于是同一 round 的 n 次尝试若结果全同 → 每个 `(group, step)` 桶方差为 0 → **整桶被官方逻辑丢弃 → 这批不更新**。我们千辛万苦想要的"全负批不训坏"，是官方逻辑白送的。而且此时所有 step 分数相同，`step_wise` 广播一个标量到全部 masked token，**等价于 `n_samples_per_prompt=n` 的普通 GRPO**——所以最小可行版本用 `grpo` 就够，不必上 `step_wise`。

**权重更新规模**（`slime/utils/arguments.py:1948`）：

```
global_batch_size = rollout_batch_size × n_samples_per_prompt // num_steps_per_rollout
```

| | rollout_batch | n_samples | steps_per_rollout | 每次更新的样本数 |
|---|---|---|---|---|
| toolcall-rl（PRM 版 `step_wise`）| 32 | 8 | 2 | **128 条轨迹** |
| toolcall-rl（非 PRM 版 `grpo`）| 32 | 8 | 2 | **128 条轨迹** |
| 我们现在 | 16 | 1 | 1 | 16 个 **turn** |

两个变体的 `n_samples_per_prompt` **都是 8**，跟 advantage estimator 无关——8 是配方核心，不是调参。

---

### 方案（待 CLI 在真实环境查证，未实现）：轨迹级样本，一个 round = 一个样本（2026-09-03）

#### 定位（重要，别当成控制变量实验）

跟"跑到 day22"那次（`20260827_163030`）相比，这个方案至少变了五件事：样本单位、奖励来源、每轮样本数、OPD hint 作用范围、多了一条长度保护。**跑出来变好或变坏都归不到单一原因上。** 它的定位是"换一种结构上站得住的形态，看能不能跑起来"，不是 A/B。

真正的单变量版本（只改奖励、不改样本单位）**已经做过了，就是 `METACLAW_MIDROUND_REWARD=outcome`，`20260831_154301` 一跑就发散**。所以"只把中间步打分换成整轮打分"这条路本身已被证伪；这个方案跟它的差别**只在于同一轮的 N 个 turn 是留成 N 个样本还是并成 1 条**，而 outcome 当初炸掉，炸的正是后者。

#### 合并样本单位带来的三个结构性后果（outcome 模式当时都没有）

1. **每一轮的梯度权重终于相等。** `sum_of_sample_mean`（`slime/backends/megatron_utils/cp_utils.py:70`）是每个样本先取均值、再对样本求和，所以**一个跑了 20 轮的 round 现在贡献 20 份梯度，1 轮答完的 round 只贡献 1 份**——"啰嗦、反复调工具"这件事自带 20 倍权重加成，跟答得对不对无关。轨迹级之后每个 round 恰好算一次。
2. **一个 round 再也无法主宰一个 batch。** day06-r7 空转 186 轮 → 一次性倒出 186 个 `-1` → 该批 1 正 15 负，正是 outcome 发散的直接形态。轨迹级下它就是 1 个样本。
3. **2~3 token 的空回复样本消失**，被吸收进整条轨迹里的一小段。

#### 一个现成的有利条件：session 已经就是 round

driver 里 `round_session_id = f"metaclaw-{test_id}-{group_id}-{round_id}"`（2026-08-19c 起，每 round 一个全新空 transcript）。**所以一个代理 session 精确对应一条轨迹，不需要额外的 round 键**；而且每轮从空 transcript 开始，round 内触发 OpenClaw context 压缩的概率比共享 session 时低（但不为零，见待查证 ②）。

#### 样本构造：前缀差分

代理侧已有 `_pending_turn_data[session_id][turn_num]`，每个 turn 存着 `prompt_ids / response_ids / response_logprobs / prompt_text / response_text / messages`。verdict 到达时按下面的方式拼一条：

```
轨迹 prompt = turn[1].prompt_text
轨迹 response = R1 + Obs1 + R2 + Obs2 + ... + RN
其中  Rn   = turn[n].response_text                                            loss_mask=1
      Obsn = turn[n+1].prompt_text 去掉 (turn[n].prompt_text + Rn) 之后剩下的部分   loss_mask=0
```

**观测段用前缀差分算出来，不去猜工具返回长什么样**，因此它天然包含 chat template 的脚手架（`<|im_end|>`、下一条 `<|im_start|>user ... <|im_start|>assistant\n`），这些正是模型条件依赖、但不该被训练的 token，mask 成 0 完全正确。

**这个算法自带失败检测**：若 `turn[n+1].prompt_text` 不以 `turn[n].prompt_text + Rn` 开头（context 压缩、消息被重写等），**当场判定重建失败、整轮丢弃并打日志**，绝不拼出一条错序列。

`rollout_log_probs` 照抄 toolcall-rl：模型输出段用该 turn 的 `response_logprobs`（官方代码已保证与 `response_ids` 等长），观测段补 `0.0`，末尾加硬断言。

**一个已知的近似**：前缀检查在**文本**层面做（权威），但各段是各自 tokenize 再拼接的，跨段边界可能与"整体 tokenize 一次"不完全一致。这跟官方现在 `response_ids = tokenizer(response_text)` 已有的近似是同一类，但轨迹级下段数更多。需要实测漂移量（待查证 ③）。

#### 提交时机

从"某个 turn 的 PRM 任务完成就提交"改成"**本轮 verdict 到达时，把整条轨迹作为唯一一个样本提交**"。中间 turn 不再触发任何 PRM/判官任务。

#### 奖励与 OPD

- **奖励 = 本轮 checker 的确定性 ±1**（`metaclaw_verdict` 分支已有），**步骤判官整个消失**。
- **OPD 的构造一行都不用改。** `_append_hint_to_messages`（`openclaw_opd_api_server.py:139`）是从后往前找最后一条 user 消息贴 hint；轨迹级下 `messages` 就是本轮初始消息列表，最后一条 user 消息正好是本轮题目，hint 落点天然正确。随后 `enhanced_full_text = enhanced_prompt_text + response_text`，把 `response_text` 换成整条轨迹之后，语义正好变成"如果这道题一开始就带着这个 hint，模型会怎么走完这一整轮"——**checker 给的 hint（比如文件名写错）本来就是针对整轮的，不是针对某个 turn 的**。
- **观测 token 不会污染蒸馏**：`openclaw_topk_select_loss.py:435` 的 `opd_loss = sum_of_sample_mean(opd_pg_tokens)`，而 `sum_of_sample_mean` 乘了 `loss_mask`，所以 GRPO 项和 OPD 项**一起**把 mask=0 的观测段排除。

#### 长度保护（前置必做，不是可选）

轨迹级下，一个失控的 round 不再是"多产负样本"，而是"**这条样本根本进不了训练**"——`--max-tokens-per-gpu 32768` 是训练侧打包上限。

分两层，**第一层保证正确性、不依赖任何未验证的杠杆**：

1. **代理侧硬丢弃**：拼完的轨迹 token 数超过阈值 → 打日志 + 丢弃整轮，不提交。永远安全，无需新机制。**阈值必须从 `--max-tokens-per-gpu` 反推**：开着 `--use-dynamic-batch-size` 时 slime 按 token 预算把样本打包成 micro-batch，**单条样本长过这个预算就永远打不进任何一包**，所以 32768 是硬天花板，不是建议值。**取值定为 31000**（贴着天花板留一点余量），理由见下面"按健康段而不是全程标定"一节——初稿写的 24576 是按被污染的全程分布定的，在健康区间会白丢 7% 的数据。
2. **轮数上限（从"优化项"升级为"选择偏差的主要缓解手段"，见下面 CLI 实测）**：两个候选杠杆——driver 的 `round_timeout`（当前 `None`，墙钟）和代理侧按 `_turn_counts` 超阈值拒绝——**副作用都没查过**（待查证 ⑤）。toolcall-rl 的锚点是 `max_turns=16 / max_tool_calls=16`。

#### 配置改动

| 参数 | 现在 | 改成 | 理由 |
|---|---|---|---|
| `--rollout-batch-size` | 16 | **8** | 每轮样本数降到 1.0；见下面的更新次数账 |
| `--n-samples-per-prompt` | 1 | **1（本阶段不动）** | 本轮先验证形态，`n_samples=8` 是下一阶段 |
| `--disable-rewards-normalization` | 开 | **保持开** | `n_samples=1` 时每组只有 1 个样本，开归一化会被 `_drop_constant_reward_groups` 整批判成常数组 |
| `--advantage-estimator` | grpo | **grpo（不动）** | 所有 step 同分时 `step_wise` 与 grpo 等价，不必上 |

#### CLI 真实日志核实结果（2026-09-03），两处修正我的估算

**④ 每轮样本数：我说的"约 2.0"是早期天数的值，不是稳定值。**

| 日志 | submits/round |
|---|---|
| `20260902_161947`（judge，早崩）| 1.49 |
| 只训到 day6 那次的早期窗口 | 2.42 |
| `20260827_163030` 全程 | **4.04** |
| `20260827_163030` 仅 day01–15 | 4.44（其中 day01–02 约 2.0，**day08 约 11.7**）|

**所以"更新次数基本不变"这个结论要收回。**`20260827_163030` 实际是 **约 56 step / 约 222 round**；轨迹级下更新次数 = `round 数 ÷ rollout_batch_size`，与 `n_samples` 无关（`rollout_batch_size` 数的是组）：

| | 更新次数 |
|---|---|
| `20260827_163030` 实测（turn 级，batch 16）| **约 56**（22 天）|
| 轨迹级 batch 8 | 346 ÷ 8 = **43**（30 天）|
| 轨迹级 batch 4 | 346 ÷ 4 = **86**（30 天）|

**结论修正为：batch 8 的更新次数明显少于 day22 那次。**具体差多少见下一节——按健康段标定之后差距还会再大一截。**读结果时必须扣掉这一条：即使形态是对的，分数也可能只是因为更新次数少而偏低。**

为什么仍选 batch 8 而不是 4：`n_samples=1` 时 batch 4 的全负批概率是 `0.83^4 ≈ 47%`，batch 8 是 23%，batch 16 是 5% 但只剩 21 次更新。**batch 8 是这三者里最不坏的折中，不是最优解**；等上了 `n_samples=8`（全负组会被常数桶逻辑丢弃、不再是均匀打压），batch 4 就变得可选了。

**⑥ 轨迹长度：比我预期大得多，而且硬丢弃会造成系统性选择偏差。**

（CLI 用 `last_prompt + last_resp` 近似整条轨迹长度——前缀性质成立时这个近似是对的。）

| 窗口 | 中位 | p90 | >24576 | >32768 |
|---|---|---|---|---|
| `20260827` day01–15 | 约 17k | 约 23k | 12/170（约 7%）| 3 |
| `20260827` 全程 | 约 18k | **约 47k** | 43/247（约 17%）| 31（约 13%）|

三条后果：

1. **中位就有 17~18k token**，对着 `--max-tokens-per-gpu 32768`，**一个 micro-batch 大概只装得下 1~2 条样本**。训练侧吞吐会明显变慢，这是之前没算到的成本。
2. **被丢掉的是系统性偏向 day08+ 和失控轮的长难 round**——这是真实的选择偏差，不是随机丢弃。**所以轮数上限从"可后置的优化"升级成"缓解偏差的主要手段"**：与其在拼完之后丢掉一条 47k 的轨迹，不如一开始就不让 round 跑到那么长。

#### 按健康段而不是全程标定（2026-09-03 用户更正，采纳）

用户指出：**`20260827_163030` 的后半段本身已经是被污染的、正是我们要解决的问题，拿全程分布当设计基准是拿病态当常态。**真正该对标的是前半段的良好训练状态。这个原则成立，但两个数的表现不一样，必须分开说。

**长度：完全成立，长尾确实全部来自污染段。**

| 窗口 | 中位 | p90 | >24576 | >32768 |
|---|---|---|---|---|
| **day01–15（健康，设计基准）**| 约 17k | **约 23k** | 12/170（**7%**）| 3/170（**约 2%**）|
| 全程（含污染段）| 约 18k | **约 47k** | 约 17% | 约 13% |

中位几乎没变，**p90 从 23k 飙到 47k 全部是后半段贡献的**。两个直接后果：

- **硬丢弃阈值改成 31000（贴天花板），不用 24576。**按健康段标定，24576 要丢 7%，31000 只丢约 2%——**在健康区间几乎不损失数据**。退化区间它会丢 13%，但那种情况本来就该被下面的检查点拦掉，不该指望阈值去兜。
- **轨迹长度分布本身就是退化的早期指标。**健康段 p90 约 23k、污染段约 47k，**这个信号比 Acc 掉下来早得多**。所以检查点 B 不只看丢弃率，直接对标健康画像：**p90 越过 30k 就报警**。

**samples/round：CLI 的数字不支持"后半段把均值拉高了"，反而相反。**

全程是 4.04，**day01–15 是 4.44**——后半段把均值**拉低了**；而峰值 `day08 ≈ 11.7` 落在 day01–15 里面，不在末段。原因是 `20260827` 的退化形态是 **thinking 膨胀**（day17 从 18k 冲到 115k），thinking 变长会挤占 context、turn 数反而减少；"多次调用工具却不怎么输出"是 09-02/09-03 那两次**工具塌陷**的形态，不是这一次的。

**这个更正对我们不利，必须写下来**：改用健康段基准之后，turn 级的基准是 170 round / 4.44 ≈ **47 step / 15 天**，折算 30 天约 **94 次更新**，而轨迹级 batch 8 只有 **43 次**——**差距从"一半到七成"扩大到"不到一半"**。

要补齐只能降 `rollout_batch_size`，于是有三选一：

| 配置 | 更新次数（30 天）| 全负批 | 说明 |
|---|---|---|---|
| **`n_samples=1` + batch 8（本轮采用）**| 43 | 23% | 更新次数不到健康基准一半，**这是本轮自带的 handicap** |
| `n_samples=1` + batch 4 | 86 | **47%** | 次数对齐了，但一半批次在均匀打压 |
| `n_samples=8` + batch 4 | 86 | 不适用（全负组被常数桶丢弃）| 两个问题同时解决，采样成本 8× |

**2026-09-03 用户决定：本轮仍按第一行跑**，先验证轨迹级这个形态本身；第三行留给下一步。**因此读这次结果时，"分数比 day22 那次低"至少有两个与形态无关的解释——更新次数不到一半、没有组内比较——不能直接当成形态失败的证据。**

**⑧ `--rollout-max-response-len 8192`：风险低但不为零。**MetaClaw 走的是 custom generate，提交时 `response_length = len(response_ids)`，slime 只要求它与 `loss_mask` 等长；8192 主要约束的是标准 sglang rollout 的 `max_new_tokens`，不是拼好之后的轨迹。**但实现时仍要显式核对**有没有别处拿这个参数分配缓冲区或做断言，不要假设零风险。

**⑨ `rollout_batch_size=8` 与 dummy 注入：风险低。**现有跑法 `data_parallel_size = 1`，只有 `len(samples) < dp_size` 才补 dummy，batch 8 不会更频繁触发。（顺带说明：之前那版批级基线里写的一大段 dummy 排除逻辑，在这个拓扑下基本是空转的。）

**①②③⑤⑦ 与 `METACLAW_*` 传播：本地验不了，仍然全开。**日志几乎不落完整的 `prompt_text` 链，压缩行为和 teacher 耗时都要运行时才知道。

**CLI 在 ① 上补了一个我没说清的区分，很关键**：现有 per-turn 代码里的 `full_text.startswith(prompt_text)` 只能说明**单个 turn 内部**的切片习惯性满足前缀关系，**完全不能替代跨 turn 的 `prompt[n+1] == prompt[n] + response[n] + obs` 这条性质**。所以 ① 至今是零证据，不是"大概率成立"。

#### 需要在真实运行中确认的项（编号沿用上文）

**已由 CLI 用真实日志给出结论**：④（数字见上，我的 2.0 已修正）、⑥（分布见上）、⑧（风险低，实现时核对）、⑨（风险低）。

**仍然全开、只能靠这次运行回答**：

1. **前缀重建成功率**——统计有多少 round 满足"每个 `turn[n+1].prompt_text` 都以 `turn[n].prompt_text + turn[n].response_text` 开头"。**这条不过，整个方案不成立；明显失败就停，不要调参硬撑。**
2. **round 内是否发生 OpenClaw context 压缩/历史重写**——① 失败的主要来源。
3. **token 边界漂移**——`len(tokenizer(拼接后的整段文本))` 与"各段分别 tokenize 后长度之和"的差值分布。
5. **轮数上限两个杠杆的副作用**——`round_timeout`（driver 侧墙钟）与代理侧按 `_turn_counts` 拒绝，哪个安全、会不会让 OpenClaw 进入异常状态。**因为 ⑥ 的结果，这一项的优先级提高了。**
7. **teacher logprob 的开销**——`_compute_teacher_log_probs` 的输入序列从约 2k token 变成 **17~18k 中位**（比方案初稿写的 10~20k 上限更贴近上界），耗时与显存是否可接受。

外加：**`METACLAW_*` 环境变量到底有没有传到训练后端进程，至今没有验证过**（`judge` 是默认值，"看起来对"证明不了传播成功）。

#### 怎么跑：直接跑满 30 天，不单独做冒烟（2026-09-03 用户决定）

**不另外跑 2 天冒烟，直接按全量 30 天跑。** 理由成立：原本设想的四项冒烟检查全部可以从**同一次全量跑的 day01~day02 日志里读出来**，冒烟 run 是全量 run 的真子集，单独跑一次只是重复一遍环境搭建。

（CLI 的评审结论里仍写着"按文档做 2 天冒烟"，那是针对方案上一版说的；本节按用户的新安排为准，检查项一条没少，只是不再单独起一次运行。）

改成**在全量跑内部设两个早期中止检查点**。设两个而不是一个，是 CLI 的 ⑥ 直接逼出来的：**day01–02 的轨迹又短又少（约 2.0 samples/round），根本代表不了 day08（约 11.7 samples/round）**——只看 day02 会得到一个过于乐观的结论。

**检查点 A（day02 跑完）——验管线是否成立：**

| 项 | 期望 | 不达标怎么办 |
|---|---|---|
| ① 前缀重建成功率 | 接近 100% | **立刻中止**，方案不成立，回来重新设计观测段的取法 |
| ② 每天提交的样本数 | 约等于当天 round 数（day01=10、day02=11）| 少很多说明重建在悄悄丢数据，中止排查 |
| ③ OPD hint 接受率 | 不显著低于现在 | 掉得多说明 round 级 hint 判官不认，需要单独处理 |
| ④ 硬丢弃命中率（阈值 31000）| 接近 0 | 这里就有命中 = 后面只会更糟，中止 |

**检查点 B（day08–day10 跑完）——验长轨迹这一段扛不扛得住。基准是健康段画像，不是全程画像：**

| 项 | 期望（对标 day01–15 健康画像）| 不达标怎么办 |
|---|---|---|
| **轨迹长度 p90** | **保持在约 23k 附近；越过 30k 就报警** | 这是**比 Acc 早得多的退化指标**（健康段 p90 约 23k、污染段约 47k）——报警就中止，先上轮数上限 |
| 硬丢弃累计命中率（31000）| 约 2%（健康段基线）| 明显高于此说明已经在往污染画像走，中止 |
| 前缀重建成功率是否随轨迹变长而下降 | 保持 | 下降说明 context 压缩在长 round 上开始生效，中止 |
| 训练侧吞吐 | 可接受 | 中位 17k token 一包只装 1~2 条，若已经拖垮节奏就要重估 |

两个检查点都过就让它跑完 30 天，**这一次要看分数**——跟 `20260827_163030`（跑到 day22）和"只训到 day6"那次比 Acc./Compl. 曲线。

**读结果的规则（先写下来，免得跑完之后事后合理化）：**

- **跑得不好不能直接判轨迹级形态死刑。**本轮自带两个与形态无关的 handicap：**没有组内比较**（约 23% 的批次全负、advantage 就是原始 ±1），以及**更新次数不到健康基准的一半**（43 vs 约 94）。那种情况下的下一步是**上 `n_samples=8` + batch 4 + 打开 `rewards_normalization`**，而不是推翻这个形态。CLI 的说法很准：`n_samples=1` 下全负批问题只是**从"186 个 -1"变成"8 个 -1"，没有消失**。
- **跑得比 day22 那次好，则 `n_samples=8` 只会更好**，可以直接接着做。
- **对标的是 `20260827_163030` 的前半段（day01–15），不是它的全程。**后半段本身就是被污染的形态，拿它当基准等于拿病态当常态。
- **无论好坏，都不能把结果归因到"只改了粒度"**——这一次至少同时变了五件事。

---

### 查证记录（九）：官方 MetaClaw 是**按天**共用 session，我们从 08-19c 起改成了**按题**隔离（2026-09-03）

CLI 在核对"为什么当前 live Acc 明显高于定版基线 17.8%"时定位到一处**与权重无关的结构差**，本地读官方源码确认了，而且比它描述的更明确。

#### 官方：一天一条 transcript，跨题共享

```
infer_cmd.py:1022   session_id = test["session"]                       # 一个 test = 一天
infer_cmd.py:864    _prepare_session(..., original_session_id)          # 在 round 循环之前，只调一次
infer_cmd.py:927    await _run_question(..., session_id=original_session_id)   # 每一轮都传同一个
```

`all_tests.json` 里 **30 个 test 对应 30 个互不相同的 session id**（`day04_4f564f09-…`），一天一个；一天只有一个 group，group 内 10~15 个 round 全部共用这条 transcript。**当天前面题的完整对话历史（含工具调用轨迹）会一直堆在后面题的上下文里。**

**官方靠 OpenClaw 自带的上下文压缩活下来，不是靠切 session**：`openclaw_cfg/openclaw.json` 里 `"contextWindow": 50000, "maxTokens": 50000`，`"compaction": {"mode": "safeguard"}`。我们用的就是官方这份配置（走官方 `_patch_agent_workspace`），压缩设置原样继承、没改过。

#### 我们：2026-08-19c 起改成按题隔离

driver 里 `round_session_id = f"metaclaw-{test_id}-{group_id}-{round_id}"`，每题一个全新空 transcript。当初改它是为了修一个真实故障（一道 file_check 的超长回复把当天后面所有题的上下文撑爆，连本来答得好的 MC 也一起死），但**它替换掉的是官方协议里那条"agent 记得自己今天早些时候做过什么"**，跨题信息只剩显式拼进 query 的 `[Previous Feedback]` 文本。

#### 三条后果

1. **绝对分数不能跟论文 Table 1 或 17.8% 定版基线比。** 定版基线跑的是官方 CLI（按天），我们跑的是按题。CLI 给的数字正是这个差：基线 day02/day03 是 0.20 / −0.01，而本次几乎未训练的 day02/day03 是 0.62 / 0.71。**这是协议差，不是训练效果。**
2. **我们自己几次跑之间仍然可比。** 08-19c 之后的全部运行——K=6（`20260820_122808`）、跑到 day22 的 `20260827_163030`、以及接下来这次——都是按题隔离。所以内部对照有效，**被污染的只有"对标官方基线"这一条**。
3. **这个偏离对 4B 是变简单了**：上下文膨胀的害处大于"记得自己写过什么"的好处。

#### 一处与轨迹级方案的强耦合，改回去之前必须先想清楚

**按题隔离恰恰是轨迹级样本能成立的前提。** 压缩正是会打断前缀不变量的那个机制（轨迹级方案的待查证 ①）。按题隔离下每轮从空 transcript 开始，压缩很难触发；**若改回按天共用 session，10~15 轮堆下来 `safeguard` 压缩会经常触发，前缀重建会大面积失败，轨迹级方案直接不成立。**

因此**现在不改回按天**：一改就跟 K=6 / day22 全部不可比，而且大概率废掉刚实现的轨迹级方案。但这是一个**已知的、待偿还的协议偏离**，如果最终目标包含"复现 MetaClaw-Bench 的数字"，按天 session 迟早要回去（配合轮数上限 + 长度守卫解决当初的上下文膨胀问题），届时轨迹级的观测段取法要重新设计。

---

#### 补充：MetaClaw 自己是怎么对付按天上下文膨胀的（2026-09-03 续查）

既然按天共用 session 会带来这么明显的退化，官方自己怎么活下来的？查完是两层，而且第二层直接关系到我们这个项目的定位。

**第一层：vanilla 跑法根本没"解决"，退化本身就是题目的一部分。**

不带 `--memory`/`--buffer-turns` 时，对抗当天上下文膨胀的**唯一机制就是 OpenClaw 自带的 `safeguard` 压缩**（`"compaction": {"mode": "safeguard"}`）。它做的事是：上下文涨到阈值时用 LLM 写一份结构化摘要替换较早的消息，带质量校验，校验不过就**取消压缩而不是丢历史**（`openclaw/src/agents/agent-hooks/compaction-safeguard.ts`）。

就这些。**定版基线 day02/day03 掉到 0.20 / −0.01（CLI 从日志读出的数字），是 benchmark 按设计工作**——"普通 agent 在长对话的一天里逐渐退化"本来就是它要测的东西之一，不是需要绕过的故障。

**第二层：MetaClaw 方法自己的答案，就是记忆系统。**

`--memory` / `--buffer-turns` 是两个 opt-in 开关（`benchmark/src/cli.py:174`），打开后：

- **写入**：每答完一题 POST `/v1/memory/buffer_turn`，缓冲满 `memory_flush_every = 5` 轮自动 flush，把这几轮抽成**记忆单元**（`_extract_turns_to_units`）外加一份 `working_summary`（`metaclaw/memory/manager.py:170-224`）。
- **读取**：每次 `/v1/chat/completions` 走 `_inject_memory`（`metaclaw/api_server.py:2483`），拿**当前这道题的文本**检索相关记忆、拼进 system 消息最前面。检索作用域是**去掉 session 后缀的 base scope**，源码注释原话是 "so memories from previous sessions are visible"——**记忆跨 session、跨天可见**。技能走同一套（`skill_manager.retrieve(task_desc, top_k=skill_top_k)`）。

**也就是说：MetaClaw 把知识从原始 transcript 搬进一个按任务文本可检索的存储里，压缩就算把原始消息扔掉，真正重要的东西也不会丢。这正是论文 meta-model = (θ, S) 里的 S。**

**对我们的意义（不舒服但必须写下来）**

本项目**明确决定不迁移 S**，唯一跨天载体是 θ（2026-08-14 查证记录三定下的）。而**按天共用 session 造成的上下文压力，恰恰就是 S 被发明出来解决的那部分任务**。

所以：**我们按题切 session，删掉的正是 benchmark 里专门用来考 S 的那一块——而那一块恰好也是纯 θ 方法做不了的。分数变高不是巧合，是因为把自己不会做的题去掉了。**

这把协议偏离的性质从"数字不可比"升级成了"**测的题目范围不同**"，写报告时必须交代。三条路：

| 选择 | 实际在测什么 |
|---|---|
| **保持按题**（现状）| "纯 θ 能不能学会单题任务"——比 MetaClaw-Bench 问的问题**更窄也更容易** |
| **改回按天、仍不带 S** | 拿纯 θ 去答一道设计上需要 S 的题；绝对分会明显更差，但**对照是忠实的** |
| 改回按天 + 迁移 S | 超出当前范围（S 迁移当初就被排除） |

**一个技术细节，将来真要改回按天时会撞上**：压缩摘要是用同一个 model 生成的（`compaction-safeguard.ts:1036` 解析 `ctx.model`/`runtime.model`），也就是会经过我们的代理、由 4B policy 自己来写——那些调用会以 turn 的形式打进代理，是另一摊要单独处理的事。

#### 待定的决策点：等 K=0 基线出来再拍（2026-09-03，用户）

**如果 `METACLAW_TRAIN_UNTIL_DAY=0` 的按题基线跟定版按天基线（17.8%）差距很大**——按 CLI 目前给的 day02/day03 对比（0.20/−0.01 vs 0.62/0.71）看大概率会很大——**那就必须重新讨论究竟该怎么做，不能默认继续按题跑下去**。届时要在上表三条路里做选择，判断依据至少包括：

- 差距有多大：这直接决定"我们测的题"和"论文测的题"偏离了多少
- 这个差距在 30 天上是均匀的还是集中在长天（day08+ 那种 round 多、轨迹长的天）——若集中在长天，说明偏离的正是最需要 S 的那部分
- 最终交付要不要包含"可与论文 Table 1 对比的数字"。要，就绕不开改回按天；不要，就保持按题但全程标注清楚

**在拍板之前不要改**：改回按天会同时（a）跟 K=6/day22 全部不可比，（b）触发 `safeguard` 压缩、打断前缀不变量、废掉刚实现的轨迹级方案。

**另一个可选的补测（未做，成本不高）**：用我们自己的 driver 跑一次**按天** K=0，跟官方 CLI 的按天基线对照——这能把"协议差"和"我们的 harness 与官方 CLI 之间的其它差异"分开，得到一个干净的 2×2。需要把 session 粒度做成 driver 里的一个开关，改动很小。等上面拍板时再决定要不要做。

---

### 方案（未跑）：按题隔离的零训练基线，`METACLAW_TRAIN_UNTIL_DAY=0`（2026-09-03）

#### 为什么必须补这一条

此前衡量迁移效果一直是"K=6 冻结实验 vs 定版基线 17.8%"。查证记录（九）之后可知这个对照**同时混着两个变量**：训练效果、以及按题/按天的协议差。**缺的是"同 driver、同 base、零训练"的那一格。**

| 实验 | session | 训练 | 作用 |
|---|---|---|---|
| 定版基线 17.8% | **按天**（官方 CLI）| 无 | 官方协议下的 base |
| **新基线（待跑）** | **按题**（我们的 driver）| **无**（K=0）| **我们协议下的 base ← 缺的就是这格** |
| K=6 冻结（`20260820_122808`）| 按题 | day1–6 | |
| 跑到 day22（`20260827_163030`）| 按题 | 全程 | |
| 轨迹级（下一次）| 按题 | 全程 | |

补上之后：

- `新基线 vs 定版基线` = **纯协议差**，跟训练无关
- `K=6 vs 新基线` = **纯训练效果**——这才是一直想测但从没干净测过的东西
- `轨迹级 vs K=6 vs 新基线` = 三方可比

#### 怎么跑：零代码改动

`METACLAW_TRAIN_UNTIL_DAY=0` 本来就是支持的，driver 里写明了"K=0 is valid -- freezes before day 1, i.e. a pure base-model pass through this same 30-day harness"（`metaclaw_rollout_driver.py:296`）。冻结信号在每个冻结日**跑之前**发送（`:1989-1992`，`if is_frozen_day: await _send_freeze_signal(...)` 在 `await run_day(...)` 之前），所以 K=0 时 day01 开跑前就已经冻结，不会有任何样本进训练队列。

```bash
METACLAW_TRAIN_UNTIL_DAY=0 bash scripts/metaclaw/run_metaclaw_migration_modelfactory.sh
```

跑满 30 天，真实 agent + checker 照常执行，产出完整的 Acc./Compl. 报告，只是权重全程不动。

#### 顺带白拿的东西：轨迹级方案的三项查证

冻结门在 `_maybe_submit_ready_samples`（提交时丢弃），而轨迹拼接在 `_fire_opd_task`，**所以冻结状态下轨迹仍然照常拼、只是拼完被丢**。也就是说这一趟基线跑会**顺带在全部 30 天上量出**：

- **① 前缀重建成功率**——轨迹级方案的成败条件，**在不烧一次训练跑的前提下就能拿到答案**
- **⑥ 轨迹长度分布 / 硬丢弃命中率**——而且是在我们自己的协议下量的，不是从 `20260827` 的日志里近似出来的
- **OPD hint 在 round 级任务上的接受率**

拿不到的：**⑦ teacher logprob 开销**（teacher logprobs 在 Megatron 侧对 `teacher_tokens` 算，没有训练就不会发生）。

#### 因此建议的顺序：先基线，后训练

这比原计划（先跑训练、day02 设检查点）更好，理由有两条：

1. **基线是读懂任何训练结果的前提**，包括轨迹级那次。
2. **它把轨迹级方案的成败条件（前缀重建）提前到一次不花训练算力的运行里回答**。如果 ① 大面积失败，我们在基线跑完就知道，不用等训练跑到 day02。

#### 已知的小风险

K=6 那次是先训了 6 天、后面 24 天空转；**K=0 是从头到尾一步都不训**，`_drain_output_queue` 会一直等不到 8 个组、每 30 秒打一行 "waiting for combine samples"。这在 K=6 的冻结尾段已经实际跑过，预期无害，但"从第一步就没有任何 batch"这个具体状态没有跑过——如果训练进程有任何"至少要完成一步"的隐含前提，会在这里暴露。观察点：训练进程不退出、SGLang 正常服务（它从 base HF checkpoint 起，本来就不依赖任何训练步）。

---

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
