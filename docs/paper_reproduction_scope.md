[← 工作记录](work_log.md)

# OpenClaw-RL 论文完整复现范围

> 基于对原文（arXiv:2603.10165）的完整阅读（2026-06-22）。  
> 本文件是复现决策的权威参考，遇到"该怎么做"的问题先看这里。

---

## 论文结果地图

论文不只是"五个 track"，共四块实验：

| 实验块 | 对应论文位置 | 核心指标 |
|--------|------------|---------|
| Personal Agent | Section 4.1 / 4.4，Table 3 | 达到优化效果所需最少 session 数 |
| General Agent | Section 4.2 / 4.5，Figure 5 | 训练曲线（rollout accuracy） |
| Hybrid RL Extension | Section 4.3 / 4.7，Figure 6 | 训练曲线（Retool accuracy / AIME score） |
| Ablation | Section 4.6–4.11，Tables 4–5，Figure 7 | 各方法对比 |

---

## 一、Personal Agent（Table 3）

### 完整配置

| 组件 | 论文值 | 说明 |
|------|--------|------|
| Policy | Qwen3-4B-Thinking-2507 | 被训练的模型 |
| PRM | Qwen3-4B-Thinking-2507 | 与 policy 同一模型 |
| Simulator | **Qwen3-32B** | 扮演 student / TA / teacher |
| Evaluator | **无独立模型** | 见下方"评估指标"说明 |
| 训练硬件 | 4 GPU policy actor + 2 GPU policy server + 1 GPU PRM Judge (SGLang，打分+生成 hint) + 1 GPU PRM Teacher (Megatron，teacher log-probs) | 共 8 GPU |
| 数据集 | GSM8K | 已在仓库中 |
| learning rate | 1×10⁻⁵ | |
| log-prob clip C | 1 | |
| batch trigger | 每收集 16 个 sample 触发一次训练 | |

### 三种用户场景

| 场景 | 适应类型 | 目标 | Simulator 判断标准 | Table 3 Joint 收敛 |
|------|---------|------|------------------|-------------------|
| Student | **Suppress**（抑制 AI 格式） | 回复不要有 AI 格式感 | 无 `**bold**`、无编号列表、无 `\boxed{}` → 说 DONE | 11.6（最慢，需主动压制已有风格）|
| TA | **Amplify**（放大输出长度）| 批改评语要足够详细 | 回复 > 100 tokens → 说 DONE | 8.2（最快，量化标准明确）|
| Teacher | **Add**（添加暖词风格）| 评论要友好耐心 | 包含 "well done" / "excellent" 等暖词 → 说 DONE | 11.4（需新增原本缺失的风格）|

### ⚠️ Table 3 的评估指标（关键，最容易理解错）

**Table 3 报告的是 session 计数，不是 0-1 分。**

- 指标定义："达到优化效果所需的最少 session 数"
- 优化效果定义："policy 对每个 session 第一条消息的回复，在**连续 3 个 session** 中均满足用户偏好"
- 判断方式：**rule-based**（见上方 Simulator 判断标准），不调用任何 LLM

```python
# 正确的 Table 3 评估逻辑（伪代码）
consecutive_pass = 0
total_sessions = 0
for session in training_sessions:
    total_sessions += 1
    first_response = get_policy_first_response(session.question)  # turn_type="side"
    if satisfies_preference(first_response, scenario):            # rule-based
        consecutive_pass += 1
    else:
        consecutive_pass = 0
    if consecutive_pass >= 3:
        return total_sessions  # <- 这就是 Table 3 的数字
```

**官方代码 `gsm8k_personal_agent.py` 测的是 0-1 LLM 打分（`personalization_evaluator.py`），不是 Table 3 的指标。** 若要复现 Table 3，需按上述逻辑修改评估部分。

### Table 3 数字参考

```
              Hybrid RL  GRPO   OPD   Mem0  Cognee
Joint opt:
  Student      11.6      15.4  30.8   13.6  14.6
  TA            8.2      12.0  34.0   15.8  15.4
  Teacher      11.4      14.8  24.4   14.2  14.8
  Average      10.3      14.1  29.7   14.5  14.9

Separate opt:
  Student      19.2      22.8  34.6   13.4  15.6
  TA           11.8      22.4  36.0   16.0  14.8
  Teacher      14.0      18.0  17.6   15.8  15.0
  Average      15.0      21.1  29.4   15.1  15.1
```
（数字越小越好）

> ⚠️ 纠错记录（2026-06-23）：Teacher Joint Hybrid RL 原记为 14.8，从论文 HTML 版核实为 **11.4**。
> 验证：(11.6 + 8.2 + 11.4) / 3 = 10.4 ≈ 10.3（与 Average 一致）。

### Joint vs Separate 区别

> ⚠️ **架构重新核实（2026-07-23）：Joint 不是"自己顺序建立 homework1/homework2 再切并行"，而是复用 Separate 跑完之后留下的真实产物。** 完整推导过程见 [`issues_log.md`](issues_log.md) 2026-07-23 条目（"Joint 训练真实编排方式重新核实"+"homework1/homework2 该怎么建立"两条）。之前 `scripts/train_with_services.sh` 里的 INIT 阶段（用刚开始训练、接近基础模型状态的模型顺序跑 72 题拼凑 homework1/homework2）**不是论文的真实做法**，已确认需要改。

**证据来源（均为直接读原文确认，非推断）：**
- 正文 4.4 节（p.12）："In addition to optimizing the model for **a single user**, OpenClaw-RL supports **multiple individuals sharing the same model**, with the model **jointly optimized**..."——明确"single user"（Separate，每个角色独立模型）是基础情况，"joint"（多角色共用同一模型）是额外能力，两者训练出来的是**不同的模型**
- 附录 A.1（p.21，Evaluation Configurations）："The TA setting can only be evaluated after the student setting has been completed. Similarly, the teacher setting can only be evaluated after the TA setting has been completed."——这句在"joint optimization setting"之前，是对 student/TA/teacher 三个独立设置本身的通用顺序约束（对应 Separate 内部依赖）
- 附录 A.1："In the joint optimization setting, we first save the directory completed by the student **as homework1**, and save the directory completed by the TA **as homework2**. We then start joint optimization... simultaneously."——**错位命名**：Student 自己完成的目录（正常语境下是 `homework/`）被存成"homework1"喂给 Joint；TA 自己完成的目录（正常语境下是 `homework1/`）被存成"homework2"喂给 Joint

| 设定 | 训练方式 | 模拟结构 |
|------|---------|---------|
| **Separate opt** | 三个独立 job，每个只训练一种 persona，模拟循环只跑对应的一个 Simulator 脚本，**必须按 Student→TA→Teacher 顺序跑**（TA 需要 Student 产出的真实 `homework/` 才能开始，Teacher 需要 TA 产出的真实 `homework1/`） | Student 独立训练 job 跑完（最多 72 题）后，`homework/` 里留下真实、由**已训练模型**产出的解答；TA 独立训练 job 用这份 `homework/` 起步（`ensure_homework_dir()` 复制成 `homework1/`），跑完后 `homework1/` 里留下真实批改；Teacher 同理产出 `homework2/` |
| **Joint opt** | 一个 job，Student/TA/Teacher 三个 persona 混合训练**同一个全新的、从基础模型开始的模型**（不接续任何 Separate 的 checkpoint，只复用 Separate 的**文件产出**）| 启动前**错位预铺**三个目录：`homework/`=空模板（Joint-Student 自己重新解，从零开始）；`homework1/`=复制 **Separate-Student** 的 `homework/`（干净、没有批改过的内容，给 Joint-TA 当起点）；`homework2/`=复制 **Separate-TA** 的 `homework1/`（干净、没有点评过的内容，给 Joint-Teacher 当起点）。三角色从一开始就同时启动，各自独立处理这批（最多 72 道）题目，互不阻塞、不需要等待信号 |

**为什么 Joint 不能直接复用 Separate 各自的"最终产出"，必须错位一层：** TA_chat.py/teacher_chat.py 是"读某个目录、把自己的产出追加进同一个目录"（`ensure_homework_dir()` 只是把源目录整体复制过来当起点，之后角色自己在这份复制品上追加）。如果 Joint-TA 直接拿 Separate-TA 已经批改完的 `homework1/` 当自己的起点，文件里会已经有一份批改内容，Joint-TA 读文件时会被"已经有答案"干扰，测不出当前（联合训练中）模型自己的真实判断。错位一层（用上一个角色**自己产出**、还没被下一个角色动过的版本）就能保证 Joint-TA/Teacher 拿到的都是干净、没被后续角色写过的内容。

**为什么必须先跑 Separate：** ①Table 3 本来就要报 Separate 这一列的数字，无论如何都要做；②`TA_chat.py`/`teacher_chat.py` 的 `ensure_homework_dir()` 要求源目录（`homework`/`homework1`）必须已存在真实内容，否则直接报错退出——Joint 没有独立的建立机制，必须依赖某次真实训练（Separate）产出的内容作为起点。

**残留的不确定性（如实记录，未被证实或证伪）：** 附录 A.1"student/TA/teacher setting"这三个独立设置是不是就是 Table 3 报告的、完整训练过的 Separate 模型，还是仅仅是 Joint 实验自己需要的一次性铺垫（可能用的是训练程度较浅的模型），论文原文没有直接点名。但不管哪种解读，Phase A（Separate-Student）都是必须先做的第一步，不影响现在的实现顺序。

**产物永久存放路径（2026-07-23 确定）：** Separate 各阶段产出的 `homework/`/`homework1/`/`homework2/` 需要跨训练任务、长期复用，不能放在 `runtime/<run_id>/workspace/` 这种每次训练都重新生成时间戳的临时目录里（后面阶段找不到）。统一放在 `/dfs/data/openclaw-rl-project/table3-artifacts/`（持久化存储，不受 GPU 空闲回收影响）：
```
/dfs/data/openclaw-rl-project/table3-artifacts/
├── separate-student/homework/     # Phase A 产出（Separate-Student 真实解答）
├── separate-ta/homework1/         # Phase B 产出（Separate-TA 真实批改）
└── separate-teacher/homework2/    # Phase C 产出（Separate-Teacher 真实点评，Joint 不需要，但 Table 3 Separate 列需要）
```

### Table 3 完整复现执行路线

> ⚠️ **优先级顺序已重排（2026-07-23）：Separate（Phase 3a-3c）从"低优先级、Joint 之后再做"改为 Joint 的前置依赖，必须先做。** 原因见上方"Joint vs Separate 区别"——Joint 需要直接复用 Separate 跑完后留下的 `homework/`/`homework1/`（错位一层），`train_with_services.sh` 现有的 INIT 阶段（自己拼凑 homework1/homework2）不是论文真实做法，需要改成直接消费 Separate 的产物。

**Phase 3：Separate Block — Hybrid RL 列（当前阶段，Joint 的前置依赖）**

三个独立 job，每个模拟循环只保留一个 persona，**必须按顺序跑**（TA 依赖 Student 产出的 `homework/`，Teacher 依赖 TA 产出的 `homework1/`），每个阶段跑完后把产出复制进 `/dfs/data/openclaw-rl-project/table3-artifacts/` 永久保存：

| 步骤 | 训练内容 | 需要的脚本 | 状态 |
|------|---------|----------|------|
| **3a** | **Separate 训练，只训 Student，最多 72 题，跑完停止训练** | `train_separate_student.sh` | 🔨 **当前正在做** |
| 3b | Separate 训练，只训 TA（依赖 3a 产出的 `homework/`，复制进这次训练的起始 workspace） | `train_separate_ta.sh` | ⬜ 需写，待 3a 完成 |
| 3c | Separate 训练，只训 Teacher（依赖 3b 产出的 `homework1/`，复制进这次训练的起始 workspace） | `train_separate_teacher.sh` | ⬜ 需写，待 3b 完成 |
| 3d | 评估三个 checkpoint 各自的收敛 session，得到 Separate / Hybrid RL 列数字 | `check_convergence.py` | ⬜ |

**Phase 1：Joint Block — Hybrid RL 列（待 Phase 3 完成后才能正确跑）**

| 步骤 | 操作 | 脚本 | 状态 |
|------|------|------|------|
| 1a | 改 `train_with_services.sh`：去掉自己拼凑 homework1/homework2 的 INIT 阶段，改成启动前从 `table3-artifacts/` 错位复制（`separate-student/homework/`→本次 `homework1/`，`separate-ta/homework1/`→本次 `homework2/`），三角色从一开始就同时启动 | `scripts/train_with_services.sh` | ⬜ 需改（待 Phase 3 产出真实数据后设计细节） |
| 1b | Joint 训练，Hybrid RL，全新基础模型开始 | 同上 | ⬜ |
| 1c | 评估：3 个 persona 各数收敛 session | `scripts/check_convergence.py` | ✅ 脚本本身已写 |
| 1d | 得到 Joint / Hybrid RL 列：Student=?, TA=?, Teacher=? | — | ⬜ |

**Phase 2：Joint Block — 基线列（GRPO、OPD）**

每列结构与 Phase 1 相同，只换训练脚本（模拟循环不变，三个 persona 仍全部保留，同样依赖 Phase 3 的产物）：

| 步骤 | 操作 | 训练后端 | 状态 |
|------|------|---------|------|
| 2a | Joint 训练，GRPO only | `openclaw-rl/run_qwen3_4b_openclaw_rl.sh` | ⬜ 需写 `train_grpo_joint.sh` |
| 2b | 评估 GRPO 列 | `check_convergence.py` | ⬜ |
| 2c | Joint 训练，OPD only | `openclaw-opd/run_qwen3_4b_openclaw_opd.sh` | ⬜ 需写 `train_opd_joint.sh` |
| 2d | 评估 OPD 列 | `check_convergence.py` | ⬜ |

**Phase 4：Separate Block — 基线列（低优先级）**

同 Phase 3，分别换 GRPO / OPD 训练后端，共再跑 6 个 job。

**Phase 5：Mem0 / Cognee 基线（独立生态，最低优先级）**

**核心区别：Mem0 / Cognee 不训练模型，模型权重全程不动。**

适应机制是"上下文注入"而非梯度更新：每个 session 结束后把用户偏好存入外部记忆库，下一个 session 推理前检索相关记忆并注入 prompt。这就是论文所说的"impose additional context overhead at inference time"。

| 项目 | 说明 |
|------|------|
| Mem0 | `pip install mem0ai`；向量数据库型记忆系统（Chhikara et al., 2025）|
| Cognee | `pip install cognee`；知识图谱型记忆系统（Markovic et al., 2025）|
| 基座模型 | 固定的 Qwen3-4B-Thinking-2507，权重不更新 |
| 评估逻辑 | 与 RL 方法完全相同：rule-based session 计数，连续 3 session 通过即收敛 |

**实现步骤（当时再写具体代码）：**

1. 安装对应库，初始化记忆客户端
2. 在 `student_chat.py` / `TA_chat.py` / `teacher_chat.py` 的 session 循环外层包一层记忆读写逻辑：
   - session 开始前：检索与当前用户相关的记忆，拼入 system prompt
   - session 结束后：把本次对话写入记忆库
3. 不启动 OpenClaw gateway（不需要 RL 训练管道），直接调用固定模型推理
4. 用同一个 `check_convergence.py` 统计收敛 session 数

**注意：** 论文中 Mem0/Cognee 在 joint 设置下的数字与 separate 设置几乎相同（joint avg 14.5 vs separate 15.1），因为记忆是 per-user 独立存储的，多用户并行不带来协同效应，这与 RL 方法的 joint 大幅提升形成对比。

---

**优先级建议（2026-07-23 更新，Separate 提前）：**  
Phase 3（Separate / Hybrid RL 列，Joint 前置依赖）→ Phase 1（Joint / Hybrid RL 列）→ Phase 2（Joint 基线）→ Phase 4（Separate 基线）→ Phase 5  
Phase 3 + Phase 1 完成后即可验证主结论（Joint Hybrid RL < Joint GRPO < Joint OPD，且 Joint < Separate 体现协同效应）。

---

## 二、General Agent（Figure 5）

### 四个 Track 配置

| Track | Policy 模型 | PRM 模型 | 训练数据集 | 并行环境数 | 评估集 |
|-------|------------|---------|----------|-----------|--------|
| Terminal | Qwen3-8B | Qwen3-4B | SETA RL data | 128 | SETA（rollout accuracy） |
| GUI | Qwen3VL-8B-Thinking（多模态）| Qwen3VL-8B-Thinking | OSWorld-Verified | 64 | OSWorld-Verified（不含 chrome/multi-apps）|
| SWE | Qwen3-4B | — | SWE-Bench-Verified | 64 | SWE-Bench-Verified |
| Tool-call | Qwen3-4B-SFT | Qwen3-4B | DAPO RL data | 32 | AIME 2024 |

> Qwen3-4B-SFT = 在 Retool dataset 上 SFT 过的 Qwen3-4B（来自 Zhu et al. 2025/slime）

### 超参数（共用）

| 参数 | 值 |
|------|-----|
| learning rate | 1×10⁻⁶ |
| KL coefficient | 0.01 |
| clip ε / εhi | 0.2 / 0.28 |
| 每步 tasks | 8（GUI/SWE）/ 16（terminal）/ 32（tool-call）|
| 每 task rollouts | 8 |
| max steps | 30（GUI）/ 20（SWE）/ 10（terminal）|

### 环境复杂度排序

```
Terminal < Tool-call < SWE < GUI
（难度递增，GUI 需要桌面 + 截图环境，最难搭建）
```

---

## 三、Hybrid RL Extension（Figure 6）

两个独立实验，与上面五个 track 无关：

### ReTool 多轮 RL
| 组件 | 值 |
|------|-----|
| Policy | Retool-4B（Qwen3-4B 在 Retool 数据集上 SFT） |
| PRM | Qwen3-8B |
| 数据 | Retool dataset |
| 评估 | Retool accuracy（训练曲线） |

### RLVR（AIME）
| 组件 | 值 |
|------|-----|
| Policy | DeepSeek-R1-Distill-Qwen-1.5B |
| PRM | Qwen3-4B |
| 数据 | DAPO RL data |
| 评估 | AIME 2024（20 次独立运行取平均）|

---

## 四、Ablation（Tables 4–5，Figure 7）

| 实验 | 变量 | Policy | 数据集 |
|------|------|--------|--------|
| Table 4：hint selection | sequence-optimal / token-optimal / random | Qwen3-32B | Personal Agent（joint opt）|
| Table 4：model size | Qwen3-32B vs Qwen3-4B policy | Qwen3-32B | Personal Agent |
| Table 5：k 值 | k=2/4/8/20 | Qwen3-4B | Personal Agent（joint opt）|
| Table 5：support set | 学生 top-k vs 重叠 top-k | Qwen3-4B | Personal Agent |
| Figure 7：log-prob clip | 有/无 clip | — | Retool + RLVR |

> Ablation 全部用 joint optimization 设定，共 5 次独立实验取均值。

---

## 模型清单

| 模型 | 大小 | 用于 | 是否需要下载 |
|------|------|------|------------|
| Qwen3-4B-Thinking-2507 | 7.6GB | Policy + PRM（Personal Agent / SWE / tool-call PRM）| ✅ 已在 modelfactory |
| **Qwen3-32B** | ~64GB BF16 | Simulator（论文原版，**已确认使用**）| ✅ 已在 modelfactory |
| Qwen3.5-122B-A10B-GPTQ-Int4 | ~65GB | Simulator 备用（已决定不用于复现）| 下载中（搁置）|
| Qwen3-8B | ~5GB | Terminal policy / ReTool PRM | ❌ 未下载 |
| Qwen3VL-8B-Thinking | ~8GB | GUI policy + PRM | ❌ 未下载 |
| Qwen3-4B-SFT | ~8GB | Tool-call policy | ❌ 未下载（需找 slime 提供的版本）|
| Retool-4B | ~8GB | ReTool RL policy | ❌ 未下载 |
| DeepSeek-R1-Distill-Qwen-1.5B | ~1.5GB | RLVR policy | ❌ 未下载 |

---

## 仓库目录归属（论文相关 vs 无关）

通过 git log 逐一确认各目录的提交时间（论文提交日期：**2026-03-11**）：

### 论文相关目录（≤ 2026-03-11 存在）

| 目录 | 对应论文方法 | 说明 |
|------|------------|------|
| `openclaw-rl/` | **Binary RL（GRPO only）** | Table 3 的 "GRPO" 基线，不是主方法 |
| `openclaw-opd/` | **OPD only** | Table 3 的 "OPD" 基线，不是主方法 |
| **`openclaw-combine/`** | **Hybrid RL = GRPO + OPD** | **Table 3 "Hybrid RL (Ours)"，论文主方法** |
| `openclaw-test/` | Personal Agent 评估套件 | `student_chat.py` / `TA_chat.py` / `teacher_chat.py`，连接 port 18789 |
| `gui-rl/` | GUI track | General Agent Figure 5 |
| `swe-rl/` | SWE track | General Agent Figure 5 |
| `terminal-rl/` | Terminal track | General Agent Figure 5 |
| `toolcall-rl/` | Tool-call track | General Agent Figure 5 |
| `Megatron-LM/` | 训练引擎 | 修改版 Megatron |
| `slime/` | RL 框架 | 异步 RL 基础设施 |

### 论文无关目录（论文提交后新增）

| 目录 | 加入时间 | 来源 | 说明 |
|------|---------|------|------|
| `openclaw-rl/oel/` | 2026-04-20 | 外部贡献者 PR #96 | OEL 模块，与论文无关 |
| `openclaw-fireworks/` | 2026-03-14+ | 内部 | 云训练集成，与论文无关 |
| `openclaw-tinker/` | 2026-03-13+ | 内部 | 论文无关 |

> **结论：复现过程中只使用论文相关目录，其余一律忽略。**

---

## ⚠️ 重要发现：OEL 模块与主论文的关系

**`openclaw-rl/oel/` 是一个独立模块，与 Table 3 的 RL 方法无关。**

通过 git log 确认：`personalization_evaluator.py` 和 `oel/eval/gsm8k_personal_agent.py` 均属于 commit `a111c68`（"feat: add openclaw-rl/oel module for OEL (Online Experiential Learning)"），不是主论文 RL 方法的一部分。

| | OEL 模块（`oel/`）| 主论文（Table 3）|
|--|---------|----------------|
| 方法 | 经验增强 distillation（top-K reverse KL）| Hybrid RL（GRPO + OPD）|
| Policy | Qwen3-**1.7B** | Qwen3-**4B**-Thinking-2507 |
| 评估指标 | 0-1 personalization score（GPT-4o 5-vote 打分）| session 计数（rule-based）|
| 数据集 | 36 道难 GSM8K（accuracy ≤ 0.25）| 全量 GSM8K（最多 72 个 session）|
| 训练脚本 | `run_qwen3_1.7b_openclaw_oel_online.sh` | `run_qwen3_4b_openclaw_rl.sh` |
| Simulator | GPT-4.1（OEL README 明确）| Qwen3-32B（主论文 Section 4.1 明确）|
| Evaluator | GPT-4o（OEL README 明确）| **无独立模型**（rule-based）|

`personalization_evaluator.py` 存在的原因：OEL 本身以 LLM 打分作为核心评估方式，不是给 Table 3 的 RL 方法用的。

**直接后果（已全部解决，2026-06-22）：**
- `oel/eval/gsm8k_personal_agent.py` 是 OEL 专用脚本，不能用于 Table 3 复现
- 主论文客户端已确认：`openclaw-test/student_chat.py` / `TA_chat.py` / `teacher_chat.py`
- Table 3 评估已确认：rule-based session 计数（见上方"评估指标"说明），无需修改为 LLM 打分

---

## ⚠️ 重大方向错误（2026-06-22 发现）

### 错误 1：训练脚本用错了

| | 我们一直在配置 | 应该用 |
|--|-------------|--------|
| 脚本 | `openclaw-rl/run_qwen3_4b_openclaw_rl.sh` | **`openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh`** |
| 方法 | Binary RL（GRPO only）= Table 3 "GRPO" 基线 | **Hybrid RL（k=4, m=3, seq-optimal）= Table 3 "Hybrid RL (Ours)"** |

`openclaw-rl/` 是单独的 GRPO 方法，不是论文的主贡献。论文主方法（Hybrid RL）对应 `openclaw-combine/` 下的 topk-select 脚本（k=4, m=3；Table 5 验证：k=4 → avg 10.3 = Table 3 主结果）。

> ⚠️ 二次更正（2026-06-29）：`openclaw-combine/` 目录下存在两个脚本，`run_qwen3_4b_openclaw_combine.sh`（basic combine，m=1，无 k）是简化版，**不是论文主方法**；`run_qwen3_4b_openclaw_topk_select.sh`（k=4, m=3）才是论文 Table 3 结果对应的脚本。

### 错误 2：客户端脚本用错了

| | 我们一直在计划用 | 应该用 |
|--|--------------|--------|
| 脚本 | `openclaw-rl/oel/eval/gsm8k_personal_agent.py` | **`openclaw-test/student_chat.py` + `TA_chat.py` + `teacher_chat.py`** |
| 所属 | OEL 模块（论文无关）| **论文评估套件** |
| 评估指标 | LLM 0-1 分（OEL 专用）| rule-based + session 计数（Table 3 指标）|

`openclaw-test/` 是论文原版的评估脚本，连接 port 18789（OpenClaw gateway）。

### ✅ 训练流程已确认（2026-06-22）

详见 `docs/implementation_path.md`。核心架构：

```
Simulator（port 30001）→ openclaw-test/*.py →
OpenClaw gateway（port 18789，rl-training-headers 扩展）→
RL proxy server（port 30000，由 openclaw-combine/run_*.sh 启动）
```

> 2026-07-07 更新：`rl-training-headers` 扩展在当前 OpenClaw 版本里端到端不生效
> （实测证实，非论文/代码问题），已改用官方 headers 静态配置 + system prompt
> Runtime 行解析的替代方案，见 `work_log.md`/`issues_log.md` 2026-07-07 条目。

正确训练脚本：`openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh`（k=4, m=3, seq-optimal）  
正确客户端：`openclaw-test/student_chat.py` / `TA_chat.py` / `teacher_chat.py`

---

## 当前复现状态（2026-06-22，历史快照）

> **当前状态请看 [`work_log.md`](work_log.md) 的「当前状态」节。** 下方为 2026-06-22 时点的历史记录，保留作决策溯源用。

### 已完成
- [x] 环境搭建（conda env + 所有 GPU 编译依赖）
- [x] Policy 模型下载 + torch_dist 格式转换
- [x] 训练脚本路径配置
- [x] API 兼容补丁（responses.create → chat.completions.create）
- [x] Simulator 替代方案选型（Qwen3.5-122B-A10B-GPTQ-Int4）
- [x] 完整阅读论文，确认复现范围

### 进行中
- [ ] Qwen3.5-122B-A10B-GPTQ-Int4 本地下载（~65GB）

### 已确认
- **Evaluator**：rule-based session 计数（无需 LLM，已确认）
- **复现顺序**：Personal Agent → Tool-call → Terminal → SWE → GUI（难度递增）→ 详见 `implementation_path.md`

### 已确认
- **Simulator 模型**：**Qwen3-32B**（论文原版）。Qwen3.5-122B 本地备用，后续优化时使用。

### 下一步
- [ ] 确认 OpenClaw 能否安装到 modelfactory（最高优先级）
- [ ] 确认 Simulator 模型
- [ ] 申请 8×H20 训练 workspace
