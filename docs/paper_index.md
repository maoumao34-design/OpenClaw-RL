[← 工作记录](work_log.md)

# OpenClaw-RL 论文索引

> **用途**：快速定位原始论文内容，所有引用必须回到原始 PDF 核实。
> **不作内容复现**：本文件不包含改写或推断，仅记录页码和 PDF 直接提取的文字/数字。
>
> **原始 PDF 路径**：`D:\MAO\Claude\openclaw-rl\openclaw-rl-paper.pdf`（33 页）
> **arXiv**：2603.10165

---

## 目录：各节页码

| 内容 | 页码 |
|------|------|
| Abstract | p.1 |
| Section 1: Introduction | p.3–5 |
| Section 2: Infrastructure | p.5–6 |
| Section 3: Methodology | p.7–9 |
| Section 4: Experiments | p.10–14 |
| Section 5: Related Work | p.15–16 |
| References | p.17–20 |
| Appendix A: Experiment Details | p.21–30 |
| Appendix A.1: Personal RL Configurations | p.21 |
| Appendix A.2: General Agentic RL Hyperparameters | p.21 |
| Appendix A.3: Hybrid RL Extension Configurations | p.22 |
| Appendix A.4: Token-level OPD | p.22 |
| Appendix A.5: Token-Level Log-Probability Shift Analysis | p.22 |
| Appendix A.6: Algorithm Pseudocode | p.23 |
| Appendix A.7: Personal RL Prompt Templates | p.24–30 |
| Appendix B: Additional Experiment Results | p.31 |
| Appendix B.1: More Optimization Examples | p.31 |
| Appendix B.2: Response Length and Truncation Ratio | p.32 |
| Appendix B.3: PRM Ablation Results | p.33 |
| Appendix C: OPD Objective is Token-level KL | p.33 |

---

## 图表索引

| 编号 | 页码 | 内容说明 |
|------|------|---------|
| Figure 1 | p.1 | OpenClaw-RL 基础架构总图（Personal Agent + General Agent 两类流） |
| Figure 2 | p.4 | 优化前/后对话示例（Student / TA / Teacher 三种 persona） |
| Figure 3 | p.7 | 方法总图：Personal Agent（binary reward + on-policy distillation）；General Agent（step-wise reward + standardization） |
| Figure 4 | p.8 | Overlap-guided hint selection 方法图 |
| Figure 5 | p.11 | General Agent 四种设置（terminal/GUI/SWE/tool-call）训练曲线，横轴 Step |
| Figure 6 | p.12 | Hybrid RL 扩展：左 ReTool multi-turn RL；右 RLVR；横轴 Step |
| Figure 7 | p.13 | (1)-(2) hint selection 方法对比曲线；(3) log-prob difference 分布（极端值动机） |
| Figure 8 | p.33 | clipping vs non-clipping：average response length 和 truncation ratio |
| **Table 1** | **p.6** | 支持的 agent 设置及环境特征（5 行，含 Tool-call Horizon=Medium） |
| **Table 2** | **p.7** | Evaluative / Directive / Hybrid 三列信号特性对比 |
| **Table 3** | **p.11** | **主结果**：优化效率（session 数），Joint + Separate，5 种方法 |
| **Table 4** | **p.12** | 消融：模型和方法（Qwen3-32B，hint selection 方式对比） |
| **Table 5** | **p.14** | 消融：k 和支持集 S_i（student top-k vs top-k overlap） |
| **Table 6** | **p.21** | General Agentic RL 完整超参数表（Appendix A.2） |
| **Table 7** | **p.33** | PRM 消融：Qwen3-4B vs Qwen3-8B 作为 teacher（Appendix B.3） |

---

## 表格数据（直接从 PDF 提取，已核实）

### Table 1（p.6）——支持的 agent 设置

| Setting | Environment | Next-state signal | Horizon |
|---------|-------------|------------------|---------|
| OpenClaw | Personal devices | user response / tool-call results | Long |
| Terminal | Shell execution sandbox | stdout/stderr, exit code | Long |
| GUI | Screen state + accessibility tree | Visual state diff, task progress | Long |
| SWE | Code repository + test suite | Test verdicts, diff, lint output | Long |
| Tool-call | API/function execution | Return values, error traces | **Medium** |

### Table 2（p.7）——信号特性

| Property | Evaluative | Directive | Hybrid (Ours) |
|----------|-----------|-----------|---------------|
| Source | Scalar PRM vote | Hint-conditioned teacher | Both |
| Granularity | Sequence-level | Token-level | Mixed |
| Information per sample | 1 scalar | \|S_i\| log-prob gaps | 1 scalar + \|S_i\| gaps |
| Frequency | Every scored turn | Turns with meaningful hint | Every scored turn |

### Table 3（p.11）——主结果

> 指标：达到优化效果所需的最少 session 数（越小越好），5 次独立运行取均值，Qwen3-4B-Thinking-2507

| Setting | Hybrid RL (Ours) | GRPO | OPD | Mem0 | Cognee |
|---------|:---:|:---:|:---:|:---:|:---:|
| **Joint** | | | | | |
| Student | 11.6 | 15.4 | 30.8 | 13.6 | 14.6 |
| TA | 8.2 | 12.0 | 34.0 | 15.8 | 15.4 |
| Teacher | 11.4 | 14.8 | 24.4 | 14.2 | 14.8 |
| Average | **10.3** | 14.1 | 29.7 | 14.5 | 14.9 |
| **Separate** | | | | | |
| Student | 19.2 | 22.8 | 34.6 | 13.4 | 15.6 |
| TA | 11.8 | 22.4 | 36.0 | 16.0 | 14.8 |
| Teacher | 14.0 | 18.0 | 17.6 | 15.8 | 15.0 |
| Average | **15.0** | 21.1 | 29.4 | 15.1 | 15.1 |

### Table 4（p.12）——消融：模型和方法

> Qwen3-32B，joint 设置，5 次运行均值

| Setting | Hybrid RL seq-optimal | Hybrid RL token-optimal | Hybrid RL random | GRPO | OPD |
|---------|:---:|:---:|:---:|:---:|:---:|
| Student | 14.0 | 13.8 | 18.6 | 17.2 | 34.4 |
| TA | 9.6 | 10.0 | 12.6 | 12.0 | 29.8 |
| Teacher | 13.8 | 13.4 | 17.0 | 18.2 | 25.6 |
| Average | **12.5** | **12.4** | 16.1 | 15.8 | 29.9 |

### Table 5（p.14）——消融：k 和支持集

> Joint 设置，Qwen3-4B

**S_i = S^q_i（student top-k）：**

| Setting | k=2 | k=4 | k=8 | k=20 | token-level |
|---------|:---:|:---:|:---:|:----:|:-----------:|
| Student | 30.4 | 11.6 | 12.8 | 11.4 | 34.4 |
| TA | 12.0 | 8.2 | 7.6 | 7.8 | 36.0 |
| Teacher | 17.2 | 11.4 | 10.0 | 10.2 | 22.6 |
| Average | 20.2 | **10.3** | 10.1 | 9.8 | 31.0 |

**S_i = S^q_i ∩ S^p_{i,h★}（top-k overlap）：**

| Setting | k=2 | k=4 |
|---------|:---:|:---:|
| Student | 31.6 | 11.8 |
| TA | 14.0 | 16.4 |
| Teacher | 18.4 | 12.2 |
| Average | 21.3 | 13.5 |

### Table 6（p.21）——General Agentic RL 超参数

| Parameter | Value | Note |
|-----------|-------|------|
| Learning rate | 1×10⁻⁶ | constant decay |
| Weight decay | 0.1 | |
| Adam β₁, β₂ | 0.9, 0.98 | |
| KL coefficient β_KL | 0.01 | k3 / low-var KL |
| Clip ε / ε_high | 0.2 / 0.28 | asymmetric PPO |
| Entropy coefficient | 0.0 | disabled |
| Batch size | 8 (GUI, SWE), 16 (terminal), 32 (tool-call) | |
| Sample per task | 8 | |
| Max response length | 8192 tokens | |
| Max context length | 16384 tokens | |
| Max interactive steps | 30 (GUI), 20 (SWE), 10 (terminal) | |
| Temperature | 1.0 | |
| Votes m | 3 (GUI), 1 (the others) | majority vote |
| PRM Temperature | 0.6 | |

### Table 7（p.33）——PRM 消融

> Qwen3-4B-Thinking-2507 作为 policy，joint 设置

| Setting | Teacher = Qwen3-4B-Thinking-2507 | Teacher = Qwen3-8B |
|---------|:---:|:---:|
| Student | 14.0 | 13.6 |
| TA | 9.6 | 9.2 |
| Teacher | 13.8 | 14.2 |
| Average | **12.5** | **12.3** |

---

## 关键定义（PDF 原文逐字引用，含页码）

### 评估指标定义（p.10）

> "We consider the optimization effect to have been achieved once the model's response to the first message satisfies the user's preferences in three consecutive sessions."

### 三种 persona 的收敛规则（p.10）

> **Student**："A response is identified as AI-like when it contains markers such as bold text, numbered lists, or over-formatting like boxed final answers."（不满足 = 不含这些）
>
> **TA**："considers a grading response insufficient when its length is below 100 tokens."（满足 = ≥100 tokens）
>
> **Teacher**："wants comments to be friendly and patient, identified by warm phrases such as 'well done!' or 'excellent!'"（满足 = 包含这类词）

### 评估配置（p.21，Appendix A.1 原文）

> "By default, we set the conversation-session limit to 72, meaning that at most 72 tasks are used for evaluation."
>
> "The TA setting can only be evaluated after the student setting has been completed. Similarly, the teacher setting can only be evaluated after the TA setting has been completed."
>
> "In the joint optimization setting, we first save the directory completed by the student as homework1, and save the directory completed by the TA as homework2. We then start joint optimization, where the three simulated users use OpenClaw to conduct their work simultaneously."

### Personal Agent 训练配置（p.10 + p.21）

> **p.10**："The OpenClaw policy and reward model in this setting is Qwen3-4B-Thinking-2507. We set the learning rate to 1×10⁻⁵ and the log-probability-difference clipping coefficient to C=1, and trigger a training step after every 16 collected samples. Users are simulated with Qwen3-32B to ensure faithful role-following."
>
> **p.21**："we set w_RL = w_OPD = 1, the clipping constant to C=1, k=4, the number of hints generated per sample to 3, and the learning rate to 1×10⁻⁵."
>
> **GPU 分配（p.21）**："we use 4 GPUs for the policy actor, 2 GPUs for the policy server, 1 GPU for the PRM actor, and 1 GPU for the PRM server."

### Hybrid RL Extension 配置（p.11）

> "We set the learning rate to 10⁻⁶, the KL coefficient to 0.01, the log-probability-difference clipping coefficient to C=2, and the lower and upper PPO clip ratios to ε_lo=0.2 and ε_hi=0.28. We sample 32 tasks per training step, with the policy drawing 8 independent rollouts per task."

---

## 实验结果快查

| 实验 | 来源 | 结论 |
|------|------|------|
| Joint Hybrid RL 平均 sessions | Table 3 p.11 | **10.3** |
| Separate Hybrid RL 平均 sessions | Table 3 p.11 | **15.0** |
| tool-call outcome→integrated reward | p.12 | 0.19 → **0.25** |
| GUI outcome→integrated reward | p.12 | 0.31 → **0.33** |
| k=4 vs k=8 差异 | Table 5 p.14 | k=4: 10.3, k=8: 10.1（几乎无差） |
| Qwen3-4B vs Qwen3-8B PRM | Table 7 p.33 | 12.5 vs 12.3（相近） |
| clipping 截断比 | Figure 8 p.33 | clipping=0.2, non-clipping=0.5 |

---

## Prompt 模板页码（Appendix A.7）

| Prompt | 页码 |
|--------|------|
| PRM Evaluative Signal（Personal）| p.24 |
| PRM Directive Signal（Personal）| p.25–26 |
| ReTool Directive Signal Prompt | p.27 |
| Student Simulator Prompt | p.28–29 |
| TA Simulator Prompt | p.29–30 |
| Teacher Simulator Prompt | p.30–31 |
