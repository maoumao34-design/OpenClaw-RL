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

| 设定 | 训练方式 | 模拟结构 |
|------|---------|---------|
| **Joint opt** | 一个 job，Student/TA/Teacher 三个 persona 混合训练同一个模型 | **INIT 阶段**（一次性顺序）：Student → TA → Teacher 各跑 72 题，建立 `homework1/` `homework2/`；**Joint 阶段**（循环并行）：三 Simulator 同时运行，各操作独立目录 |
| **Separate opt** | 三个独立 job，每个只训练一种 persona | 模拟循环只跑对应的一个 Simulator 脚本 |

### Table 3 完整复现执行路线

**Phase 1：Joint Block — Hybrid RL 列（当前阶段）**

| 步骤 | 操作 | 脚本 | 状态 |
|------|------|------|------|
| 1a | Joint 训练，Hybrid RL | `scripts/train_with_services.sh` | ✅ 已写，待跑 |
| 1b | 评估：3 个 persona 各数收敛 session | `scripts/check_convergence.py` | ✅ 已写，待跑 |
| 1c | 得到 Joint / Hybrid RL 列：Student=?, TA=?, Teacher=? | — | ⬜ |

**Phase 2：Joint Block — 基线列（GRPO、OPD）**

每列结构与 Phase 1 相同，只换训练脚本（模拟循环不变，三个 persona 仍全部保留）：

| 步骤 | 操作 | 训练后端 | 状态 |
|------|------|---------|------|
| 2a | Joint 训练，GRPO only | `openclaw-rl/run_qwen3_4b_openclaw_rl.sh` | ⬜ 需写 `train_grpo_joint.sh` |
| 2b | 评估 GRPO 列 | `check_convergence.py` | ⬜ |
| 2c | Joint 训练，OPD only | `openclaw-opd/run_qwen3_4b_openclaw_opd.sh` | ⬜ 需写 `train_opd_joint.sh` |
| 2d | 评估 OPD 列 | `check_convergence.py` | ⬜ |

**Phase 3：Separate Block — Hybrid RL 列**

三个独立 job，每个模拟循环只保留一个 persona：

| 步骤 | 训练内容 | 需要的脚本 | 状态 |
|------|---------|----------|------|
| 3a | Separate 训练，只训 Student | `train_separate_student.sh` | ⬜ 需写 |
| 3b | Separate 训练，只训 TA | `train_separate_ta.sh` | ⬜ 需写 |
| 3c | Separate 训练，只训 Teacher | `train_separate_teacher.sh` | ⬜ 需写 |
| 3d | 评估三个 checkpoint 各自的收敛 session | `check_convergence.py` | ⬜ |

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

**优先级建议：**  
Phase 1 → Phase 2 → Phase 3（Hybrid RL 列）→ Phase 3（基线）→ Phase 4 → Phase 5  
Phase 1 完成后即可验证主结论（Joint Hybrid RL < Joint GRPO < Joint OPD）。

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
