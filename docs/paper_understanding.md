# OpenClaw-RL 论文理解

> arXiv: 2603.10165 | 随时更新，记录复现过程中的新认知

## 核心问题

现有 agentic RL 系统忽略了一个普遍存在的训练信号：**next-state signal**（下一状态信号）。每次 agent 执行一个动作后，环境都会返回一个状态变化（用户回复、终端输出、GUI 变化、测试结果），这些信号既包含"做得好不好"的评估信息，也包含"应该怎么做"的指令信息，但没有现有系统系统性地利用它们。

---

## 系统架构：四个异步解耦的循环

```
Policy Server (SGLang)  ←→  Environment Server
        ↓ (aₜ, sₜ₊₁)              ↓
  PRM/Judge Server    →   Async Buffer ℬ
                                   ↓
                          Trainer (Megatron)
                          GRPO + OPD 混合损失
```

四个组件**完全异步**，互不阻塞：
- **Policy Server**：用 SGLang 对外提供推理 API，接收带 session ID 的请求
- **Environment Server**：管理四类环境（Terminal/GUI/SWE/Tool-call），产生 next-state signal
- **PRM/Judge Server**：对每个 `(aₜ, sₜ₊₁)` 对进行评分，提取评估信号和指令信号
- **Trainer**：用 Megatron 异步执行梯度更新，不干扰推理

**关键设计**：每个 HTTP 请求带 session ID，区分"主线 turn"（参与训练）和"side turn"（仅转发不训练）。

---

## 两类信号与训练目标

### 混合损失

$$\mathcal{L} = w_{RL} \cdot \mathcal{L}_{GRPO} + w_{OPD} \cdot \mathcal{L}_{OPD}$$

两者权重均为 1.0。

---

### 信号一：Evaluative Signal → GRPO

PRM Judge 对每个 turn 做 **m 次独立投票**（GUI 用 m=3，其余 m=1），输出标量奖励 `r ∈ {+1, −1, 0}`。

Step-wise 奖励整合公式（用于长 horizon 任务）：

$$r_t = o + \sum_{i=1}^{m} r_i / m$$

其中 `o` 是最终 outcome 奖励，`rᵢ` 是每个 turn 的 PRM 判分。

Advantage 计算：按 step index 分组标准化，而非全轨迹聚类。

#### ⚠️ 实际 GRPO 实现细节（与标准 GRPO 的差异）

论文说 GRPO，但代码实现是 **PPO-style clipped surrogate**，不是标准 GRPO：

```python
# slime/slime/utils/ppo_utils.py: compute_policy_loss()
ratio = exp(-ppo_kl)                          # ppo_kl = log π_old - log π_new
pg_loss1 = -ratio * advantages                # 未裁剪项
pg_loss2 = -clip(ratio, 1-eps_lo, 1+eps_hi) * advantages  # 裁剪项
loss = max(pg_loss1, pg_loss2)                # 取较大值（更保守）
```

**与标准 GRPO 的关键差异：**

| 项目 | 标准 GRPO | 实际实现 |
|------|----------|---------|
| 裁剪方式 | 无裁剪或对称裁剪 | **非对称裁剪** eps_lo=0.2, eps_hi=0.28 |
| KL 惩罚 | 通常有 | kl_loss_coef=0.0（**实际关闭**）|
| reward 归一化 | 组内归一化 | `--disable-rewards-normalization`（**关闭**）|
| 优势聚合 | 组内标准化 | per-step-index 分组标准化 |
| loss 聚合 | mean | `sum_of_sample_mean`（sample 内 mean，sample 间 sum）|

**重要**：`--disable-rewards-normalization` 是默认开启的，意味着 reward 直接用 PRM 的原始分数 {+1, -1, 0}，不做归一化处理。这使得训练信号更稀疏但更准确。

---

### 信号二：Directive Signal → OPD（Hindsight-Guided On-Policy Distillation）

**问题**：直接用 teacher（带 hint 的模型）蒸馏 student，会因分布偏移导致训练不稳定。

**解决方案：Overlap-Guided Hint Selection**

有 M 个候选 hint，选出最能与 student 分布"重叠"的那个：

$$S_i^g = \text{top-k}\{\pi_{old}(\cdot | s_t, y_{<i})\} \quad \text{(student support)}$$

$$S_{i,h}^p = \text{top-k}\{\pi_t(\cdot | s_t^h, y_{<i})\} \quad \text{(teacher support under hint h)}$$

$$h^* = \arg\max_h \sum_i |S_i^g \cap S_{i,h}^p|$$

选出 `h*` 后，在 student 的 support set `Sᵢ` 上计算 per-token OPD loss：

$$\mathcal{L}_i^{OPD} = \sum_{v \in S_i} \max(-A_v \rho_v, -A_v \cdot \text{clip}(\rho_v, 1-\varepsilon_{lo}, 1+\varepsilon_{hi}))$$

其中 advantage `Aᵥ = Δᵥ · wᵥ`，`Δᵥ` 是 log prob 差（做了 clip，防止极端梯度）。

---

## 四类环境

| 环境 | 数据来源 | Next-state Signal | 并行度 |
|------|---------|------------------|-------|
| Terminal | SETA RL | stdout/stderr/exit code | 128 envs |
| GUI | OSWorld-Verified | 视觉状态差 + a11y tree | 64 envs |
| SWE | SWE-Bench-Verified | 测试结果/代码 diff | 64 envs |
| Tool-call | DAPO RL / Retool | API 返回值/错误 | 32 envs |
| Personal (OpenClaw) | GSM8K / 用户对话 | 用户回复/纠正 | — |

---

## 基础模型选择

| 组件 | 模型 | 说明 |
|------|------|------|
| Personal Agent Policy | Qwen3-4B-Thinking-2507 | **被训练的模型**，最终产出 |
| Terminal Agent | Qwen3-8B | — |
| GUI Agent | Qwen3VL-8B-Thinking（多模态）| — |
| SWE Agent | Qwen3-4B | — |
| Tool-call Agent | Qwen3-4B-SFT | — |
| PRM Judge | Qwen3-4B / Qwen3VL-8B-Thinking | **核心训练信号**，给每步打分 |
| Personal Agent Simulator | GPT-4.1（官方代码默认值）| 扮演 student/teacher，生成训练对话 |
| Evaluator | GPT-4o（官方代码默认值）| 仅用于评估，不参与训练 |

> **注**："用户模拟器 Qwen3-32B" 是早期笔记的错误归因。官方代码 `gsm8k_personal_agent.py` 明确使用 `gpt-4.1` 作为 Personal Agent track 的 simulator。Qwen3-32B 可能对应论文中其他 environment 的模拟器，尚未核实。

---

## 各模型在训练流程中的角色

### 训练循环（Training Loop）

![各模型角色图](openclaw_model_roles.svg)

```
Simulator ──请求──▶ Policy Model ──(action, next_state)──▶ PRM Judge
    ▲                    ▲                                       │
    │                    │ 更新权重（梯度）                    奖励信号
    └──回复继续对话──────┤                                  r ∈ {+1,−1,0}
                         │                                       │
                    Trainer (Megatron) ◀─────────────────────────┘
                    GRPO + OPD 梯度更新
```

**Policy Model（Qwen3-4B-Thinking-2507）**

唯一被训练的模型，也是训练完成后实际部署使用的产出。训练前是普通助手，训练后学会了在不同用户风格（懒学生/严格老师）下调整回答方式。

**PRM Judge（Qwen3-4B）**

每次 Policy 回复一句话，PRM 就判断"这一步做得好不好"，给 +1/−1/0 分。这个分数通过 GRPO 计算 advantage，驱动 Megatron 更新 Policy 权重。**训练信号的核心来源，不替代。**

**Simulator（论文 Personal Agent track：GPT-4.1）**

扮演"懒学生"或"挑剔老师"，不断给 Policy 发消息推动多轮对话。训练完成后直接丢弃，不出现在最终产品里。它的作用是提供足够自然的对话，让 Policy 有东西可以学——但训练信号本身来自 PRM，不来自 Simulator 的质量。

**Evaluator（GPT-4o）**

只在评估阶段使用，判断 Policy 的回复是否符合"学生风格"或"老师风格"，产出 Table 3 的数值（0.17 → 0.76 → 0.81）。与训练完全无关。

### 评估阶段（Evaluation Only）

```
Evaluator ──测试请求──▶ Policy Model（turn_type="side"，不产生训练数据）
    ▲                         │
    └──── 个性化得分 ◀─────────┘
    （Table 3 数值来源）
```

---

## 本复现的模型替代方案

因无外部 API 访问，Simulator 和 Evaluator 均用开源模型自托管替代。

### 选型结论：Qwen3.5-122B-A10B

**选择依据（对比 DeepSeek V4）：**

| 维度 | Qwen3.5-122B-A10B | DeepSeek V4-Flash |
|---|---|---|
| 任务匹配 | IFBench **76.5**（全模型最高），专项强调 roleplay/多轮对话/instruction following | 擅长 coding/math，非本任务优势方向 |
| 与 Policy 兼容性 | 同 Qwen 系列，tokenizer 和 chat template 完全兼容 | 不同系列 |
| SGLang 支持 | 原生支持，官方有专项文档；AMD 官方博客有 OpenClaw+Qwen3.5+SGLang 验证案例 | Day-0 支持，较新，edge case 未充分 hardened |
| No-think 模式 | `enable_thinking: false`，roleplay 场景直接关掉推理 | Non-think 模式行为较新 |
| 激活参数 | 仅 **10B active**（总量 122B MoE），推理轻量 | ~13B active，但权重体积 ~175GB VRAM |
| H20 部署 | TP=2 即可跑，独立小 workspace | 需要较大配置，V4-Pro（500GB FP8）完全不现实 |

### 替代方案汇总

| 组件 | 论文原版 | 本复现替代 | 影响 |
|------|---------|-----------|------|
| Simulator | GPT-4.1 | **Qwen3.5-122B-A10B**（自托管）| 对话风格略有差异，训练信号（PRM）不变 |
| Evaluator | GPT-4o | **Qwen3.5-122B-A10B**（自托管）| 数值与论文不可直接对比，内部前后一致 |
| Policy | Qwen3-4B-Thinking-2507 | 同上（不替代）| — |
| PRM | Qwen3-4B | 同上（不替代）| — |

### 部署配置

```bash
# Simulator workspace（独立 2×H20）启动 Qwen3.5-122B-A10B
sglang serve \
  --model-path Qwen/Qwen3.5-122B-A10B \
  --tp 2 \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 --port 8001

# 训练侧环境变量
export OPENAI_BASE_URL="http://<simulator-host>:8001/v1"
export OPENAI_API_KEY="dummy"
export OPENAI_MODEL="Qwen3.5-122B-A10B"
```

### 切换回原始闭源模型

仅需修改环境变量，代码零改动：

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-xxxxx"
export OPENAI_MODEL="gpt-4.1"
```

---

## 关键超参数

| 参数 | Personal Agent | General Agent |
|------|---------------|---------------|
| 学习率 | 1e-5 | 1e-6 |
| OPD clip C | 1 | 2 |
| Top-k 宽度 K | 4 | 4 |
| PPO clip (εlo/εhi) | — | 0.2 / 0.28 |
| KL 系数 β | — | 0.01 |
| 最大响应长度 | 8192 tokens | 8192 tokens |
| 最大上下文长度 | 16384 tokens | 16384 tokens |

**GPU 分配**：Policy actor 4 GPU，Policy server 2 GPU，PRM actor 1 GPU，PRM server 1 GPU。

---

## 其他重要实现细节

### 1. OPD 使用全局 log-prob，不是 subset 归一化

代码注释里明确解释了为什么用 **GLOBAL ratio** 而不是 subset 内归一化：

> 如果用 subset 内归一化的 ratio，student 可以通过把 subset 外的质量压到接近零来"满足"约束，而不真正学习 teacher 的分布。用全局 ratio 才能让 IS 校正是诚实的。

```python
# ell_cur = raw_logits(v) - global_lse(raw_logits)  ← 全局 log-prob，有 autograd
# rho_v = exp(ell_cur - ell_old)                    ← 全局 ratio
```

实现上用了两个自定义 autograd Function：
- `_VocabParallelGatherRawLogits`：在 TP 分片的词表上 gather 指定 vocab id 的 raw logits
- `_VocabParallelGlobalLSE`：跨 TP 分片计算全局 log-sum-exp

**代价：** backward 时需要 materialize `[R, V_local]` 的全局 softmax，显存开销不可避免。

---

### 2. hint_opd_loss 和 openclaw_topk_select_loss 的默认权重不同

| 模块 | w_rl 默认值 | w_opd 默认值 | 含义 |
|------|-----------|------------|------|
| `hint_opd_loss.py` | **0.0** | 1.0 | 纯 OPD，无 GRPO |
| `openclaw_topk_select_loss.py` | **1.0** | 1.0 | GRPO + OPD 混合 |

论文的完整方法是 `openclaw_topk_select_loss`（w_rl=1.0, w_opd=1.0）。

---

### 3. 训练与提交的解耦机制

`openclaw_rollout.py` 里有一个关键的 **pause/resume 机制**：

```python
worker.resume_submission()   # 开放样本提交
completed_samples = drain_output_queue(...)  # 等待收集足够样本
worker.pause_submission()    # 暂停提交 + 清空 record 文件
# → 触发 Megatron 梯度更新
```

训练时 submission 被 **暂停**，防止训练期间的请求产生用旧权重生成的样本混入下一轮。这是论文里"异步但不污染"的核心保证。

---

### 4. old_log_probs 来源必须是 Megatron，不能用 SGLang rollout

代码里有明确的 assert：

```python
assert not getattr(args, "use_rollout_logprobs", False), (
    "hint_opd loss requires old-policy log-probs from Megatron old_actor, not SGLang rollout"
)
```

原因：SGLang rollout 输出的 log-prob 精度不够，OPD 需要精确的全局 log-prob 来计算 IS weight 和 advantage。

---

### 5. rollout_batch_size 决定每轮训练等待的样本数

`_drain_output_queue` 里等待 `args.rollout_batch_size` 个 group 才触发训练。
启动脚本里设置的是 `--rollout-batch-size 16`，即每收集 16 个 session turn 才更新一次参数。

等待超过 30 秒没有新样本时会打印进度日志，可以用来判断 pipeline 是否卡住。

## 复现难点

1. **异步基础设施**：四个组件完全解耦，需要实现可靠的 session 级消息路由和 buffer 同步机制
2. **Overlap-Guided OPD**：per-token 级别的 teacher/student 词表 top-k 交集计算，计算量敏感
3. **PRM Judge**：需要针对每类环境设计不同的 prompt 模板来提取评估/指令信号
4. **Megatron 集成**：论文用 Megatron-Core 做分布式训练，与 SGLang 的权重同步边界需要仔细设计
5. **环境多样性**：四类环境的 next-state signal 格式完全不同，需要统一抽象

---

## 复现进度备注

> 此节随项目推进持续更新

- [ ] 项目骨架搭建
- [ ] Async Buffer 实现
- [ ] Environment 抽象层（Terminal → GUI → SWE → Tool-call）
- [ ] PRM Judge Server
- [ ] Policy Server (SGLang)
- [ ] OPD loss 实现
- [ ] GRPO loss 实现
- [ ] Trainer (Megatron) 集成
- [ ] 端到端联调
- [ ] 实验评估
