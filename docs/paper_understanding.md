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

| 组件 | 模型 |
|------|------|
| Personal Agent Policy | Qwen3-4B-Thinking-2507 |
| Terminal Agent | Qwen3-8B |
| GUI Agent | Qwen3VL-8B-Thinking（多模态）|
| SWE Agent | Qwen3-4B |
| Tool-call Agent | Qwen3-4B-SFT |
| PRM Judge | Qwen3-4B / Qwen3VL-8B-Thinking |
| 用户模拟器 | Qwen3-32B |

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
