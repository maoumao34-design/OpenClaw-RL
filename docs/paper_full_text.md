# OpenClaw-RL: Train Any Agent Simply by Talking

> arXiv: 2603.10165 | 从 HTML 版本提取，供复现查阅原文用
> 提取日期：2026-06-23

---

## Abstract

Every agent interaction generates a next-state signal, namely the user reply, tool output, terminal or GUI state change that follows each action, yet no existing agentic RL system recovers it as a live, online learning source. We present OpenClaw-RL, a framework that employs next-state signals to optimize personal agents online through infrastructure and methodology innovations.

The system extracts two complementary training signals, **evaluative** and **directive**, via a separate asynchronous server, and introduces **overlap-guided hint selection** to stabilize distillation.

---

## Section 1: Introduction

The paper identifies a gap: "both the infrastructure and methodology for leveraging such usage data to improve language models in real time remain little explored."

OpenClaw-RL extends existing RL infrastructure to a server–client architecture where the RL server hosts the policy behind an inference API and user terminals stream interaction data back over HTTP.

**Evaluative signals** come from process reward models scoring actions implicitly (user re-queries signal dissatisfaction; passing tests signal success).

**Directive signals** are token-level hints extracted from next states—for example, "you should have checked the file first."

The framework addresses a core challenge: "training instability and reduced effectiveness due to the distribution mismatch between teacher and student," proposing solutions including hint selection based on "top-k token overlap between teacher and student distributions."

---

## Section 2: Infrastructure

### 2.1 Flexible Server–Client Architecture

"The RL server hosts the policy πθ behind a stateless completion API; users' agent frameworks, which may run on personal devices, terminals, or cloud instances, query the policy through this API and stream their interaction data back to the server over HTTP."

Each request is classified as either a **main-line turn** (trainable) or **side turn** (auxiliary). Session identifiers enable the server to "demultiplex concurrent interaction streams from multiple users and attribute each turn to the correct conversation session."

### 2.2 Asynchronous Signal Extraction

"The core architectural principle of OpenClaw-RL is full decoupling: policy serving, environment hosting, reward judging, and policy training run as four completely independent asynchronous components with no blocking dependencies between them."

The PRM operates as a separate inference server, extracting both evaluative scores (majority vote from m queries) and directive hints when next states contain "meaningful correction." This decoupling ensures "signal extraction can use a stronger model and run multiple votes per sample without affecting user-facing latency."

### 2.3 Scalability

The framework spans personal agents on user devices and general agents across terminal, GUI, SWE, and tool-call settings. Built on slime, OpenClaw-RL inherits a scalable training infrastructure for general agents and supports cloud-hosted environments across diverse agent settings.

**Table 1: Supported Agent Settings and Their Environment Characteristics**

| Agent Type | Environment | Next-State Signal | Horizon |
|-----------|-------------|------------------|---------|
| OpenClaw (Personal) | Personal devices | User responses / tool results | Long |
| Terminal | Shell sandbox | stdout / stderr / exit codes | Long |
| GUI | Screen state | Visual diffs / a11y tree | Long |
| SWE | Code repository | Test verdicts / code diffs | Long |

**Table 2: Complementary Properties of Evaluative and Directive Signals**

| Property | Evaluative Signal (Scalar PRM) | Directive Signal (Hint-Conditioned Teacher) | Hybrid |
|----------|-------------------------------|---------------------------------------------|--------|
| Granularity | Scalar per turn | Per-token distribution | Both |
| Information density per sample | Low | High | High |
| Frequency | Every scored turn | Only when next-state has correction | Mixed |

---

## Section 3: Methodology

### 3.1 Two Complementary Signals and A Hybrid RL Objective

**Evaluative Signal:**
The Process Reward Model receives queries about $(a_t, s_{t+1})$ and returns majority vote $r_t \in \{+1, -1, 0\}$. This signal is dense—every scored turn produces a sample regardless of explicit user feedback.

**Directive Signal:**
When $s_{t+1}$ contains meaningful correction, the PRM extracts a concise hint $h$ between `[HINT_START]...[HINT_END]`. A teacher distribution $\pi_T(\cdot|s_t^h)$ is obtained by conditioning on the hint-augmented prompt. This signal is sparse, firing only when the next state carries extractable correction.

**Complementary Analysis:**
The signals differ in frequency and information density. Evaluative signals appear on every scored turn but compress to scalars. Directive signals carry per-token guidance via teacher distribution but only appear when meaningful corrections exist. RLVR methods consume only evaluative signals; pure on-policy distillation consumes only directive signals.

**Hybrid Objective:**

$$\mathcal{L}^{\text{hybrid}}_i = w_{\text{RL}} \mathcal{L}^{\text{GRPO}}_i + w_{\text{OPD}} \mathcal{L}^{\text{OPD}}_i$$

where $\mathcal{L}_i^{\text{GRPO}}$ is standard PPO clipped surrogate driven by scalar advantage $A_i^{\text{grpo}}$, and $\mathcal{L}_i^{\text{OPD}}$ is distillation loss. By default $w_{\text{RL}} = w_{\text{OPD}} = 1$.

### 3.2 Overlap-Guided Hint Selection

**Overlap as Selection Signal:**

Let $S_i^q = \text{top-k}\{\pi_{\text{old}}(\cdot|s_t, y_{<i})\}$ denote student's top-k vocabulary at position $i$, and $S_{i,h}^p = \text{top-k}\{\pi_T(\cdot|s_t^h, y_{<i})\}$ denote teacher's top-k under hint $h$.

Overlap signal: $O[h,i] = |S_i^q \cap S_{i,h}^p|$

Two selection schemes are considered:
- **Sequence-level:** $h^* = \arg\max_h \sum_i O[h,i]$
- **Token-level:** $h^*(i) = \arg\max_h O[h,i]$

**Top-k OPD Loss with Log-Probability-Difference Clip:**

For $v \in S_i$:

$$w_v = \text{softmax}_{v \in S_i}(\ell_{\text{old}}(v))$$

$$\Delta_v = \text{clip}(\ell_{T,h^*}(v) - \ell_{\text{old}}(v),\ -C,\ +C)$$

$$A_v = \Delta_v \cdot w_v$$

With per-vocab ratio $\rho_v = \exp(\ell_{\text{cur}}(v) - \ell_{\text{old}}(v))$:

$$\mathcal{L}^{\text{OPD}}_i = \sum_{v \in S_i} \max\!\left(-A_v \rho_v,\ -A_v \cdot \text{clip}(\rho_v,\ 1-\varepsilon_{\text{lo}},\ 1+\varepsilon_{\text{hi}})\right)$$

where $\varepsilon_{\text{lo}} = 0.2$ and $\varepsilon_{\text{hi}} = 0.28$.

### 3.3 Step-wise Reward for General Agentic RL

Long-horizon agentic tasks suffer from sparse terminal rewards. Process Reward Models assign step-level rewards from next-state signals, enabling dense credit assignment throughout trajectories.

Outcome and process rewards are combined: $o + \sum_{i=1}^{m} r_i / m$, where $r_i$ are independently assigned by $\text{PRM}(a_t, s_{t+1})$. For advantage computation, actions with the same step index are grouped.

---

## Section 4: Experiments

### 4.1 Personal Agent Setup

Three user types interact with OpenClaw on GSM8K tasks:

| User Type | Preference | Rule |
|-----------|-----------|------|
| Student | Non-AI-like formatting | No bold, no numbered lists, no `\boxed{}` |
| TA | Specific, detailed grading | ≥ 100 tokens in response |
| Teacher | Friendly, patient comments | Contains "well done" / "excellent" / "great job" |

**Convergence criterion:** policy response to the first message of each session satisfies the rule for 3 consecutive sessions.

**Model configuration:**
- Policy: Qwen3-4B-Thinking-2507
- PRM/Judge: Qwen3-4B-Thinking-2507 (same model)
- Simulator: **Qwen3-32B** (plays student / TA / teacher)
- Evaluator: rule-based session counting, no LLM

**Hyperparameters:**
- Learning rate: $1 \times 10^{-5}$
- Log-probability clipping $C = 1$
- Top-k width $k = 4$
- Batch trigger: 16 samples

### 4.2 General Agent Setup

| Track | Policy | PRM | Dataset | Parallel Envs | Eval |
|-------|--------|-----|---------|--------------|------|
| Terminal | Qwen3-8B | Qwen3-4B | SETA RL data | 128 | SETA rollout accuracy |
| GUI | Qwen3VL-8B-Thinking | Qwen3VL-8B-Thinking | OSWorld-Verified | 64 | OSWorld-Verified |
| SWE | Qwen3-4B | — | SWE-Bench-Verified | 64 | SWE-Bench-Verified |
| Tool-call | Qwen3-4B-SFT | Qwen3-4B | DAPO RL data | 32 | AIME 2024 |

**Shared hyperparameters:**
- Learning rate: $1 \times 10^{-6}$
- KL coefficient: 0.01
- Clip: $\varepsilon_{\text{lo}} = 0.2$, $\varepsilon_{\text{hi}} = 0.28$
- Batch: 8 tasks (GUI/SWE), 16 (terminal), 32 (tool-call), 8 rollouts per task
- Max steps: 30 (GUI), 20 (SWE), 10 (terminal)

### 4.3 Hybrid RL Extension Setup

| Experiment | Policy | PRM | Dataset |
|-----------|--------|-----|---------|
| ReTool multi-turn RL | Retool-4B | Qwen3-8B | Retool dataset |
| RLVR (AIME) | DeepSeek-R1-Distill-Qwen-1.5B | Qwen3-4B | DAPO RL data |

- Learning rate: $1 \times 10^{-6}$, $C = 2$, $k = 4$
- 32 tasks per step, 8 rollouts each

### 4.4 Table 3 — Personal Agent Optimization Efficiency

Metric: minimum sessions required to reach optimization effect (lower = better).

| | **Hybrid RL** | GRPO | OPD | Mem0 | Cognee |
|-|:---:|:---:|:---:|:---:|:---:|
| **Joint Optimization** | | | | | |
| Student | **11.6** | 15.4 | 30.8 | 13.6 | 14.6 |
| TA | **8.2** | 12.0 | 34.0 | 15.8 | 15.4 |
| Teacher | **11.4** | 14.8 | 24.4 | 14.2 | 14.8 |
| Average | **10.3** | 14.1 | 29.7 | 14.5 | 14.9 |
| **Separate Optimization** | | | | | |
| Student | **19.2** | 22.8 | 34.6 | 13.4 | 15.6 |
| TA | **11.8** | 22.4 | 36.0 | 16.0 | 14.8 |
| Teacher | **14.0** | 18.0 | 17.6 | 15.8 | 15.0 |
| Average | **15.0** | 21.1 | 29.4 | 15.1 | 15.1 |

> Joint Hybrid RL significantly outperforms all baselines.
> Mem0 and Cognee are memory/skill-evolution baselines (separate optimization-only comparison).

### 4.5 General Agent Results (Figure 5)

Process rewards integrated with outcome rewards improve:
- Tool-call: 0.19 → 0.25 (over 250 steps)
- GUI: 0.31 → 0.33 (over 120 steps)

### 4.6 Hybrid RL vs. Single-Signal Methods

Hybrid RL substantially outperforms GRPO alone and OPD alone across all settings. Joint optimization yields larger gains for Hybrid RL compared to memory/skill-evolution baselines.

### 4.7 Hybrid RL Extension Results (Figure 6)

Hybrid RL extends to multi-turn agentic RL and RLVR settings, outperforming outcome-only and integrated-reward baselines in training curves.

### 4.8 Table 4 — Ablation: Hint Selection Strategy and Model Size

Policy: **Qwen3-32B**, Joint Optimization. Lower = better.

| Setting | Seq-Optimal | Token-Optimal | Random | GRPO | OPD |
|---------|:-----------:|:-------------:|:------:|:----:|:---:|
| Student | 14.0 | 13.8 | 18.6 | 17.2 | 34.4 |
| TA | 9.6 | 10.0 | 12.6 | 12.0 | 29.8 |
| Teacher | 13.8 | 13.4 | 17.0 | 18.2 | 25.6 |
| Average | **12.5** | **12.4** | 16.1 | 15.8 | 29.9 |

> Random hint selection causes training instability. Sequence-optimal ≈ Token-optimal in practice.

### 4.9 Log-Probability Difference Clipping is Vital (Figure 7)

Without clipping: truncation ratio reaches 0.5 in Retool.
With clipping ($C=1$): truncation ratio ≤ 0.2.

### 4.10 Table 5 — Ablation: k and Support Set

Joint Optimization, Qwen3-4B. Lower = better.

**Support set $S_i = S_i^q$ (student top-k only):**

| | k=2 | k=4 | k=8 | k=20 | Token-level OPD |
|-|:---:|:---:|:---:|:----:|:---------------:|
| Student | 30.4 | 11.6 | 12.8 | 11.4 | 34.4 |
| TA | 12.0 | 8.2 | 7.6 | 7.8 | 36.0 |
| Teacher | 17.2 | 11.4 | 10.0 | 10.2 | 22.6 |
| Average | 20.2 | **10.3** | 10.1 | 9.8 | 31.0 |

**Support set $S_i = S_i^q \cap S_{i,h^*}^p$ (intersection):**

| | k=2 | k=4 |
|-|:---:|:---:|
| Average | 31.6 | 11.8 |

> k ≥ 4 plateaus in performance. Token-level OPD performs significantly worse (31.0 vs 9.8).
> Student top-k support set ($S_i^q$) performs comparably to intersection set at k=4.

### 4.11 Model Size Ablation

Qwen3-32B as policy shows comparable performance to Qwen3-4B-Thinking-2507. Using Qwen3-8B as PRM teacher yields similar results to Qwen3-4B-Thinking-2507.

---

## Section 5: Conclusion

"Every agent interaction produces a next-state signal that encodes how the agent performed and, often, how it should have acted differently. OpenClaw-RL is built on a single insight: these signals are stream-agnostic, and one policy can learn from all of them simultaneously."

The framework combines "frequent evaluative signals with sparser but richer directive signals, with stable learning ensured through overlap-guided hint selection and log-probability-difference clipping."

**Limitations acknowledged:**
- Negative or adversarial user feedback (misleading corrections, malicious instructions) may poison the model if used directly for online updates.
- A model optimized for personal usage may encode user-specific preferences and private information, making it an attractive target for attacks.

---

## Figure Captions

**Figure 1:** OpenClaw-RL infrastructure overview. Interaction streams come from two agent types: Personal Agents (conversational, personalized), hosted on personal devices, and General Agents (terminal, GUI, SWE, and tool-call agents), hosted on cloud services.

**Figure 2:** Optimize your OpenClaw simply by using it. We provide simulation results here. (Shows before/after example conversations with Student, TA, Teacher personas.)

**Figure 3:** Method Overview. For personal agents, we support both binary-reward optimization and on-policy distillation training. For general agentic RL, shows RLVR with integrated step-wise rewards and standardization.

**Figure 4:** Overlap-Guided Hint Selection Method Overview. Visualizes the hint selection mechanism that picks candidate hints based on top-k vocabulary overlap between student and teacher distributions.

**Figure 5:** General Agent training curves — rollout accuracy over training steps for Terminal, GUI, SWE, and Tool-call tracks.

**Figure 6:** Hybrid RL Extension training curves — Retool accuracy and AIME score over training steps.

**Figure 7:** Ablation on log-probability clipping — stability comparison with and without clipping; three panels showing training stability and truncation ratio.

---

## Key Numbers Summary (Quick Reference)

| Item | Value |
|------|-------|
| Joint Hybrid RL average sessions | **10.3** |
| Separate Hybrid RL average sessions | **15.0** |
| Joint GRPO average | 14.1 |
| Joint OPD average | 29.7 |
| Best k | 4 (plateaus at k≥4) |
| Default clip C | 1 (Personal Agent), 2 (General Agent) |
| Policy model | Qwen3-4B-Thinking-2507 |
| Simulator | Qwen3-32B |
| Convergence criterion | 3 consecutive sessions passing rule |
| Batch trigger | 16 samples |
| Learning rate | 1e-5 (Personal), 1e-6 (General) |
