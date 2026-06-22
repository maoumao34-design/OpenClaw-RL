# 复现工作记录

汇报与整体复盘用。技术细节通过各条目链接查阅。

---

## 2026-06-17 论文理解 + 环境搭建启动

**目标：** 读懂论文，搭建基础 conda 环境

**完成内容：**
- 通读论文，整理核心机制（四组件异步架构、两类信号 GRPO+OPD、四类环境）→ [`paper_understanding.md`](paper_understanding.md)
- 创建 conda 环境 `/dfs/data/envs/openclaw-rl`（Python 3.12）
- 安装 torch 2.9.1+cu129、sglang、slime 等基础依赖

**主要问题：**
- `outlines_core` PyPI sdist 损坏无法直接安装 → 从 GitHub Releases 手动下载 wheel 本地安装，已解决
- DeepEP 需要 sm_90 架构编译，CPU workspace 上无法进行 → 确认 Qwen3-4B 是密集模型不依赖 DeepEP，跳过
- pip 调用误走系统 Python 3.13 而非 conda 环境 → 改用完整路径 `/dfs/data/envs/openclaw-rl/bin/pip`

---

## 2026-06-18 环境搭建（续）

**目标：** 完成 CPU workspace 上所有 Python 依赖安装

**完成内容：**
- 安装 slime、Megatron-LM、mbridge、megatron-bridge、megatron-core 等所有核心依赖
- 确认 Qwen3-4B 密集模型不需要 DeepEP，从依赖列表移除

**主要问题：**
- Megatron-LM / slime 等 `git+https://github.com/...` 依赖在 modelfactory 无法直连 GitHub → 在本地用 ghfast 镜像克隆后上传，或在服务器上配置 ghfast 代理后安装
- `PIP_CONSTRAINT` 环境变量与新依赖冲突 → 安装时临时 `export PIP_CONSTRAINT=` 清空绕过
- megatron-bridge 含 git submodule，拉取时子模块 URL 指向 GitHub 直连失败 → 手动修改 `.gitmodules` 改用 ghfast 镜像后 `git submodule update`

---

## 2026-06-22 GPU 编译 + 模型准备

**目标：** 在 GPU workspace 完成需要 CUDA 编译的依赖，下载并转换 Policy 模型

**完成内容：**
- GPU 编译依赖全部完成（均需 sm_90 / H20 GPU）：
  - flashinfer（SGLang 推理加速）
  - int4_qat（量化支持）
  - apex（Megatron 混合精度）
  - flash-attn 2.7.4.post1
  - TransformerEngine 2.10.0
- 下载 Qwen3-4B-Thinking-2507（7.6 GB），加载验证通过
- HF → torch_dist 格式转换，保存至 `/dfs/data/models/torch_dist/qwen3-4b-thinking-2507`（Megatron 训练所需格式）
- 本地开始下载 Qwen3.5-122B-A10B-GPTQ-Int4（Simulator 候选，~65 GB）

→ 完整环境安装步骤见 [`implementation_path.md → 环境准备`](implementation_path.md)

**⚠️ 此日同时发现重大方向错误，见下条**

---

## 2026-06-22 方向更正

**背景：** 完整阅读论文 PDF + git log 时间线核查后，发现此前复现方向存在两处根本性错误，已全部更正。

---

**错误 1：一直在配置错误的训练脚本（Binary RL 基线，而非论文主方法）**

- 原计划：`openclaw-rl/run_qwen3_4b_openclaw_rl.sh`
  - 这是 Table 3 "GRPO" 基线列（Binary RL，仅 GRPO，无 OPD）
- 正确：`openclaw-combine/run_qwen3_4b_openclaw_combine.sh`
  - 这是 Table 3 "Hybrid RL (Ours)" 主方法（GRPO + OPD 混合损失）
- 根本原因：`openclaw-rl/` 目录名容易被误认为是主方法，未在项目开始时从论文实验设计出发验证文件
- → 详见 [`WARNINGS.md → 方法与脚本对应`](WARNINGS.md)，完整步骤见 [`implementation_path.md`](implementation_path.md)

---

**错误 2：一直计划用 OEL 模块的评估脚本（与论文 Table 3 完全无关）**

- 原计划：`openclaw-rl/oel/eval/gsm8k_personal_agent.py`
  - 属于 OEL 模块，由外部贡献者 PR #96 于 2026-04-20 加入，晚于论文提交（2026-03-11）
  - 用 LLM 0-1 打分，与 Table 3 指标根本不同
- 正确：`openclaw-test/student_chat.py` + `TA_chat.py` + `teacher_chat.py`
  - Table 3 指标为 rule-based session 计数，无需 LLM
- 根本原因：未用 git log 核查文件加入时间，直接使用"看起来相关"的脚本
- → 详见 [`WARNINGS.md → 禁止使用的目录`](WARNINGS.md)，Table 3 指标定义见 [`paper_understanding.md`](paper_understanding.md)

---

**新增理解：三端口架构与 OpenClaw 的必要角色**

- 完整三端口架构：Port 30001（Simulator, Qwen3-32B SGLang）→ Port 18789（OpenClaw gateway，workspace 文件工具）→ Port 30000（RL training proxy）
- OpenClaw 是论文四组件架构中的 Environment Server，提供 homework 文件读写工具并注入训练 header（X-Session-Id / X-Turn-Type），不可替代
- → 详见 [`implementation_path.md → 系统架构`](implementation_path.md)

---

**关键决策（已确认）：**
- 评估指标：rule-based session 计数（Student/TA/Teacher 三套规则，连续 3 次满足即收敛）→ [`paper_understanding.md`](paper_understanding.md)
- Simulator：论文为 Qwen3-32B（Section 4.1），**待确认**是否直接部署或用 Qwen3.5-122B 替代

**此日工作产出：**
- [`implementation_path.md`](implementation_path.md)：完整端到端实现路径（7 个实验块，每块含所需文件和步骤）
- [`WARNINGS.md`](WARNINGS.md)：禁止使用的目录和文件清单（含时间线证据）

---

## 当前状态（2026-06-22）

### 已就绪
- [x] 环境 + 所有 GPU 编译依赖
- [x] Qwen3-4B-Thinking-2507（HF + torch_dist 格式）
- [x] GSM8K.json 数据集（已在仓库）
- [x] 完整实现路径文档（[`implementation_path.md`](implementation_path.md)）
- [x] 警示文档（[`WARNINGS.md`](WARNINGS.md)）

### 进行中
- [ ] Qwen3.5-122B-A10B-GPTQ-Int4 本地下载（~65 GB，Simulator 候选）

### 下一步（优先级顺序）
1. 确认 OpenClaw 能否安装到 modelfactory（是 Environment Server，不可跳过）
2. 确认 Simulator 模型：Qwen3-32B（完全忠实论文）vs Qwen3.5-122B（正在下载中）→ [`paper_understanding.md → Simulator 部署方案`](paper_understanding.md)
3. 申请 8×H20 训练 workspace（Actor×4 + Rollout×2 + PRM×1 + PRM Teacher×1）
