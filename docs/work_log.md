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

## 2026-06-22 OpenClaw 安装（进行中）

**目标：** 在 modelfactory 上安装 OpenClaw，打通 port 18789 gateway

**完成内容：**
- 所有文档修缮完毕，推送 GitHub，modelfactory `git pull` 同步完成
- 本地 clone OpenClaw 仓库（`github.com/openclaw/openclaw`）并上传至 `/dfs/data/openclaw-rl-project/openclaw/`
- 升级 Node.js v18.19.1 → v22.23.0（OpenClaw 要求 Node 22.19+）
- 安装 corepack，pnpm 配置放置后台运行中

**主要问题：**
- modelfactory 网速限制，OpenClaw 仓库无法直接 clone → 本地下载压缩后上传解决
- Node.js 版本不足（v18 < 要求的 v22.19+），nvm/Docker 均不可用 → NodeSource apt 安装 v22.23.0 解决

**当前状态：** pnpm 11.2.2 安装完成（2026-06-23 确认）

---

## 2026-06-23 OpenClaw 安装完成 + 训练脚本编写

**目标：** 完成 OpenClaw 配置，编写端到端训练启动脚本

**完成内容：**

**OpenClaw 安装（在 modelfactory Workspace_cuda129_CPU 完成）：**
- `rl-training-headers` 插件：从官方仓库 TypeScript 源码手动编译为 JS（`index.js` + `package.json` + `openclaw.plugin.json`），复制至 `/usr/lib/node_modules/openclaw/dist/extensions/rl-training-headers/`，`openclaw plugins enable rl-training-headers` 成功（输出 "fetch patched"）
  - 插件文件：[`scripts/rl-training-headers/`](../scripts/rl-training-headers/)
- OpenAI provider 配置：通过 `openclaw config --section model` wizard 配置，`api-key=EMPTY`，model=`openai/qwen3-4b`
- LLM provider 指向：`OPENAI_BASE_URL=http://localhost:30000/v1`（port 30000 = RL training proxy，使用 OpenAI 兼容格式，与 OpenAI 服务无关）

**三端口架构完整确认（通过读 openclaw-test/README.md 和 openclaw_opd_api_server.py）：**
- Port 30001：Simulator (Qwen3-32B)，由 `launch_user_llm.sh` 启动；供 student/TA/teacher_chat.py 使用
- Port 18789：OpenClaw gateway，调用 port 30000 作为 LLM provider
- Port 30000：RL training proxy，由训练进程自动启动；记录 X-Session-Id/X-Turn-Type，转发至内部 SGLang

**训练启动脚本：**
- [`scripts/train_with_services.sh`](../scripts/train_with_services.sh)：编排所有四个服务
  - Step 1：先启动训练（绕过 run_qwen3_4b_openclaw_combine.sh 开头的 pkill）
  - Step 2：Ray head 就绪后启动 Simulator（GPU 8）
  - Step 3：启动 OpenClaw gateway（OPENAI_BASE_URL → port 30000）
  - Step 4：运行模拟循环（student → TA → teacher，顺序依赖，循环供 rollout 数据）

**主要问题：**
- `openclaw plugins install <path>` 失败（期待预编译 dist/index.js）→ 手动编译 TypeScript，直接复制到系统目录解决
- pnpm 网络超时（3分57秒无法下载包）→ 完全绕过 pnpm，使用系统安装的 `openclaw` CLI 解决
- `openclaw config set openai.config.baseUrl` 配置 schema 不支持 → 改用环境变量 `OPENAI_BASE_URL` 方式传入

**待在 modelfactory 验证：**
- `openclaw start` 是否是启动 gateway 的正确命令（可能需要调整）
- `OPENCLAW_GATEWAY_TOKEN` 如何从 `~/.openclaw/openclaw.json` 读取

---

## 2026-06-23 论文结构理解 + 复现路线规划 + 文档建立

**目标：** 搞清楚 Table 3 的 Joint/Separate 含义、完整复现路线、论文各图表定位

**完成内容：**

**Joint vs Separate 含义确认（通过读 paper_reproduction_scope.md 中的数字）：**
- **Joint optimization**：一个训练 job，Student/TA/Teacher 三个 persona 混合训练同一个模型，对应 Table 3 上半块
- **Separate optimization**：三个独立训练 job，每个只训练一种 persona，对应 Table 3 下半块
- 当前 `train_with_services.sh` 实现的是 Joint（模拟循环每轮跑全部三个 persona）
- 要得到 Table 3 完整两块数字需要 Joint（3 个方法）+ Separate（9 个方法×persona 组合）共 12 个训练 job

**Table 3 完整复现路线制定（5 Phase）：**
- Phase 1：Joint Hybrid RL（当前，已有脚本）→ 验证主结论
- Phase 2：Joint GRPO + OPD 基线（需写 `train_grpo_joint.sh`、`train_opd_joint.sh`）
- Phase 3：Separate Hybrid RL（需写 3 个单 persona 脚本）
- Phase 4：Separate 基线（低优先级）
- Phase 5：Mem0/Cognee（独立生态，最低优先级）
- → 详见 [`paper_reproduction_scope.md → Table 3 完整复现执行路线`](paper_reproduction_scope.md)

**论文图表结构全面梳理（通过 arXiv HTML 抓取）：**
- Figure 1-4、Table 1-2：全部是方法说明图/配置表，**不是实验数据，不需要复现**
- Figure 5（step 横轴）：General Agent 四个 track 训练曲线，需要独立训练环境
- Figure 6（step 横轴）：ReTool + RLVR 扩展实验训练曲线
- Figure 7：OPD 消融曲线（log-prob clipping 有无对比）
- Table 3-5：Personal Agent 主结果 + OPD 消融，是复现核心
- Figure 2 虽然有"对话示例"但是挑选的示意图，不是定量结果

**数据纠错：**
- `paper_reproduction_scope.md` 中 Teacher Joint Hybrid RL 误记为 14.8 → 实际为 **11.4**
  - 验证：(11.6 + 8.2 + 11.4) / 3 = 10.4 ≈ 10.3 ✓
- Separate 完整数字补全（之前有 "..." 占位）

**新建文档：**
- [`docs/paper_full_text.md`](paper_full_text.md)：论文完整内容 markdown 版（从 arXiv HTML 提取）
  - 包含 Abstract、全部 Section 正文、Table 3/4/5 完整数据、所有公式（LaTeX）、Figure captions、快速查阅表

---

## 当前状态（2026-06-23）

### 已就绪
- [x] 环境 + 所有 GPU 编译依赖
- [x] Qwen3-4B-Thinking-2507（HF + torch_dist 格式）
- [x] GSM8K.json 数据集（已在仓库）
- [x] 完整实现路径文档（[`implementation_path.md`](implementation_path.md)）
- [x] 警示文档（[`WARNINGS.md`](WARNINGS.md)）
- [x] OpenClaw 安装完成（Node 22.23、rl-training-headers 插件、openai provider）
- [x] `train_with_services.sh` 编写完成（Joint Hybrid RL，modelfactory job 可直接提交）
- [x] Table 3 完整复现路线文档（Phase 1-5）
- [x] 论文完整 markdown 版（[`docs/paper_full_text.md`](paper_full_text.md)）

### 进行中
- [ ] Qwen3-32B 本地下载 → 上传至 `/dfs/data/models/Qwen/Qwen3-32B/`

### 下一步（优先级顺序）
1. **Qwen3-32B 上传完成** → 确认路径 `/dfs/data/models/Qwen/Qwen3-32B/`
2. **modelfactory 验证两点**：`openclaw start` 命令 + `OPENCLAW_GATEWAY_TOKEN` 读取方式
3. **提交 Joint Hybrid RL 训练 job**（Workspace_cuda129_CPU + 9×H20）
4. 训练完成后跑 `evaluate_table3.py` → 得到 Phase 1 三行数字
