# 复现工作记录

汇报与整体复盘用。技术细节通过各条目链接查阅。

## 工作记录规范

**与 `issues_log.md` 的分工**

| 文档 | 写什么 |
|------|--------|
| **本文件** (`work_log.md`) | 按日的目标、完成摘要、阶段状态、下一步 |
| **`issues_log.md`** | 单次失败/报错的现象、根因、修复（含日志原文） |

**每日条目格式**

```markdown
## YYYY-MM-DD

**目标：** …

**完成内容：**
- 摘要 bullet → 细节见 [`某文档.md`](path)

**主要问题：**（可选）
- 问题 → 处理方式；未闭环写「待 modelfactory 验证」
```

**文末维护「当前状态」**：`已就绪` checklist + `下一步` 列表；重大方向变更（如脚本换用、架构调整）单独成节并链到 `WARNINGS.md` / `issues_log.md`。

**其它规则**：各专题 doc 文首链回 `[← 工作记录](work_log.md)`；不在此重复贴长日志。

---

## 2026-06-17

**目标：** 读懂论文，搭建基础 conda 环境

**完成内容：**
- 通读论文，整理核心机制（四组件异步架构、两类信号、四类环境）→ [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)
- 创建 conda 环境 `/dfs/data/envs/openclaw-rl`（Python 3.12），安装 torch 2.9.1+cu129、sglang、slime 等基础依赖

**主要问题：**
- `outlines_core` PyPI sdist 损坏 → 从 GitHub Releases 手动下载 wheel 安装
- DeepEP 需要 sm_90 编译，CPU workspace 无法进行 → 确认 Qwen3-4B 密集模型不依赖 DeepEP，跳过
- pip 误走系统 Python 3.13 → 改用完整路径 `/dfs/data/envs/openclaw-rl/bin/pip`

---

## 2026-06-18

**目标：** 完成 CPU workspace 上所有 Python 依赖安装

**完成内容：**
- 安装 slime、Megatron-LM、mbridge、megatron-bridge、megatron-core 等所有核心依赖，确认 DeepEP 不需要

**主要问题：**
- `git+https://github.com/...` 依赖在 modelfactory 无法直连 GitHub → 本地用 ghfast 镜像克隆后上传
- `PIP_CONSTRAINT` 环境变量与新依赖冲突 → 安装时临时清空绕过
- megatron-bridge git submodule 拉取失败 → 手动修改 `.gitmodules` 改用 ghfast 镜像

---

## 2026-06-22

**目标：** GPU 编译依赖、模型准备、纠正复现方向、启动 OpenClaw 安装

### GPU 编译 + 模型准备

**完成内容：**
- GPU 编译依赖全部完成（flashinfer、int4_qat、apex、flash-attn 2.7.4.post1、TransformerEngine 2.10.0）
- 下载 Qwen3-4B-Thinking-2507（7.6 GB），HF → torch_dist 格式转换，保存至 `/dfs/data/models/torch_dist/qwen3-4b-thinking-2507`
- 本地开始下载 Qwen3.5-122B-A10B-GPTQ-Int4（Simulator 候选，~65 GB）

→ 完整安装步骤见 [`implementation_path.md`](openclaw-rl/docs/implementation_path.md)

### 方向更正（重大）

完整阅读论文 PDF + git log 时间线核查后，发现两处根本性错误，当日全部纠正：

- **错误 1：** 一直配置的是 `openclaw-rl/run_*.sh`（Table 3 GRPO 基线），正确脚本是 `openclaw-combine/run_qwen3_4b_openclaw_combine.sh`（论文主方法 Hybrid RL）
- **错误 2：** 计划用 `openclaw-rl/oel/eval/` 评估脚本（2026-04-20 才合入，与论文无关），正确是 `openclaw-test/student_chat.py` + `TA_chat.py` + `teacher_chat.py`

→ 详见 [`WARNINGS.md`](openclaw-rl/docs/WARNINGS.md)

**产出：**
- [`implementation_path.md`](openclaw-rl/docs/implementation_path.md)：重建完整端到端实现路径
- [`WARNINGS.md`](openclaw-rl/docs/WARNINGS.md)：禁止使用的目录和文件清单

### OpenClaw 安装（启动）

**完成内容：**
- 本地 clone OpenClaw 仓库，上传至 `/dfs/data/openclaw-rl-project/openclaw/`
- 升级 Node.js v18 → v22.23.0（OpenClaw 要求 Node 22.19+），安装 corepack，pnpm 后台运行中

**主要问题：**
- modelfactory 无法直接 clone → 本地下载压缩后上传
- Node.js 版本不足，nvm/Docker 均不可用 → NodeSource apt 安装解决

---

## 2026-06-23

**目标：** 完成 OpenClaw 安装，整理论文和复现路线，完善评估机制

### OpenClaw 安装完成 + 训练脚本编写

**完成内容：**
- `rl-training-headers` 插件：手动编译 TypeScript 源码为 JS，复制到系统目录，`openclaw plugins enable` 成功
- OpenAI provider 配置完成，LLM 指向 `http://localhost:30000/v1`（RL training proxy）
- 三端口架构完整确认：Port 30001（Simulator）→ Port 18789（OpenClaw gateway）→ Port 30000（RL proxy）→ [`codebase_analysis.md`](openclaw-rl/docs/codebase_analysis.md)
- 新建 `openclaw-rl/scripts/train_with_services.sh`：编排训练 + Simulator + OpenClaw gateway + 模拟循环四个服务

**主要问题：**
- `openclaw plugins install` 失败（期待预编译包）→ 手动编译后直接复制系统目录
- pnpm 网络超时 → 完全绕过 pnpm，使用系统已安装的 `openclaw` CLI

**待 modelfactory 验证：** `openclaw start` 命令 + `OPENCLAW_GATEWAY_TOKEN` 读取方式

### 论文整理 + 复现路线规划

**完成内容：**
- 确认 Joint（三 persona 同一训练 job）vs Separate（每 persona 独立训练 job）含义，对应 Table 3 上下两块
- Table 3 完整复现路线划分为 5 Phase → [`paper_reproduction_scope.md`](openclaw-rl/docs/paper_reproduction_scope.md)
- 论文各图表定位梳理（Figure 1-4、Table 1-2 为方法说明，不需复现；Table 3-5、Figure 5-7 为实验数据）
- 修正 Teacher Joint Hybrid RL 数值错误（14.8 → **11.4**）
- 新建 [`paper_index.md`](openclaw-rl/docs/paper_index.md)：论文页码索引 + PDF 直接提取的数据

### 收敛检测机制确认 + 脚本完善

**完成内容：**
- 读官方 `openclaw-test/` 源码，确认三个 chat 脚本已内置 `--output` 机制，收敛检测为事后分析 → [`codebase_analysis.md`](openclaw-rl/docs/codebase_analysis.md)
- 新建 `scripts/check_convergence.py`：解析 output master 文件，输出 Table 3 三行 session 数字
- 更新 `train_with_services.sh`：加 `--output` + 跨轮累积 + `SESSION_LIMIT=72` + 结束后自动调用收敛检测

### Job 提交准备 + 提交

**完成内容：**
- Qwen3-32B 上传完整确认：`/dfs/data/models/Qwen/Qwen3-32B/`（17 个 safetensors shard）
- 修复 `train_with_services.sh` 三处 bug：
  1. Token 读取路径：`gateway.token` → `gateway.auth.token`（实际 JSON 结构多一层 `auth`）
  2. conda 激活：硬编码 `/dfs/data/miniconda3` 路径，环境设为 `/dfs/data/envs/openclaw-rl`
  3. `SIMULATOR_GPU`：从 8 改为 7（平台 GPU 上限 8 张，GPU 7 与 PRM teacher 共用，Qwen3-4B 约 8GB + Qwen3-32B 约 64GB = 72GB < H20 96GB）
- 提交训练 job：`app-job-1159-1782206197366`，8×H20，64 CPU 核，128GB 内存，排队中

**关键确认：**
- job 使用保存的 workspace 镜像，openclaw CLI（`/usr/bin/openclaw`）在 job 环境中可用，无需迁移到 `/dfs/data/`

---

## 2026-06-26

**目标：** Step B 3 GPU smoke 端到端跑通；部署外部 Simulator；修复 modelfactory 上 smoke 脚本连环问题

**完成内容：**

### 外部 Simulator（Qwen3-32B vLLM）
- modelfactory 独立服务部署成功；`scripts/simulator.env` 已填写，`curl /health` → HTTP 200
- 训练 job 通过 `SIMULATOR_BASE_URL` 调用，不占训练 GPU

### OpenClaw 配置确认
- `~/.openclaw/openclaw.json` 已核对：`primary=sglang/qwen3-4b`，`baseUrl=127.0.0.1:30000`，`controlUi.enabled=false`，`rl-training-headers` 已启用
- workspace 手动测 `openclaw gateway run`（headless 参数）约 **1s** 即 `ready`（18789）

### Smoke 脚本迭代（已 push GitHub `main`）
| Commit | 内容 |
|--------|------|
| `7f657e1` | patched combine 在 `logs/` 下时 `REPO_ROOT` 解析错误 → 固定为 `OpenClaw-RL-official` |
| `96c40e5` | 先等 RL proxy `:30000` 再起 OpenClaw；headless gateway 参数；`/healthz` 检测；900s 超时 |
| `2687e58` | 新增 `run_openclaw_combine_modelfactory.sh`：Ray job 用 `SLIME_ROOT/train_async.py` + `--working-dir` |
| `01f3eb0` | smoke 3 GPU：强制 `PRM_NUM_GPUS_PER_ENGINE=1`（inference 默认 TP=2 与 3 卡布局冲突） |

### 文档 / 讨论
- GPU 布局：论文 4+2+1+1 是 Megatron 训练并行策略，非 4B 权重下限；H20 上可先 7 GPU 或 smoke 再 8 GPU
- 删除临时分支 `fix/smoke-repo-root`（fix 已合入 `main`）

**主要问题：**（细节见 [`issues_log.md`](issues_log.md) 2026-06-26 smoke 条目）
- smoke job **尚未通过**；最后一次失败为 Ray job 失败（`PRM_NUM_GPUS_PER_ENGINE=2`），`01f3eb0` 已修，**待下周重新提交 job 验证**
- 早期失败：OpenClaw 18789 超时（旧脚本启动顺序/日志缓冲）；`/workspace/train_async.py` 找不到

**GitHub：** `main` 已 push 至 `01f3eb0`；`.cursor/` 规则 commit 亦在 `main`（用户确认可上传）

---

## 2026-06-29

**目标：** 回归复现进度，3 GPU smoke 跑通

**完成内容：**

- 更换 Simulator 服务地址（新 Qwen3-32B 服务），更新 `scripts/simulator.env`；`curl /health` 验证 HTTP 200
- 确认 workspace 上直接运行 `bash smoke_train_with_services.sh` 正常：script started → log 目录创建 → conda 激活 → 训练进程启动

**主要问题：**

- **Modelfactory job 提交系统异常**（系统维护期间）：bash/python job 均无输出、无日志、脚本未执行；直接在 workspace terminal 运行完全正常 → 确认是平台层面问题，非代码问题
- 排查过程排除了：CRLF 换行、路径解析、simulator.env 空格、openclaw token、脚本内容等所有代码侧原因
- **根因**：modelfactory 系统维护，维护期间 job 提交静默失败

**处理方式：** 维护结束后改用 **workspace 直接运行**（申请 GPU 后 `bash scripts/smoke_train_with_services.sh`）；后续大训练 job 恢复后再试提交

---

## 当前状态（2026-06-29）

### 已就绪
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] 外部 Qwen3-32B Simulator（vLLM，`simulator.env` 已配，新地址验证 HTTP 200）
- [x] OpenClaw + `openclaw.json` + rl-training-headers
- [x] `scripts/smoke_train_with_services.sh`（workspace terminal 直接运行验证通过）
- [x] `scripts/train_with_services.sh`（8 GPU 正式训练脚本）
- [x] `scripts/check_convergence.py`

### 未验证 / 阻塞
- [ ] **3 GPU smoke `✅ SMOKE PASSED`**（等待 modelfactory 维护结束，在 GPU workspace 直接运行）
- [ ] 8 GPU 正式 Table 3 训练（等 smoke 通过后）

### 下一步
1. 等系统维护结束，在当前 workspace 申请 3 GPU
2. `bash scripts/smoke_train_with_services.sh 2>&1 | tee /dfs/data/openclaw-rl-project/logs/smoke_workspace_$(date +%Y%m%d_%H%M%S).txt`
3. smoke 通过后，用 GPU workspace 或恢复正常的 job 提交跑 8 GPU 正式训练

---

## 历史状态（2026-06-23，已被 6/26 架构更新取代）

### 已就绪（6/23 时点）
- [x] 环境 + 所有 GPU 编译依赖
- [x] Qwen3-4B-Thinking-2507（`/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507`）
- [x] Qwen3-32B（`/dfs/data/models/Qwen/Qwen3-32B`，17 shards）
- [x] OpenClaw 安装完成（Node 22.23、rl-training-headers 插件）
- [x] `check_convergence.py`
- [x] Table 3 完整复现路线 → [`paper_reproduction_scope.md`](paper_reproduction_scope.md)

> 6/24 8 GPU job 因 Simulator 与 GPU 7 共用失败；6/26 已改为**外部 Simulator + 8 GPU 全用于训练**。
