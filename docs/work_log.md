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
- 修复 `train_with_services.sh` 三处 bug：token 读取路径（`gateway.token` → `gateway.auth.token`）、conda 激活路径、`SIMULATOR_GPU` 从 8 改为 7
- 提交训练 job：`app-job-1159-1782206197366`，8×H20，64 CPU 核，128GB 内存，排队中
- 确认 openclaw CLI（`/usr/bin/openclaw`）在 job 环境中可用，无需迁移到 `/dfs/data/`

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

### Joint 训练实现修正（重要）

**完成内容：**
- 对照论文 Appendix A.1 + 官方源码核查，发现原实现每轮清空 homework 目录，违反 Joint 设计（INIT 一次性建立固定 `homework1/` `homework2/`，三角色并行复用）→ 重写 `train_with_services.sh` / `smoke_train_with_services.sh` 模拟部分

### 训练脚本更正：basic combine → topk-select（关键）

**完成内容：**
- 核查 Appendix A.1（k=4, m=3）+ Table 5 消融（k=4 → avg 10.3 = Table 3 主结果），确认正确脚本为 `openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh`；原用 basic combine（m=1，无 k）对应错误

**产出：**
- `scripts/run_openclaw_topk_select_modelfactory.sh`：官方 topk-select 的 modelfactory patch
- `scripts/smoke_run_qwen3_4b_openclaw_topk_select.sh`：4 GPU smoke launcher（m=1 验证流通）
- 更新 `scripts/train_with_services.sh` / `smoke_train_with_services.sh`（smoke GPU 3→4，含 PRM Teacher）

### 论文深度理解 + 源码核查

**完成内容：**
- 核查 PRM Teacher 冻结保证、三角色区分机制（session ID / homework 目录链）、三角色适应类型及收敛速度差异 → [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)
- 确认 PRM SGLang 双职能（同一次 LLM 调用产出 Judge 分 + hint 候选），更新 GPU 布局表格 → [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)

**主要问题：**
- modelfactory 系统维护，job 提交静默失败（确认为平台问题）→ 改用 workspace 直接运行

---

## 2026-06-30

**目标：** 论文深度理解收尾，核查 Actor / Rollout 完整职责

### GPU 布局表格精简 + Actor / Rollout 职责核查

**完成内容：**
- 核查 Actor / Rollout 完整职责（5 步 rollout 流程、topk-select 下 log-probs 由 Actor 重算），精简并更新 GPU 布局表格 → [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)

### OPD 信号判别机制 + 三方法对比

**完成内容：**
- 核查 OPD 逐 turn 独立判断机制、三方法兜底逻辑（GRPO 保留最低 1 sample、OPD 直接跳过、Hybrid 三路 dispatch），记录对比表格 → [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)

### GRPO 组定义与 Advantage 函数实现（源码核查）

**完成内容：**
- 核查 Personal Agent GRPO 实现（n-samples=1，无组内比较，advantage = raw PRM 分数），发现与标准 GRPO 有根本差异；修正 `paper_understanding.md` 误写 → [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)

### OpenClaw 调用架构梳理（32B Simulator ↔ 4B Policy 交互机制）

**完成内容：**
- 完整还原 Simulator↔Policy 交互链路（session 历史维护、OPD 延迟一拍打分、turn_type routing），新增"OpenClaw 调用架构"小节 → [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)

### 论文深度理解补充（下午）

**完成内容：**
- 确认 Mem0 / Cognee 是"记忆 + 上下文注入"范式（非训练），与 RL 方法形成对比；补充至 [`paper_reproduction_scope.md`](openclaw-rl/docs/paper_reproduction_scope.md) Phase 5
- 更新 `paper_understanding.md` 十四、十五节：修复编号重复 bug，补充复现难点 4 条（仓库边界、Hybrid 信号融合、GRPO 差异、General Agent 规模）

### 4 GPU smoke 调试（下午）

**主要问题：**（详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-06-26 smoke 条目）
- 问题 1：Simulator 旧 IP 残留，`simulator.env.example` 未更新 → 改为 `10.254.107.247`
- 问题 2：`nc` 未安装，port 检测永远失败 → 改用 `curl` 检测；commit `6543125`
- 问题 3：`OPENCLAW_GATEWAY_URL` 错改为 30000，绕过 OpenClaw → commit `482fdc6`（注：此修复方向错误，根因是启动顺序，见 [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-03）

---

## 2026-07-01

**目标：** 完成 4 GPU smoke 测试，打通端到端流程

### 积压文件同步（`98273cb`）

**完成内容：**
- 补提 15 个积压文件（脚本 + 文档，+1127 行），覆盖至前一日所有工作

**主要问题：**
- 多次 push 只 stage 当次操作文件，新建脚本/文档改动未入库 → push 前先 `git status` 检查全部改动（已记入长期记忆）

### 4 GPU Smoke 调试（续）

**主要问题：**（详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-01 smoke 条目）
- 问题 4：`REF_LOAD` HF 路径，Megatron 要求 bridge mode 不存在 → 改用 torch_dist；commit `672d9a7`
- 问题 5：64 GB RAM 节点 OOM（smoke TP=1 完整加载 ~24 GB×2）→ 申请 ≥128 GB 节点
- 问题 6：评估 401 Unauthorized，OPENCLAW_GATEWAY_TOKEN ≠ SGLANG_API_KEY → commit `5aa3c74`

### ✅ SMOKE PASSED

128 GB RAM 节点，commit `5aa3c74`，smoke 完整跑通：
- 训练阶段：Ray job 正常启动，Actor/Rollout/PRM/Teacher 四组件均初始化
- OpenClaw gateway 和 RL proxy（port 30000）正常就绪
- INIT 阶段：Student → TA → Teacher 顺序建立 `homework/` `homework1/` `homework2/`
- Joint 阶段：三角色并行，无文件冲突，Teacher 第 3 轮完成（max-turns=4，提前收敛）
- 输出：`✅ SMOKE PASSED`

**当前验证通过的 smoke 配置：**

| 参数 | 值 |
|------|-----|
| GPU 数 | 4（Actor×1 TP=1 / Rollout×1 / PRM SGLang×1 / Teacher×1）|
| Worker 节点 RAM | ≥128 GB |
| POLICY_MODEL_PATH | `/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507` |
| POLICY_TORCH_DIST | `/dfs/data/models/Qwen3-4B-Thinking-2507-torch-dist` |
| REF_LOAD / PRM_TEACHER_LOAD | `POLICY_TORCH_DIST` |
| Simulator | `http://10.254.107.247:8443`（Qwen3-32B vLLM）|
| smoke m | 1（正式 m=3）|
| smoke max-tokens-per-gpu | 8192（正式 32768）|

### 8GPU 正式训练前置验证脚本编写（Pre-test）

**完成内容：**
- 新建 5 GPU（2+1+1+1）前置验证脚本，在完整论文配置下跑 300 rollout（~18 步）验证整条流水线

**产出：**
- `scripts/minitest_train_with_services.sh`：5 GPU 完整流水线入口（含 Simulator + OpenClaw + 模拟 + 收敛检测）
- `scripts/minitest_run_qwen3_4b_openclaw_topk_select.sh`：5 GPU 训练 launcher（TP=2，300 rollout）
- 更新 `scripts/run_openclaw_topk_select_modelfactory.sh`：新增 `MINITEST_PROFILE=1` 分支

**与 8GPU 正式版的唯一差异：**

| 参数 | 正式（8 GPU）| Pre-test（5 GPU）|
|------|------------|----------------|
| `tensor-model-parallel-size` | 4 | 2 |
| `rollout-num-gpus-per-engine` | 2 | 1 |
| `num-rollout` | 100000000 | 300（~18 步）|
| context / batch / m / k | 32768 / 16 / 3 / 4 | **同正式**（不变）|

---

## 2026-07-02

**目标：** 提交 5 GPU pre-test，验证 8 GPU 正式训练流水线

**完成内容：**

### Pre-test 脚本完善 + 提交

- 补充 `--save-interval 100→5`（MINITEST_PROFILE sed 分支），支持可抢占式 job（被抢占后从 checkpoint 续跑，每 5 步存一次） → commit `eb518c1`
- workspace `git pull` 同步至 `eb518c1`，提交可抢占 5 GPU job

### Pre-test 运行进展（截至 16:48）

- INIT 阶段：15:42 完成（72 题 × Student→TA→Teacher 顺序建立 homework1/ homework2/）
- Joint 阶段：进行中，Round 6/12，每轮约 14 分钟，预计 18:20-18:50 完成
- 训练侧：300 rollout 尚未到上限（模拟是当前瓶颈）

### 论文理解补充

- 确认 `num-rollout 100000000` = "不限制"写法，实际靠手动 kill 或收敛判断停止
- 确认 1 rollout = 1 session（完整的一次多轮对话）
- 确认 Personal Agent max turns：论文 Appendix A.1 未写固定数字，以 context length 32768 token 作隐式上限；脚本 default `--max-turns 8` 与此一致（GSM8K 场景下 8 turn 不会撞 context limit）

---

## 2026-07-03

**目标：** 审查 5 GPU pre-test 结果，确认无论文偏离，决定是否提交 8 GPU 正式训练

### Pre-test 结果审查

**完成内容：**
- 对照论文逐项核查九项（输出写法、模型、k/m/hint_selection、权重/clip、循环、收敛、累积、训练参数）→ 全部 ✅

### 问题修复

**主要问题：**
- **Pre-test 0 训练步骤 + 无 checkpoint**：commit `482fdc6` 将 `OPENCLAW_GATEWAY_URL` 改为 30000 绕过 OpenClaw gateway，rl-training-headers 未注入 `X-Turn-Type:main` → 训练队列永远 0 → save-interval 不触发；架构组件全部核查确认 → 详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md)；三脚本改回 18789，commit `83810e4`
- **Simulator context overflow**：`launch_simulator.sh` 默认 16384，Policy 多轮后超限 → 核查官方 `launch_user_llm.sh` 确认 32768 → 已修 `scripts/launch_simulator.sh`；Simulator 需重启生效
- **`train_with_services.sh` 启动顺序**（待修复）：OpenClaw 在 30000 就绪前启动，sglang provider 连不上 → 复现原始 404；待 smoke 验证通过后修复

---

## 当前状态（2026-07-03）

### 已就绪
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist（`/dfs/data/models/Qwen3-4B-Thinking-2507-torch-dist`）
- [x] OpenClaw + `openclaw.json` + rl-training-headers（链路全部核查，详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md)）
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh`（18789 + token 修复，commit `83810e4`）
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh` / `smoke_run_qwen3_4b_openclaw_topk_select.sh` / `minitest_run_qwen3_4b_openclaw_topk_select.sh`
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768，2026-07-03 修复）

### 下一步
1. **重启 Simulator**：`git pull` + `launch_simulator.sh`（新 context=32768）
2. **重提交 smoke（4 GPU）验证**：`scripts/smoke_train_with_services.sh`，观察 `training.log` 是否出现 `combine samples: 16/16` → iter 1
3. **smoke 通过后**：修复 `train_with_services.sh` 启动顺序（先等 30000 再起 OpenClaw）→ 提交 8 GPU 正式训练

### 未验证
- [ ] smoke 重跑（18789 修复后验证训练队列）
- [ ] 8 GPU 正式 Table 3 训练

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
