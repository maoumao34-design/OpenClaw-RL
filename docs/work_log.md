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

**目标：** 审查 5 GPU pre-test 结果，确认无论文偏离；修复 smoke 18789 404 并验证训练数据生成

### Pre-test 结果审查

**完成内容：**
- 对照论文逐项核查九项（输出写法、模型、k/m/hint_selection、权重/clip、循环、收敛、累积、训练参数）→ 全部 ✅

### 问题修复（Pre-test 遗留）

**主要问题：**
- **Pre-test 0 训练步骤 + 无 checkpoint**：commit `482fdc6` 将 `OPENCLAW_GATEWAY_URL` 改为 30000 绕过 OpenClaw gateway，rl-training-headers 未注入 `X-Turn-Type:main` → 训练队列永远 0；架构核查全部确认 → 详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md)；三脚本改回 18789，commit `83810e4`
- **Simulator context overflow**：`launch_simulator.sh` 默认 16384，Policy 多轮后超限 → 核查官方确认 32768，已修；Simulator 需重启生效

### work_log 格式清理

**完成内容：**
- 审查并修复 06-23、06-29、06-30、07-01 四个日期条目的格式违规（非标字段、多行问题描述、分析级内容）→ commit `d49c574`、`f5a69a2`

### Smoke 18789 404 根因诊断与修复

**完成内容：**
- 通过 debug curl probe（`openclaw_debug.log`）确认根因：`/v1/chat/completions` 路由完全不存在于 `openclaw gateway run`，response body 为纯文本 `Not Found`（9 字节），`/v1/models` 同样 404
- 阅读 `openclaw_opd_api_server.py`，确认 `OpenClawOPDAPIServer`（port 30000）自带 `/v1/chat/completions`，通过 `X-Session-Id`（取自 `body.user`）和 `X-Turn-Type`（默认 `side`）区分 main/side turn；`openclaw gateway run` 是设备连接层，不提供 API 路由
- 实现 `scripts/rl_gateway_proxy.py`：取代 OpenClaw gateway，在 18789 接收请求、注入 `X-Session-Id` 和 `X-Turn-Type: main`，转发至 30000；smoke + minitest 两个脚本均更新 → commit `eafd060`

**主要问题：**（排查过程详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md)）
- 本次修复均不影响 `openclaw.json` 配置，workspace tools 暂未注入（不影响 smoke 验证，正式训练再评估）

**待下周验证：** smoke job（4 GPU）提交后，`training.log` 是否出现 `combine samples: 16/16` → iter 1 启动

---

## 2026-07-06

**目标：** 排查 smoke 训练队列为何仍为 0；核查 8 GPU 正式脚本是否同步了 smoke/minitest 的全部修复；深挖 18789 404 的真实根因

**完成内容：**
- 修复 smoke PRM judge 400/503（`PRM_MAX_NEW_TOKENS` 与缩配 context 冲突）→ commit `be0bc0e`；smoke 首次跑出真实训练样本提交，`update_weights()` 前的训练 step 确认执行 → 细节见 [`issues_log.md`](issues_log.md)
- 核查 8 GPU 正式脚本，补齐四处遗漏（gateway 启动方式、快速失败日志检测、断点续训 `--load`、`REPO_ROOT` 转发）→ commit `ed0aa01`、`61903e4`
- **重大方向修正**：`rl_gateway_proxy.py` 建立在 2026-07-03 的误诊之上——`openclaw gateway run` 本就内置完整 agent 循环（含工具调用）暴露 `/v1/chat/completions`，真实 404 根因是 `gateway.http.endpoints.chatCompletions.enabled` 默认关闭 + `model` 字段格式不符 → 详见 [`issues_log.md`](issues_log.md)
- 服务器验证：开启配置后认证/路由正常；撤掉 `rl_gateway_proxy.py`，三脚本改回真实 `openclaw gateway run`；新增 `scripts/prepare_openclaw_test_scripts.sh` 只打 `model` 字段补丁，官方 `openclaw-test/` 目录不动 → commit `ea19053`

**主要问题：**（细节见 [`issues_log.md`](issues_log.md) 2026-07-06 各条目）
- smoke `update_weights()` OOM（TP=1 缩配显存不足，评估 minitest/8GPU 不会复现，未在 smoke 上追加修复）
- `rl_gateway_proxy.py` 误诊 + 修复（本 session 最大方向修正，见上）

**待 modelfactory 验证：** minitest（5 GPU）排队中；smoke 用回真实 gateway 重跑一次，确认 tool call/文件读写真实发生 + 训练队列正常累积

---

## 2026-07-07

**目标：** 排查 smoke 训练队列持续为 0 的根因；确认论文原设计的 header 注入机制是否真的可行

**完成内容：**
- 修复 smoke job 内 `chatCompletions` 配置未跨环境生效（新 job 复现最初 404）→ 改为每次启动前强制 `config set` 并回读验证 → commit `9aa3c4a`
- 定位 OpenClaw 对未声明 `models[]` 的 sglang provider 请求了离谱大的 `max_completion_tokens`（178220），导致 408 → 显式声明 provider models（`contextWindow`/`maxTokens`）→ commit `18fac58`
- 修复后 smoke 首次让真实 agent 循环跑通（模型真的会调用文件工具），但训练队列仍为 0，日志显示全部请求 `[side] session=unknown`
- **全程用 CPU-only mock server 抓包排查**（不靠猜测、不占 GPU 排队）确认 `rl-training-headers` 插件端到端失效：manifest 缺 `enabledByDefault` 导致插件从未被尝试加载（对比 `browser`/`sglang`/`clickclack` 找到规律，`clickclack` 官方插件同样中招）；补上字段后插件确实加载、钩子确实触发、`fetch` 确实被 patch，但 header 依然传不到实际出站请求——确认是 **OpenClaw 本体的内部实现问题**，与论文设计、`OpenClaw-RL-official` 复现代码无关 → 细节见 [`issues_log.md`](issues_log.md)
- 相关调试/manifest 补丁全部复原；实现替代方案：`X-Turn-Type` 改用 OpenClaw 官方 `models.providers.sglang.headers` 静态配置（真官方功能，非绕过，已实测到达后端）；`X-Session-Id` 从 OpenClaw 自带的 system prompt "Runtime:" 行解析（明确记录为偏离论文设计），新增 `scripts/prepare_patched_openclaw_opd.sh`（拷贝打补丁，官方 `openclaw-opd/` 不动）→ commit `2c1e851`
- 独立单测（无需 GPU）确认补丁语法与正则提取逻辑均正确

**主要问题：**（细节见 [`issues_log.md`](issues_log.md) 2026-07-07 各条目）
- smoke Teacher 第 4 轮 context overflow（smoke 缩配 context=8192 导致，与本次早前两个问题同源，评估 minitest/8GPU 不受影响，未追加修复）
- `rl-training-headers` 端到端失效（本次最大排查成果，见上）

**待 modelfactory 验证：** smoke（4 GPU）排队中，确认 `X-Session-Id` 解析在真实链路里生效、训练队列正常累积

---

## 2026-07-08

**目标：** 提交 smoke（4 GPU）验证 07-07 的 header workaround 是否真的解决问题；排查 job 中途静默失败的原因

**完成内容：**
- 修复 smoke job 静默失败（无 Python traceback，进程直接消失）：系统级 OOM killer 因残留进程触发 cgroup 级联杀掉整个容器——07-07 手动测试时启动的 `/tmp/mock_sglang_server.py`（仍占着 30000 端口）和一个 `openclaw gateway run` 一直没有真正 kill 掉（当时只删了文件，没杀进程），清理后恢复正常 → 细节见 [`issues_log.md`](issues_log.md)
- 另一次提交失败是资源配置问题（误设 1 GPU/16GB，smoke 硬性需要 4 GPU），用户自行核实修正后重新排队
- **smoke 首次真正验证了 07-07 header workaround**：`X-Turn-Type: main` 静态 header 配置确认生效（`[main]` 出现 12 次，不再全是 `[side]`），训练队列首次真实累积样本（5 个真实样本被提交，PRM 评审也是真实投票而非全部失败）；但 `X-Session-Id` 的 Runtime 行解析**仍然全部返回 `unknown`**——PYTHONPATH、补丁文件、代码逻辑逐项核对都没问题，说明真实请求的 system prompt 内容和 07-07 手动测试时假设的不一致，具体原因还不清楚，加了调试日志（不猜测，直接打印真实内容）→ commit `593a0e0`

**主要问题：**（细节见 [`issues_log.md`](issues_log.md) 2026-07-08 各条目）
- job 静默失败：残留进程触发 cgroup OOM killer
- `X-Session-Id` Runtime 行解析在真实链路里不匹配，原因待查（已加调试日志，等下次结果）

**待明天验证：** smoke（4 GPU）排队中，等结果看 `[SESSION-ID-DEBUG]` 输出确认真实 system prompt 内容，定位为什么解析不到 Runtime 行

---

## 2026-07-09

**目标：** 验证 07-08 的 header workaround 在真实数据下是否有效；判断能不能修复论文原版 `rl-training-headers` 插件机制；提交 smoke/minitest 验证完整链路

**完成内容：**
- smoke（context 8192→32768 修复后）首次用真实数据完整验证：`X-Turn-Type` 静态 header + `X-Session-Id` Runtime 行解析**都确认生效**（真实 `session_id`、真实 19K+ token 的 `prompt_tokens`、真实样本入训练队列），之前怀疑的"session_id 解析失败"其实是 context=8192 太小导致真实轮次从未跑通的假象 → [`issues_log.md`](issues_log.md) 2026-07-09 第一部分
- 完整排查论文原版插件机制能不能用：`globalThis.fetch` 补丁、更底层的 undici `setGlobalDispatcher` 都试过，最终确认 **OpenClaw 加了一层专门的 SSRF 安全机制，对所有真实请求（非 Vitest mock）无条件绕开外部注入的 fetch/dispatcher，没有配置开关**；用直接拉取 2026 年 4 月版本 OpenClaw 源码验证，论文写插件那会儿这道机制还不存在，插件当年确实有效，是后续几个月的迭代把这条路封死的，不是论文或复现代码的问题 → [`issues_log.md`](issues_log.md) 2026-07-09 第二~五部分
- 改用 `appendSystemContext` 正文注入方案：插件把 `ctx.trigger`/`ctx.sessionId` 编码成标记塞进 system prompt，服务端补丁解析后在转发给 sglang / 计算训练样本之前清理掉，模型和训练数据都看不到这段标记——不受 SSRF 机制影响（改的是正文不是传输层）。mock server + 本地单测双重验证标记注入和清理逻辑都正确 → [`issues_log.md`](issues_log.md) 2026-07-09 第六部分，commit `be25e8b`
- 插件 + 服务端补丁部署逻辑接入 `smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三个脚本，保持一致，静态 `X-Turn-Type` header 已废弃 → commit `df22940` / `a7d1da6` / `73ccfef`
- 清理三种机制切换过程中的孤儿文件（`test_undici_header_injection.mjs`）和过时文档（`implementation_path.md` 架构描述更新到最新方案）→ commit `fab9560`
- 排查 smoke/minitest 反复静默崩溃问题：先排除 `RerunStateMachine`（查 Megatron 源码证实 DISABLED 模式下真实 NaN/Inf 依然会抛可捕获异常，跟"完全静默"对不上），加 `NCCL_DEBUG=INFO` 诊断（commit `d84b71c`）；重新提交 minitest 后**首次拿到完整 Python traceback**，确认真实根因是 **节点系统内存 OOM**（128GB 节点打满到 98.9%，Ray 自身内存监控杀掉 worker），发生在 `update_weights()` 权重同步阶段，与 07-06 记录的 smoke `update_weights()` OOM 是同一崩溃触发点、但资源种类不同（07-06 是 GPU 显存，这次是系统内存）→ [`issues_log.md`](issues_log.md) 2026-07-09 条目更新
- 权衡系统内存申请量：256GB 排队困难，改为先申请 192GB（128GB 峰值 126.63GB 之上留约 64GB 余量，比 256GB 好排很多）验证是否解决，不够再升级
- 顺带评估 CPU：当前 16 核，不是这次 OOM 的直接原因（不同资源），但崩溃日志里 top 进程列表显示 5 GPU 任务同时有 sglang scheduler/detokenizer、multiprocessing.spawn、gcs_server 等多个 CPU 侧进程，16 核偏紧；建议按比例一并提高，暂未做最终决定

**主要问题：**
- ~~smoke/minitest 连续三次...无 traceback、无 OOM 记录...根因未查清~~（已定位，见下）：反复静默崩溃的真实根因是**节点系统内存不足**（128GB 节点被 Megatron actor + rollout engine + PRM 等常驻进程打满），发生在 `update_weights()` 的 `pause_generation` 阶段；此前几次"静默无 traceback"很可能是同一个 OOM 杀在了没有异常捕获的 NCCL 集合通信调用中间，导致其余 rank 卡死，跟这次杀在 `ray.get()` 调用点上（有异常捕获、留下 traceback）是同一类问题的不同表现 → [`issues_log.md`](issues_log.md) 2026-07-09 条目更新

**待验证：**
- minitest 192GB 内存重跑，确认 OOM 是否解决（尚未排上队）
- 若不够，升级到 256GB；若解决，8GPU 正式提交同步申请更高系统内存
- `appendSystemContext` 标记会不会污染 OpenClaw 自己持久化的多轮对话历史（真实 GPU 链路里还未观察到异常，需要更长多轮对话验证）
- context-summarization 内部调用是否触发 `before_prompt_build`（决定 Task 摘要污染问题是否顺带解决）

---

## 2026-07-10

**目标：** 用新到的 A800 资源先验证 pipeline（H20 留给排队的正式跑）；确认 minitest 是否卡在网关/内存问题

**完成内容：**
- 补充 `paper_understanding.md` 第十一节：加回真正的 `openclaw gateway run` 提供的文件读写等工具执行职责（之前只列了 RL 数据管道那层职责）；顺带用官方源码核实修正了一处端口归属错误（18789=真正的 openclaw gateway，30000=`openclaw_combine_select_api_server.py`，此前标反了）；`implementation_path.md` 架构图同步更正 → commit `dc46261`/`168fa51`
- A800 提交 minitest（已申请更高内存），验证过程中追查"进度是否异常慢"：用 `minitest_20260709_172118`（上条 OOM 记录那次）做对比基准时发现该基准本身不干净——查 `simulation.log` 发现 INIT 阶段 TA/Teacher 都遇到过 `Connection refused`（网关 18789 中途短暂不可达），被脚本当"警告"静默跳过，`results_TA_init.txt`/`results_teacher_init.txt` 全是 0 字节，说明那次 homework1/homework2 数据本身不完整；确认这个网关断连问题和后面真正杀死任务的系统内存 OOM 是两个独立问题，不是同一根因 → [`issues_log.md`](issues_log.md) 2026-07-10 条目
- 修复：`run_one_persona()`（`minitest_train_with_services.sh` / `train_with_services.sh` 共用）改为每次调用前先复查网关是否可达，单次失败最多重试 3 次，不再一次失败就静默放过（`smoke_train_with_services.sh` 不用这个函数，不受影响）；确认此修复只改本地脚本源码，不影响当前已提交的 A800 job（提交时已拷贝脚本到自己的日志目录）→ commit `6324c18`
- 确认 A800 能跑真正的 Megatron 训练步（`training.log` 里 `Timer train start` 正常触发，无 kernel/import 报错）；顺带更正 07-06 `issues_log.md` 一条旧评估——"换 GPU 架构需要重新编译 flash-attn/TE/apex/flashinfer" 当时只是未经测试的猜测，现已被 A800 实测推翻；项目里唯一真正要求 sm_90 的是 DeepEP，但 Qwen3-4B 稠密模型不依赖它，环境搭建阶段就确认跳过没装（06-22 条目）→ commit `7e72777`
- 内存 OOM 修复确认有效：A800 minitest（已申请更高内存）连续跑过 10 次 `update_weights()`（`perf 23`-`perf 32`，无 OOM），此前每次都卡在第一次就死 → [`issues_log.md`](issues_log.md) 2026-07-10 条目，commit `c6ee638`
- 讨论 GPU 选型对 Table 3 复现数字的影响：架构差异（A800 vs H20）不会带来系统性偏差，但会像换随机种子一样引入正常的 run-to-run 数值波动（RL 训练本身依赖采样，对微小数值扰动敏感）；结论——A800 验证/H20 正式跑的策略没问题，但正式用来对比的数字要保持硬件一致，不要混用
- 决定：当前 A800 minitest 已达成验证目的（OOM 修复确认有效），停掉重新提交一次拉了 `6324c18` 之后代码的新 minitest，从 INIT 阶段直接验证网关断连重试修复是否生效

**主要问题：**
- INIT 阶段网关短暂不可达导致 TA/Teacher 数据静默丢失（已确认根因并修复，见上）
- A800 这次 minitest 进度明显比 07-09 那次"更慢"，但对比基准（07-09 那次）本身 INIT 不完整，不能直接下"A800 慢"的结论；已决定停掉重跑，不再纠结这个对比

**待验证：**
- 网关断连重试修复（`run_one_persona()`）——重新提交的 minitest 验证 `results_TA_init.txt`/`results_teacher_init.txt` 是否不再是 0 字节
- 8GPU 正式提交时同步应用 `run_one_persona()` 修复（`train_with_services.sh` 已同步改，无需额外操作）+ 申请更高系统内存

---

## 2026-07-13

**目标：** 检查 07-11 提交的 minitest（带网关重试修复）实际跑得怎么样；确认能不能开 wandb

**完成内容：**
- 排查 `minitest_20260711_003159`（跑了近两天）：`training.log` 显示 Ray 训练任务本身干净成功（`Job succeeded`），内存 OOM 修复持续有效（连续多次 `update_weights()` 无 OOM），但外层编排脚本没能跑到 `check_convergence.py`（训练一结束网关就被 SIGTERM、shutdown 超时）；手动用已有的 `results_*_all.txt` 补跑收敛检测 → [`issues_log.md`](issues_log.md) 2026-07-13 条目
- 收敛结果显示 TA/Teacher 228 个 session **全部**是错误占位文本（`couldn't generate a response`/`context overflow`），不是训练没学会，是从未真正生成过回复；查 `openclaw.log` 的 `[context-overflow-precheck]` 定位根因：`compaction.reserveTokens` 实际生效值 20000，导致留给 prompt 的预算被压到 12768，TA 批改任务 prompt 稳定 13.6K，超预算约 843 token
- 排查这个限制是不是像 SSRF 那次一样最近两个月新加的：查本地 `openclaw` 源码确认官方默认值是 16384（不是 20000），查 `CHANGELOG.md` 确认 precheck 机制可追溯到 2026.4.29，跟论文写插件同期甚至更早——**不是最近新加的限制**，是我们环境里这个值不知为何被设成了 20000（来源未查清，`openclaw.json` 里也搜不到）
- 修复：三个训练脚本网关启动阶段强制 `openclaw config set agents.defaults.compaction.reserveTokens 16384`，跟 `chatCompletions.enabled` 那个强制设置同一个模式 → commit `dec7ec2`
- 顺带确认 8GPU 正式脚本本来就有完整 wandb 支持（官方默认 `USE_WANDB=1`，读 `WANDB_KEY`/`WANDB_API_KEY`），不用改；minitest/smoke 之前把 `USE_WANDB` 写死成 0，改成可以被外部环境变量覆盖 → commit `36d0a9d`
- smoke（`smoke_20260713_110306`，`USE_WANDB=1`）验证：**`reserveTokens=16384` 修复确认生效**——`[verify] agents.defaults.compaction.reserveTokens = 16384`，TA 首次产生真实回复（不再是错误占位文本）；但发现一个新的独立问题：smoke 训练跑得比 INIT 模拟循环快，外层脚本一见训练进程退出就立刻杀网关，没等 INIT 跑完，TA 最后一轮被打断——已记录暂不修 → [`issues_log.md`](issues_log.md) 2026-07-13 第二条
- 重新提交 minitest（`minitest_20260713_112908`）复查，发现同一个"训练一结束就杀网关"问题也在这里出现（更正了之前"minitest 不易复现"的判断），且顺带挖出一个更严重的独立问题：这次 7 分钟就跑到 `perf 300`，查 checkpoint 目录 `latest_checkpointed_iteration.txt`=299（对应上一次跑了两天的 07-11 minitest 留下的进度）——**minitest 每次共用同一个 checkpoint 路径，`--load` 自动续训导致这次几乎没跑新训练就"续训完成"，这次结果无效**（不是真验证）→ [`issues_log.md`](issues_log.md) 2026-07-13 第三条
- 处理：清空 minitest 专用 checkpoint 目录（`rm -rf .../checkpoints/minitest-qwen3-4b-openclaw-topk-select`），确保下次 minitest 真正从头跑；不影响 8GPU 正式训练的 checkpoint（路径不同）
- 排查 wandb 一直不上报的原因：wandb.ai 需要走代理。中途踩了两个坑——① 脚本内 `source ~/.bashrc` 在 `set -u` 下被 `.bashrc` 里引用未设置的 `$PS1` 直接报错中断；② 绕过后 `pon` 报 `command not found`，最后定位到根因是同一个：**`pon` 是 alias，非交互式 bash 不展开 alias，也就不会触发 bashrc 里给 `$PS1` 兜底默认值的交互式初始化逻辑**。用户提供了同事的标准做法：`start_tools.sh`（`sing-box.sh start` + `source ~/.bashrc` + `pon`）配合 modelfactory 提交时 `代码解释器 = /bin/bash -i /dfs/data/start_tools.sh && /bin/bash -i`（`-i` 是关键，让整条链路在交互式 shell 下跑）。三脚本移除内置代理处理逻辑，改为要求用这种方式提交，头部注释同步更新 → [`issues_log.md`](issues_log.md) 2026-07-13 第四条，commit `89a27b4`
- ⚠️ 排查过程中用户不慎在对话里贴出了 `~/.bashrc` 里的明文 `WANDB_API_KEY`，已提醒去 wandb 网站撤销重新生成
- minitest/smoke 之前默认关闭 wandb，容易忘记 `export USE_WANDB=1` 导致误判代理没生效，改成默认开启 → commit `a579745`
- 用新提交方式（`start_tools.sh + bash -i`）重新提交后确认 **wandb 上报成功**（wandb 项目里出现真实 run `qwen3-4b-openclaw-topk-select_kl2fceaf-RANK_0`）；为了让我能看到图表，用户把 `openclaw_rl` 项目设成了 Public，我用浏览器访问确认能看到 Overview/Config/Summary 数据
- **在公开的 run 页面发现 "Command" 字段（wandb 自动记录的启动命令）里明文包含了 `--wandb-key <API_KEY>`**——项目一公开这个字段任何人不用登录都能看到，key 第二次暴露（第一次是贴 `.bashrc` 到对话里）。查 wandb 社区确认没有"只隐藏这个字段"的官方开关；查 `slime/utils/wandb_utils.py:40` 确认 `args.wandb_key is None` 时会跳过显式 `wandb.login()`、后续 `wandb.init()` 自己读 `WANDB_API_KEY` 环境变量兜底，不影响功能。改法：`RUNTIME_ENV_JSON.env_vars` 里加 `WANDB_API_KEY`，`WANDB_ARGS` 里去掉 `--wandb-key ${WANDB_KEY_VALUE}` 这一行，本地测试两处替换均正确匹配官方脚本、patch 后语法通过 → [`issues_log.md`](issues_log.md) 2026-07-13 条目，commit `781b602`
- 提醒用户手动做两件事（脚本解决不了）：把 `openclaw_rl` 项目 visibility 改回 Team/Private；去 wandb 网站撤销已暴露两次的 key 重新生成

**主要问题：**
- ~~TA/Teacher 全程 context overflow~~ **已确认修复生效**（见上，用 smoke 结果为准）
- 训练一结束就杀网关不等模拟循环跑完（smoke 和 minitest 都复现了，已记录暂不修）
- **minitest 复用旧 checkpoint 导致"续训到快完成直接结束"，验证结果失效**（已清空，需重新提交）
- 网关断连重试修复（`run_one_persona()`，commit `6324c18`）仍未被干净验证——两次都被别的问题盖住（07-11 是 reserveTokens 确定性失败，07-13 这次是 checkpoint 复用导致根本没跑够时间）
- ~~wandb 代理问题~~ **已验证生效**（见上），但过程中暴露了 wandb key 两次，已修复根因（改走环境变量）——用户需要手动撤销旧 key、把项目改回私有

**待验证：**
- 网关断连重试修复（`run_one_persona()`）——偶发问题，需要真的撞上才能验证
- 清空 checkpoint 后重新提交的 minitest，确认能不能观察到真正从头开始的完整流水线（INIT + Joint round + 训练）
- 新 run 的 Command 字段确认不再包含 `--wandb-key`，且 wandb 登录依然正常（验证环境变量兜底生效）

---

## 2026-07-14

**目标：** 验证前一天的修复；提交 8GPU H20 正式 Table 3 训练；排查训练中途遇到的问题

**完成内容：**
- 发现前一天提交的 minitest（`minitest_20260714_000203`）用的是 `git pull` 之前的旧代码，`reserveTokens`/wandb-key 两个修复都没生效，TA 又复现了 context overflow——确认是"提交时代码没更新"而不是新 bug，提醒以后提交前务必先 `git pull` 确认 `git log -1` 是最新 commit
- 提交首次 8GPU H20 正式训练，中途撞上 `update_weights()` 时 GPU 显存 `CUDA calloc` 失败 + NCCL 通信超时——排查后确认根因是 workspace 模式下前一次 Ctrl-C 中断没清理干净，3 个残留 `sglang::scheduler` 进程占着 GPU 4/5/6 各 80GB+ 显存，`kill -9` 清理后 8 卡恢复空闲，重新提交后训练正常推进（`rollout/step: 2`）→ [`issues_log.md`](issues_log.md) 2026-07-14 条目
- wandb 公开项目核实 key 修复生效：新 run（`8v8xutl0`）的 Command 字段确认不再包含 `--wandb-key`；确认此前暴露过 key 的旧 run 已删除
- 8GPU 训练继续跑后，INIT 阶段（Student/TA/Teacher 各 72 题）**全部以 0 字节告终**，`openclaw.log` 显示 11:40-12:00 持续大量 503。排查过程排除了四个假设（网关资源争抢、SGLang pause 时间过长、SGLang 容量不够、`submission_enabled` 被 checkpoint/eval 拖长——checkpoint 目录当时根本不存在，31 分钟训练步空档纯粹是没攒够样本的正常等待，跟 503 风暴是两个不相关的现象）→ [`issues_log.md`](issues_log.md) 2026-07-14 条目
- 排查中确认一个架构事实：训练循环不区分 INIT/Joint 阶段，INIT 还没跑完就已经开始拿 INIT 产生的对话当训练样本更新权重（`update_weights()` 11:38:17 发生时 Student 的 INIT 还在跑）。用官方 `openclaw-test/README.md` + 源码核实：**这是官方设计的常态**（官方参考流程本身训练 job 先起、模拟脚本后跑），不是 bug，不需要修
- 找到并修复一个真实设计缺陷：`run_one_persona()` 07-10 加的"失败整个重跑"逻辑，会让 `student_chat.py` 等脚本每次重新调用时清空自己的输出文件重新开始——如果第一次已经做出部分真实数据才失败，重跑反而把这部分数据也清空覆盖掉，比不重试保留得还少。改回只调用一次（保留"跑前确认网关可达"的检查），匹配官方参考流程的单次调用设计 → commit `0b25005`
- 进一步查证：Joint 阶段"每轮 6 题反复循环直到训练结束"这个结构本身也没有官方依据——`student_chat.py`/`TA_chat.py`/`teacher_chat.py` 是一次性脚本（无 `--loop` 参数），扩大到全部论文期允许目录搜索也确认没有官方编排脚本可以直接复用。改成 `run_joint_phase()`：INIT 建好 homework1/2 后三角色各自传 `JOINT_NUM_PROBLEMS=1319`（GSM8K 全量）同时并发启动一次，训练自然消耗真实样本直到自己结束 → commit `4be24ab`
- 讨论了 Joint 阶段三角色并发从题目 0 开始跑，TA/Teacher 有没有可能"超车"到 Student 还没写出的文件——查证 `TA_chat.py` 源码确认文件读取是模型自己的工具调用、不是 Python 检查，超车不会导致脚本崩溃，只会产生"对着不存在的文件对话"这种低质量轮次；官方代码没有任何防超车机制。决定不额外加保险措施，先跑起来实测观察
- 用改完的新版本重新提交 8GPU 正式训练（run `8yn4i8ml`，16:07:25 开始）。通过 wandb Overview 页确认：Command 字段已不含 `--wandb-key`（key 修复再次验证生效）、8×H20、`rollout/step` 正常推进、response 长度无截断/复读
- 全面核对了 wandb `train`（21 项）/`rollout`（22 项）全部指标的确切含义，对照官方 loss 源码 `openclaw_topk_select_loss.py`/`hint_opd_loss.py` 逐一确认，区分出训练健康度哨兵、配置常数、以及本次配置下架构上必然恒定的值，并在 wandb 上搭建了 10 张核心图的 "important" 固定分组（`train/loss`、`grpo_pg_loss`、`opd_loss`、`grad_norm`、`rollout/prm_eval_score`、`advantages`、`zero_std/count_1.0`、`zero_std/count_-1.0`、`response_len/mean`、`ref_log_probs`）；确认 wandb 自带的 "Save personal workspace template" 功能可以让这个布局自动套用到后续新 run，不需要改代码
- 排查 `train/opd_loss` 长期显示为常数 -1.0 的现象：先用 `training.log` 实测确认真实 OPD 教师信号样本占比 67%（118 条 OPD+RL + 6 条 OPD-only vs 60 条 RL-only），排除"没有真实教师数据流入"的假设；再从源码确认 `--num-steps-per-rollout 1` 导致 `rho_v`（PPO 比率）架构上精确恒等于 1，进而 `ppo_kl_sampled`/`opd_pg_clipfrac`/`grpo_pg_clipfrac` 三个指标恒为 0（wandb 图上逐一验证属实）。结论：这是早期训练阶段的正常数学性质，不是数据链路或代码 bug，预期会随训练步数增加、policy 与初始权重逐渐拉开距离后自然出现波动，暂不需要修复，记为待观察
- 用 `training.log`/`simulation.log`/`results_student_init.txt` 实测确认两处修复已生效：① GPU calloc/残留进程问题彻底解决，`update_weights()` 已成功执行 30+ 次无崩溃；② `run_one_persona()` 单次调用修复生效，Student INIT 正常推进到第 43/72 题，产出真实完整的多轮对话数据，未再复现"跑 70 分钟后 0 字节放弃"的问题

**主要问题：**
- INIT 阶段 503 风暴根因仍未 100% 精确定位到触发机制，但推测跟 Joint round 循环结构无关（该结构已经改掉，待验证问题是否随之缓解）
- Joint 阶段三角色并发可能存在"超车读空文件"的数据质量风险，官方无防护，已知不改，靠实测观察
- 截至记录时 Joint 阶段（TA/Teacher）仍未开始，Student INIT 还在第 43/72 题——`run_joint_phase()` 设计和"超车"风险都还没有真实数据可验证

**待验证：**
- Joint 阶段启动后：TA/Teacher 能否正常产出数据、训练能否持续推进、有无"超车"读空文件现象（TA/Teacher 对话里频繁出现"文件不存在"）
- `train/opd_loss`、`train/opd_teacher_student_logp_topk_abs_mean` 等目前卡在常数的指标，是否会在训练步数增多、Joint 阶段介入、样本更多样后开始正常波动，确认"早期训练巧合"这个判断成立

---

## 2026-07-15

**目标：** 排查 07-14 提交的 8GPU 训练（run `8yn4i8ml`）为何跑了约 7 小时后无声消失；解决根因，重新跑通 INIT 阶段

**完成内容：**
- 排查 `8yn4i8ml` 消失的根因：TA 从 INIT 第 0 题起就持续遇到生成失败（`stopReason=length`），在第 23/24 题命中率骤增、用尽重试仍产不出样本，导致 `RolloutManager` 卡在 `waiting for combine samples` 长达 50 分钟，GPU 实际空闲触发了 modelfactory 平台的自动回收——本地 `training.log` 与 wandb 独立上报的日志精确同时断在 23:12:39、均无报错痕迹，证实是平台强制终止而非代码崩溃 → [`issues_log.md`](issues_log.md) 2026-07-15 条目
- 对照官方 `README.md` 标准配置（`contextWindow=32768` 配 `maxTokens=8192`），发现我们三个脚本一直用的是 `maxTokens=4096`——是 07-07 那次 smoke 专用修复的历史遗留值，`contextWindow` 后来改回官方值时没有同步重新计算。改成官方一致的 8192，`train_with_services.sh`/`minitest_train_with_services.sh`/`smoke_train_with_services.sh` 均已修改 → commit `5c8c323`
- 用改完 `maxTokens` 的版本重新提交 8GPU 训练，仍然失败：Student INIT 只跑到第 35 题、TA 只跑到约第 11 题就崩溃（`ReadTimeout`/`408`）。深挖发现两者本质是同一件事——反复卡在 `[context-overflow-precheck]` 失败重试循环里空转 2 分钟左右，最后才被包装成"timeout"抛出，不是生成变慢
- 定位到真正根因：`effectiveReserveTokens` 实际生效值一直是 20000，跟脚本设置的 `reserveTokens=16384` 完全对不上；对比昨晚崩溃 run 的日志确认这个问题在改 `maxTokens` 之前就已存在，不是本次改动引入的。WebSearch 官方 GitHub 找到确切原因：[Issue #66830](https://github.com/openclaw/openclaw/issues/66830)——OpenClaw 的 `memoryFlush`/`preflight` 阈值计算逻辑根本不读 `reserveTokens` 字段，读的是另一个我们从未配置过的 `reserveTokensFloor`，不分 provider/模型都会复现（一开始误判为另一个长得很像但已修复的 Ollama 专属 issue #65465，经用户追问逻辑漏洞后排除，重新定位到 #66830）
- 评估过升级 OpenClaw（2026.6.9→2026.6.10）绕开这个 bug，但官方 release notes 无法确认包含针对性修复，且当前流水线是针对现有版本反复调好的，升级风险大于收益，未采纳；改为显式设置 `agents.defaults.compaction.reserveTokensFloor=16384`，三个脚本均已加上这个设置 → commit `d205fc7`
- 丢弃了改 `maxTokens` 那次 run 的不完整数据（Student/TA/Teacher 分别只检查了 36/23/11 个 session，"Table 3" 收敛数字没有参考价值），清理残留 GPU 进程（`sglang::scheduler`，Ctrl-C 未能完全清理，本次两次撞上），重新提交一次干净的 8GPU 训练（run 目录 `20260715_162015`）
- 用一次手动构造的诊断探测请求（直接发给正在跑的网关，绕开等待真实对话攒够长度）**确认 `reserveTokensFloor` 修复在真实运行时生效**：`effectiveReserveTokens` 终于变回 16384，不再是 20000
- 上面这次重新提交的训练（`20260715_162015`）仍然在第 40 题左右开始持续性失败（Student 最终只到 51/72、Teacher 同样崩溃）——排查确认这次根因完全不同：是 `update_weights()` 触发 `pause_generation`/清空 KV 缓存时，正好打断了"正在处理中"的对话请求（`Provider finish_reason: abort`/503），且缓存清空后长对话需要整段重新预填充，之后同一批 session 反复出现"生成结束但内容为空"（`stopReason=stop, emptyRetries=1/1`）的持续性故障，不会自己恢复
- 一开始怀疑显存/GPU 资源，`nvidia-smi` 排除；改查系统内存，`free -h` 一开始查到宿主机 1.5TB/健康，被用户指出应该查的是申请 workspace 时选的资源额度（cgroup 限额），不是宿主机总量——查 `/sys/fs/cgroup/memory.max` 确认这次 workspace 的内存限额其实是 **256GB**，当时已用到约 200GB（78%），且还在持续爬升（对话历史、Ray 缓存不断累积）。对比 07-09 那次 5GPU minitest（128GB 限额、被打到 98.9%）的历史经验，这次 8GPU（进程数是 5GPU 配置的 1.6 倍）明显需要比 256GB 更多，建议下次至少申请 512GB 起步（未采纳原有的"一步到位申请最大值"思路，沿用 07-09"实测峰值+余量，不够再加"的做法）
- 准备关闭当前 workspace 重新申请更大内存时，触发了 workspace 自己的持久化存储配额告警（"已用存储已超过最大限制（2GB）"）——确认这是跟训练内存完全独立的另一个配额（只管"已安装软件包/环境设置"这类会被保存的内容，`/dfs/data` 不受影响），排查后清理了 460MB 历史 session 转录文件（`~/.openclaw/agents/main/sessions/`，都是今天已经决定丢弃的几次失败 run 留下的对话记录）和 460MB npm 缓存，两者均确认不影响 OpenClaw 本体（`/usr/lib/node_modules/openclaw`，613MB）
- 用户重新申请了 **64 CPU / 1024GB 内存**的新 workspace，`start_tools.sh` 起代理 → `git pull` 拉到最新代码 → 确认 8 卡干净 → 重新提交训练（run 目录 `20260715_180549`），并在新 workspace 里重跑一次诊断探测请求，**再次确认 `reserveTokensFloor` 修复在新环境下同样生效**（`effectiveReserveTokens=16384`）

**主要问题：**
- TA/Student 在跑到第 10-40 题这个区间（不同 run 不完全一致）会开始持续性失败，根因是 `update_weights()` 的 pause/缓存清空打断在途请求，且暂无法从代码层面根治（属于异步 RL 架构本身的竞态，官方设计假设"训练不干扰推理"多数时候成立，但偶发打断后的恢复行为不稳定）——这次换成 256GB→1024GB 内存后是否显著改善还需要观察，如果内存不是主因，这个问题可能仍会复现
- `run_init_phase()`/`run_one_persona()` 对"角色没跑完 72 题就崩溃"没有阻塞机制的设计缺陷（07-14 就发现）今天又让两次 run 的数据作废，目前仍未修

**待验证：** 明天查看 `20260715_180549` 这次训练（64CPU/1024GB 新 workspace）INIT 阶段能否完整跑完 72 题、Joint 阶段能否正常持续，判断内存是不是今天反复出现"持续性失败"的真正主因。

---

## 2026-07-16

**目标：** 排查 `20260715_180549` 训练 `train/grad_norm` 爆炸的根因并修复；解决 workspace 代理服务故障；重新提交训练验证

**完成内容：**
- 精确核对样本-index 到训练-step 的映射，确认单步跳变最大的一次（step21→22）实际消费的 16-18 条样本里至少 7 条是 `response_len=7~8` token 的退化样本、全部 `reward=-1.0`——不是时间上巧合关联，是这一步梯度爆炸所用的训练数据本身被这类样本主导 → [`issues_log.md`](issues_log.md) 2026-07-16 条目
- 抓取实际文本内容确认退化样本本质：一个跨两次独立训练复现的乱码字符 `𬣳`（Qwen3 词表 token id=122362）。embedding 范数检查排除"固有异常权重"假设（正常，14.76 百分位）；确认 Megatron 数据侧种子、SGLang 生成侧种子均固定为 1234（`args.seed + rank`），部分解释跨 run 复现同一 token 的现象，但外部 Simulator 不受控，无法完全证实
- 补上顶格截断（`finish_reason=="length"`）时的 `reasoning_text` 完整日志（此前只记字符数），确认这是真实的"卡死循环"（反复重复车轱辘话），不是正常推理超预算——本次不过滤，先攒诊断材料
- 实施并修正生成/数据管线补丁（`prepare_patched_openclaw_opd.sh`）：用 `logit_bias` 在生成阶段直接屏蔽已知乱码 token（首版实现搞错成"生成后检测丢弃"，只保护训练数据、对话本身仍会收到坏回复，经指出后修正为生成时屏蔽）→ commit `ad56d7c`/`52c4fc6`
- 用新修复重新提交训练（run `20260716_143407`）：乱码 token 确认 0 次复现，但 grad_norm 仍缓慢爬升，且发现新问题——TA 批改作业时反复调用 OpenClaw 自带的 `memory_get`（读取按日期命名的记忆文件）或 `HEARTBEAT.md`，完全不回应批改指令。量化统计：27 道题里 **37% 撞 8 轮上限失败、30% 出现过这类干扰**
- 查证论文/官方代码均未提及如何避免这类干扰（`rl-training-headers` 插件只处理外部 heartbeat/memory/cron **触发器**发起的对话，管不到模型在正常 main turn 内**主动调用**这些工具）；查 OpenClaw 产品自身源码定位到 `memory_get` 属于 `memory-core` 插件，跟 homework 读写工具架构无关、可安全禁用，`HEARTBEAT.md` 等"agent 身份文件"属于核心代码（`src/agents/bootstrap-files.ts`）无法禁用 → `openclaw plugins disable memory-core`
- 重新评估过滤规则，去掉按 `content` 长度过滤（`<5` 字符）——像"25"这种被判 -1 的短数字回复是正常有效的 RL 训练信号（教模型"这样答不满足要求"），不该被当坏样本剔除；只保留官方原有的"完全空内容"检查 + 已知乱码 token 兜底
- 给 `tool_calls:` 日志补上 `session_id`，解决此前"并发日志天然交错、事后无法按 session 可靠关联"的分析障碍 → commit `cae49ec`/`00d9195`
- 用全部修复重新提交训练（run `20260716_182012`），当前在 INIT 阶段，结果留待明天查看

**主要问题：**
- workspace 自带的 `sing-box` 代理服务（`127.0.0.1:7893`）中途失效，`git pull`/wandb 上报连不上；排查发现 `.bashrc` 无条件导出代理环境变量，但底层代理进程没启动，跟存储配额清理无关（家目录只有 243M，远低于 2GB 配额）——`bash /dfs/data/start_tools.sh` 重新拉起代理解决；另发现 git 远程地址被配置成走 `ghproxy.net` 镜像重写，实测网络能直连 GitHub，改回直连更简单可靠
- 一开始把"退化样本过滤"方案搞错——该在生成时用 `logit_bias` 屏蔽，却做成了生成后检测丢弃，被指出后修正

**待验证：** 明天查看 `20260716_182012` 这次训练：(a) memory_get/HEARTBEAT 干扰是否消失、TA 撞轮次上限失败率是否显著下降；(b) grad_norm 是否能保持稳定，不再重演早期爬升；(c) 顶格截断的 `reasoning_text` 日志如果再次触发，能否确认是不是同类"卡死循环"。

---

## 2026-07-17

**目标：** 查看 `20260716_182012` 训练结果；排查"决策犹豫循环"（顶格截断）根源

**完成内容：**
- 确认 `20260716_182012` 跑了 8 小时后无声消失，但 wandb 显示 `train/grad_norm` 全程 2-8 波动、整体下降，**没有**复现乱码 token/memory_get 那次的爆炸式增长——两个已知根因（乱码 token、memory_get）确认修复有效：全程 0 次复现，退化过滤触发 0 次，训练数据干净 → [`issues_log.md`](issues_log.md) 2026-07-16 条目更新
- 定位新的主导性问题：`rollout/response_len/mean` 从 step 20 起顶在 7000 附近不再下降，`waiting for combine samples: 6/16` 卡住 10+ 分钟——之前特意保留只做诊断日志的"顶格截断"这次高达 298 次。抽查 `reasoning_text` 确认是同一种"决策犹豫循环"（反复重新分析同一情况、从不真正推进，直到耗尽 8192 token 预算被截断），推理原文提到 `"Non-final turn: use tools to advance, or ask for the one missing decision that blocks safe progress."`
- 定位这句话来源：OpenClaw 产品自身源码 `src/agents/system-prompt.ts:456`（`buildExecutionBiasSection()`），是产品内置、面向所有 agent 会话默认注入的工具使用指南，跟训练代码/我们的脚本无关（跟 AGENTS.md/memory_get 同一类"产品自带默认值"）。顺着 Qwen3.x 社区已知问题（`reasoning_content` 跨轮丢失导致模型"失忆"）查了 OpenClaw 的 `shouldPreserveReasoningContentReplay()` 判断逻辑，确认我们 `qwen3-4b` 模型声明已带 `"reasoning": true`，按代码逻辑推理内容应该被正常保留传回——这条线索被排除，不是根因
- 确认这次训练结束的机制还是已知的老问题：顶格截断拖慢生成速度（每条 1-3 分钟）→ 攒批跟不上 → GPU 空闲 → 触发 modelfactory 平台自动回收 workspace，只是这次的诱因从 TA 的 `stopReason=length` 换成了这个"决策犹豫循环"
- 训练结束后查 `homework/`/`homework1/`，一度怀疑内容陈旧（07-15 的旧版本）导致 TA 一直在批改过期素材，用 `stat` 核实是 workspace 2GB 配额区在"GPU空闲→平台自动回收→重启"这条路径触发后，从上次保存的快照（07-15 17:41）静默回滚导致的——只反映训练**结束后**的状态，不代表训练**进行中** TA 实际看到的也是陈旧内容（`simulation.log` 里"Written: homework/34.txt"证明进行中写入是正常的）
- 排查过程中发现一个独立的官方设计空白：Problem 34 有一次真实的"假成功"——Student 让模型写入 `homework/34.txt` 失败（`⚠️ 📝 Edit ... failed`），但模拟器在能看到这条失败警告的情况下依然生成了 `DONE_SENTINEL`（`"HOMEWORK_DONE"`），判定完成。查证 `student_chat.py:205-207` 确认官方"完成"判定完全只看模拟器文字里有没有这个哨兵字符串，不检查任何工具调用是否真的成功——这是官方原装设计，不是我们复现引入的偏差
- 提出新假设待验证：这类"写入失败但被判定完成"是否会导致对应题目的 `homework1` 素材残缺，进而在 TA 后续批改时因为素材本身矛盾/不完整而触发"决策犹豫循环"——即 Problem 34 的假成功可能是循环问题的上游诱因之一

**待验证：** 统计这次训练里所有"Student 侧工具调用失败但被误判完成"的题目，与所有触发"决策犹豫循环"的 TA session 做交叉比对，验证两者是否显著相关；如果相关，需要评估修复方向（在 `DONE_SENTINEL` 判定前加工具调用成功校验，但这会偏离官方原装代码，需先评估对复现有效性的影响）。

### 决策犹豫循环根因定位与修复

**完成内容：**
- Problem 33/36 排查：定位 Problem 36 循环的精确机制是纯输出格式判定困惑（`tool_call` 标签要不要包），不是内容困惑 → [`issues_log.md`](issues_log.md) 2026-07-16/17 条目
- 版本考古确认触发循环的 Execution Bias 章节是论文提交后 1.5 个月才加入 OpenClaw；评估版本回退可行性后暂缓，先做定点修复
- 版本差异扫描顺带发现两个更新的 retry 机制（PR #92191/#93073），影响待验证
- 修复：先用 append 方案（已 revert，用户要求更高可靠性），改用内容层直接 patch 内置 sglang 扩展 → `scripts/prepare_patched_sglang_execution_bias.sh`
- **8GPU 正式训练（run `20260717_133740`）已提交，patch 确认真实生效**（`openclaw.log` 有确认日志，Problem 0 无循环迹象）

**待验证：** 这次训练顶格截断次数是否显著下降、循环是否还会出现。

---

## 当前状态（2026-07-17）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测）
- [x] `maxTokens=8192`、`reserveTokensFloor=16384`：已验证生效（07-15）
- [x] `logit_bias` 屏蔽已知乱码 token（id=122362）：**已用真实 GPU 数据验证 0 次复现**（run `20260716_182012` 全程 0 次）
- [x] `memory-core` 插件禁用：**已用真实 GPU 数据验证 0 次复现**（run `20260716_182012` 全程 0 次 `memory_get`）
- [x] 退化样本过滤规则（只拦真正空内容 + 已知乱码 token 兜底）：**已验证生效**，全程 0 次误触发
- [x] `tool_calls` 日志补 `session_id`：支持事后按 session 可靠关联分析
- [x] Git 远程地址改回直连 GitHub
- [x] "决策犹豫循环"根因定位（Execution Bias 章节，论文提交后新加）+ 修复（`prepare_patched_sglang_execution_bias.sh` 内容层 patch）：**已用真实训练数据验证钩子真实生效**（run `20260717_133740`，见上）

### 已知限制 / 未解决
- "决策犹豫循环"修复**已确认生效但效果未知**——patch 确认真实触发，但顶格截断次数能不能显著下降、循环还会不会出现，需要这次 8GPU 训练的完整结果才能判断
- 新发现两个比 Execution Bias 更新的机制（PR #92191/#93073，2026-06-14/15 合并）：June 版本会自动重试"只有 thinking 没输出"的轮次，March 没有，可能复合放大循环伤害的影响尚未验证
- workspace 的 2GB 配额区（`~/.openclaw/workspace/`）在"GPU 空闲→平台自动回收→重启"后会静默回滚到上次保存的快照，训练**结束后**查看 workspace 文件状态时需要注意，不能直接当作训练进行中的真实状态
- 官方 `DONE_SENTINEL` 完成判定不校验工具调用是否真的成功，是官方设计本身的空白（假设待验证：是否为循环问题的上游诱因之一）
- `run_init_phase()`/`run_one_persona()` 缺乏阻塞机制的设计缺陷仍未修（07-14 起多次提及）

### 下一步
1. 观察 run `20260717_133740` 完整训练结果，确认顶格截断次数是否显著下降、循环是否消失
2. 如果这次仍复现循环，评估是否需要处理 PR #92191/#93073 这类更新机制，或重新评估版本回退
3. 交叉比对"Student 假成功"题目与"TA 决策犹豫循环"session，验证因果关系假设（次要优先级）

### 未验证
- [ ] "决策犹豫循环"修复的实际效果（钩子已确认触发，结果待观察）
- [ ] "Student 假成功"与"TA 决策犹豫循环"的因果关系
- [ ] 8 GPU 正式 Table 3 训练完整跑通

---

## 历史状态（2026-07-16，已被 7/17 结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测）
- [x] `maxTokens=8192`、`reserveTokensFloor=16384`：已验证生效（07-15）
- [x] wandb 集成：确认今天的连不上是 workspace 代理服务故障，不是训练本身问题，`start_tools.sh` 修复后正常
- [x] `logit_bias` 屏蔽已知乱码 token（id=122362）：生成阶段直接屏蔽，**已用真实 GPU 数据验证 0 次复现**（run `20260716_143407`）
- [x] 退化样本过滤规则：简化为只拦真正空内容（跟官方一致）+ 已知乱码 token 兜底，不再按 `content` 长度过滤
- [x] `memory-core` 插件禁用：解决 `memory_get` 工具干扰，**尚未验证新 run 实际效果**
- [x] `tool_calls` 日志补 `session_id`：支持事后按 session 可靠关联分析
- [x] Git 远程地址改回直连 GitHub（不再依赖易失效的 `ghproxy.net` 镜像）

### 已知限制 / 未解决
- `HEARTBEAT.md`/`AGENTS.md` 等 OpenClaw 核心自带"agent 身份文件"无法禁用（不是插件），只能靠退化过滤兜底，模型仍可能偶尔读到但预期影响远小于 `memory_get` 的卡循环模式
- grad_norm 缓慢爬升的**最初触发点**仍未 100% 定位——乱码 token、`memory_get` 卡循环只是两个已确认会放大问题的"下游因素"，是否还有更早的触发原因尚不清楚
- `run_init_phase()`/`run_one_persona()` 缺乏阻塞机制的设计缺陷仍未修（07-14 起多次提及）
- workspace 的 `sing-box` 代理服务偶发失效，需要手动 `start_tools.sh` 重启，暂无自动恢复机制

### 下一步
1. 查看 `20260716_182012` 训练结果，确认这批修复是否让训练稳定跑过之前失控的窗口
2. 如果 grad_norm 依然爬升，需要继续往前找最初触发点（不只是已知的两个放大器）
3. INIT+Joint 全部跑通后，观察 wandb 曲线 + `check_convergence.py` 结果

### 未验证
- [ ] `20260716_182012` 训练能否稳定跑完 INIT+Joint，不再复现 grad_norm 失控
- [ ] `memory-core` 禁用后 TA 撞轮次上限失败率是否显著下降
- [ ] 8 GPU 正式 Table 3 训练完整跑通

---

## 历史状态（2026-07-15，已被 7/16 结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测）
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次起 gateway 前强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`，`maxTokens` 已改为官方值 8192（原 4096 是历史遗留）
- [x] `rl-training-headers` 插件 + `openclaw_opd_api_server.py` 标记解析：**已用真实 GPU 数据验证生效**
- [x] `agents.defaults.compaction.reserveTokens=16384` + **新增 `reserveTokensFloor=16384`**：**已用诊断探测请求验证运行时真正生效**（此前 `reserveTokens` 单独设置对实际 precheck 无效，是 OpenClaw 官方已知 bug #66830）
- [x] wandb 集成：**已验证成功**，key 已改走环境变量不再暴露在 Command 字段
- [x] 系统内存 OOM 修复：**已验证**，A800/H20 上多次 `update_weights()` 无 OOM
- [x] GPU calloc / workspace 残留进程问题：机制已确认（Ctrl-C/中断后必须手动清理 `sglang::scheduler` 残留，07-15 又复现两次）
- [x] `run_one_persona()` 单次调用 + Joint 阶段一次性并发：设计已确认，但截至目前还没有一次 run 完整跑完 72 题验证到 Joint 阶段
- [x] `train/opd_loss` 常数 -1.0 现象排查完毕：确认真实教师信号占比 67%、`rho_v=1` 是架构必然，判定为早期训练正常现象，非 bug
- [x] wandb "important" 图表分组（10 张核心图）+ 个人工作区模板，后续新 run 自动套用
- [x] `scripts/check_convergence.py`

### 已知限制 / 未解决
- **8GPU 正式训练的 workspace 内存额度，之前一直是按 256GB 申请的，实测跑到 INIT 中途就用掉 78%、仍在爬升**——已改成 64CPU/1024GB 重新申请，但还没有一次完整 run 验证这个新额度是否真的够用
- `update_weights()` 触发的 pause/KV 缓存清空偶尔会打断"正在处理中"的对话请求，之后同一批 session 可能陷入持续性"生成结束但内容为空"的故障，不会自己恢复——怀疑跟内存压力有关联但未最终证实，是目前"跑到一半开始连续失败"的头号嫌疑
- workspace 模式下 GPU 长时间空闲（比如 rollout 饥饿）会触发 modelfactory 平台自动回收整个 workspace，训练进程和日志会毫无征兆地一起消失，本地日志查不出任何报错——`reserveTokensFloor` 修复降低了 context overflow 这个已知饥饿诱因，但内存压力也可能导致同样的饥饿链条，暂无系统性预防手段（讨论过 GPU keepalive 兜底，暂未实施）
- `run_init_phase()`/`run_one_persona()` 目前对"某个角色没跑完 72 题就崩溃"没有阻塞机制，只警告后放行，导致下一阶段可能建立在不完整数据上——07-15 三次 run 都因此产生了不完整/需要丢弃的数据，这个设计缺陷本身还没有修
- Joint 阶段三角色并发"超车"读空文件的风险仍未获得真实数据验证（至今没有一次 run 完整走到 Joint 阶段）
- 训练一结束就杀网关不等模拟循环跑完（已记录暂不修）
- `appendSystemContext` 标记多轮对话下的稳定性、context-summarization 是否触发 `before_prompt_build`，仍待验证
- workspace 自己的持久化存储配额只有 2GB（跟训练用的系统内存是完全独立的两个概念），关闭 workspace 前如果快超了要记得清理 `~/.openclaw/agents/main/sessions/`（历史对话转录）和 `~/.npm`（缓存），不要动 `/usr/lib/node_modules/openclaw`（本体）

### 下一步
1. 查看 `20260715_180549` 这次训练结果（64CPU/1024GB 新 workspace）：INIT 能否完整跑完 72 题、Joint 阶段能否正常推进，判断内存是否是根本瓶颈
2. 如果新内存额度下仍有角色跑不完 72 题，说明内存不是唯一原因，需要考虑给 `run_init_phase()` 加阻塞/重试机制（目前是已知设计缺陷，暂未修）
3. INIT+Joint 全部跑通后，观察 wandb 曲线 + 最终 `check_convergence.py` 结果
4. 8GPU 正式训练固定用同一种 GPU 架构和内存额度，不与其他方法/基线混用硬件配置

### 未验证
- [ ] 64CPU/1024GB 新 workspace 能否让 INIT 阶段三个角色都完整跑完 72 题（这是今天第三次尝试，前两次分别卡在 maxTokens 和 reserveTokensFloor，这次换了内存额度）
- [ ] `update_weights()` 打断在途请求导致的"持续性空回复"故障是否会因为内存余量变大而消失，还是内存无关、需要另外处理
- [ ] Joint 阶段"超车"现象是否显著影响数据质量（至今没有真实数据）
- [ ] 8 GPU 正式 Table 3 训练完整跑通

---

## 历史状态（2026-07-14，已被 7/15 结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测）
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次起 gateway 前强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`
- [x] `rl-training-headers` 插件 + `openclaw_opd_api_server.py` 标记解析：**已用真实 GPU 数据验证生效**
- [x] `agents.defaults.compaction.reserveTokens=16384` 强制设置：**已验证生效**（TA 产生真实回复）
- [x] wandb 集成：**已验证成功**，key 已改走环境变量不再暴露在 Command 字段
- [x] 系统内存 OOM 修复：**已验证**，A800/H20 上多次 `update_weights()` 无 OOM
- [x] GPU calloc / workspace 残留进程问题：**已验证解决**，`update_weights()` 已成功执行 30+ 次无崩溃（run `8yn4i8ml`）
- [x] `run_one_persona()` 改回单次调用（不再整体重跑丢数据）→ commit `0b25005`，**已用真实数据验证生效**（Student INIT 正常推进到 43/72 题，产出真实完整对话）
- [x] Joint 阶段改为一次性并发启动（不再是无官方依据的分轮次循环）→ commit `4be24ab`，**Joint 阶段本身尚未开始，仍待验证**（截至记录时 Student INIT 还在第 43/72 题）
- [x] `train/opd_loss` 常数 -1.0 现象排查完毕：确认真实教师信号占比 67%、`rho_v=1` 是架构必然，判定为早期训练正常现象，非 bug
- [x] wandb "important" 图表分组（10 张核心图）+ 个人工作区模板，后续新 run 自动套用
- [x] `scripts/check_convergence.py`

### 已知限制 / 未解决
- INIT 阶段 503 风暴根因未 100% 精确定位（已排除四个假设，见 [`issues_log.md`](issues_log.md)）
- Joint 阶段三角色并发可能"超车"读到空文件，官方无防护，已知不改，靠实测观察，Joint 阶段尚未开始所以还没有真实数据
- workspace 模式下手动中断（Ctrl-C）清理不彻底会残留 GPU 进程，每次重新提交前必须手动 `nvidia-smi` + `ps aux | grep "openclaw gateway"` 确认干净
- 训练一结束就杀网关不等模拟循环跑完（已记录暂不修）
- `appendSystemContext` 标记多轮对话下的稳定性、context-summarization 是否触发 `before_prompt_build`，仍待验证

### 下一步
1. 用今天改完的新版本（单次调用 + 一次性并发 Joint 阶段）重新提交 8GPU 训练
2. 确认 INIT 数据完整、Joint 阶段持续产出、训练正常推进后，观察 wandb 曲线 + 最终 `check_convergence.py` 结果
3. 8GPU 正式训练固定用同一种 GPU 架构，不与其他方法/基线混用硬件

### 未验证
- [ ] 新版 `run_one_persona()` + `run_joint_phase()` 在真实 8GPU 训练上的完整效果
- [ ] Joint 阶段"超车"现象是否显著影响数据质量
- [ ] 8 GPU 正式 Table 3 训练完整跑通

---

## 历史状态（2026-07-13，已被 7/14 结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测，flash-attn/APEX/TE/flashinfer 非 H20 专属编译）
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次起 gateway 前强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`（`contextWindow=32768`/`maxTokens=4096`），不再用静态 header
- [x] `scripts/prepare_patched_rl_training_headers.sh`：`rl-training-headers` 插件 `appendSystemContext` 版本，**已用真实 GPU 数据验证生效**（真实动态 `session_id`/`turn_type` 标记到达请求正文）
- [x] `scripts/prepare_patched_openclaw_opd.sh`：解析标记 + 转发前清理，**已用真实 GPU 数据验证生效**
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本已统一接入上述所有 workaround
- [x] `run_one_persona()` 网关断连重试修复，代码已就绪，**尚未被干净验证**（07-11 那次被 reserveTokens 问题盖住，见下）
- [x] `agents.defaults.compaction.reserveTokens=16384` 强制设置修复（TA/Teacher context overflow 根因），**已在 smoke 上验证生效**（TA 产生真实回复，不再是错误占位文本）
- [x] wandb 集成**已实测验证成功**（新提交方式 `代码解释器=/bin/bash -i /dfs/data/start_tools.sh && /bin/bash -i`，minitest/smoke 默认开启 `USE_WANDB=1`），wandb key 已改走环境变量不再暴露在 run 的 Command 字段里
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load` + `PATCHED_OPD_DIR` PYTHONPATH 注入
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）
- [x] 系统内存 OOM 修复：提高任务提交时申请的系统内存，A800 minitest 实测连续跑过 10 次 `update_weights()` 无 OOM

### 已知限制 / 未解决
- 训练一结束就立刻杀网关，不等模拟循环跑完（smoke、minitest 都复现过，已记录暂不修，见 [`issues_log.md`](issues_log.md) 2026-07-13 条目）
- minitest 共用同一个 checkpoint 路径，`--load` 自动续训会导致后续跑"续训到快完成直接结束"、验证结果失效——已清空 checkpoint，需要注意以后再犯（见 [`issues_log.md`](issues_log.md) 2026-07-13 第三条）
- 网关断连重试修复尚未被干净验证（偶发问题，需要真的撞上才能确认）
- ⚠️ **用户待办**：`openclaw_rl` wandb 项目当前是 Public，需要手动改回 Private；已暴露两次的 WANDB_API_KEY 需要去 wandb 网站撤销重新生成
- `appendSystemContext` 标记是否会污染 OpenClaw 自己持久化的对话历史，待更长多轮对话验证
- context-summarization 内部调用是否触发 `before_prompt_build`，待验证（决定是否顺带解决 main turn 误标问题）

### 下一步
1. 用户手动处理 wandb 项目权限 + 撤销重新生成 key（见上）
2. 清空 checkpoint 后重新提交 minitest，确认真正从头跑的完整流水线（INIT 数据完整 + 无 OOM），顺带确认新 run 的 Command 字段不再有 key
3. 提交 8 GPU 正式 Table 3 训练（`train_with_services.sh` 已就绪，需申请更高系统内存，wandb 集成已验证可用）
4. 8GPU 正式训练建议固定用同一种 GPU 架构（H20 或 A800 二选一，不要跟其他方法/基线的对比数字混用不同硬件）

### 未验证
- [ ] `run_one_persona()` 网关断连重试修复（偶发问题，待真实撞上验证）
- [ ] minitest 5 GPU 完整跑通（checkpoint 已清空，需重新提交，确认 INIT 数据完整 + 无 OOM）
- [ ] 新 run 的 Command 字段确认不再包含 `--wandb-key`
- [ ] `appendSystemContext` 标记多轮对话下的稳定性
- [ ] 8 GPU 正式 Table 3 训练

---

## 当前状态（2026-07-13）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测，flash-attn/APEX/TE/flashinfer 非 H20 专属编译）
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次起 gateway 前强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`（`contextWindow=32768`/`maxTokens=4096`），不再用静态 header
- [x] `scripts/prepare_patched_rl_training_headers.sh`：`rl-training-headers` 插件 `appendSystemContext` 版本，**已用真实 GPU 数据验证生效**（真实动态 `session_id`/`turn_type` 标记到达请求正文）
- [x] `scripts/prepare_patched_openclaw_opd.sh`：解析标记 + 转发前清理，**已用真实 GPU 数据验证生效**
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本已统一接入上述所有 workaround
- [x] `run_one_persona()` 网关断连重试修复，代码已就绪，**尚未被干净验证**（07-11 那次被 reserveTokens 问题盖住，见下）
- [x] `agents.defaults.compaction.reserveTokens=16384` 强制设置修复（TA/Teacher context overflow 根因），**已在 smoke 上验证生效**（TA 产生真实回复，不再是错误占位文本）
- [x] wandb 集成**已实测验证成功**（新提交方式 `代码解释器=/bin/bash -i /dfs/data/start_tools.sh && /bin/bash -i`，minitest/smoke 默认开启 `USE_WANDB=1`），wandb key 已改走环境变量不再暴露在 run 的 Command 字段里
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load` + `PATCHED_OPD_DIR` PYTHONPATH 注入
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）
- [x] 系统内存 OOM 修复：提高任务提交时申请的系统内存，A800 minitest 实测连续跑过 10 次 `update_weights()` 无 OOM

### 已知限制 / 未解决
- 训练一结束就立刻杀网关，不等模拟循环跑完（smoke、minitest 都复现过，已记录暂不修，见 [`issues_log.md`](issues_log.md) 2026-07-13 条目）
- minitest 共用同一个 checkpoint 路径，`--load` 自动续训会导致后续跑"续训到快完成直接结束"、验证结果失效——已清空 checkpoint，需要注意以后再犯（见 [`issues_log.md`](issues_log.md) 2026-07-13 第三条）
- 网关断连重试修复尚未被干净验证（偶发问题，需要真的撞上才能确认）
- ⚠️ **用户待办**：`openclaw_rl` wandb 项目当前是 Public，需要手动改回 Private；已暴露两次的 WANDB_API_KEY 需要去 wandb 网站撤销重新生成
- `appendSystemContext` 标记是否会污染 OpenClaw 自己持久化的对话历史，待更长多轮对话验证
- context-summarization 内部调用是否触发 `before_prompt_build`，待验证（决定是否顺带解决 main turn 误标问题）

### 下一步
1. 用户手动处理 wandb 项目权限 + 撤销重新生成 key（见上）
2. 清空 checkpoint 后重新提交 minitest，确认真正从头跑的完整流水线（INIT 数据完整 + 无 OOM），顺带确认新 run 的 Command 字段不再有 key
3. 提交 8 GPU 正式 Table 3 训练（`train_with_services.sh` 已就绪，需申请更高系统内存，wandb 集成已验证可用）
4. 8GPU 正式训练建议固定用同一种 GPU 架构（H20 或 A800 二选一，不要跟其他方法/基线的对比数字混用不同硬件）

### 未验证
- [ ] `run_one_persona()` 网关断连重试修复（偶发问题，待真实撞上验证）
- [ ] minitest 5 GPU 完整跑通（checkpoint 已清空，需重新提交，确认 INIT 数据完整 + 无 OOM）
- [ ] 新 run 的 Command 字段确认不再包含 `--wandb-key`
- [ ] `appendSystemContext` 标记多轮对话下的稳定性
- [ ] 8 GPU 正式 Table 3 训练

---

## 历史状态（2026-07-10，已被 7/13 结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖（A800/H20 均已实测，flash-attn/APEX/TE/flashinfer 非 H20 专属编译）
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次起 gateway 前强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`（`contextWindow=32768`/`maxTokens=4096`），不再用静态 header
- [x] `scripts/prepare_patched_rl_training_headers.sh`：`rl-training-headers` 插件 `appendSystemContext` 版本，**已用真实 GPU 数据验证生效**（真实动态 `session_id`/`turn_type` 标记到达请求正文）
- [x] `scripts/prepare_patched_openclaw_opd.sh`：解析标记 + 转发前清理，**已用真实 GPU 数据验证生效**
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本已统一接入上述所有 workaround
- [x] `run_one_persona()` 网关断连重试修复（`minitest_train_with_services.sh` / `train_with_services.sh`），代码已就绪，待真实 job 验证
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load` + `PATCHED_OPD_DIR` PYTHONPATH 注入
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）
- [x] 系统内存 OOM 修复：提高任务提交时申请的系统内存，A800 minitest 实测连续跑过 10 次 `update_weights()` 无 OOM

### 已知限制 / 未解决
- INIT 阶段网关断连重试修复尚未在真实 job 上验证
- `appendSystemContext` 标记是否会污染 OpenClaw 自己持久化的对话历史，待更长多轮对话验证
- context-summarization 内部调用是否触发 `before_prompt_build`，待验证（决定是否顺带解决 main turn 误标问题）

### 下一步
1. 停掉当前 A800 minitest（已达成验证目的），拉最新代码重新提交，验证网关断连重试修复
2. minitest 完整跑通（INIT 数据完整 + 无 OOM）后提交 8 GPU 正式 Table 3 训练（`train_with_services.sh` 已就绪，需申请更高系统内存）
3. 8GPU 正式训练建议固定用同一种 GPU 架构（H20 或 A800 二选一，不要跟其他方法/基线的对比数字混用不同硬件）

### 未验证
- [ ] `run_one_persona()` 网关断连重试修复
- [ ] minitest 5 GPU 完整跑通
- [ ] `appendSystemContext` 标记多轮对话下的稳定性
- [ ] 8 GPU 正式 Table 3 训练

---

## 历史状态（2026-07-09，已被 7/10 A800 实测结果取代）

### 已就绪
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次起 gateway 前强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`（`contextWindow=32768`/`maxTokens=4096`），不再用静态 header
- [x] `scripts/prepare_patched_rl_training_headers.sh`：`rl-training-headers` 插件 `appendSystemContext` 版本，**已用真实 GPU 数据验证生效**（真实动态 `session_id`/`turn_type` 标记到达请求正文）
- [x] `scripts/prepare_patched_openclaw_opd.sh`：解析标记 + 转发前清理，**已用真实 GPU 数据验证生效**
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本已统一接入上述所有 workaround
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load` + `PATCHED_OPD_DIR` PYTHONPATH 注入
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）

### 已知限制 / 未解决
- 训练进行到中途崩溃：节点系统内存 OOM，发生在 `update_weights()` 权重同步阶段（7/10 已解决，见上）
- `appendSystemContext` 标记是否会污染 OpenClaw 自己持久化的对话历史，待更长多轮对话验证
- context-summarization 内部调用是否触发 `before_prompt_build`，待验证（决定是否顺带解决 main turn 误标问题）

### 下一步
1. minitest 提交时系统内存申请提高到 192GB（不够再升 256GB），重新提交验证 OOM 是否解决
2. 若解决，8GPU 正式提交同步申请更高系统内存（+ 视情况提高 CPU 核数）
3. minitest 完整跑通后提交 8 GPU 正式 Table 3 训练（`train_with_services.sh` 已就绪）

### 未验证
- [ ] minitest 5 GPU 完整跑通
- [ ] `appendSystemContext` 标记多轮对话下的稳定性
- [ ] 8 GPU 正式 Table 3 训练

---

## 历史状态（2026-07-08，已被 7/9 实测结果和机制切换取代）

### 已就绪
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次 `launch_openclaw_gateway()` 强制设置）
- [x] `models.providers.sglang`：显式声明 `models[]`（`contextWindow`/`maxTokens`）+ 静态 `headers.X-Turn-Type=main`——**已实测确认生效**（`[main]` 出现在 training.log 里，训练队列真实累积样本）
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本统一用真实 `openclaw gateway run`
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load` + smoke `PRM_MAX_NEW_TOKENS`/`PATCHED_OPD_DIR` PYTHONPATH 注入
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）

### 已知限制 / 未解决
- `scripts/prepare_patched_openclaw_opd.sh` 的 `X-Session-Id` Runtime 行解析**实测未生效**（一直是 `unknown`），已加调试日志（`[SESSION-ID-DEBUG]`），原因待明天的 smoke 结果确认
- `rl-training-headers` 插件在当前 OpenClaw（2026.6.9）里端到端不生效，已放弃依赖
- smoke（context=8192）下真实 agent 多轮对话可能撞 context overflow；minitest/8GPU（context=32768）预期不受影响
- 提交 job 前务必确认没有残留的手动测试进程（`openclaw gateway run`、mock server 等）占用端口/内存，否则可能触发 cgroup OOM 级联杀掉整个 job

### 下一步
1. 明天查 smoke 的 `[SESSION-ID-DEBUG]` 输出，定位 `X-Session-Id` 解析失败的真实原因并修复
2. 确认训练队列能稳定累积样本（`X-Turn-Type` 这部分已验证，只差 session_id）
3. header workaround 全部验证通过后传播到 minitest/train_with_services.sh（目前只在 smoke 里生效）
4. 提交 8 GPU 正式 Table 3 训练

### 未验证
- [ ] `X-Session-Id` 解析为什么在真实链路里不匹配（已加调试日志）
- [ ] minitest 5 GPU 完整跑通
- [ ] 8 GPU 正式 Table 3 训练

---

## 历史状态（2026-07-07，已被 7/8 实测结果部分取代）

### 已就绪（7/7 时点）
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（每次 `launch_openclaw_gateway()` 强制设置，不依赖跨环境持久化）
- [x] `models.providers.sglang`：显式声明 `models[]`（`contextWindow`/`maxTokens`）+ 静态 `headers.X-Turn-Type=main`（`launch_openclaw_gateway()` 里生成）
- [x] `scripts/prepare_patched_openclaw_opd.sh`：`X-Session-Id` 从 Runtime 行解析的兜底补丁，官方 `openclaw-opd/` 不动，`PATCHED_OPD_DIR` 接入训练 job `PYTHONPATH`
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本统一用真实 `openclaw gateway run`
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load` + smoke `PRM_MAX_NEW_TOKENS`/`PATCHED_OPD_DIR` PYTHONPATH 注入
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）

> 7/7 这些改动当时都还没实测；7/8 实测确认 `X-Turn-Type` 部分生效，`X-Session-Id` 部分不生效，需要继续排查。

---

## 历史状态（2026-07-06，已被 7/7 header workaround 取代）

### 已就绪（7/6 时点）
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] `~/.openclaw/openclaw.json`：`gateway.http.endpoints.chatCompletions.enabled=true`（本次新增，是 18789 端点的真正开关）
- [x] `scripts/prepare_openclaw_test_scripts.sh`：`openclaw-test/*.py` 的 `model` 字段兼容补丁（`"default"` → `"openclaw/default"`），官方目录不动
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh` / `train_with_services.sh` 三脚本统一用真实 `openclaw gateway run`（commit `ea19053`，`rl_gateway_proxy.py` 已删除）
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`：断点续训 `--load "${SAVE_CKPT}"` + smoke `PRM_MAX_NEW_TOKENS` 修复
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768）

> 7/6 认为"插件+header"是正确方向，7/7 实测证实这套机制在当前 OpenClaw 版本里端到端不生效，已改用 headers 静态配置 + Runtime 行解析。

---

## 历史状态（2026-07-03，已被 7/6 gateway 架构修正取代）

### 已就绪
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist（`/dfs/data/models/Qwen3-4B-Thinking-2507-torch-dist`）
- [x] `openclaw.json` 配置正确（`rl-training-headers` enabled、sglang provider 已验证）
- [x] `scripts/rl_gateway_proxy.py`：替代 `openclaw gateway run`，注入 `X-Session-Id`/`X-Turn-Type: main`（commit `eafd060`）
- [x] `scripts/smoke_train_with_services.sh` / `minitest_train_with_services.sh`（已换用 proxy，commit `eafd060`）
- [x] `scripts/train_with_services.sh`（18789 URL 正确，启动顺序 bug 待修）
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh` / smoke / minitest launcher 脚本
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 32768，2026-07-03 修复）
- [x] conda env `/dfs/data/envs/openclaw-rl` 已有 fastapi / uvicorn / httpx

### 下一步
1. **重启 Simulator**（如 context=32768 版本尚未生效）：`launch_simulator.sh`
2. **重提交 smoke（4 GPU）**：`scripts/smoke_train_with_services.sh`，观察 `training.log` 出现 `combine samples: 16/16` → iter 1 即通过
3. **smoke 通过后**：修复 `train_with_services.sh` 启动顺序（先等 30000 再起 proxy）→ 提交 8 GPU 正式训练

### 未验证
- [ ] smoke 重跑（proxy 替换后，`X-Session-Id`/`X-Turn-Type: main` 注入是否使训练队列正常累积）
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
