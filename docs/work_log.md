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
- 决定：minitest 重新提交时把系统内存申请提高到 256GB 验证是否解决

**主要问题：**
- ~~smoke/minitest 连续三次...无 traceback、无 OOM 记录...根因未查清~~（已定位，见下）：反复静默崩溃的真实根因是**节点系统内存不足**（128GB 节点被 Megatron actor + rollout engine + PRM 等常驻进程打满），发生在 `update_weights()` 的 `pause_generation` 阶段；此前几次"静默无 traceback"很可能是同一个 OOM 杀在了没有异常捕获的 NCCL 集合通信调用中间，导致其余 rank 卡死，跟这次杀在 `ray.get()` 调用点上（有异常捕获、留下 traceback）是同一类问题的不同表现 → [`issues_log.md`](issues_log.md) 2026-07-09 条目更新

**待验证：**
- minitest 256GB 内存重跑，确认 OOM 是否解决（重新提交中）
- 若解决，8GPU 正式提交同步申请 256GB+ 系统内存
- `appendSystemContext` 标记会不会污染 OpenClaw 自己持久化的多轮对话历史（真实 GPU 链路里还未观察到异常，需要更长多轮对话验证）
- context-summarization 内部调用是否触发 `before_prompt_build`（决定 Task 摘要污染问题是否顺带解决）

---

## 当前状态（2026-07-09）

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
- **训练进行到中途崩溃，根因已确认**：节点系统内存 OOM（128GB 节点打满，非 GPU 显存、非 NCCL），发生在 `update_weights()` 权重同步阶段；跟今天的 header 机制改动无关；已决定重新提交时把系统内存申请提到 256GB，待验证是否解决，见上方「主要问题」
- `appendSystemContext` 标记是否会污染 OpenClaw 自己持久化的对话历史，待更长多轮对话验证
- context-summarization 内部调用是否触发 `before_prompt_build`，待验证（决定是否顺带解决 main turn 误标问题）

### 下一步
1. minitest 提交时系统内存申请提高到 256GB，重新提交验证 OOM 是否解决
2. 若解决，8GPU 正式提交同步申请更高系统内存
3. minitest 完整跑通后提交 8 GPU 正式 Table 3 训练（`train_with_services.sh` 已就绪）

### 未验证
- [ ] minitest 5 GPU 完整跑通（256GB 内存重跑中，验证 OOM 是否解决）
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
