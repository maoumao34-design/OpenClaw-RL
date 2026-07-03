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

### Joint 训练实现修正（重要）

对照论文 Appendix A.1 + 官方源码逐字审查后，发现 `train_with_services.sh` 与论文 Joint 设计存在结构性偏差：

- **原实现**：每轮 `rm -rf homework homework1 homework2` → 顺序跑 Student → TA → Teacher
- **论文要求**：一次性 INIT（顺序）→ 之后三 Simulator **同时**运行（homework1/2 固定不清空）

**关键区别**：TA 在原实现中每轮读的是当轮 Student 的输出（动态），论文中 TA 读的是 INIT 阶段固定下来的内容；`ensure_homework_dir` 只在目录不存在时复制，原实现每轮清空导致该机制失效。

**修复：** 重写两个脚本的模拟部分 → [`train_with_services.sh`](../scripts/train_with_services.sh) / [`smoke_train_with_services.sh`](../scripts/smoke_train_with_services.sh)

新结构：
1. **INIT 阶段**（一次性）：`run_init_phase()` 顺序跑 Student → TA → Teacher，各 `SESSION_LIMIT=72` 题，建立 `homework1/` `homework2/`
2. **Joint 阶段**（循环）：`run_joint_round()` 三 Simulator 并行（`&` + `wait`），各自操作独立目录无冲突

### 训练脚本更正：basic combine → topk-select（关键）

通过对照论文 Appendix A.1 参数与官方代码库，发现我们之前使用的脚本对应错误：

- **之前**：`openclaw-combine/run_qwen3_4b_openclaw_combine.sh`（basic combine，m=1，无 k）
- **正确**：`openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh`（k=4, m=3, seq-optimal）

**根据依据：**
- 论文 Appendix A.1 明确写：k=4，每样本 hint 数 m=3
- Table 5（k 消融）k=4 → avg **10.3** = Table 3 主结果 10.3，两者完全吻合
- basic combine 的 PRM_M 默认为 1，且无 `--distill-topk` 参数，无法实现 k=4 行为
- 只有 topk-select 脚本含 `OPENCLAW_TOPK_K=4`、`PRM_M=3`、`sequence_optimal` 三个参数

**修复：**
- 新建 [`scripts/run_openclaw_topk_select_modelfactory.sh`](../scripts/run_openclaw_topk_select_modelfactory.sh)（官方 topk-select 脚本的 modelfactory patch）
- 新建 [`scripts/smoke_run_qwen3_4b_openclaw_topk_select.sh`](../scripts/smoke_run_qwen3_4b_openclaw_topk_select.sh)（4 GPU smoke 配置，m=1 验证流通）
- 更新 [`scripts/train_with_services.sh`](../scripts/train_with_services.sh)：调用 topk-select launcher
- 更新 [`scripts/smoke_train_with_services.sh`](../scripts/smoke_train_with_services.sh)：GPU 3→4（topk-select 必须有 PRM Teacher GPU），调用新 launcher
- 更新 [`CLAUDE.md`](../../CLAUDE.md)：修正脚本对应关系

**smoke GPU 变化：** 3 → 4（topk-select 强制 `OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron`，PRM Teacher 占 1 张 GPU，无法省略）

### 论文深度理解 + 源码核查

**完成内容：**
- 确认 PRM Teacher 三重冻结保证（`no_load_optim=True`、init 后 `clear_memory(); return`、只调用 `compute_prm_teacher_log_probs()` 做 forward，`update_weights()` 永不调用）
- 从官方 `openclaw-test/` 源码确认三角色区分机制：独立系统 prompt + 独立 session ID（`student-hw-*` / `ta-grade-*` / `teacher-comment-*`）+ homework 目录链（`homework/` → `homework1/` → `homework2/`）
- 整理三角色适应类型（Suppress / Amplify / Add）及各自收敛速度差异（TA=8.2 最快、Teacher=11.4 次之、Student=11.6 最慢）→ [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)
- 从官方 `openclaw-opd/openclaw_opd_api_server.py` 确认 PRM SGLang 双职能：`_query_judge_once()` 单次 LLM 调用同时产出 `\boxed{±1}` 评估分和 `[HINT_START]...[HINT_END]` hint 候选，m=3 次后 `_select_best_hint()` 选出最优 h*
- 更新 GPU 布局表格：PRM Judge 说明补充双职能，PRM Teacher 说明明确为 hint-conditioned teacher log-probs（非 old log-probs）→ [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)

**主要问题：**

- **Modelfactory job 提交系统异常**（系统维护期间）：bash/python job 均无输出、无日志、脚本未执行；直接在 workspace terminal 运行完全正常 → 确认是平台层面问题，非代码问题
- 排查过程排除了：CRLF 换行、路径解析、simulator.env 空格、openclaw token、脚本内容等所有代码侧原因
- **根因**：modelfactory 系统维护，维护期间 job 提交静默失败

**处理方式：** 维护结束后改用 **workspace 直接运行**（申请 GPU 后 `bash scripts/smoke_train_with_services.sh`）；后续大训练 job 恢复后再试提交

---

## 2026-06-30

**目标：** 论文深度理解收尾，核查 Actor / Rollout 完整职责

**完成内容：**

### GPU 布局表格精简 + Actor / Rollout 职责核查

- 精简 GPU 布局表格 PRM 两行：核心职责留表格，实现细节移至下方说明段落，提升可读性
- 从 `slime/slime/backends/megatron_utils/actor.py`（`train_actor()`）+ `openclaw-combine/openclaw_topk_select_loss.py` 核查 Actor 和 Rollout 完整职责：
  - **Actor**：不只是梯度更新，每个 rollout 依次执行：① ref forward → `ref_log_probs`（KL 用）；② old_actor forward → `log_probs` + `topk_log_probs`/`topk_indices`（GRPO ratio + OPD S_i 用）；③ 接收 PRM Teacher 传来的 teacher log-probs；④ `compute_advantages_and_returns`；⑤ `train()`（backward + optimizer step）；⑥ 权重同步给 Rollout
  - **Rollout**：topk-select 模式下只生成 token（三处 loss 文件均有硬断言 `assert not use_rollout_logprobs`，log-probs 全部由 Actor Megatron 重算保证精度）
- 更新 GPU 布局表格 Actor / Rollout 两行描述 → [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)

### OPD 信号判别机制 + 三方法对比

- 确认 OPD 信号逐 turn 独立判断：每次 Simulator 回复后 PRM Judge 立即投票（不跨 turn 累积）
- 从官方代码确认三方法（GRPO / OPD / Hybrid RL）的兜底机制：
  - **GRPO**：`session_effective` 计数 + 强制保留机制——若一个 session 全部 score=0，至少保留一个有后继状态的 turn（`exclude=False`）
  - **OPD**：无兜底，`if not opd_result.get("accepted"): continue` 直接跳过整 session
  - **Hybrid RL**：无 GRPO 兜底，但若 GRPO 全 0 而 OPD 有 accepted turn → OPD 路径（reward=0.0）接管；两路均失败才丢弃 session
- 从 `openclaw_combine_api_server.py` 确认 Hybrid RL 三路 dispatch：① `opd_accepted + has_valid_rl` → GRPO+OPD；② `opd_accepted only` → OPD（reward=0.0）；③ `has_valid_rl only` → GRPO only
- 记录"三种方法信号判别对比"表格 → [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)

### GRPO 组定义与 Advantage 函数实现（源码核查）

- 从 `slime/slime/ray/rollout.py` + 训练脚本确认 Personal Agent GRPO 实现与标准 GRPO 有根本差异：
  - 标准 GRPO：同一 prompt N 次采样 → 组内 (r-mean)/std → 相对 advantage；OpenClaw：**`--n-samples-per-prompt 1`（无组内比较）+ `--disable-rewards-normalization`（禁用所有归一化）**
  - 每个 turn sample 拿到唯一 `group_index`（`next(self._group_counter)`）→ 不存在多 sample 共享同一 group
  - `_drop_constant_reward_groups` 在 `rewards_normalization=False` 时直接跳过（第 407 行短路）
  - **Advantage = raw PRM 分数 ∈ {+1, -1, 0}**，由 `get_grpo_returns` 广播到 response 所有 token
  - 本质是 REINFORCE + PPO-clip，"GRPO" 只是框架代称；PRM 替代了"组内比较"的角色，直接给出绝对信号
- 修正 `paper_understanding.md` 第 54、75 行（原误写"按 step index 分组标准化"）→ [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)

### OpenClaw 调用架构梳理（32B Simulator ↔ 4B Policy 交互机制）

- 从 `openclaw-test/student_chat.py` + `openclaw-combine/openclaw_combine_select_api_server.py` 逐层读源码，完整还原 Simulator 与 Policy 的交互链路：
  - **Simulator（Qwen3-32B）**：`generate_student_message()` 每轮调用 32B 生成下一条学生消息；`send_to_openclaw()` POST 到 18789，只传当前单条消息 + session ID，历史由 gateway 维护
  - **OpenClaw gateway（18789）**：每次请求到来时：① 用 session ID 还原完整对话历史；② 转发给 SGLang Policy（4B，port 30000）生成回复；③ 把当前消息作为上一 turn 的 next_state，trigger `_fire_opd_task(T-1)` → 异步发送给 PRM Judge 打分；④ 把当前 turn_data 写入 pending
  - **"延迟一拍" buffering**：gateway 无法在 turn T 当时打分（next_state 还没到），等 turn T+1 消息到达才能评 turn T → 打分与对话生成完全并行，零延迟
  - **turn_type routing**：`"main"` 写入 pending 产生 Sample；`"side"` 只转发不录数据（评估用）
- 更新 `paper_understanding.md`：新增"OpenClaw 调用架构"小节，含时序图 + 三设计要点 + 端口分工表 → [`paper_understanding.md`](openclaw-rl/docs/paper_understanding.md)

### 论文深度理解补充（下午）

- 确认 Mem0 / Cognee 是"记忆 + 上下文注入"范式（非训练，模型权重不变），与 RL 方法形成对比；补充至 [`paper_reproduction_scope.md`](openclaw-rl/docs/paper_reproduction_scope.md) Phase 5
- 更新 `paper_understanding.md` 十四、十五节：修复编号重复 bug；十五复现难点恢复原有 5 条并补充 4 条新内容（仓库边界识别、Hybrid 信号融合、Personal vs General GRPO 差异、General Agent 云环境规模）

### 4 GPU smoke 调试（下午）

H20 资源释放，开始正式提交 smoke job，连续排查三个问题：

**问题 1：Simulator 旧 IP 残留**
- 现象：smoke 日志持续报 `Connection timed out`，检测地址为旧 IP `10.254.28.141`
- 根因：job 提交时 `simulator.env.example` 还是旧地址，脚本回退读 example → 使用旧 IP
- 修复：`simulator.env` 和 `simulator.env.example` 均已更新为 `10.254.107.247`，重新提交即可

**问题 2：`nc` 未安装，port 30000 检测永远失败**
- 现象：训练栈（SGLang + Megatron + RL proxy）实际已正常启动（日志显示 `Uvicorn running on 0.0.0.0:30000`），但 smoke 脚本卡在"等待 RL training proxy"超过 370s
- 根因：`wait_for_port` 函数使用 `nc -z localhost ${port}`，但环境中 `nc` 未安装，命令始终返回失败
- 修复：`nc -z` → `curl -s --max-time 5 "http://localhost:${port}/"` → [`scripts/smoke_train_with_services.sh`](../scripts/smoke_train_with_services.sh) / [`train_with_services.sh`](../scripts/train_with_services.sh)；已 push GitHub `6543125`

**问题 3：`OPENCLAW_GATEWAY_URL` 指向 OpenClaw CLI（18789）而非 RL proxy（30000）**
- 现象：port 30000 检测通过后，student_chat.py 报 `404 Not Found: http://localhost:18789/v1/chat/completions`
- 根因：smoke 脚本将 `OPENCLAW_GATEWAY_URL` 设为 OpenClaw CLI（18789），而 CLI 不暴露 `/v1/chat/completions`；该 endpoint 只在 Python RL proxy（port 30000，`openclaw_combine_select_api_server.py`）上存在
- 修复：`OPENCLAW_GATEWAY_URL=http://localhost:18789` → `http://localhost:30000`；已 push GitHub `482fdc6`

**当前状态：** 三个问题均已修复并 push，正在提交下一次 smoke job 验证。

---

## 2026-07-01

**目标：** 完成 4 GPU smoke 测试，打通端到端流程

**完成内容：**

### 积压文件同步（`98273cb`）

之前多次 commit 均为"定点提交"（只提当次改动文件），导致新建脚本和文档改动未入库：
- 新建脚本 `run_openclaw_topk_select_modelfactory.sh`、`smoke_run_qwen3_4b_openclaw_topk_select.sh` 从未 `git add`
- `paper_understanding.md`、`work_log.md`、`paper_reproduction_scope.md` 等文档改动积压

一次性补提 15 个文件（+1127 行，-382 行），覆盖至前一日所有工作。

**根因归纳**：workspace 无法 push GitHub，本地负责 push，但每次 push 只 stage 了当次操作文件；已记入长期记忆，后续每次 push 前先 `git status` 检查全部积压改动。

### 4 GPU Smoke 调试（续）

继续昨日 smoke 调试，新增四个问题：

**问题 4：`REF_LOAD` / `PRM_TEACHER_LOAD` 使用 HF 路径，缺少 bridge mode**
- 现象：`AssertionError: Only bridge mode is supported for loading HF checkpoint`（`slime/backends/megatron_utils/checkpoint.py:134`）
- 根因：之前 commit `3b3e7e0` 误将 `REF_LOAD` 设为 HF 路径；官方 topk-select 脚本使用 `_torch_dist` 格式。Megatron 检测到 HF checkpoint 后要求 `--megatron-to-hf-mode bridge`，但该参数在 topk-select 脚本中不存在
- 修复：`REF_LOAD` / `PRM_TEACHER_LOAD` 改用 `POLICY_TORCH_DIST`（`/dfs/data/models/Qwen3-4B-Thinking-2507-torch-dist`）；同时修正 `POLICY_TORCH_DIST` 路径拼写（原错误路径 `torch_dist/qwen3-4b-thinking-2507`）→ 已 push `672d9a7`，**两个脚本均更新（smoke + 正式）**

**问题 5：Worker 节点 RAM 不足（64 GB）**
- 现象：`ray.exceptions.OutOfMemoryError`，节点内存 `62.x GB / 64.00 GB (0.97)`
- 根因：Ray actor 调度到 64 GB RAM 节点（非 workspace 头节点的 282 GB）；torch_dist checkpoint 含 Adam optimizer states，每个 Megatron actor 占用 ~24 GB；Actor + PRM Teacher 两个 Megatron actor 同时加载 = ~48 GB，加系统进程超过 95% 阈值
- 说明：8 GPU 正式跑 TP=4，每个 actor 只需加载 1/4 模型 = ~6 GB，无此问题；smoke TP=1 加载完整 checkpoint 是独有现象
- 修复：申请 ≥128 GB RAM 节点提交 job（内存比 GPU 好申请）

**问题 6：评估阶段 401 Unauthorized**
- 现象：`student_chat.py` 调 `http://localhost:30000/v1/chat/completions` 报 401
- 根因：`run_smoke_chat()` 传的是 `OPENCLAW_GATEWAY_TOKEN`（OpenClaw CLI 的 token），而 RL proxy（port 30000）使用 `SGLANG_API_KEY` 鉴权，两者不匹配
- 修复：`run_smoke_chat()` 中 `OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}"` → `OPENCLAW_GATEWAY_TOKEN="${SGLANG_API_KEY}"` → 已 push `5aa3c74`

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

为降低 8 GPU 正式训练风险，新建 5 GPU（2+1+1+1）前置验证脚本，在完整论文配置下跑 300 个 rollout（~18 步）验证整条流水线。

**新增脚本：**
- [`scripts/minitest_run_qwen3_4b_openclaw_topk_select.sh`](../scripts/minitest_run_qwen3_4b_openclaw_topk_select.sh)：5 GPU 训练 launcher（MINITEST_PROFILE=1，Actor×2 TP=2 / Rollout×1 / PRM×1 / Teacher×1）
- [`scripts/minitest_train_with_services.sh`](../scripts/minitest_train_with_services.sh)：5 GPU 完整流水线（含 Simulator 连通 + OpenClaw + 模拟循环 + 收敛检测），入口脚本

**修改脚本：**
- [`scripts/run_openclaw_topk_select_modelfactory.sh`](../scripts/run_openclaw_topk_select_modelfactory.sh)：新增 `MINITEST_PROFILE=1` 分支，sed 补丁：`TP 4→2`、`rollout-gpus 2→1`、`SGLang TP "2"→"1"`、`num-rollout→300`

**与 8GPU 正式版的唯一差异：**

| 参数 | 正式（8 GPU）| Pre-test（5 GPU）|
|------|------------|----------------|
| `tensor-model-parallel-size` | 4 | 2 |
| `rollout-num-gpus-per-engine` | 2 | 1 |
| `export TP` | `"2"` | `"1"` |
| `num-rollout` | 100000000 | 300（~18 步）|
| context / batch / m / k | 32768 / 16 / 3 / 4 | **同正式**（不变）|

**说明：** Pre-test 通过即说明完整流水线（4B model、torch_dist 加载、PRM Teacher、Simulator 调用链、收敛检测）均无问题，可直接提交 8 GPU job。

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

**完成内容：**

### Pre-test 结果审查

对照论文逐项核查（本地代码 + pre-test 日志）：

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 输出文件只记录第 1 轮回复（`turn==0`） | ✅ | 三个脚本全部确认 |
| Simulator 模型 Qwen3-32B | ✅ | `EXTERNAL_MODEL=qwen3-32b` |
| Policy / PRM 模型 Qwen3-4B-Thinking-2507 | ✅ | run script 确认 |
| k=4, m=3, sequence_optimal | ✅ | run script 默认值全部对齐 |
| W_RL=1.0, W_OPD=1.0, clip=1.0 | ✅ | run script 确认 |
| 模拟循环无限运行 | ✅ | commit `5833f51` 已修复 |
| 收敛判断 rule-based | ✅ | `check_convergence.py` 实现正确 |
| `_all.txt` 跨轮追加 | ✅ | `cat >>` 模式正确 |
| 训练参数（TP=2 minitest / TP=4 生产，rollout=1/2）| ✅ | MINITEST_PROFILE sed 补丁正确 |

### 发现问题：Pre-test 0 训练步骤（根因确认，已修复）

**现象：** `training.log` 持续输出 `waiting for combine samples: 0/16, queue=0`，整个 pre-test 无 checkpoint 产生；Policy 回复中出现 "I don't have access to your local file system"（无工具访问）。

**根因：commit `482fdc6` 架构错误。**

当时为修复 smoke 的 `404`，将 `OPENCLAW_GATEWAY_URL` 从 `18789` 改为 `30000`，结论写的是"18789 不暴露 `/v1/chat/completions`"——这个判断是**错的**：18789 是 OpenClaw gateway，确实暴露该 endpoint；当时 404 是因为 OpenClaw 尚未完整配置（并非 endpoint 不存在）。

改为 30000 后的实际影响：

| 影响 | 机制 |
|------|------|
| 0 训练数据 | 绕过 OpenClaw → rl-training-headers 不注入 `X-Turn-Type: main` → RL proxy 默认 "side" → 训练队列永远为空 |
| Policy 无文件访问 | 绕过 OpenClaw → Policy 无 workspace 工具 → 只能说"I don't have access" |

**架构验证（2026-07-03，所有组件已全部核查）：**

| 组件 | 状态 |
|------|------|
| `rl-training-headers` 插件 | `enabled`（`stock:rl-training-headers/index.js` v1.0.0）|
| `sglang-provider` 插件 | `enabled` |
| sglang `baseUrl` | `http://127.0.0.1:30000/v1`（精确对应 RL proxy）|
| sglang `apiKey` | `openclaw-rl-key`（与 `SGLANG_API_KEY` 一致）|
| 默认 model | `sglang/qwen3-4b`（与官方 `SERVED_MODEL_NAME="qwen3-4b"` 一致）|
| gateway token 读取路径 | `gateway.auth.token = b125280f...`（train 脚本 python 读取逻辑正确）|
| workspace 路径 | `/root/.openclaw/workspace`（与脚本 `WORKSPACE=${HOME}/.openclaw/workspace` 一致）|

**完整链路：**
```
student_chat.py
  → POST http://localhost:18789/v1/chat/completions  (model=default)
  → OpenClaw gateway (primary = sglang/qwen3-4b)
  → rl-training-headers 注入 X-Turn-Type:main + X-Session-Id
  → POST http://127.0.0.1:30000/v1  (RL proxy，看到 X-Turn-Type:main → 写训练队列)
  → SGLang (Policy Qwen3-4B)
```

**修复：** 三个脚本全部将 `OPENCLAW_GATEWAY_URL` 改回 `18789`，并修正 minitest 的 token 错误（`SGLANG_API_KEY` → `OPENCLAW_GATEWAY_TOKEN`）：
- `scripts/train_with_services.sh`
- `scripts/minitest_train_with_services.sh`
- `scripts/smoke_train_with_services.sh`

---

### 发现问题：Simulator context length 不足（已修复）

**现象：** `sim_student.log` 出现反复 400 错误：
```
maximum context length is 16384 tokens. prompt contains at least 16385 input tokens.
```

**根因：** `scripts/launch_simulator.sh` 默认 `MAX_TOKENS=16384`，而 Policy 最大回复长度 8192 token，TA 反馈等详细回复累积 2-3 轮后即超 16384。

**影响评估：**
- 收敛数据**完整**：output 写在 `turn==0`，context overflow 发生在 `turn>=1`，第一轮回复已写入，不影响收敛判断
- 会话完整性**受损**：部分 session 崩溃导致作业文件未写完，TA/Teacher 后续步骤可能读到不完整内容
- 正式训练必须修复

**修复：** `launch_simulator.sh` 默认值 `16384 → 32768`（与 Policy context 对齐）→ commit 本次

**操作要求：** Simulator 需重启以应用新 context 配置

---

## 当前状态（2026-07-03）

### 已就绪
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist（路径：`/dfs/data/models/Qwen3-4B-Thinking-2507-torch-dist`）
- [x] OpenClaw + `openclaw.json` + rl-training-headers（已验证，完整链路通）
- [x] `scripts/smoke_train_with_services.sh`（18789 修复后）
- [x] `scripts/train_with_services.sh`（8 GPU 正式，18789 修复后）
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`（支持 SMOKE_PROFILE / MINITEST_PROFILE / 生产三模式）
- [x] `scripts/smoke_run_qwen3_4b_openclaw_topk_select.sh`
- [x] `scripts/minitest_run_qwen3_4b_openclaw_topk_select.sh` / `minitest_train_with_services.sh`（18789 + token 修复后）
- [x] `scripts/check_convergence.py`
- [x] `scripts/launch_simulator.sh`（context 16384 → **32768**，2026-07-03 修复）

### 待修复（smoke 通过后执行）

**`train_with_services.sh` 启动顺序错误（影响 8 GPU 正式训练）**

当前 step 2 顺序：先起 OpenClaw → 等 18789 → 等 30000。
正确顺序应为：先等 30000 → 再起 OpenClaw → 等 18789。
（同 `smoke_train_with_services.sh` 和 `minitest_train_with_services.sh` 的做法，smoke 里甚至有注释"RL proxy :30000 必须先起来"）
提交 8 GPU 前必须修复，否则 OpenClaw 启动时 sglang provider 连不上 30000，复现原始 404。

### 待完成（下一步）
1. **重启 Simulator**：kill 旧 sglang 进程，`git pull` 后重新执行 `launch_simulator.sh`（新 context=32768）
2. **重提交 smoke（4 GPU）验证**：`scripts/smoke_train_with_services.sh`
   - 观察 `training.log` 是否出现 `combine samples: 16/16` → 训练迭代开始
   - 若 queue 仍为 0，说明 18789 调用链还有问题；若出现 `iter 1`，则修复有效
3. **smoke 通过后**：修复 `train_with_services.sh` 启动顺序（见上）→ 提交 8 GPU 正式训练

### 未验证
- [ ] smoke 重跑（18789 修复后验证训练队列）
- [ ] 8 GPU 正式 Table 3 训练

### 附：pre-test 无 checkpoint 的根因

pre-test 日志显示 `save-interval 5` 但无任何 checkpoint 产生，根因同上：训练队列永远 `0/16`（因 OPENCLAW_GATEWAY_URL=30000 绕过了 rl-training-headers，所有 session 均为 "side"），无训练步骤 → save-interval 永远不触发。

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
