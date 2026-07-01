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

## 当前状态（2026-06-30 晚）

### 已就绪
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] 外部 Qwen3-32B Simulator（vLLM，`simulator.env` 地址 `10.254.107.247`，HTTP 200）
- [x] OpenClaw + `openclaw.json` + rl-training-headers
- [x] `scripts/smoke_train_with_services.sh`（**已修复：nc→curl；GATEWAY_URL→30000**）
- [x] `scripts/train_with_services.sh`（**已修复：nc→curl；GATEWAY_URL→30000**）
- [x] `scripts/run_openclaw_topk_select_modelfactory.sh`
- [x] `scripts/smoke_run_qwen3_4b_openclaw_topk_select.sh`
- [x] `scripts/check_convergence.py`

### 未验证 / 阻塞
- [ ] **4 GPU smoke `✅ SMOKE PASSED`**（第 N 次提交中，三个已知问题已修复）
- [ ] 8 GPU 正式 Table 3 训练（等 smoke 通过后）

### 下一步
1. 等当前 smoke job 结果，查 `logs/smoke_*/` 日志
2. smoke 通过后提交 8 GPU 正式训练

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
