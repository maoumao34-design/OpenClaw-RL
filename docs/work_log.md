# 复现工作记录

汇报与整体复盘用。技术细节通过各条目链接查阅。

## 工作记录规范

**文件分工**

| 文档 | 写什么 |
|------|--------|
| **本文件** (`work_log.md`) | 顶部：唯一的「当前状态」；其下：按日的目标、完成摘要、下一步 |
| **[`status_history.md`](status_history.md)** | 已被后续结果取代的历史状态快照（倒序归档） |
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

**「当前状态」放文件顶部，全文只留一份**（2026-08-28 改，此前是「文末维护」）：`已就绪` / `已知限制` / `下一步` / `未验证` 四段。**状态被新一天取代时，整块剪到 [`status_history.md`](status_history.md) 顶部**、标题改成 `## 历史状态（YYYY-MM-DD，已被 M/D 结果取代）`，再在本文件顶部写新的当前状态——不要在 `work_log.md` 里堆积历史状态。

> 改这条的原因：原先历史状态就地留在各自日期附近、当前状态排在文末，积到 37 块 / 1024 行、占全文 39%，把最新条目和当前状态都埋在了 2000 行以下，每次找最新进展都要翻到底。

**日期以真实提交时间为准**：写完一天的条目、准备续写同一个 `## YYYY-MM-DD` 之前，先跑 `git log -1 --date=format:"%Y-%m-%d %H:%M"` 核对真实日期——会话里的「今天」经常已经跨天（此坑已踩过三次：08-20/08-21、08-21/08-25、08-25/08-26）。

**其它规则**：各专题 doc 文首链回 `[← 工作记录](work_log.md)`；不在此重复贴长日志。

---

## 当前状态（2026-09-01）

### 已就绪
**OpenClaw-RL Separate/Personal Agent Track**（同 08-13，未变）。
**MetaClaw 迁移**：同 08-26（历史状态），另加：[x] **Phase 1 打分逻辑已用真实训练数据核实正确**（4 个升级案例全部无误、反向 0 次、诊断文案具体化生效）；[x] **Traceback 泄漏进 agent 可见反馈的回归已修复**（`_compute_training_verdict` 去掉多余 fallback，合成测试已确认非空转）；[x] **`20260827_163030` 训练的退化机制已定位到 day17 thinking 断崖，链条有三项数据支撑**（格式失败 17/26 vs K=6 的 0/27、thinking 18k→115k 的时间断点、FC 派生样本占权重约 90%）；[x] **消融方案已通过 CLI 查验并实现**（中间轮次改吃本轮最终 checker 结果，`METACLAW_MIDROUND_REWARD` opt-in，默认 `judge` 行为不变）；[x] **硬负分优先级 + 结构性无效工具调用检测（规则 6）已修复**（commit `2944f87`）；[x] **outcome 消融训崩的根因已查清并在源码层面确认**（全负 batch + advantage 退化成原始 reward）；[x] **真正的根因已定位到"用着 GRPO 估计器却一个真正的组都没有"**，`toolcall-rl`/MetaClaw 的分组方式与任务形态已逐字查证；[x] **批级基线方案已出，可行性已在源码层面查证**（slime 自带 `--custom-reward-post-process-path` 钩子，不用碰 `group_index`/队列），**待 CLI 在真实环境查证 5 个点后实现**。

### 已知限制 / 未解决
同 08-26（历史状态），另加：**`20260831_154301` 的 checkpoint 已污染，不能作为后续训练起点**。**批级基线方案尚未实现，也未经真实环境查证**——5 个待查点见迁移文档。**`blend` 模式可能是多余的**：它照搬 toolcall-rl 的奖励合成，但 toolcall-rl 能靠它工作的前提是同时有 8 样本组内归一化，我们只搬了一半；默认关闭、回退成本低，视批级基线结果决定去留。**路线 A（轨迹级样本）仍未实现**，且它也只降低负样本数量、不产生正样本。**day11-15 的 FC 全 0 是能力墙**（K=6 冻结模型同样 0），跟训练信号无关。

### 下一步
1. **OpenClaw-RL 复现**：同 08-17
2. **MetaClaw 迁移**：**等 CLI 在真实环境查证批级基线方案的 5 个点**，通过后实现并跑一轮。**先单独跑批级基线、不叠加 `blend`**（`METACLAW_MIDROUND_REWARD=judge` + 自定义钩子）——这是最干净的对照，只改 advantage 算法、reward 形态完全不变，跟 K=6 那次 judge 模式唯一差别就是基线。主判据是 `batch reward` 不再出现 `0/16`、`grad_norm`/`policy drift` 保持在 K=6 量级，**不是 Acc**。**必须从干净 base 起步。**
3. 其余同 08-17

### 未验证
- [ ] **批级基线方案的 5 个真实环境问题**——`load_function` 的路径格式与 modelfactory 上的放置位置、dummy 样本会不会污染批均值、真实到达钩子的样本数、std≈0 时全 0 advantage 是否可接受、与 `step_wise` estimator 是否冲突
- [ ] **批级基线能否真的阻止发散**——源码层面确认了 advantage 会被中心化（必有正负），但真实训练里够不够稳定未知
- [ ] **中间步骤判官正奖励是不是 thinking 膨胀的上游原因**——原来的消融设计已被证明会因全负 batch 而发散，**这个问题至今没有被干净地回答过**；要等训练能稳定跑起来之后才谈得上重新验证
- [ ] **路线 A 的 token 序列重建与 logprob 拼接是否可行**——原料齐全（逐轮 `response_logprobs` 都存着），但未实测；`rollout_log_probs`/`teacher_log_probs` 都按 `response_length` 严格切片，对不齐就是硬错误
- [ ] **Phase 1 在真实训练环境下的实际效果**——打分正确性已核实，但训练效果层面未回答
- [ ] **Traceback 泄漏修复在真实训练中是否生效**——合成测试通过，需确认真实 `[Previous Feedback]` 里 Traceback 归零
- [ ] **`METACLAW_TRAIN_UNTIL_DAY` 默认关闭时是否真的与当前 `day12` 训练行为完全一致**——用户即将验证，这次改动能不能信任的前提
- [ ] **`done.log` 非追加场景真实触发率**——监控日志已埋点（`_compute_training_verdict` 里 `logger.warning`），只能等真实训练跑起来后观察
- [ ] **`_rerun_segment_official` 额外 subprocess 调用在真实训练节奏下的耗时影响**——本地未测过实际耗时，CLI 判断"可忽略"是基于全量 398 段无写操作迹象的静态扫描，不是真实计时
- [ ] **`_AGENT_PAUSE_MARKERS` 扩展在真实暂停窗口下是否真的挽回了原本会丢的样本**——下一轮训练需要确认日志里出现"pause-retry (matched 'LLM request timed out')"且题目最终计分成功
- [ ] **K=6 冻结实验的结果用官方独立 `metaclaw-bench run` 重新核实**——目前的 Frozen 窗口评测走的是训练自己的 harness，跟官方 bench 不完全同构
- [ ] `metaclaw_migration_20260820_*`（六处修复已合入）完整 30 天跑完后，`Compl.` 是否脱离 0.0%、Acc. 相对**新基线（17.8%）**有没有提升——`--agent` 修复的核心验证点，注意不要再拿旧的 8.1% 做对比
- [ ] `METACLAW_TRAIN_UNTIL_DAY` 设置为具体 K 值时，冻结是否真的生效（`[metaclaw-freeze]` 日志、样本提交数骤降为 0）、dayK 尾部竞态实际丢弃规模
- [ ] "对齐/不对齐基线 Acc. 差异" vs "`plugins.allow` 无条件排除插件"这两个结论之间的矛盾，具体机制是什么（承接 08-18，仍未解开）
- [ ] 官方 MetaClaw Compl. 非零的真实原因（OpenClaw CLI 版本差异 or 官方外层脚本另有处理）——开放问题，不阻塞
- 其余同 08-19（历史状态，见上）

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

### 503 崩溃排查与修复

**完成内容：**
- run `20260717_133740` 出现新问题：Student/TA/Teacher 因 408/503 未捕获异常集体崩溃，Joint 阶段秒结束。排查过程中用户连续指出两处逻辑错误（Joint 并发假设不成立、默认容量长期不够用假设被之前 8 小时 run 零复现证伪），最终定位真正根因：训练每步的 `submission_enabled` 正常暂停机制（`openclaw_combine_select_rollout.py`），这次因两步 rollout 收集异常快（追上了平时被流水线掩盖的约 80 秒固定训练计算耗时）导致暂停窗口首次变得肉眼可见（38秒/115秒），远超官方默认 7 秒重试预算 → [`issues_log.md`](issues_log.md) 2026-07-17 条目
- 修复：三个训练脚本的 `--max-retries` 从默认 3 加到 8（总重试预算 255 秒），只加命令行参数，不改官方 Python 源码
- 重新提交训练（run `20260717_171106`），execution-bias-fix 和 max-retries 两个补丁均已用真实日志确认生效

**待验证：** 这次训练能否稳定跑完整个流程，不再复现 503/顶格截断导致的集体崩溃。

### 训练机制答疑（供实验完成后汇报参考）

**完成内容：**
- 记录训练数据的真实粒度（是 `training.log` 里每一次内部模型生成，不只是控制台可见的 Turn，连从未产出可见回复的顶格截断样本也会被提交）、PRM 评委机制的完整代码实现（RL 打分与 hint 提取两套独立 system prompt 原文、hint 候选两阶段筛选：API server 层初筛+去重排序取 K 个候选、训练侧 seq-optimal 才是真正的重叠度最终选择）、`rollout_batch_size=16` 对应 wandb 一个 step → [`paper_understanding.md`](paper_understanding.md) 信号一/信号二小节补充

---

## 2026-07-20

**目标：** 排查上周五训练（run `20260717_171106`）跑 8 小时后自动关闭的原因；处理新发现的决策犹豫循环诱因

**完成内容：**
- 定位 run `20260717_171106` 崩溃根因：`agent-session.ts` 新增的 `"Already compacted"` 硬抛错（2026-05-15~06-01 之间加入 OpenClaw，论文版本不存在），与 `run.ts` overflow-recovery 循环的 attempt-scoped 重试计数不匹配，导致跨轮已压缩的 session 每轮必然触发 context overflow 死锁；单个 session 循环重试样本占满一批训练数据（Problem 47 占比 87.5%）→ 触发批次污染自我强化机制，是这次 8 小时后崩溃的真正诱因 → [`issues_log.md`](issues_log.md) 2026-07-17 条目
- 修复：`prepare_patched_embedded_agent_overflow_recovery.sh` 直接 patch 核心 bundle，命中 `"Already compacted"` 时按已有的优雅路径重试而非放弃。重新提交训练（run `20260720_112802`）确认 0 次复现
- 同一 run 里发现决策犹豫循环的第三个独立诱因：`## Assistant Output Directives` 章节（跟 Execution Bias 同一批 2026-04 系统提示词重构、此前一直没处理），批次污染机制再次复现（占比 37.5%）。确认批次污染不是 context overflow 独有，是这套异步训练流水线的结构性脆弱点 → issues_log.md 2026-07-20 条目
- 修复：`prepare_patched_system_prompt_output_directives.sh`，在该章节标题后插入一句显式条件说明（"仅在适用时才需要遵守，都不适用就直接发纯文本回复"），思路上是把 March 版本 `## Reply Tags` 的条件框定加回来。本地测试通过（语法、锚点唯一性、幂等性），已接入三个训练脚本，尚未用真实训练验证效果 → issues_log.md 2026-07-20 条目

**主要问题：**
- 用户指出"两个 OpenClaw 机制自己冲突"这个初始框架不准确，要求重新严谨核实——重新核查后确认是单一恢复路径的一个边界情况（attempt-scoped vs session-scoped 状态不一致），不是两个独立机制冲突
- 用户明确要求"训练数据批次污染拦截"这个通用兜底方案延后，先处理已确认的具体诱因（"做了反而会导致如果有问题发现不了"）

### 重要更正：8GPU 正式训练从未真正保存过 checkpoint

**完成内容：**
- 用户追问"07-17 的数学自我重述/纯token退化污染，是否通过训练带到了 07-20 续训、导致更容易触发 Assistant Output Directives 循环"——查证前提（`--load` 是否真加载到了 07-17 崩溃前的状态）时发现前提本身不成立：`qwen3-4b-openclaw-topk-select` 这个 checkpoint 目录从未被创建过（07-17 全程仅约 20 次 `update_weights`，远未到 `--save-interval 100`），`--load` 每次都退回 base 模型从头训练
- **结论：07-17 run 与 07-20 run 之间没有权重连续性，用户提出的这条跨任务因果链不成立**；"批次污染→同一次连续训练进程内持续数小时"这个机制本身不受影响（那条链全程在同一个 Ray job 生命周期内）→ [`issues_log.md`](issues_log.md) 2026-07-20 条目
- 衍生发现：8GPU 正式训练目前为止似乎从未连续跑到第 100 步、从未真正积累过训练进展，每次任务提交实质都是从零开始，是否要调小 `--save-interval` 待评估

### 重新提交训练（run `20260720_165444`）+ 发现第四类问题：假完成声明

**完成内容：**
- 清理上次训练（`20260720_112802`）残留进程：`ray stop --force` 只清了 Ray 托管层，3 个 sglang scheduler 子进程未被清掉、仍占满 GPU 显存，手动 `kill -9` 后确认干净
- 首次重新提交后 Ray 训练 actor 在导入 `torch`→`megatron_utils` 时 `Fatal Python error: Segmentation fault`；查证 GPU/驱动/内存均无异常（无 Xid 错误、显存已清零），判断是偶发瞬时问题（可能与前一步强制 kill 有关但无直接证据），重试后训练正常启动，四个补丁（rl-training-headers、sglang execution-bias、embedded-agent overflow-recovery、system-prompt output-directives）均确认部署成功
- 用户发现新问题：Student 明确要求"把答案追加到 homework/N.txt"，OpenClaw 回复"文件已经更新/已经包含答案"，但实际检查 `homework/46.txt` 内容完全是空模板——**模型在没有真正执行文件写入的情况下声称任务已完成**（部分案例中模型自己贴出的文件内容本身就是空的，却仍然宣称"already contains the solution"，自相矛盾）
- 写脚本系统性分类这次训练全部 72 题：明确要求写入的 35 题中，5 题（46/47/49/56/57）出现假声明话术；对照 `20260716_182012`（41 题要求写入，1 题假声明）和 `20260717_171106`（47 题要求写入，1 题假声明），基线比例约 2%，**这次训练本身把比例放大到 14.3%（约 6-7 倍）**
- 结合已确认的"checkpoint 从未真正保存、这次训练是从干净 base 模型重新开始"，排除"上次训练污染带过来"的假设——**这个倾向本身在模型里一直存在（低概率固有噪音），是这次训练自己的训练动态把它放大到了高频**，跟已知的批次污染自我强化机制是同一大类问题，但这次的具体表现（虚假完成声明）跟之前三个"绕圈子直到顶格截断"的表现完全不同，怀疑是那三个"绕圈子"逃逸出口被堵上后，模型转向了另一条更省事的退化路径（假说，未证实）
- 关联到已知的官方设计空白：`DONE_SENTINEL` 完成判定和 PRM 打分大概率都只看回复文本本身，不核实声称的文件操作是否真的发生 → 这个漏洞可能正在被训练本身放大利用，比之前三次系统提示词内容层面的问题更深一层（涉及奖励信号设计，不只是提示词歧义）

**待查（未完成，下一步继续）：** 找到这次训练里第一次出现假声明（Problem 45/46 附近）对应的具体训练 step 和批次样本构成，看是否有明确的触发事件（类似之前 Problem 47 context-overflow 污染 step 13 那次的精确定位）；确认"绕圈子出口被堵后转向假声明"这个假说是否成立

---

## 2026-07-21

**目标：** 定位"假完成声明"（模型声称已写入文件但实际未写入）的真正根因并修复

**今日新增两个补丁（均已用真实数据/真实诊断验证生效，已接入正式训练）：**

| 补丁 | 解决的问题 | 根因一句话 |
|---|---|---|
| `prepare_patched_cli_compaction.sh` | 训练里大量样本出现"模型声称已完成但实际没做"——本质是**回合已经真实执行成功，但响应被污染成报错**，客户端重试后看到的"已完成"其实是真话 | OpenClaw 论文提交后新加的 `cli_budget` 预压缩检查，命中一个已知的压缩冷却期报错时无条件抛错，把本已成功的回合打成 internal error |
| `prepare_patched_write_edit_guidance.sh` | 大量 homework 文件"Problem:"/"Solution:"结构完全丢失，只剩纯解答段落 | 模型在"追加不要覆盖"的要求下调用了 `write`（整体覆盖）而不是 `edit`（追加）；OpenClaw 自己写了一条"别把 write 当追加用"的安全提示，但因为提示词组装代码的分支问题，这条提示从未真正传到过模型面前 |

**完成内容：**
- 用户指出关键方向：论文实验结果不会有这个问题，说明我们的复现环境跟论文原始条件有不一致，问题应该出在 OpenClaw 的回复本身——没有真正写入不应该让回复看起来像真完成了。之前查的方向（官方设计空白）能解释"这个漏洞为什么可以被利用"，但不能解释"为什么这次突然大规模爆发"，改成直接查 OpenClaw 现在的实际行为
- 静态查 `system-prompt.ts` 排除了"有指引鼓励简短确认式回复"这个假设，没找到证据
- 起一次真实的 `--log-level debug` 诊断（先后踩坑：外部 Simulator API 不支持 tool_choice=auto 走不通；conda 环境没激活导致 sglang 找不到模块；网关进程名不含空格导致 `ps aux | grep` 误判杀不掉旧进程占用端口——都记录在案，最终用独立 sglang server 加载真实 Qwen3-4B-Thinking 跑通），手动模拟"读文件不写→要求追加"两轮对话
- **确凿证据：文件被真实正确写入了，但两次 HTTP 请求都收到 `internal error`**。debug 日志显示确切原因是 `[compaction-diag] trigger=cli_budget outcome=failed reason=already_compacted_recently`——这是已知的"Already compacted"报错的分类结果，命中的是一个之前没覆盖到的第二入口（`cli-compaction.ts` 的 `cli_budget` 预压缩检查，在真实工具调用已经成功执行**之后**才跑，命中时无条件抛错，没有像已修的 `run.ts` 路径那样优雅降级）
- 版本考古确认这套 `cli_budget` 压缩机制本身也是论文提交后新加的（2026-03-08 快照完全不存在，2026-06-09 已存在）
- 完整因果链条：真实写入成功 → cli_budget 预压缩检查撞冷却期抛错 → 回合响应被污染成 internal error → 客户端自动重试 → 重试看到文件已经写好，如实回复"已完成"（不是撒谎）→ PRM 只看对方满不满意，给这种简短回复正分 → 训练逐渐强化、过度泛化到真正没做的场景
- 全面排查 `src/agents/` 目录确认没有第三个入口，修复范围就是这两处 → [`issues_log.md`](issues_log.md) 2026-07-21 条目（含完整诊断记录，供答辩参考）
- 修复：`prepare_patched_cli_compaction.sh`，命中"Already compacted"时记警告日志继续、不抛错，其他压缩失败原因维持原样抛错。本地测试通过，已接入三个训练脚本，服务器部署验证待完成

**主要问题：**
- 之前一度怀疑是论文 PRM 打分方法本身的固有缺陷（reward hacking），用户明确反对这个结论、坚持要求查清楚是不是我们环境跟论文不一致——最终证明用户判断正确，是 OpenClaw 版本漂移引入的新 bug，不是论文方法缺陷

### 部署 cli-compaction 补丁重新提交训练，发现第五类问题：write 工具误用

**完成内容：**
- 清理上次训练残留进程（同样的 sglang scheduler 子进程问题，`ps aux | grep "openclaw gateway"` 因为真实进程名是 `openclaw-gateway`（无空格）一直搜不到，改按 PID 精确定位后清理干净）
- 部署 cli-compaction 补丁重新提交训练（run `20260721_122947`），确认补丁生效（`already_compacted_recently` 依然触发但不再污染回合响应）
- 用户要求继续扫描后续题目，发现新模式：0-25 题里 14/26（53.8%）的 homework 文件"Problem:"/"Solution:"标题完全丢失，只剩纯解答段落——排除"Student 合并指令"这个初始假设（Problem 2 严格分开发送依然中招），改查真实 session 文件（`.jsonl`）里的工具调用记录，**直接证实模型在"追加不要覆盖"的明确要求下调用的是 `write`（整体覆盖），不是 `edit`**
- 确认 PRM 打分规则里"工具返回成功、非报错结果"这一条会给 `write` 的"Successfully wrote N bytes"直接判正分，没有能力分辨"选对工具"和"选错工具但执行成功"
- 用户关键质疑："论文实验不可能有这个问题，是不是我们环境跟论文不一致"——推动进一步排查而非停留在"论文方法固有缺陷"
- 确认 OpenClaw 没有专门的追加工具（只有整体覆盖的 write 和精确匹配替换的 edit），版本考古确认这不是丢失的东西——查了论文提交时锁定的确切上游包版本（`@mariozechner/pi-coding-agent@0.57.1`，从 unpkg.com 直接读取当时源码），从来没有专门追加工具，`edit` 的模糊匹配兜底在两版本里都存在
- 找到真实的 OpenClaw 自身 bug：`write.ts` 带的 `promptGuidelines`（"仅用于新文件或完全重写"）因为 `buildSystemPrompt()` 的分支问题，在生产环境的 `embedded-agent-runner` 里从未被渲染进提示词——但同一次版本考古也确认上游包在论文提交时也没有这个机制，所以不能证明这就是"恢复论文条件"，诚实标注证据边界
- 用真实 debug 诊断直接验证假设（不是先部署再看结果）：手动把这句指引插入运行中的诊断环境，重跑相同失败场景，**确认 session 文件里工具调用从 write 变成了 read+edit，文件结构正确保留** → [`issues_log.md`](issues_log.md) 2026-07-21 条目（含完整版本考古与验证记录，供答辩参考）
- 修复：`prepare_patched_write_edit_guidance.sh`，在同一个 system-prompt bundle 里补一段"## File Editing"指引。技术上需要处理跟 Assistant Output Directives 补丁共享同一 bundle 文件的顺序问题（用独立命名的备份文件避免互相覆盖，本地测试过复合场景）。已接入三个训练脚本

**主要问题：**
- 我提出"补上这句指引能解决问题"时理由不够扎实（先说是恢复论文条件，实际上游包也没有这个机制），用户直接追问"为什么你认为这能解决"——纠正为"这是一个值得尝试的假设，需要先实测验证"，不能把推测包装成结论说给用户听

### 两个新补丁接入训练脚本 + 重新提交（run `20260721_152519`）

**完成内容：**
- `prepare_patched_write_edit_guidance.sh` 正式接入三个训练脚本；本地测试补丁和 Assistant Output Directives 补丁共享同一 bundle 文件时的叠加顺序（独立命名备份文件，避免互相覆盖）
- 首次重新提交因残留进程占用端口 30001（诊断测试用的临时 sglang server 忘记清理导致，`ray stop --force` 清不掉独立子进程，按 PID 手动清理后确认干净）
- 第二次提交遇到偶发 SIGSEGV（Ray actor 启动阶段崩溃，跟 07-20 那次同类问题一样，重试即可，未深究）
- 第三次提交后训练启动但 Student INIT 全部超时失败（`ECONNREFUSED`）——**排查发现是我自己的失误**：诊断调试时把 `openclaw.json` 里 sglang provider 的 `baseUrl` 改到了临时诊断端口（30001），诊断结束后忘记改回正式训练该用的端口（30000），且训练脚本本身的配置步骤不会重置这个字段（只改 `api`/`models`），导致这个残留配置一直生效
- 修复 `baseUrl` 后第四次提交成功：**六个补丁全部确认部署（rl-training-headers、sglang execution-bias、embedded-agent overflow-recovery、system-prompt output-directives、write/edit 工具选择指引、cli-compaction），Problem 0/1 真实对话确认追加成功、"Problem:"/"Solution:"结构完整保留**

**主要问题：**
- 诊断测试用的临时配置改动（sglang baseUrl）没有在测试结束后及时复原，直接导致下一次正式训练启动失败——以后做类似的服务器端手动诊断，测试完必须显式核对/复原所有被临时改动过的配置项，不能只清理进程

### 撤销 write/edit 指引补丁，改用奖励信号方向

**完成内容：**
- 用户判断：给模型加"该用 edit 别用 write"的提示词补丁不是复现该做的事（论文原始环境本来就没有这条指引），正确方向是让打分在模型选错工具时变成负分，让模型自己从训练信号里学会 → 撤销 `prepare_patched_write_edit_guidance.sh`，从三个训练脚本移除部署代码 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-21 条目
- 追溯真实奖励计算管线：`reward_func` → `_submit_turn_sample`/`_submit_rl_turn_sample` → `_build_prm_eval_prompt` 多票裁决

### PRM 打分实测：edit 失败被正确罚分，write 覆盖存在真实盲区（用户连续两次纠正了过于乐观的中间结论）

**完成内容：**
- 实测 Problem 4（edit 真失败）：训练日志确认 turn 4-12 连续 9 轮全部 `eval_score=-1.0`，官方 PRM judge 本来就正确惩罚了这个失败，不需要额外检测
- 实测 Problem 11（write 覆盖但不报错）：两次**独立**训练都命中同一模式（丢失"Problem:"/"Solution:"结构），真实 `eval_score` 都是 **+1.0**，坐实了理论盲区确实在真实训练中发生
- 用户纠正"这次覆盖没有可观测代价"的初步判断：TA 阶段读的是 student 写坏的同一批文件（`homework1/` 由 `homework/` 复制而来），且 TA 阶段末尾同样的"追加不覆盖"指令一旦误用 write，覆盖掉的是学生解答全文，后果远比丢模板标题严重；两次独立训练命中同一模式说明这不是偶发噪声，会被 RL 训练持续放大 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-21 条目

### 重大发现：论文版本混淆，一直用 v1（2026-03-10）当基准，应为 v2（2026-05-11 修订）

**完成内容：**
- 查官方 GitHub issue 库（`openclaw/openclaw`）确认 write/edit 冲突是 OpenClaw 产品早已存在的已知缺陷（[#11102](https://github.com/openclaw/openclaw/issues/11102)/[#44203](https://github.com/openclaw/openclaw/issues/44203)/[#32333](https://github.com/openclaw/openclaw/issues/32333)），跟本项目复现无关
- 用户追问"joint 分数比 separate 低更符合 session 数指标"，推翻此前 WebFetch 摘要读到的论文内容；改用浏览器直接读原始网页文本，确认 arXiv:2603.10165 有 v1（2026-03-10）/v2（2026-05-11 修订，当前唯一有效版本）两个差异巨大的版本，此前一直误读 v1
- v2 原文逐字核对：三角色（Student/TA/Teacher）、session 收敛数指标、连续 3 个 session 判据——**跟 CLAUDE.md 一直记录的内容完全一致，CLAUDE.md 本身没错，错的是本项目做"OpenClaw 版本考古"用的时间基准点**（Joint Hybrid RL 均值 10.3 < Separate 15.0，数字越小越好，跟论文原意吻合）
- 官方仓库 git log 确认训练主脚本 `run_qwen3_4b_openclaw_topk_select.sh` 实际是 2026-04-28~2026-05-12 才开发完成（与 TA_chat.py、v2 论文同期落地），非 2026-03-11；仓库最后一次提交 2026-05-22（仅 README），确认无 v3
- 更新 [CLAUDE.md](../../CLAUDE.md) 基准点说明和目录判断标准 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-21 条目

### 核实官方仓库有无遗漏代码

**完成内容：**
- 逐项核对训练管线（topk-select 脚本、三角色 INIT+Joint、Megatron PRM Teacher）与官方 README/2-node 变体脚本，确认无遗漏，无版本滞后
- 全仓库 commit message + 内容搜索确认没有任何作者（官方或外部贡献者）写过处理 write/edit 覆盖或工具结果校验的代码；GRPO 基线独立实现的 PRM prompt 逐字比对后跟 combine 用的完全一致，同样的盲区
- 确认此前"三月后屏蔽"的三个目录（`oel/`、`openclaw-fireworks/`、`openclaw-tinker/`）均为其他贡献者的独立方法或平行云端后端，不含核心方法代码 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-21 条目

### 用修正基准重新核实 4 个已部署补丁——结论待定

**完成内容：**
- 本地建 `may_2026_5_11` tag（该窗口内最接近的真实 OpenClaw 提交），逐一核实 Execution Bias / context-overflow overflow-recovery / Assistant Output Directives / cli-compaction 四个补丁——**全部在该基准点已存在**
- 用户纠正：论文自己代码库 4-5 月更新不代表作者也同步升级了外部依赖的 OpenClaw 版本，两者是独立软件；查证官方仓库全程没有锁定过 OpenClaw 具体版本号（package.json/requirements.txt/README/Dockerfile 均无），且论文作者自己代码的完整 git 历史里从未出现过处理这 4 个问题的任何 workaround
- **结论：这 4 个补丁是否该保留目前待定**——论文作者实际使用的 OpenClaw 版本号无法仅凭代码仓库证据确定 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-21 条目

### 澄清：两块自建编排逻辑（区别于上面这几个"改 OpenClaw 产品行为"的补丁）

用户要求明确记录——这两块不是补丁，是官方仓库压根没有参考实现、只能靠读论文文字描述自己写的编排代码：

- **Joint 阶段持续驱动逻辑**（`train_with_services.sh` 的 `run_init_phase()`/`run_joint_phase()`/`run_one_persona()`）：官方 `student_chat.py`/`TA_chat.py`/`teacher_chat.py` 都是一次性脚本（跑完 `--num-problems` 道题就退出，无 `--loop`/`--continuous` 参数），`openclaw-combine`/`openclaw-opd` 也没有任何调用这三个脚本的代码——官方仓库完全没有"在一次持续几小时的真实训练里反复/持续驱动三角色"这件事的参考实现。第一版设计（每轮 6 题循环直到训练结束）是我们自己发明的，07-14 改成对照 README 字面"run them together"重新实现：三角色各传 `JOINT_NUM_PROBLEMS=1319`（GSM8K 全量）同时后台启动一次，让训练循环自然消耗真实样本 → 06-29、[07-14](openclaw-rl/docs/work_log.md) 条目，commit `4be24ab`
- **收敛判定统计**（`scripts/check_convergence.py`）：官方三个脚本只把原始回复文本 dump 到 `results_*.txt`，Table 3 的"连续 3 个 session 满足 `satisfies_student`/`satisfies_ta`/`satisfies_teacher` 规则→算收敛"这套判定逻辑官方代码里完全没有实现，是我们自己读 Section 4.1 描述后重新写的事后分析脚本 → 2026-06-23 条目

---

## 2026-07-22

**目标：** 排查 run `20260721_152519` 后段（Problem 13-71）出现的新问题；核实 4 个已部署补丁是否该保留

### run `20260721_152519` 后段新发现：max-turns 激增 + 模型幻觉出"silent reply protocol"

**完成内容：**
- 确认从 Problem 36 起 `Reached max turns (8)` 大量出现（36-71 题约 22 题未达 DONE_SENTINEL），`openclaw.log` 同期持续出现同一 session 反复命中的 `stopReason=length` 未完成回合
- 定位真实推理原文：模型在正常回答后自己平白加一行 `(NO_REPLY)`，此后几十条记录里演变成完整的、自我重复升级的"silent reply protocol"幻觉文本，跟此前查的 execution-bias/output-directives 那类犹豫循环内容完全不同
- 用户提出假设：这个退化会不会是被 Problem 4/11 那类坏训练样本带偏导致的——时间线核对支持但不能坐实（"silent reply"首次出现紧跟在 Problem 11 坏样本大概率已被消费之后，但只是相关性）→ [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### 第 5 个确认版本漂移案例：Silent Reply Policy，且精确命中本项目训练场景

**完成内容：**
- 沿着"silent reply"幻觉现象查 OpenClaw 源码历史：`src/config/silent-reply.ts`/`src/shared/silent-reply-policy.ts` 在 `march_2026_3_8` 完全不存在，`may_2026_5_11` 才作为一整套新功能出现
- 该策略把对话按 session key 分类，`direct` 永远不允许静默，`group`/`internal` 允许；**我们训练用的 session key 格式匹配不到任何已知类型，会落到默认的 `internal` 分类，对应策略正是"允许静默"**——意外命中了这个新功能的放行分支
- 追踪调用方确认这个策略实际控制的是"空/沉默回复算不算合法结果"的服务端判断，March 版本没有这层特殊逻辑；如实标注证据边界：这次没做到 debug 级别实锤，只是"版本缺失+分类命中"两条强关联证据 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### 用户拍板：4 个补丁确认保留，新增第 5 个补丁并完成实现

**完成内容：**
- 累计 5 个独立发现的问题（Execution Bias、Assistant Output Directives、overflow-recovery、cli-compaction、Silent Reply Policy）全部命中"`march_2026_3_8` 缺失、之后版本已存在"这个模式，样本量增加后大幅增强"论文作者实际用的是 3 月或更早版本"这个推断——**解除此前"4 个补丁待定"的状态，确定保留**
- 服务器定位到真实 bundle 文件（`effective-reply-route-BnYlac-J.js` 是真正定义策略函数的源头模块，`dispatch-F64i6im_.js` 只是引用方，只需改前者）
- 新脚本 `scripts/prepare_patched_silent_reply_policy.sh`：把 `resolveSilentReplyPolicyFromPolicies` 恒定改为返回 `"disallow"`，恢复 3 月版本"这个功能不存在"的效果；本地用真实源码片段构造 mock 文件测试通过（含幂等性测试），接入三个训练脚本，`bash -n` 均通过 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### write/未完成误判奖励修正方案：从硬编码覆盖转向 Student 会话级事实核验

**完成内容：**
- 第一版方案（硬编码规则函数直接覆盖 reward）被否：用户指出检查 2 抓错了信号（是 Student 32B 直接判定 HOMEWORK_DONE，不是策略模型文本声称完成），且更根本的问题是"奖励打对了也没法阻止 session 被提前放弃"——这是最初"Problem 4 edit 失败却被判定完成"的根本原因
- 方案收敛为：在 `student_chat.py`（同理 `TA_chat.py`/`teacher_chat.py`）加确定性 harness 层文件核验，放行 `HOMEWORK_DONE` 前核实真实内容，不通过则注入纠正消息而非放行——评估后确认这跟此前否掉的"给策略模型加 write/edit 指引"性质不同，核验的是模拟器（Student）的判断，不是给策略模型开外挂
- 用户提出关键洞察：纠正消息会成为做错动作那一轮的 next_state，PRM 现成的"用户要求纠正=-1"规则会自动精确打分，可能完全不需要另写硬编码覆盖函数
- 追查 Student 判断不可靠的根源：`generate_student_message()` 调用 32B API **未设任何 temperature/top_p 参数**；`STUDENT_SYSTEM_PROMPT` 第 3 步指令本身已经写得很明确（"Never say HOMEWORK_DONE until..."），不存在提示词漏洞，说明是 32B 模型自身指令遵循不可靠，不是我们提示词设计问题 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### 32B Simulator 配置排查 + 全部贡献者身份核实——确认无遗漏

**完成内容：**
- 对比官方 `launch_user_llm.sh` 和我们自己的 `launch_simulator.sh`：两者都没设默认采样参数，我们的部署跟官方设计一致，不是我们的差异
- 全仓库按邮箱区分官方作者（Yinjie Wang/Ling Yang/Xuyang Chen/Xiaolong Jin）与外部贡献者：确认 Ling Yang 只改过 README/LICENSE，Xuyang Chen/Xiaolong Jin 的贡献分别全部在 `terminal-rl/`/`swe-rl/`（General Agent 赛道，不是我们复现的 Table 3 Personal Agent）——**我们实际用的 Personal Agent 路径自始至终只有 Yinjie Wang 一人编写**（另加非署名作者 Siddhant Mukherjee 早期贡献的 top-K 蒸馏基础机制，已在使用中）
- 意外发现：官方仓库早期内嵌过完整 OpenClaw 本体，提交记录写明精确版本号 `2026.3.2-beta.1`（03-03），验证了现有 `march_2026_3_8` 基准点足够接近（只差 6 天）→ [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### Student/TA/Teacher 会话级文件核验：方案定案、实现、本地测试

**完成内容：**
- 最终方案（用户确认 4+1 点）：harness 确定性核验（不靠 32B 判断）、固定模板纠正消息、严格标准（比对"session 开始前已有内容是否保留"+"是否有实质增长"+"是否包含最近认可答案的指纹"）、三角色统一处理、且核验必须在 DONE_SENTINEL 被采纳前同步生效（不能是事后补救）
- 改造 `scripts/prepare_openclaw_test_scripts.sh`：新增通用核验函数块，插入三个官方脚本；`run_one_problem`/`run_one_grading`/`run_one_commenting` 新增 `workspace_dir` 参数并在循环开始前快照 `initial_content`；把"检测到 DONE_SENTINEL 就直接 return True"改成"先核验，通过才 return，不通过就把这一轮消息替换成纠正模板、不 return、继续循环"——替换发生在 return 之前，天然阻止"没验证通过就跳到下一题"的竞态
- 本地测试：三文件语法检查通过；抽取核验函数单独跑 4 个场景（write 覆盖丢结构、正确追加、根本没写、内容对不上），**全部符合预期**
- 无需额外改训练脚本——`prepare_openclaw_test_scripts.sh` 本来就已被三个训练脚本调用且执行的是补丁后的版本，改这一个脚本自动生效 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### 文件核验补丁部署后立刻发现真实误判 bug，用户及时叫停，已修复

**完成内容：**
- 用户直接提交 8GPU 正式训练（run `20260722_124438`），全部补丁部署成功；Problem 0 第 3 轮 OpenClaw 已经正确写入文件，但核验函数误判"验证 FAILED"，注入了不该有的纠正消息——用户观察日志及时发现、要求停下来查，**没有让训练继续在这个 bug 上空耗**
- 根因：`_find_last_substantial_reply()` 取"最近一条超过 50 字的历史消息"当"认可答案"，但这条几乎总是写入确认回复本身（带一句"The solution has been added to..."开场白），取它的前 80 字当指纹，天然匹配不到文件里的真实内容——**这个 bug 幅面很大，几乎所有正确写入都会触发**，不是罕见边界情况
- 修复：指纹匹配方向反过来，改成从文件新增内容取指纹，去最近几轮对话拼接文本里搜，不用再猜"哪条历史消息才是真正认可的答案"；删掉不再需要的 `_find_last_substantial_reply`
- 用真实 Problem 0 对话数据本地复现问题、验证修复：修复前必现误判，修复后正确判定为 True；原 A-D 四个场景重跑全部保持正确，无回归 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### 核验机制升级到 v3：诊断保持确定性，纠正消息改由 32B 现场生成 + 独立复核风格

**完成内容：**
- 用修复后版本重新提交训练，真实数据暴露两个新情况：Problem 0 核验判断对了（真的 write 覆盖丢结构），但固定纠正模板太笼统，模型没能在 8 轮内改对，最终"未完成"收场（诚实但没达成"教会模型改对"）；Problem 1 核验因 Student 违反官方两步协议多绕了一轮，结果正确但不够干净
- 用户连续追问两点关键细节：(1) 复核会不会让 32B 看到带 `Problem:`/`Solution:` 标签的原始文件格式、TA 判断字数会不会被无关内容干扰——确认会，改成只展示新增部分；(2) 这个改动会不会影响 Table 3 收敛判据——核实 `results_student_*.txt` 只在 turn 0（固定开场白，从不含 DONE_SENTINEL）写入，两条路径结构上不相交，直接影响为零
- 定案 v3 方案：诊断（`_diagnose_homework_file` 返回 `overwritten`/`not_written`/`mismatch`/`None`）继续 100% 确定性；检测到 DONE_SENTINEL 统一走一次复核——有诊断只给 32B 自然语气线索（不展示文件内容），结果不能推翻"未完成"这个硬结论；无诊断才展示新增内容（不含标签/已有内容）让 32B 独立复核风格，只有这条路径它说完成才算数。复用现有 `generate_*_message` 函数加 `extra_instruction` 可选参数，API 失败 fallback 回退固定模板
- 本地测试：三文件语法检查通过；`_diagnose_homework_file` 对真实 Problem 0 场景 + 原 A/C/D 三场景全部返回正确诊断码；复核提示文本人工核对符合"不展示标签/只展示新增内容"的设计 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### v3 部署后再发现新 bug：复核仍说 DONE_SENTINEL 时哨兵词原样泄漏成聊天消息，已修复

**完成内容：**
- 用户重新提交训练（run `20260722_134555`）：Problem 0/2/3 全部顺利走完"诊断→复核→独立判断→放行"正常路径，验证了 v3 主流程本身没问题；Problem 1 正确复现"诊断出 overwritten → 拒绝放行"这条硬规则确实生效——但暴露出设计遗漏：32B 复核回复若仍只是原样 `HOMEWORK_DONE`（没听从"用自己的话自然追问"的指令），代码没做任何检查就把这句话直接当成 `student_msg` 发给 OpenClaw，导致哨兵词字面量泄漏成一条没头没尾的聊天消息，OpenClaw 顺势回复"作业已保存"，后续 4B 模型在被污染的上下文里彻底无法产出有意义内容，剩余轮次全部空转到 `max_turns`，真正的问题从未被实际提出或修复——这也直接说明用户此前悬而未决的问题（"4B 模型诊断出问题后能不能真的改对"）这次根本没被测试到
- 修复两处：(1) 加强复核提示措辞，明确加一句"这次不要回复 DONE_SENTINEL"；(2) **代码层兜底（关键）**：只要"有明确诊断 且 32B 复核仍包含哨兵词"，不管 API 是否调用成功，一律强制回退固定纠正模板，从代码层面保证哨兵词绝不可能泄漏成真实聊天消息，不依赖 32B 是否听从提示
- 本地测试：三文件语法检查通过；用真实 Problem 1 场景（overwritten 诊断 + 模拟 32B 复核仍返回 `HOMEWORK_DONE`）复现修复前必然泄漏、修复后正确替换成固定模板；另两条分支（无诊断正确放行、API 异常兜底）验证无回归 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### 按用户要求：给 32B 的线索和兜底纠正模板都改成按诊断类型区分

**完成内容：**
- 用户提出：给 32B 的内部线索不该用"隐约感觉"这类模糊措辞，应该直接说清楚是被 overwrite 还是根本没写入；兜底纠正模板也不该三角色共用一句笼统话，应该按诊断类型分开
- `_DIAGNOSIS_HINTS` 全部改写成直接陈述事实的版本（如"The original problem text ... was overwritten -- it's gone, replaced entirely by the new content"），但仍标注为"internal note"，外层仍要求 32B 转述成自然、不提技术细节的角色内追问，只是喂给模型的输入更明确
- `patch_file()` 的 `correction_template: str` 参数改成 `correction_templates: dict`（按 `overwritten`/`not_written`/`mismatch` 区分）+ `generic_correction_template: str`（仅用于"无诊断但 API 调用失败"这个边界情况）；运行时用 `_CORRECTION_TEMPLATES.get(_diagnosis, _GENERIC_CORRECTION_TEMPLATE)` 选择，顺带修好一个此前遗漏的边界情况——原来"有诊断但 API 异常"也只会用笼统模板，现在会用诊断专属消息
- 本地测试：三文件语法检查通过；6 个场景（overwritten/not_written/mismatch 各自选中专属模板、无诊断+API异常落回通用模板、无诊断+正确 DONE 保持不变）全部符合预期，且确认 TA/Teacher 正确套用了各自目录名和措辞 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### 真实数据实锤：32B 复核 10/10 不遵守指令，且笼统纠正消息本身帮不了 4B——诊断分支重新设计为跳过 32B、直接给出具体缺失内容

**完成内容：**
- 用户重新提交训练，贴出 Problem 8/9 完整 8 轮日志：两题都是第 3 轮 write 覆盖丢失 `Solution:` 标签，第 4-8 轮发送的纠正消息逐字完全相同，OpenClaw 每次都自信回复"已恢复"但展示内容里 `Solution:` 标签始终没被找回来，5 轮后耗尽 `max_turns`
- 排查 `simulation.log`（确认这是 Student/TA/Teacher 三个 `_chat.py` 实际的 stdout 落点，跟 `training.log` 是完全不同进程、不相关）：`re-check call failed` 0 次真实匹配，排除 API 异常路径，确认连续 5 轮都是"32B 复核仍返回 HOMEWORK_DONE → 触发代码层安全网"
- 用户指出根本问题：不是 32B 听不听话，是纠正消息本身没有指出具体缺什么（这次是 `Solution:` 这一行），4B 只能凭记忆重写、每次都同样漏掉这个细节
- 重新设计：诊断出确定性问题（overwritten/not_written）时**完全跳过 32B 复核调用**，直接用确定性诊断专属模板；`overwritten` 的纠正消息直接把 `initial_content`（session 开始时的文件原文快照）整段附进消息里，明确告诉 4B 文件本来该是什么样；32B 独立复核只保留在"无诊断、纯风格判断"这条分支（此前证明工作正常，不受影响）；移除不再使用的 `_DIAGNOSIS_HINTS`，`_build_recheck_instruction()` 简化为只处理无诊断情况
- 复现忠实度重新确认：把模拟器自己最初拥有的题目原文展示回给它，不算给策略模型开外挂——一个真实学生完全可能说"这里原来写的是 XXX，现在不见了"这种具体反馈
- 本地测试：三文件语法检查通过；真实 Problem 8 场景复现验证诊断仍正确判定 `overwritten`，生成消息里完整包含缺失的 `Solution:` 标签和题目原文；确认这条分支不再需要任何 API 调用；无诊断路径回归测试通过 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### workspace 从 /root 迁到 /dfs/data

**完成内容：**
- 用户提出 `/root` 分区不适合放太多太大文件，问 workspace（homework/homework1/homework2 落地位置）能不能迁到 `/dfs/data`——顺带回应此前"GPU 空闲回收重启后 workspace 静默回滚到快照"这条已知限制，怀疑是 `/root` 分区本身的平台行为
- 排查官方源码确认两套独立机制：OpenClaw 自身（`agent-scope-config.ts` 的 `resolveAgentWorkspaceDir()`）优先读 `openclaw.json` 的 `agents.defaults.workspace`，其次才是 `OPENCLAW_WORKSPACE_DIR` 环境变量；`student_chat.py` 等三个脚本读另一个变量 `OPENCLAW_WORKSPACE`。用户贴出真实 `openclaw.json` 确认 `agents.defaults.workspace` 已显式设为 `/root/.openclaw/workspace`——证实只设环境变量不够，必须直接改配置
- 新建 `/dfs/data/openclaw-rl-project/runtime/` 目录（跟 `logs/` 语义对称），内部按跟 `logs/` 相同的时间戳分 run；三训练脚本统一改：`WORKSPACE` 指向 `runtime/<run_id>/workspace`，`openclaw config set agents.defaults.workspace` 每次启动前强制覆盖，`OPENCLAW_WORKSPACE_DIR`/`OPENCLAW_WORKSPACE` 两个环境变量同步设置，`rm -rf` 清理逻辑因引用变量无需改动
- 本地测试：三脚本语法检查通过，diff 核对改动范围精确（4 处改动 × 3 脚本）→ [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### v4 部署后真实数据验证：修复本身有效，但发现"全程用 write 不用 edit"+"write 动作打分无法精确定位"两个新问题，加调试补丁

**完成内容：**
- workspace 迁移部署后重新提交训练，v4 修复验证有效：Problem 0-4 全部命中"append 时丢 Solution: 标签→给具体缺失内容→下一轮一次性修复"，5/5 全部成功，跟此前 Problem 8/9 的 10/10 失败形成鲜明对比
- 用户追问"是不是没用 edit 一直用 write"：用真实 session trajectory 文件核实工具调用次数（`edit` 出现次数精确等于工具目录声明基线值，代表零次真实调用；`read`/`write` 明显超出基线，代表真实被调用），确认策略模型全程只用 `write` 整体覆盖，两次真实调用（丢内容的坏写入 + 靠纠正消息喂原文抄回去的"修复"）都不是 `edit`——讨论后确认不能给策略模型加工具选择指引（此前已否），该起作用的是奖励信号本身
- 尝试核实"这个丢内容的 write 动作有没有被 PRM 正确扣分"：发现 4 轮真实对话对应 5 个 PRM turn 编号（数量对不上，且异步乱序落盘），按"一次工具调用拆两次模型请求"假设重新映射后，turn=3（`[1,1,1]`，score=1.0，跟 Problem 11 历史信号完全吻合）很可能就是那个 write 动作本身、依然被打了满分——但这个映射靠间接证据推出，尝试用 trajectory 里的唯一 ID 反查 training.log 未找到，无法 100% 坐实
- 新增纯诊断补丁 `scripts/prepare_patched_openclaw_combine_select.sh`：给此前从未打过补丁的 `openclaw_combine_select_api_server.py`（真正做打分决策的文件）加一行调试日志，把每个 turn 实际打分用的 response_text/next_state_text 也打出来；`run_openclaw_topk_select_modelfactory.sh` 的 PYTHONPATH 加 `PATCHED_COMBINE_SELECT_DIR` 前缀（跟现有 `PATCHED_OPD_DIR` 同模式），三训练脚本统一接入。明确标注为临时诊断，不影响任何训练逻辑，不再需要时可直接停用
- 本地测试：新补丁对真实官方文件跑通、`py_compile` 通过；launcher 补丁对真实官方脚本跑通，PYTHONPATH 两种取值场景（设置/不设置）均验证正确；全部脚本语法检查通过 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### Problem 13-16 卡顿排查：根因是 "NO_REPLY" 幻觉再次出现，Silent Reply Policy 补丁未能覆盖；顺带修复 not_written 纠正消息的一个真实设计缺口

**完成内容：**
- 排查 Problem 13-16 反复超时/无响应：先排除"GPU 请求排队过载"（`#queue-req` 全程为 0），确认真实存在的是 TP0 引擎 8 分钟吞吐骤降（150-170→1-3 token/s），跟 Megatron 侧 `train_wait_time` 逐步拉长（165.9s→215.7s→373.9s）时间吻合，但未搜到 NCCL/OOM 报错，判断是生成过程本身卡顿
- 翻真实 session trajectory 定位根因：Problem 13 对应 session 从第一轮就出现"Hey. I just came online. Who am I?"幻觉，第二轮 assistantText **字面就是 "NO_REPLY"**（`student_chat.py` 把这类回复打印成"No response from OpenClaw."，不是真没回复），同一 session 里反复出现，紧接着触发真实超时；Problem 11 也独立出现同样的"who am I"开场
- **重要修正：这次部署的 Silent Reply Policy 补丁没能覆盖这个场景**——补丁针对的是"系统层面允不允许真正的空回复"，但这里模型是在正文里真的打出了非空的 "NO_REPLY" 文字，从策略角度不算"沉默"，补丁拦不住。这是模型自身的退化生成习惯，不是这个基础设施开关能修的问题（此前"silent reply protocol 幻觉"跟这个补丁绑在一起的判断需要修正）
- 用户顺带指出 Problem 10 一个真实设计缺口：Student 还没让 OpenClaw 写入就自己说了 HOMEWORK_DONE，核验机制正确识别（not_written），但纠正消息"确保这次真的写进去了"对 OpenClaw 而言逻辑对不上（暗示"上次"，但根本没写过）——修复：新增 `_write_was_requested()` 辅助函数判断此前是否真的提过写入要求，`not_written` 分支据此二选一：真提过用原有措辞，没提过用不暗示"之前写过"的新措辞（`"not_requested"` 模板）
- 实现过程中发现并修正两次真实假阳性：FIRST_MESSAGE_TEMPLATE 固定含"don't write"导致的假阳性（跳过第一条消息解决）、"like how a person would write it"这种谈风格不谈写文件的假阳性（正则要求写入动词和 file/homework 关键词同句靠近出现）、以及真实 Problem 11 里"Don't write to the file yet."这种否定句的假阳性（正则排除否定词紧邻写入动词的情况）
- 本地测试：三文件语法检查通过；用真实 Problem 10/11 原文复现验证 4 个场景全部符合预期（含两次假阳性修正后的回归验证）→ [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目

### "who am I"/"NO_REPLY" 幻觉排查收尾：确认不是今天新加补丁引入的，暂缓处理，重新提交训练

**完成内容：**
- 用户追问 Problem 11 Turn 1 为什么也出现"who am I"幻觉：核实真实 session trajectory 的 `context.compiled` 事件，确认发给模型的 prompt 是干净的 Problem 11 开场白原文、`messages count: 0`（全新 session，无历史串位）——确认这是模型自身对干净输入的异常回复，不是 session 管理/prompt 拼接层面的 bug
- 用户追问这两类幻觉是不是今天新加的补丁引入的：结构上确认排除——Turn 1 用的固定开场白模板不受今天任何改动影响（核验补丁只在 DONE_SENTINEL 之后触发），workspace 迁移不碰对话内容，PRM 调试补丁不经过发给策略模型的请求路径；且 "NO_REPLY" 幻觉本身早在 07-21 训练（run `20260721_152519`）就已观察到，早于 Silent Reply Policy 补丁被设计出来——**两类幻觉都是训练本身早就存在的模型固有退化行为，不是今天任何一次改动引入的新问题**。更早部署的 5 个 OpenClaw 版本漂移补丁跟这个现象有没有关联，未完全排除，需要更早期训练数据对比才能确认
- 用户认可这类幻觉会造成"批次污染→自我强化"，但仍属于此前已多次要求延后处理的范畴（模型自身生成行为层面，不是 harness 能修的）——本次决定先记录、继续观察，不在这次改动范围内处理 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-22 条目
- 清理服务器残留 sglang 进程（`kill -9`，确认显存已释放），重新提交训练（`bash train_with_services.sh`），明天查看这次训练结果

---

## 2026-07-23

**目标：** 查看 07-22 晚提交的训练结果；追问 Table 3 收敛指标细节，牵出 Joint 训练真实编排方式的重新核实

### Joint 训练真实编排方式重新核实：确认现有 INIT→Joint 架构不对，真实流程是先跑完 Separate 再复用其产物启动 Joint

**背景：** 用户报告训练又是几小时就失败，追问 Table 3 的"session"单位（一道题，不是训练侧"16 条样本一步"的那个 16）、`SESSION_LIMIT=72` 是不是预设值、收敛后训练是否还会继续——这几个问题逐步牵出一个更大的疑点：现有的 INIT→Joint 两阶段架构（Student 顺序跑 72 题→TA 顺序跑 72 题→Teacher 顺序跑 72 题，再切到并行）有没有真的对应论文的训练方式。

**完成内容：**
- 确认 Table 3 的"session"= 一道题的 Turn 0 回复（`check_convergence.py` 源码核实），跟训练侧的 batch size 是完全不同的概念
- 用户发现关键反证：Table 3 里 Joint vs Separate 的 Student 收敛数字差值（7.6）明显大于 TA（3.6）和 Teacher（2.6）——如果 INIT 阶段 Student 真的完全独立跑，应该是差值最小的那个，跟实际相反；且 Table 3 图注确认"5 次独立试验取平均"，排除"只是训练噪声"这个解释
- 直接重读论文原文（`pdftoppm` 渲染 PDF 页面读图，不用 WebFetch 摘要）：p.5 明确"three simulated users using OpenClaw simultaneously"；p.21 附录 A.1 确认 72 是"at most"的评估上限（不是预先准备好的题量），且"先建立 homework1/homework2，然后开始 joint optimization...simultaneously"；全篇没有"收敛后训练终止"的表述
- 重新核查三个被禁目录（`openclaw-rl/oel/`、`openclaw-fireworks/`、`openclaw-tinker/`）：确认禁用标准从来不是按时间线（CLAUDE.md 原文早已写明），且 `openclaw-rl/oel/eval/gsm8k_personal_agent.py` 进一步确认是 v1 风格的 2 角色设计（无 TA）+ 直接拿 ground truth 顶替学生答案（不走真实文件工具），再次独立证实不能作为 v2 参考
- git 历史考古：`TA_chat.py` 2026-05-07 才加入（v2），`student_chat.py`/`teacher_chat.py` 2026-03-11 就有（v1，2 角色）；加 TA 那次 commit 的 README 改动把"two sequential phases"改成"three sequential phases"，**"run order matters"顺序执行的措辞被原样保留扩展，没有改成并行**；v1 时代 README 完全没有"joint"/"separate"/"homework1"/"homework2"这几个词——**这个对比维度本身是 v2 才新增的**
- 核实 `ensure_homework_dir()` 源码：一次性快照复制（`if os.path.isdir(target): return`），只在目标目录不存在时执行一次，之后永不更新——结构上不支持"跟上游实时进度同步的并发"，只支持"上游已完整、下游一次性复制"
- **最终结论（用户提出）：真实流程应该是先完整跑一遍 Separate（Student/TA/Teacher 各自独立训练 job，Table 3 本来就要报这一列数字），Separate 跑完后 `homework/`/`homework1/` 作为真实副产品自然留下；Joint 直接复用这些已经完整、不再变化的快照，三角色从一开始就同时启动**——不需要专门为 Joint 设计 bootstrap，也不存在死锁/追赶进度的问题（因为快照已经是完整的，TA/Teacher 不需要等 Student 实时进度）。现有 INIT 阶段（用刚开始训练、接近基础模型状态的模型拼凑 homework1/homework2）不是论文真实做法 → [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-23 条目，`docs/paper_reproduction_scope.md` 已同步更新 Joint/Separate 描述和执行路线优先级（Separate 从低优先级变成 Joint 的前置依赖）

**下一步（用户明确指定）：** 先完成 Separate-Student 部分（`train_separate_student.sh`，Table 3 复现路线 Phase 3a）。

### Joint 阶段 homework1/homework2 建立方式细化：确认需要"错位复制"，正文 4.4 节进一步确认，四阶段方案定案

**完成内容：**
- 用户追问"如果 homework1 里已经有 TA 的真实答案，Joint-TA 再训练这一题会不会被已有答案干扰"：核实 `TA_chat.py`/`teacher_chat.py` 是**读写同一个文件**（`HOMEWORK_DIR`/`SOURCE_HOMEWORK_DIR` 都是写死的模块级常量，只服务单次运行内部复制），一度提出"每个角色写独立文件"的方案后被否——跟官方硬编码协议（TA 系统提示词明确是"追加进它刚读的那份文件"）不符，write/edit 覆盖问题不会因为换文件命名方案消失，是两个独立问题
- 确认"错位复制"能正确避免污染：Joint 的 `homework1`（给 TA 用）要从 **Separate-Student 自己的 `homework/`** 复制（干净，TA 从没写过东西进去）；Joint 的 `homework2`（给 Teacher 用）要从 **Separate-TA 自己的 `homework1/`** 复制（干净，没有 Teacher 点评）——规则："下一角色要用的目录，必须从上一角色自己读写的目录复制，不能从上上个角色的目录复制"
- 重新解读附录 A.1"we first save the directory completed by the student as homework1, and save the directory completed by the TA as homework2"，确认这就是错位命名的字面意思（之前下意识按脚本内部命名习惯理解错了）
- 确认这套错位复制规则完全没有官方代码参考，是从附录原文 + 自己推理"怎么避免污染"两方面推出来的
- 用户追问系统提示词（三步协议）到底有没有生效：核实 `generate_student_message()` 每次调用都完整发送 `STUDENT_SYSTEM_PROMPT`；确认我们新加的 `extra_instruction` 只在复核（recheck）这一次特殊调用里追加，不影响正常对话轮次——排除是我们补丁干扰的可能，判断是 32B 模型自身指令遵循不可靠（此前已确认的已知短板）
- **要求完整重读论文（不只是附录，正文实验章节也要读）确认四阶段方案没有偏差**：正文 4.4 节（p.12）"In addition to optimizing the model for a single user, OpenClaw-RL supports multiple individuals sharing the same model, with the model jointly optimized..."——正文原话明确 single user（Separate）是基础情况、各角色独立模型，joint 是额外能力、共用同一模型，比 Table 3 表格列名本身更直接地确认了 Separate/Joint 是两个不同模型
- **四阶段方案定案**：Phase A（Separate-Student）→ Phase B（Separate-TA，依赖 A）→ Phase C（Separate-Teacher，依赖 B）→ Phase D（Joint，全新模型，错位复用 A/B 产出，三角色从一开始同时启动）；产物统一存到 `/dfs/data/openclaw-rl-project/table3-artifacts/`（持久化，不用 `runtime/<run_id>/` 这种每次训练都换的临时目录）→ [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-07-23 条目，`docs/paper_reproduction_scope.md` 已更新完整方案

**下一步：** 开始写 `train_separate_student.sh`。

---

## 2026-07-24

**目标：** 验证 Execution Bias 全清空补丁效果并排查新出现的训崩现象；补齐答辩用的实验流程图；对 Separate-Student 阶段收尾数据做最终核实；讨论下一阶段方向（TA/Teacher/Joint vs. General Agent vs. SEA-Eval）

### Separate-Student 阶段收尾：收敛数字 22，对齐论文 19.2

**完成内容：**
- 修复 `train_separate_student.sh` 的收敛检测 bug（占位路径不存在导致 `check_convergence.py` 从未真正生成过结果），手动补跑后确认 07-23 诊断实验（Execution Bias 补丁前）**Student 收敛于 session 22**，与论文 Table 3 的 19.2 处于同一量级
- 核实并澄清 `satisfies_student` 收敛规则本身只挡 `**`/编号列表/`\boxed{}`，不挡破折号列表/emoji，比人类直觉宽松——确认这是论文原文的规则设计，不是复现代码的偏差
→ 详见 [`issues_log.md`](issues_log.md) 2026-07-24 两条相关条目

**产出：**
- `personal_agent_full_experiment_detail.svg` / `.png`：Personal Agent 完整实验流程图（Simulator/OpenClaw/Policy 交互 + RL 训练循环 + Separate/Joint 阶段结构），答辩用

### Execution Bias 补丁验证：目标问题已解决，但暴露另一个独立的训崩机制

**完成内容：**
- Execution Bias 全清空补丁推送后重新起训练，确认原本瞄准的"41 轮不收尾"死循环（07-23 Problem 31 那种）没有复现——补丁本身有效
- 但 Problem 36/37 再次出现 context overflow，排查排除三个假设（跨 session 内容污染、继承 07-23 跑崩的 checkpoint、补丁未生效）后，确认真实机制是**模型在单个 turn 内部反复用完全相同参数调用同一工具**（`read`/`write`），且这个行为从训练极早期（Problem 5）就已存在
- 用统计脚本量化后进一步核实、并修正了两处早期误判：真正的异常不是"重复次数多少"，而是**这个重复循环会不会自己终止**——Problem 35 那种连续 7 次、被 write 打断后能恢复的循环属于正常基线；Problem 36 从第 3 轮起进入的连续 32+ 次、从未被打断的纯 read 循环才是真正导致 context 撑爆的异常。时间上，"有界→无界"这个转折精确对应训练 **step 9** 这次权重更新（此前一度误判为 step 8）
→ 详见 [`issues_log.md`](issues_log.md) 2026-07-24 条目（含完整排查过程、误判修正记录）

**关键决策：** 为进一步诊断，给 `openclaw_opd_api_server.py` 加了一个纯观测性质的调试补丁（`openclaw-rl-debug-repeat-thinking`：检测到精确重复的工具调用时，把模型完整的 `reasoning` 原文打出来）——只加日志，不改变任何训练行为，已推送，待下次训练触发后查看效果。

### 训崩根因排查：PRM 打分规则本身对重复调用有系统性盲区（已定位，暂未落地）

**主要问题：** `openclaw-opd/openclaw_opd_api_server.py` 的 `_build_prm_eval_prompt()` 判分规则写死"工具调用只要没报错就该打正分"，完全不检测这次调用是不是在原样重复上一次——Problem 36 那 32 次连续 read 每一次都因为"读取成功、没报错"被判正分，训练信号上没有"这是浪费轮次的坏行为"这个概念；进一步确认这个分数直接就是 RL/GRPO 的训练奖励，不是离线诊断专用指标 → 详见 [`issues_log.md`](issues_log.md) 2026-07-24 条目

**关键决策（待定，非最终结论）：** 一度实现了"检测到精确重复就强制把 `eval_score` 覆盖成 -1"的代码补丁并推送，随后被用户叫停并要求撤回——用户当时仍在讨论阶段，尚未最终确认要不要这样改、要不要设阈值等细节。补丁已用 `git revert` 撤销并重新推送，当前代码库不含任何打分行为改动，只保留纯观测性质的重复检测日志。**这个方向本身仍然成立、被认为很可能是训崩根因，但具体怎么修改打分规则需要后续继续讨论确定，不要在没有明确共识前再次实施。**

### 下一阶段方向调研（暂缓 TA/Teacher/Joint，评估 General Agent 与 SEA-Eval）

**背景：** 组会确定暂不继续做 TA/Teacher/Joint 部分（Personal Agent 光 Student 一部分就花了一个月，性价比现阶段判断不高），需要在收尾 Student 的同时想清楚下一步方向。最终方向待下周和导师开会后再定，本次只做前期调研。

**完成内容：**
- 梳理 General Agent 四个 track（GUI/SWE/Terminal/Tool-call）的基础设施重量级差异：`swe-rl` 只有 README、无训练脚本，需要 ECS Docker 节点；`terminal-rl` 需要独立远程 Docker worker + 跨机网络；`gui-rl` 需要视觉语言大模型（27B）+ 真实 GUI 交互环境；**`toolcall-rl`（ReTool，数学+代码解释器）单机、跟现有 Qwen3-4B 同规模、沙盒是自包含 subprocess（无额外基础设施依赖）**，是四个里复现风险最低的一个
- 核实 `toolcall-rl` 官方代码完整性：8 个文件齐全（含 4 档训练脚本、生成/沙盒/数据处理逻辑），import 均为标准库 + `slime`，无隐藏外部依赖；训练脚本结构与现有 `run_qwen3_4b_openclaw_topk_select.sh` 高度相似（同样的 `ray job submit` + `train_async.py` + PRM_ARGS 模式），数据集为公开 HuggingFace 数据集，官方甚至提供现成 SFT checkpoint 可跳过 SFT 阶段
- 核对 `paper_reproduction_scope.md`，确认 Figure 6 是两个独立于 Table 3/Figure 5 的实验（ReTool 多轮 RL：PRM 用 Qwen3-8B；RLVR/AIME：Policy 用 DeepSeek-R1-Distill-Qwen-1.5B）——`toolcall-rl` 这一套代码同时是 Figure 5 tool-call track 和 Figure 6 左图的实现基础，选它可以同时够到两张图

**关键决策（待定）：** 倾向 `toolcall-rl` 优于 SEA-Eval——前者基础设施最轻、有现成官方配方，确定性更高；SEA-Eval 虽然能复用现有训练基础设施，但没有现成任务/reward 设计可抄，是开放性研究设计问题，风险类型不同。**最终选择等下周导师会议后再定。**

**下一步：**
1. 本次已提交的诊断训练结果**下周再看**（用户明确要求，本周不再跟进）
2. 下周导师会议后确定 TA/Teacher/Joint 是否彻底搁置、General Agent（`toolcall-rl`）还是 SEA-Eval 作为下一阶段方向
3. PRM 打分规则的重复调用惩罚方案继续讨论，确定具体实现方式（是否设阈值、覆盖到什么程度等）后再实施
4. Problem 36 无界循环的具体触发原因（训练 step 9 相关性）如果继续 Student 收尾工作，仍是待查项

## 2026-07-27

**目标：** 检查上周提交的诊断训练结果，排查用户报告的"40 多题又出现 context overflow"是否与之前同一问题

**完成内容：**
- 确认这次训练比之前更严重：Problem 26（孤立、`/reset`/`/new` 均无法恢复，不影响后续题目）、Problem 29（同样卡住但最终自己收敛出真实回复）、**Problem 42 起连续 27 道全新题目（42-68，每题都是独立 session）全部 context overflow、无一恢复**——由于各题 session 互不共享上下文，这只能说明 Policy 权重本身在训练中的某个时间点进入了持续性退化状态，是目前样本里最严重的一次
- **新发现根因细节**：反查真实 `reasoning`（07-24 加的调试补丁生效）确认模型在 Problem 26/29 均把答案反复写去一个从未被提及的虚构文件（`26_answer.txt`/`29_answer.md`），不是重复读同一个真实文件；Problem 29 每次改写内容还略有不同（不是逐字重复）
- **发现调试补丁本身的检测漏洞**：判定"重复"要求参数字符串逐字相同，导致 Problem 26 的 16 次虚构文件调用一次都没被检测到（因为内容不完全相同）——之前统计的重复次数很可能系统性低估，需要把判定标准改成"路径/目标相同"而不是"参数逐字相同"
→ 详见 [`issues_log.md`](issues_log.md) 2026-07-27 条目

**关键决策：** 用户明确指出 Problem 42 这种"一旦出现、之后全部沦陷"的模式才是真正需要优先解决的问题，重要性高于 26/29 这类局部、可恢复的偶发情况——下一步排查应聚焦这里，而不是平均用力查所有重复调用现象。

### Problem 42 根因排查到底 + PRM 打分修正定案并实现

**完成内容：**
- 精确核对 `prompt_msgs` 增量，切出 Problem 42 harness 层第 1 轮对应的全部 6 个内部 turn：`read → read(重复) → sessions_send → sessions_send(重复) → read → sessions_yield`，全程没有产出过一次真正的文字回复——确认"No response"的直接原因是模型选择了 `sessions_send`/`sessions_yield` 这类多 agent 协调工具，而不是训练权重瞬间损坏
- 查清这两个工具的真实用途（`sessions_send` 本意是发消息给别的 session/子 agent，`sessions_yield` 本意是派完子任务后等结果）——都是**给多 agent 协作场景设计的工具**，这次单轮 Student-Policy 对话根本用不上；`sessions_send`/`message` 两个工具三月版本就存在（不是版本漂移新增的），只有 `sessions_yield` 是六月才新增，说明不能简单归因为"OpenClaw 版本问题"
- 核对三个工具各自的 PRM 打分：`sessions_send`/`sessions_yield` 因为"技术上没报错"全部被打正分（分别 12 次和 6 次，全部 +1.0），`message` 因为报错（没配置聊天频道）被正确打了负分（3 次，全部 -1.0，之后被模型放弃）——证实问题出在"没报错=正分"这条规则本身，不是所有工具都被误判
- **PRM 打分修正方案定案并实现**（过程中否决了两版方案）：工具黑名单（否决，没有可迁移性）→ 改 LLM 判官提示词（否决，风险不可控：判官本身是弱模型、软标准会增加打分噪声、影响面覆盖 GRPO-only 和 Hybrid RL 两条线）→ 全局 turn 计数阈值（否决，模型学不到具体因果）→ **最终采用三条"逻辑上必然成立、不针对具体任务设计"的确定性代码规则**（read 类查询工具紧邻重复、sessions_send 自问自答、sessions_yield 没有对应的 sessions_spawn），已实现并本地验证（语法检查 + 真实官方源码模拟 + 编译检查），**尚未用真实训练验证效果**
→ 详见 [`issues_log.md`](issues_log.md) 2026-07-27 两条相关条目（排查过程 + 打分修正方案）

**产出：**
- `scripts/prepare_patched_openclaw_opd.sh`：新增 `is_invalid_tool_use` 标记计算逻辑
- `scripts/prepare_patched_openclaw_combine_select.sh`：新增读取该标记、覆盖 `eval_score` 为 -1.0 的逻辑

### 补丁已拉取重新提交训练；发现"训练目标≠Table 3 评估目标"这个更深层问题

**完成内容：**
- 服务器 `git pull` 确认拉取成功，新训练已提交，**结果留到明天再看**
- 澄清了打分范围的确切边界：只有 Policy（4B）自己在 OpenClaw 对话里的动作（含工具调用）会被打分、计入训练样本；Simulator 的回复、PRM 判官/Teacher 自己的生成都不会
- **新发现（用户提出）：Simulator 主观判定"AI 感太足"要求重写后，重写版本有时会带上 `1. 2. 3.` 编号列表——这种格式反而命中论文正则的 `^\d+\.` 规则、判定不通过，而重写前的版本（比如破折号列表）本来是能通过正则的。** 说明训练时实际驱动奖励的信号（PRM 判官看"下一句话是否表现出满意"）跟 Table 3 汇报用的窄正则规则，衡量的根本不是同一件事——这能部分解释"收敛后又倒退回编号列表"这种反复
- 追查到 `STUDENT_SYSTEM_PROMPT` 原文：明确举例"bold text/numbered lists/**Final answer**:"这三种具体特征（跟论文正则高度对应），但后面跟了一句开放式兜底"or anything too AI-like"——这句没有精确边界，完全依赖具体 Simulator 模型的主观解读，是这次错位现象的直接来源
- **关键澄清：这周（含本次新提交的训练）Simulator 用的其实是 DeepSeek V4，不是论文原文规定的 Qwen3-32B**——上面这个"AI 感判断错位"的具体严重程度，可能是 DeepSeek V4 自己的主观偏好导致的，换成 Qwen3-32B 会不会有同样表现、错位程度有多大，需要真实测试才能确认，目前不能直接下结论
→ 详见 [`issues_log.md`](issues_log.md) 2026-07-27 条目（待补充：这次讨论目前只在对话中，尚未整理进 issues_log.md）

**关键决策：** 用户提出的"改用 Turn 2 是否要求重写来判断收敛"这个替代指标，讨论后确认**不能替换 Table 3 官方正则指标**（论文原文逐字给出的判定规则，换掉就没法跟论文数字比较），但可以作为独立的补充分析保留。

**新提出的问题（用户）：随着训练次数增加，难以追溯"哪次训练对应哪个问题、做了什么改动"。已实现：** `train_separate_student.sh` 启动时自动把当前 `openclaw-rl` 仓库的 git commit hash 写进该次训练自己的日志目录（`RUN_MANIFEST.txt`），**同时按用户要求同步拼进 wandb 的 run 名字里**（`run_openclaw_topk_select_modelfactory.sh` 的 `--wandb-group` 后缀），两边都能查到"这次训练对应哪个版本的补丁代码"，不再依赖人工记忆。本地验证通过（语法检查 + 真实官方源码模拟补丁生成）。**目前只覆盖 `train_separate_student.sh` 这一条训练路径，其他训练脚本还没有加**，见 [`issues_log.md`](issues_log.md) 2026-07-27 条目。

## 2026-07-28

**目标：** 落实 07-27 条目留下的两个待办——精确定位 `edit` 反复失败的技术根因，并实现两条新 PRM 打分规则（通用工具错误结果判负分 + read 覆盖范围累积追踪）

### PRM 打分修正两条新规则

**完成内容：**
- 精确定位 `edit` 反复失败根因：模型对 `oldText` 参数产生了一次 JSON 双重转义错误（LaTeX 内容本身需要大量反斜杠转义，可能是诱因），把本该是真换行符的位置写成了字面上的反斜杠+n 两个字符，跟 `cat -A` 核对确认真实文件里对应位置是真正的换行符字节——两者逻辑上永远不可能匹配，保证每次重试都会失败
- 拿到 48 次同路径 read 调用的完整 limit 参数分布（50 到 100000 之间反复横跳），确认现有"逐字参数匹配"检测对这种情况完全无效
- 实现规则："`next_state` 是 `status: error` 的工具返回结果 → 强制判负分"，通用、不挑工具，直接对 `_build_prm_eval_prompt()` 自己写的"环境报错该判负"规则做代码层兜底（真实数据证实判官不总是照着自己写的规则执行）
- 实现规则 1b（read 覆盖范围累积追踪）：**设计过程中自己发现并修正了一个 bug**——第一版实现只记"最近一次 read 的 (limit, 结果长度)"，用真实 limit 序列（大→小→大反复横跳）本地模拟测试后发现，一次小 limit 的读取会把"之前大 limit 已经证明读到过 EOF"这个信息错误覆盖掉。改成按 path 累积、sticky 的覆盖状态（`eof` 一旦确认永久保持，`max_limit` 只单调增大）后重新测试，行为符合预期
- 两个补丁脚本本地对真实官方源码完整验证：`bash -n` 语法检查、用真实 `OpenClaw-RL-official` 源码模拟生成、`py_compile` 编译检查全部通过；额外写了独立 Python 逻辑复现脚本，用真实的 48 次 limit 序列驱动跟生成代码一致的逻辑，确认修正后设计能正确处理真实观察到的 limit 波动顺序
→ 详见 [`issues_log.md`](issues_log.md) 2026-07-28 条目（含 edit 根因的完整技术细节、两条规则的设计推导过程、验证方法）

**产出：**
- `scripts/prepare_patched_openclaw_combine_select.sh`：新增 `openclaw-rl-tool-error-penalty` 规则
- `scripts/prepare_patched_openclaw_opd.sh`：新增规则 1b（`self._read_coverage`/`self._pending_read` 累积追踪逻辑）

**下一步：** 推送后请求用户在服务器 `git pull`，提交新训练观察两条新规则的真实触发效果

### Demo 视频溯源 + 官方仓库分支核查

**完成内容：**
- 用户提问 README 自带的 demo MP4 是否为 v1 版本演示、之前是否因为只看 main 分支而漏看了其他分支代码
- 核查官方仓库全部分支：除 main 外的 4 个分支中，3 个提交完全被 main 包含（无独有提交），1 个（`openclaw-topk-opd`）的唯一提交经逐文件内容比对确认是被 squash merge 进 main（4 个文件逐字节相同，另 2 个 main 版本更新更完整）——确认此前只读 main 分支没有遗漏任何代码
- 定位 demo 视频本身：git 历史显示 2026-02-26 加入 README，早于 v1（03-10）和 v2（05-11）两版论文提交/修订日期，判定为预告性质的早期演示，不严格对应任一版论文的最终训练机制；**受限于无法直接播放/解析视频内容，无法核实用户观察到的"每条 response 打一个 reward"具体展示细节**

## 2026-07-29

**目标：** 排查 07-29 新训练（两条新 PRM 规则上线后）Problem 36 再次 context overflow 的原因，并追查用户发现的"回复长度随训练持续变长"这个更根本的问题

### Context overflow 根因定位：OpenClaw 自身的压缩节流机制

**完成内容：**
- 确认这次超预算幅度很小（只超约 400 token），但每次自动压缩都以 `already_compacted_recently` 失败收场——OpenClaw 自身的压缩节流机制导致"超预算→压缩被拒→仍超预算"死循环，`/new` 也救不回来（很快再次撞上同一节流）
- `toolResultReducibleChars=0` 确认这次超预算主体不是工具调用历史，而是系统提示词+回复本身，跟长度膨胀问题直接吻合，指向同一根因
→ 详见 [`issues_log.md`](issues_log.md) 2026-07-29 条目

### 回复长度持续膨胀：定位具体机制并实施 Simulator 提示词改动

**完成内容：**
- 量化确认长度膨胀：39 题里前 19 题平均 400 字符、后 20 题平均 855 字符，且持续加速（单题峰值 2152 字符）；全程"正则判定干净"的 2 道题均为生成失败误判（56 字符的错误提示，非真实解答），真实 Turn1-干净率是 0/39，远达不到论文"平均 20 session 收敛"
- 排除三个假设：Simulator 提示词跨 session 残留（读源码确认每题每轮都重新发完整提示词）、多候选 hint 选择偏好长文本（读代码确认实际按最短优先排序）、整条奖励链路的"没有长度约束"是被动缺口（确认存在但不能单独解释主动增长）
- **实测比对 Problem 20/21 两个真实 session 定位到具体机制**：Problem 21 需要两轮重写，第一轮重写因为"加了 emoji 和场景化措辞"又被判 AI 味——模型应对"更自然"这个要求的方式是加内容、不是做减法，纠正循环本身在主动喂长度，且没有环节会清理这些新加内容
- 用户提出假设（未证实）：Simulator 用 DeepSeek V4 顶替论文原定 Qwen3-32B，可能对"AI 感"判断更严格，4B policy 跟不上
- **实施改动（用户确认后）**：`scripts/prepare_openclaw_test_scripts.sh` 新增补丁，去掉 `student_chat.py` 的 `STUDENT_SYSTEM_PROMPT` 里两处开放式"AI 味"兜底判断，只保留 bold/numbered-list/`**Final answer**:` 三个具体特征，与 Table 3 收敛正则对齐。**明确是主动偏离论文自己写定的 Simulator 提示词，不是修 bug**，需要在结果里说明这一处偏离
→ 详见 [`issues_log.md`](issues_log.md) 2026-07-29 条目（含完整排查过程、Problem 20/21 逐轮对比、改动的完整风险说明）

**产出：**
- `scripts/prepare_openclaw_test_scripts.sh`：新增 `student_chat.py` 的 AI-like 开放式兜底移除补丁

**下一步：** 推送后请求用户 `git pull`，重新提交训练观察长度膨胀是否缓解、真实 Turn1-干净率是否提升

### 新训练里人工通读 Problem 20-30，"只给答案"仍会烂尾；Steps 第 0 条方案实现后经真实训练验证无效，已撤销

**完成内容：**
- Simulator 提示词收窄改动上线后提交的新训练（`separate_student_20260729_131944`）里，用正则粗筛统计"Student 自己代答"比例（改动前 1.7% vs 改动后 4.2%），**用户指出正则筛法不完整、样本太小，要求先人工通读真实数据再下结论/做改动**——按此要求人工通读 Problem 20-30 全部 11 道题
- 真实情况：Problem 20 是唯一真正"烂尾"的（全程无人要求补步骤，最终文件只有光秒答案无解题过程）；Problem 21/22/23/29 虽然 Turn1 也是光秒答案，但后续大多能自己或在混乱追问下把真实步骤补回来；其余题目 Turn1 本身就有步骤
- **额外发现两个独立问题，本次不处理，先记录**：Problem 27 里 OpenClaw 明确返回"No response from OpenClaw."，Student 却仍宣布 HOMEWORK_DONE，违反自身规则，`homework/27.txt` 大概率未被真正写入却计入"已完成"；Problem 25 里 Student 自己编造的问题描述跟 Policy 原始答案对不上，Policy 未纠正，最终写入答案可能有误（正确性问题，不只是格式/长度问题）
- 用户提出假设：这些下游混乱大多是"Turn1 拿到光秒答案后 Student 只能自己脑补"导致的，从源头稳定要求"show me the steps"能一并解决大部分。核实提示词后确认"必须有完整步骤"这条要求本来就写在开头段落，只是没有变成 Steps 列表里第一个要检查的显式动作
- **实施改动（用户确认后）**：`scripts/prepare_openclaw_test_scripts.sh` 追加 Steps 第 0 条（跟 Step 1 同样的并列句式，无衔接语）："If the AI only gives a short answer with no steps shown, tell it to show all the steps. If it already shows the steps, no need to ask."，本地验证通过后推送
- **真实训练结果：方案未生效，且出现新故障**——用户拉取并提交新训练后，Problem 21 的 Student 仍然自己给出了完整的分步解答（带 `**Step 1**`/`**Step 2**` 加粗标题），Steps 第 0 条并未阻止"Student 自己代答"；同时出现"⚠️ Agent couldn't generate a response."的新生成失败故障。**用户明确要求不深查这次新故障，直接撤销 Steps 第 0 条这次改动**——已通过 `git revert` 撤销，AI-like 兜底收窄的改动保留不受影响
→ 详见 [`issues_log.md`](issues_log.md) 2026-07-29 条目（后续追加部分，含 Problem 20-30 逐题通读记录、Problem 27/25 两个独立问题的完整描述、Steps 第 0 条的完整实现与撤销过程）

**下一步：** "Student 自己代答"的真正成因仍未查清，仍是开放问题；下次需要先弄清楚为什么 Steps 第 0 条没能生效，而不是直接再假设新方案

### 新训练（撤销 Steps 第 0 条后）Problem 19 崩溃排查：两个并发 session 各自膨胀上下文拖垮网关，定位 edit 死循环第三种根因

**完成内容：**
- 撤销 Steps 第 0 条、清理残留 GPU 进程后重新提交训练，Problem 19 因连续 408 超时崩溃退出，同时 Problem 17 出现写入失败——分别排查，定位到两个独立根因
- **Problem 19**：session 反复用逐字节相同的参数调用 `write`（每次都成功），PRM 判官照"成功=正分"规则连续打 +1，`prompt_len` 持续膨胀（23770→24725），最终拖垮 SGLang（503）、网关被 SIGTERM 杀掉。**确认现有"精确重复调用判负分"规则只覆盖 `read`，没覆盖 `write`**，是一个待补的具体缺口
- **Problem 17**：读 OpenClaw 本地源码（`src/agents/agent-tools.params.ts`）确认 `edit` 工具要求 `oldText.trim().length > 0`，而这次模型的 `oldText` 是纯换行符 `"\n"`，trim 后为空，校验必然失败，报错"Missing required parameter: edits"措辞具有误导性（模型看不出真正问题在哪），导致 43 个 turn、24 分钟反复重试同一个注定失败的调用。**这是第三种独立的 edit 失败机制**（不同于 07-27 的 JSON 双重转义、07-28 的通用 status:error 兜底）
- 同时确认已上线规则大部分时候在正确工作（status:error 规则、read 精确重复规则均在这次追踪里正确生效），唯一没覆盖的缺口是"重复失败/成功但无意义的动作本身不会被阻止，只是打分层面兜底"
→ 详见 [`issues_log.md`](issues_log.md) 2026-07-29 条目（含完整技术细节、源码引用、两个 session 的逐 turn 追踪）

**下一步：** 均未实现修复，只是查清楚记录；后续可考虑把"精确重复调用判负分"规则从只覆盖 `read` 扩展到 `write`

### 规则 1a 扩展到 `write`（已实现）；edit 死循环解决方案讨论（三个方向，尚未实施）

**完成内容：**
- `scripts/prepare_patched_openclaw_opd.sh` 的精确重复调用判定从只覆盖 `read` 扩展到同时覆盖 `write`——理由对称：紧邻上一次把同样内容写到同一个 path 没有任何新效果，只是因为 `write` 通常返回成功、之前没被判官正确扣分，是 Problem 19 崩溃的直接对策。本地验证通过
- edit 死循环梳理出三个可能解决方向，尚未实施：(1) 维持现状，接受为已知限制（根因在 OpenClaw 自身工具实现，项目一贯不碰 OpenClaw CLI 本身）；(2) 训练服务端加"熔断"机制打断反复失败的 session（技术可行性存疑，我们的 OPD 服务是被动打分端点，没有主动终止 session 的机制）；(3) 修改 OpenClaw 自身 `edit` 工具的报错信息使其更明确（能从根源缓解，但这是从未做过的一类改动——从未修改过 `openclaw` CLI 工具本身，只改过 `OpenClaw-RL-official` 自己的训练脚本，需要评估是否超出复现范围）
→ 详见 [`issues_log.md`](issues_log.md) 2026-07-29 条目（后续追加部分，含三个方向的完整讨论）

**关键决策：** edit 死循环三个方向，用户选择方向 1（维持现状，本次不改），提交新训练验证 `write` 精确重复规则的效果。

**下一步：** 提交新训练，观察 Problem 19 型"write 重复膨胀拖垮网关"是否不再出现

## 2026-07-31

**目标：** 根据答辩交流反馈，简化训练循环架构图

**完成内容：**
- 图里"凑够 16 条样本，Megatron 反向传播"一条，"凑够 16 条样本"是批次累积的工程实现细节，跟架构本身无关，用户要求去掉，只保留"Megatron 反向传播 / 更新 Policy 权重"这一步

**产出：**
- `docs/personal_agent_dialogue_vs_training_loop.svg`/`.png`：简化训练循环图最后一步的文字

---

## 2026-08-03

**目标：** 修正训练循环图里 Hint/Eval 判官框的歧义表述

**完成内容：**
- 用户指出"Hint 判官（M 票）"/"Eval 判官（另 M 票）"里的"M 票"表述不清楚，容易被误读成"判断 M 次"这种有先后依赖的迭代精炼
- 澄清实际机制：M 票指的是用同一个提示词、同一个模型做 M 次**独立并行**采样，每次都是一次独立打分，最后多数投票决定最终结果，不是反复斟酌
- 用户确认这个表述容易引起歧义，选择直接去掉这两处括号说明，而不是改写措辞

**产出：**
- `docs/personal_agent_dialogue_vs_training_loop.svg`/`.png`：去掉 Hint/Eval 判官框里的"（M 票）"/"（另 M 票）"

## 2026-08-05

**目标：** 核实外部审阅（Cursor 通读 workspace 后的分析）建议是否可信，评估并实施可行项

**完成内容：**
- 用户提供了 Cursor 通读全部 workspace 文件后的分析建议，逐条核实其中的事实性断言（不直接采纳外部工具结论）：`max_turns=8` 只管外层对话轮次（已知）、`--save-interval 100`/`--num-rollout 100000000`（读脚本确认属实）、`student_chat.py` 单题异常无捕获会导致整场终止（读源码确认属实）
- **关键新发现**：OpenClaw 自带 `tools.loopDetection` 配置（`openclaw/src/agents/tool-loop-detection.ts`），默认 `enabled: false`，`criticalThreshold`/`globalCircuitBreakerThreshold` 默认 20/30，触发后在工具执行前直接拦截并提示模型换策略——这是在 rollout 过程中就能掐断死循环的机制，比训练端事后打分更直接；开启方式（`openclaw config set tools.loopDetection.enabled true`）跟项目里已有的 `reserveTokens` 等配置调整是同一类手法，风险较低
- 用户提问引出关键澄清：开启 `loopDetection` 后不能去掉已有的 Rule 1a/1b、status:error 判负分规则，两者工作在不同层面（loopDetection 管"动作会不会被执行"且阈值高达 20 次；PRM 规则管"训练信号该打多少分"且第 2 次重复就生效），是互补不是替代
- 核实 `tools.loopDetection.postCompactionGuard` 子机制：默认启用，但读 `run.ts` 确认它只在"压缩真正成功"之后才会被"上膛"；07-29 记录的 Problem 36 死循环是压缩从未真正成功过（一直被 `already_compacted_recently` 拒绝），这个 guard 从未被上膛，不会介入——确认它解决不了 07-29 那个问题，是另一类失败模式的对策
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-05 条目

**关键决策：** checkpoint/`--save-interval` 暂不调整，Separate-Student 跑得快，当前阶段不需要存档；`tools.loopDetection` 开启 + `student_chat.py` 单题异常捕获确认可行后动手实施

**实现：**
- `scripts/train_separate_student.sh` 新增 `openclaw config set tools.loopDetection.enabled true`（官方默认阈值，不额外调参）
- `scripts/prepare_openclaw_test_scripts.sh` 新增补丁，给 `student_chat.py` 的 `main()` 问题循环外层加 try/except，单题崩溃标记 incomplete 后继续跑下一题
- 本地验证通过：`bash -n`、真实官方源码模拟生成、`py_compile`、确认生成代码缩进和 `results` 列表长度正确

**下一步：** 提交新训练，观察 `loopDetection` 是否真实触发、Problem 17/19 型死循环是否被提前拦截、单题崩溃后训练是否能继续跑完剩余题目

## 2026-08-06

**目标：** 核实 `separate_student_20260805_204436`（commit `8c7ff43`，已含 loopDetection/write 精确重复判负分/单题容错三项补丁）里持续出现的坏行为的根因，重点排查 NO_REPLY 现象

**完成内容：**
- 用户提供了新一轮 Cursor 分析（6 类"种子行为" + 3 类"主要奖励强化成因"），逐条核对哪些行为其实从未真正修过（如 Steps 第 0 条已被撤销、false-completion 链条从未处理过）、哪些只有打分层兜底但不阻止动作本身发生（Rule 2a/2b）、哪些理论上该被 loopDetection 覆盖但需要验证
- 用户提供全量统计数据（n=342 组 thinking_chars/eval_score 配对样本）核查"verbose CoT 被奖励并自我强化"这个假说——数据不支持这个简单结论：`+1` 样本的 thinking 长度分布早晚期几乎不变（6321→6386 字符），只有 `-1` 样本明显变长（5665→9996 字符），且长度与得分是倒 U 型关系（3000-5000 字符区间 +1 率最高，达 73.5%），不是单调"越长越容易被打正分"；顶格截断（`finish_reason=length`）仅占全部 turn 的约 1%，样本量太小，不支持"顶格垃圾内容系统性拿正分"这个说法
- 用户提供 grep 结果确认 `tools.loopDetection` 全程零次触发日志（只有配置生效确认行），核实配置生效时序（`openclaw config set` 在 gateway 进程启动前执行）排除配置未生效的可能，判断更可能是本次训练的实际失败模式（内容不断变化的死循环、或纯文本无工具调用的膨胀）本身不匹配 loopDetection 的"精确 hash 匹配的工具调用重复"设计，而不是配置没生效——**这一点仍待用具体 session 的工具调用序列做最终确认**
- 用户提供 NO_REPLY 专项分析：确认 silent-reply-policy 补丁在本次训练里确实部署生效，但它管的是 OpenClaw 判定"真·空回复"要不要放行，跟模型自己写出非空字面文本 "NO_REPLY" 完全是两条路径，补丁逻辑不会被触发
- 读源码确认 NO_REPLY 根因：`NO_REPLY` 是 OpenClaw 自带真实 token（`tokens.ts` 的 `SILENT_REPLY_TOKEN`），但只有 `buildGroupChatContext()` 会把它写进 system prompt 教给模型，`buildDirectChatContext()` 完全没有；`student_chat.py` 的单会话对话被分类成 `direct`，模型从没被这次训练的 system prompt 教过这个约定，是自发套用了训练外的通用 agent 惯例。这跟 07-21/07-22 就记录过的"NO_REPLY 幻觉"是同一个现象（当时确认过不是新引入的补丁副作用，但没查出根因，一直延后处理）——本次补上了根因
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-06 条目

**关键决策：** NO_REPLY 是模型输出层面的问题，不在任何现有补丁覆盖范围内，照搬本项目"打分层兜底"的既有模式处理，不是从 system prompt 层面禁止（那样等于给策略模型加外挂指引）

**实现：**
- `scripts/prepare_patched_openclaw_opd.sh` 新增规则 3：最终回复内容 trim 后精确匹配 `NO_REPLY`（大小写不敏感，语义对齐 OpenClaw 自己 `isSilentReplyText()`），复用 Rule 1a/1b/2a/2b 共用的 `_is_invalid_tool_use` 标记，强制 `eval_score = -1.0`；下游 `openclaw_combine_select_api_server.py` 消费逻辑不用改
- 本地验证通过：真实官方源码模拟生成 + `py_compile`，正则单独测试确认只命中纯 token 回复，不误伤粘连内容或结尾恰好出现该词的实质性回答

**后续追加（同一天）：** 用户让 workspace CLI 对本轮工具调用做了针对性统计，确认 `tools.loopDetection` 零触发的根因——配置确实生效（网关启动前已 set + verify 回读正确，全程 CRITICAL=0），但本轮实测同 session 内 exact 重复工具调用最大只有 9 次（最凶的 P57 连续 9 次 `read` 同一文件），远低于官方默认阈值 20，且本轮真正的坏行为（超长 thinking 原地复读、短答、NO_REPLY、顶格/空生成）大多根本不涉及"同参数工具调用重复"这个 loopDetection 唯一能检测的模式。结论：**不是配置/时序问题，是本轮失效形态本身在 loopDetection 设计范围之外**，跟"loopDetection 与 PRM 打分规则互补、不是替代"的既有判断一致——偶发的 exact 重复本来就已经被 Rule 1a 在第 2 次就打负分覆盖了，不需要等 loopDetection 的第 20 次。唯一目前完全没有机制覆盖的一类失效是**超长 thinking 原地复读**（无固定工具调用、内容不断变化，loopDetection 看不见、精确匹配的 PRM 规则也抓不住）。
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-06 条目"后续追加"部分

**讨论"超长 thinking 原地复读"时，用户提出一个相关假设并当场拍板修复：** `student_chat.py` 的 `FIRST_MESSAGE_TEMPLATE`（第一轮直接发给 policy、不经 Student 系统提示词过滤的消息）里"Show me the answer first"可能诱导 4B 模型把"answer"理解成"只给最终答案"，而 `STUDENT_SYSTEM_PROMPT` 的重写触发条件只看格式特征（bold/编号列表/`**Final answer**:`），不带格式标记的简短答案不会被打回重写——这条因果链能解释"短答/只给 answer"这类失效为何会被当满足要求放过。给了三个改法（A 加修饰词、B 整句换成强调完整解法、C 全部换成 step-by-step 框架），用户选 **B**。
→ 详见 [`issues_log.md`](issues_log.md) 另一条 2026-08-06 条目

**实现：**
- `scripts/prepare_patched_openclaw_opd.sh`：NO_REPLY 规则 3（见上）
- `scripts/prepare_openclaw_test_scripts.sh`：`FIRST_MESSAGE_TEMPLATE` 把"Show me the answer first"改成"Show me your full solution with all the steps first"，其余文案不变
- 均已用真实官方源码模拟生成 + `py_compile` 验证通过

**"超长 thinking 原地复读"的完整讨论结论：** 用户让 CLI 拉了具体原文样本分析后（P57/58/59 若干 turn 全文），确认重复形态是"跨 turn 状态卡住 + 单 turn 内语义换皮空转"混合，不是简单的逐字复读——跨 turn 的 reasoning 文本彼此并不像（SequenceMatcher 0.02-0.15），该管跨 turn 卡死的是动作/路径重复（Rule 1a 已覆盖），单 turn 内部的换皮空转才是真正没人管的缺口。尝试用"句子精确重复次数"给这个缺口定安全阈值时发现：**全文 reasoning 目前只在两条有偏路径（repeat-thinking/TRUNCATED）下才会被记录**，`+1` 样本里只有 2/190 有全文、还都是顶格误判的毒样本，定不出干净基线。同时确认负分样本长度随训练不降反升（早期 5665→晚期 9996 字符），说明纯打分层的事后惩罚可能不足以压制这类退化生成，生成阶段的直接干预（如 `repetition_penalty`）是需要认真考虑的补充方向，但核实采样现状后（`repetition_penalty=1.0` 中性未启用，训练脚本从未碰过）判断本轮不动它——topk-select 方法依赖 k=4 采样多样性，惩罚调猛了可能伤探索、且力度难以一次调对，何况本轮复读主体是语义换皮而非字面 token 复读，`repetition_penalty` 对此可能偏弱。
→ 详见 [`issues_log.md`](issues_log.md) 又一条 2026-08-06 条目

**决策：本轮只做两项不依赖未知阈值的低风险修复，句重复正式规则和生成阶段惩罚都推迟到下一轮看 shadow 数据再定：**
1. 顶格截断（`finish_reason=="length"`）强制 `eval_score=-1`——解决的是独立的"顶格 +1 污染"问题（全量占比约 1%），不解决复读本身，但便宜精确、顺带清掉了污染阈值校准的 2 条毒样本
2. Shadow 统计日志：对每个 turn 无条件计算并记录 `max_sentence_copies`（同句原样重复次数，跟人工分析同一口径）+ `finish_reason` + `is_invalid_tool_use`，不改 reward，纯 observability，为下一轮定"句子重复 ≥N 次判负分"规则的安全阈值收集无偏全量数据

**实现：**
- `scripts/prepare_patched_openclaw_opd.sh`：新增 `_max_sentence_copies()` helper；`turn_data` 新增 `is_truncated` 字段；新增 `[openclaw-rl-shadow-sentence-repeat]` 日志（放在 `if tool_calls:` 判断之外，确保纯文本 turn 也被记录）
- `scripts/prepare_patched_openclaw_combine_select.sh`：新增 `openclaw-rl-truncation-penalty` 覆盖块，读取 `is_truncated` 强制 `eval_score=-1.0`
- 均已用真实官方源码模拟生成 + `py_compile` 验证通过；`_max_sentence_copies()` 独立测试复现了 CLI 报告的原始数字（×17→17，×5→5）

**下一步：**
1. 提交新训练，验证 NO_REPLY 规则 3、`FIRST_MESSAGE_TEMPLATE` 改动、顶格截断规则是否生效，并收集 `[openclaw-rl-shadow-sentence-repeat]` 全量数据
2. 数据回来后：用干净的 +1/-1 分布定"句子重复 ≥N 次"规则的安全阈值 N，再评估是否要加生成阶段的轻力度 `repetition_penalty`
3. CLI 提出的编排类杠杆（同 session 进批占比上限、Rule 1a 命中后提前结束/降权、Student 侧无进展熔断）——方向合理，本轮不做，待前两项效果明确后再评估

**Rule 1a 通用化：** 用户认为"只对 read/write 特判"不够普适，讨论后确认"工具重复"和"句子重复"本质是同一件事（内容/动作已出现过、无新信息），Rule 1a 的正确性前提对所有工具都成立，不是 read/write 特有性质。让 CLI 核实了作业环境实际用到的工具（`read`/`write`/`edit`/`exec`/`message`/`web_search`/`sessions_*`），唯一有真正轮询语义的是 `process`，予以豁免（连同 `args.action=="poll"` 的防御性覆盖）。**明确了这一步的边界**：只解决"紧邻同参重复、exact-match 可测"这个子类，覆盖到之前遗漏的 `edit`/`sessions_*`/`message`，但解决不了本轮长负样本主体的"换皮空转"（跨 turn 文本相似度 0.02-0.15，逐字比对根本抓不住）——那部分仍需要真正的"无状态进展"过程信号（Rule 1b 精神推广到全工具，本轮不做）或生成阶段轻量重复惩罚（已推迟），是互补关系不是替代。全历史（非紧邻）版本的通用化会有假阳性（中间发生过状态变化的重复调用会被误判），本轮只做"紧邻上一次"这个安全范围内的通用化。
→ 详见 [`issues_log.md`](issues_log.md) 又一条 2026-08-06 条目

**实现：** `scripts/prepare_patched_openclaw_opd.sh` 新增 `_is_poll_style_call()` helper；Rule 1a 判断条件去掉 `_tool_name in ("read", "write")` 限定，改为对所有工具生效、豁免轮询类调用。已用真实官方源码模拟生成 + `py_compile` 验证通过，独立测试确认豁免逻辑行为正确。

## 2026-08-07

**目标：** 核实新训练（`separate_student_20260807_104044`，commit `bf52f07`，已带上 08-06 全部修复）出现的"格式癫痫 + 拒绝写文件"新失效模式，定位诱因并修复

**完成内容：**
- 用户让 CLI 分析这次新训练，发现从 ~P30 起出现此前没见过的失效模式：满屏 `**`/emoji/表格、嘴上说不加粗手上全是加粗，P40-45 更严重，出现拒绝调用 write/edit、要求 Student 先念指定确认话术才肯写文件
- 对比 08-05 旧训练同号段（用的是 `FIRST_MESSAGE_TEMPLATE` 旧措辞）确认没有这个现象，坐实这是 08-07 新训练才出现的新问题，不是旧数据漏看的角度
- 五项核实（详见 issues_log）：拒写具体表现（P42 turn8 明确拒绝+要求确认话术）；Student 重写检测触发率随对话推进大幅下滑（P0-20 的 100% → P35-45 的 26%）；格式癫痫 turn 约 37% 仍被打 +1（打分漏检，不只是学不会）；拒写发生在对话末尾时完全不进入 PRM 打分（无 next_state，训练信号空白）；"拖延逼妥协拿正分"这个闭环基本没形成，更常见的有毒路径是 turn1 本身就已经在拿 +1
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-07 条目

**诱因定位：** `FIRST_MESSAGE_TEMPLATE` 08-06 改动用的方案 B（"Show me your full solution with all the steps first"）力度过重，把 `STUDENT_SYSTEM_PROMPT` 里本就存在的"完整步骤 vs 不能像 AI"张力显著加重，且这次训练里 turn1 应答几乎全部被判 +1，等于从一开始就在奖励这种"高调宣布要完整合规"的表演。

**实现：** `scripts/prepare_openclaw_test_scripts.sh` 把 `FIRST_MESSAGE_TEMPLATE` 改回方案 A（"Show me the full worked answer first"，只加两个词，不用"with all the steps"这种更重的表述）。已用真实官方源码模拟生成 + `py_compile` 验证通过。

**下一步：**
1. 提交新训练，验证改回方案 A 之后 P30 往后是否还会出现同样的"格式癫痫+拒写"现象
2. 另外三个真实缺口（跟 FIRST_MESSAGE_TEMPLATE 无关，是训练机制本身的空档，明确记账待后续处理）：
   - Student 重写检测触发率随对话推进大幅下滑，原因待查
   - 满屏格式违规仍有约 37% 被打 +1——可以考虑加一条确定性 PRM 规则，直接在响应文本上用正则检测 Table 3 收敛判据用的三个具体特征（bold/编号列表/`**Final answer**:`），不依赖 Student LLM 是否恰好抓到
   - "拒绝调用 write/edit、要求确认话术"这类该行动却不行动的失效，目前没有对应检测规则，且发生在对话末尾时会完全逃出 PRM 打分

**后续追加（同一天）：精确定位真正起始点。** 用户让 CLI 逐题核实时间线，确认真正起始点是 **P28**（不是之前粗略以为的 P29/P31）——P28 第一次出现"只甩空 Solution + 要求 Confirm、完全不解题"且被判官全体打 +1（第一次打错）；P31 是"假确认→空 write 也被判 +1"的最脏放大器，不是起点。确认是**两个必要因子叠加**：(1) `FIRST_MESSAGE_TEMPLATE`（已定位并改回方案 A）诱导出"先门禁、先确认"的行为种子；(2) **PRM 判官把"催解题/假 approve"这类 Student 话术系统性判成 +1，这条打分漏洞目前完全没修**，且 CLI 认为这条比软化措辞更接近根因。决策：本轮先单独测方案 A，作为单变量实验——如果 A 单独解决问题，说明行为种子是主因；如果 A 之后同类问题仍换个触发方式出现，就得单独修"demand_solve/假 approve 不该判 +1"这条打分逻辑。
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-07 条目"后续追加"部分

**再追加（同一天）：STUDENT_SYSTEM_PROMPT 的 Steps 重构。** 讨论后决定不该让 `FIRST_MESSAGE_TEMPLATE` 独自扛"防裸答"这个职责，改成让 Student 每一轮持续检查。核心设计：Step 1 拆成两层独立判断（先判断"是否是真答案"——没回应/裸答案/tool-call 空壳，命中直接要求真答案，不提风格；只有确认是真答案才轮到原有的 AI 味格式判断），并区分"解题阶段"和"写入确认阶段"（写入阶段看到 tool call 是正常预期，不能套用解题阶段的判断标准）；Step 3 新增"核实是否真的写了文件"，覆盖官方原始版本完全没预期到的"AI 拒绝写入"情况。已实现、验证（含直接打印运行时字符串确认渲染正确），是主动设计的新机制，效果需要真实训练数据验证，**明确记录了回退判断标准**：如果没有改善"格式癫痫+拒写"，或者新判断层本身出现误判（比如正常简短回复被误判成"没有真答案"），回退到 07-29 版本。
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-07 又一条条目

**再追加：让 CLI 复核 Steps 草稿，收紧两处真实漏洞。** (1) Step 3 的"写入确认"判定太松，泄文场景下一段 tool-call 样式的原始 JSON 容易被误判成"已写入"，收紧为只认大白话确认；(2) 新增"不要假装自己已读过文件/解出了什么"，堵角色串戏（真实数据里见过 Student 说"Got it, I read..."这种话）。另有两点是"限定说明"不是文字改动：这段 prompt 是软刹车不是格式通道硬约束，KL≈0 时模型仍可能在别处漂出假 tool 格式；Tier-0 判断仍靠 Simulator LLM 自身判断，边界情况不能保证 100% 准确——评估这次修复效果时需要一并考虑，不能只看有没有生效就下"彻底解决"的结论。已实现、验证，跟主条目共用同一回退标准。

**再追加：新一轮训练报了两个新问题，均已修。** (1) P17 实锤（session `a0b1e908`）：模型没写 `homework/17.txt`，反复写入/读取旁路文件 `17_solution.md`，因 read/write 交替逃过 Rule 1a 的紧邻精确匹配检测，5 条 +1、38 条 -1 全部提交进训练——新增**规则 4**：用 `session_id` 里编码的 problem_index 算出唯一正确路径，`write`/`edit` 目标路径不符就判负分（只对 `student_chat.py` 的 session 命名惯例生效，不影响 TA/Teacher）。这条不解决"交替读写但路径都对、只是没进展"这个更通用的问题，那个仍是待办。(2) Steps 的 Tier-0"是否真答案"判断真实数据里偏严格，带过程的回答也偶尔被误判——收紧改回宽松版，显式声明"哪怕简短不完整的推理也算已回答，只有真的完全没过程才触发"。
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-07 又一条条目

**再追加：撤销 FIRST_MESSAGE_TEMPLATE 补丁，完全恢复论文原始措辞。** Steps 重构的 Tier-0 检查已经把"是否给出带过程的真答案"做成每轮持续生效的机制，开场白不再需要单独扛这个职责；继续保留"full worked"这类强调只是重复限定，而且是"格式癫痫+拒写"的已证实诱因之一，纯下行风险。撤销后 `FIRST_MESSAGE_TEMPLATE` 完全等同官方原始文本，复现忠实性上净收益。
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-07 又一条条目

**再追加：规则 4 从写下来就是死代码，已修复。** 让 CLI 复核规则 4 时发现，训练日志里 `session_id` 全程是 UUID，不是当初假设的 `student-hw-{index}-{pid}`——根因是 RL-TRAINING-META 标记里的 `ctx.sessionId`（OpenClaw 自己的 UUID）在 `session_id` 派生优先级链条里排在 Runtime-line fallback 前面，规则 4 对 UUID 跑正则永远匹配不上，**部署以来从未真正拦截过任何旁路写入**。CLI 用真实落盘数据核实了修法可行性（Runtime 行每个 main 作业轮都有、`_RUNTIME_SESSION_RE` 能精确解析出 `student-hw-N-pid` 不被截断）后，改成规则 4 检查点自己单独调用已有的 `_extract_session_id_from_system_prompt(messages)`，不碰全局 `session_id` 派生逻辑（其他规则依赖它当稳定 key，UUID 够用，不该为这一条规则改全局行为）。
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-07 又一条条目"后续追加"部分

**再追加：新一轮训练（run `20260807_183828`）暴露"超长 thinking 空转顶垮训练"，新增规则 5。** thinking 从 P18-27 持续膨胀（首答 thinking_chars ~2.5k→~8.5k），P28 起系统性溃败，`couldn't generate` 大面积出现；定位到具体奖励漏洞——顶格截断已有 truncation-penalty 稳定判 -1，但"巨长 thinking 后摸到工具、`finish_reason=tool_calls`"这类样本仍能拿 +1（P16/19/23 实锤）。分四轮用真实 shadow 数据校准阈值：确认三条泄漏样本 `max_sentence_copies` 为 12/21/30；确认 `length/-1` 组 68% `copies=1`（句子重复规则对这批冗余，truncation-penalty 已覆盖）；确认候选阈值 N=12 对干净对照组 `stop/+1`（n=34）零误伤、三条泄漏全部命中，N=8 会误伤 1 条真实长解；核实 `tool_calls/+1` 里不存在"低 copies 高 thinking"的漏网形态，不需要额外规则。**定阈值 N=12，新增规则 5**：句子原样重复 ≥12 次强制判负分，复用已有的 `_max_sentence_copies()`，位置在 `tool_calls` 判断之外（复读发生在 reasoning 本身，跟有没有调用工具无关）。已实现、验证（含阈值边界行为测试：copies=12 触发、copies=9 不触发，精确对应真实边界案例）。P1（顶格/高复读样本直接不进批次）、P2 后半段（连续失败提前结束 session，本质是熔断，跟之前"熔断没有普适性"的判断冲突）、P3（尝试非零 KL）均讨论后明确不做，留待后续。
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-07 又一条条目

## 2026-08-10

**目标：** 08-07 晚些时候提交的新训练（run `20260807_183828`，已带上规则 5）暴露"超长 thinking 空转顶垮训练"问题，定位上游根因并修复

**完成内容：**
- 训练结果显示 thinking 从 P18-27 持续膨胀（首答 thinking_chars ~2.5k→~8.5k），P28 起系统性溃败，`couldn't generate` 大面积出现——规则 5 已在打分层拦截高复读样本，但这不解决"复读为什么会发生"这个更上游的问题
- 讨论了几种生成侧/物理层面的对策（降 maxTokens、采样侧 frequency/presence/repetition penalty、流式中途熔断、独立 thinking 预算），确认 `frequency_penalty`/`presence_penalty`/`repetition_penalty` 分别是"按出现次数线性扣分"/"出现过一次就固定扣分"/"按比例缩放 logit"三种不同机制，各有局限（都是 token 级信号，对"语义换皮"型复读效果有限）；决定这轮暂不动生成侧参数，先看 Student 侧修复的效果
- 让 CLI 深挖复读的**上游**根因（不只是打分层怎么补救），定位到具体因果链：Tier-0 判断经常把已经有紧凑算式的正确回答（如"5×5=25、4×10=40、40-25=15"）误判成"没有真正回答"（约 35% 的"要完整步骤"请求打在已有算式的回复上，P16/P17/P20 等题实锤）；提示词里写死的"but keep all the steps"（顶部 criteria 段落 + Step 1"otherwise"分支各出现一次）逼着已经通过 Tier-0 的回答在改语气的同时继续"加内容"；两者叠加形成"加长→重写→再加长"的两拍循环，thinking 里反复打磨口语化草稿（同一句"Let me try to break it down..."类型的句子重复 16-25 次），最终烧光 token 预算
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-10 条目

**方案（三处文字改动，缺一不可）：** Tier-0 判断补一句"紧凑算式也算有过程，不需要写成散文"；Step 1"otherwise"分支去掉"keep all the steps"，改成明确"别要求加内容"；顶部 criteria 段落同一句话一起改（只改一处会被"捡回去念"）。CLI 复核确认方向对、三处必须同发，且指出两处需要留意但不阻塞的边角：顶部"必须包含完整步骤"那句先不动（跟"紧凑算式已够"有轻微张力，观察即可）；验收标准是"假 Tier-0 触发频率下降"，不是降到零（32B/Simulator 指令遵循本来就不稳）。

**实现：** `scripts/prepare_openclaw_test_scripts.sh` 三处改动。已用真实官方源码模拟生成 + `py_compile` 验证通过；按 CLI 要求的验证方式，用 `ast` 解析打印完整运行时 `STUDENT_SYSTEM_PROMPT` 字符串，确认"keep all the steps"全文出现次数为 0（不只是检查改的两处），确认没有因为拼接方式产生重复文字。

**下一步：**
1. 提交新训练，观察假 Tier-0 触发频率是否下降、"加长→重写→再加长"两拍循环是否减少、thinking 增长趋势是否放缓、P28 式整段死亡是否消失
2. 明确不解决的问题（留待后续）：后半程 tool_call XML 元循环二次崩溃（假 Tier-0 之后的独立下游症状，预期会因上游触发频率下降而减少，但不是这次改动的直接目标）；生成侧物理限位（frequency_penalty 等，等这轮 Student 侧修复效果明确后再评估要不要加）；demand_solve/假 approve 判分漏洞（08-07 发现，仍未修）

**再追加：训练图重排（commit `ef457b3`）。** 把"两种训练顺序"和"作业文件在角色间传递"从图片底部移到顶部，homework/homework1/homework2 三个框的横向位置精确对齐 Student/TA/Teacher，一眼能看出对应关系。

**再追加：新一轮训练仍见误判，Step 1 五次修订，补上"只有答案没过程"这条此前完全没有的检测。** 跟用户和 CLI 逐轮核实：真实问题方向是"过度触发"（有过程的正确答案被误判成没答/只有答案），不是漏判；但用户澄清"没有真答复"和"只有最终答案没过程"这两种情况**都**真实导致过 Simulator 乱回复训坏模型，两条检测都要留，只是必须做到对正常回复零误触发。核实顶部"must include full solution process"是论文原文、从未被本项目改过。设计"只有答案没过程"的判断依据时，最初想用"有没有运算符号"，被用户用真实反例否掉（大白话叙述过程"4 guppies, plus 2 he bought, so that's 6 guppies"全程无符号，会被误判），改成"整段数字个数是不是只有一个"这个更通用的信号。**Step 1 从两分支改成三分支**（①有没有真答案，正向枚举证据 → ②是不是只有答案没过程，新增 → ③AI 味重写），Step 2/3（写入请求+核实）明确维持不动——用户要求保留，这次训练数据显示这部分工作良好。
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-10 又一条条目

**再追加：新训练里发现的 P54 型偶发死锁（催写入后反复 read 不写，context 撑爆无法恢复），逐层核实原因和对策，本轮不改代码，决定重装环境重新训练。**

- **CLI 提供的诊断（真实数据，P54）**：Turn 3 用户要求 append 后，模型没有写入，反而对同一文件反复用相同参数 `read`（叠加 408 超时），Turn 4 起 Student 已经在正确地说"That doesn't count — really write"，但此时 context 已经撑爆，后续几轮催写全部只能收到 overflow 文本，会话已死、无法在同一 session 内恢复。
- **两个关键澄清**：(1) 同 session 内"再催写入"只在 context 还没撑爆时有效（P55/56 证明这种情况下硬纠偏能成功），一旦真的 overflow 就无解，必须弃题/换 session；(2) 训练污染的真实形态不是"40 次 identical read 全部拿正分"（Rule 1a 已经能在第 2 次起判负），而是"催写入后**第一次**去 read（而非 write）仍会拿 +1"（Rule 1a 对首次出现无效）叠加"死会话产生大量 -1/超时坏轨迹拖累同批次的其他题目"。
- **CLI 给的四个杠杆**（A 编排熔断 / B 降低 loopDetection 阈值 / C 补"首次空转 read 仍拿 +1"这条打分漏洞 / D 调 reserveTokens）：**本轮全部不实施**，用户决定先看这轮训练结果。特别记一笔：**方案 A（熔断）本质上是之前"超长 thinking 空转"讨论时用户明确拒绝过的同一类做法**（"熔断没有普适性"），这次场景（context 真正物理性撑爆、同 session 内确认无法恢复）跟当时"担心熔断被当成学习问题主要解法"的顾虑不完全一样，但要不要为这种"确认不可恢复"的极端场景破例，仍需要用户自己决定，没有默认采纳。
- **讨论"能不能给 4B 具体指令，直接要求别再读、直接写"**：设计了两处措辞（Step 2 写入请求时预防性加"不用再读了，直接写"；Step 3 检测到"读了而不是写"时给具体纠偏话术"don't read the file again, just write/append it now"）。用户为避免影响其他正常跑通的题目，主动把范围收窄成只改 Step 3（检测到偏差后的纠正），不碰 Step 2（首次写入请求，现状对多数题目工作正常）。**最终这处收窄版也决定暂不实施**，先看这轮训练结果。
- **核实"Step 3 收紧是否导致 4B 一开场就念文件内容确认"这个疑虑**：架构上确认不可能——`STUDENT_SYSTEM_PROMPT`（含 Step 3）只喂给外部 Simulator（32B）自己判断要不要接受，`send_to_openclaw()` 实际发给 OpenClaw/4B 的只有 `messages: [{"role": "user", "content": student_msg}]`，Simulator 的系统提示词 4B 从头到尾看不见，这条因果链走不通。"念文件确认"这个行为的真实来源是论文原始的 Step 2 措辞（"not overwrite it"，07-29 至今从未改过）+ OpenClaw 自身脚手架习惯 + 预训练模型爱把任务完成过程讲清楚的倾向，三者都不是这次会话任何一次补丁引入的。
- **最终决定：不改代码，让 CLI 把 OpenClaw 和 4B 模型都清理回全新安装的干净状态，重新跑一轮训练看效果。**
→ 详见 [`issues_log.md`](issues_log.md) 2026-08-10 又一条条目

---

## 2026-08-11

**目标：** 修复 Simulator 在 Step 1 ②③上的真实误判；解决规则 5"罚了但没教清楚"的信用分配问题；追踪 turn-1 不收敛的真实瓶颈

**完成内容：**
- 规则 5（复读判负分）连续两轮训练仍高频触发。诊断为信用分配太糊——负分打在整拍，模型学不到具体是哪句话复读——不是阈值问题。走 OPD 现成的 hint 条件化 teacher 机制，命中时把该 turn 的 `accepted` hint 整体替换成写死的复读提醒（新增 `is_repeat_thinking_violation` 标记，commit `d4a584f`）
→ 详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-08-11"规则 5...连续两轮训练仍高频触发"条目
- 新训练暴露 Step 1 ②③三个真实误判样本，Simulator 未严格执行已写清楚的判断标准。加数数字 CoT 步骤 + 针对性反例锚点（Step 1 六次修订，commit `8f90615`）
→ 详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-08-11"Step 1 ②③真实误判样本"条目
- 同日晚些时候又出现同一类误判（P17），判定②这条判断对当前 Simulator 不可靠，用户决定整条删除②，七次修订（commit `5f69bc4`）——**当晚已撤销**，见下
→ 详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-08-11"Step 1 七次修订"条目
- `separate_student_20260811_170852`（commit `d4a584f`）30 题完整数据分析：**policy 侧已经学会给出合格 turn-1（P17 起分界清晰），但 Simulator 几乎全部误判打回（10/11），③型误判比②更多，还出现规则外编造的反对理由**——推翻了"训练量不足"的早先猜测，改判为 Simulator 本身系统性偏严；为让代码基线对齐这批数据，用 `git revert` 撤销七次修订（保留六次修订——170852 实际运行的 `d4a584f` 快照本身含六次修订）。撤销过程中一度误把六次修订也撤销掉，用户看 `git pull` 结果发现改动量不对当场指出，已修正，`md5sum` 核对确认当前文件与 `d4a584f` 快照哈希完全一致
→ 详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-08-11"170852 完整数据分析"条目
- 手动核对 `separate_student_20260811_141207`（commit `edd247b`）：72 题完整跑完主动停训，done 68/incomplete 4/couldn't-generate 24/overflow 11，OPD+RL +1/-1=113/129，update_weights≈21 次，是 pipeline 健康度最好的一次，记作参考基线（早于本次会话改动，仅供后续对照）

**关键决策：** turn-1 不收敛的真实瓶颈是 Simulator 系统性拒绝合格答案，不是训练量不足——170852 数据推翻了早先的猜测。下一步方向待讨论（候选：全局"默认接受、除非能具体举证"框架调整 / 重新评估 DeepSeek V4 替代模型的校准问题 / 核实误判是否在污染 PRM 打分）

**产出：**
- `scripts/prepare_openclaw_test_scripts.sh`：Step 1 六次修订（当前生效）+ 七次修订（已 revert，历史保留）
- `scripts/prepare_patched_openclaw_opd.sh` / `scripts/prepare_patched_openclaw_combine_select.sh`：规则 5 OPD hint 纠正信号
- `docs/personal_agent_self_evolution_concept.svg`/`.png`：新增自进化理念图（开篇理念图，跟实验细节图配套，两栏对话循环/训练循环结构）

---

## 2026-08-13

**目标：** Step 1 八次修订上线验证准备；排查 408/503 污染训练信号的完整链路，落地 Wave 1（A+B）+ Wave 2（D）

**完成内容：**
- Step 1 八次修订：Step 1 开头加"默认接受"框架，对冲多步骤结构诱发的挑刺倾向（commit `ae48df2`）
→ 详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-08-13"Step 1 八次修订"条目
- 完整排查 408/503 如何污染训练信号，定型 A（SGLang abort）/B（暂停期间生成）/C（网关判断与 SGLang 实际状态时序错位）/D（重复 user 重试指令当 next_state）四类。中途否定了"整个 run 因 408 而不可信"的早期假设——P17 真正做完题的 `+1`（write 成功）就发生在重试之后，按 run_id 整体拉黑会连坐好样本；进一步查明 P17 根本不是"等训练超时"类型，是模型 `edit` 工具用错精确匹配语义反复空转拖出的 408，是独立的第三类问题（潜在 Wave 3，本轮不实现）
- **Wave 1（A+B）落地**：新增 `is_aborted`/`generated_while_paused` 标记 + 新建 `prepare_patched_openclaw_combine.sh`（此前这个文件——训练实际 import 的 `_maybe_submit_ready_samples` 所在地——完全没有补丁脚本，是排查中发现的真实"补丁打错文件会静默失效"陷阱）+ `PATCHED_COMBINE_DIR` 全链路接入四条训练脚本（commit `799c500`）
- **Wave 2（D）落地**：不按 run_id 拉黑（会连坐 P17 的成功 write），改成单 turn 本地检测——`_fire_opd_task` 里比对剥离 OpenClaw 时间戳前缀后的 `next_state` 与已见 user 消息集合，命中则丢弃（不强制 `-1`，因为内容本身可能完全正常，错的是配对关系）（commit `b28968a`）
→ 详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-08-13"408/503 污染训练信号完整排查"条目
- 用 `separate_student_20260813_094000` 真实提交记录（`submitted OPD+RL`/`submitted RL` 日志）逐类核对是否真的造成训练标签污染，不只看机制：**只有 A 是实锤的标签污染**（至少 3 条错 `+1` + 1 条乱码内容进 OPD 蒸馏）；B 机制真实但这轮未捕获独立错标案例；D 独有增量这轮全部已经是 `-1`，价值是减少劣质蒸馏/占坑而非纠正错误方向；C 这轮没有独立样本；P17 的 `edit` 误用**没有标反**——失败 edit 正确拿到 `-1`、成功 write 正确拿到 `+1`，不是训练标签事故，只是同质失败样本效率低 + OpenClaw 展示层成功写入后仍挂旧的 `lastToolError` 警告（产品层 bug，不在本项目补丁范围）
- **"skip forced negative override" 诊断实验**：对比训练成功的 170852 和失败的 160713/143003，diagnosed 出 async batch 组成的"batch2 race"机制——"超长 `-1`"样本（PRM 原判 `+1` 但因工具决策空转被规则 5/截断惩罚强制改判）碰巧挤占某个训练 batch 时，loss 会被拉向"惩罚长度"而不是预期的格式信号。不提交（不是翻回 `+1`）`_original_eval_score` 为 `+1` 但最终 `eval_score` 为 `-1` 的样本，测试能否降低这种批次污染复现概率（commit `c6c9ebb`）
→ 详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-08-13"skip forced negative override 诊断实验"条目
- **Step 1 九次修订**：`20260813_161745` 真实数据显示 5/27 客观合格的 turn-1 全部被误判——新发现两类"编造清单外理由"：②结尾单独一行"Answer: X"被当成"只有答案"（未数整个回复）、③把数学符号本身当成 AI 味理由。补两条反例锚点，顺带核实 `"**Final answer**:"` 是论文原文非本项目引入（commit `a60b959`）
→ 详见 [`issues_log.md`](openclaw-rl/docs/issues_log.md) 2026-08-13"Step 1 九次修订"条目

**关键决策：** C 和 P17 的 `edit` 误用问题（潜在 Wave 3）本轮均不实现——真实数据核实后确认都不是当前紧急的训练标签污染源，A 才是唯一必须修的

**产出：**
- `scripts/prepare_openclaw_test_scripts.sh`：Step 1 八次修订、九次修订
- `scripts/prepare_patched_openclaw_opd.sh`：`is_aborted`/`generated_while_paused`/`is_duplicate_user_retry` 三个标记
- `scripts/prepare_patched_openclaw_combine.sh`（新建）：Wave 1+2 共用的 `_maybe_submit_ready_samples` 拦截点 + skip-forced-negative-override 独立检查点
- `scripts/prepare_patched_openclaw_combine_select.sh`：`_original_eval_score`/`skip_forced_negative_override` 计算与传递
- `scripts/run_openclaw_topk_select_modelfactory.sh` + 四条 `train_*.sh`：`PATCHED_COMBINE_DIR` 全链路接入

---

## 2026-08-14

**目标：** OpenClaw-RL Separate/Personal Agent Track 复现大体完成，启动下一阶段——把复现出来的 Hybrid RL 训练方法迁移到 MetaClaw（论文 arXiv:2603.17187）场景上做泛化性新实验

**完成内容：**
- 理解 MetaClaw 论文核心机制（skill-driven 快通路 + opportunistic policy optimization 慢通路），`git clone` 官方代码到 `MetaClaw-official/`，初步架构对照
- 确认迁移目标（不是复现 MetaClaw 论文结果，是用已校准的 Hybrid RL 方法在 MetaClaw-Bench 上做新实验证明方法普适性）、范围（只迁移 RL/OPD 训练方法本身，不接技能库）
- 查证 MetaClaw 官方 RL rollout 架构、`toolcall-rl`/`swe-rl`（OpenClaw-RL 论文 General Agent 赛道）的多步 tool-call 处理方式，三处早期设计假设被真实代码核查推翻并改正：OPD hint 不能直接用静态 `feedback.incorrect`（改用 checker 实际 stdout）；MetaClaw 官方"在线更新"是离散的暂停-同步训练-恢复机制、不是真并发（我们现有连续异步管线本身更强，不需要模仿）；跨天没有任何工作区/状态持久化（每天全新隔离复制，唯一能带教训跨天走的是模型权重本身）
- **round 内多轮 tool-call 训练信号设计**（本阶段耗时最长的核心问题，两篇论文都没有现成答案）：核查 `toolcall-rl`/`swe-rl` 的"自己控制生成循环+一个 Sample 共享 outcome reward"架构后发现要求放弃真实 `openclaw agent` CLI（会损失真实 `"coding"` 工具画像保真度）；核查 MetaClaw 自己怎么给 skill 演化提供中间态反馈后发现真正的技术约束不是"编排层看不见中间轮次"，而是"代理评估触发得太快"；最终定型方案 B：round 最终轮次用确定性 checker 结果，中间轮次用新写的、仿 `toolcall-rl::_judge_step_with_prm` 的任务无关步骤判官独立打分，不聚合进 round reward
- 落地第一批实验文件并本地验证（无 GPU，对着真实官方代码克隆验证 import/查询/checker 执行/`py_compile`）：新建 rollout driver，直接复用官方工作区隔离/网关/真实 CLI/inline scoring 函数；扩展两个已有代理 patch 脚本接入 verdict/步骤判官三路分派。验证过程中发现并修复两个真实 bug：漏了 `_copy_eval_scripts` 会导致 checker 恒定"文件不存在"、hint 一开始接错成静态 `feedback.incorrect`

**关键决策：** 迁移工作的详细设计过程和技术决策记录在 [`metaclaw_migration_plan.md`](openclaw-rl/docs/metaclaw_migration_plan.md)，不写入 `issues_log.md`——`issues_log.md` 是 OpenClaw-RL 复现本身的问题追踪文档，迁移是另一条独立工作线，用独立文档承接

**产出：**
- `docs/metaclaw_migration_plan.md`（新建）：迁移方案、查证记录、已实现清单、下一步任务，全部细节见此文档
- `scripts/metaclaw/metaclaw_rollout_driver.py`（新建）：day01→day30 严格顺序 rollout driver
- `scripts/prepare_patched_openclaw_opd.sh`：`_METACLAW_SESSION_RE`、`_build_metaclaw_step_judge_messages`、`turn_data["metaclaw_round_mode"]`
- `scripts/prepare_patched_openclaw_combine_select.sh`：`_opd_evaluate` 三路分派（verdict / 步骤判官 / 原逻辑）

## 2026-08-17

**目标：** MetaClaw 迁移从"方案+第一批文件"推进到"训练前准备就绪"——补齐启动脚本、核查训练信号安全性、定训练规模/checkpoint 策略，为实际提交训练做最后准备

**完成内容：**
- 确认训练起点为干净 base Qwen3-4B（不接 Personal Agent Track 训完的 checkpoint），避免"方法本身行不行"和"旧 checkpoint 是否已定型"两个因素纠缠
- 发现并补上一个真实前置依赖：`_METACLAW_SESSION_RE` 按 `session_id` 分派机制依赖 `rl-training-headers` 插件全局启用（系统级部署，不是某个 `openclaw.json` 里的配置），启动脚本必须复用现有训练脚本里部署这个插件的步骤，否则整套 round-mode 判定会静默失效
- 新建 `scripts/metaclaw/run_metaclaw_migration_modelfactory.sh`，对标 `train_separate_student.sh`——比 Personal Agent Track 少一整段：MetaClaw 不需要外部 Simulator
- **发现并修复真实漏洞**：`openclaw agent` CLI 子进程失败（网关故障/超时）时，原实现会把这次纯环境性故障当"模型任务失败"提交 `eval_score=-1` 进训练。参照 OpenClaw-RL 自己的 General Agent 赛道（`toolcall-rl`/`swe-rl` 都显式设置 `Sample.Status.ABORTED` 并在 Sample 到达 `reward_func` 之前提前返回）翻译到我们的架构修复
→ 详见 [`metaclaw_migration_plan.md`](openclaw-rl/docs/metaclaw_migration_plan.md)"已实现（续，2026-08-17）"
- **系统性核查 training-signal-safety**（四项：基础设施故障保护/步骤判官质量校验/A-B-D 环境降级规则/提交失败默认丢弃），逐项对照两篇论文有没有处理过。核心发现：四条 General Agent 赛道完全没有"重复 user 重试"检测——不是没想到，是架构上不存在这个失败模式（不走 HTTP 重试循环），据此确认 A/B/D 规则在 MetaClaw 场景下不需要调整（D 规则空转但无害）
→ 详见 `metaclaw_migration_plan.md`"查证记录（四）"
- 深挖 OPD hint 蒸馏机制的独特性：确认四条 GA 赛道 + MetaClaw 全部没有对应机制（MetaClaw 自己也有一个叫"OPD"的机制，但是跟独立更强教师模型做标准知识蒸馏，不含纠正性文字，跟 Personal Agent Track 的 hint-conditioned 自蒸馏是同名不同实）；确认 skill 库机制是梯度无关的，论文原文明确"zero weight update"，跟 OPD hint 不是同一类东西
- 写出三方对照表（迁移方法各环节 vs MetaClaw 自己 vs OpenClaw-RL tool-call），逐环节列出跟谁一致跟谁不同，核心结论：没有一个环节是整体照搬某一方
- 讨论中间轮次要不要补确定性锚点（照抄 toolcall-rl 的 `base_score+prm_step_mean` 组合公式）：技术上可行，但依赖一个未经真实验证的假设（round 边界=子进程边界、两个 round 之间无杂音请求），暂缓，留给真实环境验证后再定
- 训练规模/checkpoint 策略：确认一遍 30 天（不循环多个 epoch）跟 MetaClaw 自己实际训练方式一致（`--scene-per-train` 默认禁用，代码里没有 epoch 循环）；确认 checkpoint 是完整独立可用的模型快照，新增 `METACLAW_MIGRATION_PROFILE=1` 把 `--save-interval` 从官方默认 100 调到 10，目标一遍存约 5 个 checkpoint 供中途观察训练进度（粗估算，待真实训练后校准）
- 断点续跑：先按天粒度实现了一版，追问后直接读 MetaClaw 官方 `_run_one_test` 代码，发现它自己的 round 级"断点续传"如果真用于跨进程重启恢复，会撞上同样的 workspace 不一致问题——不是它解决了我们没解决的问题。最终决定崩溃后直接从 day01 完整重跑（一遍训练总耗时有限），撤回断点续跑机制
- 新发现一个跨 round 污染 bug（agent 在非最后一个 round 中途崩溃时，挂起轮次会被下一个 round 内容误评估），需要真实训练日志才能判断触发频率、值不值得精确修，记录后暂缓
- 查证 MetaClaw 自己用的 OpenClaw 版本：确认不锁定具体版本（插件 README 明确写"OpenClaw (any version)"，`package.json` peerDependencies 是通配符 `*`），论文提交时间 `arXiv:2603.17187v1 17 Mar 2026`，落在项目已有的 `march_2026_3_8` 版本基准附近，没有证据指向作者用的是更晚的版本
- 讨论要不要为 MetaClaw 关掉现有 5 个系统级版本漂移补丁先测试：逐个重读补丁根因说明，确认全部是 OpenClaw 核心机制层面的通用 bug（上下文压缩、系统提示词段落、silent reply 策略），不是 GSM8K 场景专属，MetaClaw 文件密集型任务只会更容易触发这些 bug——决定全部保留，不做关闭测试
- 修正 modelfactory 仓库路径大小写：CLI 确认服务器上只有大写 `/dfs/data/openclaw-rl-project/OpenClaw-RL`，`reproduction_guide.md`/`train_separate_student.sh`/`train_with_services.sh` 三处沿用的小写路径是错的，全部改回
- 用户在 modelfactory 完成两项训练前准备：`MetaClaw-official` 浅克隆（`--depth 1`，TLS 中断后改用）、`openclaw-rl` 仓库 `git pull` 同步到最新
- 新增 `METACLAW_MAX_DAYS` 环境变量，支持训练前只跑前 N 天冒烟测试，不用手动改 `all_tests.json`
- 给出训练前清单（环境变量确认、day01 冒烟测试命令、日志里要重点看的几个标记）后，**用户已实际提交训练**（`METACLAW_MAX_DAYS` 冒烟测试或正式训练，具体规模见提交时的命令），预计次日查看结果

**关键决策：** 详细设计过程和技术决策全部记录在 [`metaclaw_migration_plan.md`](openclaw-rl/docs/metaclaw_migration_plan.md)，本条目只做汇总，跳转链接查完整推理过程。**MetaClaw 迁移第一次真实训练已提交，结果待查**

**产出：**
- `scripts/metaclaw/run_metaclaw_migration_modelfactory.sh`（新建）：训练启动编排，`METACLAW_MAX_DAYS` 冒烟测试开关
- `scripts/metaclaw/metaclaw_rollout_driver.py`：rc!=0 基础设施故障保护、可选重试（`METACLAW_AGENT_RETRY`/`METACLAW_VERDICT_RETRY`，默认关闭）、`METACLAW_MAX_DAYS`
- `scripts/run_openclaw_topk_select_modelfactory.sh`：新增 `METACLAW_MIGRATION_PROFILE=1` 分支（`--save-interval` 100→10）
- `docs/reproduction_guide.md`/`scripts/train_separate_student.sh`/`scripts/train_with_services.sh`：仓库路径大小写修正
- `docs/metaclaw_migration_plan.md`：三方对照表、查证记录（四）、训练起点/checkpoint 策略/断点续跑决策/OpenClaw 版本核查全部记录

## 2026-08-18

**目标：** 补齐"给任意一个 checkpoint 打 Table 1 分数（Acc./Compl.）"的可复用方法记录，让 workspace 内的 agent（modelfactory 上跑训练/打分的 CLI agent）能直接照着做，不用每次重新问怎么打分

**完成内容：**
- 核实论文 Table 1 两个指标的准确定义：**Acc.** = 全部题目（`multi_choice`+`file_check` 混在一起）平均分（"mean per-question accuracy"）；**Compl.** = 仅 `file_check` 子集单独的通过率（"file-check completion rate"）。核实 Table 1 就是这两个指标按 Part I/Part II 分别报一遍，一共 4 列，AutoResearchClaw（论文另一张 Table 2）是完全独立的评测，跟这两个数字无关
- 核实官方 `benchmark/src/` 代码只算 Acc.（`report_cmd.py` 写进 `reports.md` 的 `Accuracy` 字段），从没算过 Compl.——全仓库搜 `completion` 关键词零命中，确认这是官方代码的真实缺口，不是我们漏看
- 新建 [`scripts/metaclaw/compute_table1_scores.py`](openclaw-rl/scripts/metaclaw/compute_table1_scores.py)：读 `metaclaw-bench run` 输出目录下所有 `scoring.json`，聚合官方 `scoring_cmd.py` 已经算好的 `score`/`question_type` 字段，一次性输出 Acc. 和 Compl. 两个数，不重新跑推理、不重新实现打分逻辑。用合成数据验证过计算逻辑（3 条样本手算 Acc.=50%/Compl.=50%，脚本输出一致）
- 完整"如何给任意一个 checkpoint 打分"三步法（起独立 SGLang 推理服务 → 配置 `openclaw.json` 环境变量指向它 → 跑官方 `metaclaw-bench run` → 用新脚本读 Compl.）已写入 `metaclaw_migration_plan.md`，含 base 模型和训练中 `torch_dist` checkpoint 两种 `MODEL_PATH` 填法（后者需先用 `convert_torch_dist_to_hf.py` 转换）、公平对比要用同一套系统级补丁的提醒
- **复盘第一次真实训练（`metaclaw_migration_20260817_181404`）：表面正常关闭，实际零有效样本**。CLI 提供详细日志诊断后逐条代码级核实（不直接采信）：346 次 `openclaw agent` 全部因 `GatewayCredentialsRequiredError` 失败。
  - **根因 1（主因）**：MetaClaw 官方 `_start_work_gateway`/`_run_openclaw_agent`（被我们直接 import、非自己代码）从不设置 `OPENCLAW_GATEWAY_TOKEN`，网关和 agent 客户端是两个独立子进程，网关自动生成的 runtime token 没有共享渠道。用 `git grep` 核实本地 `openclaw` 仓库两个版本标签：`GatewayCredentialsRequiredError` 在 `march_2026_3_8` 零命中、`may_2026_5_11` 大量命中——确认是三月之后才加入的强制鉴权，命中 CLAUDE.md"版本漂移"判断框架。**修复**：driver 启动时生成一个 `OPENCLAW_GATEWAY_TOKEN` 写回 `os.environ`，两个官方函数都用 `{**os.environ,...}` 建子进程环境，设一次自动共享，不改官方代码
  - **根因 2（次因，401，此前被静默吞掉）**：driver 自己直接 POST verdict/close 到训练代理，从没带 `Authorization` 头；代理 `_check_auth`（`openclaw_opd_api_server.py`）只要 `SGLANG_API_KEY` 有值就要求这个头，没带就 401——而 `_post_with_retry` 从没检查响应状态码，401 被当"成功"处理，driver 日志里完全看不出来。**修复**：`_post_with_retry` 统一补 `Authorization: Bearer {SGLANG_API_KEY}` + `response.raise_for_status()`，启动脚本新增把 `SGLANG_API_KEY` 传给 driver 进程
  - 两处修复均用合成数据做过功能测试（伪造 httpx client 验证 headers/状态码分支），真实网关/代理上未跑过
→ 详见 [`metaclaw_migration_plan.md`](openclaw-rl/docs/metaclaw_migration_plan.md)"训练故障复盘与修复：metaclaw_migration_20260817_181404"
- **发现并解决训练/评测数据重叠（leakage）问题——同时改变了 Acc./Compl. 数字的产生方式和 checkpoint 的用途**。用户发现训练用的 30 天数据和打分用的是同一份，追问论文怎么处理。直接读官方 `benchmark/scripts/rl_run.py`（Table 1 "MetaClaw (Full)" 那一档的真实产出脚本）核实：**论文自己是边训练边算分、同一趟跑完，没有 held-out 测试集，也完全没讨论这个问题**（`run scene-per-train 5` 单趟命令，training 和 scoring 共享同一份数据和同一次推理）。
  - **设计改动**：`metaclaw_rollout_driver.py` 现在训练过程中直接用官方 `scoring_cmd.py` 的打分函数（`_score_multi_choice`/`_score_file_check`，非简化版，multi_choice 有真实部分正确分）给每轮实时算分，跑完聚合成 Acc./Compl.，这才是跟论文方法学对得上的数字。之前那套"独立 SGLang + `metaclaw-bench run`"打分法保留作为论文没做过的、更干净的补充手段，不再是主产出
  - **连带改动：checkpoint 角色从"最终模型"变成"崩溃后能重新训练的续跑点"**——训练和打分共用一趟运行后，"崩溃从 day01 重跑"会让已算过分的天用不同权重重新生成答案、污染最终聚合分数，不再只是浪费算力。撤回 08-17"不做断点续跑"的决定，重新加回按天粒度的进度持久化（`METACLAW_PROGRESS_DIR`）：跑完一天且无异常才落盘，启动时跳过已完成的天（不重跑 agent、不重新提交 verdict），复用其分数参与聚合
  - 重新核查了之前否决按天续跑的两条理由：**都不再成立**——workspace 一致性核实每天都从 `workspace_src` 全新拷贝、从不继承前一天效果（跨天本来就无状态），跳过重跑不会丢失任何文件状态；checkpoint/天数不同步的训练侧 `--load` 自动续训已存在（早前修复），剩下风险变窄为"某天分数已记但训练贡献可能因崩溃丢失"，接受为已知权衡，不影响聚合分数本身的真实性
  - 用合成数据做了功能测试：`_score_round_official` 对 multi_choice 算出真实部分正确分（不是二元）；`main()` 级别集成测试模拟"day01 已完成"场景，确认正确跳过、正确聚合。真实崩溃/重启场景未验证
→ 详见 `metaclaw_migration_plan.md`"训练/评测数据重叠：论文自己怎么做的，我们的设计改动"
- **用户纠正两点，当场改掉**：(1) 断点续跑不能做成"设了目录就自动跳过"，必须手动触发，否则正常训练可能因为凑巧复用了旧目录而意外漏跑某些天——拆成两个独立开关：`METACLAW_PROGRESS_DIR`（只负责落盘，不改变行为）+ `METACLAW_RESUME=1`（唯一真正触发跳过的开关，必须手动显式设置，且必须搭配 `METACLAW_PROGRESS_DIR` 否则报错）。(2) 之前提出的"独立 SGLang + `metaclaw-bench run`"打分法（保留作为"更干净的补充手段"）是错误判断——那个方法打分用的还是同一份 30 天训练数据，并不比实时聚合更干净，只是把重叠往后挪了一步。**已撤回并删除** `compute_table1_scores.py`（`launch_simulator.sh` 是 Personal Agent Track 通用脚本，不受影响，保留）。用户同时确认：训练跑完保存的最终 checkpoint 仍然保留作为最终结果的一部分，这个不需要额外代码，Megatron `--save`/`--load` 本来就会存
  - 三个开关行为均用合成数据验证：`METACLAW_RESUME` 不设时即使进度文件存在也正确忽略；设为 1 时正确加载并跳过；设为 1 但没设 `METACLAW_PROGRESS_DIR` 时正确报错拒绝启动
→ 详见 `metaclaw_migration_plan.md`"已废弃：独立 SGLang + `metaclaw-bench run` 打分法"
- **训练 report 补齐成跟官方 `report.md`/`report.json` 同款格式**。用户拿到基线评测的 `report.md` 后追问训练是否也这样输出，查证发现官方训练脚本 `rl_run.py` 本身就是调 `metaclaw-bench run --scene-per-train`——跟纯基线评测走的是同一条 `infer→scoring→report` 流水线，官方两者格式本来就一致；我们自己的 driver 因为用的是 slime+Megatron 的 Hybrid RL、架构上没法塞进这条流水线，所以之前只输出简化版 `final_scores.json`。把 `report_cmd.py::run_report`/`_render_markdown` 的聚合和渲染逻辑原样搬进 driver（`_build_report`/`_render_report_markdown`，改成读内存里的 round 记录而不是扫描 `scoring.json` 文件），训练完输出**同名同结构**的 `report.json`/`report.md`（取代原来的 `final_scores.json`），可以直接跟基线报告并排对比。`_score_round_official` 返回结构同步对齐官方 `_score_one` 字段（`test_id`/`group_id`/`round_id`/`metrics`）。Token Usage 字段保留但恒为 0（driver 不落 `llm_log` 结构化文件，如实标注不是缺陷）；Compl. 不是官方 schema 概念，没塞进 `report.json`，改在 `report.md` 末尾单独加一行
  - 用合成数据验证：两天混合 multi_choice+file_check 记录，聚合数字跟手算一致；渲染出的表格列名/顺序/`-`占位跟用户贴的真实基线报告样例逐列对得上
→ 详见 `metaclaw_migration_plan.md`"训练 report 跟官方 `report.md`/`report.json` 对齐"
- **核查 CLI 给出的"6 项系统级补丁"清单+"装齐就能公平对比"结论，发现遗漏一处**。清单本身（`rl-training-headers`+5 个版本漂移补丁）跟启动脚本实际部署的六步逐一核对，准确无误；但 CLI"rl-training-headers 两边都有、不影响对比"这条结论是错的——查了插件代码：注入是无条件的（每次 `before_prompt_build` 都往系统提示词追加 `[RL-TRAINING-META]` 标记），只有训练代理（30000 端口）才会剥掉，基线走"直连 SGLang"完全不经过这层剥除，模型会看到训练时从没见过的怪异后缀。已有的基线报告（`run_20260818_101454`）大概率受此影响，`Compl.=0.0%` 可能部分由此导致。修复建议：重跑基线前 `openclaw plugins disable rl-training-headers`
→ 详见 `metaclaw_migration_plan.md`"已知风险/限制"新增条目
- **修复 report 默认不落盘的问题**。CLI 核实发现：不设 `METACLAW_PROGRESS_DIR` 时 `report.json`/`report.md`/按天分数文件全都不写，Acc./Compl. 只 print 进日志——这是把"按天续跑要不要开"（该默认关，手动触发）和"report 要不要存文件"（不该要求额外配置）错误耦合到了同一个开关上。新增独立的 `METACLAW_REPORT_DIR`：driver 里默认取这个变量，没设就退回 `METACLAW_PROGRESS_DIR`，两个都没设才是"只 print"（会打 WARNING，不再静默）；启动脚本给 `METACLAW_REPORT_DIR` 一个始终有值的默认路径（`${LOGS_DIR}/report`），正常提交训练不用任何配置就能自动拿到 report 文件
  - 三种组合（只设 REPORT_DIR / 只设 PROGRESS_DIR / 都不设）均用合成数据验证过
→ 详见 `metaclaw_migration_plan.md`"训练 report 跟官方对齐"补充修复部分

**关键决策：** 两个真实 bug（网关鉴权）均已定位到代码级根因并修复；Acc./Compl. 的产生方式和 checkpoint 的用途都发生了实质性改动，改成跟论文 Table 1（Full 档）方法学一致；断点续跑改成手动触发、废弃了一个错误的补充打分法；训练 report 格式补齐到跟官方基线评测一致、默认自动落盘；发现 rl-training-headers 对基线/训练不公平的真实风险——下一次训练提交后才能同时验证这几处改动，重跑基线前需要先关掉 rl-training-headers 插件

**产出：**
- `docs/metaclaw_migration_plan.md`："训练故障复盘与修复" + "训练/评测数据重叠" + "已废弃：独立 SGLang + `metaclaw-bench run` 打分法" + "训练 report 跟官方对齐"（含落盘修复）四节 + "已知风险/限制"新增 rl-training-headers 条目
- `scripts/metaclaw/metaclaw_rollout_driver.py`：网关鉴权两处修复；新增 `_score_round_official`（对齐官方字段）/`_aggregate_acc_compl`/`_build_report`/`_render_report_markdown`（官方 report_cmd.py 逻辑搬入）；`METACLAW_PROGRESS_DIR`（按天续跑，手动触发）+ `METACLAW_REPORT_DIR`（report 落盘，独立开关，两者解耦）；输出 `report.json`/`report.md` 取代 `final_scores.json`
- `scripts/metaclaw/run_metaclaw_migration_modelfactory.sh`：新增 `SGLANG_API_KEY`、`METACLAW_PROGRESS_DIR`、`METACLAW_RESUME`、`METACLAW_REPORT_DIR`（默认 `${LOGS_DIR}/report`，始终有值）传给 driver 进程
- `scripts/metaclaw/compute_table1_scores.py`：已删除（错误的补充打分法，同一份数据重复使用）
- **第二次真实训练（`metaclaw_migration_20260818_*`）复盘：网关鉴权修好后，撞上另一个真实问题——day01 起大面积 context overflow，全程零训练样本**。`openclaw agent` 请求 16661 输入+30313 max_tokens=46974，超过官方默认 `--sglang-context-length 32768`，SGLang 400→代理转 500→driver 正确记 `agent_succeeded=False`（这部分行为符合预期，没有提交假训练信号）→连续失败触发 router 熔断→后续全 503。**根因**：32768 是官方脚本为 Personal Agent Track（GSM8K 风格，短对话）调优的值（对比 OpenClaw-RL 自己 toolcall-rl 4B 脚本用的还更小，16384，说明 32768 不是什么安全余量）；MetaClaw 系统提示词重得多，其**官方配置模板 `openclaw_cfg/openclaw.json`/`metaclaw.json` 明确声明 `contextWindow: 50000, maxTokens: 50000`**——不是我们猜的数字。之前跑通的基线用的是临时 65536 上下文，凑巧盖过 50000 才没暴露问题
  - **修复**：`run_openclaw_topk_select_modelfactory.sh` 的 `METACLAW_MIGRATION_PROFILE=1` 分支新增三条 sed，把 `CONTEXT_LENGTH`/`--rollout-max-context-len`/`--sglang-context-length` 从 32768 统一改到 **65536**（用户决定：比官方声明的 50000 下限再留余量，应对训练过程中会话内容累积增长）；`--max-tokens-per-gpu` 不动（跟 sglang context 无关，沿用脚本既有的"不提前假设"原则）。用 sed 直接在真实官方脚本上验证过三处改对、`--max-tokens-per-gpu` 确认没被误改
  - **连带发现并修复一个 resume 设计漏洞**：全部轮次因基础设施故障失败的天，`run_day` 不会抛异常，空列表 `[]` 照样会被持久化成"跑完了"——之前的 resume 逻辑会把这种"零样本天"误判成"已完成"直接跳过，永远没机会用修好的配置重跑。修复：`main()` 里判断条件从 `if resumed is not None` 改成 `if resumed:`（`None` 和 `[]` 都是 falsy，同一个判断覆盖两种情况），空文件本身仍保留作为诊断记录，只是不再被当作"已完成"。用合成数据模拟过完整场景（day01 空进度被正确重跑、day02 有真实进度被正确跳过）
  - **明确建议：这一轮训练直接停掉，不要用它的 `METACLAW_PROGRESS_DIR` 做 resume**（里面全是空天，没有真实进度可续），开新目录、用修好 context 的脚本从 day01 完整重跑
  - CLI 同时发现请求里没有 `[RL-TRAINING-META]` 标记——暂缓排查（现在所有请求都还没走到"模型真正生成"就先溢出失败了，没法判断插件是否生效），等 context 修好、真的有请求跑通后是下一个要核实的问题：如果标记确实缺失，`_METACLAW_SESSION_RE` 整套确定性 reward/步骤判官分派可能从一开始就没生效
→ 详见 `metaclaw_migration_plan.md`"训练故障复盘与修复（二）：metaclaw_migration_20260818_*，context overflow"
- **补上跟 Personal Agent Track 对齐的人类可读训练转录**。用户反馈之前 driver 日志只有 `passed=%s agent_succeeded=%s` 结构化摘要，看不到实际问答内容，没法像 Personal Agent Track 的 `simulation.log`（`student_chat.py` 打印每轮完整对话）那样手动通读找规律——agent 查结构化字段容易漏掉的模式，人工看原始转录才看得出来。在 `_run_round` 里加了同款风格的 `print()`（天级别标题、每轮完整 `>> Query -> OpenClaw`/`<< OpenClaw -> Query` 原文不截断、verdict、OPD hint），不需要新日志文件或环境变量——`metaclaw_rollout.log` 本来就是 driver 整个进程的 stdout 重定向，只是之前没打印过这些内容。用合成数据验证过完整一轮的打印格式（mock 掉 `_run_openclaw_agent`，跑真实 `_run_round`），真实训练环境未验证
→ 详见 `metaclaw_migration_plan.md`"训练过程可读性：对齐 Personal Agent Track 的 simulation.log"
- **对齐基线跑完，定版为本次迁移唯一采信的训练前基线**（`run_20260818_141305`，30 天/346 题，6 补丁+关 `rl-training-headers` 对齐后）：**Acc.=8.1%（Correct=28.0），Compl.=0.0%（0/224 file_check）**。跟上次不对齐基线（Acc.=5.7%）对比，验证了两件事：(1) `rl-training-headers` 标记污染是真实存在的，Acc. 提升全部来自 multi_choice（满分 12→18、部分正确 19→23）；(2) **`Compl.=0.0%` 两次完全一样，不是标记污染，是 4B 模型在 file_check 类任务上的真实基线能力上限**——之前"大概率是污染导致"的怀疑被证伪。之后训练成果（driver 自己输出的 `report.json`/`report.md`）都跟这份基线对比
  - 记录了一条操作提醒（非分析结论）：这次基线打分用的独立 SGLang（4B，:30002）和关闭状态的 `rl-training-headers` 插件目前仍是这个状态，**下次提交训练前必须重新 `openclaw plugins enable rl-training-headers`**（训练走 30000 代理，代理依赖这个标记做 session 分派，忘记开等于重蹈这次迁移最早发现的那个问题）
→ 详见 `metaclaw_migration_plan.md`"基线结果（用于后续对比，2026-08-18 定版）"
- **第三次真实训练已提交，等待次日结果**：`rl-training-headers` 重新 enable，开新的 `METACLAW_PROGRESS_DIR`，用今天这一整套修复（context 32768→65536、resume 空天误判漏洞、report 格式对齐官方+自动落盘、人类可读转录）从 day01 完整重新提交。今天两次真实训练分别卡在网关鉴权和 context overflow，这次是第一次有机会真正跑出训练样本——重点看今天的修复是不是都生效了。
- **提交后发现新问题并当场修复：人类可读转录在真实训练里不是实时可见的**。CLI 反馈 `metaclaw_rollout.log` 里只有 `logger.info` 摘要，没有当天早些时候加的 `>> Query -> OpenClaw` 转录。诊断：Python `stdout` 重定向到文件（启动脚本的 `> metaclaw_rollout.log 2>&1 &`）会从行缓冲切换成全块缓冲（4-8KB 才刷新），`logging` 默认走 `stderr` 不受影响，所以只有摘要行实时可见，`print()` 内容已经执行、只是堵在缓冲区里没写进文件。**修复**：`if __name__ == "__main__":` 里加 `sys.stdout.reconfigure(line_buffering=True)`。用真实子进程+重定向到文件的方式复现过问题（不加这行时子进程没退出前文件读不到内容）、验证过修复有效（加了这行后子进程没退出前就能读到内容）——修复前后各测一遍，不是单纯读代码猜的。**这次已提交的训练跑的是旧代码，这个修复要等下一次重新提交才会生效**；这次先靠 `logger.info` 摘要判断训练是否正常——已确认 `day01` 出现过 `agent_succeeded=True`/`passed=True`，训练链路本身是通的，只是详细转录这次看不到实时更新
→ 详见 `metaclaw_migration_plan.md`"训练过程可读性"补充修复部分
- **发现并修复本次迁移最严重的根因：`rl-training-headers` 从一开始就没在任何 MetaClaw session 里真正加载过**。第三次训练（context overflow 修好后）表面正常——agent 正常答题，`day01` 10 题全部有分，GPU 利用率 80%+——但训练队列一直 `waiting for combine samples: 0/16`，权重完全没在学。CLI 现场诊断：OpenClaw 自己发出的请求（包括真正干活的 read/write 轨迹）完全没有 session_id、没有 `[RL-TRAINING-META]` 标记，代理只能记成 `session=unknown`/`turn_type=side` 直接丢弃；driver 自己手动设置 HTTP header 的 checker verdict 能对上，但这不依赖插件。CLI 把方向指对了但没找到具体机制——**直接读 OpenClaw 官方源码（`may_2026_5_11` 快照）查到精确根因**：`openclaw plugins enable` 只写全局配置的 `plugins.enabled`；MetaClaw-official 自己的 `openclaw_cfg/openclaw.json` 和 `metaclaw.json`（`_prepare_work_copy` 复制进每天隔离工作副本的模板，两份都查过）都硬编码 `"allow": ["llm-prompt-logger"]`；`config-activation-shared.ts::resolvePluginActivationDecisionShared` 有一条无条件早退：非空的 `plugins.allow` 会直接排除不在名单里的插件，不管全局有没有 enable。**结论：`rl-training-headers` 在这次迁移里从没真正加载过，`before_prompt_build` 钩子从没触发过，这是比网关鉴权、context overflow 更根本的问题——之前一直被这两个基础设施问题挡在前面没暴露出来**
  - **修复**：新增 `_ensure_plugins_allowlisted()`，每天的工作副本 `openclaw.json` 生成后（`_patch_agent_workspace` 之后、起网关之前）把 `rl-training-headers` 追加进这份文件自己的 `plugins.allow`——只改工作副本，不碰官方模板源文件。用真实的 MetaClaw 官方 `openclaw_cfg/openclaw.json` 模板验证过（改前 `allow=['llm-prompt-logger']`，改后 `allow=['llm-prompt-logger', 'rl-training-headers']`，原有条目不受影响），也验证过幂等性和无 `plugins` 字段时的防御性构建
  - **如实记录一个未解开的疑点**：如果这个白名单排除是无条件的，那"对齐/不对齐基线"两次评测理论上不该有 Acc. 差异（5.7%→8.1%），但实测确实有差异——这两个结论字面上矛盾，具体机制（全局配置和工作副本配置是否存在某种合并继承关系）本次没有查证清楚。不影响这个修复本身的必要性（把插件加进白名单在任何情况下都是训练场景下的必要条件），但那两次基线差异的真正原因还不能 100% 确定
  - **这次训练（`metaclaw_migration_20260818_175145`）已经在跑，确认零训练样本、checkpoint 学不到东西**——可以让它跑完只拿 Acc./Compl.（评测链路本身没问题），但权重不会更新。下次训练是第一次真正有条件验证确定性 reward/步骤判官分派机制是否按设计工作
→ 详见 `metaclaw_migration_plan.md`"训练故障复盘与修复（三）：metaclaw_migration_20260818_175145，rl-training-headers 从未真正加载"
- **第四次训练（`metaclaw_migration_20260818_182736`，commit `b18c791`，18:27→19:36）跑完，本次迁移第一次真正走通训练样本链路**：`[RL-TRAINING-META]` 标记确认出现，SIDE/skipped=0，累计提交 234 条 RL 样本，训练走完 step 0～13，checkpoint 存到 `iter_0000009`，day01→day30 全部跑完，`report.md` 已生成。**如实评估结果**：Acc.=8.3%（338 题）vs 对齐基线 8.1%（346 题），Compl. 两者都是 0.0%——数字跟冻结基线几乎一样，不像方法已经拉开差距；更值得注意的是逐日模式不是均匀低分，而是 `day01` 还有 38%、`day11` 之后几乎全塌到 0，`Compl.` 全程 0——这是"先好后差"的模式，不是简单的"方法完全无效"能盖棺定论的，需要看 `metaclaw_rollout.log` 里的转录和 `report.md` 逐日明细才能判断下一步往哪调（学习率/reward 设计/OPD hint 质量等）。这次训练没有被基础设施问题污染，是目前唯一一份能真正拿来讨论方法效果的数据
→ 详见 `metaclaw_migration_plan.md`"第一次真正产生训练样本的训练结果：metaclaw_migration_20260818_182736"
- **修复转录在终端看不到的问题**：`metaclaw_rollout.log` 文件里内容完整（30 个 `# Day`、346 条 `>> Query -> OpenClaw`，证实上一条 `line_buffering` 修复确实有效），但训练脚本自己的终端/job 输出里看不到——因为 driver 用的是纯重定向 `> log 2>&1`，只写文件；Personal Agent Track 的 `simulation_loop` 用的是 `tee -a`，文件和终端同时写。**修复**：改成进程替换 `> >(tee -a "$LOG") 2>&1`（不是直接 `| tee`——直接管道会导致 `$!` 拿到 `tee` 的 PID、后面 `kill "${DRIVER_PID}"` 清理逻辑杀错进程；进程替换保留 `$!` 仍是 driver 自己的 PID）。验证过三点：`$!` 确实还是 driver 自己的 PID、`kill "$DRIVER_PID"` 确实能正确杀掉真正的进程（30 秒 sleep 测试进程验证过）、日志文件内容没有因为改用 `tee` 而缺失
→ 详见 `metaclaw_migration_plan.md`"训练过程可读性"第三个补充修复部分

## 2026-08-19

**目标：** 分析 `metaclaw_migration_20260818_182736`（第一次真正产生训练样本的训练）"先好后差"塌陷模式的根因，修复训练信号污染

**完成内容：**
- **定位到根因：不是环境突然出问题，是从 step 0 起训练信号就在灌毒**。CLI 深挖训练日志和 wandb 曲线（`train_rollout_logprob_abs_diff` 从 step 0 的 3.86 掉到 step 9 的 0.23），确认两个真实 bug 叠加：
  - **Bug A（checker 分数从没打到真实最后一轮）**：`_send_verdict_turn` 把 checker 判决当合成 `main` 轮次 POST 给代理，`_opd_evaluate()` 认出 checker 分（265 次 `deterministic-reward`），但代码继续往下掉进 PRM 分支共用的 `return`，那两处都引用只在 PRM 分支才会赋值的 `_skip_forced_negative_override`——265 次评估全部 `UnboundLocalError`，真正该吃 ±1 的最终轮被静默丢弃
  - **Bug B（verdict 自己的生成残片反而进了 GRPO）**：`X-Turn-Type: main` 除了需要的"挂 next_state"效果外，还会无条件把这次调用自己的生成结果注册成新的待评估轮次；`max_tokens=8` 时 Qwen3-Thinking 把预算花在 thinking 上，留下 13-token 残片，被下一题内容挂上 next_state、被 step-judge 打分、当 RL-only 提交——234 条提交样本里 113 条（48%）是这类残片，69 条还拿到 +1。真实 checker 信号整晚没进优化器，GRPO 很快学会"少说话/空回复"，这才是 `day11` 后大面积塌陷的真正原因，之前"这次训练没被基础设施问题污染"的判断是错的，已在 `metaclaw_migration_plan.md` 里更正
  - 方案制定经过三轮 CLI 交叉核实：第一版"`max_tokens` 调到 0、指望现有空响应检查接住"被 CLI 用真实 tokenizer 实测证伪（Thinking 模板即使空内容也会补 `</think>`/`<|im_end|>` 结构性 token，约 5 个，现有检查过滤不掉）；改成代理侧控制流直接短路，不依赖生成结果长什么样
→ 详见 `metaclaw_migration_plan.md`"训练信号根因分析与修复：checker 分数丢失 + verdict 残片污染 GRPO"
- **四处改动落地**：
  1. `prepare_patched_openclaw_opd.sh` 新增 `openclaw-rl-metaclaw-verdict-signal-skip`：`_handle_request` 识别 `max_tokens==0` 信号，完全不调用 SGLang，只执行"挂 next_state"+ 完整 `session_done` 清理（含 `_seen_user_messages.pop`）
  2. `prepare_patched_openclaw_combine_select.sh` 新增 `openclaw-rl-metaclaw-verdict-early-return`：`_metaclaw_verdict` 分支末尾显式 `return`（结构照抄 step-judge 分支），不再掉进会崩溃的共用出口——代价是原设计"长 hint 走 OPD"这条从没成功执行过的路径正式改成 RL-only，如实记在补丁注释里
  3. 同一脚本，step-judge 分支新增 `openclaw-rl-metaclaw-step-judge-truncation-penalty`：补上跟现有 PRM 分支一致的 `is_truncated` 强制 -1（延续 08-06 已确定的策略，不是另选弱方案）
  4. `metaclaw_rollout_driver.py`：`_send_verdict_turn` 的 `max_tokens` 8→0（`_send_session_close_only` 不动）；新增 generate-fail 兜底文案检测，**只做转录标注**，不改变 `agent_succeeded`/打分/verdict 逻辑（草稿曾想复用 infra 失败通道处理，被指出语义不对，已改正）
  - 四处改动均用官方源文件跑过完整补丁链验证（`py_compile` 通过、代码位置正确），driver 侧新增端到端回归测试（mock agent 调用，确认普通轮次不受影响、verdict payload 正确变成 `max_tokens=0`、close payload 不变、generate-fail 检测只加标注）。**真实训练环境完全未验证**，冒烟清单见迁移文档
→ 详见同一节

**关键决策：** 这是本次迁移目前为止最严重的一处训练信号污染，两个 bug 叠加导致 234 条样本里近一半是被误标的垃圾残片，真实 checker 信号完全没进优化器——`metaclaw_migration_20260818_182736` 这个 checkpoint 不能代表训练方法的真实效果，之前"如实评估"里"数字接近基线"这个结论本身没错，但背后原因不是"方法效果不明显"，是"训练信号从一开始就没对"。下次训练才是第一次有条件看到干净信号下的真实效果

**产出：**
- `scripts/prepare_patched_openclaw_opd.sh`：新增 `openclaw-rl-metaclaw-verdict-signal-skip` 补丁
- `scripts/prepare_patched_openclaw_combine_select.sh`：新增 `openclaw-rl-metaclaw-verdict-early-return` + `openclaw-rl-metaclaw-step-judge-truncation-penalty` 两处补丁
- `scripts/metaclaw/metaclaw_rollout_driver.py`：`_send_verdict_turn` max_tokens 改 0，新增 `_GENERATE_FAIL_MARKERS` 转录标注
- `docs/metaclaw_migration_plan.md`：新增"训练信号根因分析与修复"一节，更正了上一节"没被基础设施问题污染"的错误判断
- **发现并修复一个独立的问题：默认提交训练会静默加载上一次的（可能训坏的）权重**。用户追问"权重续训"跟"按天续跑"是不是两套机制，CLI 核对 `run_openclaw_topk_select_modelfactory.sh` 确认：`--load "${SAVE_CKPT}"` 是无条件加的（更早为 Personal Agent Track 真实崩溃续训加的），`run_metaclaw_migration_modelfactory.sh` 里 `SAVE_CKPT` 默认值原来是固定路径（不带时间戳）——只要上次训练在这个目录存过 checkpoint，下次提交训练**即使完全不碰 `METACLAW_PROGRESS_DIR`/`METACLAW_RESUME`** 也会静默加载那份权重继续训，不是从干净 base 开始，跟"实验阶段每次都要重新开始跑"这个明确要求直接冲突。之前"训练侧 checkpoint 本来就有 `--load` 自动续训，不受这次改动影响"这个判断没有意识到这正是问题所在
  - **修复**：`SAVE_CKPT` 默认值改成带时间戳（跟 `LOGS_DIR` 共用脚本开头统一生成的 `RUN_TIMESTAMP`）——不显式设置就天然是新目录，`--load` 自动回退到干净预训练权重；旧 checkpoint 不删，留在各自时间戳目录下，需要接着训某次跑出来的权重仍可以手动指定 `SAVE_CKPT`。用 bash 验证过默认解析和显式覆盖两种情况都符合预期
→ 详见 `metaclaw_migration_plan.md`"修复：默认提交训练会静默加载上一次的（可能训坏的）权重"
- **发现并修复第三个独立问题：训练暂停期间的 503 被当成基础设施失败，整段天数被空转吃掉**。用上面两个修复后重新提交的 `metaclaw_migration_20260819_132608` 从 `day06` 起大段"没答题"，CLI 沿时间线核对 `submission paused`、权重同步和 rollout 失败，定位到跟前一天那两个 bug不是同一类——训练信号是干净的，是 slime 攒满一个 batch（16 条）后 `pause_submission()`（真实观测一次完整暂停窗口 4 分 20 秒，含 66.5 秒 `save_model`）期间，代理对所有请求直接 503（`submission_enabled` 检查在最前面），OpenClaw 自己不重试，`rc=1` 退出，driver 原来的 `AGENT_RETRY` 循环两次尝试之间没有任何等待，扛不住几分钟的暂停，一次暂停窗口就能连续吃掉后面好几天
  - CLI 用真实日志核实：日志里其实有三类失败——503（还没开始处理就拒绝，能靠等）、timeout（已经生成很久才死，不该跟 503 用同一套长等待，风险是把真实的 GPU 争用问题也拖成空转）、generate-fail（`rc==0`，已在前一次修复里确认走正常打分通道）——不能一概而论
  - **修复**：只针对 503 单独加一个独立、耐心的等待重试环，不消耗现有 `AGENT_RETRY`/`VERDICT_RETRY` 预算，其余失败类型（含 timeout）完全不受影响：`_run_round`（agent 子进程）用 `stderr` 文本匹配 `"503 status code"`（OpenClaw 自己 `FailoverError` 的确切文案）；`_post_with_retry`（verdict/close 提交）用 `httpx` 结构化 `response.status_code == 503`（更精确，httpx 自己的错误文案跟 OpenClaw 不是同一段文字）。15 秒重试间隔，900 秒（15 分钟）总预算，用 `time.monotonic()` 测墙钟时间（不是单纯累加 sleep，因为每次失败的进程启动本身也要 1-2 秒）
  - 5 项合成数据回归测试全部通过（503-then-成功、503 耗尽预算后正确放弃、timeout 完全不进等待环——三点都覆盖了 `_run_round`/`_post_with_retry` 两条路径）。真实训练环境（真实的分钟级暂停窗口）尚未验证
→ 详见 `metaclaw_migration_plan.md`"修复：训练暂停期间的 503 被当成基础设施失败，整段天数被空转吃掉"
- **发现并修复第四个独立问题：checker 算出的 OPD hint 被无条件丢弃，file_check/多选题 verdict 轮次从未真正走过 OPD 蒸馏**。CLI 对着 `metaclaw_migration_20260819_153518` 排查"越写越长、`write` 调用消失"这个模式，核对 driver 的判分和 hint 构造，发现上面第一条修复（`openclaw-rl-metaclaw-verdict-early-return`）为了堵 `UnboundLocalError` 崩溃，把返回值写死成 `accepted: False, hint: ""`——但 `_metaclaw_hint`（driver 侧 `_build_opd_hint` 算出的真实失败原因）当时已经算出来了，只打进日志没真正使用，OPD 蒸馏这条路径事实上从未生效过，一直只有 checker ±1 的 GRPO 信号
  - 两处独立问题，分两层修，Layer 1 必须先于 Layer 2：**Layer 1**（`_build_opd_hint`，只影响 file_check）——checker 静默失败（stdout/stderr 都空）时原来退回题面写死的静态 `feedback.incorrect` 文案，可能文不对题（day01 r5 实锤：模型没碰某个 task，hint 却说该 task 的日期格式错了），改成直接返回 `""`，不再猜；CLI 用 `20260819_153518` 真实数据核对：约 55 道失败 file_check 里 48 道有真实 checker stdout、6 道是留在 stderr 的 traceback、只有 1 道落进"两边都空"这个分支，改动后覆盖率几乎不受影响。**Layer 2**（`prepare_patched_openclaw_combine_select.sh`，`_metaclaw_verdict` 分支）——`_metaclaw_hint` 非空且长度 > 10 时，照抄父类 `accepted=True` 材料化代码（`_append_hint_to_messages` → `_normalize_messages_for_template` → `apply_chat_template` → tokenizer 编码），真正返回 `accepted=True` 让 hint 进入 teacher 序列，`eval_score` 仍是 checker ±1；空 hint 维持 `accepted=False` 不变；材料化代码包了 try/except，模板/tokenizer 在 MetaClaw 消息结构上一旦报错就安全退回现有行为，不丢样本
  - CLI 核实过一个容易想歪的点：官方原始调度确实是"没 hint 就整条丢弃"，但 MetaClaw 走的是 `Combine → Combine-Select` 链路，本项目早先给 Personal Agent Track 打的调度补丁本来就有"`accepted=False` 但 `eval_score` 有效就走纯 RL"这条路，这次改动不需要碰调度逻辑，只改 `_opd_evaluate` 返回值
  - `py_compile` + 完整补丁链对真实官方源文件生成输出验证通过，人工核对生成代码的分支结构正确。**真实训练环境完全未验证**——下一轮需要在日志里确认出现 `[openclaw-rl-metaclaw-verdict-opd-hint] ... accepted K_i=1` 这行，不能只有 `submitted RL sample`
  - **明确没有一起做**：CLI 同一轮诊断还指出"越写越长"另有两个更致命的独立成因——下一轮 `_build_feedback_text` 静态反馈文案本身可能文不对题（跟这次的 verdict hint 是完全不同的代码路径）、中间轮次 step-judge 对"没有实际调用 write/edit 的长分析"经常打 +1（75-100% 命中率）。这两处都只有诊断没有方案，决定不跟这次两层改动捆在一起，等下一轮训练数据出来分别单独设计
→ 详见 `metaclaw_migration_plan.md`"修复：checker 算出的 OPD hint 被无条件丢弃，file_check/多选题 verdict 轮次从未真正走过 OPD 蒸馏"
- **发现并修复第五个独立问题：session 拆分——从"一天一个 session"改成"每题一个 session"**。CLI 继续排查上面第 (1)(2) 点"越写越长"的另外两个成因时，挖出第三个、对 Acc. 影响最直接的独立问题：`153518`/`173654` 两次训练日志显示 `day01-06` 选择题正确率稳定在 85-97%（权重没训坏），但从 `day07` 起，一旦某道 file_check 题写崩（`day07 r5` 约 1.7 万字），**同一天后面所有题目**（含本来答得好的选择题）一起 Context overflow——`day08`/`day09`/`day10` 同一模式反复出现。根因是这次迁移一直对齐 MetaClaw 官方评测代码"一天一个 session"（`_run_group` 的设计），导致某题写崩的长文原样累积进当天后续所有题目的上下文
  - 读代码核实（不是猜）：`_prepare_session` 只是 touch 一个 `.jsonl` transcript 文件，跟 workspace/gateway 无关；day 级资源（workspace/gateway/plugins allowlist）都不依赖 session_id；proxy 侧 `_pending_turn_data`/`_prm_tasks` 按 session_id 分桶，新 session 天然是白纸；`[Previous Feedback]` 靠纯文本拼接不靠共享历史，不受影响；MetaClaw 自己的训练侧代码（`openclaw_env_rollout.py`）跟这次迁移的 day/round/feedback 结构本来就没对齐过，不存在"按题拆 session 会破坏跟官方训练方法一致性"的顾虑
  - CLI 额外核实三件事：全部 30 天 346 题的 `round_record["id"]` 扫描（清一色 `r1`...`r15`，字符安全，每天恰好 1 个 group）；`openclaw agent --session-id` 真隔离（官方文档+代码解析确认，`sessions_*` 调用在真实日志里 0 次，无记忆类插件）；overflow 真实模式不止 day07（day08/09/10 同一模式，均由真实日志核对）
  - **修复**：只改 `metaclaw_rollout_driver.py`——day 级 `session_id`/`_prepare_session` 删除，round 循环内部现算 `round_session_id = f"metaclaw-{test_id}-{group_id}-{round_id}"`（`{group_id}-` 真实数据里冗余但保留作防御）；`_run_round`/`_send_verdict_turn`/`_send_session_close_only` 三处统一用这个新 id；`session_done`/`_send_session_close_only` 都从"只有当天最后一题才收尾"改成**每题无条件收尾**——不这样做的话非最后一题失败的挂起轮次会卡在一个再也不会收到消息的 session 里，永远等不到清理
  - **连带效果**：顺手结构性解决了 08-17 记录的"跨 round 污染"老问题（挂起轮次被下一个 round 误判成 next_state）——每个 round 现在是独立 session 且每次都无条件 `session_done`，那个 bug 的前提不再成立，不需要再单独判断触发频率或设计修复，08-17 那条"待观察"记录已关闭
  - **预期效果**（如实记录）：会让 day07-10 那种"选择题被前面题目长文拖垮"的模式基本消失，Acc. 里选择题部分能重新反映真实能力；不会让 `Compl.` 变好——单题依然可能写崩/打 0 分，这只是不再连坐，"file_check 学不会写文件"仍要靠 OPD/奖励设计解决
  - 跟 MetaClaw 官方评测代码"一天一个 session"是主动分歧，明确记账：官方确定性打分从不读 transcript，这次改动不影响打分口径
  - `py_compile` 通过。**真实训练环境完全未验证**——CLI 明确要求不热补正在跑的 `153518`/`173654`，这次改动落地后由用户决定何时提交新训练；如果跟 OPD hint 接线修复同一轮验证，Acc. 提升需要分开看归因（"选择题不再被 overflow 拖累" vs "OPD 让 file_check 真的变好"，是两件事）
→ 详见 `metaclaw_migration_plan.md`"改动：session 拆分——从'一天一个 session'改成'每题一个 session'"
- **发现并修复第六个独立问题，也是迄今最大的一个：`openclaw agent` 子进程从未传 `--agent`，`write` 实际写进了 checker 看不到的默认 agent workspace**。CLI 排查 `Compl.` 为什么一直是 0 时发现，模型经常真的在写文件（session transcript 里有真实的 `Successfully wrote N bytes` 成功回执），但文件落在 `openclaw_state_.../workspace-main/day05/`，checker 读的是 `work/workspace_day05_.../day05/`——两个完全不同的目录，`stdout` 永远是 `FAIL: cannot read ...` 不是因为没写，是写到了别的地方
  - 沿 OpenClaw CLI 自己的源码逐层追出根因：`MetaClaw-official/benchmark/src/infer/infer_cmd.py::_run_openclaw_agent` 有 `agent_id` 参数但从未拼进 `openclaw agent` 的命令行，它唯一的官方调用方 `_run_question` 也从未传——**这个缺口在 MetaClaw 官方代码本身，driver 是直接 import 复用的，缺口原样带进来**。不传 `--agent` 时，OpenClaw 自己的 session-key 解析其实内部算对过一次默认 agent id（`resolveDefaultAgentId`，只配一个 agent 时会返回 `metaclaw_agent`），但新 session-id 第一次出现、任何 store 都找不到匹配时，兜底分支用的是原始 `opts.agentId`（`undefined`）而不是那个已经算对的结果，`normalizeAgentId(undefined)` 硬编码回落成字面量 `"main"`——OpenClaw 自己 session-key 解析代码的一处真实不一致，CLI 用本机实际安装的 OpenClaw 编译产物核对过跟源码仓库对得上
  - **"官方 Compl. 不为 0 应该做过调整吧"——查证结果：查不到确凿规避机制，如实记录不卡修复**。仓库里唯一显式传 `--agent` 的地方是另一条无关路径（`metaclaw/utils.py::run_turn`），`_register_session_in_json` 只在题目 `"update"` 字段下才触发、不是每题都走。真正原因不确定（候选：官方论文用的 OpenClaw CLI 版本更老、这段兜底当年行为不同；或官方外层脚本在别处传了等价参数），两条都没有确凿证据，留作开放问题
  - **影响范围**：这个缺口从 driver 第一次跑起来那天就存在，**至今为止每一次 MetaClaw 迁移训练/基线的 `Compl.`＝0.0% 都可能主要是这个原因**，不是 checker 真没找到文件，是文件从一开始就没写到 checker 会看的地方。修完不会让 `Compl.` 直接变成论文级别数字——文件内容对不对、`generate-fail`、同天 session 污染这些问题都还在，只是不再被"写到隔壁目录"锁死成 0
  - **修复**：给 `openclaw agent` 子进程调用显式加 `--agent {agent_id}`。没有采用"整份拷贝 `infer_cmd.py` 打补丁"（`prepare_patched_*.sh` 那套模式）——那套是给会被 slime/proxy 直接 import 的模块用的；这里训练路径只用到 `_run_openclaw_agent` 这一个函数，`_run_group`/`_run_question` 完全没被用到，为两个 argv 参数拷贝 1400 行文件过重。改法是在 `metaclaw_rollout_driver.py` 里本地复制这一个函数（~40 行，逐行对照官方版本，只加一行 `--agent`），不 import 官方版本、不碰 `MetaClaw-official/` 任何文件——跟这次迁移一贯处理官方代码缺口的方式一致（比如 `OPENCLAW_GATEWAY_TOKEN` 也是写进 driver 自己的环境变量）。`agent_id` 在本地版本里是必填参数（官方是 `str | None = None`），训练路径漏传应该直接报错，不能静默retreat回这个 bug
  - **明确没做**：MetaClaw-Bench 自己的离线评测路径（`_run_question`/`_run_group`，不经过这个 driver）目前仍未修，这次完全没用到那条路径，等真用到再处理
  - `py_compile` 通过，新增合成测试（mock 子进程调用）确认最终 argv 精确等于 `(..., "--session-id", <id>, "--agent", "metaclaw_agent", "--message", <msg>)`。**真实训练环境完全未验证**——下一轮需要确认真实 session key 变成 `agent:metaclaw_agent:explicit:...`、写入的文件出现在正确的 `workspace_{test_id}_*` 目录、checker stdout 不再清一色 `FAIL: cannot read`。同样不热补正在跑的 job
→ 详见 `metaclaw_migration_plan.md`"修复：`openclaw agent` 从未传 `--agent`，`write` 实际写进了 checker 看不到的默认 agent workspace"

## 2026-08-20

**目标：** 设计并实现"可调 K 天训练窗口 + 冻结评测剩余天数"（`METACLAW_TRAIN_UNTIL_DAY`）这个**额外、纯附加**的能力——用户和 CLI 之前讨论过，边训边考的混合 Acc./Compl. 没法直接回答"训练到底有没有提高能力"，需要一种方式能"训 K 天、冻结权重、继续跑完剩余天数"，拿冻结段的干净数字跟训练前对比。**硬性前提**：这次改动落地时，`metaclaw_migration_20260820_*` 那条训练已经跑到 day12、效果不错，这次改动完全不能影响这条正在跑的训练——默认（不设置开关）必须和当前这条训练逐字节一致，这是能不能合入的前提，不是"尽量做到"。

**完成内容：**
- 方案经过三轮 CLI 交叉审阅才定稿：第一轮 CLI 确认方向可行，指出"未设置=零触碰"要用"环境变量是否存在"而不是数字哨兵、冻结检查必须放在 proxy 分发点（driver 不改 `_run_round`）、每天都发信号（防漏发）；第二轮 CLI 核对代码后指出四处必须改准的地方——冻结信号必须绕开 `submission_enabled` 的 503（否则控制面消息本身会被暂停窗口卡住）、标志位在 OPD 但拦截必须打在 combine 的 `_maybe_submit_ready_samples`（不能只改 OPD/combine-select，会"写了标志没人读"）、dayK 尾部竞态接受不做 drain、报告只在设了 K 时才拆分；第三轮 CLI 用真实日志（`logs/metaclaw_migration_20260820_094611/...`）指出第一版方案把冻结识别写进 `_handle_request` 是错的——真实代码里 `submission_enabled` 的 503 检查在 `chat_completions` 路由函数里、`_handle_request` 之前就已经发生，冻结识别必须打在 `chat_completions` 本身，鉴权之后、503 检查之前
- **driver 侧**（`metaclaw_rollout_driver.py`）：新增 `METACLAW_TRAIN_UNTIL_DAY`（`int | None`，未设置即 `None`，`"0"` 合法且不等于禁用）；新增 `_send_freeze_signal`（复用 `_post_with_retry`，专用 header `X-Metaclaw-Freeze-Training: true`，占位 body）；`main()` 天数循环用 1-based `day_index` 判断 `is_frozen_day`，冻结的每一天调用 `run_day` 前都发一次冻结信号（不是只发一次）；`run_day`/`_run_round`/`_send_verdict_turn` 一行未改。同时把每天的 `official_score` 分流进 `train_round_scores`/`frozen_round_scores` 两个桶
- **proxy 侧，两处**：①`prepare_patched_openclaw_opd.sh`——`chat_completions` 路由新增 header 参数 `x_metaclaw_freeze_training`，在 `_check_auth` 之后、`submission_enabled` 503 检查之前识别并置位 `owner._metaclaw_training_frozen = True`、立即返回（不解析 body）；`__init__` 里显式初始化 `self._metaclaw_training_frozen = False`。②`prepare_patched_openclaw_combine.sh`——`_maybe_submit_ready_samples` 里、`skip_forced_negative_override` 之后、`eval_score` 赋值之前，加 `if getattr(self, "_metaclaw_training_frozen", False): ...continue`，跟现有 `is_aborted`/`generated_while_paused`/`is_duplicate_user_retry` 是同一个拦截模式的延伸，同时挡住 OPD 和 RL-only 两条提交路径。`openclaw_combine_select_api_server.py` 不需要改动
- dayK 尾部异步样本竞态：采纳"接受，写清楚"这个选项（CLI 认同），不做等待清空/session_id 解析，失败模式是"dayK 尾部少丢几个样本"不是"训错"
- 报告：只有设置了 `METACLAW_TRAIN_UNTIL_DAY` 才会在 `report.json` 新增 `metaclaw_train_until_day`/`metaclaw_train_window`/`metaclaw_frozen_window` 三个字段，`report.md` 新增 Train/Frozen 对照表并标注"Train window 只是过程监控，Frozen window 才是回答训练有没有用的数字"；未设置时字段集合不变
- **验证**：三个代理补丁脚本依次跑通完整补丁链，对真实官方源文件生成输出全部 `py_compile` 通过，人工核对冻结检查在 `chat_completions` 里确实排在 `submission_enabled` 之前；新增合成测试覆盖环境变量未设置/`"0"`/`"5"` 时 `TRAIN_UNTIL_DAY` 解析正确、`is_frozen_day` 公式在多组输入下结果正确、`_send_freeze_signal` 的 header/body 形状正确。**没有做也做不到端到端"默认配置逐字节对比"的本地合成测试**——`main()` 依赖真实 `openclaw agent`/代理/checker，验证交给用户接下来另开一次默认配置训练、跟当前 `day12` 这条正在跑的训练直接对比
→ 详见 `metaclaw_migration_plan.md`"方案：可调 K 天训练窗口 + 冻结评测剩余天数"

**关键决策：** 这是一个额外能力，不是对现有训练逻辑的修改或替代——`metaclaw_migration_20260820_*` 这条正在跑的训练（day12，效果不错）不受这次改动影响，也不会被热改。用户明确表示会先跑一次默认配置（不设置 `METACLAW_TRAIN_UNTIL_DAY`）的训练，跟当前这条对比，验证"改了代码但没打开开关"这件事本身没有引入行为差异，然后才会决定要不要用这个开关做实际的短训练窗口实验。

**产出：**
- `scripts/metaclaw/metaclaw_rollout_driver.py`：新增 `TRAIN_UNTIL_DAY` 常量、`_send_freeze_signal`、`main()` 天数循环/报告section 改动
- `scripts/prepare_patched_openclaw_opd.sh`：新增 `openclaw-rl-metaclaw-train-until-day` 补丁（`__init__` 标志位 + `chat_completions` 冻结识别）
- `scripts/prepare_patched_openclaw_combine.sh`：新增 `openclaw-rl-metaclaw-train-until-day` 拦截补丁（`_maybe_submit_ready_samples`）
- `docs/metaclaw_migration_plan.md`：新增"方案：可调 K 天训练窗口 + 冻结评测剩余天数"一节，含三轮 CLI 审阅过程和最终实现细节

**完成内容（续，同一天）：**
- **发现并修复第八个独立问题：day12-14 那种"超长 thinking 空转"老问题再次出现，根因是难度阶梯（P2 命名规范引入）+ 反复灌同一份刻板反馈**。CLI 排查 `metaclaw_migration_20260820_094611`（全量 30 天）时确认：day01-09 正常，day10-11 开始"只说不做"，day12-14 陷入 thinking 循环（7 万→22 万字，同一句"满足用户要求"复读几百到近两千次）——跟 Personal Agent Track 08-07~08-10 那次"超长 thinking 空转"是同一机制，换了个触发场景（P2 文件命名反复失败）
  - 经过多轮 CLI 用真实数据核实（含专门验证"不能影响 P1 现在跑得不错的部分"这个约束），定稿并实现三处独立修复：
  - **修复 1**：`_build_next_round_feedback`（新函数）给 file_check 失败的 next-round 反馈追加真实 checker stdout，带过滤——只在 stdout 以 `FAIL` 开头、不含 `Traceback` 时追加，否则整段跳过退回纯静态反馈（P1 用的 `check_iso8601` 约 1/4 失败是脚本自己崩了产出 Traceback，原样拼会更糊）
  - **修复 2**：仅当 `round_record["eval"]["command"]` 命中 `check_filename.py --dir`（P2 宽松日期模式）时，追加"日期不用精确匹配"的说明；day11 起的精确 glob 模式完全不加（题面无专门字段区分，靠解析 command 判断，检测不到就不加，不猜）——确认 P1 的 35 道 file_check 全部走 `check_iso8601`，这条改动对 P1 结构性零影响
  - **修复 3**：选择题格式失败（`_build_multi_choice_feedback` 返回官方常量 `FORMAT_ERROR`）时追加模型这次实际输出的原文片段，打破连续 20+ 次字节级相同反馈；一处改动同时覆盖 next-round 反馈和 OPD hint（两边调用同一个官方函数）；`094611` 数据里 P1 阶段格式失败 0 次，几乎不受影响
  - **修复 4**：`is_invalid_tool_use`（含规则 5 复读检测，`N=12`，已用 P1 好样本/day12-14 坏样本两面校准过，无需重新调参）接线到 MetaClaw 的 `_metaclaw_verdict`/`metaclaw_round_mode` 两条打分分支——CLI 用真实 shadow 日志确认这个标记之前只在 Personal Agent Track 分支强制 -1，MetaClaw 两条路径 52 次命中、0 次真正生效，不是重做检测，是把已有信号接到之前漏掉的地方，跟 08-19 给 step-judge 补 `is_truncated` 覆盖是同一类缺口
  - **验证**：`py_compile` 通过（driver + 三个代理补丁脚本对真实官方源文件完整跑通）；新增合成测试覆盖 `_filtered_checker_stdout`（干净/Traceback/非FAIL/超长四种情况）、`_is_dir_mode_filename_check`（`--dir`/精确glob/缺失三种情况）、`_build_next_round_feedback`（五种场景：宽松失败/精确失败/Traceback失败/通过/MC格式失败）全部符合预期；`is_invalid_tool_use` 接线位置人工核对（`_metaclaw_verdict` 分支在 hint 材料化前、`metaclaw_round_mode` 分支跟 `is_truncated` 并列）。**真实训练环境完全未验证**——不热改正在跑的训练
→ 详见 `metaclaw_migration_plan.md`"方案：next-round 反馈 + FORMAT_ERROR + is_invalid_tool_use 三处修复"

## 2026-08-21

**目标：** 记录/核实 08-20 提交的两个真实实验结果（`--agent` 修复后的新基线、`METACLAW_TRAIN_UNTIL_DAY=6` 冻结实验），过程中被用户两次纠正方法学表述错误，一并改正并补全记录。

**完成内容：**
- **`--agent` 修复后重新定版训练前基线，取代 2026-08-18 版**。用带 `--agent metaclaw_agent` 的新 harness（"agentfix"）跑了三个不同 SGLang seed 的独立基线：`247444587`（Acc=17.7%，overflow 17.9%）、`589953305`（Acc=15.3%，overflow 5.5%）、`465485731`（Acc=17.8%，overflow 4.6%）——选 Acc 最高且 overflow 最低的 `465485731` 定为正式基线，结果目录 `metaclaw-baseline-eval-aligned/run_20260820_192625_agentfix_seed465485731/run_20260820_192725/`
  - 主指标：**Acc.=17.8%**（Correct=61.676/346）、**Compl.=0.0%**（0/224 file_check，跟旧基线一样没变）、MC 均分 50.6%、MC format_valid 73.0%、Context overflow 全题仅 4.6%（MC 10 道/FC 6 道）
  - **关键结论**：旧基线（8.1%）偏低的主因确认是 `--agent` 缺口导致的大量 Context overflow（49.1%）被记成未完成，不是"4B 模型能力真的这么差"——修复后 overflow 降到 4.6%，Acc. 提升几乎全部来自 multi_choice（真正生成出答案时两版 `\boxed{}` 合规率都约 98%，模型格式遵循能力本身没变，变的是"能不能把题答完"）。**`Compl.`（file_check）新旧基线都是 0.0%，这条没变**——file_check 确实是真实能力上限，不是链路/overflow 问题，这条旧结论继续保留
  - 旧基线（`run_20260818_141305`，Acc.=8.1%/Compl.=0.0%）明确标注已作废，不再当主基线用；连带确认 08-18"`rl-training-headers` 只污染 multi_choice、不影响 file_check"那条实验结论（5.7%→8.1%）作为**相对比较**仍然成立（两次跑用的是同一套有缺陷的旧 harness），只是 5.7%/8.1% 这两个绝对数字不再当参考
→ 详见 `metaclaw_migration_plan.md`"基线结果（用于后续对比，2026-08-20 定版）"

- **里程碑：`METACLAW_TRAIN_UNTIL_DAY=6` 冻结实验，第一次拿到可信的正面训练效果证据**。用 08-20 实现的 `METACLAW_TRAIN_UNTIL_DAY`（`commit 0882f69`）跑了一次真实实验：day1-6 正常训练，day7 起冻结权重继续跑完剩余天数，日志目录 `logs/metaclaw_migration_20260820_122808`
  - 初版记录把 Frozen 窗口（day7-30）当成了主结果，**被用户指出错了**：主结果必须是本趟 driver 全程 live 聚合的 Acc./Compl.（跟论文 MetaClaw Full/Table 1 的出分方式一致——同一趟训练运行实时聚合，不是训后另开 held-out，这是这次迁移从 08-14 起就确认的方法学主线）。Train/Frozen 拆分只是同一次 run 的**分段诊断**，不能替代全程聚合当成绩，已改正
  - **主结果（全程 live 聚合）**：**Acc.=37.3%（Correct=128.069/343）、Compl.=13.9%**——相对训练前基线（seed465485731，Acc.=17.8%/Compl.=0%）**Acc. +19.5pt、Compl. +13.9pt**，这是本次迁移**第一次观测到非零的 `Compl.`**
  - 分段（诊断用）：Train day1-6（滚动权重）Acc.=61.4%/Compl.=45.0%（n=63）；Frozen day7-30（固定 ckpt）Acc.=31.9%/Compl.=7.1%（n=280）——Frozen 窗口单独看也比基线同范围（约 16.2%）高约 16pt，进一步支持"提升不是混合权重状态的假象"；未计分 3 题：`day03/r11`（MC）、`day05/r13`（MC）、`day06/r4`（FC）
  - **关键前提**：这次实验用的是 `0882f69`，**不包含同一天早些时候诊断/实现的"day12-14 thinking 空转"三处修复（`455a54f`）**——冻结窗口这次没有出现 `094611` 那种后期 Acc→0 塌陷，结构性原因是 day6 后不再继续训练，不代表 thinking 空转问题已经被验证解决
  - 跟论文 Kimi-K2.5 Baseline/Full（21.4%/40.6%，Compl. 2.0%/16.5%）仅作数量级参照，模型和方法都不同，不是同条件对比
  - 局限：这次冻结窗口的评测走的仍是训练自己的 harness（proxy/`rl-training-headers`），跟官方独立 `metaclaw-bench run` 不完全同构，严格 apples-to-apples 还需要另外单独用官方 bench 重跑一次这个冻结 checkpoint
→ 详见 `metaclaw_migration_plan.md`"里程碑：K=6 冻结实验——首次观测到训练带来的真实、可信的能力提升"

- **`METACLAW_TRAIN_UNTIL_DAY` 设计动机表述被用户两次纠正，已改正**：
  1. 第一次：写"方案：可调 K 天训练窗口"这节时，只记了 CLI 最早提议的两个理由之一（混合数字没法单独回答训练有没有提高能力），漏了另一个（训练继续太久容易训坏、K 是止损旋钮）——已补全两条理由。
  2. 第二次、更根本：**"混合数字没法直接回答训练有没有提高能力"这句话本身说得太绝对，被用户指出方向错了**。这套"边训边考、全程混合数字直接跟基线比"就是论文自己验证 Hybrid RL 有没有用的办法，没有缺陷；真正的问题是**这次迁移现在还没有能力产出一个"训练全程 30 天不崩"的完整结果去跟基线比**（`094611` 证明了继续训练到 day12+ 会撞上 thinking 空转塌陷）。`METACLAW_TRAIN_UNTIL_DAY`/冻结窗口是这段时间的**临时权宜工具**，不是要长期维持的替代方法学——训练能稳定跑满 30 天之后，应该直接切回"训练前 30 天基线 vs 训练后 30 天混合数字"这套论文原生比法，冻结窗口就不再需要了。已改正 `metaclaw_migration_plan.md` 的"动机"段落和 K=6 里程碑的"意义"段落，并把这条"最终目标"写进了下一步的优先级排序（训练稳定跑完 30 天 > 继续做冻结实验）
  - 这次纠正也存成了一条通用记忆（`feedback_workaround_framing`）：写临时权宜工具的设计动机时，必须明确写"最终目标仍是标准方法，这只是当前某个前置条件没达成时的临时替代"，不能把临时现状包装成永久方法学结论
→ 详见 `metaclaw_migration_plan.md`"方案：可调 K 天训练窗口"节的"动机"段落 + K=6 里程碑节的"意义"段落

**关键决策：** `METACLAW_TRAIN_UNTIL_DAY`/冻结窗口明确定位为"训练还没法稳定跑满 30 天"这个现状下的临时诊断工具，不是永久方法学——迁移完成的最终目标是训练能稳定跑满完整 30 天，直接用"训练前基线 vs 训练后混合数字"这套论文原生方法学对比，不需要任何冻结/分段技巧。

**产出：**
- `docs/metaclaw_migration_plan.md`：新增 08-20 版基线记录（取代 08-18 版）、K=6 里程碑记录（含出分口径更正）、"方案：可调 K 天训练窗口"动机段落两次改正
- `C:\Users\maozh2\.claude\projects\D--MAO-Claude\memory\feedback_workaround_framing.md`（新建，通用记忆，非项目文档）

## 2026-08-25

**目标：** 排查 K=6 实验里 3 题未计分的根因，修复发现的新问题（暂停窗口砍断在飞生成时被误判成 timeout、没吃到 08-19 的 503 耐心等待逻辑）。

**完成内容：**
- **发现并修复：暂停窗口把正在飞的生成砍断后，OpenClaw 报的是 timeout 不是 503，漏出 08-19 那次修复的覆盖范围**。K=6 冻结实验里未计分的 3 题（`day03/r11`/`day05/r13`/`day06/r4`）查出同一个根因：不是"新请求撞上已暂停的 503"，是**暂停发生那一刻这道题的生成正在飞、被 SGLang `pause_generation` 中途砍断**——代理侧 `is_aborted`/`degraded-turn-drop` 正确丢弃了这个残缺样本（训练信号没被污染），但 OpenClaw 网关把这次中断报给 driver 的方式是 `GatewayClientRequestError: FailoverError: LLM request timed out`，不是 `"503 status code"`，08-19 那次修复的 `_AGENT_PAUSE_MARKER` 只认后者，这次匹配不上，`AGENT_RETRY=0` 直接判 infra failure，题目从聚合里被剔除
  - 跟 08-19 的关系是"同一诱因换了表现形态"，不是推翻——CLI 当时明确把 timeout 排除在耐心等待之外，理由是"timeout 通常是模型真的在深度生成中挂了，跟一上来就被拒的 503 不一样，盲目等可能拖长真实的 GPU 争用问题"。这次核实：至少这一句特定的 OpenClaw 文案，在能查到的样本里 100% 是暂停窗口导致的——K=6 这 3 题全部对齐，回溯扫描全部历史 migration run 找到 28 次真正的 `openclaw agent failed + LLM request timed out`，**28/28 对齐、零反例**
  - **修复**：`_AGENT_PAUSE_MARKER`（单字符串）扩成 `_AGENT_PAUSE_MARKERS`（元组，加入 `"LLM request timed out"`），命中任意一个都进现有耐心等待重试环，完全复用现有的 `PAUSE_RETRY_INTERVAL_SECONDS`/`PAUSE_RETRY_MAX_WAIT_SECONDS` 预算，不新增参数；日志措辞从"503 pause-retry"改成"pause-retry (matched %r)"带上具体命中哪个 marker
  - **收窄不是收回**：只加这一句特定稳定文案，不是"所有 timeout 都当 503 处理"——driver 自己未来若真设置了 `round_timeout`、触发它自己的 `asyncio.wait_for` 超时，那种超时跟暂停窗口无关，仍然立刻判失败，这个边界没变
  - Plan B（proxy `/healthz` 暴露暂停状态、精确轮询）暂不做——28/28 零反例说明现有文案匹配已经够可靠，等以后真出现"timeout 明显跟暂停无关"的反例再考虑
  - **验证**：`py_compile` 通过，新增合成测试覆盖三种场景（timeout 一次后成功重试、持续 timeout 耗尽预算判失败、跟暂停无关的普通失败零等待立刻判失败）全部符合预期。**真实训练环境完全未验证**——不热改正在跑的训练
→ 详见 `metaclaw_migration_plan.md`"修复：暂停窗口把正在飞的生成砍断后，OpenClaw 报的是 timeout 不是 503"

**产出：**
- `scripts/metaclaw/metaclaw_rollout_driver.py`：`_AGENT_PAUSE_MARKER`→`_AGENT_PAUSE_MARKERS`，`_run_round` 匹配逻辑、日志措辞改动
- `docs/metaclaw_migration_plan.md`：新增"修复：暂停窗口把正在飞的生成砍断后，OpenClaw 报的是 timeout 不是 503"一节

**完成内容（续，同一天）——一项发现 + 一次自我纠正：**
- **CLI 深挖 `094611`（全量 30 天）时间线，先划出三阶段真实数据**（阶段 A day1-10：最终失败题里 71% 中段步骤仍是 +1，57/83 道全部中段步都是 +1；阶段 B day11-16：中段 RL+ 比例 91%→15%，工具开始反复乱调；阶段 C day17-22：几乎只剩 `finish_reason=length`+`idle timeout` 文案，样本被 `is_duplicate_user_retry` 整段 drop，训练样本变 0），当时据此推出"主根因是中段 step-judge 长期跟最终结果脱钩（信用分配错误），`455a54f` 只动了表面症状"——**这个"主根因"判断后来被用户指出、CLI 复核后撤回**：阶段 A 里中段 +1（写了文件、已经很接近正确答案）本身是合理的过程 shaping，不是训练信号污染；真正该解释的问题是"持续的最终 -1/OPD，为什么没能把已经很接近的近似解拧成精确解，反而滑向空转"——这个问题的答案就是下面这条 `check_filename` 累计计数发现，不需要另外假设"中段奖励结构错了"。阶段 A/B/C 的真实时间线数据本身没错，保留；"中段脱钩是主根因"这个结论和对应的"重做中段奖励"这个待讨论方向已撤回
  - 用户追问 MetaClaw 官方对中间步骤是不是完全不管，核实（读 `metaclaw/prm_scorer.py` 源码）：**不是不管，是每一轮（不分中间/最终）都用同一个判官打分，且判官 prompt 明确写"不看后续轮次"**——MetaClaw 自己的训练奖励从来没有确定性最终结果参与，这次迁移是自己主动引入"最终用 checker 确定性 ±1"这个设计才第一次制造出"中段可能跟最终脱钩"这个矛盾，MetaClaw 没有对应解法可抄——这部分判断不受上面撤回影响，仍然成立，只是现在更多是背景知识，不再是某个决策的直接依据
- **发现（用户，已用真实代码+真实题面核实）：`check_filename.py --dir --min-count` 是累计跨轮次计数，一旦落后就结构性补不回来——正确的做法也会被判负分**。读了 `check_filename.py` 全文和真实 `day06/questions.json` 的 `eval.command` 序列（`r1` 默认 min-count=1、`r2` 改成 2、`r4`→3、`r7`→4、`r9`→5，逐题递增），确认：checker 每次现场数当天共享目录里有多少个合规文件，跟 `min_count` 比大小，不是只看这一轮新写的文件对不对。题干只要求每轮写 1 个新文件，min-count 的设计假设"每题都写对、累计数正好跟上"——**一旦某道题落后（比如文件名不合规），后面每题只让新写 1 个，永远追不平，即使某一轮自己写得完全正确，checker 依然会判 -1**。**这现在是"越写越长/thinking 空转"这整条链路的真正解释，不是一个平行的独立根因**——模型确实在往正确方向靠近，但因为当天更早的题目已经欠账，怎么改都还是 -1，持续收到这种"看似纠偏、实际拧不动"的负反馈才是把模型逼向空转的机制
→ 详见 `metaclaw_migration_plan.md`"查证记录：`094611`'超长 thinking 空转'复现的时间线"+"重大发现：`check_filename.py --dir --min-count` 是累计跨轮次计数"

**完成内容（续二，同一天）——分阶段核查 day01-30 全部检查方式，发现第二类更严重的累计问题：**
- 用户要求核查 day15 及之后是否每 5 天一个阶段、用不同检查方式——读了全部 30 天 `questions.json` 的 `eval.command` 加 `eval/scripts/` 下全部 5 个 checker 脚本源码（`check_filename.py`/`check_iso8601.py`/`check_metadata.py`/`check_backup.py`/`check_done_log.py`），确认**不是同一种检查方式**，分六个阶段：day01-05 单文件内容检查（`check_iso8601.py`，无累计问题）；day06-10 `check_filename.py --dir --min-count`（已知累计缺陷）；day11-15 题面自带 `python -c` 手写 `len(glob(...))>=N`（结构同 day06-10，只是没有 `--dir` 这种干净 CLI flag，且常 `&&` 接一个对选中文件的 `check_metadata.py`）；day16-20 新引入 `check_backup.py`（单文件、无累计问题）与 `--dir --min-count` 混用；day21-25 新引入 `check_done_log.py --min-entries`（**更严重的累计问题**，见下）；day26-30 前几阶段的检查方式全部混用。
- **新发现：`check_done_log.py --min-entries` 不是"数量不够就 FAIL"，是每次调用都把 `done.log` 从头到尾整份重新逐行校验格式**——只要曾经有一轮往里面写错过一行（哪怕很多轮之前），之后所有轮次不管新写的行多规范，永远判 FAIL，是**永久性**损坏，比 `--dir --min-count` 的"门槛追不上"更彻底（后者理论上门槛不再涨还有机会追平，这个只要历史里有一行格式错就再也过不去）。`check_backup.py`/`check_metadata.py`/`check_iso8601.py` 三个逐个核实过，均为单文件校验，本身没有累计问题，但常被 `&&` 接在有累计问题的检查后面，一旦前面短路 FAIL，后面这几个原本没问题的检查根本没机会跑。
- **修复方向讨论收敛（仍未实现，未交 CLI review）**：核心是"round 开始前/结束后 diff"——文件累计类（day06-23 部分题目）diff 文件集合、找出这一轮新写的文件，单独跑跟 day01-05 单文件模式同口径的命名+扩展名判定；日志行累计类（day21-30）diff 日志行、只对新追加的行单独校验格式，不看历史行。两类都只改"训练用 `eval_score` 怎么算"，官方 Acc./Compl. 继续用 checker 原始口径不动，保住跟论文可比性。反馈文案（`_build_opd_hint`/`_build_next_round_feedback`）要同步改，现在这两类失败反馈都只有聚合数字（"found 3, need 5"），从不像单文件模式那样指出"这一轮新写的文件/日志行具体哪里错"，需要用同一次 diff 顺带给出具体诊断，避免奖励判断和反馈文字互相打架。明确否决了两个替代方案：让 agent 回头补写早前欠账的文件（改变题面指令）、driver 判负后自己塞一个"正确"文件把计数补平（伪造 workspace 状态，比现状更失真）。
→ 详见 `metaclaw_migration_plan.md`"补充查证：day01-30 全部检查方式分阶段核查，发现第二类更严重的累计问题（`check_done_log.py`）+ 修复方向讨论"

**完成内容（续三，同一天）——把修复方向落成具体设计，写入文档待 CLI review：**
- 新增"方案：round 开始前/结束后 diff 判定训练奖励 + 复用同一次 diff 生成具体反馈"一节：文件累计类（day06-23）用"round 前后扫目录取差集"判定这一轮是否新增合规文件，`--dir` 模式复用参数解析、glob 模式直接对解析出的字面量表达式跑 `glob.glob()`；日志行累计类（day21-30）用同样的 diff 原理对 `done.log` 只校验新追加的行，不重新校验历史行；`&&` 复合命令拆开分段独立判定，累计段用新逻辑、其余段（backup/metadata/iso8601）照官方逻辑单独跑，不再依赖短路。同一次 diff 顺带给出具体反馈文案（区分"没写新文件"/"写了但命名不对"/"命名对但扩展名不对"），不再是笼统的聚合数字。只改训练用 `eval_score`，官方 Acc./Compl. 口径不动。
- 明确列了 4 个未细化、留给 CLI 的问题：glob 表达式解析在全部 30 天真实数据上是否稳定；拆分 `&&` 后多出的 subprocess 调用对耗时的影响；`done.log` 非纯追加场景（真实数据里有没有出现过）；训练侧接线点（`_metaclaw_verdict`/`metaclaw_round_mode` 分支）怎么改，这次讨论完全没涉及
→ 详见 `metaclaw_migration_plan.md`"方案：round 开始前/结束后 diff 判定训练奖励 + 复用同一次 diff 生成具体反馈"

**完成内容（续四，同一天）——CLI 两轮真实数据核对，方案定稿为 v2，Phase 1 确认可进入实现：**
- **第一轮核对**：CLI 用 30 天真实 `questions.json` + 现有 driver 代码做只读核对，方向确认可行，但指出 v1 的 4 类边界问题——(1) glob 段不能一律走"抠 glob+PATTERN diff"，实际有存在性检查（27 题）、内容校验（2 题，需 fallback）、过滤式 glob（1 题，需 fallback）、双 glob 双阈值（1 题，需两组独立 diff）四种子类型；(2) `--dir` 范围写窄了，实际 70 题（day16-23 还有 33 题），不应按 day 段划分应按 `eval.command` 特征分类；(3) 训练判 pass 时反馈也要跟着切，不能只改 `eval_score`/OPD hint 漏掉 `_build_next_round_feedback`；(4) `done.log` 非追加场景兜底策略合理但需 runtime 监控。同时确认了接线点比预想简单——`prepare_patched_openclaw_combine_select.sh` 不用改，只需改 driver 主循环（约 1420 行）三行。
- **据此修订为 v2**：新增 relax-only 安全约束（`seg_training_pass = seg_official_pass or seg_round_local_pass`，diff 只能把官方判负翻成正，不能反过来，防止在存在性检查题上比官方更严）；按 `eval.command` 特征分 8 类（A1/A2/A3/B/C1/C2/C3/D），C1/C2/C3 明确 fallback 官方或留到 Phase 2；引入统一 `training_passed` 驱动 `eval_score`/OPD hint/next-round 反馈三处。
- **第二轮核对**：CLI 确认 relax-only 逻辑正确、正好堵住 A3 假阴性；分类表与 30 天数据一致；累计段也跑一遍官方命令的额外开销经全量 398 段扫描确认安全（无写操作迹象）；额外钉了一句此前模糊的实现约定——失败文案以 diff 诊断为主，官方静态聚合文案只能是次要补充，不能盖过具体诊断。**结论：Phase 1（A1/A2/A3/B/D + relax-only + 三处接线）可以进入实现**，C1/C2/C3 fallback 官方。
- 已把 v2 定稿写入 `metaclaw_migration_plan.md`，取代 v1 那一节
→ 详见 `metaclaw_migration_plan.md`"方案 v2：round 前后 diff 判定训练奖励"

## 2026-08-26

**目标：** 把 08-25 定稿、经 CLI 两轮核对确认的 diff-based 修复方案（v2）落成代码，覆盖 Phase 1 范围。

**完成内容：**
- **已实现**（`scripts/metaclaw/metaclaw_rollout_driver.py`）：`_split_command_segments`/`_classify_segment`（8 类分类）、`_snapshot_segment`/`_prepare_before_snapshots`（before/after 快照）、`_diagnose_file_segment`/`_diagnose_log_segment`（具体反馈诊断）、`_rerun_segment_official`（segment 级独立重跑）、`_compute_training_verdict`（relax-only 汇总，`training_passed`/`training_hint` 唯一产出点）；`_run_round`/`_build_next_round_feedback`/`run_day` 主循环三处接线全部改成消费 `training_passed`/`training_hint`；`combine_select` 补丁按 CLI 确认未改动。
- **实现中发现并修复一个 v2 设计没预料到的分类 bug**：`check_metadata.py $(python -c "...glob.glob(...)...")` 这种"用 glob 选目标文件、本身是内容检查"的写法，最初被误分类成 A2A3——已加前置守卫（段内出现 `check_metadata.py`/`check_backup.py`/`check_iso8601.py` 直接判 OFFICIAL）修复。
- **验证**：`py_compile` 通过；单元级合成测试（tmp workspace）覆盖净增升级、无新文件/命名错误的具体诊断、官方已过不碰 diff、复合命令部分净增仍判负、`done.log` 历史改写正确退化并打日志，全部通过；**全量 30 天真实数据分类扫描**（不是单个样例）得到 A1=70、A2A3=75（=CLI 报告的 48+27）、B=75 段，**跟 CLI 两轮真实数据核对报告的数字完全一致**。
- **仍未验证**：真实训练环境下的实际效果（day12+ 是否还复现 `094611` 崩溃模式）、`done.log` 非追加场景真实触发率（监控日志已埋点）、`_rerun_segment_official` 额外 subprocess 调用在真实训练节奏下的耗时影响
- 提交并推送（commit `d7b231f`，`feat(metaclaw): implement Phase 1 relax-only training verdict via round-local diff`，由 modelfactory 侧的 Cursor agent 代为 commit+push，非本地直接操作）
→ 详见 `metaclaw_migration_plan.md`"Phase 1 已实现"一节

**产出：**
- `scripts/metaclaw/metaclaw_rollout_driver.py`：新增约 300 行分类/快照/诊断/汇总函数，`_run_round`/`_build_next_round_feedback`/`run_day` 三处接线改动
- `docs/metaclaw_migration_plan.md`：新增"Phase 1 已实现"一节

## 2026-08-27

**目标：** 核实 Phase 1 修复在首次真实训练里的实际表现，处理核实中发现的问题。

**完成内容：**
- **CLI 用真实训练数据核实打分侧，结论是"改对了"**：
  - **4 个升级案例全部核实无误**（`day06/r2`、`day08/r5`、`day08/r11`、`day09/r2`）——四个都是标准目标场景：本轮按要求写对了一个合规文件（如 `20260827_test_results_summary.json`），只因前面轮次欠账、累计总数没到 `--min-count` 阈值而被官方判 FAIL，现在全部拿到训练 +1
  - **`passed=True` 而 `training_passed=False` 的反向情况 0 次**——relax-only 约束在真实数据上成立，不只是合成测试里成立
  - **诊断文案确实具体化了**：`'ci_build_report.json' does not match YYYYMMDD_snake_case.ext pattern`，取代了原来的 `expected >= 2, found 1`；53 次 `no new file was created in this round` 对应工具调用坍缩那些轮，诊断准确
  - 顺带确认设计预期成立：升级案例里 agent 用的是真实日期 `20260827` 而非题面场景日期 `20260326`——`--dir` 模式下任何 8 位日期都合规，判 +1 正确；day11+ 精确日期 glob 下同样写法不会被升级
- **修复一处我在 Phase 1 自己引入的反馈质量回归（CLI 发现）：14 处 Python Traceback 泄漏进 agent 可见的 `[Previous Feedback]`**（判分不受影响）
  - **根因不是"少了一层过滤"，是我多加了一层不该加的 fallback**：这个 hint 有两个消费者、需要两种不同的兜底——OPD hint（`run_day`）要 `_build_opd_hint` 的原始 stdout（本机制存在前就是这个行为），agent 可见反馈（`_build_next_round_feedback`）要 `_filtered_checker_stdout`（专门丢弃含 Traceback 的 stdout，这正是 08-20 加这个函数时的原始动机）。两个调用点本来各自都写好了自己的 fallback，但我在 `_compute_training_verdict` 内部又补了一次 `else _build_opd_hint(...)`，**等于在上游把两种策略强行统一成了 OPD 那一套**，agent 可见路径就走进 `if training_hint:` 分支、跳过了自己的过滤
  - **修复（一行）**：`_compute_training_verdict` 只返回 diff 推导的诊断、没有就返回 `""`，不在函数内部做任何 fallback；两个调用点的 fallback 自动各自恢复。两处 docstring 补了说明，讲清楚"这两个 fallback 故意不同、不能在上游合并"，避免以后被当重复代码合掉
  - **验证**：`py_compile` 通过；新增合成测试复现 CLI 报的真实场景（checker 崩溃吐 Traceback 的 OFFICIAL 类轮次），确认 agent 可见反馈不含 Traceback、官方静态文案仍在、**OPD hint 仍拿到原始 stdout（行为未变）**、diff 诊断存在时仍优先于聚合计数行；**并确认该测试非空转**——手动模拟修复前行为后重跑，Traceback 确实泄漏
  - **教训**：Phase 1 的合成测试只覆盖了"diff 能产出诊断"的主路径，没覆盖"diff 产不出诊断、走 fallback"的兜底路径，而 bug 恰恰在后者；全量 30 天分类扫描也帮不上忙（只验证分类计数，不验证反馈文本内容）。**主路径验证通过不等于兜底路径验证通过**
  - 提交并推送（commit `40c5450`）
→ 详见 `metaclaw_migration_plan.md`"首次真实训练核实：打分改对了，但暴露一处我自己引入的反馈质量回归"

- **清掉 `git status` 里长期挂着的 `M scripts/launch_simulator.sh`（本地环境噪音，非项目内容，记一次免得反复排查）**：现象是 `git diff` 空、`git diff --numstat` 也空，但 `status` 一直显示 `M`、`update-index --refresh` 报 `needs update`。真实机制不是"CRLF 换行符差异"这种含糊说法（这是第一次给出的不完整解释，被追问后才查到底）：**索引 stat 缓存记的 `size=2195` 是 CRLF 检出形态，工作区实际文件是 LF 形态 `size=2124`**，差的 71 字节正好是该文件行数；git 每次看 size 不符就标记"可能改动"，读内容比对后又发现一致，于是 `diff` 空而 `M` 不消。三个哈希（索引 blob / 工作区 raw / 工作区经 clean filter）实测完全相同，确认零内容差异。**修法是 `git add` 该文件**——用当前 stat 重写索引缓存，内容经 clean filter 后与索引 blob 一致，因此暂存区为空、无任何东西可提交、工作区文件一字节未动。`core.filemode=false`，工作区 755 vs 索引 100644 的模式差异 git 本来就忽略，不是原因。

**产出：**
- `scripts/metaclaw/metaclaw_rollout_driver.py`：`_compute_training_verdict` 去掉内部 fallback，两处 docstring 补充双 fallback 策略说明
- `docs/metaclaw_migration_plan.md`：新增"首次真实训练核实"一节，"Phase 1 已实现"标题日期从 08-25 更正为 08-26


---

## 2026-08-28

**目标：** 归档实习小结材料；重整工作记录结构，解决"最新进展被埋在文件底部"的问题。

**完成内容：**
- **实习小结材料入库**（commit `c400ec6`，由 modelfactory 侧 agent 提交）：`report/` 全套 113 个文件纳入版本控制——最终 PPT（`实习小结_毛泽辉.pptx`）、生成/预览脚本、渲染图（Table 3 / Table 1 / 迁移架构 / 对话证据）、讲稿、QA 备份、简历实习经历初稿。`.gitignore` 里原有的 `report/` 条目同步删除，否则新增文件会被静默忽略、误以为已提交。该提交同时捎带了 08-27 条目里补的 3 行（commit hash + `launch_simulator.sh` 排查记录）
- **重整 `work_log.md` 结构**：拆出 [`status_history.md`](status_history.md)，把 37 个已被取代的历史状态块（1024 行）全部迁入、按日期倒序排列；唯一的「当前状态」移到 `work_log.md` 顶部。`work_log.md` 从 2689 行降到 1665 行
  - 动机：历史状态块此前散落全文各处、合计占 39%，当前状态排在文末，导致每次找最新进展都要翻过 2000 行
  - **迁移前后做了逐行内容校验**（Counter 比对全部非空行）：唯一差异是 8 个被改名的标题（`###`→`##` 统一层级）和归档文件的 5 行头部说明，**零内容丢失**
  - 顺带修正一个孤儿块：`## 当前状态（2026-07-13）` 从未改名成"历史状态"，一直冒充当前状态，迁移时更正为 `## 历史状态（2026-07-13，已被 7/14 结果取代）`
  - 「工作记录规范」一节同步更新：新增文件分工表、「当前状态放顶部且全文只留一份」的维护规则、以及**日期以真实提交时间为准**这条（此坑已踩过三次）

**完成内容（续，同一天）——定位 `20260827_163030` 训练的退化机制，出消融方案待查验：**
- **经过两次自我纠正才定位到真正的机制**（两次都被 CLI 用真实数据否掉，过程见迁移文档）——
  - 第一次：拿 `day04/r5/turn3` 当"任务完成后继续改无关文件仍拿 +1"的证据。CLI 查出该轮 checker 实际是 `passed=False`（`check_iso8601.py` 抛 `AttributeError`），任务压根没完成，样本混了两个因素，只能证明"判官不惩罚改动本任务未要求的文件"
  - 第二次：据此提的统计口径（"完成后额外操作比例随天数上升"）**在现有日志里根本算不出来**（没有逐 turn checker 状态），且实际统计**否证**了该预期——中间步骤 +1 高峰在 day07-12，day13 后反而下降
  - 第三次（当前结论）：用户指出前两轮框架本身有问题——K=6 不是目标而是控制组；**MC 退化更可能是 FC 训练的连带损害**。这直接推翻了我"差距在 MC 则判官消融无关"的说法：样本来源是 FC 不代表影响只限于 FC，改的是同一份权重。按样本构成算，**权重约 90% 由 FC 派生信号塑造，MC 只占约 10%**
- **CLI 补的两组数据让链条闭合**：day16-22 的 MC 失败**主要是格式失败**（本次 17/26，K=6 为 0/27），且 **day17 出现 thinking 断崖**（day16 平均 18k → day17 115k，`finish_reason` 从 stop 变 length/abort；K=6 同期全部是 stop）。**真正的转折点是 day17，不是此前一直以为的 day12-14**。链条：FC 中间步骤判官 +1（64 个样本 thinking ≥20k 且 +1）→ `loss_mask` 整段含 `<think>` 进 loss → thinking 膨胀 → 撞 length 上限 → MC 输不出 `\bbox{}` → MC 崩塌
- **顺带澄清**：day11-15 的 FC 全 0 两边都一样，是 `check_metadata.py` 三字段硬要求造成的**共同能力墙**，跟训练信号无关，不要跟上面的链条混在一起
- **出了消融方案（纯设计，代码未动，待 CLI 查验）**：中间轮次**保留为样本、样本量不变**，只把 reward 从判官分换成本轮最终 checker 的确定性 ±1；`METACLAW_MIDROUND_REWARD` opt-in 开关，默认沿用现有行为。否掉了"直接删掉中间轮次"的形态——那样样本掉 72%，跟"训得少所以好"混在一起无法归因，且只留最终轮等于训总结、丢动作
  - **核实到一件事让这个方案变得可行**：08-17 暂缓该设计的理由（"代理没有 round 概念"）**已随 08-19c 每轮一个 session 的改动自动失效**，`session_id` 现在就是 round 边界，`_pending_turn_data[session_id]` 天然覆盖整轮
  - 采纳 CLI 的三条实现修正：verdict 不是新 turn 而是触发最后一个真实 turn 的评估、滞留时必须缓存完整 `opd_result`（含两个对照分数）、赋分与提交必须早于 cleanup 且要处理 verdict 后才完成的 judge task
  - 自己补了一条 CLI 没提的竞态：**基础设施失败路径不产生 verdict，滞留轮次移出 `pending` 后 `force_drop` 就看不见了会永久泄漏**，必须由 OPD 层显式置位"verdict 已触发"标记来区分，靠任务完成状态推断不可靠
→ 详见 `metaclaw_migration_plan.md`"诊断：day17 thinking 断崖…"与"方案（待 CLI 查验，未实现）：中间轮次改吃本轮最终 checker 结果的消融实验"

**产出（08-28）：**
- `docs/status_history.md`（新建）：37 个历史状态快照倒序归档
- `docs/work_log.md`：当前状态移至顶部，规范一节更新
- `docs/metaclaw_migration_plan.md`：新增诊断一节 + 消融方案设计一节

---

## 2026-08-31

**目标：** 按 CLI 查验结果实现消融开关。

**完成内容：**
- **CLI 查验补了一条必要边界（已采纳）**：原设计漏了"verdict 已触发但 `_opd_evaluate` task 抛异常/返回无效结果"这种情况——`force_drop` 因 `verdict_turn` 已置位而不丢，异常分支又永远产不出 outcome，**滞留样本既不提交也不清理、永久留在内存**。已补成四个明确终态（`pending`/`succeeded`/`failed`/`no_verdict`），失败一律 discard-and-cleanup。实现上有个细节：task 抛异常时 `opd_result` 根本不存在、读不到 `metaclaw_verdict` 标记，**唯一能识别"失败的是 verdict task"的办法是拿 `turn_num` 跟记录的 `verdict_turn` 比对**——这也是该字段存轮次号而非布尔值的原因
- CLI 另外三条验收点也已落实：现有丢弃门控排在滞留逻辑之前（滞留不能复活本该丢弃的样本）；滞留逻辑插在 `_eval_scores.append` 之前（判官分不会被记成实际训练 reward）；"逐字节不变"改为语义不变（提交数量/reward/丢弃规则/verdict hint/非 MetaClaw session 五项）
- **已实现**：`prepare_patched_openclaw_combine_select.sh`（加显式标记 + 两个对照分数，判分逻辑零改动）、`prepare_patched_openclaw_combine.sh`（开关读取 + 异常路径终态 + 滞留/继承/flush 主逻辑 + 循环后基础设施失败清理）、`prepare_patched_openclaw_opd.sh`（`_metaclaw_round` 状态初始化 + 记录 `verdict_turn`）、`run_metaclaw_migration_modelfactory.sh`（**把 `METACLAW_MIDROUND_REWARD` 传给训练后端进程**——读它的是被训练进程 import 的代理侧代码，只传给 driver 会静默失效）
- **验证**：三个补丁脚本对**真实官方源码**跑完整补丁链全部成功（九处锚点全部匹配）、`py_compile` 通过；行为测试从**补丁生成的真实代码**里抽出 `_maybe_submit_ready_samples` 执行，22 项断言覆盖八个场景（judge 模式三项、outcome 正常路径、判官 task 晚于 verdict 的竞态、verdict 抛异常、verdict 返回无效值、基础设施失败、verdict 在飞时不得丢弃、两个既有门控仍先生效）全部通过；**测试非空转**——场景 6 与场景 7 互为反例，判据写死成任一方向都会有一项失败
- **真实训练完全未验证**
→ 详见 `metaclaw_migration_plan.md`"已实现（2026-08-28…）"

**产出：**
- `scripts/prepare_patched_openclaw_{opd,combine,combine_select}.sh`、`scripts/metaclaw/run_metaclaw_migration_modelfactory.sh`：消融开关三层实现 + 启动脚本传参
- `docs/metaclaw_migration_plan.md`：新增"CLI 查验补的一条边界"+"已实现（2026-08-28…）"两节

**完成内容（续，同一天）——outcome 消融首次真实训练，训崩；查清根因并修两处独立缺陷：**
- **首次 `METACLAW_MIDROUND_REWARD=outcome` 训练发散，checkpoint 已污染不能作后续起点**。CLI 逐 step 核实：step 0 起 batch reward 就偏负（3/13），step 4 grad_norm 跳到 16.1、policy drift 0.86，**step 6 起全 16 个样本都是 -1**，grad_norm 冲到 2543.9、drift 21.94，响应长度从约 1370 掉到 160。同期 K=6 的 judge 模式 batch reward 全为正、grad_norm 1.4-3.3
- **机制已在源码层面确认**：`slime/utils/ppo_utils.py::get_grpo_returns` 把原始 reward 直接广播到每个 token，配上 `--n-samples-per-prompt 1` + `--disable-rewards-normalization`（我们的脚本原样继承官方），**advantage 就等于原始 reward、无任何组内归一化**——全 -1 的批次等于把模型刚产出的一切统一往下压
- **为什么 outcome 模式必然全负**：round 通过率只有约 17%，而 judge 模式下中间步骤约 69% 是 +1——**step judge 一直是这套训练里唯一稳定的正信号来源**。而且失败 round 轮次更多，样本层面的负样本占比比 round 通过率还低。**重要推论：用户最初设想的"只留最终轮"（V1）有同样的问题**，最终轮通过率同样 17%、每批只有 16 个样本更容易凑出全负批——不是思路不对，是两者共享同一个致命前提
- **修复两个独立缺陷（commit `2944f87`）**：(1) 硬负分被 outcome 继承覆盖——`day02/r2/turn2` 日志链完整（`eval_score 1.0 -> -1.0` → `inherited_outcome=1.0` → `submitted score=1.0`），重复/退化的工具调用被正向强化；改成 `_opd_evaluate` 上报 `hard_negative` 标记、两条提交路径都遵守，继承只替换判官分不替换硬规则判定。(2) 结构性无效工具调用首发漏检——查证发现缺口比 CLI 描述的更大：**规则 4 整段只对 Personal Agent Track 生效**（靠 `student-hw-N-` 推期望路径），MetaClaw 这边空参数 `write` 一条规则都没覆盖；新增任务无关的规则 6（参数无法解析、或 write/edit 无 path）
- **两个修复都会让负样本变多**，对全负 batch 是雪上加霜——**在 reward 方差问题解决前，它们不构成"可以重新提交训练"的条件**
→ 详见 `metaclaw_migration_plan.md`"`metaclaw_migration_20260831_154301` 复盘"

**完成内容（续二，同一天）——查清 toolcall-rl 与 MetaClaw 各自怎么处理中间步骤，据此出两条候选路线：**
- **修正了"三方对照"表里一处不准确的记载**：此前把我们的步骤判官记成"刻意对齐 toolcall-rl"。逐字读源码后确认——同样是二档 ±1，但 **toolcall-rl 的步骤分是整条轨迹取平均后加权加到 outcome 上**（`final_score = base_score + prm_step_coef * prm_step_mean`，判错被平均稀释、确定性 `base_score` 永远在场当锚点），我们是**每个中间步骤直接拿自己那份 ±1 当完整 reward**（判错原封不动进训练）。**风险等级完全不同**
- 另查实 toolcall-rl 有两处我们没有的机制：失败时按工具调用次数给补偿且**负分下限钳在 -0.6**、`max_tool_calls` **硬性轮次上限**
- **MetaClaw 侧**：每次 LLM 调用一个 sample，但 **`score == 0.0`（判官拿不准）的样本整段 `loss_mask=0` 不参与训练**（三档判官），配 `at-least-one guarantee` 兜底。我们的判官是二档、没有"弃权"档
- **共同点**：两边都不会让一个 round 产出十几二十个独立样本（toolcall-rl 结构上不可能 + 轮次上限；MetaClaw 靠 `score==0` 排除）——**我们两条都没有，这正是全负 batch 的直接成因**
- **据此记录两条候选路线，决定分别实验、不合在一起跑**（否则无法归因）：路线 A 轨迹级样本（照搬 toolcall-rl 结构，含 CLI 核实的 6 项必做工作和"OPD 怎么办"这个最大岔路）；路线 B toolcall-rl 式奖励合成（不动样本结构，代价远小，但需配合样本数上限）
→ 详见 `metaclaw_migration_plan.md`"查证记录（六）"+"方案：两条候选路线，分别实验"

**产出（续）：**
- `scripts/prepare_patched_openclaw_{opd,combine,combine_select}.sh`：硬负分优先级 + 规则 6
- `docs/metaclaw_migration_plan.md`：新增训练复盘、查证记录（六）、两条候选路线三节

**完成内容（续三，同一天）——查证归一化能否救全负 batch：不能，且反转了两条路线的优先级：**
- **查证结果是否定的**：`openclaw_combine_api_server.py:88-89/133-134` 显示 **`sample.group_index = next(self._group_counter)`——每个样本都被放进只含它自己的组**。而 `slime/ray/rollout.py::normalize_vals` 对单元素组恒等于零（`vals - vals.mean()` 长度 1 时恒为 0；带 std 归一化时直接 `zeros_like`），`dynamic_history` 与否两条分支同构。**开启归一化会把每个 reward 都变成 0，训练信号整个消失**；此外 `_drop_constant_reward_groups` 会把每个单元素组都判成"常数组"、整批丢弃只留一组
- **所以 `--disable-rewards-normalization` 不是可调选项、是这套架构的必需项**（`--n-samples-per-prompt 1` + 一样本一组，组内没有方差可归一）。**全负 batch 无法靠配置解决，只能改 reward 分布本身**
- **顺带修正自己前一条记录里的误读**：toolcall-rl 的 `min(-0.6, score + tool_call_reward)` 被我记成"负分下限钳在 -0.6"，**方向说反了**——`min` 取小值，实际是封顶，失败样本最好也只能到 -0.6。真实作用是"按工具调用次数把 -1 抬高到最多 -0.6，但绝不让它变正"，目的正是**让失败样本的幅度产生差异**
- **据此反转两条路线的优先级**（此前把 A 当主方案是错的）：路线 B 才对症（连续 reward 分布直接打散全负批）；路线 A 只降低负样本**数量**、每个样本仍是干脆 ±1，全负批概率从必然降到约 5%，是改善不是解决。**执行顺序改为先做 B**
→ 详见 `metaclaw_migration_plan.md`"查证结果（2026-08-31）：靠打开归一化来救全负 batch 这条路走不通"

**完成内容（续四，同一天）——实现路线 B（`METACLAW_MIDROUND_REWARD=blend`）：**
- **新增第三种模式 `blend`**，`judge`（默认）/`outcome` 两种行为完全不变。核心是新的模块级函数 `_metaclaw_blend_reward(outcome, judge_score, hard_negative)`：**outcome 决定符号、判官分只调幅度**，对齐 toolcall-rl 的 `base_score + prm_step_coef * prm_step_mean`——判官判错会被稀释，而不是像 outcome 模式那样成为整个 reward
- **两个可调系数**：`METACLAW_MIDROUND_JUDGE_COEF`（默认 0.3，保证 `|0.3×判官分| < 1`、outcome 永远决定符号）、`METACLAW_MIDROUND_FAIL_CEILING`（默认 -0.6，失败 round 的**封顶**不是下限）。**实现时发现并写进注释：默认 coef 下封顶是惰性的**（-1+0.3=-0.7 本来就低于 -0.6），它的存在意义是给 coef 调大时当符号保护——coef=1.0 时不封顶的话失败样本会到 0.0、符号丢失
- **硬负分优先级在 blend 下同样最高**：`hard_negative` 直接短路成 -1.0，不参与混合（否则把已知无效的工具调用往零的方向拉，等于抵消 override）
- **效果**：一个失败 round 的中间轮次不再是清一色 -1，而是按判官分散成 -1.3 / -1.0 / -0.7 三档——这正是全负 batch 需要的幅度差异，而归一化在这套架构下提供不了（每个样本自成一组）
- **验证**：补丁链对真实官方源码重跑通过、`py_compile` 通过；22 项断言覆盖 blend 算术（含符号保持的 6 种组合）、失败 round 产生三档不同幅度、成功 round 仍能区分干净/浪费轨迹、`judge`/`outcome` 两模式逐条未变、系数与封顶可配置且封顶在高 coef 下确实起到符号保护作用。**两处测试预期写错被实际结果纠正**（误以为默认 coef 下封顶会生效），已按真实行为改正
- 启动脚本三处接好：传参给训练后端、`RUN_MANIFEST.txt` 落盘、blend 模式下额外打印两个系数
→ 详见 `metaclaw_migration_plan.md`"方案：两条候选路线"路线 B

**产出（续二）：**
- `scripts/prepare_patched_openclaw_combine.sh`：`blend` 模式 + `_metaclaw_blend_reward` + 两个系数
- `scripts/metaclaw/run_metaclaw_migration_modelfactory.sh`：两个系数的默认值、传参、日志、manifest

**完成内容（续五，同一天）——查清 GRPO 的"组"机制，推翻了刚做完的路线 B 的定位：**
- **查清三方分组方式（全部读源码）**：slime 标准路径里"组"= 同一 prompt 的多次采样（`data_source.py:102-111`，同 prompt 复制 N 份、共享 `group_index`）。**toolcall-rl** 用 `--n-samples-per-prompt 8`——同一道题独立采 8 次，组内有对有错，减组均值后必有正负，全同的组被 `_drop_constant_reward_groups` 整组丢弃。**MetaClaw** 没有组概念，`data_formatter.py::compute_advantages` 直接对**整批**做 `(r - mean) / (std + eps)`——注释自称 "GRPO style" 但严格说是批级基线，**正因如此对"整批全负"免疫**
- **我们是三者里唯一会崩的配置**：`--n-samples-per-prompt 1` + 每样本自成一组 → **用着 GRPO 估计器却一个真正的组都没有**，advantage 退化成原始 reward。而 `n_samples_per_prompt=1` 不是随便设的——我们一次采样是真实 agent 跑一遍会改 workspace，采 8 次要开 8 份独立 workspace 且不知道后续轮次接哪份，是真实架构障碍
- **这推翻了上一条记录里"路线 B 才是对症的"这个判断**：blend 只提供幅度差异（-1.3/-1.0/-0.7），**在失败 round 里一个正样本都产生不了**，而崩溃的直接形态是 `0/16 正样本`。blend 是相对塑形、跟基线方案正交可叠加，但不应指望它单独解决问题
- **记下两条待查证路径**：① 学 toolcall-rl 提高 `n_samples_per_prompt`（架构障碍大，初步判断不现实但未正式评估）；② 学 MetaClaw 做批级基线（改动最小且有官方先例，已知坑是 `_drain_output_queue` 的 `completed_groups[group_id] = group` 会让共享 group_id 的样本互相覆盖，需先改成累加，且还没查有没有别处假设 `group_index` 唯一）。**两条都要先查清可行性再选，不先动手**
→ 详见 `metaclaw_migration_plan.md`"查证记录（七）：GRPO 的'组'到底是什么"

---

## 2026-09-01

**目标：** 查清 GRPO 分组机制、对照 toolcall-rl 与 MetaClaw 的任务形态，定出真正命中根因的方案。

**完成内容：**
- **日期更正**：昨天写的"查证记录（七）"实际提交于今天（`66e5b49`，09-01 10:03），文档里两处标题日期已从 08-31 改为 09-01
- **查清 toolcall-rl 的任务形态，确认路径 ①（提高 `n_samples_per_prompt`）应排除**：它是**数学题 + Python 代码解释器**（ReTool/DAPO-Math-17k），一次采样 = 跑一遍沙箱、**用完即弃、彼此完全独立**，所以能开 8 份并发做 GRPO 组内比较。我们一次采样 = 真实 agent 跑一遍、**读写真实 workspace**，同一天 round 共享 workspace、day01→day30 严格顺序，采 8 次要开 8 份 workspace 且跑完状态各异、后续轮次不知接哪份。**GRPO 要求"多次采样可独立可丢弃"，我们的任务结构上违反这个前提**——这也解释了官方脚本本来就设 1，不是我们改小的
- **反过来印证路径 ②**：MetaClaw 面对的正是同一种任务形态（连续、有持久 workspace、无法重复采样），它的解法就是放弃组内比较、改用批级基线。**这不是权宜之计，是同类任务下的合理设计**
- **查清路径 ② 的可行性，结论是可行且比预想干净得多**：slime 自带 `--custom-reward-post-process-path` 钩子（`rollout.py:339-341`），在 `_post_process_rewards` 最顶部短路。**只要挂一个自定义函数就能实现批级基线，完全不用碰 `group_index`、不用改 `_drain_output_queue`**——之前担心的"共享 group_id 会互相覆盖"根本不会遇到。三个前提逐条核实：钩子拿到的是拍平后的完整一批（`rollout.py:253-256`）；可以保留 `--disable-rewards-normalization`（钩子不看这个标志，而关着它正好避开"单元素组被判常数组、整批丢弃"的陷阱）；`get_reward_value` 与我们写的 `{"score": ...}` 加官方 `--reward-key score` 对得上
- **出方案待 CLI 在真实环境查证**（本地无法验证的 5 点：`load_function` 的路径格式与 modelfactory 上的放置位置、dummy 样本会不会污染批均值、真实到达钩子的样本数、std≈0 时全 0 advantage 是否可接受、与 `step_wise` estimator 是否冲突）
- **记录了这一轮走过的弯路**：同一个问题被诊断了五次，前四次（中段脱钩已撤回 / 累计计数缺陷 / outcome 消融 / blend）**都在改 reward 的值，而问题出在 reward 到 advantage 的那一步**。`blend` 可能是多余的，默认关闭、回退成本低
→ 详见 `metaclaw_migration_plan.md`"任务形态对照"+"方案（待 CLI 在真实环境查证）：批级基线"

**产出：**
- `docs/metaclaw_migration_plan.md`：新增任务形态对照、批级基线方案（含 5 个待查证点、实验设计、弯路记录）两节；两处标题日期更正
