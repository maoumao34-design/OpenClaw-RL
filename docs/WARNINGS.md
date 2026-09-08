[← 工作记录](work_log.md)

# 复现警示文档

**所有参与复现的 agent 在开始工作前必须阅读本文件。**

---

## ❌ 禁止使用的目录和文件

以下内容是论文提交后由外部贡献者或内部扩展加入的，**与本论文（arXiv:2603.10165）无关**，复现过程中禁止读取、引用或修改：

| 路径 | 原因 |
|------|------|
| `openclaw-rl/oel/`（整个目录）| OEL 模块，外部贡献者 PR #96，2026-04-20 加入，是完全独立的研究，不属于本论文 |
| `openclaw-fireworks/`（整个目录）| Fireworks 云训练集成，论文提交后加入，与本论文无关 |
| `openclaw-tinker/`（整个目录）| 论文提交后加入，与本论文无关 |

**特别警告：以下两个文件极易误用：**

- `openclaw-rl/oel/eval/gsm8k_personal_agent.py` — OEL 的实验脚本，**不是** Table 3 的复现脚本
- `openclaw-rl/oel/eval/personalization_evaluator.py` — OEL 专用 LLM 打分器，**不是** Table 3 的评估方式

---

## ✅ 论文对应的正确目录

| 目录 | 对应论文内容 |
|------|------------|
| `openclaw-combine/` | **论文主方法：Hybrid RL（GRPO + OPD）= Table 3 "Hybrid RL (Ours)"** |
| `openclaw-rl/` | Binary RL 基线（Table 3 "GRPO" 列） |
| `openclaw-opd/` | OPD 基线（Table 3 "OPD" 列） |
| `openclaw-test/` | Personal Agent 评估套件（`student_chat.py` / `TA_chat.py` / `teacher_chat.py`）|
| `gui-rl/` | GUI agent track（Figure 5）|
| `swe-rl/` | SWE agent track（Figure 5）|
| `terminal-rl/` | Terminal agent track（Figure 5）|
| `toolcall-rl/` | Tool-call agent track（Figure 5）|

---

## 关键事实速查

**⚠️ 2026-07-07 更正：下面这条曾经写反过（曾导致 2026-06-29 用错脚本），已核对
`paper_index.md` 的 Table 3/5 数据修正——Table 3 "Hybrid RL (Ours)" 平均 10.3，
与 Table 5 k=4 这一列完全一致，`run_qwen3_4b_openclaw_topk_select.sh` 才是复现
Table 3 的正确脚本，`run_qwen3_4b_openclaw_combine.sh`（不带 k 的 basic combine）
在 Table 3/4/5/7 里都没有对应的消融列，已作为死代码删除
（`scripts/run_openclaw_combine_modelfactory.sh` / `smoke_run_qwen3_4b_openclaw_combine.sh`
一并删除，见 git log `c40619b`）。**

**论文主方法的训练脚本：**
```
openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh
# k=4, m=3, seq-optimal hint selection；Table 3 "Hybrid RL (Ours)" 平均 10.3
# 即 Table 5（p.14）k=4 这一列，复现首选
```

**完整三端口架构（缺一不可）：**
```
port 30001 → Simulator LLM（Qwen3-32B，SGLang）
    ↕ 扮演 student/TA/teacher
port 18789 → OpenClaw 应用 gateway（workspace 文件工具，真正的 openclaw gateway run）
    ↕ X-Turn-Type 靠 models.providers.sglang.headers 静态配置；
      X-Session-Id 靠解析 system prompt 里的 Runtime 行
      （rl-training-headers 插件在当前 OpenClaw 版本里端到端不生效，
      2026-07-07 实测证实，详见 work_log.md/issues_log.md 同日条目）
port 30000 → RL 训练代理服务器（运行 run_qwen3_4b_openclaw_topk_select.sh 后自动启动）
```

**客户端编排脚本（直连 port 18789，非 port 30000）：**
```
openclaw-test/student_chat.py  →  OPENCLAW_GATEWAY_URL=http://localhost:18789
openclaw-test/TA_chat.py
openclaw-test/teacher_chat.py
```

**Table 3 评估方式：rule-based session 计数，不是 LLM 打分**
- 指标 = 达到优化效果所需的最少 session 数
- Student 满足条件：回复无 `**bold**`、无编号列表、无 `\boxed{}`
- TA 满足条件：回复 > 100 词
- Teacher 满足条件：包含 "well done" / "excellent" 等暖词
- 收敛 = 连续 3 个 session 第一条回复满足上述规则

**论文 simulator 模型：Qwen3-32B**（Section 4.1 原文，非 GPT-4.1）

**详细说明见：`docs/paper_reproduction_scope.md`**

---

# ⚠️ MetaClaw 迁移阶段的警示（arXiv:2603.17187，另一篇论文）

> 以上全部是 **OpenClaw-RL（arXiv:2603.10165）** 的内容。下面这一节属于 **MetaClaw 迁移**阶段，两者是不同的论文和代码库，**不要混用**。

## ❌ 最大的坑：Part I 和 Part II 是两套完全独立的东西

MetaClaw-Bench 分 Part I / Part II，**它们不只是两份题集，而是两套独立的 agent 栈、两个任务域**。仓库里两套代码并存，极易误认。

| | **Part I** | **Part II** |
|---|---|---|
| 实现 | `metaclaw/openclaw_env_rollout.py` | `benchmark/` |
| system prompt | **附录 A.1 逐字自带**（"The single tool exposed to the agent is `run_command`"）| **一个都没有**，用 OpenClaw 原生 |
| 工具集 | **单个 `run_command`** | `profile: coding` 全套 read/write/edit/exec |
| 任务域 | **操作/配置一个 OpenClaw 安装**（CLI 运维）| **在工作区里产出文件**（文档产出）|
| 工作区 | **无**（`_exec_command` 是裸 shell，不传 `cwd`）| 每个 test 一个隔离副本 |
| 评分 | **无**（`reward=0.0`，交产品的 PRM）| checker，`cwd=workspace_path` |
| 任务数据 | `examples/train.jsonl`（72 条）| `benchmark/data/metaclaw-bench/`（30 天/346 题）|

## ❌ 千万别以为"用了官方数据集 = 用了对应 Part 的方法"

**本项目在这上面栽了很久**（2026-08 中旬～09-07，多轮结论反复）：

1. 公开数据集**物理上就嵌在 `benchmark/data/` 里**，用它只能走 `benchmark/src/` 这条 harness——**这不是一个选择，是目录结构决定的唯一默认路径**
2. 更关键：**`benchmark/data/metaclaw-bench/` 目录里同时打包了 agent 配置**——`openclaw_cfg/openclaw.json`（`profile: coding` 全套工具）和 `workspaces/shared/`（IDENTITY.md / SOUL.md / USER.md / AGENTS.md / TOOLS.md）。**`benchmark/src/` 本身是中立的，"Part II 特征"是随数据集发的**
3. 而这份数据集的规模（30 天 / 346 题 / 10–15 每天）与论文 **Part I** 定义 7/7 吻合

**→ 结果就是「用 Part II 的方法跑 Part I 的数据集」，分数与 Table 1 任何一列都对不上。**多日的"4B 零训练贴平 GPT-5.2"异常根源在此，不是模型问题、不是 harness bug、也不是评测口径。

## ❌ 论文附录的 Part 标签不可靠，不要当证据

已证实附录至少三处 Part 归属存疑（A.2 的身份文件随 Part I 数据发布；A.3 的示例标签与实际数据对不上）。**判定归属时只采信可数事实**（天数 / 题数 / 每天题数 / 轮次号 / 题型配比），不要采信文字标签。

有效的判定法（A.3 对称性测试）：**拿附录示例的内容去仓库里找**——Part I 示例在 `examples/train.jsonl` 第 1 条逐字存在；Part II 的两个示例（day01/r1 多选、day01/r21 `decision_log`）**在库里完全不存在**（本库 day01/r1 是 file_check、day01 只到 r10、全库无 `decision_log`、最大轮次号 15）。

## ❌ 不要拿 Table 1 当锚点

**两列都不 like-for-like**：Part I 列方法不同（单工具 vs 全套），Part II 列数据集不同（14 天/588 题/MC 74% vs 我们 30 天/346 题/MC 35%）。公开仓库里**没有任一 Part 的"方法 + 数据集"完整组合**。

**→ 训练效果一律以我们自己的 K=0 为基准**（2026-09-04 定，09-07 复核仍有效）。

## ❌ 已作废的基线，不要再引用

- **17.8% / 0%（定版基线）**：跑在带 OpenClaw session-key 兜底 bug 的构建上，文件写进 `workspace-main/` 而非 checker 读的目录，`Compl=0` 是构建产物不是能力上限。且其 "agentfix" 很可能只补了 `--agent` 四层链条中的 L1、因 L2/L3 断链而**静默无效**
- **K=6 的"正面训练效果"**：已被证伪（K=0 零训练即 34.9%/13.4%，与 K=6 的 37.3%/13.9% 基本重合，训练增益接近 0）

## ⚠️ `--agent` 是一条四处断链，缺一处静默失效

`_run_openclaw_agent` argv / `_run_question`→`_run_openclaw_agent` / `_run_group`→`_run_question` / `_run_group`→末轮 standalone feedback。**缺任何一处，`agent_id` 一路默认成 `None`，argv 里的 `--agent` 消失，行为与完全没修逐字节相同。**

判断时**按 caller+callee 配对，不要按行号**——`infer_cmd.py` 里有一个长得极像的 `_execute_update(agent_id=agent_id, ...)` 就在同一个函数里，本项目已因此误判过一次。校验器见 `scripts/metaclaw/run_official_baseline_modelfactory.sh`。

**详细说明见：[`metaclaw_migration_plan.md`](metaclaw_migration_plan.md)「🔴 当前结论」与查证记录（十）～（十六）**
