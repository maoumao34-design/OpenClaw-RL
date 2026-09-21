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
| 任务数据 | **未发布**⚠️ | `benchmark/data/metaclaw-bench/`（30 天/346 题）|

## ❌ 千万别以为"用了官方数据集 = 用了对应 Part 的方法"

**本项目在这上面栽了很久**（2026-08 中旬～09-07，多轮结论反复）：

1. 公开数据集**物理上就嵌在 `benchmark/data/` 里**，用它只能走 `benchmark/src/` 这条 harness——**这不是一个选择，是目录结构决定的唯一默认路径**
2. 更关键：**`benchmark/data/metaclaw-bench/` 目录里同时打包了 agent 配置**——`openclaw_cfg/openclaw.json`（`profile: coding` 全套工具）和 `workspaces/shared/`（IDENTITY.md / SOUL.md / USER.md / AGENTS.md / TOOLS.md）。**`benchmark/src/` 本身是中立的，"Part II 特征"是随数据集发的**
3. 而这份数据集的规模（30 天 / 346 题 / 10–15 每天）与论文 **Part I** 定义 7/7 吻合

**→ 结果就是「用 Part II 的方法跑 Part I 的数据集」，分数与 Table 1 任何一列都对不上。**多日的"4B 零训练贴平 GPT-5.2"异常根源在此，不是模型问题、不是 harness bug、也不是评测口径。

## ❌ `examples/train.jsonl` 不是 Part I 的数据集

**本项目误认过很久。**它是**一整段真实对话**——`user_msg_001…072`，时间戳全在 `Sat 2026-02-21 07:25→09:25 EST` 同一天两小时内，内容如 `Retry.` / `Confirm.`，单条无意义。**没有标准答案、没有 checker、没有天/轮结构，算不出 Acc 或 Compl。**

它是 `examples/` 下产品 RL 通路的演示数据（`trainer.py` 仅在配置 `openclaw_env_data_dir` 时才读）。附录 A.3 引其第 1 条只是示范"Part I 的用户指令长什么样"（caption：Part I **User Instruction Template**）。

**Part I 真正没发布的只有一样：它自己的那 346 道题。**（2026-09-11 逐条复核后收窄——原先写的"三样都没发布"里有两样说重了。）

`metaclaw/openclaw_env_rollout.py` 作为 Part I 的 agent 是**完整的**：A.1 提示词（`:104`，那个 `records/system_prompt_cache.json` 覆盖文件实测不存在）、单 `run_command` schema、agent 循环、trainer 接线齐全。**唯一的硬缺口是"基于事实的评分"**——`reward: 0.0`（`:266`），评分交给 `prm_scorer.py` 的 **LLM 判官**（默认 `gpt-5.2`、m=3 多数票），那是训练 reward，**算不出 Acc/Compl**。天/轮结构与工作区隔离（`_exec_command` 无 `cwd=`）确实也没有，但那是补几十行代码的事。

**→ 「把 Part I 的 agent 跑在 benchmark 那 346 道题上」是可行的**（评分借 `benchmark/` 的 checker，驱动用我们现成的 driver），**文件夹不在一起不构成障碍**。**但跑出来的数不能叫「论文 Part I 基线」**——题不是 Part I 的题。另注 A.1 提示词写的是"controlling an OpenClaw installation"，与这批 workspace 文档产出题**领域错位**，照搬会给消融混入第二个变量。

## ❌ 论文附录的 Part 标签不可靠，不要当证据

已证实附录至少三处 Part 归属存疑（A.2 的身份文件随 Part I 数据发布；A.3 的示例标签与实际数据对不上）。**判定归属时只采信可数事实**（天数 / 题数 / 每天题数 / 轮次号 / 题型配比），不要采信文字标签。

有效的判定法（A.3 对称性测试）：**拿附录示例的内容去仓库里找**——Part I 示例在 `examples/train.jsonl` 第 1 条逐字存在；Part II 的两个示例（day01/r1 多选、day01/r21 `decision_log`）**在库里完全不存在**（本库 day01/r1 是 file_check、day01 只到 r10、全库无 `decision_log`、最大轮次号 15）。

## ❌ 不要拿 Table 1 当锚点

**两列都不 like-for-like**：Part I 列方法不同（单工具 vs 全套），Part II 列数据集不同（14 天/588 题/MC 74% vs 我们 30 天/346 题/MC 35%）。公开仓库里**没有任一 Part 的"方法 + 数据集"完整组合**。

**→ 训练效果一律以我们自己的 K=0 为基准**（2026-09-04 定，09-07 复核仍有效）。

## ❌ 已作废的基线，不要再引用

- **17.8% / 0%（定版基线）**：跑在带 OpenClaw session-key 兜底 bug 的构建上，文件写进 `workspace-main/` 而非 checker 读的目录，`Compl=0` 是构建产物不是能力上限。且其 "agentfix" 很可能只补了 `--agent` 四层链条中的 L1、因 L2/L3 断链而**静默无效**
- **K=6 的"正面训练效果"**：已被证伪（K=0 零训练即 34.9%/13.4%，与 K=6 的 37.3%/13.9% 基本重合，训练增益接近 0）

## ❌ checker 会给**零写入**的轮次白送 +1

`benchmark/src/infer/infer_cmd.py:608` 的 `_run_file_check` 在 workspace 里跑 checker，
**只数目录里已有的合规文件，完全不看最终回复**。按天共享 workspace ⇒ 计数跨轮累计。

**实证**（`20260914_181842` day09 r8）：最终回复只有 `idle timeout`、本轮零写入，
判分是 `--dir day09/ --ext json --min-count 4`，而**本轮写入之前目录里已有 4 个合规 json**
（全来自更早轮次）→ `official_score = 1.0`。

> **这不只是评测虚高，是把错的 +1 喂进优化器——被强化的正是退化行为。**
> 且污染**不对称**：超时轮越多的跑次白拿越多。
>
> `agent_succeeded=True` 只表示 CLI 进程没崩，**不等于这轮答完了**。

`--dir --min-count` 的**负方向**（早轮欠账 → 本轮做对也判 −1，31% 的 FC 轮次）此前已记，
**正方向 2026-09-17 才发现**。两个方向都错，从未打过补丁。

## ❌ 别拿 rollout 的 `passed=True` 计数当 Acc

`report.md` 的 Acc **含部分分**：基线 day02 是 `Correct 8.8 / 11` → **80.3%**，
而 rollout 里 `passed=True` 的离散数是 8/11 ≈ **72.7%**。**两者差 7.6pt，混用会得出相反结论。**

**闸门/对照一律对齐 report Acc。** K=0 定版基线 `20260907_112320` 的前三天是
**68.3% / 80.3% / 33.3%**——注意 **day03 本来就只有 33.3%**，用错基线会把"正常"判成"崩了"。

## ⚠️ 训练必崩，而零训练基线不崩

截至 2026-09-18：

| | 结果 |
|---|---|
| **零训练基线 K=0** | 跑完 30 天，超长轮次熔断计数 **0** |
| **纯 RL / 纯 OPD / 混合** | **三臂全部退化成复读，无一跑完** |

**退化完全是训练造成的，不是题集的性质。** 三臂的崩法还不一样：
纯 OPD 是**复读 thinking → idle timeout**；纯 RL 是**复读 write → transcript 撑爆 →
`Compaction timed out`（180s 预算）→ rc=1 无训练样本**。

**起病原因至今未知**——批次构成、优化器标量、OPD 强度三条独立线索全阴性。

## ⚠️ 09-03 以来的训练侧改动，没有一个被一趟跑完的训练验证过

**「跑过」不等于「已验证」。** 一趟在 step 10 OOM 的训练，其中的改动只是跑过。
逐条状态见 [`change_ledger.md`](change_ledger.md)——**引用任何训练侧改动的效果之前先查它。**

## ⚠️ `--agent` 是一条四处断链，缺一处静默失效

`_run_openclaw_agent` argv / `_run_question`→`_run_openclaw_agent` / `_run_group`→`_run_question` / `_run_group`→末轮 standalone feedback。**缺任何一处，`agent_id` 一路默认成 `None`，argv 里的 `--agent` 消失，行为与完全没修逐字节相同。**

判断时**按 caller+callee 配对，不要按行号**——`infer_cmd.py` 里有一个长得极像的 `_execute_update(agent_id=agent_id, ...)` 就在同一个函数里，本项目已因此误判过一次。校验器见 `scripts/metaclaw/run_official_baseline_modelfactory.sh`。

**详细说明见：[`metaclaw_migration_plan.md`](metaclaw_migration_plan.md)「🔴 当前结论」与查证记录（十）～（十七）**

## ❌ 别用 submit 日志去测「退化样本是否被奖励」

**这个方法本身不成立，不是结果阴性。**

超长轮次在**成为训练样本之前**就被熔断器（`METACLAW_MAX_TRAJECTORY_TOKENS=32768`）
整轮丢弃，**所以想测的那群样本结构性地不出现在 submit 日志里**。

2026-09-20 实测（臂 A `20260917_180600`，起病窗 day08–13，按 `response_len` 分桶）：

| 桶 | n | mean | frac_pos |
|---|---|---|---|
| short (<4k) | 64 | −0.375 | 0.312 |
| mid | 2 | 0.000 | 0.500 |
| **long (≥8k)** | **1** | −1.000 | 0.000 |

`long` 只有 **1 条**——不是"没有退化轮"，是它们被丢掉了。而且窗口本身被挤掉：
day08 那一声是瞬态、随后回落四天；day13+ 退化爆发时 `DROPPED×16` 同步开始。
**可测的窗口和有信号的窗口是错开的。**

> 要测必须换数据源：`record_*_archive.jsonl`（**含被丢弃的轮次**）。
> ⚠️ **但有一个未验证的前提**：被熔断器丢弃的轮次**是否仍被打了分**。
> 若丢弃发生在 scoring 之前，这些轮次根本没有分可查，**这个问题就是不改代码
> 不可测的**。动手之前先验证这一条。

## ⚠️ MetaClaw 路径 `K_i ≡ 1`——`--hint-m` / `--hint-selection` 是死参数

**这是迁移的设计后果，不是配置缺陷，别当 bug 去"修"。**

`prepare_patched_openclaw_combine_select.sh` 的 verdict 分支（2026-08-19 起）直接返回：

```python
"teacher_tokens_candidates": [_enhanced_ids],
"hints": [_metaclaw_hint],
"votes": [],
```

hint 来源已从**PRM 三票判官**换成 **checker 自己的 stdout / 错选项反馈**，
这种 hint 对一个轮次**天然只有一条**，从不进入官方那条
`seen_hints` 去重 → `MAX_CAND` 截断的链路。

实测四趟（A/B/H0/E01）accepted 共 **323 条，K_i 全部 = 1，无一 ≥2**。

**后果**（读日志和改配置时都要记得）：

| 参数 | 名义值 | 实际 |
|---|---|---|
| `--hint-m` | 3 | **死参数**，候选恒为 1 |
| `--hint-selection` | `sequence_optimal` | **从未执行**——`_select_k_star_per_token` 在 `if K == 1` 就短路返回全零 |
| `sel_k_star_mean` | — | **恒为精确 0.0，是 K==1 短路的签名，不是"总选中 0 号候选"** |

> 论文主方法是 k=4 **m=3** 多候选（见本文档「关键事实速查」）。
> **"我们跑的不是论文的多候选版本"是事实**；要不要恢复多候选是**方法选择**，
> 不是缺陷修复——checker 反馈本来就只有一条。

## ⚠️ advantage **没有 baseline**，是裸 ±1——而且这是官方设定

`run_qwen3_4b_openclaw_topk_select.sh:153,180,181`（**官方脚本自带，不是我们加的**）：

```
--n-samples-per-prompt 1
--advantage-estimator grpo
--disable-rewards-normalization
```

组大小 = 1 且关掉归一化 ⇒ **advantage = 原始奖励 ±1，没有减任何均值**。
项目早前独立验证过的恒等式 `grpo_pg_loss = (n_neg − n_pos)/8`
（每个样本恰好贡献 ∓1）本身就是证据。

> **推论一：不存在"组内对比"。** 任何写着"同一组里退化候选 vs 干净候选"的
> 分析都是空的——每组只有一个样本。本项目 2026-09-20 在这上面推演过一整轮。
>
> **推论二：「白拿 +1」不会被任何 baseline 抵消**，拿到 +1 的样本被无条件推高。
>
> **推论三：这三个 flag 是官方设定，擅自改动即偏离复现基线。**
