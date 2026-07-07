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
