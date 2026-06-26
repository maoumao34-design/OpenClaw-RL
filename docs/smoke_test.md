# Smoke Test（Step B）— 3 GPU 端到端冒烟

> **⚠️ 本文档与 `scripts/smoke_*.sh` 仅用于提交正式 8 GPU job 前的连通性验证。**  
> **不可用于 Table 3 论文复现；指标与 8 GPU 正式训练不可对比。**

## 脚本对照

| 用途 | 脚本 | GPU |
|------|------|-----|
| **Smoke 冒烟** | `scripts/smoke_train_with_services.sh` | **3**（缩配） |
| **正式 Table 3** | `scripts/train_with_services.sh` | **8**（论文布局） |

内部调用关系：

```text
smoke_train_with_services.sh
  └── smoke_run_qwen3_4b_openclaw_combine.sh   # 临时 patch 官方 combine
        └── OpenClaw-RL-official/.../run_qwen3_4b_openclaw_combine.sh
```

## Smoke 与正式训练的差异

| 项 | Smoke (3 GPU) | 正式 (8 GPU) |
|----|---------------|--------------|
| Actor | 1 (TP=1) | 4 (TP=4) |
| Rollout | 1 | 2 |
| PRM | 1 (inference OPD) | 1 + Megatron Teacher |
| OPD teacher | SGLang inference | Megatron（默认） |
| 模拟轮数 | 1 轮 × 1 题 | 72 session |
| 训练时长 | 验证通过后自动退出 | 完整训练 |

## 前置条件

1. 外部 **Qwen3-32B** vLLM 服务已就绪（Simulator）
2. `~/.openclaw/openclaw.json` 已配置：
   - `primary`: `sglang/qwen3-4b`
   - `baseUrl`: `http://127.0.0.1:30000/v1`
3. OpenClaw-RL-official、conda 环境、模型与 torch_dist 路径可用

## 1. 配置 Simulator（只改一个文件）

**文件路径：**

```text
/dfs/data/openclaw-rl-project/OpenClaw-RL/scripts/simulator.env
```

首次使用（只需一次）：

```bash
cd /dfs/data/openclaw-rl-project/OpenClaw-RL/scripts
cp simulator.env.example simulator.env
```

之后换 vLLM 服务、换地址或 key，**只编辑 `simulator.env` 这三行**：

```bash
SIMULATOR_BASE_URL=http://<32B-vllm-内网地址>/v1
SIMULATOR_API_KEY=EMPTY
EXTERNAL_MODEL=qwen3-32b
```

验证：

```bash
source simulator.env
curl -s "${SIMULATOR_BASE_URL%/v1}/health"
```

> `simulator.env` 不要 push 到 GitHub（已在 .gitignore）。模板见 `simulator.env.example`。

## 2. modelfactory 提交 job

| 字段 | 填什么 |
|------|--------|
| **代码解释器** | `bash`（不是 `python`） |
| **代码** | `.../OpenClaw-RL/scripts/smoke_train_with_services.sh` |
| **参数** | 留空 |
| **GPU** | **3× H20**（不要选 CPU / 0 GPU） |

**不需要**在 job 表单填环境变量；脚本会自动读 `scripts/simulator.env`。

## 通过标准

日志末尾出现：

```text
✅ SMOKE PASSED
```

且 `logs/smoke_*/` 下存在：

- `training.log` — Ray / SGLang / Megatron 无致命错误
- `openclaw.log` — gateway 正常
- `simulation.log` — student / TA / teacher 各 1 题跑完

## 失败后排查

```bash
tail -120 /dfs/data/openclaw-rl-project/logs/smoke_*/training.log
tail -40  /dfs/data/openclaw-rl-project/logs/smoke_*/openclaw.log
tail -80  /dfs/data/openclaw-rl-project/logs/smoke_*/simulation.log
```

常见原因：Simulator URL 不可达、torch_dist 路径错误、openclaw.json 未指向 30000、GPU 数不足 3。

## 通过后

使用 **`scripts/train_with_services.sh`** 提交 **8 GPU** 正式训练。
