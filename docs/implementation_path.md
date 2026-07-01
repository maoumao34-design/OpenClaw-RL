[← 工作记录](work_log.md)

# OpenClaw-RL 完整实现路径

> 基于对论文（arXiv:2603.10165）和代码库的完整分析（2026-06-22）。  
> 本文件是"怎么做"的权威参考。"需要什么"见 `paper_reproduction_scope.md`。  
> **每一步都标明对应的具体文件。未列入的文件不得使用。**

---

## 总览：四块实验的复现顺序

```
① Personal Agent（Table 3）← 入门首选，本文档重点
② Tool-call Agent（Figure 5）← 无外部环境，相对简单
③ Terminal Agent（Figure 5）← 需要远程 sandbox 环境
④ SWE Agent（Figure 5）←需要 Docker + 4 节点
⑤ GUI Agent（Figure 5）← 最复杂，需要桌面 VM 环境
⑥ Hybrid RL Extension（Figure 6）← 与 ①② 共享基础设施
⑦ Ablation（Tables 4-5，Figure 7）← 最后做，需要 ①② 的结果
```

---

## 共用准备（所有实验均需要）

### P1. 环境（已完成）
- **文件**：`instructions/README.md`
- **conda 环境**：`/dfs/data/envs/openclaw-rl`（Python 3.12）
- **状态**：✅ 所有 GPU 编译依赖已安装

### P2. 基础模型格式转换（已完成）
- **文件**：`slime/tools/convert_hf_to_torch_dist.py`
- **输入**：HF checkpoint
- **输出**：torch_dist checkpoint（Megatron 格式）
- **状态**：✅ Qwen3-4B-Thinking 已转换至 `/dfs/data/models/torch_dist/qwen3-4b-thinking-2507`

### P3. 模型清单（按实验块）

| 模型 | 大小 | 用于哪个实验 | 路径 / 状态 |
|------|------|------------|------------|
| Qwen3-4B-Thinking-2507 | 7.6 GB | ①③⑥⑦ Policy + PRM | ✅ `/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507` |
| Qwen3-4B-Thinking-2507 torch_dist | — | ①⑥⑦ Megatron actor | ✅ `/dfs/data/models/torch_dist/qwen3-4b-thinking-2507` |
| **Qwen3-32B**（论文原版 simulator）| ~16 GB Int4 | ① Simulator | ❌ 未下载 |
| Qwen3.5-122B-A10B-GPTQ-Int4 | ~65 GB | ① Simulator（替代方案）| 本地下载中 |
| Qwen3-8B | ~5 GB | ③ Terminal policy / ⑥ ReTool PRM | ❌ 未下载 |
| Qwen3VL-8B-Thinking | ~8 GB | ⑤ GUI policy + PRM | ❌ 未下载 |
| Qwen3-4B-Instruct-2507 | ~8 GB | ④ SWE policy | ❌ 未下载 |
| Qwen3-4B-SFT（Retool 上 SFT 版）| ~8 GB | ② Tool-call policy | ❌ 未下载（需从 slime/Retool 项目获取）|
| Retool-4B | ~8 GB | ⑥ ReTool RL policy | ❌ 未下载 |
| DeepSeek-R1-Distill-Qwen-1.5B | ~1.5 GB | ⑥ RLVR policy | ❌ 未下载 |

---

## ① Personal Agent（Table 3）— 完整路径

**目标数字**（数字越小越好）：

| 方法 | Student | TA | Teacher |
|------|---------|-----|---------|
| **Hybrid RL (Ours)** | 11.6 | 8.2 | 14.8 |
| GRPO | 15.4 | 12.0 | 24.4 |
| OPD | 30.8 | 34.0 | — |

### 架构（三端口，缺一不可）

```
port 30001 ─── Simulator（Qwen3-32B / 替代，SGLang）
                    ↕ 扮演 student/TA/teacher
port 18789 ─── OpenClaw gateway（workspace 文件工具 + rl-training-headers）
                    ↕ X-Session-Id / X-Turn-Type 训练头
port 30000 ─── RL 训练代理（openclaw-combine 启动脚本负责启动）
```

### GPU 配置阶梯（三步走）

| 阶段 | GPU 数 | Actor TP | Rollout | PRM | OPD 路径 | 目的 |
|------|--------|---------|---------|-----|---------|------|
| **Smoke**（已写脚本）| 3 | TP=1 | 1 GPU | 1 GPU | inference（无 Megatron teacher）| 管道联调，不产生论文质量训练信号 |
| **小规模验证**（下一步）| 6 | TP=2 | 2 GPU | 1+1 GPU | **megatron（论文原版路径）**| 第一次真实 Hybrid RL 训练，验证 OPD 闭环 |
| **完整论文配置** | 8 | TP=4 | 2 GPU | 1+1 GPU | megatron | Table 3 正式复现 |

**Smoke 的本质限制**：`OPD_SRC=inference` 意味着没有独立的 Megatron PRM Teacher，OPD 教师 log-prob 精度不够，只能验证三端口管道是否通畅，不能产生论文质量的训练信号。

**为什么 6 GPU 比直接跳 8 GPU 更合理**：第一次跑 megatron mode，Actor TP=2 已足够验证 OPD Megatron 路径，失败时日志更容易定位；8 GPU 资源需求更高，调试成本也更高。

#### 6 GPU 脚本改动（相对于 `run_qwen3_4b_openclaw_combine.sh`）

需要新建 `scripts/train_6gpu_with_services.sh`，在调用官方脚本前覆盖三个变量：

```bash
export NUM_GPUS=6
export ACTOR_GPUS=2
# 在 PERF_ARGS 里将 --tensor-model-parallel-size 从 4 改为 2
# 其余保持不变：ROLLOUT_GPUS=2, PRM_GPUS=1, PRM_TEACHER_GPUS=1, OPD_SRC=megatron（默认）
```

与 smoke 的区别：
- `OPD_SRC` 不设置（走默认 `megatron`）→ PRM Teacher 独立 Megatron 实例启动
- `ACTOR_GPUS=2` 而非 1 → TP=2，序列并行开启
- `ROLLOUT_GPUS=2` 保持和论文一致（SGLang TP=2）

---

### 步骤

**Step A：申请 workspace**
- 训练机：6×H20（小规模验证）或 8×H20（完整论文配置）+ CUDA 12.9
  - 6 GPU 分配：Actor×2 + Rollout×2 + PRM×1 + PRM Teacher×1
  - 8 GPU 分配：Actor×4 + Rollout×2 + PRM×1 + PRM Teacher×1
- Simulator 机：1×H20 96 GB（托管 Simulator LLM，外部独立）

**Step B：安装 OpenClaw + 配置** ⚠️ *当前阻塞项，需优先确认能否安装*
- 文件：`extensions/rl-training-headers/`（整个目录）
- 动作：
  1. 安装 OpenClaw 应用（https://github.com/openclaw/openclaw）
  2. 启用 rl-training-headers 扩展
  3. 修改 `~/.openclaw/openclaw.json`，设置 `providers.baseUrl = "http://0.0.0.0:30000/v1"`，`apiKey` = `SGLANG_API_KEY` 的值
- 验证：`curl http://localhost:18789/healthz` 返回 200

**Step C：上传 Simulator 模型并启动**
- 文件：无（标准 SGLang 启动，无自定义代码）
- 命令：
  ```bash
  python -m sglang.launch_server \
    --model-path /dfs/data/models/Qwen/<simulator-model> \
    --tp 1 \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 --port 30001
  ```
- 验证：`curl http://localhost:30001/health` 返回 200

**Step D：启动 Hybrid RL 训练服务器**
- 文件：`slime/train_async.py`（被脚本调用，无需直接修改）
- 启动脚本（按阶段选择）：
  - **Smoke（3 GPU）**：`scripts/smoke_train_with_services.sh`（已写，管道联调用）
  - **小规模验证（6 GPU）**：`scripts/train_6gpu_with_services.sh`（待写，第一次真实训练）
  - **完整论文配置（8 GPU）**：`scripts/train_with_services.sh`（已写，Table 3 正式复现）
  - 论文原版脚本：`openclaw-combine/run_qwen3_4b_openclaw_combine.sh`（官方，被上述脚本调用）
- 依赖文件（自动加载，需在 PYTHONPATH 里）：
  - `openclaw-combine/openclaw_combine_api_server.py`（proxy server，port 30000）
  - `openclaw-combine/openclaw_combine_rollout.py`（rollout 循环）
  - `openclaw-combine/combine_loss.py`（Hybrid RL loss）
  - `openclaw-opd/openclaw_opd_api_server.py`（父类，hint judge + PRM eval）
- 关键环境变量：
  ```bash
  export HF_CKPT=/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507
  export REF_LOAD=/dfs/data/models/torch_dist/qwen3-4b-thinking-2507
  export SAVE_CKPT=/dfs/data/models/ckpt/qwen3-4b-openclaw-combine
  export PRM_MODEL_PATH=${HF_CKPT}
  export PRM_TEACHER_LOAD=${REF_LOAD}
  export SGLANG_API_KEY="<自定义 key>"
  ```
- 验证：等待日志出现 `[OpenClaw-OPD] model is ready`，port 30000 可访问

**Step E：运行训练会话（三角色，三阶段顺序执行）**
- 文件（训练 + 评估共用这三个）：
  - `openclaw-test/student_chat.py`
  - `openclaw-test/TA_chat.py`
  - `openclaw-test/teacher_chat.py`
- 数据集：`openclaw-test/GSM8K.json` ✅（已在仓库）
- 环境变量：
  ```bash
  export OPENCLAW_GATEWAY_TOKEN="<openclaw token>"
  export OPENAI_API_KEY="dummy"
  export OPENCLAW_GATEWAY_URL="http://localhost:18789"
  export OPENCLAW_WORKSPACE="$HOME/.openclaw/workspace"
  export OPENAI_BASE_URL="http://<simulator-ip>:30001/v1"
  export EXTERNAL_MODEL="<simulator-model-name>"
  ```
- 命令（顺序执行，student → TA → teacher）：
  ```bash
  python openclaw-test/student_chat.py  --dataset openclaw-test/GSM8K.json --num-problems 36 --max-turns 8
  python openclaw-test/TA_chat.py       --dataset openclaw-test/GSM8K.json --num-problems 36 --max-turns 8
  python openclaw-test/teacher_chat.py  --dataset openclaw-test/GSM8K.json --num-problems 36 --max-turns 8
  ```
- 训练会循环进行：收集 16 个 sample → 梯度更新 → 继续收集

**Step F：Table 3 评估（rule-based session 计数）**
- 文件：同 Step E 三个脚本 + 人工统计（无额外评估脚本）
- 方法：从脚本输出的 `results_student.txt` / `results_TA.txt` / `results_teacher.txt` 取**第一条回复**，按以下规则统计收敛 session 数：
  ```python
  # Student：无 bold / 编号列表 / \boxed{}
  satisfies = not re.search(r'\*\*|^\d+\.|\\boxed\{', reply, re.M)
  # TA：> 100 词
  satisfies = len(reply.split()) > 100
  # Teacher：包含暖词
  satisfies = any(w in reply.lower() for w in ['well done', 'excellent', 'great job'])
  # 收敛：连续 3 个 session 均满足
  ```
- 输出：与 Table 3 比较（Hybrid RL Ours：Student 11.6 / TA 8.2 / Teacher 14.8）

⚠️ **注意**：目前没有现成的 rule-based 评估脚本，Step F 的统计逻辑需要自己写（约 30 行 Python）

---

## ② Tool-call Agent（Figure 5）— 完整路径

**目标**：Figure 5 右图，Tool-call track 的训练曲线（横轴 steps，纵轴 rollout accuracy）

### 步骤

**Step A：申请 workspace**
- 8×H20 + CUDA 12.9（ACTOR×2 + ROLLOUT×4 + PRM×2）

**Step B：准备数据集**
- 训练数据：DAPO RL dataset（`PROMPT_DATA` 路径，格式为 jsonl）
- 评估数据：AIME 2024 dataset（`EVAL_DATA` 路径）
- ⚠️ 两个数据集来源需确认（论文引用了 Zhu et al. 2025 / slime）

**Step C：准备 Qwen3-4B-SFT 模型**
- 需要：Qwen3-4B 在 Retool dataset 上做过 SFT 的版本
- 文件：`toolcall-rl/retool_qwen3_4b_sft.sh`（SFT 脚本，需先跑 SFT 再做 RL）
- ⚠️ 或从 slime/ReTool 项目直接获取预训练的 SFT checkpoint

**Step D：转换模型格式**
- 文件：`slime/tools/convert_hf_to_torch_dist.py`
- 输入：Qwen3-4B-SFT HF checkpoint
- 输出：torch_dist checkpoint

**Step E：启动 Tool-call RL 训练**
- 启动脚本：`toolcall-rl/retool_qwen3_4b_prm_rl.sh`
- 依赖文件（自动加载）：
  - `toolcall-rl/` 下的自定义 rollout 文件（需查看脚本内 `--rollout-function-path`）
- 关键环境变量（需覆盖脚本默认值）：
  ```bash
  export HF_CKPT=/dfs/data/models/Qwen/Qwen3-4B-SFT
  export REF_LOAD=/dfs/data/models/torch_dist/qwen3-4b-sft
  export SAVE_CKPT=/dfs/data/models/ckpt/qwen3-4b-retool-prm-rl
  export PROMPT_DATA=/dfs/data/datasets/dapo-math-17k.jsonl
  export EVAL_DATA=/dfs/data/datasets/aime-2024.jsonl
  export PRM_MODEL_PATH=/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507
  ```
- 评估：训练中自动评估（AIME accuracy），每 N 步记录一次，输出到 wandb / 日志

---

## ③ Terminal Agent（Figure 5）— 完整路径

**目标**：Figure 5 左图，Terminal track 的训练曲线

### 步骤

**Step A：申请 workspace**
- 训练机：8×H20（ACTOR×4 + ROLLOUT×4）
- 环境服务器：若干台机器运行 terminal sandbox pool（128 个并行环境）

**Step B：搭建 Terminal 环境 pool**
- 文件：`terminal-rl/remote/run_pool_server.sh`（启动 sandbox pool）
- 文件：`terminal-rl/remote/setup.sh`（环境机器初始化）
- 数据：SETA RL dataset（需从 SETA 项目获取）
- 环境变量：`ENV_SERVER_URL`、`WORKER_URLS`（pool 机器 IP 列表）

**Step C：准备 Qwen3-8B 模型**
- 下载 Qwen3-8B HF checkpoint
- 转换为 torch_dist 格式（同 P2）

**Step D：启动 Terminal RL 训练**
- 启动脚本：`terminal-rl/terminal-rl_qwen3-8b.sh`
- 依赖文件：`terminal-rl/configs/rollout_qwen3.yaml`（rollout 配置）
- 关键环境变量：
  ```bash
  export HF_CKPT=/dfs/data/models/Qwen/Qwen3-8B
  export REF_LOAD=/dfs/data/models/torch_dist/qwen3-8b
  export ROLLOUT_PROMPT_DATA=/dfs/data/datasets/seta-rl-data.jsonl
  export ENV_SERVER_URL=http://<pool-server-ip>:18080
  ```
- 评估：rollout accuracy，训练中自动记录

---

## ④ SWE Agent（Figure 5）— 完整路径

**目标**：Figure 5，SWE track 训练曲线

### 步骤

**Step A：申请 workspace**
- 4 节点 × 8 H20 = 32 GPU（colocate 模式，actor + rollout 共用）
- 额外：Docker 环境机器（64 个并行 SWE 环境）

**Step B：搭建 SWE-Bench Docker 环境**
- 文件：`swe-rl/data/pull_swebench_verified_images.sh`（拉取 Docker 镜像）
- 文件：`swe-rl/server/setup_ecs_seed.sh`（ECS 环境机器初始化）
- 数据：SWE-Bench-Verified 数据集

**Step C：准备 Qwen3-4B-Instruct-2507 模型**
- 下载 + 转换 torch_dist

**Step D：启动 SWE RL 训练**
- 启动脚本：`swe-rl/scripts/run_swe_rl_4b_4nodes_colocate.sh`
- 评估脚本：`swe-rl/scripts/eval_4b_4nodes.sh`

---

## ⑤ GUI Agent（Figure 5）— 完整路径

**目标**：Figure 5，GUI track 训练曲线

### 步骤

**Step A：申请 workspace**
- 8×H20（ACTOR×4 + ROLLOUT×3 + PRM×1）
- 额外：64 台 GUI 虚拟机（OSWorld 环境）

**Step B：搭建 OSWorld 环境**
- 参考：`gui-rl/desktop_env/`（整个子目录，包含各云服务商指南）
- 文件：`gui-rl/desktop_env/providers/` 下选择对应云平台（AWS / Aliyun / Volcengine / Docker）

**Step C：准备 Qwen3VL-8B-Thinking 模型（多模态）**
- 下载 Qwen3VL-8B-Thinking HF checkpoint（含视觉编码器）
- 转换 torch_dist

**Step D：启动 GUI RL 训练**
- 启动脚本：`gui-rl/gui_qwen3vl_8b_prm_rl.sh`
- 关键环境变量：`GUI_ENV_SERVER_HOST`、`GUI_ENV_SERVER_PORT`

---

## ⑥ Hybrid RL Extension（Figure 6）— 完整路径

两个子实验，与 Personal Agent（①）和 Tool-call（②）共享基础设施。

### 子实验 A：ReTool 多轮 RL

- **目标**：Figure 6 左图，Retool accuracy 训练曲线
- **Policy**：Retool-4B（Qwen3-4B 在 Retool 数据上 SFT 版）
- **PRM**：Qwen3-8B
- **启动脚本**：`toolcall-rl/retool_qwen3_4b_prm_rl.sh`（覆盖 `PRM_MODEL_PATH=Qwen3-8B`）
- 与 ② Tool-call 的区别：PRM 换成 Qwen3-8B（更大）

### 子实验 B：RLVR（AIME）

- **目标**：Figure 6 右图，AIME score 训练曲线
- **Policy**：DeepSeek-R1-Distill-Qwen-1.5B
- **PRM**：Qwen3-4B
- **启动脚本**：`toolcall-rl/retool_qwen3_4b_prm_rl.sh`（覆盖 `HF_CKPT=DeepSeek-R1-Distill-Qwen-1.5B`）
  - ⚠️ 需要确认是否有独立脚本或直接复用 tool-call 脚本
- 评估：AIME 2024，20 次独立运行取平均

---

## ⑦ Ablation（Tables 4-5，Figure 7）— 完整路径

**前提**：① Personal Agent 基础设施已跑通

### Table 4：hint selection 策略对比

- 变量：`sequence-optimal` / `token-optimal` / `random`
- **启动脚本**：`openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh`
- 控制变量：`OPENCLAW_TOPK_HINT_SELECTION=sequence_optimal|token_optimal|shortest`
- Policy：Qwen3-32B（ablation 用更大 policy）→ 需要 Qwen3-32B 的训练脚本
  - ⚠️ 目前只有 4B 脚本，32B 可能需要多节点；需确认是否有现成脚本

### Table 4：model size 对比

- Qwen3-4B vs Qwen3-32B policy
- 分别运行对应脚本，比较 session 数

### Table 5：k 值和 support set 对比

- 变量：`OPENCLAW_TOPK_K=2|4|8|20`
- 变量：`OPENCLAW_TOPK_SUBSET_MODE=student|overlap`
- 启动脚本同 Table 4

### Figure 7：log-prob clip 影响

- 变量：`OPENCLAW_TOPK_ADV_DIFF_CLIP` 有/无
- 使用 ReTool + RLVR 两个数据集

---

## 文件-步骤 对应表（完整索引）

| 文件 | 属于哪步 | 作用 |
|------|---------|------|
| `instructions/README.md` | P1 | 环境安装指南 |
| `slime/tools/convert_hf_to_torch_dist.py` | P2 | HF→Megatron 格式转换 |
| `extensions/rl-training-headers/` | ① Step B | OpenClaw 训练头插件 |
| **`openclaw-combine/run_qwen3_4b_openclaw_combine.sh`** | ① Step D | **Hybrid RL 主训练脚本（论文原版）** |
| `scripts/smoke_train_with_services.sh` | ① Step D | Smoke（3 GPU）管道联调 |
| `scripts/train_6gpu_with_services.sh` | ① Step D | 小规模验证（6 GPU，待写）|
| `scripts/train_with_services.sh` | ① Step D | 完整论文配置（8 GPU）|
| `openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh` | ① Step D / ⑦ | 改进版 Hybrid RL / Ablation |
| `openclaw-combine/openclaw_combine_api_server.py` | ① Step D | RL proxy server（port 30000）|
| `openclaw-combine/openclaw_combine_rollout.py` | ① Step D | Rollout 循环控制 |
| `openclaw-combine/combine_loss.py` | ① Step D | Hybrid RL loss（GRPO + OPD）|
| `openclaw-opd/openclaw_opd_api_server.py` | ① Step D | Hint judge + PRM eval（父类）|
| `openclaw-test/student_chat.py` | ① Step E/F | Student 训练 + 评估 |
| `openclaw-test/TA_chat.py` | ① Step E/F | TA 训练 + 评估 |
| `openclaw-test/teacher_chat.py` | ① Step E/F | Teacher 训练 + 评估 |
| `openclaw-test/GSM8K.json` | ① Step E/F | 数学题数据集 |
| `openclaw-test/launch_user_llm.sh` | ① Step C | Simulator SGLang 启动脚本 |
| `toolcall-rl/retool_qwen3_4b_prm_rl.sh` | ② Step E | Tool-call RL 训练 |
| `toolcall-rl/retool_qwen3_4b_sft.sh` | ② Step C | Qwen3-4B SFT（Tool-call 前置）|
| `terminal-rl/terminal-rl_qwen3-8b.sh` | ③ Step D | Terminal RL 训练 |
| `terminal-rl/remote/run_pool_server.sh` | ③ Step B | Terminal 环境 pool |
| `swe-rl/scripts/run_swe_rl_4b_4nodes_colocate.sh` | ④ Step D | SWE RL 训练 |
| `swe-rl/data/pull_swebench_verified_images.sh` | ④ Step B | SWE Docker 镜像 |
| `gui-rl/gui_qwen3vl_8b_prm_rl.sh` | ⑤ Step D | GUI RL 训练 |
| `openclaw-rl/run_qwen3_4b_openclaw_rl.sh` | **⛔ 禁用** | GRPO-only 基线，不是主方法 |
| `openclaw-rl/oel/eval/gsm8k_personal_agent.py` | **⛔ 禁用** | OEL 评估脚本，与 Table 3 无关 |

---

## 当前状态与阻塞项

### 已就绪
- [x] 环境 + GPU 编译依赖
- [x] Qwen3-4B-Thinking HF + torch_dist
- [x] GSM8K.json 数据集

### 最近完成（本次分析）
- [x] 完整理解 Hybrid RL 三端口架构
- [x] 确认正确训练脚本（openclaw-combine vs openclaw-rl）
- [x] 确认正确客户端脚本（openclaw-test 而非 oel/eval）
- [x] 创建本文档

### 当前阻塞项（按优先级）

1. **OpenClaw 能否安装到 modelfactory？**（最高优先级，阻塞整个 ① Personal Agent）
   - 需要安装 OpenClaw 应用 + rl-training-headers 扩展
   - OpenClaw 是论文四组件架构中的 Environment Server，不可替代

2. **Simulator 模型选择**（影响 ① 的论文一致性）
   - 论文：Qwen3-32B
   - 当前：Qwen3.5-122B-A10B-GPTQ-Int4（下载中）

3. **rule-based 评估脚本**（影响 ① 的指标可信度）
   - 需要自己写约 30 行统计脚本，才能得到 Table 3 的 session 计数

4. **Qwen3-4B-SFT / DAPO dataset / AIME dataset**（阻塞 ②）
   - 需要找到这些资源的来源

5. **Terminal sandbox pool / SWE Docker / GUI VM**（阻塞 ③④⑤）
   - 各需要独立的环境搭建
