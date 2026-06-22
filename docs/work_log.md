# 复现工作记录

记录每次工作的进展，用于汇报。

---

## 2026-06-17 论文理解 + 环境搭建启动

**目标：** 读懂论文，开始搭建复现环境

**完成内容：**

### 论文理解
- 通读 OpenClaw-RL 论文（arXiv:2603.10165）
- 整理论文核心机制文档：`docs/paper_understanding.md`
  - 四组件异步架构（Policy Server / Environment Server / PRM Server / Trainer）
  - GRPO 实为 PPO-style 非对称裁剪（eps_lo=0.2, eps_hi=0.28）
  - OPD（Off-Policy Distillation）使用全局 log-prob 校正
  - Binary RL：用二值奖励替代 PRM 分步打分

### 环境初始化
- 创建 conda 环境 `/dfs/data/envs/openclaw-rl`（Python 3.12）
- 安装 torch 2.9.1+cu129

### 依赖安装
- 安装 sglang（解决 outlines_core 0.1.26 PyPI sdist 损坏问题，下载 wheel 本地安装）
- 安装 slime（commit 02fef7e9）

### 遇到的主要问题
1. **outlines_core==0.1.26 PyPI sdist 元数据损坏**（version=0.0.0）→ 本地下载 wheel 安装，patch outlines METADATA 移除依赖声明
2. **DeepEP 编译失败**：CPU workspace 默认目标 sm_75，DeepEP 需要 sm_90 指令 → 决定申请 H20+CUDA12.9 GPU workspace 编译（H20 为 sm_90）
3. **pip 误用系统 Python 3.13**：conda activate 后 which pip 仍指向 miniconda3 base → 改用 `/dfs/data/envs/openclaw-rl/bin/pip` 全路径调用

---

## 2026-06-18 环境搭建（Phase 1）

**目标：** 在 modelfactory 上搭建 OpenClaw-RL 复现环境

**完成内容：**

### 环境初始化
- 创建 conda 环境：`/dfs/data/envs/openclaw-rl`（Python 3.12，持久化跨 workspace）
- 安装 torch 2.9.1+cu129（CUDA 12.9 编译版本）

### 依赖安装（CPU workspace，CUDA 12.9）
以下依赖已安装完成：

| 依赖 | 版本/提交 | 状态 |
|------|-----------|------|
| torch | 2.9.1+cu129 | ✅ |
| sglang | commit d566816d | ✅ |
| slime | commit 02fef7e9 | ✅ |
| Megatron-LM (megatron-core) | commit 3714d81d | ✅ |
| mbridge | commit 89eb1088 | ✅ |
| torch_memory_saver | commit dc689760 | ✅ |
| megatron-bridge | commit 35b4ebfc | ✅ |
| requirements.txt（主依赖列表） | — | ✅ |

### 待完成（需要 GPU workspace）
以下依赖需要在 H20 + CUDA 12.9 的 GPU workspace 上编译：

| 依赖 | 命令 | 备注 |
|------|------|------|
| int4_qat kernel | `pip install -e slime/slime/backends/megatron_utils/kernels/int4_qat --no-build-isolation` | — |
| apex | `APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .` | 需 keepalive 防 idle 关机 |
| flash-attn 2.7.4.post1 | `MAX_JOBS=8 pip install --no-build-isolation -v flash-attn==2.7.4.post1` | 编译时间长 |
| flashinfer-jit-cache 0.6.3 | `pip install "flashinfer-jit-cache==0.6.3" --index-url https://flashinfer.ai/whl/cu129` | — |
| TransformerEngine 2.10.0 | `NVTE_FRAMEWORK=pytorch pip install --no-build-isolation "transformer_engine[pytorch,core_cu12]==2.10.0"` | — |

**编译注意：** 在 H20（sm_90）上编译，若后续在 A100（sm_80）训练，需加 `TORCH_CUDA_ARCH_LIST="8.0"`

### 跳过的依赖
| 依赖 | 原因 |
|------|------|
| DeepEP | 需要 sm_90（H100/H20），仅用于 MoE 模型 Expert Parallelism，Qwen3-4B（密集模型）不需要 |

### 遇到的主要问题
1. **outlines_core PyPI sdist 损坏** → 下载 wheel 本地安装，patch outlines METADATA 移除依赖声明
2. **pip 使用系统 Python 3.13 而非 conda env** → 改用完整路径 `/dfs/data/envs/openclaw-rl/bin/pip`
3. **Megatron-LM/mbridge 等 git+ 依赖无法直连 GitHub** → 先本地克隆（ghfast 镜像）再 `pip install -e`
4. **系统级 PIP_CONSTRAINT 冲突（protobuf 4.24.4 vs 6.33.5）** → `PIP_CONSTRAINT=""` 临时绕过（conda env 与系统隔离，无风险）
5. **megatron-bridge 子模块拉取失败** → 手动配置子模块 URL 为 ghfast 镜像后初始化

---

---

## 2026-06-22 GPU 编译 + 模型下载

**目标：** 在 H20 + CUDA 12.9 workspace 上完成 GPU 编译依赖安装，下载模型权重

**完成内容：**

### 根目录清理
- 删除临时文件：slime.zip、req_no_git.txt、src/、tmp_wheels/ 内容、Untitled*.ipynb
- req_install.log 移至 /dfs/data/logs/

### GPU Workspace 申请
- 申请 H20 × 1 + CUDA 12.9 小型编译专用 workspace（16核 64GB）
- 原因：H20-8-PREMIUM 排队，先用小 workspace 做编译，编译结果保存在 conda env（持久化）

### GPU 编译依赖安装

| 依赖 | 状态 | 备注 |
|------|------|------|
| flashinfer-jit-cache 0.6.3 | ✅ | 本地下载 wheel 上传安装（modelfactory 限速，本地 10MB/s 更快） |
| int4_qat kernel (fake_int4_quant_cuda) | ✅ | — |
| apex 0.1 | ✅ | `APEX_CUDA_EXT=1 APEX_CPP_EXT=1 MAX_JOBS=16` |
| flash-attn 2.7.4.post1 | ✅ | 从源码编译（torch2.9 无预编译 wheel）；需先拉 cutlass + composable_kernel 子模块 |
| TransformerEngine 2.10.0 | ✅ | 下载预编译 cu12 wheel（287MB） |

---

### 模型下载

| 模型 | 路径 | 状态 |
|------|------|------|
| Qwen3-4B-Thinking-2507 | `/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507` | ✅ 7.6GB，加载验证通过 |

---

### 模型格式转换

| 步骤 | 命令 | 状态 |
|------|------|------|
| HF → torch_dist | `/dfs/data/envs/openclaw-rl/bin/torchrun --nproc-per-node 1 slime/tools/convert_hf_to_torch_dist.py --megatron-to-hf-mode bridge --hf-checkpoint /dfs/data/models/Qwen/Qwen3-4B-Thinking-2507 --save /dfs/data/models/torch_dist/qwen3-4b-thinking-2507 --num-layers 36 --hidden-size 2560 --ffn-hidden-size 9728 --num-attention-heads 32 --num-query-groups 8 --rotary-base 5000000 --vocab-size 151936 --kv-channels 128 --qk-layernorm --swiglu` | ✅ |

转换后路径：`/dfs/data/models/torch_dist/qwen3-4b-thinking-2507`

### 运行脚本路径配置

修改 `openclaw-rl/run_qwen3_4b_openclaw_rl.sh` 中四个路径占位符：

```bash
HF_CKPT=${HF_CKPT:-/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-/dfs/data/models/ckpt/qwen3-4b-openclaw-rl}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-/dfs/data/models/Qwen/Qwen3-4B-Thinking-2507}
```

以及 PYTHONPATH：`/dfs/data/openclaw-rl-project/OpenClaw-RL-official/Megatron-LM/`

（官方仓库无 push 权限，通过 `sed -i` 直接在 modelfactory 上应用。）

---

## 2026-06-22 训练数据流分析

**目标：** 理解 Personal Agent 训练数据如何产生、外部依赖是什么

**完成内容：**

### 训练数据流全貌

OpenClaw-RL Personal Agent Track 有两套独立的交互方式：

**方式 A：直接训练（`gsm8k_personal_agent.py`）**  
→ 连接 **port 30000**（slime 训练服务器直接暴露的推理 API）  
→ 请求携带 `session_id`、`turn_type`（main/side）、`session_done` 字段  
→ `turn_type="main"` 的 turn 进入训练 buffer；`turn_type="side"` 仅用于评估不训练  
→ 外部 LLM（GPT-4.1）扮演 student/teacher 模拟器  
→ **这是复现 Table 3 的完整脚本**

**方式 B：文件系统工作流（`openclaw-test/` 三个脚本）**  
→ 连接 **port 18789**（`openclaw_api_server.py` gateway）  
→ OpenClaw 作为文件操作 Agent，读写 workspace 目录下的 homework 文件  
→ 三阶段：student 写作业 → TA 批改 → teacher 评论

### 三阶段工作流

| 脚本 | 角色 | 输入 | 输出（workspace 文件）|
|------|------|------|------|
| `student_chat.py` | 懒学生（外部 LLM）| GSM8K.json | `homework/i.txt`（解答）|
| `TA_chat.py` | 助教（外部 LLM）| 上一步 homework | `homework1/i.txt`（批改）|
| `teacher_chat.py` | 老师（外部 LLM）| 上一步 homework1 | `homework2/i.txt`（评论）|

### 关键发现

1. **GSM8K.json 已存在** ✅：路径 `/dfs/data/openclaw-rl-project/OpenClaw-RL-official/openclaw-test/GSM8K.json`

2. **外部 LLM 要求**：
   - 模拟器（simulator）：GPT-4.1（默认），通过 `OPENAI_API_KEY` + `OPENAI_BASE_URL` 配置
   - 评估器（evaluator）：GPT-4o（默认）
   - 注意：`gsm8k_personal_agent.py` 使用 `client.responses.create()`（新版 Responses API），需要支持该接口的端点

3. **`openclaw_api_server.py` 双重角色**：
   - 作为独立 FastAPI 服务暴露在 port 18789（`openclaw-test/` 脚本用）
   - 作为 slime 的回调模块：`--custom-generate-function-path openclaw_api_server.generate`

4. **Port 30000**：slime 训练服务器（SGLang + openclaw hooks），`gsm8k_personal_agent.py` 直连此端口发送 `session_id`/`turn_type` 字段

5. **`rollout_batch_size=16`**：每收集 16 个 session turn 触发一次梯度更新

---

## 下一步计划

### Simulator / Evaluator 开源替代选型

论文原版 simulator 用 GPT-4.1，evaluator 用 GPT-4o，均无外部 API 访问权限。

**选定替代方案：Qwen3.5-122B-A10B（自托管）**

选择依据：
- IFBench 76.5（全模型最高），roleplay / instruction following 场景最优
- AMD 官方有 OpenClaw + Qwen3.5 + SGLang 验证案例
- 同 Qwen 系列，与 Policy（Qwen3-4B）tokenizer 完全兼容
- MoE 架构仅激活 10B 参数，2×H20 即可部署
- DeepSeek V4 对比：V4-Flash 需 175GB VRAM，V4-Pro 500GB，均不现实；且优势在 coding/math，非本任务所需

代码只需修改 `openai_api.py` 的一行（`responses.create` → `chat.completions.create`），其余通过环境变量切换：
```bash
export OPENAI_BASE_URL="http://<simulator-host>:8001/v1"
export OPENAI_API_KEY="dummy"
export OPENAI_MODEL="Qwen3.5-122B-A10B"
```

---

## 下一步计划

**Phase 1 启动前的准备：**

1. **申请 8×H20 + CUDA 12.9 workspace**（训练：actor×4 + rollout×2 + PRM×2）
2. **申请独立小 workspace（2×H20）**跑 Qwen3.5-122B-A10B simulator 服务
3. **修改 `openai_api.py`**：`client.responses.create()` → `client.chat.completions.create()`
4. **创建 checkpoint 保存目录**：`mkdir -p /dfs/data/models/ckpt/`
5. **运行训练**：`bash run_qwen3_4b_openclaw_rl.sh`
6. **运行客户端（复现 Table 3）**：
   ```bash
   export OPENAI_BASE_URL="http://<simulator-host>:8001/v1"
   export OPENAI_API_KEY="dummy"
   export OPENAI_MODEL="Qwen3.5-122B-A10B"
   cd /dfs/data/openclaw-rl-project/OpenClaw-RL-official/openclaw-rl/oel/eval
   python gsm8k_personal_agent.py --method combined --training-rounds 16
   ```
