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

## 下一步计划

1. 转换模型格式（HF → torch_dist for Megatron）
2. 修改启动脚本路径配置
3. 运行 Phase 1：openclaw-rl Binary RL 验证 pipeline
