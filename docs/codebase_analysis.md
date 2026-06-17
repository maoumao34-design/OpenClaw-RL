# OpenClaw-RL 官方仓库代码分析

> 基于 https://github.com/Gen-Verse/OpenClaw-RL 的完整代码阅读
> 作为复现工作的指导文件，随理解深入持续更新

---

## 整体架构

官方仓库是**完整的工程实现**，核心是 `slime/` 框架 + 各场景插件模块。
启动方式极简：一条 shell 脚本拉起所有组件，通过 Ray 协调分布式执行。

```
slime/train_async.py          ← 主入口（Ray + Megatron + SGLang）
    ↑
openclaw-combine/run_*.sh     ← 场景启动脚本（设置环境变量 + ray job submit）
```

---

## 各模块详解

### 1. `slime/` — 核心 RL 框架

**职责：** 通用异步 RL 训练框架，所有场景都跑在它上面，不需要修改。

| 文件 | 作用 |
|------|------|
| `train_async.py` | 主入口：启动 Ray + Megatron actor + SGLang rollout |
| `slime/ray/` | Ray actor 管理，GPU 分配，进程间通信 |
| `slime/rollout/sglang_rollout.py` | SGLang 推理引擎的 rollout 封装 |
| `slime/utils/ppo_utils.py` | PPO/GRPO 基础数学工具（compute_policy_loss 等） |
| `slime_plugins/` | 模型适配插件（Qwen3、Qwen3.5、GLM4 等） |

**关键设计：** `sum_of_sample_mean` 聚合函数——每个 sample 内做 mean，sample 间做 sum，用于 loss 聚合。

---

### 2. `openclaw-rl/` — Personal Agent Binary RL（最轻量，复现起点）

**职责：** 只用 GRPO（evaluative signal），不用 OPD，是最简单的场景。

| 文件 | 作用 |
|------|------|
| `openclaw_api_server.py` | 核心代理：转发请求→SGLang，识别 session/turn_type，触发 PRM，提交训练样本 |
| `openclaw_rollout.py` | 连接 API server 和 slime trainer 的 rollout 桥接 |
| `run_qwen3_4b_openclaw_rl.sh` | 启动脚本 |

**`openclaw_api_server.py` 核心逻辑（重点理解）：**

```
请求进来
  ↓
识别 turn_type: "main" 或 "side"
  ↓ (main)
转发到 SGLang，拿到 response
  ↓
buffer 当前 turn 的数据（等待 next-state）
  ↓
下一个请求进来时，flush 上一个 turn（此时有了 next-state）
  ↓
触发 PRM 异步评分（m 次投票 → 多数决 → score ∈ {+1, -1, 0}）
  ↓
PRM 完成后提交 Sample 到 slime 的 output_queue
  ↓ (side)
只转发，不产生训练数据
```

**PRM Prompt 设计（已在代码中固定）：**
- `\boxed{1}`：next-state 显示任务推进（用户继续、工具成功返回）
- `\boxed{-1}`：next-state 显示失败（用户要求重做/纠正/重述）
- `\boxed{0}`：next-state 模糊，无法判断

**特殊逻辑：** 每个 session 至少保证一个 effective sample（score=0 时如果是首个样本会被 promote）。

---

### 3. `openclaw-combine/` — GRPO + OPD 混合（论文核心方法）

**职责：** 在 openclaw-rl 基础上加入 OPD（directive signal），是论文完整方法的实现。

| 文件 | 作用 |
|------|------|
| `openclaw_topk_select_loss.py` | **核心 loss**：GRPO + top-K OPD 混合，完整实现论文公式 |
| `openclaw_combine_select_api_server.py` | 在 PRM 打分基础上额外生成 M 个 hint 候选，计算 teacher top-K |
| `openclaw_combine_select_rollout.py` | 带 hint 的 rollout，传递 teacher tensor 到 loss |
| `hint_opd_loss.py` | OPD 单样本 loss 核心：`_opd_one_sample()` |
| `hint_opd_select_loss.py` | hint 选择辅助函数：`_overlap_count_per_token`, `_select_k_star_per_token` |
| `prm_teacher_postprocess.py` | teacher tensor 后处理工具 |
| `run_qwen3_4b_openclaw_topk_select.sh` | 完整启动脚本（论文标准配置） |

**Loss 计算流程：**

```python
# 1. GRPO branch（evaluative signal）
pg_loss_tokens = compute_policy_loss(ppo_kl, rl_advantages, eps_lo, eps_hi)
grpo_loss = sum_of_sample_mean(pg_loss_tokens)

# 2. OPD branch（directive signal）
# 2a. 对每个样本，计算 M 个 hint 候选的 top-K overlap
overlap_kr = _overlap_count_per_token(student_idx, teacher_native_idx_cand)
# 2b. 选出 k*（sequence_optimal / token_optimal / shortest）
k_star = _select_k_star_per_token(overlap_kr, hint_selection, ...)
# 2c. 用 k* 对应的 teacher 计算 OPD loss
pg_t, clip_t = _opd_one_sample(logits, student_idx, student_lp, teacher_idx_sel, teacher_lp_sel, ...)
opd_loss = sum_of_sample_mean(pg_t)

# 3. 合并
loss = w_rl * grpo_loss + w_opd * opd_loss  # 默认 w_rl=w_opd=1.0
```

**9-cell 支持矩阵（`--distill-subset-mode` × `--hint-selection`）：**

|  | `shortest` | `token_optimal` | `sequence_optimal` |
|--|-----------|-----------------|-------------------|
| `student` | ✓ | ✓ | ✓ |
| `overlap` | ✓ | ✓ | ✓ |
| `teacher` | ✓ | ✓ | ✓ |

**论文默认配置：** `subset_mode=student`，`hint_selection=sequence_optimal`，K=4，M=3。

---

### 4. `Megatron-LM/` — 分布式训练后端

**职责：** 负责实际的梯度更新，通过 `megatron-bridge` 与 slime 集成。

**关键点：**
- 模型权重需要先转换成 `torch_dist` 格式才能被 Megatron 加载
- 转换命令：`python tools/convert_hf_to_torch_dist.py`
- **基本不需要修改**，安装是最复杂的环节

---

### 5. 四类 General Agent 环境

| 模块 | 场景 | Next-state Signal | 复杂度 |
|------|------|------------------|-------|
| `toolcall-rl/` | API/函数调用 | 返回值/错误 | 低 |
| `terminal-rl/` | Shell 执行 | stdout/stderr/exit code | 中 |
| `swe-rl/` | 代码仓库+测试 | 测试结果/diff | 高 |
| `gui-rl/` | 屏幕+a11y tree | 视觉状态变化 | 最高 |

---

## 启动脚本关键参数解读

以 `run_qwen3_4b_openclaw_topk_select.sh` 为例：

```bash
NUM_GPUS=8          # 总 GPU 数
ACTOR_GPUS=4        # Megatron trainer 占用
ROLLOUT_GPUS=2      # SGLang policy server 占用
PRM_GPUS=1          # PRM SGLang server 占用
PRM_TEACHER_GPUS=1  # Megatron teacher 占用

HF_CKPT=...         # HuggingFace 格式模型路径（SGLang 用）
REF_LOAD=...        # torch_dist 格式路径（Megatron 用）
SAVE_CKPT=...       # checkpoint 保存路径

# 需要改成自己路径的变量（3个）
HF_CKPT
REF_LOAD
SAVE_CKPT
```

---

## 复现步骤规划

### 阶段一：环境配置

按官方 `instructions/README.md` 顺序安装，**顺序不能乱**：

```
1. conda create python=3.12
2. PyTorch 2.9.1 + cu129
3. pip install -r requirements.txt
4. DeepEP（源码编译）
5. int4_qat kernels
6. NVIDIA APEX（源码编译，最慢）
7. flash-attn 2.7.4（源码编译）
8. flashinfer 0.6.3（cu129 wheel）
9. megatron-bridge（特定 commit）
10. TransformerEngine 2.10.0
```

**风险：** workspace 是 CUDA 13，官方要求 CUDA 12.9。flashinfer 和 TransformerEngine 的 cu129 wheel 能否在 cu130 上运行需要实测。

### 阶段二：模型准备

```bash
# 下载 Qwen3-4B-Thinking-2507
# 转换为 torch_dist 格式
cd slime
python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /path/to/Qwen3-4B-Thinking-2507 \
  --rotary-base 5000000 \
  --save /path/to/Qwen3-4B-Thinking-2507_torch_dist
```

### 阶段三：修改启动脚本

最小改动：替换 3 个路径变量，调整 GPU 数量。

### 阶段四：先跑 openclaw-rl（Binary RL）

验证整个 pipeline 通路，不需要 OPD 相关组件，GPU 需求更低。

### 阶段五：升级到 openclaw-combine（完整方法）

Binary RL 跑通后，切换到 `run_qwen3_4b_openclaw_topk_select.sh`。

---

## 单 GPU 降配方案

官方脚本有强制检查：`ACTOR + ROLLOUT + PRM + PRM_TEACHER ≤ NUM_GPUS`。

1 张 A800 的可行方案：

**方案 A：跑 openclaw-rl（Binary RL，无 PRM_TEACHER）**
```bash
NUM_GPUS=1
ACTOR_GPUS=1    # Megatron 时分复用
ROLLOUT_GPUS=0  # 与 ACTOR 共享（需要代码级调整）
PRM_GPUS=0      # 禁用 PRM（无奖励信号，用于 pipeline 联调）
```

**方案 B（推荐起点）：先禁用 PRM，只验证数据流通路**

关掉 PRM 后 reward 全为 0，但可以验证 slime + SGLang + Megatron 的数据流是否跑通，再逐步开启。

---

## 待确认问题

- [ ] CUDA 13 与 cu129 wheel 的兼容性
- [ ] 单卡时 ACTOR/ROLLOUT 共享 GPU 的代码层面修改点
- [ ] Qwen3-4B-Thinking-2507 模型文件大小和下载方式（HuggingFace vs 国内镜像）
- [ ] workspace 系统盘 2GB 是否够装所有编译产物（建议 conda env 也放 /dfs/data/）
