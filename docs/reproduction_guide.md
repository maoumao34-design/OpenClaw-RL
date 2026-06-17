# 复现操作指南

> 给 Cursor CLI 的执行指南，每一步都要严格按顺序，遇到问题记录到 `docs/issues_log.md`

---

## 环境信息

| 项目 | 值 |
|------|-----|
| GPU | 8× A100-PCIE-40GB |
| CUDA 驱动 | 13.0 |
| Python | 3.12.3 |
| conda 路径 | `/dfs/data/miniconda3` |
| conda env | `/dfs/data/envs/openclaw-rl` |
| 项目根目录 | `/dfs/data/openclaw-rl-project/` |
| 官方仓库 | `/dfs/data/openclaw-rl-project/OpenClaw-RL-official/` |
| 我们的仓库 | `/dfs/data/openclaw-rl-project/openclaw-rl/` |

---

## 第一阶段目标

**跑通 `openclaw-rl`（Binary RL）**，验证整个数据流：
```
SGLang（推理）→ API Server（PRM 评分）→ Sample Buffer → Megatron Trainer（梯度更新）
```
不需要跑完整评估，看到 loss 下降 + `submitted sample` 日志就算跑通。

---

## 安装步骤（严格按顺序）

### Step 0：激活环境（每次新开终端都要运行）

```bash
export PATH="/dfs/data/miniconda3/bin:$PATH"
conda activate /dfs/data/envs/openclaw-rl
```

### Step 1：PyTorch（正在进行）

```bash
pip install torch==2.9.1+cu129 torchvision==0.24.1+cu129 torchaudio==2.9.1+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
```

**验证：**
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 期望：2.9.1+cu129  True
```

### Step 2：requirements.txt

```bash
cd /dfs/data/openclaw-rl-project/OpenClaw-RL-official
pip install -r requirements.txt
```

### Step 3：DeepEP

```bash
cd /dfs/data
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
pip install -e . --no-build-isolation
```

### Step 4：int4_qat kernels

```bash
pip install -e /dfs/data/openclaw-rl-project/OpenClaw-RL-official/slime/slime/backends/megatron_utils/kernels/int4_qat --no-build-isolation
```

### Step 5：APEX（最慢，1~2小时，用 tmux）

```bash
tmux new -s apex
cd /dfs/data
git clone https://github.com/NVIDIA/apex.git
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
# Ctrl+B D 退出 tmux，让它后台跑
```

**验证：**
```bash
python -c "import apex; print('apex ok')"
```

### Step 6：flash-attn（30~60分钟，用 tmux）

```bash
tmux new -s flash
export MAX_JOBS=8
pip install --no-build-isolation -v flash-attn==2.7.4.post1
```

**验证：**
```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

### Step 7：flashinfer（⚠️ 风险点，CUDA 13 兼容性未知）

```bash
pip install "flashinfer-jit-cache==0.6.3" --index-url https://flashinfer.ai/whl/cu129
```

**验证：**
```bash
python -c "import flashinfer; print('flashinfer ok')"
```

> 如果报错，把完整错误发给本地 Claude Code 分析，可能需要从源码编译或找 cu130 版本。

### Step 8：megatron-bridge

```bash
pip install "megatron-bridge @ git+https://github.com/fzyzcjy/Megatron-Bridge.git@35b4ebfc486fb15dcc0273ceea804c3606be948a" --no-build-isolation
```

### Step 9：TransformerEngine（⚠️ 风险点）

```bash
export NVTE_FRAMEWORK=pytorch
pip install --no-build-isolation "transformer_engine[pytorch,core_cu12]==2.10.0"
```

**验证：**
```bash
python -c "import transformer_engine; print('TE ok')"
```

---

## 模型准备

### 下载 Qwen3-4B-Thinking-2507

```bash
# 方式一：HuggingFace
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-4B-Thinking-2507 --local-dir /dfs/data/models/Qwen3-4B-Thinking-2507

# 方式二：modelscope（国内更快）
pip install modelscope
modelscope download --model Qwen/Qwen3-4B-Thinking-2507 --local_dir /dfs/data/models/Qwen3-4B-Thinking-2507
```

### 转换为 torch_dist 格式（Megatron 需要）

```bash
cd /dfs/data/openclaw-rl-project/OpenClaw-RL-official/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/dfs/data/openclaw-rl-project/OpenClaw-RL-official/Megatron-LM \
  python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /dfs/data/models/Qwen3-4B-Thinking-2507 \
    --rotary-base 5000000 \
    --save /dfs/data/models/Qwen3-4B-Thinking-2507_torch_dist
```

---

## 启动训练

### 修改路径（必须）

编辑 `openclaw-rl/run_qwen3_4b_openclaw_rl.sh`，替换以下三个变量：

```bash
HF_CKPT=/dfs/data/models/Qwen3-4B-Thinking-2507
REF_LOAD=/dfs/data/models/Qwen3-4B-Thinking-2507_torch_dist
SAVE_CKPT=/dfs/data/openclaw-rl-project/checkpoints/qwen3-4b-openclaw-rl
```

### 启动

```bash
cd /dfs/data/openclaw-rl-project/OpenClaw-RL-official/slime
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh
```

---

## 关键验证节点

| 检查点 | 验证命令 | 期望结果 |
|--------|---------|---------|
| PyTorch + CUDA | `python -c "import torch; print(torch.cuda.is_available())"` | `True` |
| Ray 集群 | `ray status` | 看到 8 GPU |
| SGLang 启动 | `curl http://localhost:30000/health` | `{"status": "ok"}` |
| PRM server | `curl http://localhost:30001/health` | `{"status": "ok"}` |
| 第一个 sample | 日志里出现 `submitted sample session=...` | 有 score 输出 |
| 第一次梯度更新 | 日志里出现 loss 数值 | loss 不为 NaN |

---

## 注意事项

1. **编译步骤必须在 GPU 在线时运行**（APEX、flash-attn、flashinfer、TE）
2. **tmux 防断连**：长时间编译任务一律用 `tmux new -s <name>` 跑
3. **每步装完立即验证**，不要等到最后才发现前面某步装错了
4. **遇到报错**：把完整错误信息记录到 `docs/issues_log.md`，发给本地 Claude Code 分析
5. **GPU 空闲超时**：平台会回收空闲 GPU，编译时保持 GPU 使用（或调大超时阈值）

---

## 问题记录

遇到问题请记录到 `docs/issues_log.md`，格式：

```
## [日期] 问题描述
**步骤：** Step X
**报错：** 
\```
错误信息粘贴在这里
\```
**解决方案：** 
```
