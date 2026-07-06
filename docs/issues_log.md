# 复现问题记录

> 遇到问题按格式记录，发给本地 Claude Code 分析

---

## [2026-06-24] Simulator port 30001 600s 超时，job 失败

**步骤：** train_with_services.sh Step 2/4（等待 Simulator）

**现象：**
```
14:21:01 === [2/4] 启动 Qwen3-32B Simulator（GPU 7，port 30001）===
14:21:01 等待 Simulator (port 30001)...
14:23:51   已等待 170s...
（约 10min 后 job 退出，ray stop --force 清理日志）
```

**根因：** GPU 7 被训练侧 Megatron PRM Teacher（~8GB）与 Simulator Qwen3-32B BF16（~64GB）共用，Simulator 无法完成加载，port 30001 一直未就绪。

**解决方案（已改脚本）：**
1. 训练改为 GPU 0-6（NUM_TRAINING_GPUS=7），OPD teacher 改 `inference` 模式（PRM_TEACHER_GPUS=0）
2. GPU 7 独占 Simulator
3. `openclaw start` → `openclaw gateway run --allow-unconfigured --force`
4. REF_LOAD / PRM_TEACHER_LOAD 改为 torch_dist 路径
5. Simulator 超时 900s，失败时自动 tail simulator.log

> 注：上述 inference 方案已弃用。最终方案见下一条（外部 Simulator + 8 GPU 论文布局）。

---

## [2026-06-26] 外部 Simulator + 8 GPU 论文布局（基于 0f4582c5）

**基线 commit：** `0f4582c5`（9 GPU 设计：GPU 0-7 训练 + GPU 8 Simulator，最接近论文）

**改动：**
- 训练 job 仅 8 GPU，全部用于 megatron 布局（4+2+1+1），不再本地起 Simulator
- Simulator 在独立机器运行 `scripts/launch_simulator.sh`
- 训练 job 通过 `SIMULATOR_BASE_URL` + `SIMULATOR_API_KEY` 调用外部服务
- `REF_LOAD` 使用 torch_dist；OpenClaw 使用 `gateway run`

**训练 job 必填环境变量：**
```
SIMULATOR_BASE_URL=http://<simulator-host>:30001/v1
SIMULATOR_API_KEY=<与 SGLang api-key 一致，无 auth 填 EMPTY>
```

---

## [2026-06-26] 3 GPU smoke 连环失败（modelfactory）

**步骤：** `scripts/smoke_train_with_services.sh`

### 1) patched combine `REPO_ROOT` → `logs/`

**现象：** `training.log` 在 `ray stop` 后结束，8265 未就绪  
**根因：** patched 脚本在 `logs/smoke_*/` 下，`REPO_ROOT=SCRIPT_DIR/..` 指向 `logs/`，找不到 `slime/`  
**修复：** commit `7f657e1`，`run_openclaw_combine_modelfactory.sh` 固定 `REPO_ROOT`

### 2) OpenClaw gateway 18789 超时（300s）

**现象：** `openclaw.log` 仅 `loading configuration` + `force: no listeners`  
**根因：** 旧脚本与训练并行起 OpenClaw、无 headless 参数、日志块缓冲；workspace 手动测同配置 ~1s ready  
**修复：** commit `96c40e5`，先等 `:30000`、headless gateway、`/healthz`、900s

### 3) Ray job：`/workspace/train_async.py` 不存在

**现象：**
```
python3: can't open file '/workspace/train_async.py'
Job 'raysubmit_...' failed
```
**根因：** modelfactory Ray 默认 cwd 为 `/workspace`；入口在 `slime/train_async.py`  
**修复：** commit `2687e58`，`--working-dir=${SLIME_ROOT}` + 绝对路径 `train_async.py`

### 4) inference 模式 PRM TP=2 vs 3 GPU

**现象：** log 中 `PRM_GPUS=1` 但 `PRM_NUM_GPUS_PER_ENGINE=2`；Ray job 失败  
**根因：** OPD inference 默认 PRM 2 卡 TP=2，smoke 仅 3 GPU（1+1+1）  
**修复：** commit `01f3eb0`，export + sed 强制 `PRM_NUM_GPUS_PER_ENGINE=1`  
**状态：** 待下周重新提交 smoke job 验证

---

## [2026-07-01] 4 GPU Smoke 续（REF_LOAD / RAM / 鉴权）

**步骤：** `scripts/smoke_train_with_services.sh`（接 [2026-06-26] 3 GPU smoke 连环失败，问题 4 后新提交）

### 4) REF_LOAD / PRM_TEACHER_LOAD 使用 HF 路径

**现象：** `AssertionError: Only bridge mode is supported for loading HF checkpoint`（`slime/backends/megatron_utils/checkpoint.py:134`）  
**根因：** commit `3b3e7e0` 误将 `REF_LOAD` 设为 HF 路径；Megatron 检测到 HF checkpoint 时要求 `--megatron-to-hf-mode bridge`，该参数在 topk-select 脚本中不存在  
**修复：** `REF_LOAD` / `PRM_TEACHER_LOAD` 改用 `POLICY_TORCH_DIST`（`/dfs/data/models/Qwen3-4B-Thinking-2507-torch-dist`）；同时修正路径拼写错误；commit `672d9a7`（smoke + 正式脚本均更新）

### 5) Worker 节点 RAM 不足（64 GB OOM）

**现象：** `ray.exceptions.OutOfMemoryError`，节点内存 `62.x GB / 64.00 GB (0.97)`  
**根因：** smoke TP=1 每个 Megatron actor 加载完整 checkpoint（~24 GB）；Actor + PRM Teacher 同时加载 ~48 GB，加系统进程超 64 GB 节点 95% OOM 阈值；正式 8 GPU TP=4 每 actor 仅 ~6 GB，无此问题  
**修复：** 申请 ≥128 GB RAM 节点提交 job

### 6) 评估阶段 401 Unauthorized

**现象：** `student_chat.py` 调 `http://localhost:30000/v1/chat/completions` 报 401  
**根因：** `run_smoke_chat()` 传的是 `OPENCLAW_GATEWAY_TOKEN`（OpenClaw CLI token），RL proxy（port 30000）使用 `SGLANG_API_KEY` 鉴权，两者不同  
**修复：** `run_smoke_chat()` 中 token 改为 `SGLANG_API_KEY`；commit `5aa3c74`

---

## [2026-07-03] Pre-test 0 训练步骤 + 无 checkpoint（commit `482fdc6` 架构绕过）

**步骤：** 5 GPU pre-test `scripts/minitest_train_with_services.sh`

**现象：**
- `training.log` 持续 `waiting for combine samples: 0/16, queue=0`，无训练步骤
- Policy 回复出现 "I don't have access to your local file system"
- `save-interval 5` 但无任何 checkpoint 产生
- 所有 session 日志显示 `[side] session=unknown`

**根因：** commit `482fdc6`（2026-06-30）将 `OPENCLAW_GATEWAY_URL` 从 18789 改为 30000

| 影响 | 机制 |
|------|------|
| 0 训练数据 | 绕过 OpenClaw → rl-training-headers 不注入 `X-Turn-Type:main` → RL proxy 默认 "side" → 训练队列永远为空 |
| Policy 无文件访问 | 绕过 OpenClaw gateway → Policy 缺少 workspace 工具 |

`482fdc6` 的原始诊断（"18789 不暴露 `/v1/chat/completions`"）是错的。真实 404 根因是当时 OpenClaw 在 30000 就绪前启动，sglang provider 初始化失败；应只改启动顺序而非 URL。

**架构核查（2026-07-03）：**
- `rl-training-headers`：`enabled`（`stock:rl-training-headers/index.js` v1.0.0）
- `sglang-provider`：`enabled`，`baseUrl=http://127.0.0.1:30000/v1`，`apiKey=openclaw-rl-key`
- 默认 model：`sglang/qwen3-4b`（= 官方 `SERVED_MODEL_NAME`）
- gateway token 读取路径 `gateway.auth.token`：脚本读取逻辑正确

**修复：** commit `83810e4`，三脚本改回 `OPENCLAW_GATEWAY_URL=http://localhost:18789`；minitest token 由 `SGLANG_API_KEY` 改回 `OPENCLAW_GATEWAY_TOKEN`

**遗留：** `train_with_services.sh` 启动顺序仍错（先起 OpenClaw 再等 30000）→ 待 smoke 验证通过后修复

---

## [2026-07-03] Simulator context overflow（16384 → 32768）

**步骤：** 5 GPU pre-test 模拟阶段（`sim_student.log`）

**现象：** 反复 400 错误 `maximum context length is 16384 tokens. prompt contains at least 16385 input tokens`

**根因：** `scripts/launch_simulator.sh` 默认 `MAX_TOKENS=16384`；Policy 单次回复最大 8192 token，TA/Teacher 的详细反馈在 2-3 轮后累积超限

**影响：** 收敛数据完整（output 写 turn==0，overflow 在 turn>=1）；会话完整性受损（部分作业文件未写完，TA/Teacher 可能读到不完整内容）

**修复：** `launch_simulator.sh` 默认值 16384 → 32768（核查官方 `openclaw-test/launch_user_llm.sh` 确认）；Simulator 需重启生效

---

## [2026-07-06] smoke PRM judge 400/503（PRM_MAX_NEW_TOKENS 与缩配 context 冲突）

**步骤：** 4 GPU smoke `scripts/smoke_train_with_services.sh`

**现象：** `training.log` 里每轮 PRM judge/eval 调用均失败：
```
[OpenClaw-OPD] PRM eval query failed (vote 0): Client error '400 Bad Request' for url 'http://<ip>:3794/generate'
（此后全部变为）Server error '503 Service Unavailable' for url '...'
[OpenClaw-Combine-Select] session=... turn=N no valid hint (votes=[None]), sample dropped
```
`combine samples` 队列从 0 GPU job 开始到结束全程停在 `0/4`。

**根因：** `run_openclaw_topk_select_modelfactory.sh` 的 `SMOKE_PROFILE` sed 补丁把 `--sglang-context-length`/`CONTEXT_LENGTH` 从官方 32768 缩到 8192（省显存），但没有同步缩小 `--prm-max-new-tokens`（仍为官方默认 8192）。PRM judge 引擎与 rollout 共用同一个 `sglang_context_length`，`prompt_len + 8192 > 8192` 对任何非空 prompt 恒成立，sglang 直接 400；PRM 引擎疑似因此卡死，此后全部 503。8 GPU 正式配置（context=32768）不受影响。

**修复：** `SMOKE_PROFILE` 补丁增加一行 `--prm-max-new-tokens 8192 → 4096`；commit `be0bc0e`

---

## [2026-07-06] smoke `update_weights()` OOM（TP=1 缩配显存不足）

**步骤：** 4 GPU smoke，PRM 修复后首次真实跑通数据流

**现象：**
```
ray.exceptions.RayTaskError(OutOfMemoryError): ray::MegatronTrainRayActor.update_weights()
```
崩溃前日志已出现 `submitted OPD+RL sample session=... index=7 reward=-1.0`，说明 `async_train()`（真正的前向/反向/optimizer step）已执行完毕，OOM 发生在训练步之后的权重同步（推给 sglang 推理引擎）阶段。训练进程死后，18789 → 30000 转发失败（`httpx.ConnectError`），表现为 student/teacher_chat.py 端的 500。

**根因：** smoke 把 Actor 从官方 TP=4 强行缩到 TP=1，整个 4B 模型 + 梯度 + 权重同步缓冲全部挤在一张卡，无张量并行分摊显存。此前从未真正走到 `update_weights()`（训练队列一直是 0），OOM 是链路修复后第一次暴露。

**评估：** minitest（TP=2）/ 8GPU 正式（TP=4）显存分摊更充分，理论上不会复现；若 minitest 也 OOM 则需进一步排查 `update_weight_from_distributed.py` 是否有不必要的全量 gather。**未在 smoke 上追加修复**（低优先级，smoke 本不要求跑完整训练）。

---

## [2026-07-06] 8GPU 正式脚本与 smoke/minitest 存在未同步的修复

**步骤：** 8 GPU 正式训练前置核查（用户要求：确认 smoke/minitest 的改动是否都同步到 `train_with_services.sh`）

**现象：** 逐条比对 `git log` + `diff` 后发现 `train_with_services.sh` 遗漏四处已在 smoke/minitest 验证过的修复：

| 遗漏项 | 影响 |
|---|---|
| 仍用 `openclaw gateway run` 而非当时的 `rl_gateway_proxy.py` | 会直接复现最初的 18789 404（此项后被 [下一条] 整体推翻重做）|
| `wait_for_port` 缺少 Traceback/CUDA OOM 快速失败检测 | 训练崩溃时会傻等满 900s 超时，而非立刻报错 |
| `REPO_ROOT` 未转发进训练启动子进程 | 目前因默认值恰好一致而未触发，但存在潜在风险 |
| 启动顺序（先起 gateway 还是先等 30000）| smoke/minitest 已改为先等 30000 |

**修复：** 全部补齐；commit `ed0aa01`（gateway proxy + 断点续训 `--load` + 启动顺序）、`61903e4`（快速失败检测 + `REPO_ROOT`）

---

## [2026-07-06] rl_gateway_proxy.py 是基于误诊的绕过方案；真实根因是配置开关默认关闭

**步骤：** 用户追问"能不能让 `openclaw gateway run` 自己有 workspace 工具能力"，触发重新排查

**背景：** 2026-07-03 曾诊断 `openclaw gateway run` 完全不提供 `/v1/chat/completions` 路由（"设备连接层，非 API 路由"），据此实现了 `scripts/rl_gateway_proxy.py`——绕开 OpenClaw、直连 30000 端口的裸转发代理（手动注入 `X-Session-Id`/`X-Turn-Type: main`）。该方案让 smoke 首次跑通训练数据流，但存在已知缺陷：policy 完全没有 workspace 文件读写工具，homework 文件互动这一论文核心设计无法真正发生。

**根因（本次查明）：** 本地发现 OpenClaw 本体源码克隆（`D:\MAO\Claude\openclaw`），读 `docs/gateway/openai-http-api.md` 确认：
- `/v1/chat/completions` 内置在 `openclaw gateway run` 里，走"a normal Gateway agent run"（与 `openclaw agent` 同代码路径），天然带完整工具调用能力，`rl-training-headers` 插件挂的 `before_prompt_build` 钩子正是这条路径触发的
- 该端点**默认禁用**，需 `gateway.http.endpoints.chatCompletions.enabled=true`
- OpenAI `model` 字段必须是 agent-target 格式（`openclaw`/`openclaw/<agentId>`/兼容别名 `openclaw:<agentId>`、`agent:<agentId>`），官方 `openclaw-test/*.py` 硬编码的 `"model": "default"` 不在支持列表内

服务器实测验证链条：
1. `openclaw config get gateway.http.endpoints.chatCompletions.enabled` → `Config path not found`（确认默认未开）
2. `openclaw config set ... true` + `openclaw config validate` → 通过
3. 重启 `openclaw gateway run --allow-unconfigured --force`，`model: "openclaw/default"` → `401 Unauthorized`（新终端 `$OPENCLAW_GATEWAY_TOKEN` 未设置）→ 改用 Python 读取原始 JSON（绕开 `config get` 对敏感字段的脱敏 `__OPENCLAW_REDACTED__`）拿到真实 token → 认证通过，卡在 `upstream provider timeout`
4. 确认 30000 端口当时确实无进程监听（`ps aux` 为空）→ 判定 timeout 是预期结果，非新 bug
5. 用官方脚本实际发送的 `"model": "default"` 复测 → `400 Invalid model. Use openclaw or openclaw/<agentId>.`，坐实兼容性问题

**结论：** 2026-07-03 的诊断不完整——404 根源是配置开关默认关闭，不是端点不存在。`rl_gateway_proxy.py` 方案虽然让训练队列跑通，但牺牲了论文要求的 agent 工具调用能力，是不必要的绕过。

**修复：**
- 撤掉 `scripts/rl_gateway_proxy.py`（已删除），三脚本（smoke/minitest/train_with_services.sh）改回真实 `openclaw gateway run --allow-unconfigured --force`
- 服务器 `~/.openclaw/openclaw.json` 已设置 `gateway.http.endpoints.chatCompletions.enabled=true`（持久化，无需每次重设）
- 新增 `scripts/prepare_openclaw_test_scripts.sh`：生成 `student_chat.py`/`TA_chat.py`/`teacher_chat.py` 的补丁副本（仅改 `"model": "default"` → `"model": "openclaw/default"` 一处），官方 `openclaw-test/` 目录本身不动；三脚本 `OPENCLAW_DIR` 改指向补丁副本
- commit `ea19053`

**待验证：** 重跑 smoke，确认（a）真实 agent 循环下 Student/TA/Teacher 对话里模型确有文件读写行为，（b）训练队列仍能正常累积（换成官方 `rl-training-headers` 插件注入 header 而非手工伪造）

---

<!-- 格式模板：

## [YYYY-MM-DD] 问题描述
**步骤：** Step X（对应 reproduction_guide.md 的步骤编号）
**报错：**
```
错误信息粘贴在这里
```
**解决方案：**
（填写后）

-->
