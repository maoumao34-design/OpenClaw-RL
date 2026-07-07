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

## [2026-07-07] smoke chatCompletions 配置未在新 job 环境生效（二次 404）

**步骤：** 4 GPU smoke `scripts/smoke_train_with_services.sh`（承接 2026-07-06 gateway 修复后的验证）

**现象：** 前一天已在交互式 shell 里确认 `gateway.http.endpoints.chatCompletions.enabled=true` 生效，但新提交的 smoke job 又复现了最初的 `404 Not Found`

**根因：** `openclaw` 本体（`/usr/lib/node_modules/openclaw/`）是系统级安装，job 容器和交互式 shell 都能找到；但 `~/.openclaw/openclaw.json` 是用户配置，job 容器很可能是从早于这次手动配置改动的镜像/模板生成的，交互式 shell 里的临时改动不会传播到新 job

**修复：** 不依赖"配置是否跨环境持久"这个不确定的前提，改为在 `launch_openclaw_gateway()` 里每次启动前强制 `openclaw config set gateway.http.endpoints.chatCompletions.enabled true`，并打印 `openclaw config get` 回读结果到日志验证；commit `9aa3c4a`

---

## [2026-07-07] OpenClaw 请求 max_completion_tokens=178220 导致 408

**步骤：** 4 GPU smoke，上一条修复后的新 job

**现象：** `student_chat.py` 报 `408 Client Error: Request Timeout`；`openclaw.log` 显示反复 `[agent/embedded] ... error=LLM request failed ... reason=timeout`，5 次重试后放弃

**根因：** 用 CPU-only mock server（Python + FastAPI，监听 30000 端口顶替 sglang 后端，不需要 GPU/训练进程）单独抓包，发现 OpenClaw 实际转发的请求体里 `"max_completion_tokens": 178220`，远超 sglang `context_length=8192`，被 400 拒绝；`OpenClawOPDAPIServer` 把这个 400 转成 500 抛给 OpenClaw agent，agent 归类为 timeout，重试耗尽后 408 传导回客户端。根因是 `~/.openclaw/openclaw.json` 的 `models.providers.sglang` 从未显式声明 `models[]`（无 `contextWindow`/`maxTokens`），OpenClaw 走自动发现，不知道模型真实的输出上限

**修复：** `launch_openclaw_gateway()` 里显式声明 `sglang.models=[{id:"qwen3-4b", contextWindow:8192, maxTokens:4096, ...}]`（maxTokens 明显小于 contextWindow 留出 prompt 空间，同 `PRM_MAX_NEW_TOKENS` 那次的道理）；commit `18fac58`

---

## [2026-07-07] smoke Teacher 第 4 轮 context overflow

**步骤：** 4 GPU smoke，上两条修复后真实 agent 循环首次跑通

**现象：** Teacher INIT 阶段第 4 轮（`--max-turns 4` 上限）返回 `Context overflow: prompt too large for the model. Try /reset...`；该题最终 `0/1 problems commented within turn limit`；脚本自身仍打印 `✅ SMOKE PASSED`（该判定只检查训练进程是否存活，见 2026-07-06 条目）

**根因：** smoke 把 `--sglang-context-length` 缩到 8192（省显存）；真实 agent 循环（工具调用 schema、系统提示词、reasoning token）本身开销就不小，累积到第 3-4 轮很容易突破 8192。与本次 session 早前两个问题（`PRM_MAX_NEW_TOKENS`、`max_completion_tokens`）同源——都是 smoke 缩配 context 过紧带来的连锁反应，这是第三次发作

**评估：** minitest/8GPU（context=32768）预期不受影响；未在 smoke 上追加修复（低优先级，smoke 本不要求跑完整对话）

---

## [2026-07-07] rl-training-headers 插件端到端失效（OpenClaw 内部实现问题，非论文/复现代码问题）

**步骤：** 上一条 context overflow 修复后，`training.log` 里 `combine samples` 持续 `0/4`，全部请求日志显示 `[side] session=unknown`

**排查过程**（全程用 CPU-only mock server 在 30000 端口直接抓包验证，不占 GPU 排队，不靠猜测）：

1. 用 mock server 抓包确认 `X-Session-Id`/`X-Turn-Type` 完全没有出现在 OpenClaw 实际发出的请求头里
2. `openclaw --log-level debug/trace gateway run` 确认 `rl-training-headers` 从未进入"尝试加载"列表（`loaded N plugin(s) (N attempted)` 里没有它），而单独执行 `openclaw plugins enable rl-training-headers` 能正常触发插件的 `register()`（打印 "activated (fetch patched)"）——说明插件代码本身没问题，问题在加载阶段
3. 对比能加载的 `browser`/`sglang` 与不能加载的 `clickclack`/`rl-training-headers` 的 `openclaw.plugin.json`：能加载的都有 `"enabledByDefault": true`；**`clickclack` 是官方原生插件，同样缺这个字段、同样加载不了**——排除"手动装的插件才有问题"这个假设，确认是这个 OpenClaw 版本（2026.6.9）本身的加载器行为
4. 给 `rl-training-headers` 的 manifest 补上 `"enabledByDefault": true` + `"activation": {"onStartup": true}`，插件成功加载（9 个插件里出现它），`register()` 正确触发，`before_prompt_build` 钩子也确认触发（`[hooks] running before_prompt_build (1 handlers, sequential)`）
5. 用 mock server 复测——header **依然没有到达实际出站请求**。确认问题不在"是否加载"，而在"patch `globalThis.fetch`"这套注入机制本身，在这个 OpenClaw 版本里对实际的 provider 调用不起作用（请求头里 `user-agent: OpenAI/JS 6.39.1` 表明走的是官方 `openai` npm SDK 客户端，大概率在构造时就缓存了一份 `fetch` 引用，不会每次请求重新读 `globalThis.fetch`）

**结论：** 这是 **OpenClaw 本体的内部实现问题**，跟论文设计、`OpenClaw-RL-official` 复现代码都无关。相关 manifest 补丁（`enabledByDefault`/`activation`）、调试补丁（`openclaw_opd_api_server.py` 里的临时 header 打印）已全部复原，不再依赖这个插件。

**替代方案**（偏离论文原设计的 header 注入机制，明确记录）：
- **`X-Turn-Type`**：改用 OpenClaw **官方**的 `models.providers.sglang.headers` 静态 header 配置，固定注入 `"main"`——这是官方支持的功能（`ModelProviderConfig.headers: Record<string, SecretInput>`），不是绕过；能用固定值是因为这条复现流水线里的调用全部是真实 main turn，从不触发 OpenClaw 的 heartbeat/memory/cron。实测确认 mock server 收到 `'x-turn-type': 'main'`
- **`X-Session-Id`**：没有静态配置的等价物（按对话区分）。发现 OpenClaw 自己会把 `session=agent:<agentId>:openai-user:<user>` 嵌入每次调用的 system prompt "Runtime:" 行里，`<user>` 正好就是 `student_chat.py`/`TA_chat.py`/`teacher_chat.py` 传的 `user` 字段。新增 `scripts/prepare_patched_openclaw_opd.sh`，拷贝打补丁给 `openclaw_opd_api_server.py` 加这个兜底解析（官方 `openclaw-opd/` 目录不动），`PATCHED_OPD_DIR` 注入训练 job 的 `PYTHONPATH`（需排在官方 `openclaw-opd/` 之前）

**修复：** commit `9aa3c4a`（配置强制设置）、`18fac58`（sglang provider 声明）、`2c1e851`（header workaround，含 `prepare_patched_openclaw_opd.sh`）

**待验证：** 真实 smoke job 里 `X-Session-Id` 解析是否生效（本地已单测通过正则提取、补丁文件语法检查）

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
