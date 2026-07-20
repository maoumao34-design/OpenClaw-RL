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

**评估（已被后续实测部分推翻，见下方更正）：** minitest（TP=2）/ 8GPU 正式（TP=4）显存分摊更充分，理论上不会复现；若 minitest 也 OOM 则需进一步排查 `update_weight_from_distributed.py` 是否有不必要的全量 gather。**未在 smoke 上追加修复**（低优先级，smoke 本不要求跑完整训练）。

**更正（2026-07-09）：** 这条评估只针对 **GPU 显存**维度成立。minitest（TP=2）在 `update_weights()` 同一触发点确实复现了 OOM，但资源种类是**节点系统内存**（128GB 节点打满），跟这里说的 TP 显存分摊无关——是两种不同资源、同一个崩溃触发点。详见 [2026-07-09 update_weights() 节点内存 OOM] 条目。

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

## [2026-07-08] smoke job 静默失败（无 traceback）——残留进程触发 cgroup OOM

**步骤：** 提交 smoke（4 GPU）验证 07-07 的 header workaround

**现象：** 训练进程在模型加载阶段（日志停在 `> number of parameters on (tensor, pipeline) model parallel rank (0, 0): 4022468096`）之后再无任何输出；`ps aux` 确认训练进程已不存在；`training.log` 全文（1524 行）搜索 `error|traceback|killed|oom|exception|fatal` 均无匹配——不是 Python 异常崩溃

**根因：** `dmesg`/`journalctl -k` 确认系统级 OOM killer 杀掉了一个 **16GB RSS 的 `node` 进程**，触发 cgroup `memory.oom.group`，把整个容器里的所有进程（包括这次训练进程）一起杀掉。进一步查当前存活进程发现两个 07-07 遗留、一直没清干净的进程：`/tmp/mock_sglang_server.py`（07-07 验证 `X-Turn-Type` header 时起的假后端，仍占着 30000 端口）和一个手动测试用的 `openclaw gateway run`。之前清理时只删了 `.py` 文件（`rm -f`），没有真正 `kill` 掉正在跑的进程，导致它们一直挂着（此时已跑了 41 分钟）

**修复：** 按 PID 手动 `kill -9` 两个残留进程，确认端口释放后重新提交 job 恢复正常。**教训**：以后清理测试产物时，删文件和杀进程是两件事，必须都做（`ps aux | grep` 确认没有残留后再提交新 job）

---

## [2026-07-08] smoke job 资源配置错误（1 GPU/16GB 而非 4 GPU）

**步骤：** 清理残留进程后重新提交 smoke，仍然卡在等待 port 30000

**现象：** 跟上一条现象类似（`等待 RL training proxy (port 30000)` 一直不就绪），但这次不是残留进程问题

**根因：** 用户核实发现这次 job 提交参数误设为 1 GPU / 16GB 内存。smoke（topk-select）硬性需要 4 GPU：官方脚本内置断言 `if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS + PRM_TEACHER_GPUS > NUM_GPUS )); then exit 1; fi`，四个角色（Actor×1 + Rollout×1 + PRM SGLang×1 + PRM Teacher×1）之和恰好是 4，PRM Teacher 必须有独立 GPU（Megatron 路径不能共享），无法压缩到 3 GPU 以下

**修复：** 用户自行修正 job 提交参数为 4 GPU、≥128GB 内存，重新排队

**附带确认：** 换用非 H 系列 GPU 测试是否可行——理论上 bf16 训练不是 H 系列专属，但（a）smoke 用 TP=1 单卡需扛下整个 4B 模型+KV cache，实测占用约 77GB/95GB，换成更小显存的卡大概率 OOM；（b）flash-attn/TransformerEngine/apex/flashinfer 等扩展是针对当前 GPU 架构专门编译的，换架构大概率需要重新编译，不是简单换卡就能跑。未做实际测试，仅为可行性评估。

**更正（2026-07-10）：** (b) 这条猜测已被实测推翻。A800（Ampere，sm_80）minitest 上 Megatron 训练步正常触发（`training.log` 里 `Timer train start` 顺利执行，无任何 kernel/import 报错），说明当时装的 flash-attn/TE/apex/flashinfer 构建产物并不是 H20（sm_90）专属编译，本来就支持多架构（或默认构建配置本就覆盖了 Ampere）。项目里唯一真正要求 sm_90 的是 DeepEP（`work_log.md` 06-22 条目），但 Qwen3-4B 是稠密模型不依赖 DeepEP，当时就没装，从未被本项目实际用上。

---

## [2026-07-08] smoke 首次验证 header workaround：X-Turn-Type 生效，X-Session-Id 仍未生效

**步骤：** 修复上述两个问题后，smoke job（`smoke_20260708_171455`）正常跑完，脚本打印 `✅ SMOKE PASSED`

**验证结果：**
- ✅ **`X-Turn-Type: main`（07-07 加的官方 `models.providers.sglang.headers` 静态配置）确认生效**：`training.log` 里 `[main]` 出现 12 次，不再是清一色 `[side]`
- ✅ **训练队列首次真实累积样本**：`combine samples` 从 `0/4` 涨到 5 个真实样本被 `submitted`（`OPD+RL sample`/`RL sample`，`index=0` 到 `index=4`），PRM 评审也是真实投票（`eval_votes=[1]`/`eval_votes=[-1]`），不再全部 `fail`——这是本项目第一次确认训练数据真实流入队列
- ❌ **`X-Session-Id`（07-07 加的 Runtime 行解析 fallback）仍然全部是 `unknown`**（33 次）。逐项核对：
  1. `training.log` 里确认 `PYTHONPATH` 正确把 `patched-openclaw-opd/` 排在官方 `openclaw-opd/` 之前
  2. 补丁文件确实存在、确实被导入过（目录下有 `__pycache__`）
  3. 补丁代码本身（`_extract_session_id_from_system_prompt` 函数定义 + `session_id = (...)` 拼接）逐行核对与预期一致
  4. 但函数显然返回了 `None`（否则不会落到 `"unknown"` 兜底），说明真实请求的 system prompt 内容和 07-07 手动测试时用 mock server 抓到的样本不完全一致，具体差异未知

**处理：** 不猜测原因，在 `_extract_session_id_from_system_prompt` 里加调试日志——没匹配到 Runtime 行时打印最后一条 system message 内容的尾部（500 字符）；commit `593a0e0`

**待验证：** 下一次 smoke 跑完后查 `training.log` 里的 `[SESSION-ID-DEBUG]` 输出，确认真实 system prompt 内容到底长什么样，定位解析失败的具体原因

---

## [2026-07-09] header 注入机制完整调查：确认是 OpenClaw 版本迭代破坏的架构问题，非配置/安装问题

**背景：** 上一条遗留的 `X-Session-Id` 全为 `unknown` 问题，追查后发现根源比预想的更深，牵出一整条完整的调查链。记录完整过程和证据，即使最终仍用 workaround 方案，这份记录能说明"为什么不用官方机制"是有根据的判断，不是绕不过去就放弃。

### 第一部分：smoke context 修复验证（真实数据首次证实 Runtime 行解析有效）

`smoke_20260709_103410`（context 已从 8192 改回官方值 32768，见前一天 commit `b2fe9ea`）跑出真实数据：`session=ta-grade-0-5410`（真实会话 ID，不再是 `unknown`）、`prompt_tokens=19075`（真实 TA 批改轮次的大 prompt，不是内部摘要调用的 275 token）、`submitted OPD+RL sample ... prompt_len=18862`（真实样本入训练队列）。**结论：`X-Turn-Type` 静态 header workaround + `X-Session-Id` Runtime 行解析 workaround，在真实 context=32768 场景下确认完全生效**，之前怀疑的"session_id 解析失败"其实是 smoke context=8192 太小、真实轮次从未跑通导致的假象（见前一条 07-08 记录）。

任务因残留进程触发的 cgroup OOM 级联杀而失败（同类问题第二次发作，`ps aux --sort=-rss`/`ps -eo pid,etime,rss,cmd` 排查后确认残留进程已被 OOM killer 自己清理，无需手动 kill，环境已干净），核心修复（context 大小、header workaround）已确认有效，不受这次失败影响。

打了 git tag `working-static-header-workaround` 标记这个已验证可用的版本，作为后续任何进一步改动的回退点。

### 第二部分：尝试用官方插件机制替代 workaround（undici dispatcher，最终判定不可行）

**动机：** static header + Runtime 行解析是偏离论文设计的临时方案，`X-Turn-Type` 是写死的静态值（无法区分真实 main turn 和 OpenClaw 内部的 context-summarization 兜底调用）。想改用官方插件（`OpenClaw-RL-official/extensions/rl-training-headers/`）真正拿到 `ctx.trigger`/`ctx.sessionId`。

**尝试 1（失败，根源已查清）：** 官方插件用 `globalThis.fetch` 打补丁注入 header——07-07 已确认这条路在当前 OpenClaw（2026.6.9）上完全不生效（mock server 抓包证实 header 从不到达）。本次进一步用源码追证：`src/llm/providers/openai-completions.ts` 的 `createClient()` **每次调用都全新构造** `new OpenAI({...})`（无缓存），但从不显式传 `fetch` 参数，依赖 SDK 自身默认解析——大概率是 `openai` npm 包自己在模块加载时就把 `fetch` 引用锁死，跟插件何时打补丁、`globalThis.fetch` 后来指向哪里都无关。

**尝试 2（失败，根源已查清）：** 改用 `undici.setGlobalDispatcher()`，比 `globalThis.fetch` 更底层，理论上能绕开引用缓存问题。独立隔离测试（`scripts/test_undici_header_injection.mjs`）验证机制本身可行（mock server 收到注入的 header）。但部署到真实 OpenClaw gateway 后，用带调试日志的插件版本（多轮迭代：先怀疑 undici 版本不匹配——Node 内置 6.27.0 vs npm 装的 8.5.0，读 `undici` v8.5.0 源码确认它确实做了向后兼容桥接（`Symbol.for('undici.globalDispatcher.1')` legacy key + `Dispatcher1Wrapper`），排除版本不匹配这个假设）最终查到真实根因：**OpenClaw 自己的 `src/agents/embedded-agent-runner/run/attempt.ts:836` 每处理一轮 agent 交互都无条件调用 `ensureGlobalUndiciDispatcherStreamTimeouts()`**（强制设置 30 分钟流超时），设计上应该幂等（值不变就不重设 dispatcher），但实测每一轮都会真的触发 `setGlobalDispatcher`，把 dispatcher 换成全新的普通 `Agent`，覆盖掉插件设置的拦截链——怀疑是 `resolveUndiciAutoSelectFamily()` 返回值不稳定导致幂等检查失效。**这跟代理配置无关**（`config.proxy.enabled` 本来就是 false，代理相关分支本来就是空操作），是每轮对话都会触发的核心逻辑，无法通过配置关闭。

用"每秒轮询、检测到被换就重新 compose"这种方式实测：日志显示每次检查都发现被换了（`was Agent, now Agent`），说明这个重置发生的频率至少跟我们的检查频率一样快，是持续性的，不是一次性的启动期竞态，用更高频轮询硬扛属于跟框架设计对抗，投入产出比不合适。

### 第三部分：版本考古——确认论文当时的 OpenClaw 版本机制原本是有效的

**动机：** 用户提出疑问——会不会是我们漏看了某个配置项，而不是 OpenClaw 真的把这条路封死了？

**核查：** `OpenClaw-RL-official` 仓库全文搜索找不到任何 OpenClaw 版本锁定（`package.json`/`README`/CI 配置均无版本号）。但插件所在路径 `extensions/rl-training-headers/` 的 git 提交历史只有两次真实提交（作者 Yinjie Wang，均为 "Add files via upload"）：`2026-03-20` 和 `2026-04-04`。对照 OpenClaw 自己的 CHANGELOG，这个时间区间对应大约 OpenClaw `2026.3.22 ~ 2026.4.5` 版本，比我们现在跑的 `2026.6.9` 早 2-2.5 个月。

**直接验证（关键）：** 本地 `openclaw` 仓库原本是浅克隆（只有最新一个 commit），用 `git fetch --depth=1 origin <2026-04-04 附近的具体 commit sha>`（sha 通过 GitHub API `https://api.github.com/repos/openclaw/openclaw/commits?until=2026-04-04T23:59:59Z` 查到）单独拉取了这个历史时间点的完整代码树，直接读当时的源码：

- 当时的 OpenAI 传输代码在 `src/agents/openai-transport-stream.ts`（**更正**：这次继续调查证实这个文件现在仍然存在，不是被移除了——见下方第四部分，当初这句判断是错的，是没有确认清楚就下的结论），客户端构造函数 `createOpenAIResponsesClient` **显式传了 `fetch: buildGuardedModelFetch(model)`**，并且 `buildOpenAIClientHeaders(model, context, optionHeaders, turnHeaders)` **有一个显式的 `turnHeaders` 参数**会被合并进 `defaultHeaders`——每轮动态 header 在当时是官方代码里的第一等公民特性，不是要靠外部插件 hack 进去的东西。
- 追进 `buildGuardedModelFetch` → `fetchWithSsrFGuard`（`src/infra/net/fetch-guard.ts`），关键一行：`const fetcher: FetchLike | undefined = params.fetchImpl ?? globalThis.fetch;`——**当时 OpenClaw 自己的传输层代码，每次发请求都会动态重新读取一次 `globalThis.fetch`**。

**结论（有源码证据，非推测，2026-07-09 当天后续已修正细节，见第四部分）：** 论文写插件那会儿（约 2026.3-4 月），OpenClaw 自身的传输代码有意设计成动态读取 `globalThis.fetch`，所以插件的 `globalThis.fetch` 打补丁机制在当时是真实有效的。**不是我们装的版本有问题、也不是漏看了配置，是 OpenClaw 项目本身在这几个月的快速迭代（同期 CHANGELOG 显示单周合并 400+ PR）中，把论文依赖的这个底层机制改掉了**——但"改掉"的具体方式比最初判断的更精细，见下方第四部分。

### 第四部分：修正——机制没有被移除，是被一道新加的门槛挡住了

**背景：** 打算按上面的思路，在当前版本的 `createClient()`（`src/llm/providers/openai-completions.ts`，编译后 `dist/openai-completions-D8IP0i-n.js`）里补一个动态读取 `globalThis.fetch` 的 `fetch` 参数。备份原文件后插入调试日志（`console.error` 打印 `sessionId`/`compat.sendSessionAffinityHeaders`），部署到服务器，用真实请求测试。

**关键发现（推翻了第三部分的部分结论）：** 调试日志**从未触发**——说明 `openai-completions.ts` 的 `createClient()` 根本没有被我们的 sglang provider 请求调用过。追下去发现：OpenClaw 有独立的 `extensions/sglang/` provider 插件，只注册了 `auth`/`catalog.run`/消息重放逻辑，**没有自己的 HTTP 客户端构造代码**；真正的请求分发是通过 `src/agents/provider-transport-stream.ts` 的 `hasOpenClawTransportRequirement(model)` 判断（检查 `request.proxy || request.tls || getModelProviderLocalService(model)`）：

- 条件为真 → 走 `src/agents/openai-transport-stream.ts` 的 `createOpenAICompletionsClient()`——**这个函数现在依然存在，依然显式传 `fetch: buildGuardedModelFetch(model)`（动态读取 `globalThis.fetch`），一行没有被移除**
- 条件为假（我们目前的情况）→ 走 `openai-completions.ts` 的 `createClient()`，也就是我们插了调试日志、确认从未被触发的那个函数

**用户提出的关键问题：** 会不会是论文原作者的部署本来就配置了真的本地服务管理（`localService`），而不是我们漏看了什么？

**验证（用已经拉取的 2026-04-04 附近历史代码）：**
```
git grep -n "hasOpenClawTransportRequirement" FETCH_HEAD -- '*.ts'   → 零匹配
```
**`hasOpenClawTransportRequirement` 这道门槛判断在 4 月版本里完全不存在**。当时 `createOpenAICompletionsClient`（动态读取 `globalThis.fetch` 那个版本）是被**无条件直接调用**的（`openai-transport-stream.ts:949`），不需要任何 `localService`/`proxy`/`tls` 配置。

**修正后的结论：** 论文插件当年能用，不是因为默认配置里悄悄开了 `localService`，而是因为 4 月那会儿根本没有这道门槛——所有 OpenAI 兼容类型的 provider，默认都会走动态读取 `globalThis.fetch` 的那条路径。**这道门槛判断是 4 月到 6 月之间新加进去的**，把原本无条件的默认行为收窄成了需要满足特定条件才能触达——`globalThis.fetch` 动态读取这个机制本身**从未被移除**，只是现在默认路径不再通向它。

**下一步（待验证，测试中）：** 给 `models.providers.sglang` 加一个 `localService` 配置块（`command` 指向一个无害的空操作可执行文件如 `/bin/true`，真实请求仍然走已配置的 `baseUrl`），让 `getModelProviderLocalService(model)` 返回真值，从而满足 `hasOpenClawTransportRequirement`，把请求引导到仍然保留着动态 `globalThis.fetch` 读取行为的 `openai-transport-stream.ts` 路径上——这样官方插件原始的 `globalThis.fetch` 补丁机制应该就能不做任何代码改动直接生效。**这不是滥用 `localService` 这个功能**（我们的 sglang 由训练框架自己管理生命周期，不能真的交给 OpenClaw spawn，所以用无害空操作命令只是为了满足这道后加的门槛判断，恢复到 4 月版本本来就有的无条件默认行为，性质上更接近"绕开一个新加的限制"，不是钻空子）。

如果这条路验证通过：把之前的调试日志（`createClient()` 里的 `console.error`）从 `dist/openai-completions-D8IP0i-n.js` 移除，恢复到备份版本（`.bak-original`），确认不需要碰这个核心文件。如果不通过：回退到 `working-static-header-workaround` tag。

### 第五部分：`localService` 配置验证通过，但发现最终的、真正无法绕过的结构性死路

**验证 `localService` 路由是否生效：** 给 `models.providers.sglang` 加了 `localService: {command: "/bin/true", healthUrl: "http://127.0.0.1:30000/health"}`（`command` 指向无害空操作程序，真实请求仍走已配置的 `baseUrl`；第一次用 `/v1/models` 当 `healthUrl` 导致请求超时——因为 mock server 只实现了 `/health` 这个 GET 路径，改过来就正常了）。在 `openai-transport-stream-*.js` 的 `createOpenAICompletionsClient()` 插了调试日志，确认**这次真的被调用了**（`provider=sglang model=qwen3-4b`）——`localService` 配置成功把路由切到了保留着 `fetch: buildGuardedModelFetch(model)` 的正确代码路径。

**但 header 依然没有到达。** 同时给恢复的原版插件（`globalThis.fetch` 补丁版）加了调试日志，完整链路证据如下：
```
before_prompt_build fired: trigger=user sessionId=<真实值> turnType=main   ← 正确
createOpenAICompletionsClient CALLED provider=sglang model=qwen3-4b        ← 路由确认正确
patched fetch invoked: method=undefined hasScopedHeaders=false （×2）      ← 但明显不是真实的那次 POST
```
mock server 收到了正常的 POST 请求，但完全没经过插件的 header 合并逻辑——说明打了补丁的 `globalThis.fetch` 被调用了两次，但都不是真正发给 sglang 的那次请求（`method=undefined`，真实请求应该是 `POST`），大概率是某个无关的次要调用（如健康检查）。

**追进 `fetchWithSsrFGuardInternal`（`src/infra/net/fetch-guard.ts`）找到真正原因：**
```ts
const supportsDispatcherInit = params.fetchImpl !== void 0 && !isAmbientGlobalFetch({...}) || isUsingMockedFetch;
const shouldUseRuntimeFetch = Boolean(dispatcher) && !supportsDispatcherInit;
const response = shouldUseRuntimeFetch
  ? await fetchWithRuntimeDispatcher(parsedUrl.toString(), init)
  : await defaultFetch(parsedUrl.toString(), init);
```
真实网络请求（尤其是要走 DNS 钉死防护的，几乎是全部）会构造出一个 `dispatcher`。只要 `dispatcher` 是真值、且当前 `fetch` 函数不满足 `supportsDispatcherInit`，就完全绕开 `defaultFetch`（= `params.fetchImpl ?? globalThis.fetch`），改用 `fetchWithRuntimeDispatcher`（`src/infra/net/runtime-fetch.ts:96`，直接调用 undici 自己的内部 runtime fetch）。而 `isUsingMockedFetch = isMockedFetch(defaultFetch)`——查 `isMockedFetch` 定义（`runtime-fetch.ts:88`）：
```ts
/** Returns true for Vitest-style mocked fetch functions that should stay injectable. */
export function isMockedFetch(fetchImpl) {
  if (typeof fetchImpl !== "function") return false;
  return typeof fetchImpl.mock === "object";
}
```
**这是专门给 Vitest 测试框架的 mock 函数识别用的**（Vitest mock 函数有 `.mock` 属性），不是给插件这种改写 `globalThis.fetch` 的场景设计的——我们的补丁函数不满足这个检测，所以真实请求 100% 会走 `fetchWithRuntimeDispatcher`，完全绕开 `globalThis.fetch`。

**核查这道绕过逻辑是不是也是新加的：**
```
git grep -n "isMockedFetch|fetchWithRuntimeDispatcher|isAmbientGlobalFetch" FETCH_HEAD -- '*.ts'   → 零匹配
```
**4 月版本里同样不存在**——跟第三/四部分的门槛判断一样，是 4-6 月这段时间新加的。

**最终结论：** 论文机制失效，横跨两层独立的、都是这几个月新加的架构变化：(1) `hasOpenClawTransportRequirement` 门槛判断，把大多数 provider 的默认路由改离了动态读取 `globalThis.fetch` 的代码路径（可以用 `localService` 配置绕过，已验证）；(2) 即使绕过第一层、走到正确代码路径，还有一道专门的 SSRF 安全机制，只认 Vitest 测试环境的 mock 函数，其余一律绕开 `globalThis.fetch` 直接用内部 dispatcher 发请求（**没有配置开关，无法绕过**，是刻意的安全边界）。第二层是真正无法逾越的死路——不存在"再调一个配置就通"的可能性，这不是配置问题，是 OpenClaw 故意不再信任外部代码接管真实网络请求。

**处理：** 撤销所有临时调试改动，恢复到 `working-static-header-workaround` tag（`git checkout working-static-header-workaround`，服务器上手动删除临时的 `localService` 配置、恢复 `openai-completions-D8IP0i-n.js`/`openai-transport-stream-*.js`/插件 `index.js` 的 `.bak-original`/`.bak-clean-original` 备份），继续使用已验证有效的静态 header + Runtime 行解析方案。这份调查记录作为"为什么论文的 header 注入机制在当前 OpenClaw 版本上无法使用"的完整依据保留。

### 第六部分：改用 appendSystemContext 动态正文注入，替代已确认失效的 header 注入（已实现并验证）

**动机：** header/dispatcher 两层注入机制均已确认结构性失效（第四、五部分），但今天早些时候已经验证过（见本文档更早的插件重写讨论）`before_prompt_build` 的 `appendSystemContext` 返回字段能把内容真实写入发给策略模型的 system prompt——这条路径完全不经过 `fetchWithSsrFGuardInternal`/`fetchWithRuntimeDispatcher` 这层，不受第五部分那道 SSRF 安全机制影响。

**方案：** 插件在 `before_prompt_build` 里把 `ctx.trigger`/`ctx.sessionId` 编码成标记文字，通过 `appendSystemContext` 写入 system prompt；`openclaw_opd_api_server.py` 补丁从 `messages` 里解析出这段标记作为 `session_id`/`turn_type` 的兜底来源（`x_session_id`/`x_turn_type` header 仍然优先，标记只是兜底），**并且在转发给 sglang 之前、以及计算训练样本的 `prompt_ids` 之前，都要把这段标记文字从 `messages` 里清理掉**——保证策略模型和训练数据看到的都是干净版本，跟论文设计的效果等价（不是同一份代码，是我们另外实现的，需要如实记录）。`turn_type` 没有标记时的默认值会从现在写死的 `"main"` 改回官方默认的 `"side"`。

**两个未验证点（实现后必须用真实多轮对话测试确认，不能假设）：**
1. `appendSystemContext` 追加的内容会不会被 OpenClaw 自己持久化进对话历史，导致后续轮次的 system prompt 里标记文字重复出现或者累积——如果会，需要额外处理（比如换一种更不容易被误存的写法，或者接受这个副作用但要记录清楚）
2. OpenClaw 内部的 context-summarization 调用到底触不触发 `before_prompt_build`——如果不触发，标记天然缺失，`turn_type` 默认回落到 `"side"`，等于顺带解决了 2026-07-08/09 记录过的"内部摘要调用被误标成 main"问题；如果触发但 `ctx.trigger` 不是 `"user"`，也需要看它具体是什么值再决定要不要扩充 `SIDE_TRIGGERS`

**实现 + 验证结果：**
- `scripts/prepare_patched_rl_training_headers.sh`（插件）、`scripts/prepare_patched_openclaw_opd.sh`（服务端解析+清理）已按上述方案实现，commit `be25e8b`
- **mock server 实测确认插件生效**：真实动态标记（`session_id='5077dd70-...' turn_type='main'`）到达了发给 sglang 的请求正文，完全绕开了第四/五部分那道 SSRF 安全机制
- **本地单测确认服务端解析+清理逻辑正确**：用真实抓到的标记值验证，能正确解析出 `session_id`/`turn_type`，清理后 system message 内容干净（不含标记残留），无标记时正确返回 `None`/回退默认值，原始 `messages` 对象不会被意外修改
- 部署逻辑接入 `smoke_train_with_services.sh`（commit `df22940`）、`minitest_train_with_services.sh`（commit `a7d1da6`）、`train_with_services.sh`（commit `73ccfef`），三脚本保持一致，废弃的静态 `X-Turn-Type` header workaround 已移除
- 真实 GPU minitest 跑通期间（`session_id` 全程真实、`prompt_tokens` 正常增长）未观察到标记污染多轮历史的迹象（待验证点1），但训练本身反复中途崩溃（见第七部分），还没有跑够长的多轮对话做充分验证；待验证点2（context-summarization 是否触发 `before_prompt_build`）尚未专门验证

---

## [2026-07-09] minitest 训练进行到中途反复静默崩溃——根因已确认：节点系统内存 OOM（非 GPU 显存、非 NCCL），与 [2026-07-06 update_weights() OOM] 同类问题

**现象：** 提交 minitest（5 GPU，TP=2）验证 header workaround 完整链路，连续多次在训练开始后几分钟内（第一轮 rollout 完成、进入真实梯度/优化器训练步不久）整个任务无预警断线，modelfactory 平台判定为失败。`training.log`/`openclaw.log` 均无 Python traceback，`dmesg -T`/`journalctl -k` 精确时间窗口查无 OOM killer 记录。累计复现 4 次（2 次 smoke + 2 次 minitest，TP=1 和 TP=2 都有），排除了"smoke TP=1 显存吃紧"这个最初的猜测（minitest TP=2 显存分摊更充分，同样复现）。

**排查过程：**
1. 一开始误判 `ps aux` 查不到训练进程 = 任务已死，后来发现这个交互终端和实际计算节点是分离的（`/tmp/ray` 在这个 shell 里完全不存在），`ps aux` 本来就看不到远程节点的进程，跟任务死没死无关。改用"`training.log` 最新时间戳是否还在跟系统时间同步更新"作为判断任务存活的可靠依据。
2. 崩溃前的日志里，每次都能看到 `MegatronTrainRayActor` 打印 `rerun_state_machine.py:1300 - Implicit initialization of Rerun State Machine!` + `RerunStateMachine initialized in mode RerunMode.DISABLED`（TP=2 时两个 Actor 进程同时打印），一开始怀疑这是崩溃的直接原因。
3. 直接查本地 `Megatron-LM` 源码（`rerun_state_machine.py`）确认：
   - "Implicit initialization" 只是懒加载——`train_step()` 每次迭代第一行都调用 `get_rerun_state_machine()`，第一次调用时才初始化，纯粹是"第一次真正的训练步骤开始了"的时间标记，不是被异常触发的
   - `RerunMode.DISABLED` 模式下，`validate_result()` 如果真检测到 NaN/Inf，依然会抛出 `RuntimeError`（有 traceback）——跟观察到的"完全静默、无 traceback"对不上，**这条线索被排除，不是崩溃的直接原因**
4. 顺带查出一个真实存在、但与本次崩溃无直接关联的 bug：`slime`（RL 框架）从未调用 Megatron 官方的 `initialize_megatron()`/`initialize_rerun_state_machine()`，导致 `--rerun-mode` 参数（官方默认 `validate_results`）从未真正生效，`RerunStateMachine` 永远走隐式初始化的硬编码默认值 `DISABLED`。这是 `slime`/`Megatron-LM` 集成上的缺口，`CLAUDE.md` 里这两个目录标注为"可以读、改、使用"，理论上能修，但因为（a）不确定跟这次崩溃有没有关系，（b）改分布式训练初始化时机风险较高，暂缓修复，优先级排在下面的 NCCL 诊断之后。

**当前最可能的方向（未确认）：** "完全静默、无 traceback、无 OOM、总在第一次真正的分布式梯度同步/优化器步骤开始后不久发生"这个模式，更符合 NCCL 层面的分布式通信挂起/死锁特征——这类问题通常不产生 Python 异常（进程卡住而不是崩溃）。

**处理：** 在 `run_openclaw_topk_select_modelfactory.sh` 的 `RUNTIME_ENV_JSON` 里加 `NCCL_DEBUG=INFO`，对 smoke/minitest/8GPU 统一生效，commit `d84b71c`。这是临时诊断手段（日志会很啰嗦），目的是下次复现时拿到 NCCL 自己的诊断输出作为真实证据，而不是继续猜测。

**待验证（已解决，见下方更新）：** ~~下次 minitest 重跑，若再次复现同样的静默崩溃，检查 NCCL_DEBUG 输出定位真实原因。~~

### 更新（2026-07-09 当天，加 NCCL_DEBUG=INFO 后重跑）：真实根因浮出水面

**现象：** 加 `NCCL_DEBUG=INFO` 后重新提交 minitest，这次训练推进得更远（INIT 阶段基本走完，进入 Joint round，`Final collected 16 samples from rollout to train`、`perf 1: {'rollout/prm_eval_score': 0.5, ...}` 均为真实数据），随后崩溃——但这次 `training.log` 第一次留下了完整 Python traceback（此前 4 次全部静默无 traceback）：

```
Traceback (most recent call last):
  File ".../slime/train_async.py", line 156, in train
    actor_model.update_weights()
  File ".../slime/slime/ray/actor_group.py", line 125, in update_weights
    return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])
...
ray.exceptions.RayTaskError(OutOfMemoryError): ray::MegatronTrainRayActor.update_weights()
  File ".../update_weight_from_distributed.py", line 81, in update_weights
    ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.
Memory on the node (IP: 172.18.250.100, ...) where the lease (name=SGLangEngine.__init__, pid=3719, memory used=0.68GB)
was running was 126.63GB / 128.00GB (0.989315), which exceeds the memory usage threshold of 0.95.
Ray killed this worker...
Top 10 memory users:
PID     MEM(GB) COMMAND
3565    45.16   ray::MegatronTrainRayActor
3312    31.83   ray::MegatronTrainRayActor.train
3566    18.05   ray::MegatronTrainRayActor
```

**根因（有完整 traceback 证据，非推测）：** 节点系统内存（128GB，非 GPU 显存）在 `update_weights()` 的 `pause_generation` 阶段被打满到 98.9%，触发 Ray 自身的内存监控主动 kill 掉一个 worker（这次 kill 中的恰好是 `pause_generation.remote()` 依赖的 `SGLangEngine.__init__` lease，导致 `ray.get()` 抛出可捕获异常）。仅 3 个 `MegatronTrainRayActor` 相关进程就占了约 95GB，加上 sglang engine/PRM 等其余进程把 128GB 节点内存打满。

**与此前 4 次"静默无 traceback"崩溃的关系（推断，未逐一验证）：** 很可能是同一个系统内存 OOM，只是每次 Ray 内存监控杀掉的具体进程不同——这次刚好杀在一个有 `ray.get()`/Python 异常捕获包裹的调用点上，所以能看到完整 traceback；若杀在某个底层 NCCL 集合通信调用中间（没有异常捕获），其余 rank 会一直卡等一个已死的 peer，表现为"静默挂起、无 traceback"，与此前 4 次观察到的症状一致。**NCCL_DEBUG=INFO 输出本身未发现异常**（channel/tree/P2P 建连全部正常完成），进一步排除了 NCCL 协议层本身的问题——挂起是内存 OOM 的下游后果，不是 NCCL 的锅。

**与 [2026-07-06] `smoke update_weights() OOM（TP=1 缩配显存不足）` 的关系：** 同一个崩溃触发点（`update_weights()` 权重同步阶段），但资源种类不同——07-06 那次是 **GPU 显存**（TP=1 单卡挤爆），这次是 **节点系统内存**（TP=2，与 GPU 显存无关）。07-06 条目当时的评估"**minitest（TP=2）/8GPU（TP=4）显存分摊更充分，理论上不会复现**"——这个评估只覆盖了 GPU 显存维度，没有覆盖系统内存维度，**已被本次 minitest（TP=2）实测结果推翻**：系统内存 OOM 与 TP 并行度无关，是节点总内存预算（128GB）相对于 Megatron actor + rollout engine + PRM 等全部常驻进程的内存footprint 本身就不够用。

**处理：** 权衡排队难度后，先把 minitest 任务提交时申请的系统内存提高到 **192GB**（128GB 峰值 126.63GB 之上留约 64GB 余量，比一次性申请 256GB 好排很多），重新提交验证；若 192GB 仍复现，再升级到 256GB。若某个内存档位下不再复现，说明是纯粹的资源申请不足（跟 07-08 "误设 1 GPU/16GB" 同类：任务提交参数问题，非代码 bug）；若 256GB 仍复现，需要进一步定位是否 Megatron actor 本身内存泄漏/常驻内存过大。

顺带评估了 CPU（当前 16 核）：不是这次 OOM 的直接原因（不同资源维度），但崩溃日志 top 进程列表显示 5 GPU 任务同时有 sglang scheduler/detokenizer、multiprocessing.spawn、gcs_server 等多个 CPU 侧进程，16 核偏紧，建议按比例一并提高，暂未做最终决定。

**待验证：** 192GB 内存重跑 minitest（尚未排上队），确认是否解决；若不够升级到 256GB；解决后 8GPU 正式提交时同步申请更高系统内存（+ 视情况提高 CPU 核数，8GPU 版本进程数更多）。

**更新（2026-07-10，A800 minitest 已验证解决）：** `minitest_20260710_150305`（A800，提交时已申请更高系统内存）连续跑过 10 次 `update_weights()`（`training.log` 里 `perf 23` 到 `perf 32`，每次 `perf/update_weights_time` 仅 0.45-1.5s，全部成功，无 OOM/Traceback）——此前每次都是卡在**第一次** `update_weights()` 就死，这次稳定跑过 10 次，**确认提高系统内存能解决这个 OOM**。不需要等它跑完全部 300 步（预计 25-30 小时）即可确认；8GPU 正式提交时按同样思路申请更高系统内存。

---

## [2026-07-10] INIT 阶段 TA/Teacher 与网关连接中途被拒绝，静默跳过导致 homework 数据不完整（与上条 OOM 是同一次跑，但相互独立的两个问题）

**背景：** 追查"这次 A800 minitest 进度是不是异常慢"时，用 `minitest_20260709_172118`（就是上一条 OOM traceback 那次）作对比基准，结果发现这个基准本身就不干净。

**现象：** `training.log` 里完全没有 INIT 阶段的记录——`grep "INIT"` / `grep "Problem.*session"` 都是 0 条，说明 INIT 阶段的输出根本不写进 `training.log`，而是写进同目录下的 `simulation.log`。查 `simulation.log` 发现：

```
results_TA_init.txt / results_TA_all.txt / results_teacher_init.txt / results_teacher_all.txt 均为 0 字节
```

`simulation.log` 里 TA 和 Teacher 的 INIT 阶段都各自重试 3-4 次后报同样的错：

```
requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=18789): Max retries exceeded with url: /v1/chat/completions
(Caused by NewConnectionError("HTTPConnection(host='localhost', port=18789): Failed to establish a new connection: [Errno 111] Connection refused"))
警告：TA 模拟未完全完成，继续训练
...
警告：Teacher 模拟未完全完成，继续训练
```

**根因：** `openclaw.log` 显示网关本身 17:24:31 就已经 `ready`，晚于 TA/Teacher 尝试连接的时间，且同一次跑里 Joint round 阶段 Student 的后续请求又能正常打通网关、真实样本也正常提交——说明网关不是没启动、也没有彻底挂掉，是中途有一段短暂不可达的窗口，TA/Teacher 的 INIT 恰好撞上了这段窗口。`run_one_persona()`（`minitest_train_with_services.sh` / `train_with_services.sh` 共用同一份逻辑）只在整个脚本最开始 `wait_for_port` 检查一次网关就绪，之后 Student/TA/Teacher 依次调用时不再复查；单次调用失败就直接把 stderr 打成"警告...继续训练"放过，不重试。

**影响：** 这次 Joint round 收集到的"16 个真实样本"是在 TA 没有真正批改 homework1、Teacher 没有真正评论 homework2 的情况下产生的——`homework1`/`homework2` 数据本身不完整，不是一次干净的训练信号。跟同一次跑里后面发生的 `update_weights()` 系统内存 OOM 是两个独立问题：网关断连发生在 INIT 阶段（~17:24-17:31），没有让任务崩溃，任务是后面才被系统内存 OOM 真正杀死的（~17:37 之后，见上条）。

**修复：** `run_one_persona()` 改为每次调用前先用 `wait_for_port` 复查网关是否仍可达（已就绪时开销接近零），网关确认可达后再执行；单次 Python 脚本调用失败时不再直接放过，最多重试 3 次（每次间隔 10s），3 次都失败才保留原有的"警告...继续训练"兜底（措辞加了"数据可能不完整"提示）。`smoke_train_with_services.sh` 不用 `run_one_persona`（不受影响）。commit（待补）。已确认此次修复只改本地脚本源码，不影响当前正在跑的 A800 minitest job（该 job 提交时已经把脚本拷贝/生成到自己的日志目录，不会读取后续修改）。

**待验证：** 下次提交的 minitest/8GPU 如果再遇到 18789 中途不可达，观察是否能在重试窗口内自愈，`results_TA_init.txt`/`results_teacher_init.txt` 是否不再是 0 字节。

---

## [2026-07-13] TA/Teacher 全程 context overflow / 生成不了回复——根因：compaction.reserveTokens 实际生效值 20000，跟官方默认 16384 不一致

**背景：** 2026-07-11 提交的 minitest（带 07-10 网关重试修复）在服务器上挂了近两天（`minitest_20260711_003159`），Ray 训练任务本身干净跑完（`Job succeeded`，`training.log` 里连续多次 `update_weights()` 无 OOM，确认内存修复持续有效），但外层编排脚本没能跑到 `check_convergence.py`（训练一结束网关就被 SIGTERM，`shutdown timed out`），手动用已有的 `results_*_all.txt` 补跑收敛检测：

```
Student : converged at session   3  (checked 228 sessions)
TA      : NOT converged  (checked 228 sessions)
Teacher : NOT converged  (checked 228 sessions)
```

**排查：** `results_TA_all.txt` 全部 228 条记录，`--verbose` 显示 `consec` 从未超过 0——查看原始内容发现全部是错误占位文本（`⚠️ Agent couldn't generate a response`／`Context overflow: prompt too large for the model`），不是真实模型输出。查具体 session（`ta-grade-0-198647`）在 `openclaw.log` 里的 `[context-overflow-precheck]` 日志：

```
estimatedPromptTokens=13611  promptBudgetBeforeReserve=12768  reserveTokens=20000  overflowTokens=843
```

`32768（总 context）- 20000（reserveTokens）= 12768（剩给 prompt 的预算）`，TA 批改任务的 prompt 稳定在 13.6K 左右，超预算约 843 token（约 6.6%），差一点点就撞上——不是"任务天然需要巨大 context"，是预算分配太紧。

**关键疑问排查（用户问：这个限制是不是也像 SSRF 那次一样是最近两个月新加的）：** 查了本地 `openclaw` 源码（`src/agents/sessions/settings-manager.ts:721`）：`getCompactionReserveTokens(): number { return this.settings.compaction?.reserveTokens ?? 16384; }`——**官方代码默认值是 16384，不是 20000**。查 `CHANGELOG.md`，`preemptive overflow precheck`/`midTurnPrecheck` 机制可追溯到 **2026.4.29**，跟论文写插件的时间（约 2026.3-4 月）同期甚至更早——**不是像 SSRF 那次"最近两个月新加的限制"，precheck 机制本身一直都在**。真正的问题是我们环境里这个值不知为何被设成了 20000，不是官方默认，也不是我们自己的脚本设的（`~/.openclaw/openclaw.json` 里 grep 不到 `reserveTokens`/`compaction`，`openclaw config get agents.defaults.compaction` 显示路径未设置）——来源未查清，可能是运行时按 thinking 模式动态计算，不是简单读配置。

**修复：** 不再深究 20000 具体从哪算出来的，直接在三个训练脚本的网关启动阶段显式设置 `openclaw config set agents.defaults.compaction.reserveTokens 16384`（跟 `chatCompletions.enabled` 那个强制设置是同一个模式），把它拉回官方默认值——论文本身没有理由用非默认配置，16384 应该就是论文实际用的值。已加入 `smoke_train_with_services.sh`/`minitest_train_with_services.sh`/`train_with_services.sh` 三脚本，commit（待补）。

**待验证（已在 smoke 上确认修复生效，见下）：** ~~下次提交 minitest，确认...~~

**更新（2026-07-13，smoke 验证）：** `smoke_20260713_110306` 确认修复生效——`openclaw.log` 里 `[verify] agents.defaults.compaction.reserveTokens = 16384`；`results_TA_smoke.txt` 里 TA 第一次产生真实回复（`"The solution section is empty. The correct answer is 36."`），不再是错误占位文本。TA 后续 turn 4 又失败了，但那是另一个独立问题（网关被提前 SIGTERM，见下一条），不是 context overflow 复发。minitest/8GPU 上仍建议留意确认，但根因已确认解决，风险很低。

---

## [2026-07-13] smoke 训练一结束就立刻杀网关，没等 INIT 模拟循环跑完——TA 最后一轮被打断（已记录，暂不修）

**背景：** 验证上一条 `reserveTokens=16384` 修复时，用 smoke 快速测试（`smoke_20260713_110306`），确认修复生效（TA 前 3 轮都是真实回复），但 TA 第 4 轮（也是 INIT 流程最后一轮，写入批改评论）撞上新问题。

**现象：** `simulation.log` 显示 TA turn 4/4 先是 `408 Client Error: Request Timeout`，内部重试 3 次（1s/2s/4s backoff）全部失败，`TA_chat.py` 抛未捕获异常直接退出进程；`openclaw.log` 同一时间窗口显示网关收到 `SIGTERM`、开始关闭。

**根因：** smoke 的训练规模很小（只有 `perf 0`/`perf 1` 两步），训练进程比 INIT 阶段的 Student→TA→Teacher 顺序模拟更快跑完；外层脚本一见训练进程（`TRAINING_PID`）退出就立即执行清理、杀掉网关，**没有等 INIT 模拟循环也跑完**，正好在 TA 写最后一条评论时把网关杀了。这是脚本收尾时序上的一个真实 bug，跟 `reserveTokens`/context overflow 无关（TA 前 3 轮已经证明能正常生成，第 4 轮失败纯粹是被提前杀掉）。

**更正（2026-07-13，minitest 上也复现了）：** 原以为"minitest 训练耗时远超 INIT，不容易撞上"，`minitest_20260713_112908` 证明这个判断错了——这次训练在 `perf 300`（`--num-rollout` 目标）达成后 6 秒内网关就被 SIGTERM，跟 smoke 是同一个 bug，只是触发时机不同：smoke 是训练规模小、比 INIT 先跑完；minitest 这次是训练**提前跑完了全部目标步数**（原因见下一条：checkpoint 复用导致训练几乎没跑新步数就"完成"），本质上都是"训练进程一退出就立刻清理，不管模拟循环还在不在跑"。

**处理：** 已记录，暂不修——问题本身没有直接损害数据有效性（网关关闭前的真实数据仍然有效），优先级仍然不高。如果后续 8GPU 正式训练也复现（训练进程提前退出、模拟循环没跑完就被打断），需要修：让清理逻辑等 `simulation_loop`/`SIM_LOOP_PID` 也退出再杀网关，而不是只等 `TRAINING_PID`。

---

## [2026-07-13] minitest 复用了旧 checkpoint，"续训"到接近目标直接完成，本次验证结果无效

**背景：** 排查 `minitest_20260713_112908`为什么能在 7 分钟内就跑到 `perf 300`（不合理——A800 实测一步要 5-6 分钟）。

**根因：** `run_openclaw_topk_select_modelfactory.sh` 里我们自己加的断点续训补丁（`--load "${SAVE_CKPT}"`，07-08 条目）会在 `SAVE_CKPT` 目录已有 checkpoint 时自动续训。但 **minitest 每次都写到同一个固定路径**（`/dfs/data/openclaw-rl-project/checkpoints/minitest-qwen3-4b-openclaw-topk-select`），查该目录 `latest_checkpointed_iteration.txt` = 299，保存时间 `Jul 13 01:56`，正好对应上一次跑（`minitest_20260711_003159`，跑了两天才到 299 步）。这次新提交的 minitest 自动接上了这个几乎跑满的 checkpoint，几乎没做任何新训练就达成 `--num-rollout 300` 目标，7 分钟就"完成"了。

**影响：** 这次 minitest 的"训练 300 步无 OOM"结论无效（不是真跑的）；TA/Teacher 也没有意义（7 分钟内 INIT 还没轮到 TA 就被收尾逻辑打断，见上一条）。内存 OOM 修复的有效性仍以 07-10 A800 那次连续 10 次全新 `update_weights()` 的实测为准，不受影响；`reserveTokens` 修复的有效性以同一天 smoke 的结果为准，不受影响。

**这是一个通用陷阱**：断点续训对 8GPU 正式训练是有意设计的功能（防止意外中断丢进度），但对 minitest 这种"每次都要验证一遍完整流水线"的场景是有害的——一旦某次 minitest 的 checkpoint 攒够了 `--num-rollout` 步数，之后所有指向同一路径的 minitest 都会"续训到快完成直接结束"，失去验证意义，且不会有任何报错提示这个情况，容易被误判为"这次跑通过了"。

**处理：** 清空 minitest 专用的 checkpoint 目录后重新提交：
```bash
rm -rf /dfs/data/openclaw-rl-project/checkpoints/minitest-qwen3-4b-openclaw-topk-select
```
清空后 `--load` 找不到有效 checkpoint 会自动回退到 `--ref-load` 从预训练权重重新开始（断点续训补丁本身的设计行为）。不影响 8GPU 正式训练的 checkpoint（路径不同）。当前选择手动清空而不是改脚本让每次 minitest 用独立路径——如果以后经常忘记清空导致重复踩坑，再考虑把 `SAVE_CKPT` 绑定到 minitest 各自的 `LOGS_DIR`。

**待验证：** 清空 checkpoint 后重新提交的 minitest，确认能不能观察到真正从头开始的 INIT + Joint round + 训练全过程。

---

## [2026-07-13] wandb.ai 连不上——需要代理，且必须用 bash -i 提交才能让 pon 生效

**背景：** 开了 `USE_WANDB=1` 之后 wandb 一直没有在网页上出现对应的 run，怀疑是 wandb.ai 被墙、需要走代理（这个 modelfactory 环境访问境外服务通常要过内部代理）。

**排查过程：**
1. 一开始尝试在训练脚本开头加 `sing-box.sh start && source ~/.bashrc && pon`，跑起来先是报 `/root/.bashrc: line 6: PS1: unbound variable`——脚本用 `set -euo pipefail`，`.bashrc` 里判断交互式 shell 的写法引用了未设置的 `$PS1`，在非交互式脚本里直接触发 `set -u` 报错，`source` 提前中断，后面 sing-box 加进 `.bashrc` 的代理配置根本没机会执行。
2. 用 `set +u; source ~/.bashrc || true; set -u` 绕过后，`PS1` 报错消失，但换成 `pon: command not found`——查 `/root/.local/bin/`、`.bashrc` 最后几十行都找不到 `pon` 相关内容。
3. 用户提供了同事的 `start_tools.sh`（同样的 `sing-box.sh start` + `source ~/.bashrc` + `pon` 三步）和对应的 modelfactory 提交方式：`代码解释器` 填 `/bin/bash -i /dfs/data/start_tools.sh && /bin/bash -i`（注意 `-i`）。这才定位到根本原因：**`pon` 是 `.bashrc`/`.bash_aliases` 里定义的 alias，bash 只在交互式模式下才展开 alias**，训练脚本用 `bash script.sh` 起的是非交互式 shell，`.bashrc` 哪怕跑完了，`pon` 这个 alias 也不会被识别成命令。这跟第 1 步的 `PS1` 报错是**同一个根因的两个症状**——真正交互式 shell 里 bash 会自动给 `$PS1` 设默认值，根本不会触发那个 unbound variable 错误。

**中间尝试过的绕过方案（已废弃）：** 不依赖 `pon`，从 `sing-box.sh start` 自己的输出里 `grep` 出监听端口，直接 `export http_proxy`/`https_proxy`，配合重试的连通性检查。这个方案本身是可行的，但既然团队已经有标准化的 `start_tools.sh + bash -i` 提交方式，改用这个更统一、不用维护自己的一套代理解析逻辑。

**修复：** 三个训练脚本移除内置的代理处理逻辑，改为要求提交时用 `代码解释器 = /bin/bash -i /dfs/data/start_tools.sh && /bin/bash -i`（`start_tools.sh` 路径：`/dfs/data/start_tools.sh`，同事提供，内容是 `sing-box.sh start` + `source ~/.bashrc` + `pon`，失败即 `exit 1`）。脚本头部注释同步更新提交说明。commit `89a27b4`。

**⚠️ 安全提醒（已处理）：** 排查过程中用户贴出的 `~/.bashrc` 内容里包含了明文 `WANDB_API_KEY`，已提醒用户去 wandb 网站撤销重新生成。

**注意：** 这种提交方式下代理失败会直接 `exit 1` 中断整个训练 job（`start_tools.sh` 自己的设计，不是"警告后继续"），跟之前脚本内建的"代理失败不阻断训练"的处理哲学不一样——如果代理经常不稳定导致训练 job 频繁提交失败，需要回头评估要不要恢复脚本内的非致命兜底逻辑。

**待验证：** 用新的提交方式重新跑一次 smoke/minitest，确认 wandb 项目 `openclaw_rl` 里能看到对应 run。

**更新（同一天，验证成功但暴露新的安全问题）：** 新提交方式确认有效，wandb run 正常出现（`qwen3-4b-openclaw-topk-select_kl2fceaf-RANK_0`）。用户为了让我能查看图表，把 `openclaw_rl` 项目设成了 Public。查看 run 的 Overview 页面时发现 **"Command" 字段（wandb 自动记录的启动命令）里明文包含了 `--wandb-key <API_KEY>`**——项目公开后这个字段任何人不用登录都能看到，key 第二次暴露（第一次是在对话里贴 `.bashrc`）。

---

## [2026-07-13] wandb key 从 CLI 参数暴露在 Command 字段——改走环境变量

**背景：** 上一条发现 wandb run 的 "Command" 字段公开可见，且里面有明文 API key。

**根因：** 官方 `run_qwen3_4b_openclaw_topk_select.sh` 把 `--wandb-key ${WANDB_KEY_VALUE}` 作为命令行参数传给 `train_async.py`；wandb SDK 会自动把完整的 `sys.argv` 记录到 run 的 "Command" 字段用于可复现性——这是 wandb 的标准行为，不是 bug，但意味着任何用 CLI flag 传的敏感值都会被公开记录，且**没有"只隐藏这个字段"的官方开关**（查过 wandb 社区确认）。

**修复：** 查 `slime/utils/wandb_utils.py:40` 确认 `args.wandb_key is None` 时会跳过显式 `wandb.login()`，后续 `wandb.init()` 自己会读 `WANDB_API_KEY` 环境变量完成认证，不影响功能。改法：
1. 在 `run_openclaw_topk_select_modelfactory.sh` 的 `RUNTIME_ENV_JSON.env_vars` 里加 `WANDB_API_KEY`（Ray runtime env 传递，不是 CLI 参数）
2. 从 `WANDB_ARGS` 里去掉 `--wandb-key ${WANDB_KEY_VALUE}` 这一行

本地测试两处字符串替换均正确匹配官方脚本原文、patch 后 bash 语法通过。commit `781b602`。

**用户需要手动做的两件事（不是脚本能解决的）：**
1. 把 `openclaw_rl` 项目 visibility 改回 Team/Private，直到确认新 run 的 Command 字段不再有 key
2. 去 wandb 网站撤销当前这个已经暴露两次的 key，重新生成一个新的

**待验证：** 下次提交后检查新 run 的 "Command" 字段确认不再包含 `--wandb-key`，且 wandb 登录/上报依然正常（验证 `WANDB_API_KEY` 环境变量兜底生效）。

---

## [2026-07-14] 8GPU 正式训练首次尝试：update_weights() 时 GPU 显存 calloc 失败——根因是 workspace 残留僵尸进程，非参数问题

**背景：** 8GPU H20 正式训练（`train_with_services.sh`）第一次真正提交（跳过了完整 minitest 验证，直接上 8GPU，见前一天讨论），提交前有过一次手动 Ctrl-C 中断。重新提交后训练正常起步（Ray 42 进程、gateway 就绪、INIT 阶段开始），第一次 `update_weights()` 时失败。

**报错：**
```
(SGLangEngine) Failed to CUDA calloc 268435456 bytes. The full weights of the ModelRunner are partially updated. Please discard the whole weights.
(SGLangEngine) [TP0] Failed to update parameter online: NCCL error in: .../NCCLUtils.cpp:94, unhandled cuda error
(SGLangEngine) ncclUnhandledCudaError: Call to CUDA function failed.
(SGLangEngine) "POST /update_weights_from_distributed HTTP/1.1" 400 Bad Request
[Rank 3] Some NCCL operations have failed or timed out...
```

**排查：** GPU 显存报错（256MB calloc 失败）+ 后续 NCCL 通信卡死超时，看起来像是 `--sglang-mem-fraction-static 0.8`（官方默认值，未改动）在这次 `--rollout-num-gpus-per-engine 2`（rollout 引擎自己也做 2 卡 TP，minitest/smoke 从未测过这个场景，一直用的是单卡引擎）配置下显存余量不够。但下结论前先用 `nvidia-smi` 排查了一遍——**真实原因是之前那次 Ctrl-C 中断后，`trap cleanup` 没能杀干净 SGLang 进程**，3 个残留的 `sglang::scheduler`/`sglang::scheduler_TP0`/`sglang::scheduler_TP1` 进程分别占着 GPU 4/5/6 各 80GB+ 显存（对应 `--sglang-mem-fraction-static 0.8` 的静态预留量），这次新训练的进程被分配到部分重叠的 GPU 上，实际可用显存远低于预期，NCCL 权重广播需要的临时缓冲区分配失败。

**根因确认：** `kill -9` 杀掉三个残留 PID 后 `nvidia-smi` 确认 8 张卡全部恢复空闲（4MiB），不是 `sglang-mem-fraction-static` 参数问题，是 **workspace 模式下手动 Ctrl-C 中断的清理不彻底**——这是持久化 workspace 相比一次性 job 容器的一个新风险点（跟同一天早些时候发现的"reserveTokens 没生效是因为 workspace 里残留网关进程"是同一类问题：workspace 状态会在多次尝试之间残留，job 容器每次是全新的）。

**处理：** 手动 `kill -9` 清理残留进程后确认 GPU 全部空闲，重新提交。

**经验（记录下来，避免以后重犯）：** 在 workspace 模式下（跟提交 job 不一样），每次训练中断/失败后重新提交前，都要先 `nvidia-smi` 确认 8 张卡显存干净、`ps aux | grep "openclaw gateway"` 确认没有残留网关进程，两者都要检查，不能想当然认为环境是干净的。

**待验证：** 清理后重新提交的 8GPU 训练，确认 `update_weights()` 能正常完成，不再复现这个 CUDA calloc 失败。

---

## [2026-07-14] 8GPU 正式训练 INIT 阶段三个角色全部失败（0 字节）——根因尚未定位，已排除多个假设

**背景：** 8GPU H20 正式训练（`train_with_services.sh`），残留进程清理后重新提交，INIT 阶段（Student→TA→Teacher 各 72 题）全部以 0 字节告终：`results_student_init.txt`（mtime 11:41:12）、`results_TA_init.txt`（11:49:45）、`results_teacher_init.txt`（12:00:53），三个文件均为空。`openclaw.log` 显示 11:40-12:00 这段时间大量 `[agent/embedded] ... error=LLM request failed. rawError=503 status code (no body)`。

**已排除的假设（按顺序，均有实测证据支持排除）：**
1. ~~网关被训练进程抢占资源~~——`nvidia-smi`/进程状态显示网关本身运行正常，且同一窗口内 Student 早期（10:32）有过真实成功对话
2. ~~SGLang `pause_generation` 暂停时间过长~~——查了 11:38-12:08 全部 `update_weights` 事件，每次 `pause_generation`→`continue_generation` 都在 1-2 秒内完成，跟论文"不干扰推理"的设计描述一致
3. ~~SGLang 引擎容量不够/被挤爆~~——同一时间窗口 `token usage` 只有 1-2%，`#queue-req: 0`，引擎明显不忙（但后来发现可能查的不是 Student 实际请求经过的那个 SGLang 进程，这条本身也需要重新确认，见下）
4. ~~`submission_enabled` 因为 checkpoint 保存/eval 被拖长暂停窗口~~——检查过 13:46-14:18 那次 31 分钟的训练步间隙，checkpoint 目录当时根本不存在（还没到 save-interval），eval 也没有触发记录；且那次间隙期间 `submission` 全程是开着的、模拟脚本日志里也没有报错——这 31 分钟纯粹是"没攒够真实样本"的正常等待，跟 11:40-12:00 那次 503 风暴是两个不相关的现象，之前误认为是同一件事

**排查中发现的一个更深层次问题（架构层面，独立于上面的 503 根因排查）：** `run_init_phase()` 和 `run_joint_round()` 调用 `student_chat.py`/`TA_chat.py`/`teacher_chat.py` 的方式完全一样（只有题目数量不同），**没有任何机制告诉底层训练管道"当前是 INIT 阶段，还不该把这些对话当训练样本消费"**。实测证据：第一次 `update_weights()` 发生在 11:38:17，此时 Student 的 INIT 还在进行（直到 11:41:12 才结束）——说明训练循环在 INIT 尚未完成时就已经开始拿 INIT 产生的对话当训练样本、并据此更新了策略权重。这意味着 `homework1`（Student 解题产出）可能不是论文描述的"单一策略版本生成的冻结快照"，而是**由多个不同权重版本的策略混合生成的**——TA 后续基于这个不一致的 homework1 做批改，跟论文"先冻结 homework1 再开始联合优化"的设计顺序不符。这个问题不会导致训练报错/崩溃，但可能影响训练数据的正确性，需要单独评估要不要修、怎么修。

**待查：**
1. 官方论文/代码库对"INIT 阶段该不该跟训练并发、homework1 该不该是单一策略快照"这件事是怎么处理的——`train_with_services.sh`（我们自己写的编排脚本）目前是训练 job 提交后立刻并发跑 INIT，这个时序选择是不是我们自己的设计缺陷，官方有没有对应的参考做法
2. 11:40-12:00 那次 503 风暴的真正根因——需要重新确认 Student 实际请求经过的是哪个 SGLang 进程（这次跑起了 `pid=34571`/`pid=36012` 两个 SGLang 进程，之前查健康状态时不确定看的是不是对的那个），精确到秒对比失败时刻和 SGLang/`submission_enabled` 状态

**更新（同一天，待查 1 已确认）：** 用官方 `openclaw-test/README.md` 的 Step-by-Step 流程核实——官方参考流程本身就是训练 job 先提交、Student/TA/Teacher 后跑，边跑边训练，`_maybe_submit_ready_samples()`（`openclaw_combine_api_server.py:151-213`）完全不区分 INIT/joint 阶段，谁的对话先评估完就先入训练队列，没有等 homework1/2 建好才开始训练的逻辑。**结论：homework1 由多个策略版本混合生成是官方设计的常态，不是 bug**，`train_with_services.sh` 里"INIT 全部跑完才进 Joint round"是我们自己额外加的更严格顺序，不是必需的，但也不算错，不需要改。这一条排查到此结束。

**更新（同一天，待查 2 有新进展，修了一个真实设计缺陷）：** 排查过程中发现 `run_one_persona()` 的外层重试（07-10 加的，网关中途不可达时最多重试 3 次）有一个副作用——`student_chat.py`/`TA_chat.py`/`teacher_chat.py` 每次被重新调用都会**清空自己的输出文件重新开始**，如果第一次已经做完一部分真实题目才失败，外层重跑会把这部分已经拿到的真实数据一起清空覆盖掉，比"只跑一次不重试"保留的数据反而更少；而且 Student 的 INIT 从约 10:32（网关就绪）持续到 11:41:12 才彻底放弃，耗时超过一小时，对于"72 题、最多重试 3 次"这种设计来说明显偏长，符合"多次整体重来"的迹象。核实官方 `openclaw-test/README.md` 参考流程本身也是每个角色只跑一次，没有整体重来的机制。**修复：`run_one_persona()` 改回只调用一次**（保留 07-10 那次"跑之前先确认网关可达"的前置检查，去掉"失败重跑整个脚本"），commit `0b25005`。

**待验证：** 用这个修复重新提交 8GPU 训练，确认 INIT 阶段 `results_*_init.txt` 是否不再是 0 字节（哪怕不能 100% 跑完 72 题，至少应该保留部分真实进度，不会被重跑清空）。

**更新（同一天，发现 Joint round 的"重复小批次循环"结构本身也没有官方依据）：** 排查过程中用户指出既然"INIT 全部跑完才进 Joint round"这道等待官方没有要求，我们也不需要照搬——于是回头查了一下"Joint round"这个循环本身是不是官方设计。查证结果（只在论文期允许目录内查找，`openclaw-tinker`/`openclaw-fireworks`/`openclaw-rl/oel` 均未涉及）：

1. `student_chat.py`/`TA_chat.py`/`teacher_chat.py`（`openclaw-test/`）的 `argparse` 只有 `--dataset`/`--num-problems`/`--max-turns`/`--max-retries`/`--output`，**没有** `--loop`/`--continuous`/`--num-rounds` 这类参数，`main()` 就是 `for i in range(count)` 处理完指定数量题目就退出进程——脚本设计上是**一次性**的，不是可以反复调用小批次的常驻服务
2. `openclaw-combine/`、`openclaw-opd/` 目录下搜索 `student_chat|TA_chat|teacher_chat` **零命中**——没有任何官方脚本组织"INIT + 反复循环跑 Joint round"这套编排
3. `openclaw-test/README.md` 唯一提到 joint 的原话："The test consists of three sequential phases (**you can also run them together**, but you need to obtain homework1 and homework2 first if you want to try this joint optimization)"——只要求"先有 homework1/2"，没有"之后循环跑很多轮小批次"这个说法

**结论：`train_with_services.sh` 里 `run_joint_round()` 那个"每轮 6 题、`while kill -0 TRAINING_PID` 反复循环直到训练结束"的结构，是我们自己发明的，官方代码里找不到对应实现可以直接抄，也没有文档依据。**最符合官方"three sequential phases...run them together"字面意思、也最符合三个脚本本身"一次性处理完给定题目退出"设计的实现，应该是：INIT（Student→TA→Teacher 依次各跑一次全部 72 题，建好 homework1/2，这部分不变，官方明确要求）之后，Joint 阶段是**三个脚本各自传完整题目数、同时后台启动一次**，不是分成一轮一轮的小批次反复循环。

**待确认后再动手：** 用户要求先确认清楚这个结论没有跟之前确认过的其他事实冲突（已核对 `paper_understanding.md` 547-558 行，不冲突），并且要确认官方是不是真的没有可以直接复用的编排代码（不是自己瞎写），确认完再决定要不要改、怎么改。

**更新（同一天，已确认无冲突、无官方代码可复用，已实施修复）：** 扩大范围搜遍论文期允许的全部目录（仓库根目录、`openclaw-test/` 完整文件列表、`Megatron-LM/`、`slime/`、全部允许目录的 `.md` 文档）确认没有遗漏的官方编排脚本；`openclaw-test/README.md` 原话"three sequential phases (you can also run them together, but you need homework1/2 first)"本身就说明联合运行是留给用户自己组织的，官方没有配套自动化编排。同时核对了论文原文逐字引用（`paper_index.md` p.21 Appendix A.1，与之前理解一致，无新信息）。

**修复：** `train_with_services.sh`/`minitest_train_with_services.sh` 的 `run_joint_round()`（每轮 6 题、`while kill -0 TRAINING_PID` 反复循环）改为 `run_joint_phase()`——INIT 建好 homework1/2 后，三角色各自传 `JOINT_NUM_PROBLEMS=1319`（GSM8K 全量数据集）、同时后台启动一次，训练循环自然消耗真实样本直到自己结束（num-rollout 跑完或被手动停止），模拟进程跟着训练进程一起收尾（谁先结束就收尾，避免互相干等）。`smoke_train_with_services.sh` 本来就只做 1 轮 Joint 验证并发，不受影响。commit `4be24ab`。

**待验证：** 用这个新设计重新提交 8GPU 训练，确认 INIT 阶段能正常建好 homework1/2，Joint 阶段三角色能持续产出真实数据、训练能正常推进，最终 `check_convergence.py` 能拿到有意义的 Table 3 收敛数字。

---

## [2026-07-15] 8GPU 正式训练（run `8yn4i8ml`）跑了约 7 小时后无声消失——根因已确认：TA 生成失败链式导致 rollout 饥饿，GPU 空闲触发平台自动回收

**背景：** 用 07-14 修复过的新版本（`run_one_persona()` 单次调用 + `run_joint_phase()` 一次性并发）重新提交的 8GPU 正式训练，16:07:25 开始，Student INIT 阶段这次验证正常（进度推进到 43/72 题，产出真实完整数据，`run_one_persona()` 修复确认生效，见 `work_log.md` 2026-07-14 条目）。但训练在 23:12:39 前后彻底停止响应，wandb 上该 run 状态变为 `Crashed`。

**排查过程（按顺序，每一步都有实测证据）：**
1. 确认不是磁盘写满（`df -h /dfs/data` 仅用 6%）、不是系统 OOM（`dmesg` 里的 OOM 记录是 7/9 的旧记录，进程名 `cicc`/`VLLM::EngineCor`，跟本次训练无关）、不是 workspace 会话断连（用户确认未断连）
2. `training.log` 结尾显示：22:22-23:12 这 50 分钟里，`[OpenClawCombineSelectWorker] waiting for combine samples: 11/16, queue=0` 反复打印、样本数字卡住不再增长，SGLang 健康检查全程 200 OK；23:12:39 后彻底没有任何输出，无报错、无 Traceback、无 NCCL 错误
3. `grep error/exception` 在故障时刻附近无真实匹配（只有启动阶段配置项里带"error"字样的参数名，属误报）
4. 查 `simulation.log`，第一次出现 `⚠️ Agent couldn't generate a response. Please try again.` 早在 Student INIT 第 1 题（Turn 3，请求内容是"append that to homework/1.txt"，重试一次后 Turn 4 即成功）——说明这个故障从训练一开始就存在，不是 Joint 阶段才出现的新问题
5. `openclaw.log` 确认这个报错的技术信号是 `stopReason=length` + `[model-fallback/decision] decision=candidate_failed reason=format`——模型生成回复过程中，还没输出完整、可解析的结果就把输出长度预算耗尽，被判定失败，跟"文件读/写"这个动作本身无关（Student 那次是写入请求、TA 那次是读取+生成评语请求，共同点更可能是"turn 内需要先完成一次工具调用"，而不是具体是读还是写）
6. 按小时统计 `openclaw.log` 里这个报错的出现次数：16点 21次，17-21点稳定在 70-96次/小时（撑了 5 小时以上系统仍正常运作，说明这个故障本身是可以被重试机制扛住的，不是它直接杀死训练），22点骤降到 17次，23点归零——不是故障变少了，是系统 22点后基本停止发起新请求
7. `simulation.log` 显示：22点左右 TA 在批改第 23 题时连续 8 轮全部命中这个故障，`Reached max turns (8) for problem 23`，随后第 24 题 Turn 1、Turn 2 也连续失败——TA 这个角色被这个故障"钉住"了，产不出新的合格批改样本
8. wandb Logs 面板（`train_async.py` 进程自己独立上报的 stdout，跟本地 `training.log` 是完全不同的两条日志链路）拉到最后确认：**最后一条日志同样精确停在 23:12:39**，内容也是普通的健康检查 `GET /health 200 OK`，跟本地 `training.log` 完全吻合。两条独立链路精确同时归零，且都没有任何报错/退出信息，排除了"进程内部报错/崩溃"（那样至少会有一条链路先记录到异常），指向"整个容器从外部被一次性杀掉"

**根因确认：** TA 反复遇到 `stopReason=length` 生成失败（这个故障本身从训练一开始就存在，是个长期慢性问题）→ 22点左右在批改第 23、24 题时命中率突然变得极高，TA 用完所有重试机会仍产不出合格样本 → `RolloutManager` 拿不到新的合格样本，卡在 `waiting for combine samples: 11/16` 长达 50 分钟不动 → 这段时间里 SGLang 引擎除了应答健康检查，没有真正的生成任务在跑，8 张 GPU 实际处于空闲状态 → GPU 空闲持续超过 modelfactory 平台的自动回收阈值（约 1 小时）→ 平台强制关闭整个 workspace → 容器内所有进程（本地日志写入 + wandb 上报）瞬间同时终止，不留报错痕迹。**不是代码 bug、不是 OOM、不是磁盘问题，是"训练进程仍在但 GPU 真实空闲"触发了平台层面的资源回收。**

**更新（同一天，子问题已查清并修复）：** 直接查 TA 那次 INIT 运行的原始输出文件 `results_TA_init.txt`（不是靠 `simulation.log` 片段推测）：**从第 0 题（`session: ta-grade-0-227040`）到崩溃前的第 37 题，每一题都是唯一一行 `⚠️ Agent couldn't generate a response. Please try again.`，无一例外，100% 失败率。** 这推翻了"TA 在第 23/24 题才开始集中爆发"的说法——TA 从 INIT 阶段第一题起就没有成功过一次，是从头到尾彻底失效，跟 Student 同期基本正常（43 题里只有 1 次可重试的失败）形成鲜明对比。之前按小时统计出来的"稳定 70-96 次/小时"，绝大部分应该就是 TA 在不停重试失败，不是 Student/TA 均摊的随机噪音。

**根因定位（对照官方配置发现的真实原因）：** 查官方 `README.md`（[第 312-344 行](../../OpenClaw-RL-official/README.md)，"Slime-based RL server" 配置示例）给出的标准 OpenClaw 模型 provider 配置是 `contextWindow: 32768` 配 `maxTokens: 8192`。我们三个脚本（`train_with_services.sh`/`minitest_train_with_services.sh`/`smoke_train_with_services.sh`）实际用的是 `maxTokens: 4096`——只有官方值的一半。追溯这个 4096 的来历：最早是 07-07 那次修复引入的，当时 smoke 的 `contextWindow` 被临时缩到 8192（省显存），`maxTokens=4096` 是"设成 contextWindow 一半、给 prompt 留空间"这个逻辑下算出来的，在那个场景里合理；但后来（07-09）`contextWindow` 已经改回官方的 32768，`maxTokens` 却没有跟着重新计算，就这样原封不动地被复制进了 minitest 和 8GPU 正式训练脚本，形成了"`contextWindow=32768` 配 `maxTokens=4096`"这个不匹配组合，一直没人发现。TA 的批改任务比 Student 复杂得多（先要读文件工具调用，再要求生成结构化多点评语，Qwen3-Thinking 生成前还会先输出一大段 `<think>` 推理），系统性地更容易撞到这个偏小的输出预算触底；Student 的任务通常更短，大多数时候能在 4096 内说完，所以基本没事——这个不对称完全解释了"Student 正常、TA 100% 失败"的现象。

**修复：** 三个脚本里的 `maxTokens` 从 `4096` 改为跟官方一致的 `8192`（`contextWindow` 不变，仍是 32768）。`train_with_services.sh`、`minitest_train_with_services.sh`、`smoke_train_with_services.sh` 均已修改。

**待验证：** 同步跑的 minitest 用这个修复验证 TA 能否正常产出真实批改结果（不再 100% 命中 `stopReason=length`）；确认后再重新提交 8GPU 正式训练，同时观察是否还需要处理"处理方向"里提到的另外两点（`waiting for combine samples` 超时报警、GPU keepalive 兜底）——如果这次 maxTokens 修复能让 TA 恢复正常，rollout 饥饿的根源就解决了，那两个可能就不再是必须做的事，视 minitest 结果决定。

**更新（2026-07-15，maxTokens 修复不够，找到真正根因）：** 用改完 `maxTokens=8192` 的版本重新提交 8GPU 训练（run `8yn4i8ml` 之后的新 run，log 目录 `20260715_130612`），仍然反复失败：

1. Student INIT **只跑到第 35 题就崩溃**（`results_student_init.txt` 只有 36 个 session，非 72），TA INIT 也**只跑到第 11 题左右**（同一个流水线里两个角色都提前中断）——`run_init_phase()`/`run_one_persona()` 在任一角色失败时只打印警告就放行、不阻塞，导致"INIT 完成"这个标志本质上只代表"三个角色的脚本各自被调用过一次"，不代表真的跑完了全部 72 题。Teacher 的 `ensure_homework_dir()` 会在这个不完整的 `homework1/` 基础上直接复制出 `homework2/`，导致 Teacher 面对大部分题目时根本没有 TA 评语可用
2. 逐条排查失败原因：Student 报 `requests.exceptions.ReadTimeout`（客户端自己 180 秒等超时，服务端根本没回复），TA 报服务端明确返回的 `408`——看起来像两种不同的超时，但深挖 `openclaw.log` 发现两边都是同一件事：`[agent/embedded][context-overfow-precheck]` 反复报 `Context overflow: prompt too large for the model (precheck)`，卡在这个 precheck 失败重试循环里空转 2 分钟左右，最后才被包装成"LLM request timed out"抛出来——**表面是超时，真正根因是 context overflow precheck 循环**，不是生成变慢
3. 关键异常：precheck 日志里 `reserveTokens=20000 effectiveReserveTokens=20000`，跟脚本里设置的 `16384` 完全对不上；核实 `openclaw config get`/`~/.openclaw/openclaw.json` 都确认配置文件里明明白白是 `16384`，且当时只有一个网关进程在跑（排除残留进程用旧配置的可能）
4. 对比昨晚崩溃的 run（`maxTokens` 还是旧的 4096）的 `openclaw.log`，发现 `effectiveReserveTokens=20000` **早就存在**，不是这次 maxTokens 改动引入的新问题——说明 07-13 那次"改配置为 16384"的修复从一开始就没有真正在这条 OpenAI 兼容请求路径上生效过，配置层面看着对，实际运行时一直在用 20000
5. WebSearch 官方 GitHub 找到确切原因：[Issue #66830](https://github.com/openclaw/openclaw/issues/66830)（"reserveTokens vs reserveTokensFloor asymmetry"，不分 provider/模型都会复现）——**`memoryFlush`/`preflight` 这两条阈值计算路径根本不读 `reserveTokens` 字段，读的是另一个从未配置过的 `reserveTokensFloor`**，没设置就用 OpenClaw 自己的内部默认值。（另外查到一个高度相似但已 Closed 的 [Issue #65465](https://github.com/openclaw/openclaw/issues/65465)，一开始误判为同一个根因，但那次是 Ollama provider 专属且已在我们当前版本 2026.6.9 之前修复完，逻辑上不该是这次问题的原因，已排除，改用 #66830 这个更贴合"不分模型都复现"描述的版本）
6. `openclaw config get agents.defaults.compaction.reserveTokensFloor` 确认这个字段**从未配置过**（"Config path not found"）；手动 `openclaw config set agents.defaults.compaction.reserveTokensFloor 16384` 后 `config get agents.defaults.compaction` 显示两个字段都是 16384，确认设置本身有效

**修复：** 三个脚本（`train_with_services.sh`/`minitest_train_with_services.sh`/`smoke_train_with_services.sh`）在原有 `reserveTokens 16384` 设置之后，追加 `openclaw config set agents.defaults.compaction.reserveTokensFloor 16384`（每次启动网关前都强制设置 + 回读验证，跟 `reserveTokens` 同一套模式）。

**未采纳的方案：** 升级 OpenClaw 到 2026.6.10——搜了官方 release notes 没有确认这个版本包含针对性修复，而且当前整套流水线是针对 2026.6.9 反复调好的，换版本风险大于收益，先不做。

**待验证：** 用这个修复重新提交一次干净的 8GPU 训练（当前跑到一半、数据已不完整的这次直接放弃重跑），确认 `effectiveReserveTokens` 终于变成 16384、Student/TA 能否完整跑完 72 题不再中途崩溃。

---

## [2026-07-16] 8GPU 正式训练（run `20260715_180549`，maxTokens+reserveTokensFloor 修复后）train/grad_norm 爆炸——根因确认：模型崩坏输出乱码字符，被当正常样本喂回训练形成雪崩

**背景：** 用 07-15 两个修复（`maxTokens=8192` + `reserveTokensFloor=16384`）后、64CPU/1024GB workspace 重新提交的 8GPU 正式训练（log 目录 `20260715_180549`）。Student INIT 阶段这次 72/72 全部跑完（过程中出现 121 次可自愈的 `stopReason=length` 重试，均恢复），但 TA 从第 0 题起 100% 失败（`⚠️ Agent couldn't generate a response. Please try again.`，连续多轮无一例外）。当晚 wandb `train/grad_norm` 从 step 0-12 稳定的 3-8 区间，在 step 13 首次跳变到 22.30，之后持续攀升，step 26 达到 1776、step 34 达到 2243，`train/train_rollout_logprob_abs_diff` 同步从 0.02 涨到 233+；训练最终因同样的 GPU 空闲超时被平台回收。这次的排查目标是回应一个更早的、更根本的问题："我们的训练数据管线、训练超参、GPU 布局都已核对过跟论文一致，那这个不稳定到底是我们环境里哪里跟论文验证过的环境不一样"。

**排查过程（按顺序，每一步都有实测证据）：**

1. **排除"陈旧化单独致因"和"批次同构单独致因"两个过早假设**：
   - 核对论文 Appendix A.1 对 Personal Agent 训练方法的原文描述（INIT 顺序一次性 + Joint 并行持续两阶段设计、`rollout_batch_size=16` 按 session turn 计数、`lr=1e-5`/`KL系数=0`/`w_RL=w_OPD=1.0` 等超参），确认我们 `train_with_services.sh` 的 `run_init_phase()`+`run_joint_phase()` 实现与论文描述的两阶段结构完全一致，INIT 用 72 题、Joint 用 GSM8K 全量 1319 题也都是论文文档里明确记录的数字，不是我们自己定的偏差——如果"INIT 阶段单一 persona 顺序跑、batch 同构"本身就会导致训坏，论文自己的 Table 3 结果不该收敛，所以这个结构差异被排除为唯一根因
   - 核实 `openclaw_opd_api_server.py:722-726` 的 `_handle_request()`：SGLang 返回非 200 状态时代理直接 `raise_for_status()` 抛异常，不会走到 `_submit_turn_sample`——证明"网关兜底文案 `couldn't generate a response` 被当训练样本提交"这个猜测是错的，硬失败（后端 500/超时）确实会被正确拦截，不会污染训练队列

2. **对齐 `train/grad_norm`/`train/train_rollout_logprob_abs_diff` 逐 step 数值与训练耗时**：发现 step 0-23 每步稳定 4-6 分钟，**step 24 起单步耗时暴涨到 15-29 分钟**（21:41→22:02→22:25→22:43→23:12...），说明从这附近开始"攒够 16 条 session turn"本身变得极慢——是生成侧卡住，不是 Megatron 训练变慢

3. **直接抓取 `_submit_turn_sample`/`_submit_rl_turn_sample` 提交日志（`[OpenClaw-Combine-Select] submitted ... sample`）逐条核对 index/response_len/reward**：在 index 363-399（对应 19:54-20:20，正好是 grad_norm 从 step 20 的 188.87 冲到 step 23 的 415.61 那个窗口），发现两种截然不同的畸形生成扎堆出现：
   - 近乎空的回复：`response_len=7~8` token，多个不同 session（`da498b53`/`34c3ea5c`/`6c895508`）反复出现，全部 `reward=-1.0`
   - 顶格跑满的超长回复：`response_len=8197`（正好卡在 `maxTokens=8192` 上限），同样全部 `reward=-1.0`/`0.0`，且每条要 1-3 分钟才生成完，直接解释了上一步发现的单步耗时暴涨

4. **最初误判"短回复=坏样本"是凭 response_len 推断，未看实际文本内容**——用户指出这个推断不严谨（可能是模型学会了简洁但有效的回复），于是回查 `openclaw_opd_api_server.py:742-750` 确认生成阶段本身有把 `content` 原文打进日志（`thinking=%d chars, response:\n%s`），据此抓取这几个 session 的实际文本：

   **确认这些短回复不是"简洁但有效"，而是模型输出崩坏**——`thinking=8~9 chars`（推理内容几乎是空的），正文只有孤零零一个字符 `𬣳`（一个跟教育对话场景毫无关系的生僻 CJK 扩展汉字）。更关键的是这个**完全相同的乱码字符跨越三个互不相关的 session 反复出现**：`da498b53` 从 turn 5 开始第一次出现，紧接着的下一个 session `34c3ea5c` 从 turn 1 起就直接吐这个字符，再下一个 session `6c895508` 除了继续吐这个字符，turn 7 起还出现另一种畸形（`thinking=8198 chars`，卡进 8197 token 顶格胡言乱语，两种畸形交替出现）。跨 session 复现同一个具体字符，说明这不是单次生成抽风，是**模型权重本身已经坏了**，且没有任何机制拦截，这些技术上"非空"（`response_text.strip()` 非空、`response_ids` 非空）但语义完全退化的样本被 `_submit_turn_sample` 正常提交进了训练队列。

**根因确认：** 模型在训练早期（grad_norm 首次跳变发生在 step 13，19:18:39；目前确认的乱码首次出现在 19:35:37 附近，晚于 step 13，说明还有更早的、尚未定位的初始触发点）某个节点开始输出退化内容（重复乱码字符 / 顶格跑满两种模式交替），**这些退化样本没有被任何环节过滤，被当作正常训练样本喂回了训练队列**。Personal Agent 用的简化版 GRPO 没有组内归一化（`A_t = r_t` 直接取 ±1/0），一个 16 条一批的训练 batch 里混进几条语义退化但同方向、且顶格超长（8197 token）的极端样本，聚合梯度自然远超正常范围，形成"退化样本 → 梯度爆炸 → 模型进一步退化 → 更多退化样本、生成更慢 → 攒批更久、陈旧化更严重"的自我强化雪崩，与 wandb 上 grad_norm 和 train_rollout_logprob_abs_diff 同步失控的曲线完全吻合。**"陈旧化"和"梯度爆炸"不是两个互斥的根因，是同一条因果链的下游表现，真正的根因是"退化生成没有被过滤就被喂进了训练队列"。**

**待查（根源尚未完全定位）：**
1. 乱码字符 `𬣳` 最早出现的确切时刻和触发它的具体样本——目前只确认了 19:35:37 这次，但 grad_norm 从 step 13（19:18:39）就已经开始跳变，中间还有近 17 分钟、约 4-5 个 step 的窗口没有查过，需要往前追这段时间提交的样本，找到真正意义上"第一次"退化发生的位置
2. 这个具体的乱码字符 `𬣳`（或者这一类退化模式）本身有没有已知成因——是否与 Qwen3-4B-Thinking 在某类数值不稳定状态下的已知 tokenizer/精度问题有关，还是纯粹由某次异常大的梯度更新触发的权重级坍塌，需要进一步查证（社区/已知 issue 层面）

**待做（解决方向，尚未实施）：** 需要在 RL 数据代理层（`_submit_turn_sample`/`_submit_rl_turn_sample` 提交前）加一道样本级过滤，拦截明显退化的生成（比如 response 顶格卡在 `max_tokens` 上限、或 `response_ids` 极短但内容跟上下文/任务要求明显不符），不让这类样本进入训练队列——具体过滤规则、加在哪个文件哪个函数，下一步继续设计。

**更新（同一天，找到乱码 `𬣳` 最早出现的精确时刻，定位到具体退化机制）：** 直接在 `training.log` 全文搜索这个具体乱码字符（而不是靠猜时间窗口），找到全文件**第一次出现**在 `19:29:58`，session `9778b1c5` 的 turn 15：`thinking=429 chars`（推理内容完全正常，不短），但最终 `content` 字段只有这一个孤立字符 `𬣳`。这比此前确认的"完全崩溃"案例（19:35 起，`da498b53`/`34c3ea5c`/`6c895508` 连推理本身都退化成 8-9 字符）早了约 5-6 分钟，且表现更轻微——说明这不是"模型瞬间整体崩了"，而是一个**渐进过程**：一开始只是"思考"到"最终答案"这个切换点偶尔采样出这一个具体怪字符（推理过程本身仍正常），几分钟后才恶化成连推理都退化、通篇只有这一个字符。

**可能机制（推断，未直接验证）：** `𬣳`（U+2C8F3，CJK 扩展生僻汉字）这类词表里几乎从未被训练数据覆盖过的 token，其 embedding 在预训练阶段可能本身就带有异常值；RL 微调中一旦偶然采样到它、产生哪怕很小的原始梯度，Adam 优化器基于这个 token 极小的二阶矩估计给出的自适应学习率会把这个小梯度放大成一次远超其他参数的有效更新——该 token 被采样概率进一步升高，下次更容易被采出来，形成自我强化的正反馈。这与观察到的"偶发（19:29:58 一次）→加剧（19:35 起持续）→全面崩溃（19:52 起 grad_norm 破百破千）"时间线吻合，且能解释为什么这个具体问题出现在我们这次跑的环境里——**这更像是训练过程中的一次随机事件（哪个生僻 token 的 embedding 恰好不稳定、什么时候被采样到），不是我们代码/配置的错，论文原始跑法大概率也有非零概率遇到同类问题，只是这次被我们的随机种子/具体样本序列触发了**。这个机制推断尚未通过直接检查模型 embedding 验证，仅基于时间线吻合程度判断为高度可能。

**解决方向（两条互补，待用户选择后实施）：**
1. **数据管线防御性过滤（治标，优先级更高、改动小）：** 在 `_submit_turn_sample`/`_submit_rl_turn_sample` 提交前检测明显退化的生成——比如 `thinking` 很长但最终 `content` 异常短、或 `response_ids` 顶格卡在 `max_tokens`——直接丢弃，不让这类样本进入训练队列，阻断"退化样本 → 梯度雪崩 → 更多退化样本"这条正反馈链路。不需要动训练超参，风险最低。
2. **训练侧根源缓解（治本，改动更大、需评估是否偏离论文默认配置）：** 如果确认是 Adam + 稀有 token 的已知交互问题，可以考虑对 embedding/输出层单独做梯度裁剪或使用更保守的 Adam epsilon，但这类改动会偏离论文/官方脚本的默认超参配置，需要先评估对复现有效性的影响，不能贸然改。

**更新（同一天，因果关系已用直接证据核实，非仅时间重合）：** 精确核对了 `rollout_batch_size=16` 的样本-index 到训练-step 的映射（`data.py:480 Dynamic batching: num_samples=16` 逐段核对），确认 step 21→22（19:58:07→20:04:54，grad_norm 224.90→556.07，单步跳变最大的一次）实际消费的 16-18 条样本（index 361-378）里，**至少 7 条是 `response_len=7~8` token 的退化样本，全部 `reward=-1.0`，接近半个 batch**——不再是时间上的巧合关联，是这一步梯度爆炸所用的训练数据本身就被这类样本主导。

**更新（同一天，排查"顶格截断"是不是能查清根源，发现关键日志缺口）：** 用户追问顶格截断（`finish_reason=="length"`）到底该不该一并过滤、要不要先查根源。拉取了具体一条顶格截断样本（session `6c895508` turn 7/8，`thinking=8198 chars`）的完整原文尝试判断是"卡死循环"还是"正常推理没写完"，发现两个关键事实：
1. `content`（最终答案字段）是**完全空的**——模型耗尽整个 8192 token 预算，从未写出 `</think>` 闭合标签和最终答案，不是"内容超长被截断显示"而是真的没写出结果
2. 同一 session 的 turn 7/8/9/10**连续四轮**，`thinking` 长度精确停在同一个数字（8198 字符）——如果是"题目难、需要更多推理空间"这种正常情况，每次卡住的长度该有波动，连续精确撞在同一天花板更像是卡死不收敛
3. 但**无法进一步判断**——`openclaw_opd_api_server.py:742-750` 的日志只记录了 `reasoning_content` 的字符数（`len(reasoning)`），从未把推理原文打印出来，且 `OPENCLAW_RECORD_ENABLED` 未开、没有完整转录留存，现有日志material 不足以判断这 8192 token 内容到底是重复车轱辘话还是发散但不重复的无效探索

**决策（用户拍板）：** 本次只过滤"最终答案异常短"和"命中已知乱码 token（id=122362）"这两类已确认的退化样本，**不过滤顶格截断**——根因未查清前不能确定这是不是真实 bug，贸然过滤会丢失诊断材料。同时把 `reasoning_content` 原文补进日志（仅在 `finish_reason=="length"` 时记录全文，避免正常场景日志膨胀），供下次复现时判断顶格截断到底是不是卡死循环。

**已实施：** 沿用既有的"补丁副本"机制（`scripts/prepare_patched_openclaw_opd.sh`，官方 `openclaw-opd/` 目录本身不动，训练时通过 `PATCHED_OPD_DIR` 优先加载补丁副本），在这个脚本的 Python 补丁块里追加两处改动（commit 见下）：
1. `thinking=%d chars` 日志之后，新增：`finish_reason=="length"` 时把 `reasoning` 原文完整打进日志（`TRUNCATED (finish_reason=length) reasoning_text:\n%s`）
2. 已有的"空回复跳过"检查之后，新增：`content.strip()` 长度 `<5` 字符，或 `response_ids` 命中 `{122362}`（"𬣳"），直接判定退化、不提交训练样本

改动只影响 `openclaw-opd/openclaw_opd_api_server.py`（Personal Agent OPD/Combine/Combine-Select 共用的基类），General Agent 各 track（Terminal/GUI/SWE/Tool-call）用完全独立的代码路径，不受影响。

**待验证：** 下次提交 8GPU 正式训练后确认：(a) 退化样本过滤生效（日志里能看到 "degenerate response...skipping"）、grad_norm 不再因这类样本雪崩；(b) 如果再次出现顶格截断，`TRUNCATED...reasoning_text` 日志能打印出完整推理原文，供判断是否为卡死循环——如果确认是循环，再决定是否需要单独修复或过滤。

**更新（同一天，纠正一个实现错误：乱码 token 的屏蔽方式搞反了）：** 上面"已实施"部分描述的乱码 token 拦截，第一版实现成了**生成完之后检测 `response_ids` 里有没有这个 token、有就丢弃样本**——但这只保护了训练数据，`return {"response": output}` 返回给调用方（真实 OpenClaw 网关→模拟对话）的仍然是 SGLang 原始生成内容，**对话本身依然会收到这条坏回复**，只是不会被算作训练样本。用户指出这跟最初说的"生成时候就屏蔽"不是一回事，确认是实现错误。

WebSearch 确认 SGLang 的 `/v1/chat/completions` 支持标准 OpenAI 兼容的 `logit_bias` 参数（-100~100，-100 视为基本不会再被采样到），但也有已知 issue（[sgl-project/sglang#6171](https://github.com/sgl-project/sglang/issues/6171)、[#3059](https://github.com/sgl-project/sglang/issues/3059)）显示个别版本上 `logit_bias` 不是 100% 可靠。

**修正：** 在 `forward_body` 构造之后（发给 SGLang 之前）加 `forward_body["logit_bias"]["122362"] = -100`，真正做到生成阶段就屏蔽，对话本身也不会再看到这条坏回复；原来事后检测 `response_ids` 的 token 检查**保留**，降级为兜底（防 `logit_bias` 在个别 SGLang 版本上失效），不再是主力机制。三处改动都在 `scripts/prepare_patched_openclaw_opd.sh` 里，commit 见下。

这个修正提醒了一件事：generation-time token 屏蔽本身也不是完全没代价的操作——用 `logit_bias` 人为压低某个 token 的概率，意味着实际采样分布不完全是模型自己的原始策略分布，跟 RL 训练"log-prob 要真实反映模型策略"这个前提有一点张力（类似需要 off-policy 修正的情况）。对这一个几乎不会被正常任务用到的生僻 token 而言影响应该极小，但记录下来供以后类似决策参考。

**更新（2026-07-16，重新提交后发现新问题：模型被 OpenClaw 自带的"记忆/心跳"工具带偏）：** 用上述修复（logit_bias 屏蔽 + 退化过滤 + wandb 代理修好）重新提交 8GPU 训练（run `20260716_143407`）。乱码 token 确认 0 次出现（logit_bias 生效），退化过滤也在正常拦截（224 次）。但 grad_norm 仍缓慢爬升（step 0 的 2.3 到 step 10 的 41.8），且用户从实时日志里发现一种新的畸形模式：TA 批改作业时反复调用 `{"name": "memory_get", "arguments": {"path": "2026-07-16.md", ...}}`，或读取 `HEARTBEAT.md`，完全不回应 TA"请读 homework1/N.txt 并写批改意见"的明确指令，连续多轮卡死，最终撞 8 轮上限失败。

**量化影响：** 截至排查时 TA 已处理 27 道题，其中 **10 道（37%）撞 8 轮上限失败**，**8 道（30%）过程中出现过 memory_get/HEARTBEAT 干扰**；且已确认至少一例（问题17）全部 8 轮都在调 memory_get、从未真正读过 homework1，但因为撞到轮次上限，日志仍显示"TA confirmed grading...done"——说明"正常完成"的计数里可能还混有这种空壳完成，实际受干扰比例只会更高。

**根因排查（用户要求先查论文/官方代码，确认不是我们独有的偏离）：** 用 Explore 子代理查证 `paper_index.md`/`paper_understanding.md`/`paper_reproduction_scope.md` 和整个官方仓库克隆，确认论文和官方代码均未提及如何避免模型被这类工具干扰。唯一相关机制（`extensions/rl-training-headers/index.ts:18,44-51`）解决的是不同问题——把 `heartbeat`/`memory`/`cron` **外部触发器**发起的对话标记为 `side`（不算训练数据），管不到模型在正常 main turn **内部主动调用**这些工具的情况。`TA_chat.py`/`README.md` 均无工具白名单配置，也没有 workspace 清理步骤。结论：这是论文和官方代码都未覆盖的真实空白，最可能的解释是论文原始跑法也暴露在同样的环境风险下，只是那次训练轨迹没有偶然走到"模型开始探索这些无关工具"这条路（跟乱码 token 的性质类似——环境隐患是否被触发看运气）。

**定位工具来源：** 直接查了 OpenClaw 产品自身源码（`D:\MAO\Claude\openclaw`，CLI 本体，非训练仓库）。`memory_get` 定义在 `extensions/memory-core/src/tools.ts:661`，描述是"从 MEMORY.md 或 memory/*.md 读取片段"，属于 `memory-core` 插件（`openclaw plugins list` 确认当前是 enabled），跟 homework 读写用的 `read`/`write`/`append` 工具是完全不同的代码路径，架构上互不依赖，禁用安全。`HEARTBEAT.md` 及同一批"agent 自我身份"文件（`AGENTS.md`/`IDENTITY.md`/`SOUL.md`/`TOOLS.md`/`USER.md`）属于 OpenClaw **核心**代码（`src/agents/bootstrap-files.ts`、`src/auto-reply/heartbeat.ts`），不是插件，没法用 `plugins disable` 关闭，workspace 初始化时会自动生成。

**修复：** `openclaw plugins disable memory-core`（提示"Restart the gateway to apply"，训练重新提交时网关自然重启生效）。`HEARTBEAT.md` 类核心文件关不掉，继续靠已有过滤兜底。

**同时补的诊断日志：** `tool_calls:` 这行日志原本不带 `session_id`，事后分析时没法跟同一请求的其他日志行可靠关联（并发场景下日志天然交错，多次尝试按行号/时间邻近关联都因为交错出错）。在 `prepare_patched_openclaw_opd.sh` 里补上 `session=%s`，供以后分析用。

**更新（同一天，重新评估过滤规则，去掉按 content 长度过滤）：** 用户指出一个此前没意识到的问题——`content.strip()<5` 这条规则会把"TA 要求展开、模型只回复个位数字（比如'25'）"这种情况也当退化样本过滤掉，但**这类样本被判 -1 恰恰是正常、有效的 RL 训练信号**（教模型"这样答不满足格式要求"），不是需要剔除的损坏数据；只有真正空的内容（`content=''`）才需要拦截。之前把"内容异常短"和"内容完全损坏（乱码/空）"这两类混在一起处理是错的。**修复：** 去掉 `len(content.strip())<5` 这条判断，只保留已知乱码 token 的兜底检查（防 logit_bias 万一失效）+ 官方原有的"完全空回复"检查，不再额外按长度过滤。三处改动（disable memory-core、tool_calls 加 session_id、过滤规则简化）均已提交。

**待验证：** 停掉旧的 `20260716_143407` run（GPU 已清理干净），用上述全部修复重新提交训练，观察：(a) memory_get/HEARTBEAT 干扰是否消失；(b) TA 撞轮次上限失败率是否显著下降；(c) grad_norm 是否能保持稳定，不再重演早期爬升。

**更新（重新提交后，run `20260716_182012`，跑 8 小时后无声消失，查明是同一个老问题的新触发方式）：** 用全部修复（logit_bias、memory-core 禁用、简化过滤、session_id 日志）重新提交训练。wandb 确认关键指标跟之前完全不同：`train/grad_norm` 全程在 2-8 之间波动、整体呈下降趋势，**没有**复现乱码 token/memory_get 那次的爆炸式增长——这两个根因确认修复有效（乱码 token 0 次、memory_get 0 次、退化过滤触发 0 次，训练数据很干净）。但 `rollout/response_len/mean` 从 step 20 起死死顶在 7000 附近（接近 8192 上限）不再下降，`training.log` 结尾显示 `waiting for combine samples: 6/16, queue=0` 反复打印超过 10 分钟纹丝不动——是吞吐被拖垮致死，不是梯度爆炸致死，两次是完全不同的死法。

**新主因定位：** 之前特意保留、只做诊断日志不过滤的"顶格截断"（`finish_reason=="length"`）这次高达 **298 次**，成为唯一剩下的、明显主导性的问题。抽查这次的 `reasoning_text` 完整原文，确认是同一种"决策犹豫循环"——模型反复重新分析同一个情况（"用户说了别写文件"、"工具调用成功了"、"用户还没回复"），每次措辞略有不同但从不真正推进，直到耗尽 8192 token 预算被硬截断。推理原文里出现"Looking at the tooling guidelines: 'Non-final turn: use tools to advance, or ask for the one missing decision that blocks safe progress.'"这句话。

**排查这句话的来源：** 查证论文/官方训练代码/我们自己的脚本均无此表述，精确定位到 OpenClaw 产品自身源码 `src/agents/system-prompt.ts:456`（`buildExecutionBiasSection()` 函数），是 OpenClaw CLI 产品内置、面向所有 agent 会话默认注入的"Execution Bias"工具使用指南，与训练用哪个模型/数据集无关——跟 AGENTS.md/memory_get 是同一类"产品自带默认值"，不是我们训练代码引入的。WebSearch 未找到专门针对这句话本身的已知 issue，但查到 Qwen3.x 系列模型在 agentic 多轮工具调用场景下的相似报告，常见根因是 `reasoning_content` 在多轮之间未被正确保留/回传导致模型"失忆"式反复重新分析——顺着这个线索查了 OpenClaw 源码 `openai-transport-stream.ts` 里 `shouldPreserveReasoningContentReplay()` 的判断逻辑，**排除**：我们当前 `openclaw.json` 里 `qwen3-4b` 模型声明已经带了 `"reasoning": true`，按代码逻辑推理内容应该是被正常保留传回的，这条线索没有成立。

**训练结束机制确认：** 8 小时后消失的根因还是**已知的老机制**——顶格截断样本拖慢生成速度（每条 1-3 分钟）→ 攒够 16 条样本的速度跟不上 → `RolloutManager` 卡在 `waiting for combine samples` → GPU 实际空闲 → 触发 modelfactory 平台自动回收 workspace（跟 07-15 第一次遇到的死法本质相同，这次的诱因从 TA 的 `stopReason=length` 卡死换成了这个"决策犹豫循环"）。

**意外发现：workspace 2GB 配额区在中断重启后会静默回滚（独立于本次问题，但值得记录）：** 训练结束后查 `homework/`/`homework1/` 目录，发现内容像是两天前（07-15 17:41）的旧版本，一度怀疑是"TA 一直在批改陈旧内容"导致的连锁问题。用 `stat` 核实：文件 `Birth`（inode 诞生时间）是当天早上、但 `Modify`（内容修改时间）停留在两天前——这是典型的"从快照恢复、且恢复过程保留了原始 mtime"的特征。用户指出原因：**workspace 中断后重新启动时，2GB 配额区（`~/.openclaw/workspace/` 所在的持久化区域，跟 `/dfs/data/` 完全独立）会从上次保存的快照恢复**——这次训练结束触发的正是"GPU 空闲→平台自动回收→重启"这条路径，回收重启后配额区被打回了 07-15 那次存储清理时保存的状态。`/dfs/data/` 下的日志、checkpoint、代码仓库完全不受影响（训练日志能完整还原整个过程）。**结论：查到的"陈旧内容"只反映训练结束后、workspace 被回收重启之后的状态，不代表训练进行中 TA 实际看到的内容也是陈旧的**（`simulation.log` 里"Written: homework/34.txt"这行日志证明训练进行中 `prepare_homework_files()` 确实执行了）——"TA 批改陈旧内容导致连锁问题"这个猜测目前证据不支持，暂不作为主因。

**发现的独立缺陷（官方设计本身的空白，不是这次训练失败的直接原因，但值得记录）：** 排查过程中发现 Problem 34（`20260716_182012` 训练进行中，非事后回滚状态）有一次真实的"假成功"——Student 让 policy 模型把答案追加到 `homework/34.txt`，日志显示 `⚠️ 📝 Edit: in homework/34.txt failed`（工具调用真实失败），但 policy 模型的文字回复没有承认这个失败（"The solution is already in the file"），且 Qwen3-32B 模拟器**在能看到这条失败警告的情况下**依然生成了包含 `DONE_SENTINEL`（`student_chat.py:66` 定义为字符串 `"HOMEWORK_DONE"`）的消息，判定"Turn 3: Student confirmed problem 34 is done!"。查证 `student_chat.py:205-207` 确认"完成"判定**完全只看模拟器自己的文字里有没有这个哨兵字符串，不检查任何工具调用是否真的成功**——这是官方原装代码的设计，不是我们复现引入的偏差。当场核实 `homework/34.txt` 实际内容确认 Solution 部分确实是空的，写入确实没有成功。

**待查（用户提出的新假设，尚未验证）：** 这类"写入实际失败但被判定成功"的情况，是否会导致该问题的 `homework1/*.txt` 内容不完整（Solution 部分为空或缺失），进而在 TA 后续批改这道题时，因为素材本身就残缺/矛盾，触发了"决策犹豫循环"这类困惑行为——即 Problem 34 的假成功可能是循环问题的**上游诱因**之一，不只是一个独立现象。需要检查：(a) 其他触发了顶格截断/决策循环的 session，是否也对应着某个 Student 侧写入失败但被判"完成"的问题；(b) 这次训练里 Student 侧"Edit/Append 失败但被确认完成"的问题总共发生了多少次，占顶格截断 298 次的比例有多大。

---

## [2026-07-17] 排查"决策犹豫循环"根源——确认是 Student 侧问题（这次训练根本没跑到 TA），定位到 lane 排队堆积机制，DONE_SENTINEL 误判范围比预期严重得多

**背景：** 承接上一条目。`20260716_182012` 训练 8 小时后因顶格截断（298 次，"决策犹豫循环"模式）拖垮吞吐、GPU 空闲被平台回收而结束。用户观察到 Problem 34 存在一次真实的"Edit 失败但被 `DONE_SENTINEL` 误判完成"的案例，最初假设：这类问题产生的残缺 homework1 素材，可能是后续 TA 批改时触发"决策犹豫循环"的上游诱因。

**更新（原假设的前提不成立）：** 查证 `simulation.log` 全文搜索 "Grading problem"（TA 批改的标志性文本）**零命中**——这次训练 Student INIT 阶段自己就卡死了，从头到尾没有跑到 TA。原假设"Student 假成功 → TA 批改时循环"的因果链条不存在，问题从一开始就是 **Student 自己的循环**，不涉及 TA。

**更新（DONE_SENTINEL 误判的范围比预期严重得多）：** 抽查 Problem 34/36/37/44 这四道"显示 confirmed done"的题目，**全部** 是零真实进展的误判：
- Problem 34：Edit 失败，模型回复掩盖失败，被判完成
- Problem 36：模型自己说"The file remains untouched as requested"（明确承认没写），依然被判完成
- Problem 37：全部 8 轮没有一次拿到真实回复（"No response from OpenClaw" / "No tool calls generated" 交替出现，其中一次遭遇 180 秒 read timeout），最后仍判完成
- Problem 44：6 轮里 5 轮生成失败，模拟器绝望到自己写出完整解答让模型帮忙保存，模型依然生成失败，仍判完成

统计 Problem 36-65 区间："confirmed done"只在 36/37/38/44/45/50/52/53 出现过（且这几个抽查后全部证实是误判），**从 Problem 54 起到区间末尾（65），连续 12 道题一次"confirmed done"都没出现过**——即便是 DONE_SENTINEL 这种不做任何验证的兜底机制，到这里也彻底失效了，说明这段时期已经不是"偶发误判"，是持续性的完全瘫痪。

**定位到具体机制——lane 排队堆积：** 在 `openclaw.log` 里发现 `[diagnostic] lane wait exceeded: lane=session:agent:main:openai-user:<session>` 这类诊断日志，每个 session 有自己独立的 lane（不是全局共享队列）。查 Problem 36/37 各自的 lane，发现等待时间持续增长：Problem 36 从 51 秒涨到 132 秒（约 15 分钟内），Problem 37 一度到 148 秒、`queueAhead=2`。同时发现 `lane task error: ... durationMs=89706 error="FailoverError: LLM request timed out."`——一次生成真实跑了 89.7 秒才被判超时。机制推断：生成本身卡进"决策犹豫循环"耗时 1-3 分钟 → 触及客户端超时阈值（180秒 read timeout 等）→ `student_chat.py` 自带的重试逻辑发起新请求 → 新请求排在同一 session 的 lane 队列里等前一个仍在处理的请求 → 等待时间累积增长，表现为"connection timeout"/"no response"，但根子是生成慢，不是真的网络问题。

**时间线交叉验证（33→37）：**
- Problem 33：一次孤立的 11.8 秒超时（`lane task error`），随后恢复正常
- Problem 34：openclaw.log 里搜"student-hw-34"**零匹配**——这次交互本身很快，没有 lane 排队问题，唯一的问题是语义层面（Edit 失败），跟"耗时变慢"这条机制无关
- Problem 35：完全正常，无任何异常
- Problem 36：lane 排队堆积从这里**突然、崭新地**开始（51秒→132秒），之前 33/34/35 都没有铺垫或延续的痕迹

**结论（用户确认、本人过度断言已收回）：** "Problem 34 的假成功直接导致 36 的堆积"这个具体因果链条，目前的 lane 排队时间证据**不支持**（34 本身干净利落，35 也完全正常，中间没有过渡）。但用户指出一个尚未验证的替代假设——33 号那次超时重试，有没有可能在更底层造成了请求错位/重复（不会体现在 lane 排队时长上，但可能造成对话状态不一致）——这个可能性还没有排除，需要检查 33 号超时重试后有没有产生重复/悬空的请求。**唯一可以确认的是**：从 Problem 36 开始，无论最初根源是什么，训练进入了持续性、不可逆的恶化（一路到 54 题起完全瘫痪），33/34/35 都不存在这种一路恶化不回头的迹象——**Problem 36（或其代表的这个时间点触发的某个具体事件）是这次训练彻底失效的直接肇因**，找到并解决"36 为什么会开始卡"是接下来的核心目标。

**待查：**
1. ~~Problem 33 超时重试后，有没有产生重复/悬空请求~~ → 已查，见下方更新
2. Problem 36 第一次触发"决策犹豫循环"级别耗时（1-3分钟+生成）的具体那次请求，其 prompt/上下文有没有异常之处，找到"36 为什么会开始卡"的真正触发点
3. 是否可以从代码层面验证 lane 排队机制的具体实现（重试是否真的会与仍在处理的原请求并发排队，而不是取消重发）

**更新（查完 Problem 33 的 `training.log` 完整视角，排除"33 导致错位"假设，但发现一个弱信号——负反馈后 thinking 变长可能是通用倾向）：**

Problem 33 真实内部 session（`b61ee04f-e1cb-493d-9f43-d5b9b4700e08`）完整 turn 序列：

| Turn | 时间 | thinking chars | response_tokens |
|---|---|---|---|
| 1 | 19:10:05 | 3790 | 1043 |
| 2 | 19:10:10 | 3189 | 760 |
| 3 | 19:10:24 | 9171 | 2061 |
| （PRM eval turn=2 @19:11:04，eval_score=**-1.0**，K_i=3） | | | |
| 4 | 19:11:05 | **20260** | 4572 |
| （PRM eval turn=3 @19:11:19，eval_score=-1.0） | | | |
| 5 | 19:11:34 | 13860 | 3595 |
| 6 | 19:11:43 | 6758 | 1538 |
| （PRM eval turn=4/5 @19:11:56/19:12:18，均 +1.0） | | | |

**结论一：Problem 33 全程没有真正顶格截断**——6 个 turn 的 `response_tokens` 最高 4572，远没碰到 8192 上限，无 `TRUNCATED` 标记，session 干净收敛完成。turn/PRM-eval 交替序列完全顺序、无重复 turn 号或并发 session_id 冲突的痕迹——**应用层日志不支持"33 超时重试导致错位"这个假设**（但日志只是应用层视角，无法排除更底层的网络层重复，只能说没找到痕迹）。

**结论二（弱信号，不是根因，先记录）：** turn=4 的 thinking 从前三轮的 3000-9000 字符量级，一次性跳到 **20260 字符**，且紧跟在 PRM eval **-1.0**（K_i=3，需要 3 个候选才选出）之后；随后 turn 5/6 thinking 逐步回落（13860→6758），同时 PRM eval 也转回 +1.0、+1.0。这跟 Problem 36 的触发条件同源（负反馈后 thinking 变长），量级约为 36 的 54%（20260 vs 37495），但 33 这次自己在预算内收住了，没有失控。**推测：**"负反馈后 thinking 变长"可能是模型本身的常规倾向（不算 bug），36 的特殊之处不在于"变长"这件事本身，而在于这次没能在 8192 token 预算内收敛——是同一机制下的量变到质变临界点，而不是 33/34/35 存在什么特殊铺垫或残留状态。33/34/35 本身没有表现出向 36 过渡的渐进恶化迹象。

**更新（挖出 Problem 36 turn=2 完整 344 行 `reasoning_text` 原文，定位到"决策犹豫循环"的精确触发机制——不是内容困惑，是输出格式判定困惑）：**

先纠正一个此前的错误猜测：一开始怀疑是"学生要求先看答案、别写文件"跟"系统提示要求非最终轮用工具推进"这两条指令冲突导致循环。查完 turn=2 完整 reasoning_text 后确认**不是**——真正的机制更精确：

- Turn 1（19:18:06）：模型正常读取 `homework/36.txt`，工具调用成功，无异常
- Turn 2（19:18:57，紧接着）：模型在推理的**前 10% 就已经确定了要回复的内容**——`"I didn't modify the file. Do you need anything else?"`，这句话在后续 300+ 段重复推理里**一字未变**
- 但之后陷入无限循环，反复纠结的是**同一个问题**：这句纯文字回复要不要包上 `tool_call` XML 标签、算不算一次"函数调用"。反复引用几条互相看似矛盾的系统提示词规则来回论证：
  - `"For each function call, return a json object with function name and arguments within tool_call XML tags."`
  - `"Non-final turn: use tools to advance, or ask for the one missing decision that blocks safe progress."`（`buildExecutionBiasSection()`，见上条已确认来源）
  - `"Do not invent commands."`
  - `"Silent Replies: When you have nothing to say, respond with ONLY: NO_REPLY"`
  - `"Attach media in the final visible reply with MEDIA:<path-or-url> on its own line."`
- 每次都重新得出"这是纯文本回复，不需要 tool_call 标签"的结论，然后立刻自我怀疑"但规则说每次都要走 tool_call 格式"，如此循环几十轮，直到 8197 token 耗尽被硬截断，**从未真正输出终止 turn 所需的内容/格式**

**结论：这是纯粹的输出格式判定困惑，不是内容/决策困惑。** 模型早就知道该说什么，卡在"这段话该怎么打包发出去"这个纯格式问题上。根源是 OpenClaw 系统提示词里"每次函数调用都要 tool_call 标签"和"非工具的纯文字回复怎么输出"这两类规则之间存在真实的表述歧义，Qwen3-4B-Thinking 遇到这种自相矛盾的格式指令容易触发这类"反复重新论证同一结论"的循环。因为这是 OpenClaw 产品默认系统提示词的内容（论文原始实验大概率也用的同一份），符合此前"环境隐患是否被触发看运气"的判断——只是这次触发的具体机制比之前设想的更精确。

**更新（用户追问：这段矛盾表述是不是论文提交后新加的——版本考古确认，不是"运气"问题，是真实的版本断层）：** 用 2026-07-09 那次调查已验证过的方法（GitHub API 按日期查 commit SHA、`git fetch --depth=1` 单独拉某个历史时间点的完整代码树），直接检查触发循环的关键文字——`buildExecutionBiasSection()`（含 `"Non-final turn: use tools to advance, or ask for the one missing decision that blocks safe progress."`）——在论文提交前后的 OpenClaw 版本里是否存在。

全仓库 `git grep`（不只是 `system-prompt.ts` 单文件）确认：

| 时间点 | OpenClaw 版本 | `ExecutionBias`/冲突表述 |
|---|---|---|
| 论文提交日（2026-03-11）前一天 | 2026.3.10 | **不存在**（零匹配） |
| 2026-04-15 | 2026.4.15-beta.1 | 不存在 |
| 2026-04-30 | 2026.4.30 | **存在** |
| 我们训练用的版本 | 2026.6.9 | 存在 |

**这段导致 Problem 36 循环的文字，是论文提交后约 1.5 个月（2026-04-15~2026-04-30 之间）才加进 OpenClaw 的全新内容，论文原始实验用的版本不可能触发这个循环，因为这段文字当时根本不存在。** 同时确认 March 版本里 `"Do not invent commands."` 虽然文字部分匹配，但上下文完全无关（属于"## OpenClaw CLI Quick Reference"章节，指"不要编造 CLI 子命令"，跟 tool_call JSON 格式毫无关系）；`"tool_call XML tags"` 那句本来就不是 OpenClaw 自己的代码（Qwen 模型 chat template 自带，两个版本共有）。

**这跟 2026-07-09 那次 `rl-training-headers` 机制被破坏是同一类问题的第二次发作**——不是我们复现引入的偏差，是 OpenClaw 项目本身在论文提交后的快速迭代中（同期 CHANGELOG 显示单周 400+ PR 合并）新增了论文时间点根本不存在的内容，而我们固定使用的 2026.6.9 版本继承了这些新行为。**这也改变了"要不要动系统提示词"这个决策的保真度风险判断**：去掉这段文字不是"主动偏离论文环境"，而是"消除一个已经存在、由版本断层带来的偏差"——论文原版环境里这段文字本来就不存在。

**更新（定位到 Problem 36 第一次卡循环的确切样本，发现一条可能的自我强化机制）：** 在 `training.log` 里找到 Problem 36 对应的真实内部 session（`7f2bc33a-01b6-404f-841a-79d3e08ae4bc`）：Turn 1（19:18:06）正常读取 `homework/36.txt`；**Turn 2（19:18:57，紧接着）`thinking=37495 chars`、`TRUNCATED (finish_reason=length)`、`response_tokens=8197`（顶格撞满）**——这是这个 session 第一次出现"决策犹豫循环"级别的生成。19:19:55 确认这条样本被正常提交进了训练队列：`submitted OPD+RL sample ... index=162 reward=-1.0 prompt_len=17437 response_len=8197`。

这条样本的 token 数（8197）是同批其他样本（1000-5000 左右）的 2-8 倍，reward 却是同方向的 -1.0——按 Personal Agent 这套简化版 GRPO（`A_t=r_t` 直接用，无组内归一化），这类"超长+同方向负奖励"样本的梯度贡献会被放大，跟此前查乱码 token 那次"梯度爆炸"的机制是**同一种模式**。而这次顶格截断（`finish_reason=="length"`）的样本我们此前特意选择不过滤（想留着诊断卡循环原因），意味着这类样本至今一直在正常提交进训练队列——不排除存在一条"卡循环生成 → 作为超长负奖励样本提交训练 → 反过来推动模型更不稳定 → 更容易再次卡循环"的自我强化链路，这可能是问题一旦出现就不断恶化、不会自愈的部分原因（跟"生成慢拖累吞吐/lane 排队堆积"是并行的另一条伤害路径，不是互斥关系）。

**更新（用户追问：这段矛盾表述是否论文提交后新加、能否回退版本——顺带做了一次更全面的 March vs June 版本差异扫描）：** 用户提出两个问题：(1) 这段导致循环的文字是不是最近几个月新加的；(2) 有没有可能把 OpenClaw 回退到论文提交时（2026-03-11）附近的版本。

**版本考古方法：** 复用 2026-07-09 那次已验证的方法（GitHub API 按发布时间查 npm 版本对应的 commit SHA、`git fetch --depth=1` 单独拉取该历史时间点的完整代码树），在本地 `D:\MAO\Claude\openclaw` 仓库里打了两个 tag：`march_2026_3_8`（npm 2026.3.8，发布于 2026-03-09，论文提交前 2 天，是最接近论文时间点的**实际可安装**发布版——精确的"2026.3.10"字符串本身不是一个 npm 发布版本，只是某次开发中间态的 package.json 版本号）、`june_2026_6_9`（npm 2026.6.9，发布于 2026-06-21，我们训练流水线当前固定使用的版本）。

**问题(1) 的结论：确认是新加的。** 全仓库 `git grep` 在 `march_2026_3_8` tag 上零命中 `Execution Bias`/`Non-final turn: use tools to advance`/`Attach media in the final visible reply`；二分查到 2026-04-15（`2026.4.15-beta.1`）仍不存在，2026-04-30（`2026.4.30`）首次出现——这段文字是论文提交后约 1.5 个月才加入的全新内容。

**问题(2) 的结论：技术上可行，但代价不小，关键发现：**

| 检查项 | 结果 |
|---|---|
| Node 版本要求 | March `>=22.12.0` vs June `>=22.19.0`，回退方向要求更低，不是阻塞项 |
| `extensions/sglang/` 是否存在于 March | **不存在**。但 March 版本已有等价的通用机制（`docs/providers/vllm.md`：`models.providers.<id>` + 自定义 baseUrl 指向任意 OpenAI 兼容 `/v1` 端点），SGLang 和 vLLM 暴露同一套 OpenAI 兼容 API，理论上可以复用这条通用路径连接我们自己的 SGLang 服务，不是硬阻塞，但需要实测 |
| `hasOpenClawTransportRequirement`（07-09 那次破坏 `rl-training-headers` 官方插件的门槛）| **March 版本确认不存在**——回退不仅能避开这次循环 bug，还能顺带恢复论文原版的 header 注入机制，扔掉 `localService: {command: "/bin/true"}` 绕开方案 |
| `memory-core` 插件 | March 版本同样存在（代码相同），不是版本差异导致的问题，回退与否都要手动禁用 |

判断：没有发现硬阻塞项，且回退能一次性解决两个已知问题（循环 bug + header 注入失效），但代价是整套流水线是针对 2026.6.9 反复调试稳定的（cgroup OOM、workspace 快照回滚、gateway 稳定性等经验都基于这个版本），回退 3 个月等于要在一个从没跑过的旧版本上重新做一轮稳定性验证，可能遇到当时存在、后来才修的未知 bug。这一步暂缓，用户选择先在当前版本上做定点修复。

**顺带做的更广范围版本差异扫描（用户要求：查一下除了已发现的两处，还有没有其他可能影响这次实验的改动）：** 用 Explore 子代理系统扫描 `march_2026_3_8` 到 `june_2026_6_9` 的差异（17333 个文件、370 万行插入，规模上不可能逐行看完，只做了 CHANGELOG 全文 + 几个训练流水线核心必经文件的定向 diff），本人抽查核实了其中最关键的几条（用 GitHub API 查了具体 PR 的合并时间）：

- **PR #92191**「retry thinking-only errored turns」（合并于 **2026-06-14**）、**PR #93073**「retry empty post-tool final turns」（合并于 **2026-06-15**）——March 版本没有这两个机制。June 版本里，一轮如果输出是"只有 thinking、没有实际输出"（跟我们的顶格截断样本高度类似）或者"工具调用完之后最终回复是空"，框架会**自动重试**这一轮，March 会让它直接失败/结束。**这两个改动离我们现在用的 2026.6.9（发布于 2026-06-21）只差不到一周**，比 4 月加入的 Execution Bias 章节还要新。可能的影响：同一次决策犹豫循环，在 June 版本上不是"卡死一次就结束"，而是"卡死→自动重试→可能再次卡死"，把单次循环的伤害复合放大，值得作为后续排查方向记录下来，但因果关系尚未验证
- **PR #91361**「compaction 默认超时 900s→180s」（合并于 **2026-06-10**）——只影响自动上下文压缩场景，目前 session prompt 长度在 17-19K token 左右，大概率还没到触发 compaction 的门槛，相关性存疑，优先级较低
- 排除的候选：新增的"跨轮次重复相同工具调用"熔断机制——我们这次的循环是单轮内部推理文本重复，从没真正发起过第二次工具调用，机制类型不匹配，大概率拦不到这类问题；`thinking.ts` 大重写（53→753行）——看起来偏 Anthropic signature 相关，我们用的是 SGLang/OpenAI-completions 路径，且之前已确认 `reasoning_content` replay 在我们的配置下没问题，优先级较低
- **诚实说明覆盖范围**：`openai-transport-stream.ts`（4427行）、内置 read/write/edit 工具实现语义、`provider-runtime.ts`（1019行新文件）体量太大，只做了存在性确认，没有逐行核实是否还有其他相关改动，不保证查全

**决策（用户拍板）：** 先不回退版本，在当前 2026.6.9 上做定点修复。

**修复尝试一（已回退，改用方案二）：** 最初用 `before_prompt_build` + `appendSystemContext` 追加一条消歧规则（原因：`resolveSystemPromptContribution` 整段替换机制要求是 provider 唯一绑定的插件，`sglang` 已被内置扩展占用，不是零接触方案）。用户明确要求修复必须尽量可靠（"不然就会导致整个训练失效"），指出 append 方案没有真正删除冲突文字、只是在后面加了句解释，原文依然完整留在上下文里，可靠性天然不如直接删除。用户同时评估了"基础设施层兜底"（限制单次循环拖垮训练的连锁反应）方案，明确不采用——会掩盖问题、发现不了。整段 append 方案的插件文件和三个训练脚本的部署代码块已 `git revert` 撤销。

**修复方案二（内容层，正在实施）：** 直接 patch 内置 `extensions/sglang` 扩展本身，通过 `resolveSystemPromptContribution`（`src/plugins/types.ts:1675`）的 `sectionOverrides.execution_bias` 真正替换掉 Execution Bias 章节内容——不是事后追加澄清，而是从根上让那句冲突的话不出现在提示词里。

实现细节：
- 确认 `resolveSystemPromptContribution` 必须挂在 `api.registerProvider({...})` 传入的对象上（`ProviderPlugin` 类型，`src/plugins/types.ts:1248`，字段跟 `id`/`label`/`docsPath`/`auth` 同级），这是 `ensureProviderRuntimePluginHandle` → `resolveProviderRuntimePlugin` 返回值的实际类型，逐层跟踪调用链确认，不是靠猜测
- 服务器上实际部署的 `/usr/lib/node_modules/openclaw/dist/extensions/sglang/index.js` 是编译后的干净单文件（不是像 core src 那样被打包成哈希文件名），跟本地 TypeScript 源码结构一致但不是逐字相同，已用真实文件内容核对过（`api.registerProvider({ id, label, docsPath: "/providers/sglang", envVars, auth, catalog, ...buildProviderReplayFamilyHooks(...), wizard })`，没有 `resolveSystemPromptContribution` 字段）
- 这个文件不属于 OpenClaw-RL-official（论文训练代码仓库），是 OpenClaw CLI 产品自身内置扩展，没有可以在本仓库里维护的"官方源码副本"可 patch——跟 `prepare_patched_openclaw_opd.sh`/`prepare_patched_rl_training_headers.sh`"从仓库里的官方源码生成补丁副本"的模式不同，这次是直接 patch 服务器上实际部署的活文件
- 新脚本 `scripts/prepare_patched_sglang_execution_bias.sh`：第一次运行时自动备份未修改原文件（`${LIVE_FILE}.orig-unpatched`），之后每次都从这份干净备份重新生成补丁（不是从当前可能已被 patch 过的活文件生成），保证多次重跑训练脚本不会重复打补丁或产生漂移；补丁内容是 `buildExecutionBiasSection()` fallback 原文逐字复制，只删掉"Non-final turn: use tools to advance..."这一行，其余六条不变
- 已确认 patch 部署发生在 `openclaw gateway run` 启动之前（`train_with_services.sh`/`minitest_train_with_services.sh`/`smoke_train_with_services.sh` 三处都验证过顺序），保证 gateway 进程启动时读到的是打过补丁的文件

**本地已验证（用用户实际贴出的服务器文件内容原样复现测试，非纯理论审查）：**
- 用真实粘贴的 `index.js` 内容在本地跑了一遍完整补丁脚本逻辑：锚点匹配成功，输出通过 `node --check` 语法校验
- 二次运行幂等性验证：模拟同一台机器上重复跑训练脚本（第二次运行时备份文件已存在），两次生成的补丁产物逐字节相同，不会重复打补丁或产生漂移
- 幂等性保护验证：故意把已打过补丁的文件当作"原始备份"喂给脚本，正确报错拒绝（不会静默产生双重补丁）
- 加了一条一次性确认日志（`console.error("[execution-bias-fix] resolveSystemPromptContribution invoked...")`，靠 `_executionBiasFixLogged` 标记只打一次、不刷屏），训练启动后可以立刻从 `openclaw.log` 里确认这个钩子真的被调用了，不用等一次完整训练跑完才能间接判断有没有生效

**待验证（下次训练前必须确认，尤其是"不能失效"这个要求）：**
1. ~~patch 脚本的锚点字符串匹配和实际部署文件一致~~ → 已用真实文件内容本地验证，见上
2. **启动前确认没有残留的 OpenClaw gateway 进程**——如果上一次训练遗留的 gateway 进程还在跑，Node 已经把旧版 `index.js` 加载进内存，这次的文件级 patch 不会生效（这是文件补丁类修复的通用限制，`rl-training-headers` 那次也有同样的依赖，不是这次新引入的风险，但鉴于这次要求"必须生效"，需要额外强调）
3. ~~训练启动后第一时间查 `openclaw.log` 里有没有出现确认日志~~ → 已确认，见下
4. 真实训练观察 `TRUNCATED` 次数是否显著下降、Problem 36 那类决策犹豫循环是否还会出现

**更新（重新提交 8GPU 正式训练，run `20260717_133740`，patch 确认生效）：** 用户 GPU 已申请，选择跳过 smoke 测试直接提交正式训练（评估依据：smoke 规模太小测不出概率性的循环问题，只能测部署机制，而部署机制已经过本地模拟验证；GPU 闲置本身也是浪费）。启动前确认无残留 gateway 进程。

`openclaw.log` 里查到确认日志：
```
2026-07-17T13:41:24.089+08:00 [execution-bias-fix] resolveSystemPromptContribution invoked, overriding execution_bias section (first call only, not logged per-turn)
```
**patch 已确认真实生效**——`resolveSystemPromptContribution` 钩子被内置 sglang provider 实际调用，Execution Bias 章节的替换内容已经在发给模型的真实提示词里生效，不再是"应该生效"，是"已验证生效"。

Problem 0 INIT 阶段顺利完成（3 轮：读文件给答案 → 按要求改写风格 → 追加写入文件确认完成），无循环迹象。后续持续观察顶格截断次数和是否再出现 Problem 36 那类循环。

---

## [2026-07-17] 新问题：新发现的 503 风暴导致 Student/TA/Teacher INIT 阶段集体崩溃，Joint 阶段秒结束

**背景：** run `20260717_133740` 跑了一段后，Student INIT 在第 49 题崩溃（`session=student-hw-49-31138`），紧接着 TA INIT 第 0 题就崩溃（`session=ta-grade-0-82181`），Teacher INIT 第 0 题前两轮成功、第三轮崩溃（`session=teacher-comment-0-82379`）。三者报错特征完全一致：`send_to_openclaw` 3 次重试（1s/2s/4s 退避）全部命中 `408 Client Error: Request Timeout`，`main()` 里未捕获异常直接终止整个 Python 进程。之后 `run_init_phase()`/`run_one_persona()` 已知的"不阻塞"缺陷（见 07-15 条目）放行进入 Joint 阶段，但 Joint 阶段"开始"和"结束"是日志里紧邻的两行，中间零输出，`模拟循环结束（INIT 1次 + Joint阶段1次）`后直接进收敛检测（Student 只 check 到 50 session，TA 0 个，Teacher 1 个）。用户手动 `^C` 终止了这次训练。

**排查过程（用户最初怀疑是"Student 49 题的报错引发后面连锁失败"，已用时间戳证伪，纠正为"共同受害于同一场服务风暴"）：**

1. 先查 `openclaw.log` 里这几次失败对应的真实错误，发现跟此前排查的"决策犹豫循环"（生成慢、卡很久超时）完全不是同一种：这次是 `FailoverError: 503 status code (no body)`，每次失败都在 1.5-2 秒内就返回，是**快速拒绝**，不是慢生成。
2. 查这个时间点前后 SGLang 引擎（`training.log` 里 `SGLangEngine pid=29192`）自己的活动：完全健康，`/generate` 请求持续 200 OK，训练侧 `RolloutManager` 也在正常提交样本——说明底层 SGLang 引擎没有整体挂掉，问题出在更靠近 OpenClaw 的某一层（`RL training proxy`，即我们自己的 `openclaw_combine_select_api_server.py`，或它前面的排队/准入控制）。查这层代理自己的日志，没有找到任何 Python 异常/traceback，只有正常的 PRM eval/submitted sample 记录——`503 (no body)` 这种极简响应更像是 SGLang 自身内置的过载保护/准入控制主动拒绝，而不是我们代码抛出的异常（**未 100% 实锤，只是目前证据指向这个方向**）。
3. 统计全部 `503 status code (no body)` 出现次数：**156 次，但全部集中在 15:06:58~15:10:27 这 3.5 分钟窗口内**，不是全程持续出现——训练前后大部分时间是正常的，是一次集中的短暂风暴，不是持续性容量不足。
4. **关键证据（推翻"Student 引发连锁"假设）：** 查 `student-hw-49-31138` 第一次 503 的精确时间戳——`15:06:58.535`，跟整场风暴最早出现的时间点（15:06:58）几乎完全重合。说明不是 Student 的失败传导给了 TA/Teacher，而是 **Student/TA/Teacher 三者都是同一场外部风暴的独立受害者**，只是脚本按顺序调用（Student→TA→Teacher），谁的请求刚好落在这 3.5 分钟窗口里谁就中招，看起来像是"依次传导"，实际是巧合的时间重叠。

**结论：** 专门修"Student 第 49 题"这个具体请求解决不了根本问题——如果这场风暴下次发生在别的题号/别的角色，还是会照样打崩当时正在跑的那个进程。真正要查的是这场 503 风暴本身的触发原因（目前只到"疑似 SGLang 准入控制在 Joint 启动 + 训练自身 rollout 流量叠加时短暂过载"这一步，没有拿到 SGLang 自己的原生日志实锤）。

**用户明确要求：跳过失败继续下一题只是保险措施，不是解决问题的方法，必须先找到 503 风暴的根因。** 同时指出：这类风暴级联失败在 INIT 阶段影响是结构性的——风暴持续 3.5 分钟，如果只是"跳过继续"，一次风暴窗口里可能连续跳过几十道题（不是零星几道），而 INIT 阶段的完整性直接决定 TA（读 `homework1`）、Teacher（读 `homework2`）后续能不能有真实内容可批改/评论——批量跳过不是"丢几个训练样本"，是"这一整段 homework1/homework2 基础数据本身就是空的/残缺的"，跳过机制防不住这种结构性数据缺失，只能防止"进程死亡导致这个角色接下来几小时完全没有任何数据"这个更严重的次生问题。

**待查（根因排查，优先级高于保险措施）：**
1. SGLang 自己的原生服务日志（不是 `training.log` 里 `RolloutManager`/`SGLangEngine` 转发的那部分，是 SGLang server 进程自己的完整日志，可能在别的位置）有没有"queue full"/"reject"/"admission control"这类原生限流记录，实锤 503 的真正来源
2. 这场风暴触发的具体条件——是不是 Joint 阶段三个角色同时启动 + 训练自身 rollout 流量叠加导致的瞬时并发峰值超过某个容量上限；如果是，能不能通过错峰启动 Joint 阶段三个角色、或者给 SGLang/代理层加大队列容量来解决
3. 三个角色脚本的容错保险措施（跳过继续、更长退避重试等）作为独立于根因排查的次要改动，优先级低于第 1/2 点

**更新（找到很可能的根因——SGLang Router 层默认容量限制，官方代码从未配置过）：** 先用精确时间戳证伪了"Student 49 题报错引发 TA/Teacher 连锁失败"这个假设——`student-hw-49-31138` 第一次 503 的时间戳（15:06:58.535）跟整场风暴最早出现的时间点几乎完全重合，说明 Student/TA/Teacher 是同一场外部风暴的独立受害者，不是因果传导，只是脚本按顺序调用导致看起来像依次传导。

**排查方向调整：** 检查过 SGLang 引擎自身原生日志（`training.log` 里能看到 `SGLangEngine pid=29192`/`pid=27181` 的完整原生输出，`log_level='warn'` 下没有找到显式的"reject/queue full"字样，但这个日志级别可能压掉了熔断器状态切换这类事件，未 100% 证伪也未 100% 证实）。

**关键发现：** `training.log` 里有 `RolloutManager pid=26108 ... rollout.py:1164 - Launch router with args: RouterArgs(...)` 完整参数转储，显示 OpenClaw 实际连接的是 SGLang **Router 层**（不是引擎本身），这一层有明确的容量限制：
- `queue_size=100`：请求队列上限 100，超过直接拒绝（不排队等待）
- `disable_circuit_breaker=False`：**熔断器开启**
- `cb_failure_threshold=10` / `cb_window_duration_secs=120`：120 秒内失败 10 次触发熔断
- `cb_timeout_duration_secs=60`：熔断后 60 秒内无差别拒绝所有请求，需连续 3 次成功（`cb_success_threshold=3`）才重新关闭

这个组合精确匹配观察到的症状——"极快返回（1.5-2秒）、无 body 的 503"是队列拒绝/熔断的典型特征（不是慢生成超时）；熔断一旦触发是**无差别拒绝所有客户端**的，直接解释了 Student/TA/Teacher 为什么会在同一窗口"同时"中招；3.5 分钟的持续时间也对得上熔断器 60 秒一轮、可能连续触发了好几轮。

**确认这是纯默认值，不是我们或论文官方代码配置的**：全仓库（`OpenClaw-RL-official`）搜索找不到任何地方设置过 `queue_size`/`cb_failure_threshold`/`disable_circuit_breaker` 等参数。追到源头：`RouterArgs.from_cli_args(args, use_router_prefix=True)`（`slime/slime/ray/rollout.py:1154`），这些参数由 slime 训练框架通过带 `router-` 前缀的命令行参数（如 `--router-queue-size`、`--router-cb-failure-threshold`）暴露，但官方训练启动脚本从没传过这些参数，全部落在 SGLang 包自己的默认值上——**这不是我们复现引入的偏差，论文原始实验大概率也在用同一套默认值，只是他们的并发触发模式可能没撞到这个上限**。

**下一步（待用户确认）：** 给训练启动脚本加 `--router-queue-size`（加大队列容量）和/或 `--router-cb-failure-threshold`（放宽熔断阈值）/`--router-disable-circuit-breaker`（直接关掉熔断器）这类参数，属于纯基础设施/网络容量调整，不碰模型、提示词、奖励计算，跟论文方法论无关，保真度风险低。需要先确认这些具体 CLI 参数名（`RouterArgs.add_cli_args()` 的实际实现），再决定加大到多少合适。

**更新（用户连续指出两个逻辑漏洞，撤回两个错误假设，改用训练步 perf 数据交叉比对，找到真实异常但因果方向仍不确定）：**

**撤回的假设一（Joint 阶段三角色并发导致峰值）：** 用户指出这次 Student(49题)→TA(0题)→Teacher(0题) 三次失败全部发生在 **INIT 阶段**，INIT 是三个角色顺序单独跑的（不是并发），Student 崩溃时 TA/Teacher 都还没启动，不存在"三角色同时打过去"这个前提。之前把这次的 INIT 阶段失败和另一次 Joint 阶段的独立崩溃搞混了，撤回这个解释。

**撤回的假设二（默认容量长期不够用）：** 用户指出之前 8 小时的 run（`20260716_182012`）跑得比这次更久却从没出现过这个问题——如果真是默认队列/熔断参数长期不够用，应该是持续性、可复现的问题，不该只在这一次出现。直接验证：`grep -c "503 status code (no body)" .../20260716_182012/openclaw.log` = **0**，实锤了"默认容量不够用"不是持续性问题，这次运行本身有特殊情况。

**新证据（用户要求"既然是这一步触发的，就该跟其他几十次同类事件比对"）：** 提取这次 run 全部 15 个训练步（`perf 0` 到 `perf 14`）的 `perf/rollout_time`/`perf/tokens_per_gpu_per_sec`：

| 步骤 | rollout_time | tokens/gpu/s |
|---|---|---|
| 0-12（正常范围）| 229.78~530.81 秒 | 27.86~62.52 |
| **13**（15:06:56，风暴开始前 2 秒）| **191.66 秒**（全场最短之一）| **78.99**（明显偏高）|
| **14**（15:09:52，风暴仍在进行）| **138.20 秒**（全场最短）| **148.41**（全场最高，接近正常值 3 倍）|

步骤 13/14 相对前 13 步确认异常，但**异常方向是变快、不是变慢**——这跟"某个环节堵塞导致变慢"的直觉相反，提出一个新的因果方向假设：会不会不是"这一步触发了风暴"，而是反过来——**某个原因先导致 OpenClaw→30000 端口这条路被拒绝/挡住，训练自己的 rollout 请求因为少了竞争反而跑得更快**。支持这个方向的线索：`training.log` 开头有两次独立的"Launch router"调用（line 1505/1508，端口不同），说明训练自己的 rollout 流量和 OpenClaw 走的 30000 端口流量是**两个独立的 Router 实例**——如果它们背后共享同一批 GPU worker，"OpenClaw 这条路被拒绝→worker 空出来→训练 rollout 变快"说得通；如果两条路完全独立不共享 worker，这个解释就不成立，纯属时间巧合。

**当前状态：** 确认步骤 13/14 客观异常且与风暴同时发生，但**因果方向仍未确定**（步骤边界触发风暴 / 风暴释放容量导致步骤加速 / 纯粹时间巧合，三者都未排除）。需要更底层的证据（SGLang 两个 Router 实例是否共享 worker 池、SGLang 自己的原生队列/连接状态日志）才能继续往下查，现有 `training.log` 里能看到的都是转发日志，不够用。

**更新（确认 OpenClaw 流量和训练自己的 rollout 流量共用同一个 Router/worker 池，"两条独立路径"这个假设不成立）：** 直接查我们自己的代理源码（`openclaw-opd/openclaw_opd_api_server.py:241`）：

```python
url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
```

这行代码确认：port 30000 这层代理（OpenClaw 走的路）把请求转发到的是 `args.sglang_router_ip`/`sglang_router_port`——跟 `_start_router()`（`slime/slime/ray/rollout.py`）里 `router_ip_attr == "sglang_router_ip"` 那次调用是**同一个属性**，也是训练自己 rollout 用的那个 Router。也就是说**之前查到的"两次独立 Launch router 调用"里，其中一次（`sglang_router_*`）本来就是 OpenClaw 和训练共用的同一个 Router/同一批 2 GPU worker**，不是两条互相独立的路径——"两条路径完全独立、不共享 worker"这个假设已被排除。

**结论：** "OpenClaw 这条路的请求被拒绝 → 空出容量 → 训练自己的 rollout 变快（对应 perf 13/14 异常）"这个因果方向，从"架构上说得通但没实锤"变成**架构上已确认成立**（两者本就共用资源）。但**还没查清楚的是：为什么熔断/拒绝只打在 OpenClaw 这条路上，训练自己的 rollout 请求不但没被拒，反而变快**——如果是同一个 Router 的同一个熔断器，理论上应该无差别拒绝所有调用方，不该看请求来源区别对待。这需要 SGLang Router 内部实现细节（`policy='cache_aware'` 路由策略是否按 worker 分别统计失败率、熔断器统计维度是按 Router 全局还是按 worker/client 分组）才能查清，静态读代码到这已经是本地能查到的极限，需要 SGLang 包自己的源码或运行时状态才能继续。

**最终更新（彻底找到真正根因——不是 SGLang Router 熔断器，是我们自己代码里一个正常设计好的暂停机制，且找到了具体触发原因）：** 用户要求"先加日志能抓到这个问题，再重新跑"。检查加日志之前，先看 `openclaw_opd_api_server.py` 里已有的诊断日志（`_handle_request` 里 `if sglang_resp.status_code != 200: logger.error("[OpenClaw-OPD] SGLang returned %d: %s", ...)`），查 `training.log` 里这条日志——**零匹配**。说明这次的 503 根本没有走到"SGLang 返回了非 200 响应"这一步，是在更早的地方就被拦下了。

顺着这个线索查同一个文件的 `/v1/chat/completions` 入口（`openclaw_opd_api_server.py:334-345`），找到真正的根源：

```python
@app.post("/v1/chat/completions")
async def chat_completions(...):
    ...
    if not owner.submission_enabled.is_set():
        raise HTTPException(status_code=503, detail="submission paused for weight update")
```

这是**我们自己代码里明确设计好的行为**：`openclaw_combine_select_rollout.py:85-94` 的 `pause_submission()`/`resume_submission()`，在**每一轮训练步骤里，rollout 攒够样本之后、训练计算（log_probs+反向传播+权重同步）完成之前**，会主动暂停接受新的 `/v1/chat/completions` 请求，直接秒返回 503。训练自己的 rollout `generate()` 函数（`openclaw_opd_api_server.py:232`）是 slime 进程内直接 Python 调用，根本不走这个 HTTP 端点、不受这个开关影响——**这才是"为什么只拒绝 OpenClaw、训练自己完全不受影响"的真正原因，跟 SGLang Router 的熔断器/队列容量完全无关，之前那条排查方向整体排除**。

**精确定位这次的暂停时长（查 `"submission paused"`/`"submission resumed"` 这两行 print 日志的时间戳）：**

| 暂停 | 开始 | 结束（下一次 `update_weights_from_distributed` 完成后立即触发 resume）| 时长 | 对应失败 |
|---|---|---|---|---|
| 第一次 | 15:06:56（`perf 13` 收集完成后）| 15:07:34 | **~38 秒** | Student 49题、TA 0题、Teacher 0题turn3（INIT 阶段）|
| 第二次 | 15:09:52（`perf 14` 收集完成后）| 15:11:47 | **~115 秒** | student-hw-0-83882（Joint 阶段）|

之前以为是一次连续 3.5 分钟的风暴，实际是**两次独立的暂停**，中间有约 2 分钟正常运行的间隙——因为最初只看了全部 503 的最早/最晚时间戳，没有按暂停周期拆开看。

**为什么这次会触发、之前 8 小时的 run 没触发：** 这个暂停机制是**每一步训练都会发生的正常设计**（这次 run 里已经有十几次同样的 pause/resume 配对），不是新问题。但 Student/TA/Teacher 脚本（`student_chat.py`/`TA_chat.py`/`teacher_chat.py`）的重试预算只有 **3 次、1s+2s+4s=7 秒**——平时如果暂停只有几秒钟，这个重试预算早就悄悄扛过去了，请求最终成功，不会有任何可见异常。**这次这两次暂停异常地长（38秒、115秒），远超 7 秒的重试预算，才第一次把这个一直存在、平时被重试悄悄掩盖掉的机制暴露成了可见的崩溃**——所谓"这次运行本身有特殊情况"，特殊之处就是这两步训练计算耗时明显偏长（原因待查：可能是这两步本身样本更难/更长，或者服务器资源竞争，暂不深究，不影响修复方向）。

**修复方向（待用户确认）：** 加长 Student/TA/Teacher 脚本的重试预算/退避策略，让它能扛住这类正常但偶尔偏长的暂停窗口（比如把重试总窗口从 7 秒拉长到 2-3 分钟）。这不是"绕开问题的保险措施"，是**直接针对这个已经查清楚、且是官方设计本身自带的暂停机制做的合理适配**——暂停期间收到 503 是预期行为，客户端理应能扛住，扛不住才是真正的缺陷。

**已实施：** `student_chat.py`/`TA_chat.py`/`teacher_chat.py` 官方自带 `--max-retries` CLI 参数（默认 3），不需要碰 Python 源码，只在调用处多传一个参数即可。改成 `--max-retries 8`（退避算法不变，仍是 `2**attempt` 秒，总预算 1+2+4+8+16+32+64+128=255 秒，覆盖实测最长 115 秒暂停约 2.2 倍余量）。三处调用点都已更新：`train_with_services.sh`/`minitest_train_with_services.sh` 的共用函数 `run_one_persona()`，以及 `smoke_train_with_services.sh` 的 `run_smoke_chat()`。

**待验证：** 下次训练确认 `--max-retries 8` 生效后，类似时长的暂停窗口不再导致 Student/TA/Teacher 未捕获异常崩溃；同时继续观察这两步训练计算耗时异常偏长（38秒、115秒 vs 正常应该更短）的原因，虽然不影响这次的修复方向，但如果这类"异常慢的训练步"变得更频繁/更长，255 秒的新预算也可能不够用。

**最终更新（查清"暂停为什么这么长"——不是训练计算变慢了，是平时被流水线掩盖的固定耗时这次意外露出来了）：** 用户追问"到底为什么这次暂停这么久，有没有办法解决"。对比正常暂停（15:03:43→15:03:44，仅 1 秒）和异常暂停（15:06:56→15:07:34，38 秒）完整过程日志，找到关键数据：

```
Timer actor_train end (elapsed: 54.0s)
Timer train end (elapsed: 80.2s)
perf/log_probs_time: 24.5s
perf/actor_train_time: 54.0s
```

**核心洞察：actor 训练计算（log_probs + 反向传播，合计约 80 秒）在这个模型/GPU 配置下是每一步都要花的固定成本，不是这次异常变慢。** 正常情况下这约 80 秒的训练计算，是跟**下一轮 rollout 收集流水线重叠**着做的——rollout 收集正常要 300~500 秒，比 80 秒训练计算长得多，所以等收集攒够 16 个样本、真正触发 `pause_submission()` 的时候，训练计算早就在后台悄悄算完了，暂停窗口几乎瞬间恢复（对应查到的"正常暂停只有 1 秒"）。

**而触发这次风暴的两步（`perf 13`/`perf 14`，此前已查到 rollout 收集时间异常快：191.66 秒、138.20 秒，全场最短）——收集提前完成，训练计算这边还没来得及在后台默默算完，两者的时间差就变成了这次真正能观测到的暂停窗口（38秒、115秒，量级正好对上约 80 秒的训练计算耗时）。**

**完整因果链条（自洽，无遗留矛盾）：** 这两批 rollout 收集异常快 → 追上了平时被流水线掩盖的固定训练计算耗时 → `submission_enabled` 暂停窗口第一次变得肉眼可见地长 → 超出 Student/TA/Teacher 脚本 7 秒的默认重试预算 → 未捕获异常崩溃 → 触发 `run_init_phase()`/`run_joint_phase()` 已知的不阻塞缺陷继续放行 → Joint 阶段三个后台进程全灭、秒结束。

**"有没有办法解决"——真正需要解决的问题已经解决（重试预算加到 255 秒，见上）。** 剩下"这两批 rollout 收集为什么这么快"这个问题，更可能是正常的批次间波动（这一批的题目/对话内容恰好更快完成），不是需要修的 bug——固定的训练计算耗时和波动的收集耗时之间存在这种"谁先谁后"的关系，是这套异步流水线架构的固有特性，只要重试预算覆盖住训练计算的固定耗时上限，这类波动造成的暂停就不会再导致崩溃。

---

## [2026-07-17] run `20260717_171106`（8 小时后自动关闭）：发现两个新问题——context overflow 死循环、决策犹豫循环的新触发方式（非 tool_call 格式歧义）

**背景：** 用户回来查上次训练为什么只跑了 8 小时（17:11 启动，`training.log` 最后一条记录在次日 01:18，约 8 小时 7 分钟，跟目录 mtime 不完全一致但 log 内容才是真实结束时间），结尾卡在 `waiting for combine samples: 9/16, queue=0` 持续 19+ 分钟不再推进——是典型的"生成卡住→GPU空闲→平台自动回收"死法。往前查是哪里开始卡住的。

### 问题一：Problem 47 context overflow 死循环（新问题，此前未见过）

`openclaw.log` 显示：18:43:18 先有一次 108.8 秒的 `FailoverError: LLM request timed out.`；之后该 session 上下文涨到近 20000 token（`estimatedPromptTokens=19816`），超过 16384 token 的 prompt 预算（`overflowTokens=3432`）；系统尝试自动压缩瘦身但**每次都失败**，原因是 `already_compacted_recently`（压缩冷却期挡住）；压缩失败不影响 Student 脚本继续发新消息重试，**每次重试都会把新消息计入这个已经压不动的 session**（`messages=70→71→72` 持续增长），永远回不到预算内，形成死循环，8 轮内没有恢复。

### 问题二：Problem 49 起决策犹豫循环复现，但触发原因跟 execution-bias-fix 修的不是同一个

`training.log` 里两个 session（`student-hw-47/49-19019`）用 session-id 搜索**零匹配**——请求走的是 OpenClaw 内部重试路径（`incomplete turn detected ... stopReason=length ... reasoningRetries=0/2 emptyRetries=0/1 missingAssistantRetries=0/1 — surfacing error to user`），OpenClaw 自己内部重试耗尽后才把"⚠️ Agent couldn't generate a response"这个兜底文案以**正常 200 响应**返回给 Student——因为不是 408/503，Student 脚本自己的重试逻辑根本不会触发，只会当成"收到回复但没用"，正常推进到下一轮，8 轮全部耗光。这也解释了为什么这次 `training.log` 按 session-id 搜不到：需要按**时间窗口**去找对应的 `TRUNCATED (finish_reason=length)` 记录。

按时间窗口找到对应 session（`287f67eb-e7e5-49b9-8040-00ae352bbe4a`），`thinking=31699 chars`，`TRUNCATED (finish_reason=length)`。**查完整推理原文，确认是完全不同的循环模式**——不是"要不要给纯文本回复包 tool_call 标签"那种格式判定困惑，是**纯数学应用题理解上的反复自我重述**：模型反复用略微不同的措辞重新解释同一段已经理解过的条件（"第一阶段 6 个月 $8/月"），绕了十几段都没有真正推进到计算第二、第三阶段和最终总价，从头到尾没有出现 `tool_call`/Execution Bias 相关字眼。

**结论：** "决策犹豫循环"是一类问题的统称，不是单一原因——`execution-bias-fix` 补丁修复的只是其中一个已确认诱因（Execution Bias 章节造成的 tool_call 格式歧义），这次撞到的是**另一个独立诱因**（Qwen3-Thinking 在某些数学应用题上的固有倾向——反复自我重述条件而不收敛），补丁本身没有失效，只是覆盖范围不包括这一类触发方式。

**待查/待定：**
1. context overflow 死循环（问题一）是否是这次训练崩溃的真正开端——需要确认 `waiting for combine samples: 9/16` 卡住的批次里，是不是正好包含了 47/49 题这类无法在 8 轮内正常完成的 session
2. 数学应用题反复自我重述这类循环（问题二）要不要单独修——如果是 Qwen3-Thinking 本身的固有倾向，修复思路（是否也能用类似"追加消歧规则"或其他机制）还没评估，需要用户决定优先级
3. context overflow 的"压缩冷却期挡住重新压缩"（`already_compacted_recently`）这个机制本身是否有配置项可以调整，还没查

**更新（用户追问：50题以及之后是不是都跟49题同一种问题——统计规模+抽样后发现比预期严重得多，且至少存在两种不同性质的模式）：**

`incomplete turn detected ... stopReason=length` 这个模式（顶格截断的下游表现，OpenClaw 内部重试耗尽后返回"⚠️ Agent couldn't generate a response"）**整个 run 里总共出现 297 次**，从 Problem 49（18:57）持续到训练日志倒数第几分钟（次日 00:15），跨度超过 5 个小时——期间不断有**不同的** session（`cc93a2fa`→`915eacb9`→`89c0a0e7`→...→`876ae8d2`→`1ccf8b13`→...→`4c69c6b9` 等一长串）依次中招，不是同一个 session 卡住不放，是新开的 session 也会撞上同一个问题，说明触发条件不是这一道题本身，而是某种持续存在、影响后续所有 session 的系统性状态。

**抽样查最后一次出现（00:15:04，session=`4c69c6b9-...`）的完整推理原文，发现是第三种、性质完全不同的模式——不是 Problem 49 那种"语义连贯但反复自我重述"，是纯粹的 token 级别退化：推理原文就是"try try try try try..."这一个词一路重复到 8192 token 上限，没有任何语义内容。** 这更像是采样/解码层面的退化重复（degenerate repetition），不是内容理解或指令歧义问题，`execution-bias-fix` 这类"追加消歧规则"的修复思路对这种模式大概率无效。

**这次"try"退化样本提交（00:15:11，`submitted OPD+RL sample ... index=533 reward=-1.0 response_len=8197`）之后，`training.log` 紧接着就开始出现 `waiting for combine samples: 6/16`，此后一路卡顿最终演变成训练结尾那次永久卡死在 9/16——时间点直接衔接，这次"try"退化很可能就是压垮这次训练、导致最终收尾的直接导火索之一。**

**结论（重新评估问题严重程度）：** 这不是"Problem 49 一道题的孤立问题"，是从 Problem 49 起持续 5+ 小时、跨越几十个不同 session 的系统性大面积失控，且已确认背后至少有两种不同性质的触发模式（语义连贯的反复自我重述 / 纯 token 退化重复），可能还有更多没抽样到的模式。297 次太多，没法逐条核实，但已经能确认"决策犹豫循环"这个统称下至少包含 3 类独立诱因（execution-bias 的 tool_call 格式歧义已修复；数学应用题反复自我重述；纯 token 退化重复），后两类目前都还没有修复方案。

**更新（用户追问：这个 297 次的连锁是不是 49 题本身"传染"给后面所有题的——查证后确认不是字面意义上的传染，是同一种倾向独立反复触发）：** 查 Problem 49 结束后紧跟着的下一个内部 session（`cc93a2fa-...`，19:01:48 第一次 TRUNCATED）完整推理原文：题目内容是**热狗餐饮问题**，跟 49 题的流媒体订阅问题完全不同——是一道全新的题，不是 49 题状态延续下去的同一个上下文。但模式跟 49 题**完全一致**：语义连贯、反复重新解析同一句话（"Let's parse this... Wait, maybe it's that... Let's think..."），绕圈子不收敛。

**结论：不是"49 题的问题传染给了后面所有题"这种字面意义上的因果传导，是"反复自我重述不收敛"这种倾向本身会独立地在不同题目上反复触发**——不管题目内容是什么都可能撞上，更像是 Qwen3-Thinking 在这类应用题上的一种固有倾向在这段时间内反复独立发作，不是一次状态污染引发的连锁反应。而 00:15 那次"纯 token 退化"（"try try try..."）跟 49 题之间隔了 5 个多小时、中间发生过多次训练权重更新，目前没有证据支持这两者是同一条因果链，更可能是训练进行到更晚阶段才出现的、性质不同的另一个独立问题。

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
