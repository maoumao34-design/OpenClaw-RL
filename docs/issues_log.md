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
