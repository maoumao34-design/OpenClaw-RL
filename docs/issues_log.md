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
