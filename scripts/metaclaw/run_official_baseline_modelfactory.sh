#!/bin/bash
# run_official_baseline_modelfactory.sh
#
# 一条命令跑完"官方代码路径基线"：官方 MetaClaw-Bench 的 `metaclaw-bench run`
# （infer → scoring → report 一条龙），文件落地已修复，其余一律不动。
#
# 这一格补的是什么
# ----------------
# 现有两个基线都答不了"完全按官方代码库的方法跑，基线是多少"：
#   - 17.8% / 0%（定版基线）：官方 metaclaw-bench run，但 Compl 恒 0
#   - 34.4% / 12.1%（K=0）  ：文件落地正确，但走的是我们自己的 driver
# 本脚本给的是中间那一格：官方方法 + 文件落地正确。
# 详见 docs/metaclaw_migration_plan.md「计划（2026-09-07）」。
#
# 本脚本不打补丁，只做校验
# ------------------------
# `--agent` 修复已经在 MetaClaw-official 的工作区里（四处全在，见下），
# 所以这里只**校验**它完整存在，缺任何一处就退出。不自动补——自动补会在
# 上游改动后悄悄补歪，而"悄悄补歪"正是 17.8%/0% 那次的教训。
#
# 四处缺一不可（缺一处则 agent_id 为 None，argv 里的 --agent 消失，
# 整条修复静默失效、行为与完全没修一模一样）：
#   L1  _run_openclaw_agent   argv 里 append --agent
#   L2  _run_question    -> _run_openclaw_agent   传 agent_id
#   L3  _run_group       -> _run_question         传 agent_id
#   L4  _run_group       -> _run_openclaw_agent（末轮 standalone feedback）
#
# modelfactory job 提交：
#   代码解释器: /bin/bash -i /dfs/data/start_tools.sh && /bin/bash -i
#   代码路径:   /dfs/data/openclaw-rl-project/OpenClaw-RL/scripts/metaclaw/run_official_baseline_modelfactory.sh
#   GPU 数量:   1（只需要一个 SGLang 推理实例，不训练）
#
# 需要外部先就位（本脚本会校验，不会代劳）：
#   - SGLang 已起，seed 465485731，BENCHMARK_BASE_URL 指向它
#   - OpenClaw 系统补丁 2-6 已部署；rl-training-headers **关闭**
#
# 环境变量：
#   METACLAW_ROOT        必填，MetaClaw-official 检出路径
#   BENCHMARK_BASE_URL   必填
#   BENCHMARK_API_KEY    必填
#   BENCHMARK_MODEL      必填
#   BENCH_WORKERS        默认 4   （-w；不影响分数，只影响墙钟与超时风险）
#   BENCH_RETRY          默认 3   （-n；官方 baseline_run.py 默认值，只在
#                                  进程失败/超时时重试，不会拿错答案重摇）
#   BASELINE_SMOKE       默认 1   （先跑 day30 冒烟；=0 跳过直接全量）
#   BASELINE_SMOKE_ONLY  默认 0   （=1 只跑冒烟就停，用来先探压缩/溢出）
#   OUT_ROOT             默认 ${METACLAW_ROOT}/benchmark/results

set -euo pipefail

SCRIPTS_DIR=$(dirname "$(realpath "$0")")
OPENCLAW_RL_ROOT=$(cd "${SCRIPTS_DIR}/../.." && pwd)
OPENCLAW_RL_GIT_SHA=$(cd "${OPENCLAW_RL_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo unknown)

METACLAW_ROOT="${METACLAW_ROOT:?METACLAW_ROOT 必填：MetaClaw-official 检出路径}"
export METACLAW_ROOT

: "${BENCHMARK_BASE_URL:?BENCHMARK_BASE_URL 必填}"
: "${BENCHMARK_API_KEY:?BENCHMARK_API_KEY 必填}"
: "${BENCHMARK_MODEL:?BENCHMARK_MODEL 必填}"

BENCH_WORKERS="${BENCH_WORKERS:-4}"
BENCH_RETRY="${BENCH_RETRY:-3}"
BASELINE_SMOKE="${BASELINE_SMOKE:-1}"
BASELINE_SMOKE_ONLY="${BASELINE_SMOKE_ONLY:-0}"
OUT_ROOT="${OUT_ROOT:-${METACLAW_ROOT}/benchmark/results}"

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUT_ROOT}/official_baseline_${RUN_TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

DATA_DIR="${METACLAW_ROOT}/benchmark/data/metaclaw-bench"
ALL_TESTS="${DATA_DIR}/all_tests.json"
INFER_CMD="${METACLAW_ROOT}/benchmark/src/infer/infer_cmd.py"

echo "==================================================================="
echo " 官方代码路径基线"
echo " 输出目录 : ${RUN_DIR}"
echo " -w / -n  : ${BENCH_WORKERS} / ${BENCH_RETRY}"
echo "==================================================================="

# =====================================================================
# 1. 校验 --agent 修复四处齐全（缺一处整条静默失效）
# =====================================================================
echo "[1/6] 校验 --agent 修复 ..."

python3 - "$INFER_CMD" <<'PYEOF'
import io, re, sys

path = sys.argv[1]
src = io.open(path, encoding="utf-8").read()
lines = src.split("\n")


def enclosing(idx):
    """Name of the def that encloses 1-based line *idx*."""
    for j in range(idx - 1, 0, -1):
        m = re.match(r"^(?:async )?def (\w+)", lines[j - 1])
        if m:
            return m.group(1)
    return "<module>"


problems = []

# L1: --agent must reach the openclaw subprocess argv.
if '"--agent"' not in src:
    problems.append('L1: infer_cmd.py 里根本没有 "--agent" 字面量')

# L2/L3/L4: every call that must forward agent_id, identified by the
# function it sits in -- NOT by line number. (Matching on line number is
# how this check was got wrong once: infer_cmd.py has an
# `_execute_update(agent_id=agent_id, ...)` call inside _run_group that
# looks identical to the one we care about.)
want = {
    ("_run_question", "_run_openclaw_agent"): "L2",
    ("_run_group", "_run_question"): "L3",
    ("_run_group", "_run_openclaw_agent"): "L4",
}
seen = {}
for i, line in enumerate(lines, 1):
    m = re.search(r"(?:await\s+)?(_run_openclaw_agent|_run_question)\(\s*$", line)
    if not m:
        continue
    callee, caller = m.group(1), enclosing(i)
    key = (caller, callee)
    if key not in want:
        continue
    # scan the call's argument block for agent_id=
    depth, forwarded = 1, False
    for k in range(i, min(i + 40, len(lines))):
        seg = lines[k]
        if re.search(r"\bagent_id\s*=", seg):
            forwarded = True
        depth += seg.count("(") - seg.count(")")
        if depth <= 0:
            break
    seen.setdefault(key, []).append(forwarded)

for key, label in want.items():
    got = seen.get(key)
    if not got:
        problems.append(f"{label}: 找不到 {key[0]}() 里对 {key[1]}() 的调用")
    elif not all(got):
        problems.append(f"{label}: {key[0]}() -> {key[1]}() 有调用未传 agent_id")

if problems:
    print("FATAL: --agent 修复不完整，跑下去会得到一个静默无效的基线：")
    for p in problems:
        print("  - " + p)
    print("\n  修复必须四处齐全（L1 argv / L2 / L3 / L4）。缺任何一处，")
    print("  agent_id 一路默认成 None，argv 里的 --agent 消失，文件重新")
    print("  落进 workspace-main/，Compl 会恒为 0——这正是定版基线 17.8%/0%")
    print("  的成因。参见 docs/metaclaw_migration_plan.md。")
    raise SystemExit(1)

print("  ok  L1/L2/L3/L4 四处齐全")
PYEOF

# =====================================================================
# 2. 记录 MetaClaw-official 的实际状态（含非官方改动）
# =====================================================================
echo "[2/6] 记录环境 ..."

MC_SHA=$(cd "${METACLAW_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo unknown)
(cd "${METACLAW_ROOT}" && git --no-pager diff) > "${RUN_DIR}/metaclaw_official.diff" 2>/dev/null || true
MC_DIFF_SHA=$(sha256sum "${RUN_DIR}/metaclaw_official.diff" 2>/dev/null | cut -c1-12 || echo none)

command -v metaclaw-bench >/dev/null 2>&1 || {
    echo "FATAL: metaclaw-bench 不在 PATH。先 cd ${METACLAW_ROOT}/benchmark && pip install -e ." >&2
    exit 1
}

cat > "${RUN_DIR}/RUN_MANIFEST.txt" <<EOF
run: 官方代码路径基线（metaclaw-bench run，文件落地已修）
started: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
openclaw-rl commit: ${OPENCLAW_RL_GIT_SHA}
metaclaw-official commit: ${MC_SHA}
metaclaw-official 工作区改动: metaclaw_official.diff (sha256:${MC_DIFF_SHA})
  -- 已知并**有意保留**的两类非官方改动：
     (a) --agent 四处修复（文件落地，缺之则 Compl 恒 0）
     (b) _wait_for_gateway timeout 10.0 -> 30.0（只影响等网关启动，
         不触及任何评分逻辑；服务器负载高时 10s 会误判失败）
model: ${BENCHMARK_MODEL}
base_url: ${BENCHMARK_BASE_URL}
dataset: ${ALL_TESTS}
workers (-w): ${BENCH_WORKERS}
retry   (-n): ${BENCH_RETRY}
smoke: ${BASELINE_SMOKE} (only=${BASELINE_SMOKE_ONLY})
EOF
cat "${RUN_DIR}/RUN_MANIFEST.txt"

# =====================================================================
# 共用：跑一次 bench + 抓三个计数
# =====================================================================
run_bench() {
    local input="$1" out="$2" workers="$3" log="$4"
    mkdir -p "${out}"
    echo "  metaclaw-bench run -i $(basename "${input}") -w ${workers} -n ${BENCH_RETRY}"
    set +e
    metaclaw-bench run -i "${input}" -o "${out}" -w "${workers}" -n "${BENCH_RETRY}" 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    set -e
    return ${rc}
}

report_counters() {
    local log="$1"
    echo "  --- 压缩 / 溢出 / infra 计数 ---"
    echo "  compaction-skip : $(grep -c "openclaw-rl-cli-compaction-patch" "${log}" 2>/dev/null || echo 0)"
    echo "  context overflow: $(grep -ci "context overflow\|context_length\|exceeds.*context" "${log}" 2>/dev/null || echo 0)"
    echo "  infra 失败      : $(grep -c "Exit code\|Timeout after" "${log}" 2>/dev/null || echo 0)"
    echo "  提示：三个数都小 → 结果可直接归因；任一显著 → 该次 Compl 只能当上界。"
}

# Compl 官方 report 不出，必须自己从 scoring.json 算。
# 判别方式：只有 _score_file_check 会写 metrics.passed；multi_choice 写的是
# iou/precision/recall/f1/exact_match。所以"有 passed 键"即 file_check。
report_compl() {
    local out="$1"
    python3 - "$out" <<'PYEOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.rglob("scoring.json"))
if not files:
    print("  [warn] 没找到 scoring.json —— scoring 阶段可能没跑到")
    raise SystemExit(0)

total = passed = 0
mc_n = 0
for f in files:
    try:
        rec = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue
    for r in (rec if isinstance(rec, list) else [rec]):
        metrics = r.get("metrics") or {}
        if "passed" in metrics:
            total += 1
            passed += 1 if metrics["passed"] else 0
        else:
            mc_n += 1

print(f"  scoring.json 总数: {len(files)}  (file_check {total} / multi_choice {mc_n})")
if total:
    print(f"  Compl. (file-check completion) = {passed}/{total} = {passed/total:.1%}")
else:
    print("  [warn] 一个 file_check 都没识别出来——把任意一个 scoring.json 贴出来核对结构")
PYEOF
}

# =====================================================================
# 3. day30 冒烟（15 轮，全库最长的一天 + 难度最高 → context 最坏情况）
# =====================================================================
if [[ "${BASELINE_SMOKE}" == "1" ]]; then
    echo "[3/6] day30 冒烟 ..."
    SMOKE_INPUT="${RUN_DIR}/day30_only.json"
    python3 - "$ALL_TESTS" "$SMOKE_INPUT" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
d["test"] = [t for t in d["test"] if t["id"] == "day30"]
if not d["test"]:
    raise SystemExit("FATAL: all_tests.json 里没有 day30")
json.dump(d, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"  冒烟输入已生成：1 天")
PYEOF

    SMOKE_OUT="${RUN_DIR}/smoke_day30"
    SMOKE_LOG="${LOG_DIR}/smoke_day30.log"
    run_bench "${SMOKE_INPUT}" "${SMOKE_OUT}" 1 "${SMOKE_LOG}" || {
        echo "FATAL: 冒烟失败，不进全量。日志：${SMOKE_LOG}" >&2
        report_counters "${SMOKE_LOG}"
        exit 1
    }

    echo "[4/6] 冒烟判读 ..."
    report_counters "${SMOKE_LOG}"
    report_compl "${SMOKE_OUT}"

    # -----------------------------------------------------------------
    # 文件落地实证 —— 否定侧 + 正向侧都要查
    #
    # 只查"workspace-main 里没有新文件"是不够的：如果模型压根没写成功，
    # 它同样会通过。必须同时确认文件**确实出现在了 checker 会读的那个
    # 副本里**。这两侧合起来才能区分三种情况：
    #   落对了 / 落进 workspace-main / 根本没写
    # -----------------------------------------------------------------
    echo "  --- 文件落地实证 ---"

    STRAY=$(find "${METACLAW_ROOT}" -path "*workspace-main*" -type f \
            -newer "${RUN_DIR}/RUN_MANIFEST.txt" 2>/dev/null | head -5)
    LANDED=$(find "${METACLAW_ROOT}" -path "*workspace_day30_*/day30/*" -type f \
             -newer "${RUN_DIR}/RUN_MANIFEST.txt" 2>/dev/null | head -20)
    LANDED_N=$(printf '%s\n' "${LANDED}" | grep -c . || true)

    if [[ -n "${STRAY}" ]]; then
        echo "  FATAL: 文件落进了 workspace-main/ —— --agent 在 bench 路径上没生效：" >&2
        printf '    %s\n' ${STRAY} >&2
        echo "  （这正是定版基线 Compl=0 的成因，不要继续跑全量。）" >&2
        exit 1
    fi
    echo "  ok  workspace-main/ 无新文件"

    if [[ "${LANDED_N}" -eq 0 ]]; then
        echo "  FATAL: checker 读的 workspace_day30_*/day30/ 里一个新文件都没有。" >&2
        echo "  两侧都空 = 模型根本没写成功，不是落地问题。先看 ${SMOKE_LOG}：" >&2
        echo "    - 是不是每轮都 context overflow / infra 失败（见上面计数）" >&2
        echo "    - session transcript 里有没有 'Successfully wrote N bytes'" >&2
        exit 1
    fi
    echo "  ok  workspace_day30_*/day30/ 下有 ${LANDED_N} 个新文件 → 落地正确"

    if [[ "${BASELINE_SMOKE_ONLY}" == "1" ]]; then
        echo "BASELINE_SMOKE_ONLY=1，到此为止。结果：${SMOKE_OUT}"
        exit 0
    fi
else
    echo "[3/6] 跳过冒烟（BASELINE_SMOKE=0）"
    echo "[4/6] —"
fi

# =====================================================================
# 5. 全量 30 天
# =====================================================================
echo "[5/6] 全量 30 天 / 346 题 ..."
FULL_OUT="${RUN_DIR}/full"
FULL_LOG="${LOG_DIR}/full.log"
run_bench "${ALL_TESTS}" "${FULL_OUT}" "${BENCH_WORKERS}" "${FULL_LOG}" || {
    echo "WARN: metaclaw-bench 返回非 0；已产出的部分仍会汇总。" >&2
}

# =====================================================================
# 6. 汇总
# =====================================================================
echo "[6/6] 汇总 ..."
report_counters "${FULL_LOG}"
report_compl "${FULL_OUT}"

if [[ -f "${FULL_OUT}/report.json" ]]; then
    python3 - "${FULL_OUT}/report.json" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8")).get("summary", {})
print(f"  Acc. = {s.get('correct',0):.3f}/{s.get('total_questions',0)}"
      f" = {s.get('accuracy',0):.1%}")
PYEOF
else
    echo "  [warn] 没有 report.json（${FULL_OUT}）"
fi

echo
echo "==================================================================="
echo " 完成。结果目录: ${RUN_DIR}"
echo " 对照：定版基线 17.8% / 0%（同为官方 metaclaw-bench run，落地未修）"
echo "       我们 driver 的 scope=day K=0（day01-17）49.9% / 36.3%"
echo "==================================================================="
