#!/usr/bin/env bash
# 在独立 GPU 机器上启动 Qwen3-32B Simulator（供 train_with_services.sh 远程调用）
#
# 本脚本不在训练 job 内运行。先在 Simulator 机器上跑通，再提交训练 job。
#
# 必需：
#   MODEL_PATH=/path/to/Qwen3-32B
#
# 可选：
#   HOST=0.0.0.0
#   PORT=30001
#   TP_SIZE=1
#   MAX_TOKENS=16384
#   MODEL_NAME=qwen3-32b
#   SGLANG_API_KEY=your-secret        # 训练 job 里 SIMULATOR_API_KEY 需一致
#   MEM_FRACTION_STATIC=0.90
#
# 启动后验证（任意能访问该机器的终端）：
#   curl http://<simulator-host>:30001/health
#
# 训练 job 环境变量示例：
#   export SIMULATOR_BASE_URL=http://<simulator-host>:30001/v1
#   export SIMULATOR_API_KEY=your-secret
#   export EXTERNAL_MODEL=qwen3-32b

set -euo pipefail

if [ -z "${MODEL_PATH:-}" ]; then
    echo "Error: MODEL_PATH is not set." >&2
    echo "Usage: MODEL_PATH=/path/to/Qwen3-32B bash $0" >&2
    exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30001}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MODEL_NAME="${MODEL_NAME:-qwen3-32b}"
MEM_FRACTION="${MEM_FRACTION_STATIC:-0.90}"
API_KEY="${SGLANG_API_KEY:-}"

API_KEY_ARGS=()
if [ -n "${API_KEY}" ]; then
    API_KEY_ARGS=(--api-key "${API_KEY}")
fi

echo "============================================"
echo "  External Simulator (Qwen3-32B)"
echo "  Model:       ${MODEL_PATH}"
echo "  Listen:      ${HOST}:${PORT}"
echo "  TP:          ${TP_SIZE}"
echo "  Context:     ${MAX_TOKENS}"
echo "  Served name: ${MODEL_NAME}"
echo "  API key:     ${API_KEY:+set}${API_KEY:-none}"
echo "============================================"
echo ""
echo "训练 job 请设置："
echo "  SIMULATOR_BASE_URL=http://<this-host>:${PORT}/v1"
echo "  SIMULATOR_API_KEY=${API_KEY:-EMPTY}"
echo "  EXTERNAL_MODEL=${MODEL_NAME}"
echo ""

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --context-length "${MAX_TOKENS}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --served-model-name "${MODEL_NAME}" \
    "${API_KEY_ARGS[@]}"
