#!/bin/bash
# Patch openclaw-test/{student,TA,teacher}_chat.py to use the agent-target
# model format ("openclaw/<agentId>") that the installed OpenClaw CLI expects.
#
# The official scripts hardcode `"model": "default"`. OpenClaw 2026.6.9's
# /v1/chat/completions endpoint rejects that with:
#   "Invalid `model`. Use `openclaw` or `openclaw/<agentId>`."
# (see docs/gateway/openai-http-api.md in the openclaw repo -- "default" was
# never a documented agent-target alias, only "openclaw", "openclaw/default",
# "openclaw/<agentId>", "openclaw:<agentId>", "agent:<agentId>").
#
# This only rewrites the literal model-field string; no training logic,
# reward, or data path is touched. The official openclaw-test/ directory is
# left untouched -- this writes a patched copy to DEST_DIR instead.
set -euo pipefail

REPO_ROOT=${1:?usage: prepare_openclaw_test_scripts.sh <repo_root> <dest_dir>}
DEST_DIR=${2:?usage: prepare_openclaw_test_scripts.sh <repo_root> <dest_dir>}
SRC_DIR="${REPO_ROOT}/openclaw-test"

mkdir -p "${DEST_DIR}"

for f in student_chat.py TA_chat.py teacher_chat.py; do
    sed 's/"model": "default"/"model": "openclaw\/default"/' \
        "${SRC_DIR}/${f}" > "${DEST_DIR}/${f}"
done

if [ ! -e "${DEST_DIR}/GSM8K.json" ]; then
    ln -sf "${SRC_DIR}/GSM8K.json" "${DEST_DIR}/GSM8K.json"
fi

echo "已生成兼容 patch: ${DEST_DIR}（model: default -> openclaw/default）"
