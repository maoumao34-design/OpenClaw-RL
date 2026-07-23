#!/bin/bash
# Patch openclaw-test/{student,TA,teacher}_chat.py:
#
#   Rewrite the literal `"model": "default"` field to `"model": "openclaw/default"`,
#   the agent-target format OpenClaw 2026.6.9's /v1/chat/completions endpoint
#   actually expects. This is a pure API-compatibility shim (this OpenClaw CLI
#   version's routing format, unrelated to the paper's method) -- without it
#   every request 400s immediately, nothing can run at all.
#
# 2026-07-23: the homework-verification-gate patch (deterministic file-write
# check + 32B recheck before honoring DONE_SENTINEL) that used to live here
# has been removed for this experiment, per explicit user request, to isolate
# whether the write/overwrite-compliance problems we kept hitting are actually
# caused by the external Simulator model (Qwen3-32B) itself being too weak/
# non-compliant to reliably notice its own mistakes, rather than a gap in the
# harness. This run therefore uses the *unmodified* official DONE_SENTINEL
# logic (no session-continuation gate at all) -- see docs/issues_log.md
# 2026-07-23 entry for the removal rationale and the full prior design
# (several real-data-driven revisions) if it ever needs to be restored.
#
# Reproduction-fidelity note: swapping the Simulator model away from Qwen3-32B
# (paper Section 4.1) makes this run NOT a valid Table 3 data point -- it is a
# diagnostic-only experiment to isolate the 32B model's contribution to the
# write-compliance failures, not a replacement for the official Table 3
# reproduction runs.
#
# This only rewrites a known, literal source line; no training logic, reward,
# or data path is touched. The official openclaw-test/ directory is left
# untouched -- this writes patched copies to DEST_DIR instead.
set -euo pipefail

REPO_ROOT=${1:?usage: prepare_openclaw_test_scripts.sh <repo_root> <dest_dir>}
DEST_DIR=${2:?usage: prepare_openclaw_test_scripts.sh <repo_root> <dest_dir>}
SRC_DIR="${REPO_ROOT}/openclaw-test"

mkdir -p "${DEST_DIR}"

if [ ! -e "${DEST_DIR}/GSM8K.json" ]; then
    ln -sf "${SRC_DIR}/GSM8K.json" "${DEST_DIR}/GSM8K.json"
fi

for filename in student_chat.py TA_chat.py teacher_chat.py; do
    src_path="${SRC_DIR}/${filename}"
    dest_path="${DEST_DIR}/${filename}"
    old_model='"model": "default"'
    count=$(grep -o "${old_model}" "${src_path}" | wc -l | tr -d ' ')
    if [ "${count}" != "1" ]; then
        echo "patch failed: expected exactly 1 occurrence of ${old_model} in ${filename}, found ${count}" >&2
        exit 1
    fi
    sed "s/${old_model}/\"model\": \"openclaw\/default\"/" "${src_path}" > "${dest_path}"
    echo "patched -> ${dest_path}"
done

echo "已生成 openclaw-test 补丁: ${DEST_DIR}（仅 model 字段兼容修复，homework-verification-gate 已移除，见 docs/issues_log.md 2026-07-23）"
