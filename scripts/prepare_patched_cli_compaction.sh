#!/bin/bash
# Patch OpenClaw's core "cli_budget"-triggered CLI-turn compaction lifecycle
# (src/agents/command/cli-compaction.ts) to stop treating "Already compacted"
# as a fatal error that corrupts an otherwise-successful turn.
#
# Root cause (see docs/issues_log.md, 2026-07-21 entry): OpenClaw's
# cli-compaction.ts runs a "cli_budget"-triggered compaction check as part
# of normal CLI-turn housekeeping (this whole file/mechanism did not exist
# at the paper's 2026-03-08 submission snapshot -- confirmed via git
# archaeology, first appears by 2026-06-09). It has two call sites (native
# harness path and context-engine path) that BOTH unconditionally
# `throw new Error(...)` whenever the underlying compaction attempt fails
# for ANY reason -- including "Already compacted" (the same manual-mode
# compaction guard, added to OpenClaw between 2026-05-15 and 2026-06-01,
# that we already patched a DIFFERENT call site for via
# prepare_patched_embedded_agent_overflow_recovery.sh). Unlike that other
# call site (src/agents/embedded-agent-runner/run.ts's overflow-recovery
# loop), this one has no graceful-degradation branch at all.
#
# The bug: this compaction check runs as part of normal turn housekeeping,
# AFTER the actual generation/tool-call work for that turn has already
# completed successfully. When it hits "Already compacted", the resulting
# thrown error corrupts the HTTP response for that turn into a generic
# "internal error" -- even though the real work (e.g. a file write via the
# edit/write tool) already genuinely happened. The client (student_chat.py
# in our training pipeline) retries on this error; the retry re-sends the
# same instruction, and since the underlying work is already done from the
# lost-response attempt, the retry's (successful) response accurately
# reports "already done" / "no further changes needed" -- this is NOT the
# model lying, it's an accurate observation of session state the client
# never got confirmation of. Confirmed via a live debug-level diagnostic
# (2026-07-21): a manually-driven two-turn conversation against a real
# Qwen3-4B-Thinking sglang backend showed the target file was genuinely
# written correctly on disk, while BOTH HTTP responses returned
# {"error":{"message":"internal error"}}, with the debug log showing
# `[compaction-diag] ... trigger=cli_budget ... outcome=failed
# reason=already_compacted_recently` firing right after each turn's
# generation completed.
#
# Downstream training impact: because the PRM/judge reward signal used
# during training only evaluates whether the conversational partner seems
# satisfied (see docs/issues_log.md's PRM eval prompt findings -- it has no
# way to verify ground-truth task completion), these short "already done"
# retry responses get rewarded identically to genuine, verified completions.
# Over many training steps this reinforces the model toward this cheap
# phrasing pattern, which eventually over-generalizes to genuinely
# incomplete turns (producing what looked like the model fabricating
# completion claims out of nothing).
#
# Fix: at both throw sites, check whether the failure reason specifically
# mentions "Already compacted"; if so, log a warning and continue without
# throwing (the turn's own generation/tool-call work already succeeded and
# should not be invalidated by a housekeeping compaction attempt failing).
# Any OTHER compaction failure reason still throws exactly as before --
# this is a narrow, targeted exception, not a blanket suppression of
# compaction-failure errors.
#
# Why this patches a LIVE, hash-named bundle file instead of a repo-vendored
# copy: same reasoning as the sibling embedded-agent-runner patch -- this is
# OpenClaw's own core CLI-turn compaction logic, bundled into a
# content-hashed dist file, with no supported plugin extension point. The
# exact bundle filename (cli-compaction-B6C2IDnn.js) is specific to this
# OpenClaw build and WILL change on any OpenClaw upgrade -- if this patch
# starts failing with "anchor not found", re-locate the file first via:
#   grep -rl "CLI transcript compaction failed for" /usr/lib/node_modules/openclaw/dist/
#
# Idempotency: always patches from a pristine backup
# (${LIVE_FILE}.orig-unpatched, created on first run) rather than the live
# file, so re-running this across multiple training script invocations never
# double-patches or drifts.
set -euo pipefail

LIVE_FILE=${1:?usage: prepare_patched_cli_compaction.sh <live_file> <dest_dir>}
DEST_DIR=${2:?usage: prepare_patched_cli_compaction.sh <live_file> <dest_dir>}
BACKUP_FILE="${LIVE_FILE}.orig-unpatched"

if [ ! -f "${LIVE_FILE}" ]; then
    echo "错误：找不到 cli-compaction bundle ${LIVE_FILE}" >&2
    echo "这个文件名是内容哈希命名的，OpenClaw 升级后会变化，需要重新定位：" >&2
    echo "  grep -rl \"CLI transcript compaction failed for\" /usr/lib/node_modules/openclaw/dist/" >&2
    exit 1
fi

if [ ! -f "${BACKUP_FILE}" ]; then
    cp "${LIVE_FILE}" "${BACKUP_FILE}"
    echo "已备份未修改原文件 -> ${BACKUP_FILE}"
fi

mkdir -p "${DEST_DIR}"

python3 - "${BACKUP_FILE}" "${DEST_DIR}/cli-compaction-B6C2IDnn.js" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(src_path, encoding="utf-8").read()

marker = "openclaw-rl-cli-compaction-patch"
if marker in text:
    raise SystemExit(
        "patch failed: patch marker already present in backup file "
        f"({src_path}) -- the backup may itself be already patched. "
        "Investigate before proceeding; do not blindly re-run."
    )

# Anchor 1: native harness compaction failure throw. Confirmed against the
# real deployed bundle (openclaw 2026.6.9,
# /usr/lib/node_modules/openclaw/dist/cli-compaction-B6C2IDnn.js) on
# 2026-07-21. If missing, OpenClaw's cli-compaction.ts has changed upstream
# and this patch needs re-verification against the new source.
anchor1 = (
    '} else if (nativeOutcome.failureReason) throw new Error(`CLI native harness '
    'compaction failed for ${params.provider}/${params.model}: '
    '${nativeOutcome.failureReason ?? "compaction did not reduce context"}`);'
)
count1 = text.count(anchor1)
if count1 != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of anchor1, found {count1}"
    )

# Anchor 2: context-engine compaction failure throw.
anchor2 = (
    'if (!compacted && contextOutcome.failureReason) throw new Error(`CLI transcript '
    'compaction failed for ${params.provider}/${params.model}: '
    '${contextOutcome.failureReason ?? "compaction did not reduce context"}`);'
)
count2 = text.count(anchor2)
if count2 != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of anchor2, found {count2}"
    )

idx1 = text.find(anchor1)
line_start1 = text.rfind("\n", 0, idx1) + 1
indent1 = text[line_start1:idx1]

# Replace the unconditional throw with a check: if this is specifically an
# "Already compacted" failure, log and continue (skip context-engine
# fallback too, since it would hit the same session-level guard); any other
# failure reason still throws exactly as before.
replacement1 = (
    f'}} else if (nativeOutcome.failureReason) {{'
    f'\n{indent1}\tif (nativeOutcome.failureReason.includes("Already compacted")) {{'
    f'\n{indent1}\t\tlog.warn(`CLI native harness compaction skipped (already compacted '
    f'recently) for ${{params.provider}}/${{params.model}}; continuing without '
    f'compaction ({marker})`);'
    f'\n{indent1}\t\tuseContextEngineCompaction = false;'
    f'\n{indent1}\t}} else throw new Error(`CLI native harness compaction failed for '
    f'${{params.provider}}/${{params.model}}: ${{nativeOutcome.failureReason ?? '
    f'"compaction did not reduce context"}}`);'
    f'\n{indent1}}}'
)

text = text[:idx1] + replacement1 + text[idx1 + len(anchor1):]

# Re-locate anchor2 after the first replacement shifted offsets.
idx2 = text.find(anchor2)
if idx2 == -1:
    raise SystemExit("patch failed: anchor2 not found after applying patch 1 (unexpected)")
line_start2 = text.rfind("\n", 0, idx2) + 1
indent2 = text[line_start2:idx2]

replacement2 = (
    f'if (!compacted && contextOutcome.failureReason) {{'
    f'\n{indent2}\tif (contextOutcome.failureReason.includes("Already compacted")) '
    f'log.warn(`CLI transcript compaction skipped (already compacted recently) for '
    f'${{params.provider}}/${{params.model}}; continuing without compaction ({marker})`);'
    f'\n{indent2}\telse throw new Error(`CLI transcript compaction failed for '
    f'${{params.provider}}/${{params.model}}: ${{contextOutcome.failureReason ?? '
    f'"compaction did not reduce context"}}`);'
    f'\n{indent2}}}'
)

new_text = text[:idx2] + replacement2 + text[idx2 + len(anchor2):]

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(new_text)
print(f"patched -> {dest_path}")
PY

echo "已生成 cli-compaction 补丁: ${DEST_DIR}/cli-compaction-B6C2IDnn.js"
