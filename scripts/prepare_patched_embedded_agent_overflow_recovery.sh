#!/bin/bash
# Patch OpenClaw's core embedded-agent-runner overflow-recovery logic to stop
# treating "Already compacted" as a fatal failure, restoring March-era
# behavior (retry the prompt without additional compaction) instead of
# immediately giving up and surfacing "Context overflow" to the user.
#
# Root cause (see docs/issues_log.md, 2026-07-17/20 entries): OpenClaw's
# overflow-recovery retry loop (src/agents/embedded-agent-runner/run.ts) has
# two branches that both existed in the paper's original ~March 2026 version:
#   1. If this attempt already compacted once, don't compact again -- just
#      retry the prompt ("retrying prompt without additional compaction").
#   2. If this attempt has NOT compacted yet, attempt an explicit compaction.
# Branch 2 calls into src/agents/sessions/agent-session.ts's compact() method,
# which is hardcoded to mode="manual". That method's manual-mode gate
# (`if (isManual) throw new Error(lastEntry?.type === "compaction" ?
# "Already compacted" : ...)`) was added to OpenClaw between 2026-05-15 and
# 2026-06-01 -- confirmed via git archaeology, did not exist at the paper's
# submission time (2026-03-11, version 2026.3.8) or as late as 2026.5.16.
#
# The bug: branch 2's `hadAttemptLevelCompaction` check only looks at
# compactions that happened WITHIN the current attempt. If a PRIOR attempt
# (an earlier turn) already compacted this session, the session's last
# transcript entry is still type="compaction", but the NEW attempt's own
# counter is 0, so it takes branch 2 (explicit compact) -- which then hits
# the new May/June "manual" mode guard and throws "Already compacted". The
# catch handler treats this as a generic compaction failure, falls through
# past tool-result truncation (a no-op here since there are no oversized
# tool results to truncate), and gives up, surfacing "Context overflow:
# prompt too large for the model..." to the user. Because this happens
# every subsequent turn on a session that's already compacted (and Student/
# TA/Teacher's client keeps retrying with new messages that never shrink the
# context), this is a permanent, unrecoverable deadlock for that session.
#
# Confirmed real-world impact: this deadlock (Problem 47 in run
# 20260717_171106) produced 14 of 16 samples in one training batch (all
# near-duplicate retries of the same stuck session, growing prompt_len),
# polluting that batch's training signal. The resulting weight update
# preceded, by ~3 minutes, the onset of a 5+ hour cascade of unrelated
# sessions repeatedly hitting decision-paralysis-style stopReason=length
# failures -- i.e. this deadlock is the confirmed root trigger for
# training run 20260717_171106's eventual collapse, not merely a cosmetic
# per-problem annoyance.
#
# Fix: when the explicit-compaction attempt's catch block specifically
# catches "Already compacted", treat it exactly like branch 1 above (retry
# the prompt without further compaction) instead of falling through to the
# generic failure/give-up path. This is a strict behavioral restoration of
# what the paper's original OpenClaw version did (branch 1 is unmodified;
# this just routes one more case into it) -- bounded by the same
# MAX_OVERFLOW_COMPACTION_ATTEMPTS the existing retry loop already enforces,
# so it cannot loop forever; it just gets a genuine chance to recover
# instead of giving up on the very first hit.
#
# Why this patches a LIVE, hash-named bundle file instead of a repo-vendored
# copy: this is OpenClaw's own core embedded-agent-runner logic, bundled
# into a content-hashed dist file (not a clean per-plugin directory like
# extensions/sglang was), and there is no supported plugin extension point
# for this behavior. The exact bundle filename
# (embedded-agent-Cv16r2d1.js) is specific to this OpenClaw build (2026.6.9)
# and WILL change on any OpenClaw upgrade -- if this patch starts failing
# with "anchor not found", re-locate the file first via:
#   grep -rl "retrying prompt without additional compaction" /usr/lib/node_modules/openclaw/dist/
#
# Idempotency: always patches from a pristine backup
# (${LIVE_FILE}.orig-unpatched, created on first run) rather than the live
# file, so re-running this across multiple training script invocations never
# double-patches or drifts.
set -euo pipefail

LIVE_FILE=${1:?usage: prepare_patched_embedded_agent_overflow_recovery.sh <live_file> <dest_dir>}
DEST_DIR=${2:?usage: prepare_patched_embedded_agent_overflow_recovery.sh <live_file> <dest_dir>}
BACKUP_FILE="${LIVE_FILE}.orig-unpatched"

if [ ! -f "${LIVE_FILE}" ]; then
    echo "错误：找不到 embedded-agent-runner bundle ${LIVE_FILE}" >&2
    echo "这个文件名是内容哈希命名的，OpenClaw 升级后会变化，需要重新定位：" >&2
    echo "  grep -rl \"retrying prompt without additional compaction\" /usr/lib/node_modules/openclaw/dist/" >&2
    exit 1
fi

if [ ! -f "${BACKUP_FILE}" ]; then
    cp "${LIVE_FILE}" "${BACKUP_FILE}"
    echo "已备份未修改原文件 -> ${BACKUP_FILE}"
fi

mkdir -p "${DEST_DIR}"

python3 - "${BACKUP_FILE}" "${DEST_DIR}/embedded-agent-Cv16r2d1.js" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(src_path, encoding="utf-8").read()

marker = "openclaw-rl-overflow-recovery-patch"
if marker in text:
    raise SystemExit(
        "patch failed: patch marker already present in backup file "
        f"({src_path}) -- the backup may itself be already patched. "
        "Investigate before proceeding; do not blindly re-run."
    )

# Unique anchor: the log line inside the catch block that fires whenever the
# explicit overflow-recovery compaction call throws for any reason. Confirmed
# against the real deployed bundle (openclaw 2026.6.9,
# /usr/lib/node_modules/openclaw/dist/embedded-agent-Cv16r2d1.js) on
# 2026-07-20. If this string is missing, OpenClaw's bundle has changed
# upstream (or the hashed filename moved) and this patch needs to be
# re-verified against the new source before reapplying blindly.
anchor = (
    'log$1.warn(`contextEngine.compact() threw during overflow recovery '
    'for ${provider}/${modelId}: ${String(compactErr)}`);'
)
idx = text.find(anchor)
if idx == -1:
    raise SystemExit(
        "patch failed: expected anchor line not found in embedded-agent "
        "bundle (OpenClaw's overflow-recovery logic may have changed "
        "upstream -- re-verify this patch against the new source before "
        "reapplying)"
    )

line_start = text.rfind("\n", 0, idx) + 1
indent = text[line_start:idx]
anchor_end = idx + len(anchor)

# Inserted right after the existing "compact() threw" log line, before the
# generic `compactResult = { ok: false, ... }` failure assignment. If the
# caught error is specifically "Already compacted" (the manual-mode
# compaction guard added to OpenClaw between 2026-05-15 and 2026-06-01,
# absent from the paper's original ~March 2026 version), treat it exactly
# like the sibling "hadAttemptLevelCompaction" branch above: retry the
# prompt without attempting further compaction, instead of falling through
# to the generic compaction-failure / give-up path. Bounded by the same
# MAX_OVERFLOW_COMPACTION_ATTEMPTS the surrounding loop already enforces.
insertion = (
    f"\n{indent}if (String(compactErr).includes(\"Already compacted\")) {{"
    f"\n{indent}\tlog$1.warn(`context overflow persisted after already-compacted "
    f"session for ${{provider}}/${{modelId}}; retrying prompt without additional "
    f"compaction ({marker})`);"
    f"\n{indent}\tif (preflightRecovery?.source === \"mid-turn\") continueFromCurrentTranscript();"
    f"\n{indent}\tcontinue;"
    f"\n{indent}}}"
)

new_text = text[:anchor_end] + insertion + text[anchor_end:]

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(new_text)
print(f"patched -> {dest_path}")
PY

echo "已生成 embedded-agent overflow-recovery 补丁: ${DEST_DIR}/embedded-agent-Cv16r2d1.js"
