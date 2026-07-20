#!/bin/bash
# Patch OpenClaw's core system-prompt "## Assistant Output Directives" section
# to make its five directives explicitly conditional, restoring the spirit of
# March-era (2026.3.8, pre-paper-submission) behavior instead of the
# unconditional-checklist framing OpenClaw introduced later.
#
# Root cause (see docs/issues_log.md, 2026-07-20 entry): OpenClaw's
# src/agents/system-prompt.ts::buildAssistantOutputDirectivesSection() emits
# (for our config: isMinimal=false, sourceMessageToolOnly=false) five bullet
# directives -- MEDIA: attachment syntax, audio_as_voice hint, reply_to_current
# tag, etc. -- with NO framing telling the model these are opt-in / only
# relevant when the corresponding situation actually applies. Confirmed via
# git archaeology: this section (or its unconditional five-bullet form) was
# added to OpenClaw between 2026-04-08 (absent) and 2026-04-15 (present,
# v2026.4.15-beta.1) -- after the paper's 2026-03-11 submission / 2026.3.8
# snapshot, which instead had a much narrower, explicitly conditional
# "## Reply Tags" section (system-prompt.ts:109, paper-era) covering only
# reply_to_current/reply_to:<id>, framed as "To request a native reply/quote
# on supported surfaces, include one tag in your reply" -- i.e. clearly
# optional, not a checklist to verify every turn.
#
# Confirmed real-world impact: two independent reasoning_text samples in run
# 20260720_112802 (12:44 and 13:44) show the policy repeatedly, unproductively
# re-deriving whether MEDIA/voice-hint/reply_to_current apply to a plain-text
# math reply, never converging -- the same decision-paralysis pattern
# previously confirmed for the Execution Bias section (tool_call-wrapping
# ambiguity, fixed via the sglang provider-hook patch) and for the
# context-overflow "Already compacted" deadlock (fixed via the
# embedded-agent-runner patch). This is the THIRD independent, confirmed
# OpenClaw-version-drift-introduced trigger of this same failure mode.
# Same batch-pollution mechanism confirmed again: one struggling session's
# retried samples dominated 37.5% of a single 16-sample training batch.
#
# Fix: insert one clarifying sentence immediately after the section heading,
# in the sourceMessageToolOnly=false variant only (the only variant confirmed
# active in our training config), telling the model these directives are
# conditional and that a plain reply needs no checking against this list.
# This mirrors the disambiguation approach already used for Execution Bias,
# but must be applied as a direct core-bundle content edit (not via the
# `resolveSystemPromptContribution`/sectionOverrides provider hook), because
# ProviderSystemPromptSectionId only supports "interaction_style" /
# "tool_call_style" / "execution_bias" -- there is no override slot for the
# Assistant Output Directives section.
#
# Why this patches a LIVE, hash-named bundle file instead of a repo-vendored
# copy: this is OpenClaw's own core system-prompt assembly logic, bundled
# into a content-hashed dist file (not a clean per-plugin directory like
# extensions/sglang was), and there is no supported plugin extension point
# for this content. The exact bundle filename
# (system-prompt-config-CLAPATdy.js) is specific to this OpenClaw build and
# WILL change on any OpenClaw upgrade -- if this patch starts failing with
# "anchor not found", re-locate the file first via:
#   grep -rl "Attach media in the final visible reply" /usr/lib/node_modules/openclaw/dist/
#
# Idempotency: always patches from a pristine backup
# (${LIVE_FILE}.orig-unpatched, created on first run) rather than the live
# file, so re-running this across multiple training script invocations never
# double-patches or drifts.
set -euo pipefail

LIVE_FILE=${1:?usage: prepare_patched_system_prompt_output_directives.sh <live_file> <dest_dir>}
DEST_DIR=${2:?usage: prepare_patched_system_prompt_output_directives.sh <live_file> <dest_dir>}
BACKUP_FILE="${LIVE_FILE}.orig-unpatched"

if [ ! -f "${LIVE_FILE}" ]; then
    echo "错误：找不到 system-prompt bundle ${LIVE_FILE}" >&2
    echo "这个文件名是内容哈希命名的，OpenClaw 升级后会变化，需要重新定位：" >&2
    echo "  grep -rl \"Attach media in the final visible reply\" /usr/lib/node_modules/openclaw/dist/" >&2
    exit 1
fi

if [ ! -f "${BACKUP_FILE}" ]; then
    cp "${LIVE_FILE}" "${BACKUP_FILE}"
    echo "已备份未修改原文件 -> ${BACKUP_FILE}"
fi

mkdir -p "${DEST_DIR}"

python3 - "${BACKUP_FILE}" "${DEST_DIR}/system-prompt-config-CLAPATdy.js" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(src_path, encoding="utf-8").read()

# Content-based idempotency check: look for the exact sentence we insert
# below, rather than a separate marker token, since the inserted text
# itself becomes part of the live system prompt shown to the model (unlike
# the embedded-agent-runner patch's marker, which only ever appears in a
# log line).
sentinel = "do not enumerate or re-check this list"
if sentinel in text:
    raise SystemExit(
        "patch failed: sentinel text already present in backup file "
        f"({src_path}) -- the backup may itself be already patched. "
        "Investigate before proceeding; do not blindly re-run."
    )

# Unique anchor: the first bullet of the sourceMessageToolOnly=false variant
# (the raw MEDIA:/[[audio_as_voice]]/[[reply_to_current]] directive framing
# -- confirmed as the only variant active in our training config). Confirmed
# against the real deployed bundle (openclaw build with hash CLAPATdy) on
# 2026-07-20 via `sed -n '255,300p' system-prompt-config-CLAPATdy.js`. If
# this string is missing or appears more than once, OpenClaw's system-prompt
# section has changed upstream and this patch needs to be re-verified
# against the new source before reapplying blindly.
anchor = '"- Attach media in the final visible reply with `MEDIA:<path-or-url>` on its own line.",'
count = text.count(anchor)
if count != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of anchor, found {count} "
        "(OpenClaw's Assistant Output Directives section may have changed "
        "upstream -- re-verify this patch against the new source before "
        "reapplying)"
    )

idx = text.find(anchor)
line_start = text.rfind("\n", 0, idx) + 1
indent = text[line_start:idx]

# Inserted as a new bullet immediately after the "## Assistant Output
# Directives" heading and before the existing MEDIA: bullet. States plainly
# that the five directives below are conditional and that a plain-text reply
# needs no checking against them -- directly targeting the observed
# unproductive re-derivation loop (repeatedly checking whether MEDIA/
# voice-hint/reply_to_current apply to a reply that uses none of them).
insertion = (
    f'{indent}"These directives are conditional -- follow only the one(s) that '
    f'actually apply to this specific reply. If none apply (no media, no '
    f'audio, no quote/reply target), send the plain-text reply directly; '
    f'{sentinel}.",\n'
)

new_text = text[:line_start] + insertion + text[line_start:]

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(new_text)
print(f"patched -> {dest_path}")
PY

echo "已生成 system-prompt output-directives 补丁: ${DEST_DIR}/system-prompt-config-CLAPATdy.js"
