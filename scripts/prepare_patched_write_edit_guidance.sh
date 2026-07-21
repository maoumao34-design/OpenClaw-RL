#!/bin/bash
# Patch OpenClaw's core system-prompt assembly (src/agents/sessions/system-prompt-config.ts,
# function buildConfiguredAgentSystemPrompt) to add an explicit "prefer edit over write for
# existing files" guideline, working around a dead-code bug that silently drops this exact
# guidance today.
#
# Root cause (see docs/issues_log.md, 2026-07-21 entries): OpenClaw ships a `write` tool
# (always overwrites the whole file) and an `edit` tool (exact-text-replacement) but has NO
# dedicated "append" tool -- confirmed by checking both the current OpenClaw source and the
# upstream @mariozechner/pi-coding-agent package at the exact version (0.57.1) OpenClaw was
# pinned to at the paper's 2026-03-08 submission snapshot; neither ever had a dedicated
# append tool. Separately, OpenClaw's own `write.ts` carries a `promptGuidelines: ["Use
# write only for new files or complete rewrites."]` field meant to steer the model away from
# misusing `write` -- but this field is only rendered into the system prompt by
# buildSystemPrompt()'s "default/synthetic" branch; the real embedded-agent-runner always
# supplies a `customPrompt` (via buildConfiguredAgentSystemPrompt), which takes an early-return
# branch that never reads `promptGuidelines` at all. Confirmed via the project's own test
# (src/agents/sessions/sdk.test.ts:301-346), which asserts this drop as expected behavior for
# that code path. Net effect: the model never sees ANY guidance steering it away from using
# `write` to "append" to an existing file.
#
# Confirmed real-world impact: across three GSM8K homework-writing runs, the model frequently
# calls `write` in response to "append the answer, do not overwrite" instructions, replacing
# the entire file (destroying the original "Problem:\n...\n\nSolution:\n" structure) with just
# the new content -- while confidently reporting success ("has been appended"/"added to the
# file"), because `write` genuinely does succeed (just at the wrong operation). Confirmed via a
# live debug-level diagnostic (2026-07-21) against a real Qwen3-4B-Thinking sglang backend:
# without this patch, an "append, don't overwrite" instruction produced a `write` tool call
# and the file lost its original structure; with this patch's guideline text injected, the
# SAME scenario produced a `read` + `edit` tool call sequence and the file was correctly
# appended with structure intact. This also connects to the PRM reward gap already documented:
# `write`'s "Successfully wrote N bytes" result is exactly the kind of "tool returns a
# successful, non-error result" signal the PRM judge scores positively (see
# openclaw_opd_api_server.py's `_build_prm_eval_prompt`), with no way to detect that the
# operation destroyed pre-existing content -- so this misuse is not corrected by training,
# it's likely reinforced.
#
# Scope note: unlike the other four patches this project has shipped, this one is NOT a
# "restore paper-era behavior" fix -- the upstream package at the paper's pinned version
# (0.57.1) never provided this guidance either, so this is a genuinely new addition (fixing
# an OpenClaw bug where OpenClaw's own intended safety nudge fails to reach the model), not a
# reversion. Kept deliberately narrow: a single explicit guideline sentence, not a broader
# rewrite of tool descriptions or behavior.
#
# Fix: insert a "## File Editing" section directly after the existing
# buildAssistantOutputDirectivesSection(...) call site inside the `lines` array that
# buildConfiguredAgentSystemPrompt assembles, so it reaches the model exactly like the other
# sections we've already patched into this same file (Assistant Output Directives).
#
# Why this patches a LIVE, hash-named bundle file instead of a repo-vendored copy: same
# reasoning as the sibling system-prompt/cli-compaction patches -- this is OpenClaw's own core
# prompt-assembly logic, bundled into a content-hashed dist file, with no supported plugin
# extension point. This is the SAME bundle file already patched by
# prepare_patched_system_prompt_output_directives.sh (Assistant Output Directives); this
# script is designed to be independent/composable with that one -- it anchors on the
# buildAssistantOutputDirectivesSection(...) CALL SITE structure (not its section content), so
# it works whether or not that other patch has already been applied to the same file. The
# exact bundle filename (system-prompt-config-CLAPATdy.js) is specific to this OpenClaw build
# and WILL change on any OpenClaw upgrade -- if this patch starts failing with "anchor not
# found", re-locate the file first via:
#   grep -rl "buildConfiguredAgentSystemPrompt" /usr/lib/node_modules/openclaw/dist/
#
# Idempotency + composability with the Assistant Output Directives patch (same bundle
# file): this script uses its OWN backup file
# (${LIVE_FILE}.orig-unpatched-write-edit-guidance), NOT the shared
# "${LIVE_FILE}.orig-unpatched" convention the other patches use. Reason: both this
# patch and prepare_patched_system_prompt_output_directives.sh target the exact same
# live bundle file. If this script reused the shared backup name, it would patch from
# a pristine state that predates the OTHER patch's changes, and copying its output
# over the live file would silently revert that other patch. Using a distinct backup
# name means: whenever this script first runs (normally right after the Assistant
# Output Directives patch, in the same deployment sequence in train_with_services.sh),
# it snapshots whatever the live file looks like AT THAT MOMENT (i.e. already including
# the other patch's change) as ITS OWN "pristine" baseline, then always re-patches from
# that baseline on subsequent runs -- so the two patches compose correctly and neither
# double-patches nor drifts.
set -euo pipefail

LIVE_FILE=${1:?usage: prepare_patched_write_edit_guidance.sh <live_file> <dest_dir>}
DEST_DIR=${2:?usage: prepare_patched_write_edit_guidance.sh <live_file> <dest_dir>}
BACKUP_FILE="${LIVE_FILE}.orig-unpatched-write-edit-guidance"

if [ ! -f "${LIVE_FILE}" ]; then
    echo "错误：找不到 system-prompt bundle ${LIVE_FILE}" >&2
    echo "这个文件名是内容哈希命名的，OpenClaw 升级后会变化，需要重新定位：" >&2
    echo "  grep -rl \"buildConfiguredAgentSystemPrompt\" /usr/lib/node_modules/openclaw/dist/" >&2
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

# Content-based idempotency check: this inserted text becomes part of the live
# system prompt shown to the model, so we check for a distinctive substring of
# it directly (same approach as the sibling Assistant Output Directives patch).
sentinel = "Never use write to append content"
if sentinel in text:
    raise SystemExit(
        "patch failed: sentinel text already present in backup file "
        f"({src_path}) -- the backup may itself be already patched. "
        "Investigate before proceeding; do not blindly re-run."
    )

# Anchor on the unique "## Workspace Files (injected)" marker that precedes the
# buildAssistantOutputDirectivesSection(...) call site in the `lines` array
# assembly (confirmed unique via real deployed bundle, openclaw build with hash
# CLAPATdy, on 2026-07-21). We deliberately anchor on the CALL SITE structure
# (not the Assistant Output Directives section's own content), so this patch
# works independently of whether prepare_patched_system_prompt_output_directives.sh
# has already been applied to the same file.
call_anchor = "## Workspace Files (injected)"
count_call = text.count(call_anchor)
if count_call != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of call_anchor, found {count_call} "
        "(OpenClaw's system-prompt assembly may have changed upstream -- re-verify "
        "this patch against the new source before reapplying blindly)"
    )
idx_call = text.find(call_anchor)

call_site = "buildAssistantOutputDirectivesSection({"
idx_call_site = text.find(call_site, idx_call)
if idx_call_site == -1:
    raise SystemExit(
        "patch failed: buildAssistantOutputDirectivesSection( call site not found "
        "after call_anchor -- OpenClaw's system-prompt assembly may have changed upstream"
    )

close_paren = text.find("})", idx_call_site)
if close_paren == -1:
    raise SystemExit("patch failed: could not find closing '})' for the call site")
close_bracket = text.find("];", close_paren)
if close_bracket == -1:
    raise SystemExit("patch failed: could not find closing '];' for the lines array")

# Sanity check: the gap between close_paren and close_bracket should be small
# (just whitespace) -- if it's large, our anchor logic likely drifted to the
# wrong location and we should refuse rather than silently patch the wrong spot.
gap = text[close_paren + 2:close_bracket]
if len(gap.strip()) > 0 or len(gap) > 20:
    raise SystemExit(
        "patch failed: unexpected content between call site and array close "
        f"(gap={gap!r}) -- anchor logic may have matched the wrong location, "
        "re-verify against the new source before reapplying blindly"
    )

insertion = (
    ",\n"
    '"## File Editing",\n'
    '"When modifying an EXISTING file, always prefer edit over write. write unconditionally '
    "overwrites the entire file and destroys any existing content -- only use write for "
    "brand-new files or when a full, intentional rewrite is requested. To add content to an "
    "existing file (e.g. appending an answer), use edit with the trailing existing text as "
    'oldText and that same text plus your addition as newText. Never use write to append '
    'content -- it will erase everything already in the file.",\n'
    '""'
)

new_text = text[:close_bracket] + insertion + text[close_bracket:]

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(new_text)
print(f"patched -> {dest_path}")
PY

echo "已生成 write/edit 工具选择指引补丁: ${DEST_DIR}/system-prompt-config-CLAPATdy.js"
