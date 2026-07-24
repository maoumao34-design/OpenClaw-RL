#!/bin/bash
# Patch the bundled sglang provider extension to completely blank out
# OpenClaw's built-in "## Execution Bias" system-prompt section (all 7
# guidance lines, not just one).
#
# History: this started 2026-07-17 as a narrow, one-line removal --
# "Non-final turn: use tools to advance, or ask for the one missing decision
# that blocks safe progress." was confirmed (via full reasoning_text capture,
# docs/issues_log.md 2026-07-16/17, Problem 36) to make the model unable to
# decide whether a plain-text reply needs <tool_call> XML wrapping (colliding
# with the Qwen chat template's own tool-call-format instruction), observed
# spending its entire 8192-token budget re-deriving the same already-decided
# reply text without ever emitting it. The other 6 lines were deliberately
# left alone at the time as a minimal, surgical fix.
#
# 2026-07-23: a second line from the same section, independently confirmed
# harmful with new real production data (docs/issues_log.md 2026-07-23,
# Problem 31/session 86d5e924-e206-4513-89ab-98d7299d7f2e): "Weak/empty tool
# result: vary query, path, command, or source before concluding." A single
# Student-only session ran 41 consecutive internal MAIN turns over 12
# minutes (16:03-16:15) without ever converging on a final reply -- turn 1's
# `read` call succeeded immediately and correctly (confirmed via the session
# transcript: the real problem text was faithfully present in every
# subsequent turn's context, never corrupted or lost), yet the model kept
# repeating `read` on the same file and trying unrelated tools
# (`memory_search` against "memory" then "wiki" corpora, both returning
# empty) roughly every ~20 seconds, never producing a final answer, until
# the accumulating session finally overflowed its context budget. This
# instruction directly matches that behavior: an empty/weak result (the
# memory_search misses) is explicit license to keep trying alternate
# sources "before concluding" -- and combined with "Continue until done or
# genuinely blocked", a Thinking model with no v1-era grounding for when a
# GSM8K read is "enough" has no natural stopping condition. (The literal
# reasoning_content text was not directly quotable -- training.log's debug
# logging only prints the `content` field, which was empty for these
# tool-call-only turns, and the character count of `thinking`, not its text
# -- so this is strong structural/behavioral evidence, not a verbatim
# citation.)
#
# User's judgment (2026-07-23): rather than continuing to whack-a-mole
# individual lines as each one is separately caught causing harm, blank out
# the entire section. Rationale: this whole "Execution Bias" section is an
# optimization/nudge that assumes a model capable enough to correctly judge
# "when do I have enough" -- Qwen3-4B-Thinking is not that model, and the
# added guidance measurably backfires (both confirmed lines make an already
# weak model second-guess itself) rather than helping. Restoring silence on
# all 7 lines is lower-risk than trying to identify exactly which subset is
# safe.
#
# This section (buildExecutionBiasSection() in OpenClaw's src/agents/
# system-prompt.ts) was added 2026-04-15~2026-04-30 -- after the paper's
# 2026-03-11 submission. Git archaeology against the real OpenClaw repo
# history (docs/issues_log.md) confirmed the ENTIRE section (not just these
# two lines) did not exist in march_2026_3_8 at all -- and per the
# 2026-07-23 CLAUDE.md working assumption (paper authors' actual OpenClaw CLI
# dependency is most likely pinned to ~March 2026, not upgraded in lockstep
# with the paper's own May v2 methodology revisions), removing the whole
# section restores that original (no Execution Bias guidance at all)
# behavior rather than introducing a new deviation.
#
# Why this patches the LIVE installed file instead of a repo-vendored copy
# (unlike prepare_patched_openclaw_opd.sh / prepare_patched_rl_training_headers.sh):
# extensions/sglang is part of the OpenClaw CLI product itself (bundled,
# npm-installed), not part of OpenClaw-RL-official -- there is no copy of it
# in this training repo to patch from. The only supported way to override a
# core system-prompt section without editing OpenClaw's core
# src/agents/system-prompt.ts is the provider hook `resolveSystemPromptContribution`,
# which is invoked for exactly the one plugin registered as owner of a given
# provider id (single-owner lookup -- see docs/issues_log.md,
# ensureProviderRuntimePluginHandle/resolveProviderRuntimePluginHandle). So it
# has to be added to whichever plugin owns the "sglang" provider, i.e. this
# bundled extension itself.
#
# An earlier attempt (docs/issues_log.md) used before_prompt_build's
# appendSystemContext to append a disambiguation rule instead of removing the
# conflicting line -- rejected because appending leaves the original
# contradictory text in the prompt (append-only, cannot erase context), which
# is a materially weaker guarantee than actually removing it. Given the user's
# requirement that this fix must reliably hold (a recurrence would waste an
# entire training run), literal content removal is used instead.
#
# Idempotency: always patches from a pristine backup
# (${LIVE_FILE}.orig-unpatched, created on first run) rather than the live
# file, so re-running this across multiple training script invocations never
# double-patches or drifts.
set -euo pipefail

LIVE_FILE=${1:?usage: prepare_patched_sglang_execution_bias.sh <live_file> <dest_dir>}
DEST_DIR=${2:?usage: prepare_patched_sglang_execution_bias.sh <live_file> <dest_dir>}
BACKUP_FILE="${LIVE_FILE}.orig-unpatched"

if [ ! -f "${LIVE_FILE}" ]; then
    echo "错误：找不到 sglang 扩展 ${LIVE_FILE}" >&2
    exit 1
fi

if [ ! -f "${BACKUP_FILE}" ]; then
    cp "${LIVE_FILE}" "${BACKUP_FILE}"
    echo "已备份未修改原文件 -> ${BACKUP_FILE}"
fi

mkdir -p "${DEST_DIR}"

python3 - "${BACKUP_FILE}" "${DEST_DIR}/index.js" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(src_path, encoding="utf-8").read()

if "resolveSystemPromptContribution" in text:
    raise SystemExit(
        "patch failed: resolveSystemPromptContribution already present in "
        f"backup file ({src_path}) -- the backup may itself be already "
        "patched (e.g. from a stale prior run against a different OpenClaw "
        "build). Investigate before proceeding; do not blindly re-run."
    )

# Two anchors, both confirmed against the real deployed file (openclaw
# 2026.6.9, /usr/lib/node_modules/openclaw/dist/extensions/sglang/index.js)
# on 2026-07-17. If either string is missing, the bundled extension has
# changed upstream and this patch needs to be re-verified against the new
# source before reapplying blindly.

# Anchor 1: last import line, top of file -- insertion point for a
# module-scope "have I logged the confirmation line yet" flag, so the
# training log can positively confirm (once, not once-per-turn) that this
# hook is actually being invoked and returning the expected override,
# instead of relying purely on "no more TRUNCATED samples" as indirect
# evidence after a full (expensive) training run.
import_anchor = 'import "../../api-D_xHk_js.js";'
import_idx = text.find(import_anchor)
if import_idx == -1:
    raise SystemExit(
        "patch failed: expected import anchor line not found in sglang "
        "extension (bundled OpenClaw sglang provider may have changed "
        "upstream -- re-verify this patch against the new source before "
        "reapplying)"
    )
import_line_end = text.find("\n", import_idx)
flag_insertion = "\nlet _executionBiasFixLogged = false;"
text = text[:import_line_end] + flag_insertion + text[import_line_end:]

# Anchor 2: unique, short anchor line inside the object literal passed to
# api.registerProvider({...}).
anchor = 'docsPath: "/providers/sglang",'
idx = text.find(anchor)
if idx == -1:
    raise SystemExit(
        "patch failed: expected anchor line not found in sglang extension "
        "(bundled OpenClaw sglang provider may have changed upstream -- "
        "re-verify this patch against the new source before reapplying)"
    )

line_start = text.rfind("\n", 0, idx) + 1
indent = text[line_start:idx]
line_end = text.find("\n", idx)

# All 7 guidance lines of buildExecutionBiasSection()'s fallback content
# (src/agents/system-prompt.ts) are blanked out -- only the bare section
# header survives. Two of the 7 lines are independently confirmed (real
# production data, see comment block at the top of this file) to trigger
# decision-paralysis / non-converging tool-call loops in Qwen3-4B-Thinking;
# the remaining 5 were never individually confirmed safe, and the entire
# section is absent from march_2026_3_8 anyway, so there is no principled
# subset to keep. Cannot override with a literal empty string -- OpenClaw's
# own normalizeStructuredPromptSection() .trim()s the override value, and
# buildOverridablePromptSection() treats a falsy (post-trim-empty) override
# as "no override provided", silently falling back to the full original
# 7-line section instead of suppressing it. The section header alone is
# non-empty (survives trim), carries no behavioral guidance, and is the
# smallest content that reliably keeps the override path active.
execution_bias_override = "## Execution Bias"

insertion = (
    f'\n{indent}resolveSystemPromptContribution(_ctx) {{\n'
    f'{indent}\tif (!_executionBiasFixLogged) {{\n'
    f'{indent}\t\t_executionBiasFixLogged = true;\n'
    f'{indent}\t\tconsole.error("[execution-bias-fix] resolveSystemPromptContribution invoked, '
    f'overriding execution_bias section (first call only, not logged per-turn)");\n'
    f'{indent}\t}}\n'
    f'{indent}\treturn {{ sectionOverrides: {{ execution_bias: "{execution_bias_override}" }} }};\n'
    f'{indent}}},'
)

new_text = text[:line_end] + insertion + text[line_end:]

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(new_text)
print(f"patched -> {dest_path}")
PY

echo "已生成 sglang execution-bias 补丁: ${DEST_DIR}/index.js"
