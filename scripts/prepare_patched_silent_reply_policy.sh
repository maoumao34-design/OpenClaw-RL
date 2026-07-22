#!/bin/bash
# Patch OpenClaw's core "Silent Reply Policy" (src/shared/silent-reply-policy.ts,
# bundled into dist/effective-reply-route-*.js) so it always resolves to
# "disallow", restoring March-era behavior where this feature did not exist at
# all -- no code path could ever treat an empty/silent final reply as an
# accepted, non-error outcome.
#
# Root cause (see docs/issues_log.md, 2026-07-22 entry): this entire module
# (src/config/silent-reply.ts, src/shared/silent-reply-policy.ts, plus their
# test files) is CONFIRMED ABSENT from the paper's 2026-03-08 submission
# snapshot (git archaeology: `git ls-tree -r march_2026_3_8 --name-only | grep
# -i silent` -> zero matches) and CONFIRMED PRESENT by 2026-05-11 (same
# grep against may_2026_5_11 -> six files, including this policy module).
# This is the fifth independent OpenClaw behavior confirmed to have this exact
# "absent at March, present after" pattern, following Execution Bias,
# Assistant Output Directives, embedded-agent-runner overflow-recovery, and
# cli-compaction's cli_budget check -- strong cumulative evidence that the
# paper's real experiments ran against a March-era (or earlier) OpenClaw
# build that never had any of these five behaviors.
#
# The bug (as it interacts with our training setup): DEFAULT_SILENT_REPLY_POLICY
# is { direct: "disallow", group: "allow", internal: "allow" }.
# classifySilentReplyConversationType() falls back to "internal" whenever the
# session key does not match any of :group:/:channel:/:direct:/:dm: and the
# surface isn't "webchat" -- exactly the case for our training session keys
# (agent:main:openai-user:student-hw-<i>-<pid>), which were never designed
# with this messaging-channel feature in mind. So our sessions land in the
# "internal" bucket, where silent/empty final replies are policy-"allow"ed by
# a feature the paper's original OpenClaw build never had.
#
# Evidence boundary (documented honestly, unlike the cli-compaction patch this
# was NOT confirmed via a live debug-level diagnostic): we traced
# emptyFinalAllowedAsSilent's construction in dispatch-from-config.ts/
# dispatch-*.js (it governs whether an empty/silent-looking final reply is
# treated as an accepted outcome vs. routed through OpenClaw's normal
# error/retry handling for that turn) but did not fully trace every
# downstream consumer. The fix is deliberately narrow and conservative: force
# the SAME "disallow" outcome already hardcoded for conversationType="direct"
# to apply unconditionally, rather than attempting to reproduce the absence
# of the whole feature module (which would require removing call sites in
# multiple bundles). Any other reply-routing behavior is untouched.
#
# Why this patches a LIVE, hash-named bundle file instead of a repo-vendored
# copy: same reasoning as the sibling embedded-agent-runner/cli-compaction
# patches -- this is OpenClaw's own core reply-dispatch logic, bundled into a
# content-hashed dist file, with no supported plugin extension point. The
# exact bundle filename (effective-reply-route-BnYlac-J.js) is specific to
# this OpenClaw build and WILL change on any OpenClaw upgrade -- if this patch
# starts failing with "anchor not found", re-locate the file first via:
#   grep -rl "resolveSilentReplyPolicyFromPolicies" /usr/lib/node_modules/openclaw/dist/
#
# Idempotency: always patches from a pristine backup (${LIVE_FILE}.orig-unpatched,
# created on first run) rather than the live file, so re-running this across
# multiple training script invocations never double-patches or drifts.
set -euo pipefail

LIVE_FILE=${1:?usage: prepare_patched_silent_reply_policy.sh <live_file> <dest_dir>}
DEST_DIR=${2:?usage: prepare_patched_silent_reply_policy.sh <live_file> <dest_dir>}
BACKUP_FILE="${LIVE_FILE}.orig-unpatched"

if [ ! -f "${LIVE_FILE}" ]; then
    echo "错误：找不到 effective-reply-route bundle ${LIVE_FILE}" >&2
    echo "这个文件名是内容哈希命名的，OpenClaw 升级后会变化，需要重新定位：" >&2
    echo "  grep -rl \"resolveSilentReplyPolicyFromPolicies\" /usr/lib/node_modules/openclaw/dist/" >&2
    exit 1
fi

if [ ! -f "${BACKUP_FILE}" ]; then
    cp "${LIVE_FILE}" "${BACKUP_FILE}"
    echo "已备份未修改原文件 -> ${BACKUP_FILE}"
fi

mkdir -p "${DEST_DIR}"

python3 - "${BACKUP_FILE}" "${DEST_DIR}/effective-reply-route-BnYlac-J.js" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(src_path, encoding="utf-8").read()

marker = "openclaw-rl-silent-reply-policy-patch"
if marker in text:
    raise SystemExit(
        "patch failed: patch marker already present in backup file "
        f"({src_path}) -- the backup may itself be already patched. "
        "Investigate before proceeding; do not blindly re-run."
    )

# Unique anchor: the full body of resolveSilentReplyPolicyFromPolicies as
# confirmed against the real deployed bundle (effective-reply-route-BnYlac-J.js)
# on 2026-07-22. If missing, OpenClaw's silent-reply-policy.ts has changed
# upstream and this patch needs re-verification against the new source.
anchor = (
    'function resolveSilentReplyPolicyFromPolicies(params) {\n'
    '\tif (params.conversationType === "direct") return "disallow";\n'
    '\treturn params.surfacePolicy?.[params.conversationType] ?? params.defaultPolicy?.[params.conversationType] ?? DEFAULT_SILENT_REPLY_POLICY[params.conversationType];\n'
    '}'
)
count = text.count(anchor)
if count != 1:
    raise SystemExit(
        f"patch failed: expected exactly 1 occurrence of anchor, found {count} "
        "(OpenClaw's silent-reply-policy.ts may have changed upstream -- "
        "re-verify this patch against the new source before reapplying)"
    )

# Force "disallow" unconditionally -- restores the effective absence of any
# "silence is allowed here" code path, matching the paper's original OpenClaw
# build where this whole feature did not exist. Keeps a one-time (not
# per-turn) log line so real training logs can positively confirm the patched
# function is actually being invoked, matching the diagnostic rigor used for
# the sibling execution-bias patch.
replacement = (
    f'let _silentReplyPolicyFixLogged = false;\n'
    f'function resolveSilentReplyPolicyFromPolicies(params) {{\n'
    f'\tif (!_silentReplyPolicyFixLogged) {{\n'
    f'\t\t_silentReplyPolicyFixLogged = true;\n'
    f'\t\tconsole.error("[{marker}] resolveSilentReplyPolicyFromPolicies invoked, '
    f'forcing disallow (first call only, not logged per-turn)");\n'
    f'\t}}\n'
    f'\treturn "disallow";\n'
    f'}}'
)

new_text = text.replace(anchor, replacement, 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(new_text)
print(f"patched -> {dest_path}")
PY

node --check "${DEST_DIR}/effective-reply-route-BnYlac-J.js"
echo "已生成 silent-reply-policy 补丁: ${DEST_DIR}/effective-reply-route-BnYlac-J.js"
