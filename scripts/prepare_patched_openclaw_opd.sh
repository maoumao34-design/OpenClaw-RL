#!/bin/bash
# Patch openclaw_opd_api_server.py's session_id/turn_type derivation.
#
# Why: the paper's intended mechanism for X-Session-Id/X-Turn-Type is the
# official `rl-training-headers` OpenClaw plugin, which injects HTTP headers
# via a before_prompt_build hook. Confirmed via extensive source-level
# investigation (2026-07-09, docs/issues_log.md) that on this OpenClaw build
# (2026.6.9), NO header/dispatcher-based injection mechanism can reach a real
# outbound request -- OpenClaw added a deliberate SSRF-safety layer
# (fetchWithSsrFGuardInternal / fetchWithRuntimeDispatcher) that bypasses any
# externally-patched fetch/dispatcher for all non-Vitest-mock requests, with
# no config escape hatch. This is a structural, intentional security boundary
# in the current OpenClaw version, not present when the paper's plugin was
# written (~2026.3-4), confirmed by reading OpenClaw's actual source from
# that era.
#
# Working alternative: the patched plugin (scripts/prepare_patched_rl_training_headers.sh)
# uses before_prompt_build's `appendSystemContext` return field instead --
# this modifies request CONTENT, not transport, so it is unaffected by the
# SSRF-safety bypass. It encodes ctx.trigger/ctx.sessionId as a
# "[RL-TRAINING-META] session_id=... turn_type=..." marker appended to the
# system prompt. This file's patch:
#   1. Parses that marker out of `messages` as a fallback for session_id/turn_type
#      (still lower priority than explicit X-Session-Id/X-Turn-Type headers or
#      body fields, in case those ever start working again on a future OpenClaw
#      version).
#   2. Strips the marker text out of `messages` BEFORE forwarding to sglang and
#      BEFORE computing training-sample prompt_ids -- so neither the policy
#      model nor the training data ever sees the marker. Verified this keeps
#      model-facing and training-facing content identical to what the paper's
#      (never-working-here) header mechanism would have produced.
#
# The old Runtime-line session_id fallback (from OpenClaw's own "Runtime:"
# line, unrelated to the plugin) is kept as a lower-priority fallback below
# the marker -- harmless to keep, doesn't hurt if the marker is ever absent.
#
# This is a deviation from the paper's plugin-based design -- documented in
# docs/issues_log.md. Official openclaw-opd/ directory is left untouched; this
# writes a patched copy to DEST_DIR and the caller must prepend DEST_DIR to
# PYTHONPATH ahead of openclaw-opd/ so `import openclaw_opd_api_server`
# resolves to the patched copy.
set -euo pipefail

REPO_ROOT=${1:?usage: prepare_patched_openclaw_opd.sh <repo_root> <dest_dir>}
DEST_DIR=${2:?usage: prepare_patched_openclaw_opd.sh <repo_root> <dest_dir>}
SRC="${REPO_ROOT}/openclaw-opd/openclaw_opd_api_server.py"

if [ ! -f "${SRC}" ]; then
    echo "错误：找不到官方文件 ${SRC}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

python3 - "${SRC}" "${DEST_DIR}/openclaw_opd_api_server.py" <<'PY'
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(src_path, encoding="utf-8").read()

old_block = (
    '            body = await request.json()\n'
    '            session_id = x_session_id or body.get("session_id") or "unknown"\n'
    '            turn_type = (x_turn_type or body.get("turn_type") or "side").strip().lower()\n'
)
if old_block not in text:
    raise SystemExit(
        "patch failed: expected chat_completions body/session_id/turn_type block not found "
        "in openclaw_opd_api_server.py (official file may have changed upstream -- update this patch)"
    )

class_marker = "class OpenClawOPDAPIServer"
if class_marker not in text:
    raise SystemExit("patch failed: OpenClawOPDAPIServer class not found")

helper = '''
_RUNTIME_SESSION_RE = re.compile(r"session=agent:[^:]+:openai-user:(\\S+)")
_RL_META_RE = re.compile(r"\\[RL-TRAINING-META\\] session_id=(\\S*) turn_type=(\\S*)")


def _extract_session_id_from_system_prompt(messages):
    """Fallback session id when X-Session-Id/body.session_id/RL-TRAINING-META
    marker are all absent.

    OpenClaw's own Runtime line (embedded in the first system message) carries
    "session=agent:<agentId>:openai-user:<user>", where <user> is exactly the
    OpenAI `user` field the caller passed (student_chat.py etc. set this to a
    stable per-conversation id). Independent of the rl-training-headers plugin.
    """
    if not isinstance(messages, list):
        logger.warning("[SESSION-ID-DEBUG] messages is not a list: %r", type(messages))
        return None
    system_contents = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            system_contents.append(f"<non-str:{type(content)!r}>")
            continue
        system_contents.append(content)
        match = _RUNTIME_SESSION_RE.search(content)
        if match:
            return match.group(1)
    logger.warning(
        "[SESSION-ID-DEBUG] no Runtime session match; %d system message(s), tail=%r",
        len(system_contents),
        system_contents[-1][-500:] if system_contents else None,
    )
    return None


def _extract_rl_meta_from_messages(messages):
    """Read the (session_id, turn_type) pair the patched rl-training-headers
    plugin appends to the system prompt via before_prompt_build's
    appendSystemContext (header injection is confirmed structurally
    impossible on this OpenClaw build -- see docs/issues_log.md). Either
    element may be None if the marker is absent or a field is empty.
    """
    if not isinstance(messages, list):
        return None, None
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        match = _RL_META_RE.search(content)
        if match:
            session_id = match.group(1) or None
            turn_type = match.group(2) or None
            return session_id, turn_type
    return None, None


def _strip_rl_meta_from_messages(messages):
    """Remove the RL-TRAINING-META marker from system message content.

    Must run before the messages are forwarded to sglang and before training
    prompt_ids are computed, so neither the policy model nor the training
    data ever sees this plugin-internal marker text.
    """
    if not isinstance(messages, list):
        return messages
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system" and isinstance(msg.get("content"), str):
            new_content = _RL_META_RE.sub("", msg["content"]).rstrip()
            if new_content != msg["content"]:
                msg = {**msg, "content": new_content}
        cleaned.append(msg)
    return cleaned

'''

text = text.replace(class_marker, helper + "\n" + class_marker, 1)

new_block = (
    '            body = await request.json()\n'
    '            _rl_meta_session_id, _rl_meta_turn_type = _extract_rl_meta_from_messages(body.get("messages"))\n'
    '            if isinstance(body.get("messages"), list):\n'
    '                body["messages"] = _strip_rl_meta_from_messages(body["messages"])\n'
    '            session_id = (\n'
    '                x_session_id\n'
    '                or body.get("session_id")\n'
    '                or _rl_meta_session_id\n'
    '                or _extract_session_id_from_system_prompt(body.get("messages"))\n'
    '                or "unknown"\n'
    '            )\n'
    '            turn_type = (x_turn_type or body.get("turn_type") or _rl_meta_turn_type or "side").strip().lower()\n'
)
text = text.replace(old_block, new_block, 1)

if "\nimport re\n" not in text:
    text = text.replace("import json\n", "import json\nimport re\n", 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched -> {dest_path}")
PY

echo "已生成 session_id 兼容补丁: ${DEST_DIR}/openclaw_opd_api_server.py"
