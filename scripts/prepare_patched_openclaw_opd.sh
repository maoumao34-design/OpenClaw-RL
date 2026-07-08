#!/bin/bash
# Patch openclaw_opd_api_server.py's session_id derivation.
#
# Why: the paper's intended mechanism for X-Session-Id/X-Turn-Type is the
# official `rl-training-headers` OpenClaw plugin (patches globalThis.fetch,
# hooks before_prompt_build). Verified empirically on this OpenClaw build
# (2026.6.9) that the hook fires and fetch is patched, but the headers never
# reach the actual outbound HTTP call -- OpenClaw's provider transport (OpenAI
# JS SDK client) does not re-read globalThis.fetch per request. This is an
# OpenClaw-internal incompatibility, not something in the paper or in
# OpenClaw-RL-official.
#
# X-Turn-Type is recovered via OpenClaw's own official static-header config
# (models.providers.sglang.headers -> "X-Turn-Type": "main"; safe as a
# constant here because this pipeline never triggers OpenClaw's
# heartbeat/memory/cron "side" traffic -- every call is a real student/TA/
# teacher turn). No code change needed for turn_type.
#
# X-Session-Id has no static equivalent (it's per-conversation). OpenClaw
# embeds "session=agent:<agentId>:openai-user:<user>" in its own Runtime line
# inside the first system message on every call, where <user> is exactly the
# OpenAI `user` field student_chat.py/TA_chat.py/teacher_chat.py already set
# to a stable per-session id. This patch extracts it as a fallback when the
# header/body fields are absent.
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

old_session_line = 'session_id = x_session_id or body.get("session_id") or "unknown"'
if old_session_line not in text:
    raise SystemExit(
        "patch failed: expected session_id line not found in openclaw_opd_api_server.py "
        "(official file may have changed upstream -- update this patch)"
    )

class_marker = "class OpenClawOPDAPIServer"
if class_marker not in text:
    raise SystemExit("patch failed: OpenClawOPDAPIServer class not found")

helper = '''
_RUNTIME_SESSION_RE = re.compile(r"session=agent:[^:]+:openai-user:(\\S+)")


def _extract_session_id_from_system_prompt(messages):
    """Fallback session id when X-Session-Id/body.session_id are absent.

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

'''

text = text.replace(class_marker, helper + "\n" + class_marker, 1)

new_session_lines = (
    'session_id = (\n'
    '                x_session_id\n'
    '                or body.get("session_id")\n'
    '                or _extract_session_id_from_system_prompt(body.get("messages"))\n'
    '                or "unknown"\n'
    '            )'
)
text = text.replace(old_session_line, new_session_lines, 1)

if "\nimport re\n" not in text:
    text = text.replace("import json\n", "import json\nimport re\n", 1)

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"patched -> {dest_path}")
PY

echo "已生成 session_id 兼容补丁: ${DEST_DIR}/openclaw_opd_api_server.py"
