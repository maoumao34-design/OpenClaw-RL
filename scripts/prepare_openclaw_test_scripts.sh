#!/bin/bash
# Patch openclaw-test/{student,TA,teacher}_chat.py:
#
#   1. (pre-existing) Rewrite the literal `"model": "default"` field to
#      `"model": "openclaw/default"`, the agent-target format OpenClaw
#      2026.6.9's /v1/chat/completions endpoint actually expects.
#
#   2. (new) Insert a deterministic, harness-level ground-truth file check
#      before honoring the DONE-style sentinel (HOMEWORK_DONE / GRADING_DONE /
#      COMMENT_DONE), so the session cannot end and advance to the next
#      problem unless the target homework file was genuinely, correctly
#      updated. If the check fails, the sentinel is replaced with a fixed
#      correction message instead of being honored, and the loop continues.
#
# Why (see docs/issues_log.md, 2026-07-22 entries): the Student/TA/Teacher
# simulator (external Qwen3-32B, base model, unmodified prompt, no sampling
# params ever set -- confirmed via full git history of these three files,
# this is exactly how the paper's own original design has always worked, not
# a gap introduced by our deployment) has no file-reading capability and
# repeatedly confirms "done" based purely on conversational impression --
# empirically confirmed via real training data to (a) accept a genuinely
# failed edit as complete (Problem 4) and (b) accept a `write` call that
# silently overwrote/dropped prior content while still reporting success
# (Problem 11, hit twice independently). Fixing the *reward* for a turn does
# not stop the *session* from ending prematurely and moving to the next
# problem with the task never actually completed -- these are two different
# mechanisms. This patch fixes the session-continuation side.
#
# Reproduction-fidelity note: this is NOT the same category of change as the
# (reverted) write/edit prompt-guidance patch. That one gave the POLICY model
# technical help the paper's original environment never had. This one makes
# the SIMULATOR (a stand-in for a real human student/TA/teacher) behave more
# like a real person actually checking their own homework file before
# declaring it done -- it changes nothing the policy model perceives as
# input; the policy still receives exactly the same conversational messages
# it always would, just possibly one more (a correction) if it got something
# wrong. The 32B simulator itself does not perform this check or see its
# internals -- it is a deterministic Python-side gate, matching the project's
# established preference for rule-based checks over LLM judgment.
#
# This only rewrites known, literal source blocks; no training logic,
# reward, or data path is touched. The official openclaw-test/ directory is
# left untouched -- this writes patched copies to DEST_DIR instead.
set -euo pipefail

REPO_ROOT=${1:?usage: prepare_openclaw_test_scripts.sh <repo_root> <dest_dir>}
DEST_DIR=${2:?usage: prepare_openclaw_test_scripts.sh <repo_root> <dest_dir>}
SRC_DIR="${REPO_ROOT}/openclaw-test"

mkdir -p "${DEST_DIR}"

if [ ! -e "${DEST_DIR}/GSM8K.json" ]; then
    ln -sf "${SRC_DIR}/GSM8K.json" "${DEST_DIR}/GSM8K.json"
fi

python3 - "${SRC_DIR}" "${DEST_DIR}" <<'PY'
import sys

src_dir, dest_dir = sys.argv[1], sys.argv[2]

marker = "openclaw-rl-homework-verification-gate"

# Shared helper block, inserted verbatim (module-level) into each of the
# three scripts, right after their DONE_SENTINEL constant definition.
HELPERS = '''

# --- {marker} ---
# Deterministic ground-truth file check, run before honoring the DONE-style
# sentinel. See scripts/prepare_openclaw_test_scripts.sh for full rationale.
_WHITESPACE_RE = re.compile(r"\\s+")


def _normalize_for_compare(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def _read_homework_file(workspace_dir: str, homework_dir: str, problem_index: int) -> str:
    """Read the current content of the target homework file. Empty string if missing."""
    filepath = os.path.join(workspace_dir, homework_dir, f"{{problem_index}}.txt")
    try:
        with open(filepath, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _find_last_substantial_reply(conversation_history: list[dict], min_len: int = 50) -> str | None:
    """Scan backwards for the most recent OpenClaw reply that looks like a
    real answer/comment rather than a short confirmation. In this script's
    conversation_history convention, OpenClaw's replies are stored under
    role="user" (from the simulator's own chat-completions perspective)."""
    marker_text = "The AI assistant replied:\\n\\n"
    for entry in reversed(conversation_history):
        if entry.get("role") != "user":
            continue
        content = entry.get("content", "")
        reply = content[len(marker_text):] if content.startswith(marker_text) else content
        if len(reply.strip()) >= min_len:
            return reply
    return None


def _verify_homework_file_ground_truth(
    workspace_dir: str,
    homework_dir: str,
    problem_index: int,
    initial_content: str,
    conversation_history: list[dict],
) -> bool:
    """Returns True only if the target file was genuinely, correctly updated:
      1. still contains whatever content was already there before this
         session started (catches a destructive `write` overwrite that
         silently dropped prior content), and
      2. actually grew by a non-trivial amount (something was appended, not
         a no-op / never-written case), and
      3. contains a recognizable fingerprint of the most recent
         answer/comment shown in the conversation (catches "wrote something
         unrelated" as well as "claimed done with nothing written")."""
    current_content = _read_homework_file(workspace_dir, homework_dir, problem_index)
    if not current_content:
        return False

    normalized_current = _normalize_for_compare(current_content)
    normalized_initial = _normalize_for_compare(initial_content)
    if normalized_initial and normalized_initial not in normalized_current:
        return False
    if len(current_content) <= len(initial_content) + 5:
        return False

    approved = _find_last_substantial_reply(conversation_history)
    if approved:
        fingerprint = _normalize_for_compare(approved)[:80]
        if fingerprint and fingerprint not in normalized_current:
            return False

    return True
'''.format(marker=marker)


def patch_file(filename, done_sentinel, homework_dir, role_label, run_fn_anchor, run_fn_call_anchor, done_check_anchor, msg_var, correction_template):
    src_path = f"{src_dir}/{filename}"
    dest_path = f"{dest_dir}/{filename}"
    text = open(src_path, encoding="utf-8").read()

    if marker in text:
        raise SystemExit(
            f"patch failed: marker already present in {src_path} -- "
            "the source may already be patched. Investigate before "
            "proceeding; do not blindly re-run."
        )

    # 1. Pre-existing: model field fix.
    old_model = '"model": "default"'
    if text.count(old_model) != 1:
        raise SystemExit(
            f"patch failed: expected exactly 1 occurrence of {old_model!r} in "
            f"{filename}, found {text.count(old_model)}"
        )
    text = text.replace(old_model, '"model": "openclaw/default"', 1)

    # 2. Insert shared helpers right after the DONE_SENTINEL constant.
    sentinel_line = f'DONE_SENTINEL = "{done_sentinel}"'
    if text.count(sentinel_line) != 1:
        raise SystemExit(
            f"patch failed: expected exactly 1 occurrence of {sentinel_line!r} "
            f"in {filename}, found {text.count(sentinel_line)} (openclaw-test "
            "script may have changed upstream -- re-verify this patch)"
        )
    text = text.replace(sentinel_line, sentinel_line + HELPERS, 1)

    # 3. Thread workspace_dir into the run_one_* function signature.
    if run_fn_anchor not in text:
        raise SystemExit(
            f"patch failed: run-function signature anchor not found in {filename} "
            "(openclaw-test script may have changed upstream -- re-verify this patch):\\n"
            f"{run_fn_anchor!r}"
        )
    text = text.replace(run_fn_anchor, run_fn_anchor.replace(
        "    problem_index: int,",
        "    problem_index: int,\n    workspace_dir: str,",
        1,
    ), 1)

    # 4. Capture initial_content right after conversation_history is set up,
    #    and thread it into the DONE_SENTINEL check block below.
    if 'conversation_history: list[dict] = []' not in text:
        raise SystemExit(
            f"patch failed: conversation_history initialization not found in {filename}"
        )
    text = text.replace(
        'conversation_history: list[dict] = []',
        'conversation_history: list[dict] = []\n'
        f'    initial_content = _read_homework_file(workspace_dir, "{homework_dir}", problem_index)',
        1,
    )

    # 5. Replace the "if DONE_SENTINEL in <msg_var>: ... return True" block
    #    with the ground-truth-gated version.
    if done_check_anchor not in text:
        raise SystemExit(
            f"patch failed: DONE_SENTINEL check anchor not found in {filename} "
            "(openclaw-test script may have changed upstream -- re-verify this patch):\\n"
            f"{done_check_anchor!r}"
        )
    correction_msg = correction_template.format(homework_dir=homework_dir)
    new_check = (
        f'if DONE_SENTINEL in {msg_var}:\n'
        f'            if _verify_homework_file_ground_truth(\n'
        f'                workspace_dir, "{homework_dir}", problem_index, initial_content, conversation_history,\n'
        f'            ):\n'
        f'                print(f"\\n  Turn {{turn + 1}}: {role_label} confirmed problem {{problem_index}} is done! (file verified, {marker})")\n'
        f'                return True\n'
        f'            print(\n'
        f'                f"\\n  Turn {{turn + 1}}: {role_label} said {done_sentinel} but file verification "\n'
        f'                f"FAILED ({marker}) -- injecting correction instead of ending session"\n'
        f'            )\n'
        f'            {msg_var} = {correction_msg!r}.format(index=problem_index)'
    )
    text = text.replace(done_check_anchor, new_check, 1)

    # 6. Pass workspace_dir at the run_one_* call site.
    if run_fn_call_anchor not in text:
        raise SystemExit(
            f"patch failed: run-function call-site anchor not found in {filename} "
            "(openclaw-test script may have changed upstream -- re-verify this patch):\\n"
            f"{run_fn_call_anchor!r}"
        )
    text = text.replace(run_fn_call_anchor, run_fn_call_anchor.replace(
        "problem_index=i,",
        "problem_index=i,\n            workspace_dir=workspace,",
        1,
    ), 1)

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"patched -> {dest_path}")


CORRECTION_TEMPLATE = (
    "Wait, that doesn't look right -- I don't think it actually got saved "
    "correctly. Can you check the file {homework_dir}/{{index}}.txt again "
    "and make sure it's really there?"
)

patch_file(
    filename="student_chat.py",
    done_sentinel="HOMEWORK_DONE",
    homework_dir="homework",
    role_label="Student",
    run_fn_anchor=(
        "def run_one_problem(\n"
        "    problem_index: int,\n"
        "    gateway_url: str,"
    ),
    run_fn_call_anchor=(
        "        completed = run_one_problem(\n"
        "            problem_index=i,"
    ),
    done_check_anchor=(
        'if DONE_SENTINEL in student_msg:\n'
        '            print(f"\\n  Turn {turn + 1}: Student confirmed problem {problem_index} is done!")\n'
        '            return True'
    ),
    msg_var="student_msg",
    correction_template=CORRECTION_TEMPLATE,
)

patch_file(
    filename="TA_chat.py",
    done_sentinel="GRADING_DONE",
    homework_dir="homework1",
    role_label="TA",
    run_fn_anchor=(
        "def run_one_grading(\n"
        "    problem_index: int,\n"
        "    question: str,"
    ),
    run_fn_call_anchor=(
        "        completed = run_one_grading(\n"
        "            problem_index=i,"
    ),
    done_check_anchor=(
        'if DONE_SENTINEL in ta_msg:\n'
        '            print(f"\\n  Turn {turn + 1}: TA confirmed grading for problem {problem_index} is done!")\n'
        '            return True'
    ),
    msg_var="ta_msg",
    correction_template=CORRECTION_TEMPLATE,
)

patch_file(
    filename="teacher_chat.py",
    done_sentinel="COMMENT_DONE",
    homework_dir="homework2",
    role_label="Teacher",
    run_fn_anchor=(
        "def run_one_commenting(\n"
        "    problem_index: int,\n"
        "    question: str,"
    ),
    run_fn_call_anchor=(
        "        completed = run_one_commenting(\n"
        "            problem_index=i,"
    ),
    done_check_anchor=(
        'if DONE_SENTINEL in teacher_msg:\n'
        '            print(f"\\n  Turn {turn + 1}: Teacher confirmed commenting for problem {problem_index} is done!")\n'
        '            return True'
    ),
    msg_var="teacher_msg",
    correction_template=CORRECTION_TEMPLATE,
)
PY

echo "已生成 openclaw-test 补丁: ${DEST_DIR}（model 字段兼容 + HOMEWORK_DONE/GRADING_DONE/COMMENT_DONE 前置文件核验）"
