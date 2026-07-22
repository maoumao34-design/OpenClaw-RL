#!/bin/bash
# Patch openclaw-test/{student,TA,teacher}_chat.py:
#
#   1. (pre-existing) Rewrite the literal `"model": "default"` field to
#      `"model": "openclaw/default"`, the agent-target format OpenClaw
#      2026.6.9's /v1/chat/completions endpoint actually expects.
#
#   2. Insert a deterministic, harness-level ground-truth file check before
#      honoring the DONE-style sentinel (HOMEWORK_DONE / GRADING_DONE /
#      COMMENT_DONE), so the session cannot end and advance to the next
#      problem unless the target homework file was genuinely, correctly
#      updated. The check never trusts the simulator's own claim of "done" --
#      it always re-derives a fresh decision:
#        - If the file is missing prior content or has no meaningful new
#          content (a definite, deterministic diagnosis), the simulator
#          sends a fixed, diagnosis-specific correction message directly --
#          no 32B call. For "overwritten" specifically, the message includes
#          the actual original file content (session-start snapshot) so the
#          policy model is told exactly what needs to still be there, not
#          just "something's wrong, check it" (real data showed the vague
#          version left the policy re-guessing from memory and re-losing the
#          same detail every retry -- see docs/issues_log.md, 2026-07-22).
#          This is a hard rule: this path can never finalize the session.
#        - Only if the deterministic check finds no problem does the
#          simulator get shown the actual newly-written content (just the
#          new portion, not the whole file with its Problem:/prior-content
#          scaffolding, so pre-existing content can't distort a length/style
#          judgment -- e.g. inflating a TA's perceived comment length) and
#          asked to make one more independent judgment call via a real 32B
#          call (matching its own step-1 requirements, e.g. non-AI-like
#          style for Student). Only in this branch can the simulator's own
#          response actually finalize the session.
#
# Why (see docs/issues_log.md, 2026-07-22 entries): the Student/TA/Teacher
# simulator (external Qwen3-32B, base model, unmodified prompt, no sampling
# params ever set -- confirmed via full git history of these three files,
# this is exactly how the paper's own original design has always worked, not
# a gap introduced by our deployment) has no file-reading capability and
# repeatedly confirms "done" based purely on conversational impression --
# empirically confirmed via real training data to (a) accept a genuinely
# failed edit as complete (Problem 4), (b) accept a `write` call that
# silently overwrote/dropped prior content while still reporting success
# (Problem 11, hit twice independently), and (c) accept a write whose actual
# saved content was never independently checked for style. Fixing the
# *reward* for a turn does not stop the *session* from ending prematurely
# and moving to the next problem with the task never actually completed --
# these are two different mechanisms. This patch fixes the
# session-continuation side. It went through several real-data-driven
# revisions before landing here: v1 fingerprinted a *past reply* instead of
# the file's actual new content and false-positived on a genuinely correct
# write (fixed same day); v2 added a fixed generic correction message that
# proved too vague for the policy to self-correct from; v3 tried having the
# 32B simulator phrase the diagnosed-problem correction itself, but real
# data showed it reliably ignored the "don't just say done" instruction
# (10/10 observed attempts across two problems) and, even when it fell back
# to the fixed template, that template was still too vague for the policy to
# fix (it kept re-losing the same missing detail on every retry); this
# version drops the 32B call for the diagnosed-problem path entirely and
# instead sends a deterministic, diagnosis-specific message that includes
# the concrete original content, while keeping the 32B independent style
# recheck for the no-problem-found path (which real data showed works fine).
#
# Reproduction-fidelity note: this is NOT the same category of change as the
# (reverted) write/edit prompt-guidance patch. That one gave the POLICY
# model technical help the paper's original environment never had. This one
# makes the SIMULATOR (a stand-in for a real human student/TA/teacher)
# behave more like a real person actually checking their own homework file
# before declaring it done -- it changes nothing the policy model perceives
# as input; the policy still receives exactly the same conversational
# messages it always would, just possibly one more (a follow-up pointing out
# a concrete discrepancy) if something looked wrong -- a real student who
# knows their own homework problem absolutely can and would point out "hey,
# it originally said X, that's gone now" instead of a vague "something feels
# off". The simulator only ever sees content scoped to what it would
# plausibly have access to (its own original problem text, or its own
# newly-requested content), never raw file-format internals.
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
HELPERS_TEMPLATE = '''

# --- {marker} ---
# Deterministic ground-truth file check, run before honoring the DONE-style
# sentinel. See scripts/prepare_openclaw_test_scripts.sh for full rationale.
_WHITESPACE_RE = re.compile(r"\\s+")

# Diagnosis-specific fixed correction messages, sent directly (no 32B call)
# whenever a real content problem is deterministically found. Keyed by
# diagnosis code so the message actually names what's wrong and, for
# "overwritten", includes the concrete original content -- a generic
# "something's off, check it" message left the policy re-guessing from
# memory and re-losing the same detail on every retry (real data, see
# docs/issues_log.md 2026-07-22).
_CORRECTION_TEMPLATES = {templates}

# Fallback for the no-diagnosis path only, used if the 32B recheck call
# itself raises an exception (network/API failure, not a content problem).
_GENERIC_CORRECTION_TEMPLATE = {generic}


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


def _diagnose_homework_file(
    workspace_dir: str,
    homework_dir: str,
    problem_index: int,
    initial_content: str,
    conversation_history: list[dict],
    recent_turns: int = 6,
) -> str | None:
    """Returns None if the file appears genuinely, correctly updated (content
    preserved, something new actually added, and that new content is
    recognizable in the recent conversation), else a short diagnosis code:
      "overwritten"  -- prior content is gone (destructive write)
      "not_written"  -- no meaningful growth (nothing actually saved)
      "mismatch"     -- new content isn't recognizable in recent conversation

    The mismatch check fingerprints the FILE's new content and searches for
    it in the conversation, not the other way around: fingerprinting a
    specific past reply (e.g. "the most recent substantial message") is
    fragile, because the message right before a DONE-style sentinel is
    typically the write tool's own confirmation reply, which often opens
    with meta-commentary before it happens to echo the actual file content --
    fingerprinting just its first N characters can land on that preamble and
    never match, even for a genuinely correct write."""
    current_content = _read_homework_file(workspace_dir, homework_dir, problem_index)
    if not current_content:
        return "not_written"

    normalized_current = _normalize_for_compare(current_content)
    normalized_initial = _normalize_for_compare(initial_content)
    if normalized_initial and normalized_initial not in normalized_current:
        return "overwritten"
    if len(current_content) <= len(initial_content) + 5:
        return "not_written"

    new_content = current_content[len(initial_content):]
    fingerprint = _normalize_for_compare(new_content)[:80]
    if fingerprint:
        recent_text = _normalize_for_compare(
            " ".join(entry.get("content", "") for entry in conversation_history[-recent_turns:])
        )
        if fingerprint not in recent_text:
            return "mismatch"

    return None


def _build_recheck_instruction(new_content: str, done_sentinel: str) -> str:
    """Builds the extra system-role instruction used to force a grounded,
    final re-check before honoring a DONE-style sentinel, for the
    no-deterministic-problem-found path only (a diagnosed problem is handled
    entirely by a fixed correction message, no 32B call -- see
    scripts/prepare_openclaw_test_scripts.sh for why).

    The ACTUAL new content is shown (scoped to just what was newly added,
    not the whole file with its Problem:/prior scaffolding) so the simulator
    can independently judge it against its own stated requirements one more
    time, grounded in real content instead of conversational impression
    alone."""
    return (
        "(Internal note, not shown to the AI: here is exactly what was newly "
        f"added:\\n\\n{{new_content}}\\n\\n"
        "Take a close look and judge it the same way you would in step 1 -- does "
        "it satisfy everything you actually care about? If something is still "
        "off, point it out naturally and ask for a fix, the same way you always "
        f"would. If it genuinely looks good, respond with exactly {{done_sentinel}}.)"
    )
'''


def patch_file(
    filename,
    done_sentinel,
    homework_dir,
    role_label,
    generate_fn_name,
    generate_fn_anchor,
    run_fn_anchor,
    run_fn_call_anchor,
    done_check_anchor,
    msg_var,
    correction_templates,
    generic_correction_template,
):
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
    helpers = HELPERS_TEMPLATE.format(
        marker=marker,
        templates=repr(correction_templates),
        generic=repr(generic_correction_template),
    )
    text = text.replace(sentinel_line, sentinel_line + helpers, 1)

    # 3. Thread an optional extra_instruction param into generate_*_message,
    #    appended as one more system-role message right before the API call.
    if generate_fn_anchor not in text:
        raise SystemExit(
            f"patch failed: {generate_fn_name} signature anchor not found in "
            f"{filename} (openclaw-test script may have changed upstream -- "
            f"re-verify this patch):\n{generate_fn_anchor!r}"
        )
    text = text.replace(generate_fn_anchor, generate_fn_anchor.replace(
        "    max_retries: int = 3,\n) -> str:",
        "    max_retries: int = 3,\n    extra_instruction: str | None = None,\n) -> str:",
        1,
    ).replace(
        "        *conversation_history,\n    ]\n",
        "        *conversation_history,\n    ]\n"
        "    if extra_instruction:\n"
        '        messages.append({"role": "system", "content": extra_instruction})\n',
        1,
    ), 1)

    # 4. Thread workspace_dir into the run_one_* function signature.
    if run_fn_anchor not in text:
        raise SystemExit(
            f"patch failed: run-function signature anchor not found in {filename} "
            f"(openclaw-test script may have changed upstream -- re-verify this patch):\n"
            f"{run_fn_anchor!r}"
        )
    text = text.replace(run_fn_anchor, run_fn_anchor.replace(
        "    problem_index: int,",
        "    problem_index: int,\n    workspace_dir: str,",
        1,
    ), 1)

    # 5. Capture initial_content right after conversation_history is set up.
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

    # 6. Replace the "if DONE_SENTINEL in <msg_var>: ... return True" block
    #    with the diagnose-then-recheck version.
    if done_check_anchor not in text:
        raise SystemExit(
            f"patch failed: DONE_SENTINEL check anchor not found in {filename} "
            f"(openclaw-test script may have changed upstream -- re-verify this patch):\n"
            f"{done_check_anchor!r}"
        )
    new_check = (
        f'if DONE_SENTINEL in {msg_var}:\n'
        f'            _diagnosis = _diagnose_homework_file(\n'
        f'                workspace_dir, "{homework_dir}", problem_index, initial_content, conversation_history,\n'
        f'            )\n'
        f'            if _diagnosis:\n'
        f'                {msg_var} = _CORRECTION_TEMPLATES[_diagnosis].format(\n'
        f'                    index=problem_index, original=initial_content,\n'
        f'                )\n'
        f'                print(\n'
        f'                    f"\\n  Turn {{turn + 1}}: {role_label} said {done_sentinel} but file check failed "\n'
        f'                    f"(diagnosis={{_diagnosis}}, {marker}) -- continuing instead of ending session"\n'
        f'                )\n'
        f'            else:\n'
        f'                _new_content = _read_homework_file(workspace_dir, "{homework_dir}", problem_index)[len(initial_content):]\n'
        f'                _recheck_instruction = _build_recheck_instruction(_new_content, DONE_SENTINEL)\n'
        f'                try:\n'
        f'                    _recheck_msg = {generate_fn_name}(\n'
        f'                        external_client, model, problem_index, conversation_history,\n'
        f'                        max_retries=max_retries, extra_instruction=_recheck_instruction,\n'
        f'                    )\n'
        f'                except Exception as _e:\n'
        f'                    print(f"  [warn] re-check call failed ({{_e}}), falling back to fixed correction message")\n'
        f'                    _recheck_msg = None\n'
        f'                if _recheck_msg is None:\n'
        f'                    _recheck_msg = _GENERIC_CORRECTION_TEMPLATE.format(index=problem_index)\n'
        f'                if DONE_SENTINEL in _recheck_msg:\n'
        f'                    print(f"\\n  Turn {{turn + 1}}: {role_label} confirmed problem {{problem_index}} is done! (file + re-check verified, {marker})")\n'
        f'                    return True\n'
        f'                print(\n'
        f'                    f"\\n  Turn {{turn + 1}}: {role_label} said {done_sentinel} but re-check did not confirm "\n'
        f'                    f"done ({marker}) -- continuing instead of ending session"\n'
        f'                )\n'
        f'                {msg_var} = _recheck_msg'
    )
    text = text.replace(done_check_anchor, new_check, 1)

    # 7. Pass workspace_dir at the run_one_* call site.
    if run_fn_call_anchor not in text:
        raise SystemExit(
            f"patch failed: run-function call-site anchor not found in {filename} "
            f"(openclaw-test script may have changed upstream -- re-verify this patch):\n"
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


patch_file(
    filename="student_chat.py",
    done_sentinel="HOMEWORK_DONE",
    homework_dir="homework",
    role_label="Student",
    generate_fn_name="generate_student_message",
    generate_fn_anchor=(
        "def generate_student_message(\n"
        "    client: OpenAI,\n"
        "    model: str,\n"
        "    problem_index: int,\n"
        "    conversation_history: list[dict],\n"
        "    max_retries: int = 3,\n"
        ") -> str:"
    ),
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
    correction_templates={
        "overwritten": (
            "Wait, I think part of what was originally in homework/{index}.txt is "
            "now missing -- here's exactly what it's supposed to say before the "
            "answer:\n\n{original}\n\n"
            "Can you check the file and make sure all of that is still there, with "
            "the answer added after it, not replacing it?"
        ),
        "not_written": "Wait, I don't think anything actually got saved to homework/{index}.txt -- it looks exactly the same as before. Can you check and make sure the answer is really written to the file this time?",
        "mismatch": "Wait, what's in homework/{index}.txt doesn't look like what you just showed me -- can you double check what actually got saved?",
    },
    generic_correction_template=(
        "Wait, that doesn't look right -- I don't think it actually got saved "
        "correctly. Can you check the file homework/{index}.txt again and make "
        "sure it's really there?"
    ),
)

patch_file(
    filename="TA_chat.py",
    done_sentinel="GRADING_DONE",
    homework_dir="homework1",
    role_label="TA",
    generate_fn_name="generate_ta_message",
    generate_fn_anchor=(
        "def generate_ta_message(\n"
        "    client: OpenAI,\n"
        "    model: str,\n"
        "    problem_index: int,\n"
        "    conversation_history: list[dict],\n"
        "    max_retries: int = 3,\n"
        ") -> str:"
    ),
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
    correction_templates={
        "overwritten": (
            "Wait, I think part of what was originally in homework1/{index}.txt is "
            "now missing -- here's exactly what it's supposed to contain before "
            "your grading comments:\n\n{original}\n\n"
            "Can you check the file and make sure all of that is still there, with "
            "your comments added after it, not replacing it?"
        ),
        "not_written": "Wait, I don't think your grading comments actually got saved to homework1/{index}.txt -- it looks exactly the same as before. Can you check and make sure they're really written to the file this time?",
        "mismatch": "Wait, what's in homework1/{index}.txt doesn't look like the grading comments you just showed me -- can you double check what actually got saved?",
    },
    generic_correction_template=(
        "Wait, that doesn't look right -- I don't think it actually got saved "
        "correctly. Can you check the file homework1/{index}.txt again and make "
        "sure it's really there?"
    ),
)

patch_file(
    filename="teacher_chat.py",
    done_sentinel="COMMENT_DONE",
    homework_dir="homework2",
    role_label="Teacher",
    generate_fn_name="generate_teacher_message",
    generate_fn_anchor=(
        "def generate_teacher_message(\n"
        "    client: OpenAI,\n"
        "    model: str,\n"
        "    problem_index: int,\n"
        "    conversation_history: list[dict],\n"
        "    max_retries: int = 3,\n"
        ") -> str:"
    ),
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
    correction_templates={
        "overwritten": (
            "Wait, I think part of what was originally in homework2/{index}.txt is "
            "now missing -- here's exactly what it's supposed to contain before "
            "your comments:\n\n{original}\n\n"
            "Can you check the file and make sure all of that is still there, with "
            "your comments added after it, not replacing it?"
        ),
        "not_written": "Wait, I don't think your comments actually got saved to homework2/{index}.txt -- it looks exactly the same as before. Can you check and make sure they're really written to the file this time?",
        "mismatch": "Wait, what's in homework2/{index}.txt doesn't look like the comments you just showed me -- can you double check what actually got saved?",
    },
    generic_correction_template=(
        "Wait, that doesn't look right -- I don't think it actually got saved "
        "correctly. Can you check the file homework2/{index}.txt again and make "
        "sure it's really there?"
    ),
)
PY

echo "已生成 openclaw-test 补丁: ${DEST_DIR}（model 字段兼容 + HOMEWORK_DONE/GRADING_DONE/COMMENT_DONE 前置事实核验 + 32B 复核）"
