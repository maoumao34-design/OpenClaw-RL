#!/usr/bin/env python3
"""Does the chat template render historical `reasoning_content` into tokens?

This is the SECOND gate, and the one that decides what the model actually saw.

Gate 1 is OpenClaw: does it put `reasoning_content` on each historical
assistant message when it calls the model? Settled -- `transcript-policy.ts:113`
preserves whenever `model.reasoning === true`, and the official
`openclaw_cfg/openclaw.json` declares exactly that.

Gate 2 is the serving side. The proxy takes those messages and calls
`self.tokenizer.apply_chat_template(...)` (see the combine-select patch). The
OpenAI wire format is a LIST of messages; the flat token sequence the model
reads is produced by the template. `reasoning_content` is not a standard
OpenAI field, so whether the template emits it -- for the last assistant turn,
for every assistant turn, or never -- is a property of the template alone.

Nothing downstream can recover thinking the template dropped. So if the answer
here is "never", then within one question the model sees only what was
actually DONE, regardless of what OpenClaw sent -- and the flat-trajectory
design has to reckon with that rather than with OpenClaw's policy.

CPU only, no GPU, no weights -- the tokenizer is enough.

Usage:
    python3 check_chat_template_reasoning.py <tokenizer_path_or_hf_id>
"""
import sys

R1 = "PROBE_REASONING_TURN1_8f3a"
R2 = "PROBE_REASONING_TURN2_2b6e"
C1 = "PROBE_CONTENT_TURN1_1c7d"
C2 = "PROBE_CONTENT_TURN2_9d40"

MESSAGES = [
    {"role": "system", "content": "You are a test agent."},
    {"role": "user", "content": "Do the task."},
    {
        "role": "assistant",
        "content": C1,
        "reasoning_content": f"{R1} first I inspect the directory.",
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "exec", "arguments": {"command": "ls"}},
        }],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "a.txt b.txt"},
    {
        "role": "assistant",
        "content": C2,
        "reasoning_content": f"{R2} now I write the report file.",
        "tool_calls": [{
            "id": "call_2", "type": "function",
            "function": {"name": "exec", "arguments": {"command": "touch r.md"}},
        }],
    },
    {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    path = sys.argv[1]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    text = tok.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True,
    )

    found = {
        "turn1 reasoning": R1 in text,
        "turn2 reasoning": R2 in text,
        "turn1 content": C1 in text,
        "turn2 content": C2 in text,
    }

    print("=" * 72)
    print(f"tokenizer: {path}")
    print(f"rendered prompt: {len(text)} chars, "
          f"{len(tok(text, add_special_tokens=False)['input_ids'])} tokens")
    print("-" * 72)
    for k, v in found.items():
        print(f"  {'IN PROMPT' if v else 'DROPPED  '}  {k}")
    print("-" * 72)

    n_reason = found["turn1 reasoning"] + found["turn2 reasoning"]
    if not (found["turn1 content"] and found["turn2 content"]):
        print("VERDICT: UNEXPECTED -- even the assistant CONTENT did not survive.")
        print("         The message list is not being rendered the way this probe")
        print("         assumes; read the printed prompt below before concluding")
        print("         anything about reasoning.")
    elif n_reason == 2:
        print("VERDICT: reasoning from BOTH prior turns is in the token sequence.")
        print("         Within one question the model does see every turn's")
        print("         thinking, and a flat trajectory over those turns is")
        print("         well-founded on this gate.")
    elif n_reason == 0:
        print("VERDICT: the template DROPS historical reasoning entirely.")
        print("         Within one question the model sees only what was DONE")
        print("         (tool calls and their results), never the earlier")
        print("         thinking -- no matter what OpenClaw put on the wire.")
    else:
        kept = "turn 1" if found["turn1 reasoning"] else "turn 2"
        print(f"VERDICT: the template keeps only {kept}'s reasoning.")
        print("         Partial retention -- a flat trajectory would claim the")
        print("         model saw thinking it never received.")
    print("=" * 72)
    print()
    print("--- rendered prompt (verbatim) ---")
    print(text)


if __name__ == "__main__":
    main()
