#!/usr/bin/env python3
"""CPU-only probe: how faithfully does OpenClaw replay an assistant turn back
to the model on the NEXT turn? -- no GPU, no real model, just request inspection.

Why this exists
---------------
2026-09-03 concluded that OpenClaw strips reasoning when replaying history
(`dropReasoningFromHistory`), and the trajectory-level sample design was
deleted on that basis -- roughly 96 of 120 rounds had been dropped because a
flat sequence cannot honestly claim turn 2 saw turn 1's thinking.

2026-09-10, first run of this probe: reasoning came back intact. But that run
only generated ONE turn and read it back, and turn 1 is the wrong place to
look -- `shouldPreserveCurrentToolTurnReasoning` (thinking.ts:365) exempts the
first assistant tool turn after the latest user message EVEN WHEN the drop
policy is on. A one-turn probe therefore cannot tell the two policies apart.
It now generates two turns: under "off" both keep their reasoning, under "on"
turn 1 keeps it and turn 2 does not.

Independent of the probe, the config settles it: `transcript-policy.ts:113`
preserves whenever `model.reasoning === true`, and the official
`openclaw_cfg/openclaw.json` declares exactly that for the `metaclaw-bench`
provider. So the 09-03 diagnosis was wrong and the trajectory design was
deleted for the wrong reason.

But 96/120 rounds really were dropped. If reasoning was not the cause, the real
one is still unidentified -- and rebuilding on an unexamined failure would just
repeat it. Reading the same policy fallback, three other history-mutating
settings are ALSO on for a strict-OpenAI-compatible provider like ours:

    sanitizeToolCallIds: true, toolCallIdMode: "strict"
    applyAssistantFirstOrderingFix: true
    validateGeminiTurns / validateAnthropicTurns: true

Any of these can break the byte-level prefix containment a flat trajectory
needs -- tool-call id rewriting being the most obvious. So this probe no longer
just answers "is reasoning kept"; it diffs what was sent against what came back,
field by field, and reports every difference.

Usage
-----
    python3 mock_sglang_reasoning_replay_probe.py [port]     # default 30099

Point the agent's provider baseUrl at http://127.0.0.1:<port>/v1, fire ONE real
request through the OpenClaw gateway, and read this process's stdout. A tool the
agent is allowed to call must exist, or the follow-up requests never happen --
adjust TOOL_NAME/TOOL_ARGS if the agent under test exposes different tools.

WHAT THIS DOES **NOT** COVER
----------------------------
This probe reads the JSON that OpenClaw puts on the wire. Whether the
`reasoning_content` field then survives into the actual token sequence is a
SECOND gate, decided by the serving side's chat template, not by OpenClaw.
To answer that, look at a real recorded `prompt_text` from a later turn in
the RL proxy's turn_data and check whether the earlier turns' thinking is in
it. Only that second measurement describes what the model really saw.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

TOOL_NAME = "exec"
TOOL_ARGS = {"command": "echo reasoning-replay-probe"}

# One marker set per generated turn. Two tool-calling turns are needed, not
# one: even when dropReasoningFromHistory is ON, the FIRST assistant tool turn
# after the latest user message is exempt (`shouldPreserveCurrentToolTurnReasoning`,
# thinking.ts:365 -- it walks backwards and bails the moment it sees another
# assistant). So a two-request probe reads the one message that survives under
# either policy, and cannot tell the policies apart. Turn 2 is the discriminator:
#   flag OFF -> turn 1 AND turn 2 both keep their reasoning
#   flag ON  -> turn 1 keeps it, turn 2 is stripped
REASONING_MARKERS = ["PROBE_REASONING_MARKER_8f3a", "PROBE_REASONING_MARKER_2b6e"]
CONTENT_MARKERS = ["PROBE_CONTENT_MARKER_1c7d", "PROBE_CONTENT_MARKER_9d40"]
TOOL_CALL_IDS = ["call_probe_marker_5e21", "call_probe_marker_7a13"]

GENERATED_TURNS = 2

_state = {"request": 0, "sent": []}


def _describe(msg, step):
    """Field-level shape of one message -- names only, no long bodies."""
    out = {"role": msg.get("role"), "keys": sorted(msg.keys())}
    c = msg.get("content")
    if isinstance(c, str):
        out["content"] = f"str(len={len(c)})"
    elif isinstance(c, list):
        out["content"] = [(b.get("type") if isinstance(b, dict) else type(b).__name__)
                          for b in c]
    else:
        out["content"] = type(c).__name__
    blob = json.dumps(c, ensure_ascii=False) if c is not None else ""
    out["content_has_think_tag"] = "<think>" in blob
    out["content_has_marker"] = CONTENT_MARKERS[step] in blob
    for f in ("reasoning_content", "reasoning", "thinking"):
        if f in msg:
            v = msg[f]
            out[f] = (f"present(len={len(v)})" if isinstance(v, str)
                      else f"present({type(v).__name__})")
            if isinstance(v, str) and REASONING_MARKERS[step] in v:
                out[f] += " CONTAINS_MARKER"
    tcs = msg.get("tool_calls")
    if tcs:
        out["tool_calls"] = [
            {"id": tc.get("id"),
             "name": (tc.get("function") or {}).get("name")}
            for tc in tcs if isinstance(tc, dict)
        ]
    return out


def _diff_replay(sent, got, step):
    """What changed between the assistant message we sent and the one replayed.

    This is the property a flat-trajectory sample needs: the replayed turn must
    be byte-identical to what was generated, or the concatenated sequence is a
    fiction.
    """
    diffs = []

    s_reason = sent.get("reasoning_content") or ""
    g_reason = got.get("reasoning_content") or ""
    if not isinstance(g_reason, str):
        g_reason = json.dumps(g_reason, ensure_ascii=False)
    if REASONING_MARKERS[step] not in json.dumps(got, ensure_ascii=False):
        diffs.append("reasoning: MARKER GONE (stripped)")
    elif REASONING_MARKERS[step] not in g_reason:
        # Still there, but no longer in reasoning_content -- folded into the
        # content as a <think> block or a thinking part. Survives, but the
        # token sequence is not the one that was generated.
        diffs.append("reasoning: MOVED out of reasoning_content into content")
    elif s_reason != g_reason:
        diffs.append(f"reasoning: text differs (sent {len(s_reason)} chars, "
                     f"got {len(g_reason)})")

    s_content = sent.get("content") or ""
    g_content = got.get("content")
    g_blob = json.dumps(g_content, ensure_ascii=False) if g_content is not None else ""
    if CONTENT_MARKERS[step] not in g_blob:
        diffs.append("content: MARKER GONE")
    if type(s_content) is not type(g_content):
        diffs.append(f"content: type changed {type(s_content).__name__} -> "
                     f"{type(g_content).__name__}")

    s_ids = [tc["id"] for tc in (sent.get("tool_calls") or [])]
    g_ids = [tc.get("id") for tc in (got.get("tool_calls") or []) if isinstance(tc, dict)]
    if s_ids != g_ids:
        diffs.append(f"tool_call ids REWRITTEN: sent {s_ids} -> got {g_ids}")

    s_names = [tc["function"]["name"] for tc in (sent.get("tool_calls") or [])]
    g_names = [(tc.get("function") or {}).get("name")
               for tc in (got.get("tool_calls") or []) if isinstance(tc, dict)]
    if s_names != g_names:
        diffs.append(f"tool names differ: {s_names} -> {g_names}")

    return diffs


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {}
        messages = parsed.get("messages") or []
        _state["request"] += 1
        req = _state["request"]

        assistants = [m for m in messages
                      if isinstance(m, dict) and m.get("role") == "assistant"]
        n_sent = len(_state["sent"])

        print("=" * 72)
        print(f"[probe] REQUEST #{req}  ({len(messages)} messages, "
              f"{len(assistants)} assistant)")

        if n_sent < GENERATED_TURNS:
            step = n_sent
            reply = {
                "role": "assistant",
                "content": CONTENT_MARKERS[step],
                "reasoning_content": (
                    f"{REASONING_MARKERS[step]} step {step + 1}: I will run one "
                    "command to inspect the environment."
                ),
                "tool_calls": [{
                    "id": TOOL_CALL_IDS[step],
                    "type": "function",
                    "function": {"name": TOOL_NAME, "arguments": json.dumps(TOOL_ARGS)},
                }],
            }
            _state["sent"].append(reply)
            print(f"[probe] generating turn {step + 1}/{GENERATED_TURNS} -- "
                  "reasoning + content + tool_call, all marked")
            finish = "tool_calls"

        elif not assistants:
            # No prior assistant in this request -- a fresh session or a
            # heartbeat, NOT a replay. Rendering a verdict here would be
            # meaningless; say so instead of implying 'stripped'.
            print("[probe] NO VERDICT: this request carries no prior assistant "
                  "message, so it is not a replay (fresh session / heartbeat). "
                  "Ignore it.")
            reply = {"role": "assistant", "content": "probe idle"}
            finish = "stop"

        else:
            print(f"[probe] inspecting the replay of {GENERATED_TURNS} generated turns")
            all_diffs = []
            for step, sent in enumerate(_state["sent"]):
                got = assistants[step] if step < len(assistants) else None
                if got is None:
                    all_diffs.append((step, ["message MISSING from replay entirely"]))
                    print(f"    turn {step + 1}: MISSING from replay")
                    continue
                print(f"    turn {step + 1} sent: "
                      f"{json.dumps(_describe(sent, step), ensure_ascii=False)}")
                print(f"    turn {step + 1} got : "
                      f"{json.dumps(_describe(got, step), ensure_ascii=False)}")
                d = _diff_replay(sent, got, step)
                if d:
                    all_diffs.append((step, d))

            print()
            if not all_diffs:
                print("[probe] VERDICT: replay is FAITHFUL for ALL "
                      f"{GENERATED_TURNS} turns -- reasoning, content and "
                      "tool_call ids all came back unchanged")
                print("[probe]          dropReasoningFromHistory is OFF (under the "
                      "drop policy, turn 2 would have lost its reasoning while "
                      "turn 1 kept it)")
            else:
                print(f"[probe] VERDICT: replay is NOT faithful -- "
                      f"{len(all_diffs)} of {GENERATED_TURNS} turns differ:")
                for step, d in all_diffs:
                    for one in d:
                        print(f"           - turn {step + 1}: {one}")
                only_later = all(step > 0 for step, _ in all_diffs)
                if only_later and all(
                    any("MARKER GONE" in x for x in d) for _, d in all_diffs
                ):
                    print("[probe]          turn 1 kept its reasoning and the later "
                          "turn(s) did not -- this is exactly the "
                          "dropReasoningFromHistory ON signature")
                print("[probe]          a flat trajectory built from these turns "
                      "would not match what the model actually saw")
            reply = {"role": "assistant", "content": "probe done"}
            finish = "stop"

        print("=" * 72, flush=True)

        payload = json.dumps({
            "id": f"probe-{req}", "object": "chat.completion", "created": 0,
            "model": parsed.get("model", "probe"),
            "choices": [{"index": 0, "message": reply, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 30099
    print(f"[probe] listening on http://127.0.0.1:{port}  (CPU only, no model)")
    print("[probe] point the provider baseUrl here, then send ONE message "
          "through the agent")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
