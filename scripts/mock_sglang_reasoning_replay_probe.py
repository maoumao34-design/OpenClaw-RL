#!/usr/bin/env python3
"""CPU-only probe: how faithfully does OpenClaw replay an assistant turn back
to the model on the NEXT turn? -- no GPU, no real model, just request inspection.

Why this exists
---------------
2026-09-03 concluded that OpenClaw strips reasoning when replaying history
(`dropReasoningFromHistory`), and the trajectory-level sample design was
deleted on that basis -- roughly 96 of 120 rounds had been dropped because a
flat sequence cannot honestly claim turn 2 saw turn 1's thinking.

2026-09-10, first run of this probe: reasoning is PRESERVED. The marker planted
in turn 1's reasoning_content came back intact in turn 2's replay. So the 09-03
diagnosis was wrong, and the trajectory design was deleted for the wrong reason.

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
agent is allowed to call must exist, or turn 2 never happens -- adjust
TOOL_NAME/TOOL_ARGS if the agent under test exposes different tools.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

TOOL_NAME = "exec"
TOOL_ARGS = {"command": "echo reasoning-replay-probe"}

REASONING_MARKER = "PROBE_REASONING_MARKER_8f3a"
CONTENT_MARKER = "PROBE_CONTENT_MARKER_1c7d"
TOOL_CALL_ID = "call_probe_marker_5e21"

_state = {"turn": 0, "sent": None}


def _describe(msg):
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
    out["content_has_marker"] = CONTENT_MARKER in blob
    for f in ("reasoning_content", "reasoning", "thinking"):
        if f in msg:
            v = msg[f]
            out[f] = (f"present(len={len(v)})" if isinstance(v, str)
                      else f"present({type(v).__name__})")
            if isinstance(v, str) and REASONING_MARKER in v:
                out[f] += " CONTAINS_MARKER"
    tcs = msg.get("tool_calls")
    if tcs:
        out["tool_calls"] = [
            {"id": tc.get("id"),
             "name": (tc.get("function") or {}).get("name")}
            for tc in tcs if isinstance(tc, dict)
        ]
    return out


def _diff_replay(sent, got):
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
    if REASONING_MARKER not in json.dumps(got, ensure_ascii=False):
        diffs.append("reasoning: MARKER GONE (stripped)")
    elif REASONING_MARKER not in g_reason:
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
    if CONTENT_MARKER not in g_blob:
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
        _state["turn"] += 1
        turn = _state["turn"]

        assistants = [m for m in messages
                      if isinstance(m, dict) and m.get("role") == "assistant"]

        print("=" * 72)
        print(f"[probe] REQUEST #{turn}  ({len(messages)} messages, "
              f"{len(assistants)} assistant)")

        if _state["sent"] is None:
            reply = {
                "role": "assistant",
                "content": f"{CONTENT_MARKER}",
                "reasoning_content": (
                    f"{REASONING_MARKER} I will run one command to inspect the environment."
                ),
                "tool_calls": [{
                    "id": TOOL_CALL_ID,
                    "type": "function",
                    "function": {"name": TOOL_NAME, "arguments": json.dumps(TOOL_ARGS)},
                }],
            }
            _state["sent"] = reply
            print("[probe] turn 1 -- replying with reasoning + content + tool_call, "
                  "all marked")
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
            sent = _state["sent"]
            got = assistants[0]
            print(f"[probe] turn {turn} -- inspecting the replayed assistant message")
            print(f"    sent: {json.dumps(_describe(sent), ensure_ascii=False)}")
            print(f"    got : {json.dumps(_describe(got), ensure_ascii=False)}")

            diffs = _diff_replay(sent, got)
            print()
            if diffs:
                print("[probe] VERDICT: replay is NOT faithful -- "
                      f"{len(diffs)} difference(s):")
                for d in diffs:
                    print(f"           - {d}")
                print("[probe]          a flat trajectory built from these turns "
                      "would not match what the model actually saw")
            else:
                print("[probe] VERDICT: replay is FAITHFUL -- reasoning, content and "
                      "tool_call ids all came back unchanged")
                print("[probe]          a flat trajectory over these turns is "
                      "well-founded")
            reply = {"role": "assistant", "content": "probe done"}
            finish = "stop"

        print("=" * 72, flush=True)

        payload = json.dumps({
            "id": f"probe-{turn}", "object": "chat.completion", "created": 0,
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
