#!/usr/bin/env python3
"""CPU-only probe: does OpenClaw replay an assistant turn's reasoning back to
the model on the NEXT turn? -- no GPU, no real model, just request inspection.

Why this exists
---------------
2026-09-03 concluded that OpenClaw strips reasoning when replaying history
(`dropReasoningFromHistory`), and the trajectory-level sample design was
deleted on that basis -- roughly 96 of 120 rounds had been dropped because a
flat sequence cannot honestly claim turn 2 saw turn 1's thinking.

Reading transcript-policy.ts on 2026-09-10 contradicts that. The policy comes
from buildUnownedProviderTransportReplayFallback (confirmed: no runtime plugin
claims the metaclaw-bench provider), whose strict-OpenAI-compatible branch sets

    dropReasoningFromHistory = !shouldPreserveReasoningContentReplay(params)

and shouldPreserveReasoningContentReplay is true when `model.reasoning === true`
-- which MetaClaw's own openclaw_cfg/openclaw.json declares. By that reading
reasoning should be PRESERVED, and the 09-03 diagnosis was wrong.

Source reading has been wrong in both directions on this project, so settle it
by observation. The question is about OpenClaw's replay behaviour, not about
the model, so no GPU is needed: this stub plays the model's part.

What it does
------------
Turn 1: answers with reasoning_content AND a tool_call, so OpenClaw runs the
        tool and comes back for a second turn.
Turn 2: dumps the structure of the assistant message OpenClaw replayed, and
        prints the verdict.

Usage
-----
    python3 mock_sglang_reasoning_replay_probe.py [port]     # default 30099

Point the agent's provider baseUrl at http://127.0.0.1:<port>/v1, fire ONE
real request through the OpenClaw gateway, and read this process's stdout.
A tool the agent is allowed to call must exist; `exec`/`bash` with a harmless
command is the default below -- adjust TOOL_NAME/TOOL_ARGS if the agent under
test exposes different tools, otherwise turn 2 never happens.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# The tool turn 1 asks for. It only has to be a tool the agent will actually
# run and report back on; the command itself is irrelevant.
TOOL_NAME = "exec"
TOOL_ARGS = {"command": "echo reasoning-replay-probe"}

# Marker text placed in turn 1's reasoning. If OpenClaw replays reasoning,
# this string comes back in turn 2's request; that is the whole test.
REASONING_MARKER = "PROBE_REASONING_MARKER_8f3a"

_state = {"turn": 0}


def _describe(msg):
    """Field-level shape of one message -- names only, no long bodies."""
    out = {"role": msg.get("role"), "keys": sorted(msg.keys())}
    c = msg.get("content")
    if isinstance(c, str):
        out["content"] = f"str(len={len(c)})"
        out["content_has_think_tag"] = "<think>" in c
        out["content_has_marker"] = REASONING_MARKER in c
    elif isinstance(c, list):
        out["content"] = [
            (b.get("type") if isinstance(b, dict) else type(b).__name__) for b in c
        ]
        blob = json.dumps(c, ensure_ascii=False)
        out["content_has_think_tag"] = "<think>" in blob
        out["content_has_marker"] = REASONING_MARKER in blob
    else:
        out["content"] = type(c).__name__
        out["content_has_think_tag"] = False
        out["content_has_marker"] = False
    for f in ("reasoning_content", "reasoning", "thinking"):
        if f in msg:
            v = msg[f]
            out[f] = f"present(len={len(v)})" if isinstance(v, str) else f"present({type(v).__name__})"
            if isinstance(v, str) and REASONING_MARKER in v:
                out[f] += " CONTAINS_MARKER"
    if msg.get("tool_calls"):
        out["tool_calls"] = len(msg["tool_calls"])
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence the default access log
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

        print("=" * 72)
        print(f"[probe] REQUEST #{turn}  ({len(messages)} messages)")

        if turn == 1:
            print("[probe] turn 1 -- replying with reasoning_content + a tool_call")
            reply = {
                "role": "assistant",
                "content": "",
                "reasoning_content": (
                    f"{REASONING_MARKER} I will run one command to inspect the environment."
                ),
                "tool_calls": [{
                    "id": "call_probe_1",
                    "type": "function",
                    "function": {"name": TOOL_NAME,
                                 "arguments": json.dumps(TOOL_ARGS)},
                }],
            }
            finish = "tool_calls"
        else:
            assistants = [m for m in messages
                          if isinstance(m, dict) and m.get("role") == "assistant"]
            print(f"[probe] turn {turn} -- {len(assistants)} assistant message(s) replayed")
            for i, m in enumerate(assistants):
                print(f"    assistant[{i}]: {json.dumps(_describe(m), ensure_ascii=False)}")

            blob = json.dumps(messages, ensure_ascii=False)
            found = REASONING_MARKER in blob
            print()
            print(f"[probe] VERDICT: turn 1's reasoning was "
                  f"{'PRESERVED -- dropReasoningFromHistory is OFF' if found else 'STRIPPED -- dropReasoningFromHistory is ON'}")
            print(f"[probe]          (marker {REASONING_MARKER!r} "
                  f"{'found' if found else 'absent'} anywhere in the replayed messages)")
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
    print(f"[probe] point the provider baseUrl here, then send ONE message through the agent")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
