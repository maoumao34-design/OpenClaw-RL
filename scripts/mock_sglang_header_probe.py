#!/usr/bin/env python3
"""Minimal CPU-only mock sglang backend for verifying whether OpenClaw's
outbound /v1/chat/completions request carries X-Session-Id/X-Turn-Type
headers, and/or a "[RL-TRAINING-META] session_id=... turn_type=..." marker
in a system message -- no GPU, no real model, just request inspection.

Usage:
    python3 mock_sglang_header_probe.py [port]   # default port 30000

Point OpenClaw's sglang provider baseUrl at this server, fire one real
request through the OpenClaw gateway, and check this process's stdout for
the headers/marker it received.
"""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

_RL_META_RE = re.compile(r"\[RL-TRAINING-META\] session_id=(\S*) turn_type=(\S*)")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        print("=" * 70)
        print(f"[mock-sglang] {self.command} {self.path}")
        print("[mock-sglang] headers received:")
        for k, v in self.headers.items():
            marker = " <---" if k.lower() in ("x-session-id", "x-turn-type") else ""
            print(f"    {k}: {v}{marker}")
        session_id = self.headers.get("X-Session-Id")
        turn_type = self.headers.get("X-Turn-Type")
        if session_id or turn_type:
            print(f"[mock-sglang] HEADER RESULT: X-Session-Id={session_id!r} X-Turn-Type={turn_type!r}")
        else:
            print("[mock-sglang] HEADER RESULT: neither header present")

        marker_match = None
        try:
            parsed = json.loads(body)
            for msg in parsed.get("messages", []):
                if isinstance(msg, dict) and msg.get("role") == "system" and isinstance(msg.get("content"), str):
                    m = _RL_META_RE.search(msg["content"])
                    if m:
                        marker_match = m
                        break
        except (json.JSONDecodeError, AttributeError):
            pass
        if marker_match:
            print(f"[mock-sglang] BODY MARKER RESULT: session_id={marker_match.group(1)!r} turn_type={marker_match.group(2)!r}")
        else:
            print("[mock-sglang] BODY MARKER RESULT: no [RL-TRAINING-META] marker found in any system message")
        print("=" * 70, flush=True)

        response = {
            "id": "mock-probe",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen3-4b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok (mock probe response)"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # keep stdout to just our own prints above


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"[mock-sglang] listening on 127.0.0.1:{port}")
    server.serve_forever()
