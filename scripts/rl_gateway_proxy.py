#!/usr/bin/env python3
"""Minimal proxy replacing OpenClaw gateway at port 18789.

Adds X-Session-Id (from request body.user) and X-Turn-Type: main before
forwarding to the RL training proxy (OpenClawOPDAPIServer) at port 30000.
"""

import os
import httpx
import uvicorn
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

OPENCLAW_GATEWAY_TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
SGLANG_API_KEY = os.environ.get("SGLANG_API_KEY", "openclaw-rl-key")
RL_PROXY_URL = os.environ.get("RL_PROXY_URL", "http://127.0.0.1:30000")
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "18789"))
GATEWAY_BIND = os.environ.get("GATEWAY_BIND", "127.0.0.1")

app = FastAPI()


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
):
    if OPENCLAW_GATEWAY_TOKEN and authorization != f"Bearer {OPENCLAW_GATEWAY_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    session_id = body.get("user") or body.get("session_id") or "unknown"

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(
            f"{RL_PROXY_URL}/v1/chat/completions",
            json=body,
            headers={
                "Authorization": f"Bearer {SGLANG_API_KEY}",
                "Content-Type": "application/json",
                "X-Session-Id": session_id,
                "X-Turn-Type": "main",
            },
        )

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


if __name__ == "__main__":
    uvicorn.run(app, host=GATEWAY_BIND, port=GATEWAY_PORT, log_level="info")
