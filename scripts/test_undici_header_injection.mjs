// Standalone, OpenClaw-independent test: does patching the undici global
// dispatcher (instead of globalThis.fetch) let a per-call dynamic header
// reach a freshly-constructed `openai` SDK client's outbound request?
//
// Why this test exists: the rl-training-headers plugin's original approach
// (patch globalThis.fetch) was confirmed not to work -- headers never reach
// the wire. Source tracing (2026-07-09) showed OpenClaw's own
// `createClient()` constructs a brand-new `OpenAI` client on every single
// completion call (no caching), and never passes an explicit `fetch` option,
// so it relies on the SDK's own default fetch resolution. That default
// resolution is suspected to capture a `fetch` reference at `openai` package
// module-load time (before any plugin code runs), not per-instantiation --
// which would make `globalThis.fetch` patching permanently ineffective
// regardless of timing.
//
// undici's `setGlobalDispatcher` operates one layer below any cached fetch
// function reference: it registers a dispatcher under a global Symbol.for()
// key that Node's built-in fetch (and openai SDK's default fetch, which is
// Node's built-in fetch) consults at actual network-call time, not at
// module-load or client-construction time. If this holds, header injection
// should work without needing to fix `globalThis.fetch` patch timing at all.
//
// This script is fully standalone -- no OpenClaw, no GPU. Run with:
//   npm install openai undici   (in a scratch dir, or reuse OpenClaw's own
//                                 node_modules if resolvable)
//   node test_undici_header_injection.mjs

import { getGlobalDispatcher, setGlobalDispatcher } from "undici";
import { AsyncLocalStorage } from "node:async_hooks";
import http from "node:http";
import OpenAI from "openai";

const headerStore = new AsyncLocalStorage();

const rlHeaderInterceptor = (dispatch) => {
  return function InterceptedDispatch(opts, handler) {
    const scoped = headerStore.getStore();
    if (scoped) {
      opts.headers = { ...(opts.headers || {}), ...scoped };
    }
    return dispatch(opts, handler);
  };
};

setGlobalDispatcher(getGlobalDispatcher().compose(rlHeaderInterceptor));

let observedHeaders = null;

const server = http.createServer((req, res) => {
  observedHeaders = req.headers;
  console.log("[mock-server] received headers:\n" + JSON.stringify(req.headers, null, 2));
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({
    id: "test",
    object: "chat.completion",
    created: 0,
    model: "test-model",
    choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
  }));
});

server.listen(30099, async () => {
  console.log("[mock-server] listening on 127.0.0.1:30099");

  // Mirror OpenClaw's own createClient(): no explicit `fetch` option passed.
  const client = new OpenAI({ apiKey: "test-key", baseURL: "http://127.0.0.1:30099/v1" });

  await headerStore.run(
    { "x-session-id": "test-session-123", "x-turn-type": "main" },
    async () => {
      await client.chat.completions.create({
        model: "test-model",
        messages: [{ role: "user", content: "hello" }],
      });
    },
  );

  server.close(() => {
    const ok =
      observedHeaders &&
      observedHeaders["x-session-id"] === "test-session-123" &&
      observedHeaders["x-turn-type"] === "main";
    console.log(ok ? "\n=== RESULT: PASS (headers reached the wire) ===" : "\n=== RESULT: FAIL (headers did NOT reach the wire) ===");
    process.exit(ok ? 0 : 1);
  });
});
