#!/bin/bash
# Patch the official rl-training-headers OpenClaw plugin's injection mechanism.
#
# Why: the plugin's original injection technique -- monkey-patching
# globalThis.fetch inside before_prompt_build -- is confirmed non-functional
# on this OpenClaw build (2026.6.9): headers never reach the actual outbound
# request (verified via CPU-only mock-server packet capture, 2026-07-07).
# Root cause (verified via OpenClaw source tracing, 2026-07-09): OpenClaw's
# provider transport constructs a brand-new `openai` SDK client on every
# single completion call (openai-completions.ts createClient(), no caching),
# and never passes an explicit `fetch` option -- it relies on the SDK's own
# default fetch resolution, which most likely captures a `fetch` reference at
# `openai` package module-load time, before any plugin code ever runs. This
# makes patching globalThis.fetch permanently ineffective regardless of
# plugin-registration timing.
#
# Fix: intercept one layer below the cached fetch reference, at undici's
# global dispatcher (Node's built-in fetch is undici-based; the dispatcher is
# resolved per-call from a global Symbol.for() registry, not baked into
# whichever fetch function got cached). Verified working in isolation via
# scripts/test_undici_header_injection.mjs (2026-07-09, PASS: headers reached
# a freshly-constructed openai SDK client's outbound request).
#
# ONLY the injection mechanism changes. The classification logic --
# before_prompt_build hook, ctx.trigger/ctx.sessionId, SIDE_TRIGGERS set -- is
# left byte-for-byte identical to the official plugin. This is a deviation
# from the paper's shipped plugin code, documented in docs/issues_log.md:
# the official injection technique doesn't work on this OpenClaw version: we
# swapped the transport-layer technique, not the design.
#
# Official OpenClaw-RL-official/extensions/rl-training-headers/ is left
# untouched; this writes a patched copy (as plain JS, matching the
# already-verified-loadable deployment format from 2026-06-23) to DEST_DIR.
set -euo pipefail

REPO_ROOT=${1:?usage: prepare_patched_rl_training_headers.sh <repo_root> <dest_dir>}
DEST_DIR=${2:?usage: prepare_patched_rl_training_headers.sh <repo_root> <dest_dir>}
SRC_DIR="${REPO_ROOT}/extensions/rl-training-headers"
SRC="${SRC_DIR}/index.ts"

if [ ! -f "${SRC}" ]; then
    echo "错误：找不到官方插件源码 ${SRC}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

python3 - "${SRC}" "${DEST_DIR}/index.js" <<'PY'
import re
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
text = open(src_path, encoding="utf-8").read()

old_block = '''import { AsyncLocalStorage } from "node:async_hooks";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

type RlTrainingConfig = {
  sessionIdHeader: string;
  turnTypeHeader: string;
};

function resolveConfig(api: OpenClawPluginApi): RlTrainingConfig {
  const cfg = (api.pluginConfig ?? {}) as Partial<RlTrainingConfig>;
  return {
    sessionIdHeader: cfg.sessionIdHeader ?? "X-Session-Id",
    turnTypeHeader: cfg.turnTypeHeader ?? "X-Turn-Type",
  };
}

// Triggers classified as "side" (non-user-facing housekeeping runs).
const SIDE_TRIGGERS = new Set(["heartbeat", "memory", "cron"]);

export default function register(api: OpenClawPluginApi) {
  const config = resolveConfig(api);
  const headerStore = new AsyncLocalStorage<Record<string, string>>();

  const originalFetch = globalThis.fetch;

  globalThis.fetch = function rlPatchedFetch(
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> {
    const scopedHeaders = headerStore.getStore();
    if (scopedHeaders && init?.method?.toUpperCase() === "POST") {
      const merged = new Headers(init.headers);
      for (const [k, v] of Object.entries(scopedHeaders)) {
        // Plugin headers go first; per-request headers can still override.
        if (!merged.has(k)) {
          merged.set(k, v);
        }
      }
      return originalFetch.call(globalThis, input, { ...init, headers: merged });
    }
    return originalFetch.call(globalThis, input, init);
  } as typeof globalThis.fetch;

  api.on("before_prompt_build", (_event, ctx) => {
    const sessionId = ctx.sessionId ?? "";
    const turnType = SIDE_TRIGGERS.has(ctx.trigger ?? "") ? "side" : "main";
    headerStore.enterWith({
      [config.sessionIdHeader]: sessionId,
      [config.turnTypeHeader]: turnType,
    });
    return {};
  });

  api.logger.info("rl-training-headers: activated (fetch patched)");
}'''

if old_block not in text:
    raise SystemExit(
        "patch failed: expected index.ts content not found "
        "(official plugin may have changed upstream -- update this patch)"
    )

new_js = '''import { AsyncLocalStorage } from "node:async_hooks";
import { getGlobalDispatcher, setGlobalDispatcher } from "undici";

function resolveConfig(api) {
  const cfg = api.pluginConfig ?? {};
  return {
    sessionIdHeader: cfg.sessionIdHeader ?? "X-Session-Id",
    turnTypeHeader: cfg.turnTypeHeader ?? "X-Turn-Type",
  };
}

// Triggers classified as "side" (non-user-facing housekeeping runs).
// Identical to the official plugin -- only the injection mechanism below
// (undici global dispatcher instead of globalThis.fetch) changed, because
// globalThis.fetch patching is confirmed dead on this OpenClaw build: the
// openai SDK's default fetch resolution never re-reads it (see
// docs/issues_log.md 2026-07-09).
const SIDE_TRIGGERS = new Set(["heartbeat", "memory", "cron"]);

const RL_DISPATCHER_SYMBOL = Symbol.for("undici.globalDispatcher.2");
const RL_LEGACY_DISPATCHER_SYMBOL = Symbol.for("undici.globalDispatcher.1");
let rlRegisterCallCount = 0;

export default function register(api) {
  rlRegisterCallCount += 1;
  const registerInstanceId = rlRegisterCallCount;
  const config = resolveConfig(api);
  const headerStore = new AsyncLocalStorage();

  const rlHeaderInterceptor = (dispatch) => {
    return function InterceptedDispatch(opts, handler) {
      const scopedHeaders = headerStore.getStore();
      api.logger.info(
        `[RL-HEADERS-DEBUG#${registerInstanceId}] interceptor invoked: method=${opts.method} ` +
        `hasScopedHeaders=${!!scopedHeaders} scopedHeaders=${JSON.stringify(scopedHeaders)} ` +
        `origin=${opts.origin} path=${opts.path}`,
      );
      if (scopedHeaders && opts.method?.toUpperCase() === "POST") {
        opts.headers = { ...(opts.headers || {}), ...scopedHeaders };
        api.logger.info(`[RL-HEADERS-DEBUG#${registerInstanceId}] headers merged into opts: ${JSON.stringify(opts.headers)}`);
      }
      return dispatch(opts, handler);
    };
  };

  setGlobalDispatcher(getGlobalDispatcher().compose(rlHeaderInterceptor));
  const dispatcherAfterSet = globalThis[RL_DISPATCHER_SYMBOL];
  const legacyAfterSet = globalThis[RL_LEGACY_DISPATCHER_SYMBOL];
  api.logger.info(
    `[RL-HEADERS-DEBUG#${registerInstanceId}] global dispatcher composed. ` +
    `dispatcher.constructor=${dispatcherAfterSet?.constructor?.name} ` +
    `legacy.constructor=${legacyAfterSet?.constructor?.name}`,
  );

  // Re-check a few seconds later: did something else in OpenClaw's own
  // startup sequence call setGlobalDispatcher again after us and silently
  // replace our composed dispatcher?
  setTimeout(() => {
    const dispatcherNow = globalThis[RL_DISPATCHER_SYMBOL];
    const stillOurs = dispatcherNow === dispatcherAfterSet;
    api.logger.info(
      `[RL-HEADERS-DEBUG#${registerInstanceId}] delayed recheck (3s later): ` +
      `stillSameObject=${stillOurs} dispatcher.constructor=${dispatcherNow?.constructor?.name}`,
    );
  }, 3000);

  api.on("before_prompt_build", (_event, ctx) => {
    const sessionId = ctx.sessionId ?? "";
    const turnType = SIDE_TRIGGERS.has(ctx.trigger ?? "") ? "side" : "main";
    api.logger.info(
      `[RL-HEADERS-DEBUG#${registerInstanceId}] before_prompt_build fired: ` +
      `trigger=${ctx.trigger} sessionId=${sessionId} turnType=${turnType}`,
    );
    headerStore.enterWith({
      [config.sessionIdHeader]: sessionId,
      [config.turnTypeHeader]: turnType,
    });
    return {};
  });

  api.logger.info("rl-training-headers: activated (undici dispatcher patched)");
}
'''

with open(dest_path, "w", encoding="utf-8") as f:
    f.write(new_js)
print(f"patched -> {dest_path}")
PY

python3 - "${SRC_DIR}/openclaw.plugin.json" "${DEST_DIR}/openclaw.plugin.json" <<'PY'
import json
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]
manifest = json.loads(open(src_path, encoding="utf-8").read())

# 2026-07-07 finding: this OpenClaw build's plugin loader skips any plugin
# manifest missing these fields -- confirmed via A/B against the stock
# `clickclack` plugin, which has the exact same gap and also fails to load.
manifest["enabledByDefault"] = True
manifest["activation"] = {"onStartup": True}

with open(dest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"patched -> {dest_path}")
PY

cat > "${DEST_DIR}/package.json" <<'JSON'
{
  "name": "@openclaw/rl-training-headers",
  "version": "1.0.0-patched",
  "private": true,
  "description": "Injects X-Session-Id and X-Turn-Type HTTP headers into LLM API requests for RL training data classification (undici-dispatcher injection, patched for OpenClaw 2026.6.9 -- see docs/issues_log.md)",
  "type": "module",
  "openclaw": {
    "extensions": [
      "./index.js"
    ]
  }
}
JSON

echo "已生成 rl-training-headers 补丁插件目录: ${DEST_DIR}"
