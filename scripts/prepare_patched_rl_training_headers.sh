#!/bin/bash
# Patch the official rl-training-headers OpenClaw plugin's injection mechanism.
#
# Why: exhaustively investigated (docs/issues_log.md, 2026-07-09) two
# transport-layer injection techniques -- the official plugin's
# globalThis.fetch monkey-patch, and an undici setGlobalDispatcher variant --
# and confirmed BOTH are structurally impossible on this OpenClaw build
# (2026.6.9): OpenClaw added a deliberate SSRF-safety layer
# (fetchWithSsrFGuardInternal / fetchWithRuntimeDispatcher) that bypasses any
# externally-patched fetch/dispatcher for every real (non-Vitest-mock)
# outbound request, with no config escape hatch. Confirmed via reading
# OpenClaw's own source from ~2026-03/04 (when the paper's plugin was
# authored) that this bypass layer did not exist then -- it's a security
# hardening added between April and June 2026, not a config gap or bad
# install.
#
# Fix: inject via request CONTENT instead of transport. before_prompt_build's
# `appendSystemContext` return field writes into the system prompt that
# actually gets sent -- this is unaffected by the SSRF-safety bypass since it
# never touches fetch/dispatcher machinery at all. The classification logic
# (ctx.trigger/ctx.sessionId, SIDE_TRIGGERS) is unchanged from the official
# plugin. The counterpart patch (prepare_patched_openclaw_opd.sh) parses this
# marker out of `messages` server-side AND strips it before forwarding to
# sglang / before computing training prompt_ids, so neither the policy model
# nor the training data ever sees it -- verified this keeps model-facing and
# training-facing content identical to what the paper's (never-working-here)
# header mechanism would have produced.
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

new_js = '''// Triggers classified as "side" (non-user-facing housekeeping runs).
// Identical to the official plugin's classification logic. Only the
// injection mechanism changed: the official plugin patches globalThis.fetch
// to add HTTP headers, confirmed structurally impossible on this OpenClaw
// build (a deliberate SSRF-safety layer bypasses any externally-patched
// fetch/dispatcher for real requests -- see docs/issues_log.md 2026-07-09).
// This version instead appends a machine-parseable marker to the system
// prompt via before_prompt_build's appendSystemContext field, which is
// unaffected by that bypass since it modifies request content, not
// transport. The server-side counterpart (openclaw_opd_api_server.py patch)
// parses this marker out AND strips it before forwarding to the policy
// model / before computing training data, so this marker never reaches the
// model or the training set.
const SIDE_TRIGGERS = new Set(["heartbeat", "memory", "cron"]);
const RL_META_PREFIX = "[RL-TRAINING-META]";

export default function register(api) {
  api.on("before_prompt_build", (_event, ctx) => {
    const sessionId = ctx.sessionId ?? "";
    const turnType = SIDE_TRIGGERS.has(ctx.trigger ?? "") ? "side" : "main";
    return {
      appendSystemContext: `\\n\\n${RL_META_PREFIX} session_id=${sessionId} turn_type=${turnType}`,
    };
  });

  api.logger.info("rl-training-headers: activated (appendSystemContext marker)");
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
  "description": "Marks X-Session-Id/X-Turn-Type equivalent metadata via an appendSystemContext marker for RL training data classification (header/dispatcher injection confirmed structurally blocked on OpenClaw 2026.6.9 -- see docs/issues_log.md)",
  "type": "module",
  "openclaw": {
    "extensions": [
      "./index.js"
    ]
  }
}
JSON

echo "已生成 rl-training-headers 补丁插件目录: ${DEST_DIR}"
