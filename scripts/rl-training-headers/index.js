import { AsyncLocalStorage } from "node:async_hooks";

const SIDE_TRIGGERS = new Set(["heartbeat", "memory", "cron"]);

export default function register(api) {
  const cfg = api.pluginConfig ?? {};
  const config = {
    sessionIdHeader: cfg.sessionIdHeader ?? "X-Session-Id",
    turnTypeHeader:  cfg.turnTypeHeader  ?? "X-Turn-Type",
  };

  const headerStore = new AsyncLocalStorage();
  const originalFetch = globalThis.fetch;

  globalThis.fetch = function rlPatchedFetch(input, init) {
    const scopedHeaders = headerStore.getStore();
    if (scopedHeaders && init?.method?.toUpperCase() === "POST") {
      const merged = new Headers(init.headers);
      for (const [k, v] of Object.entries(scopedHeaders)) {
        if (!merged.has(k)) merged.set(k, v);
      }
      return originalFetch.call(globalThis, input, { ...init, headers: merged });
    }
    return originalFetch.call(globalThis, input, init);
  };

  api.on("before_prompt_build", (_event, ctx) => {
    const sessionId = ctx.sessionId ?? "";
    const turnType = SIDE_TRIGGERS.has(ctx.trigger ?? "") ? "side" : "main";
    headerStore.enterWith({
      [config.sessionIdHeader]: sessionId,
      [config.turnTypeHeader]:  turnType,
    });
    return {};
  });

  api.logger.info("rl-training-headers: activated (fetch patched)");
}
