// Fix for the "decision-paralysis loop" confirmed in docs/issues_log.md
// (2026-07-16/17 entries, Problem 36): OpenClaw's "## Execution Bias" system
// prompt section (added 2026-04-15~2026-04-30, i.e. after the paper's
// 2026-03-11 submission -- see issues_log.md version-archaeology entry) says
// "Non-final turn: use tools to advance, or ask for the one missing decision
// that blocks safe progress." This reads, to the model, as if EVERY non-final
// reply must be packaged as a tool call. Combined with the Qwen chat
// template's own "return a json object ... within tool_call XML tags"
// instruction for actual function calls, Qwen3-4B-Thinking was observed
// (full reasoning_text captured via prepare_patched_openclaw_opd.sh's
// TRUNCATED logging) spending its entire 8192-token budget re-deriving the
// same already-decided reply text, unable to resolve whether a plain-text
// reply needs <tool_call> wrapping -- never emitting output.
//
// There's no supported way to remove/replace an existing core system-prompt
// section without owning the "sglang" provider plugin registration (single
// owner per provider, see issues_log.md), so this appends a short,
// unambiguous disambiguation rule via before_prompt_build's
// appendSystemContext instead -- the same mechanism already proven to work
// for the rl-training-headers plugin on this OpenClaw build.
const DISAMBIGUATION_RULE =
  "\n\nThe Execution Bias guideline \"Non-final turn: use tools to advance, " +
  "or ask for the one missing decision that blocks safe progress\" is fully " +
  "satisfied by sending a plain-text reply -- it does NOT require wrapping " +
  "that reply in <tool_call> tags. Plain-text replies are not function " +
  "calls: only an actual tool invocation (name + arguments) uses " +
  "<tool_call> format. If you have already decided what to say and it does " +
  "not call a tool, output that text directly as your reply and stop -- do " +
  "not re-analyze whether it needs <tool_call> wrapping.";

export default function register(api) {
  api.on("before_prompt_build", () => {
    return {
      appendSystemContext: DISAMBIGUATION_RULE,
    };
  });

  api.logger.info("execution-bias-fix: activated (appendSystemContext disambiguation rule)");
}
