# Default tool plugin: web_search (OpenAI). Delete to disable; see memory.py for the contract.
#
# Unlike the others this is a *built-in*, not a Tool class: OpenAI's Responses API runs the
# search server-side and the harness never executes it. So the plugin sets `builtin` (the wire
# name) instead of `impl`. It requires the OpenAI provider AND the Responses surface — under
# Chat Completions, which has no such built-in, it self-excludes; under xAI, the xai_search
# plugin claims the `web_search` name instead (exactly one activates per config).
from basecradle_harness import OpenAISurface, ToolPlugin, Vendor

# Powerful (web search) → opt_in everywhere (issue #168): off by default, overlay opt-in only.
# `requires` gates availability (openai + responses surface), never the safety default.
PLUGIN = ToolPlugin(
    builtin="web_search",
    requires=(Vendor("openai"), OpenAISurface("responses")),
    opt_in=True,
)
