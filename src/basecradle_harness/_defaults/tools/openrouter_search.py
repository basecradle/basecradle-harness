# Default tool plugin: OpenRouter web search (server tool). Delete to disable.
#
# This is a *built-in*, not a Tool class: OpenRouter runs the search entirely server-side (its
# `openrouter:web_search` server tool) and the harness never executes it — the same
# safe-by-construction shape as OpenAI's `web_search` and xAI's Live Search. When the model
# decides it needs current information it calls the tool; OpenRouter searches and feeds the
# results (with `url_citation` annotations) back into the same turn.
#
# It shares the model-facing name `web_search` with the OpenAI and xAI built-ins, but carries a
# different requirement, so exactly one search built-in activates per config — the resolver
# settles it. The wire type it maps to (`openrouter:web_search`) and its optional `parameters`
# (from search_params.json) are the native `OpenRouterProvider`'s to emit.
#
# `Sdk("openrouter")` on top of `Vendor("openrouter")` scopes it to the **native** OpenRouter SDK
# (AI_SDK=openrouter — @glm-5.2's brain), the path whose chat request actually wires the server
# tool. The openai-SDK-at-OpenRouter cell is chat-only and ships no server-side built-ins (a
# documented limit), so there the plugin self-excludes rather than activating inert.
#
# Powerful (web search) → opt_in everywhere (issue #168): off by default on every provider,
# activates only when this file is dropped into a persona's tools/ overlay. `requires` gates
# *availability* (provider + SDK), never the safety default.
from basecradle_harness import Sdk, ToolPlugin, Vendor

PLUGIN = ToolPlugin(
    builtin="web_search",
    requires=(Vendor("openrouter"), Sdk("openrouter")),
    note=(
        "OpenRouter server-side web search — the model searches and cites live sources itself. "
        "Tune it (engine, max_results, domains, …) in search_params.json."
    ),
    opt_in=True,
)
