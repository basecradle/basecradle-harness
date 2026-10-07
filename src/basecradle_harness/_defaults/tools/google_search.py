# Default tool plugin: Google Search for a Gemini brain on Vertex AI. Delete to disable.
#
# Not a built-in, unlike every other provider's search: Vertex documents no support for a search
# tool beside function declarations in one request, and every harness turn carries function
# declarations. So this is a `web_search` function tool the harness runs: one grounded Gemini call
# of its own, `google_search` its only tool, on the brain's own Vertex account and model (issue
# #656, `_google_search.py`). The agent gets the grounded answer and its sources as the tool result.
#
# It shares the model-facing name `web_search` with the OpenAI, xAI and OpenRouter search plugins
# and carries a different requirement, so exactly one activates per config.
#
# Powerful (web search, and a per-query fee) → opt_in everywhere (issue #168): off by default on
# every provider, active only when this file is dropped into an agent's tools/ overlay. Its spend is
# logged: the grounded call's own llm line (purpose=helper kind=search.grounding) and a priced media
# line for the grounding fee.
from basecradle_harness import GoogleSearchTool, ToolPlugin, Vendor

PLUGIN = ToolPlugin(
    impl=GoogleSearchTool,
    requires=(Vendor("google"),),
    note=(
        "Google Search through Gemini's grounding — returns a grounded answer with its sources. "
        "Each search is a separate grounded call, with its own cost."
    ),
    opt_in=True,
)
