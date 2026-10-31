# Default tool plugin: Anthropic web search (server tool). Delete to disable.
#
# A *built-in*, not a Tool class: Anthropic runs the search server-side (`web_search_20260318`) and
# the harness never executes it — the same safe-by-construction shape as OpenAI's `web_search`, xAI's
# Live Search and OpenRouter's server tool. The model searches, reads the results and cites them in
# the same turn; the citations come back as a `Sources:` footer on its reply.
#
# It shares the model-facing name `web_search` with the OpenAI, xAI, OpenRouter and Google search
# plugins and carries a different requirement, so exactly one activates per config.
#
# Powerful (web search, and a per-search fee) → opt_in everywhere (issue #168): off by default on
# every provider, active only when this file is dropped into an agent's tools/ overlay. Its spend is
# logged: $10 per 1,000 searches as a priced media line (kind=search.web), with the call's own search
# count on its llm line (web_search_requests=).
from basecradle_harness import ToolPlugin, Vendor

PLUGIN = ToolPlugin(
    builtin="web_search",
    requires=(Vendor("anthropic"),),
    note=(
        "Anthropic server-side web search — you search, read and cite live sources yourself. Each "
        "search has its own small fee."
    ),
    opt_in=True,
)
