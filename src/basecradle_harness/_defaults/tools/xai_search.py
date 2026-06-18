# Default tool plugins: xAI Live Search (web_search + x_search). Delete to disable.
#
# These are *built-ins*, not Tool classes: xAI's Responses API runs the search server-side and
# the harness never executes it (the same shape as OpenAI's web_search). They require the xAI
# profile (AI_PROVIDER_API=xai); under it, grok answers are grounded in live web and X results
# with url_citation sources. `web_search` shares its name with OpenAI's Responses built-in but
# carries a different requirement, so exactly one activates per config — the resolver settles it.
from basecradle_harness import ProviderAPI, ToolPlugin

PLUGINS = [
    ToolPlugin(
        builtin="web_search",
        requires=(ProviderAPI("xai"),),
        note="xAI Live Search of the web — grok searches and cites live sources itself.",
    ),
    ToolPlugin(
        builtin="x_search",
        requires=(ProviderAPI("xai"),),
        note="xAI Live Search of X (Twitter) — grok searches posts and cites them itself.",
    ),
]
