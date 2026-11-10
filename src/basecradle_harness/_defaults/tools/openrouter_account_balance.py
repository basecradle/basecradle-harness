# Default tool plugin: openrouter_account_balance — read the credit remaining on the agent's own
# OpenRouter account, and what the key it reads with has spent (issues #425, #853).
#
# A plain read-only function Tool (not a server-side built-in): it GETs openrouter.ai/api/v1/credits
# (total_credits purchased to date, less total_usage used to date) and /key (the calling key's own
# usage and, when it has one, its spending limit) with an OpenRouter key the agent already holds:
# OPENROUTER_MANAGEMENT_KEY when set (an optional override, never a requirement), else the brain's
# AI_API_KEY when AI_PROVIDER is openrouter, else the reranker's HARNESS_MEMPALACE_RERANK_API_KEY,
# else the describer's HARNESS_DESCRIBER_API_KEY when HARNESS_DESCRIBER_PROVIDER is openrouter.
# An ordinary inference key reads both endpoints (measured live, 2026-11-10).
#
# Powerful (it reaches an account/billing surface) → opt_in everywhere (issue #168): off by
# default on every provider, activates only when this file is dropped into an agent's tools/
# overlay (basecradle-harness-install --opt-in openrouter_account_balance).
#
# `requires=()` — NO Vendor gate, and that is deliberate (issue #425). Its xAI sibling gates on
# Vendor("xai"), but the case here includes an agent brained by *another* provider whose
# OpenRouter key serves a helper (the reranker, the describer): a Vendor("openrouter") gate would
# self-exclude exactly that agent. The tool checks AI_PROVIDER itself before it reads AI_API_KEY,
# so another vendor's key is never sent to OpenRouter. It is not gated on a credential either: a
# missing key must reach the model as a soft, readable "not configured" reason it can act on, not
# as a capability that silently is not there (issue #374).
#
# `needs_env=()` — deliberately empty since issue #853. The field declares a variable the tool
# cannot work without (issue #427), and no single variable is that any more: any one of the four
# above serves. Declaring OPENROUTER_MANAGEMENT_KEY, as it did before, would report a gap on every
# healthy agent reading with its own key; declaring all four would report three gaps on each. The
# cost, stated rather than left to be found: an agent with this tool opted in and no OpenRouter key
# at all reports nothing missing in `--resolved-config`'s `tool_env`, and learns it only from the
# tool's own "no OpenRouter key is configured" answer, which names every variable it reads.
#
# Delete this file to disable the tool.
from basecradle_harness import OpenRouterAccountBalanceTool, ToolPlugin

PLUGIN = ToolPlugin(
    impl=OpenRouterAccountBalanceTool,
    note=(
        "Reads the credit remaining on your own OpenRouter account and what your key has spent "
        "against its own limit (read-only; uses an OpenRouter key you already hold)."
    ),
    opt_in=True,
)
