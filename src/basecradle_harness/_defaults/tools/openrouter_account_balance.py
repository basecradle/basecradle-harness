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
# On by default (issue #862): @origin's rule of 2026-11-10 is that every agent can read the
# available balance at its main LLM vendor, as long as that read exposes no other sensitive account
# data, and that an agent whose reranker is on OpenRouter reads the OpenRouter balance too. This read
# exposes the account's credit totals and the key's own spend and limit, nothing else, so it is a
# benign default, laid down on every agent like the platform tools. Before #862 it was opt_in
# (issue #168's account/billing class); xai_account_balance stays opt_in, because it needs a
# dedicated Management key, which makes it a case-by-case grant.
#
# `requires=(OpenRouterKeyHeld(),)` — present exactly where the agent already holds an OpenRouter key
# (any one of the four above, in that order), absent where it holds none: an agent with no OpenRouter
# key gets no tool, never a tool that answers "not configured" on every wake. The gate asks the same
# function the tool reads its key through (`_openrouter_account.held_key`), so the two cannot
# disagree. NOT a Vendor gate, deliberately (issue #425): an agent brained by *another* provider whose
# OpenRouter key serves a helper (the reranker, the describer) is the case it exists for, and a
# Vendor("openrouter") gate would self-exclude exactly that agent. The tool checks AI_PROVIDER itself
# before it reads AI_API_KEY, so another vendor's key is never sent to OpenRouter.
#
# `needs_env=()` — deliberately empty since issue #853. The field declares a variable the tool
# cannot work without (issue #427), and no single variable is that any more: any one of the four
# above serves. Declaring OPENROUTER_MANAGEMENT_KEY, as it did before, would report a gap on every
# healthy agent reading with its own key; declaring all four would report three gaps on each. The
# gate names all four as the credentials `basecradle-harness-resolve` assumes, which is a different
# question (what the gate tests), so the resolver still states the assumption it made.
#
# Delete this file to disable the tool.
from basecradle_harness import OpenRouterAccountBalanceTool, OpenRouterKeyHeld, ToolPlugin

PLUGIN = ToolPlugin(
    impl=OpenRouterAccountBalanceTool,
    requires=(OpenRouterKeyHeld(),),
    note=(
        "Reads the credit remaining on your own OpenRouter account and what your key has spent "
        "against its own limit (read-only; uses an OpenRouter key you already hold)."
    ),
)
