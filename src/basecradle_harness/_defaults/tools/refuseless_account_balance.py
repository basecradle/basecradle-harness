# Default tool plugin: refuseless_account_balance — read the credit on the agent's own refuseless
# account (issue #848).
#
# A plain read-only function Tool (not a server-side built-in): it GETs <base_url>/credits — the
# host the brain's own calls go to, AI_BASE_URL or api.refuseless.com/v1 — with the agent's own
# inference key, AI_API_KEY. The call is free. It reports what a new request can spend
# (`available`), the balance, the part held by requests in flight (`reserved`), and the key's own
# spending limit (`key_limit`), each as the vendor states it.
#
# Powerful (it reaches an account/billing surface) → opt_in everywhere (issue #168): off by default
# on every provider, activates only when this file is dropped into an agent's tools/ overlay
# (basecradle-harness-install --opt-in refuseless_account_balance).
#
# `requires=(Vendor("refuseless"),)` — the xAI sibling's gate, not the OpenRouter sibling's absence
# of one, and the reason is the credential. The OpenRouter tool can be offered beside any brain
# because it also reads OpenRouter keys that serve helpers (the reranker, the describer). This one
# reads only AI_API_KEY, which is a refuseless key only when the brain is refuseless: beside any
# other brain it would send that vendor's key to refuseless's host. So it self-excludes everywhere
# else.
#
# `needs_env` names AI_API_KEY because the tool cannot work without it, so it rides
# `credentials.wanted` and `tool_env` like the xAI sibling's Management key (issue #427). It is the
# brain's own key, so on a healthy refuseless box it is always present and never reads as a gap.
#
# Delete this file to disable the tool.
from basecradle_harness import RefuselessAccountBalanceTool, ToolPlugin, Vendor

PLUGIN = ToolPlugin(
    impl=RefuselessAccountBalanceTool,
    requires=(Vendor("refuseless"),),
    needs_env=("AI_API_KEY",),
    note="Reads the live credit on your own refuseless account (read-only, free; refuseless only).",
    opt_in=True,
)
