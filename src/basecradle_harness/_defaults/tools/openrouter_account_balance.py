# Default tool plugin: openrouter_account_balance — read the credit remaining on the agent's own
# OpenRouter account (issue #425).
#
# A plain read-only function Tool (not a server-side built-in): it GETs openrouter.ai/api/v1/credits
# with a dedicated OpenRouter *Management key* (OPENROUTER_MANAGEMENT_KEY), an account-administration
# credential distinct from the inference AI_API_KEY. The figure is total_credits (purchased to date)
# less total_usage (used to date) — one endpoint, one subtraction, no ledger/preview dichotomy.
#
# Powerful (it reaches an account/billing surface with a dedicated credential) → opt_in
# everywhere (issue #168): off by default on every provider, activates only when this file is
# dropped into a persona's tools/ overlay
# (basecradle-harness-install --opt-in openrouter_account_balance).
#
# `requires=()` — NO Vendor gate, and that is deliberate (issue #425). Its xAI sibling gates on
# Vendor("xai"), but the ordering case here is an agent brained by *another* provider that holds a
# separate OpenRouter account: a Vendor("openrouter") gate would self-exclude exactly that agent.
# The credential is dedicated and provider-independent, so provider affinity says nothing about
# whether this tool can work. It is not gated on the credential either (the way the DM tool gates
# on NTFY_DM_TOKEN): a missing key must reach the model as a soft, readable "not configured"
# reason it can act on, not as a capability that silently is not there (issue #374).
#
# `needs_env` is how that ungated dependency stays *visible* anyway (issue #427): it reports —
# through `basecradle-harness-resolve`'s `credentials.wanted` and `--resolved-config`'s `tool_env`
# — without gating, so an operator provisioning this agent can ask the box which keys its
# configuration wants instead of reading the README and knowing to. Before it existed this key was
# named in exactly one place in the repo: a table in README.md.
#
# Delete this file to disable the tool.
from basecradle_harness import OpenRouterAccountBalanceTool, ToolPlugin

PLUGIN = ToolPlugin(
    impl=OpenRouterAccountBalanceTool,
    needs_env=("OPENROUTER_MANAGEMENT_KEY",),
    note=(
        "Reads the credit remaining on your own OpenRouter account (read-only; needs a "
        "Management key, not the inference key)."
    ),
    opt_in=True,
)
