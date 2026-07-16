# Default tool plugin: xai_account_balance — read the agent's own xAI prepaid credit balance.
#
# A plain read-only function Tool (not a server-side built-in): it calls the xAI *Management API*
# (management-api.x.ai) with a dedicated read-only Management Key (XAI_MANAGEMENT_KEY), a
# billing/account surface distinct from the inference endpoint and its AI_API_KEY. So an xAI
# persona whose charter treats capital as first-class can see its own remaining runway.
#
# Powerful (it reaches an account/billing surface with a dedicated credential) → opt_in
# everywhere (issue #168): off by default on every provider, activates only when this file is
# dropped into a persona's tools/ overlay (basecradle-harness-install --opt-in xai_account_balance).
# `requires=(Vendor("xai"),)` gates *availability* to the xAI provider — it self-excludes on
# OpenAI/OpenRouter, which expose no equivalent balance surface — never the safety default.
#
# Delete this file to disable the tool.
from basecradle_harness import ToolPlugin, Vendor, XaiAccountBalanceTool

PLUGIN = ToolPlugin(
    impl=XaiAccountBalanceTool,
    requires=(Vendor("xai"),),
    note="Reads your own xAI prepaid credit balance (read-only billing; xAI provider only).",
    opt_in=True,
)
