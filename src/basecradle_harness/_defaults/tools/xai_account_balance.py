# Default tool plugin: xai_account_balance — read the credit remaining on the agent's own xAI
# account.
#
# A plain read-only function Tool (not a server-side built-in): it calls the xAI *Management API*
# (management-api.x.ai) with a dedicated read-only Management Key (XAI_MANAGEMENT_KEY), a
# billing/account surface distinct from the inference endpoint and its AI_API_KEY. So an xAI
# persona whose charter treats capital as first-class can see its own remaining runway.
#
# The figure is the *live* one — the invoice preview's prepaid credit less what this billing cycle
# has already drawn from it (issue #388). It is neither the posted prepaid ledger, which settles at
# cycle close and overstates runway mid-cycle (issue #384), nor that undrawn prepaid figure alone.
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
    note="Reads the live credit remaining on your own xAI account (read-only billing; xAI only).",
    opt_in=True,
)
