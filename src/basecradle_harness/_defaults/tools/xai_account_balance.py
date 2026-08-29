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
# `requires=(Vendor("xai"),)` gates *availability* to the xAI provider — it self-excludes
# elsewhere, because this endpoint reads an *xAI* account and nothing else — never the safety
# default. An agent that also holds an OpenRouter account reads that one with the separate,
# provider-agnostic openrouter_account_balance tool (issue #425).
#
# `needs_env` reports the Management Key without gating on it (issue #427), so the key is visible
# to `basecradle-harness-resolve` and `--resolved-config` rather than named only in the README.
# XAI_TEAM_ID is deliberately NOT declared: the team is discovered from the key when it is unset,
# so its absence costs one HTTP call and nothing else — declaring it would read as a provisioning
# gap on every correctly-configured box, and a report that reddens on a healthy agent is one
# nobody reads twice. `needs_env` means *cannot work without*, never *reads*.
#
# Delete this file to disable the tool.
from basecradle_harness import ToolPlugin, Vendor, XaiAccountBalanceTool

PLUGIN = ToolPlugin(
    impl=XaiAccountBalanceTool,
    requires=(Vendor("xai"),),
    needs_env=("XAI_MANAGEMENT_KEY",),
    note="Reads the live credit remaining on your own xAI account (read-only billing; xAI only).",
    opt_in=True,
)
