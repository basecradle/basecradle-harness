# Default tool plugin: grok_generate_video (xAI). Delete to disable; see memory.py for the contract.
#
# xAI-native video generation (text-to-video and image-to-video) — the harness's first video
# capability. Requires the xAI provider (AI_PROVIDER=xai); self-excludes everywhere else.
# Powerful (video generation — off by default on EVERY provider, full stop) → opt_in everywhere
# (issue #168): overlay opt-in only. `requires` gates availability (the xai provider), not the default.
#
# `needs_env=("AI_API_KEY",)` reports the ungated call-time read (issue #427). The tool gates on the
# *provider*, not the key, and soft-fails with a readable "no API key" string when it is unset — so
# without this declaration an xAI agent's media stack named no credential at all, while its OpenAI
# counterpart (gating on OpenAIKey, an EnvSet) named AI_API_KEY. Same dependency, opposite
# visibility, decided by nothing.
from basecradle_harness import GrokGenerateVideoTool, ToolPlugin, Vendor

PLUGIN = ToolPlugin(
    impl=GrokGenerateVideoTool, requires=(Vendor("xai"),), needs_env=("AI_API_KEY",), opt_in=True
)
