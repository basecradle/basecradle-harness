# Default tool plugin: grok_generate_image (xAI). Delete to disable; see memory.py for the contract.
#
# xAI-native image generation. Requires the xAI provider (AI_PROVIDER=xai) — under any other
# provider it self-excludes (the OpenAI generate_image tool covers that case instead), so an
# xAI agent's media stack touches no OpenAI surface.
# Powerful (media generation) → opt_in everywhere (issue #168): off by default, overlay opt-in
# only. `requires` gates availability (the xai provider), never the safety default.
#
# `needs_env=("AI_API_KEY",)` reports the ungated call-time read (issue #427). The tool gates on the
# *provider*, not the key, and soft-fails with a readable "no API key" string when it is unset — so
# without this declaration an xAI agent's media stack named no credential at all, while its OpenAI
# counterpart (gating on OpenAIKey, an EnvSet) named AI_API_KEY. Same dependency, opposite
# visibility, decided by nothing.
from basecradle_harness import GrokGenerateImageTool, ToolPlugin, Vendor

PLUGIN = ToolPlugin(
    impl=GrokGenerateImageTool, requires=(Vendor("xai"),), needs_env=("AI_API_KEY",), opt_in=True
)
