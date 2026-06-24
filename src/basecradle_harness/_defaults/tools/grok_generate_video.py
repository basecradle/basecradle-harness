# Default tool plugin: grok_generate_video (xAI). Delete to disable; see memory.py for the contract.
#
# xAI-native video generation (text-to-video and image-to-video) — the harness's first video
# capability. Requires the xAI provider (AI_PROVIDER=xai); self-excludes everywhere else.
# Powerful (video generation — off by default on EVERY provider, full stop) → opt_in everywhere
# (issue #168): overlay opt-in only. `requires` gates availability (the xai provider), not the default.
from basecradle_harness import GrokGenerateVideoTool, ToolPlugin, Vendor

PLUGIN = ToolPlugin(impl=GrokGenerateVideoTool, requires=(Vendor("xai"),), opt_in=True)
