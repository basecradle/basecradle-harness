# Default tool plugin: grok_generate_video (xAI). Delete to disable; see memory.py for the contract.
#
# xAI-native video generation (text-to-video and image-to-video) — the harness's first video
# capability. Requires the xAI profile (AI_PROVIDER_API=xai); self-excludes everywhere else.
from basecradle_harness import GrokGenerateVideoTool, ProviderAPI, ToolPlugin

PLUGIN = ToolPlugin(impl=GrokGenerateVideoTool, requires=(ProviderAPI("xai"),))
