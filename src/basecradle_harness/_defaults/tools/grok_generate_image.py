# Default tool plugin: grok_generate_image (xAI). Delete to disable; see memory.py for the contract.
#
# xAI-native image generation. Requires the xAI provider (AI_PROVIDER=xai) — under any other
# provider it self-excludes (the OpenAI generate_image tool covers that case instead), so an
# xAI agent's media stack touches no OpenAI surface.
from basecradle_harness import GrokGenerateImageTool, ToolPlugin, Vendor

PLUGIN = ToolPlugin(impl=GrokGenerateImageTool, requires=(Vendor("xai"),))
