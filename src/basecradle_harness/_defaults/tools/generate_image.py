# Default tool plugin: generate_image. Delete to disable; see memory.py for the contract.
#
# This tool calls OpenAI's Images API with the agent's AI_PROVIDER_API_KEY (under either
# provider), so it declares OpenAIKey() as its activation requirement: with no OpenAI key it
# self-excludes rather than registering a tool that can only error.
from basecradle_harness import GenerateImageTool, OpenAIKey, ToolPlugin

PLUGIN = ToolPlugin(impl=GenerateImageTool, requires=(OpenAIKey(),))
