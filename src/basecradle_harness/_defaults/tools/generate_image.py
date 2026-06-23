# Default tool plugin: generate_image. Delete to disable; see memory.py for the contract.
#
# This tool calls OpenAI's Images API (through the openai SDK) with the agent's AI_API_KEY, so
# it declares OpenAIKey() as its activation requirement: under the openai provider with a key
# set. Under any other provider, or with no key, it self-excludes rather than registering a
# tool that can only error.
from basecradle_harness import GenerateImageTool, OpenAIKey, ToolPlugin

PLUGIN = ToolPlugin(impl=GenerateImageTool, requires=(OpenAIKey(),))
