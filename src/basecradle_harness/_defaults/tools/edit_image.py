# Default tool plugin: edit_image. Delete to disable; see memory.py for the contract.
#
# Like generate_image, this calls OpenAI's Images API (the /edits endpoint, through the openai
# SDK) with the agent's AI_API_KEY, so it requires OpenAIKey() and self-excludes under any
# other provider or when no key is configured.
from basecradle_harness import EditImageTool, OpenAIKey, ToolPlugin

PLUGIN = ToolPlugin(impl=EditImageTool, requires=(OpenAIKey(),))
