# Default tool plugin: edit_image. Delete to disable; see memory.py for the contract.
#
# Like generate_image, this calls OpenAI's Images API (the /edits endpoint) with the agent's
# AI_PROVIDER_API_KEY under either provider, so it requires OpenAIKey() and self-excludes when
# no OpenAI key is configured.
from basecradle_harness import EditImageTool, OpenAIKey, ToolPlugin

PLUGIN = ToolPlugin(impl=EditImageTool, requires=(OpenAIKey(),))
