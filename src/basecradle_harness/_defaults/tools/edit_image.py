# Default tool plugin: edit_image. Delete to disable; see memory.py for the contract.
#
# Like generate_image, this calls OpenAI's Images API (the /edits endpoint, through the openai
# SDK) with the agent's AI_API_KEY, so it requires OpenAIKey() and self-excludes under any
# other provider or when no key is configured.
# Powerful (media generation) → opt_in everywhere (issue #168): off by default, overlay opt-in
# only. `requires` gates availability (openai + key), never the safety default.
from basecradle_harness import EditImageTool, OpenAIKey, ToolPlugin

PLUGIN = ToolPlugin(impl=EditImageTool, requires=(OpenAIKey(),), opt_in=True)
