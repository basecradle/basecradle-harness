# Default tool plugin: listen (transcribe an audio asset). Delete to disable; see memory.py
# for the contract.
#
# Like generate_image, this calls an OpenAI API (Audio/transcription, through the openai SDK)
# with the agent's AI_API_KEY, so it requires OpenAIKey() and self-excludes under any other
# provider or when no key is configured.
# Powerful (named in the issue #168 opt-in set — a provider-call media tool) → opt_in
# everywhere: off by default, overlay opt-in only. `requires` gates availability, not the default.
from basecradle_harness import HearAudioTool, OpenAIKey, ToolPlugin

PLUGIN = ToolPlugin(impl=HearAudioTool, requires=(OpenAIKey(),), opt_in=True)
