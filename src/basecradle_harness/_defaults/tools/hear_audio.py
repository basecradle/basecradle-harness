# Default tool plugin: listen (transcribe an audio asset). Delete to disable; see memory.py
# for the contract.
#
# Like generate_image, this calls an OpenAI API (Audio/transcription) with the agent's
# AI_PROVIDER_API_KEY under either provider, so it requires OpenAIKey() and self-excludes
# when no OpenAI key is configured.
from basecradle_harness import HearAudioTool, OpenAIKey, ToolPlugin

PLUGIN = ToolPlugin(impl=HearAudioTool, requires=(OpenAIKey(),))
