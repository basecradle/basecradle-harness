# Default tool plugin: web_search. Delete to disable; see memory.py for the contract.
#
# Unlike the others this is a *built-in*, not a Tool class: OpenAI's Responses API runs the
# search server-side and the harness never executes it. So the plugin sets `builtin` (the
# wire name) instead of `impl`, and requires the Responses provider API — under Chat
# Completions, which has no such built-in, it self-excludes and the active set drops it.
from basecradle_harness import ProviderAPI, ToolPlugin

PLUGIN = ToolPlugin(builtin="web_search", requires=(ProviderAPI("responses"),))
