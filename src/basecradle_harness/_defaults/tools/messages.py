# Default tool plugin: messages. Delete to disable; see memory.py for the contract. A platform
# tool (read the backlog + post messages, incl. cross-timeline) with no activation requirements
# (provider-agnostic). Default-on, not opt-in: posting carries no new safety surface — the platform
# authorizes every post server-side (viewer-only, locked timelines reject, mutual trust gates).
from basecradle_harness import MessagesTool, ToolPlugin

PLUGIN = ToolPlugin(impl=MessagesTool)
