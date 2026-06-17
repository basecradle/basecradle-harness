# Default tool plugin: messages. Delete to disable; see memory.py for the contract. A platform
# read tool with no activation requirements (provider-agnostic).
from basecradle_harness import MessagesTool, ToolPlugin

PLUGIN = ToolPlugin(impl=MessagesTool)
