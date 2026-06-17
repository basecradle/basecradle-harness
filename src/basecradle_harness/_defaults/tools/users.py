# Default tool plugin: users. Delete to disable; see memory.py for the contract. A platform
# read tool with no activation requirements (provider-agnostic).
from basecradle_harness import ToolPlugin, UsersTool

PLUGIN = ToolPlugin(impl=UsersTool)
