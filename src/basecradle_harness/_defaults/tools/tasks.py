# Default tool plugin: tasks. Delete to disable; see memory.py for the contract. A platform
# tool with no activation requirements.
from basecradle_harness import TasksTool, ToolPlugin

PLUGIN = ToolPlugin(impl=TasksTool)
