# Default tool plugin: timelines. Delete to disable; see memory.py for the contract. A
# platform tool with no activation requirements.
from basecradle_harness import TimelinesTool, ToolPlugin

PLUGIN = ToolPlugin(impl=TimelinesTool)
