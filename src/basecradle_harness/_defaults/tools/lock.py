# Default tool plugin: lock. Delete to disable; see memory.py for the contract. A platform
# tool with no activation requirements (provider-agnostic). The irreversible emergency stop,
# guarded by an explicit confirm=true.
from basecradle_harness import LockTool, ToolPlugin

PLUGIN = ToolPlugin(impl=LockTool)
