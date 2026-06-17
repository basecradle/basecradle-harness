# Default tool plugin: trust. Delete to disable; see memory.py for the contract. A platform
# tool with no activation requirements.
from basecradle_harness import ToolPlugin, TrustTool

PLUGIN = ToolPlugin(impl=TrustTool)
