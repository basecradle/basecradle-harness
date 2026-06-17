# Default tool plugin: assets (view/list/upload timeline assets). Delete to disable; see
# memory.py for the contract. A platform tool — bound to the agent's client + timeline by
# the hosting agent at startup — with no activation requirements.
from basecradle_harness import AssetsTool, ToolPlugin

PLUGIN = ToolPlugin(impl=AssetsTool)
