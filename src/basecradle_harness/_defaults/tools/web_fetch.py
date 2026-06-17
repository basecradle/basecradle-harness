# Default tool plugin: web_fetch. Delete this file to disable the tool; see memory.py for
# the full plugin contract. web_fetch is a plain tool with no activation requirements.
from basecradle_harness import ToolPlugin, WebFetchTool

PLUGIN = ToolPlugin(impl=WebFetchTool)
