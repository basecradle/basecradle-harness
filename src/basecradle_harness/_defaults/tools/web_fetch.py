# Default tool plugin: web_fetch. Delete this file to disable the tool; see memory.py for
# the full plugin contract. web_fetch is a plain tool with no activation requirements on every
# provider but one: under AI_PROVIDER=anthropic, Claude's own server-side fetch claims the name
# (anthropic_web_fetch.py), and this tool steps aside for it.
from basecradle_harness import NotVendor, ToolPlugin, WebFetchTool

PLUGIN = ToolPlugin(impl=WebFetchTool, requires=(NotVendor("anthropic"),))
