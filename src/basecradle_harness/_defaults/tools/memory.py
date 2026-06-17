# Default tool plugin: memory.
#
# This is a real, editable file in your config home's tools/ dir. To DISABLE this tool,
# delete this file (the upgrader will not resurrect it). To OVERRIDE it, set the same name
# on a plugin in another file. To ADD a tool, drop a new *.py file here exposing a PLUGIN.
#
# A plugin is (name + requires + impl): `impl` is the Tool class, `requires` is what the
# active config must provide for the tool to load. Memory needs nothing — it is always on.
from basecradle_harness import MemoryTool, ToolPlugin

PLUGIN = ToolPlugin(impl=MemoryTool)
