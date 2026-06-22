# Default tool plugin: delete. Delete this file to disable; see memory.py for the contract. A
# platform tool with no activation requirements (provider-agnostic). The destructive owner
# power — permanently delete a timeline and ALL its content — guarded by uuid-confirm, the same
# gate as lock (see _confirmed.py).
#
# `note` is the one-line gotcha the function schema can't convey — rendered into the Turn-0
# brief's tool manifest so the model is told, every wake, that this destroys everything and
# cannot be undone.
from basecradle_harness import DeleteTool, ToolPlugin

PLUGIN = ToolPlugin(
    impl=DeleteTool,
    note="permanently destroys a timeline and ALL its content — irreversible; confirm must equal the uuid.",
)
