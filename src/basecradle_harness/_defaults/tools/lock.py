# Default tool plugin: lock. Delete to disable; see memory.py for the contract. A platform
# tool with no activation requirements (provider-agnostic). The irreversible emergency stop,
# guarded by an explicit confirm=true.
#
# `note` is the one-line gotcha the function schema can't convey — rendered into the Turn-0
# brief's tool manifest so the model is told, every wake, that this action is irreversible.
from basecradle_harness import LockTool, ToolPlugin

PLUGIN = ToolPlugin(
    impl=LockTool,
    note="one-way and irreversible — an emergency stop; needs confirm=true.",
)
