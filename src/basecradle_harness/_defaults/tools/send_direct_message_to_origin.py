# Default tool plugin: send_direct_message_to_origin — a push notification to @origin's iPhone,
# off the platform entirely (issue #341). See memory.py for the full plugin contract.
#
# Powerful (it interrupts a human wherever he is) → opt_in everywhere (issue #168): off by default
# on every provider, activating only when this file is dropped into a persona's tools/ overlay
# (basecradle-harness-install --opt-in send_direct_message_to_origin). An interruption channel that
# arrives switched on for everyone is a spam channel.
#
# `requires=(EnvSet("NTFY_DM_TOKEN"),)` gates *availability* on the publish credential, so an agent
# provisioned without one never sees a tool that could only fail; it is provider-agnostic (the
# safety default is the capability, never the vendor). The tool re-checks the token at call time
# anyway, for a registry wired by hand through the library API.
#
# Delete this file to disable the tool.
from basecradle_harness import DirectMessageTool, EnvSet, ToolPlugin

PLUGIN = ToolPlugin(
    impl=DirectMessageTool,
    requires=(EnvSet("NTFY_DM_TOKEN"),),
    note=(
        "Rings @origin's phone — use it only when he asks you for a direct message. "
        "Plain text, 4,096 bytes max; nothing is truncated for you."
    ),
    opt_in=True,
)
