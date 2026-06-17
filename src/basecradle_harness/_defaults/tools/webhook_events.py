# Default tool plugin: webhook_events. Delete to disable; see memory.py for the contract. A
# platform tool with no activation requirements.
from basecradle_harness import ToolPlugin, WebhookEventsTool

PLUGIN = ToolPlugin(impl=WebhookEventsTool)
