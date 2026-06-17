# Default tool plugin: webhook_endpoints. Delete to disable; see memory.py for the contract.
# A platform tool with no activation requirements.
from basecradle_harness import ToolPlugin, WebhookEndpointsTool

PLUGIN = ToolPlugin(impl=WebhookEndpointsTool)
