# Default tool plugin: assets — the timeline's files, and the senses that open them (list / read /
# view / watch / listen / create / post_image). Delete to disable; see memory.py for the contract.
# A platform tool — bound to the agent's client + timeline by the hosting agent at startup — with
# no activation requirements: every agent gets files, and every agent gets eyes.
#
# `configure=assets_options` is what builds the action set for *this* agent (issue #484): `listen`
# needs a transcription provider, so it is present only where one is configured, and absent from
# the schema and the description everywhere else — never a door that does not open. The gate lives
# in `_assets.assets_options`, not here, because the installer reads a plugin file's provider
# affinity from this file's *source*: an activation-requirement call naming a vendor or its key,
# written here, would mark the universal assets plugin single-provider and stop it loading on every
# other agent in the fleet. Keep this file free of them — `test_plugins.py` pins that.
from basecradle_harness import AssetsTool, ToolPlugin, assets_options

PLUGIN = ToolPlugin(impl=AssetsTool, configure=assets_options)
