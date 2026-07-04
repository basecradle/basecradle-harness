# Default tool plugin: system_prompt (read + edit). Delete to disable; see memory.py for the
# contract. Provider-agnostic (universal — no Vendor/OpenAIKey affinity): it touches only the
# agent's own config-home file, not any model endpoint.
#
# The MOST powerful tool in the kit — an agent that can rewrite its own personality charter
# (issue #241). So both tools are opt_in=True: off by default on EVERY provider, never
# auto-scaffolded and never loaded from the packaged defaults, active only when an operator
# deliberately drops this file into a persona's tools/ overlay. Enablement is a founder decision,
# per-agent; as of #241 no agent has it. `requires` is empty because availability is universal;
# per issue #168 the opt_in safety default is capability-based, never gated on availability.
#
# Structural safety (see _system_prompt.py): neither tool takes a path/agent argument, so it edits
# only THIS agent's own system-prompt.md and can never reach initialize.md (the input-security
# floor stays above self-authorship). Edits are guarded by a compare-and-swap confirm token and
# snapshot a .bak first; they take effect next wake.
#
# `note` rides into the Turn-0 tool manifest so the model is reminded, every wake, that this
# rewrites its own persona, is guarded, and lands next wake.
from basecradle_harness import SystemPromptEditTool, SystemPromptReadTool, ToolPlugin

PLUGINS = [
    ToolPlugin(impl=SystemPromptReadTool, opt_in=True),
    ToolPlugin(
        impl=SystemPromptEditTool,
        opt_in=True,
        note=(
            "rewrites your OWN persona (system-prompt.md, never initialize.md); guarded by a "
            "confirm token, snapshots a .bak, and takes effect next wake."
        ),
    ),
]
