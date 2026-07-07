"""The safety boundary: which tools a profile is allowed to load.

A `Tool` declares the capabilities it needs (`Tool.requires`). A `Policy`
forbids a set of capabilities. The `ToolRegistry` consults the policy at
registration time and refuses any tool that needs something forbidden — so the
boundary is enforced where tools enter the system, not left to a tool author's
good behavior.

Two Harness profiles bracket the design (CLAUDE.md spine #2). Both are Harness;
the engine is policy-neutral and the policy is the only difference between them:

- `Policy.locked()` — what Harness ships. Forbids `SHELL` (subprocess / arbitrary
  command execution). Combined with the fact that Harness ships **no** shell or
  exec tool and **no** primitive to spawn a subprocess, the shipped default
  forbids the shell capability outright. This is safe by *default*, not a standing
  guarantee: an operator opts out deliberately — by passing `unlocked()`, or by
  dropping a `tools/` tool or MCP server that reaches a denied capability into the
  config home. Leaving the safe zone is a deliberate, auditable operator act.
- `Policy.unlocked()` — forbids nothing. The unlocked profile an operator selects
  to grant shell, sudo, and self-modification. Present here only as the other end
  of the same dial; the shipped Harness never selects it for you.

Capabilities are plain strings so a tool author can invent new ones and a future
policy can forbid them. `SHELL` is the one the shipped boundary cares about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from basecradle_harness._tools import Tool

# The capability to run a subprocess or otherwise execute arbitrary commands.
# The locked profile forbids it; this is the line between the safe default and
# the unlocked profile — both of them Harness.
SHELL = "shell"

# The capability to act on the BaseCradle platform through the SDK — what every
# platform-aware tool (assets, and the later Phase-2 tranches) needs. It is *not*
# dangerous: the shipped locked profile permits it, because reading and posting on
# the platform is the whole point of a peer. It is named as a capability anyway so
# the boundary is honest — a profile that wanted a platform-blind agent could
# forbid it, exactly as `SHELL` is forbidden, with no change to any tool.
BASECRADLE = "basecradle"

# What the safe, shipped profile refuses. A frozenset so it cannot be mutated.
# `BASECRADLE` is deliberately absent — platform I/O is permitted under `locked()`.
DANGEROUS_CAPABILITIES = frozenset({SHELL})


@dataclass(frozen=True)
class Policy:
    """A decision about which capabilities a profile will allow a tool to need.

    `forbidden` is the set of capability names the policy refuses. A tool is
    permitted iff none of its `requires` are forbidden.
    """

    forbidden: frozenset[str] = field(default=DANGEROUS_CAPABILITIES)

    def permits(self, tool: Tool) -> bool:
        """True if none of the tool's required capabilities are forbidden."""
        return tool.requires.isdisjoint(self.forbidden)

    @classmethod
    def locked(cls) -> Policy:
        """The safe Harness profile: forbids shell/exec. The default everywhere."""
        return cls(forbidden=DANGEROUS_CAPABILITIES)

    @classmethod
    def unlocked(cls) -> Policy:
        """Forbids nothing — the unlocked profile. Never selected for you by the shipped Harness."""
        return cls(forbidden=frozenset())
