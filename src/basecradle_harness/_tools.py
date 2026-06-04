"""The extension surface: a tool is a small class; a registry holds them.

Writing a tool is the one thing a Harness hacker does most, so the contract is
as small as it can be: set a `name`, a `description`, and a JSON-Schema
`parameters`, then implement `run`. That's a usable tool.

    class Echo(Tool):
        name = "echo"
        description = "Echo a phrase back."
        parameters = {
            "type": "object",
            "properties": {"phrase": {"type": "string"}},
            "required": ["phrase"],
        }

        def run(self, phrase: str) -> str:
            return phrase

The `ToolRegistry` collects tools, gates each one through a `Policy` as it is
registered, and produces the `ToolSpec` list a `Provider` needs. The policy gate
is the safety boundary — see `_policy.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from basecradle_harness._exceptions import PolicyError
from basecradle_harness._messages import ToolSpec
from basecradle_harness._policy import Policy

# The JSON Schema for a tool that takes no arguments.
NO_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}


class Tool(ABC):
    """A capability the model can invoke.

    Subclasses set three class attributes and implement `run`:

    - `name` — the identifier the model calls (unique within a registry).
    - `description` — what it does, written for the model to read.
    - `parameters` — a JSON-Schema object describing `run`'s keyword arguments;
      defaults to "no arguments".
    - `requires` — capability names this tool needs (e.g. ``{SHELL}``). Empty by
      default — a pure tool needs nothing and loads under any policy.
    """

    name: str
    description: str
    parameters: dict[str, Any] = NO_PARAMETERS
    requires: frozenset[str] = frozenset()

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return its result as a string."""

    def to_spec(self) -> ToolSpec:
        """The provider-neutral schema the model is shown."""
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)


class ToolRegistry:
    """A policy-gated collection of tools.

    Defaults to `Policy.locked()` — safe by default. Loading a tool that needs a
    forbidden capability raises `PolicyError` at `register` time; you never get a
    registry holding a tool the policy would refuse.
    """

    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy or Policy.locked()
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        """Add a tool, after the policy permits it. Returns the tool for chaining."""
        if not getattr(tool, "name", None):
            raise ValueError(f"{type(tool).__name__} must set a non-empty `name`.")
        if not self.policy.permits(tool):
            blocked = sorted(tool.requires & self.policy.forbidden)
            raise PolicyError(
                f"Policy forbids tool {tool.name!r}: it requires {blocked}, "
                f"which this profile does not allow."
            )
        if tool.name in self._tools:
            raise ValueError(f"A tool named {tool.name!r} is already registered.")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        """The tool registered under `name`, or `KeyError` if absent."""
        return self._tools[name]

    def run(self, name: str, **kwargs: Any) -> str:
        """Invoke a registered tool by name."""
        return self.get(name).run(**kwargs)

    def specs(self) -> list[ToolSpec]:
        """The `ToolSpec` list to hand a `Provider`, in registration order."""
        return [tool.to_spec() for tool in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
