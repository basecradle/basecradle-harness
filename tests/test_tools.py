"""The tool contract, the registry, and the policy boundary.

The headline invariant: the shipped (locked) profile cannot load a tool that
needs a shell. That is asserted directly here.
"""

import pytest

from basecradle_harness import (
    SHELL,
    Message,
    Policy,
    PolicyError,
    Tool,
    ToolRegistry,
    ToolSpec,
)
from basecradle_harness._openai_wire import chat_tool_to_wire
from basecradle_harness._tools import NO_PARAMETERS


class EchoTool(Tool):
    name = "echo"
    description = "Echo a phrase back."
    parameters = {
        "type": "object",
        "properties": {"phrase": {"type": "string"}},
        "required": ["phrase"],
    }

    def run(self, phrase: str) -> str:
        return phrase


class PingTool(Tool):
    name = "ping"
    description = "Reply with pong. Takes no arguments."

    def run(self) -> str:
        return "pong"


class ShellTool(Tool):
    """A tool that needs to run commands — exactly what the safe profile forbids."""

    name = "shell"
    description = "Run a shell command."
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    requires = frozenset({SHELL})

    def run(self, command: str) -> str:  # pragma: no cover - never loads under locked
        return f"pretended to run: {command}"


# --- The Tool contract -------------------------------------------------------


def test_tool_to_spec_mirrors_its_attributes():
    spec = EchoTool().to_spec()
    assert spec == ToolSpec(
        name="echo",
        description="Echo a phrase back.",
        parameters=EchoTool.parameters,
    )


def test_tool_defaults_to_no_parameters_and_no_requirements():
    ping = PingTool()
    assert ping.parameters == NO_PARAMETERS
    assert ping.requires == frozenset()


# --- The registry ------------------------------------------------------------


def test_register_then_get_and_run():
    registry = ToolRegistry()
    registry.register(EchoTool())

    assert "echo" in registry
    assert len(registry) == 1
    assert registry.get("echo").name == "echo"
    assert registry.run("echo", phrase="hello") == "hello"


def test_specs_lists_every_tool_in_registration_order():
    registry = ToolRegistry()
    registry.register(PingTool())
    registry.register(EchoTool())

    assert [s.name for s in registry.specs()] == ["ping", "echo"]


def test_duplicate_name_is_rejected():
    registry = ToolRegistry()
    registry.register(EchoTool())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())


def test_tool_without_a_name_is_rejected():
    class Nameless(Tool):
        description = "no name"

        def run(self) -> str:
            return ""

    with pytest.raises(ValueError, match="non-empty `name`"):
        ToolRegistry().register(Nameless())


def test_get_missing_tool_raises_key_error():
    with pytest.raises(KeyError):
        ToolRegistry().get("absent")


def test_iteration_yields_the_tools():
    registry = ToolRegistry()
    echo, ping = EchoTool(), PingTool()
    registry.register(echo)
    registry.register(ping)
    assert list(registry) == [echo, ping]


# --- The policy boundary -----------------------------------------------------


def test_locked_profile_cannot_load_a_shell_tool():
    """The headline safety invariant."""
    registry = ToolRegistry(policy=Policy.locked())
    with pytest.raises(PolicyError, match="shell"):
        registry.register(ShellTool())
    assert "shell" not in registry


def test_default_registry_is_locked_by_default():
    """No policy argument still means safe — a shell tool is refused."""
    with pytest.raises(PolicyError):
        ToolRegistry().register(ShellTool())


def test_unlocked_profile_loads_and_runs_the_shell_tool():
    """The Cradle seam: the same engine, an unlocked policy, the tool loads."""
    registry = ToolRegistry(policy=Policy.unlocked())
    registry.register(ShellTool())
    assert registry.run("shell", command="ls") == "pretended to run: ls"


def test_safe_tools_load_under_the_locked_profile():
    registry = ToolRegistry(policy=Policy.locked())
    registry.register(EchoTool())
    registry.register(PingTool())
    assert len(registry) == 2


def test_policy_permits_is_capability_disjointness():
    assert Policy.locked().permits(EchoTool()) is True
    assert Policy.locked().permits(ShellTool()) is False
    assert Policy.unlocked().permits(ShellTool()) is True


# --- The seam to the provider -----------------------------------------------


def test_registry_specs_serialize_through_the_provider_adapter():
    """A registry's specs are exactly what the OpenAI adapter turns into tools."""
    registry = ToolRegistry()
    registry.register(EchoTool())

    wire = chat_tool_to_wire(registry.specs()[0])
    assert wire == {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo a phrase back.",
            "parameters": EchoTool.parameters,
        },
    }


def test_specs_feed_a_provider_as_its_tools_argument():
    """End to end through the seam: registry specs satisfy Provider.chat's tools."""
    captured: dict[str, object] = {}

    class RecordingProvider:
        def chat(self, messages, tools=None):
            captured["tools"] = tools
            return Message.assistant(content="ok")

    registry = ToolRegistry()
    registry.register(EchoTool())
    RecordingProvider().chat([Message.user("hi")], tools=registry.specs())

    assert [t.name for t in captured["tools"]] == ["echo"]
