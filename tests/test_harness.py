"""The engine loop and the public `Harness`, driven by a scripted provider.

No HTTP and no real model: `ScriptedProvider` returns pre-set assistant turns in
order and records what it was asked, so the tests pin the loop's behavior — tool
dispatch, error recovery, the step guard, the safe default, and memory carrying
across turns.
"""

import pytest

from basecradle_harness import (
    SHELL,
    Engine,
    EngineError,
    Harness,
    MemoryTool,
    Message,
    Policy,
    PolicyError,
    Tool,
    ToolCall,
    ToolRegistry,
)


class ScriptedProvider:
    """A `Provider` that replays prepared assistant messages and records calls."""

    def __init__(self, *replies: Message) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[list[Message], object]] = []

    def chat(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        if not self._replies:
            raise AssertionError("ScriptedProvider ran out of replies")
        return self._replies.pop(0)


def text(content: str) -> Message:
    return Message.assistant(content=content)


def calls_tool(call_id: str, name: str, **arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


def tool_results(history: list[Message]) -> list[Message]:
    return [m for m in history if m.role == "tool"]


# --- A plain turn ------------------------------------------------------------


def test_send_returns_the_text_reply():
    agent = Harness(ScriptedProvider(text("Hello, peer.")))
    assert agent.send("Hi") == "Hello, peer."
    assert [m.role for m in agent.history] == ["user", "assistant"]


def test_system_prompt_seeds_history():
    agent = Harness(ScriptedProvider(text("ok")), system_prompt="be terse")
    assert agent.history[0].role == "system"
    agent.send("hi")
    assert [m.role for m in agent.history] == ["system", "user", "assistant"]


def test_tools_are_offered_to_the_provider(tmp_path):
    provider = ScriptedProvider(text("ok"))
    Harness(provider, tools=[MemoryTool(path=tmp_path / "m.json")]).send("hi")
    _, tools = provider.calls[0]
    assert [spec.name for spec in tools] == ["memory"]


def test_no_tools_means_none_is_passed():
    provider = ScriptedProvider(text("ok"))
    Harness(provider).send("hi")
    assert provider.calls[0][1] is None


# --- A think → tool → respond cycle -----------------------------------------


def test_tool_call_cycle_actually_runs_the_tool(tmp_path):
    path = tmp_path / "m.json"
    provider = ScriptedProvider(
        calls_tool("c1", "memory", action="write", key="city", value="Dallas"),
        text("Done — I'll remember that."),
    )
    agent = Harness(provider, tools=[MemoryTool(path=path)])

    reply = agent.send("Remember my city is Dallas.")

    assert reply == "Done — I'll remember that."
    # The write really happened: a fresh tool reads it back from disk.
    assert MemoryTool(path=path).run(action="read", key="city") == "Dallas"
    # The transcript is user → assistant(tool_calls) → tool → assistant(text).
    assert [m.role for m in agent.history] == ["user", "assistant", "tool", "assistant"]
    result = tool_results(agent.history)[0]
    assert result.tool_call_id == "c1"
    assert result.content == "Remembered 'city'."
    # The model's second call saw the tool result.
    assert any(m.role == "tool" for m in provider.calls[1][0])


def test_multi_step_tool_calls(tmp_path):
    """Two tool calls in a row, then a final answer — three provider calls."""
    provider = ScriptedProvider(
        calls_tool("c1", "memory", action="write", key="city", value="Dallas"),
        calls_tool("c2", "memory", action="read", key="city"),
        text("You're in Dallas."),
    )
    agent = Harness(provider, tools=[MemoryTool(path=tmp_path / "m.json")])

    assert agent.send("Where am I?") == "You're in Dallas."
    assert len(provider.calls) == 3
    assert tool_results(agent.history)[1].content == "Dallas"


def test_memory_persists_across_sends(tmp_path):
    """A fact written on one send is recalled on the next — across-turns memory."""
    path = tmp_path / "m.json"
    provider = ScriptedProvider(
        calls_tool("c1", "memory", action="write", key="city", value="Dallas"),
        text("Got it."),
        calls_tool("c2", "memory", action="read", key="city"),
        text("Dallas."),
    )
    agent = Harness(provider, tools=[MemoryTool(path=path)])

    assert agent.send("Remember city = Dallas.") == "Got it."
    assert agent.send("What city am I in?") == "Dallas."
    assert tool_results(agent.history)[-1].content == "Dallas"


# --- Resilience: tool failures become model-readable results -----------------


def test_unknown_tool_call_is_fed_back_not_crashed():
    provider = ScriptedProvider(
        calls_tool("c1", "nonexistent"),
        text("sorry, my mistake"),
    )
    agent = Harness(provider)  # no tools registered

    assert agent.send("hi") == "sorry, my mistake"
    assert "no tool named 'nonexistent'" in tool_results(agent.history)[0].content


def test_a_tool_that_raises_is_fed_back_not_crashed():
    class BoomTool(Tool):
        name = "boom"
        description = "Always raises."

        def run(self, **kwargs) -> str:
            raise RuntimeError("kaboom")

    provider = ScriptedProvider(calls_tool("c1", "boom"), text("recovered"))
    agent = Harness(provider, tools=[BoomTool()])

    assert agent.send("hi") == "recovered"
    result = tool_results(agent.history)[0].content
    assert "Error running 'boom'" in result and "kaboom" in result


# --- The step guard ----------------------------------------------------------


def test_max_steps_guard_raises_engine_error(tmp_path):
    never_done = [calls_tool(f"c{i}", "memory", action="list") for i in range(10)]
    provider = ScriptedProvider(*never_done)
    agent = Harness(provider, tools=[MemoryTool(path=tmp_path / "m.json")], max_steps=3)

    with pytest.raises(EngineError, match="3 steps"):
        agent.send("loop forever")
    assert len(provider.calls) == 3


# --- Safe by default ---------------------------------------------------------


class ShellTool(Tool):
    name = "shell"
    description = "Run a command."
    requires = frozenset({SHELL})

    def run(self, **kwargs) -> str:  # pragma: no cover - never loads under locked
        return "ran"


def test_a_shell_tool_is_refused_at_construction():
    with pytest.raises(PolicyError):
        Harness(ScriptedProvider(), tools=[ShellTool()])


def test_unlocked_profile_admits_the_shell_tool():
    """The Cradle seam: same Harness, unlocked policy, the tool loads."""
    agent = Harness(ScriptedProvider(text("ok")), tools=[ShellTool()], policy=Policy.unlocked())
    assert "shell" in agent.tools


# --- The engine directly -----------------------------------------------------


def test_engine_appends_the_full_transcript():
    provider = ScriptedProvider(text("done"))
    engine = Engine(provider, ToolRegistry())
    messages = [Message.user("hi")]

    final = engine.run(messages)

    assert final.content == "done"
    # run() extended the same list with the assistant turn.
    assert [m.role for m in messages] == ["user", "assistant"]
