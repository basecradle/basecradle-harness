"""The engine loop, with a focus on how it carries images through a tool turn.

No model and no platform: a scripted fake provider returns a canned sequence of
turns, and a fake tool returns a `ToolResult` carrying an image. These tests pin
the vision plumbing the issue called "the real work" — a tool's image becomes
model *input* on the next turn, then the pixels are evicted from the transcript
once the model has answered, so a viewed image is never re-sent.
"""

import pytest

from basecradle_harness import (
    Engine,
    EngineError,
    ImageContent,
    Message,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
)


class ScriptedProvider:
    """A `Provider` that replays a fixed list of assistant turns, one per call.

    It records the `messages` it was handed on each call, so a test can assert
    what the model actually saw (e.g. an image part) on a given turn.
    """

    def __init__(self, *replies: Message) -> None:
        self._replies = list(replies)
        self.seen: list[list[Message]] = []

    def chat(self, messages, tools=None):
        self.seen.append([_clone(m) for m in messages])
        return self._replies.pop(0)


def _clone(message: Message) -> Message:
    """A shallow snapshot of a turn, so later mutation (eviction) doesn't rewrite history."""
    return Message(
        role=message.role,
        content=message.content,
        tool_calls=list(message.tool_calls),
        tool_call_id=message.tool_call_id,
        images=list(message.images),
    )


class ViewTool(Tool):
    """A fake 'view' tool: returns text plus one image, like the assets tool's view."""

    name = "view"
    description = "Look at an image."

    def run(self, **kwargs):
        return ToolResult(
            text="Looking at cat.png now.",
            images=[ImageContent(url="data:image/png;base64,AAAA", alt="cat.png")],
        )


class EchoTool(Tool):
    """A fake plain tool: returns a string, the common case."""

    name = "echo"
    description = "Echo."

    def run(self, **kwargs):
        return "echoed"


def _engine(provider, tool):
    registry = ToolRegistry()
    registry.register(tool)
    return Engine(provider, registry)


def test_a_tool_image_becomes_model_input_on_the_next_turn():
    provider = ScriptedProvider(
        Message.assistant(tool_calls=[ToolCall(id="c1", name="view", arguments={})]),
        Message.assistant(content="It's a tabby cat."),
    )
    engine = _engine(provider, ViewTool())

    reply = engine.run([Message.user("look at cat.png")])

    assert reply.content == "It's a tabby cat."
    # On the SECOND provider call, the model saw the injected image turn.
    second_call = provider.seen[1]
    image_turn = next(m for m in second_call if m.images)
    assert image_turn.role == "user"
    assert image_turn.images[0].url == "data:image/png;base64,AAAA"
    assert image_turn.content == "(Showing image: cat.png)"


def test_viewed_image_pixels_are_evicted_after_the_reply():
    provider = ScriptedProvider(
        Message.assistant(tool_calls=[ToolCall(id="c1", name="view", arguments={})]),
        Message.assistant(content="A cat."),
    )
    engine = _engine(provider, ViewTool())
    history = [Message.user("look")]

    engine.run(history)

    # The injected image turn survives as a breadcrumb, but its pixels are gone —
    # so the next turn (or a reload) never re-sends the image.
    image_turn = next(m for m in history if m.content == "(Showing image: cat.png)")
    assert image_turn.images == []
    assert not any(m.images for m in history)


def test_the_tool_result_text_is_the_tool_message():
    provider = ScriptedProvider(
        Message.assistant(tool_calls=[ToolCall(id="c1", name="view", arguments={})]),
        Message.assistant(content="done"),
    )
    engine = _engine(provider, ViewTool())
    history = [Message.user("look")]

    engine.run(history)

    tool_turn = next(m for m in history if m.role == "tool")
    assert tool_turn.content == "Looking at cat.png now."


class AlwaysViewProvider:
    """A provider that never stops calling the view tool — drives the loop to max_steps."""

    def chat(self, messages, tools=None):
        return Message.assistant(tool_calls=[ToolCall(id="c", name="view", arguments={})])


def test_images_are_evicted_even_when_the_step_limit_is_hit():
    """The eviction guarantee holds on the error path too (try/finally), not just success."""
    registry = ToolRegistry()
    registry.register(ViewTool())
    engine = Engine(AlwaysViewProvider(), registry, max_steps=2)
    history = [Message.user("look")]

    with pytest.raises(EngineError):
        engine.run(history)

    # No base64 lingers in the mutated-in-place transcript to be re-sent next turn.
    assert not any(m.images for m in history)


def test_a_plain_string_tool_injects_no_image_turn():
    provider = ScriptedProvider(
        Message.assistant(tool_calls=[ToolCall(id="c1", name="echo", arguments={})]),
        Message.assistant(content="ok"),
    )
    engine = _engine(provider, EchoTool())
    history = [Message.user("echo please")]

    engine.run(history)

    # No image turns at all — a str tool result behaves exactly as it always has.
    assert not any(m.images for m in history)
    assert [m.role for m in history] == ["user", "assistant", "tool", "assistant"]
