"""The engine loop, with a focus on how it carries images through a tool turn.

No model and no platform: a scripted fake provider returns a canned sequence of
turns, and a fake tool returns a `ToolResult` carrying an image. These tests pin
the vision plumbing the issue called "the real work" — a tool's image becomes
model *input* on the next turn, then the pixels are evicted from the transcript
once the model has answered, so a viewed image is never re-sent.
"""

import re

import pytest

from basecradle_harness import (
    Engine,
    EngineError,
    ImageContent,
    Message,
    ProviderAuthError,
    ProviderResponseError,
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
    """Calls the view tool while tools are offered; answers in text once they're withheld.

    Drives the loop to the budget, then settles on the engine's out-of-budget reserve call
    (``tools=None``) so a viewed image is exercised on the reserve path too.
    """

    def chat(self, messages, tools=None):
        if tools is None:
            return Message.assistant(content="Out of steps; here's the summary.")
        return Message.assistant(tool_calls=[ToolCall(id="c", name="view", arguments={})])


def _is_counter(m: Message) -> bool:
    return m.role == "system" and bool(m.content) and bool(re.search(r"Step \d+ of \d+", m.content))


def test_images_are_evicted_even_when_the_step_limit_is_hit():
    """The eviction guarantee holds when the budget is spent (try/finally), not just on success."""
    registry = ToolRegistry()
    registry.register(ViewTool())
    engine = Engine(AlwaysViewProvider(), registry, max_steps=2)
    history = [Message.user("look")]

    reply = engine.run(history)  # the reserve summary, not an EngineError

    assert reply.content == "Out of steps; here's the summary."
    # No base64 lingers in the mutated-in-place transcript to be re-sent next turn.
    assert not any(m.images for m in history)


class ReserveBlowsUpProvider:
    """Calls the view tool while budgeted; the out-of-budget reserve call itself raises."""

    def chat(self, messages, tools=None):
        if tools is None:
            raise RuntimeError("reserve call blew up")
        return Message.assistant(tool_calls=[ToolCall(id="c", name="view", arguments={})])


def test_reserve_failure_still_evicts_images_and_raises():
    """When the reserve call itself errors, EngineError raises and images still evict (finally)."""
    registry = ToolRegistry()
    registry.register(ViewTool())
    engine = Engine(ReserveBlowsUpProvider(), registry, max_steps=2)
    history = [Message.user("look")]

    with pytest.raises(EngineError):
        engine.run(history)

    assert not any(m.images for m in history)


class ReserveReturnsToolCallProvider:
    """Never stops calling a tool — even the reserve call answers with a lone tool call, no text.

    Models a server-tool persona: `tools=None` withholds the harness's function tools but a
    server-side built-in can still resolve, and the model can come back with no usable text.
    """

    def chat(self, messages, tools=None):
        return Message.assistant(tool_calls=[ToolCall(id="c", name="echo", arguments={})])


def test_reserve_reply_with_no_text_raises_and_persists_no_dangling_tool_call():
    """A textless reserve reply (a lone tool call) falls back to EngineError, and its dangling
    assistant tool-call turn is NOT persisted — else the next wake's transcript is malformed."""
    engine = _engine(ReserveReturnsToolCallProvider(), EchoTool())
    engine.max_steps = 2
    history = [Message.user("go")]

    with pytest.raises(EngineError, match="produced no text"):
        engine.run(history)

    # The transcript must not end on an assistant turn with tool_calls and no following tool
    # result — every persisted assistant tool-call turn is answered by a tool turn.
    for i, m in enumerate(history):
        if m.role == "assistant" and m.tool_calls:
            assert i + 1 < len(history) and history[i + 1].role == "tool"


class ReserveTextPlusToolCallProvider:
    """Budgeted turns call a tool; the reserve turn answers with text AND a stray tool call."""

    def chat(self, messages, tools=None):
        if tools is None:
            return Message.assistant(
                content="Here's my progress.",
                tool_calls=[ToolCall(id="stray", name="echo", arguments={})],
            )
        return Message.assistant(tool_calls=[ToolCall(id="c", name="echo", arguments={})])


def test_reserve_summary_persists_text_only_dropping_stray_tool_calls():
    """A reserve reply with text keeps the text but drops any stray tool calls, so the persisted
    turn is a clean assistant text turn (no dangling tool-call to poison the next wake)."""
    engine = _engine(ReserveTextPlusToolCallProvider(), EchoTool())
    engine.max_steps = 2
    history = [Message.user("go")]

    reply = engine.run(history)

    assert reply.content == "Here's my progress."
    assert reply.tool_calls == []  # the stray call was dropped
    assert history[-1] is reply and history[-1].tool_calls == []


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
    # The step-counter notes are filtered out; the underlying flow is unchanged.
    assert [m.role for m in history if not _is_counter(m)] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


# --- the live step counter (issue #243) --------------------------------------


def _counters(history):
    return [m for m in history if _is_counter(m)]


def test_a_step_counter_note_precedes_every_provider_call():
    provider = ScriptedProvider(
        Message.assistant(tool_calls=[ToolCall(id="c1", name="echo", arguments={})]),
        Message.assistant(content="done"),
    )
    engine = _engine(provider, EchoTool())
    history = [Message.user("go")]

    engine.run(history)

    # Two provider calls → two counter notes, and each lands immediately before its call.
    counters = _counters(history)
    assert [c.content.splitlines()[-1] for c in counters] == ["Step 1 of 24.", "Step 2 of 24."]
    # The note is a trailing system turn: the assistant reply follows right after it.
    idx = history.index(counters[0])
    assert history[idx + 1].role == "assistant"


def test_the_counter_note_carries_a_fresh_timestamp():
    from datetime import datetime, timezone

    stamp = datetime(2026, 7, 4, 20, 4, 12, tzinfo=timezone.utc)
    provider = ScriptedProvider(Message.assistant(content="done"))
    engine = Engine(provider, ToolRegistry(), clock=lambda: stamp)
    history = [Message.user("go")]

    engine.run(history)

    note = _counters(history)[0].content
    assert note.startswith("Current Time: 2026-07-04 20:04:12 UTC")
    assert note.endswith("Step 1 of 24.")


def test_the_counter_escalates_in_the_final_stretch():
    # With a budget of 3, every step is within the last 5 → the escalation guidance shows.
    provider = ScriptedProvider(Message.assistant(content="done"))
    engine = Engine(provider, ToolRegistry(), max_steps=3)
    history = [Message.user("go")]

    engine.run(history)

    note = _counters(history)[0].content
    assert "Step 1 of 3." in note
    assert "running low" in note
    assert "final action step" in note  # tells it to land with a text reply


def test_counter_notes_persist_as_a_ledger_and_are_not_evicted():
    provider = ScriptedProvider(
        Message.assistant(tool_calls=[ToolCall(id="c1", name="view", arguments={})]),
        Message.assistant(content="done"),
    )
    engine = _engine(provider, ViewTool())
    history = [Message.user("look")]

    engine.run(history)

    # The image pixels are evicted, but the (tiny) counter notes stay as the step ledger.
    assert not any(m.images for m in history)
    assert len(_counters(history)) == 2


# --- per-step logging (issue #244) -------------------------------------------


def test_each_step_logs_its_tools_and_a_final_summary_line(caplog):
    import logging

    provider = ScriptedProvider(
        Message.assistant(tool_calls=[ToolCall(id="c1", name="echo", arguments={})]),
        Message.assistant(content="done"),
    )
    engine = _engine(provider, EchoTool())

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        engine.run([Message.user("go")])

    messages = [r.getMessage() for r in caplog.records]
    assert any("step 1/24: tools=echo" in m for m in messages)
    assert any("step 2/24: final reply" in m for m in messages)
    assert any("wake used 2/24 steps" in m for m in messages)


def test_a_capped_wake_logs_the_reserve_summary_marker(caplog):
    import logging

    class Loops:
        def chat(self, messages, tools=None):
            if tools is None:
                return Message.assistant(content="summary")
            return Message.assistant(tool_calls=[ToolCall(id="c", name="echo", arguments={})])

    engine = _engine(Loops(), EchoTool())
    engine.max_steps = 2

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        engine.run([Message.user("go")])

    assert any("wake used 2/2 steps + reserve summary" in r.getMessage() for r in caplog.records)


# --- server-side built-in called as a function (issue #245) ------------------


def test_a_server_builtin_called_as_a_function_gets_targeted_guidance():
    provider = ScriptedProvider(
        Message.assistant(tool_calls=[ToolCall(id="c1", name="web_search", arguments={})]),
        Message.assistant(content="ok"),
    )
    engine = Engine(provider, ToolRegistry(), server_builtins=["web_search"])
    history = [Message.user("look it up")]

    engine.run(history)

    result = next(m for m in history if m.role == "tool")
    assert "runs server-side" in result.content
    assert "Do not retry" in result.content
    assert "no tool named" not in result.content  # not the generic error


def test_an_unknown_tool_still_gets_the_generic_error():
    provider = ScriptedProvider(
        Message.assistant(tool_calls=[ToolCall(id="c1", name="nonesuch", arguments={})]),
        Message.assistant(content="ok"),
    )
    engine = Engine(provider, ToolRegistry(), server_builtins=["web_search"])
    history = [Message.user("go")]

    engine.run(history)

    result = next(m for m in history if m.role == "tool")
    assert "no tool named 'nonesuch'" in result.content


# --- bounded retry of a truncated / unparseable provider response (issue #259) ---


class FlakyProvider:
    """Raises `ProviderResponseError` on its first `fails` calls, then returns `reply`.

    Models the observed GLM-5.2 flake: a provider call that comes back unparseable (a truncated
    body / EOF-mid-JSON) and then, re-issued, succeeds. Records how many times it was called.
    """

    def __init__(self, fails: int, reply: Message) -> None:
        self._fails = fails
        self._reply = reply
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls <= self._fails:
            raise ProviderResponseError(f"EOF while parsing a value (call {self.calls})")
        return self._reply


class AlwaysTruncatedProvider:
    """Every call comes back unparseable — the wedged case the retry must still bound."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        raise ProviderResponseError(f"EOF while parsing a value (call {self.calls})")


class AuthFailingProvider:
    """Raises a *permanent* provider error — the class the retry must NOT re-attempt."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        raise ProviderAuthError("bad key", status_code=401)


def _no_sleep():
    """A sleep spy: records the backoff delays it was asked to wait, and never actually sleeps."""
    delays: list[float] = []
    return delays, delays.append


def test_a_truncated_response_is_retried_then_succeeds():
    """The first attempt fails validation, the retry succeeds — the engine returns the reply
    instead of aborting the wake (issue #259, definition-of-done point 1)."""
    provider = FlakyProvider(fails=1, reply=Message.assistant(content="the real answer"))
    delays, spy = _no_sleep()
    engine = Engine(provider, ToolRegistry(), sleep=spy)
    history = [Message.user("hi")]

    reply = engine.run(history)

    assert reply.content == "the real answer"
    assert provider.calls == 2  # one failure, one success
    assert delays == [0.5]  # exactly one backoff, at the base delay


def test_retries_are_bounded_and_the_last_error_propagates():
    """When every attempt fails, the engine gives up after response_retries+1 tries and re-raises
    the ProviderResponseError — the wake then aborts cleanly (definition-of-done point 1/2)."""
    provider = AlwaysTruncatedProvider()
    delays, spy = _no_sleep()
    engine = Engine(provider, ToolRegistry(), response_retries=2, sleep=spy)

    with pytest.raises(ProviderResponseError):
        engine.run([Message.user("hi")])

    assert provider.calls == 3  # 1 initial + 2 retries
    assert delays == [0.5, 1.0]  # a backoff before each retry, scaled by attempt


def test_the_give_up_leaves_a_diagnosable_log_trail(caplog):
    """On exhaustion the engine logs a WARNING per retry and an ERROR naming the attempt count, so
    a dropped wake is diagnosable from logs alone (definition-of-done point 2)."""
    provider = AlwaysTruncatedProvider()
    _delays, spy = _no_sleep()
    engine = Engine(provider, ToolRegistry(), response_retries=2, sleep=spy)

    with caplog.at_level("WARNING", logger="basecradle_harness"):
        with pytest.raises(ProviderResponseError):
            engine.run([Message.user("hi")])

    warnings = [
        r for r in caplog.records if r.levelname == "WARNING" and "unparseable" in r.message
    ]
    errors = [r for r in caplog.records if r.levelname == "ERROR" and "unparseable" in r.message]
    assert len(warnings) == 2  # one per retry attempt
    assert len(errors) == 1  # the final give-up
    assert "all 3 attempt(s)" in errors[0].message  # names the attempt count


def test_a_permanent_provider_error_is_not_retried():
    """A permanent failure (auth) is never re-attempted — retrying it only repeats it, so it
    propagates on the first raise with no backoff (definition-of-done: only the response class)."""
    provider = AuthFailingProvider()
    delays, spy = _no_sleep()
    engine = Engine(provider, ToolRegistry(), response_retries=5, sleep=spy)

    with pytest.raises(ProviderAuthError):
        engine.run([Message.user("hi")])

    assert provider.calls == 1  # not retried
    assert delays == []  # no backoff


def test_response_retries_zero_disables_the_retry():
    """response_retries=0 → a single attempt, no retry — the pre-issue behavior, opt-in."""
    provider = AlwaysTruncatedProvider()
    delays, spy = _no_sleep()
    engine = Engine(provider, ToolRegistry(), response_retries=0, sleep=spy)

    with pytest.raises(ProviderResponseError):
        engine.run([Message.user("hi")])

    assert provider.calls == 1
    assert delays == []
