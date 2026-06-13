"""The unified-identity model: per-source sessions atop one shared memory.

These pin the constitution's requirement that an agent is one memory-and-charter
locus addressed over many channels — channels share memory and charter, never
conversation — and that a past session's reasoning stays answerable from another.

As elsewhere, a `ScriptedProvider` replays prepared assistant turns, so the loop
runs with no model and no network.
"""

import pytest

from basecradle_harness import (
    Harness,
    ImageContent,
    MemoryTool,
    Message,
    Session,
    ToolCall,
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


# --- Channels share memory, not conversation ---------------------------------


def test_sessions_keep_separate_transcripts():
    """Two channels of one agent do not bleed into one transcript."""
    agent = Harness(ScriptedProvider(text("a1"), text("b1")))

    agent.send("from A", source="github:pr-1")
    agent.send("from B", source="timeline:x")

    a = agent.session("github:pr-1").history
    b = agent.session("timeline:x").history
    assert [m.content for m in a] == ["from A", "a1"]
    assert [m.content for m in b] == ["from B", "b1"]


def test_every_session_starts_from_the_shared_charter():
    """The charter (system prompt) seeds each new conversation, on every channel."""
    agent = Harness(ScriptedProvider(text("ok"), text("ok")), system_prompt="be terse")

    one = agent.session("github:pr-1")
    two = agent.session("timeline:x")

    assert one.history[0].role == "system" and one.history[0].content == "be terse"
    assert two.history[0].role == "system" and two.history[0].content == "be terse"
    assert one is not two  # distinct conversations…
    assert one.engine is two.engine  # …on one shared brain + memory


def test_memory_written_on_one_channel_is_read_on_another(tmp_path):
    """The motivating convergence: shared durable memory across sessions."""
    path = tmp_path / "m.json"
    provider = ScriptedProvider(
        calls_tool("c1", "memory", action="write", key="why-pr-123", value="retry was flaky"),
        text("Logged."),
        calls_tool("c2", "memory", action="read", key="why-pr-123"),
        text("Because the retry was flaky."),
    )
    agent = Harness(provider, tools=[MemoryTool(path=path)])

    # Work happens on the GitHub channel…
    agent.send("Note why I changed PR #123.", source="github:pr-123")
    # …and a peer asks about it on the timeline — a different conversation, same memory.
    reply = agent.send("Why did you change PR #123?", source="timeline:abc")

    assert reply == "Because the retry was flaky."
    # The two channels never shared a transcript…
    assert (
        len(agent.session("github:pr-123").history) == 4
    )  # user, assistant(call), tool, assistant
    assert len(agent.session("timeline:abc").history) == 4
    # …but the fact crossed between them through the one memory store.


# --- Cross-session answerability via readable transcripts --------------------


def test_transcript_reads_another_live_sessions_history():
    agent = Harness(ScriptedProvider(text("I shipped the retry fix.")))
    agent.send("status?", source="github:pr-123")

    seen = agent.transcript("github:pr-123")

    assert [m.content for m in seen] == ["status?", "I shipped the retry fix."]
    # It is a copy: mutating the returned list does not touch the session.
    seen.clear()
    assert len(agent.session("github:pr-123").history) == 2


def test_transcript_of_unknown_source_is_empty():
    agent = Harness(ScriptedProvider())
    assert agent.transcript("never-spoke-here") == []


# --- Durable across a restart, when a home is given --------------------------


def test_transcripts_persist_and_reload_across_instances(tmp_path):
    """With a home, a session's reasoning survives the instance that produced it."""
    provider = ScriptedProvider(text("Did the thing because X."))
    first = Harness(provider, home=tmp_path)
    first.send("do the thing", source="github:pr-9")

    # A fresh agent over the same home — a restart — can still read the old session.
    second = Harness(ScriptedProvider(), home=tmp_path)
    reloaded = second.transcript("github:pr-9")
    assert [m.content for m in reloaded] == ["do the thing", "Did the thing because X."]

    # And re-opening the session resumes its transcript rather than starting fresh.
    resumed = second.session("github:pr-9")
    assert [m.content for m in resumed.history] == ["do the thing", "Did the thing because X."]


def test_no_home_means_no_transcript_files_are_written(tmp_path):
    agent = Harness(ScriptedProvider(text("ok")), home=None)
    agent.send("hi", source="github:pr-1")
    assert not (tmp_path / "sessions").exists()


def test_source_with_separators_maps_to_one_safe_file(tmp_path):
    agent = Harness(ScriptedProvider(text("ok")), home=tmp_path)
    agent.send("hi", source="github:org/repo#123")

    files = list((tmp_path / "sessions").iterdir())
    assert len(files) == 1  # the ':' and '/' did not spawn nested dirs or collide


# --- The default session is just a named session -----------------------------


def test_default_send_and_explicit_default_source_are_the_same_session():
    agent = Harness(ScriptedProvider(text("a"), text("b")))
    agent.send("one")  # default source
    agent.send("two", source="default")  # explicitly the same one
    assert [m.content for m in agent.history] == ["one", "a", "two", "b"]


# --- Message persistence round-trip ------------------------------------------


def test_message_to_dict_from_dict_round_trips_a_tool_call():
    original = Message.assistant(
        tool_calls=[ToolCall(id="c1", name="memory", arguments={"action": "list"})]
    )
    restored = Message.from_dict(original.to_dict())
    assert restored.role == "assistant"
    assert restored.tool_calls[0].name == "memory"
    assert restored.tool_calls[0].arguments == {"action": "list"}


def test_session_is_exported_and_constructible_directly():
    # Session is part of the public surface for callers wiring their own routing.
    assert isinstance(Session, type)


# --- Presented images (vision) are shown once, then evicted ------------------


def test_send_presents_images_then_evicts_the_pixels():
    """An image passed to `send` reaches the model on that turn, then its pixels are
    dropped from history (the text breadcrumb stays) so it is never re-sent later."""
    image = ImageContent(url="data:image/png;base64,AAAA", alt="cat.png")

    class CapturesImages:
        """Snapshots the last turn's images at chat time, before the session evicts them."""

        def __init__(self):
            self.seen = None

        def chat(self, messages, tools=None):
            self.seen = list(messages[-1].images)
            return text("I see a cat.")

    provider = CapturesImages()
    session = Session("timeline:x", Harness(provider).engine)

    reply = session.send("look at this", images=[image])

    assert reply == "I see a cat."
    assert provider.seen == [image]  # the model saw the pixels on the turn they were sent
    # …but they were evicted from the stored history afterward (text breadcrumb kept).
    user_turn = session.history[0]
    assert user_turn.content == "look at this"
    assert user_turn.images == []


def test_presented_images_are_evicted_even_when_the_turn_fails():
    """COST DISCIPLINE: if the engine raises, the pixels are still evicted, so a failed
    turn cannot leave base64 in history to be re-sent (or persisted) on a later send."""

    class Boom:
        def chat(self, messages, tools=None):
            raise RuntimeError("model exploded mid-turn")

    session = Session("timeline:x", Harness(Boom()).engine)
    image = ImageContent(url="data:image/png;base64,AAAA", alt="cat.png")

    with pytest.raises(RuntimeError):
        session.send("look at this", images=[image])

    assert session.history[-1].images == []  # evicted on the error path too
