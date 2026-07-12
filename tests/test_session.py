"""The unified-identity model: per-source sessions atop one shared memory.

These pin the constitution's requirement that an agent is one memory-and-charter
locus addressed over many channels — channels share memory and charter, never
conversation — and that a past session's reasoning stays answerable from another.

As elsewhere, a `ScriptedProvider` replays prepared assistant turns, so the loop
runs with no model and no network.
"""

import json
import re

import pytest

from basecradle_harness import (
    Harness,
    ImageContent,
    MemoryTool,
    Message,
    Session,
    Tool,
    ToolCall,
)
from basecradle_harness._session import TOOL_RESULT_CAP


class ScriptedProvider:
    """A `Provider` that replays prepared assistant messages and records calls."""

    def __init__(self, *replies: Message) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[list[Message], object]] = []
        #: `(role, content)` per turn, snapshotted at *chat time*. `calls` holds the live
        #: `Message` objects, which the session later mutates in place (evicting pixels,
        #: capping tool results) — so only a snapshot can answer "what did the model actually
        #: read on this call?", which is the whole question issue #275 turns on.
        self.snapshots: list[list[tuple[str, str | None]]] = []

    def chat(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        self.snapshots.append([(m.role, m.content) for m in messages])
        if not self._replies:
            raise AssertionError("ScriptedProvider ran out of replies")
        return self._replies.pop(0)


def text(content: str) -> Message:
    return Message.assistant(content=content)


def calls_tool(call_id: str, name: str, **arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


def _is_counter(m: Message) -> bool:
    """A live step-counter note the engine injects before each provider call (issue #243).

    Matched by its `Step N of M` line — never the brief's `Current Time:` anchor or its
    `Step budget:` statement, whose wording is deliberately close but carries no `N of M`.
    """
    return m.role == "system" and bool(m.content) and bool(re.search(r"Step \d+ of \d+", m.content))


def convo(history: list[Message]) -> list[Message]:
    """The transcript with the injected step-counter notes removed.

    These tests pin the conversation shape (separate transcripts, shared memory), not the
    per-step ledger the counter notes form — so they read the transcript without them.
    """
    return [m for m in history if not _is_counter(m)]


# --- Channels share memory, not conversation ---------------------------------


def test_sessions_keep_separate_transcripts():
    """Two channels of one agent do not bleed into one transcript."""
    agent = Harness(ScriptedProvider(text("a1"), text("b1")))

    agent.send("from A", source="github:pr-1")
    agent.send("from B", source="timeline:x")

    a = agent.session("github:pr-1").history
    b = agent.session("timeline:x").history
    assert [m.content for m in convo(a)] == ["from A", "a1"]
    assert [m.content for m in convo(b)] == ["from B", "b1"]


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
    # The two channels never shared a transcript… (counter notes filtered out)
    assert (
        len(convo(agent.session("github:pr-123").history)) == 4
    )  # user, assistant(call), tool, assistant
    assert len(convo(agent.session("timeline:abc").history)) == 4
    # …but the fact crossed between them through the one memory store.


# --- Cross-session answerability via readable transcripts --------------------


def test_transcript_reads_another_live_sessions_history():
    agent = Harness(ScriptedProvider(text("I shipped the retry fix.")))
    agent.send("status?", source="github:pr-123")

    seen = agent.transcript("github:pr-123")

    assert [m.content for m in convo(seen)] == ["status?", "I shipped the retry fix."]
    # It is a copy: mutating the returned list does not touch the session.
    seen.clear()
    assert len(convo(agent.session("github:pr-123").history)) == 2


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
    assert [m.content for m in convo(reloaded)] == ["do the thing", "Did the thing because X."]

    # And re-opening the session resumes its transcript rather than starting fresh.
    resumed = second.session("github:pr-9")
    assert [m.content for m in convo(resumed.history)] == [
        "do the thing",
        "Did the thing because X.",
    ]


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
    assert [m.content for m in convo(agent.history)] == ["one", "a", "two", "b"]


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
            # The engine appends a step-counter note as the last turn, so the image no longer
            # rides messages[-1]; a real adapter renders images wherever they sit, so scan all.
            self.seen = [img for m in messages for img in m.images]
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

    # No base64 anywhere in the persisted-on-failure transcript (the user turn's pixels
    # are evicted before the failure marker is appended).
    assert not any(m.images for m in session.history)


def test_partial_transcript_persists_on_engine_failure(tmp_path):
    """Issue #244: a failed run still writes its partial transcript to disk, marked failed,
    rather than discarding the whole ledger the way a save-only-on-success path did."""

    class Boom:
        def chat(self, messages, tools=None):
            raise RuntimeError("model exploded")

    path = tmp_path / "transcript.json"
    session = Session("timeline:x", Harness(Boom(), max_steps=2).engine, path=path)

    with pytest.raises(RuntimeError):
        session.send("do the thing")

    # The turns accumulated before the failure are on disk, with a failure marker at the tail.
    reloaded = Session("timeline:x", Harness(Boom(), max_steps=2).engine, path=path)
    assert any(t.content == "do the thing" for t in reloaded.history)  # the user turn survived
    marker = reloaded.history[-1]
    assert marker.role == "system"
    assert "turn failed" in marker.content and "RuntimeError" in marker.content


# --- The ephemeral brief: shown to the model, never persisted (issue #275) ----


def test_brief_is_shown_to_the_model_but_never_persisted():
    """The brief reaches the provider on this turn and leaves no trace in the transcript.

    It is recomposed fresh every wake (current time, step budget, live dashboard), so a
    persisted copy is a stale duplicate the model would re-read — and re-pay for — on every
    later turn. That is the bloat #275 exists to end.
    """
    provider = ScriptedProvider(text("Understood."))
    session = Session("timeline:x", Harness(provider).engine)

    session.send("what's up?", brief="Current Time: 2026-07-12. You are Nova.")

    read = [content for _, content in provider.snapshots[0]]
    assert any("You are Nova." in (c or "") for c in read)  # the model saw it…
    assert not any("You are Nova." in (m.content or "") for m in session.history)  # …not stored


def test_brief_rides_at_the_tail_immediately_before_the_newest_user_turn():
    """CACHE INVARIANT: stable content first, volatile content last.

    Provider prefix caching only pays out on a byte-stable prefix. The frozen transcript comes
    first and the per-wake brief is spliced in at the tail, just ahead of the newest user turn —
    so the cacheable prefix is the whole prior conversation. Hoisting the brief to position 0
    ("system prompts go first") would change the prefix on every request and silently destroy
    caching fleet-wide, while fixing nothing.
    """
    provider = ScriptedProvider(text("one"), text("two"))
    session = Session("timeline:x", Harness(provider).engine)

    session.send("first", brief="BRIEF-A")
    frozen = [m.content for m in session.history]  # the transcript the next call must not disturb
    session.send("second", brief="BRIEF-B")

    read = [content for _, content in provider.snapshots[1]]
    brief_idx = read.index("BRIEF-B")
    assert read[brief_idx + 1] == "second"  # brief, then the turn it governs
    # Everything ahead of the brief is exactly the frozen transcript — the stable, cacheable
    # prefix — and last wake's BRIEF-A is nowhere in it.
    assert read[:brief_idx] == frozen
    assert "BRIEF-A" not in read


def test_a_failed_turn_persists_no_brief():
    """A wake that errors must not grow the transcript by a brief it never got value from."""

    class Boom:
        def chat(self, messages, tools=None):
            raise RuntimeError("model exploded")

    session = Session("timeline:x", Harness(Boom()).engine)

    with pytest.raises(RuntimeError):
        session.send("do the thing", brief="EPHEMERAL BRIEF")

    assert not any("EPHEMERAL BRIEF" in (m.content or "") for m in session.history)
    assert "turn failed" in session.history[-1].content  # the failure ledger still persists


# --- Tool results: read in full, persisted capped (issue #275) ----------------


def test_an_oversized_tool_result_is_shown_whole_then_persisted_capped(tmp_path):
    """The model reads the full result on the turn it ran; what *persists* is head + tail.

    One mailbox listing (142 KB, live) otherwise taxes every future wake for the life of the
    timeline — the same permanent-cost trap the engine already closed for images. Text gets the
    same discipline: seen once in full, never re-billed in full.
    """
    dump = "MAILBOX\n" + ("x" * 60_000) + "\nEND-OF-MAILBOX"

    class Dumps(Tool):
        name = "mailbox"
        description = "list the mailbox"

        def run(self, **kwargs):
            return dump

    path = tmp_path / "t.json"
    provider = ScriptedProvider(calls_tool("c1", "mailbox"), text("You have mail."))
    session = Session("timeline:x", Harness(provider, tools=[Dumps()]).engine, path=path)

    session.send("check my mail")

    # The model read the whole thing on the turn the tool ran…
    assert ("tool", dump) in provider.snapshots[1]
    # …but what persisted is the capped excerpt: head, an honest marker, tail.
    stored = next(m for m in session.history if m.role == "tool")
    assert len(stored.content) < TOOL_RESULT_CAP
    assert stored.content.startswith("MAILBOX")  # the head survives…
    assert stored.content.endswith("END-OF-MAILBOX")  # …and so does the tail
    assert f"of {len(dump)}" in stored.content  # the marker names the original size
    assert "elided" in stored.content
    # Pairing is intact — the content was edited, never the message dropped (a dangling
    # assistant tool-call would make the next wake's transcript malformed).
    assert stored.tool_call_id == "c1"
    assert any(m.role == "assistant" and m.tool_calls for m in session.history)
    # And it is what reloads from disk, so the next wake pays the capped price, not the raw one.
    reloaded = Session("timeline:x", Harness(ScriptedProvider()).engine, path=path)
    assert next(m for m in reloaded.history if m.role == "tool").content == stored.content


def test_a_normal_tool_result_is_persisted_untouched():
    """The cap is a bound, not a haircut: an ordinary tool answer persists byte for byte."""
    answer = "Dallas, Texas."

    class Small(Tool):
        name = "lookup"
        description = "look something up"

        def run(self, **kwargs):
            return answer

    provider = ScriptedProvider(calls_tool("c1", "lookup"), text("You're in Dallas."))
    session = Session("timeline:x", Harness(provider, tools=[Small()]).engine)

    session.send("where am I?")

    assert next(m for m in session.history if m.role == "tool").content == answer


def test_a_pre_cap_transcript_heals_on_load(tmp_path):
    """A transcript written before the cap existed is bounded the moment it is read.

    Otherwise an agent that ran the old code keeps paying for its old mailbox dumps forever,
    and only a hand-prune on the box could save it.
    """
    path = tmp_path / "t.json"
    bloat = "y" * 50_000
    path.write_text(
        json.dumps(
            [
                {"role": "user", "content": "check my mail"},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "name": "mailbox", "arguments": {}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": bloat},
            ]
        )
    )

    session = Session("timeline:x", Harness(ScriptedProvider()).engine, path=path)

    stored = next(m for m in session.history if m.role == "tool")
    assert len(stored.content) < TOOL_RESULT_CAP
    assert f"of {len(bloat)}" in stored.content
    assert stored.tool_call_id == "c1"  # pairing survives the heal


def test_note_records_a_system_turn_without_calling_the_model(tmp_path):
    """`Session.note` carries an out-of-band fact (a reply that couldn't be delivered)
    into the transcript at zero model cost, and persists it like any turn."""

    class NeverCalled:
        def chat(self, messages, tools=None):
            raise AssertionError("note must not invoke the model")

    path = tmp_path / "transcript.json"
    session = Session("timeline:x", Harness(NeverCalled()).engine, path=path)

    session.note("(Couldn't post that reply to the timeline: it is locked.)")

    assert session.history[-1].role == "system"
    assert "Couldn't post" in session.history[-1].content
    # Persisted: a fresh session over the same path reloads the note.
    reloaded = Session("timeline:x", Harness(NeverCalled()).engine, path=path)
    assert any("Couldn't post" in t.content for t in reloaded.history)
