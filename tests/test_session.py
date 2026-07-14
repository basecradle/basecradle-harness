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
    Engine,
    Harness,
    ImageContent,
    MemoryTool,
    Message,
    Session,
    Tool,
    ToolCall,
    ToolRegistry,
)
from basecradle_harness._context import persisted_step_cap
from basecradle_harness._idempotency import creates, interrupted
from basecradle_harness._session import (
    INTERRUPTED,
    TOOL_ARGS_CAP,
    TOOL_RESULT_CAP,
    _cap_arguments,
    _elide_argument,
    _json_size,
    turn_work,
)

#: A real, well-formed UUIDv7, per the repo's test-data convention.
TIMELINE = "0198e3f1-0000-7000-8000-000000000001"


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


def test_a_crash_mid_save_leaves_the_previous_transcript_intact(tmp_path, monkeypatch):
    """The transcript is published atomically, so a killed wake cannot tear it (issue #297).

    A bare `write_text` truncates and *then* writes. A signal in between — `SIGKILL`, the default
    disposition of `SIGTERM`, the OOM killer, a box reset — leaves half a file, and half a
    transcript is not a degraded transcript: it is invalid JSON. `_load` raises on it, so the wake
    dies on load — and so does every wake after it. The agent is bricked on that timeline, and its
    memory of the conversation is gone, until a human deletes the file.

    Here the crash is simulated where the real one is fatal: after the new bytes are written, before
    they are published. What must survive is the *old* transcript, complete and loadable — never a
    splice of the two.
    """
    agent = Harness(ScriptedProvider(text("The first answer.")), home=tmp_path)
    agent.send("first", source="timeline:t1")
    path = next((tmp_path / "sessions").iterdir())
    before = path.read_text()

    def die(_src, _dst):  # the kill lands between the write and the rename
        raise OSError("killed")

    monkeypatch.setattr("basecradle_harness._session.os.replace", die)

    second = Harness(ScriptedProvider(text("The second answer.")), home=tmp_path)
    with pytest.raises(OSError):
        second.send("second", source="timeline:t1")

    # Not truncated, not spliced, not empty: byte-for-byte the transcript we had before the crash.
    assert path.read_text() == before
    assert [
        m.content
        for m in convo(Harness(ScriptedProvider(), home=tmp_path).transcript("timeline:t1"))
    ] == [
        "first",
        "The first answer.",
    ]
    # …and the staged copy is gone. It held the whole conversation; an exception must not leave it
    # lying in the sessions dir (a *killed* process still can — which is why the sweep knows it).
    assert [p.name for p in (tmp_path / "sessions").iterdir()] == [path.name]


def test_two_sessions_on_one_path_never_share_a_staging_file(tmp_path):
    """The temp is per-`Session`, not per-process — a pid does not identify a writer (issue #297).

    Two `Harness` instances over one home hold two `Session` objects on the *same* transcript path
    in the *same* process. Keyed on the pid alone they would stage into one file and could tear each
    other's temp — which `os.replace` would then publish, which is the exact corruption the atomic
    write exists to prevent.
    """
    home = tmp_path
    one = Harness(ScriptedProvider(text("a")), home=home).session("timeline:t1")
    two = Harness(ScriptedProvider(text("b")), home=home).session("timeline:t1")

    assert one.path == two.path  # same transcript…
    assert one._temp(one.path) != two._temp(two.path)  # …never the same staging file


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


# --- Tool call arguments: sent in full, persisted capped (issue #301) ---------
#
# The half of a call nobody bounded. `_payload` capped a tool *result* and wrote the *arguments*
# whole, so an `assets create` carrying a 200 KB document sat in the transcript forever and was
# replayed to the model on every wake for the life of the timeline — the one class of persisted
# content with no bound at all, against Context Discipline's first invariant.


def _stored_calls(path) -> list[dict]:
    """Every tool call as it actually reached the disk — the only thing a later wake re-reads."""
    return [call for m in json.loads(path.read_text()) for call in m.get("tool_calls", [])]


def test_an_oversized_argument_is_sent_whole_then_persisted_capped(tmp_path):
    """The tool runs with the whole document; what *persists* is head + tail around a marker.

    Exactly the shape the result cap already had, applied to the other half of the call — and the
    live offender from issue #301: an agent that posts a long document paid for it on every wake,
    forever, because the transcript kept the document.
    """
    document = "REPORT\n" + ("x" * 200_000) + "\nEND-OF-REPORT"
    posted: list[str] = []

    class Assets(Tool):
        name = "assets"
        description = "store an asset"

        def run(self, **kwargs):
            posted.append(kwargs["content"])
            return "asset 0198e3f1-0000-7000-8000-000000000001 created"

    path = tmp_path / "t.json"
    provider = ScriptedProvider(
        calls_tool("c1", "assets", action="create", title="Q3 Report", content=document),
        text("Posted the report."),
    )
    session = Session("timeline:x", Harness(provider, tools=[Assets()]).engine, path=path)

    session.send("write up Q3")

    # The tool ran with the document intact: the cap bounds what is *kept*, never what is *sent*.
    assert posted == [document]

    (stored,) = _stored_calls(path)
    assert _json_size(stored["arguments"]) <= TOOL_ARGS_CAP
    # The blob is elided; the small arguments beside it survive whole, so the call is still legible.
    assert stored["arguments"]["action"] == "create"
    assert stored["arguments"]["title"] == "Q3 Report"
    body = stored["arguments"]["content"]
    assert body.startswith("REPORT")  # the head survives…
    assert body.endswith("END-OF-REPORT")  # …and so does the tail
    assert f"elided from {len(document)} chars" in body  # and the marker names what was cut
    assert "elided" in body


def test_ordinary_arguments_persist_byte_for_byte(tmp_path):
    """The cap is a bound, not a haircut. A normal call — the overwhelming majority — is untouched.

    A message body of a few hundred characters is an agent's own speech, and it keeps it verbatim.
    """
    body = "Dallas, Texas. " * 20

    class Messages(Tool):
        name = "messages"
        description = "post a message"

        def run(self, **kwargs):
            return "posted"

    path = tmp_path / "t.json"
    provider = ScriptedProvider(
        calls_tool("c1", "messages", action="create", body=body), text("Said it.")
    )
    session = Session("timeline:x", Harness(provider, tools=[Messages()]).engine, path=path)

    session.send("where am I?")

    (stored,) = _stored_calls(path)
    assert stored["arguments"] == {"action": "create", "body": body}


def test_the_live_call_is_never_edited_only_what_reaches_the_disk(tmp_path):
    """Bounding happens **on the way out**, never by mutating the call the engine is holding.

    This is the guard against the tidy-up that would "simplify" `_payload` by capping `history` in
    place, the way `_cap_tool_results` does. It cannot be done here, and the reason is not style: a
    save lands **mid-turn** (the pre-dispatch write the whole delivery guarantee rests on), and the
    recovery re-issues an interrupted platform create *from these very arguments*. An in-place cap
    would reach into the call about to be dispatched — and into the body a resumed wake is about to
    re-post — and cut it down. The transcript on disk is bounded; the conversation in hand is whole.
    """
    document = "x" * 200_000

    class Assets(Tool):
        name = "assets"
        description = "store an asset"

        def run(self, **kwargs):
            return "stored"

    path = tmp_path / "t.json"
    provider = ScriptedProvider(
        calls_tool("c1", "assets", action="create", content=document), text("Stored.")
    )
    session = Session("timeline:x", Harness(provider, tools=[Assets()]).engine, path=path)

    session.send("store this")

    live = next(m for m in session.history if m.tool_calls).tool_calls[0]
    assert live.arguments["content"] == document, "the cap edited the live call, not the payload"
    assert _json_size(_stored_calls(path)[0]["arguments"]) <= TOOL_ARGS_CAP


def test_a_call_too_big_to_bound_by_eliding_its_values_is_stubbed_and_still_classifiable(tmp_path):
    """The backstop: hundreds of *medium* arguments, none of them individually elidable.

    Rare to the point of pathological — but "rare" is not a bound, and an unbounded fallback would be
    the very defect this cap exists to close, hiding behind a shape nobody expected. `action` survives
    even here, because `create_kind` reads it and the idempotency ordinal must be the same number
    counted off the live transcript and the reloaded one.
    """
    arguments = {"action": "create", **{f"field_{i}": "y" * 100 for i in range(200)}}

    class Assets(Tool):
        name = "assets"
        description = "store an asset"

        def run(self, **kwargs):
            return "stored"

    path = tmp_path / "t.json"
    provider = ScriptedProvider(calls_tool("c1", "assets", **arguments), text("Stored."))
    session = Session("timeline:x", Harness(provider, tools=[Assets()]).engine, path=path)

    session.send("store this")

    (stored,) = _stored_calls(path)
    assert _json_size(stored["arguments"]) <= TOOL_ARGS_CAP
    assert stored["arguments"]["action"] == "create"  # still a create, to anyone counting them
    assert "elided" in json.dumps(stored["arguments"])


def test_cutting_an_argument_never_makes_it_bigger():
    """**A cap that can grow a transcript is not a cap** — the shrink check in `_elide_argument`.

    The marker costs ~120 characters, so "eliding" a 200-character value would *add* to it. Water-
    filling hands every argument a share and cuts the ones that overflow theirs, so it does reach
    values in that band. Comparing the two sizes is what forecloses it, and this tests the guard
    directly rather than through `_cap_arguments`, where the stub backstop can mask the difference.
    """
    for size in (40, 120, 200):  # the band where the marker costs more than it saves
        assert _elide_argument("x" * size, budget=1024) == "x" * size

    excerpt = _elide_argument("x" * 200_000, budget=1024)
    assert _json_size(excerpt) <= 1024
    assert "elided from 200000 chars" in excerpt  # and it says how much is gone

    # A structure has no honest head-and-tail, so it is replaced outright — but only when that shrinks
    # it. A tiny list would grow, and is left alone.
    assert _elide_argument(["z" * 290], budget=512) != ["z" * 290]
    assert _elide_argument([1, 2], budget=512) == [1, 2]


def test_a_call_is_capped_by_the_language_it_is_written_in_never_by_the_script():
    """**A character is a character, whatever script it is written in** (found in review of #301).

    `json.dumps` defaults to `ensure_ascii=True`, which escapes every non-Latin character to a six-
    character `\\uXXXX`. Sizing the cap with it priced one Japanese character at six, so an *ordinary*
    500-character message body blew a 2,048-character cap — and the call collapsed to the stub, which
    keeps nothing but `action`. A peer answered in Japanese lost its own words, its timeline uuid and
    its subject from the transcript, while the identical message in English persisted whole.

    That is not a cost bug. It is an agent that cannot remember what it said because of the language it
    said it in, on a platform whose founding claim is that its peers are equals. The cap bounds
    **context**, and context is billed on the decoded string.
    """
    body = "こんにちは、今日の状況を共有します。" * 28  # ~500 characters: an ordinary post
    japanese = {"action": "create", "timeline": TIMELINE, "body": body}

    assert _cap_arguments(japanese, TOOL_ARGS_CAP) == japanese, (
        "an ordinary Japanese post did not survive the cap"
    )

    # The same post in English is likewise untouched — which is the entire point: same size, same fate.
    english = {
        "action": "create",
        "timeline": TIMELINE,
        "body": "Hello, here is today's status. " * 17,
    }
    assert _cap_arguments(english, TOOL_ARGS_CAP) == english

    # And a Japanese body that *is* genuinely over the cap is excerpted like any other — in Japanese,
    # with its siblings intact — never stubbed away.
    long_form = {"action": "create", "timeline": TIMELINE, "body": "日本語の長い本文です。" * 800}
    capped = _cap_arguments(long_form, TOOL_ARGS_CAP)
    assert _json_size(capped) <= TOOL_ARGS_CAP
    assert capped["action"] == "create" and capped["timeline"] == TIMELINE
    assert capped["body"].startswith("日本語の長い本文です。")


def test_a_call_of_several_medium_arguments_keeps_all_of_them(tmp_path):
    """Water-filling, and why the cap does not simply elide the biggest argument until the call fits.

    An excerpt costs a marker, so an argument can be **too small to be worth eliding and still too big
    to keep** — and a call made of several of them (an ordinary `tasks create` with three 700-character
    fields) could never be brought under the cap at all. It fell through to the stub, and *every*
    argument was lost to save the 159 characters that were over. Giving each argument a fair share of
    the budget has no such cliff: it always makes progress, and it takes the room from where the room is.
    """
    arguments = {
        "action": "create",
        "timeline": TIMELINE,
        "title": "Ship the thing",
        "description": "D" * 700,
        "acceptance": "A" * 700,
        "notes": "N" * 700,
    }

    capped = _cap_arguments(arguments, TOOL_ARGS_CAP)

    assert _json_size(capped) <= TOOL_ARGS_CAP
    assert list(capped) == list(arguments), (
        "an argument was dropped to bring the call under the cap"
    )
    # The short ones are kept byte for byte — their surplus is what pays for the long ones' excerpts.
    assert capped["action"] == "create"
    assert capped["timeline"] == TIMELINE
    assert capped["title"] == "Ship the thing"
    for name in ("description", "acceptance", "notes"):
        assert capped[name].startswith(arguments[name][:50])
        assert "elided from 700 chars" in capped[name]


def test_re_saving_a_capped_transcript_is_a_fixed_point(tmp_path):
    """A transcript is re-saved on every turn for the life of the timeline, so the cap must not ratchet.

    An already-capped call is *under* the cap, so it passes back through untouched — no marker of a
    marker of a marker, each naming a size that is no longer true. The property is the point; the
    mechanism (the early return in `_cap_arguments`) is what delivers it.
    """
    path = tmp_path / "t.json"
    provider = ScriptedProvider(
        calls_tool("c1", "assets", action="create", content="x" * 200_000), text("Stored.")
    )

    class Assets(Tool):
        name = "assets"
        description = "store an asset"

        def run(self, **kwargs):
            return "stored"

    session = Session("timeline:x", Harness(provider, tools=[Assets()]).engine, path=path)
    session.send("store this")
    once = _stored_calls(path)

    reloaded = Session("timeline:x", Harness(ScriptedProvider()).engine, path=path)
    reloaded.note("a later turn re-saves the whole transcript")

    assert _stored_calls(path) == once


# --- A step, not a call, is the unit of the cap (issue #304) ------------------
#
# `max_steps` bounds the model's *calls*, never the tools it dispatched: a model may emit several
# tool calls in one assistant turn, and every model the fleet runs does. A per-*call* cap therefore
# let one step's persisted growth scale with a fan-out nothing bounds — and the compaction
# threshold's safety proof (`worst_case_turn_tokens`, which counts one call per step) understated
# the worst case by exactly that factor, silently and without limit. These pin the fix: the step
# shares one budget, whatever it fans out into.


def call(call_id: str, name: str, **arguments) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def fans_out(*calls: ToolCall) -> Message:
    """One assistant turn carrying several tool calls — the parallel-call shape a real model emits."""
    return Message.assistant(tool_calls=list(calls))


def _stored_steps(path) -> list[int]:
    """What each step's **tool payload** actually cost the transcript, in characters.

    The proof's own quantity, read off the disk: per assistant turn, the arguments of every call it
    made plus the content of every result answering them. This is the term
    `_context.worst_case_turn_tokens` multiplies by the step budget — so what it counts here is what
    that arithmetic must not understate.
    """
    stored = json.loads(path.read_text())
    steps = []
    for index, message in enumerate(stored):
        if not message.get("tool_calls"):
            continue
        ids = {c["id"] for c in message["tool_calls"]}
        payload = sum(_json_size(c["arguments"]) for c in message["tool_calls"])
        for later in stored[index + 1 :]:
            if later.get("role") == "assistant":
                break
            if later.get("role") == "tool" and later.get("tool_call_id") in ids:
                payload += len(later.get("content") or "")
        steps.append(payload)
    return steps


def test_a_step_that_fans_out_shares_one_result_budget(tmp_path):
    """Three parallel calls, three mailbox dumps — and **one** `TOOL_RESULT_CAP` between them.

    The defect this closes: each call used to get its own 4 KB, so a step's persisted growth was
    `fan-out x cap` and the compaction proof — which counts one call per step — was quietly untrue.
    The model still reads all three dumps in full on the turn they ran; what every *future* wake
    re-reads is bounded by the step, not by how wide the model chose to fan out.
    """
    dumps = {box: f"BOX-{box}\n" + ("x" * 60_000) + f"\nEND-{box}" for box in ("a", "b", "c")}

    class Mailboxes(Tool):
        name = "mailbox"
        description = "list a mailbox"

        def run(self, **kwargs):
            return dumps[kwargs["box"]]

    path = tmp_path / "t.json"
    provider = ScriptedProvider(
        fans_out(
            call("c1", "mailbox", box="a"),
            call("c2", "mailbox", box="b"),
            call("c3", "mailbox", box="c"),
        ),
        text("You have mail in all three."),
    )
    session = Session("timeline:x", Harness(provider, tools=[Mailboxes()]).engine, path=path)

    session.send("check all my mail")

    # Sent whole: the model read every one of the three dumps in full on the turn they ran.
    for dump in dumps.values():
        assert ("tool", dump) in provider.snapshots[1]

    # Kept capped — and capped **together**. Per-call, this step would have persisted three times
    # the budget; the whole point is that it persists one.
    results = [m for m in session.history if m.role == "tool"]
    assert len(results) == 3
    assert sum(len(m.content or "") for m in results) <= TOOL_RESULT_CAP

    # Each one is still worth reading: its own head, its own tail, its own honest marker. A cap that
    # degrades to a shrug would be cheaper and useless.
    for (box, dump), stored in zip(dumps.items(), results):
        assert stored.content.startswith(f"BOX-{box}")
        assert stored.content.endswith(f"END-{box}")
        assert f"of {len(dump)}" in stored.content
    assert [m.tool_call_id for m in results] == ["c1", "c2", "c3"]  # pairing intact


def test_a_wide_fan_out_of_small_results_keeps_every_one_of_them_whole(tmp_path):
    """Fan-out alone costs nothing — **only fan-out that is also fat pays.**

    This is why the shared budget is water-filled rather than sliced evenly. Ten parallel lookups
    returning a line each fit the budget between them, and the smallest of any remaining set is never
    larger than their mean — which *is* its share — so every one of them is kept byte for byte. An
    even 1/10th slice would have taken a haircut off ten results that were never the problem.
    """
    answer = "Dallas, Texas."

    class Lookup(Tool):
        name = "lookup"
        description = "look something up"

        def run(self, **kwargs):
            return answer

    path = tmp_path / "t.json"
    provider = ScriptedProvider(
        fans_out(*(call(f"c{n}", "lookup", of=str(n)) for n in range(10))),
        text("Ten answers."),
    )
    session = Session("timeline:x", Harness(provider, tools=[Lookup()]).engine, path=path)

    session.send("look up ten things")

    results = [m for m in session.history if m.role == "tool"]
    assert len(results) == 10
    assert all(m.content == answer for m in results), "a small result paid for a wide fan-out"


def test_a_step_that_fans_out_shares_one_argument_budget_and_still_reads_as_creates(tmp_path):
    """The other half of the step's payload — and the trap that makes it dangerous to cap.

    Three parallel `assets create` calls, each carrying a document. They share one `TOOL_ARGS_CAP`,
    so the step is bounded. But **capping may never change what `create_kind` reads off a call**: the
    idempotency ordinal is "the nth create of this kind in this turn", counted once by the live mint
    and again by the recovery over the *reloaded* transcript. Cap an `action` away and a create
    vanishes from the recovery's count, the ordinal drifts, and a key the platform has never seen is
    a message posted twice. Water-filling keeps the short arguments whole, which is what makes the
    two counts agree — so this asserts the bound *and* the thing the bound must not break.
    """
    document = "REPORT\n" + ("x" * 200_000) + "\nEND-OF-REPORT"

    class Assets(Tool):
        name = "assets"
        description = "store an asset"

        def run(self, **kwargs):
            return "stored"

    path = tmp_path / "t.json"
    provider = ScriptedProvider(
        fans_out(
            *(
                call(f"c{n}", "assets", action="create", title=f"Part {n}", content=document)
                for n in range(3)
            )
        ),
        text("Stored all three."),
    )
    session = Session("timeline:x", Harness(provider, tools=[Assets()]).engine, path=path)

    session.send("store these")

    stored = _stored_calls(path)
    assert len(stored) == 3
    assert sum(_json_size(c["arguments"]) for c in stored) <= TOOL_ARGS_CAP

    # …and every one of them is still, unmistakably, a create — which is what the recovery counts.
    reloaded = Session("timeline:x", Harness(ScriptedProvider()).engine, path=path)
    work = turn_work(reloaded.history, reloaded.history[0])
    assert [(c.kind, c.ordinal) for c in creates(work)] == [
        ("asset", 1),
        ("asset", 2),
        ("asset", 3),
    ]


class Wide(Tool):
    """Takes a lot and gives a lot back — one call of it would swamp a transcript on its own."""

    name = "wide"
    description = "take a lot and give a lot back"

    def __init__(self, chars: int = 100_000) -> None:
        self.chars = chars

    def run(self, **kwargs):
        return "RESULT\n" + ("z" * self.chars)


def _fan_out_step(path, fan_out: int, *, chars: int = 100_000) -> int:
    """Run one step of `fan_out` fat calls and report what its tool payload cost the transcript."""
    provider = ScriptedProvider(
        fans_out(*(call(f"c{n}", "wide", body="y" * chars) for n in range(fan_out))),
        text("Done."),
    )
    session = Session("timeline:x", Harness(provider, tools=[Wide(chars)]).engine, path=path)
    session.send("do a lot at once")
    (step,) = _stored_steps(path)
    return step


def test_no_fan_out_can_make_a_step_persist_more_than_the_proof_allows(tmp_path):
    """**The invariant, stated as the compaction proof needs it** (issue #304).

    `worst_case_turn_tokens` = `persisted_step_cap() x max_steps`, and that is a real upper bound only
    if a *step* — not a call — is what the cap governs. So: whatever the model fans out into, one
    step's persisted tool payload stays inside `persisted_step_cap()`. Widen the fan-out and the
    excerpts get thinner; the total does not move. Per-call, a 30-way step would have persisted thirty
    times the budget, and the proof would have understated it by thirty.
    """
    for fan_out in (1, 2, 5, 10, 20, 30):
        step = _fan_out_step(tmp_path / f"t{fan_out}.json", fan_out)
        assert step <= persisted_step_cap(), (
            f"a step that fanned out {fan_out} ways persisted {step} chars, over the "
            f"{persisted_step_cap()} the compaction proof is built on"
        )


def test_a_steps_growth_is_bounded_by_what_the_model_wrote_never_by_what_its_tools_returned(
    tmp_path,
):
    """The bound underneath the bound — and the one that holds at **every** fan-out, without exception.

    Above ~50 parallel calls the total creeps past `persisted_step_cap()`, and that is not a leak: it
    is the floor (`_gone`). A result cannot be dropped (its call would dangle, permanently) and neither
    can a call's arguments (`create_kind` reads them), so each call keeps one short record saying how
    much is gone — of the same order as the `id`+`name` envelope the transcript must keep for that call
    anyway. **That residue scales with what the model emitted, never with what its tools returned**,
    which is the thing "nothing replayed per wake may be unbounded" actually asks for: the harness
    bounds the class of content that can be arbitrarily large *independently of what the model wrote* —
    a three-token call can return a 200 KB mailbox — and the model's own emission is the provider's to
    bound, at every response's max-output-tokens.

    So: multiply the tools' output fiftyfold and the transcript does not move.
    """
    modest = _fan_out_step(tmp_path / "modest.json", 20, chars=100_000)
    enormous = _fan_out_step(tmp_path / "enormous.json", 20, chars=5_000_000)

    assert modest <= persisted_step_cap()
    assert abs(enormous - modest) < 100, (
        f"50x the tool output moved the transcript by {enormous - modest} chars: a step's persisted "
        f"size is scaling with what its tools returned"
    )


def test_a_killed_wide_step_keeps_the_evidence_the_recovery_reads(tmp_path):
    """**The shared budget must never cut the `INTERRUPTED` marker** — found reviewing this diff.

    It is a *sentinel*, matched exactly: `_replayable` keeps an interrupted create's arguments whole
    because of it, and `_idempotency.interrupted` finds the calls to re-issue by it. Per call it was
    never at risk (~300 characters against a 4 KB cap). Share that cap across a step and a fan-out of
    ~14 drives every share below it — at which point the marker is elided, and the damage is silent
    and severe: the create's arguments are capped and the recovery re-posts the peer's message with
    its body cut out, and the call is no longer even recognizable as one to re-issue.

    So an interrupted result is outside the pool — never charged, never elided. The mirror of the
    exception `_replayable` already makes on the arguments side, for the same reason: **the recovery's
    evidence is never bounded away.**
    """
    body = "The full text of the message the peer is owed. " * 30
    path = tmp_path / "t.json"
    # A wake killed mid-chain: 20 creates issued, none recorded — exactly what `_load` heals.
    path.write_text(
        json.dumps(
            [
                {"role": "user", "content": "post these"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"c{n}",
                            "name": "messages",
                            "arguments": {"action": "create", "timeline": TIMELINE, "body": body},
                        }
                        for n in range(20)
                    ],
                },
            ]
        )
    )

    session = Session("timeline:x", Harness(ScriptedProvider()).engine, path=path)
    session.note("a later turn re-saves the whole transcript")

    healed = [m for m in session.history if m.role == "tool"]
    assert len(healed) == 20
    assert all(m.content == INTERRUPTED for m in healed), "the cap ate the recovery's own sentinel"

    # …and on disk, where the recovering wake will actually read it.
    reloaded = Session("timeline:x", Harness(ScriptedProvider()).engine, path=path)
    work = turn_work(reloaded.history, reloaded.history[0])
    pending = interrupted(work, INTERRUPTED)
    assert [(c.kind, c.ordinal) for c in pending] == [("message", n + 1) for n in range(20)]
    # The whole point of finding them: each is re-issued from arguments that still carry the body.
    assert all(c.call.arguments["body"] == body for c in pending), (
        "an interrupted create's body was elided — the recovery would re-post it truncated"
    )


def test_re_saving_a_fanned_out_step_is_a_fixed_point(tmp_path):
    """The transcript is re-saved on every turn for the life of the timeline, so the cap must settle.

    A shared budget is where a ratchet would hide: cap a step's results *together*, and the capped
    set is what the next save measures. Water-filling is what forecloses it — a set that already fits
    its budget is kept whole, every item of it — so the second save writes the same bytes as the
    first, and there is never a marker of a marker naming a size that is no longer true.
    """

    class Mailboxes(Tool):
        name = "mailbox"
        description = "list a mailbox"

        def run(self, **kwargs):
            return "BOX\n" + ("x" * 60_000)

    path = tmp_path / "t.json"
    provider = ScriptedProvider(
        fans_out(*(call(f"c{n}", "mailbox", box=str(n)) for n in range(4))), text("Mail.")
    )
    session = Session("timeline:x", Harness(provider, tools=[Mailboxes()]).engine, path=path)
    session.send("check my mail")
    once = path.read_text()

    reloaded = Session("timeline:x", Harness(ScriptedProvider()).engine, path=path)
    reloaded.note("a later turn re-saves the whole transcript")

    twice = json.loads(path.read_text())
    assert twice[:-1] == json.loads(once), "the cap ground the step down a second time"


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


# --- issue #297: the turn persists as it runs, and a killed one still loads ---


def test_the_rolled_back_build_is_removed_from_the_file_not_just_from_memory(tmp_path):
    """`rollback` rewrites the transcript. Incremental persistence changed the premise (#297).

    The staleness guard discards a stale (tool-free, so speechless) build and regenerates. That used
    to be a purely in-memory `del`, because nothing had been written yet. Now the build being thrown
    away is **already on disk**, so an in-memory-only rollback would leave it there — and a wake
    killed before the replacement build persisted would come back to a transcript carrying a turn
    that was deliberately unmade.
    """
    provider = ScriptedProvider(Message.assistant(content="a stale answer"))
    session = Session("timeline:t1", Engine(provider, ToolRegistry()), path=tmp_path / "s.json")
    base = len(session.history)

    session.send("what do you think?")
    assert any(
        t["content"] == "a stale answer" for t in json.loads((tmp_path / "s.json").read_text())
    )

    session.rollback(base)

    on_disk = json.loads((tmp_path / "s.json").read_text())
    assert on_disk == [], "the unmade build is still on disk; a killed wake would read it as real"
    assert session.history == []


def test_turn_work_does_not_stop_at_an_injected_turn(tmp_path):
    """A turn's *work* runs past the turns the engine injects into it (issue #297).

    The engine appends a `user`-role turn to show the model an image; the code bridge appends one
    naming the Assets a run produced. Neither is a new turn of the conversation. Stopping at one
    would cut the turn in half — the narration behind it would vanish, and a turn that **finished**
    would read as one that was **interrupted**, which is a re-run of everything it already did.
    """
    turn = Message(role="user", content="look at this", items=["m-1"])
    history = [
        Message.system("charter"),
        turn,
        Message.assistant(tool_calls=[ToolCall(id="c1", name="assets", arguments={})]),
        Message.tool(tool_call_id="c1", content="here"),
        Message(role="user", content="(Showing image: owl.png)", injected=True),
        Message.assistant(content="A barn owl."),
        Message(
            role="user", content="and this?", items=["m-2"]
        ),  # the next real turn: the boundary
        Message.assistant(content="A tawny."),
    ]

    work = turn_work(history, turn)

    assert [m.role for m in work] == ["assistant", "tool", "user", "assistant"]
    assert work[-1].content == "A barn owl."  # the narration is inside the turn, not behind it


def test_the_turn_is_located_by_identity_so_a_compaction_cannot_misplace_it(tmp_path):
    """Two turns with identical text are equal but not the same turn.

    `list.index` compares with `==`, and `Message` is a dataclass — so a peer who asks the same
    question twice would have the *first* turn found for the second one. Identity is also what
    survives a compaction, which rewrites the list by moving objects rather than copying them.
    """
    # A transcript written before #297 carries no uuids, so two identical questions really are
    # `==` — which is exactly when an equality-based lookup silently returns the wrong turn.
    first = Message(role="user", content="are you there?")
    second = Message(role="user", content="are you there?")
    history = [first, Message.assistant(content="yes"), second, Message.assistant(content="still")]

    assert first == second  # equal...
    assert turn_work(history, second) == [history[-1]]  # ...but the right one is found anyway
    assert turn_work(history, first) == [history[1]]
