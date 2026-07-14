"""The context budget: the transcript is bounded, so a standing agent never hits the wall.

Issue #276. These pin the four things that make compaction safe rather than merely clever:

- the **trigger** is the provider's own reported usage, never a client-side count;
- the **limit** resolves env → adapter capability → conservative floor, never a static table;
- a **cut never strands a tool result from its call** (the load-bearing invariant: a dangling
  `tool_call_id` is malformed *forever*, breaking every later wake — worse than the bloat);
- an agent **already past the ceiling self-heals**, instead of failing identically on every wake.

No model and no network: a scripted provider replays prepared turns and reports whatever usage
the test tells it to.
"""

from __future__ import annotations

import logging

import pytest

from basecradle_harness import (
    Engine,
    Harness,
    Message,
    Policy,
    Session,
    Tool,
    ToolCall,
    ToolRegistry,
)
from basecradle_harness._context import (
    COMPACT_AT,
    DEFAULT_CONTEXT_LIMIT,
    TOOL_ARGS_CAP,
    TOOL_RESULT_CAP,
    WORST_CASE_CHARS_PER_TOKEN,
    Compactor,
    ContextBudget,
    is_context_overflow,
    max_safe_steps,
    min_safe_limit,
    persisted_step_cap,
    worst_case_turn_tokens,
)
from basecradle_harness._engine import DEFAULT_MAX_STEPS
from basecradle_harness._exceptions import ProviderContextLengthError

# The fabricated cast, per repo convention.
JOHN = "John Doe"
NOVA = "Nova Digital"


class ScriptedProvider:
    """A `Provider` that replays prepared replies and reports the usage the test dictates.

    `last_tokens_in` is the capability the budget triggers on — here it is set explicitly per call
    (from `usage`), exactly as a real adapter sets it from what the endpoint reported.
    """

    def __init__(self, *replies: Message, usage: list[int | None] | None = None) -> None:
        self._replies = list(replies)
        self._usage = list(usage or [])
        self.last_tokens_in: int | None = None
        self.calls: list[list[Message]] = []
        #: The images in front of the model *at chat time*, per call. `calls` holds the live
        #: `Message` objects, which the session mutates afterwards (it evicts the pixels once the
        #: model has answered), so only a snapshot can say what the model actually saw.
        self.shown: list[list[object]] = []
        self.limit_calls = 0
        self.limit: int | None = None

    def chat(self, messages, tools=None):
        self.calls.append(list(messages))
        self.shown.append([image for m in messages for image in m.images])
        if self._usage:
            self.last_tokens_in = self._usage.pop(0)
        if not self._replies:
            raise AssertionError("ScriptedProvider ran out of replies")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def context_limit(self) -> int | None:
        self.limit_calls += 1
        return self.limit


def budget(provider, *, override=None, max_steps=DEFAULT_MAX_STEPS) -> ContextBudget:
    return ContextBudget(provider, override=override, max_steps=max_steps)


def compactor(provider, *, override=None, on_summary=None) -> Compactor:
    return Compactor(provider, budget(provider, override=override), on_summary=on_summary)


def conversation(exchanges: int, *, chars: int = 400) -> list[Message]:
    """A plain user/assistant transcript — no tool calls, `exchanges` turns each way."""
    history: list[Message] = []
    for i in range(exchanges):
        history.append(Message.user(f"[{i}] {JOHN} asks something. " + "x" * chars))
        history.append(Message.assistant(content=f"[{i}] {NOVA} answers. " + "y" * chars))
    return history


def tool_exchange(index: int, *, chars: int = 400) -> list[Message]:
    """One exchange that *drives a tool*: user → assistant(tool_call) → tool result → assistant."""
    call = ToolCall(id=f"call_{index}", name="memory", arguments={"action": "write"})
    return [
        Message.user(f"[{index}] {JOHN}: do the thing. " + "x" * chars),
        Message.assistant(tool_calls=[call]),
        Message.tool(tool_call_id=call.id, content="ok. " + "z" * chars),
        Message.assistant(content=f"[{index}] done. " + "y" * chars),
    ]


def dangling(history: list[Message]) -> list[str]:
    """Every `tool` message in `history` whose assistant tool-call is *not* also in `history`.

    The one thing a compaction may never produce. A provider rejects a transcript whose tool result
    answers a call it cannot see — and because the transcript persists, it would fail that way on
    every wake from then on, forever.
    """
    called = {call.id for m in history for call in m.tool_calls}
    return [m.tool_call_id for m in history if m.role == "tool" and m.tool_call_id not in called]


# --- the limit: env → adapter → floor, never a table -------------------------


def test_limit_prefers_the_operator_override_over_everything():
    provider = ScriptedProvider()
    provider.limit = 1_000_000

    resolved = budget(provider, override=200_000).limit()

    assert (resolved.tokens, resolved.source) == (200_000, "env")
    assert provider.limit_calls == 0  # the operator answered; the adapter is never even asked


def test_limit_falls_to_the_adapter_capability_when_no_override():
    provider = ScriptedProvider()
    provider.limit = 1_048_576

    resolved = budget(provider).limit()

    assert (resolved.tokens, resolved.source) == (1_048_576, "adapter")


def test_limit_falls_to_the_conservative_floor_when_the_adapter_cannot_answer():
    provider = ScriptedProvider()
    provider.limit = None  # an OpenAI-direct adapter: the models API states no context window

    resolved = budget(provider).limit()

    assert (resolved.tokens, resolved.source) == (DEFAULT_CONTEXT_LIMIT, "default")


def test_limit_is_resolved_once_and_cached():
    provider = ScriptedProvider()
    provider.limit = 400_000
    live = budget(provider)

    live.limit()
    live.limit()
    live.limit()

    assert provider.limit_calls == 1  # a metadata call per turn would be a per-wake network tax


def test_a_raising_adapter_degrades_to_the_floor_rather_than_breaking_the_wake():
    class Broken(ScriptedProvider):
        def context_limit(self):
            raise RuntimeError("the models endpoint is down")

    resolved = budget(Broken()).limit()

    assert (resolved.tokens, resolved.source) == (DEFAULT_CONTEXT_LIMIT, "default")


def test_an_adapter_with_no_capability_at_all_still_works():
    class Minimal:
        """The whole `Provider` protocol: one `chat`. No capabilities."""

        def chat(self, messages, tools=None):
            return Message.assistant(content="hi")

    assert budget(Minimal()).limit().source == "default"


# --- the trigger: the provider's own count, and nothing paid for when quiet ---


def test_compaction_triggers_above_half_the_ceiling():
    provider = ScriptedProvider()
    provider.limit = 200_000
    live = budget(provider)

    assert live.should_compact(int(200_000 * COMPACT_AT) - 1) is False
    assert live.should_compact(int(200_000 * COMPACT_AT) + 1) is True


def test_a_quiet_agent_never_pays_for_the_adapter_lookup():
    provider = ScriptedProvider()
    provider.limit = 1_048_576
    live = budget(provider)

    # A call well under half the *floor* cannot have crossed half of any ceiling at or above it,
    # so the question needs no answer — and the adapter's live metadata call is never made.
    assert live.should_compact(10_000) is False
    assert provider.limit_calls == 0


def test_no_usage_reported_means_no_compaction():
    provider = ScriptedProvider()

    assert budget(provider).should_compact(None) is False


def test_zero_disables_compaction_outright():
    provider = ScriptedProvider()
    provider.limit = 1_000

    live = budget(provider, override=0)

    assert live.enabled is False
    assert live.should_compact(999_999) is False


# --- the rewrite -------------------------------------------------------------


def test_compaction_replaces_the_old_region_with_one_summary_and_keeps_the_recent_tail():
    provider = ScriptedProvider(Message.assistant(content="SUMMARY: they discussed the thing."))
    provider.last_tokens_in = 100_000  # over half of the 128k floor
    history = conversation(20)
    newest = history[-1]

    assert compactor(provider).maybe_compact(history) is True

    assert history[0].role == "system"
    assert "SUMMARY: they discussed the thing." in history[0].content
    assert "compacted" in history[0].content  # labelled, so the model knows what it is reading
    assert history[-1] is newest  # the recent window survives verbatim, object-identical
    assert len(history) < 40


def test_the_charter_survives_compaction():
    provider = ScriptedProvider(Message.assistant(content="SUMMARY"))
    provider.last_tokens_in = 100_000
    charter = Message.system("You are Nova Digital, an AI peer on BaseCradle.")
    history = [charter, *conversation(20)]

    compactor(provider).maybe_compact(history)

    # The charter is standing context, not conversation: it is never summarized away.
    assert history[0] is charter
    assert history[1].role == "system" and "SUMMARY" in history[1].content


def test_a_cut_never_strands_a_tool_result_from_its_call():
    """The load-bearing invariant, across every cut point the retention budget could pick.

    A dangling `tool_call_id` is not a bad turn — it is a *permanently malformed transcript*, and
    the wake that replays it fails, then fails again, forever. So this sweeps the whole space of
    retention sizes over a tool-heavy transcript and asserts the property holds at every one.
    """
    history_template = [m for i in range(12) for m in tool_exchange(i)]

    for limit in range(2_000, 200_000, 2_000):
        provider = ScriptedProvider(Message.assistant(content="SUMMARY"))
        provider.last_tokens_in = limit  # whatever the ceiling, the cut must be well-formed
        history = [Message.from_dict(m.to_dict()) for m in history_template]

        compactor(provider, override=limit).maybe_compact(history)

        assert dangling(history) == [], f"orphaned tool result at limit={limit}"
        # And the head is always the summary or the charter — never a tool result.
        assert history[0].role in ("system", "user")


def test_the_summary_is_asked_for_the_work_not_just_the_words():
    """Requirement 7: tool-driven work must survive the turns that carried it."""
    provider = ScriptedProvider(Message.assistant(content="SUMMARY"))
    provider.last_tokens_in = 100_000
    history = [m for i in range(10) for m in tool_exchange(i)]

    compactor(provider).maybe_compact(history)

    instruction, excerpt = provider.calls[0]
    assert "WORK DONE" in instruction.content
    assert "no trace that it ever happened" in instruction.content
    # The excerpt names the tools that ran, so the summary can record what was actually done —
    # an assistant turn that only *called* a tool says nothing on its own.
    assert "called: memory" in excerpt.content


def test_the_summary_is_written_to_durable_memory():
    written: list[str] = []
    provider = ScriptedProvider(Message.assistant(content="SUMMARY: I posted the report."))
    provider.last_tokens_in = 100_000
    history = conversation(20)

    compactor(provider, on_summary=written.append).maybe_compact(history)

    assert written == ["SUMMARY: I posted the report."]


def test_a_memory_failure_never_blocks_the_compaction():
    def explode(summary):
        raise RuntimeError("the palace is on fire")

    provider = ScriptedProvider(Message.assistant(content="SUMMARY"))
    provider.last_tokens_in = 100_000
    history = conversation(20)

    assert compactor(provider, on_summary=explode).maybe_compact(history) is True
    assert "SUMMARY" in history[0].content  # memory is best-effort; the transcript bound is not


def test_compaction_is_cumulative_the_previous_summary_folds_into_the_next():
    provider = ScriptedProvider(
        Message.assistant(content="FIRST SUMMARY"),
        Message.assistant(content="SECOND SUMMARY"),
    )
    provider.last_tokens_in = 100_000
    history = conversation(20)
    live = compactor(provider)

    live.maybe_compact(history)
    history.extend(conversation(20))
    live.maybe_compact(history)

    # The second summarize call was *shown* the first summary (it sat in the dropped region), so
    # nothing is orphaned by a chain of compactions — each one folds its predecessor in.
    assert "FIRST SUMMARY" in provider.calls[1][1].content
    assert len([m for m in history if m.role == "system" and "compacted" in (m.content or "")]) == 1
    assert "SECOND SUMMARY" in history[0].content


def test_a_failed_summarization_leaves_the_transcript_untouched(caplog):
    provider = ScriptedProvider(RuntimeError("the model is down"))
    provider.last_tokens_in = 100_000
    history = conversation(20)
    before = [m.content for m in history]

    with caplog.at_level(logging.WARNING):
        assert compactor(provider).maybe_compact(history) is False

    assert [m.content for m in history] == before
    assert "summarization call errored" in caplog.text


def test_compaction_declines_rather_than_produce_a_transcript_it_cannot_prove_safe(caplog):
    """No safe cut → decline. A malformed transcript is worse than a long one."""
    provider = ScriptedProvider(Message.assistant(content="SUMMARY"))
    provider.last_tokens_in = 100_000
    # One user turn and an unbroken tool chain after it: there is no *second* user turn to cut at,
    # so no cut can be made without stranding a tool result.
    call = ToolCall(id="call_1", name="memory", arguments={})
    history = [
        Message.user("go " + "x" * 5_000),
        Message.assistant(tool_calls=[call]),
        Message.tool(tool_call_id=call.id, content="y" * 5_000),
    ]

    with caplog.at_level(logging.WARNING):
        assert compactor(provider).maybe_compact(history) is False

    assert dangling(history) == []
    assert "no safe cut point" in caplog.text


def test_the_compaction_log_line_names_the_numbers(caplog):
    provider = ScriptedProvider(Message.assistant(content="SUMMARY"))
    provider.last_tokens_in = 100_000
    history = conversation(20)

    with caplog.at_level(logging.INFO):
        compactor(provider).maybe_compact(history)

    line = next(r.message for r in caplog.records if r.message.startswith("context compact"))
    assert "tokens_in=100000" in line
    assert f"limit={DEFAULT_CONTEXT_LIMIT}" in line
    assert "source=default" in line
    assert "messages=" in line and "→" in line


# --- the session: where the trigger is actually read -------------------------


def session_for(provider, tmp_path, *, override=None, source="timeline:t1"):
    engine = Engine(provider, ToolRegistry(policy=Policy.locked()))
    return Session(
        source,
        engine,
        path=tmp_path / f"{source}.json",
        compactor=compactor(provider, override=override),
    )


def test_a_settled_turn_compacts_when_the_provider_says_the_call_was_too_big(tmp_path):
    provider = ScriptedProvider(
        Message.assistant(content="my reply"),
        Message.assistant(content="SUMMARY"),
        usage=[100_000],  # what the endpoint reported for the reply call
    )
    session = session_for(provider, tmp_path)
    session.history.extend(conversation(20))

    session.send("one more thing?")

    assert session.history[0].role == "system" and "SUMMARY" in session.history[0].content
    assert len(session.history) < 40
    # And it persisted — the compacted transcript on disk *is* the record of the decision, which is
    # why no sidecar usage file has to survive the process.
    assert "SUMMARY" in session.path.read_text()


def test_a_small_turn_leaves_the_transcript_alone(tmp_path):
    provider = ScriptedProvider(Message.assistant(content="my reply"), usage=[5_000])
    session = session_for(provider, tmp_path)
    session.history.extend(conversation(5))
    before = len(session.history)

    session.send("hello")

    # The turn, the engine's step-counter note, and the reply. Nothing summarized.
    assert len(session.history) == before + 3
    assert not any("compacted" in (m.content or "") for m in session.history)


def test_a_session_without_a_compactor_is_unchanged(tmp_path):
    provider = ScriptedProvider(Message.assistant(content="my reply"), usage=[900_000])
    engine = Engine(provider, ToolRegistry(policy=Policy.locked()))
    session = Session("timeline:t1", engine, path=tmp_path / "t.json")
    session.history.extend(conversation(20))

    session.send("hello")

    # The pre-#276 behavior: a huge reported usage and the transcript just grows anyway.
    assert len(session.history) == 43
    assert not any("compacted" in (m.content or "") for m in session.history)


def test_one_shared_provider_never_attributes_one_sessions_usage_to_anothers_transcript(tmp_path):
    """The capital's note: `last_tokens_in` lives on the *provider*, which sessions share.

    It is read the moment a session's own run returns, on one thread, so the last call is always
    that session's own. This pins the assumption — if a future change ever pools or parallelizes
    providers, this test is what fails.
    """
    provider = ScriptedProvider(
        Message.assistant(content="big-channel reply"),
        Message.assistant(content="SUMMARY"),
        Message.assistant(content="small-channel reply"),
        usage=[100_000, None, 900],  # the summarize call reports nothing; the small channel, little
    )
    harness = Harness(provider, home=tmp_path, compactor=compactor(provider))
    big = harness.session("timeline:busy")
    small = harness.session("github:pr-1")
    big.history.extend(conversation(20))
    small.history.extend(conversation(3))

    big.send("one more thing?")  # 100k reported → this session compacts
    small.send("hi")  # 900 reported → this one must not

    assert "SUMMARY" in big.history[0].content
    assert not any("compacted" in (m.content or "") for m in small.history)


# --- the wall: an agent already past the ceiling self-heals -------------------


def test_an_over_length_400_compacts_and_retries_the_turn_once(tmp_path):
    """The founding defect: without this, every later wake rebuilds the same doomed request."""
    provider = ScriptedProvider(
        ProviderContextLengthError("maximum context length exceeded", status_code=400),
        Message.assistant(content="SUMMARY of the long conversation"),
        Message.assistant(content="the reply, against a transcript that now fits"),
        usage=[None, None, 40_000],
    )
    session = session_for(provider, tmp_path)
    session.history.extend([m for i in range(12) for m in tool_exchange(i)])

    reply = session.send("are you still there?")

    assert reply == "the reply, against a transcript that now fits"
    assert "SUMMARY of the long conversation" in session.history[0].content
    assert dangling(session.history) == []
    # The failed attempt left no residue: the retry ran against a clean transcript, and the turn
    # appears once, not twice.
    assert sum(1 for m in session.history if m.content == "are you still there?") == 1
    assert not any("[turn failed" in (m.content or "") for m in session.history)


def _fails_after(n: int, monkeypatch):
    """Let the first `n` saves through, then make every later one hit a full disk.

    The turn now persists in more than one place (issue #297), and *which* save fails decides which
    guard is under test: the one before the model call (the turn must die), or one after it (the
    turn's own exception must survive).
    """
    real = open
    saves = {"count": 0}

    def maybe_full_disk(*args, **kwargs):
        saves["count"] += 1
        if saves["count"] > n:
            raise OSError(28, "No space left on device")
        return real(*args, **kwargs)

    monkeypatch.setattr("basecradle_harness._session.open", maybe_full_disk, raising=False)


def test_a_failing_save_never_masks_the_overflow_the_self_heal_is_keyed_on(tmp_path, monkeypatch):
    """A save that fails in the `finally` must not eat the turn's own exception (issue #297).

    `_drive` persists in a `finally`, and an exception raised in a `finally` **replaces** the one
    propagating through it. The one propagating through this one is load-bearing: `send` catches
    `ProviderContextLengthError` to compact and retry — the self-heal that keeps an agent at its
    context ceiling from failing identically on every wake, forever. Masked, it becomes an `OSError`
    that no one catches, and the transcript stays over the wall.

    The atomic write is what makes this reachable, which is why the guard ships with it: staging a
    temp needs the old transcript and the new one on disk at once, and `fsync` is where a filesystem
    reports the deferred write errors a buffered close used to swallow. **A near-full box is
    precisely where an over-long transcript turns up.**

    What the guard promises is exactly this and no more: the turn's own exception is what comes out.
    (It does not promise the agent survives a full disk — nothing can persist on one.)
    """
    provider = ScriptedProvider(
        ProviderContextLengthError("maximum context length exceeded", status_code=400),
        usage=[None],
    )
    engine = Engine(provider, ToolRegistry(policy=Policy.locked()))
    session = Session("timeline:t1", engine, path=tmp_path / "s.json")

    # The disk fills *after* the user turn is safely down — so the model runs, overflows, and the
    # save that fails is the one in the `finally`. That is the window the guard is for.
    _fails_after(1, monkeypatch)

    # No compactor on this session, so the overflow is re-raised rather than healed — which is what
    # lets the test see *which* exception survived the `finally`. It must be the model's, not the
    # disk's: an OSError here is the self-heal's trigger, eaten.
    with pytest.raises(ProviderContextLengthError):
        session.send("are you still there?")


def test_a_turn_that_cannot_record_the_message_never_calls_the_model(tmp_path, monkeypatch):
    """A save that fails *before* the model call takes the turn down with it (issue #297).

    This is the other side of the guard above, and the asymmetry is deliberate. A save that fails
    mid-turn is stood down, because the turn may already have posted and killing it would undo
    nothing. A save that fails **before** the model has been called is the opposite case in every
    respect: nothing has run, nothing has posted, and proceeding would mean engaging the model on a
    peer's message with **no durable record that we ever read it** — the precise state the whole
    recovery design exists to make impossible (a wake killed after posting would then look, to its
    successor, exactly like a wake that never started, and the reply would be sent twice).

    So it fails, loudly and early. The claim on the message stays in-flight and the high-water mark
    stays behind it, so the next wake re-drives it cleanly. An agent that cannot write down what it
    was asked is an agent that must not answer.
    """
    provider = ScriptedProvider(Message.assistant(content="Hello, John."), usage=[None])
    engine = Engine(provider, ToolRegistry(policy=Policy.locked()))
    session = Session("timeline:t1", engine, path=tmp_path / "s.json")

    _fails_after(0, monkeypatch)  # the very first save — the user turn — hits the wall

    with pytest.raises(OSError):
        session.send("are you still there?")

    assert provider.calls == [], "the model was engaged on a message we could not record"


def test_the_retry_still_shows_the_model_the_image_the_peer_posted(tmp_path):
    from basecradle_harness import ImageContent

    provider = ScriptedProvider(
        ProviderContextLengthError("prompt is too long", status_code=400),
        Message.assistant(content="SUMMARY"),
        Message.assistant(content="I see the picture."),
        usage=[None, None, 1_000],
    )
    session = session_for(provider, tmp_path)
    session.history.extend(conversation(20))
    picture = ImageContent(url="data:image/png;base64,iVBORw0KGgo=", alt="a chart")

    session.send("what do you make of this?", images=[picture])

    # The retry is a real perception, not a degraded one: the pixels are restored before it runs, so
    # a peer's posted image is still *seen* rather than silently dropped by the rescue.
    assert provider.shown[0] == [picture]  # the attempt that overflowed
    assert provider.shown[-1] == [picture]  # and the retry that succeeded


def test_an_over_length_400_with_no_compactor_propagates(tmp_path):
    provider = ScriptedProvider(ProviderContextLengthError("context length", status_code=400))
    engine = Engine(provider, ToolRegistry(policy=Policy.locked()))
    session = Session("timeline:t1", engine, path=tmp_path / "t.json")

    with pytest.raises(ProviderContextLengthError):
        session.send("hello")


def test_an_unrecoverable_overflow_propagates_rather_than_retrying_forever(tmp_path):
    """Nothing to compact → let the error out. A retry of an unchanged request only repeats it."""
    provider = ScriptedProvider(
        ProviderContextLengthError("context window exceeded", status_code=400)
    )
    session = session_for(provider, tmp_path)  # an empty transcript: no cut is possible

    with pytest.raises(ProviderContextLengthError):
        session.send("hello")


# --- the one heuristic: classifying an over-length error ----------------------


@pytest.mark.parametrize(
    "text",
    [
        "This model's maximum context length is 128000 tokens, however you requested 200000",
        "Input is too long for requested model",
        "prompt is too long: 250000 tokens > 200000 maximum",
        "Please reduce the length of the messages",
        "context_length_exceeded",
        "The request exceeds the model's context window",
    ],
)
def test_the_over_length_shapes_the_vendors_actually_return(text):
    assert is_context_overflow(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Rate limit exceeded",
        "Invalid API key provided",
        "The model produced an invalid tool call",
        "",
    ],
)
def test_an_unrelated_error_is_never_mistaken_for_the_wall(text):
    # Fails safe: an unrecognized phrasing simply behaves as it did before this existed.
    assert is_context_overflow(text) is False


# --- the rescue must never repeat a side effect -------------------------------


class _Recorder(Tool):
    """A tool with a side effect worth not repeating — it counts how often it ran."""

    name = "post_message"
    description = "Post a message to the timeline."
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.runs = 0

    def run(self, **kwargs) -> str:
        self.runs += 1
        return "posted"


def test_an_overflow_after_a_tool_already_ran_does_not_re_run_the_turn(tmp_path, caplog):
    """A run can cross the ceiling *mid-flight*, after tools have already fired.

    Rewinding there would erase the record that they ran, and the re-run would very likely call them
    again — posting the same message twice. `ClaimStore` makes each *item* exactly-once; nothing
    makes a *turn* replay-safe. So the turn is not retried: the work stays, the error stands, and the
    transcript still compacts so the next wake comes in under the ceiling.
    """
    tool = _Recorder()
    registry = ToolRegistry(policy=Policy.locked())
    registry.register(tool)
    provider = ScriptedProvider(
        Message.assistant(tool_calls=[ToolCall(id="call_1", name="post_message", arguments={})]),
        ProviderContextLengthError("maximum context length exceeded", status_code=400),
        Message.assistant(content="SUMMARY"),
        usage=[20_000, None, None],
    )
    session = Session(
        "timeline:t1",
        Engine(provider, registry),
        path=tmp_path / "t.json",
        compactor=compactor(provider),
    )
    session.history.extend(conversation(20))

    with caplog.at_level(logging.WARNING), pytest.raises(ProviderContextLengthError):
        session.send("post the report")

    assert tool.runs == 1  # the side effect happened exactly once — never replayed
    assert "NOT retried" in caplog.text
    # The record of the work survives, and the transcript still shrank for the next wake.
    assert any(m.role == "tool" and m.content == "posted" for m in session.history)
    assert any("compacted" in (m.content or "") for m in session.history)


def test_zero_means_off_even_at_the_wall(tmp_path, caplog):
    """`HARNESS_MAX_CONTEXT_TOKENS=0` says "I manage this myself" — the rescue honors it."""
    provider = ScriptedProvider(
        ProviderContextLengthError("context length exceeded", status_code=400)
    )
    session = session_for(provider, tmp_path, override=0)
    session.history.extend(conversation(20))
    before = [m.content for m in session.history]

    with caplog.at_level(logging.WARNING), pytest.raises(ProviderContextLengthError):
        session.send("hello")

    # An escape hatch that rewrites the operator's transcript anyway is not an escape hatch.
    assert [m.content for m in session.history][: len(before)] == before
    assert not any("compacted" in (m.content or "") for m in session.history)
    assert "compaction is disabled" in caplog.text


def test_a_disabled_budget_never_reports_a_ceiling_it_was_told_not_to_use():
    provider = ScriptedProvider()
    provider.limit = 1_048_576

    resolved = budget(provider, override=0).limit()

    # A falsy-zero bug here would fall through to the adapter and report a limit the operator
    # explicitly opted out of.
    assert (resolved.tokens, resolved.source) == (0, "env")
    assert provider.limit_calls == 0


@pytest.mark.parametrize(
    "text",
    [
        "The uploaded file exceeds the maximum size of 20 MB",
        "Value exceeds the maximum allowed",
    ],
)
def test_an_unrelated_exceeds_the_maximum_is_not_the_wall(text):
    # The openai error mapper this feeds is shared with the image/audio tools, where an
    # "exceeds the maximum size" 400 is an ordinary file-too-big error. A false positive there
    # would compact a transcript that was never too long; a false negative only costs one wake.
    assert is_context_overflow(text) is False


# --- the 50% proof has a precondition, and the harness says so (issue #287) ---
#
# The threshold is only a *safe* distance while one turn's worst-case growth fits in the headroom
# above it: `limit x (1 - COMPACT_AT) > persisted_step_cap() x max_steps / chars-per-token`. Two operator
# knobs move those terms — `HARNESS_MAX_CONTEXT_TOKENS` shrinks the left, `HARNESS_MAX_STEPS` grows
# the right — and either can walk an agent out of the guarantee silently. These pin that it warns,
# that it warns from the *arithmetic* rather than a magic number, and that it never refuses.


def test_the_worst_case_is_derived_from_the_constants_not_hardcoded():
    # The whole point of deriving it: tune a cap or the step budget and the guarantee's arithmetic
    # follows automatically. A literal 30_000 here would rot the day either constant moved.
    assert worst_case_turn_tokens(DEFAULT_MAX_STEPS) == int(
        (TOOL_RESULT_CAP + TOOL_ARGS_CAP) * DEFAULT_MAX_STEPS / WORST_CASE_CHARS_PER_TOKEN
    )
    assert worst_case_turn_tokens(2 * DEFAULT_MAX_STEPS) == 2 * worst_case_turn_tokens(
        DEFAULT_MAX_STEPS
    )
    # `min_safe_limit` is that same inequality solved for the ceiling.
    assert min_safe_limit(DEFAULT_MAX_STEPS) * (1 - COMPACT_AT) >= worst_case_turn_tokens(
        DEFAULT_MAX_STEPS
    )


def test_the_arguments_cap_is_an_input_to_the_proof_not_a_neighbor_of_it(monkeypatch):
    """Both halves of a persisted tool call are in the arithmetic — its result *and* its arguments.

    Until issue #301 only the result was, and the omitted term was not merely small: it was
    **unbounded**, so the inequality this file is built on did not hold at all. A change that caps the
    arguments without teaching the proof about them would leave exactly that gap, one size smaller and
    just as silent — which is why `persisted_step_cap` exists and why this test watches it rather than
    a literal.
    """
    assert persisted_step_cap() == TOOL_RESULT_CAP + TOOL_ARGS_CAP

    worst_case = worst_case_turn_tokens(DEFAULT_MAX_STEPS)
    needed = min_safe_limit(DEFAULT_MAX_STEPS)
    affordable = max_safe_steps(DEFAULT_CONTEXT_LIMIT)

    # Raise what one step may leave behind and *both* sides of the guarantee move with it: the ceiling
    # it takes to stay safe goes up, and the step budget a given ceiling can afford comes down.
    monkeypatch.setattr("basecradle_harness._context.TOOL_ARGS_CAP", 2 * TOOL_ARGS_CAP)
    assert worst_case_turn_tokens(DEFAULT_MAX_STEPS) > worst_case
    assert min_safe_limit(DEFAULT_MAX_STEPS) > needed
    assert max_safe_steps(DEFAULT_CONTEXT_LIMIT) < affordable


def test_the_shipped_caps_keep_the_default_install_inside_the_guarantee():
    """The floor must clear the bar *with the arguments counted* — or every stock agent warns.

    This is the constraint that sizes `TOOL_ARGS_CAP`. The 128 K floor leaves 64 K of headroom above
    the threshold, and one turn can now add ~49 K of it (6 KB per step x 24 steps at 3 chars/token).
    Raise the argument cap to the result cap's 4 KB and the minimum safe ceiling lands at 131_072 —
    *above* the floor — and the default install starts warning about a guarantee it has not actually
    lost. A warning that fires on every stock agent is a warning nobody reads.
    """
    assert min_safe_limit(DEFAULT_MAX_STEPS) < DEFAULT_CONTEXT_LIMIT


def test_an_env_budget_too_small_for_the_guarantee_warns_and_names_the_numbers(caplog):
    # The live case, exactly: basecradle-noc#218 set HARNESS_MAX_CONTEXT_TOKENS=20000 on @pinky to
    # force a compaction, leaving 10_000 tokens of headroom against a ~32_768 worst-case turn — and
    # the harness said nothing about the guarantee it had just dropped.
    provider = ScriptedProvider()

    with caplog.at_level(logging.WARNING):
        resolved = budget(provider, override=20_000).limit()

    warning = next(r.message for r in caplog.records if r.message.startswith("context budget"))
    assert "20000" in warning and "source=env" in warning
    assert str(worst_case_turn_tokens(DEFAULT_MAX_STEPS)) in warning  # what a turn can add
    assert "10000 tokens of headroom" in warning  # what it actually has
    assert f"at least {min_safe_limit(DEFAULT_MAX_STEPS)}" in warning  # how to restore it
    assert "emergency compaction + retry" in warning  # what still protects them
    # Warn, never refuse: the operator's number stands, untouched. The escape hatch always wins.
    assert (resolved.tokens, resolved.source) == (20_000, "env")


def test_the_shipped_floor_satisfies_the_guarantee_and_says_nothing(caplog):
    # The default install must never emit this warning — the 128 K floor clears the bar by
    # construction (64_000 of headroom against a ~32_768 turn). A warning that cried wolf on every
    # stock agent would be trained away within a week.
    provider = ScriptedProvider()
    provider.limit = None

    with caplog.at_level(logging.WARNING):
        resolved = budget(provider).limit()

    assert (resolved.tokens, resolved.source) == (DEFAULT_CONTEXT_LIMIT, "default")
    assert "context budget" not in caplog.text


def test_an_adapter_ceiling_that_clears_the_bar_says_nothing(caplog):
    provider = ScriptedProvider()
    provider.limit = 1_048_576  # every model the fleet runs today

    with caplog.at_level(logging.WARNING):
        budget(provider).limit()

    assert "context budget" not in caplog.text


def test_a_small_adapter_ceiling_warns_but_never_tells_you_to_raise_it_past_the_wall(caplog):
    # A genuinely small-context model (a local model, a budget endpoint) forfeits the guarantee too,
    # and the operator did not even choose it. But the remedy MUST NOT be "raise the budget": that
    # ceiling is the model's real window, and moving the threshold above it would push compaction
    # *past the wall* — never firing in time — which is strictly worse than the problem. The only
    # honest fix at a fixed ceiling is to spend fewer steps per turn.
    provider = ScriptedProvider()
    provider.limit = 32_000

    with caplog.at_level(logging.WARNING):
        budget(provider).limit()

    warning = next(r.message for r in caplog.records if r.message.startswith("context budget"))
    assert "source=adapter" in warning
    assert f"Lower HARNESS_MAX_STEPS to {max_safe_steps(32_000)}" in warning
    # The env path's remedy sentence must never appear here. An operator skimming the journal for an
    # actionable number would otherwise raise the budget past the model's real ceiling — turning the
    # warning into the outage it exists to prevent.
    assert "raise HARNESS_MAX_CONTEXT_TOKENS to at least" not in warning
    assert "Do not raise HARNESS_MAX_CONTEXT_TOKENS" in warning


def test_a_raised_step_budget_forfeits_the_guarantee_from_the_other_side(caplog):
    # The inequality has two operator-tunable terms. A budget that is perfectly safe at the shipped
    # 24 steps stops being safe when HARNESS_MAX_STEPS is raised far enough, because the worst-case
    # turn grows with it. A guard that watched only the context knob would be half a guard.
    provider = ScriptedProvider()

    with caplog.at_level(logging.WARNING):
        budget(provider, override=DEFAULT_CONTEXT_LIMIT, max_steps=24).limit()
    assert "context budget" not in caplog.text  # safe at the shipped step budget

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        budget(provider, override=DEFAULT_CONTEXT_LIMIT, max_steps=64).limit()

    warning = next(r.message for r in caplog.records if r.message.startswith("context budget"))
    assert f"x {64} steps" in warning
    assert str(worst_case_turn_tokens(64)) in warning


def test_a_budget_at_the_minimum_safe_limit_is_silent(caplog):
    provider = ScriptedProvider()

    with caplog.at_level(logging.WARNING):
        budget(provider, override=min_safe_limit(DEFAULT_MAX_STEPS)).limit()

    assert "context budget" not in caplog.text


def test_compaction_switched_off_entirely_is_not_nagged_about_the_guarantee(caplog):
    # `0` is the operator saying "I manage this agent's context myself." There is no compaction
    # threshold to prove anything about, so a warning framed around one would be noise — and it
    # would repeat, since a disabled budget resolves no cached limit.
    provider = ScriptedProvider()

    with caplog.at_level(logging.WARNING):
        budget(provider, override=0).limit()

    assert "context budget" not in caplog.text


def test_the_warning_fires_once_not_on_every_wake(caplog):
    provider = ScriptedProvider()
    live = budget(provider, override=20_000)

    with caplog.at_level(logging.WARNING):
        live.limit()
        live.limit()
        live.limit()

    # The limit resolves once and caches; the warning rides with it. A per-call warning would bury
    # the journal of an agent that is otherwise working fine.
    assert sum(1 for r in caplog.records if r.message.startswith("context budget")) == 1


def test_the_remedy_the_warning_quotes_actually_clears_the_warning():
    # A warning that hands the operator a number which does not fix the problem is worse than
    # silence — they change the setting, the warning persists, and they stop believing it. Both
    # remedies are exact inverses of the trigger, so this sweeps them against the real condition.
    def warns(limit: int, steps: int) -> bool:
        return (limit - int(limit * COMPACT_AT)) < worst_case_turn_tokens(steps)

    for limit in range(1_000, 300_000, 337):
        steps = max_safe_steps(limit)
        if steps >= 1:
            assert not warns(limit, steps)  # the quoted step budget clears it...
            assert warns(limit, steps + 1)  # ...and is the largest that does

    for steps in range(1, 200):
        assert not warns(min_safe_limit(steps), steps)  # the quoted budget clears it


def test_a_cut_never_lands_on_an_injected_turn(tmp_path):
    """The compactor's "newest user turn always survives" floor must mean the **peer's** turn.

    Issue #297. The engine injects a `user`-role turn to *show* the model an image, and the
    code-execution bridge injects one naming the Assets a run produced. Both wear the role because
    it is the only one that content may ride on — but they are a turn's own *work*, not a new turn
    of the conversation.

    Counting one as a boundary is not merely untidy. It can become the **floor** — the newest "user"
    turn, the one compaction promises to keep — at which point the transcript keeps the image caption
    and summarizes the peer's actual message away. The result is a perfectly valid transcript and a
    broken agent: the recovery classifier can no longer find the turn that carried a message, so a
    wake killed while holding it re-drives a turn that already posted, and the peer is answered twice.
    """
    history = [
        Message.system("charter"),
        *conversation(20),
        Message(role="user", content=f"[t] {JOHN}: look at this", items=["m-1"]),
        Message.assistant(tool_calls=[ToolCall(id="c1", name="assets", arguments={})]),
        Message.tool(tool_call_id="c1", content="here it is"),
        # What the engine appends to put the picture in front of the model.
        Message(role="user", content="(Showing image: owl.png)", injected=True),
        Message.assistant(content="A barn owl."),
    ]
    provider = ScriptedProvider(Message.assistant(content="SUMMARY"), usage=[None])
    compactor = Compactor(provider, ContextBudget(provider, override=1_000))

    assert compactor.emergency_compact(history)

    # The peer's real turn survived; the injected one did not become the floor.
    survivors = [m for m in history if m.role == "user"]
    assert any(m.items == ["m-1"] for m in survivors), (
        "compaction summarized the peer's message away and kept the image caption"
    )


def test_a_compaction_records_the_items_whose_turns_it_destroys():
    """**Compaction summarizes the conversation away; it may never erase the recovery's evidence.**

    Issue #289. The delivery guarantee's classifier reads one thing off the transcript — *is there a
    turn carrying this item?* — and treats "no" as proof the dead wake never reached the model, which
    licenses a **re-drive**. Compaction can make that a lie: it replaces a region of `history` with a
    single summary, so a turn that ran tools — that posted, that bought an image at fal.ai — ceases
    to exist while the item's claim is still in flight. The next wake re-drives a turn that already
    spoke, and every tool in it fires again.

    So the summary inherits the uuids of the turns it drops. That is what lets the recovery tell "the
    model never saw this" (re-drive) from "the model saw it and the evidence is gone" (abandon, and
    say so). `_cut_index`'s own docstring already named this outcome; this is what enforces it.
    """
    history = [
        Message.system("charter"),
        Message(role="user", content=f"[t] {JOHN}: one", items=["m-1"]),
        Message.assistant(content="Answered one."),
        Message(role="user", content=f"[t] {JOHN}: two", items=["m-2", "m-3"]),
        Message.assistant(content="Answered two."),
        *conversation(20),
        Message(role="user", content=f"[t] {JOHN}: newest", items=["m-9"]),
        Message.assistant(content="Answered newest."),
    ]
    provider = ScriptedProvider(Message.assistant(content="SUMMARY"), usage=[None])
    compactor = Compactor(provider, ContextBudget(provider, override=1_000))

    assert compactor.emergency_compact(history)

    summary = next(m for m in history if m.role == "system" and m.items)
    assert summary.items == ["m-1", "m-2", "m-3"]  # every item whose turn is now gone
    # The newest turn survived the cut, so its uuid is *not* evidence of a destroyed turn.
    assert "m-9" not in summary.items


def test_a_later_compaction_carries_the_earlier_summarys_items_forward():
    """The evidence must survive **repeated** compaction, or it only ever delays the double-post.

    A previous summary sits inside the region the next compaction drops, so its uuids ride along
    with the rest — otherwise the second compaction quietly erases what the first one preserved, and
    an orphan from two compactions ago reads as "never seen" again.
    """
    history = [
        Message.system("charter"),
        Message.system("[Earlier conversation summarized] older stuff"),
        Message(role="user", content=f"[t] {JOHN}: recent", items=["m-4"]),
        Message.assistant(content="Answered."),
        *conversation(20),
        Message(role="user", content=f"[t] {JOHN}: newest", items=["m-9"]),
        Message.assistant(content="Answered newest."),
    ]
    history[1].items = ["m-1", "m-2"]  # what the *first* compaction destroyed
    provider = ScriptedProvider(Message.assistant(content="SUMMARY"), usage=[None])
    compactor = Compactor(provider, ContextBudget(provider, override=1_000))

    assert compactor.emergency_compact(history)

    summary = next(m for m in history if m.role == "system" and m.items)
    assert summary.items[:2] == ["m-1", "m-2"], (
        "the earlier summary's evidence was dropped — the second compaction erased what the first "
        "one preserved, and an orphan from two compactions ago reads as 'never seen' again"
    )
    assert "m-9" not in summary.items  # the newest turn survived the cut, so it is not evidence
