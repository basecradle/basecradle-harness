"""A brain turn the vendor cut off mid-answer is **unfinished**, never terminal narration.

Issue #490, and the shape of the defect is the reason it went unseen. `Engine.run` returns on
``not reply.tool_calls and not extend`` — and a final text stopped at ``length`` is exactly that
shape. The Delivery Guarantee reads a turn's terminal narration as its **commit record**: the turn
finished, its claim settles, the mark advances, and no later wake ever looks at that message again.
So a model cut off mid-thought was filed as a model that finished and chose what to say. Nothing
raised, nothing logged, and the peer was simply never answered.

Two adjacent cases were already covered, and naming them draws the gap precisely: truncation
*inside tool-call arguments* arrives as undecodable JSON (`ProviderResponseError`) and is retried
under the bounded transient policy (#259); a context *overflow* has its own compact-and-retry. The
uncovered one was a truncated **final text**, silent on every provider.

The fix is three small things and one decision the founder made (2026-09-09, `unfinished`):

- **The fact is reachable where the decision is made.** Every adapter already read the vendor's
  finish reason for its `llm` line (#488); it now also keeps it as `last_finish_reason`, the same
  capability shape as `last_tokens_in`, so `_observability.finish_reason` stays the one reader and
  no adapter grows a vendor branch.
- **The engine unfinishes the turn** — a trailing marker in the transcript (which is what the
  recovery reads) plus `Engine.output_truncated` (which is what this wake reads), written together
  so the live verdict and the persisted one cannot disagree.
- **The wake leaves the item pending** instead of committing it, on all three paths a turn ends —
  the batched message reply, `_act_on`'s one-item turn, and a resume — so the next wake finds an
  orphan whose turn is unfinished and **finishes it**, through the machinery #297 already built.

No budget override: the output budget is the operator's `model_params.json`, and a harness that
quietly raised a number it does not own to paper over a truncation would be tuning the agent behind
the operator's back.
"""

from __future__ import annotations

import inspect
import json
import logging

import httpx
import pytest
import respx

from basecradle_harness import (
    Engine,
    Message,
    MessagesTool,
    Session,
    Tool,
    ToolCall,
    ToolRegistry,
)
from basecradle_harness._engine import _TRUNCATED_NOTE, is_truncation_note
from basecradle_harness._memory_provider import MemoryExchange, MemoryProvider
from basecradle_harness._openai import OpenAIProvider
from basecradle_harness._openrouter import OpenRouterProvider
from basecradle_harness._wake import (
    ClaimStore,
    MarkStore,
    SeenStore,
    _turn_cut_off,
    _turn_narration,
)
from basecradle_harness._xai_sdk import XaiSdkProvider
from tests.test_wake import (
    BC_URL,
    M0,
    M1,
    M2,
    PNG_BYTES,
    REPLY,
    TIMELINE_UUID,
    _posts,
    asset_page,
    build_wake,
    dashboard,
    event_page,
    message,
    page,
    serve_messages,
    task,
    task_page,
    timeline,
)

BODY = "write me the whole history of the barn owl"


class _Echo(Tool):
    """A tool that does nothing interesting, so a turn can have a tool call in it."""

    name = "echo"
    description = "Echo."

    def run(self, **kwargs):
        return "echoed"


@pytest.fixture
def platform():
    """The respx-mocked platform — its own copy, for the reason `test_resume` keeps one."""
    with respx.mock(base_url=BC_URL, assert_all_called=False) as router:
        router.get("/users/dashboard").mock(return_value=httpx.Response(200, json=dashboard()))
        router.get(f"/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline())
        )
        router.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
            return_value=httpx.Response(
                201, json={"message": message(uuid=REPLY, body="reply", mine=True)}
            )
        )
        router.get("/assets").mock(return_value=httpx.Response(200, json=asset_page()))
        router.get("/webhook_events").mock(return_value=httpx.Response(200, json=event_page()))
        router.get("/tasks").mock(return_value=httpx.Response(200, json=task_page()))
        router.get(path__regex=r"^/blobs/").mock(
            return_value=httpx.Response(200, content=PNG_BYTES)
        )
        yield router


class _CutOff:
    """A brain whose replies stop for want of room — the vendor's own word, on the capability.

    `reasons` is one finish reason per call, in order; the last one repeats forever, so a provider
    built with a single ``"length"`` truncates every turn it is ever asked for.
    """

    provider, model = "openai", "gpt-4o"

    def __init__(self, *replies: Message, reasons: tuple[str, ...] = ("length",)) -> None:
        self._replies = list(replies) or [Message.assistant(content="The barn owl is a sp")]
        self._reasons = list(reasons)
        self.last_finish_reason: str | None = None
        self.seen: list[list[Message]] = []
        self.calls = 0

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        self.last_finish_reason = self._reasons[min(self.calls, len(self._reasons) - 1)]
        self.calls += 1
        reply = self._replies[min(self.calls - 1, len(self._replies) - 1)]
        return reply


class _Finishes:
    """A brain that answers whole — `last_finish_reason` is the vendor's ordinary ``stop``."""

    provider, model = "openai", "gpt-4o"

    def __init__(self, content: str = "…ecies of owl. That is the whole of it.") -> None:
        self.content = content
        self.last_finish_reason: str | None = None
        self.seen: list[list[Message]] = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        self.last_finish_reason = "stop"
        return Message.assistant(content=self.content)


class _RecordingMemory(MemoryProvider):
    """A memory provider that only records what it was asked to mine."""

    def __init__(self) -> None:
        self.observed: list[MemoryExchange] = []

    def observe(self, exchange: MemoryExchange) -> None:
        self.observed.append(exchange)


def _transcript(home) -> list[dict]:
    """The persisted transcript for this timeline, straight off the disk."""
    path = home / "sessions" / f"timeline%3A{TIMELINE_UUID}.json"
    return json.loads(path.read_text())


def _claim(home, uuid=M0, kind="messages"):
    return ClaimStore(home).read(TIMELINE_UUID, uuid, kind=kind)


# === the engine: a cut-off final text does not end the turn ===================


def test_a_truncated_final_text_is_marked_unfinished():
    """The headline at engine level: the loop returns the fragment and says the turn is not done.

    Both halves are asserted because both are load-bearing and they are written together: the
    marker is what the *recovery* reads (after this process is gone), the flag is what *this* wake
    reads. Either one alone would leave the other reaching the opposite verdict about one turn.
    """
    engine = Engine(_CutOff(), ToolRegistry())
    messages = [Message.user(BODY)]

    reply = engine.run(messages)

    assert reply.content == "The barn owl is a sp"  # the fragment is real and is still returned
    assert engine.output_truncated is True
    assert is_truncation_note(messages[-1])
    # And the single definition of the commit record now answers "unfinished" for this turn.
    assert _turn_narration(messages) is None
    assert _turn_cut_off(messages) is True


def test_an_ordinary_final_text_is_untouched():
    """The regression bar: an agent whose model stops normally is byte-identical to pre-#490."""
    engine = Engine(_Finishes(), ToolRegistry())
    messages = [Message.user("hi")]

    engine.run(messages)

    assert engine.output_truncated is False
    assert not any(is_truncation_note(m) for m in messages)
    assert _turn_narration(messages) == "…ecies of owl. That is the whole of it."


def test_an_adapter_that_reports_nothing_is_untouched_too():
    """The capability **fails safe**: unanswered, a truncation goes undetected exactly as before.

    A `Provider` is a protocol with one required method, so an adapter (or a library caller's own
    object) that has never heard of `last_finish_reason` must keep working — it costs the detection,
    never the turn.
    """

    class _Bare:
        provider, model = "custom", "whatever"

        def chat(self, messages, tools=None):
            return Message.assistant(content="done")

    engine = Engine(_Bare(), ToolRegistry())
    messages = [Message.user("hi")]

    engine.run(messages)

    assert engine.output_truncated is False
    assert not any(is_truncation_note(m) for m in messages)


def test_the_verdict_is_per_run_not_per_engine():
    """One engine drives every turn of a wake, so a truncated turn must not taint the next one."""
    brain = _CutOff(
        Message.assistant(content="cut off"),
        Message.assistant(content="whole"),
        reasons=("length", "stop"),
    )
    engine = Engine(brain, ToolRegistry())

    engine.run([Message.user("first")])
    assert engine.output_truncated is True

    second = [Message.user("second")]
    engine.run(second)
    assert engine.output_truncated is False
    assert not any(is_truncation_note(m) for m in second)


def test_a_truncation_mid_tool_chain_is_not_the_end_of_the_turn():
    """Only the reply that would **end** the turn is judged.

    A truncated reply that still carries tool calls is not a terminal narration in the first place —
    the loop runs the calls and asks again — so it must not be marked. (Truncation *inside* the
    arguments is a different fault entirely: undecodable JSON, raised and retried by #259.)
    """
    brain = _CutOff(
        Message.assistant(tool_calls=[ToolCall(id="c1", name="echo", arguments={})]),
        Message.assistant(content="all done"),
        reasons=("length", "stop"),  # a vendor *can* cut a reply off around its tool calls
    )

    registry = ToolRegistry()
    registry.register(_Echo())
    engine = Engine(brain, registry)
    messages = [Message.user("echo please")]

    engine.run(messages)

    assert engine.output_truncated is False
    assert not any(is_truncation_note(m) for m in messages)
    assert _turn_narration(messages) == "all done"


def test_a_truncation_a_hook_extends_past_is_logged_but_not_marked(caplog):
    """A hook extends on exactly the shape a truncated narration has, so this is the common case.

    `Engine.run` ends on *no tool calls and no extension*, and both shipped turn hooks extend on
    "no tool calls" — the no-reply informer nudges an agent that fell silent when it was addressed;
    the code-execution bridge feeds a run's output files back. So a model whose closing narration
    ran out of room is very often *not* at the end of its turn: the loop carries on, and some later
    reply is what settles it. Marking here would stamp a turn that is still working. Saying nothing
    would hide the thing this whole change exists to surface — so the line is written either way,
    and ``outcome=`` is what tells the two apart.
    """
    brain = _CutOff(
        Message.assistant(content="I was about to say th"),
        Message.assistant(content="…and there it is."),
        reasons=("length", "stop"),
    )
    extended = [False]

    def hook(reply, messages):
        if extended[0]:
            return False
        extended[0] = True
        messages.append(Message.system("say something"))
        return True

    engine = Engine(brain, ToolRegistry(), turn_hook=hook)
    messages = [Message.user(BODY)]

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        engine.run(messages)

    line = next(r for r in caplog.records if r.getMessage().startswith("turn truncated"))
    assert "outcome=continued" in line.getMessage()
    # The turn went on and finished, so it is committed — not marked, not left pending.
    assert engine.output_truncated is False
    assert not any(is_truncation_note(m) for m in messages)
    assert _turn_narration(messages) == "…and there it is."


def test_the_truncation_is_logged_loudly(caplog):
    """WARNING, for the reason the step cap is: the turn returns a perfectly good-looking string.

    Nothing downstream *looks* wrong, which is exactly why the event has to be findable by a
    severity filter rather than buried in the INFO stream.
    """
    engine = Engine(_CutOff(), ToolRegistry())

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        engine.run([Message.user(BODY)])

    line = next(r for r in caplog.records if r.getMessage().startswith("turn truncated"))
    assert line.levelno == logging.WARNING
    assert "finish_reason=length" in line.getMessage()  # the vendor's own word, not a paraphrase
    assert "outcome=unfinished" in line.getMessage()  # …and what the harness did about it


def test_the_marker_reaches_disk_with_the_turn(tmp_path):
    """The marker is persisted, not merely held in memory.

    A marker that lived only in RAM would leave the *on-disk* turn reading as finished — which is
    the original bug with an extra step, because the transcript is the only thing a later wake has.
    """
    session = Session("timeline:x", Engine(_CutOff(), ToolRegistry()), path=tmp_path / "s.json")

    session.send(BODY)

    saved = json.loads((tmp_path / "s.json").read_text())
    assert saved[-1]["role"] == "system"
    assert saved[-1]["content"] == _TRUNCATED_NOTE


def test_a_truncated_reserve_summary_still_ends_the_turn():
    """The reserve summary is deliberately **outside** this check, and that is stated, not implied.

    A step-capped turn has already spent the whole budget it was promised; resuming it is the one
    thing that must not happen, or a wake that hit the cap would be resumed on every later wake,
    forever, spending a fresh budget each time. Move the truncation read one level up — into
    `_chat`, or into `_reserve_summary` — and every other test in this file still passes while that
    loop opens. So it is pinned here.
    """
    call = Message.assistant(tool_calls=[ToolCall(id="c1", name="echo", arguments={})])
    brain = _CutOff(
        call,  # step 1 — still working
        call,  # step 2 — the budget is spent, still working
        Message.assistant(content="Here is where I got t"),  # the reserve report, cut off
        reasons=("tool_calls", "tool_calls", "length"),
    )
    registry = ToolRegistry()
    registry.register(_Echo())
    engine = Engine(brain, registry, max_steps=2)
    messages = [Message.user("echo forever")]

    engine.run(messages)

    assert engine.reserve_used is True
    assert engine.output_truncated is False
    assert not any(is_truncation_note(m) for m in messages)
    assert _turn_narration(messages) == "Here is where I got t"


def test_every_shipped_adapter_records_the_finish_reason():
    """The standing rule, made mechanical: no adapter ships without recording it.

    `last_finish_reason` fails **safe** when unanswered — a truncation simply goes undetected — so
    this is not the hard gate `cache_mode` is. It is here for the reason that gate exists anyway:
    an adapter that forgets loses the whole of issue #490 on its provider, silently, and nothing
    else in this suite would go red. Mirrors `test_caching.test_every_shipped_adapter_declares_a_
    cache_mode`, deliberately, so the two read as one habit.
    """
    for adapter in (OpenAIProvider, XaiSdkProvider, OpenRouterProvider):
        source = inspect.getsource(adapter)
        assert "self.last_finish_reason" in source, adapter.__name__


# === the wake: a truncated turn is not committed ==============================


def test_a_truncated_turn_leaves_the_message_pending(platform, tmp_path):
    """The whole point. The mark does not move and the claim stays in-flight.

    Committing here is the defect: the mark is a **cursor**, so moving it past this message hides
    it from every future wake, forever — and the peer, who can see the agent said nothing, has no
    way to tell that from an agent that read them and chose silence.
    """
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    agent, _ = build_wake(tmp_path, _CutOff())

    agent.wake()

    assert MarkStore(tmp_path).get(TIMELINE_UUID) is None  # the cursor did not pass it
    assert _claim(tmp_path).phase == "in-flight"  # and the item is still owed an answer
    # The evidence the *next* wake will classify on is on disk, not just in this process.
    assert _transcript(tmp_path)[-1]["content"] == _TRUNCATED_NOTE


def test_the_next_wake_finishes_the_turn_rather_than_starting_it_again(platform, tmp_path):
    """Founder ruling, 2026-09-09: **resume**, through the path issue #297 already built.

    A re-drive would be *safe* here (a pure-text turn fired nothing) and it would not *work*: the
    same input under the same output budget truncates in the same place, so the peer is answered by
    nobody, forever, one wake at a time. Continuing from the fragment converges.
    """
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    first, _ = build_wake(tmp_path, _CutOff())
    first.wake()

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    second, brain = build_wake(tmp_path, _Finishes())
    second.wake()

    # The model was handed the turn it was cut off in — its own fragment included — and not a
    # second copy of the question.
    replayed = brain.seen[0]
    assert sum(1 for m in replayed if m.role == "user" and not m.injected) == 1
    assert any(m.role == "assistant" and m.content == "The barn owl is a sp" for m in replayed)
    assert any(is_truncation_note(m) for m in replayed)
    # And now that the turn finished, the item settles: the cursor moves and the claim is done.
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M0
    assert _claim(tmp_path).settled


def test_a_truncated_turn_that_already_spoke_never_speaks_twice(platform, tmp_path):
    """The at-most-once half, on the shape that makes it dangerous.

    A turn that posted through the `messages` tool and *then* truncated its narration is unfinished
    **and** committed to a side effect. It must be continued, never re-run — the resume re-fires
    nothing, because the tool's result is already on disk.
    """
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    speaks = _CutOff(
        Message.assistant(
            tool_calls=[
                ToolCall(id="c1", name="messages", arguments={"action": "create", "body": "Owls!"})
            ]
        ),
        Message.assistant(content="I told them about the ow"),
        reasons=("stop", "length"),
    )
    first, _ = build_wake(tmp_path, speaks, tools=[MessagesTool()])
    first.wake()

    assert _posts(platform) == ["Owls!"]
    assert MarkStore(tmp_path).get(TIMELINE_UUID) is None

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    second, _ = build_wake(tmp_path, _Finishes(), tools=[MessagesTool()])
    second.wake()

    assert _posts(platform) == ["Owls!"]  # ONE post, not two
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M0


def test_a_resume_that_truncates_again_stays_pending(platform, tmp_path):
    """Repeating means the output budget is too small for what this turn is trying to say.

    The harness says so — loudly, every time — and leaves the item pending rather than committing a
    fragment. Raising the budget is the operator's call, and the harness never makes it for them.
    """
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    build_wake(tmp_path, _CutOff())[0].wake()

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    second, brain = build_wake(tmp_path, _CutOff(Message.assistant(content="ecies of ow")))
    second.wake()

    assert brain.calls == 1  # it did resume
    assert MarkStore(tmp_path).get(TIMELINE_UUID) is None  # and it is still not finished
    assert _claim(tmp_path).phase == "in-flight"


def test_a_truncated_batch_turn_holds_every_message_in_it(platform, tmp_path):
    """A turn answers a *batch*, so an unfinished one leaves the whole batch unanswered.

    The mark is a cursor and stops at the **oldest** undecided item, so a partial commit here would
    be worse than none: it would let the cursor pass a message whose answer was never written.
    Finishing the turn once, on the next wake, settles all of them together.
    """
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)  # a prior wake answered M0; M1 and M2 are new
    inbox = page(
        message(uuid=M2, body="and another thing"),
        message(uuid=M1, body=BODY),
        message(uuid=M0, body="answered already"),
    )
    serve_messages(platform, inbox)
    first, _ = build_wake(tmp_path, _CutOff())
    first.wake()

    claims = ClaimStore(tmp_path)
    assert claims.read(TIMELINE_UUID, M1, kind="messages").phase == "in-flight"
    assert claims.read(TIMELINE_UUID, M2, kind="messages").phase == "in-flight"
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M0  # the cursor did not move at all

    serve_messages(platform, inbox)
    second, brain = build_wake(tmp_path, _Finishes())
    second.wake()

    # ONE model call finished the turn for both, and both settle together.
    assert len(brain.seen) == 1
    assert claims.read(TIMELINE_UUID, M1, kind="messages").phase == "done"
    assert claims.read(TIMELINE_UUID, M2, kind="messages").phase == "done"
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M2


def test_a_finished_resume_leaves_a_transcript_that_reads_as_finished(platform, tmp_path):
    """The marker unfinishes a turn; a completed continuation must un-unfinish it, on disk.

    `_turn_narration` asks whether the turn's work **ends** on the model's own text, so a resume
    that lands its continuation *after* the marker settles the question by construction. Were the
    marker read as a sticky property of the turn instead, a finished turn would read as unfinished
    forever, and the item behind it would be resumed on every wake for the life of the timeline.

    (The marker rides inside the build's own span, so `_generate_settled`'s staleness rollback —
    `del history[base_len:]` — removes it with the build it belongs to, by construction rather than
    by a second rule.)
    """
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    build_wake(tmp_path, _CutOff())[0].wake()

    assert any(t.get("content") == _TRUNCATED_NOTE for t in _transcript(tmp_path))

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    build_wake(tmp_path, _Finishes())[0].wake()

    history = _transcript(tmp_path)
    assert history[-1]["role"] == "assistant"  # the turn now ends on the model's finished text
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M0

    # And a third wake neither resumes nor re-drives it: the turn is committed.
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    third, idle = build_wake(tmp_path, _Finishes())
    third.wake()
    assert idle.seen == []


def test_a_truncated_turn_is_journaled_as_truncated(platform, tmp_path, caplog):
    """Four unspoken endings now, and they read nothing alike to whoever digs this up later.

    The narration line is the flight recorder; a fragment filed under ``kind=narration`` would be
    indistinguishable from an agent that said its piece and stopped.
    """
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    agent, _ = build_wake(tmp_path, _CutOff())

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake()

    unspoken = next(r for r in caplog.records if r.getMessage().startswith("unspoken "))
    assert "kind=truncated" in unspoken.getMessage()
    # …and the wake's own greppable record of what it decided to do about it. Asserted by **value
    # and by level**, not by presence: the line exists to name the item a forensic dig starts from,
    # and to be findable by a severity filter — a version that dropped either would still contain
    # the words.
    decision = next(r for r in caplog.records if "wake truncated_turn" in r.getMessage())
    assert decision.levelno == logging.WARNING
    assert f"item={M0}" in decision.getMessage()
    assert "kind=messages" in decision.getMessage()
    assert f"timeline={TIMELINE_UUID}" in decision.getMessage()


def test_a_truncated_turn_is_mined_like_any_other(platform, tmp_path):
    """The peer's words reach memory now, not on the strength of a resume that may never come.

    The tempting version — skip the mine, let the turn that finishes it mine the whole exchange —
    bets the peer's message on a resume, and a resume has three documented ways never to happen:
    `_drop` on a context overflow, `_drop` on a payload-too-large, and `_abandon` when a compaction
    destroyed the turn. On any of those the peer's words would be in **no** memory, on any
    timeline, forever. A fragment is still the model's own words, and the peer's half is real
    whatever happened on our side — the same call `_observe` already makes for a degraded turn.

    The price is one duplicate drawer when the resume lands, which is a retrieval cost, not a loss.
    """
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    memory = _RecordingMemory()
    agent, _ = build_wake(tmp_path, _CutOff(), memory_provider=memory)

    agent.wake()

    assert len(memory.observed) == 1
    assert BODY in memory.observed[0].user
    assert memory.observed[0].assistant == "The barn owl is a sp"  # the fragment, honestly


def test_the_guarantee_covers_every_kind_not_just_messages(platform, tmp_path):
    """`_act_on`'s three kinds run the same machinery, so they get the same verdict (issue #289).

    An activated task whose turn was cut off is as unfinished as a message's, and the queue that
    re-offers it is the platform's own `activated` list rather than a cursor — so what must not
    happen here is the *seen-set* recording it.
    """
    serve_messages(platform, page())
    platform.get("/tasks").mock(
        return_value=httpx.Response(
            200, json=task_page(task(uuid=M1, instructions="check the owl feeder"))
        )
    )
    agent, brain = build_wake(tmp_path, _CutOff())

    agent.wake()

    assert brain.calls == 1  # the task did drive a turn, and that turn was cut off
    assert _claim(tmp_path, M1, kind="tasks").phase == "in-flight"
    assert SeenStore(tmp_path).all(TIMELINE_UUID, kind="tasks") == set()


def test_a_wake_that_left_a_turn_unfinished_starts_no_more_model_work(platform, tmp_path):
    """**The latch, and it is what keeps the resume's evidence alive.**

    A turn left cut off is waiting on a resume, and a resume reads the *transcript*. Every further
    turn this wake runs can compact that transcript (`Session.send` ends in `_compact_if_needed`,
    and an over-length rescue compacts hard), and a compaction that destroys the unfinished turn
    turns the next wake's verdict from **resume** into **abandon** — the peer dropped, loudly, by
    the machinery meant to answer them. So the wake stops starting new turns, exactly as the
    out-of-funds wall does.

    The deferred items are not dropped: unclaimed, unrecorded, re-read on the next wake. That is
    the same outcome `_generate_settled` already gives a message that lands during its final build.
    """
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    platform.get("/tasks").mock(
        return_value=httpx.Response(
            200, json=task_page(task(uuid=M1, instructions="check the owl feeder"))
        )
    )
    agent, brain = build_wake(tmp_path, _CutOff())

    agent.wake()

    # ONE model call: the message batch. The task behind it was never engaged.
    assert brain.calls == 1
    assert _claim(tmp_path, M1, kind="tasks") is None  # not even claimed
    assert SeenStore(tmp_path).all(TIMELINE_UUID, kind="tasks") == set()  # …and not recorded

    # The next wake, with a brain that finishes, takes both: the message's turn is resumed and the
    # task is driven fresh. Nothing was lost by stopping.
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    second, healthy = build_wake(tmp_path, _Finishes())
    second.wake()

    assert len(healthy.seen) == 2
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M0
    assert SeenStore(tmp_path).all(TIMELINE_UUID, kind="tasks") == {M1}
