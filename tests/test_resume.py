"""A wake killed by a **signal** answers the peer once — never twice, never not at all.

Issue #297. The delivery guarantee is *at-least-once for the read, at-most-once for every side
effect; the turn is the unit of commit* — and until this issue it was true of an **exception** and
false of a **signal**. `Session._exchange` persisted the transcript in a `finally`, and a `finally`
does not run on `SIGKILL`, on `SIGTERM`'s default disposition, on the OOM killer, or on a box reset.
So a wake killed *after* the `messages` tool posted left no evidence that it ever ran, the recovery
read that emptiness as "the model never saw it", re-drove the message, and the peer was answered
twice — with every non-idempotent tool in the turn firing a second time.

These tests pin the three pieces that close it, and the one bug they uncovered on the way:

- **The evidence survives the kill.** The turn persists as it runs, and the assistant turn naming a
  tool call reaches disk **before that tool is dispatched** — which is what makes the classifier's
  central claim true: *a tool call absent from the transcript is a tool call that never ran.*
- **The transcript stays sendable.** A kill mid-tool-chain leaves a call with no result, and a
  dangling `tool_call_id` is malformed *permanently* — the provider 400s on it forever. Naive
  incremental persistence is therefore strictly worse than the bug it fixes; healing on load is what
  makes it a fix at all.
- **The dead turn is resumed, not re-driven.** Its tool results are on disk, so it needs neither
  re-running nor abandoning. An interrupted *platform create* is re-issued under the deterministic
  key its dead wake minted, so the platform returns the original record; an interrupted
  *non-idempotent* effect is surfaced to the model rather than re-run, because no key can un-spend
  money.
- **`_turn_of` could not see a message with a newline in it.** The classifier matched a
  peer-controlled *rendered line* against the turn's content split on newlines, so a body with a
  second paragraph never matched and was re-driven regardless of what the dead wake had done. It was
  live in the field. The turn now carries the uuids it rendered.

**How a signal is simulated.** `Session._persist` is the save in the `finally` — precisely the one a
signal skips. Neutering it, and leaving the incremental writes real, reproduces a `SIGKILL` exactly:
everything the turn wrote *as it went* is on disk, and the closing write never happened. No test
kills a real process, and none needs to.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from basecradle_harness import (
    Harness,
    Message,
    MessagesTool,
    Session,
    Tool,
    ToolCall,
)
from basecradle_harness._idempotency import MESSAGE, IdempotencyKeys, key
from basecradle_harness._session import INTERRUPTED, TOOL_ARGS_CAP, _json_size, turn_work
from basecradle_harness._wake import Claim, ClaimStore, _turn_narration
from tests.test_wake import (
    BC_URL,
    M0,
    M1,
    PNG_BYTES,
    REPLY,
    TIMELINE_UUID,
    _posts,
    asset_page,
    build_wake,
    crashed_wake_owning,
    dashboard,
    event_page,
    message,
    page,
    serve_messages,
    task_page,
    timeline,
)

BODY = "generate me an image of a barn owl"

#: A peer message with a **newline in the body** — the shape that broke the old text-matching
#: classifier, and now the default shape these tests use, so the repair stays pinned.
MULTILINE = "Can you look at this?\n\nIt is the second paragraph that used to break the recovery."


@pytest.fixture
def platform():
    """The respx-mocked platform — its own copy, so the fixture name is not an imported symbol.

    Deliberately not imported from `test_wake`: a fixture pulled in by name is shadowed by every
    test's own `platform` parameter, which is exactly the kind of "works, but the linter is right"
    arrangement this repo does not ship. The wire builders *are* shared; only the wiring is local.
    """
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


@pytest.fixture
def kill_the_finally(monkeypatch):
    """Simulate a signal: the incremental writes land, the closing one never happens.

    `Session._persist` is the save in `_drive`'s `finally`. A `finally` runs on an exception and
    **not** on a signal, so neutering exactly this one — and nothing else — is what a `SIGKILL`
    looks like from the disk's point of view.
    """
    monkeypatch.setattr(Session, "_persist", lambda self, *, masking: None)


class _Speaks:
    """Posts one message through the `messages` tool, then dies — the killed-after-posting wake."""

    provider, model = "openai", "gpt-4o"

    def __init__(self, body: str = "Here is your owl.") -> None:
        self.body = body
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return Message.assistant(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="messages",
                        arguments={"action": "create", "body": self.body},
                    )
                ]
            )
        raise RuntimeError("SIGKILL: the box went down after the message was posted")


class _Finishes:
    """Picks an interrupted turn up and settles it, saying nothing new."""

    provider, model = "openai", "gpt-4o"

    def __init__(self) -> None:
        self.seen: list[list[Message]] = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return Message.assistant(content="I had already answered that; nothing more to add.")


def _transcript(home) -> list[dict]:
    """The persisted transcript for this timeline, straight off the disk."""
    path = home / "sessions" / f"timeline%3A{TIMELINE_UUID}.json"
    return json.loads(path.read_text())


# --- the headline: a signal-killed wake posts once, not twice -----------------


def test_a_wake_killed_after_it_posted_does_not_post_again(platform, tmp_path, kill_the_finally):
    """**The bug, closed.** A wake posts, is killed by a signal, and the peer is answered ONCE.

    This is the whole issue in one test. Before it, the killed wake left no transcript at all, the
    recovery concluded "the model never saw this message", re-drove it, and the peer got the answer
    twice. Now the turn's own work is on disk as it happens, the recovery sees the `messages` call
    the dead wake made, and it **finishes** that turn instead of re-running it.

    The assertion that matters is the count of POSTs the *platform* received across both wakes.
    """
    serve_messages(platform, page(message(uuid=M0, body=MULTILINE)))
    first, _ = build_wake(tmp_path, _Speaks())
    with pytest.raises(RuntimeError):
        first.wake()

    assert _posts(platform) == ["Here is your owl."]  # the dead wake did speak

    serve_messages(platform, page(message(uuid=M0, body=MULTILINE)))
    second, brain = build_wake(tmp_path, _Finishes(), tools=[MessagesTool()])
    second.wake()

    # ONE message on the timeline, not two. The peer is answered exactly once.
    assert _posts(platform) == ["Here is your owl."]
    # And the resumed model was handed the turn it was cut off in — the peer's message, its own
    # tool call, and the result of it — rather than a fresh copy of the question.
    replayed = brain.seen[0]
    assert sum(1 for m in replayed if m.role == "user" and not m.injected) == 1
    assert any(m.tool_calls and m.tool_calls[0].name == "messages" for m in replayed)


def test_the_evidence_a_signal_leaves_is_exactly_what_the_classifier_needs(
    platform, tmp_path, kill_the_finally
):
    """A `SIGKILL` mid-turn leaves the user turn, the assistant tool-call turn, and every result."""
    serve_messages(platform, page(message(uuid=M0, body=MULTILINE)))
    agent, _ = build_wake(tmp_path, _Speaks())
    with pytest.raises(RuntimeError):
        agent.wake()

    roles = [
        (t["role"], t.get("tool_calls", []) and t["tool_calls"][0]["name"])
        for t in _transcript(tmp_path)
        if t["role"] != "system"
    ]
    assert ("user", []) in [(r, c) for r, c in roles]  # the peer's message
    assert ("assistant", "messages") in roles  # the call it made
    assert ("tool", []) in [(r, c) for r, c in roles]  # and the result of it
    # The turn carries the uuid it rendered — the classifier's evidence, not a text match.
    user_turn = next(t for t in _transcript(tmp_path) if t["role"] == "user")
    assert user_turn["items"] == [M0]


# --- the ordering invariant that everything else rests on --------------------


def test_the_tool_call_is_on_disk_before_the_tool_runs(platform, tmp_path):
    """**The load-bearing ordering.** A tool cannot fire before the transcript names its call.

    This is what licenses the classifier's central inference — *a tool call absent from the
    transcript is a tool call that never ran* — and it is asserted from inside the tool itself, at
    the only instant that can prove it: the tool reads its own session file while it is running and
    finds the call that invoked it already there.

    Reverse the two writes and nothing fails, nothing logs, and the guarantee is silently gone: a
    wake killed in the gap posts a message the transcript never mentions, and the next wake says it
    again.
    """
    seen: dict[str, object] = {}

    class ReadsItsOwnTranscript(Tool):
        name = "noop"
        description = "Does nothing, but checks the transcript first."

        def run(self, **kwargs) -> str:
            seen["transcript"] = _transcript(tmp_path)
            return "ok"

    class CallsIt:
        provider, model = "openai", "gpt-4o"

        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return Message.assistant(
                    tool_calls=[ToolCall(id="call_1", name="noop", arguments={})]
                )
            return Message.assistant(content="Done.")

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    agent, _ = build_wake(tmp_path, CallsIt(), tools=[ReadsItsOwnTranscript()])
    agent.wake()

    written = seen["transcript"]
    calls = [t for t in written if t["role"] == "assistant" and t.get("tool_calls")]
    assert calls, "the tool ran before its own call was durable — the guarantee is void"
    assert calls[0]["tool_calls"][0]["id"] == "call_1"


def test_a_turn_that_cannot_be_recorded_never_dispatches_a_tool(platform, tmp_path, monkeypatch):
    """The pre-dispatch write is allowed to **fail the turn** — the one progress call that is not
    swallowed.

    Nothing has run at that point, so stopping costs nobody anything and the claim stays in-flight
    for the next wake. Swallowing it would dispatch tools with no record that they exist, which is
    the bug itself — reached silently, on the near-full box where it is likeliest.
    """
    ran: list[str] = []

    class Records(Tool):
        name = "noop"
        description = "Does nothing."

        def run(self, **kwargs) -> str:
            ran.append("ran")
            return "ok"

    class CallsIt:
        provider, model = "openai", "gpt-4o"

        def chat(self, messages, tools=None):
            return Message.assistant(tool_calls=[ToolCall(id="c1", name="noop", arguments={})])

    saves = {"n": 0}
    real = Session._save

    def fail_after_the_user_turn(self, messages=None):
        saves["n"] += 1
        if saves["n"] > 1:  # the user turn lands; the pre-dispatch write hits the wall
            raise OSError(28, "No space left on device")
        return real(self, messages)

    monkeypatch.setattr(Session, "_save", fail_after_the_user_turn)
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    agent, _ = build_wake(tmp_path, CallsIt(), tools=[Records()])

    with pytest.raises(OSError):
        agent.wake()
    assert ran == [], "a tool was dispatched with no durable record that it was ever called"


# --- healing: a killed transcript must still load, and still be sendable ------


def test_an_interrupted_call_is_healed_on_load_so_the_next_wake_is_not_bricked(tmp_path):
    """A dangling `tool_call_id` is malformed **permanently** — every later wake 400s on it.

    This is why naive incremental persistence would be strictly worse than the bug it fixes: it
    would trade an occasional double post for an agent that can never speak on that timeline again,
    until a human deletes the file. Healing on load is not hygiene; it is the reason the rest of the
    design is allowed to exist.
    """
    path = tmp_path / "sessions" / "timeline%3At1.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            [
                {"role": "user", "content": "two things please", "items": [M0]},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1", "name": "noop", "arguments": {}},
                        {"id": "c2", "name": "noop", "arguments": {}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                # c2's result never made it to disk — the wake was killed while it ran.
            ]
        )
    )

    class Quiet:
        def chat(self, messages, tools=None):
            return Message.assistant(content="ok")

    session = Harness(Quiet(), home=tmp_path).session("timeline:t1")

    answered = [m.tool_call_id for m in session.history if m.role == "tool"]
    assert answered == ["c1", "c2"], "every call must have a result, or the provider rejects it"
    healed = next(m for m in session.history if m.tool_call_id == "c2")
    assert healed.content == INTERRUPTED
    assert "NOT been re-run" in healed.content  # it says what is true, and nothing else


# --- the keys: deterministic, and reproducible by the wake that recovers ------


def test_the_key_is_the_same_number_computed_twice(tmp_path):
    """The ordinal is read off the transcript, so the dead wake and its recoverer agree by
    construction.

    A **counter** would be a second source of truth, and it would drift the first time a create-shaped
    call was recorded without ever reaching the tool's create branch (the model passes an unknown
    kwarg; `run(**arguments)` raises `TypeError`; the engine feeds the error back as the result). The
    counter would say 1 and the transcript would say 2 — and an ordinal off by one is a key the
    platform has never seen, which is a message posted twice.
    """

    class Quiet:
        def chat(self, messages, tools=None):
            return Message.assistant(content="ok")

    session = Harness(Quiet(), home=tmp_path).session("timeline:t1")
    keys = IdempotencyKeys()
    session.history.append(Message(role="user", content="hi", items=[M0]))
    session._turn = session.history[-1]
    keys.begin(session, timeline=TIMELINE_UUID, anchor=M0)

    # The first create of the turn.
    assert keys.mint(MESSAGE) == key(timeline=TIMELINE_UUID, anchor=M0, kind=MESSAGE, ordinal=1)

    # A create-shaped call that never reached the tool (bad kwargs) still gets a result, and still
    # consumes its place in the count — because the transcript records the call, and the transcript
    # is what both halves count.
    session.history.append(
        Message.assistant(
            tool_calls=[ToolCall(id="c1", name="messages", arguments={"action": "create"})]
        )
    )
    session.history.append(Message.tool(tool_call_id="c1", content="Error running 'messages'"))
    assert keys.mint(MESSAGE) == key(timeline=TIMELINE_UUID, anchor=M0, kind=MESSAGE, ordinal=2)


def test_an_interrupted_create_is_re_issued_under_the_dead_wakes_key(platform, tmp_path):
    """The resume re-issues an interrupted platform create with the key its dead wake minted.

    The platform recognizes the key and returns the original record, so the message the dead wake
    posted stands and nothing is posted twice. This is the *only* class of interrupted call that can
    be safely re-executed, and the key is the entire reason it can.
    """
    serve_messages(platform, page(message(uuid=M0, body=MULTILINE)))

    # The dead wake's transcript: it issued the `messages` create and was killed before its result
    # could be written. That is the one genuinely unknowable state — the POST may or may not have
    # landed — and the key is what makes acting on it safe.
    crashed_wake_owning(tmp_path, M0)
    session = Harness(_Finishes(), home=tmp_path).session(f"timeline:{TIMELINE_UUID}")
    session.history.append(
        Message(
            role="user",
            content=f"[2026-06-04T00:00:00.000Z] john: {MULTILINE}",
            items=[M0],
        )
    )
    session.history.append(
        Message.assistant(
            tool_calls=[
                ToolCall(
                    id="c1", name="messages", arguments={"action": "create", "body": "Here you go."}
                )
            ]
        )
    )
    session._save()

    agent, _ = build_wake(tmp_path, _Finishes(), tools=[MessagesTool()])
    agent.wake()

    assert _posts(platform) == ["Here you go."]  # replayed from the transcript, byte for byte
    assert _keys(platform) == [key(timeline=TIMELINE_UUID, anchor=M0, kind=MESSAGE, ordinal=1)], (
        "the re-issue must carry the key the dead wake minted, or the platform posts it twice"
    )


def test_a_non_idempotent_effect_is_surfaced_never_re_run(platform, tmp_path):
    """No key can un-spend money at fal.ai, so an interrupted `generate_image` is **not** re-run.

    The model is told plainly that the outcome is unknown and left to decide — the Unspoken
    Channel's own stance applied to recovery: full visibility, never forcing. It can read the
    timeline and see for itself.
    """
    runs: list[str] = []

    class Expensive(Tool):
        name = "generate_image"
        description = "Costs real money."

        def run(self, **kwargs) -> str:
            runs.append("charged")
            return "an image"

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    crashed_wake_owning(tmp_path, M0)
    session = Harness(_Finishes(), home=tmp_path).session(f"timeline:{TIMELINE_UUID}")
    session.history.append(
        Message(role="user", content=f"[2026-06-04T00:00:00.000Z] john: {BODY}", items=[M0])
    )
    session.history.append(
        Message.assistant(
            tool_calls=[ToolCall(id="c1", name="generate_image", arguments={"prompt": "an owl"})]
        )
    )
    session._save()

    agent, brain = build_wake(tmp_path, _Finishes(), tools=[MessagesTool(), Expensive()])
    agent.wake()

    assert runs == [], "the recovery re-ran a tool that spends money"
    # And the model was told, rather than left to assume the call had simply failed.
    shown = brain.seen[0]
    interrupted = next(m for m in shown if m.role == "tool" and m.tool_call_id == "c1")
    assert interrupted.content == INTERRUPTED


# --- the arguments the recovery replays from (issue #301) ---------------------
#
# The transcript now caps a tool call's *arguments* as well as its result — everything replayed per
# wake must be bounded. But the resume re-issues an interrupted platform create **from exactly those
# arguments**, so the cap has one exception, and these pin both halves of it: the arguments the
# recovery still needs survive whole, and they are bounded the moment it no longer needs them.


def test_an_interrupted_creates_body_survives_the_cap_and_is_re_posted_whole(platform, tmp_path):
    """**The trap the naive fix walks into.** A killed create is re-issued byte for byte, at any size.

    Capping a call's arguments the obvious way — elide anything over the cap, like a tool result —
    would truncate the body of the very message the recovery is about to re-post. Under the dead
    wake's idempotency key that is harmless *if* the original POST landed (the platform hands back
    the original record and the elided body is never used) — and if it did **not** land, the elided
    body is what the peer reads, forever. That is worse than the cost it saves, and it is why issue
    #297 left the arguments alone rather than capping them wrong.
    """
    body = "Here is the full report you asked for.\n\n" + ("x" * 200_000) + "\n\nEND-OF-REPORT"
    serve_messages(platform, page(message(uuid=M0, body=MULTILINE)))

    # The dead wake got as far as writing the call and no further: no result, so the POST may or may
    # not have landed. The one genuinely unknowable state — and the arguments are the only record of
    # what it was trying to say.
    crashed_wake_owning(tmp_path, M0)
    session = Harness(_Finishes(), home=tmp_path).session(f"timeline:{TIMELINE_UUID}")
    session.history.append(
        Message(role="user", content=f"[2026-06-04T00:00:00.000Z] john: {MULTILINE}", items=[M0])
    )
    session.history.append(
        Message.assistant(
            tool_calls=[
                ToolCall(id="c1", name="messages", arguments={"action": "create", "body": body})
            ]
        )
    )
    session._save()

    # It is on disk **whole**, over the cap and unelided, because it is still load-bearing.
    (call,) = [c for m in _transcript(tmp_path) for c in m.get("tool_calls", [])]
    assert call["arguments"]["body"] == body

    agent, _ = build_wake(tmp_path, _Finishes(), tools=[MessagesTool()])
    agent.wake()

    assert _posts(platform) == [body], "the recovery re-posted a body the cap had cut down"
    assert _keys(platform) == [key(timeline=TIMELINE_UUID, anchor=M0, kind=MESSAGE, ordinal=1)]

    # And now that the re-issue has settled it, the arguments stop being evidence and start being
    # cost — so the very next save bounds them. The exception is *transient*, never a loophole.
    (call,) = [c for m in _transcript(tmp_path) for c in m.get("tool_calls", [])]
    assert _json_size(call["arguments"]) <= TOOL_ARGS_CAP
    assert call["arguments"]["body"].startswith("Here is the full report")
    assert f"elided from {len(body)} chars" in call["arguments"]["body"]


def test_an_unsettled_create_keeps_its_arguments_across_every_save_until_it_is_re_issued(tmp_path):
    """The "outcome unknown" marker is a **durable** flag, not a one-wake grace period.

    A wake can load an interrupted turn, save the transcript for some other reason, and die again
    before it re-issues anything — `Session.excise` and `Session.note` both write, and a resume can
    lose its take-over race and never run at all. If any of those writes capped the arguments, the
    *next* wake would replay a mangled create. So the cap keys on the marker, and the marker stands
    until a real result replaces it.
    """
    body = "y" * 200_000
    path = tmp_path / "t.json"
    session = Session("timeline:x", Harness(_Finishes()).engine, path=path)
    session.history.append(Message.user("say something long"))
    session.history.append(
        Message.assistant(
            tool_calls=[
                ToolCall(id="c1", name="messages", arguments={"action": "create", "body": body})
            ]
        )
    )
    session.history.append(Message.tool(tool_call_id="c1", content=INTERRUPTED))

    session.note("some unrelated turn wrote the transcript again")

    (call,) = [c for m in json.loads(path.read_text()) for c in m.get("tool_calls", [])]
    assert call["arguments"]["body"] == body


def test_a_reused_tool_call_id_never_makes_an_interrupted_create_look_settled(tmp_path):
    """A call is paired with a result **from its own turn's run** — never by a global id lookup.

    A `tool_call_id` is the provider's own string and nothing normalizes it: a model that numbers its
    calls per response (`call_0`, `call_1` — what an OpenRouter-fronted model emits, and the fleet's
    primary agent is one) reuses the same ids on every turn. Look a call's result up across the whole
    transcript and an **interrupted** create finds the *previous* turn's "posted" — reads as settled,
    has its arguments elided, and the resume then re-posts the peer's message with its body cut out.
    The same trap `heal_interrupted_calls` and `_idempotency.creates` each avoid, in the same way.
    """
    body = "x" * 200_000
    path = tmp_path / "t.json"
    session = Session("timeline:x", Harness(_Finishes()).engine, path=path)
    session.history += [
        Message.user("first"),
        Message.assistant(
            tool_calls=[
                ToolCall(id="call_0", name="messages", arguments={"action": "create", "body": body})
            ]
        ),
        Message.tool(tool_call_id="call_0", content="posted"),  # settled
        Message.assistant(content="Said it."),
        Message.user("second"),
        Message.assistant(  # the SAME id, a turn later — and this one was interrupted
            tool_calls=[
                ToolCall(id="call_0", name="messages", arguments={"action": "create", "body": body})
            ]
        ),
        Message.tool(tool_call_id="call_0", content=INTERRUPTED),
    ]

    session._save()

    settled, unsettled = [c for m in json.loads(path.read_text()) for c in m.get("tool_calls", [])]
    assert _json_size(settled["arguments"]) <= TOOL_ARGS_CAP  # its fate is decided: bounded
    assert unsettled["arguments"]["body"] == body  # still replayable: whole, at any size


def test_an_interrupted_call_the_recovery_will_never_re_run_is_capped_from_the_first_save(tmp_path):
    """The exception covers the four platform creates and **nothing else** — or it is a loophole.

    A `generate_image` whose outcome is unknown is never re-run: no idempotency key can un-spend money
    at fal.ai. So its arguments are dead weight the instant it is interrupted, and they are bounded
    like any other call's. Keeping *every* interrupted call whole would have been the easy rule — and
    it would have left an unbounded blob in the transcript forever, which is the bug, one size smaller.
    """
    prompt = "z" * 200_000
    path = tmp_path / "t.json"
    session = Session("timeline:x", Harness(_Finishes()).engine, path=path)
    session.history.append(Message.user("draw me something"))
    session.history.append(
        Message.assistant(
            tool_calls=[ToolCall(id="c1", name="generate_image", arguments={"prompt": prompt})]
        )
    )
    session.history.append(Message.tool(tool_call_id="c1", content=INTERRUPTED))

    session.note("a later turn writes the transcript")

    (call,) = [c for m in json.loads(path.read_text()) for c in m.get("tool_calls", [])]
    assert _json_size(call["arguments"]) <= TOOL_ARGS_CAP
    assert f"elided from {len(prompt)} chars" in call["arguments"]["prompt"]


# --- the live bug the review found -------------------------------------------


def test_a_multi_line_body_is_found_by_the_classifier(platform, tmp_path, kill_the_finally, caplog):
    """A body with a newline in it used to be invisible to the recovery — **and was, in the field.**

    `_turn_of` matched the message's rendered line against the turn's content *split on newlines*,
    so a two-paragraph body produced a multi-line needle that can never be an element of a list of
    single lines. It matched nothing; the classifier concluded the model had never seen the message;
    it re-drove a turn that had already posted. Every recovery test in the suite used a single-line
    body, which is why it survived.

    The turn now carries the uuids it rendered, so the match is exact — and the body, which is
    **peer-controlled**, is no longer the substrate of a safety decision at all.
    """
    serve_messages(platform, page(message(uuid=M0, body=MULTILINE)))
    first, _ = build_wake(tmp_path, _Speaks())
    with pytest.raises(RuntimeError):
        first.wake()

    serve_messages(platform, page(message(uuid=M0, body=MULTILINE)))
    second, _ = build_wake(tmp_path, _Finishes(), tools=[MessagesTool()])
    with caplog.at_level(logging.WARNING):
        second.wake()

    assert any(r.message.startswith("resuming") for r in caplog.records), (
        "the classifier lost a multi-line message and re-drove a turn that had already posted"
    )
    assert _posts(platform) == ["Here is your owl."]  # once, not twice


def _keys(platform) -> list[str | None]:
    """The `Idempotency-Key` header on every message POST, in order.

    Read from the router's **call log**, never by registering a route: `platform.post(...)` mid-test
    adds a *fresh* route behind the fixture's own, which then never fires — the trap `_posts` in
    `test_wake` is documented against. The call log is inert.
    """
    return [
        call.request.headers.get("idempotency-key")
        for call in platform.calls
        if call.request.method == "POST" and call.request.url.path.endswith("/messages")
    ]


# --- the stall the review found: a take-over that dies must not pin the mark ---


def test_a_take_over_that_died_before_writing_its_claim_is_itself_recoverable(tmp_path):
    """`reclaim()` wins a token, then writes the claim. **A crash between them used to be fatal.**

    Not fatal to the wake — fatal to the *message*, forever. The token was taken and the claim still
    named the dead wake, so every future wake judged that stale owner, tried to reclaim from it, lost
    a token it could never win, and returned `_PENDING`. `_settle` stops at the first `_PENDING`, so
    the high-water mark was **pinned behind that message permanently** and nobody ever answered it.
    A stall is not better than a drop; it *is* a drop, with the cursor stuck behind it.

    It needed no race to reach — an `ENOSPC` on the claim write would do — and it is pre-existing.
    The token now carries its own record, so the wake that comes next can see who really holds the
    item, judge *that* wake, and compete on a fresh token keyed to it.
    """
    store = ClaimStore(tmp_path)
    store.claim(TIMELINE_UUID, M0, kind="messages")
    store._write(
        store._path(TIMELINE_UUID, "messages", M0),
        Claim(phase="in-flight", pid=1, wake="the-original-wake", at=0.0),
    )

    # A recovering wake wins the take-over token — and dies before it can rewrite the claim.
    recoverer = ClaimStore(tmp_path, wake="the-recovering-wake")
    token = recoverer._token(
        recoverer._path(TIMELINE_UUID, "messages", M0), M0, "the-original-wake"
    )
    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text(
        json.dumps({"phase": "in-flight", "pid": 1, "wake": "the-recovering-wake", "at": 0.0})
    )

    # The next wake reads a claim that still names the *original* owner — and must not be fooled.
    third = ClaimStore(tmp_path, wake="the-third-wake")
    claim = third.read(TIMELINE_UUID, M0, kind="messages")
    effective = third.effective_owner(TIMELINE_UUID, M0, kind="messages", claim=claim)

    assert effective.wake == "the-recovering-wake"  # the token says who really held it
    assert third.orphaned(effective)  # and that wake is dead too, so the item is still recoverable
    # It competes on a token keyed to the *real* owner, which is free — so recovery is repeatable
    # rather than a one-shot that strands the message the first time a recoverer dies.
    assert third.reclaim(TIMELINE_UUID, M0, kind="messages", owner=effective.wake) is True
    assert third.read(TIMELINE_UUID, M0, kind="messages").wake == "the-third-wake"


def test_only_one_wake_can_take_an_orphan_over_from_the_same_owner(tmp_path):
    """The take-over is still a compare-and-swap: two recovering wakes, one winner."""
    store = ClaimStore(tmp_path)
    store.claim(TIMELINE_UUID, M0, kind="messages")
    store._write(
        store._path(TIMELINE_UUID, "messages", M0),
        Claim(phase="in-flight", pid=1, wake="the-dead-wake", at=0.0),
    )

    a = ClaimStore(tmp_path, wake="wake-a")
    b = ClaimStore(tmp_path, wake="wake-b")

    assert a.reclaim(TIMELINE_UUID, M0, kind="messages", owner="the-dead-wake") is True
    assert b.reclaim(TIMELINE_UUID, M0, kind="messages", owner="the-dead-wake") is False
    assert store.read(TIMELINE_UUID, M0, kind="messages").wake == "wake-a"


def test_a_claim_is_never_observable_empty(tmp_path):
    """A claim is born carrying its record — an empty one is the legacy `done` sentinel."""
    store = ClaimStore(tmp_path)
    assert store.claim(TIMELINE_UUID, M0, kind="messages") is True

    claim = store.read(TIMELINE_UUID, M0, kind="messages")
    assert claim.phase == "in-flight" and claim.wake == store.wake


# --- what the second adversarial pass found, and would have shipped without ---


def test_a_resumed_turns_narration_lands_in_its_own_turn_not_at_the_end(tmp_path):
    """**The silent drop the review caught.** A resume must splice its work into the turn it finishes.

    A resume can fail (the provider blips), and the wake goes on to answer newer messages — leaving
    an *older* turn unfinished **behind** a newer one. Appending the eventual continuation to the end
    of the transcript then files an old turn's narration under the newer turn. The recovery reads it
    as the newer turn's own terminal text, commits a message that was never answered, and lets the
    high-water mark sail past it. Gone, silently — the exact failure class this issue exists to end,
    manufactured by the machinery built to prevent it.
    """

    class Finishes:
        def chat(self, messages, tools=None):
            return Message.assistant(content="Finishing turn A.")

    session = Harness(Finishes(), home=tmp_path).session("timeline:t1")
    turn_a = Message(role="user", content="the first question", items=["m-1"])
    session.history.extend(
        [
            turn_a,
            Message.assistant(tool_calls=[ToolCall(id="c1", name="noop", arguments={})]),
            Message.tool(tool_call_id="c1", content="ok"),
            # A *newer* turn, complete, sitting after the unfinished one.
            Message(role="user", content="the second question", items=["m-2"]),
            Message.assistant(content="Answered the second."),
        ]
    )

    session.resume(turn_a)

    roles = [(m.role, (m.content or "")[:24]) for m in session.history]
    assert roles[-1] == ("assistant", "Answered the second.")[:2] or roles[-1][1] == (
        "Answered the second."
    ), f"the continuation was appended to the end instead of spliced into turn A: {roles}"
    # Turn A now ends on its own narration; turn B still ends on its own.
    assert turn_work(session.history, turn_a)[-1].content == "Finishing turn A."
    second = next(m for m in session.history if m.items == ["m-2"])
    assert turn_work(session.history, second)[-1].content == "Answered the second."


def test_a_turn_the_hook_was_still_extending_is_not_read_as_finished(tmp_path):
    """A terminal narration is the **last** thing in a turn's work, not merely present in it.

    `Engine.run` returns on `not reply.tool_calls and not extend` — and *both* shipped turn hooks
    extend on exactly that shape (the mention informer nudges an agent that was addressed and did
    nothing; the code bridge harvests a run's files and feeds their uuids back). Scanning backwards
    for "an assistant turn with text and no tool calls" therefore finds a turn that was still being
    extended and calls it finished: the claim settles, the mark advances, and no wake ever looks at
    that message again.
    """
    # The shape a wake killed *after* the informer's nudge leaves behind.
    work = [
        Message.system("Step 1 of 24."),
        Message.assistant(content="I could reply here."),
        Message.system("You were mentioned and this turn is ending with nothing said..."),
    ]
    assert _turn_narration(work) is None, "a turn the hook was still extending read as finished"

    # And the genuine article — the work *ends* on the model's own final text.
    assert _turn_narration(work + [Message.assistant(content="Done.")]) == "Done."

    # A turn that *raised* ends on the failure marker, not a narration: finish it, never file it.
    failed = [
        Message.assistant(content="halfway thought"),
        Message.system("[turn failed: RuntimeError — boom]"),
    ]
    assert _turn_narration(failed) is None


def test_a_dead_turn_carrying_a_batch_is_resumed_once_not_once_per_message(platform, tmp_path):
    """A dead wake's turn answered a *batch*. Finishing it once answers all of them."""
    serve_messages(
        platform,
        page(message(uuid=M1, body="and another thing"), message(uuid=M0, body=MULTILINE)),
    )
    crashed_wake_owning(tmp_path, M0, M1)

    brain = _Finishes()
    session = Harness(brain, home=tmp_path).session(f"timeline:{TIMELINE_UUID}")
    session.history.append(Message(role="user", content="[t] john: two questions", items=[M0, M1]))
    session.history.append(
        Message.assistant(tool_calls=[ToolCall(id="c1", name="noop", arguments={})])
    )
    session.history.append(Message.tool(tool_call_id="c1", content="ok"))
    session._save()

    agent, live = build_wake(tmp_path, _Finishes(), tools=[MessagesTool()])
    agent.wake()

    # ONE model call: the turn was finished once, for both messages — not once per message.
    assert len(live.seen) == 1
    claims = ClaimStore(tmp_path)
    assert claims.read(TIMELINE_UUID, M0, kind="messages").phase == "done"
    assert claims.read(TIMELINE_UUID, M1, kind="messages").phase == "done"
