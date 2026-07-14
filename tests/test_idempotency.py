"""Deterministic `Idempotency-Key`s: the same number, computed twice, by two different wakes.

Issue #297. A key is only useful if the wake that *dies* and the wake that *recovers it* derive the
same one — otherwise the platform has never seen it, and the "safe" re-issue posts the peer's answer
a second time. Everything here exists to make that equality true by construction rather than by
agreement.

The ordinal is the whole trick, and it is read **off the transcript**, never off a counter:

    ordinal = 1 + (create calls of this kind in this turn that already have a result)

At the instant a tool call runs, the create-shaped calls that have results are exactly the ones
*earlier in call order* — the engine appends each result before dispatching the next call. So
counting answered calls while the turn runs (`completed`) and counting positions after a crash
(`ordinal_of`) are the same number. A counter would be a second source of truth, and it would drift.
"""

from __future__ import annotations

import uuid as uuidlib

from basecradle_harness import Message, ToolCall
from basecradle_harness._idempotency import (
    ASSET,
    MESSAGE,
    NAMESPACE,
    TASK,
    WEBHOOK_ENDPOINT,
    completed,
    create_kind,
    creates,
    interrupted,
    key,
)

INTERRUPTED = "[Interrupted: ...]"


def ordinal_of(work, call_id):
    """The `(kind, ordinal)` of `call_id` — read off the one walk, exactly as the recovery does."""
    for c in creates(work):
        if c.call.id == call_id:
            return (c.kind, c.ordinal)
    return None


TIMELINE = "019e7750-66ee-7f53-829f-13a8a710b6da"
ANCHOR = "019e7751-4a1b-7c2d-8e3f-1a2b3c4d5e6f"


def _create_call(call_id: str, tool: str = "messages") -> ToolCall:
    return ToolCall(id=call_id, name=tool, arguments={"action": "create", "body": "hi"})


# --- the key itself ----------------------------------------------------------


def test_the_key_is_a_pure_function_of_the_four_ingredients():
    """Same inputs → same key, in any process, on any box, forever. That is the entire contract."""
    once = key(timeline=TIMELINE, anchor=ANCHOR, kind=MESSAGE, ordinal=1)
    again = key(timeline=TIMELINE, anchor=ANCHOR, kind=MESSAGE, ordinal=1)

    assert once == again
    assert once == str(uuidlib.uuid5(NAMESPACE, f"{TIMELINE}:{ANCHOR}:{MESSAGE}:1"))


def test_every_ingredient_changes_the_key():
    """Each of the four is load-bearing — collapse any one and two distinct creates collide.

    A collision is not a near-miss: the platform returns the *first* record for the second create,
    so the agent's second message is silently swallowed while it is told it posted.
    """
    base = key(timeline=TIMELINE, anchor=ANCHOR, kind=MESSAGE, ordinal=1)

    assert key(timeline="other", anchor=ANCHOR, kind=MESSAGE, ordinal=1) != base
    assert key(timeline=TIMELINE, anchor="other", kind=MESSAGE, ordinal=1) != base
    assert key(timeline=TIMELINE, anchor=ANCHOR, kind=ASSET, ordinal=1) != base
    assert key(timeline=TIMELINE, anchor=ANCHOR, kind=MESSAGE, ordinal=2) != base


def test_the_namespace_is_a_constant_not_a_knob():
    """Changing it silently re-mints every key, so a recovery stops matching and duplicates."""
    assert NAMESPACE == uuidlib.UUID("9f2b0c4e-6a1d-4f7e-9c3b-8d5a2e1f0b74")


# --- which calls are creates -------------------------------------------------


def test_the_four_creates_are_recognized_from_the_recorded_call():
    """Read from the call's *persisted* name and arguments, so a reloaded transcript answers the
    same question the live call did."""
    assert create_kind(_create_call("c1", "messages")) == MESSAGE
    assert create_kind(_create_call("c1", "assets")) == ASSET
    assert create_kind(_create_call("c1", "tasks")) == TASK
    assert create_kind(_create_call("c1", "webhook_endpoints")) == WEBHOOK_ENDPOINT


def test_a_read_is_not_a_create_and_neither_is_an_unrelated_tool():
    assert create_kind(ToolCall(id="c1", name="messages", arguments={"action": "list"})) is None
    assert create_kind(ToolCall(id="c1", name="memory", arguments={"action": "create"})) is None
    assert create_kind(ToolCall(id="c1", name="generate_image", arguments={})) is None


def test_a_generated_image_consumes_no_ordinal():
    """It uploads an Asset, but it is not an `assets create` call — and that is deliberate.

    A recovery never re-issues a `generate_image` (no key can un-spend money at fal.ai), so keying
    its upload would buy nothing and cost the one thing that matters: the ordinal count must see
    exactly the creates the transcript's *tool calls* describe, or the two halves disagree.
    """
    work = [
        Message.assistant(tool_calls=[ToolCall(id="c1", name="generate_image", arguments={})]),
        Message.tool(tool_call_id="c1", content="an image"),
        Message.assistant(tool_calls=[_create_call("c2")]),
        Message.tool(tool_call_id="c2", content="posted"),
    ]

    assert ordinal_of(work, "c2") == (MESSAGE, 1)  # the first *message* create, not the second
    assert ordinal_of(work, "c1") is None


# --- the two halves of the ordinal must agree --------------------------------


def test_the_live_count_and_the_replay_count_are_the_same_number():
    """`completed` (while the turn runs) and `ordinal_of` (after it dies) must never disagree.

    Simulated exactly as the engine produces it: each call's result is appended before the next call
    is dispatched, so at the moment call N runs, calls 1..N-1 are answered and N is not.
    """
    calls = [_create_call(f"c{i}") for i in range(1, 4)]
    work: list[Message] = []

    for index, call in enumerate(calls, start=1):
        work.append(Message.assistant(tool_calls=[call]))
        # The live ordinal, computed the instant this call runs — before its own result exists.
        assert completed(work, MESSAGE) + 1 == index
        work.append(Message.tool(tool_call_id=call.id, content="posted"))

    # And after a crash, read back off the finished transcript: the same numbers.
    for index, call in enumerate(calls, start=1):
        assert ordinal_of(work, call.id) == (MESSAGE, index)


def test_a_create_call_that_never_reached_the_tool_still_consumes_its_ordinal():
    """**The case a counter gets wrong.**

    The model passes an unknown kwarg; `run(**arguments)` raises `TypeError`; the engine feeds the
    error back as the call's result. The tool's create branch was never entered, so a counter would
    not have incremented — but the *transcript* records the call, and the transcript is what the
    recovery counts. The counter would say 1 where the transcript says 2, the recovery would re-issue
    under a key the platform has never seen, and the peer would be answered twice.

    Counting answered calls makes both halves read the same evidence, so they cannot drift.
    """
    work = [
        Message.assistant(tool_calls=[_create_call("c1")]),
        Message.tool(tool_call_id="c1", content="Error running 'messages': unexpected kwarg"),
        Message.assistant(tool_calls=[_create_call("c2")]),
    ]

    assert completed(work, MESSAGE) + 1 == 2  # the live mint for c2
    assert ordinal_of(work, "c2") == (MESSAGE, 2)  # and the replay agrees


def test_kinds_are_counted_separately():
    """A turn that posts a message *and* uploads an asset keys each from its own sequence.

    This is what lets a resumed model do its work in a different order and still dedupe both.
    """
    work = [
        Message.assistant(tool_calls=[_create_call("c1", "assets")]),
        Message.tool(tool_call_id="c1", content="uploaded"),
        Message.assistant(tool_calls=[_create_call("c2", "messages")]),
        Message.tool(tool_call_id="c2", content="posted"),
    ]

    assert ordinal_of(work, "c1") == (ASSET, 1)
    assert ordinal_of(work, "c2") == (MESSAGE, 1)


def test_parallel_tool_calls_in_one_assistant_turn_are_ordered_by_call_order():
    """A model may ask for two creates at once; the engine still runs them one at a time."""
    work = [
        Message.assistant(tool_calls=[_create_call("c1"), _create_call("c2")]),
        Message.tool(tool_call_id="c1", content="posted"),
        Message.tool(tool_call_id="c2", content="posted"),
    ]

    assert ordinal_of(work, "c1") == (MESSAGE, 1)
    assert ordinal_of(work, "c2") == (MESSAGE, 2)


def test_a_reused_tool_call_id_does_not_confuse_the_pairing():
    """A model that numbers its calls per response reuses ids across turns (issue #297).

    Tool-call ids come straight off the wire — nothing normalizes them — and an OpenRouter-fronted
    model routinely emits `call_0`, `call_1`, per *response*. Pairing calls to results by a global
    id lookup then matches the **previous** turn's result: the interrupted call looks answered (so
    it is never healed, and the provider 400s on that transcript forever) and the live ordinal runs
    one ahead of the recovery's (so the re-issue mints a key the platform has never seen, and the
    peer is answered twice).

    Pairing within the issuing assistant turn's own run is what makes both halves right.
    """
    work = [
        Message.assistant(tool_calls=[_create_call("call_0")]),
        Message.tool(tool_call_id="call_0", content="posted"),
        Message.system("Step 2 of 24."),
        Message.assistant(tool_calls=[_create_call("call_0")]),  # the SAME id, a new turn
        # ...and killed here: this second call has no result.
    ]

    found = creates(work)
    assert [(c.kind, c.ordinal, c.result is not None) for c in found] == [
        (MESSAGE, 1, True),
        (MESSAGE, 2, False),  # the live call is NOT answered by its namesake two turns ago
    ]
    assert completed(work, MESSAGE) == 1  # so the ordinal it mints is 2 — which is the truth


def test_interrupted_finds_only_the_creates_whose_result_is_the_marker():
    work = [
        Message.assistant(tool_calls=[_create_call("c1")]),
        Message.tool(tool_call_id="c1", content="posted"),
        Message.assistant(tool_calls=[_create_call("c2"), _create_call("c3")]),
        Message.tool(tool_call_id="c2", content=INTERRUPTED),
        Message.tool(tool_call_id="c3", content=INTERRUPTED),
    ]

    found = interrupted(work, INTERRUPTED)

    assert [(c.call.id, c.ordinal) for c in found] == [("c2", 2), ("c3", 3)]
