"""Prompt caching as a declared adapter capability (issue #277).

The failure this guards against is the quiet one: an explicit-cache vendor (Anthropic) returns a
perfectly good answer whether or not you marked the cacheable prefix — it just bills you full
freight, forever, if you didn't. Nothing raises; no log line changes. So these tests pin the two
halves that make the mode *mechanical* rather than remembered: what each adapter **declares**, and
where the engine puts the one breakpoint that declaration earns.

No Anthropic adapter ships yet, so `ExplicitProvider` stands in for one — which is exactly the
point: the placement is a property of the engine, not of any vendor, and it is provable with no key
and no network.
"""

import pytest

from basecradle_harness import Harness, Message, ToolCall
from basecradle_harness._caching import (
    AUTOMATIC,
    CACHE_MODES,
    EXPLICIT,
    NONE,
    anchor_cacheable_prefix,
    cache_mode,
)
from basecradle_harness._openai import OpenAIProvider
from basecradle_harness._openai_wire import chat_message_to_wire
from basecradle_harness._openrouter import OpenRouterProvider
from basecradle_harness._xai_sdk import XaiSdkProvider


class ScriptedProvider:
    """A `Provider` that replays prepared assistant turns and records what it was shown.

    Each snapshot is `(role, content, cache_anchor)` per message **at chat time** — the question
    these tests ask is what the model was actually sent, which only a snapshot can answer (the
    session goes on to mutate the live `Message` objects afterwards).
    """

    cache_mode = AUTOMATIC

    def __init__(self, *replies: Message) -> None:
        self._replies = list(replies)
        self.snapshots: list[list[tuple[str, str, bool]]] = []

    def chat(self, messages, tools=None):
        self.snapshots.append([(m.role, m.content or "", m.cache_anchor) for m in messages])
        if not self._replies:
            raise AssertionError("ScriptedProvider ran out of replies")
        return self._replies.pop(0)


class ExplicitProvider(ScriptedProvider):
    """The Anthropic-shaped adapter that does not exist yet: no caching unless the client marks it."""

    cache_mode = EXPLICIT


def text(content: str) -> Message:
    return Message.assistant(content=content)


def anchored(snapshot: list[tuple[str, str, bool]]) -> list[tuple[str, str]]:
    """The `(role, content)` of every turn carrying a breakpoint, in order."""
    return [(role, content) for role, content, anchor in snapshot if anchor]


# --- The declaration ---------------------------------------------------------


def test_every_shipped_adapter_declares_a_cache_mode():
    """The standing rule, made mechanical: no adapter ships without declaring one.

    This is the test that fails when someone adds an adapter and forgets — which is the whole point
    of the rule, because forgetting on an explicit-cache vendor costs money and says nothing.
    """
    for adapter in (OpenAIProvider, XaiSdkProvider, OpenRouterProvider):
        assert adapter.cache_mode in CACHE_MODES, adapter.__name__


def test_the_shipped_adapters_declare_automatic():
    """Today's endpoints all cache by themselves — so the engine must put nothing on their wire."""
    for adapter in (OpenAIProvider, XaiSdkProvider, OpenRouterProvider):
        assert adapter.cache_mode == AUTOMATIC, adapter.__name__


@pytest.mark.parametrize(
    ("declared", "resolved"),
    [
        (EXPLICIT, EXPLICIT),
        (AUTOMATIC, AUTOMATIC),
        (NONE, NONE),
        (None, AUTOMATIC),  # an adapter written before the capability existed
        ("Explicit", AUTOMATIC),  # a typo must never silently arm the wire
        ("cache-please", AUTOMATIC),
    ],
)
def test_cache_mode_resolves_and_fails_closed(declared, resolved):
    """An undeclared or unrecognized mode reads as `automatic` — the do-nothing default.

    Failing *closed* matters because the only thing a mode can make the engine do is put a vendor
    field on the wire; doing that on a typo, at an endpoint that never asked for it, is a 400 on
    every wake. The worst case of a bad declaration must be the status quo, never a broken agent.
    """

    class Adapter:
        def chat(self, messages, tools=None): ...  # pragma: no cover - never called

    adapter = Adapter()
    if declared is not None:
        adapter.cache_mode = declared

    assert cache_mode(adapter) == resolved


# --- Where the breakpoint lands ----------------------------------------------


def test_no_anchor_for_an_automatic_provider():
    """The shipped path is untouched: an automatic provider is never handed a breakpoint."""
    provider = ScriptedProvider(text("one"), text("two"))
    session = Harness(provider, system_prompt="be terse").session("timeline:x")

    session.send("first", brief="BRIEF-A")
    session.send("second", brief="BRIEF-B")

    assert all(anchored(snapshot) == [] for snapshot in provider.snapshots)


def test_the_anchor_lands_on_the_last_frozen_turn_ahead_of_the_brief():
    """The breakpoint marks the stable/volatile boundary — the same one the brief is spliced at.

    On the second wake the model is sent (the engine's step-counter notes elided):

        [charter][user "first"][assistant "one"] │ [brief][user "second"]
                                     ↑ anchor      └── recomposed this wake

    Everything left of the bar is byte-identical to what the last wake sent, so it is exactly what
    is worth caching. The brief is a snapshot of a *moment*: caching through it would buy a cache
    write that can never be read.
    """
    provider = ExplicitProvider(text("one"), text("two"))
    session = Harness(provider, system_prompt="be terse").session("timeline:x")

    session.send("first", brief="BRIEF-A")
    session.send("second", brief="BRIEF-B")

    wake = provider.snapshots[-1]
    # Exactly one breakpoint, and it is the previous wake's assistant reply — the last frozen turn.
    assert anchored(wake) == [("assistant", "one")]
    # Everything the brief and the newest turn contributed stays volatile, behind the anchor.
    at = [i for i, (_, _, anchor) in enumerate(wake) if anchor][0]
    assert ("system", "BRIEF-B", False) in wake[at + 1 :]
    assert ("user", "second", False) in wake[at + 1 :]


def test_the_first_wake_anchors_the_charter():
    """The charter is frozen content from the very first wake, and it is the largest byte-stable
    block an agent has (system prompt + operating guidance) — so it is cached from turn one, which
    is what makes turn two a cache *read* rather than another full-freight write.
    """
    provider = ExplicitProvider(text("one"))
    session = Harness(provider, system_prompt="be terse").session("timeline:x")

    session.send("first", brief="BRIEF-A")

    assert anchored(provider.snapshots[0]) == [("system", "be terse")]


def test_a_session_with_no_charter_has_nothing_to_anchor_on_its_first_turn():
    """With no charter there is no frozen prefix at all on wake one, so mark nothing — a breakpoint
    there would buy a cache write over content no later wake can match."""
    provider = ExplicitProvider(text("one"))
    session = Harness(provider).session("timeline:x")  # no system_prompt

    session.send("first", brief="BRIEF-A")

    assert anchored(provider.snapshots[0]) == []


def test_the_anchor_never_persists_and_never_accumulates():
    """The bug this forecloses: an anchor written into the *stored* transcript is still there on the
    next wake, when that turn is no longer the boundary — and each wake adds another, walking
    straight into the vendor's four-breakpoint ceiling.

    So the anchor is stamped on a copy: the persisted history carries none, and every wake sends
    exactly one no matter how long the conversation runs.
    """
    provider = ExplicitProvider(text("one"), text("two"), text("three"), text("four"))
    session = Harness(provider, system_prompt="be terse").session("timeline:x")

    for turn in ("first", "second", "third", "fourth"):
        session.send(turn, brief=f"BRIEF-{turn}")

    assert [len(anchored(snapshot)) for snapshot in provider.snapshots] == [1, 1, 1, 1]
    # And nothing leaked into what is stored — and therefore replayed, and re-paid for, forever.
    assert not any(m.cache_anchor for m in session.history)


def test_the_anchor_is_not_serialized_into_the_transcript():
    """`cache_anchor` is a property of the request, never of the conversation — so a session file
    written by one wake can never resurrect a stale breakpoint on the next."""
    assert "cache_anchor" not in Message(role="user", content="hi", cache_anchor=True).to_dict()
    assert Message.from_dict({"role": "user", "content": "hi"}).cache_anchor is False


# --- The wire ----------------------------------------------------------------


def test_an_anchored_turn_carries_the_breakpoint_as_a_content_block():
    """A breakpoint rides a *content block*, never a bare string — so an anchored turn is emitted as
    a one-element block list carrying `cache_control`. Everything through it is the cached prefix.
    """
    wire = chat_message_to_wire(Message(role="assistant", content="frozen", cache_anchor=True))

    assert wire["content"] == [
        {"type": "text", "text": "frozen", "cache_control": {"type": "ephemeral"}}
    ]


def test_an_unanchored_turn_is_unchanged_on_the_wire():
    """Every shipped adapter declares `automatic`, so nothing they send changes shape: a plain string
    stays a plain string, and no endpoint sees a field it never asked for."""
    assert chat_message_to_wire(Message(role="assistant", content="frozen"))["content"] == "frozen"


# --- The placement rule, directly --------------------------------------------


@pytest.mark.parametrize("mode", [AUTOMATIC, NONE])
def test_anchor_is_a_no_op_for_every_mode_but_explicit(mode):
    messages = [Message.system("charter"), Message.user("hi")]

    result = anchor_cacheable_prefix(messages, stable=1, mode=mode)

    assert result is messages
    assert not any(m.cache_anchor for m in result)


def test_anchor_leaves_the_callers_messages_untouched():
    """Copy-on-write: the caller's `Message` objects are the ones the session persists."""
    messages = [Message.system("charter"), Message.user("hi")]

    result = anchor_cacheable_prefix(messages, stable=1, mode=EXPLICIT)

    assert result[0].cache_anchor is True
    assert result[0] is not messages[0]
    assert messages[0].cache_anchor is False  # the original was not mutated


@pytest.mark.parametrize("stable", [0, -1])
def test_anchor_declines_when_there_is_no_frozen_prefix(stable):
    messages = [Message.system("charter"), Message.user("hi")]

    result = anchor_cacheable_prefix(messages, stable=stable, mode=EXPLICIT)

    assert not any(m.cache_anchor for m in result)


def test_the_anchor_walks_back_past_a_tool_result():
    """A tool result's content is a bare string keyed to its `tool_call_id` — it has no content-block
    form to hang a `cache_control` on, so a breakpoint landing there would be **silently dropped**,
    and a dropped breakpoint on an explicit-cache vendor is full freight on the whole transcript.

    So the anchor walks back to the nearest turn that can actually carry it.
    """
    messages = [
        Message.system("charter"),
        Message.assistant(content="thinking"),
        Message.tool("call-1", "tool output"),
        Message.user("hi"),
    ]

    result = anchor_cacheable_prefix(messages, stable=3, mode=EXPLICIT)

    assert [m.cache_anchor for m in result] == [False, True, False, False]


def test_the_anchor_walks_back_past_a_contentless_tool_call_turn():
    """An assistant turn that is *purely* tool calls carries `None` content — the wire's explicit
    null — so there is no text block to hang a `cache_control` on and a breakpoint there would be
    silently dropped. Walk back to the last turn that actually has text.
    """
    messages = [
        Message.system("charter"),
        Message.assistant(content="thinking"),
        Message.assistant(tool_calls=[ToolCall(id="call-1", name="memory", arguments={})]),
        Message.user("hi"),
    ]

    result = anchor_cacheable_prefix(messages, stable=3, mode=EXPLICIT)

    assert [m.cache_anchor for m in result] == [False, True, False, False]


def test_anchor_declines_a_prefix_that_can_carry_nothing():
    """Nothing in the frozen prefix can hold a breakpoint — mark nothing, rather than one the wire
    would drop on the floor and leave the agent paying full freight in silence."""
    messages = [Message.tool("call-1", "tool output"), Message.user("hi")]

    assert not any(
        m.cache_anchor for m in anchor_cacheable_prefix(messages, stable=1, mode=EXPLICIT)
    )


def test_a_contentless_anchor_never_reaches_the_wire_as_an_empty_block():
    """The guard's whole purpose, stated at the wire: a turn with no text emits no cache block."""
    tool_calls = [ToolCall(id="call-1", name="memory", arguments={})]
    wire = chat_message_to_wire(Message(role="assistant", tool_calls=tool_calls, cache_anchor=True))

    assert wire["content"] is None
