"""Prompt caching as a declared adapter capability — so no agent silently pays full freight.

Caching is the difference between paying full price for a standing agent's transcript on every
wake and paying the cache-read rate for it (~5.4× cheaper, measured live on @glm-5.2). The harness
already *observes* it — ``cached_tokens=`` rides the per-call log line (issue #274) — but observing
is not the same as **reaching** it, and how a cache is reached differs by vendor in a way that is
not cosmetic:

- **automatic** — the endpoint caches a repeated prefix by itself, with nothing on the wire. OpenAI,
  xAI, and (verified live) OpenRouter's GLM endpoints. The engine does nothing.
- **explicit** — the client must *mark* the cacheable prefix, or it gets **nothing at all**.
  Anthropic is the one that matters: a Claude agent shipped without breakpoints pays full freight on
  every token of every wake, silently, forever. Nothing errors; the bill just arrives.
- **none** — the endpoint has no prompt cache. The engine does nothing.

The asymmetry is the whole point: *automatic* and *none* fail safe (do nothing, lose nothing), and
*explicit* fails **expensive and invisible**. So the mode is a thing an adapter **declares**, not a
thing the engine guesses, and the standing rule (`CLAUDE.md` → Provider Capabilities) is that no new
adapter ships without declaring one. Read as a capability, never a vendor branch: the engine asks
every adapter the same question and does exactly one thing with the answer.

Where the breakpoint goes, and why it is the same boundary the cache already turns on
-------------------------------------------------------------------------------------
The message list is already built stable-prefix-first, volatile-tail-last, precisely so a provider's
prefix cache pays out (`CLAUDE.md` → Context Discipline; `_session.Session._exchange`):

    [ ...frozen transcript... ][ per-wake brief ][ newest user turn ]
                              ↑
                     the anchor goes here

Everything left of that line is byte-identical to what the last wake sent, and everything right of it
was recomposed *this* wake. So the boundary an explicit breakpoint wants is the boundary that already
exists — the last message of the frozen transcript — and marking it needs no new notion of what is
stable. Anchoring any further right would write a cache entry over the brief (a snapshot of a moment,
different on the next wake) and buy a cache write that can never be read.

Two constraints from the wire, both load-bearing:

- **A breakpoint rides a *content block*, not a plain string**, so the anchored turn's content is
  emitted as a one-element block list carrying ``cache_control`` (`_openai_wire`). This is why the
  anchor is a mark on a `Message` rather than a request-level field: only the adapter knows the shape.
- **It is a Chat-Completions/Messages-wire feature.** The Responses API does not expose per-block
  breakpoints at all, so an adapter reaching an explicit-cache model over the Responses surface
  cannot place one and must not claim it can.

The anchor is **copy-on-write**: it is stamped onto a *copy* of the turn, never onto the object the
session persists. A `cache_anchor` written into the stored history would still be there on the next
wake, when that turn is no longer the boundary — and the wake after that would add another, walking
straight into the vendor's four-breakpoint ceiling. Transient by construction beats remembering to
clear it.
"""

from __future__ import annotations

from dataclasses import replace

from basecradle_harness._messages import Message

#: The endpoint caches a repeated prefix on its own; nothing goes on the wire.
AUTOMATIC = "automatic"
#: The client must mark the cacheable prefix or there is no caching at all (Anthropic).
EXPLICIT = "explicit"
#: The endpoint has no prompt cache.
NONE = "none"

#: The declared modes. An adapter's ``cache_mode`` is one of these, and the engine's behavior is a
#: function of it alone — there is no ``if provider == ...`` anywhere above the adapter layer.
CACHE_MODES = (AUTOMATIC, EXPLICIT, NONE)


def cache_mode(provider: object) -> str:
    """The adapter's declared cache mode — `AUTOMATIC` when it declares none, or declares nonsense.

    Absent (a third-party adapter written before this capability existed) resolves to `AUTOMATIC`,
    which is the same **do nothing** the engine already did — so an old adapter keeps working
    untouched, and the capability stays a question rather than a contract. `AUTOMATIC` is also the
    right answer for an *unrecognized* value: the only thing a mode can make the engine do is put a
    vendor field on the wire, and doing that on a typo — to an endpoint that never asked for it — is
    a 400 on every wake. Failing closed here means the worst case of a bad declaration is the status
    quo (no breakpoint), never a broken agent.
    """
    declared = getattr(provider, "cache_mode", None)
    return declared if declared in CACHE_MODES else AUTOMATIC


def anchor_cacheable_prefix(messages: list[Message], *, stable: int, mode: str) -> list[Message]:
    """Mark the end of the cacheable prefix, for an `EXPLICIT` provider only.

    `stable` is the count of messages that are byte-identical to what the previous wake sent — the
    frozen transcript, everything left of the per-wake brief. The anchor lands on the **last** of
    them (index ``stable - 1``), which is the stable/volatile boundary the module docstring draws.

    Returns a list with that one turn **replaced by an anchored copy**; the caller's `Message`
    objects are never mutated, so nothing leaks into the persisted transcript (see the docstring —
    a persisted anchor accumulates across wakes and eventually trips the four-breakpoint ceiling).

    On an agent's *first* wake the frozen prefix is just the charter — and anchoring it is the
    point, not an edge case: the charter (system prompt + operating guidance) is the single largest
    byte-stable block an agent has, and caching it on wake one is what makes wake two a cache read.
    Only a session with no charter at all has nothing to anchor (`stable <= 0`).

    A no-op for every mode but `EXPLICIT`.
    """
    if mode != EXPLICIT:
        return messages
    at = _anchorable(messages, stable)
    if at is None:
        return messages
    anchored = list(messages)
    anchored[at] = replace(anchored[at], cache_anchor=True)
    return anchored


def _anchorable(messages: list[Message], stable: int) -> int | None:
    """The last index of the frozen prefix that can actually *carry* a breakpoint, or ``None``.

    A breakpoint rides a **text content block**, so a turn with no text to hang one on cannot hold
    it, and two kinds are skipped:

    - a **`tool` turn** — on the chat wire its content is a bare string keyed to its
      ``tool_call_id``, with no content-block form at all; and
    - a turn with **no content** — an assistant turn that is purely tool calls carries ``None``
      content (the wire's explicit null), and an empty string is no better.

    Skipping them is not fussiness. A breakpoint aimed at either would be **silently dropped** by
    `_openai_wire.chat_message_to_wire`, and a dropped breakpoint on an explicit-cache vendor is
    full freight on the *entire* transcript, forever, with nothing raised and no log line changed —
    precisely the invisible bill this module exists to prevent. So the anchor walks back to the
    nearest turn that can genuinely hold it rather than aiming at one that cannot.

    In practice a frozen prefix ends with the assistant's text reply (or the engine's own note), so
    the walk-back is the guard for the case that isn't practice — a wake that failed mid-chain, a
    transcript compacted to an odd tail.
    """
    for index in reversed(range(min(stable, len(messages)))):
        message = messages[index]
        if message.role != "tool" and message.content:
            return index
    return None
