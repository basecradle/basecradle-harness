"""Prompt caching as a declared adapter capability — so no agent silently pays full freight.

Caching is the difference between paying full price for a standing agent's transcript on every
wake and paying the cache-read rate for it (~5.4× cheaper, measured live on @glm-5.2). The harness
already *observes* it — ``cached_tokens=`` rides the per-call log line (issue #274) — but observing
is not the same as **reaching** it, and how a cache is reached differs by vendor in a way that is
not cosmetic:

- **automatic** — the endpoint caches a repeated prefix by itself, so nothing has to *mark* it.
  OpenAI, xAI, and (verified live) most of OpenRouter's GLM endpoints. The engine places no
  breakpoint. (*Reaching* that cache can still take a routing key — see `bind_conversation` below;
  the mode answers what must be **marked**, never what must be **sent**.)
- **explicit** — the client must *mark* the cacheable prefix, or it gets **nothing at all**.
  Anthropic is the one that matters: a Claude agent shipped without breakpoints pays full freight on
  every token of every wake, silently, forever. Nothing errors; the bill just arrives.
- **none** — the endpoint has no prompt cache. The engine does nothing.

``automatic`` behind a router is a weaker promise than it looks (issue #372)
---------------------------------------------------------------------------
The mode answers *"what must the client put on the wire?"* — and for a router the honest answer is
still **nothing**. It does not answer *"will there be a hit?"*, and behind a router those two come
apart in a way worth stating once: caching is a property of **the upstream that serves the call**,
so a hit needs consecutive calls to reach *the same* upstream, and which upstream that is belongs to
the router, not the client. Measured on ``z-ai/glm-5.2`` across 33 endpoints: most cache a repeated
prefix in full (StreamLake, Z.AI, SiliconFlow, AtlasCloud, Alibaba, BaseTen, Chutes — ~99.9% of a
287 K-token prefix, at ~5.4× cheaper), some cache **nothing at all** (Novita, DeepInfra), and one
caches about half (Fireworks). Whether a given wake pays full freight is therefore decided by where
it lands, and the engine has no honest lever on that:

- OpenRouter's **sticky routing** pins a conversation to one upstream, but its *implicit* form
  "only activates after a cache hit is detected" — which a cold first call on a fresh endpoint can
  never produce. Nothing in the message list breaks that circle.
- Passing an explicit ``session_id`` was **measured and rejected** as a fix: it pins eagerly, which
  makes a landing on a *non-caching* endpoint durable instead of transient, and it drifted anyway.
  Across four A/B trials it never beat sending nothing, and at production scale it cost 2.75× more.
- The only lever that worked is the operator's own ``provider`` routing preference in
  ``model_params.json``, which the SDK already carries to the wire. That is a **routing policy**
  choice — cost, latency, and quantization all ride on it — and deliberately not the harness's to
  make. What the harness owes it is honesty about the consequences, which is why `context_limit`
  reads the same pin when it computes the ceiling (`_openrouter._pinned_slugs`).

So the rule for a router adapter: declare the mode by what the wire needs, and **never read a hit
rate as a capability**. `AUTOMATIC` here means *nothing to send*, not *caching achieved* — the
per-call ``cached_tokens=`` on the log line is the only thing that says what actually happened.

The asymmetry is the whole point: *automatic* and *none* fail safe (do nothing, lose nothing), and
*explicit* fails **expensive and invisible**. So the mode is a thing an adapter **declares**, not a
thing the engine guesses, and the standing rule (`CLAUDE.md` → Provider Capabilities) is that no new
adapter ships without declaring one. Read as a capability, never a vendor branch: the engine asks
every adapter the same question and does exactly one thing with the answer.

Direct-to-vendor, `automatic` does not promise a hit either — xAI's cache is **per-server**
-------------------------------------------------------------------------------------------
The same gap opens without a router in sight, for a different reason, and it cost real money before
anyone saw it (issue #431). xAI runs a fleet, and a cache entry lives on **the one server that
served the call** — so a request that lands anywhere else re-pays for a prefix xAI already has.
Their own guidance is to send a stable conversation id — ``x-grok-conv-id`` as an HTTP header on the
Chat Completions surface, ``prompt_cache_key`` in the Responses body, and ``x-grok-conv-id`` as
**gRPC metadata** on the native surface — precisely so consecutive calls route back to the same
server. The harness never sent one, and the bill said so: measured live 2026-08-29,
@briggs re-sent a byte-stable ~210 K-token prefix roughly 45 seconds apart and hit
**0.2%–18%** — several calls at ``cached_tokens=512``, one at **0**, against the 92–99% every other
adapter was earning on the identical engine and the identical message layout. Roughly half of a
~$50 xAI burn should have been discounted and was not. The sporadic partial hits were luck: a call
landing on a server that happened to still be warm.

So the harness **binds a conversation** to the adapter before every turn (`bind_conversation`), and
the key is **the session id** — the string the `Session` is already keyed by, e.g.
``timeline:019f6e71-…``. That is exactly the unit whose transcript *is* the repeated prefix (one
transcript per session), so affinity aligns 1:1 with the cacheable bytes; cross-session calls share
only the charter, so a coarser key would herd unrelated prefixes onto one server and a finer one
would not exist. Keying on the session id rather than on any timeline-specific notion covers every
session kind, present and future (``default``, a hypothetical ``github:pr-123``), with no
special-casing anywhere. The id is opaque plumbing to xAI — a routing key, never content.

**Where the adapter puts that key is the adapter's problem, and it is not always a request field**
(issue #433). This capability hands over a string; it does not say what goes on the wire, because
only the adapter knows. On the ``xai-sdk`` the answer turned out to be gRPC call metadata rather
than a ``chat.create`` field — the SDK accepts a ``conversation_id`` keyword and spends it on an
OpenTelemetry span attribute, so 0.110.0 bound the key, passed every test, and reached no server
with it. The lesson generalizes past xAI and is why this capability is deliberately thin: **an
adapter has not implemented `bind_conversation` until something proves the key left the process.**

**This is not the OpenRouter session pin, and it must not be "consistency"-ed away.** Issue #372
measured an eager `session_id` pin *for OpenRouter* and **rejected** it — and the reasoning does not
transfer, because it turned on the one property xAI does not have: OpenRouter fans one model id out
across **dozens of third-party upstreams that do not behave alike**, some of which cache nothing at
all, so pinning made a landing on a non-caching endpoint *durable* instead of transient (at
production scale, 2.75× more expensive). xAI is a single vendor's **homogeneous** fleet reached
directly, where every server caches the same way and the only question is whether the next call
finds the one holding your prefix. Same-looking knob, opposite situation: there, a pin risks
sticking to a bad endpoint; here, there is no bad endpoint to stick to. Read #372 as *never read a
hit rate as a capability* — never as *never send a routing key*.

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

import logging
from dataclasses import replace

from basecradle_harness._messages import Message

_log = logging.getLogger("basecradle_harness")

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


def bind_conversation(provider: object, conversation: str | None) -> None:
    """Tell an adapter which conversation its next calls belong to — for cache **affinity** (#431).

    A capability, read the same way as `cache_mode`: an adapter that wants a routing key declares
    ``bind_conversation``; one that does not is left alone, and this is a no-op. There is no
    ``if provider == xai`` above the adapter layer, and a third-party adapter written before this
    existed keeps working untouched.

    `conversation` is the `Session`'s ``source`` — the string a transcript is keyed by, which is
    exactly the unit whose bytes repeat. An empty or absent one binds ``None``, and an adapter's
    contract on ``None`` is **omit the field**, never invent a value: a made-up id is a *new*
    conversation to the vendor on every call, which is a cache miss dressed up as a fix.

    The binding is **sticky** until the next one, and that is deliberate rather than incidental:
    every model call the harness makes while working a session belongs to that session — the turn
    itself, the engine's retries and its reserve summary, and the compaction summarize call that
    runs after it (`_context.Compactor._summarize`). `Session._drive` rebinds before each turn, so
    the value in force is always the session about to be driven. A caller that drives an `Engine`
    directly, with no `Session`, never binds and is unchanged.

    Best-effort by construction: a raising adapter costs a log line and a worse hit rate, never a
    wake. A routing hint is not worth a dropped peer message.
    """
    bind = getattr(provider, "bind_conversation", None)
    if not callable(bind):
        return
    try:
        bind(conversation or None)
    except Exception as exc:  # noqa: BLE001 - affinity is an optimization; never break a wake
        _log.warning("Could not bind the conversation id for cache affinity: %s", exc)


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
