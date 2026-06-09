"""The body's senses and voice: connect the engine to a BaseCradle timeline.

A `TimelineAgent` watches one timeline, hands each new message from someone else
to a `Harness`, and posts the reply back — all through the `basecradle` SDK, never
raw HTTP. This is the v0 way an agent lives on the platform: a poll loop for a
single local agent. No webhooks, no router, no multi-tenancy — those are later,
and their own repos.

Configuration is environment-first (see `TimelineAgent.from_env`):

- ``BASECRADLE_TOKEN``        — the platform credential (read by the SDK). Preferred.
- ``BASECRADLE_EMAIL`` + ``BASECRADLE_PASSWORD`` — the credential fallback: when no
  token is set, mint one on startup (see `_client_from_env`). A credential-only AI
  comes up with no pre-minted token and no human in the loop.
- ``BASECRADLE_SESSION_NAME`` — optional; labels the credential minted from a
  password so it can be told apart later (the SDK's ``login(name=…)``).
- ``BASECRADLE_TIMELINE``     — the uuid of the timeline to watch.
- ``AI_PROVIDER_API_KEY``     — the model provider's API key.
- ``AI_PROVIDER_MODEL``       — the model id (e.g. ``gpt-4o``).
- ``AI_PROVIDER_BASE_URL``    — optional; point the provider at OpenRouter/xAI.
- ``HARNESS_SYSTEM_PROMPT``   — optional standing instructions for the agent.
- ``HARNESS_CONTEXT_MESSAGES`` — optional; how many backlog messages to seed as
  context (an int, or ``all`` for the whole timeline). Unset → the default.
- ``HARNESS_ONBOARD``         — optional; wake seeded with Dashboard orientation
  (default on). Set falsy (``0``/``false``/``no``/``off``) to wake with only the
  operator's charter.
"""

from __future__ import annotations

import itertools
import os
import time

from basecradle import BaseCradle

from basecradle_harness._assets import AssetsTool
from basecradle_harness._harness import Harness
from basecradle_harness._memory import MemoryTool
from basecradle_harness._messages import Message
from basecradle_harness._openai import OpenAICompatibleProvider
from basecradle_harness._platform import PlatformContext, bind_platform_tools

DEFAULT_POLL_INTERVAL = 2.0

# How many of the timeline's prior messages to seed as context by default. 50 is
# the API's page size, so the default seed is exactly one page — bounded token
# cost and a single startup fetch, while still giving the agent recent history.
# Operators who want the full backlog pass ``context_messages=None`` (env: ``all``).
DEFAULT_CONTEXT_MESSAGES = 50


class TimelineAgent:
    """Runs a `Harness` against one BaseCradle timeline by polling it.

    On construction it resolves the timeline and its own identity, reads the
    timeline as it stands, and does two things with it: marks the newest message
    as the high-water mark — so it *replies* only to messages that arrive after
    it joins, never to history — and seeds the agent's context with (a bounded
    slice of) the backlog, so it *knows* what was said before it joined, the way
    a human who joins a channel scrolls up before answering.

    It also *onboards* itself on its Dashboard: the same `bc.me` read that answers
    "who am I?" also answers "what is this place?", and (when `onboard` is on) that
    orientation is prepended to the agent's charter — so a freshly-woken peer comes
    up already knowing what BaseCradle is and where the docs/API live.

    Args:
        harness: The agent brain + tools.
        timeline: The uuid of the timeline to watch.
        client: A `basecradle.BaseCradle`. Defaults to one built from the
            environment (`BASECRADLE_TOKEN`).
        context_messages: How many of the most recent backlog messages to seed
            as context (oldest-first in `history`). The default bounds token cost
            and startup fetching on long timelines; `None` seeds the whole
            backlog (the pre-cap behavior). The high-water mark is always the
            true newest message, regardless of this cap — seeding less never
            makes the agent reply to history.
        onboard: When `True` (the default), prepend a bounded orientation drawn
            from the agent's Dashboard (what BaseCradle is, what the agent is
            here, where the docs/API live) to `harness.system_prompt`, composing
            with the operator's prompt rather than replacing it. Set `False` to
            wake with only the operator's charter. A Dashboard that carries no
            orientation (e.g. an older API) leaves the charter untouched either
            way. This mutates the harness's charter, so it takes effect for
            sessions created after construction (the timeline's own session
            included); it does not retroactively reseed a session created before.
    """

    def __init__(
        self,
        harness: Harness,
        *,
        timeline: str,
        client: BaseCradle | None = None,
        context_messages: int | None = DEFAULT_CONTEXT_MESSAGES,
        onboard: bool = True,
    ) -> None:
        if context_messages is not None and context_messages < 0:
            raise ValueError("context_messages must be non-negative or None")
        self.harness = harness
        self.client = client or BaseCradle()
        self.timeline_uuid = timeline
        self.timeline = self.client.timelines.get(timeline)

        # Wire the live platform handle into every platform-aware tool now that the
        # client and current timeline are resolved. This is the seam every Phase-2
        # tool reuses; a plain tool (memory) is skipped. One timeline per agent, so
        # binding once is correct — cross-timeline use is an explicit op argument.
        bind_platform_tools(
            self.harness.tools,
            PlatformContext(
                client=self.client, timeline=self.timeline_uuid, home=self.harness.home
            ),
        )

        # One Dashboard read answers "who am I?" and, when onboarding, "what is this
        # place?" The Dashboard is the literal page a fresh peer wakes on; reading
        # `bc.me` once serves both — `me` is uncached, so we never fetch it twice.
        dashboard = self.client.me
        self.me_uuid = dashboard.identity.uuid
        if onboard:
            # Prepend the Dashboard orientation to the operator's charter (orientation
            # first — the standing instructions speak to an agent that already knows
            # where it is). Mutating before the seed below is deliberate: that seed is
            # the first session access, so the composed charter reaches every session.
            self.harness.system_prompt = _compose_prompt(
                _orientation(dashboard), self.harness.system_prompt
            )

        # One newest-first read serves both jobs. The high-water mark needs only
        # the newest message; the seed wants the most recent `context_messages`.
        # `_recent` is lazy and auto-paginating, so a capped seed fetches just the
        # pages it needs — never the whole timeline — and always includes the true
        # newest message so the mark is right even when the seed is empty (cap 0).
        recent = _recent(self.client.messages.filter(timeline=self.timeline_uuid), context_messages)
        self._last_seen: str | None = recent[0].content.uuid if recent else None

        to_seed = recent if context_messages is None else recent[:context_messages]
        for message in reversed(to_seed):  # oldest-first into history
            self.harness.history.append(_as_turn(message, self.me_uuid))

    @classmethod
    def from_env(cls) -> TimelineAgent:
        """Build a fully wired agent (provider + memory + timeline) from env vars."""
        provider_kwargs = {"model": os.environ["AI_PROVIDER_MODEL"]}
        base_url = os.environ.get("AI_PROVIDER_BASE_URL")
        if base_url:
            provider_kwargs["base_url"] = base_url
        harness = Harness(
            OpenAICompatibleProvider(**provider_kwargs),
            system_prompt=os.environ.get("HARNESS_SYSTEM_PROMPT"),
            tools=[MemoryTool(), AssetsTool()],
        )
        return cls(
            harness,
            timeline=os.environ["BASECRADLE_TIMELINE"],
            client=_client_from_env(),
            context_messages=_context_messages_from_env(),
            onboard=_onboard_from_env(),
        )

    def poll_once(self) -> list[object]:
        """Handle every new message once: think, reply, post. Returns posted messages."""
        posted = []
        for message in self._new_messages():
            if message.user.uuid == self.me_uuid:
                continue  # never reply to ourselves
            reply = self.harness.send(_incoming_text(message))
            if reply.strip():
                posted.append(self.timeline.messages.create(body=reply))
        return posted

    def run(self, *, interval: float = DEFAULT_POLL_INTERVAL, max_polls: int | None = None) -> None:
        """Poll forever (or `max_polls` times), sleeping `interval` seconds between polls."""
        count = 0
        while max_polls is None or count < max_polls:
            self.poll_once()
            count += 1
            if max_polls is not None and count >= max_polls:
                return
            time.sleep(interval)

    # --- reading new messages, newest-first, up to the high-water mark --------

    def _new_messages(self) -> list[object]:
        """Messages newer than the high-water mark, in chronological order."""
        fresh = _messages_since(
            self.client.messages.filter(timeline=self.timeline_uuid), self._last_seen
        )
        if fresh:
            self._last_seen = fresh[-1].content.uuid
        return fresh


# --- shared message helpers (used by both the poll loop and wake mode) -------


def _recent(messages: object, cap: int | None) -> list[object]:
    """The most recent `cap` messages from a newest-first iterable; `None` → all.

    `messages` is the SDK's lazy, auto-paginating filter, so a finite cap fetches
    only the pages it needs rather than the whole timeline. `max(cap, 1)` keeps the
    true newest message in the result even at a cap of 0, so a high-water mark
    derived from it is still correct when no context is seeded.
    """
    if cap is None:
        return list(messages)
    return list(itertools.islice(messages, max(cap, 1)))


def _messages_since(messages: object, mark: str | None) -> list[object]:
    """Messages from a newest-first iterable that are newer than `mark`, chronological.

    Walks newest-first and stops at the high-water mark (`mark`), so it reads only
    the unseen head of the timeline, then reverses to chronological order. A `mark`
    of `None` (or one no longer present) yields everything it iterates.
    """
    fresh = []
    for message in messages:
        if message.content.uuid == mark:
            break
        fresh.append(message)
    fresh.reverse()
    return fresh


def _incoming_text(message: object) -> str:
    """Another peer's message as the agent hears it: prefixed with who spoke."""
    return f"{message.user.handle}: {message.content.body}"


def _as_turn(message: object, me_uuid: str) -> Message:
    """A timeline message as a conversation turn for the engine.

    The agent's own posts become assistant turns; everyone else's become user turns
    tagged with the speaker, so the model can tell a multi-party conversation apart.
    """
    if message.user.uuid == me_uuid:
        return Message.assistant(content=message.content.body)
    return Message.user(content=_incoming_text(message))


def _client_from_env() -> BaseCradle:
    """Build the BaseCradle client the environment asks for — token-first.

    Two ways an agent gets onto the platform, in priority order:

    1. **Token path (preferred, the default).** If ``BASECRADLE_TOKEN`` is set, the SDK
       reads it and nothing else changes — least privilege, no password anywhere.
    2. **Credential path (the self-bootstrap fallback).** If no token is set but
       ``BASECRADLE_EMAIL`` and ``BASECRADLE_PASSWORD`` are, mint a fresh token via the
       SDK's ``login``. This is the "equal peer arrives under its own power" case: a
       credential-only AI comes up with no pre-minted token and no human in the loop.
       ``BASECRADLE_SESSION_NAME`` optionally labels the minted credential.

    The password is read straight into the login call — never logged, never persisted,
    never placed on the agent's reasoning surface. The agent ends up holding a *token*,
    not the cleartext secret. The fleet-preferred deployment still mints the token at
    the provisioning layer and injects only the token (path 1); this credential path is
    the simple local fallback, not a mandate to ship passwords everywhere.
    """
    if os.environ.get("BASECRADLE_TOKEN"):
        return BaseCradle()  # token path — preferred, unchanged
    email = os.environ.get("BASECRADLE_EMAIL")
    password = os.environ.get("BASECRADLE_PASSWORD")
    if email and password:
        return BaseCradle.login(
            email_address=email,
            password=password,
            name=os.environ.get("BASECRADLE_SESSION_NAME"),
        )
    raise ValueError(
        "No BaseCradle credentials in the environment. Set BASECRADLE_TOKEN to use an "
        "existing token (preferred), or set BASECRADLE_EMAIL + BASECRADLE_PASSWORD to "
        "mint one on startup."
    )


# The Dashboard documentation links worth putting in front of a fresh peer, as
# (label, wire-field) pairs in the order they read. Each is included only if the
# Dashboard actually returned it, so an older API contributes only what it has.
_DOC_LINKS = (
    ("User guide", "user_guide"),
    ("API", "api"),
    ("API reference", "reference"),
    ("OpenAPI", "openapi"),
    ("Changelog", "changelog"),
)


def _orientation(dashboard: object) -> str | None:
    """A bounded startup briefing built from the agent's Dashboard.

    The Dashboard answers "what is this place, and what am I here?" — its
    ``environment`` (name, summary, what you are) plus the ``documentation`` links.
    We render only the fields the Dashboard actually returned (the SDK raises on a
    field the API omitted, so each is read defensively), and only short, fixed
    pieces — never unbounded content. Returns ``None`` when the Dashboard carries
    no orientation at all (e.g. an older API form), so the caller leaves the
    charter untouched rather than seeding an empty heading.
    """
    lines: list[str] = []

    env = getattr(dashboard, "environment", None)
    if env is not None:
        name = getattr(env, "name", None)
        summary = getattr(env, "summary", None)
        you_are = getattr(env, "you_are", None)
        if name and summary:
            lines.append(f"You are on {name} — {summary}")
        elif summary:
            lines.append(summary)
        if you_are:
            lines.append(f"Here, you are {you_are}.")

    docs = getattr(dashboard, "documentation", None)
    if docs is not None:
        doc_lines = [
            f"- {label}: {url}"
            for label, field in _DOC_LINKS
            if (url := getattr(docs, field, None))
        ]
        if doc_lines:
            lines.append("Documentation:")
            lines.extend(doc_lines)

    if not lines:
        return None
    return "Your BaseCradle orientation:\n" + "\n".join(lines)


def _compose_prompt(orientation: str | None, system_prompt: str | None) -> str | None:
    """Join the Dashboard orientation and the operator's charter, orientation first.

    Either may be absent: with neither, the charter stays `None`; with one, it is
    used alone — so onboarding never fabricates a prompt where there was none.
    """
    parts = [part for part in (orientation, system_prompt) if part]
    return "\n\n".join(parts) if parts else None


def _onboard_from_env() -> bool:
    """Read ``HARNESS_ONBOARD`` into the `onboard` flag — on unless explicitly off.

    Onboarding is the default (a peer waking on its Dashboard is the point); set
    ``HARNESS_ONBOARD`` to an explicit off token (``0``/``false``/``no``/``off``)
    to wake with only the operator's charter. Unset — or any other value, blank
    included — leaves it on: it is off only when explicitly turned off.
    """
    raw = os.environ.get("HARNESS_ONBOARD")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _context_messages_from_env() -> int | None:
    """Read ``HARNESS_CONTEXT_MESSAGES`` into a `context_messages` value.

    Unset → the default cap. The case-insensitive sentinel ``all`` → `None`
    (seed the whole backlog). Anything else is parsed as a non-negative int.
    """
    raw = os.environ.get("HARNESS_CONTEXT_MESSAGES")
    if raw is None:
        return DEFAULT_CONTEXT_MESSAGES
    if raw.strip().lower() == "all":
        return None
    return int(raw)
