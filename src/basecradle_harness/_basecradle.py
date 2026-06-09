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
"""

from __future__ import annotations

import itertools
import os
import time

from basecradle import BaseCradle

from basecradle_harness._harness import Harness
from basecradle_harness._memory import MemoryTool
from basecradle_harness._messages import Message
from basecradle_harness._openai import OpenAICompatibleProvider

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
    """

    def __init__(
        self,
        harness: Harness,
        *,
        timeline: str,
        client: BaseCradle | None = None,
        context_messages: int | None = DEFAULT_CONTEXT_MESSAGES,
    ) -> None:
        if context_messages is not None and context_messages < 0:
            raise ValueError("context_messages must be non-negative or None")
        self.harness = harness
        self.client = client or BaseCradle()
        self.timeline_uuid = timeline
        self.timeline = self.client.timelines.get(timeline)
        self.me_uuid = self.client.me.identity.uuid

        # One newest-first read serves both jobs. The high-water mark needs only
        # the newest message; the seed wants the most recent `context_messages`.
        # `filter()` is a lazy, auto-paginating iterator, so `islice` fetches
        # just the pages it needs — a capped seed never paginates the whole
        # timeline. We read `max(cap, 1)` so the mark is still the true newest
        # even when the seed is empty (cap of 0).
        newest_first = self.client.messages.filter(timeline=self.timeline_uuid)
        if context_messages is None:
            recent = list(newest_first)
        else:
            recent = list(itertools.islice(newest_first, max(context_messages, 1)))
        self._last_seen: str | None = recent[0].content.uuid if recent else None

        to_seed = recent if context_messages is None else recent[:context_messages]
        for message in reversed(to_seed):  # oldest-first into history
            self.harness.history.append(self._as_turn(message))

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
            tools=[MemoryTool()],
        )
        return cls(
            harness,
            timeline=os.environ["BASECRADLE_TIMELINE"],
            client=_client_from_env(),
            context_messages=_context_messages_from_env(),
        )

    def poll_once(self) -> list[object]:
        """Handle every new message once: think, reply, post. Returns posted messages."""
        posted = []
        for message in self._new_messages():
            if message.user.uuid == self.me_uuid:
                continue  # never reply to ourselves
            reply = self.harness.send(self._incoming_text(message))
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

    # --- turning timeline messages into conversation turns --------------------

    def _incoming_text(self, message: object) -> str:
        """Another peer's message as the agent hears it: prefixed with who spoke."""
        return f"{message.user.handle}: {message.content.body}"

    def _as_turn(self, message: object) -> Message:
        """A historical timeline message as a conversation turn for the engine.

        The agent's own posts become assistant turns; everyone else's become user
        turns tagged with the speaker, so the model can tell a multi-party
        conversation apart.
        """
        if message.user.uuid == self.me_uuid:
            return Message.assistant(content=message.content.body)
        return Message.user(content=self._incoming_text(message))

    # --- reading new messages, newest-first, up to the high-water mark --------

    def _new_messages(self) -> list[object]:
        """Messages newer than the high-water mark, in chronological order."""
        fresh = []
        for message in self.client.messages.filter(timeline=self.timeline_uuid):
            if message.content.uuid == self._last_seen:
                break
            fresh.append(message)
        fresh.reverse()
        if fresh:
            self._last_seen = fresh[-1].content.uuid
        return fresh


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
