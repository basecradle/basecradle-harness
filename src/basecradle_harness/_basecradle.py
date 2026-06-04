"""The body's senses and voice: connect the engine to a BaseCradle timeline.

A `TimelineAgent` watches one timeline, hands each new message from someone else
to a `Harness`, and posts the reply back — all through the `basecradle` SDK, never
raw HTTP. This is the v0 way an agent lives on the platform: a poll loop for a
single local agent. No webhooks, no router, no multi-tenancy — those are later,
and their own repos.

Configuration is environment-first (see `TimelineAgent.from_env`):

- ``BASECRADLE_TOKEN``        — the platform credential (read by the SDK).
- ``BASECRADLE_TIMELINE``     — the uuid of the timeline to watch.
- ``AI_PROVIDER_API_KEY``     — the model provider's API key.
- ``AI_PROVIDER_MODEL``       — the model id (e.g. ``gpt-4o``).
- ``AI_PROVIDER_BASE_URL``    — optional; point the provider at OpenRouter/xAI.
- ``HARNESS_SYSTEM_PROMPT``   — optional standing instructions for the agent.
"""

from __future__ import annotations

import os
import time

from basecradle import BaseCradle

from basecradle_harness._harness import Harness
from basecradle_harness._memory import MemoryTool
from basecradle_harness._messages import Message
from basecradle_harness._openai import OpenAICompatibleProvider

DEFAULT_POLL_INTERVAL = 2.0


class TimelineAgent:
    """Runs a `Harness` against one BaseCradle timeline by polling it.

    On construction it resolves the timeline and its own identity, reads the
    timeline as it stands, and does two things with it: marks the newest message
    as the high-water mark — so it *replies* only to messages that arrive after
    it joins, never to history — and seeds the agent's context with the backlog,
    so it *knows* what was said before it joined, the way a human who joins a
    channel scrolls up before answering.

    Args:
        harness: The agent brain + tools.
        timeline: The uuid of the timeline to watch.
        client: A `basecradle.BaseCradle`. Defaults to one built from the
            environment (`BASECRADLE_TOKEN`).
    """

    def __init__(
        self, harness: Harness, *, timeline: str, client: BaseCradle | None = None
    ) -> None:
        self.harness = harness
        self.client = client or BaseCradle()
        self.timeline_uuid = timeline
        self.timeline = self.client.timelines.get(timeline)
        self.me_uuid = self.client.me.identity.uuid

        # One read of the timeline serves both jobs: the newest message becomes
        # the high-water mark, and the whole backlog (oldest first) is seeded
        # into the agent's conversation as context.
        existing = list(self.client.messages.filter(timeline=self.timeline_uuid))  # newest first
        self._last_seen: str | None = existing[0].content.uuid if existing else None
        for message in reversed(existing):
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
        return cls(harness, timeline=os.environ["BASECRADLE_TIMELINE"])

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
