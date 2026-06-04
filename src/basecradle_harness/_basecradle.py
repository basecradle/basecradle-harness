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
from basecradle_harness._openai import OpenAICompatibleProvider

DEFAULT_POLL_INTERVAL = 2.0


class TimelineAgent:
    """Runs a `Harness` against one BaseCradle timeline by polling it.

    On construction it resolves the timeline and its own identity, and marks the
    timeline's current newest message as the high-water mark — so it replies only
    to messages that arrive *after* it joins, never to history.

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
        self._last_seen: str | None = self._newest_message_uuid()

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
            reply = self.harness.send(message.content.body)
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

    def _newest_message_uuid(self) -> str | None:
        for message in self.client.messages.filter(timeline=self.timeline_uuid):
            return message.content.uuid
        return None

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
