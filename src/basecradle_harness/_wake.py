"""One-shot, per-event wake: the entrypoint a router invokes once per platform event.

Poll mode (`TimelineAgent.run`) is a long-lived loop that holds its high-water
mark in memory. Under a router ([basecradle-router](https://github.com/basecradle/basecradle-router))
the model inverts: the router runs a *command* once per platform event, the
process answers the timeline's unseen messages, and exits. Two things the poll
loop never needed become load-bearing:

1. **The high-water mark must persist across processes.** Each wake is a fresh
   process, so the mark cannot live in memory — a router retry or two events
   arriving close together would re-answer the same message. It is stored under
   the agent's `home`, one small file per timeline (`MarkStore`), and advanced
   after every reply so a crash mid-batch resumes without duplicating.
2. **The conversation must persist across processes too** — otherwise the agent
   would re-seed the whole backlog on every wake. That is exactly what a persisted
   `Session` already gives us: with `Harness(home=...)` and a `timeline:<uuid>`
   source, each wake reloads the prior transcript instead of starting blank.

Everything else — identity, Dashboard onboarding, turning messages into turns,
the newest-first scan up to the mark — is shared with `TimelineAgent`.

The command is `basecradle-harness-wake --timeline <uuid>` (also runnable as
`python -m basecradle_harness`); see `main`. It works from the timeline uuid
alone; an optional `--message <uuid>` names the triggering message for a precise
first-wake bootstrap (see `_bootstrap_split`).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import quote

from basecradle import BaseCradle

from basecradle_harness._assets import AssetsTool
from basecradle_harness._audio import HearAudioTool
from basecradle_harness._basecradle import (
    DEFAULT_CONTEXT_MESSAGES,
    _as_turn,
    _client_from_env,
    _compose_prompt,
    _context_messages_from_env,
    _incoming_text,
    _messages_since,
    _onboard_from_env,
    _orientation,
    _provider_from_env,
    _recent,
)
from basecradle_harness._exceptions import HarnessError, ProviderError
from basecradle_harness._governance import TimelinesTool, TrustTool
from basecradle_harness._harness import Harness
from basecradle_harness._images import GenerateImageTool
from basecradle_harness._memory import MemoryTool
from basecradle_harness._platform import PlatformContext, bind_platform_tools
from basecradle_harness._session import Session
from basecradle_harness._tasks import TasksTool
from basecradle_harness._webfetch import WebFetchTool
from basecradle_harness._webhooks import WebhookEndpointsTool, WebhookEventsTool


class MarkStore:
    """Per-timeline high-water marks, persisted under the agent's home.

    One file per timeline (`<root>/marks/<timeline>.txt`, the uuid percent-encoded
    into a safe filename), holding the uuid of the newest message the agent has
    handled. This is what makes wake mode idempotent across separate processes:
    the next wake reads the mark and skips everything at or before it.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def get(self, timeline: str) -> str | None:
        path = self._path(timeline)
        if not path.exists():
            return None
        return path.read_text().strip() or None

    def set(self, timeline: str, uuid: str) -> None:
        path = self._path(timeline)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(uuid)

    def _path(self, timeline: str) -> Path:
        return self.root / "marks" / f"{quote(timeline, safe='')}.txt"


class WakeAgent:
    """Answers one timeline's unseen messages in a single process, then is done.

    A wake runs the agent's `Harness` against the `timeline:<uuid>` session,
    replying to every message newer than the persisted high-water mark and
    advancing the mark as it goes. Unlike `TimelineAgent` it does not loop and does
    not hold state between invocations — durability lives entirely in the persisted
    transcript (the session) and the `MarkStore`.

    Args:
        harness: The agent brain + tools. It **must** have a `home` (or pass an
            explicit `marks`) — wake mode needs somewhere to persist the mark and
            the transcript across processes.
        timeline: The uuid of the timeline this wake is scoped to.
        client: A `basecradle.BaseCradle`. Defaults to one built from the
            environment (`BASECRADLE_TOKEN`).
        marks: Where high-water marks persist. Defaults to a `MarkStore` under the
            harness's `home`.
        context_messages: How many backlog messages to seed as context on the
            first wake (see `_bootstrap`). The default bounds token cost; `None`
            seeds the whole backlog.
        onboard: Prepend the Dashboard orientation to the charter (as
            `TimelineAgent` does). On by default.
    """

    def __init__(
        self,
        harness: Harness,
        *,
        timeline: str,
        client: BaseCradle | None = None,
        marks: MarkStore | None = None,
        context_messages: int | None = DEFAULT_CONTEXT_MESSAGES,
        onboard: bool = True,
    ) -> None:
        if context_messages is not None and context_messages < 0:
            raise ValueError("context_messages must be non-negative or None")
        if marks is None and harness.home is None:
            raise ValueError(
                "Wake mode needs somewhere to persist its high-water mark across "
                "processes. Set HARNESS_HOME (or pass home= to Harness, or marks= here)."
            )
        self.harness = harness
        self.client = client or BaseCradle()
        self.timeline_uuid = timeline
        self.source = f"timeline:{timeline}"
        self.context_messages = context_messages
        self.marks = marks or MarkStore(harness.home)  # type: ignore[arg-type]
        self.timeline = self.client.timelines.get(timeline)

        # Bind the live platform handle into every platform-aware tool — the same
        # seam the poll loop uses, so a router-woken peer can act on the timeline
        # exactly as a polling one. One wake serves one timeline; bind once.
        bind_platform_tools(
            self.harness.tools,
            PlatformContext(
                client=self.client, timeline=self.timeline_uuid, home=self.harness.home
            ),
        )

        # One Dashboard read answers "who am I?" and, when onboarding, "what is this
        # place?" — exactly as the poll loop does, so a router-woken peer is oriented
        # the same as a polling one. Mutating the charter here (before any session is
        # created in `wake`) means the composed prompt reaches a fresh session.
        dashboard = self.client.me
        self.me_uuid = dashboard.identity.uuid
        if onboard:
            self.harness.system_prompt = _compose_prompt(
                _orientation(dashboard), self.harness.system_prompt
            )

    @classmethod
    def from_env(cls, *, timeline: str | None = None) -> WakeAgent:
        """Build a fully wired wake agent from env vars — `TimelineAgent.from_env`'s twin.

        Reads the same provider/credential/charter vars, plus `HARNESS_HOME`, which
        wake mode requires: it is where the transcript and the high-water mark
        persist across the separate processes a router spawns. `timeline` overrides
        `BASECRADLE_TIMELINE` (the router passes it on the command line).
        """
        home = os.environ.get("HARNESS_HOME")
        if not home:
            raise ValueError(
                "Wake mode requires HARNESS_HOME — the directory where the agent's "
                "transcript and high-water mark persist across wakes."
            )
        harness = Harness(
            _provider_from_env(),
            system_prompt=os.environ.get("HARNESS_SYSTEM_PROMPT"),
            tools=[
                MemoryTool(),
                WebFetchTool(),
                AssetsTool(),
                HearAudioTool(),
                TasksTool(),
                TimelinesTool(),
                TrustTool(),
                GenerateImageTool(),
                WebhookEndpointsTool(),
                WebhookEventsTool(),
            ],
            home=home,
        )
        return cls(
            harness,
            timeline=timeline or os.environ["BASECRADLE_TIMELINE"],
            client=_client_from_env(),
            context_messages=_context_messages_from_env(),
            onboard=_onboard_from_env(),
        )

    def wake(self, *, trigger: str | None = None) -> list[object]:
        """Answer this timeline's unseen messages once and return what was posted.

        With a persisted mark, replies to everything newer than it. Without one
        (the first wake), bootstraps a mark first (see `_bootstrap`). Either way, if
        nothing is unseen, no provider call is made and nothing is posted.

        `trigger` is the optional uuid of the message that fired the event; it only
        sharpens the first-wake bootstrap and is ignored once a mark exists.
        """
        session = self.harness.session(self.source)
        mark = self.marks.get(self.timeline_uuid)
        if mark is None:
            return self._bootstrap(session, trigger)
        unseen = _messages_since(self.client.messages.filter(timeline=self.timeline_uuid), mark)
        return self._respond(session, unseen)

    # --- the first wake: infer a high-water mark from the timeline ------------

    def _bootstrap(self, session: Session, trigger: str | None) -> list[object]:
        """The first wake for a timeline: there is no persisted mark yet.

        Read the recent backlog (newest-first, bounded by `context_messages`), then
        split it into the messages to *reply* to and the older ones to seed as
        *context* — so the agent answers the right thing while knowing what came
        before, the way a human scrolls up before speaking. The context is seeded
        only into a fresh transcript, so a restart that kept the transcript but lost
        the mark does not re-seed. Finally, record the true newest message as the
        mark, so the next wake is a normal incremental one.
        """
        recent = _recent(
            self.client.messages.filter(timeline=self.timeline_uuid), self.context_messages
        )
        if not recent:
            return []  # empty timeline: nothing to answer, nothing to mark
        split = self._bootstrap_split(recent, trigger)
        to_reply = list(reversed(recent[: split + 1]))  # chronological; empty when split is -1
        context = recent[split + 1 :]  # older than the reply set, still newest-first
        if not _has_conversation(session):
            for message in reversed(context):  # oldest-first into history
                session.history.append(_as_turn(message, self.me_uuid))
        posted = self._respond(session, to_reply)
        # Everything up to the newest message is now seen (replied to or seeded as
        # context), so the mark is the true newest — even if `to_reply` was empty.
        self.marks.set(self.timeline_uuid, recent[0].content.uuid)
        return posted

    def _bootstrap_split(self, recent: list[object], trigger: str | None) -> int:
        """The newest-first index of the *oldest* backlog message to reply to.

        Resolves "what counts as unseen?" on the first wake, in priority order:

        1. **The triggering message**, if named and present — reply to it and
           anything newer. The precise answer; the router can pass it.
        2. **Everything since the agent last spoke here** — its own newest post is
           the natural high-water mark, which makes the poll→wake cutover lossless
           (it picks up exactly where polling left off) and answers a burst that
           arrived at once.
        3. **The newest message only** — when the agent has never spoken here (a
           fresh join), reply to the message that woke it without flooding history.

        Returns -1 when there is nothing newer than the agent's own latest post
        (e.g. it just spoke and nothing followed): reply to nothing, but still mark.
        """
        if trigger is not None:
            for i, message in enumerate(recent):
                if message.content.uuid == trigger:
                    return i
            # Trigger older than the fetched window — fall through to the defaults.
        for i, message in enumerate(recent):  # newest-first → first hit is the latest post
            if message.user.uuid == self.me_uuid:
                return i - 1  # reply to everything newer than our own last message
        return 0  # never spoken here → reply to the newest message only

    # --- replying ------------------------------------------------------------

    def _respond(self, session: Session, messages: list[object]) -> list[object]:
        """Reply to each message in chronological order, advancing the mark per reply.

        Persisting the mark after *each* message (not once at the end) is what makes
        a crash or router retry mid-batch safe: the next wake resumes after the last
        message actually handled, never re-answering one. Our own posts are skipped
        but still advance the mark.

        Order matters: reply is posted *before* the mark advances, so the semantics
        are at-least-once — if the post fails, the mark does not advance and the
        retry re-posts. We deliberately favour a possible duplicate over a dropped
        reply, which on a comms platform is the worse failure.
        """
        posted = []
        for message in messages:
            if message.user.uuid != self.me_uuid:
                reply = session.send(_incoming_text(message))
                if reply.strip():
                    posted.append(self.timeline.messages.create(body=reply))
            self.marks.set(self.timeline_uuid, message.content.uuid)
        return posted


def _has_conversation(session: Session) -> bool:
    """Has this session said or heard anything yet (beyond a seeded system charter)?"""
    return any(turn.role in ("user", "assistant") for turn in session.history)


def main(argv: list[str] | None = None) -> int:
    """The `basecradle-harness-wake` entrypoint: one wake, then exit.

    Exit code 0 on success including "nothing to do"; non-zero on a hard
    config/auth/credential failure, so the router can surface it.
    """
    parser = argparse.ArgumentParser(
        prog="basecradle-harness-wake",
        description="Answer one BaseCradle timeline's unseen messages, then exit (router wake mode).",
    )
    parser.add_argument(
        "--timeline",
        default=os.environ.get("BASECRADLE_TIMELINE"),
        help="uuid of the timeline to process (or set BASECRADLE_TIMELINE).",
    )
    parser.add_argument(
        "--message",
        default=os.environ.get("BASECRADLE_MESSAGE"),
        help="optional uuid of the triggering message; sharpens the first-wake bootstrap.",
    )
    args = parser.parse_args(argv)
    if not args.timeline:
        parser.error("a timeline uuid is required (--timeline or BASECRADLE_TIMELINE)")

    try:
        agent = WakeAgent.from_env(timeline=args.timeline)
        agent.wake(trigger=args.message)
    except (HarnessError, ProviderError, ValueError, KeyError) as error:
        print(f"basecradle-harness-wake: {error}", file=sys.stderr)
        return 1
    return 0
