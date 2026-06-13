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

A wake reconciles **every** kind of unseen actionable item on the timeline, not just
messages. Three cases the message scan would otherwise miss:

- A peer's posted **asset**: a file (image, doc, audio) shared on the timeline is an
  item like a message, so it rides the same high-water mark — but the message scan
  reads only messages, so the wake also scans assets and *perceives* a peer's file. An
  image is fetched and shown to the model inline, so a vision-capable agent actually
  sees the picture on wake; media it cannot yet fully perceive (a doc, audio, video)
  degrades to a description naming the file and its type, with the `view`/`read`/`listen`
  tools available to engage further on demand.
- An inbound **webhook delivery**: a received `webhook_event` is not a timeline item,
  so the wake fetches unseen ones through the SDK's webhook-events read surface, under
  their own high-water mark, and acts on them.
- A newly-**activated task**: a `task.activated` wake fires when a scheduled task comes
  due, but the activation is not a fresh timeline item the scan surfaces — so the wake
  lists the timeline's *activated* tasks and carries out the instructions of any it has
  not handled yet. Activated tasks are tracked by a persisted seen-set rather than a
  high-water mark, because activation order does not track creation order (a task
  scheduled earlier can come due later) and a task has no terminal status to mark done.

So a peer woken by `webhook_event.received`, `task.activated`, or an asset post
perceives and acts on the trigger, with the same idempotency across processes the
message path has. **The actor self-filter is the safety property running through all of
it:** messages and assets are skipped when the agent authored them, so it never reacts
to — or wake-loops on — its own posts, most importantly an image it just generated.

Everything else — identity, Dashboard onboarding, turning messages into turns,
the newest-first scan up to the mark — is shared with `TimelineAgent`.

The command is `basecradle-harness-wake --timeline <uuid>` (also runnable as
`python -m basecradle_harness`); see `main`. **In production it works from the timeline
uuid alone** — the router wakes a harness agent with `--timeline <uuid>` and nothing
else (see basecradle-router `wake_command`), never naming the item that fired the wake.
So the reconcile cannot lean on a trigger: each kind surfaces its own newest unseen item
on a first wake and everything past its mark thereafter, with no router help. The optional
`--message`, `--event`, and `--asset` uuids still name a triggering item when a manual or
future-router invocation passes one — each sharpens its own first-wake bootstrap (see
`_bootstrap_split` and `_bootstrap_stream`) — but they are not required for a delivery,
asset, or task to be acted on.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import quote

import httpx
from basecradle import BaseCradle, BaseCradleError

from basecradle_harness._assets import AssetsTool, _describe, _is_image, image_input
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
from basecradle_harness._messages import ImageContent
from basecradle_harness._platform import PlatformContext, bind_platform_tools
from basecradle_harness._probe import ack_line, verify_probe
from basecradle_harness._session import Session
from basecradle_harness._tasks import TasksTool
from basecradle_harness._version import __version__
from basecradle_harness._webfetch import WebFetchTool
from basecradle_harness._webhooks import WebhookEndpointsTool, WebhookEventsTool

# The kinds of timeline item the wake reconciles. Messages and webhook events are
# creation-ordered streams tracked by a high-water mark (below); activated tasks are
# not (a task scheduled earlier can activate later, and a task has no "done" status),
# so they are tracked by a persisted seen-set instead. `messages` keeps the original
# on-disk mark location so a deployed agent's existing marks still resolve.
_MESSAGES = "messages"
_EVENTS = "webhook_events"
_TASKS = "tasks"
_ASSETS = "assets"


class MarkStore:
    """Per-timeline, per-kind high-water marks, persisted under the agent's home.

    One file per (kind, timeline) — `<root>/marks/<timeline>.txt` for messages (the
    original location, kept for backward compatibility) and
    `<root>/marks/<kind>/<timeline>.txt` for any other kind, the uuid percent-encoded
    into a safe filename — holding the uuid of the newest item of that kind the agent
    has handled. This is what makes wake mode idempotent across separate processes:
    the next wake reads the mark and skips everything at or before it. Messages and
    webhook events advance their marks independently, so reconciling one never
    re-surfaces the other.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def get(self, timeline: str, *, kind: str = _MESSAGES) -> str | None:
        path = self._path(timeline, kind)
        if not path.exists():
            return None
        return path.read_text().strip() or None

    def set(self, timeline: str, uuid: str, *, kind: str = _MESSAGES) -> None:
        path = self._path(timeline, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(uuid)

    def _path(self, timeline: str, kind: str = _MESSAGES) -> Path:
        # Messages live directly under `marks/` (the original layout); every other
        # kind gets its own subdirectory, so the namespaces never collide.
        base = self.root / "marks"
        folder = base if kind == _MESSAGES else base / kind
        return folder / f"{quote(timeline, safe='')}.txt"


class SeenStore:
    """Per-timeline sets of handled item uuids, persisted under the agent's home.

    A high-water mark works for a creation-ordered stream (messages, webhook events):
    everything at or before the mark is seen. Activated **tasks** are not such a
    stream — a task scheduled earlier can activate later, so activation order does not
    track creation order, and a task carries no terminal "done" status the agent could
    set. So idempotency here is a *set*: the uuids already acted on, persisted one per
    line under `<root>/seen/<kind>/<timeline>.txt`. Appended to (not rewritten) after
    each item is handled, so a crash mid-batch resumes without re-acting on the rest.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def all(self, timeline: str, *, kind: str) -> set[str]:
        path = self._path(timeline, kind)
        if not path.exists():
            return set()
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}

    def add(self, timeline: str, uuid: str, *, kind: str) -> None:
        path = self._path(timeline, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(uuid + "\n")

    def _path(self, timeline: str, kind: str) -> Path:
        return self.root / "seen" / kind / f"{quote(timeline, safe='')}.txt"


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
        probe_secret: str | None = None,
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
        # The shared HMAC key for the NOC synthetic-probe marker (see `_probe`). Set → the
        # message, webhook, and task reconciles each recognize a signed probe in their own
        # carrier field and ack it token-free, before the model. Unset → the short-circuit
        # is off and every item goes to the model.
        self.probe_secret = probe_secret
        self.marks = marks or MarkStore(harness.home)  # type: ignore[arg-type]
        # Activated tasks are tracked by a seen-set, not a high-water mark (see
        # `SeenStore`); it lives beside the marks, under the same home root.
        self.seen = SeenStore(self.marks.root)
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

        Token reuse matters most here: each wake is a fresh process, so without
        persistence a credential-only agent would mint a new token (a new platform
        Session) on every wake. `_client_from_env` mints only when the token is missing
        or dead and writes it back to `BASECRADLE_ENV_FILE`, so the next wake — which
        sources that same file — reuses it. See `_token` for the full lifecycle.

        `NOC_PROBE_SECRET` (optional) is the shared HMAC key for the NOC's synthetic-probe
        marker. Set → a recognized probe is acked token-free before the model (see
        `_probe`), across the message, webhook, and task seams alike; unset → the
        short-circuit is off. The same var name the NOC box uses, so one provisioned value
        serves the harness half of all three seam contracts.
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
            # `or None` so a set-but-blank NOC_PROBE_SECRET (an exported-but-unfilled
            # secret) reads as *off*, not as "enabled with an empty HMAC key".
            probe_secret=os.environ.get("NOC_PROBE_SECRET") or None,
        )

    def wake(
        self,
        *,
        trigger: str | None = None,
        event_trigger: str | None = None,
        asset_trigger: str | None = None,
    ) -> list[object]:
        """Reconcile this timeline once — unseen messages, posted assets, inbound webhook
        deliveries, *and* newly-activated tasks — and return everything posted in response.

        A wake surfaces every kind of unseen actionable item on the timeline, not just
        new messages, because each kind arrives by a different route: a message and an
        asset are read from the timeline, a webhook delivery and a task activation are not
        items the timeline scan would surface on their own. Each kind has its own
        idempotency record so reconciling one never re-surfaces another, and all reply
        through the one session, so the agent perceives them in a single coherent
        conversation. Messages and assets pass through the **actor self-filter**, so the
        agent never reacts to its own posts. If nothing is unseen, no provider call is
        made and nothing is posted.

        `trigger`, `event_trigger`, and `asset_trigger` are the optional uuids of the
        message, webhook event, or asset that fired the wake. **The router never passes
        them** — it wakes a harness agent with the timeline uuid alone — so each kind
        finds its own unseen items unaided: the newest on a first wake, everything past its
        mark thereafter. The triggers remain accepted for a manual or future-router
        invocation that does name an item (each only sharpens its own first-wake bootstrap
        and is ignored once that kind's record exists), but nothing depends on them. Task
        activations never used a trigger — the reconcile finds every activated-but-unhandled
        task, keeping the router thin.
        """
        session = self.harness.session(self.source)
        posted = self._wake_messages(session, trigger)
        posted += self._wake_assets(session, asset_trigger)
        posted += self._wake_events(session, event_trigger)
        posted += self._wake_tasks(session)
        return posted

    # --- messages ------------------------------------------------------------

    def _wake_messages(self, session: Session, trigger: str | None) -> list[object]:
        """Answer unseen messages: bootstrap on the first wake, else reply incrementally."""
        mark = self.marks.get(self.timeline_uuid)
        if mark is None:
            return self._bootstrap(session, trigger)
        unseen = _messages_since(self.client.messages.filter(timeline=self.timeline_uuid), mark)
        return self._respond(session, unseen)

    # --- posted assets -------------------------------------------------------

    def _wake_assets(self, session: Session, asset_trigger: str | None) -> list[object]:
        """Surface a peer's posted files (assets) and let the agent engage with them.

        An asset is a timeline item, like a message, so it rides the same high-water
        mark — but the wake's *message* scan reads only messages, so a file a peer
        shares would otherwise go unseen. This is the founder's minimum wake set: a
        peer posts an image or a doc, and the agent perceives it (and can `view`,
        `read`, or `listen` to it via the assets/audio tools). The router wakes with the
        timeline uuid alone, so on the first wake (no mark, no `asset_trigger`) the agent
        acts on the **newest** unseen asset — the one that almost certainly woke it —
        bounding a fresh agent to a single action rather than replaying every file posted
        before it arrived. A named `asset_trigger`, when a manual/future invocation passes
        one, acts on that asset and everything newer.

        The **actor self-filter** is load-bearing here: an asset the agent itself posted
        — most importantly an image it just generated with `generate_image` — is skipped
        (never acted on), while its mark still advances, so the agent cannot react to its
        own output and spiral into generating image after image. `_respond_assets`
        applies that filter through the shared `_act_on` loop, so it holds on the trigger
        path too (a wake somehow fired on the agent's own asset is still a no-op).
        """
        mark = self.marks.get(self.timeline_uuid, kind=_ASSETS)
        if mark is None:
            return self._bootstrap_stream(
                session,
                self.client.assets.filter(timeline=self.timeline_uuid),
                asset_trigger,
                _ASSETS,
                self.client.assets.get,
                self._respond_assets,
            )
        unseen = _assets_since(self.client.assets.filter(timeline=self.timeline_uuid), mark)
        return self._respond_assets(session, unseen)

    def _respond_assets(self, session: Session, assets: list[object]) -> list[object]:
        """Perceive each newly-posted asset, skipping the agent's own (the self-filter).

        `_perceive_asset` fetches the file and presents an image **inline** so the agent
        actually sees a peer's picture (not just a description of it); media it cannot yet
        fully perceive degrades to a description rather than erroring. A recognized NOC
        synthetic-probe — read from the asset's **description**, the asset analog of a
        message body / task instructions / webhook payload (see `_asset_marker_carrier`) —
        is acked token-free before the model (`probe=`), the 4th seam's short-circuit. The
        probe check runs *before* the fetch, so a probe asset is acked without any download.
        """
        return self._act_on(
            session,
            assets,
            self._perceive_asset,
            lambda asset: self.marks.set(self.timeline_uuid, asset.content.uuid, kind=_ASSETS),
            skip=self._is_own,
            probe=lambda asset: self._probe_nonce(_asset_marker_carrier(asset)),
        )

    def _perceive_asset(self, asset: object) -> tuple[str, list[ImageContent]]:
        """A peer's posted asset as the agent perceives it: an image shown, else described.

        The asset-wake's perception step, and the reason the seam is more than a
        notification: for a viewable image the bytes are fetched and handed to the model as
        input (via `image_input`, the same gate the `view` tool uses), so a vision-capable
        agent *sees* the shared picture on wake. Anything it cannot yet fully perceive —
        a non-image file, an unviewable/oversized image, or an image whose download fails —
        degrades to the text description (`_incoming_asset_text`), which names the file and
        its type and points at the tools, so the seam is graceful, never an error. Audio/
        video perception *depth* is out of scope here (it rides the perception done-bar
        thread); this handles their seam by acknowledging the file rather than choking on it.
        """
        file = asset.content.file
        if _is_image(file.content_type):
            try:
                shown = image_input(file)
            except httpx.HTTPError:
                shown = None  # fetch failed: degrade to a description, never an error
            if isinstance(shown, ImageContent):
                intro = f"{asset.user.handle} posted a file to this timeline: {_describe(asset)}."
                return f"{intro} Looking at it now.", [shown]
        # Non-image, unviewable/oversized image, or a failed fetch: describe, don't show.
        return _incoming_asset_text(asset), []

    # --- webhook events ------------------------------------------------------

    def _wake_events(self, session: Session, event_trigger: str | None) -> list[object]:
        """Surface unseen inbound webhook deliveries and let the agent act on them.

        A received webhook event is *not* a timeline item the way a message or an
        activated task is — so a woken agent never sees one unless it goes looking.
        This is that lookup: the same idempotent high-water-mark discipline used for
        messages, over the SDK's webhook-events read surface. The router wakes with the
        timeline uuid alone, so on the first wake (no mark, no `event_trigger`) the agent
        acts on the **newest** unseen delivery — the one that almost certainly woke it.
        This is the fix for the live bug where a `webhook_event.received` wake surfaced
        nothing: the old first-wake baselined silently when no trigger was passed, and the
        router never passes one, so every first delivery was dropped. A named
        `event_trigger`, when a manual/future invocation passes one, acts on that delivery
        and everything newer.
        """
        mark = self.marks.get(self.timeline_uuid, kind=_EVENTS)
        if mark is None:
            return self._bootstrap_stream(
                session,
                self.client.webhook_events.filter(timeline=self.timeline_uuid),
                event_trigger,
                _EVENTS,
                self.client.webhook_events.get,
                self._respond_events,
            )
        unseen = _events_since(self.client.webhook_events.filter(timeline=self.timeline_uuid), mark)
        return self._respond_events(session, unseen)

    # --- the first wake for a creation-ordered stream (events, assets) --------

    def _bootstrap_stream(self, session, items_filter, trigger, kind, fetch_one, respond):
        """First wake for a creation-ordered, mark-tracked stream (webhook events, assets).

        There is no persisted mark yet. **The router wakes a harness agent with the
        timeline uuid alone — it never names the triggering item** (see basecradle-router
        `wake_command`), so in production `trigger` is always `None`. A no-trigger first
        wake therefore acts on the **newest** unseen item — the one that almost certainly
        woke us — exactly as the message bootstrap replies to the newest message on a fresh
        join; `respond` advances the mark to it (and applies the asset self-filter, so the
        agent's own newest post is marked-but-not-acted, never a wake-loop). Older items
        are left behind, not replayed, so a fresh agent is bounded to a single action
        rather than flooded by a backlog it was never woken for. Acting on the newest is
        what makes a webhook delivery (and a peer's posted asset) actually surface under
        the real router contract; baselining silently was the live bug where the first
        delivery of each kind was dropped.

        With a trigger (a manual or future-router invocation that does name the item), act
        on that item and everything newer, then baseline to the true newest so the next
        wake is incremental. `fetch_one` retrieves the trigger by uuid if a burst pushed it
        past the fetched window, so a named item is never silently dropped.

        **Known bound — newest-only acts on one item on a *cold* first wake.** This is the
        first reconcile of this kind *ever* (no mark yet), so "newest only" is intentional:
        it bounds a fresh agent to a single action instead of replaying a backlog. Two
        edges fall out of it, both strictly better than the old baseline-silently behavior
        (which acted on nothing) and both confined to the cold first wake — every wake
        after the mark exists acts on **all** unseen items past it:
        - *Cold-start burst:* if several items of this kind predate the very first wake,
          only the newest is acted on and the rest are marked seen. In the live router flow
          each new item triggers its own wake, so this only bites items delivered before the
          agent ever reconciled this kind.
        - *Self-authored newest:* if the newest item is the agent's own (a just-generated
          asset), the self-filter makes the wake a no-op and the mark still advances past an
          older unseen peer item. Acceptable for the same reason — in the per-event flow the
          peer item had its own earlier wake.
        If a future need makes dropping a cold-start burst unacceptable (e.g. webhook
        deliveries that must each be processed), act on all of `recent` here for that kind
        rather than `recent[0]`, accepting the join-time replay that trades against.
        """
        recent = _recent(items_filter, self.context_messages)
        if trigger is None:
            if not recent:
                return []
            # Act on the newest unseen item; `respond` advances the mark to it. See the
            # "Known bound" note above for why a cold first wake acts on only the newest.
            return respond(session, [recent[0]])
        to_act = self._from_trigger(recent, trigger, fetch_one)
        if not to_act:
            return []  # no such item and an empty stream: nothing to do, nothing to mark
        posted = respond(session, to_act)
        newest = recent[0].content.uuid if recent else to_act[-1].content.uuid
        self.marks.set(self.timeline_uuid, newest, kind=kind)
        return posted

    def _from_trigger(self, recent: list[object], trigger: str, fetch_one) -> list[object]:
        """The items to act on for a named trigger, oldest-first.

        Normally the trigger is in the fetched window: act on it and everything newer. If
        a burst pushed it *past* the window (more than `context_messages` newer items
        arrived before the wake fired), the windowed scan would miss it — so fetch the
        named item directly and act on it together with the window, rather than silently
        dropping the one item the router explicitly woke us for.
        """
        for i, item in enumerate(recent):
            if item.content.uuid == trigger:
                return list(reversed(recent[: i + 1]))  # trigger and newer, chronological
        # Trigger not in the window: fetch it directly so it is never dropped, and act on
        # it before the window (it is older than everything fetched).
        try:
            triggered = fetch_one(trigger)
        except BaseCradleError:
            triggered = None  # gone or unreadable — act on what we can still see
        window = list(reversed(recent))  # all newer than the trigger, chronological
        return [triggered, *window] if triggered is not None else window

    def _respond_events(self, session: Session, events: list[object]) -> list[object]:
        """Act on each webhook delivery in order, advancing the event mark per delivery.

        A recognized NOC probe — read from the delivery's **payload** — is acked token-free
        before the model (`probe=`), plain at-least-once like messages (post the ack, then
        advance the mark): no `claim_first` subtlety here, that is the task seam's concern.
        The short-circuit runs *inside* `_act_on`, after `_bootstrap_stream` has already
        selected the item, so the #100 cold-first-wake bootstrap (newest unseen delivery
        only — which on a quiet probe timeline is the probe) is preserved unchanged.
        """
        return self._act_on(
            session,
            events,
            _incoming_event_text,
            lambda event: self.marks.set(self.timeline_uuid, event.content.uuid, kind=_EVENTS),
            probe=lambda event: self._probe_nonce(event.content.payload),
        )

    # --- activated tasks -----------------------------------------------------

    def _wake_tasks(self, session: Session) -> list[object]:
        """Carry out every activated-but-unhandled task on the timeline.

        A task activation is not a timeline item the wake's message scan would surface,
        so — like a webhook delivery — the agent must go looking. Unlike messages and
        events, an activated task is not a creation-ordered stream a high-water mark can
        track (a task scheduled earlier can activate later) and carries no terminal
        status the agent could set, so idempotency is a persisted *seen-set* (see
        `SeenStore`): act on each activated task whose uuid is not yet recorded. The task
        is recorded **before** it is acted on (`claim_first`), not after — a task is
        at-most-once, so an action that re-wakes the agent (a generated image posts an
        `asset.created`) cannot re-surface the still-`activated` task and re-run it. That
        re-fire was the live monkey pile-up: the task stayed `activated`, its image-post
        re-woke the agent, and an act-then-record order let the unrecorded task fire again
        and again. An activated-but-unhandled task is genuinely undone work — not stale
        history — so acting on all of them is correct, and needs no router-passed
        trigger, which keeps the router thin.

        The full activated list is drained (no `context_messages` window): a task is
        acted on one at a time, never seeded as history, so there is no token-cost
        argument for a cap — and a cap would silently drop tasks beyond it, since a
        newly-due task can sit anywhere in the (creation-ordered) list and no high-water
        mark guarantees a re-scan. The per-wake cost is the count of activated tasks on
        the timeline, the same order as the seen-set read this already does.
        """
        activated = list(self.client.tasks.filter(timeline=self.timeline_uuid, status="activated"))
        if not activated:
            return []
        seen = self.seen.all(self.timeline_uuid, kind=_TASKS)
        # Oldest-first, so the agent works the backlog in the order it was scheduled.
        unhandled = [task for task in reversed(activated) if task.content.uuid not in seen]
        return self._act_on(
            session,
            unhandled,
            _activated_task_text,
            lambda task: self.seen.add(self.timeline_uuid, task.content.uuid, kind=_TASKS),
            claim_first=True,
            # A NOC probe task is recognized from its **instructions** and acked token-free.
            # `_act_on` checks `probe` *before* `claim_first`, so the probe is acked
            # at-least-once (post, then record) — NOT claim-first. That is correct and must
            # be preserved: a probe's only side-effect is @jt's own ack (self-filtered, and
            # router wakes are serialized), so there is no re-fire hazard to guard against,
            # and at-least-once is the safe failure direction for a monitor — a crash before
            # recording re-acks (harmless) rather than marking the task seen with no ack ever
            # posted, which would manufacture a false FAIL. See task-seam.md §4.
            probe=lambda task: self._probe_nonce(task.content.instructions),
        )

    # --- the shared act-on loop ----------------------------------------------

    def _act_on(
        self,
        session: Session,
        items,
        render,
        record,
        skip=None,
        claim_first: bool = False,
        probe=None,
    ) -> list[object]:
        """Engage the model on each item in order, posting any reply and recording it.

        The one loop behind all four reconcilers — messages, webhook events, activated
        tasks, and posted assets. `render` turns an item into the text the model reads, or
        a `(text, images)` pair when the item carries pictures to *show* the model (the
        asset-perception path); `record` marks it handled (a mark advance, or a seen-set add).

        `probe(item)` is the **NOC synthetic-probe short-circuit**, passed by all four
        reconcilers — messages, webhook events, activated tasks, and posted assets — each
        reading the marker from its own carrier field (a message body, a webhook payload, a
        task's instructions, an asset's description). When it returns a nonce (the item is a
        valid signed probe — see
        `_probe`), the loop posts the deterministic ack and records the item **without
        calling the model** — no provider request, no tokens, nothing into the transcript —
        so the message-, webhook-, task-, and asset-seam heartbeats all run token-free at
        rest (an asset probe is acked before its file is ever fetched). It
        is checked *after* the self-filter (a probe is from a distinct peer, never the
        agent's own) and *before* both `claim_first` and `render`/`session.send`, and
        follows the same at-least-once order as a normal reply (post, then record), so a
        crash mid-batch re-acks — a duplicate ack is harmless (the prober matches the first
        ack carrying its nonce). The probe-before-`claim_first` order is **load-bearing for
        the task seam**: a probe task is acked at-least-once, never claim-first, so a crash
        before recording re-acks rather than silently marking it seen with no ack (which
        would manufacture a false monitor FAIL — task-seam.md §4).

        **Two recording disciplines, chosen by `claim_first`:**

        - *At-least-once* (`claim_first=False`, the default — messages, events, assets):
          the reply is posted *before* the record advances, so a crash or router retry
          mid-batch re-acts on the un-recorded item. A possible duplicate over a dropped
          action is the better failure on a comms platform.
        - *At-most-once* (`claim_first=True` — activated tasks): the record advances
          *before* the action runs, so an action that fails part-way (most importantly
          one that already posted a side effect — a generated image whose `asset.created`
          will wake the agent again) can **never re-fire**. A self-scheduled task must run
          exactly once even if its own output re-wakes the agent; a one-time dropped task
          on a crash is far better than the runaway re-execution it would otherwise cause.

        Either way the record advances after (or before) *each* item independently, so the
        batch is crash-resumable at item granularity.

        `skip(item)` is the **actor self-filter** — the safety property. When it returns
        true (the item is the agent's *own* authored post — a message it sent, an image
        it generated), the item is *not* acted on, but its record **still advances**, so
        the agent never reacts to its own output and never wake-loops on it (a generated
        image that re-woke the agent to generate another would be a runaway), while the
        skipped item is still marked seen so it is not re-scanned forever.

        `skip` and `claim_first` are independent axes that **may** both apply: skip is
        checked first, so an own item is recorded once and never acted on regardless of
        `claim_first`. (No current reconciler passes both — tasks are at-most-once with no
        self-filter; assets/messages self-filter at-least-once — but the precedence is
        well-defined if one ever does.)
        """
        posted = []
        for item in items:
            if skip is not None and skip(item):
                record(item)  # own post: never acted on, but marked so it is not re-scanned
                continue
            if probe is not None and (nonce := probe(item)) is not None:
                # A verified NOC probe: ack token-free (no model call), then record — the
                # same at-least-once order as a normal reply, so a crash re-acks safely.
                posted.append(self.timeline.messages.create(body=ack_line(nonce)))
                record(item)
                continue
            if claim_first:
                record(item)  # at-most-once: claim before acting so a crash cannot re-fire it
            # `render` returns the text the model reads, or a (text, images) pair when the
            # item carries pictures to *show* the model (the asset-perception path). Images
            # ride into the model's input on this turn and are evicted after (see `Session`).
            rendered = render(item)
            text, images = rendered if isinstance(rendered, tuple) else (rendered, [])
            reply = session.send(text, images=images)
            if reply.strip():
                posted.append(self.timeline.messages.create(body=reply))
            if not claim_first:
                record(item)  # at-least-once: mark after the post, so a crash re-acts
        return posted

    def _is_own(self, item: object) -> bool:
        """Whether an item was authored by this agent — the actor self-filter's test."""
        return item.user.uuid == self.me_uuid

    def _probe_nonce(self, carrier: str) -> str | None:
        """The nonce if `carrier` holds a valid signed NOC probe, else `None` (the short-circuit).

        Off unless `NOC_PROBE_SECRET` was provisioned (`probe_secret` set and non-empty):
        with no secret there is nothing to verify against, so every item falls through to
        the model, exactly as before this capability existed. A falsy secret (`None` *or*
        an empty string) keeps it off, so a blank env var never enables verification against
        an empty key. `carrier` is the field each reconciler reads the marker from — a
        message body, a task's instructions, a webhook payload, or an asset's description —
        so one signed-marker check serves all four seams (the NOC's message, task, webhook,
        and asset contracts).
        """
        if not self.probe_secret:
            return None
        return verify_probe(carrier, self.probe_secret)

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

        Our own posts are skipped (the actor self-filter) but still advance the mark, and a
        recognized NOC probe — read from the message **body** — is acked token-free before
        the model (`probe=`). The task and webhook reconcilers pass the same `probe=` over
        their own carrier fields, so the short-circuit spans all three seams. Delegates to
        the shared `_act_on` loop — see it for the crash-safe, at-least-once ordering that
        every reconciler shares.
        """
        return self._act_on(
            session,
            messages,
            _incoming_text,
            lambda message: self.marks.set(self.timeline_uuid, message.content.uuid),
            skip=self._is_own,
            probe=lambda message: self._probe_nonce(message.content.body),
        )


def _has_conversation(session: Session) -> bool:
    """Has this session said or heard anything yet (beyond a seeded system charter)?"""
    return any(turn.role in ("user", "assistant") for turn in session.history)


# A webhook event and an asset each share a message's newest-first ordering and
# `.content.uuid` shape, so the same high-water-mark scan walks all three. Aliased
# (not re-implemented) to keep them in lockstep.
_events_since = _messages_since
_assets_since = _messages_since


def _asset_marker_carrier(asset: object) -> str:
    """The asset field the NOC synthetic-probe marker rides in — the 4th seam's carrier.

    The asset analog of a message body / task instructions / webhook payload. The chosen
    field is the asset's **description**: the one free-text field a peer (or the NOC
    prober) controls on an upload, so a synthetic asset probe is minted by creating a tiny
    asset whose `description` is the `BCNOC1 …` marker. This is the contract the NOC's
    asset probe must agree with byte-for-byte. The field is optional, so an absent (or
    `None`) description reads as empty — an ordinary file with no description simply falls
    through to perception, never mistaken for a probe.
    """
    return getattr(asset.content, "description", None) or ""


def _incoming_asset_text(asset: object) -> str:
    """A peer's posted file as the agent hears it: who shared what, and how to open it.

    The description fallback for the asset-perception path: used for media the wake cannot
    show inline (a non-image file, an unviewable/oversized image, or one whose download
    failed). Surfaces the asset's metadata (the shared `_describe` rendering) and points
    the agent at the tools that actually open it, rather than inlining the bytes — looking
    is a deliberate, on-demand step, the same discipline `view`/`read`/`listen` follow.
    """
    return (
        f"{asset.user.handle} posted a file to this timeline: {_describe(asset)}. "
        "Use the assets tool to 'read' it (or 'view' an image / 'listen' to audio) if "
        "you want to engage with it."
    )


# How much of a webhook payload to put in front of the model directly. A large body
# is truncated with a pointer to the webhook_events tool, which reads it in full —
# the same describe-don't-dump discipline the assets tool uses.
_MAX_EVENT_PAYLOAD = 8 * 1024


def _incoming_event_text(event: object) -> str:
    """An inbound webhook delivery as the agent hears it: what arrived, and its payload.

    A large payload is truncated with a pointer to the `webhook_events` tool (which
    reads the full headers and body by uuid), so a firehose delivery cannot blow up
    the model's context.
    """
    content = event.content
    payload = content.payload
    if len(payload) > _MAX_EVENT_PAYLOAD:
        payload = (
            payload[:_MAX_EVENT_PAYLOAD].rstrip()
            + f"\n… (payload truncated — use the webhook_events tool to read {content.uuid} in full)"
        )
    return (
        "An inbound webhook was just delivered to this timeline "
        f"(event {content.uuid}, endpoint {event.webhook_endpoint.uuid}, "
        f"content_type {content.content_type}). Decide whether and how to act on it. "
        f"Its payload:\n{payload}"
    )


def _activated_task_text(task: object) -> str:
    """A newly-activated task as the agent hears it: its instructions, to carry out now.

    A task is work the agent (or a peer) scheduled for this moment, so it is phrased as
    a directive to act, not a notification to consider — the activation *is* the cue.
    """
    content = task.content
    return (
        "A task you scheduled has activated and is due now "
        f"(task {content.uuid}, scheduled for {content.activate_at}). "
        f"Carry out its instructions:\n{content.instructions}"
    )


def main(argv: list[str] | None = None) -> int:
    """The `basecradle-harness-wake` entrypoint: one wake, then exit.

    Exit code 0 on success including "nothing to do"; non-zero on a hard
    config/auth/credential failure, so the router can surface it.
    """
    parser = argparse.ArgumentParser(
        prog="basecradle-harness-wake",
        description="Answer one BaseCradle timeline's unseen messages, then exit (router wake mode).",
    )
    # A token-free, model-free, timeline-free way to ask a deployed box "what version
    # are you actually running?" — the cheap probe a fleet drift-guard runs on-box to
    # catch a published-but-not-deployed release before it goes silent. Prints
    # "basecradle-harness-wake <version>" and exits 0 (argparse's built-in action).
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="print the installed basecradle-harness version and exit.",
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
    parser.add_argument(
        "--event",
        default=os.environ.get("BASECRADLE_EVENT"),
        help=(
            "optional uuid of the triggering webhook event; the router passes it on a "
            "webhook_event.received wake so the first wake acts on that delivery."
        ),
    )
    parser.add_argument(
        "--asset",
        default=os.environ.get("BASECRADLE_ASSET"),
        help=(
            "optional uuid of the triggering asset; the router passes it on an "
            "asset.created wake so the first wake perceives that posted file."
        ),
    )
    args = parser.parse_args(argv)
    if not args.timeline:
        parser.error("a timeline uuid is required (--timeline or BASECRADLE_TIMELINE)")

    try:
        agent = WakeAgent.from_env(timeline=args.timeline)
        agent.wake(trigger=args.message, event_trigger=args.event, asset_trigger=args.asset)
    except (HarnessError, ProviderError, ValueError, KeyError) as error:
        print(f"basecradle-harness-wake: {error}", file=sys.stderr)
        return 1
    return 0
