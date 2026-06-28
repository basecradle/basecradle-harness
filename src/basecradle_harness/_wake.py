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
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from urllib.parse import quote

import httpx
from basecradle import BaseCradle, BaseCradleError

from basecradle_harness._assets import _describe, _is_image, image_input
from basecradle_harness._basecradle import (
    DEFAULT_CONTEXT_MESSAGES,
    _as_turn,
    _client_from_env,
    _config_from_env,
    _context_messages_from_env,
    _incoming_text,
    _messages_since,
    _onboard_from_env,
    _recent,
    _resolve_tools,
    _resolve_tools_and_provider,
)
from basecradle_harness._brief import (
    compose_brief,
    fetch_dashboard_md,
    render_defects,
    render_manifest,
    render_safety,
)
from basecradle_harness._code import CodeExecutionBridge
from basecradle_harness._exceptions import EngineError, HarnessError, ProviderError
from basecradle_harness._harness import Harness
from basecradle_harness._install import charter_from_env, prompt_text, system_prompt_text
from basecradle_harness._memory_provider import MemoryExchange, MemoryProvider, MemoryScope
from basecradle_harness._messages import ImageContent
from basecradle_harness._platform import PlatformContext, bind_platform_tools, explain
from basecradle_harness._probe import ack_line, verify_probe
from basecradle_harness._session import Session
from basecradle_harness._version import __version__

_log = logging.getLogger("basecradle_harness")

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


class ClaimStore:
    """Per-item atomic claims, so concurrent wakes handle each item exactly once.

    A high-water mark (`MarkStore`) and a seen-set (`SeenStore`) make a wake idempotent
    across *sequential* processes — a later wake reads the record and skips. Neither is
    safe across *concurrent* ones: two wakes firing at once (an upload posts
    `asset.created` and `message.created` together, so the router spawns two) both read the
    same record, both find the same message unseen, and both reply — the live duplicate, or
    worse a duplicated tool action. A claim closes that race with the one operation a POSIX
    filesystem makes atomic: an exclusive create. The first wake to create
    `<root>/claims/<kind>/<timeline>/<uuid>` wins and acts; a concurrent create raises
    `FileExistsError`, so the loser knows the item is already owned and skips it.

    Claiming *before* acting also subsumes single-process crash-idempotency: a wake that
    claims an item and then dies (or fails partway) leaves the claim behind, so the retry
    skips it rather than re-running its turn and re-firing its tool actions — the live
    reprocess loop. This is the at-most-once discipline a message needs: a one-time dropped
    reply on a crash is far better than a backlog re-answered, with tool side effects, on
    every later wake.

    **Known bound — claims are not pruned.** One tiny empty file accrues per handled item,
    the same unbounded-growth shape the task `SeenStore` already has. Items are small and
    the files are empty; if it ever matters, claims at or below a kind's high-water mark are
    dead (that item is never re-scanned) and prunable by UUIDv7 order. Out of scope here.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def claim(self, timeline: str, uuid: str, *, kind: str) -> bool:
        """Atomically claim `(kind, timeline, uuid)`. True if this wake won it, else False.

        Wins by exclusively creating the claim file: the first caller succeeds, any
        concurrent (or later) caller gets `FileExistsError` and is told the item is already
        owned. The create is the whole synchronization — no lock, no read-modify-write race.
        """
        path = self._path(timeline, kind, uuid)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        os.close(handle)
        return True

    def _path(self, timeline: str, kind: str, uuid: str) -> Path:
        folder = self.root / "claims" / kind / quote(timeline, safe="")
        return folder / f"{quote(uuid, safe='')}.claim"


# The wake-breaker's generous safe defaults (Phase 2 · Group 6). A genuine cross-wake
# runaway fires continuously — many wakes per second — so over a 60 s window it racks up
# far more than this cap, while a human-paced multi-peer conversation almost never reaches
# 10 *inbound* items to one timeline in a minute (the agent's own replies are self-filtered
# and never wake it, so only peer items count). Tunable via env for the rare firehose
# timeline; the router's cross-agent breaker (basecradle-router) is the complementary layer.
_DEFAULT_BREAKER_MAX = 10
_DEFAULT_BREAKER_WINDOW = 60.0

# The breaker's loud alert text. Full sentences, so sentence case with terminal punctuation
# (the Title-Case exception). The trip alert is posted once on the trip transition (the
# durable marker is the one-time guard) and the reset alert once on auto-reset.
_BREAKER_TRIP_ALERT = (
    "I appear to be in a wake loop here, so I'm pausing to avoid a runaway. "
    "I'll resume automatically once the burst clears; an operator can also reset me."
)
_BREAKER_RESET_ALERT = "The wake burst here has cleared, so I've resumed normal operation."


@dataclass(frozen=True)
class BreakerDecision:
    """The wake-breaker's verdict for one wake: what it decided, and why.

    `short_circuit` is the load-bearing field — when True this wake **self-declines**: it
    makes no provider call and acts on nothing. `tripped` and `reset` flag the one-time
    state *transitions* (a trip or an auto-reset happened on *this* wake), so the caller
    posts the loud alert exactly once per cycle rather than on every tripped wake. `count`
    is the number of wakes counted in the rolling window, for the alert and the log line.
    """

    short_circuit: bool
    tripped: bool
    reset: bool
    count: int


class WakeBreaker:
    """Per-timeline cross-wake circuit-breaker — the backstop for an *unknown* runaway loop.

    The runaway this defends against is a **cross-wake loop**: the agent is woken, it posts,
    the post fires a platform event, the router wakes it again → a tight cycle burning
    provider tokens and box resources. The in-wake `max_steps` cap, the actor self-filter,
    and the known B3/B8 fixes each stop a *specific* loop; this is the generic backstop for a
    *novel* one — most plausibly introduced by a custom `tools/` plugin (Group 2) or a
    drop-in MCP server (Group 5).

    It is a rolling-window rate limiter on **wakes per timeline**, persisted beside the
    `marks/`/`seen/`/`claims/` stores under the agent's home so it survives the
    process-per-wake model:

    - `breaker/<timeline>.wakes` — the timestamps of recent wakes, pruned to the window on
      every wake (so the file stays bounded even under a fast runaway).
    - `breaker/<timeline>.tripped` — the durable **trip marker**: present iff the timeline is
      currently tripped, holding the trip timestamp.

    On each wake `record_and_check` records the wake and returns a `BreakerDecision`:

    - Over the cap within the window → **TRIP**: write the marker, return `short_circuit`
      (the wake self-declines, **no provider call** — the whole point is to stop the burn)
      with `tripped=True` so the caller alerts once.
    - Already tripped → keep short-circuiting (every wake is still *counted*, so a runaway
      that keeps firing keeps the window saturated and stays tripped).
    - **Auto-reset** (the preferred reset, stated in CLAUDE.md): once the burst subsides —
      the window clears back under the cap *and* the cooldown has elapsed since the trip —
      clear the marker, restart the window from now, and return `reset=True` (normal
      operation resumes; the caller posts the recovery alert). A transient burst self-heals
      while the loud alert still leaves a human a breadcrumb. Clearing the marker by hand is
      the equivalent operator reset.

    Disabled by setting the cap to 0 (or below) — an operator escape hatch; the default is a
    generous always-on sanity cap.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_wakes: int = _DEFAULT_BREAKER_MAX,
        window: float = _DEFAULT_BREAKER_WINDOW,
        cooldown: float | None = None,
        now=None,
    ) -> None:
        self.root = Path(root)
        self.max_wakes = max_wakes
        self.window = float(window)
        # The cooldown defaults to the window: hysteresis so a trip cannot reset until at
        # least one clear window has passed, which prevents flapping at the threshold.
        self.cooldown = float(cooldown) if cooldown is not None else float(window)
        # An injectable clock keeps the breaker deterministically testable; production uses
        # the wall clock (a synthetic burst in a test drives `now` directly).
        self._now = now or time.time

    @classmethod
    def from_env(cls, root: str | Path, *, now=None) -> WakeBreaker:
        """Build a breaker from `HARNESS_WAKE_BREAKER_MAX`/`_WINDOW`/`_COOLDOWN` (generous defaults)."""
        return cls(
            root,
            max_wakes=_breaker_max_from_env(),
            window=_breaker_window_from_env(),
            cooldown=_breaker_cooldown_from_env(),
            now=now,
        )

    @property
    def enabled(self) -> bool:
        """Off when the cap is 0 or below — the operator escape hatch."""
        return self.max_wakes > 0

    def tripped(self, timeline: str) -> bool:
        """Whether this timeline currently holds a durable trip marker."""
        return self._read_trip(timeline) is not None

    def record_and_check(self, timeline: str) -> BreakerDecision:
        """Record this wake for `timeline` and decide whether it must self-decline.

        Always appends the wake to the rolling window first (a tripped wake is still counted,
        so a continuing runaway keeps the window saturated and stays tripped), then evaluates
        trip/reset state. See the class docstring for the state machine.
        """
        if not self.enabled:
            return BreakerDecision(short_circuit=False, tripped=False, reset=False, count=0)
        now = self._now()
        recent = [t for t in self._read_window(timeline) if t > now - self.window]
        recent.append(now)
        self._write_window(timeline, recent)
        count = len(recent)

        trip_at = self._read_trip(timeline)
        if trip_at is not None:
            # Currently tripped. Auto-reset only once the burst has genuinely subsided — the
            # window cleared back under the cap — *and* the cooldown has elapsed since the
            # trip, so a runaway still firing every few seconds keeps it tripped.
            if count <= self.max_wakes and now - trip_at >= self.cooldown:
                self._clear_trip(timeline)
                self._write_window(timeline, [now])  # fresh window: re-measure from here
                return BreakerDecision(short_circuit=False, tripped=False, reset=True, count=count)
            return BreakerDecision(short_circuit=True, tripped=False, reset=False, count=count)

        if count > self.max_wakes:
            self._write_trip(timeline, now)
            return BreakerDecision(short_circuit=True, tripped=True, reset=False, count=count)
        return BreakerDecision(short_circuit=False, tripped=False, reset=False, count=count)

    # --- storage -------------------------------------------------------------

    def _window_path(self, timeline: str) -> Path:
        return self.root / "breaker" / f"{quote(timeline, safe='')}.wakes"

    def _trip_path(self, timeline: str) -> Path:
        return self.root / "breaker" / f"{quote(timeline, safe='')}.tripped"

    def _read_window(self, timeline: str) -> list[float]:
        path = self._window_path(timeline)
        if not path.exists():
            return []
        out: list[float] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(float(line))
            except ValueError:
                continue  # a corrupt line is dropped, never crashes the wake
        return out

    def _write_window(self, timeline: str, times: list[float]) -> None:
        path = self._window_path(timeline)
        path.parent.mkdir(parents=True, exist_ok=True)
        # `repr` round-trips a float exactly, so a re-read window is byte-faithful.
        path.write_text("".join(f"{t!r}\n" for t in times))

    def _read_trip(self, timeline: str) -> float | None:
        path = self._trip_path(timeline)
        if not path.exists():
            return None
        try:
            return float(path.read_text().strip())
        except ValueError:
            return None

    def _write_trip(self, timeline: str, when: float) -> None:
        path = self._trip_path(timeline)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(repr(when))

    def _clear_trip(self, timeline: str) -> None:
        self._trip_path(timeline).unlink(missing_ok=True)


def _breaker_max_from_env() -> int:
    """`HARNESS_WAKE_BREAKER_MAX` → the wake cap; unset/blank → the generous default."""
    raw = os.environ.get("HARNESS_WAKE_BREAKER_MAX")
    if raw is None or not raw.strip():
        return _DEFAULT_BREAKER_MAX
    return int(raw)


def _breaker_window_from_env() -> float:
    """`HARNESS_WAKE_BREAKER_WINDOW` → the rolling-window seconds; unset/blank → the default."""
    raw = os.environ.get("HARNESS_WAKE_BREAKER_WINDOW")
    if raw is None or not raw.strip():
        return _DEFAULT_BREAKER_WINDOW
    return float(raw)


def _breaker_cooldown_from_env() -> float | None:
    """`HARNESS_WAKE_BREAKER_COOLDOWN` → the reset cooldown seconds; unset/blank → the window."""
    raw = os.environ.get("HARNESS_WAKE_BREAKER_COOLDOWN")
    if raw is None or not raw.strip():
        return None  # default: tie the cooldown to the window length
    return float(raw)


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
        onboard: Re-assert the persistent operating brief on every wake (see
            `_reassert_brief`). On by default. When on, the brief — `initialize.md`
            + the live tool manifest + the live `dashboard.md` + `system-prompt.md`
            — supersedes a static turn-0 charter, so the agent's standing context
            stays recent in a long transcript rather than aging out at turn 1. Off
            wakes with only the operator's charter, seeded once at turn 0.
        tool_manifest: ``(name, note)`` for the agent's active tools, rendered into
            the brief so it names exactly what the model can call. Defaults to the
            harness's registered function tools (no notes) when not supplied;
            `from_env` threads the precise `ResolvedTools.manifest` (built-ins and
            notes included).
        memory_provider: The agent's pluggable memory backend (see
            `basecradle_harness._memory_provider`). Its `observe` hook fires after every
            exchange and its `context` hook injects recalled memory into the persistent
            brief. ``None`` (the default for a directly-constructed wake) disables both
            hooks — the memory tool, if any, still comes from the harness's registry.
            `from_env` passes the env-selected provider (default SQLite, whose hooks are
            no-ops, so behavior is unchanged).
        breaker: The cross-wake circuit-breaker (see `WakeBreaker`). ``None`` (the default)
            constructs one over the harness's home with the generous default cap; `from_env`
            threads the env-tuned breaker (`HARNESS_WAKE_BREAKER_MAX`/`_WINDOW`/`_COOLDOWN`).
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
        tool_manifest: list[tuple[str, str | None]] | None = None,
        memory_provider: MemoryProvider | None = None,
        safety_notices: list[str] | None = None,
        defect_notices: list[str] | None = None,
        breaker: WakeBreaker | None = None,
        code_bridge: CodeExecutionBridge | None = None,
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
        self.onboard = onboard
        self.tool_manifest = tool_manifest
        self.memory_provider = memory_provider
        # Safe-by-default opt-out notices (active MCP servers, policy-refused drop-in tools)
        # surfaced into the persistent brief, so "all bets off" is stated and auditable —
        # empty for a pure-Harness config.
        self.safety_notices = safety_notices
        # Broken-shipped-default defect notices surfaced into the brief under their own loud
        # heading (issue #160), so a capability silently disabled by a stale overlay or a
        # packaging bug is impossible to miss — empty when every shipped default loaded.
        self.defect_notices = defect_notices
        # The persistent brief is composed and injected at most once per `wake()` — lazily,
        # right before the first time the model is actually engaged — so an idle or probe-only
        # wake neither fetches the live dashboard nor bloats the transcript. Reset each wake.
        self._brief_asserted = False
        # The shared HMAC key for the NOC synthetic-probe marker (see `_probe`). Set → the
        # message, webhook, and task reconciles each recognize a signed probe in their own
        # carrier field and ack it token-free, before the model. Unset → the short-circuit
        # is off and every item goes to the model.
        self.probe_secret = probe_secret
        self.marks = marks or MarkStore(harness.home)  # type: ignore[arg-type]
        # Activated tasks are tracked by a seen-set, not a high-water mark (see
        # `SeenStore`); it lives beside the marks, under the same home root.
        self.seen = SeenStore(self.marks.root)
        # Per-item atomic claims (see `ClaimStore`): the exactly-once guard that makes
        # `_act_on` safe across concurrent wakes and crash-resumable without reprocessing.
        # Lives beside the marks and the seen-set, under the same home root.
        self.claims = ClaimStore(self.marks.root)
        # Cross-wake circuit-breaker (see `WakeBreaker`): the generic backstop for an unknown
        # runaway wake loop. Records every wake and self-declines (no provider call) over the
        # cap. Lives beside the other stores, under the same home root. A directly-constructed
        # wake gets the generous defaults; `from_env` threads the env-tuned breaker.
        self.breaker = breaker or WakeBreaker(self.marks.root)
        self.timeline = self.client.timelines.get(timeline)

        # Bind the live platform handle into every platform-aware tool — the same
        # seam the poll loop uses, so a router-woken peer can act on the timeline
        # exactly as a polling one. One wake serves one timeline; bind once. The
        # code-execution bridge (when active) is bound the same way, and rides the
        # context so the `code_attach` tool can reach it.
        self.code_bridge = code_bridge
        context = PlatformContext(
            client=self.client,
            timeline=self.timeline_uuid,
            home=self.harness.home,
            code_bridge=code_bridge,
        )
        bind_platform_tools(self.harness.tools, context)
        if code_bridge is not None:
            code_bridge.bind(context)

        # One Dashboard read answers "who am I?" — the identity uuid the actor self-filter
        # tests every item against. (The brief's orientation is the *live* `dashboard.md`
        # primer, fetched per wake in `_reassert_brief`, not this structured read.)
        self.me_uuid = self.client.me.identity.uuid
        if onboard:
            # The persistent brief is re-asserted on every wake (see `_reassert_brief`) and
            # carries the personality charter (`system-prompt.md`) itself, so a static turn-0
            # seed would only duplicate it. Clear it; the brief is the charter now. With
            # onboarding off the operator's turn-0 charter stands as before.
            self.harness.system_prompt = None

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
        provider, resolved, memory, bridge = _resolve_tools_and_provider()
        harness = Harness(
            provider,
            system_prompt=charter_from_env(),
            tools=resolved.tools,
            home=home,
            # The code-execution Asset bridge (when active) harvests a run's output files +
            # source into Assets after each code-exec turn, then feeds their uuids back. None
            # when code execution isn't opted in → the engine loop is unchanged.
            turn_hook=bridge.on_reply if bridge is not None else None,
        )
        return cls(
            harness,
            timeline=timeline or os.environ["BASECRADLE_TIMELINE"],
            client=_client_from_env(),
            context_messages=_context_messages_from_env(),
            onboard=_onboard_from_env(),
            code_bridge=bridge,
            # `or None` so a set-but-blank NOC_PROBE_SECRET (an exported-but-unfilled
            # secret) reads as *off*, not as "enabled with an empty HMAC key".
            probe_secret=os.environ.get("NOC_PROBE_SECRET") or None,
            # The active tool manifest (built-ins + notes) for the persistent brief, so it
            # names exactly the tools this config resolved — never a present-but-broken one.
            tool_manifest=resolved.manifest,
            # The env-selected memory provider, whose observe/context hooks fire in the wake
            # loop. Default SQLite → no-op hooks → behavior unchanged for @jt.
            memory_provider=memory,
            # Safe-by-default opt-out notices from tool resolution (active MCP servers,
            # policy-refused drop-ins). Empty by default → no safety section in the brief.
            safety_notices=resolved.notices,
            # Broken-shipped-default defects from tool resolution (issue #160). Empty when
            # every shipped default loaded → no defect section in the brief.
            defect_notices=resolved.broken,
            # The env-tuned cross-wake circuit-breaker, persisted under HARNESS_HOME beside
            # the marks/seen/claims stores. Generous defaults; off only if explicitly capped
            # to 0. The router's cross-agent breaker is the complementary layer.
            breaker=WakeBreaker.from_env(home),
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

        **The cross-wake circuit-breaker runs first.** Before any reconcile, this wake is
        recorded on the per-timeline `WakeBreaker`; if this timeline is in a runaway wake
        loop (too many wakes in the rolling window), the wake **self-declines** — it makes no
        provider call, posts a single loud alert on the trip transition, and returns nothing.
        It auto-resets once the burst clears. This is the backstop the in-wake `max_steps`
        cap and the actor self-filter don't cover: an *unknown* cross-wake loop.

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
        if self._breaker_short_circuits():
            return []  # a tripped timeline self-declines: no session, no provider call
        session = self.harness.session(self.source)
        self._brief_asserted = False  # re-assert the brief once this wake, before the model
        posted = self._wake_messages(session, trigger)
        posted += self._wake_assets(session, asset_trigger)
        posted += self._wake_events(session, event_trigger)
        posted += self._wake_tasks(session)
        return posted

    # --- the cross-wake circuit-breaker --------------------------------------

    def _breaker_short_circuits(self) -> bool:
        """Record this wake on the breaker; trip/reset loudly, and report whether to decline.

        The first thing a wake does — *before* the session is loaded or the model is ever
        engaged — so a tripped timeline self-declines token-free, exactly like the NOC probe
        ack short-circuit. Posts the loud alert **once** on each state transition (the
        durable trip marker is the one-time guard) with a WARNING log, then returns the
        breaker's verdict: True → this wake makes no provider call. A reset transition does
        *not* short-circuit — it alerts that the burst cleared, then the wake proceeds
        normally (`record_and_check` already restarted the window).

        **Known bound — a tripped timeline also stops acking NOC probes.** The probe ack
        lives per-item inside `_act_on`, downstream of this early return, so a tripped wake
        skips it along with everything else. That is acceptable: a probe timeline is quiet
        and low-cadence, so it never reaches the cap in practice, and a timeline genuinely in
        a runaway *failing* its heartbeat is honest signal, not a false FAIL — the loud trip
        alert is itself the louder out-of-band notice. We do **not** read the timeline to
        rescue a probe before deciding, because that would forfeit the whole point: a
        token-free decline before any platform work.
        """
        decision = self.breaker.record_and_check(self.timeline_uuid)
        if decision.tripped:
            _log.warning(
                "Wake breaker TRIPPED for timeline %s: %d wakes within %ss exceeds the cap "
                "of %d. Self-declining (no provider call) until the burst clears; an operator "
                "can reset by clearing the trip marker under HARNESS_HOME.",
                self.timeline_uuid,
                decision.count,
                self.breaker.window,
                self.breaker.max_wakes,
            )
            self._breaker_alert(_BREAKER_TRIP_ALERT)
        elif decision.reset:
            _log.warning(
                "Wake breaker RESET for timeline %s: the wake burst cleared; resuming normal "
                "operation.",
                self.timeline_uuid,
            )
            self._breaker_alert(_BREAKER_RESET_ALERT)
        return decision.short_circuit

    def _breaker_alert(self, body: str) -> None:
        """Post the breaker's loud alert straight to the timeline, degrading on refusal.

        Fired once per transition (trip or reset), so the alert never loops — and the actor
        self-filter keeps the agent from waking on its own alert post. There is no session
        and no model call: a tripped wake is model-free by definition, so the alert goes
        directly through the timeline client. A refused post (a locked timeline) is logged
        and swallowed — the breaker's job, stopping the burn, is already done.
        """
        try:
            self.timeline.messages.create(body=body)
        except BaseCradleError:
            _log.warning(
                "Couldn't post the wake-breaker alert to timeline %s (continuing).",
                self.timeline_uuid,
                exc_info=True,
            )

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
            kind=_ASSETS,
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
                intro = (
                    f"[{asset.created_at}] {asset.user.handle} posted a file to this "
                    f"timeline: {_describe(asset)}."
                )
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
        before the model (`probe=`), at-least-once (post the ack, then record): the probe
        seam is never claimed, so a refused ack re-acks next wake rather than going silent.
        The short-circuit runs *inside* `_act_on`, after `_bootstrap_stream` has already
        selected the item, so the #100 cold-first-wake bootstrap (newest unseen delivery
        only — which on a quiet probe timeline is the probe) is preserved unchanged.
        """
        return self._act_on(
            session,
            events,
            _incoming_event_text,
            lambda event: self.marks.set(self.timeline_uuid, event.content.uuid, kind=_EVENTS),
            kind=_EVENTS,
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
        `SeenStore`), now backed by an atomic per-task claim (`ClaimStore`) for safety across
        concurrent wakes too: act on each activated task whose uuid is not yet recorded. The
        task is recorded and claimed **before** it is acted on (claim-first), not after — a
        task is at-most-once, so an action that re-wakes the agent (a generated image posts
        an `asset.created`) cannot re-surface the still-`activated` task and re-run it. That
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
            kind=_TASKS,
            # A NOC probe task is recognized from its **instructions** and acked token-free.
            # `_act_on` checks `probe` *before* claiming, so the probe is acked at-least-once
            # (post, then record) and never claimed. That is correct and must be preserved: a
            # probe's only side-effect is @jt's own ack (self-filtered, and router wakes are
            # serialized), so there is no re-fire hazard to guard against, and at-least-once
            # is the safe failure direction for a monitor — a crash before recording re-acks
            # (harmless) rather than leaving the task claimed-but-unacked, which would
            # manufacture a false FAIL. See task-seam.md §4.
            probe=lambda task: self._probe_nonce(task.content.instructions),
        )

    # --- the shared act-on loop ----------------------------------------------

    def _act_on(
        self,
        session: Session,
        items,
        render,
        record,
        kind: str,
        skip=None,
        probe=None,
    ) -> list[object]:
        """Engage the model on each item in order, posting any reply and recording it.

        The one loop behind all four reconcilers — messages, webhook events, activated
        tasks, and posted assets. `render` turns an item into the text the model reads, or
        a `(text, images)` pair when the item carries pictures to *show* the model (the
        asset-perception path); `record` marks it handled (a mark advance, or a seen-set
        add); `kind` namespaces its atomic claim (see `ClaimStore`). **`kind` must match the
        namespace `record` writes to** — they are the same per-reconciler constant
        (`_MESSAGES`/`_ASSETS`/`_EVENTS`/`_TASKS`); if they ever diverge, claims land in one
        namespace while marks land in another and the exactly-once guard silently goes dead.

        **Exactly-once, claim-first (`ClaimStore`).** Before acting on an item, the loop
        *atomically claims* it. The first wake to claim wins and acts; a concurrent wake
        loses the claim and skips — so two near-simultaneous wakes (an upload firing
        `asset.created` + `message.created` spawns two) handle the same message exactly
        once instead of double-replying. Because the claim lands *before* the action, it
        also makes a crashed or partial wake non-reprocessing: the claim persists, so the
        retry skips the item rather than re-running its turn and re-firing its tool actions.
        The `record` (mark/seen advance) likewise lands before the action. The **claim is the
        authoritative exactly-once guard**; the high-water mark is a best-effort scan bound
        that may briefly rewind under truly concurrent wakes (two wakes claiming adjacent
        items in either order) — harmless, because a rewound mark only re-scans an item whose
        claim then blocks the duplicate. This is at-most-once by design —
        a one-time dropped reply on a hard crash beats a backlog re-answered, with side
        effects, on every later wake (the live reprocess loop). Tasks gain the same
        cross-process exclusivity their image-post re-fire never had.

        `probe(item)` is the **NOC synthetic-probe short-circuit**, passed by all four
        reconcilers — messages, webhook events, activated tasks, and posted assets — each
        reading the marker from its own carrier field (a message body, a webhook payload, a
        task's instructions, an asset's description). When it returns a nonce (the item is a
        valid signed probe — see `_probe`), the loop posts the deterministic ack and records
        the item **without calling the model** — no provider request, no tokens, nothing
        into the transcript — so the message-, webhook-, task-, and asset-seam heartbeats
        all run token-free at rest (an asset probe is acked before its file is ever
        fetched). It is checked *after* the self-filter (a probe is from a distinct peer,
        never the agent's own) and **before the claim**, and follows an at-least-once order
        (post the ack, *then* record), so a crash mid-batch re-acks. This probe-before-claim
        order is **load-bearing for the task seam**: a probe is never claimed, so a crash
        before recording re-acks (harmless — the prober matches the first ack carrying its
        nonce) rather than leaving the item claimed-but-unacked, which would manufacture a
        false monitor FAIL (task-seam.md §4).

        **Never crash the wake (B2).** Both platform-touching steps degrade instead of
        raising: a refused post (`self._post`, most pointedly a locked timeline) becomes an
        in-conversation note and the loop carries on; an engine that hits its step cap
        (`self._engage`) becomes a short "I got stuck" reply. So a wake that hits a locked
        timeline or the step cap posts a graceful note and exits 0 — no unhandled exception
        reaches the entrypoint, and the item is still recorded (no reprocess).

        `skip(item)` is the **actor self-filter** — the safety property. When it returns
        true (the item is the agent's *own* authored post — a message it sent, an image it
        generated), the item is *not* acted on, but its record **still advances** (no claim
        needed — there is no action to make exclusive), so the agent never reacts to its own
        output and never wake-loops on it, while the skipped item is still marked seen so it
        is not re-scanned forever.
        """
        posted = []
        for item in items:
            if skip is not None and skip(item):
                record(item)  # own post: never acted on, but marked so it is not re-scanned
                continue
            if probe is not None and (nonce := probe(item)) is not None:
                # A verified NOC probe: ack token-free (no model call), and record only
                # *after* a successful ack — at-least-once, never claimed. A refused ack
                # degrades *silently* (`note=False` keeps the probe seam trace-free) and
                # leaves the item unrecorded, so the next wake re-acks rather than marking it
                # seen with no ack ever posted (a false monitor FAIL — task-seam.md §4).
                ack = self._post(session, ack_line(nonce), note=False)
                if ack is not None:
                    posted.append(ack)
                    record(item)
                continue
            if not self.claims.claim(self.timeline_uuid, item.content.uuid, kind=kind):
                continue  # a concurrent (or crashed prior) wake already owns this item
            record(item)  # claim-first: mark seen before acting, so a failed wake won't reprocess
            # `render` returns the text the model reads, or a (text, images) pair when the
            # item carries pictures to *show* the model (the asset-perception path). Images
            # ride into the model's input on this turn and are evicted after (see `Session`).
            rendered = render(item)
            text, images = rendered if isinstance(rendered, tuple) else (rendered, [])
            reply = self._engage(session, text, images)
            if reply.strip():
                sent = self._post(session, reply)
                if sent is not None:
                    posted.append(sent)
                # The exchange happened — hand it to the memory provider to capture, whether
                # or not the post landed (a locked timeline still produced a real exchange).
                self._observe(text, reply)
        return posted

    def _engage(self, session: Session, text: str, images: list[ImageContent]) -> str:
        """Run the model on one item, degrading the engine's step-cap to a graceful reply.

        The think→act loop bounds runaway tool use with `max_steps` (`_engine`), raising
        `EngineError` when the model never settles on a reply. A wake must not crash on
        that — so it degrades to a short, honest note the peer can read, and the batch
        carries on. Other failures still propagate (the entrypoint reports them cleanly);
        this catches only the step-cap, the one the issue names.

        The persistent brief is re-asserted here, just before the user turn — so it is
        present and recent for *this* model call. It fires at most once per wake (lazily, so
        a probe-only or self-skip-only wake never pays the live dashboard fetch) and lands
        ahead of the item it should govern. The item's text is the retrieval query the memory
        provider's `context` hook ranks against, so recalled memory is relevant to this turn.
        """
        self._reassert_brief(session, query=text)
        try:
            return session.send(text, images=images)
        except EngineError:
            return "I got stuck working through that and stopped before reaching an answer."

    def _reassert_brief(self, session: Session, *, query: str | None = None) -> None:
        """Inject the persistent operating brief as a system turn — once per wake, when onboarding.

        This is what makes Turn 0 *persistent*: rather than a one-time onboarding seed that
        ages into the distant past of a long transcript, the brief is re-asserted at the head
        of every wake's work, so the agent's standing context (how to operate, what tools it
        has, where it is, who it is) is always recent in the conversation. Composed lazily and
        injected at most once per wake — a no-op when onboarding is off or already asserted.
        """
        if not self.onboard or self._brief_asserted:
            return
        self._brief_asserted = True  # set first: a failed compose/note still won't retry-loop
        brief = self._compose_brief(query)
        if brief:
            session.note(brief)
        else:
            # Onboarding is on yet the brief composed empty (every part absent — deleted
            # prompts, a failed dashboard, no tools). Not an error, but log it: an agent
            # waking with no standing context is worth a breadcrumb, not a silent gap.
            _log.info("Persistent brief composed empty this wake; proceeding without it.")

    def _compose_brief(self, query: str | None = None) -> str | None:
        """Compose the persistent brief: now + initialize + manifest + defects + safety + dashboard + memory + charter.

        The parts, in order (see `basecradle_harness._brief`): the **current-time anchor**
        (`_now_line` — the absolute "now" the model reasons every item's age against, fresh
        each wake since the brief is re-composed per wake), the provider-independent
        `initialize.md` operating guidance, the generated manifest of the agent's *active*
        tools, the **safe-by-default opt-out notice** (active MCP servers / policy-refused
        drop-ins — omitted when there are none), the live `dashboard.md` primer (fetched
        fresh each wake; a fetch failure degrades to omitting it, never breaking the wake),
        the memory provider's recalled
        **context** for this turn (its `context` hook — omitted when there is none or the
        provider's hook is a no-op), and the operator's `system-prompt.md` personality
        charter. Any part may be absent and the brief is composed from the rest.

        **Never break the wake.** `fetch_dashboard_md` already swallows its (network) failures
        and `_memory_context` swallows the provider's, but the prompt-file reads can also raise
        (a permission/IO error on `prompts/*.md` mid-wake) — so the whole composition is
        guarded too: any failure degrades to *no brief* and the wake carries on, the same
        invariant the dashboard fetch is held to.
        """
        try:
            return compose_brief(
                now=_now_line(),
                initialize=prompt_text("initialize.md"),
                manifest=render_manifest(self._manifest_entries()),
                defects=render_defects(self.defect_notices),
                safety=render_safety(self.safety_notices),
                dashboard=fetch_dashboard_md(self.client),
                memory=self._memory_context(query),
                system_prompt=system_prompt_text(),
            )
        except Exception:  # noqa: BLE001 - the brief must never break the wake; degrade to none
            _log.warning(
                "Failed to compose the persistent brief; proceeding without it.", exc_info=True
            )
            return None

    def _memory_context(self, query: str | None) -> str | None:
        """The memory provider's recalled context for this turn, guarded — never breaks the wake.

        Calls the provider's `context` hook with the agent-scoped `MemoryScope` (the query is
        the incoming turn's text, so a relevance-ranked provider retrieves against it). A no
        provider, or a provider whose hook is a no-op, yields ``None`` and the brief omits the
        section. A hook that *raises* degrades to ``None`` — a memory backend hiccup must not
        drop the whole brief, let alone the wake — distinct from `_compose_brief`'s outer guard
        so a memory failure costs only the memory section, not the rest of the brief.
        """
        if self.memory_provider is None:
            return None
        try:
            return self.memory_provider.context(
                MemoryScope(agent=self.me_uuid, timeline=self.timeline_uuid, query=query)
            )
        except Exception:  # noqa: BLE001 - a memory hook must never break the brief; degrade to none
            _log.warning(
                "Memory provider context() failed; omitting recalled memory.", exc_info=True
            )
            return None

    def _observe(self, user: str, assistant: str) -> None:
        """Hand a completed exchange to the memory provider's `observe` hook, guarded.

        Fires after each real exchange (never on a probe ack or a self-skip — those never
        reach here). A no provider, or one whose hook is a no-op (the default SQLite
        provider), does nothing. A hook that *raises* is swallowed: auto-capture is a
        best-effort side channel and must never break the wake or drop the reply that already
        posted.
        """
        if self.memory_provider is None:
            return
        try:
            self.memory_provider.observe(
                MemoryExchange(
                    user=user,
                    assistant=assistant,
                    scope=MemoryScope(agent=self.me_uuid, timeline=self.timeline_uuid),
                )
            )
        except Exception:  # noqa: BLE001 - a memory hook must never break the wake; swallow it
            _log.warning("Memory provider observe() failed; continuing.", exc_info=True)

    def _manifest_entries(self) -> list[tuple[str, str | None]]:
        """The ``(name, note)`` pairs for the tool manifest — the resolved set, else the registry.

        `from_env` threads the precise `ResolvedTools.manifest` (built-ins and notes included);
        a `WakeAgent` built directly (a test, or an embedder wiring its own `Harness`) falls
        back to the registered function tools by name, so the brief still names what the model
        can call even without the resolver's metadata.
        """
        if self.tool_manifest is not None:
            return list(self.tool_manifest)
        return [(tool.name, None) for tool in self.harness.tools]

    def _post(self, session: Session, body: str, *, note: bool = True) -> object | None:
        """Post a reply to the timeline, degrading any SDK refusal instead of crashing.

        The reply-post is the line that crashed the live wake: a locked timeline (even one
        self-locked earlier in the same turn) raises `TimelineLockedError`, and with no
        guard the whole wake died (`exit 1`) — *before* the message was marked seen, so the
        same prompt reprocessed on every later wake. Here any `basecradle` SDK error is
        caught and `None` is returned so the caller carries on (the item is still marked
        seen); the wake exits 0.

        For a normal reply (`note=True`) the failure is also recorded as a note in the
        transcript, so the agent's own record stays honest. A NOC probe ack passes
        `note=False`: the probe seam is deliberately trace-free (model-free, nothing into the
        transcript), so a refused ack must degrade *silently* — a note would both break that
        invariant and mislabel the heartbeat ack as a "reply".
        """
        try:
            return self.timeline.messages.create(body=body)
        except BaseCradleError as error:
            if note:
                session.note(f"(Couldn't post that reply to the timeline: {explain(error)})")
            return None

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
            kind=_MESSAGES,
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


def _now_line() -> str:
    """The current-time anchor injected at the head of every wake's brief.

    Renders the anchor line ``Current Time: 2026-06-21 17:09:49 UTC (+00:00, Sunday)`` —
    Title Case label, absolute UTC with an explicit ``+00:00`` offset, and the day-of-week,
    with no trailing period (it's a label, not a sentence) — **followed by** a one-line
    conversion instruction. The brief is re-composed and re-injected on every wake, so this is
    always current; the anchor is the reference every inbound item's ``[created_at]`` stamp is
    reasoned against. Accuracy rides on the host clock's NTP sync.

    The clock is UTC by design (every agent runs UTC on the box), but a bare UTC day/date was
    being parroted as if it were local — wrong whenever UTC has rolled to the next day but the
    asked-about locale hasn't (issue #180, live-confirmed on @jt). So the offset is now
    explicit and a conversion instruction rides with the anchor: when a peer names a locale,
    convert from UTC to that timezone first, because the local day can differ from the UTC day.
    """
    n = datetime.now(timezone.utc)
    anchor = f"Current Time: {n:%Y-%m-%d %H:%M:%S} UTC (+00:00, {n:%A})"
    instruction = (
        "This clock is UTC. For a question about a specific locale's date or time, convert "
        "from UTC to that timezone first — the local day can differ from the UTC day (e.g. "
        "US Central is UTC-5 in summer / UTC-6 in winter)."
    )
    return f"{anchor}\n{instruction}"


def _incoming_asset_text(asset: object) -> str:
    """A peer's posted file as the agent hears it: who shared what, and how to open it.

    The description fallback for the asset-perception path: used for media the wake cannot
    show inline (a non-image file, an unviewable/oversized image, or one whose download
    failed). Surfaces the asset's metadata (the shared `_describe` rendering) and points
    the agent at the tools that actually open it, rather than inlining the bytes — looking
    is a deliberate, on-demand step, the same discipline `view`/`read`/`listen` follow.

    The leading ``[created_at]`` stamp is the asset item's own timeline timestamp, read
    against the brief's `Current Time:` anchor so the model can reason about its age.
    """
    return (
        f"[{asset.created_at}] {asset.user.handle} posted a file to this timeline: "
        f"{_describe(asset)}. Use the assets tool to 'read' it (or 'view' an image / "
        "'listen' to audio) if you want to engage with it."
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

    The leading ``[created_at]`` stamp is the event item's own timeline timestamp, read
    against the brief's `Current Time:` anchor so the model can reason about its age.
    """
    content = event.content
    payload = content.payload
    if len(payload) > _MAX_EVENT_PAYLOAD:
        payload = (
            payload[:_MAX_EVENT_PAYLOAD].rstrip()
            + f"\n… (payload truncated — use the webhook_events tool to read {content.uuid} in full)"
        )
    return (
        f"[{event.created_at}] An inbound webhook was delivered to this timeline "
        f"(event {content.uuid}, endpoint {event.webhook_endpoint.uuid}, "
        f"content_type {content.content_type}). Decide whether and how to act on it. "
        f"Its payload:\n{payload}"
    )


def _activated_task_text(task: object) -> str:
    """A newly-activated task as the agent hears it: its instructions, to carry out now.

    A task is work the agent (or a peer) scheduled for this moment, so it is phrased as
    a directive to act, not a notification to consider — the activation *is* the cue.

    The leading ``[created_at]`` stamp is the task **item**'s own timeline timestamp. A task's
    Item is created when it *activates* (the platform's `Task::ActivationJob` inserts the Item
    in the same transaction it flips the task to activated), not when it was scheduled — so
    this reads ≈ now, consistent with every other inbound item, never "days ago." The
    complementary ``scheduled for {activate_at}`` text stays.
    """
    content = task.content
    return (
        f"[{task.created_at}] A task you scheduled has activated and is due now "
        f"(task {content.uuid}, scheduled for {content.activate_at}). "
        f"Carry out its instructions:\n{content.instructions}"
    )


# The PyPI distribution name for each AI_SDK value, so `--version` can report the *installed*
# vendor-SDK version. Only `openai` ships an adapter in Milestone 1; the others are listed so a
# later milestone's drift report names them without code change.
_SDK_DISTRIBUTIONS = {"openai": "openai", "xai-sdk": "xai-sdk", "anthropic": "anthropic"}


def _sdk_version(sdk: str) -> str | None:
    """The installed version of the vendor SDK named by ``AI_SDK``, or ``None`` if absent."""
    dist = _SDK_DISTRIBUTIONS.get(sdk, sdk)
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _version_string() -> str:
    """``basecradle-harness-wake <harness> · <sdk> SDK <ver>`` — both versions an upgrade tracks.

    Reports the harness version *and* the configured vendor SDK's installed version, so the
    fleet drift alarm catches a stale SDK as well as a stale harness (released ≠ deployed
    applies to the SDK pin too). With the SDK not installed it says so plainly — the same
    "no LLM, by design" signal the provider build gives.
    """
    sdk = (os.environ.get("AI_SDK") or "openai").strip().lower()
    version = _sdk_version(sdk)
    sdk_note = f"{sdk} SDK {version}" if version else f"{sdk} SDK not installed"
    return f"basecradle-harness-wake {__version__} · {sdk_note}"


def resolved_config() -> dict[str, object]:
    """The agent's live, *resolved* configuration + active tool set, as machine-readable ground truth.

    The introspection the fleet deployer (the NOC) reads to verify a deploy converged — by
    **ground truth, never self-report** (the basecradle#307 failure class, where a capability is a
    corpse while every version/health signal still reads green). It resolves through the **same
    code paths the running agent uses**: the validated ``(provider, sdk, surface)`` triple
    (`_config_from_env`, which hard-fails an unknown provider or an SDK-mismatched surface) and the
    active tool set after the full plugin/memory/MCP/locked-policy resolution (`_resolve_tools`) —
    so the output is what the agent would actually do, not a declared list.

    **Side-effect-free**, so it is safe to run repeatedly over SSH against a live agent home:

    - It does **not** build the model provider (no ``AI_API_KEY`` required, no client constructed),
      so ``ai_model`` is reported as the raw ``AI_MODEL`` env value (``None`` if unset) rather than
      raising the "AI_MODEL is required" the provider build would.
    - It does **not** run the config-home upgrade reconcile (which *writes* refreshed defaults), so
      it reports the overlay **as it is on disk**. The deploy order is install (which reconciles)
      then verify, so a verifier reads the post-install state.
    - Loading MCP drop-ins briefly starts each *configured* server to ``tools/list`` it (read-only
      at the protocol level — no ``tools/call`` — the same connection a wake makes, with failed
      servers self-excluding); with the default empty ``mcp/`` dir this is a no-op.

    The returned field set is an **additive contract** a downstream tool can depend on:

    - ``harness_version`` — the installed ``basecradle-harness`` version.
    - ``ai_provider`` / ``ai_sdk`` / ``ai_sdk_surface`` — the validated config triple (``surface``
      is ``""`` for a single-surface SDK that declares none).
    - ``ai_sdk_version`` — the installed version of the vendor SDK named by ``AI_SDK``, or ``None``
      if that SDK is not installed.
    - ``ai_model`` — the ``AI_MODEL`` env value, or ``None`` if unset.
    - ``tools`` — the resolved active **function** tool names, sorted.
    - ``builtins`` — the resolved active server-side **built-in** wire names, sorted (e.g. the
      Responses ``web_search``); a live capability the tool-set axis must count.
    - ``skipped`` — the names of plugins that did **not** activate, sorted — the auditable "why
      isn't this tool here?" trail (a diagnostic, not part of the active set).
    - ``opt_in_tools`` — the active **opt-in** (powerful) tools' source-file **stems**, sorted
      (issue #181). The stem is the unit the fleet inventory keys a powerful tool on, and is
      **not** 1:1 with the resolved ``tools``/``builtins`` names (one stem can fan out — e.g.
      ``code_execution`` → the ``code_interpreter`` built-in **+** the ``code_attach`` tool —
      and a name can differ from its stem — ``hear_audio`` → ``listen``). Reporting the stems
      lets the NOC's fleet-drift audit compare declared-vs-active inventory like-for-like,
      holding no stem→name map of its own. ``[]`` for a safe default config (no opt-in tool).
    """
    provider_name, sdk, surface = _config_from_env()
    resolved, _memory = _resolve_tools(provider_name, sdk, surface)
    return {
        "harness_version": __version__,
        "ai_provider": provider_name,
        "ai_sdk": sdk,
        "ai_sdk_surface": surface,
        "ai_sdk_version": _sdk_version(sdk),
        "ai_model": os.environ.get("AI_MODEL") or None,
        "tools": sorted(tool.name for tool in resolved.tools),
        "builtins": sorted(resolved.builtins),
        "skipped": sorted(name for name, _reason in resolved.skipped),
        "opt_in_tools": list(resolved.opt_in_stems),
    }


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
    # catch a published-but-not-deployed release before it goes silent. The harness reaches
    # an LLM only through a vendor SDK, so an upgrade tracks **harness + vendor-SDK version
    # together** (issue #158): this reports both, e.g.
    # "basecradle-harness-wake 0.33.0 · openai SDK 2.43.0", so the drift alarm catches a stale
    # SDK as well as a stale harness. Exits 0 (argparse's built-in action).
    parser.add_argument(
        "--version",
        action="version",
        version=_version_string(),
        help="print the installed basecradle-harness and vendor-SDK versions, then exit.",
    )
    # The deploy verifier's ground-truth probe (issue #174): print the live, *resolved* config +
    # active tool set as machine-readable JSON, then exit — no timeline, no model call, no writes.
    # The NOC's `fleet-drift` check reads this to verify a converged deploy by ground truth rather
    # than self-report (the basecradle#307 failure class). JSON (pretty-printed, stable key order)
    # is both what a verifier parses and human-readable enough on its own, so it is the one format.
    parser.add_argument(
        "--resolved-config",
        action="store_true",
        help=(
            "print the resolved config + active tool set as JSON (ground truth for fleet drift), "
            "then exit. Read-only and timeline-free."
        ),
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

    if args.resolved_config:
        # A read-only introspection — resolve the config + tool set (no model call, no writes) and
        # emit it as stable, pretty-printed JSON. A resolution error (an unknown AI_PROVIDER, an
        # SDK-mismatched AI_SDK_SURFACE) is the verifier's honest signal that the agent is
        # misconfigured, so it surfaces as a clean non-zero exit, never a raw traceback.
        try:
            print(json.dumps(resolved_config(), indent=2, sort_keys=True))
        except (HarnessError, ProviderError, BaseCradleError, ValueError, KeyError) as error:
            print(f"basecradle-harness-wake: {error}", file=sys.stderr)
            return 1
        return 0

    if not args.timeline:
        parser.error("a timeline uuid is required (--timeline or BASECRADLE_TIMELINE)")

    try:
        agent = WakeAgent.from_env(timeline=args.timeline)
        agent.wake(trigger=args.message, event_trigger=args.event, asset_trigger=args.asset)
    except (HarnessError, ProviderError, BaseCradleError, ValueError, KeyError) as error:
        # `_act_on` degrades the per-item failures (a locked-timeline post, the engine's
        # step cap) in flight, so a wake reaching a locked timeline still exits 0. A
        # BaseCradleError caught here is a harder failure — setup, an unreadable timeline —
        # which the router should see as a clean non-zero exit, never a raw traceback.
        print(f"basecradle-harness-wake: {error}", file=sys.stderr)
        return 1
    return 0
