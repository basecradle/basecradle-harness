"""The cross-wake circuit-breaker — the backstop for an *unknown* runaway loop.

The runaway this defends against is a **cross-wake loop**: the agent is woken, it acts, the act
fires a platform event, the router wakes it again → a tight cycle burning provider tokens and box
resources. The in-wake `max_steps` cap, the actor self-filter, and read-speed pacing each stop a
*specific* loop; this is the generic backstop for a *novel* one — most plausibly introduced by a
custom `tools/` plugin or a drop-in MCP server.

It is a rolling-window rate limiter on **engaged wakes per timeline** — over ``max_wakes`` in
``window`` seconds, it trips — persisted under the agent's home beside `marks/`/`seen/`/`claims/`
so it survives the process-per-wake model:

- ``breaker/<timeline>.wakes`` — the timestamps of recent engaged wakes, pruned to the window on
  every admission (so the file stays bounded however fast a runaway fires).
- ``breaker/<timeline>.tripped`` — the durable **trip marker**: present iff the timeline is
  tripped, holding the trip timestamp.

Three decisions shape it, and each one reverses a design that shipped and failed live (issue #592,
@briggs, 2026-09-29 07:14Z: a founder's direct question dropped, and nothing answered it until a
third party posted twelve minutes later).

**It counts the thing it exists to stop: a wake that is about to call a model.** It used to record
every process start, first thing, before the wake had read anything — on the premise that only
peer items wake an agent. In production that premise is false: the router delivers the agent's own
echoes and replays the backlog a long wake queued, and every delivery is a process start. So ten
replays that made **zero** provider calls in 25 seconds tripped it, and the wake that tripped it
had the founder's message in hand. A wake is now counted when it reaches its **first model work**
(`WakeAgent._admit`) and never before — so a wake that found nothing to do costs the breaker
nothing, however many of them the router sends.

**A tripped wake holds; it never drops.** It used to *self-decline* — return with its items
unanswered — and reset only lazily, inside whichever wake came next. When the burst's last event is
the one that matters (it usually is: the burst is the backlog, the last event is the live one),
nothing comes next, and the timeline goes silent until somebody else speaks. So the wake that trips
it now **waits out the cooldown in-process, then resets and does its work** — the declined events
answered from the marks, by the same wake, with nobody having to notice. Three alternatives were
weighed and are worth recording, because each looks right until its cost is named:

- *A platform task scheduled for the cooldown's end* is a harness-authored item on the timeline in
  the agent's name, which the Unspoken Channel forbids: the harness speaks for the agent only where
  the model cannot be reached, and here it can.
- *A wake launched from outside the router* (a timer, a detached process) breaks the one-writer
  guarantee the transcript depends on — the router's per-agent serialization is what makes two
  wakes never clobber one session file, and nothing in this repo replaces it.
- *A deferred re-wake in the router* would be the right owner of scheduling, but it is a contract
  another repository does not offer.

Holding in-process keeps both invariants by construction — it touches no timeline, and it *is* the
serialized wake — and it has a precedent: `ReadPacer` already holds a wake to simulate reading. The
price is stated rather than hidden. The router runs one wake per agent at a time, so while a wake
holds, **every** timeline of that agent waits behind it. A NOC probe is never skipped, as it was
when a tripped wake declined — but one can wait up to a cooldown for its ack: behind a hold on
another timeline, behind a hold taken on another item before it is read, or behind a resume held
ahead of the batch it sits in. At the default 60 s cooldown that is a pause; an operator who sets a
cooldown of hours sets a pause of hours. And the
hold is part of the wake's wall-clock, so a long turn that also held can cross the fleet's
wake-duration outlier line and page a second time for the same trip.

**A trip is witnessed.** ``Wake breaker TRIPPED`` feeds the fleet's ``breaker_tripped`` column and
its *Circuit Breaker Tripped* alert, and like every needle line it exists only on the failure path,
so nothing on a healthy fleet exercises it. `breaker_tripped_line` is therefore the **only** author
of those bytes, and `_log_grammar` emits them synthetically through it on a cadence — the
``billing_blocked`` arrangement (basecradle-noc#509), one column over.

What it deliberately gives up: a genuine runaway is **throttled** — at most ``max_wakes`` engaged
wakes per window, then a cooldown — rather than stopped. Any design that answers the held work
without a third party re-arms a loop whose next link is that work; the stop was only ever as
durable as the silence after it. Each trip pages, which is what keeps a throttled runaway loud.

Disabled by setting the cap to 0 (or below) — an operator escape hatch; the default is a generous
always-on sanity cap.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from basecradle_harness._observability import GREEN, YELLOW, head, kv
from basecradle_harness._report import probe_prefix

# The generous safe defaults. A genuine cross-wake runaway fires continuously — many engaged wakes
# a minute — while a human-paced multi-peer conversation almost never needs 10 model-engaging wakes
# on one timeline in a minute: unseen messages batch into one turn, so the deliveries behind the
# first one mostly find nothing left to do, and those are no longer counted. Tunable via env for
# the rare high-volume timeline; the router's cross-agent breaker (basecradle-router) is the
# complementary layer.
_DEFAULT_BREAKER_MAX = 10
_DEFAULT_BREAKER_WINDOW = 60.0

#: The three tunables, named once for the readers and for the errors that must name them.
MAX_VAR = "HARNESS_WAKE_BREAKER_MAX"
WINDOW_VAR = "HARNESS_WAKE_BREAKER_WINDOW"
COOLDOWN_VAR = "HARNESS_WAKE_BREAKER_COOLDOWN"

#: The heads of the two breaker lines. ``Wake breaker TRIPPED`` is a **consumed literal** — the
#: fleet's ``breaker_tripped`` column matches exactly these bytes — so it is spelled once, here, and
#: never re-typed: a rename is the #501 failure (an alarm silently dark), and the probe
#: (`_log_grammar`) renders through the same constant.
TRIPPED_HEAD = "Wake breaker TRIPPED"
RESET_HEAD = "Wake breaker RESET"


@dataclass(frozen=True)
class BreakerDecision:
    """The breaker's verdict for one wake's admission to model work.

    ``hold`` is the load-bearing field: the seconds this wake must wait before its first model call
    (``0.0`` → proceed now). ``tripped`` flags the one-time state *transition* — this admission
    tripped the breaker — so the caller alerts exactly once per trip. ``reset`` says a trip is to be
    cleared (`WakeBreaker.release`) once the hold is served: set on the tripping admission, and on
    one that found a trip already standing (a wake killed while holding, or a concurrent one).
    ``count`` is the number of engaged wakes in the rolling window, for the log line.
    """

    tripped: bool
    hold: float
    reset: bool
    count: int


class WakeBreaker:
    """Per-timeline cross-wake circuit-breaker. See the module docstring for the design.

    `admit` is called once per wake, at its first model work, and returns a `BreakerDecision`:

    - Under the cap → record the wake and proceed (``hold == 0``).
    - Over the cap → **TRIP**: write the marker and return ``tripped`` with ``hold`` = the
      cooldown. The caller logs the trip, waits the hold, calls `release`, and proceeds.
    - A trip already standing → do not record; return ``hold`` = what is left of its cooldown
      (``0`` when it has already elapsed) with ``reset``, and the caller finishes it the same way.

    `release` clears the marker and restarts the window from now — counting the wake that is about
    to engage — so a trip always ends in a clean window rather than re-tripping on its own history.
    Nothing is recorded while a timeline is tripped, because nothing engages while it is: the only
    wakes that reach `admit` then are the ones about to hold.

    Two seams keep it deterministically testable, mirroring `ReadPacer`: an injectable clock
    (``now``, default `time.time`) and an injectable ``sleep`` behind `wait` (default `time.sleep`).

    The files are not locked. The router serializes an agent's wakes, so two never meet here; a
    hand-launched wake running beside a routed one can overwrite the other's window write, which
    loses a count and never adds one.

    When the breaker is enabled, ``window`` must be a positive, finite number of seconds and
    ``cooldown`` a finite one that is not negative, or the constructor raises `ValueError`. The
    failures they prevent are silent ones: a window of zero or ``nan`` never holds a timestamp, so a
    breaker that reports itself enabled never trips; and an infinite cooldown is a hold that never
    ends, wedging every timeline of the agent. A disabled breaker validates nothing, because the
    escape hatch must work whatever else is configured.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_wakes: int = _DEFAULT_BREAKER_MAX,
        window: float = _DEFAULT_BREAKER_WINDOW,
        cooldown: float | None = None,
        now=None,
        sleep=None,
    ) -> None:
        self.root = Path(root)
        self.max_wakes = max_wakes
        self.window = float(window)
        # The cooldown defaults to the window: hysteresis, so a trip holds for at least one full
        # window and the loop it interrupted has a whole window's worth of silence to die in.
        self.cooldown = float(cooldown) if cooldown is not None else float(window)
        if self.enabled:
            _require_valid(
                self.window, self.cooldown, window_name="window", cooldown_name="cooldown"
            )
        self._now = now or time.time
        self._sleep = sleep or time.sleep

    @classmethod
    def from_env(cls, root: str | Path, *, now=None, sleep=None) -> WakeBreaker:
        """Build a breaker from `HARNESS_WAKE_BREAKER_MAX`/`_WINDOW`/`_COOLDOWN` (generous defaults)."""
        max_wakes, window, cooldown = breaker_settings_from_env()
        return cls(
            root, max_wakes=max_wakes, window=window, cooldown=cooldown, now=now, sleep=sleep
        )

    @property
    def enabled(self) -> bool:
        """Off when the cap is 0 or below — the operator escape hatch."""
        return self.max_wakes > 0

    def tripped(self, timeline: str) -> bool:
        """Whether this timeline currently holds a durable trip marker."""
        return self._read_trip(timeline) is not None

    def admit(self, timeline: str) -> BreakerDecision:
        """Decide whether this wake may start its model work on `timeline` now, or must hold first.

        Called once per wake, at its first model work — never at process start, which is what made
        empty replays count (see the module docstring). See the class docstring for the verdicts.
        """
        if not self.enabled:
            return BreakerDecision(tripped=False, hold=0.0, reset=False, count=0)
        now = self._now()
        # Inside the window and not ahead of the clock: a stamp from before a backwards clock step
        # is not recent, and kept it would sit in the window for as long as the step was long.
        recent = [t for t in self._read_window(timeline) if now - self.window < t <= now]
        trip_at = self._read_trip(timeline)
        if trip_at is not None:
            # A trip is already standing — its holder was killed mid-hold, or another wake is
            # holding it right now. Serve what is left of the cooldown and finish the reset; this
            # wake is counted in the fresh window `release` starts, not here. Never more than the
            # cooldown itself: the stamp is wall-clock, and a clock stepped backwards would
            # otherwise turn a one-minute trip into a hold as long as the step.
            hold = min(self.cooldown, max(0.0, trip_at + self.cooldown - now))
            return BreakerDecision(tripped=False, hold=hold, reset=True, count=len(recent))
        recent.append(now)
        self._write_window(timeline, recent)
        if len(recent) > self.max_wakes:
            self._write_trip(timeline, now)
            return BreakerDecision(tripped=True, hold=self.cooldown, reset=True, count=len(recent))
        return BreakerDecision(tripped=False, hold=0.0, reset=False, count=len(recent))

    def release(self, timeline: str) -> None:
        """End a trip: clear the marker and restart the window from now, counting this wake."""
        self._clear_trip(timeline)
        self._write_window(timeline, [self._now()])

    def wait(self, seconds: float) -> None:
        """Sleep `seconds` through the injectable seam — the hold is served in `WakeAgent._hold`."""
        if seconds > 0:
            self._sleep(seconds)

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


# --- the two breaker log lines: one author, two callers ----------------------------------------


def _configured(seconds: float | None) -> str | None:
    """A configured duration, spelled as the router's breaker line spells it (``60s``)."""
    return None if seconds is None else f"{seconds:g}s"


def _measured(seconds: float | None) -> str | None:
    """A measured duration, spelled as the wake bookend spells one (``57.25s``)."""
    return None if seconds is None else f"{seconds:.2f}s"


def breaker_tripped_line(
    *,
    timeline: str | None = None,
    count: int | None = None,
    threshold: int | None = None,
    window: float | None = None,
    cooldown: float | None = None,
    hold: float | None = None,
    source: str | None = None,
    agent: str | None = None,
) -> str:
    """The **trip** line — logged once per trip, at WARNING, before the hold begins.

    ``Wake breaker TRIPPED timeline=… count=… threshold=… window=…s cooldown=…s hold=…s``

    This function is the *only* author of these bytes, for the reason `_report.billing_onset_line`
    is the only author of its own: the fleet's ``breaker_tripped`` column matches
    ``Wake breaker TRIPPED`` and powers the *Circuit Breaker Tripped* alert, the line exists only
    on the failure path, and `_log_grammar` proves it by rendering it here — so a refactor that
    changes the real line changes the probe's in the same edit.

    The fields after the head mirror the router's own trip line (``count= threshold= window=
    cooldown=``), so one reading serves both layers' breakers; ``hold=`` is what this wake is about
    to wait. None of them is a field any fleet column extracts, and ``duration=`` is deliberately
    not used — it is the extractor key for wake and model-call durations.

    ``source`` and ``agent`` are the **probe-only** fields, under the capital's ruling as amended on
    2026-09-29 (issue #593): they trail the grammar, and a probe line also *leads* with ``PROBE``,
    rendered by `_report.probe_prefix` from the same value as the stamp. Production passes neither,
    so a real trip carries no ``source=`` and no token — which is what keeps it inside the alert's
    ``!= 'probe'`` block-list predicate, and what keeps a person from waving it away.
    """
    line = f"{head(TRIPPED_HEAD, YELLOW)} " + kv(
        timeline=timeline,
        count=count,
        threshold=threshold,
        window=_configured(window),
        cooldown=_configured(cooldown),
        hold=_measured(hold),
        source=source,
        agent=agent,
    )
    return probe_prefix(source) + line


def breaker_reset_line(*, timeline: str | None = None, held: float | None = None) -> str:
    """The **reset** line — logged when a trip ends, after the hold, as the wake resumes its work.

    ``Wake breaker RESET timeline=… held=…s``

    It pairs with every trip line (the wake that trips is the wake that resets, unless it is killed
    mid-hold — then the next wake to reach model work resets it, with what was left to hold), so the
    journal shows a trip's whole span. No fleet column reads it today; it has one author for the
    day one does.
    """
    return f"{head(RESET_HEAD, GREEN)} " + kv(timeline=timeline, held=_measured(held))


def breaker_settings_from_env() -> tuple[int, float, float]:
    """The breaker's tunables from the environment, validated exactly as a wake validates them.

    Returns ``(max_wakes, window, cooldown)``, the cooldown already defaulted to the window. Shared
    by `WakeBreaker.from_env` and ``--resolved-config``, so a value that would fail every wake fails
    the deploy verifier too — the report is the resolution the wake runs, never a second reading of
    the same variables. The error names the variable an operator has to change.
    """
    max_wakes = _breaker_max_from_env()
    window = _breaker_window_from_env()
    configured = _breaker_cooldown_from_env()
    cooldown = window if configured is None else configured
    if max_wakes > 0:
        # A defaulted cooldown *is* the window, so a bad one is always reported as the window.
        _require_valid(window, cooldown, window_name=WINDOW_VAR, cooldown_name=COOLDOWN_VAR)
    return max_wakes, window, cooldown


def _require_valid(window: float, cooldown: float, *, window_name: str, cooldown_name: str) -> None:
    """Refuse the two values that make an enabled breaker fail silently (see `WakeBreaker`)."""
    if not (math.isfinite(window) and window > 0):
        raise ValueError(f"{window_name} must be a positive number of seconds, not {window!r}")
    if not (math.isfinite(cooldown) and cooldown >= 0):
        raise ValueError(
            f"{cooldown_name} must be a number of seconds, zero or more, not {cooldown!r}"
        )


def _breaker_max_from_env() -> int:
    """`HARNESS_WAKE_BREAKER_MAX` → the wake cap; unset/blank → the generous default."""
    raw = os.environ.get(MAX_VAR)
    if raw is None or not raw.strip():
        return _DEFAULT_BREAKER_MAX
    return int(raw)


def _breaker_window_from_env() -> float:
    """`HARNESS_WAKE_BREAKER_WINDOW` → the rolling-window seconds; unset/blank → the default."""
    raw = os.environ.get(WINDOW_VAR)
    if raw is None or not raw.strip():
        return _DEFAULT_BREAKER_WINDOW
    return float(raw)


def _breaker_cooldown_from_env() -> float | None:
    """`HARNESS_WAKE_BREAKER_COOLDOWN` → the hold seconds after a trip; unset/blank → the window."""
    raw = os.environ.get(COOLDOWN_VAR)
    if raw is None or not raw.strip():
        return None  # default: tie the cooldown to the window length
    return float(raw)
