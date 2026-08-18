"""Report a provider failure to the timeline — mechanically, with the model unavailable (issue #336).

The three-way provider-failure taxonomy (`_exceptions`) is *classified* in the adapters and
*handled* in the wake; this module is the small, model-free machinery in between. Two founder
decisions (2026-07-21) shape every line of it:

- **The reporter is mechanical — no LLM anywhere in the failure path.** The agent's own model is the
  thing that just failed, so it cannot be asked to compose the notice. The harness posts the notice
  itself, through the BaseCradle SDK, under the agent's identity (the platform API costs no vendor
  credit). This module only *builds the text and holds the debounce state*; the actual post is the
  wake's existing mechanical poster (`_wake.WakeAgent._post` → ``timeline.messages.create``), the
  same path the NOC probe ack uses.

- **The vendor error is relayed verbatim, never softened** (decision 3). `verbatim` digs the vendor's
  own words out of the error and the report carries them unchanged, so the human — and any peer AI on
  the timeline — sees the real cause and can act on it (shrink the file; add funds).

**This is a sanctioned exception to the Unspoken Channel** (issue #293: "the harness never speaks for
the agent"). That invariant's own boundary is *the model decides when the agent speaks* — and here
the model **cannot be reached at all**, which is exactly the case it could not cover. The founder
decided the harness must speak in this one narrow, model-unavailable case (the CLAUDE.md Unspoken
Channel section records it). It is not the breaker-alert mistake the invariant warns against: there
the model was available and *chose* not to mention the outage; here there is no model to ask.

Two report shapes, one per handled class:

- **Permanent** (`ProviderPayloadTooLargeError` — a payload too large to ever accept — and a context
  overflow that could not self-heal): the same *content* fails forever, and only the human changing it
  can resolve it, so the wake reports once, marks the item handled, and exits clean. The report names
  what could not be processed and the verbatim reason. (A *generic* malformed-request 4xx is **not**
  here: it is a fixable config defect, so it propagates and stays re-drivable — see `_exceptions`.)
- **Billing** (`ProviderBillingError`): the account is out of credit and heals only when a human
  funds it. The report says so in plain language, the notice is **debounced** (one per outage per
  timeline — `BillingState`), and the pending work is left pending so it resumes on the first
  successful call after funding.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from basecradle_harness._exceptions import (
    ProviderBillingError,
    ProviderContextLengthError,
    ProviderError,
    ProviderPayloadTooLargeError,
)
from basecradle_harness._media import _detail_from_body
from basecradle_harness._observability import RED, YELLOW, head, kv

_log = logging.getLogger("basecradle_harness")

#: The two reported classes (the transient class is retried, never reported). They ride the
#: `reported_failure` log line's ``kind=`` field, which the NOC alarms on (basecradle-noc#317).
PERMANENT = "permanent"
BILLING = "billing"

#: The reason slug on the log line — a stable, greppable name for *which* fault, one per handled
#: exception type. Coarser than the verbatim vendor error (which rides the timeline post), but stable
#: across vendors so a dashboard can group by it.
_PAYLOAD_TOO_LARGE = "payload_too_large"
_CONTEXT_LENGTH = "context_length"
_OUT_OF_FUNDS = "out_of_funds"

#: How a raw provider label (``AI_PROVIDER`` / the adapter's ``provider``) reads in a peer-facing
#: notice. An unknown label passes through unchanged rather than being mangled.
_PROVIDER_LABELS = {"xai": "xAI", "openai": "OpenAI", "openrouter": "OpenRouter"}


@dataclass(frozen=True)
class ReportClass:
    """How a provider failure is reported: which taxonomy class, and a stable reason slug."""

    kind: str  # PERMANENT | BILLING
    reason: str  # a `_*` slug above


def classify(exc: BaseException) -> ReportClass | None:
    """Which report class a provider failure falls in, or ``None`` if it is not a reported one.

    The adapters have already classified the fault *by raising the right exception type* — this only
    reads the type. Exactly three types are reported: out-of-funds (billing), a payload too large, and
    a context overflow that could not self-heal. Everything else — a transient fault (retried), an
    auth error, a rate limit, and a **generic** malformed-request `ProviderAPIError` (which propagates,
    a fixable config defect, not a permanent property of the peer's content) — returns ``None`` and
    keeps its existing behavior.
    """
    if isinstance(exc, ProviderBillingError):
        return ReportClass(BILLING, _OUT_OF_FUNDS)
    if isinstance(exc, ProviderPayloadTooLargeError):
        return ReportClass(PERMANENT, _PAYLOAD_TOO_LARGE)
    if isinstance(exc, ProviderContextLengthError):
        # A context overflow only reaches the reporter when its own self-heal (compact + retry in
        # `Session.send`) could not run — the residual, genuinely-stuck case. Reporting it then is
        # decision 4's "a blown context tells the human", not a substitute for the self-heal.
        return ReportClass(PERMANENT, _CONTEXT_LENGTH)
    return None


def verbatim(exc: BaseException) -> str:
    """The vendor's own error text, dug out and relayed **unchanged** (decision 3).

    For an HTTP-shaped error the real cause lives in the response ``body`` under ``error.message``
    (`_media._detail_from_body` handles the common shapes); for the native xAI gRPC path the
    exception's own ``str`` already *is* the verbatim ``xAI gRPC error (...): <detail>`` line. Never
    softened, never translated — the human decides what to do from the real message.
    """
    body = getattr(exc, "body", "") or ""
    if body:
        detail = _detail_from_body(body)
        if detail:
            return detail
    return str(exc)


def provider_label(provider: str | None) -> str:
    """A peer-facing name for a provider label (``xai`` → ``xAI``); unknown labels pass through."""
    if not provider:
        return "the model provider"
    return _PROVIDER_LABELS.get(provider, provider)


def report_body(rc: ReportClass, *, item: str, provider: str | None, exc: ProviderError) -> str:
    """The peer-facing notice for a reported failure — verbatim vendor error, plain-language framing.

    Written for a non-technical human *and* a peer AI on the timeline (decision 5): the billing notice
    says plainly to add funds; the permanent notice names what could not be processed and, for a
    too-large payload, that the original is untouched and a smaller version may work (decision 1). The
    vendor's own words ride inside, unchanged.
    """
    name = provider_label(provider)
    detail = verbatim(exc)
    if rc.kind == BILLING:
        return (
            f"I can't respond right now — my {name} account is out of credit ({detail}). "
            f"Add funds to the {name} account to resume; pending messages will be handled then."
        )
    base = f"I couldn't process {item}: {name} rejected the request — {detail}."
    if rc.reason == _PAYLOAD_TOO_LARGE:
        return base + " The original file is untouched; a smaller or cropped version may work."
    if rc.reason == _CONTEXT_LENGTH:
        return base + " This conversation has grown too long for my context window here."
    return base


# --- the two billing log lines: one author, two callers (basecradle-noc#509) ------------------


#: The stamp that marks a line as manufactured by the fleet's own instrumentation rather than real
#: traffic — the wake-origin contract's vocabulary (basecradle-noc#473, @origin 2026-08-11),
#: generalized to this line class by the capital on 2026-08-18: ``source=probe`` means *synthetic*,
#: on a wake line and on a log-grammar line alike. It is the one token that keeps a probe-emitted
#: billing line out of the founder-named *LLM Vendor Payment Failed* page, via the block-list
#: predicate the NOC's four wake charts already carry.
PROBE_SOURCE = "probe"


def billing_onset_line(
    *,
    reason: str,
    provider: str | None = None,
    timeline: str | None = None,
    delivery: str | None = None,
    source: str | None = None,
    agent: str | None = None,
) -> str:
    """The **onset** line — the one notice per out-of-funds outage, per timeline.

    ``ERROR wake reported_failure kind=billing reason=… provider=… timeline=… delivery=…``

    This function is the *only* author of these bytes, and that is the whole reason it exists
    (basecradle-noc#509). The NOC's ``billing_blocked`` column matches
    ``wake reported_failure.*kind=billing`` on it, and that column powers **LLM Vendor Payment
    Failed** — a founder-named page-the-human alert (basecradle-noc#317). A needle line like this
    one exists *only on the failure path*, so nothing arrives on a healthy fleet for the NOC's
    extraction guard to watch, and a rename would take the alarm silently dark. `_log_grammar`
    therefore emits the same line synthetically on a cadence — through **this** function, so a
    refactor that changes the real line changes the probe's line in the same edit. Two spellings
    would let the probe keep proving a grammar production no longer writes.

    ``source`` and ``agent`` are the **probe-only** trailing fields (the capital's ruling 3,
    2026-08-18: the grammar-under-proof bytes are never altered or interleaved; probe-only fields
    trail them). Production passes neither, and `kv` drops them — so the real line is byte-for-byte
    what it was before this function existed, and *carrying no* ``source=`` is what keeps it inside
    the alarm's block-list predicate. See `_log_grammar` for why the probe passes no ``provider``.
    """
    return f"{head('wake reported_failure', RED)} " + kv(
        kind=BILLING,
        reason=reason,
        provider=provider,
        timeline=timeline,
        delivery=delivery,
        source=source,
        agent=agent,
    )


def billing_repeat_line(
    *,
    reason: str,
    provider: str | None = None,
    timeline: str | None = None,
    delivery: str | None = None,
    source: str | None = None,
    agent: str | None = None,
) -> str:
    """The **debounced repeat** — the breadcrumb every subsequent blocked wake leaves in the journal.

    ``WARNING wake billing_blocked reason=… provider=… timeline=… delivery=…``

    The onset is posted once per outage, so without this line an outage would look like a single
    instant rather than a duration, and the NOC's alert would auto-resolve after the first line
    while the agent is *still* unfunded and still being woken. That is why ``billing_blocked``
    matches **two** clauses, and why the probe emits both: the extraction guard asks only whether
    the column extracted *anything*, so one working clause would green a column whose other clause
    has gone deaf (the two-clause finding on basecradle-noc#509, adopted by the capital).

    Same single-author contract as `billing_onset_line`; same probe-only trailing fields.
    """
    return f"{head('wake billing_blocked', YELLOW)} " + kv(
        reason=reason,
        provider=provider,
        timeline=timeline,
        delivery=delivery,
        source=source,
        agent=agent,
    )


class BillingState:
    """Per-timeline out-of-funds debounce + self-heal state (issue #336).

    Billing is an *account-level* outage, but a notice is posted *per timeline* — each conversation
    (and each peer AI on it) deserves to be told once. So the debounce is per-timeline: a marker file
    beside the wake's other stores (`marks/`, `seen/`, `claims/`, `breaker/`) under the agent's home,
    holding the time the outage was first noticed on that timeline.

    - `note_and_check` records the outage and returns whether *this* wake should post the notice —
      ``True`` on the first billing failure of an outage (the marker was absent), ``False`` on every
      one after (already notified → stay quiet). Idempotent per outage.
    - `recovered` is the **self-heal**: called after any successful model call, it clears the marker
      (if present) and returns whether it cleared one, so a wake can log the recovery. The next
      outage re-notifies from a clean slate.

    It mirrors `WakeBreaker`'s storage shape (one small file per timeline, an injectable clock for
    deterministic tests) and, like it, never breaks a wake: a filesystem hiccup degrades to "not
    blocked" rather than raising.
    """

    def __init__(self, root: str | Path, *, now=None) -> None:
        self.root = Path(root)
        self._now = now or time.time

    def note_and_check(self, timeline: str) -> bool:
        """Record a billing outage on `timeline`; return whether this wake should post the notice.

        ``True`` only when the marker was absent (the first failure of an outage) — otherwise the
        outage is already announced and this wake stays quiet ("fail fast and quiet"). A write failure
        degrades to ``True`` (better a possible second notice than a silent outage).
        """
        path = self._path(timeline)
        if path.exists():
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(repr(self._now()))
        except OSError as error:
            _log.warning("Could not persist the billing-blocked marker for %s: %s", timeline, error)
        return True

    def blocked(self, timeline: str) -> bool:
        """Whether `timeline` currently holds a billing-blocked marker."""
        return self._path(timeline).exists()

    def recovered(self, timeline: str) -> bool:
        """Clear any billing-blocked marker on `timeline`; return whether one was cleared (self-heal).

        Called after a successful model call: a call got through, so the account is funded again and
        the next outage should re-notify. Returns ``True`` (and the caller logs the recovery) only
        when a marker was actually present, so a healthy wake pays nothing but one `exists()` check.
        """
        path = self._path(timeline)
        if not path.exists():
            return False
        try:
            path.unlink()
        except OSError as error:
            _log.warning("Could not clear the billing-blocked marker for %s: %s", timeline, error)
            return False
        return True

    def _path(self, timeline: str) -> Path:
        return self.root / "billing" / f"{quote(timeline, safe='')}.blocked"
