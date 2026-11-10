"""The ``key spend`` line: what the brain's own key has been billed, read once per wake (issue #868).

The fleet reconciles each agent's logged ``cost=`` against what its vendor billed. An account's
balance cannot separate one agent from another when several share an account, so the per-agent
billed figure is the calling **key's** cumulative spend, and only a vendor that states it can be
reconciled this way. Two do, and both are read with the brain's own ``AI_API_KEY``:

- **refuseless** — ``GET <root>/credits`` → ``key_spent`` and ``key_spent_since``, plus the
  vendor's ``updated_at`` (`_refuseless_account.read_credits`).
- **OpenRouter** — ``GET <root>/key`` → ``data.usage``, to the vendor's own precision rather than
  the balance tool's cents (`_openrouter_account.read_key_usage`). It states no counter start and
  no read time, so its line carries neither.

The shape is the contract's (`docs/telemetry.md` → `key spend`), settled with the NOC's consumer
(basecradle-noc#1365) before it shipped. Four properties are load-bearing:

- **``spent``, never ``cost``.** The fleet's *Spend by Provider* sums every ``cost=`` on any head, so
  a cumulative figure there would be counted again on every wake.
- **A failed read is a WARNING on the same head, ``outcome=error`` with its slug**, never silence:
  the consumer can tell "no figure this wake, because X" from "the read never ran". Only the slug is
  logged — never the key, the URL, the vendor's prose or its body.
- **It never holds or fails a wake.** It runs as the wake ends (`Wake._wake`), is skipped by a wake
  that was stopped (the router is waiting for it to die), is bounded by `TIMEOUT` as a deadline on
  the whole attempt (each read's client carries `_deadline.bound_attempts`), and turns anything it did not foresee into ``reason=exception``.
- **It reads only where the key is the brain's and the brain is the vendor's.** ``AI_PROVIDER``
  must name the vendor (as the balance tools require before they read ``AI_API_KEY``) **and** the
  wired brain must report it, so a library caller's test double, or a brain built by hand beside a
  stale environment, sends nothing anywhere.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import httpx

from basecradle_harness import _openrouter_account, _refuseless_account
from basecradle_harness._basecradle import (
    _PROVIDER_BASE_URLS,
    base_url_override,
    normalize_provider,
)
from basecradle_harness._observability import kv

_log = logging.getLogger("basecradle_harness")

#: The line's head: a fact, so plain (`_observability.head`'s rule).
HEAD = "key spend"

#: Five seconds to connect and five to answer: `_deadline` holds the whole attempt to their sum,
#: ten seconds, however a vendor keeps the connection alive. The read runs while the wake ends.
TIMEOUT = httpx.Timeout(5.0)


@dataclass(frozen=True)
class _Spend:
    """One vendor's statement of what the key has been billed."""

    #: US dollars, cumulative, exactly as the vendor stated it.
    spent: float
    #: When the vendor's counter started; ``None`` when it states no start.
    since: datetime | None = None
    #: When the vendor read the figure; ``None`` when it states no such moment.
    as_of: datetime | None = None


class _Unavailable(Exception):
    """The read did not produce a figure; `slug` is the line's ``reason``."""

    def __init__(self, slug: str) -> None:
        super().__init__(slug)
        self.slug = slug


def log_key_spend(brain: str, *, timeline: str | None, delivery: str | None) -> None:
    """Write this wake's ``key spend`` line, or nothing when the brain's vendor states no key spend.

    `brain` is the vendor the wired brain reports (`describe_provider`). Never raises.
    """
    vendor = _active_provider()
    reader = _READERS.get(vendor)
    if reader is None or brain != vendor:
        return
    try:
        key = (os.environ.get("AI_API_KEY") or "").strip()
        if not key:
            raise _Unavailable("config:missing_api_key")
        spend = reader(key)
    except _Unavailable as exc:
        _log_failed(vendor, exc.slug, timeline=timeline, delivery=delivery)
        return
    except Exception:  # noqa: BLE001 - nothing the read does may raise into the end of a wake
        _log_failed(vendor, "exception", timeline=timeline, delivery=delivery)
        return
    _log.info(
        "%s %s",
        HEAD,
        kv(
            provider=vendor,
            outcome="ok",
            spent=_fixed(spend.spent),
            since=_stamp(spend.since),
            as_of=_stamp(spend.as_of),
            timeline=timeline,
            delivery=delivery,
        ),
    )


def _log_failed(vendor: str, slug: str, *, timeline: str | None, delivery: str | None) -> None:
    """The WARNING a read that produced no figure writes, naming why by its slug and nothing else."""
    _log.warning(
        "%s %s",
        HEAD,
        kv(provider=vendor, outcome="error", reason=slug, timeline=timeline, delivery=delivery),
    )


def _refuseless(key: str) -> _Spend:
    root = (base_url_override() or _PROVIDER_BASE_URLS["refuseless"]).rstrip("/")
    try:
        credits = _refuseless_account.read_credits(key, root, timeout=TIMEOUT)
    except _refuseless_account._BalanceUnavailable as exc:
        raise _Unavailable(exc.slug) from None
    if credits.key_spend is None:
        raise _Unavailable(credits.key_spend_fault or "not_reported")
    return _Spend(credits.key_spend.spent, credits.key_spend.since, credits.updated_at)


def _openrouter(key: str) -> _Spend:
    root = base_url_override() or _openrouter_account.DEFAULT_BASE_URL
    try:
        spent = _openrouter_account.read_key_usage(key, root, timeout=TIMEOUT)
    except _openrouter_account._BalanceUnavailable as exc:
        raise _Unavailable(exc.slug) from None
    if spent is None:
        raise _Unavailable("not_reported")
    return _Spend(spent)


#: The vendors whose key spend can be read with the brain's key, each compared by equality.
_READERS: dict[str, Callable[[str], _Spend]] = {
    "refuseless": _refuseless,
    "openrouter": _openrouter,
}


def _active_provider() -> str:
    """``AI_PROVIDER`` as the wake reads it, or ``""`` when it is not a slug."""
    try:
        return normalize_provider(os.environ.get("AI_PROVIDER"))
    except ValueError:
        return ""


def _stamp(moment: datetime | None) -> str | None:
    """A moment as the contract writes one: ISO 8601 in UTC, to the second, with a ``Z``."""
    if moment is None:
        return None
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def _fixed(usd: float) -> str:
    """Dollars in plain fixed-point, to every digit the vendor stated: ``1.047832516``, ``0.001405``.

    Not `_observability._money`, which keeps eight decimals: a reconciliation diffs this figure
    against the sum of every logged ``cost=``, so a digit cut here is drift the consumer would
    report as the vendor's. ``repr`` is the shortest text that reads back as the same float, which
    for a figure parsed from JSON is the figure as written; `Decimal` spells it without an exponent.
    """
    text = format(Decimal(repr(usd + 0.0)), "f")  # ``+ 0.0`` turns a ``-0.0`` into ``0.0``
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"
