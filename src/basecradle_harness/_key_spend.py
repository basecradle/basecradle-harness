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
- **It never holds or fails a wake.** It is the last of a wake's work (`Wake._wake`), so a wake
  that failed or was stopped before it writes none, and a SIGTERM during it is an ordinary stop. It
  is bounded by `TIMEOUT`, held as a deadline on the whole attempt (each read's client carries
  `_deadline.bound_attempts`, which checks it as each part of the answer arrives), and turns
  anything it did not foresee into ``reason=exception``. A stop (`WakeStopped`, a
  `BaseException`) is the one thing it lets through, because a stop is never swallowed.
- **It reads only where the key is the brain's and the brain is the vendor's.** ``AI_PROVIDER``
  must name the vendor (as the balance tools require before they read ``AI_API_KEY``) **and** the
  wired brain must report it, so a library caller's test double, or a brain built by hand beside a
  stale environment, sends nothing anywhere.

The line's bytes have **one author each**, `key_spend_line` and `key_spend_error_line`, because the
log-grammar probe renders the head too (issue #871): only five of the fleet's harness agents have a
brain that writes it, so the NOC's extraction guard cannot watch it off live traffic and watches
the probe's synthetic instead (`_log_grammar`, basecradle-noc#1369). A second spelling would let
the probe keep proving a head production no longer writes.
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
from basecradle_harness._report import probe_prefix

_log = logging.getLogger("basecradle_harness")

#: The line's head: a fact, so plain (`_observability.head`'s rule).
HEAD = "key spend"

#: The longest rendering of a figure the line will carry. A ``float`` the vendor stated can spell
#: out hundreds of digits (``1e300``, ``5e-324``), which `kv`'s bound would drop or cut, leaving an
#: ``outcome=ok`` line with no figure or a broken one; no real key's spend is that, so such a
#: figure is ``invalid_response``.
SPENT_CHARS = 32

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
    spent = _fixed(spend.spent)
    if len(spent) > SPENT_CHARS:
        _log_failed(vendor, "invalid_response", timeline=timeline, delivery=delivery)
        return
    _log.info(
        "%s",
        key_spend_line(
            provider=vendor,
            spent=spent,
            since=_stamp(spend.since),
            as_of=_stamp(spend.as_of),
            timeline=timeline,
            delivery=delivery,
        ),
    )


def _log_failed(vendor: str, slug: str, *, timeline: str | None, delivery: str | None) -> None:
    """The WARNING a read that produced no figure writes, naming why by its slug and nothing else."""
    _log.warning(
        "%s",
        key_spend_error_line(provider=vendor, reason=slug, timeline=timeline, delivery=delivery),
    )


def key_spend_line(
    *,
    provider: str,
    spent: str,
    since: str | None = None,
    as_of: str | None = None,
    timeline: str | None = None,
    delivery: str | None = None,
    source: str | None = None,
    agent: str | None = None,
) -> str:
    """The ``outcome=ok`` line: the figure the vendor stated, already rendered (`_fixed`, `_stamp`).

    ``key spend provider=… outcome=ok spent=… since=… as_of=… timeline=… delivery=…``

    The only author of these bytes, for the wake and for the log-grammar probe alike (see the
    module docstring). ``source`` and ``agent`` are the probe-only fields and trail the grammar; a
    real line passes neither, so it carries no stamp and no leading ``PROBE`` token.
    """
    line = f"{HEAD} " + kv(
        provider=provider,
        outcome="ok",
        spent=spent,
        since=since,
        as_of=as_of,
        timeline=timeline,
        delivery=delivery,
        source=source,
        agent=agent,
    )
    return probe_prefix(source) + line


def key_spend_error_line(
    *,
    provider: str,
    reason: str,
    timeline: str | None = None,
    delivery: str | None = None,
    source: str | None = None,
    agent: str | None = None,
) -> str:
    """The ``outcome=error`` line: no figure this wake, and the slug that says why.

    ``key spend provider=… outcome=error reason=… timeline=… delivery=…``

    Same single-author contract and probe-only fields as `key_spend_line`.
    """
    line = f"{HEAD} " + kv(
        provider=provider,
        outcome="error",
        reason=reason,
        timeline=timeline,
        delivery=delivery,
        source=source,
        agent=agent,
    )
    return probe_prefix(source) + line


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
