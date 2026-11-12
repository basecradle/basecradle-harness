"""What a refuseless call cost, priced from the vendor's own live price list (issue #844).

refuseless.com returns token usage and **no price**: a Chat Completions body's ``usage`` is exactly
``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``, a Responses body's is ``input_tokens`` /
``output_tokens`` / ``total_tokens`` / ``input_tokens_details`` (issue #847), and no header states a
cost on either. What it
does publish is a ``pricing`` object on every model of ``GET /v1/models``, added by the vendor at
BaseCradle's request so that its price could be read rather than transcribed. So this module is
deliberately **not** a rate table, and there is no ``_refuseless_rates.py``: a hand copy of a price
the vendor publishes live would only be a second, rotting statement of it (@origin). The adapter
reads the list once per wake per key (`_openai.ModelCatalog`), again after a backoff when a read
fails (issue #874) or answers with a row that could not price a call (issue #876), and hands each
row here.

The rules every rate table keeps hold here too:

- **A figure the harness cannot state is omitted, never estimated.** No row for the model, a row
  with no ``pricing``, a ``currency`` other than ``USD``, a rate the call needs that is missing or
  not a non-negative number, a usage block missing either count (`usage_of`): each returns no cost, and the adapter
  states the gap once per wake as a ``cost unpriced charge=tokens`` WARNING and on each call it
  covers as ``cost_gap=`` (issue #876). Nothing falls back to a
  constant. The one list read elsewhere that may price a call is the one this process read on the
  same host with the same key, no older than `_openai.LIST_FALLBACK_MAX_AGE`, for a catalog whose
  own read failed (issue #874): the vendor's own figure, read moments before, is not an estimate.
- **The arithmetic is the call's own facts × the vendor's own rates.** The top-level rates are the
  **effective** price in US dollars per million tokens, discounts applied; ``discounted`` and
  ``regular`` (the list price) are informational and never read.

The shape, as the vendor serves it (a dated reading, never a contract this module carries)::

    {"currency": "USD", "input_per_m": …, "cached_input_per_m": …, "output_per_m": …,
     "discounted": true, "regular": {…}}

**Cached input.** The vendor serves both of the ``openai`` SDK's surfaces, and only one states a
cached count (issue #847). A Chat Completions usage block states none, so every prompt token there
is priced at ``input_per_m`` and no split is invented. A Responses usage block states
``input_tokens_details.cached_tokens``, a part of ``input_tokens``, so those tokens are priced at
``cached_input_per_m`` and the rest of the input at ``input_per_m``; a row without the cached rate
leaves a call that cached something unpriced. A rate the call does not need is never required:
a row missing ``cached_input_per_m`` still prices a call that cached nothing, because the figure is
then exact. A cache *write* count has no published rate at all, so a call that reports one is
unpriced rather than charged a rate the vendor never stated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from basecradle_harness._observability import ComputedCost
from basecradle_harness._openai_rates import Usage

#: The only currency the ``cost=`` field is denominated in (`docs/telemetry.md` → Common).
CURRENCY = "USD"

#: What `cost_rates_source` reports for this vendor in ``--resolved-config``: where the figure is
#: read, by path relative to the agent's own API root, and when. No date, because nothing here was
#: read on a date: the price is the one the vendor serves on the wake that uses it.
SOURCE = "GET {base_url}/models → data[].pricing, read live by each wake"


@dataclass(frozen=True)
class Pricing:
    """One model's effective rates, in US dollars per million tokens.

    ``cached_input`` is ``None`` when the row does not state it, which only matters to a call that
    reports cached tokens.
    """

    input: float
    output: float
    cached_input: float | None = None


def pricing_of(row: Mapping[str, Any] | None) -> Pricing | None:
    """A models-list row's ``pricing`` as a `Pricing`, or ``None`` when it states no usable USD price.

    ``None`` for a row with no ``pricing`` object, a ``currency`` other than ``USD`` (an absent one
    included: a figure in an unnamed currency is not a dollar), and an input or output rate that is
    missing or not a finite, non-negative number. A malformed ``cached_input_per_m`` is read as
    absent rather than failing the row, because the call that needs it is refused on its own.
    """
    if not isinstance(row, Mapping):
        return None
    pricing = row.get("pricing")
    if not isinstance(pricing, Mapping) or pricing.get("currency") != CURRENCY:
        return None
    rate_in = _rate(pricing.get("input_per_m"))
    rate_out = _rate(pricing.get("output_per_m"))
    if rate_in is None or rate_out is None:
        return None
    return Pricing(
        input=rate_in, output=rate_out, cached_input=_rate(pricing.get("cached_input_per_m"))
    )


#: The two surfaces' spellings of one usage block, as ``(input, output, details)``: Chat Completions
#: (``prompt_tokens``) and Responses (``input_tokens``, issue #847). The vendor serves both.
_SPELLINGS = (
    ("prompt_tokens", "completion_tokens", "prompt_tokens_details"),
    ("input_tokens", "output_tokens", "input_tokens_details"),
)


def usage_of(usage: Any) -> Usage | None:
    """A ``usage`` block of either surface as a `Usage`, or ``None`` unless it states **both** counts.

    Read in the Chat Completions spelling (``prompt_tokens`` / ``completion_tokens`` /
    ``prompt_tokens_details``) or the Responses one (``input_tokens`` / ``output_tokens`` /
    ``input_tokens_details``, issue #847), whichever states both counts, Chat first; the two are
    never mixed, because a count from one spelling beside a count from the other is not one call's
    usage. The cached count is part of the input count on both (the vendor says so of Responses),
    which is what `call_cost` subtracts.

    Stricter than the OpenAI table's reader (`_openai_rates.usage_of`), which reads a missing count
    as zero: here the formula is ``input × input_per_m + output × output_per_m``, and a block
    missing either term would price the call low rather than not at all, which the dashboard would
    sum as if it were true. Both must be non-negative integers (never a bool). The cached and
    cache-write counts are read from that spelling's ``…_details``: absent is zero, and present but
    unreadable (a ``…_details`` that is not an object, a count that is not a non-negative integer)
    is ``None``, because reading it as zero would price a cached token at the full rate, or a cache
    write as though there were none — an estimate either way.
    """
    if not isinstance(usage, Mapping):
        return None
    for input_key, output_key, details_key in _SPELLINGS:
        prompt, completion = _count(usage.get(input_key)), _count(usage.get(output_key))
        if prompt is None or completion is None:
            continue
        details = usage.get(details_key)
        if details is None:
            details = {}
        if not isinstance(details, Mapping):
            return None
        cached, cache_write = (_optional(details, key) for key in _DETAIL_COUNTS)
        if cached is None or cache_write is None:
            return None
        return Usage(input=prompt, cached=cached, cache_write=cache_write, output=completion)
    return None


#: The optional counts a ``…_details`` object may state, in `Usage` order.
_DETAIL_COUNTS = ("cached_tokens", "cache_write_tokens")


def _optional(details: Mapping[str, Any], key: str) -> int | None:
    """An optional count: zero when absent or ``null``, ``None`` when present and unreadable."""
    value = details.get(key)
    return 0 if value is None else _count(value)


def beyond_any_row(usage: Usage | None) -> bool:
    """Whether the call's own usage leaves it unpriceable by **any** row (issue #876).

    A block `usage_of` could not read, and a cache write, which the vendor publishes no rate for.
    Every other way `call_cost` refuses is the row's (no cached rate for a call that cached), and a
    later read of the list could answer it, so the adapter asks again for those and never for these.
    """
    return usage is None or usage.cache_write > 0


def call_cost(pricing: Pricing, usage: Usage | None) -> ComputedCost | None:
    """This call's dollars at `pricing`, or ``None`` when the arithmetic would need a guess.

    ``None`` for a call with no counts, a call that reports cached tokens on a row with no cached
    rate, and a call that reports a cache write (a charge the vendor publishes no rate for).
    """
    if beyond_any_row(usage):
        return None
    cached = min(usage.cached, usage.input)
    if cached and pricing.cached_input is None:
        return None
    total = (usage.input - cached) * pricing.input + usage.output * pricing.output
    if cached:
        total += cached * pricing.cached_input  # type: ignore[operator]
    return ComputedCost(total / 1_000_000)


def _count(value: Any) -> int | None:
    """`value` as a token count: a non-negative integer, never a bool."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _rate(value: Any) -> float | None:
    """`value` as a rate: a finite, non-negative real number, never a bool or a numeric string."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    rate = float(value)
    return rate if math.isfinite(rate) and rate >= 0 else None
