"""What a refuseless call cost, priced from the vendor's own live price list (issue #844).

refuseless.com returns token usage and **no price**: a Chat Completions body's ``usage`` is exactly
``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``, and no header states a cost. What it
does publish is a ``pricing`` object on every model of ``GET /v1/models``, added by the vendor at
BaseCradle's request so that its price could be read rather than transcribed. So this module is
deliberately **not** a rate table, and there is no ``_refuseless_rates.py``: a hand copy of a price
the vendor publishes live would only be a second, rotting statement of it (@origin). The adapter
reads the list once per wake per key (`_openai.ModelCatalog`) and hands each row here.

The rules every rate table keeps hold here too:

- **A figure the harness cannot state is omitted, never estimated.** No row for the model, a row
  with no ``pricing``, a ``currency`` other than ``USD``, a rate the call needs that is missing or
  not a non-negative number, a usage block missing either count (`usage_of`): each returns no cost, and the adapter
  states the gap once per wake as a ``cost unpriced charge=tokens`` WARNING. Nothing falls back to a
  constant or to a price read on an earlier wake.
- **The arithmetic is the call's own facts × the vendor's own rates.** The top-level rates are the
  **effective** price in US dollars per million tokens, discounts applied; ``discounted`` and
  ``regular`` (the list price) are informational and never read.

The shape, as the vendor serves it (a dated reading, never a contract this module carries)::

    {"currency": "USD", "input_per_m": …, "cached_input_per_m": …, "output_per_m": …,
     "discounted": true, "regular": {…}}

**Cached input.** The usage block states no cached-token count today, so every prompt token is
priced at ``input_per_m`` and no split is invented. The day it states one (OpenAI's
``prompt_tokens_details.cached_tokens``), those tokens are priced at ``cached_input_per_m``, and a
row without that rate leaves such a call unpriced. A rate the call does not need is never required:
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
SOURCE = "GET {base_url}/models → data[].pricing, read live once per wake"


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


def usage_of(usage: Any) -> Usage | None:
    """A Chat Completions ``usage`` block as a `Usage`, or ``None`` unless it states **both** counts.

    Stricter than the OpenAI table's reader (`_openai_rates.usage_of`), which reads a missing count
    as zero: here the formula is ``prompt_tokens × input + completion_tokens × output``, and a block
    missing either term would price the call low rather than not at all, which the dashboard would
    sum as if it were true. Both must be non-negative integers (never a bool). The cached and
    cache-write counts are OpenAI's ``prompt_tokens_details`` spellings, read only if present.
    """
    if not isinstance(usage, Mapping):
        return None
    prompt, completion = _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens"))
    if prompt is None or completion is None:
        return None
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, Mapping) else {}
    return Usage(
        input=prompt,
        cached=_count(details.get("cached_tokens")) or 0,
        cache_write=_count(details.get("cache_write_tokens")) or 0,
        output=completion,
    )


def call_cost(pricing: Pricing, usage: Usage | None) -> ComputedCost | None:
    """This call's dollars at `pricing`, or ``None`` when the arithmetic would need a guess.

    ``None`` for a call with no counts, a call that reports cached tokens on a row with no cached
    rate, and a call that reports a cache write (a charge the vendor publishes no rate for).
    """
    if usage is None or usage.cache_write > 0:
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
