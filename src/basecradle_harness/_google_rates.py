"""What a Gemini call on Vertex AI cost, priced from Google's published rates (issue #655).

Every other adapter's ``cost=`` is the vendor's own figure: OpenRouter states ``usage.cost`` and xAI
states ticks. **Vertex states tokens and no dollars**, so the ``llm provider=google`` line would
otherwise carry no cost at all, and an agent brained on Gemini would be invisible on the fleet's
spend dashboard. The capital's ruling on #655 is to price the call from a maintained table rather
than leave the hole, and this module is that table. It is the one place in the harness that
*computes* a dollar figure instead of reading one, so three rules hold it honest:

- **A figure the table cannot state is omitted, never estimated.** An unknown model, a rate the
  page lists as ``N/A``, a tier or an endpoint class the page prices separately and the table does
  not carry, a call served on a non-standard traffic tier: each returns ``None``, and the line has
  no ``cost=`` — exactly what a vendor that reports nothing gets. A wrong number on the line is
  worse than a missing one, because the dashboard sums it.
- **Every number is transcribed, never derived.** Each `Tier` below is one row of Google's pricing
  page as it stood on the date in `SOURCE`, including the rows that look like typos (the one place a
  row was implausible, the cell is left unstated rather than corrected). When the page changes, the
  table is edited from the page, in a reviewed PR, with the new date.
- **Which row applies is decided by the call's own facts.** The model id the call ran, the location
  it ran at (``global`` vs everything else, which Google prices separately for its Gemini 3 families),
  the prompt size (the >200K tier reprices *every* token of the call, input and output alike), the
  modality split the usage block reports (audio input is priced apart on some models), the traffic
  tier the response names, and the billing date (the 3.6/3.7/3.8 Flash introductory rates end on
  2026-12-31, and the standard rates that follow are already in the table, so the switch needs no
  release).

Pricing is the **Standard pay-as-you-go** tier only. Priority, Flex, batch and Provisioned
Throughput are priced elsewhere on the page (or not per call at all), and a call the response says
ran on one of them gets no cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

#: Where every number in `RATES` was read, and when. Edit the table from this page, and move the
#: date with it.
SOURCE = (
    "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing "
    "(Standard tier, read 2026-10-06)"
)

#: Above this many prompt tokens a call is priced on the long-context row — **all** of its tokens,
#: input and output alike ("If a query input context is longer than 200K tokens, all tokens (input
#: and output) are charged at long context rates").
LONG_CONTEXT_THRESHOLD = 200_000

#: The location Google prices as "Global". Every other location is "Non-global", which for the GA
#: Gemini 3 families costs 10% more; the ``us`` and ``eu`` multi-regions are non-global.
GLOBAL_LOCATION = "global"

#: The traffic tier the Standard rows price. A response that names no tier — or names it
#: ``TRAFFIC_TYPE_UNSPECIFIED`` — is read as standard only when the request asked for the standard
#: tier too (see `call_cost`).
_STANDARD_TRAFFIC = frozenset({"ON_DEMAND"})
_UNNAMED_TRAFFIC = frozenset({"", "TRAFFIC_TYPE_UNSPECIFIED"})

#: What a request's ``service_tier`` is called when it asks for the standard tier: unset, or one of
#: the SDK's two names for it (``ServiceTier.STANDARD``, and ``UNSPECIFIED``, "which is standard").
_STANDARD_TIERS = frozenset({"", "STANDARD", "UNSPECIFIED", "SERVICE_TIER_UNSPECIFIED"})

#: Google bills by the Pacific-time day, so a rate's last day ends at midnight in Los Angeles.
_BILLING_TZ = "America/Los_Angeles"


@dataclass(frozen=True)
class Tier:
    """One context tier's rates, in USD per million tokens, as the page states them.

    ``None`` means the page states no rate for that cell (``N/A``, or a value not transcribed); a
    call that needs it is not priced. ``audio_input`` / ``audio_cached`` are set only where the page
    prices audio input apart from the other modalities; left ``None`` there, audio is priced as
    ``input`` / ``cached`` like everything else, because that row says
    "Input (text, image, video, audio)".
    """

    input: float | None
    cached: float | None
    output: float | None
    audio_input: float | None = None
    audio_cached: float | None = None
    #: Whether audio has a separate row. When it does and that row's cell is ``None``, audio tokens
    #: cannot be priced; when it does not, they ride ``input``.
    audio_apart: bool = False


@dataclass(frozen=True)
class Rates:
    """A model's prices for one period: each endpoint class, each context tier."""

    #: Inclusive first and last billing day this row applies to; ``None`` is open-ended.
    start: date | None
    end: date | None
    #: ``(≤200K, >200K)`` per endpoint class. ``None`` for a class the page does not price for
    #: this model (``gemini-3.1-pro-preview`` lists Global only), which leaves a call there unpriced.
    global_: tuple[Tier, Tier] | None
    non_global: tuple[Tier, Tier] | None


def _flat(tier: Tier) -> tuple[Tier, Tier]:
    """A model whose long-context row repeats its short one (the page states both, identically)."""
    return (tier, tier)


# --- the table --------------------------------------------------------------------------------
#
# Transcribed row by row from `SOURCE`. The Gemini 3.6 / 3.7 / 3.8 Flash introductory rows are
# marked on the page "Promotional pricing provided through 50% credits back on net spend": the
# introductory figure is what the operator pays net of that credit, which is the figure a cost
# line exists to show.

_FLASH_3X_INTRO = Rates(
    start=None,
    end=date(2026, 12, 31),
    global_=_flat(Tier(input=0.75, cached=0.075, output=3.75)),
    non_global=_flat(Tier(input=0.825, cached=0.0825, output=4.125)),
)
_FLASH_3X_STANDARD = Rates(
    start=date(2027, 1, 1),
    end=None,
    global_=_flat(Tier(input=1.50, cached=0.15, output=7.50)),
    non_global=_flat(Tier(input=1.65, cached=0.165, output=8.25)),
)

#: Model id → its rows, oldest first. Keyed by the id the call is made with, exactly as Google
#: publishes it on each model's page.
RATES: Mapping[str, tuple[Rates, ...]] = {
    "gemini-3.8-flash": (_FLASH_3X_INTRO, _FLASH_3X_STANDARD),
    "gemini-3.7-flash": (_FLASH_3X_INTRO, _FLASH_3X_STANDARD),
    "gemini-3.6-flash": (_FLASH_3X_INTRO, _FLASH_3X_STANDARD),
    "gemini-3.5-flash": (
        Rates(
            start=None,
            end=None,
            global_=_flat(Tier(input=1.50, cached=0.15, output=9.00)),
            non_global=_flat(Tier(input=1.65, cached=0.165, output=9.90)),
        ),
    ),
    "gemini-3.5-flash-lite": (
        Rates(
            start=None,
            end=None,
            global_=_flat(Tier(input=0.30, cached=0.03, output=2.50)),
            non_global=_flat(Tier(input=0.33, cached=0.033, output=2.75)),
        ),
    ),
    "gemini-3.1-flash-lite": (
        Rates(
            start=None,
            end=None,
            global_=_flat(
                Tier(
                    input=0.25,
                    cached=0.025,
                    output=1.50,
                    audio_input=0.50,
                    audio_cached=0.05,
                    audio_apart=True,
                )
            ),
            non_global=_flat(
                Tier(
                    input=0.275,
                    cached=0.0275,
                    output=1.65,
                    audio_input=0.55,
                    audio_cached=0.055,
                    audio_apart=True,
                )
            ),
        ),
    ),
    "gemini-3.1-pro-preview": (
        Rates(
            start=None,
            end=None,
            global_=(
                Tier(input=2.00, cached=0.20, output=12.00),
                Tier(input=4.00, cached=0.40, output=18.00),
            ),
            # The page prices this preview at Global only.
            non_global=None,
        ),
    ),
    # The 2.5 family predates Google's non-global surcharge (which applies to GA Gemini 3 and
    # later), so one set of rows serves every location.
    "gemini-2.5-pro": (
        Rates(
            start=None,
            end=None,
            global_=(
                Tier(input=1.25, cached=0.125, output=10.00),
                Tier(input=2.50, cached=0.25, output=15.00),
            ),
            non_global=(
                Tier(input=1.25, cached=0.125, output=10.00),
                Tier(input=2.50, cached=0.25, output=15.00),
            ),
        ),
    ),
    "gemini-2.5-flash": (
        Rates(
            start=None,
            end=None,
            # The page lists this model's long-context **audio** input as cheaper than its short one
            # ($0.30 against $1.00), which no other row on the page does. It is left unstated rather
            # than corrected: a long-context call carrying audio is not priced.
            global_=(
                Tier(
                    input=0.30,
                    cached=0.03,
                    output=2.50,
                    audio_input=1.00,
                    audio_cached=0.10,
                    audio_apart=True,
                ),
                Tier(
                    input=0.30,
                    cached=0.03,
                    output=2.50,
                    audio_input=None,
                    audio_cached=0.10,
                    audio_apart=True,
                ),
            ),
            non_global=(
                Tier(
                    input=0.30,
                    cached=0.03,
                    output=2.50,
                    audio_input=1.00,
                    audio_cached=0.10,
                    audio_apart=True,
                ),
                Tier(
                    input=0.30,
                    cached=0.03,
                    output=2.50,
                    audio_input=None,
                    audio_cached=0.10,
                    audio_apart=True,
                ),
            ),
        ),
    ),
    "gemini-2.5-flash-lite": (
        Rates(
            start=None,
            end=None,
            global_=_flat(
                Tier(
                    input=0.10,
                    cached=0.01,
                    output=0.40,
                    audio_input=0.30,
                    audio_cached=0.03,
                    audio_apart=True,
                )
            ),
            non_global=_flat(
                Tier(
                    input=0.10,
                    cached=0.01,
                    output=0.40,
                    audio_input=0.30,
                    audio_cached=0.03,
                    audio_apart=True,
                )
            ),
        ),
    ),
}


@dataclass(frozen=True)
class Usage:
    """The token counts a Vertex response reported, in the shape pricing needs.

    Built by the adapter from the SDK's usage block (`from_usage_metadata`). ``prompt`` includes
    ``cached`` (Google's ``prompt_token_count`` counts the cache hit too), and ``output`` includes the
    thinking tokens, which the page bills at the output rate ("Text output (response and
    reasoning)"). ``tool_prompt`` is ``tool_use_prompt_token_count``: what a built-in tool's results
    fed back in, billed as input.
    """

    prompt: int
    cached: int
    output: int
    tool_prompt: int = 0
    audio_prompt: int = 0
    audio_cached: int = 0
    traffic: str | None = None


def from_usage_metadata(meta: Any) -> Usage | None:
    """The SDK's ``usage_metadata`` as a `Usage`, or ``None`` when it stated no token counts.

    Read by attribute, defensively: this is the logging path, where a surprise in a vendor object
    must cost the cost field and never the turn.
    """
    if meta is None:
        return None
    prompt = _count(getattr(meta, "prompt_token_count", None))
    candidates = _count(getattr(meta, "candidates_token_count", None))
    thoughts = _count(getattr(meta, "thoughts_token_count", None))
    if prompt is None and candidates is None:
        return None
    traffic = getattr(meta, "traffic_type", None)
    return Usage(
        prompt=prompt or 0,
        cached=_count(getattr(meta, "cached_content_token_count", None)) or 0,
        output=(candidates or 0) + (thoughts or 0),
        tool_prompt=_count(getattr(meta, "tool_use_prompt_token_count", None)) or 0,
        audio_prompt=_modality(getattr(meta, "prompt_tokens_details", None), "AUDIO"),
        audio_cached=_modality(getattr(meta, "cache_tokens_details", None), "AUDIO"),
        traffic=None if traffic is None else _name(traffic),
    )


def call_cost(
    model: str,
    location: str,
    usage: Usage | None,
    *,
    service_tier: Any = None,
    on: date | None = None,
) -> float | None:
    """What this call cost in USD at Google's published Standard rates, or ``None``.

    `location` is the Vertex location the call ran at; `service_tier` is whatever the request asked
    for (``None`` when it asked for none), so a response that names no traffic tier is priced as
    standard only when the request did not ask for another one. `on` is the billing day, Pacific
    time, defaulting to today.

    ``None`` whenever any rate the call needs is not in the table — see the module docstring.
    """
    if usage is None:
        return None
    traffic = usage.traffic or ""
    if traffic in _UNNAMED_TRAFFIC:
        if _name(service_tier) not in _STANDARD_TIERS:
            return None
    elif traffic not in _STANDARD_TRAFFIC:
        return None
    tiers = _tiers(model, location, on or billing_day())
    if tiers is None:
        return None
    tier = tiers[1] if usage.prompt > LONG_CONTEXT_THRESHOLD else tiers[0]
    return _price(tier, usage)


def known(model: str) -> bool:
    """Whether the table carries any row at all for `model` — the adapter warns once when not."""
    return _model_key(model) in RATES


def billing_day(now: datetime | None = None) -> date:
    """The billing day it is in Google's billing timezone — UTC where the zone data is missing."""
    now = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        return now.astimezone(ZoneInfo(_BILLING_TZ)).date()
    except Exception:  # noqa: BLE001 - a box without tz data prices by the UTC day, never fails
        return now.astimezone(timezone.utc).date()


def _tiers(model: str, location: str, day: date) -> tuple[Tier, Tier] | None:
    """The ``(≤200K, >200K)`` rows that apply to this model, location and day."""
    for rates in RATES.get(_model_key(model), ()):
        if rates.start is not None and day < rates.start:
            continue
        if rates.end is not None and day > rates.end:
            continue
        return rates.global_ if location.strip().lower() == GLOBAL_LOCATION else rates.non_global
    return None


def _price(tier: Tier, usage: Usage) -> float | None:
    """The dollar sum of one call's tokens at `tier`'s rates — ``None`` if a needed rate is unstated.

    Cached tokens are a subset of the prompt and are priced at the cache rate *instead of* the input
    rate, never on top of it. The same holds for the audio share where audio is priced apart.
    """
    cached = min(usage.cached, usage.prompt)
    audio_cached = min(usage.audio_cached, cached) if tier.audio_apart else 0
    audio = max(0, min(usage.audio_prompt, usage.prompt) - audio_cached) if tier.audio_apart else 0
    uncached = max(0, usage.prompt - cached - audio) + usage.tool_prompt
    parts = (
        (uncached, tier.input),
        (cached - audio_cached, tier.cached),
        (audio, tier.audio_input),
        (audio_cached, tier.audio_cached),
        (usage.output, tier.output),
    )
    total = 0.0
    for tokens, rate in parts:
        if tokens <= 0:
            continue
        if rate is None:
            return None
        total += tokens * rate
    return total / 1_000_000


def _model_key(model: str) -> str:
    """`model` as the table keys it: the bare id, whatever resource path the caller wrote around it.

    Vertex accepts ``publishers/google/models/<id>`` and the full ``projects/…/models/<id>`` path as
    well as the bare id, and all three run (and bill) the same model.
    """
    return model.strip().rsplit("/", 1)[-1].lower()


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _modality(details: Any, modality: str) -> int:
    """How many of a token breakdown's tokens were `modality` — ``0`` when it names none."""
    total = 0
    for entry in details or ():
        if _name(getattr(entry, "modality", None)) == modality:
            total += _count(getattr(entry, "token_count", None)) or 0
    return total


def _name(value: Any) -> str:
    """An SDK enum (or the string it wraps) as its upper-case name."""
    return str(getattr(value, "value", value) or "").upper()
