"""What an OpenAI call cost, priced from OpenAI's published rates (issue #657).

OpenAI's API returns token usage and **no price**: no field on a Responses or Chat Completions body,
an Images response, or a transcription says what the call cost. So every ``llm provider=openai``
line the fleet wrote carried no ``cost=`` (206 brain calls in 30 days, none with a cost —
basecradle#645), and an agent brained on GPT-6 Sol was invisible on *Total Spend by Provider* and
*Spend by Agent* alike. The capital's ruling on #657 is the one it made for Google on #655: price the
call from a maintained table, one mechanism for both vendors (`basecradle_harness._google_rates` is
the other half). A vendor that states dollars (OpenRouter, xAI) keeps stating them; nothing here is
ever summed with a stated figure or used to second-guess one.

The three rules that hold `_google_rates` honest hold this table too:

- **A figure the table cannot state is omitted, never estimated.** An unknown model, a cell the page
  leaves blank, a call above the long-context threshold on a model the page gives no long-context row,
  a call served on a tier other than Standard, a call sent to any host but OpenAI's own: each returns
  ``None`` and the line carries no ``cost=``. A wrong number is worse than a missing one, because the
  dashboard sums it.
- **Every number is transcribed, never derived** — row by row from `SOURCE`, as it stood on the date
  there. When the page moves, the table is edited from the page in a reviewed PR, with the new date.
- **Which row applies is decided by the call's own facts**: the model id, the prompt size (above
  `LONG_CONTEXT_THRESHOLD` the long-context row prices **every** token of the call), the cache split
  the usage block reports (a cache *write* is its own rate on GPT-5.6 and later), and the service tier
  the response names.

What is priced, and what is not
-------------------------------
- **Model calls** (`call_cost`): Standard tier only. Flex, Fast (formerly Priority), Ultrafast, Batch
  and Scale are priced elsewhere on the page or not per call, and a response that names one gets no
  cost.
- **Image generation** (`image_cost`): by the Images API's token usage, image and text priced apart.
  The page's cached-input image rates "only apply to images generated with the Responses API", and the
  Images endpoint the harness calls reports no cached count, so none is applied.
- **Transcription** (`transcription_cost`): per minute of audio where the response reports a duration
  (``gpt-transcribe``), by tokens where it reports tokens (``gpt-4o-transcribe``).
- **Web search** (`web_search_cost`): $10 per 1,000 search calls, as its own priced line — the search
  content tokens it feeds back are already in the call's input tokens ("billed at model rates").
- **Not priced: code-interpreter containers.** The page bills a container per 20-minute session by
  memory size, and "eligible" sessions by the minute with a five-minute minimum. Which applies, and how
  long a session lived, is nowhere in a response, so any figure would be a guess. The adapter states
  that gap once per wake as a WARNING (`CONTAINER_GAP`) rather than log a number it cannot stand
  behind or stay silent about a charge it knows exists.
- **Not priced: data-residency and FedRAMP endpoints.** The page adds a 10% uplift there; the table is
  for OpenAI's own host (`PRICED_HOST`), and a call anywhere else is unpriced and warned about.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from basecradle_harness._observability import ComputedCost

#: Where every number in this module was read, and when. Edit the tables from this page, and move
#: the date with them.
SOURCE = "https://developers.openai.com/api/docs/pricing (Standard tier, read 2026-10-07)"

#: Above this many input tokens a call is priced on the long-context row — all of its tokens
#: ("Short context: ≤272K input tokens. Long context: >272K input tokens.").
LONG_CONTEXT_THRESHOLD = 272_000

#: The one host these rates are OpenAI's list price on. Regional-processing (data residency) and
#: FedRAMP endpoints are "charged a 10% uplift", and a proxy's price is its own business.
PRICED_HOST = "api.openai.com"

#: What a response's ``service_tier`` says when it ran on the Standard tier.
_STANDARD_SERVED = frozenset({"default"})
#: What a request's ``service_tier`` is when it asked for nothing in particular, so a response that
#: names no tier is read as standard (``auto`` is "the service tier configured in the Project
#: settings", which is ``default`` unless an operator changed it — and a response served on another
#: tier says so).
_STANDARD_ASKED = frozenset({"", "auto", "default"})

#: OpenAI's charge per web search call: "$10.00 / 1k calls" (Web search, all models).
WEB_SEARCH_PER_CALL = 10.00 / 1_000

#: The WARNING the adapter writes, once per wake, when a code-interpreter container ran — see the
#: module docstring for why it is a stated gap rather than a figure.
CONTAINER_GAP = (
    "OpenAI bills code-interpreter containers per session (a 20-minute session at $0.03 for 1 GB, "
    "or by the minute with a five-minute minimum for eligible sessions), and no response says which "
    "applies or how long the session lived. Container charges are therefore not priced on any line "
    "this wake; read them off OpenAI's usage dashboard."
)


@dataclass(frozen=True)
class Tier:
    """One context tier's rates, in USD per million tokens, as the page states them.

    ``cache_write`` is ``None`` for a model the page lists no cache-write rate for: those models
    charge nothing extra to write the cache ("No additional cache-write charge"), so a written token
    is an ordinary input token.
    """

    input: float
    cached: float
    output: float
    cache_write: float | None = None


@dataclass(frozen=True)
class Rates:
    """A model's Standard rates: the short-context row, and the long one where the page states one."""

    short: Tier
    #: ``None`` where the page leaves the long-context row blank; a call above the threshold there is
    #: not priced.
    long: Tier | None


# --- the table --------------------------------------------------------------------------------
#
# Transcribed row by row from `SOURCE`, "Standard pricing data". Every OpenAI model the fleet runs
# or has run is here (gpt-6-sol today; gpt-5.6-terra and gpt-5.4-mini before it — basecradle-noc
# fleet/inventory.json history), with the rest of the GPT-6 and GPT-5.6 families beside them so an
# operator moving within a family is priced from the first call. The ``-pro`` models are left out:
# the page states them no cached-input rate, and an agent's every call reads cache.

#: Model id → its Standard rates. Keyed by the alias OpenAI publishes; a dated snapshot of it
#: (``gpt-6-sol-2026-08-14``) is priced as its alias (`_model_key`).
RATES: Mapping[str, Rates] = {
    "gpt-6-astra": Rates(
        short=Tier(input=10.00, cached=1.00, cache_write=12.50, output=50.00),
        long=Tier(input=20.00, cached=2.00, cache_write=25.00, output=75.00),
    ),
    "gpt-6.1-sol": Rates(
        short=Tier(input=2.00, cached=0.10, cache_write=2.50, output=10.00),
        long=Tier(input=4.00, cached=0.20, cache_write=5.00, output=15.00),
    ),
    "gpt-6-sol": Rates(
        short=Tier(input=2.00, cached=0.20, cache_write=2.50, output=10.00),
        long=Tier(input=4.00, cached=0.40, cache_write=5.00, output=15.00),
    ),
    "gpt-6-luna": Rates(
        short=Tier(input=0.10, cached=0.01, cache_write=0.125, output=0.50),
        long=Tier(input=0.20, cached=0.02, cache_write=0.25, output=0.75),
    ),
    # The page: "GPT-5.6 Sol's promotional pricing is available at least through November 21, 2026."
    # No end date is stated, so none is encoded; the NOC's list-price check is what notices a move.
    "gpt-5.6-sol": Rates(
        short=Tier(input=4.00, cached=0.40, cache_write=5.00, output=20.00),
        long=Tier(input=8.00, cached=0.80, cache_write=10.00, output=30.00),
    ),
    "gpt-5.6-terra": Rates(
        short=Tier(input=2.00, cached=0.20, cache_write=2.50, output=12.00),
        long=Tier(input=4.00, cached=0.40, cache_write=5.00, output=18.00),
    ),
    "gpt-5.6-luna": Rates(
        short=Tier(input=0.20, cached=0.02, cache_write=0.25, output=1.20),
        long=Tier(input=0.40, cached=0.04, cache_write=0.50, output=1.80),
    ),
    "gpt-5.5": Rates(
        short=Tier(input=5.00, cached=0.50, output=30.00),
        long=Tier(input=10.00, cached=1.00, output=45.00),
    ),
    "gpt-5.4": Rates(
        short=Tier(input=2.50, cached=0.25, output=15.00),
        long=Tier(input=5.00, cached=0.50, output=22.50),
    ),
    "gpt-5.4-mini": Rates(short=Tier(input=0.75, cached=0.075, output=4.50), long=None),
    "gpt-5.4-nano": Rates(short=Tier(input=0.20, cached=0.02, output=1.25), long=None),
}


@dataclass(frozen=True)
class ImageRates:
    """An image model's Standard rates per million tokens, image and text priced apart.

    ``text_output`` is ``None`` where the page states no text-output rate; a call that reports text
    output tokens there is not priced.
    """

    image_input: float
    text_input: float
    image_output: float
    text_output: float | None = None


#: The image models the harness calls (`_images.GENERATE_MODEL` / `EDIT_MODEL`) and their family,
#: from the page's "Image generation models", Standard.
IMAGE_RATES: Mapping[str, ImageRates] = {
    "gpt-image-2.5-flare": ImageRates(image_input=8.00, text_input=5.00, image_output=30.00),
    "gpt-image-2.5-sunburst": ImageRates(image_input=8.00, text_input=5.00, image_output=30.00),
    "gpt-image-2": ImageRates(image_input=8.00, text_input=5.00, image_output=30.00),
    "gpt-image-1.5": ImageRates(
        image_input=8.00, text_input=5.00, image_output=32.00, text_output=10.00
    ),
}


@dataclass(frozen=True)
class TranscriptionRates:
    """A transcription model's price: per minute of audio, and per million tokens where stated."""

    per_minute: float | None
    input: float | None = None
    output: float | None = None


#: From the page's "Transcription models". ``gpt-transcribe`` is the harness's default
#: (`_audio.DEFAULT_MODEL`) and is priced per minute only. The token-billed models carry the one
#: input rate the page states for them (no separate audio-input row as of the date in `SOURCE`);
#: if the page splits audio from text again, the table grows a column, never a guessed ratio.
TRANSCRIPTION_RATES: Mapping[str, TranscriptionRates] = {
    "gpt-transcribe": TranscriptionRates(per_minute=0.0045),
    "gpt-4o-transcribe": TranscriptionRates(per_minute=0.006, input=2.50, output=10.00),
    "gpt-4o-mini-transcribe": TranscriptionRates(per_minute=0.003, input=1.25, output=5.00),
    "whisper-1": TranscriptionRates(per_minute=0.006),
}


@dataclass(frozen=True)
class Usage:
    """The token counts an OpenAI model call reported, in the shape pricing needs.

    ``input`` includes ``cached`` and ``cache_write`` (OpenAI: "input tokens use the uncached-input,
    cached-input, or cache-write rate" — one rate each, never two); ``output`` includes the reasoning
    tokens, which OpenAI bills as output.
    """

    input: int
    cached: int
    output: int
    cache_write: int = 0


def usage_of(usage: Any) -> Usage | None:
    """A Responses or Chat Completions ``usage`` block as a `Usage`, or ``None`` if it has no counts.

    Reads both surfaces' spellings (``input_tokens`` / ``prompt_tokens`` and their ``…_details``), as
    a mapping or as attributes, defensively: this is the logging path, where a surprise in a vendor
    object must cost the cost field and never the turn.
    """
    if usage is None:
        return None
    prompt = _count(_read(usage, "input_tokens"))
    if prompt is None:
        prompt = _count(_read(usage, "prompt_tokens"))
    output = _count(_read(usage, "output_tokens"))
    if output is None:
        output = _count(_read(usage, "completion_tokens"))
    if prompt is None and output is None:
        return None
    details = _read(usage, "input_tokens_details") or _read(usage, "prompt_tokens_details")
    return Usage(
        input=prompt or 0,
        cached=_count(_read(details, "cached_tokens")) or 0,
        cache_write=_count(_read(details, "cache_write_tokens")) or 0,
        output=output or 0,
    )


def call_cost(
    model: str,
    usage: Usage | None,
    *,
    served_tier: Any = None,
    asked_tier: Any = None,
) -> ComputedCost | None:
    """What this model call cost in USD at OpenAI's Standard rates, or ``None``.

    `served_tier` is the response's ``service_tier`` (the tier OpenAI says it actually used);
    `asked_tier` is what the request asked for, consulted only when the response names none.
    """
    if usage is None or not _standard(served_tier, asked_tier):
        return None
    rates = RATES.get(_model_key(model))
    if rates is None:
        return None
    tier = rates.long if usage.input > LONG_CONTEXT_THRESHOLD else rates.short
    if tier is None:
        return None
    cached = min(usage.cached, usage.input)
    written = min(usage.cache_write, usage.input - cached)
    write_rate = tier.input if tier.cache_write is None else tier.cache_write
    total = (
        (usage.input - cached - written) * tier.input
        + cached * tier.cached
        + written * write_rate
        + usage.output * tier.output
    )
    return ComputedCost(total / 1_000_000)


def image_cost(model: str, usage: Any) -> ComputedCost | None:
    """What an Images API call cost, from its ``usage`` block, or ``None``.

    The input split is required (``input_tokens_details``), because image and text input are priced
    apart and guessing the split is guessing the price. Output tokens are image tokens unless the
    response breaks them down: an image model's output *is* the image, and the page states no
    text-output rate for the current models — so where a breakdown reports text output on such a
    model, the call is not priced. A breakdown that names one half has the other as the remainder
    of ``output_tokens``, and one whose halves do not add up to it is not priced: dropping the
    unexplained tokens would drop $30/M image output from a figure still tagged as computed.
    """
    rates = IMAGE_RATES.get(_model_key(model))
    if rates is None or usage is None:
        return None
    details = _read(usage, "input_tokens_details")
    image_in = _count(_read(details, "image_tokens"))
    text_in = _count(_read(details, "text_tokens"))
    output = _count(_read(usage, "output_tokens"))
    if image_in is None or text_in is None or output is None:
        return None
    out_details = _read(usage, "output_tokens_details")
    image_out = _count(_read(out_details, "image_tokens"))
    text_out = _count(_read(out_details, "text_tokens"))
    if image_out is None and text_out is None:
        image_out, text_out = output, 0
    elif image_out is None:
        image_out = output - text_out
    elif text_out is None:
        text_out = output - image_out
    if image_out < 0 or text_out < 0 or image_out + text_out != output:
        return None
    if text_out and rates.text_output is None:
        return None
    total = (
        image_in * rates.image_input
        + text_in * rates.text_input
        + image_out * rates.image_output
        + text_out * (rates.text_output or 0.0)
    )
    return ComputedCost(total / 1_000_000)


def transcription_cost(model: str, usage: Any) -> ComputedCost | None:
    """What a transcription cost, from its ``usage`` block (a duration or token counts), or ``None``."""
    rates = TRANSCRIPTION_RATES.get(_model_key(model))
    if rates is None or usage is None:
        return None
    kind = _read(usage, "type")
    if kind == "duration":
        seconds = _read(usage, "seconds")
        if rates.per_minute is None or isinstance(seconds, bool):
            return None
        if not isinstance(seconds, (int, float)) or seconds < 0:
            return None
        return ComputedCost(seconds / 60 * rates.per_minute)
    if kind == "tokens":
        tokens_in = _count(_read(usage, "input_tokens"))
        tokens_out = _count(_read(usage, "output_tokens"))
        if rates.input is None or rates.output is None or tokens_in is None or tokens_out is None:
            return None
        return ComputedCost((tokens_in * rates.input + tokens_out * rates.output) / 1_000_000)
    return None


def web_search_cost(calls: int) -> ComputedCost | None:
    """The per-call fee for `calls` web search calls — ``None`` for none."""
    return ComputedCost(calls * WEB_SEARCH_PER_CALL) if calls > 0 else None


def search_calls(output: Any) -> int:
    """How many billed web search calls a Responses ``output`` list holds.

    "Search actions incur a tool call cost"; a ``web_search_call`` whose action is ``open_page`` or
    ``find_in_page`` (a reasoning model reading a result) is not a search and is not charged.
    """
    calls = 0
    for item in output or ():
        if _read(item, "type") != "web_search_call":
            continue
        if _read(_read(item, "action"), "type") == "search":
            calls += 1
    return calls


def ran_container(output: Any) -> bool:
    """Whether a Responses ``output`` list shows a code-interpreter call (a container session)."""
    return any(_read(item, "type") == "code_interpreter_call" for item in output or ())


def known(model: str) -> bool:
    """Whether the model table carries `model` at all — the adapter warns once when it does not."""
    return _model_key(model) in RATES


def priced_host(base_url: str | None) -> bool:
    """Whether `base_url` is OpenAI's own API host, the one these rates are the list price on."""
    if not base_url:
        return True
    return (urlsplit(base_url).hostname or "").lower() == PRICED_HOST


#: A dated snapshot suffix (``-2026-08-14``), priced as the alias it pins.
_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _model_key(model: str) -> str:
    return _SNAPSHOT.sub("", model.strip().lower())


def _standard(served: Any, asked: Any) -> bool:
    served = _name(served)
    if served:
        return served in _STANDARD_SERVED
    return _name(asked) in _STANDARD_ASKED


def _name(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


def _read(payload: Any, name: str) -> Any:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        return payload.get(name)
    return getattr(payload, name, None)


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
