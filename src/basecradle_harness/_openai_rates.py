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
  there. The numbers live in the package's public prices file (``data/prices.json``, read through
  `basecradle_harness._prices`, issue #997), never here: when the page moves, the file's ``openai``
  section is edited from the page in a reviewed PR, with the new date.
- **Which row applies is decided by the call's own facts**: the model id, the prompt size (above
  `LONG_CONTEXT_THRESHOLD` the long-context row prices **every** token of the call), the cache split
  the usage block reports (a cache *write* is its own rate on GPT-5.6 and later), and the service tier
  the response names.

**Its watcher is the NOC's fleet price check** (basecradle-noc#1025 / #1147), which checks each
row against OpenRouter's listed price or the vendor's billed cost; a table it does not check is
not shipped (issue #755).

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
- **Web search** (`web_search_cost`): $10 per 1,000 search calls, on the call's ``web_search`` ``tool``
  line — the search content tokens it feeds back are already in the call's input tokens ("billed at
  model rates").
- **Not priced: code-interpreter containers.** The page bills a container per 20-minute session by
  memory size, and "eligible" sessions by the minute with a five-minute minimum. Which applies, and how
  long a session lived, is nowhere in a response, so any figure would be a guess. The adapter states
  that gap once per wake as a ``cost unpriced charge=container_time reason=not_reported`` WARNING
  rather than log a number it cannot stand behind or stay silent about a charge it knows exists.
- **Not priced: data-residency and FedRAMP endpoints.** The page adds a 10% uplift there; the table is
  for OpenAI's own host (`PRICED_HOST`), and a call anywhere else is unpriced and warned about.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from basecradle_harness import _prices
from basecradle_harness._observability import ComputedCost

_DATA = _prices.vendor("openai")

#: Where every number in this module's tables was read, and when. Edit the file's tables from this page, and
#: move the date with them.
SOURCE = _prices.source(_DATA)

#: Above this many input tokens a call is priced on the long-context row — all of its tokens
#: ("Short context: ≤272K input tokens. Long context: >272K input tokens.").
LONG_CONTEXT_THRESHOLD: int = _DATA["long_prompt"]["threshold"]

#: Whether a model with no long-context row is priced on its short row above the threshold: no —
#: the page leaves that row blank, and a blank is not a price.
_SHORT_ROW_WITHOUT_LONG = _prices.short_row_without_long(_DATA)

#: The one host these rates are OpenAI's list price on. Regional-processing (data residency) and
#: FedRAMP endpoints are "charged a 10% uplift", and a proxy's price is its own business.
PRICED_HOST: str = _DATA["priced_host"]

#: What a response's ``service_tier`` says when it ran on the Standard tier.
_STANDARD_SERVED = frozenset(_DATA["standard_service_tiers_served"])
#: What a request's ``service_tier`` is when it asked for nothing in particular (the empty name is
#: an unset field), so a response that names no tier is read as standard (``auto`` is "the service
#: tier configured in the Project settings", which is ``default`` unless an operator changed it —
#: and a response served on another tier says so).
_STANDARD_ASKED = frozenset(_DATA["standard_service_tiers_asked"])

#: OpenAI's charge per web search call: "$10.00 / 1k calls" (Web search, all models).
WEB_SEARCH_PER_CALL: float = _DATA["web_search_usd_per_1000"] / 1_000


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
# Transcribed row by row from `SOURCE`, "Standard pricing data", into the prices file's ``openai``
# section, whose ``note`` says which models are carried and why.

#: Model id → its Standard rates. Keyed by the alias OpenAI publishes; a dated snapshot of it
#: (``gpt-6-sol-2026-08-14``) is priced as its alias (`_model_key`).
RATES: Mapping[str, Rates] = {
    model: Rates(
        short=_prices.row(Tier, rows["short"]),
        long=None if rows["long"] is None else _prices.row(Tier, rows["long"]),
    )
    for model, rows in _DATA["models"].items()
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
    model: _prices.row(ImageRates, row) for model, row in _DATA["image_models"].items()
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
#: if the page splits audio from text again, the file grows a field, never a guessed ratio.
TRANSCRIPTION_RATES: Mapping[str, TranscriptionRates] = {
    model: _prices.row(TranscriptionRates, row)
    for model, row in _DATA["transcription_models"].items()
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
    over = usage.input > LONG_CONTEXT_THRESHOLD
    if over and rates.long is None and not _SHORT_ROW_WITHOUT_LONG:
        return None
    tier = rates.long if over and rates.long is not None else rates.short
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


def image_known(model: str) -> bool:
    """Whether `IMAGE_RATES` carries `model` — an unpriced image call's ``unknown_model`` otherwise."""
    return _model_key(model) in IMAGE_RATES


def transcription_known(model: str) -> bool:
    """Whether `TRANSCRIPTION_RATES` carries `model` — ``unknown_model`` otherwise."""
    return _model_key(model) in TRANSCRIPTION_RATES


def priced_host(base_url: str | None) -> bool:
    """Whether `base_url` is OpenAI's own API host, the one these rates are the list price on."""
    if not base_url:
        return True
    return (urlsplit(base_url).hostname or "").lower() == PRICED_HOST


#: The file's model-key rule: a dated snapshot suffix (``-2026-08-14``) is priced as the alias it
#: pins.
_model_key = _prices.model_key(_DATA)


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
