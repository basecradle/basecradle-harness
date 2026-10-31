"""What a Claude call cost, priced from Anthropic's published rates (issue #725).

Anthropic's Messages API returns token usage and **no price**, so a Claude brain's ``llm`` line
would carry no ``cost=`` and the agent would be invisible on the fleet's spend dashboard. The ruling
the capital made for Google (#655) and OpenAI (#657) applies unchanged: price the call from a
maintained table, tag it ``cost_basis=computed``, and never sum it with a stated figure.

The three rules that hold `_google_rates` and `_openai_rates` honest hold this table too:

- **A figure the table cannot state is omitted, never estimated.** An unknown model, a call served
  on a tier other than standard, a fast-mode call, an inference geography the page prices nowhere:
  each returns ``None`` and the line carries no ``cost=``. A wrong number is worse than a missing
  one, because the dashboard sums it.
- **Every number is transcribed, never derived** — row by row from `SOURCE`, as it stood on the date
  there. When the page moves, the table is edited from the page in a reviewed PR, with the new date.
- **Which row applies is decided by the call's own facts**: the model id, the prompt length (on
  Claude Haiku 5.5, a prompt over `HAIKU_LONG_PROMPT_THRESHOLD` prices **every** token of the call
  on the higher row — "a request's prompt length counts all of its input tokens, including cache
  reads and cache writes"), the cache split the usage block reports (a 5-minute write, a 1-hour
  write and a read are three different rates), and the inference geography the request asked for.

**Its watcher is the NOC's fleet price check** (basecradle-noc#1025 / #1147), which checks each
row against OpenRouter's listed price or the vendor's billed cost; a table it does not check is
not shipped (issue #755).

What is priced, and what is not
-------------------------------
- **Model calls** (`call_cost`): standard tier, standard speed. ``inference_geo="us"`` applies the
  page's 1.1x multiplier to every token category; global routing (the default) is the list price.
- **Web search** (`web_search_cost`): $10 per 1,000 searches, counted from
  ``usage.server_tool_use.web_search_requests``, as its own priced ``media`` line (``kind=search.web``,
  the head the fleet's tool spend is read from). The search results it feeds back are already in the
  call's input tokens. "If an error occurs during web search, the web search will not be billed",
  and the count Anthropic reports is the billed one.
- **Web fetch**: "no additional charges beyond standard token costs" — nothing to price.
- **Not priced: code-execution container time on a request that declares neither web search nor web
  fetch.** With either of those declared "there are no additional charges for code execution tool
  calls"; without them Anthropic bills execution time per container beyond 1,550 free hours a month
  per organization, and no response states the time or where the organization stands against the
  allowance. Any figure would be a guess, so the adapter states the gap once per wake as a WARNING
  (`CONTAINER_GAP`) instead.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from basecradle_harness._observability import ComputedCost

#: Where every number in this module was read, and when. Edit the tables from this page, and move
#: the date with them.
SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing (read 2026-10-31)"

#: On Claude Haiku 5.5, a request whose prompt is over this many tokens pays the higher row for all
#: of them ("for prompts over 100,000 tokens"). Every other current model is priced flat across its
#: whole context window ("A 900k-token request is billed at the same per-token rate as a 9k-token
#: request").
HAIKU_LONG_PROMPT_THRESHOLD = 100_000

#: "$10 USD per 1,000 searches" (Web search tool).
WEB_SEARCH_PER_SEARCH = 10.00 / 1_000

#: "Specifying US-only inference through the `inference_geo` parameter incurs a 1.1x multiplier on
#: all token pricing categories … Global routing (the default) uses standard pricing."
US_INFERENCE_MULTIPLIER = 1.1

#: The ``inference_geo`` values a request may carry and the multiplier the page states for each.
#: Unset is global routing. A value the page prices nowhere is not priced.
_GEO_MULTIPLIERS = {"": 1.0, "global": 1.0, "us": US_INFERENCE_MULTIPLIER}

#: The WARNING the adapter writes, once per wake, when code execution ran on a request that declared
#: neither web search nor web fetch — see the module docstring for why it is a stated gap.
CONTAINER_GAP = (
    "Anthropic bills code-execution container time beyond 1,550 free hours a month per organization "
    "($0.05 per container-hour) on a request that declares neither web search nor web fetch, and no "
    "response states the time used. Code-execution charges are therefore not priced on any line this "
    "wake; read them off the Anthropic Console's usage page."
)


@dataclass(frozen=True)
class Tier:
    """One row of the page's model table, in USD per million tokens."""

    input: float
    write_5m: float
    write_1h: float
    read: float
    output: float


@dataclass(frozen=True)
class Rates:
    """A model's rates: the row for every prompt, or — Claude Haiku 5.5 — the two prompt-length rows."""

    short: Tier
    #: The row for a prompt over `HAIKU_LONG_PROMPT_THRESHOLD`; ``None`` for a model priced flat.
    long: Tier | None = None


# --- the table --------------------------------------------------------------------------------
#
# Transcribed row by row from `SOURCE`, "Model pricing". The current generation and the one before
# it, so an operator moving within a family is priced from the first call; retired models are left
# out. Claude Mythos is left out too: it is limited-availability, and no fleet agent can run it.

#: Model id → its rates. Keyed by the id Anthropic publishes; a dated snapshot of it
#: (``claude-haiku-4-5-20251001``) is priced as that id (`_model_key`).
RATES: Mapping[str, Rates] = {
    "claude-fable-5-1": Rates(
        Tier(input=10.0, write_5m=12.50, write_1h=20.0, read=0.25, output=50.0)
    ),
    "claude-fable-5": Rates(Tier(input=10.0, write_5m=12.50, write_1h=20.0, read=1.0, output=50.0)),
    "claude-opus-5-5": Rates(Tier(input=4.0, write_5m=5.0, write_1h=8.0, read=0.20, output=20.0)),
    "claude-opus-5": Rates(Tier(input=5.0, write_5m=6.25, write_1h=10.0, read=0.50, output=25.0)),
    "claude-opus-4-8": Rates(Tier(input=5.0, write_5m=6.25, write_1h=10.0, read=0.50, output=25.0)),
    "claude-opus-4-7": Rates(Tier(input=5.0, write_5m=6.25, write_1h=10.0, read=0.50, output=25.0)),
    "claude-opus-4-6": Rates(Tier(input=5.0, write_5m=6.25, write_1h=10.0, read=0.50, output=25.0)),
    "claude-sonnet-5-5": Rates(
        Tier(input=2.0, write_5m=2.50, write_1h=4.0, read=0.10, output=10.0)
    ),
    "claude-sonnet-5": Rates(Tier(input=2.0, write_5m=2.50, write_1h=4.0, read=0.20, output=10.0)),
    "claude-sonnet-4-6": Rates(
        Tier(input=3.0, write_5m=3.75, write_1h=6.0, read=0.30, output=15.0)
    ),
    "claude-haiku-5-5": Rates(
        short=Tier(input=0.10, write_5m=0.125, write_1h=0.20, read=0.01, output=0.50),
        long=Tier(input=0.50, write_5m=0.625, write_1h=1.0, read=0.05, output=2.50),
    ),
    "claude-haiku-4-5": Rates(Tier(input=1.0, write_5m=1.25, write_1h=2.0, read=0.10, output=5.0)),
}


@dataclass(frozen=True)
class Usage:
    """The token counts a Messages call reported, in the shape pricing needs.

    Anthropic reports the prompt in three disjoint parts — ``input_tokens`` (uncached),
    ``cache_creation_input_tokens`` and ``cache_read_input_tokens`` — and "total input tokens in a
    request is the summation" of the three. ``write_5m`` / ``write_1h`` split the written part by
    TTL; ``output`` includes the thinking, which Anthropic bills as output.
    """

    uncached: int
    write_5m: int
    write_1h: int
    read: int
    output: int

    @property
    def prompt(self) -> int:
        """Every input token of the call — what the Haiku 5.5 prompt-length rule counts."""
        return self.uncached + self.write_5m + self.write_1h + self.read


def usage_of(usage: Any, *, written_ttl: str = "5m") -> Usage | None:
    """A Messages ``usage`` block as a `Usage`, or ``None`` if it carries no token counts.

    The written tokens are split by the ``cache_creation`` breakdown when the response carries one;
    a response that reports a write and no breakdown is read at `written_ttl`, the one TTL the
    adapter asked for on every breakpoint it placed. Read defensively, as a mapping or as attributes: this
    is the logging path, where a surprise in a vendor object costs the cost field, never the turn.
    """
    if usage is None:
        return None
    uncached = _count(_read(usage, "input_tokens"))
    output = _count(_read(usage, "output_tokens"))
    if uncached is None and output is None:
        return None
    written = _count(_read(usage, "cache_creation_input_tokens")) or 0
    breakdown = _read(usage, "cache_creation")
    five = _count(_read(breakdown, "ephemeral_5m_input_tokens"))
    hour = _count(_read(breakdown, "ephemeral_1h_input_tokens"))
    if five is None and hour is None:
        five, hour = (0, written) if written_ttl == "1h" else (written, 0)
    return Usage(
        uncached=uncached or 0,
        write_5m=five or 0,
        write_1h=hour or 0,
        read=_count(_read(usage, "cache_read_input_tokens")) or 0,
        output=output or 0,
    )


def call_cost(
    model: str, usage: Usage | None, *, geo: Any = None, tier: Any = None, speed: Any = None
) -> ComputedCost | None:
    """What this Messages call cost in USD at Anthropic's published rates, or ``None``.

    `geo` is the ``inference_geo`` the request asked for; `tier` and `speed` are what the response's
    usage says the call was served on (``service_tier``, ``speed``). Anything but standard on either
    is not priced: the priority and batch tiers and fast mode are other rows of the page, or none.
    """
    if usage is None:
        return None
    multiplier = _GEO_MULTIPLIERS.get(_name(geo))
    if (
        multiplier is None
        or _name(tier) not in ("", "standard")
        or _name(speed)
        not in (
            "",
            "standard",
        )
    ):
        return None
    rates = RATES.get(_model_key(model))
    if rates is None:
        return None
    row = (
        rates.long
        if rates.long is not None and usage.prompt > HAIKU_LONG_PROMPT_THRESHOLD
        else (rates.short)
    )
    total = (
        usage.uncached * row.input
        + usage.write_5m * row.write_5m
        + usage.write_1h * row.write_1h
        + usage.read * row.read
        + usage.output * row.output
    )
    return ComputedCost(total * multiplier / 1_000_000)


def web_search_cost(searches: int) -> ComputedCost | None:
    """The per-search fee for `searches` web searches — ``None`` for none."""
    return ComputedCost(searches * WEB_SEARCH_PER_SEARCH) if searches > 0 else None


def server_tool_requests(usage: Any, name: str) -> int:
    """How many server-tool requests of kind `name` a ``usage`` block reports (``0`` for none).

    `name` is the field under ``usage.server_tool_use`` — ``web_search_requests``,
    ``web_fetch_requests``, ``code_execution_requests``. The last is not a declared field of the
    SDK's ``ServerToolUsage`` model, so it is read off the model's extra fields as well as its
    attributes: the page documents it, and the SDK keeps a field it was not told about.
    """
    block = _read(usage, "server_tool_use")
    value = _read(block, name)
    if value is None and block is not None:
        extra = getattr(block, "model_extra", None)
        value = extra.get(name) if isinstance(extra, Mapping) else None
    return _count(value) or 0


def known(model: str) -> bool:
    """Whether the model table carries `model` at all — the adapter warns once when it does not."""
    return _model_key(model) in RATES


#: A dated snapshot suffix (``-20251001``), priced as the id it pins.
_SNAPSHOT = re.compile(r"-\d{8}$")


def _model_key(model: str) -> str:
    return _SNAPSHOT.sub("", model.strip().lower())


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
