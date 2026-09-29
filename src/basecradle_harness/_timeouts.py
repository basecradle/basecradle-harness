"""How long a model call may take — one policy, every adapter (issue #589).

On 2026-09-29 @glm-5.2 was handed a six-step job, and by the ninth step its context had grown to
~40 K tokens with 2–3 K-token replies. One generation legitimately needed more than a minute, and
every adapter-level answer to "how long may a call take?" was a **constant that knew nothing about
the call**: the `openrouter` and `openai` adapters gave the SDK a flat 60 s for every phase of every
request, and the native xAI adapter gave none at all (the SDK's own 27 minutes). The 60 s wall cut
the answer off, the retry re-sent the identical request into the identical wall twice more, and the
recovery replayed the whole thing on the next wake. Four wakes, fifteen timeouts, no answer — on a
provider whose uptime for that model was 100% the whole time.

The defect was one number doing two jobs. A timeout on a non-streaming call is there to catch a
**dead provider**; it was also bounding the **model's thinking**. So the policy separates them:

- **Connect** — `CONNECT_TIMEOUT`, short and fixed whatever the call. Reaching an endpoint does not
  get slower as a conversation grows, so a provider that cannot be reached in ten seconds is not
  going to be. It is an HTTP phase: the native xAI adapter speaks gRPC, whose own connection
  machinery fails an unreachable call fast (``UNAVAILABLE``) and whose keepalive pings notice a
  connection that dies mid-call, so there the fitted budget is the whole call's deadline.
- **Generation** — everything after the request is accepted, **fitted to the call**: a floor, plus
  time to read the request, plus time to write the answer, capped. `generation_timeout` is the fit.

Why a fitted timeout and not streaming
--------------------------------------
Streaming is the other honest answer — an idle timeout between chunks tells a dead provider from a
slow one exactly — and it was considered and not taken. Every adapter is non-streaming **by
contract** (`_basecradle._OWNED_*` strips ``stream`` for that reason), and the `openrouter` SDK ships
no accumulator: the harness would have to reassemble tool-call deltas, usage, finish reason, routing
metadata and citations from chunks itself, on the hot path of every agent, and the raw-body capture
that recovers web-search citations reads a whole body. That is a rewrite of three adapters' wire
handling to fix a number. A fit bounds the answer by what the call is, which is what the defect
needed; streaming stays the stronger mechanism if a fit ever proves too coarse.

The fit, and where each number comes from
-----------------------------------------
``generation = FLOOR + prefill_tokens / PREFILL_RATE + output_tokens / DECODE_RATE``, rounded **up**
to a whole `GENERATION_STEP` and capped at `GENERATION_CEILING`.

- The rates are **slow-but-alive floors, not predictions**: a bound must exceed every legitimate
  answer, so it is built from the slowest throughput still worth waiting for. The incident's own
  successful call ran 35.6 K tokens in / 2.7 K out in 48 s (~55 tok/s end to end); the decode floor
  is under half of that.
- ``prefill_tokens`` is the request measured in characters as the model reads them
  (`_context.request_chars`) and converted at `_context.WORST_CASE_CHARS_PER_TOKEN` — the estimate
  that counts the *most* tokens, which is the safe direction for a timeout.
- ``output_tokens`` is the cap the call actually carries (`output_cap` — ``max_tokens``,
  ``max_completion_tokens`` or ``max_output_tokens``, whichever the SDK spells), because that is the
  most it can write. An uncapped call is assumed to write `UNCAPPED_OUTPUT_TOKENS`, which is where a
  reasoning model at high effort lands.
- The floor is the old wall. No call gets **less** time than it had before this policy.
- The **whole-step rounding** is not cosmetic: the native xAI SDK fixes a deadline per client, so
  that adapter rebuilds its client when the deadline changes, and a context that grows ~2 K tokens a
  step would otherwise change it on every call. Rounded, it changes every few dozen steps.

At the shipped constants (``chars`` measured the way `request_chars` measures them)::

    ~10 K-token request, uncapped      60 + 10 + 410  ->  480 s
    ~40 K (the incident), uncapped     60 + 40 + 410  ->  540 s (510, rounded up)
    ~200 K, uncapped                   60 + 200 + 410 ->  720 s (670, rounded up)
    a described image, 2,048 cap       60 + ~2 + 103  ->  180 s

The retry
---------
A timed-out call is retried **once, at `TIMEOUT_RETRY_SCALE` times the budget** — every phase of it,
connect included, so a connect timeout's retry is not the identical request either — and never
re-sent into the same wall (`_retry.Retry`). Once, because a fit that has already been exceeded twice is not
describing a slow answer. The engine and the describer hand the scale to the adapter they call
through `bind_scale`, a capability read the way `_caching.bind_conversation` is: an adapter that
does not declare it keeps its own timeout, and nothing breaks. The reranker fits its own call, so it
reads the scale directly.

**The worst case is stated, not hidden.** A provider that accepts a request and then never answers
costs one fit plus twice the fit before the wake fails — ~27 minutes for the incident's call — where
the 60 s wall cost three minutes and then looped for an hour. That is the price of a non-streaming
contract, it is paid only on a genuine hang, and the NOC's Wake Duration Outlier alert is the thing
that sees it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger("basecradle_harness")

#: Seconds to establish a connection (and to wait for one from the pool). Fixed whatever the call,
#: because reaching an endpoint does not depend on it: a provider that cannot be reached in ten
#: seconds is down. Scaled only for a timeout's one retry, like the rest of the budget.
CONNECT_TIMEOUT = 10.0

#: The least any generation is given — the old flat wall, kept as a floor so no call gets less time
#: than it had before issue #589. It is what a small request's queueing and time-to-first-token ride.
GENERATION_FLOOR = 60.0

#: The slowest request-reading rate still worth waiting for, in tokens per second. Hosted endpoints
#: prefill at several thousand; a cached prefix is faster still.
PREFILL_TOKENS_PER_SECOND = 1_000

#: The slowest writing rate still worth waiting for, in tokens per second — reasoning tokens
#: included, since a vendor bills and times them the same. Under half the incident's own end-to-end
#: rate (~55 tok/s on a call that succeeded), so an answer that is merely slow fits.
DECODE_TOKENS_PER_SECOND = 20

#: What a call that carries no output cap is assumed to write. Where a reasoning model at high effort
#: lands on a long answer — and the term that dominates an uncapped brain call's fit.
UNCAPPED_OUTPUT_TOKENS = 8_192

#: The most a first attempt is ever given, however large the call. Fifteen minutes: past every fit a
#: real request has produced, and short enough that a hang is not an hour.
GENERATION_CEILING = 900.0

#: The resolution a fit is rounded **up** to. See the module docstring for why a native gRPC client
#: needs it; for the HTTP adapters it only ever adds time.
GENERATION_STEP = 60.0

#: The budget for an adapter's **metadata** reads — a model's context window, its input modalities —
#: which are catalog lookups, not model calls, and are not fitted. The flat minute every call used to
#: get is right for them, and it is spelled here so they keep it: with no client-wide timeout left on
#: the HTTP adapters, a read made before the first turn would otherwise take the transport's own
#: default (five seconds on `httpx`), and one made after it the last turn's fitted budget.
METADATA_TIMEOUT = 60.0

#: How much larger the one retry a timeout earns is. "Materially larger" is the requirement (issue
#: #589): the same request with the same budget would fail the same way, so the retry buys time,
#: not luck.
TIMEOUT_RETRY_SCALE = 2.0

#: The keys a call's output cap goes by — one per wire (`_basecradle._openai_budget_key`). Read in
#: this order; a call carries one of them at most.
OUTPUT_CAP_KEYS = ("max_output_tokens", "max_completion_tokens", "max_tokens")


@dataclass(frozen=True)
class CallTimeout:
    """The two budgets one model call is given: reaching the endpoint, and everything after."""

    connect: float
    generation: float

    def phases(self) -> dict[str, float]:
        """The budget as an HTTP client's four phases — the one place they are mapped.

        Both HTTP stacks the adapters drive (`httpx`, and the `openai` SDK's HTTPX2) take these four
        names. ``read`` is the one that matters: a non-streaming response sends nothing until the
        answer is done, so the read wait **is** the generation. ``write`` gets the same allowance,
        because a request carrying a picture is large and the link may be slow; ``pool`` is a wait
        for a connection and gets the connect budget.
        """
        return {
            "connect": self.connect,
            "read": self.generation,
            "write": self.generation,
            "pool": self.connect,
        }


def output_cap(params: Mapping[str, Any]) -> int | None:
    """The output cap a call carries — whichever spelling its SDK takes — or ``None`` if uncapped.

    A value that is not a positive integer is not a cap (``True`` is an ``int`` in Python and would
    read as one token). An unusable one is ignored rather than trusted: the vendor will say what it
    thinks of it, and a timeout must never be *shortened* by a value the call cannot actually mean.
    """
    for key in OUTPUT_CAP_KEYS:
        value = params.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def generation_timeout(chars: int, *, output_tokens: int | None, scale: float = 1.0) -> float:
    """Seconds a call of `chars` request characters and `output_tokens` output may take to answer.

    See the module docstring for the fit and where each constant comes from. `chars` is measured by
    `_context.request_chars`; `output_tokens` is `output_cap` of the call's parameters (``None`` →
    `UNCAPPED_OUTPUT_TOKENS`). `scale` multiplies the capped, rounded fit, so the retry a timeout
    earns is always exactly `TIMEOUT_RETRY_SCALE` times the first attempt's budget — including at
    the ceiling, where a cap applied after the scale would make the retry no larger at all.
    """
    # Imported here, not at module top: `_context` imports `_engine`, which imports this module, so a
    # top-level import would be a cycle. The constant is the one `_context`'s own bounds use, and a
    # second spelling of it would be two answers to "how many tokens is a character?".
    from basecradle_harness._context import WORST_CASE_CHARS_PER_TOKEN

    prefill = max(0, chars) / WORST_CASE_CHARS_PER_TOKEN / PREFILL_TOKENS_PER_SECOND
    written = (output_tokens or UNCAPPED_OUTPUT_TOKENS) / DECODE_TOKENS_PER_SECOND
    fit = min(GENERATION_FLOOR + prefill + written, GENERATION_CEILING)
    # Rounded to nine places before the ceiling, so float noise in the sum (``1.1 + 2.2`` is
    # ``3.3000000000000003``) cannot push a fit that lands on a step up a whole step.
    rounded = math.ceil(round(fit / GENERATION_STEP, 9)) * GENERATION_STEP
    return rounded * max(1.0, scale)


def call_timeout(
    chars: int, *, output_tokens: int | None, scale: float = 1.0, fixed: float | None = None
) -> CallTimeout:
    """The whole budget for one call: the fixed connect, and the generation fitted to it.

    `scale` multiplies both, so the one retry a timeout earns is larger in every phase — a
    connect timeout's retry included. `fixed` is an adapter's explicit ``timeout=`` — a library
    caller that knows its calls and wants a set generation budget instead of the fit. It is scaled
    like the fit. No deployment passes one: the config layer owns the key.
    """
    factor = max(1.0, scale)
    generation = (
        generation_timeout(chars, output_tokens=output_tokens, scale=scale)
        if fixed is None
        else fixed * factor
    )
    return CallTimeout(connect=CONNECT_TIMEOUT * factor, generation=generation)


def bind_scale(provider: object, scale: float) -> None:
    """Tell an adapter how much of its fitted budget its next calls get — ``1.0``, or the retry's.

    A capability, read the way `_caching.bind_conversation` is: an adapter that fits its timeouts
    declares ``bind_timeout_scale``, and one that does not (a third-party adapter, a test double) is
    left alone — it keeps whatever timeout it has, and the retry still happens, just without the
    larger budget. The binding is **sticky** until the next one, so every caller that raises it
    lowers it again in a ``finally`` (`Engine._chat`, `Describer._attempt`); a compaction summarize
    call made after a turn therefore runs at the ordinary budget.

    Best-effort by construction, for the same reason affinity is: a budget hint is not worth a wake.
    """
    bind = getattr(provider, "bind_timeout_scale", None)
    if not callable(bind):
        return
    try:
        bind(scale)
    except Exception as exc:  # noqa: BLE001 - a timeout hint must never break a wake
        _log.warning("Could not bind the timeout scale on the model provider: %s", exc)


def last_timeout(provider: object) -> float | None:
    """The generation budget `provider` applied to its most recent call, in seconds, if it says.

    Read for the retry and give-up lines, so a timeout in the journal names the budget it ran out
    of. An adapter that does not fit its timeouts answers nothing, and the line omits the field.
    """
    value = getattr(provider, "last_timeout", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
