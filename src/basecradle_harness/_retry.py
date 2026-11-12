"""One bounded retry for every model call the harness makes (issue #506).

On 2026-09-16 four of eighteen reranks fell back to plain hybrid retrieval, every one of them on a
transient OpenRouter 429, and every one of them recoverable: at 06:08:42 @briggs's rerank got a 429
and its **next** rerank succeeded on Parasail 1.3 seconds later; at 21:53:38 @briggs got a 429 while
@glm-5.2's rerank succeeded on Parasail *in the same second*. One bounded wait would have recovered
all four. The founder's ruling: **a rerank that waits up to three seconds and works beats one that
silently runs without the rerank.**

The three call sites this serves — the MemPalace reranker, the blind-model describer, and the
engine's own brain call — had three different answers to the same question (two had *no* retry; the
engine had one that excluded 429 by design). One policy, in one module, is the point: a fault class
is retried because of **what it is**, never because of which caller happened to hit it.

**The reason set is a ceiling, and each caller's own `except` is its floor** — worth knowing,
because the two are not the same and the gap between them is where a class goes missing. `Retry`
never sees a fault its caller did not catch, so a word in `RETRYABLE_REASONS` that no call site's
``except`` can reach is a policy nothing implements. **Issue #545 is that sentence happening**: this
module has declared ``timeout`` and ``transport`` retryable since it shipped, and the engine caught
`_TRANSIENT` — which did not name `ProviderConnectionError` — so @glm-5.2's *rerank* would have
retried the read timeout that killed its wake and its *brain* did not. The floor is now the ceiling
on all three call sites, which is what the paragraph above actually asks for.

**The argument that used to sit here was that an aborted wake is *recovered*** — the claim is
two-phase and the router re-wakes (#285), so the peer is answered one wake later, whereas a rerank
has no second chance at all. Both halves are still true and neither is a reason to skip the retry.
Recovery is the floor under a *lost* wake, not a substitute for surviving a blip: the live event
cost ~12 minutes and a second full step budget where a ≤3s wait would have cost nothing, and the
recovery path is not itself free of risk — a compaction inside its window turns a **resume** into an
*abandon, loudly* (#289, #490), which is the peer genuinely dropped. Cheap insurance is still worth
buying when the fallback works.

**And the objection worth answering out loud, because it is the reason to hesitate: a timed-out call
may have run and been billed.** True, and it buys at most one duplicate *generation* — never a
duplicate *act*. The harness's side effects are tool dispatches, and a tool is dispatched only from
a parsed response: a call that never returned dispatched nothing and persisted nothing (the
assistant turn is written after the reply arrives and before any tool runs, `_engine.run`), so a
retry inside `_chat` is invisible to the transcript, to the claim, and to the idempotency ordinal.
The double-spend is real, bounded by the attempt count, and **already paid on every `openai`-SDK
agent in the fleet**, whose SDK has retried `APITimeoutError` internally all along.

What is retried, and what is not
--------------------------------
`RETRYABLE_REASONS` is the whole gate, read off the taxonomy the call sites already speak
(`_rerank._fault_of`, `_retry.fault_slug`, `_engine._reason`): the **runtime** class, and only
its members that a second identical request can actually fix. Deliberately **out**:

- **config-class** (``config:auth``, ``config:billing``, ``config:model_not_found``) — dead until a
  human acts, so a retry spends a call to receive the identical refusal and delays the ERROR that
  pages.
- ``context_length`` and ``payload_too_large`` — deterministic verdicts about *this* request. The
  first has its own, better remedy (compact and re-run); the second can only be fixed by a human
  sending different content.
- ``parse`` — a reply that arrived and parsed badly is not a transport fault. The model answered;
  it answered unusably. (``invalid_response`` *is* retried, and the distinction is real: that one is
  the SDK failing to parse a **truncated or malformed body** — the EOF-mid-JSON class of issue #259
  — where the bytes never arrived whole.)

A timeout is its own class, and it is retried once, with more time
-----------------------------------------------------------------
``timeout`` is in `RETRYABLE_REASONS`, but it is not retried the way the others are (issue #589).
Every other member is a fault a second **identical** request can fix: a 429 clears, a 5xx was the
server's hiccup, a mangled body arrives whole next time, a dropped connection reconnects. A timeout
is the one where the identical request is the one thing guaranteed **not** to help — it waits the
identical time and runs out the identical way. On 2026-09-29 that cost @glm-5.2 three full minutes
per step (a 60 s wall, then 0.5 s and 1 s later the same request into the same wall), four wakes in
a row.

So a timeout earns **at most one** retry (`TIMEOUT_RETRIES`), and that retry runs at
`_timeouts.TIMEOUT_RETRY_SCALE` times the budget: `Retry.timeout_scale` is what the caller binds on
the adapter before the next attempt (`_timeouts.bind_scale`). The line says so — a timeout retry
carries ``timeout_scale=`` and, when the adapter reports what it applied, the budget it ran out of
(``timeout=``) and the one the retry gets (``next_timeout=``). A second timeout is not retried: a
fit that has been exceeded at twice its size is not describing a slow answer.

The budget, and why a cap is not an expectation
-----------------------------------------------
At most `RETRY_ATTEMPTS` retries and at most `RETRY_BUDGET_SECONDS` of **total sleep per call**. The
observed 429s cleared in about a second, so three seconds is the ceiling on a wait that is normally
one — not the cost anyone expects to pay. A vendor's own ``Retry-After`` wins over the schedule and
is then **clamped to what is left of the budget**; a hint larger than the whole budget still retries
rather than giving up, because OpenRouter re-routes on the retry — another pinned upstream can serve
while the limited one recovers, which is the same reasoning that made ``allow_fallbacks`` true
inside the pinned list (issue #468).

A retry is an outcome of the call, never a call
-----------------------------------------------
A failed attempt returned no answer and no usage to price, so it is not a call, carries no cost, and
must join no series that counts calls (a vendor may still bill one that failed after it began
generating, which no ``llm`` line can show: `docs/telemetry.md` → Outcomes). Since issue #782 it is
written on the one ``llm`` head with ``outcome=retrying``, and the final give-up with
``outcome=failed``. The fleet counts a
call as a line on `` llm provider=`` whose ``outcome`` is neither of those (basecradle-noc#1212), so
the split is a verdict the line states, not a spelling that happens to dodge a regex.
`tests/test_retry.py` runs the NOC's expressions against these lines to prove it.

The corollary, which is the one a later "consistency" pass will get backwards: an attempt that the
vendor **answered** still writes its own ``outcome=fallback`` line even when its answer was
unusable. A truncated description was generated and billed (issue #488), and suppressing its line to
reach "one line per describe" would delete real spend from ``helper_cost``. The rule is *one line
per attempt*, and only an answered attempt's line carries tokens and a cost.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from basecradle_harness._exceptions import (
    ProviderAPIError,
    ProviderAuthError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderContextLengthError,
    ProviderPayloadTooLargeError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderToolSchemaError,
    ProviderTruncatedError,
)
from basecradle_harness._faults import routing_refusal
from basecradle_harness._observability import (
    FAILED,
    MAIN,
    _secs,
    describe_provider,
    log_llm_call,
    log_llm_retry,
)
from basecradle_harness._timeouts import TIMEOUT_RETRY_SCALE

#: How many **extra** attempts a transient fault earns: 2 retries, so 3 attempts in all. Small on
#: purpose — the faults this covers clear in about a second or not at all, and a call that has been
#: refused three times inside three seconds is telling us something a fourth will not change.
RETRY_ATTEMPTS = 2

#: The ceiling on **total sleep for one call**, across every retry it makes. A cap, not an expected
#: cost: the live 429s cleared in ~1s. It is a *budget* rather than a per-retry limit because that
#: is the number the agent actually pays — a wake waits for the sum, never for the largest wait.
RETRY_BUDGET_SECONDS = 3.0

#: The wait before the nth retry when the vendor hinted nothing: 1s, then 2s. Their sum is exactly
#: `RETRY_BUDGET_SECONDS`, so the default schedule runs the budget to the edge and never past it.
#: A retry beyond the schedule's length reuses its last entry (the budget stops it regardless).
RETRY_BACKOFF: tuple[float, ...] = (1.0, 2.0)

#: The ``reason=`` values a second identical request can plausibly fix — the gate, spelled as the
#: vocabulary the call sites already log rather than as a tuple of exception classes, because the
#: taxonomy is what the fleet's dashboards read and an exception class is not.
#:
#: Pinned against **all three** producers by test (`tests/test_retry.py`): a reason a producer stops
#: emitting, or starts emitting for a different class, must not quietly leave this set describing a
#: policy nothing implements — and a producer missing from that pin is how issue #545 hid, since the
#: engine was not in it and its ``except`` could reach neither of the transport words.
RETRYABLE_REASONS = frozenset(
    {"rate_limited", "server_error", "transport", "timeout", "invalid_response", "truncated"}
)


#: How many retries a **timeout** earns, whatever the call's attempt count (issue #589): one, at
#: `_timeouts.TIMEOUT_RETRY_SCALE` times the budget. See the module docstring for why a timeout is
#: never re-sent into the same wall.
TIMEOUT_RETRIES = 1

#: How far `connection_reason` walks an exception's cause chain. An adapter chains what its SDK
#: raised, and an SDK chains what *its* transport raised, so the fact being looked for is one or two
#: hops down — never deeper in practice. The bound exists because a chain can be circular, and a log
#: word is not worth an infinite loop.
_CAUSE_DEPTH = 5

#: What a class is called when it means *this call ran out of time*. Matched as a **suffix** against
#: every class in an exception's MRO, which is the only test that works across the families
#: involved: the stdlib spells it ``TimeoutError``, both HTTPX distributions spell it
#: ``TimeoutException``, the `openai` SDK's own wrapper is ``APITimeoutError``, and ``requests``
#: spells its base class bare. Matching the MRO rather than the leaf is what makes a concrete
#: ``ReadTimeout`` answer through whichever of these its family derives from. See `connection_reason`
#: for why this is a name test rather than an `isinstance` against imported classes.
_TIMEOUT_NAMES = ("Timeout", "TimeoutError", "TimeoutException")


def _is_timeout(exc: BaseException) -> bool:
    """Does any class this exception inherits from name itself a timeout?"""
    return any(base.__name__.endswith(_TIMEOUT_NAMES) for base in type(exc).__mro__)


def connection_reason(exc: object) -> str:
    """``timeout`` or ``transport`` — the taxonomy's word for one `ProviderConnectionError`.

    **It lives here, alone, because it is the one fault whose word is not fixed by its class.**
    Every other member of the taxonomy is answered by an ``isinstance`` the producers can each spell
    for themselves; this one has to be read off the cause, and the moment a second call site spells
    that read, a third answer appears. It did: `_rerank._fault_of` split the class while
    `_describer._fault_of` returned ``transport`` flat, so the *same* read timeout was ``timeout`` in
    one line of the journal and ``transport`` in the next — and `tests/test_retry.py` carried a
    comment tolerating the drift, which is a comment standing in for a rule. One spelling, three
    readers, for exactly the reason `diagnostics` is one function: so they cannot disagree about what
    a failure looked like.

    **The distinction it draws is the one the adapters erase, and it is not cosmetic.** An adapter
    maps DNS, TCP, TLS, connect timeouts, read timeouts and mid-body transport failures onto the one
    `ProviderConnectionError`, under a comment that read *"nothing reached the model"* — true of the
    connect half and **false of the read half**. A read timeout means the request was accepted and
    the answer did not come back in time: the model may well have run, and may well have been
    billed. Both are worth retrying and neither can double-*act* (a call that never returned
    dispatched no tools and persisted nothing), but a journal that cannot tell them apart cannot
    tell a network that is down from a vendor that is slow — which was the first question asked of
    the wake that opened issue #545.

    **The cause is *walked*, not read one level deep, because an SDK is a layer.** The `openrouter`
    adapter chains the raw transport failure; the `openai` adapter chains its SDK's
    ``APITimeoutError``, whose *own* cause is the transport one. A read at depth 1 would answer
    correctly on one provider and wrongly on the other — the same decided-by-nobody asymmetry
    between adapters that issue #545 exists to remove, reappearing inside the fix for it.

    **And it is a *name* test rather than an ``isinstance``, which is deliberate and was learned the
    hard way.** The obvious spelling — ``isinstance(cause, httpx.TimeoutException)``, which is what
    `_rerank._fault_of` had — is **wrong for the `openai` path**, because since its 3.0 that SDK
    runs on **HTTPX2**: `httpx2.ReadTimeout` is a different distribution's class and no subclass of
    `httpx`'s, so a real OpenAI read timeout read ``transport``. Nothing errored, no test failed
    (the first draft's double chained an `httpx` exception, which only ever proved the assumption),
    and the word was wrong on one provider and right on another — this fix reproducing its own
    defect. Importing every family instead would put a list of HTTP client packages in the core, one
    of which is an optional extra, and a fourth family would fall out of it silently. A class that
    calls itself a timeout is the fact all of them share.

    **A `ProviderTimeoutError` answers first, and the walk is now the fallback** (issue #589). Since
    a timeout is retried differently from a transport failure — once, with more time, rather than
    as it was — the word is no longer only a log word, and a misread now costs a retry of the wrong
    shape. So every shipped adapter types its timeouts at the boundary that can see them, and the
    walk remains for an adapter that only ever raises the parent class.

    For such an adapter, anything else reads as ``transport``: the fallback is the broader word,
    retried as it always was. One known imprecision, stated rather than left to be discovered: a
    class deliberately named so as to end in ``Timeout`` without being one would read ``timeout``
    (no library names an exception that way; a test written to be adversarial does). The native xAI
    ``DEADLINE_EXCEEDED`` that used to read ``transport`` is typed by its adapter now.
    """
    if isinstance(exc, ProviderTimeoutError):
        # The adapter knew, and said so with the type (issue #589) — the answer that needs no
        # inference. It is also the only answer the native xAI path can give: a gRPC
        # ``DEADLINE_EXCEEDED`` names no timeout class, so the cause walk below reads it
        # ``transport`` and a deadline would be retried as if the network had dropped.
        return "timeout"
    seen = exc
    for _ in range(_CAUSE_DEPTH):
        cause = getattr(seen, "__cause__", None)
        if cause is None:
            break
        if isinstance(cause, BaseException) and _is_timeout(cause):
            return "timeout"
        seen = cause
    return "transport"


def retryable(reason: str | None, *, is_config: bool = False) -> bool:
    """Is this fault worth one more identical request?

    Two conditions, and the config one is not redundant: the ``config:`` prefix is the class the
    reranker and describer *log* with, and a future member of that class must never become
    retryable by accident just because someone added its bare word to `RETRYABLE_REASONS`.
    """
    return not is_config and reason in RETRYABLE_REASONS


def diagnostics(exc: object) -> dict[str, Any]:
    """The vendor's own account of one failed attempt, as log fields (issue #506).

    Read as **attributes**, never by re-parsing JSON at three call sites: the OpenRouter adapter
    parses its own error body once, at the boundary of the vendor that wrote it
    (`basecradle_harness._openrouter._diagnostics`), and every other adapter simply carries
    ``None``s — which `kv` omits, so a non-OpenRouter line is byte-identical to what it always was.

    Why these four fields and not the error text alone: every one of the four live fallbacks logged
    ``detail="OpenRouter rate-limited the request (HTTP 429)."`` — the adapter's own fixed sentence,
    which names nothing. A 429 on a *paid* model is the **upstream** limiting or at capacity, and
    OpenRouter excludes 429s from its published provider-uptime statistics, so this line is the only
    place the offending endpoint can ever be seen. Without it nobody can say which of
    ``baseten, coreweave, parasail, modal`` refused.

    Returned as a dict so the same fields, under the same names and in the same order, ride
    **both** the retry line and the final ``outcome=failed`` line — one function, so the two can
    never disagree about what a failure looked like.

    The fifth, ``generation_id``, is the vendor's own id for the failed attempt (issue #634), which
    the adapter that saw the response stamped on the error (`ProviderError.generation_id`). Both
    emitters render it last on the line, wherever it sits in this dict.
    """
    hinted = _retry_after(exc)
    return {
        "retry_after": None if hinted is None else _secs(hinted),
        "provider_code": getattr(exc, "provider_code", None),
        # OpenRouter's **own** attempt counter (``openrouter_metadata.attempt``): how far its router
        # got, where ``0`` means it reached no provider at all. Deliberately not spelled ``attempt=``
        # — that field on the retry line is *the harness's* attempt of this call, and two different
        # counters under one key is a dashboard nobody can read.
        "routing_attempt": getattr(exc, "routing_attempt", None),
        "attempts": getattr(exc, "routing_attempts", None),
        "generation_id": getattr(exc, "generation_id", None),
    }


class Retry:
    """The bounded retry of one model call: the decision, the line, and the wait.

    Held by the caller for the life of **one** call — a rerank, a describe, one engine step — so
    "how many attempts have I made" and "how much of the budget is spent" need no clock and no
    shared state. Each call site keeps its own control flow (they are genuinely different: the
    reranker reports and returns ``[]``, the describer falls back to a withheld caption, the engine
    re-raises), and shares the only two things that must not drift — *what is retried*, and *what a
    retry says in the journal*.

    Args:
        provider: The endpoint vendor, for the line's ``provider=``.
        model: The model id the call was made against.
        purpose: ``main`` / ``memory`` / ``helper`` — the same category its ``llm`` line carries.
        kind: The job within that category (``rerank``, ``image.describe``), where there is one.
        attempts: Total attempts including the first. The engine passes its own
            ``HARNESS_RESPONSE_RETRIES``-derived count; nothing else takes an override, because a
            second env knob for the same axis is one way to disagree with yourself.
        budget: Total seconds this call may spend sleeping, across every retry.
        backoff: ``(retry number, reason) -> seconds`` when the vendor hinted nothing. The engine
            supplies its own so its long-standing 0.5s/1s schedule for a 5xx or an unparseable body
            is untouched by this change; everything else takes `RETRY_BACKOFF`.
        sleep: Injectable, so a test proves the schedule without waiting it.
        extra: The call site's own trailing fields (``surface=`` for a rerank, ``subject=`` for a
            describe), so a retry line is greppable beside the ``llm`` line it belongs to.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        purpose: str,
        kind: str | None = None,
        attempts: int = RETRY_ATTEMPTS + 1,
        budget: float = RETRY_BUDGET_SECONDS,
        backoff: Callable[[int, str], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.purpose = purpose
        self.kind = kind
        self.attempts = max(1, attempts)
        self.budget = max(0.0, budget)
        self._backoff = backoff or _default_backoff
        self._sleep = sleep or time.sleep
        self._extra = dict(extra or {})
        #: Attempts **made** so far — incremented by `again`, so the caller never counts twice.
        self.attempt = 0
        #: Seconds actually slept so far. The budget is read off this, never off a wall clock: the
        #: thing being bounded is the wait this code chose, not how slow the vendor was.
        self.slept = 0.0
        #: How much of its fitted budget the **next** attempt gets (issue #589). ``1.0`` until a
        #: timeout is retried, then `_timeouts.TIMEOUT_RETRY_SCALE` for every attempt after it —
        #: more time never hurts an attempt that failed for another reason. The caller binds it on
        #: the adapter before each attempt (`_timeouts.bind_scale`); `Retry` never reaches the
        #: adapter itself, because what a caller calls is the caller's to know.
        self.timeout_scale = 1.0
        #: Timeout retries made so far — capped at `TIMEOUT_RETRIES` whatever `attempts` allows.
        self._timeout_retries = 0

    @property
    def label(self) -> str:
        """The attempts made so far, as the line's ``attempt=`` renders them: ``n/total``."""
        return f"{self.attempt}/{self.attempts}"

    @property
    def retried(self) -> bool:
        """Did this call take more than one attempt?

        Read by a caller deciding **whose** duration its line should carry. An adapter measures the
        one HTTP call it made, which is the better number right up until there were refused attempts
        and sleeps it knows nothing about — at which point the honest answer is the caller's own
        wall clock. One predicate, so the two cases are chosen rather than blended.
        """
        return self.attempt > 1

    def again(
        self,
        exc: object,
        *,
        reason: str | None,
        is_config: bool = False,
        timed_out_after: float | None = None,
    ) -> bool:
        """Record this attempt's failure; wait and answer ``True`` when it is worth another.

        ``False`` — do not retry — is the answer for a fault outside `RETRYABLE_REASONS`, for a
        config-class fault, for a retryable one that has run out of attempts, and for a **second**
        timeout (`TIMEOUT_RETRIES`). **Nothing is logged in that case**: the caller is about to
        write the final line, and a retry line for a retry that never happened would put a WARNING
        in the journal for an event that did not occur.

        A timeout that *is* retried raises `timeout_scale` for the attempts after it (issue #589).
        ``timed_out_after`` is the generation budget the adapter applied to the attempt that timed
        out (`_timeouts.last_timeout`), when it says; with it the line names both the budget that
        ran out and the one the retry gets, so "retried with a larger budget" is read off the
        journal rather than inferred from it.
        """
        self.attempt += 1
        if not retryable(reason, is_config=is_config) or self.attempt >= self.attempts:
            return False
        timing: dict[str, Any] = {}
        if getattr(exc, "vendor_deadline", False) is True:
            # The deadline that ran out was the vendor's own (a gateway's 504), not the budget the
            # harness fitted, so that budget is not the one the attempt ran out of (issue #759).
            timed_out_after = None
        if reason == "timeout":
            if self._timeout_retries >= TIMEOUT_RETRIES:
                return False
            self._timeout_retries += 1
            self.timeout_scale = TIMEOUT_RETRY_SCALE
            timing = {
                "timeout": None if timed_out_after is None else _secs(timed_out_after),
                "timeout_scale": _scale(TIMEOUT_RETRY_SCALE),
                "next_timeout": (
                    None
                    if timed_out_after is None
                    else _secs(timed_out_after * TIMEOUT_RETRY_SCALE)
                ),
            }
        wait = self._wait_for(exc, reason or "")
        log_llm_retry(
            provider=self.provider,
            model=self.model,
            purpose=self.purpose,
            kind=self.kind,
            attempt=self.label,
            reason=reason,
            next_in=wait,
            diagnostics=diagnostics(exc),
            extra={**timing, **self._extra},
        )
        self.slept += wait
        if wait > 0:
            self._sleep(wait)
        return True

    def _wait_for(self, exc: object, reason: str) -> float:
        """How long to wait before the next attempt — the vendor's hint, clamped to what is left.

        A hint **larger than the whole budget** still yields a retry (at whatever time remains,
        possibly none): OpenRouter re-routes on the re-issued request, so the wait a limited
        upstream asked for is not the wait a *different* pinned upstream needs. Giving up there
        would hand one vendor's bad minute the power to defeat the feature — the issue #468 lesson,
        one layer up.
        """
        remaining = max(0.0, self.budget - self.slept)
        hinted = _retry_after(exc)
        wait = self._backoff(self.attempt, reason) if hinted is None else hinted
        return min(max(0.0, float(wait)), remaining)


def _scale(value: float) -> str:
    """A budget multiplier as the line prints it: ``2``, not ``2.0`` — a factor, not a duration."""
    return f"{value:g}"


def _default_backoff(attempt: int, reason: str) -> float:
    """`RETRY_BACKOFF` indexed by retry number, holding at its last entry past the end.

    ``reason`` is unread here and is part of the signature anyway: the engine's own backoff keys on
    it (a 429 takes this schedule, a 5xx keeps its long-standing 0.5s/1s one), and one callable
    shape for both is what keeps that difference at the call site rather than inside `Retry`.
    """
    return RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)) - 1]


def _retry_after(exc: object) -> float | None:
    """The vendor's ``Retry-After`` in seconds, when it stated a usable one.

    Defensive about the type because it is the vendor's number reaching us through an SDK: a bool is
    an ``int`` in Python and would sleep for a second on ``True``, and a zero or negative hint is
    *no* hint rather than an instruction to retry with no wait at all — which would be a hint-shaped
    way to lose the backoff entirely.
    """
    hinted = getattr(exc, "retry_after", None)
    if isinstance(hinted, bool) or not isinstance(hinted, (int, float)):
        return None
    return float(hinted) if hinted > 0 else None


def fault_slug(exc: BaseException) -> str:
    """A provider fault → the ``reason=`` its `llm` line carries, its class by the ``config:`` prefix.

    The describer's and, since issue #782, the brain's: a brain call that ends ``outcome=failed``
    names its fault in the same words a describe does. The same taxonomy `_rerank._fault_of` draws
    (which keeps its own for its unknown-model read), and for the same reason: **config-class** is *dead
    until a human acts* — a rejected key, an unfunded account, a model id that does not exist —
    while **runtime-class** is everything that can succeed next time unchanged. A 402 sits with the
    bad key rather than beside the 429 above it precisely because a rate limit heals with time and
    an empty account heals only when somebody puts money in it.

    OpenRouter's two routing refusals (issue #651) are config-class as well, and are named for their
    cause rather than filed with a missing model: both arrive as a 404, and an ``AI_BASE_URL`` on a
    regional host is where they come from — no endpoint for the describer model in the region, or
    none the ``HARNESS_DESCRIBER_PROVIDERS`` list permits there.

    A generic 4xx is runtime-class deliberately: it is far more likely a fixable harness/config
    defect than a permanent property of the pool, and reporting it as config-class would page a
    human for something the next release fixes.

    Ordered most-specific first, because these classes subclass one another.
    """
    if isinstance(exc, ProviderToolSchemaError):
        # A vendor refused one tool's schema and the adapter could not recover by dropping it.
        return "tool_schema"
    if isinstance(exc, ProviderAuthError):
        return "config:auth"
    if isinstance(exc, ProviderBillingError):
        return "config:billing"
    if isinstance(exc, ProviderRateLimitError):
        return "rate_limited"
    if isinstance(exc, ProviderServerError):
        return "server_error"
    if isinstance(exc, ProviderContextLengthError):
        return "context_length"
    if isinstance(exc, ProviderPayloadTooLargeError):
        return "payload_too_large"
    if isinstance(exc, ProviderTruncatedError):
        return "truncated"
    if isinstance(exc, ProviderResponseError):
        return "invalid_response"
    if isinstance(exc, ProviderConnectionError):
        # ``timeout`` or ``transport``, off the cause — the shared read, not a flat word. This
        # returned ``transport`` unconditionally until issue #545, so a describe that timed out and
        # a rerank that timed out named the same fault differently on the same box.
        return connection_reason(exc)
    if isinstance(exc, ProviderAPIError):
        refusal = routing_refusal(f"{exc} {getattr(exc, 'body', '') or ''}")
        if refusal:
            return f"config:{refusal}"
        return "config:model_not_found" if getattr(exc, "status_code", None) == 404 else "api_error"
    return "provider_error"


def log_brain_failure(
    provider: object,
    exc: BaseException,
    *,
    attempt: str | None,
    seconds: float | None,
    level: int = logging.WARNING,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """The `llm` line, ``outcome=failed``, for a brain call that ended with no answer (issue #782).

    A brain call that raised used to write no line at all: the wake's failure lines, or a prose
    ERROR, were its only record, so "where did this wake die, and of what" was read off prose, and
    a producer with no wake had nothing to compare it with. This is the brain's one final line, in
    the same words a describe's failure uses (`fault_slug`) and with the vendor's own account of
    the fault (`diagnostics`). It carries no tokens and no cost: the attempt returned no usage to
    price.

    ``level`` is the caller's, because severity is a page: the engine's transient give-up keeps
    the ERROR its prose line always had, and every other failure is WARNING, since the wake's own
    lines already carry the verdict a human is paged on.
    """
    name, model = describe_provider(provider)
    log_llm_call(
        provider=name,
        model=model,
        seconds=seconds,
        purpose=MAIN,
        outcome=FAILED,
        attempt=attempt,
        reason=fault_slug(exc),
        detail=str(exc),
        extra={**diagnostics(exc), **(extra or {})},
        level=level,
    )
