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
(`_rerank._fault_of`, `_describer._fault_of`, `_engine._reason`): the **runtime** class, and only
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

The budget, and why a cap is not an expectation
-----------------------------------------------
At most `RETRY_ATTEMPTS` retries and at most `RETRY_BUDGET_SECONDS` of **total sleep per call**. The
observed 429s cleared in about a second, so three seconds is the ceiling on a wait that is normally
one — not the cost anyone expects to pay. A vendor's own ``Retry-After`` wins over the schedule and
is then **clamped to what is left of the budget**; a hint larger than the whole budget still retries
rather than giving up, because OpenRouter re-routes on the retry — another pinned upstream can serve
while the limited one recovers, which is the same reasoning that made ``allow_fallbacks`` true
inside the pinned list (issue #468).

The line a retry writes is **not** an ``llm`` line
---------------------------------------------------
A failed attempt that never reached a model generated nothing and was billed nothing, so it is not a
call and must join no series that counts calls. The fleet's dashboard keys its whole model-call
family on the literal `` llm provider=`` (`basecradle-noc` ``observability/ai-box.json``:
``llm_calls``, ``memory_calls``, ``helper_calls`` and every cost/duration/outcome column hanging off
them), so a retry line matching that head would inflate *Calls by Model*, drag failed attempts into
the duration average, and put a NULL outcome on the rerank-outcomes chart.
`basecradle_harness._observability.RETRY_HEAD` is
``llm retry``, which does not contain `` llm provider=`` — the space-then-``provider=`` immediately
after ``llm`` is the discriminator — and `tests/test_retry.py` runs the NOC's own expression against
both lines to prove it, because a head that *looks* safe is not the same as one that has been
checked against the regex that reads it.

The corollary, which is the one a later "consistency" pass will get backwards: an attempt that the
vendor **answered** still writes its own ``llm`` line even when its answer was unusable — a
truncated description was generated and billed (issue #488), and suppressing its line to reach "one
line per describe" would delete real spend from ``helper_cost``. The rule is *a line per billed
call*, and a 429 is not one.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from basecradle_harness._observability import _secs, log_llm_retry

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
    {"rate_limited", "server_error", "transport", "timeout", "invalid_response"}
)


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

    Anything else reads as ``transport``: the fallback is the broader, safer word, and both are in
    `RETRYABLE_REASONS`, so a misread in **either** direction costs a less precise line and never a
    lost retry — which is what makes a suffix match the right precision for this question. Two known
    imprecisions, stated rather than left to be discovered: the native xAI gRPC path's
    ``DEADLINE_EXCEEDED`` names no timeout class at all and reads ``transport``, and a class
    deliberately named so as to end in ``Timeout`` without being one would read ``timeout`` (no
    library names an exception that way; a test written to be adversarial does).
    """
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

    Returned as a dict so the same four fields, under the same names and in the same order, ride
    **both** the retry line and the final ``outcome=fallback`` line — one function, so the two can
    never disagree about what a failure looked like.
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

    @property
    def retried(self) -> bool:
        """Did this call take more than one attempt?

        Read by a caller deciding **whose** duration its line should carry. An adapter measures the
        one HTTP call it made, which is the better number right up until there were refused attempts
        and sleeps it knows nothing about — at which point the honest answer is the caller's own
        wall clock. One predicate, so the two cases are chosen rather than blended.
        """
        return self.attempt > 1

    def again(self, exc: object, *, reason: str | None, is_config: bool = False) -> bool:
        """Record this attempt's failure; wait and answer ``True`` when it is worth another.

        ``False`` — do not retry — is the answer for a fault outside `RETRYABLE_REASONS`, for a
        config-class fault, and for a retryable one that has run out of attempts. **Nothing is
        logged in that case**: the caller is about to write the final line, and a retry line for a
        retry that never happened would put a WARNING in the journal for an event that did not
        occur.
        """
        self.attempt += 1
        if not retryable(reason, is_config=is_config) or self.attempt >= self.attempts:
            return False
        wait = self._wait_for(exc, reason or "")
        log_llm_retry(
            provider=self.provider,
            model=self.model,
            purpose=self.purpose,
            kind=self.kind,
            attempt=f"{self.attempt}/{self.attempts}",
            reason=reason,
            next_in=wait,
            diagnostics=diagnostics(exc),
            extra=self._extra,
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
