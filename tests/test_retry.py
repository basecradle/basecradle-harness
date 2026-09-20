"""The shared bounded retry (issue #506) — the policy, the budget, and the head that must not match.

Three things are pinned here and each has an "obviously fine" broken form.

**The gate is the reason vocabulary, and it is pinned against its producers.** `RETRYABLE_REASONS`
is a set of words, not of exception classes, because the words are what the fleet's dashboards read.
A set that drifted from `_rerank._fault_of` / `_describer._fault_of` would describe a policy nothing
implements — retrying nothing, or retrying a config fault that is dead until a human acts — and no
test anywhere else would fail.

**The budget is a total, not a per-wait cap.** What the agent pays is the sum, so that is what is
bounded; a per-retry limit with an unbounded count is the shape this replaced.

**The retry line must not be read as a call.** The fleet keys its whole model-call family on the
literal `` llm provider=``, so the NOC's own expressions are run here against real lines from the
real emitters. A head that *looks* safe is not the same as one checked against the regex that reads
it — this repo has already paid for that distinction twice (issues #414, #504).
"""

from __future__ import annotations

import logging
import re

import httpx
import pytest

from basecradle_harness._describer import _fault_of as describer_fault_of
from basecradle_harness._engine import _reason as engine_reason
from basecradle_harness._exceptions import (
    ProviderAuthError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderContextLengthError,
    ProviderPayloadTooLargeError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
)
from basecradle_harness._observability import (
    HELPER,
    LOG_FORMAT,
    MAIN,
    MEMORY,
    RETRY_HEAD,
    log_llm_call,
)
from basecradle_harness._rerank import RERANK_KIND
from basecradle_harness._rerank import _fault_of as rerank_fault_of
from basecradle_harness._retry import (
    RETRY_ATTEMPTS,
    RETRY_BACKOFF,
    RETRY_BUDGET_SECONDS,
    RETRYABLE_REASONS,
    Retry,
    connection_reason,
    diagnostics,
    retryable,
)

#: The NOC's model-call gate, byte-for-byte as `basecradle-noc` ``observability/ai-box.json`` spells
#: it. Every column in the family is anchored on it — ``llm_calls`` (and therefore the *Log
#: Extraction Broken* alert's denominator), ``llm_purpose``, ``llm_kind``, ``model``, ``endpoint``,
#: ``llm_duration_s``, and the ``memory_*`` / ``helper_*`` / ``main_*`` families that AND it with a
#: ``purpose=``. ClickHouse's ``match()`` is a partial match, which is what `re.search` is.
LLM_CALL_GATE = r" llm provider="
PURPOSE_MEMORY = r"purpose=memory"
PURPOSE_HELPER = r"purpose=helper"

#: The other expressions a new line class can move by accident, each ungated or gated on something
#: a retry line might plausibly grow into. A retry line must move **none** of them.
MEMORY_OUTCOME = r"outcome=(ok|fallback)"
MEMORY_DURATION = r"duration=([0-9.]+)s"
RERANK_ON = r" rerank=on"
STAGE_LABEL = r"stage=([a-z_]+)"
SOURCE_LABEL = r"source=([A-Za-z0-9_-]+)"
WAKE_OUTCOME_GATE = r"stage=|event=wake_end|wake end| tool name="


def record(level: int, message: str) -> str:
    """One line as journald receives it — the severity token the fleet's ``level`` column reads.

    The NOC matches over the whole message, so a test that matched a bare ``message`` would be
    asking a different question than the dashboard asks.
    """
    return LOG_FORMAT % {"levelname": logging.getLevelName(level), "message": message}


def lines(caplog) -> list[str]:
    return [record(r.levelno, r.getMessage()) for r in caplog.records]


def retry_line(caplog) -> str:
    return next(line for line in lines(caplog) if RETRY_HEAD in line)


def rerank_llm_line(caplog) -> str:
    return next(line for line in lines(caplog) if re.search(LLM_CALL_GATE, line))


def _no_sleep():
    delays: list[float] = []
    return delays, delays.append


def rate_limit(retry_after=None, **diagnostics_kwargs):
    return ProviderRateLimitError(
        "OpenRouter rate-limited the request (HTTP 429).",
        status_code=429,
        retry_after=retry_after,
        **diagnostics_kwargs,
    )


def retry(**kwargs):
    kwargs.setdefault("provider", "openrouter")
    kwargs.setdefault("model", "z-ai/glm-5.3-flash")
    kwargs.setdefault("purpose", MEMORY)
    kwargs.setdefault("kind", RERANK_KIND)
    kwargs.setdefault("extra", {"surface": "turn0"})
    return Retry(**kwargs)


# === The gate: what is retried, and what is never ============================


def _chained(exc: ProviderConnectionError, cause: BaseException) -> ProviderConnectionError:
    """A transport error carrying the cause its adapter chained — the shape `connection_reason` reads."""
    exc.__cause__ = cause
    return exc


@pytest.mark.parametrize(
    ("exc", "reason"),
    [
        (rate_limit(), "rate_limited"),
        (ProviderServerError("oops", status_code=503), "server_error"),
        (ProviderResponseError("EOF while parsing a value"), "invalid_response"),
        (ProviderConnectionError("no route"), "transport"),
        (
            _chained(ProviderConnectionError("timed out"), httpx.ReadTimeout("read")),
            "timeout",
        ),
    ],
)
def test_every_retryable_reason_is_one_a_producer_actually_emits(exc, reason):
    """The set is pinned against the taxonomy it gates, in **all three** spellings of it.

    `RETRYABLE_REASONS` is words, and words drift. A reason a `_fault_of` stopped emitting would
    leave this policy describing retries that can never happen — a dead mechanism that fails no
    test, which is this repo's Green-While-Absent shape one level down.

    **The engine is in this list now, and that is the whole of issue #545.** It was absent because
    it had no producer to pin: it caught three exception classes and `ProviderConnectionError` was
    not one, so ``timeout`` and ``transport`` sat in the gate above with nothing on the brain path
    able to reach them — the set describing a policy two call sites implemented and the third could
    not. A word is only retryable if some ``except`` can hand it over.
    """
    assert rerank_fault_of(exc) == (reason, False)
    assert describer_fault_of(exc) == reason
    assert engine_reason(exc) == reason
    assert reason in RETRYABLE_REASONS
    assert retryable(reason)


def test_one_fault_gets_one_word_however_deep_its_adapter_buried_the_cause():
    """The adapters chain at different depths; the word must not.

    `openrouter` chains the raw transport failure, `openai` chains its SDK's own wrapper around one.
    A read at depth 1 answers correctly on one provider and calls the other a ``transport`` fault —
    the decided-by-nobody asymmetry between adapters that issue #545 exists to remove, reappearing
    inside the fix for it. (The real chains are pinned against the SDKs in `test_provider.py`.)
    """
    wrapped = RuntimeError("the SDK's own wrapper, which names no timeout")
    wrapped.__cause__ = httpx.ReadTimeout("The read operation timed out")

    assert connection_reason(_chained(ProviderConnectionError("x"), wrapped)) == "timeout"
    assert (
        connection_reason(_chained(ProviderConnectionError("x"), httpx.ReadTimeout("y")))
        == "timeout"
    )


def test_a_timeout_is_recognized_by_what_it_calls_itself_not_by_which_package_defines_it():
    """The bug the first draft of issue #545 shipped, pinned so it cannot come back.

    ``isinstance(cause, httpx.TimeoutException)`` is the obvious spelling and it is **wrong for the
    `openai` path**: since its 3.0 that SDK runs on HTTPX2, whose `ReadTimeout` is a different
    distribution's class and no subclass of `httpx`'s. A real OpenAI read timeout therefore logged
    ``reason=transport`` with nothing failing anywhere. Importing every family instead would put a
    list of HTTP client packages in the core, one of them an optional extra, and a fourth family
    would drop out of it silently — so the test is what a timeout **calls itself**.
    """

    class Timeout(Exception):
        """A fourth family's base class, spelled bare — the way ``requests`` spells its own."""

    class ReadTimeout(Timeout):
        """Its concrete leaf, which is what an adapter would actually chain."""

    class ConnectionRefused(Exception):
        """A transport failure that is not a timeout — the other half of the distinction."""

    assert connection_reason(_chained(ProviderConnectionError("x"), ReadTimeout())) == "timeout"
    assert connection_reason(_chained(ProviderConnectionError("x"), TimeoutError())) == "timeout"
    assert (
        connection_reason(_chained(ProviderConnectionError("x"), ConnectionRefused()))
        == "transport"
    )


def test_a_cause_chain_that_says_nothing_reads_as_transport_and_always_terminates():
    """The fallback is the broader, safer word — and both are retryable, so a misread costs a less
    precise line and never a lost retry. The walk is bounded because a cause chain can be circular,
    and a log word is not worth an infinite loop (the native xAI gRPC path has no ``httpx`` in its
    chain at all, which is the one stated case that reads ``transport`` by omission)."""
    assert connection_reason(ProviderConnectionError("no cause")) == "transport"
    assert (
        connection_reason(_chained(ProviderConnectionError("x"), OSError("no route")))
        == "transport"
    )

    loop = ProviderConnectionError("a")
    other = ProviderConnectionError("b")
    loop.__cause__ = other
    other.__cause__ = loop

    assert connection_reason(loop) == "transport"  # terminates rather than spinning


def test_the_reason_set_is_exactly_the_runtime_faults_a_second_request_can_fix():
    """Named one by one, because each exclusion is a decision rather than an omission."""
    assert RETRYABLE_REASONS == {
        "rate_limited",
        "server_error",
        "transport",
        "timeout",
        "invalid_response",
    }


@pytest.mark.parametrize(
    "exc",
    [
        ProviderAuthError("bad key", status_code=401),
        ProviderBillingError("no credit", status_code=402),
        ProviderContextLengthError("too long", status_code=400),
        ProviderPayloadTooLargeError("too big", status_code=413),
    ],
)
def test_a_fault_a_retry_cannot_fix_is_never_retried(exc):
    """Config-class is dead until a human acts; the deterministic pair fails identically forever.

    Both would spend a call to receive the identical refusal — and the config half would *delay the
    ERROR that pages* while doing it, which is worse than useless.
    """
    reason, is_config = rerank_fault_of(exc)
    delays, spy = _no_sleep()

    assert retry(sleep=spy).again(exc, reason=reason, is_config=is_config) is False
    assert delays == []


def test_a_parse_failure_is_not_a_transport_fault():
    """The model answered; it answered unusably. Asking again buys the same unusable answer.

    The distinction from ``invalid_response`` is real and is the reason both words exist: that one
    is the SDK failing on a body that never arrived whole.
    """
    assert "parse" not in RETRYABLE_REASONS
    assert retryable("parse") is False
    assert retryable("invalid_response") is True


def test_a_config_prefixed_reason_is_refused_even_if_someone_adds_its_bare_word():
    """The `is_config` flag is not redundant with the set — it is the second lock on the same door."""
    assert retryable("rate_limited", is_config=True) is False


# === The budget ==============================================================


def test_the_default_schedule_spends_the_budget_and_never_more():
    """Two retries, 1s then 2s — exactly `RETRY_BUDGET_SECONDS`, by construction not by luck."""
    assert sum(RETRY_BACKOFF) == RETRY_BUDGET_SECONDS
    assert len(RETRY_BACKOFF) == RETRY_ATTEMPTS

    delays, spy = _no_sleep()
    bounded = retry(sleep=spy)

    assert bounded.again(rate_limit(), reason="rate_limited") is True
    assert bounded.again(rate_limit(), reason="rate_limited") is True
    assert bounded.again(rate_limit(), reason="rate_limited") is False  # attempts exhausted
    assert delays == [1.0, 2.0]
    assert sum(delays) <= RETRY_BUDGET_SECONDS


def test_the_vendors_retry_after_wins_over_the_schedule():
    delays, spy = _no_sleep()

    assert retry(sleep=spy).again(rate_limit(retry_after=0.25), reason="rate_limited") is True
    assert delays == [0.25]


def test_a_retry_after_past_the_budget_is_clamped_and_still_retried():
    """Giving up there would let one vendor's bad minute defeat the whole call.

    OpenRouter re-routes on the re-issued request, so the wait a limited upstream asked for is not
    the wait a *different* pinned upstream needs — the issue #468 lesson, one layer up.
    """
    delays, spy = _no_sleep()
    bounded = retry(sleep=spy)

    assert bounded.again(rate_limit(retry_after=30), reason="rate_limited") is True
    assert delays == [RETRY_BUDGET_SECONDS]
    # The budget is spent, so the second retry is issued immediately rather than not at all: the
    # attempt count is what bounds the loop, and the wait is what the budget bounds.
    assert bounded.again(rate_limit(retry_after=30), reason="rate_limited") is True
    assert delays == [RETRY_BUDGET_SECONDS]  # a zero wait is not slept at all


@pytest.mark.parametrize("hint", [0, -5, True, "soon", None])
def test_an_unusable_retry_after_falls_back_to_the_schedule(hint):
    """A vendor's number reaches us through an SDK, so it is checked rather than trusted.

    ``True`` is the one worth naming: a bool is an ``int`` in Python and would have slept for a
    second on a value that is not a duration at all.
    """
    delays, spy = _no_sleep()

    assert retry(sleep=spy).again(rate_limit(retry_after=hint), reason="rate_limited") is True
    assert delays == [RETRY_BACKOFF[0]]


def test_a_retry_that_never_happened_writes_no_line(caplog):
    """A WARNING for an event that did not occur is a journal that lies about its own quiet."""
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        retry(sleep=lambda _s: None).again(
            ProviderAuthError("bad key", status_code=401), reason="config:auth", is_config=True
        )

    assert lines(caplog) == []


# === The diagnostics =========================================================


def test_the_vendors_own_account_rides_every_failed_attempt(caplog):
    """Without it a fallback says only what the adapter's fixed sentence says, which names nobody.

    A 429 on a paid model is the **upstream** refusing, and OpenRouter excludes 429s from its
    published uptime statistics — so this line is the only place the offending endpoint can ever be
    seen.
    """
    exc = rate_limit(
        retry_after=1,
        provider_code="429",
        routing_attempt=2,
        routing_attempts="Parasail:429,CoreWeave:429",
    )
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        retry(sleep=lambda _s: None).again(exc, reason="rate_limited")

    line = retry_line(caplog)
    for field in (
        "attempt=1/3",
        "reason=rate_limited",
        "retry_after=1.00s",
        "provider_code=429",
        "routing_attempt=2",
        "attempts=Parasail:429,CoreWeave:429",
        "next_in=1.00s",
        "surface=turn0",
    ):
        assert field in line, line


def test_an_adapter_with_nothing_to_say_carries_nothing(caplog):
    """Every other adapter reports ``None``s, which `kv` omits — the line is what it always was."""
    assert diagnostics(ProviderServerError("oops", status_code=503)) == {
        "retry_after": None,
        "provider_code": None,
        "routing_attempt": None,
        "attempts": None,
    }
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        retry(purpose=MAIN, kind=None, extra={}, sleep=lambda _s: None).again(
            ProviderServerError("oops", status_code=503), reason="server_error"
        )

    line = retry_line(caplog)
    assert "provider_code=" not in line and "attempts=" not in line and "retry_after=" not in line


def test_a_routing_attempt_of_zero_survives(caplog):
    """``0`` means *the router reached no provider at all* — the most diagnostic value there is."""
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        retry(sleep=lambda _s: None).again(rate_limit(routing_attempt=0), reason="rate_limited")

    assert "routing_attempt=0" in retry_line(caplog)


# === The head: a retry is not a call ==========================================


@pytest.fixture
def both_lines(caplog):
    """A real retry line and a real final line, from the emitters production uses.

    Never hand-written strings: the whole point of this file is that the bytes the fleet reads are
    the bytes this package writes, and a fixture that spelled them itself would prove nothing about
    the emitter.
    """
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        retry(sleep=lambda _s: None).again(rate_limit(retry_after=1), reason="rate_limited")
        log_llm_call(
            provider="openrouter",
            purpose=MEMORY,
            kind=RERANK_KIND,
            model="z-ai/glm-5.3-flash",
            seconds=1.42,
            outcome="ok",
            extra={"surface": "turn0", "pool": 20, "picked": 10},
        )
    return retry_line(caplog), rerank_llm_line(caplog)


def test_the_retry_line_is_invisible_to_every_column_that_counts_calls(both_lines):
    """The NOC's own gate, run against both lines — the assertion this feature turns on.

    A retry line matching `` llm provider=`` would inflate *Calls by Model*, drag a failure's wait
    into the duration average, and put a NULL on the rerank-outcomes chart the founder reads. The
    discriminator is the space-then-``provider=`` immediately after ``llm``.
    """
    retried, final = both_lines

    assert not re.search(LLM_CALL_GATE, retried), retried
    assert re.search(LLM_CALL_GATE, final), final
    # `memory_calls` / `memory_cost` / `memory_duration_s` / `memory_outcome` all AND the two.
    assert PURPOSE_MEMORY in retried  # it *is* a memory-purpose event …
    assert not re.search(rf"{LLM_CALL_GATE}.*{PURPOSE_MEMORY}", retried)  # … and still not a call
    assert re.search(rf"{LLM_CALL_GATE}.*{PURPOSE_MEMORY}", final)


def test_the_retry_line_moves_no_other_ungated_column(both_lines):
    """The columns a new line class moves by accident are the ungated ones.

    `_log_grammar`'s rule, applied to a second synthetic-looking line: carry no field that is a
    witness parent for another column, and never a field an ungated extractor reads.
    """
    retried, _final = both_lines

    assert not re.search(MEMORY_OUTCOME, retried)  # no outcome: nothing was decided
    assert not re.search(MEMORY_DURATION, retried)  # no duration: nothing was measured
    assert not re.search(RERANK_ON, retried)  # `memory_rerank_on`'s denominator
    assert not re.search(STAGE_LABEL, retried)
    assert not re.search(SOURCE_LABEL, retried)
    assert not re.search(WAKE_OUTCOME_GATE, retried)


def test_the_retry_line_is_a_warning_and_names_its_provider(both_lines):
    """WARNING because a retry is a degradation that recovered — not an error, not routine.

    ``provider=`` is deliberately *kept*: it is an ungated label that is also written on
    ``wake start``, so a retry line populating it is honest rather than contaminating.
    """
    retried, _final = both_lines

    assert retried.startswith("WARNING ")
    assert "provider=openrouter" in retried


def test_a_helper_retry_reads_the_same_way(caplog):
    """One grammar for three purposes: a brain, a rerank and a describe retry are one grep."""
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        retry(
            purpose=HELPER,
            kind="image.describe",
            extra={"subject": "cat.png"},
            sleep=lambda _s: None,
        ).again(rate_limit(), reason="rate_limited")

    line = retry_line(caplog)
    assert "purpose=helper" in line and "kind=image.describe" in line and "subject=cat.png" in line
    assert not re.search(LLM_CALL_GATE, line)
