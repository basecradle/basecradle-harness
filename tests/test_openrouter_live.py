"""Live smoke for the native OpenRouter adapter (`OpenRouterProvider`) — issue #234.

The one check the mocked suite **structurally cannot** make: the adapter tests inject a fake SDK
client (respx mocks the transport), so a request the *real* OpenRouter endpoint rejects — a
model-params key the live API refuses, a wire shape that drifted on an SDK bump — still passes
them. The same blindness runs the other way, on the *response*: a mocked body proves only that the
adapter reads the fields **we told it to expect**, never that those are still the fields OpenRouter
sends (issue #274). This test builds a **real** ``openrouter`` client and hits ``openrouter.ai`` for
real, so a regression to a server-rejected wiring — or to an observability field that quietly
stopped landing — fails loudly here.

It is an explicitly-marked **live** job (`@pytest.mark.live`), deselected from the default offline
run by ``addopts = -m 'not live'`` and skipped when no key is present. Run it deliberately::

    OPENROUTER_API_KEY=sk-or-... uv run pytest -m live tests/test_openrouter_live.py -v

The capital re-runs it (with a valid OpenRouter key) at the release gate; this file makes that a
repeatable command rather than a one-off manual probe.
"""

from __future__ import annotations

import os

import pytest

from basecradle_harness import Message, OpenRouterProvider
from basecradle_harness._exceptions import ProviderAPIError

pytestmark = pytest.mark.live

KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "z-ai/glm-5.2"


@pytest.mark.skipif(not KEY, reason="set OPENROUTER_API_KEY to run the live OpenRouter probe")
def test_native_openrouter_returns_a_reply():
    """A real turn against ``openrouter.ai`` returns non-empty assistant text (@glm-5.2's brain)."""
    provider = OpenRouterProvider(model="z-ai/glm-5.2", api_key=KEY)
    try:
        reply = provider.chat([Message.user("Reply with a single short greeting.")])
    finally:
        provider.close()

    assert reply.role == "assistant"
    assert reply.content  # a real, non-empty answer


@pytest.mark.skipif(not KEY, reason="set OPENROUTER_API_KEY to run the live OpenRouter probe")
def test_live_model_params_reach_the_endpoint():
    """A ``model_params.json``-style tuning key (``temperature``) is accepted by the live API.

    The mocked suite proves the key lands in the request body; only this proves the *real*
    endpoint accepts it rather than rejecting the request.

    ``max_tokens`` must leave room to actually *answer*: ``glm-5.2`` is a **reasoning** model, and
    its reasoning tokens are drawn from the same budget — at the 32 this test used to pass, the
    whole allowance went to reasoning and the reply came back with ``content=None``, failing the
    release gate for a reason that had nothing to do with model params. A false-failing gate is
    worse than no gate: it trains you to ignore it.
    """
    provider = OpenRouterProvider(model=MODEL, api_key=KEY, temperature=0.2, max_tokens=512)
    try:
        reply = provider.chat([Message.user("Say hi.")])
    finally:
        provider.close()

    assert reply.role == "assistant"
    assert reply.content


def _llm_line(caplog) -> str:
    return next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("llm "))


def _field(line: str, key: str) -> str | None:
    return next((f.split("=", 1)[1] for f in line.split() if f.startswith(f"{key}=")), None)


def _live_pool(provider: OpenRouterProvider) -> set[str]:
    """Every upstream that actually serves this model, straight from OpenRouter's endpoints API."""
    author, _, slug = MODEL.partition("/")
    response = provider._client.endpoints.list(author=author, slug=slug)
    return {e.provider_name for e in response.data.endpoints}


@pytest.mark.skipif(not KEY, reason="set OPENROUTER_API_KEY to run the live OpenRouter probe")
def test_the_live_endpoint_is_a_real_member_of_the_models_pool(caplog):
    """The observability fields (#274) are read off the **live** API, and the mocked suite cannot
    prove the field names are still the real ones — nor that the value is *true*.

    This asserts the second part, and it is the lesson of issue #280: it is not enough that
    ``endpoint=`` is **present**. The defect that shipped logged ``endpoint=OpenAI`` on every
    search-enabled ``z-ai/glm-5.2`` wake — a vendor serving **no endpoint in that model's pool** —
    and the previous version of this test passed the entire time, because it only checked that the
    field was there. A wrong endpoint is worse than an absent one: it does not leave a gap in the
    routing review, it invents a distribution. So the value is checked **against the live pool**.

    ``cached_tokens`` is deliberately not asserted: a cache hit needs a prefix the endpoint has seen
    before, so a cold one-shot probe legitimately reports none.
    """
    import logging

    provider = OpenRouterProvider(model=MODEL, api_key=KEY, max_tokens=16)
    try:
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            provider.chat([Message.user("Say hi.")])
        pool = _live_pool(provider)
    finally:
        provider.close()

    line = _llm_line(caplog)
    endpoint = _field(line, "endpoint")
    assert endpoint, f"the live response named no serving upstream: {line}"
    assert endpoint in pool, (
        f"endpoint={endpoint!r} serves no {MODEL} endpoint — a fabricated routing datum, the exact "
        f"defect of issue #280. Real pool: {sorted(pool)}"
    )
    assert "cost=" in line, f"the live usage block reported no cost: {line}"


@pytest.mark.skipif(not KEY, reason="set OPENROUTER_API_KEY to run the live OpenRouter probe")
def test_the_live_endpoint_stays_real_when_a_server_side_search_runs(caplog):
    """The exact shape that produced issue #280 — and the one no offline test can honestly prove.

    With ``openrouter:web_search`` active, OpenRouter's undocumented top-level ``provider`` reports
    **the search tool's** upstream (``OpenAI``), not the model's. That is a *live* fact about
    OpenRouter's wire, so only a live call can pin it: a mocked body proves we read the fields we
    told ourselves to expect, never that they still mean what we thought.

    OpenRouter's web-search plugin returns intermittent 5xx (measured ~2-in-5, present *before* this
    change and unrelated to it), so an upstream outage must not read as our regression: the probe
    retries, and skips only when every attempt fails upstream.
    """
    import logging

    provider = OpenRouterProvider(
        model=MODEL, api_key=KEY, builtin_tools=("web_search",), max_tokens=16
    )
    try:
        pool = _live_pool(provider)
        for attempt in range(4):
            caplog.clear()
            try:
                with caplog.at_level(logging.INFO, logger="basecradle_harness"):
                    provider.chat([Message.user("What is the capital of France? One word.")])
                break
            except ProviderAPIError:  # OpenRouter's own 5xx on its search plugin — not ours
                if attempt == 3:
                    pytest.skip("openrouter:web_search is 5xx-ing upstream; nothing to prove")
    finally:
        provider.close()

    endpoint = _field(_llm_line(caplog), "endpoint")
    assert endpoint in pool, (
        f"endpoint={endpoint!r} with a server-side search active — issue #280 exactly: the search "
        f"tool's upstream logged as the model's. Real pool: {sorted(pool)}"
    )
