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

**Something runs this on a schedule now, and a SKIP is RED** (issue #450). Nothing did before: ``-m
'not live'`` hides this file from every default run and from CI, and it skips itself green with no
key — three states, *passed* / *skipped* / *never invoked*, and from outside the box the last two
look exactly like the first. That is this repo's own named failure shape, Green-While-Absent,
pointed at the one suite that can catch a wire shape the live endpoint refuses, or an observability
field that quietly stopped landing. The trigger is the NOC prober (``basecradle-noc#563``, grown
from one pinned path to a **registry** of five arms in ``basecradle-noc#575``): this file is the
**``openrouter``** arm, and the prober clones the tip of ``main`` and runs this file's own
invocation with a **dedicated** ``OPENROUTER_API_KEY`` that lives only on the NOC box — never a
copy of a running agent's runtime key, per @origin's per-consumer ruling on #441. Weekly on green,
daily on red, per arm. The verdict is read off a JUnit report rather than ``returncode``, because
pytest exits **0** when every collected test skips, so an absent key is byte-identical to a pass at
the process boundary; a run reporting ``skipped > 0`` is a failure there.

Two consequences for anyone editing this file. **The path and the marker are a cross-repo
contract**: the arm selects ``-m live tests/test_openrouter_live.py``, so renaming either makes it
collect zero tests, which the prober reads as *never invoked* and calls red — correct and loud, but
tell the NOC rather than leaving it to fire. And **the assertions stay ours** — the prober runs
this suite instead of re-implementing it, so what this file proves is what gets proven on a
cadence, and adding a case here needs no coordination at all.
"""

from __future__ import annotations

import os

import pytest

from basecradle_harness import Message, OpenRouterProvider

from .conftest import plain

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

    **No ``max_tokens`` here, and that is load-bearing** (issue #284). A server-side search has to
    *summarize* what it found, and it draws that from the same completion budget — starve it and
    OpenRouter's search pipeline 500s instead of answering (measured: at ``max_tokens=16``, 6 of 8
    calls fail; unset, 8 of 8 succeed). The first cut of this test pinned 16 and then wrapped itself
    in a retry loop to survive the failures it was causing. A test that needs retries to pass is
    telling you something; listen to it rather than muffling it.
    """
    import logging

    provider = OpenRouterProvider(model=MODEL, api_key=KEY, builtin_tools=("web_search",))
    try:
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            provider.chat([Message.user("What is the capital of France? One word.")])
        pool = _live_pool(provider)
    finally:
        provider.close()

    endpoint = _field(_llm_line(caplog), "endpoint")
    assert endpoint in pool, (
        f"endpoint={endpoint!r} with a server-side search active — issue #280 exactly: the search "
        f"tool's upstream logged as the model's. Real pool: {sorted(pool)}"
    )


# === the MemPalace reranker (issue #464) ======================================

#: The rerank model the fleet runs (a founder decision, not this file's choice).
RERANK_MODEL = "z-ai/glm-5.3-flash"

#: A US-only routing pin, the shape every production rerank call sends. Several slugs rather than
#: one because a single pinned endpoint having a bad minute would be a red release gate for a
#: reason that is not this code — and a rerank pin is a *list* in production too, so this stays the
#: production shape rather than a test-only narrowing. Since issue #468 the call also sends
#: ``allow_fallbacks: true``, so a limited upstream is retried *within* this list rather than
#: failing the call; that is the behaviour this probe exercises.
RERANK_PROVIDERS = ("deepinfra", "baseten", "fireworks", "together")


def _rerank_line(caplog) -> str:
    """The rerank line with its ANSI verdict color stripped — ``outcome=`` is a colored field."""
    return plain(
        next(
            m for m in (r.getMessage() for r in caplog.records) if m.startswith("mempalace rerank")
        )
    )


@pytest.mark.skipif(not KEY, reason="set OPENROUTER_API_KEY to run the live OpenRouter probe")
def test_the_live_reranker_picks_and_reports_what_it_cost(caplog):
    """One real rerank call — the release gate for issue #464.

    Everything the offline suite proves about the reranker is proved against a body **we** wrote.
    Four things only a live call can settle, and each has already burned this repo once in another
    form: that ``provider: {only, allow_fallbacks: true, data_collection: "deny"}`` is *accepted*
    rather than 400'd; that ``reasoning: {effort: "low"}`` alongside ``response_format:
    json_object`` is a combination the model actually honors; that the model returns the documented
    ``{"picks": [...]}`` shape well enough to survive validation; and that ``usage.cost`` still
    lands, since a cost field that quietly stopped arriving would take the rerank spend series dark
    with nothing failing.

    The pool is built so the **right answer is unambiguous** and sits *below* the hybrid cut — a
    reranker that simply echoed ``1, 2, 3`` would fail here, which is what makes this a check on
    the ranking rather than on the plumbing.
    """
    import logging

    from basecradle_harness._rerank import SURFACE_TURN0, MemPalaceReranker

    pool = [
        {"text": "> john: the office coffee machine is broken again"},
        {"text": "> john: my favourite colour is green"},
        {"text": "> nova: the sprint demo moved to Thursday"},
        {"text": "> john: remember the staging endpoint is api.staging.example.com"},
        {"text": "> nova: lunch is at noon"},
    ]
    reranker = MemPalaceReranker(
        model=RERANK_MODEL, api_key=KEY, providers=RERANK_PROVIDERS, timeout=120.0
    )
    try:
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            chosen = reranker.rerank(
                "what was that staging endpoint we discussed?", pool, 1, surface=SURFACE_TURN0
            )
    finally:
        reranker.close()

    line = _rerank_line(caplog)
    assert _field(line, "outcome") == "ok", f"the live rerank fell back: {line}"
    # The ranking, not the plumbing: the answer is the fourth candidate, below a top-3 cut.
    assert chosen[0]["text"].endswith("api.staging.example.com"), line
    assert _field(line, "picked") == "1"
    assert _field(line, "pool") == "5"
    # A *value*, not a presence: a cost field that renders `0` is a spend series that reads free.
    cost = _field(line, "cost")
    assert cost and float(cost) > 0, f"the live usage block reported no cost: {line}"
    # The routing pin was honored. `only` is the hard restriction — `allow_fallbacks: true` lets
    # OpenRouter retry *inside* this list (issue #468), never outside it — so a served call is a
    # call one of these endpoints served.
    assert _field(line, "endpoint"), f"the live response named no serving upstream: {line}"


@pytest.mark.skipif(not KEY, reason="set OPENROUTER_API_KEY to run the live OpenRouter probe")
def test_a_live_rerank_against_a_nonexistent_model_is_config_class(caplog):
    """The loud half of the taxonomy, proved against the real endpoint's own 4xx.

    A silently-dead reranker is the failure issue #464 exists to prevent, and ERROR is what makes
    the fleet's "Error on AI Server" alert fire — so "a misconfigured model id pages" must not rest
    on a status code this repo *assumed* OpenRouter returns for one. The offline suite can only
    assert the mapping from a status we chose; this asserts the status.
    """
    import logging

    from basecradle_harness._rerank import SURFACE_TOOL, MemPalaceReranker

    reranker = MemPalaceReranker(
        model="z-ai/glm-does-not-exist-5.3", api_key=KEY, providers=RERANK_PROVIDERS
    )
    try:
        with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
            chosen = reranker.rerank("q", [{"text": "a"}, {"text": "b"}], 2, surface=SURFACE_TOOL)
    finally:
        reranker.close()

    # It falls back rather than raising — a broken reranker never costs the agent its memories.
    assert [hit["text"] for hit in chosen] == ["a", "b"]
    record = next(r for r in caplog.records if r.getMessage().startswith("mempalace rerank"))
    assert record.levelno == logging.ERROR, record.getMessage()
    assert _field(
        record.getMessage(),
        "reason",
    ).startswith("config:"), record.getMessage()
