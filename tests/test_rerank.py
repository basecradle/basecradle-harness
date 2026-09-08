"""The MemPalace LLM reranker (issue #464), against a real ``openrouter`` SDK over respx.

Two things about the shape of these tests are deliberate.

**The client is real and the transport is mocked**, exactly as `test_openrouter.py` does it — never
a hand-written stand-in for ``chat.send``. The near miss this repo already paid for (issue #433) was
a routing key the SDK *accepted* and never put on the wire, invisible to a whole tranche of tests
because every one of them asked what the adapter *decided*. So the request-shape assertions here
read the **bytes respx saw**: the ``provider`` pin, ``reasoning.effort``, ``response_format``, the
numbered candidates. A capability a mock both defines and verifies is a capability nobody checked.

**The failure classes are asserted by severity and by ``reason=``**, not merely by "it fell back".
The whole point of the two-class taxonomy is that a config-class fault *pages* and a runtime one
does not, and a fault filed under the wrong class is a silently-dead reranker again — this repo's
Green-While-Absent shape, one level up.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx
from openrouter import OpenRouter

from basecradle_harness._mempalace import (
    DEFAULT_N_RESULTS,
    MemPalaceMemoryProvider,
)
from basecradle_harness._rerank import (
    POOL_FLOOR,
    RERANK_API_KEY_VAR,
    RERANK_MODEL_VAR,
    RERANK_PROVIDERS_VAR,
    SURFACE_TOOL,
    SURFACE_TURN0,
    MemPalaceReranker,
    pool_size,
    providers_from_env,
    reranker_from_env,
    validated_picks,
)

# A fabricated OpenRouter endpoint and a correctly-shaped fake key — never a real credential.
BASE_URL = "https://openrouter.test/api/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
FAKE_KEY = "sk-or-v1-0123456789abcdef0123456789abcdef"
MODEL = "z-ai/glm-5.3-flash"
PROVIDERS = ("deepinfra", "baseten")


def completion(content, *, usage=None):
    """A chat-completions body the ``openrouter`` SDK's typed ``ChatResult`` accepts.

    ``system_fingerprint`` is **required** by that model — omit it and the SDK raises
    ``ResponseValidationError`` before the reranker ever sees the body.
    """
    return {
        "id": "gen-rerank0001",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "system_fingerprint": "fp_test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 4812,
            "completion_tokens": 611,
            "total_tokens": 5423,
            "completion_tokens_details": {"reasoning_tokens": 540},
            "cost": 0.000846,
        },
    }


def error(code, message):
    """An OpenRouter error body its SDK's typed error model accepts (``code`` is required)."""
    return {"error": {"message": message, "code": code}}


def _routing_metadata(selected, *considered):
    """OpenRouter's ``openrouter_metadata`` block — the endpoints it weighed and the one it picked.

    Modeled by the SDK's typed ``ChatResult``, so it must be complete or the response fails
    validation before the reranker sees it. It is the *only* trustworthy source of ``endpoint=``
    (issue #280), which is why the rerank call asks for it on every request.
    """
    available = [{"model": MODEL, "provider": name, "selected": False} for name in considered]
    available.append({"model": MODEL, "provider": selected, "selected": True})
    return {
        "attempt": 1,
        "endpoints": {"available": available, "total": len(available)},
        "is_byok": False,
        "region": None,
        "requested": MODEL,
        "strategy": "direct",
        "summary": f"routed to {selected}",
    }


def picks(*numbers):
    return json.dumps({"picks": list(numbers)})


def hits(n):
    """``n`` searcher-shaped hits, each distinguishable by its own text."""
    return [{"text": f"memory {i}"} for i in range(1, n + 1)]


def reranker(**kwargs):
    """A reranker over a **real** SDK client whose transport respx intercepts."""
    client = OpenRouter(api_key=FAKE_KEY, server_url=BASE_URL, retry_config=None)
    kwargs.setdefault("model", MODEL)
    kwargs.setdefault("api_key", FAKE_KEY)
    kwargs.setdefault("providers", PROVIDERS)
    return MemPalaceReranker(client=client, **kwargs)


@pytest.fixture
def router():
    with respx.mock(assert_all_called=False) as r:
        yield r


# === The pool: rerank only pays when it can reach past the cut =================


def test_pool_reaches_past_the_requested_count():
    """A pool the size of the request is a reorder, which the Turn-0 surface cannot even see."""
    assert pool_size(1) == POOL_FLOOR
    assert pool_size(DEFAULT_N_RESULTS) == POOL_FLOOR  # 10 → 2k lands exactly on the floor
    assert pool_size(20) == 40
    assert pool_size(0) == POOL_FLOOR  # a degenerate request still gets a sane pool


# === The happy path ===========================================================


def test_picks_are_honored_in_the_models_order(router):
    """The reranked list is the model's ranking, not the hybrid's — that is the whole feature."""
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(picks(11, 3, 7))))
    pool = hits(20)

    chosen = reranker().rerank("q", pool, 3, surface=SURFACE_TURN0)

    assert [hit["text"] for hit in chosen] == ["memory 11", "memory 3", "memory 7"]


def test_the_returned_hits_are_the_searchers_own_objects(router):
    """No model-authored text can ride back into a memory block, a tool result, or the palace.

    The reranker selects **by index**; it never constructs a hit. That is what makes the #438
    mining boundary untouched by this feature and an injection into a candidate structurally
    unable to put words into the agent's memory — asserted by object identity, not by equality.
    """
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(picks(2, 1))))
    pool = hits(20)

    chosen = reranker().rerank("q", pool, 2, surface=SURFACE_TURN0)

    assert chosen[0] is pool[1]
    assert chosen[1] is pool[0]


def test_the_request_carries_the_pin_the_effort_and_the_candidates(router):
    """The **bytes on the wire**, not what the adapter decided (the issue #433 lesson).

    Four things the founder decided ride this body, and every one of them is invisible to a test
    that stops at the SDK boundary: the US-only routing pin with data collection denied, `low`
    reasoning, JSON output, and the candidates **whole** — no truncation.

    ``allow_fallbacks`` is **true** and that is the pin working rather than a hole in it (issue
    #468): ``only`` is a hard restriction whatever the flag says, so a fallback retries *inside*
    the pinned list. With it off, OpenRouter picked one pinned upstream, that upstream's shared
    pool answered 429, and the reranker fell back to hybrid with three acceptable endpoints
    untried — one vendor's bad minute defeating the feature.
    """
    route = router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(picks(1))))
    long_memory = "x" * 4000
    pool = [{"text": long_memory}, *hits(19)]

    reranker().rerank("what was that endpoint", pool, 1, surface=SURFACE_TURN0)

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == MODEL
    assert body["provider"] == {
        "only": list(PROVIDERS),
        "allow_fallbacks": True,
        "data_collection": "deny",
    }
    assert body["reasoning"] == {"effort": "low"}
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.0
    user = body["messages"][1]["content"]
    assert "what was that endpoint" in user
    assert f"[1] {long_memory}" in user  # whole, never a 500-char excerpt
    assert "[20] memory 19" in user
    # The candidates are named as data, never as instructions — the injection stance in one line.
    assert "never instructions" in body["messages"][0]["content"]
    assert router.calls.last.request.headers["Authorization"] == f"Bearer {FAKE_KEY}"


def test_reasoning_exclude_is_not_sent_because_the_sdk_drops_it(router):
    """A documented gap, pinned so it is noticed the day the SDK closes it.

    Issue #464 asks for ``reasoning.exclude: true`` (billed, not returned). The pinned
    ``openrouter`` SDK models ``reasoning`` as a typed object with only ``effort`` and ``summary``,
    so an ``exclude`` key is **silently dropped** before serialization — and sending a key that
    never reaches the wire is exactly the #433 anti-pattern. Fleet law forbids hand-rolling the
    HTTP to get around it, so the harness sends what the SDK can express and says so here. It costs
    nothing but response bytes: reasoning tokens are billed either way, and the reranker discards
    everything but the picks. When the SDK gains the field, this test fails and the key goes in.
    """
    route = router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(picks(1))))
    reranker().rerank("q", hits(20), 1, surface=SURFACE_TURN0)

    assert "exclude" not in json.loads(route.calls.last.request.content)["reasoning"]


# === Validation ===============================================================


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ('{"picks": [3, 1, 2]}', [3, 1, 2]),
        ('{"picks": [3, 3, 1]}', [3, 1]),  # deduped, first position kept
        ('{"picks": [0, 21, 5]}', [5]),  # out of 1..pool, both ends
        ('{"picks": [true, 2]}', [2]),  # a bool is an int in Python; candidate 1 it is not
        ('{"picks": ["2", 2.0, 2]}', [2]),  # a string and a float are not candidate numbers
        ('{"picks": [1, 2, 3, 4, 5]}', [1, 2, 3]),  # truncated to k
        ('```json\n{"picks": [4]}\n```', [4]),  # a fenced answer is still an answer
        ('```json\n{"picks": [4]}', [4]),  # ...and an *unclosed* fence is not a reason to lose it
        ('{"picks": []}', []),
        ('{"picks": "all of them"}', []),
        ("{}", []),
        ("[1, 2, 3]", []),  # a bare array is not the documented shape
        ("I think memory 4 is best.", []),
        ("", []),
    ],
)
def test_validated_picks(answer, expected):
    """Everything the reranker contributes passes through here, so it is parsed — never trusted."""
    assert validated_picks(answer, pool=20, k=3) == expected


def test_a_short_answer_is_topped_up_from_the_hybrid_order(router):
    """Fewer than ``k`` valid picks degrades to *partial* reranking, never to a short recall.

    The same "a cap degrades, never collapses" rule the transcript's caps keep: a lazy or truncated
    answer must not cost the agent memories the palace already found for it.
    """
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(picks(9, 4))))
    pool = hits(20)

    chosen = reranker().rerank("q", pool, 5, surface=SURFACE_TURN0)

    assert [hit["text"] for hit in chosen] == [
        "memory 9",
        "memory 4",
        "memory 1",  # then the hybrid order, skipping what was already picked
        "memory 2",
        "memory 3",
    ]


def test_an_unusable_answer_falls_back_to_the_hybrid_top_k(router):
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion("no idea, sorry")))
    chosen = reranker().rerank("q", hits(20), 3, surface=SURFACE_TURN0)
    assert [hit["text"] for hit in chosen] == ["memory 1", "memory 2", "memory 3"]


def test_an_empty_pool_never_calls_the_model(router):
    route = router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(picks(1))))
    assert reranker().rerank("q", [], 5, surface=SURFACE_TURN0) == []
    assert not route.called


# === The log lines ============================================================


def test_the_rerank_line_carries_the_call_and_never_the_llm_head(router, caplog):
    """The observability contract, field by field — and the one head it must never wear.

    The fleet dashboard splits **LLM spend** from everything else on the literal `` llm provider=``
    head. A reranker billed into that series would inflate every agent's model-cost rollup with a
    second, unrelated spend, which is why the NOC asked for its own head — so this asserts the
    absence as hard as it asserts the fields.
    """
    body = completion(picks(*range(1, 9)))
    body["openrouter_metadata"] = _routing_metadata("DeepInfra", "Novita")
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=body))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        reranker().rerank("q", hits(20), 8, surface=SURFACE_TURN0)

    line = next(r for r in caplog.records if r.getMessage().startswith("mempalace rerank"))
    message = line.getMessage()
    assert line.levelno == logging.INFO
    for field in (
        "surface=turn0",
        "provider=openrouter",
        "endpoint=DeepInfra",
        f"model={MODEL}",
        "tokens_in=4812",
        "tokens_out=611",
        "tokens_reasoning=540",
        "cost=0.000846",
        "pool=20",
        "picked=8",
        "outcome=ok",
    ):
        assert field in message, message
    assert "duration=" in message
    assert " llm provider=" not in message
    assert "reason=" not in message  # a clean call names no fault


def test_the_recall_line_reports_the_pool_and_what_was_injected(fake_mempalace, tmp_path, caplog):
    """``mempalace recall`` at INFO, on both surfaces — the founder's explicit one-line-per-wake ask.

    It is the A/B: without ``rerank=`` and ``pool=`` on a shipped line, whether the reranker helped
    is a question nobody outside the box can answer.
    """
    _, searcher = fake_mempalace
    searcher.result = {"results": hits(20)}
    palace = tmp_path / "palace"
    palace.mkdir()
    provider = MemPalaceMemoryProvider(palace_path=palace, reranker=_always_picks([2, 1]))

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.search("q", 2, surface=SURFACE_TOOL)

    line = next(r for r in caplog.records if r.getMessage().startswith("mempalace recall"))
    assert line.levelno == logging.INFO
    message = line.getMessage()
    assert "surface=tool" in message
    assert "rerank=on" in message
    assert "pool=20" in message
    assert "injected=2" in message
    assert "chars=16" in message  # "memory 2" + "memory 1", the chunk text only
    assert "duration=" in message


def test_a_rerank_off_agent_says_so_and_emits_no_rerank_line(fake_mempalace, tmp_path, caplog):
    _, searcher = fake_mempalace
    searcher.result = {"results": hits(3)}
    palace = tmp_path / "palace"
    palace.mkdir()

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        MemPalaceMemoryProvider(palace_path=palace).search("q", 3, surface=SURFACE_TURN0)

    messages = [r.getMessage() for r in caplog.records]
    assert any("mempalace recall" in m and "rerank=off" in m for m in messages)
    assert not any(m.startswith("mempalace rerank") for m in messages)


# === Off by absence ===========================================================


def test_rerank_off_is_byte_identical_to_the_pre_rerank_search(fake_mempalace, tmp_path):
    """The regression bar: no model configured → the same query, the same pool, the same slice."""
    _, searcher = fake_mempalace
    searcher.result = {"results": hits(3)}
    palace = tmp_path / "palace"
    palace.mkdir()

    out = MemPalaceMemoryProvider(palace_path=palace).search("q", 3, surface=SURFACE_TURN0)

    _query, _palace, kwargs = searcher.queries[0]
    assert kwargs == {"n_results": 3, "candidate_strategy": "union"}  # no pool widening
    assert [hit["text"] for hit in out] == ["memory 1", "memory 2", "memory 3"]


def test_no_model_configured_is_no_reranker(monkeypatch):
    monkeypatch.delenv(RERANK_MODEL_VAR, raising=False)
    assert reranker_from_env() is None
    monkeypatch.setenv(RERANK_MODEL_VAR, "   ")  # set-but-blank is still unconfigured
    assert reranker_from_env() is None


def test_a_configured_model_binds_a_reranker(monkeypatch):
    monkeypatch.setenv(RERANK_MODEL_VAR, MODEL)
    monkeypatch.setenv(RERANK_API_KEY_VAR, FAKE_KEY)
    monkeypatch.setenv(RERANK_PROVIDERS_VAR, "deepinfra, baseten ,")
    bound = reranker_from_env()
    assert bound is not None
    assert bound.model == MODEL
    assert bound.providers == PROVIDERS


def test_provider_slugs_keep_their_order_and_drop_the_blanks():
    assert providers_from_env(" deepinfra ,baseten,, ") == ("deepinfra", "baseten")
    assert providers_from_env(None) == ()
    assert providers_from_env("") == ()


# === Failure classes ==========================================================


def _reason_of(records):
    line = next(r for r in records if r.getMessage().startswith("mempalace rerank"))
    reason = line.getMessage().partition("reason=")[2].split(" ")[0]
    return line.levelno, reason


@pytest.mark.parametrize(
    ("status", "message", "level", "reason"),
    [
        (401, "bad key", logging.ERROR, "config:auth"),
        (403, "forbidden", logging.ERROR, "config:auth"),
        (402, "insufficient credits", logging.ERROR, "config:billing"),
        (404, "no such model", logging.ERROR, "config:model_not_found"),
        (429, "slow down", logging.WARNING, "rate_limited"),
        (500, "oops", logging.WARNING, "server_error"),
        (418, "teapot", logging.WARNING, "api_error"),
    ],
)
def test_a_provider_fault_is_classified_and_logged_at_its_class(
    router, caplog, status, message, level, reason
):
    """Config-class pages; runtime-class does not. A fault filed wrong is a dead reranker nobody sees."""
    router.post(CHAT_URL).mock(return_value=httpx.Response(status, json=error(status, message)))
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        chosen = reranker().rerank("q", hits(20), 2, surface=SURFACE_TOOL)

    assert [hit["text"] for hit in chosen] == ["memory 1", "memory 2"]  # always falls back
    assert _reason_of(caplog.records) == (level, reason)


def test_a_timeout_is_distinguished_from_a_transport_failure(router, caplog):
    """The SDK collapses both into one class; the taxonomy wants *we waited* apart from *we never got there*."""
    router.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("too slow"))
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        reranker().rerank("q", hits(20), 2, surface=SURFACE_TOOL)
    assert _reason_of(caplog.records) == (logging.WARNING, "timeout")

    caplog.clear()
    router.post(CHAT_URL).mock(side_effect=httpx.ConnectError("no route"))
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        reranker().rerank("q", hits(20), 2, surface=SURFACE_TOOL)
    assert _reason_of(caplog.records) == (logging.WARNING, "transport")


def test_an_unparseable_answer_is_runtime_class(router, caplog):
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion("nope")))
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        reranker().rerank("q", hits(20), 2, surface=SURFACE_TOOL)
    level, reason = _reason_of(caplog.records)
    assert (level, reason) == (logging.WARNING, "parse")
    # It still cost money, so the line still carries what it cost.
    line = next(r for r in caplog.records if r.getMessage().startswith("mempalace rerank"))
    assert "cost=0.000846" in line.getMessage()


def test_an_unexpected_internal_failure_still_falls_back(caplog):
    """The promise is structural: no exception leaves `rerank`, whatever raised it.

    Both callers happen to be guarded — `_wake._memory_context` catches, and the engine turns a tool
    exception into a tool result — which is exactly why this cannot be left to them: a reranker
    fault escaping here would read as *memory itself* failing, and the agent would lose the hits the
    palace had already found for it.
    """

    class _Exploding(MemPalaceReranker):
        def _pick(self, query, hits, k, *, surface):
            raise RuntimeError("the SDK did something new")

    subject = _Exploding(model=MODEL, api_key=FAKE_KEY, providers=PROVIDERS)
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        chosen = subject.rerank("q", hits(20), 2, surface=SURFACE_TOOL)

    assert [hit["text"] for hit in chosen] == ["memory 1", "memory 2"]
    assert _reason_of(caplog.records) == (logging.WARNING, "internal")


def test_a_missing_key_is_config_class_and_never_calls_the_model(router, caplog):
    route = router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(picks(1))))
    dead = MemPalaceReranker(model=MODEL, fault="config:missing_api_key")

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        chosen = dead.rerank("q", hits(20), 2, surface=SURFACE_TURN0)

    assert not route.called
    assert [hit["text"] for hit in chosen] == ["memory 1", "memory 2"]
    assert _reason_of(caplog.records) == (logging.ERROR, "config:missing_api_key")


def test_a_model_with_no_key_or_no_providers_is_a_fault_not_a_none(monkeypatch, caplog):
    """Configured-and-dead is loud; unconfigured is a choice. They must never be the same object."""
    monkeypatch.setenv(RERANK_MODEL_VAR, MODEL)
    monkeypatch.delenv(RERANK_API_KEY_VAR, raising=False)
    monkeypatch.setenv(RERANK_PROVIDERS_VAR, "deepinfra")
    assert reranker_from_env() is not None

    monkeypatch.setenv(RERANK_API_KEY_VAR, FAKE_KEY)
    monkeypatch.delenv(RERANK_PROVIDERS_VAR, raising=False)
    dead = reranker_from_env()
    assert dead is not None
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        assert [hit["text"] for hit in dead.rerank("q", hits(3), 2, surface=SURFACE_TOOL)] == [
            "memory 1",
            "memory 2",
        ]
    assert _reason_of(caplog.records) == (logging.ERROR, "config:missing_providers")


def test_a_config_fault_is_reported_once_per_wake(router, caplog):
    """ERROR once, then quiet: one defect must not become a storm on a chatty wake.

    "Once per wake" is exactly the life of the provider object — it is built once per wake — so the
    repeats drop to DEBUG rather than being counted or timestamped.
    """
    router.post(CHAT_URL).mock(return_value=httpx.Response(401, json=error(401, "bad key")))
    subject = reranker()

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        for _ in range(3):
            subject.rerank("q", hits(20), 2, surface=SURFACE_TOOL)

    levels = [r.levelno for r in caplog.records if r.getMessage().startswith("mempalace rerank")]
    assert levels == [logging.ERROR, logging.DEBUG, logging.DEBUG]


def test_a_rerank_fault_never_raises_into_the_tool_result(fake_mempalace, tmp_path, router):
    """The worst outcome of a broken reranker is the retrieval the agent had before it existed."""
    _, searcher = fake_mempalace
    searcher.result = {"results": hits(20)}
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(500, json=error(500, "upstream fell over"))
    )
    palace = tmp_path / "palace"
    palace.mkdir()
    provider = MemPalaceMemoryProvider(palace_path=palace, reranker=reranker())

    answer = provider.tools()[0].run(query="anything")

    assert "- memory 1" in answer


# === Helpers ==================================================================


def _always_picks(numbers):
    """A reranker double that returns a fixed ranking without any client at all."""

    class _Fixed(MemPalaceReranker):
        def _pick(self, query, hits, k, *, surface):
            return list(numbers)

    return _Fixed(model=MODEL, api_key=FAKE_KEY, providers=PROVIDERS)
