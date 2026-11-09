"""refuseless is a vendor the harness knows, and every call is priced from its live price list.

Issue #844. refuseless.com is an OpenAI-compatible vendor reached through the ``openai`` SDK at its
own host. A completion states tokens and no dollars; the vendor's models list states a ``pricing``
object per model, so the harness reads that list once per wake and prices each call from it. There
is deliberately no rate table: a figure the list cannot state is a ``cost unpriced`` gap, never an
estimate.

The wire claims are read off a **real loopback HTTP server** (issue #433's rule); the pricing paths
use respx at the transport, so the real SDK parses every body.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import logging
import re
import threading

import httpx
import pytest
import respx

from basecradle_harness import Message
from basecradle_harness._basecradle import (
    _PROVIDER_BASE_URLS,
    _config_from_env,
    _provider_from_config,
    _require_no_server_builtins,
    _surface_for,
)
from basecradle_harness._observability import ComputedCost
from basecradle_harness._openai import _AFFINITY, CACHE_KEY_FIELD, OpenAIProvider
from basecradle_harness._openai_rates import Usage
from basecradle_harness._refuseless_pricing import Pricing, call_cost, pricing_of
from basecradle_harness._resolve import resolve_stems
from basecradle_harness._wake import _cost_report

from .conftest import completion

VENDOR = "refuseless"
HOST = "https://api.refuseless.com/v1"
MODEL = "glm-5.3-flash-abliterated"
OTHER = "deepseek-v4.1-flash-abliterated"
FAKE_KEY = "sk-rl-0123456789abcdef0123456789abcdef"

#: The two `pricing` shapes the vendor serves (read by the capital on 2026-11-09): the plain one,
#: and a discounted one whose top level is the effective price and `regular` the list price.
PLAIN = {"currency": "USD", "input_per_m": 0.12, "cached_input_per_m": 0.025, "output_per_m": 0.5}
DISCOUNTED = {
    "currency": "USD",
    "input_per_m": 0.12,
    "cached_input_per_m": 0.025,
    "output_per_m": 0.45,
    "discounted": True,
    "regular": {"input_per_m": 0.14, "cached_input_per_m": 0.035, "output_per_m": 0.5},
}
USAGE = {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}


def _row(model: str, pricing: dict | None, *, context: int = 1048576) -> dict:
    row = {
        "id": model,
        "object": "model",
        "created": 0,
        "owned_by": "refuseless",
        "abliterated": True,
        "context_length": context,
        "reasoning_default": "low",
        "reasoning_efforts": ["low", "high", "max"],
        "vision": True,
    }
    if pricing is not None:
        row["pricing"] = pricing
    return row


def _models(*rows: dict) -> dict:
    return {"object": "list", "data": list(rows)}


def _reply(model: str = MODEL, usage: dict | None = None) -> dict:
    body = completion(content="ok")
    body["model"] = model
    body["usage"] = USAGE if usage is None else usage
    return body


@pytest.fixture
def refuseless_env(monkeypatch):
    """An agent configured as the issue says the first two will be: the slug and nothing else."""
    monkeypatch.setenv("AI_PROVIDER", VENDOR)
    monkeypatch.setenv("AI_MODEL", MODEL)
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    for var in ("AI_SDK", "AI_SDK_SURFACE", "AI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def gaps(caplog):
    """The ``cost unpriced`` lines this test's adapter wrote."""
    caplog.set_level(logging.INFO, logger="basecradle_harness")

    def read() -> list[str]:
        return [
            r.getMessage() for r in caplog.records if r.getMessage().startswith("cost unpriced")
        ]

    return read


def _adapter(**kwargs) -> OpenAIProvider:
    return OpenAIProvider(
        MODEL, api_key=FAKE_KEY, base_url=HOST, provider=VENDOR, surface="chat", **kwargs
    )


def _priced(router, models, *, calls: int = 1, reply: dict | None = None) -> list[float | None]:
    """Run `calls` chat turns on one adapter (one wake) and return the ``cost=`` each line carried.

    `models` is what the list read answers: a response, or an exception the transport raises.
    """
    route = router.get(f"{HOST}/models")
    if isinstance(models, Exception):
        route.mock(side_effect=models)
    else:
        route.mock(return_value=models)
    router.post(f"{HOST}/chat/completions").mock(
        return_value=httpx.Response(200, json=reply or _reply())
    )
    handler = _Lines()
    logger = logging.getLogger("basecradle_harness")
    logger.addHandler(handler)
    try:
        adapter = _adapter()
        for _ in range(calls):
            adapter.chat([Message.user("hi")])
        adapter.close()
    finally:
        logger.removeHandler(handler)
    costs = [
        re.search(r" cost=([0-9.]+)", line) for line in handler.lines if line.startswith("llm ")
    ]
    return [float(found.group(1)) if found else None for found in costs]


class _Lines(logging.Handler):
    """Every line the package writes while installed, as rendered text."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


# --- the vendor is known, with its conventions ----------------------------------------------------


def test_the_slug_alone_configures_the_openai_sdk_on_chat(refuseless_env):
    assert _config_from_env() == (VENDOR, "openai", "chat")


def test_the_build_aims_the_openai_sdk_at_the_vendors_own_host(refuseless_env):
    provider = _provider_from_config(*_config_from_env())
    assert isinstance(provider, OpenAIProvider)
    assert provider.provider == VENDOR
    assert provider.surface == "chat"
    assert provider.base_url == HOST == _PROVIDER_BASE_URLS[VENDOR]
    assert provider._affinity is None
    provider.close()


def test_ai_base_url_still_overrides_the_default_host(refuseless_env):
    refuseless_env.setenv("AI_BASE_URL", "https://eu.refuseless.example/v1")
    provider = _provider_from_config(*_config_from_env())
    assert provider.base_url == "https://eu.refuseless.example/v1"
    provider.close()


def test_the_affinity_cell_is_a_decided_no():
    assert (VENDOR, "chat") in _AFFINITY
    assert _AFFINITY[(VENDOR, "chat")] is None
    assert (VENDOR, "responses") not in _AFFINITY


def test_responses_is_refused_at_startup_because_the_vendor_documents_none(refuseless_env):
    refuseless_env.setenv("AI_SDK_SURFACE", "responses")
    with pytest.raises(ValueError, match=r"'refuseless' documents only the \('chat',\) surface"):
        _config_from_env()


def test_every_reader_of_a_surface_refuses_the_same_cell():
    """The describer's own stack and the off-box resolver share `_surface_for`'s rule."""
    assert _surface_for("openai", None, VENDOR) == "chat"
    assert _surface_for("openai", "chat", VENDOR) == "chat"
    with pytest.raises(ValueError, match="documents only"):
        _surface_for("openai", "responses", VENDOR)
    with pytest.raises(ValueError, match="documents only"):
        resolve_stems(provider=VENDOR, surface="responses")
    resolved = resolve_stems(provider=VENDOR)
    assert resolved["ai_sdk_surface"] == "chat"
    assert resolved["builtins"] == []


@pytest.mark.parametrize("sdk", ["xai-sdk", "openrouter", "google-genai", "anthropic"])
def test_a_native_sdk_cannot_reach_it(refuseless_env, sdk):
    with pytest.raises(ValueError, match="requires AI_PROVIDER"):
        _provider_from_config(VENDOR, sdk, "", model=MODEL, api_key=FAKE_KEY)


def test_a_server_side_built_in_is_refused_rather_than_silently_dropped(refuseless_env):
    with pytest.raises(ValueError, match="'refuseless' runs no server-side built-in"):
        _provider_from_config(VENDOR, "openai", "chat", builtins=["web_search"])
    with pytest.raises(ValueError, match="runs no server-side built-in"):
        _require_no_server_builtins(VENDOR, ["web_search"])
    _require_no_server_builtins(VENDOR, [])


def test_a_misspelled_slug_is_still_a_generic_vendor_that_needs_a_host(refuseless_env):
    refuseless_env.setenv("AI_PROVIDER", "refuselss")
    with pytest.raises(ValueError, match=r"AI_PROVIDER='refuselss'.*set AI_BASE_URL"):
        _config_from_env()


def test_resolved_config_reports_a_live_computed_basis_and_claims_no_row(refuseless_env):
    report = _cost_report(VENDOR, "openai", MODEL)
    assert report == {
        "cost_basis": "computed",
        "cost_rates_source": f"GET {HOST}/models → data[].pricing, read live once per wake",
        "cost_model_priced": None,
    }
    refuseless_env.setenv("AI_BASE_URL", "https://eu.refuseless.example/v1/")
    source = _cost_report(VENDOR, "openai", MODEL)["cost_rates_source"]
    assert source.startswith("GET https://eu.refuseless.example/v1/models ")


# --- the pricing arithmetic -----------------------------------------------------------------------


def test_the_plain_shape_prices_tokens_at_the_top_level_rates():
    # 1,000 × $0.12/M + 200 × $0.50/M
    cost = call_cost(pricing_of(_row(MODEL, PLAIN)), Usage(input=1000, cached=0, output=200))
    assert isinstance(cost, ComputedCost)
    assert cost == pytest.approx(0.00022)


def test_the_discounted_shape_prices_at_the_effective_rate_never_the_list_price():
    # 1,000 × $0.12/M + 200 × $0.45/M; `regular` ($0.14 / $0.50) is never read.
    cost = call_cost(pricing_of(_row(MODEL, DISCOUNTED)), Usage(input=1000, cached=0, output=200))
    assert cost == pytest.approx(0.00021)


def test_a_stated_cached_count_is_priced_at_the_cached_rate():
    # 600 uncached × $0.12/M + 400 cached × $0.025/M + 200 × $0.50/M
    cost = call_cost(pricing_of(_row(MODEL, PLAIN)), Usage(input=1000, cached=400, output=200))
    assert cost == pytest.approx((600 * 0.12 + 400 * 0.025 + 200 * 0.5) / 1e6)


def test_a_rate_the_call_does_not_need_is_not_required():
    pricing = Pricing(input=0.12, output=0.5)
    assert call_cost(pricing, Usage(input=10, cached=0, output=1)) == pytest.approx(0.0000017)
    assert call_cost(pricing, Usage(input=10, cached=4, output=1)) is None


def test_a_cache_write_has_no_published_rate_so_the_call_is_unpriced():
    pricing = pricing_of(_row(MODEL, PLAIN))
    assert call_cost(pricing, Usage(input=1000, cached=0, output=200, cache_write=10)) is None


def test_a_call_with_no_counts_is_unpriced():
    assert call_cost(pricing_of(_row(MODEL, PLAIN)), None) is None


@pytest.mark.parametrize(
    "pricing",
    [
        None,
        {**PLAIN, "currency": "EUR"},
        {key: value for key, value in PLAIN.items() if key != "currency"},
        {key: value for key, value in PLAIN.items() if key != "input_per_m"},
        {key: value for key, value in PLAIN.items() if key != "output_per_m"},
        {**PLAIN, "input_per_m": "0.12"},
        {**PLAIN, "output_per_m": True},
        {**PLAIN, "output_per_m": -0.5},
        {**PLAIN, "input_per_m": float("nan")},
        {**PLAIN, "input_per_m": float("inf")},
        "0.12",
    ],
)
def test_a_row_that_states_no_usable_dollar_price_has_none(pricing):
    row = _row(MODEL, None)
    if pricing is not None:
        row["pricing"] = pricing
    assert pricing_of(row) is None


def test_a_malformed_cached_rate_reads_as_absent_not_as_a_broken_row():
    pricing = pricing_of(_row(MODEL, {**PLAIN, "cached_input_per_m": "cheap"}))
    assert pricing == Pricing(input=0.12, output=0.5, cached_input=None)


# --- the adapter: one read per wake, every gap stated once ----------------------------------------


def test_a_call_is_priced_from_the_list_and_the_line_says_computed(caplog):
    caplog.set_level(logging.INFO, logger="basecradle_harness")
    with respx.mock(assert_all_mocked=True) as router:
        models = router.get(f"{HOST}/models").mock(
            return_value=httpx.Response(200, json=_models(_row(MODEL, DISCOUNTED)))
        )
        router.post(f"{HOST}/chat/completions").mock(
            return_value=httpx.Response(200, json=_reply())
        )
        adapter = _adapter()
        adapter.chat([Message.user("hi")])
        adapter.chat([Message.user("again")])
        adapter.close()
    assert models.call_count == 1
    assert models.calls[0].request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm ")]
    assert len(lines) == 2
    for line in lines:
        assert "provider=refuseless" in line
        assert f"model={MODEL}" in line
        assert "tokens_in=1000 tokens_out=200" in line
        assert "cost=0.00021" in line
        assert "cost_basis=computed" in line
    assert not [r for r in caplog.records if r.getMessage().startswith("cost unpriced")]


def test_the_model_the_response_names_is_the_row_that_prices_it(gaps):
    with respx.mock(assert_all_mocked=True) as router:
        listed = _models(_row(MODEL, PLAIN), _row(OTHER, DISCOUNTED))
        (cost,) = _priced(router, httpx.Response(200, json=listed), reply=_reply(model=OTHER))
    assert cost == pytest.approx(0.00021)
    assert not gaps()


@pytest.mark.parametrize(
    "models",
    [
        httpx.Response(503, json={"error": {"message": "unavailable"}}),
        httpx.Response(401, json={"error": {"message": "bad key"}}),
        # A 200 that is not the list: an edge page answering in place of the API.
        httpx.Response(200, text="<!DOCTYPE html><title>Attention Required!</title>"),
    ],
    ids=["server_error", "refused", "not_json"],
)
def test_an_unreadable_list_is_one_rates_unavailable_gap_and_one_request(gaps, models):
    with respx.mock(assert_all_mocked=True) as router:
        costs = _priced(router, models, calls=3)
        assert router.routes[0].call_count == 1
    assert costs == [None, None, None]
    assert gaps() == [
        f"cost unpriced provider=refuseless model={MODEL} charge=tokens reason=rates_unavailable"
    ]


def test_an_unreachable_list_is_a_gap_too(gaps):
    with respx.mock(assert_all_mocked=True) as router:
        costs = _priced(router, httpx.ConnectError("refused"), calls=2)
    assert costs == [None, None]
    assert gaps() == [
        f"cost unpriced provider=refuseless model={MODEL} charge=tokens reason=rates_unavailable"
    ]


def test_a_model_the_list_has_no_row_for_is_unknown(gaps):
    with respx.mock(assert_all_mocked=True) as router:
        costs = _priced(router, httpx.Response(200, json=_models(_row(OTHER, PLAIN))), calls=2)
    assert costs == [None, None]
    assert gaps() == [
        f"cost unpriced provider=refuseless model={MODEL} charge=tokens reason=unknown_model"
    ]


@pytest.mark.parametrize("pricing", [None, {**PLAIN, "currency": "EUR"}])
def test_a_row_with_no_usable_price_is_not_reported(gaps, pricing):
    with respx.mock(assert_all_mocked=True) as router:
        (cost,) = _priced(router, httpx.Response(200, json=_models(_row(MODEL, pricing))))
    assert cost is None
    assert gaps() == [
        f"cost unpriced provider=refuseless model={MODEL} charge=tokens reason=not_reported"
    ]


def test_the_context_limit_comes_from_the_same_one_read(gaps):
    with respx.mock(assert_all_mocked=True) as router:
        models = router.get(f"{HOST}/models").mock(
            return_value=httpx.Response(200, json=_models(_row(MODEL, PLAIN, context=524288)))
        )
        router.post(f"{HOST}/chat/completions").mock(
            return_value=httpx.Response(200, json=_reply())
        )
        adapter = _adapter()
        assert adapter.context_limit() == 524288
        adapter.chat([Message.user("hi")])
        adapter.close()
    assert models.call_count == 1
    assert not gaps()


def test_an_unreadable_list_leaves_the_context_limit_to_its_floor(caplog):
    caplog.set_level(logging.INFO, logger="basecradle_harness")
    with respx.mock(assert_all_mocked=True) as router:
        router.get(f"{HOST}/models").mock(return_value=httpx.Response(500, json={}))
        adapter = _adapter()
        assert adapter.context_limit() is None
        assert adapter.context_limit() is None
        adapter.close()
        assert router.routes[0].call_count == 1
    (warning,) = [r for r in caplog.records if "Could not read the model list" in r.getMessage()]
    assert HOST in warning.getMessage()


def test_no_other_vendor_reads_the_list_on_a_call():
    """The list read is refuseless's: a generic vendor's call sends nothing but the call."""
    with respx.mock(assert_all_mocked=True) as router:
        router.post("https://api.wiro.example/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_reply())
        )
        adapter = OpenAIProvider(
            MODEL,
            api_key=FAKE_KEY,
            base_url="https://api.wiro.example/v1",
            provider="wiro",
            surface="chat",
        )
        adapter.chat([Message.user("hi")])
        adapter.close()
        assert len(router.calls) == 1


# --- the bytes on the wire ------------------------------------------------------------------------


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


@contextlib.contextmanager
def _vendor():
    """A fake refuseless on loopback: its models list and its chat endpoint, recording each request."""
    seen: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def _answer(self, body: dict) -> None:
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _record(self, body) -> None:
            seen.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                }
            )

        def do_GET(self):
            self._record(None)
            self._answer(_models(_row(MODEL, PLAIN)))

        def do_POST(self):
            self._record(json.loads(self.rfile.read(int(self.headers.get("content-length") or 0))))
            self._answer(_reply())

        def log_message(self, *args):
            pass

    server = _Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", seen
    finally:
        server.shutdown()
        server.server_close()


def test_a_wake_sends_the_call_and_one_list_read_with_the_agents_key_and_nothing_else(
    refuseless_env, caplog
):
    caplog.set_level(logging.INFO, logger="basecradle_harness")
    with _vendor() as (root, seen):
        refuseless_env.setenv("AI_BASE_URL", root)
        provider = _provider_from_config(*_config_from_env(), default_output_tokens=32768)
        provider.bind_conversation("timeline:019f6e71-2a12-7b69-a204-0fec1497b9c2")
        provider.chat([Message.user("hi")])
        provider.chat([Message.user("again")])
        provider.close()
    reads = [r for r in seen if r["method"] == "GET"]
    calls = [r for r in seen if r["method"] == "POST"]
    assert [r["path"] for r in reads] == ["/v1/models"]
    assert [r["path"] for r in calls] == ["/v1/chat/completions"] * 2
    for request in seen:
        assert request["headers"]["authorization"] == f"Bearer {FAKE_KEY}"
        assert "x-grok-conv-id" not in request["headers"]
        assert "x-openrouter-metadata" not in request["headers"]
    for call in calls:
        body = call["body"]
        assert body["model"] == MODEL
        assert body["max_tokens"] == 32768
        for field in (CACHE_KEY_FIELD, "search_parameters", "provider", "tools"):
            assert field not in body
    llm = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm ")]
    assert all("cost=0.00022" in line and "cost_basis=computed" in line for line in llm)
