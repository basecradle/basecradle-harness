"""The refuseless_account_balance tool: the vendor's figures, degraded modes, caching — issue #848.

All HTTP is mocked with respx at the transport; no test reaches the network. The healthy body is the
exact shape the capital read off ``GET https://api.refuseless.com/v1/credits`` on 2026-11-10, with the
per-key spend the vendor added the same day (issue #868), and the error bodies follow the vendor's documented envelope (``{"error": {"code", "message", "type"}}``).

Three properties carry the tool, each pinned below in its "obviously fine" broken form:

- **The figures are the vendor's.** ``available`` is the headline because a request is refused when
  it cannot reserve, whatever the balance says; nothing is re-derived or rounded to cents, so the six
  decimals the vendor states are the six the model reads.
- **The key is the brain's, so it goes only where the brain's calls go** — the default host or
  ``AI_BASE_URL``, read at call time and only under ``AI_PROVIDER=refuseless``, and the cache is
  keyed on both without ever holding the key.
- **Nothing but figures leaves the tool**: never the key, never the vendor's error prose.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

import httpx
import pytest
import respx

from basecradle_harness import Engine, Policy, RefuselessAccountBalanceTool, ToolRegistry
from basecradle_harness._basecradle import _PROVIDER_BASE_URLS
from basecradle_harness._refuseless_account import _dollars

KEY = "sk-rl-0123456789abcdef0123456789abcdef"
HOST = "https://api.refuseless.com/v1"
CREDITS_URL = f"{HOST}/credits"
#: 2026-11-09T23:57:41Z, the `updated_at` the capital read.
UPDATED_AT = 1794268661
#: A sentinel in the vendor's error prose, which must never reach the model.
PROSE = "account 019f6e71-2a12-7b69-a204-0fec1497b9c2 belongs to john@example.com"


def _credits(**overrides) -> dict:
    """``GET /credits`` exactly as read live, with `overrides` applied (``...`` removes a field)."""
    body = {
        "object": "credits",
        "balance": 69.999906,
        "reserved": 0,
        "available": 69.999906,
        "key_limit": None,
        "key_spent": 0.001405,
        "key_spent_since": None,
        "unit": "usd",
        "updated_at": UPDATED_AT,
    }
    for field, value in overrides.items():
        if value is ...:
            body.pop(field)
        else:
            body[field] = value
    return body


def _ok(body: dict) -> httpx.Response:
    """A 200 carrying `body` as Python's `json` writes it, ``NaN`` and ``Infinity`` included.

    httpx refuses to encode a non-finite float, but Python's `json` (and so a hostile or broken
    intermediary) will, and the parser must meet it.
    """
    return httpx.Response(
        200, content=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )


def _error(status: int, code: str) -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"code": code, "message": PROSE, "type": "api_error"}}
    )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """A refuseless brain with no ambient credential or host; each test sets what it means."""
    monkeypatch.setenv("AI_PROVIDER", "refuseless")
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("AI_BASE_URL", raising=False)


@pytest.fixture
def tool():
    return RefuselessAccountBalanceTool(api_key=KEY, cache_ttl=0)


def _read(tool, response) -> tuple[str, respx.Route]:
    with respx.mock(assert_all_mocked=True) as mock:
        route = mock.get(CREDITS_URL).mock(
            return_value=response if isinstance(response, httpx.Response) else None,
            side_effect=None if isinstance(response, httpx.Response) else response,
        )
        return tool.run(), route


# --- the figures --------------------------------------------------------------------------------


def test_the_live_shape_reads_as_available_with_every_figure_behind_it(tool):
    result, route = _read(tool, httpx.Response(200, json=_credits()))
    assert result == (
        "refuseless credit available: $69.999906 USD (as of 2026-11-09T23:57:41Z).\n"
        "Balance $69.999906, of which $0.00 is held by requests in flight (an unused hold returns "
        "to the balance when its request settles). This key has no spending limit of its own. "
        "This key has been billed $0.001405 in total."
    )
    request = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.method == "GET"


def test_the_headline_is_what_a_request_can_spend_not_the_balance(tool):
    body = _credits(balance=70.0, reserved=12.5, available=57.5)
    result, _ = _read(tool, httpx.Response(200, json=body))
    headline, context = result.splitlines()
    assert headline == "refuseless credit available: $57.50 USD (as of 2026-11-09T23:57:41Z)."
    assert "Balance $70.00, of which $12.50 is held by requests in flight" in context


def test_a_key_limit_is_stated_as_the_vendor_states_it(tool):
    body = _credits(available=4.25, key_limit=4.25)
    result, _ = _read(tool, httpx.Response(200, json=body))
    assert result.startswith("refuseless credit available: $4.25 USD")
    assert "This key's spending limit allows $4.25 more." in result
    assert "no spending limit" not in result


# --- the key's own spend (issue #868) -------------------------------------------------------------


def test_a_counter_with_a_start_says_since_when(tool):
    body = _credits(key_spent=12.5, key_spent_since=UPDATED_AT - 86400)
    result, _ = _read(tool, httpx.Response(200, json=body))
    assert result.endswith("This key has been billed $12.50 since 2026-11-08T23:57:41Z.")


def test_a_key_that_has_spent_nothing_says_so(tool):
    result, _ = _read(tool, httpx.Response(200, json=_credits(key_spent=0)))
    assert result.endswith("This key has been billed $0.00 in total.")


@pytest.mark.parametrize(
    "overrides",
    [
        {"key_spent": ...},
        {"key_spent": None},
        {"key_spent": ..., "key_spent_since": ...},
        {"key_spent": float("nan")},
        {"key_spent": float("inf")},
        {"key_spent": -0.5},
        {"key_spent": True},
        {"key_spent": "0.001405"},
        {"key_spent": 10**400},
        {"key_spent_since": "2026-11-01T00:00:00Z"},
        {"key_spent_since": -1},
        {"key_spent_since": True},
        {"key_spent_since": 1794268661.5},
        {"key_spent_since": 10**20},
    ],
    ids=[
        "absent",
        "null",
        "both_absent",
        "nan",
        "inf",
        "negative",
        "bool",
        "string",
        "overflow",
        "since_string",
        "since_negative",
        "since_bool",
        "since_float",
        "since_overflow",
    ],
)
def test_a_spend_the_vendor_did_not_state_usably_leaves_the_balance_standing(tool, overrides):
    """The spend is additive: without it the balance still answers, and only its sentence goes."""
    result, _ = _read(tool, _ok(_credits(**overrides)))
    assert result.startswith("refuseless credit available: $69.999906 USD")
    assert result.endswith("This key has no spending limit of its own.")
    assert "billed" not in result


def test_nothing_available_is_called_out_first(tool):
    body = _credits(balance=3.0, reserved=3.0, available=0)
    result, _ = _read(tool, httpx.Response(200, json=body))
    headline, context = result.splitlines()
    assert headline.startswith("refuseless credit available: $0.00 USD")
    assert context.startswith("No new request can be paid for until the account is topped up.")


@pytest.mark.parametrize("available", [0.0000004, -0.0000004, -1.5])
def test_a_figure_that_reads_as_nothing_is_called_out_however_it_rounds(tool, available):
    body = _credits(balance=1.0, reserved=1.0, available=available)
    result, _ = _read(tool, httpx.Response(200, json=body))
    assert result.splitlines()[1].startswith("No new request can be paid for")


def test_a_healthy_account_carries_no_callout(tool):
    result, _ = _read(tool, httpx.Response(200, json=_credits()))
    assert "topped up" not in result


@pytest.mark.parametrize(
    ("usd", "text"),
    [
        (69.999906, "$69.999906"),
        (70, "$70.00"),
        (0.1, "$0.10"),
        (0, "$0.00"),
        (-0.0, "$0.00"),
        (1234.5, "$1,234.50"),
        (0.000001, "$0.000001"),
        (-5.25, "-$5.25"),
        (0.9999995, "$1.00"),
        (1.0000005, "$1.000001"),
        (-0.0000004, "$0.00"),
        (-0.0000006, "-$0.000001"),
        (0.00000012, "$0.00"),
    ],
)
def test_a_figure_is_shown_to_the_vendors_precision(usd, text):
    assert _dollars(usd) == text


# --- where the key goes -------------------------------------------------------------------------


def test_the_key_and_host_are_the_brains_read_at_call_time(monkeypatch):
    tool = RefuselessAccountBalanceTool(cache_ttl=0)
    monkeypatch.setenv("AI_API_KEY", KEY)
    result, route = _read(tool, httpx.Response(200, json=_credits()))
    assert result.startswith("refuseless credit available:")
    assert route.calls.last.request.headers["authorization"] == f"Bearer {KEY}"
    assert HOST == _PROVIDER_BASE_URLS["refuseless"]


def test_ai_base_url_moves_the_read_with_the_brains_calls(monkeypatch, tool):
    monkeypatch.setenv("AI_BASE_URL", "https://eu.refuseless.example/v1/")
    with respx.mock(assert_all_mocked=True) as mock:
        route = mock.get("https://eu.refuseless.example/v1/credits").mock(
            return_value=httpx.Response(200, json=_credits())
        )
        assert tool.run().startswith("refuseless credit available:")
    assert route.call_count == 1


@pytest.mark.parametrize("provider", [None, "openai", "openrouter", "wiro", "Not A Slug!"])
def test_the_environment_is_read_only_on_a_refuseless_brain(monkeypatch, provider):
    """Beside another brain, AI_API_KEY is that vendor's key: it must never go to refuseless."""
    if provider is None:
        monkeypatch.delenv("AI_PROVIDER")
    else:
        monkeypatch.setenv("AI_PROVIDER", provider)
    monkeypatch.setenv("AI_API_KEY", "sk-proj-another-vendors-key-0123456789")
    for tool in (
        RefuselessAccountBalanceTool(cache_ttl=0),
        RefuselessAccountBalanceTool(api_key=KEY, cache_ttl=0),  # its host would be AI_BASE_URL's
        RefuselessAccountBalanceTool(base_url=HOST, cache_ttl=0),
    ):
        with respx.mock(assert_all_mocked=True) as mock:
            result = tool.run()
            assert not mock.calls
        assert result.startswith("refuseless account balance unavailable — AI_PROVIDER is ")
        assert "another-vendors-key" not in result


def test_a_caller_that_hands_over_both_reads_nothing_from_the_environment(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    tool = RefuselessAccountBalanceTool(api_key=KEY, base_url=HOST, cache_ttl=0)
    result, route = _read(tool, httpx.Response(200, json=_credits()))
    assert result.startswith("refuseless credit available:")
    assert route.calls.last.request.headers["authorization"] == f"Bearer {KEY}"


def test_an_unusable_base_url_is_a_soft_config_fault(monkeypatch):
    monkeypatch.setenv("AI_BASE_URL", "http://[::1")
    with respx.mock(assert_all_mocked=True):
        result = RefuselessAccountBalanceTool(api_key=KEY, cache_ttl=0).run()
    assert result == (
        "refuseless account balance unavailable — AI_BASE_URL is not a usable URL, so there is "
        "no host to read the account at."
    )


def test_no_key_is_a_config_fault_and_sends_nothing():
    tool = RefuselessAccountBalanceTool(cache_ttl=0)
    with respx.mock(assert_all_mocked=True) as mock:
        result = tool.run()
        assert not mock.calls
    assert result.startswith("refuseless account balance unavailable — AI_API_KEY is not")


def test_a_pasted_non_ascii_key_is_a_soft_config_fault():
    tool = RefuselessAccountBalanceTool(api_key="sk-rl-“pasted”", cache_ttl=0)
    with respx.mock(assert_all_mocked=True):
        result = tool.run()
    assert "non-ASCII character" in result


# --- every failure is soft, and says nothing it shouldn't ---------------------------------------


@pytest.mark.parametrize(
    ("response", "says"),
    [
        (_error(401, "invalid_key"), "refuseless rejected the key (HTTP 401). Check AI_API_KEY."),
        (_error(429, "rate_limited"), "rate-limiting balance reads (HTTP 429"),
        (
            _error(503, "gateway_down"),
            "refuseless's billing service is unreachable (HTTP 503, gateway_down).",
        ),
        (_error(503, "upstream_error"), "refuseless returned HTTP 503 reading the credit balance."),
        (_error(404, "not_found"), "refuseless returned HTTP 404 reading the credit balance."),
        (_error(504, "gateway_down"), "unreachable (HTTP 504, gateway_down)."),
        (httpx.Response(302, headers={"location": "https://elsewhere.example/"}), "HTTP 302"),
        (httpx.Response(200, text="<!DOCTYPE html>"), "an unreadable response"),
        (httpx.Response(200, json=[1, 2]), "an unexpected response"),
    ],
    ids=[
        "auth",
        "rate_limit",
        "gateway_down",
        "other_5xx",
        "4xx",
        "gateway_timeout",
        "redirect",
        "not_json",
        "not_object",
    ],
)
def test_a_failed_read_is_a_clear_reason_and_never_the_vendors_prose(tool, response, says):
    result, _ = _read(tool, response)
    assert result.startswith("refuseless account balance unavailable — ")
    assert says in result
    assert PROSE not in result
    assert KEY not in result


@pytest.mark.parametrize(
    ("error", "says"),
    [
        (httpx.ConnectError("refused"), "couldn't reach the refuseless API (ConnectError)."),
        (httpx.ReadTimeout("slow"), "the refuseless API did not answer in time."),
    ],
)
def test_an_unreachable_host_is_a_soft_failure(tool, error, says):
    result, _ = _read(tool, error)
    assert result == f"refuseless account balance unavailable — {says}"


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"balance": ...}, "balance"),
        ({"reserved": ...}, "reserved"),
        ({"available": ...}, "available"),
        ({"key_limit": ...}, "key_limit"),
        ({"updated_at": ...}, "updated_at"),
        ({"available": True}, "available"),
        ({"balance": "69.99"}, "balance"),
        ({"reserved": float("nan")}, "reserved"),
        ({"available": float("inf")}, "available"),
        ({"available": 10**400}, "available"),
        ({"key_limit": "none"}, "key_limit"),
        ({"updated_at": True}, "updated_at"),
        ({"updated_at": -1}, "updated_at"),
        ({"updated_at": 1794268661.5}, "updated_at"),
        ({"updated_at": 10**20}, "updated_at"),
    ],
)
def test_a_figure_the_vendor_did_not_state_usably_is_unavailable_never_guessed(
    tool, overrides, missing
):
    result, _ = _read(tool, _ok(_credits(**overrides)))
    assert result.startswith("refuseless account balance unavailable — ")
    assert f"carried no usable {missing}" in result
    assert "$" not in result


@pytest.mark.parametrize(
    "overrides",
    [{"unit": "eur"}, {"unit": ...}, {"object": "list"}, {"object": ...}],
)
def test_a_response_that_is_not_dollar_credits_is_unavailable(tool, overrides):
    result, _ = _read(tool, httpx.Response(200, json=_credits(**overrides)))
    assert result.startswith("refuseless account balance unavailable — ")
    assert "$" not in result


# --- the tool line ------------------------------------------------------------------------------


def _line(tool, caplog, response) -> str:
    registry = ToolRegistry(policy=Policy.locked())
    registry.register(tool)
    with respx.mock(assert_all_mocked=True) as mock:
        if isinstance(response, httpx.Response):
            mock.get(CREDITS_URL).mock(return_value=response)
        elif response is not None:
            mock.get(CREDITS_URL).mock(side_effect=response)
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            Engine(provider=None, tools=registry)._run_tool(tool.name, {})  # type: ignore[arg-type]
    (line,) = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("tool ") and "outcome=" in r.getMessage()
    ]
    return re.sub(r"\x1b\[[0-9;]*m", "", line)  # the verdict token is colored


@pytest.mark.parametrize(
    ("response", "outcome"),
    [
        (httpx.Response(200, json=_credits()), "outcome=ok"),
        (_error(401, "invalid_key"), "outcome=error reason=config:auth"),
        (_error(429, "rate_limited"), "outcome=error reason=rate_limited"),
        (_error(503, "gateway_down"), "outcome=error reason=server_error"),
        (_error(402, "insufficient_credits"), "outcome=error reason=config:billing"),
        (_error(404, "not_found"), "outcome=error reason=config:model_not_found"),
        (_error(413, "too_large"), "outcome=error reason=payload_too_large"),
        (_error(504, "gateway_down"), "outcome=error reason=timeout"),
        (_error(500, "upstream_error"), "outcome=error reason=server_error"),
        (_error(418, "teapot"), "outcome=error reason=api_error"),
        (httpx.Response(302), "outcome=error reason=api_error"),
        (httpx.Response(200, json=_credits(unit="eur")), "outcome=error reason=invalid_response"),
        (httpx.ConnectError("refused"), "outcome=error reason=transport"),
        (httpx.ReadTimeout("slow"), "outcome=error reason=timeout"),
    ],
    ids=[
        "ok",
        "auth",
        "rate_limit",
        "gateway_down",
        "402",
        "404",
        "413",
        "504",
        "500",
        "418",
        "302",
        "shape",
        "transport",
        "read_timeout",
    ],
)
def test_one_tool_line_per_call_with_its_verdict(caplog, response, outcome):
    line = _line(RefuselessAccountBalanceTool(api_key=KEY, cache_ttl=0), caplog, response)
    assert line.startswith("tool name=refuseless_account_balance executor=harness ")
    assert outcome in line
    if outcome == "outcome=ok":
        assert "reason=" not in line


def test_a_missing_key_is_a_config_fault_on_the_line(caplog):
    line = _line(RefuselessAccountBalanceTool(cache_ttl=0), caplog, None)
    assert line.endswith("outcome=error reason=config:missing_api_key")


# --- caching ------------------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_fresh_figure_is_reused_and_a_stale_one_reread(monkeypatch):
    clock = _Clock()
    tool = RefuselessAccountBalanceTool(api_key=KEY, cache_ttl=30, clock=clock)
    with respx.mock(assert_all_mocked=True) as mock:
        route = mock.get(CREDITS_URL).mock(return_value=httpx.Response(200, json=_credits()))
        first = tool.run()
        clock.now += 29
        assert tool.run() == first
        assert route.call_count == 1
        clock.now += 2
        tool.run()
        assert route.call_count == 2


def test_a_cached_figure_is_never_served_for_another_key_or_host(monkeypatch):
    clock = _Clock()
    tool = RefuselessAccountBalanceTool(cache_ttl=30, clock=clock)
    monkeypatch.setenv("AI_API_KEY", KEY)
    with respx.mock(assert_all_mocked=True) as mock:
        default = mock.get(CREDITS_URL).mock(return_value=httpx.Response(200, json=_credits()))
        other = mock.get("https://eu.refuseless.example/v1/credits").mock(
            return_value=httpx.Response(200, json=_credits(available=1.0))
        )
        tool.run()
        monkeypatch.setenv("AI_API_KEY", "sk-rl-ffffffffffffffffffffffffffffffff")
        tool.run()
        assert default.call_count == 2
        monkeypatch.setenv("AI_BASE_URL", "https://eu.refuseless.example/v1")
        assert tool.run().startswith("refuseless credit available: $1.00 USD")
        assert other.call_count == 1


def test_the_cache_never_holds_the_key(monkeypatch):
    """The #599 leak: a plain attribute holding the key is emitted by `vars()` and crash reporters."""
    monkeypatch.setenv("AI_API_KEY", KEY)
    for tool in (RefuselessAccountBalanceTool(api_key=KEY), RefuselessAccountBalanceTool()):
        with respx.mock(assert_all_mocked=True) as mock:
            mock.get(CREDITS_URL).mock(return_value=httpx.Response(200, json=_credits()))
            tool.run()
        assert tool._cached is not None
        assert KEY not in repr(vars(tool))
        assert KEY.encode() not in b"".join(
            part for part in tool._cached if isinstance(part, bytes)
        )


def test_a_failure_is_never_cached():
    clock = _Clock()
    tool = RefuselessAccountBalanceTool(api_key=KEY, cache_ttl=30, clock=clock)
    with respx.mock(assert_all_mocked=True) as mock:
        route = mock.get(CREDITS_URL).mock(
            side_effect=[_error(503, "gateway_down"), httpx.Response(200, json=_credits())]
        )
        assert "unavailable" in tool.run()
        assert tool.run().startswith("refuseless credit available:")
        assert route.call_count == 2


def test_the_stamp_is_the_vendors_read_time_not_ours(tool):
    stamp = datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)
    body = _credits(updated_at=int(stamp.timestamp()))
    result, _ = _read(tool, httpx.Response(200, json=body))
    assert "(as of 2026-11-10T12:00:00Z)." in result
