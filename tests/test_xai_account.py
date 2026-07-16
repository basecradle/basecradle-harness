"""The xai_account_balance tool: the balance read, team discovery, degraded modes, caching.

All HTTP is mocked with respx — no test reaches the network. The payload shapes and the
(inverted) sign convention mirror the real xAI Management API, verified live against a real
account in issue #179: `total.val` is a string of USD cents whose sign is inverted, so a stored
`-4250` is `$42.50` of *available* credit. (The figures here are fabricated — no real account
data lives in the repo; the live smoke test hits the real endpoint.)
"""

from __future__ import annotations

import httpx
import pytest
import respx

from basecradle_harness import Policy, ToolRegistry, XaiAccountBalanceTool
from basecradle_harness._xai_account import DEFAULT_BASE_URL

# A fabricated, well-formed UUIDv7 standing in for the agent's team — never a real team id.
TEAM = "019510a0-2b3c-7d4e-8f01-23456789abcd"
KEY = "xai-mgmt-fake-key-000"
VALIDATE_URL = f"{DEFAULT_BASE_URL}/auth/management-keys/validation"
BALANCE_URL = f"{DEFAULT_BASE_URL}/v1/billing/teams/{TEAM}/prepaid/balance"


def _balance_body(total_cents: str) -> dict:
    """A prepaid-balance response in the real shape: total + a reconciling changes ledger.

    Fabricated figures: a $50 purchase (stored negative) and a $7.50 spend (positive) net to
    -4250 cents = $42.50 available — the inverted-sign shape, none of it real account data.
    """
    return {
        "total": {"val": total_cents},
        "changes": [
            {"changeOrigin": "PURCHASE", "amount": {"val": "-5000"}, "topupStatus": "SUCCEEDED"},
            {"changeOrigin": "SPEND", "amount": {"val": "750"}},
        ],
    }


def _validation_body(team: str = TEAM) -> dict:
    """A management-key validation response — the shape the tool reads `teamId` out of."""
    return {"teamId": team, "scopeId": team, "acls": ["team-token:endpoint:BillingRead"]}


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """No ambient credential leaks into a test — each drives the key/team explicitly."""
    monkeypatch.delenv("XAI_MANAGEMENT_KEY", raising=False)
    monkeypatch.delenv("XAI_TEAM_ID", raising=False)


@pytest.fixture
def tool():
    # An explicit team id skips the discovery call; caching off unless a test asks for it.
    return XaiAccountBalanceTool(management_key=KEY, team_id=TEAM, cache_ttl=0)


# --- the happy path & the sign convention ------------------------------------


def test_reads_the_balance_and_inverts_the_cents_sign(tool):
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(BALANCE_URL).mock(
            return_value=httpx.Response(200, json=_balance_body("-4250"))
        )
        result = tool.run()

    assert "$42.50" in result  # -(-4250)/100, the *available* balance — not -$42.50
    assert "-$42.50" not in result
    assert "xAI prepaid credit balance" in result
    assert "as of" in result  # the timestamp stamp is present
    # The Authorization header carried the key; the result never does.
    assert route.calls.last.request.headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in result


def test_a_whole_dollar_balance_formats_cleanly(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(BALANCE_URL).mock(return_value=httpx.Response(200, json=_balance_body("-5000")))
        assert "$50.00" in tool.run()


def test_an_overdrawn_account_is_reported_as_negative(tool):
    # A positive stored total means the account has gone into the red (spent past its credit).
    with respx.mock(assert_all_called=True) as mock:
        mock.get(BALANCE_URL).mock(return_value=httpx.Response(200, json=_balance_body("500")))
        result = tool.run()

    assert "-$5.00" in result
    assert "overdrawn" in result


# --- team discovery ----------------------------------------------------------


def test_discovers_the_team_from_the_key_when_no_override():
    tool = XaiAccountBalanceTool(management_key=KEY, cache_ttl=0)  # no team_id → discover it
    with respx.mock(assert_all_called=True) as mock:
        validate = mock.get(VALIDATE_URL).mock(
            return_value=httpx.Response(200, json=_validation_body())
        )
        mock.get(BALANCE_URL).mock(return_value=httpx.Response(200, json=_balance_body("-4250")))
        result = tool.run()

    assert "$42.50" in result
    assert validate.called  # the team was resolved from the key, not assumed


def test_an_explicit_team_id_skips_discovery(tool):
    # assert_all_called is off: the validation route is registered precisely to prove it is NOT hit.
    with respx.mock(assert_all_called=False) as mock:
        validate = mock.get(VALIDATE_URL).mock(return_value=httpx.Response(200))
        balance = mock.get(BALANCE_URL).mock(
            return_value=httpx.Response(200, json=_balance_body("-4250"))
        )
        tool.run()

    assert balance.called  # the balance was read directly
    assert not validate.called  # the override means the validation call is never made


def test_team_id_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("XAI_MANAGEMENT_KEY", KEY)
    monkeypatch.setenv("XAI_TEAM_ID", TEAM)
    tool = XaiAccountBalanceTool(cache_ttl=0)  # both from env
    with respx.mock(assert_all_called=True) as mock:
        mock.get(BALANCE_URL).mock(return_value=httpx.Response(200, json=_balance_body("-4250")))
        assert "$42.50" in tool.run()


def test_discovery_without_a_team_in_the_response_degrades():
    tool = XaiAccountBalanceTool(management_key=KEY, cache_ttl=0)
    with respx.mock(assert_all_called=True) as mock:
        mock.get(VALIDATE_URL).mock(return_value=httpx.Response(200, json={"scopeId": "x"}))
        result = tool.run()

    assert "unavailable" in result
    assert "XAI_TEAM_ID" in result  # tells the operator how to bypass discovery


# --- degraded modes (the DoD's required failures) ----------------------------


def test_no_key_is_a_clean_unavailable_not_a_crash():
    tool = XaiAccountBalanceTool(cache_ttl=0)  # nothing configured
    result = tool.run()  # makes no HTTP call at all
    assert "unavailable" in result
    assert "XAI_MANAGEMENT_KEY" in result


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_names_the_scope_and_never_echoes_the_key(tool, status):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(BALANCE_URL).mock(return_value=httpx.Response(status, text="forbidden: bad key"))
        result = tool.run()

    assert "unavailable" in result
    assert "BillingRead" in result  # points at the missing read-only billing scope
    assert KEY not in result  # the key is never surfaced


def test_an_endpoint_error_status_degrades(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(BALANCE_URL).mock(return_value=httpx.Response(500, text="boom"))
        result = tool.run()

    assert "unavailable" in result
    assert "500" in result
    assert "boom" not in result  # the raw body is never relayed


def test_an_unreachable_endpoint_degrades(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(BALANCE_URL).mock(side_effect=httpx.ConnectError("no route"))
        result = tool.run()

    assert "unavailable" in result
    assert "couldn't reach" in result


def test_a_missing_total_degrades(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(BALANCE_URL).mock(return_value=httpx.Response(200, json={"changes": []}))
        assert "unavailable" in tool.run()


def test_a_non_numeric_balance_degrades(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(BALANCE_URL).mock(return_value=httpx.Response(200, json={"total": {"val": "abc"}}))
        assert "unavailable" in tool.run()


def test_the_raw_changes_ledger_never_reaches_the_model(tool):
    # The security invariant: only the computed figure leaves the tool — not the purchase/invoice
    # history the payload carries.
    body = _balance_body("-4250")
    body["changes"][0]["invoiceNumber"] = "INV-SECRET-42"
    with respx.mock(assert_all_called=True) as mock:
        mock.get(BALANCE_URL).mock(return_value=httpx.Response(200, json=body))
        result = tool.run()

    assert "$42.50" in result
    assert "INV-SECRET-42" not in result
    assert "PURCHASE" not in result


# --- caching -----------------------------------------------------------------


class _FakeClock:
    """A hand-advanced monotonic clock, so cache expiry is deterministic (no sleeping)."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_a_second_call_within_the_ttl_is_served_from_cache():
    clock = _FakeClock()
    tool = XaiAccountBalanceTool(management_key=KEY, team_id=TEAM, cache_ttl=30, clock=clock)
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(BALANCE_URL).mock(
            return_value=httpx.Response(200, json=_balance_body("-4250"))
        )
        first = tool.run()
        clock.t += 10  # still inside the 30s window
        second = tool.run()

    assert first == second
    assert route.call_count == 1  # the second call never hit the network


def test_the_cache_expires_after_the_ttl():
    clock = _FakeClock()
    tool = XaiAccountBalanceTool(management_key=KEY, team_id=TEAM, cache_ttl=30, clock=clock)
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(BALANCE_URL).mock(
            return_value=httpx.Response(200, json=_balance_body("-4250"))
        )
        tool.run()
        clock.t += 31  # past the window
        tool.run()

    assert route.call_count == 2  # a stale balance is re-read


def test_ttl_zero_disables_caching(tool):
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(BALANCE_URL).mock(
            return_value=httpx.Response(200, json=_balance_body("-4250"))
        )
        tool.run()
        tool.run()

    assert route.call_count == 2  # cache_ttl=0 → every call re-reads


# --- it is a safe, locked-profile tool ---------------------------------------


def test_loads_under_the_locked_default_profile():
    # A plain read-only tool: no policy capability, so it registers under the shipped safe policy.
    registry = ToolRegistry(Policy.locked())
    registry.register(XaiAccountBalanceTool())
    assert "xai_account_balance" in registry
    assert XaiAccountBalanceTool().parameters == {"type": "object", "properties": {}}
