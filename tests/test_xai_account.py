"""The xai_account_balance tool: the live figure, the ledger's role, degraded modes, caching.

All HTTP is mocked with respx — no test reaches the network. The payload shapes and the
(inverted) sign convention mirror the real xAI Management API, verified live against a real
account in issues #179 and #384: a figure lives at `<field>.val` as a string of USD cents whose
sign is inverted for *credit*, so a stored `-4250` is `$42.50` of credit.

The central test here is `test_the_live_figure_wins_when_the_ledger_disagrees` — the #384
regression. The posted prepaid ledger settles only at cycle close, so mid-cycle it reports
credit already spent; the tool must lead with the preview's live remaining, never the ledger's
larger, staler total. (The figures are fabricated — no real account data lives in the repo; the
live smoke test hits the real endpoint.)
"""

from __future__ import annotations

import re

import httpx
import pytest
import respx

from basecradle_harness import Policy, ToolRegistry, XaiAccountBalanceTool
from basecradle_harness._xai_account import DEFAULT_BASE_URL

# A fabricated, well-formed UUIDv7 standing in for the agent's team — never a real team id.
TEAM = "019510a0-2b3c-7d4e-8f01-23456789abcd"
KEY = "xai-mgmt-fake-key-000"
VALIDATE_URL = f"{DEFAULT_BASE_URL}/auth/management-keys/validation"
PREVIEW_URL = f"{DEFAULT_BASE_URL}/v1/billing/teams/{TEAM}/postpaid/invoice/preview"
LEDGER_URL = f"{DEFAULT_BASE_URL}/v1/billing/teams/{TEAM}/prepaid/balance"


def _preview_body(
    remaining_cents: str = "-11847",
    usage_cents: str | None = "7138",
    prepaid_used_cents: str | None = "-7138",
) -> dict:
    """A postpaid invoice preview in the real shape — the live mid-cycle picture.

    Fabricated figures in the *live* proportions from issue #384: $118.47 of prepaid still
    available, $71.38 of usage this cycle, all of it drawn from prepaid. Credit figures are
    stored negative; `totalWithCorr` is a spend figure and is positive.
    """
    core: dict = {"prepaidCredits": {"val": remaining_cents}}
    if usage_cents is not None:
        core["totalWithCorr"] = {"val": usage_cents}
    if prepaid_used_cents is not None:
        core["prepaidCreditsUsed"] = {"val": prepaid_used_cents}
    return {"coreInvoice": core}


def _ledger_body(total_cents: str = "-51749") -> dict:
    """A prepaid-balance response in the real shape: total + a reconciling changes ledger.

    Fabricated figures: purchases (stored negative) far outweighing the two *posted* spend rows,
    which is exactly the #384 shape — the ledger's $517.49 lags the cycle's real burn.
    """
    return {
        "total": {"val": total_cents},
        "changes": [
            {"changeOrigin": "PURCHASE", "amount": {"val": "-52500"}, "topupStatus": "SUCCEEDED"},
            {"changeOrigin": "SPEND", "amount": {"val": "751"}},
        ],
    }


def _validation_body(team: str = TEAM) -> dict:
    """A management-key validation response — the shape the tool reads `teamId` out of."""
    return {"teamId": team, "scopeId": team, "acls": ["team-token:endpoint:BillingRead"]}


def _mock_both(mock, *, preview=None, ledger=None):
    """Route both billing calls; each defaults to the healthy fabricated body."""
    preview_route = mock.get(PREVIEW_URL).mock(
        return_value=preview if preview is not None else httpx.Response(200, json=_preview_body())
    )
    ledger_route = mock.get(LEDGER_URL).mock(
        return_value=ledger if ledger is not None else httpx.Response(200, json=_ledger_body())
    )
    return preview_route, ledger_route


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """No ambient credential leaks into a test — each drives the key/team explicitly."""
    monkeypatch.delenv("XAI_MANAGEMENT_KEY", raising=False)
    monkeypatch.delenv("XAI_TEAM_ID", raising=False)


@pytest.fixture
def tool():
    # An explicit team id skips the discovery call; caching off unless a test asks for it.
    return XaiAccountBalanceTool(management_key=KEY, team_id=TEAM, cache_ttl=0)


# --- the #384 regression: live remaining, never the posted ledger -------------


def test_the_live_figure_wins_when_the_ledger_disagrees(tool):
    """Mid-cycle the ledger is stale and larger; the tool must lead with the live figure.

    These are the live proportions from the issue: the ledger says $517.49 while only $118.47
    is actually spendable. Before #384 the tool returned the $517.49.
    """
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock)
        result = tool.run()

    headline = result.splitlines()[0]
    assert headline.startswith("xAI credits remaining: $118.47 USD (as of ")
    assert "$517.49" not in headline  # the posted ledger is nowhere near the headline
    # ...and the ledger total is present only as labelled context, never as "remaining".
    assert "Posted prepaid ledger: $517.49" in result
    assert "not what you have left to spend" in result


def test_the_headline_never_labels_a_posted_total_as_remaining(tool):
    """The AC in words: no posted-only number is called available/remaining unqualified."""
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock)
        result = tool.run()

    remaining_line, context = result.split("\n", 1)
    assert "remaining" in remaining_line
    assert "$517.49" in context  # the stale figure lives *below* the fold...
    assert "remaining" not in context.split("Posted prepaid ledger")[1]  # ...and is never so named


def test_the_cycle_burn_is_reported_as_context(tool):
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock)
        result = tool.run()

    assert "This cycle: $71.38 used, $71.38 of it drawn from prepaid credit." in result


def test_a_preview_without_the_breakdown_still_answers(tool):
    # Only prepaidCredits is required — the rest is context the render simply omits.
    body = _preview_body(usage_cents=None, prepaid_used_cents=None)
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=body))
        result = tool.run()

    assert "xAI credits remaining: $118.47 USD" in result
    assert "This cycle:" not in result


def test_usage_beyond_prepaid_reports_both_figures(tool):
    # Once prepaid is exhausted the cycle's usage exceeds what prepaid covered — they diverge.
    body = _preview_body(remaining_cents="0", usage_cents="9000", prepaid_used_cents="-2500")
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=body))
        result = tool.run()

    assert "xAI credits remaining: $0.00 USD" in result
    assert "This cycle: $90.00 used, $25.00 of it drawn from prepaid credit." in result


# --- the sign convention ------------------------------------------------------


def test_the_credit_sign_is_inverted_not_absolute(tool):
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=_preview_body(remaining_cents="-4250")))
        result = tool.run()

    assert "$42.50" in result  # -(-4250)/100, the *available* credit — not -$42.50
    assert "xAI credits remaining: $42.50" in result


def test_a_whole_dollar_figure_formats_cleanly(tool):
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=_preview_body(remaining_cents="-5000")))
        assert "xAI credits remaining: $50.00 USD" in tool.run()


def test_an_overdrawn_account_is_reported_as_negative_never_absolute(tool):
    """A positive stored value means overdrawn. `abs()` here would invert the whole warning."""
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=_preview_body(remaining_cents="500")))
        result = tool.run()

    assert result.startswith("xAI credits remaining: -$5.00 USD (as of ")
    assert "The account is overdrawn." in result
    assert "xAI credits remaining: $5.00" not in result  # never rendered as healthy credit


@pytest.mark.parametrize(
    ("preview", "label"),
    [
        pytest.param(None, "xAI credits remaining", id="live"),
        pytest.param(httpx.Response(500), "xAI posted prepaid ledger total", id="fallback"),
    ],
)
@pytest.mark.parametrize("cents", ["-11847", "500"])
def test_the_headline_shape_is_uniform_across_renders_and_signs(tool, preview, label, cents):
    """The figure sits in the same place whatever happened — no sometimes-there parenthetical.

    An overdraft is stated in the body instead, so `<label>: <$figure> USD (as of <stamp>).` holds
    for the live figure, the ledger fallback, credit, and overdraft alike.
    """
    body = _preview_body(remaining_cents=cents)
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(
            mock,
            preview=preview if preview is not None else httpx.Response(200, json=body),
            ledger=httpx.Response(200, json=_ledger_body(total_cents=cents)),
        )
        headline = tool.run().splitlines()[0]

    assert re.fullmatch(rf"{re.escape(label)}: -?\$[\d,]+\.\d{{2}} USD \(as of \S+Z\)\.", headline)


def test_the_key_never_leaks_and_the_header_carries_it(tool):
    with respx.mock(assert_all_called=True) as mock:
        preview, _ = _mock_both(mock)
        result = tool.run()

    assert preview.calls.last.request.headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in result


# --- team discovery ----------------------------------------------------------


def test_discovers_the_team_from_the_key_when_no_override():
    tool = XaiAccountBalanceTool(management_key=KEY, cache_ttl=0)  # no team_id → discover it
    with respx.mock(assert_all_called=True) as mock:
        validate = mock.get(VALIDATE_URL).mock(
            return_value=httpx.Response(200, json=_validation_body())
        )
        _mock_both(mock)
        result = tool.run()

    assert "$118.47" in result
    assert validate.called  # the team was resolved from the key, not assumed


def test_an_explicit_team_id_skips_discovery(tool):
    # assert_all_called is off: the validation route is registered precisely to prove it is NOT hit.
    with respx.mock(assert_all_called=False) as mock:
        validate = mock.get(VALIDATE_URL).mock(return_value=httpx.Response(200))
        preview, _ = _mock_both(mock)
        tool.run()

    assert preview.called  # the figures were read directly
    assert not validate.called  # the override means the validation call is never made


def test_team_id_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("XAI_MANAGEMENT_KEY", KEY)
    monkeypatch.setenv("XAI_TEAM_ID", TEAM)
    tool = XaiAccountBalanceTool(cache_ttl=0)  # both from env
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock)
        assert "$118.47" in tool.run()


def test_discovery_without_a_team_in_the_response_degrades():
    tool = XaiAccountBalanceTool(management_key=KEY, cache_ttl=0)
    with respx.mock(assert_all_called=True) as mock:
        mock.get(VALIDATE_URL).mock(return_value=httpx.Response(200, json={"scopeId": "x"}))
        result = tool.run()

    assert "unavailable" in result
    assert "XAI_TEAM_ID" in result  # tells the operator how to bypass discovery


# --- the ledger as fallback: labelled, never passed off as live ---------------


@pytest.mark.parametrize(
    "preview",
    [
        pytest.param(httpx.Response(500, text="boom"), id="preview-5xx"),
        pytest.param(httpx.Response(404, text="gone"), id="preview-404"),
        pytest.param(httpx.Response(200, json={}), id="preview-no-coreInvoice"),
        pytest.param(
            httpx.Response(200, json={"coreInvoice": {"totalWithCorr": {"val": "7138"}}}),
            id="preview-no-prepaidCredits",
        ),
        pytest.param(
            httpx.Response(200, json=_preview_body(remaining_cents="not-a-number")),
            id="preview-non-numeric",
        ),
    ],
)
def test_an_unusable_preview_falls_back_to_a_labelled_ledger(tool, preview):
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=preview)
        result = tool.run()

    assert "$517.49" in result
    assert "NOT your remaining credit" in result  # the #384 invariant, in the model's face
    assert "upper bound" in result
    assert result.startswith("xAI posted prepaid ledger total:")  # labelled at the headline
    assert "boom" not in result and "gone" not in result  # raw bodies never relayed


def test_the_fallback_names_why_the_live_figure_was_missing(tool):
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(503))
        result = tool.run()

    assert "503" in result
    assert "read the live credits remaining" in result  # the action that failed, named


def test_a_healthy_preview_survives_a_broken_ledger(tool):
    # The ledger is context, not a dependency: losing it must not lose the answer.
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, ledger=httpx.Response(500, text="boom"))
        result = tool.run()

    assert "xAI credits remaining: $118.47 USD" in result
    assert "Posted prepaid ledger" not in result  # simply omitted, not faked
    assert "unavailable" not in result


def test_both_surfaces_failing_is_a_clean_unavailable(tool):
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(500), ledger=httpx.Response(500))
        result = tool.run()

    assert result.startswith("xAI account balance unavailable — ")
    assert "500" in result


# --- degraded modes (the DoD's required failures) ----------------------------


def test_no_key_is_a_clean_unavailable_not_a_crash():
    tool = XaiAccountBalanceTool(cache_ttl=0)  # nothing configured
    result = tool.run()  # makes no HTTP call at all
    assert "unavailable" in result
    assert "XAI_MANAGEMENT_KEY" in result


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_names_the_scope_and_never_echoes_the_key(tool, status):
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(
            mock,
            preview=httpx.Response(status, text="forbidden: bad key"),
            ledger=httpx.Response(status, text="forbidden: bad key"),
        )
        result = tool.run()

    assert "unavailable" in result
    assert "BillingRead" in result  # points at the missing read-only billing scope
    assert KEY not in result  # the key is never surfaced


def test_an_unreachable_endpoint_degrades(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(PREVIEW_URL).mock(side_effect=httpx.ConnectError("no route"))
        mock.get(LEDGER_URL).mock(side_effect=httpx.ConnectError("no route"))
        result = tool.run()

    assert "unavailable" in result
    assert "couldn't reach" in result


def test_an_unreadable_response_degrades(tool):
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(
            mock,
            preview=httpx.Response(200, text="not json"),
            ledger=httpx.Response(200, text="not json"),
        )
        result = tool.run()

    assert "unavailable" in result
    assert "unreadable" in result


def test_a_json_list_response_degrades(tool):
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=[]), ledger=httpx.Response(200, json=[]))
        assert "unavailable" in tool.run()


@pytest.mark.parametrize(
    "ledger",
    [
        pytest.param(httpx.Response(200, json={"changes": []}), id="ledger-no-total"),
        pytest.param(httpx.Response(200, json={"total": {"val": "abc"}}), id="ledger-non-numeric"),
    ],
)
def test_an_unusable_ledger_alongside_an_unusable_preview_degrades(tool, ledger):
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(500), ledger=ledger)
        assert "unavailable" in tool.run()


def test_the_raw_ledger_history_never_reaches_the_model(tool):
    # The security invariant: only computed figures leave the tool — not the purchase/invoice
    # history the payload carries.
    body = _ledger_body()
    body["changes"][0]["invoiceNumber"] = "INV-SECRET-42"
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, ledger=httpx.Response(200, json=body))
        result = tool.run()

    assert "$118.47" in result
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
        preview, ledger = _mock_both(mock)
        first = tool.run()
        clock.t += 10  # still inside the 30s window
        second = tool.run()

    assert first == second
    assert preview.call_count == 1  # the second call never hit the network
    assert ledger.call_count == 1


def test_the_cache_expires_after_the_ttl():
    clock = _FakeClock()
    tool = XaiAccountBalanceTool(management_key=KEY, team_id=TEAM, cache_ttl=30, clock=clock)
    with respx.mock(assert_all_called=True) as mock:
        preview, _ = _mock_both(mock)
        tool.run()
        clock.t += 31  # past the window
        tool.run()

    assert preview.call_count == 2  # a stale figure is re-read


def test_ttl_zero_disables_caching(tool):
    with respx.mock(assert_all_called=True) as mock:
        preview, _ = _mock_both(mock)
        tool.run()
        tool.run()

    assert preview.call_count == 2  # cache_ttl=0 → every call re-reads


def test_an_unavailable_result_is_never_cached():
    # A transient outage must not pin "unavailable" for the whole TTL — the next call retries.
    clock = _FakeClock()
    tool = XaiAccountBalanceTool(management_key=KEY, team_id=TEAM, cache_ttl=30, clock=clock)
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(500), ledger=httpx.Response(500))
        assert "unavailable" in tool.run()
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock)  # the outage clears; the clock has not moved
        assert "xAI credits remaining: $118.47 USD" in tool.run()


# --- it is a safe, locked-profile tool ---------------------------------------


def test_loads_under_the_locked_default_profile():
    # A plain read-only tool: no policy capability, so it registers under the shipped safe policy.
    registry = ToolRegistry(Policy.locked())
    registry.register(XaiAccountBalanceTool())
    assert "xai_account_balance" in registry
    assert XaiAccountBalanceTool().parameters == {"type": "object", "properties": {}}
