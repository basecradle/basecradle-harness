"""The xai_account_balance tool: the live figure, the ledger's role, degraded modes, caching.

All HTTP is mocked with respx — no test reaches the network. The payload shapes and the
(inverted) sign convention mirror the real xAI Management API, verified live against a real
account in issues #179, #384 and #388: a figure lives at `<field>.val` as a string of USD cents
whose sign is inverted for *credit*, so a stored `-4250` is `$42.50` of credit.

Two regressions are pinned here, and they are the same mistake at different depths — a figure
this account can show being mistaken for the runway:

- `test_the_live_figure_wins_when_the_ledger_disagrees` (#384): the posted prepaid ledger settles
  only at cycle close, so mid-cycle it reports credit already spent. The tool must lead with the
  preview, never the ledger's larger, staler total.
- `test_the_live_figure_nets_the_cycles_prepaid_draw` (#388): the preview's `prepaidCredits` is
  the credit the cycle draws *against*, not what is left of it. The tool must lead with
  `prepaidCredits − prepaidCreditsUsed`, never the first field alone.

A third thing pinned here belongs to the *live* suite rather than to the tool: that suite's #388
assertion runs against an account which never stops spending, so the last two tests in this file
drive the real live function against a preview whose draw advances between reads — once proving it
survives the drift, once proving it still catches the #388 defect through it. A live-marked test
cannot prove its own race-freedom (it needs a real key, and the race is a coincidence of timing), so
the proof lives offline, beside the fixtures that can produce the condition on demand (issue #450).

The fixture bodies are one coherent (fabricated) snapshot in the live proportions of 2026-08-02
21:14 UTC, when the Console showed **$52.14** remaining: `$168.47` of prepaid credit, `$116.66`
drawn this cycle → **$51.81** remaining, against a `$567.49` posted ledger. Those proportions are
the point — no real account data lives in the repo, and the live smoke test hits the real endpoint.
"""

from __future__ import annotations

import re

import httpx
import pytest
import respx

from basecradle_harness import Policy, ToolRegistry, XaiAccountBalanceTool
from basecradle_harness._xai_account import DEFAULT_BASE_URL, _dollars
from tests import test_xai_account_live as live

# A fabricated, well-formed UUIDv7 standing in for the agent's team — never a real team id.
TEAM = "019510a0-2b3c-7d4e-8f01-23456789abcd"
KEY = "xai-mgmt-fake-key-000"
VALIDATE_URL = f"{DEFAULT_BASE_URL}/auth/management-keys/validation"
PREVIEW_URL = f"{DEFAULT_BASE_URL}/v1/billing/teams/{TEAM}/postpaid/invoice/preview"
LEDGER_URL = f"{DEFAULT_BASE_URL}/v1/billing/teams/{TEAM}/prepaid/balance"

# `prepaidCreditsUsed` advancing $0.04 a read: half the fleet draw measured on the NOC prober box
# (2026-08-31, $0.08 in 32 s) and four times the $0.01 the live suite used to compare against.
DRIFT = ["-46430", "-46434", "-46438"]


def _preview_body(
    prepaid_cents: str | None = "-16847",
    prepaid_used_cents: str | None = "-11666",
    usage_cents: str | None = "11666",
) -> dict:
    """A postpaid invoice preview in the real shape — the live mid-cycle picture.

    Fabricated figures in the *live* proportions of 2026-08-02 21:14 UTC (issue #388): $168.47 of
    prepaid credit for the cycle to draw against, $116.66 already drawn, $116.66 of usage in total
    — so $51.81 actually remaining. Credit figures are stored negative; `totalWithCorr` is a spend
    figure and is positive.
    """
    core: dict = {}
    if prepaid_cents is not None:
        core["prepaidCredits"] = {"val": prepaid_cents}
    if prepaid_used_cents is not None:
        core["prepaidCreditsUsed"] = {"val": prepaid_used_cents}
    if usage_cents is not None:
        core["totalWithCorr"] = {"val": usage_cents}
    return {"coreInvoice": core}


def _ledger_body(total_cents: str = "-56749") -> dict:
    """A prepaid-balance response in the real shape: total + a reconciling changes ledger.

    Fabricated figures: purchases (stored negative) far outweighing the two *posted* spend rows,
    which is exactly the #384 shape — the ledger's $567.49 lags the cycle's real burn.
    """
    return {
        "total": {"val": total_cents},
        "changes": [
            {"changeOrigin": "PURCHASE", "amount": {"val": "-57500"}, "topupStatus": "SUCCEEDED"},
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

    These are the live proportions from the issues: the ledger says $567.49 while only $51.81
    is actually spendable. Before #384 the tool returned the $567.49.
    """
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock)
        result = tool.run()

    headline = result.splitlines()[0]
    assert headline.startswith("xAI credits remaining: $51.81 USD (as of ")
    assert "$567.49" not in headline  # the posted ledger is nowhere near the headline
    # ...and the ledger total is present only as labelled context, never as "remaining".
    assert "Posted prepaid ledger: $567.49" in result
    assert "not what you have left to spend" in result


def test_the_headline_never_labels_a_posted_total_as_remaining(tool):
    """The AC in words: no posted-only number is called available/remaining unqualified."""
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock)
        result = tool.run()

    remaining_line, context = result.split("\n", 1)
    assert "remaining" in remaining_line
    assert "$567.49" in context  # the stale figure lives *below* the fold...
    assert "remaining" not in context.split("Posted prepaid ledger")[1]  # ...and is never so named


# --- the #388 regression: remaining nets the cycle's prepaid draw -------------


def test_the_live_figure_nets_the_cycles_prepaid_draw(tool):
    """`prepaidCredits` alone is the credit drawn *against*, never what is left of it.

    The live payload of 2026-08-02 21:14 UTC, when the Console showed $52.14 remaining: $168.47
    of prepaid credit less $116.66 drawn = $51.81. 0.96.0 reported the $168.47 — right surface,
    wrong arithmetic, ~3× high.
    """
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock)
        result = tool.run()

    assert result.startswith("xAI credits remaining: $51.81 USD (as of ")
    assert "xAI credits remaining: $168.47" not in result  # the undrawn figure is never the answer
    # The gross is shown — but only as a term of the subtraction that produced the headline.
    assert "$168.47 of prepaid credit less the $116.66" in result


def test_the_cycle_burn_is_reported_as_context(tool):
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock)
        result = tool.run()

    assert "This cycle: $116.66 used in total." in result


def test_a_preview_missing_the_draw_is_not_a_live_figure(tool):
    """Both terms are required: without the draw there is no remaining figure to report.

    The tempting "tolerant" parse — fall back to `prepaidCredits` when the draw is absent — *is*
    the #388 defect, so the tool degrades to the labelled ledger instead of guessing.
    """
    body = _preview_body(prepaid_used_cents=None)
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=body))
        result = tool.run()

    assert "$168.47" not in result  # never surfaced as a figure of any kind
    assert result.startswith("xAI posted prepaid ledger total: $567.49")
    assert "NOT your remaining credit" in result
    assert "prepaidCreditsUsed" in result  # the reason names the field that was missing


def test_the_total_spend_is_never_substituted_for_the_prepaid_draw(tool):
    """`totalWithCorr` runs past prepaid onto the postpaid invoice; it is a different quantity.

    Prepaid is exhausted here ($25 available, all $25 drawn) while the cycle spent $90 — the
    remaining credit is $0.00, not the phantom -$65.00 that subtracting total spend would give.
    """
    body = _preview_body(prepaid_cents="-2500", prepaid_used_cents="-2500", usage_cents="9000")
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=body))
        result = tool.run()

    assert result.startswith("xAI credits remaining: $0.00 USD")
    assert "The account is overdrawn." not in result  # exhausted is not overdrawn
    assert "$25.00 of prepaid credit less the $25.00" in result
    assert "This cycle: $90.00 used in total." in result


def test_a_preview_without_the_total_spend_still_answers(tool):
    # `totalWithCorr` is pure context — the subtraction that answers the question does not need it.
    body = _preview_body(usage_cents=None)
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=body))
        result = tool.run()

    assert "xAI credits remaining: $51.81 USD" in result
    assert "This cycle:" not in result


# --- the sign convention ------------------------------------------------------


def test_the_credit_sign_is_inverted_not_absolute(tool):
    body = _preview_body(prepaid_cents="-4250", prepaid_used_cents="0", usage_cents=None)
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=body))
        result = tool.run()

    assert "$42.50" in result  # -(-4250)/100, the *available* credit — not -$42.50
    assert "xAI credits remaining: $42.50" in result


def test_a_whole_dollar_figure_formats_cleanly(tool):
    body = _preview_body(prepaid_cents="-7500", prepaid_used_cents="-2500", usage_cents=None)
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=body))
        assert "xAI credits remaining: $50.00 USD" in tool.run()


def test_an_overdrawn_account_is_reported_as_negative_never_absolute(tool):
    """More prepaid drawn than the cycle had means overdrawn — and `abs()` would invert the warning.

    The subtraction is where an overdraft is *born* now (#388), not only where a stored sign is
    read: $1.00 of prepaid credit against $6.00 drawn from it is -$5.00, and rendering that as
    healthy credit is the one condition this tool exists to shout about.
    """
    body = _preview_body(prepaid_cents="-100", prepaid_used_cents="-600", usage_cents="600")
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=body))
        result = tool.run()

    assert result.startswith("xAI credits remaining: -$5.00 USD (as of ")
    assert "The account is overdrawn." in result
    assert "xAI credits remaining: $5.00" not in result  # never rendered as healthy credit


def test_a_positive_stored_prepaid_figure_is_still_an_overdraft(tool):
    """The stored sign convention holds through the subtraction: positive stored = overdrawn."""
    body = _preview_body(prepaid_cents="500", prepaid_used_cents="0", usage_cents=None)
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=httpx.Response(200, json=body))
        result = tool.run()

    assert result.startswith("xAI credits remaining: -$5.00 USD (as of ")
    assert "The account is overdrawn." in result


@pytest.mark.parametrize(
    ("preview", "label"),
    [
        pytest.param(None, "xAI credits remaining", id="live"),
        pytest.param(httpx.Response(500), "xAI posted prepaid ledger total", id="fallback"),
    ],
)
@pytest.mark.parametrize("cents", ["-16847", "500"])
def test_the_headline_shape_is_uniform_across_renders_and_signs(tool, preview, label, cents):
    """The figure sits in the same place whatever happened — no sometimes-there parenthetical.

    An overdraft is stated in the body instead, so `<label>: <$figure> USD (as of <stamp>).` holds
    for the live figure, the ledger fallback, credit, and overdraft alike.
    """
    body = _preview_body(prepaid_cents=cents)
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

    assert "xAI credits remaining: $51.81 USD" in result
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
    monkeypatch.setattr(live, "KEY", KEY)  # never build the oracle's header from an ambient key
    tool = XaiAccountBalanceTool(cache_ttl=0)  # both from env
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock)
        assert "xAI credits remaining: $51.81 USD" in tool.run()


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
            httpx.Response(200, json=_preview_body(prepaid_cents=None)),
            id="preview-no-prepaidCredits",
        ),
        pytest.param(
            httpx.Response(200, json=_preview_body(prepaid_used_cents=None)),
            id="preview-no-prepaidCreditsUsed",
        ),
        pytest.param(
            httpx.Response(200, json=_preview_body(prepaid_cents="not-a-number")),
            id="preview-non-numeric",
        ),
        pytest.param(
            httpx.Response(200, json=_preview_body(prepaid_used_cents="not-a-number")),
            id="preview-non-numeric-draw",
        ),
    ],
)
def test_an_unusable_preview_falls_back_to_a_labelled_ledger(tool, preview):
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=preview)
        result = tool.run()

    assert "$567.49" in result
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

    assert "xAI credits remaining: $51.81 USD" in result
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

    assert "xAI credits remaining: $51.81 USD" in result
    assert "INV-SECRET-42" not in result
    assert "PURCHASE" not in result


# --- run() never raises: the malformed-response cases -------------------------


def test_an_oversized_figure_degrades_instead_of_raising(tool):
    """A numeric string has no width limit; `cents / 100.0` does.

    A 400-digit `val` parses to a perfectly good Python `int` and then raises `OverflowError`
    at the division every caller performs — straight out of `run()`, which the module's whole
    contract says can never happen.
    """
    huge = "-" + "1" * 400
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(
            mock,
            preview=httpx.Response(200, json=_preview_body(prepaid_cents=huge)),
            ledger=httpx.Response(200, json=_ledger_body(total_cents=huge)),
        )
        result = tool.run()  # must not raise

    assert result.startswith("xAI account balance unavailable — ")
    assert "111" not in result  # nothing of the figure is surfaced


def test_a_deeply_nested_body_degrades_instead_of_raising(tool):
    # `json` raises `RecursionError`, not a `ValueError`, past a certain nesting depth.
    raw = "[" * 2000 + "]" * 2000
    nested = httpx.Response(200, content=raw, headers={"content-type": "application/json"})
    with respx.mock(assert_all_called=True) as mock:
        _mock_both(mock, preview=nested, ledger=nested)
        result = tool.run()  # must not raise

    assert result.startswith("xAI account balance unavailable — ")


def test_a_non_ascii_key_degrades_instead_of_raising():
    """httpx ASCII-encodes a header value at *client construction*, before any request exists.

    So a key carrying a smart quote or a non-breaking space — what pasting a credential actually
    produces — raised `UnicodeEncodeError` straight out of `run()`, past the `httpx.RequestError`
    guard, which never sees it. A misconfigured credential is the case this tool most owes a soft
    answer to.
    """
    tool = XaiAccountBalanceTool(
        management_key="xai-\u201cnot-ascii\u201d", team_id=TEAM, cache_ttl=0
    )
    with respx.mock(assert_all_called=False) as mock:
        preview, _ = _mock_both(mock)
        result = tool.run()  # must not raise

    assert not preview.called  # it never got as far as a request
    assert result.startswith("xAI account balance unavailable — ")
    assert "non-ASCII" in result
    assert "XAI_MANAGEMENT_KEY" in result
    assert "not-ascii" not in result  # the key is never echoed back


def test_the_renderer_never_prints_a_sign_on_the_wrong_side_of_the_dollar():
    """`-0.0 < 0` is `False`, so a negative zero would fall through `_dollars`' positive arm.

    Not reachable through the parse today — every figure is `-int / 100.0` and `-0 / 100.0` is
    `+0.0` — so this pins the renderer directly, which is the point: a future change to how a
    figure is derived must not silently arm `$-0.00`.
    """
    assert _dollars(-0.0) == "$0.00"
    assert _dollars(0.0) == "$0.00"
    assert _dollars(-5.25) == "-$5.25"
    assert _dollars(1234.5) == "$1,234.50"


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
        assert "xAI credits remaining: $51.81 USD" in tool.run()


# --- it is a safe, locked-profile tool ---------------------------------------


def test_loads_under_the_locked_default_profile():
    # A plain read-only tool: no policy capability, so it registers under the shipped safe policy.
    registry = ToolRegistry(Policy.locked())
    registry.register(XaiAccountBalanceTool())
    assert "xai_account_balance" in registry
    assert XaiAccountBalanceTool().parameters == {"type": "object", "properties": {}}


# --- the live suite's own race, pinned offline (issue #450) -------------------


def _a_drifting_preview(prepaid_cents: str, drawn_cents: list[str]):
    """A preview whose `prepaidCreditsUsed` advances on every read — a live account, spending.

    The one condition the live suite runs under and no fixture used to reproduce: three xAI
    personas draw on that account continuously, so two oracle reads seconds apart are two
    different numbers.
    """
    reads = iter(drawn_cents)
    last = drawn_cents[-1]

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal last
        last = next(reads, last)
        return httpx.Response(200, json=_preview_body(prepaid_cents, last))

    return respond


def test_the_live_bracket_survives_an_account_that_moves_between_reads(monkeypatch):
    """The live `test_the_live_figure_nets_the_cycles_prepaid_draw` passes while the account moves.

    The live suite's #388 assertion used to be an **equality** against a single oracle read, which
    compares two figures fetched at two different instants. Measured on the NOC prober box
    2026-08-31, ordinary fleet traffic moved `prepaidCreditsUsed` by $0.08 in 32 seconds — eight
    times the $0.01 tolerance — so the arm that would have run it daily was held disarmed rather
    than install a known-intermittent false page whose red is indistinguishable from the real
    Management-API drift it exists to catch (basecradle-noc#573).

    A live-marked test cannot prove its own race-freedom (it needs a real key, and the race is a
    coincidence of timing), so it is proven **here**, offline, by driving the real live function
    against a preview that advances between every read. `$0.04` per read is half the measured
    fleet rate and four times the old tolerance.
    """
    monkeypatch.setenv("XAI_MANAGEMENT_KEY", KEY)
    monkeypatch.setenv("XAI_TEAM_ID", TEAM)
    monkeypatch.setattr(live, "KEY", KEY)  # never build the oracle's header from an ambient key

    with respx.mock(assert_all_called=True) as mock:
        mock.get(PREVIEW_URL).mock(side_effect=_a_drifting_preview("-51847", DRIFT))
        mock.get(LEDGER_URL).mock(return_value=httpx.Response(200, json=_ledger_body()))
        live.test_the_live_figure_nets_the_cycles_prepaid_draw()


def test_the_live_bracket_still_catches_the_388_defect_while_the_account_moves(monkeypatch):
    """Widening to a bracket must not widen away the regression the assertion exists for.

    The adversarial half of the test above, and the one that makes the fix a fix rather than a
    loosened tolerance: under the *same* drifting account, a tool that reports `prepaidCredits`
    itself as the runway — issue #388 exactly — still fails. A few seconds of draw spans cents;
    the #388 defect overstates by the whole cycle's draw.
    """
    monkeypatch.setenv("XAI_MANAGEMENT_KEY", KEY)
    monkeypatch.setenv("XAI_TEAM_ID", TEAM)
    monkeypatch.setattr(live, "KEY", KEY)  # never build the oracle's header from an ambient key

    class _Regressed:
        """0.96.0's defect: the undrawn prepaid credit rendered as what is left to spend."""

        def __init__(self, **_kwargs):
            pass

        def run(self) -> str:
            return (
                "xAI credits remaining: $518.47 USD (as of 2026-08-31T05:10:59Z).\n"
                "Posted prepaid ledger: $567.49 — not what you have left to spend."
            )

    monkeypatch.setattr(live, "XaiAccountBalanceTool", _Regressed)
    with respx.mock(assert_all_called=True) as mock:
        mock.get(PREVIEW_URL).mock(side_effect=_a_drifting_preview("-51847", DRIFT))
        with pytest.raises(AssertionError, match="outside the"):
            live.test_the_live_figure_nets_the_cycles_prepaid_draw()
