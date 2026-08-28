"""The openrouter_account_balance tool: the subtraction, degraded modes, caching — issue #425.

All HTTP is mocked with respx — no test reaches the network. The payload shape mirrors the real
`GET https://openrouter.ai/api/v1/credits`, verified live on 2026-08-28: two lifetime cumulative
USD totals as plain positive JSON numbers, `{"data": {"total_credits": …, "total_usage": …}}`.
The figures here are fabricated; only the *shape* and the proportions are real.

Three properties carry the tool, and each has an "obviously fine" broken form pinned below:

- **Remaining is a subtraction, and both terms are required.** `total_credits` alone is the
  credit ever *bought*, not what is left of it — an account that bought $500 and spent $500 has
  no runway. Reporting the first field when the second is missing is xAI's issue #388 defect one
  vendor over, so an incomplete payload is *unavailable*, never a guess.
- **The arithmetic shown is the arithmetic done.** Both terms are quantized to cents at the parse
  boundary, so the context line's subtraction lands exactly on the headline — a model that
  re-does it does not find a discrepancy.
- **Nothing but computed figures leaves the tool.** Not the key, not a response body: OpenRouter's
  error envelope carries `error.message` and `user_id`.
"""

from __future__ import annotations

import re

import httpx
import pytest
import respx

from basecradle_harness import OpenRouterAccountBalanceTool, Policy, ToolRegistry
from basecradle_harness._openrouter_account import DEFAULT_BASE_URL

KEY = "sk-or-mgmt-fake-key-000"
CREDITS_URL = f"{DEFAULT_BASE_URL}/credits"


def _credits_body(total_credits=500.0, total_usage=123.456789) -> dict:
    """A `/credits` response in the real shape — two lifetime cumulative USD totals.

    Fabricated figures: $500.00 purchased to date against $123.46 used to date, so $376.54
    remains. Pass a sentinel of `None` to omit a field entirely.
    """
    data: dict = {}
    if total_credits is not None:
        data["total_credits"] = total_credits
    if total_usage is not None:
        data["total_usage"] = total_usage
    return {"data": data}


def _mock(mock, response=None):
    """Route the one billing call; defaults to the healthy fabricated body."""
    return mock.get(CREDITS_URL).mock(
        return_value=response if response is not None else httpx.Response(200, json=_credits_body())
    )


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """No ambient credential leaks into a test — each drives the key explicitly."""
    monkeypatch.delenv("OPENROUTER_MANAGEMENT_KEY", raising=False)


@pytest.fixture
def tool():
    # Caching off unless a test asks for it.
    return OpenRouterAccountBalanceTool(management_key=KEY, cache_ttl=0)


# --- the subtraction ----------------------------------------------------------


def test_remaining_is_purchased_less_used(tool):
    """The headline is the difference, and the two terms behind it are shown."""
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock)
        result = tool.run()

    assert result.startswith("OpenRouter credits remaining: $376.54 USD (as of ")
    # The gross is shown — but only as a term of the subtraction that produced the headline.
    assert "$500.00 of credits purchased to date less the $123.46 used to date." in result


def test_the_purchased_total_is_never_the_headline(tool):
    """`total_credits` is credit ever bought, not credit left — the #388 defect, one vendor over."""
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock)
        headline = tool.run().splitlines()[0]

    assert "$500.00" not in headline
    assert "$376.54" in headline


def test_an_exhausted_account_reports_zero_not_the_purchased_total(tool):
    body = _credits_body(total_credits=250.0, total_usage=250.0)
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, json=body))
        result = tool.run()

    assert result.startswith("OpenRouter credits remaining: $0.00 USD")
    assert "The account is overdrawn." not in result  # exhausted is not overdrawn


def test_an_overdrawn_account_is_reported_as_negative_never_absolute(tool):
    """Usage can overrun purchased credit; `abs()` would invert the one warning that matters."""
    body = _credits_body(total_credits=100.0, total_usage=105.25)
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, json=body))
        result = tool.run()

    assert result.startswith("OpenRouter credits remaining: -$5.25 USD (as of ")
    assert "The account is overdrawn." in result
    assert "OpenRouter credits remaining: $5.25" not in result  # never rendered as healthy credit


def test_the_wording_is_lifetime_to_date_never_a_billing_cycle(tool):
    """These totals never reset, so cycle wording would be false (the issue's explicit note)."""
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock)
        result = tool.run()

    assert "to date" in result
    assert "cycle" not in result.lower()


# --- the arithmetic shown is the arithmetic done ------------------------------


_TERMS = re.compile(r"\$([\d,]+\.\d{2}) of credits purchased to date less the \$([\d,]+\.\d{2})")
_HEADLINE = re.compile(r"^OpenRouter credits remaining: (-?)\$([\d,]+\.\d{2}) USD")


@pytest.mark.parametrize(
    ("purchased", "used"),
    [
        pytest.param(375, 268.464928179, id="live-shape-integer-credits"),
        pytest.param(500.0, 123.456789, id="fabricated-default"),
        pytest.param(10.0, 0.005, id="sub-cent-usage"),
        pytest.param(1.005, 0.005, id="both-sub-cent"),
        pytest.param(100.0, 100.004, id="sub-cent-overrun"),
        pytest.param(0.0, 0.0, id="brand-new-account"),
    ],
)
def test_the_rendered_subtraction_lands_on_the_rendered_headline(tool, purchased, used):
    """Quantizing at the parse boundary is what makes the three printed figures agree.

    A model that re-does the context line's arithmetic must land on the headline; rendering an
    unrounded difference beside rounded terms lets them disagree by a cent, and lets a
    four-tenths-of-a-cent overrun print as an alarming `-$0.00`.
    """
    body = _credits_body(total_credits=purchased, total_usage=used)
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, json=body))
        result = tool.run()

    headline, terms = _HEADLINE.search(result), _TERMS.search(result)
    assert headline and terms, result
    shown = float(terms.group(1).replace(",", "")) - float(terms.group(2).replace(",", ""))
    reported = (-1.0 if headline.group(1) else 1.0) * float(headline.group(2).replace(",", ""))
    assert reported == pytest.approx(shown, abs=1e-9)
    # ...and a negative-zero headline is never printed with an overdraft warning beside it.
    assert not (reported == 0.0 and "overdrawn" in result)


def test_an_integer_credit_total_is_accepted(tool):
    """The live reading returned `total_credits` as a bare `375`, not `375.0`."""
    body = _credits_body(total_credits=375, total_usage=268.464928179)
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, json=body))
        result = tool.run()

    assert result.startswith("OpenRouter credits remaining: $106.54 USD")
    assert "$375.00 of credits purchased to date less the $268.46 used to date." in result


def test_a_large_figure_is_thousands_separated(tool):
    body = _credits_body(total_credits=12500.0, total_usage=1000.0)
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, json=body))
        assert "OpenRouter credits remaining: $11,500.00 USD" in tool.run()


@pytest.mark.parametrize("used", [123.456789, 705.0])
def test_the_headline_shape_is_uniform_across_signs(tool, used):
    """`<label>: <$figure> USD (as of <stamp>).` holds for credit and overdraft alike.

    The overdraft is stated in the body instead of spliced into the headline, so the figure is
    always in the same place — no sometimes-there parenthetical to parse around.
    """
    body = _credits_body(total_usage=used)
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, json=body))
        headline = tool.run().splitlines()[0]

    assert re.fullmatch(
        r"OpenRouter credits remaining: -?\$[\d,]+\.\d{2} USD \(as of \S+Z\)\.", headline
    )


# --- the key: carried, never leaked ------------------------------------------


def test_it_reads_exactly_one_documented_endpoint(tool):
    """The literal URL, pinned — everything else routes off `DEFAULT_BASE_URL` and so cannot
    catch the constant itself drifting. One call, not xAI's three: there is no second surface.
    """
    with respx.mock(assert_all_called=True) as mock:
        route = _mock(mock)
        tool.run()

    assert route.call_count == 1
    assert str(route.calls.last.request.url) == "https://openrouter.ai/api/v1/credits"


def test_the_key_never_leaks_and_the_header_carries_it(tool):
    with respx.mock(assert_all_called=True) as mock:
        route = _mock(mock)
        result = tool.run()

    assert route.calls.last.request.headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in result


def test_the_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", KEY)
    tool = OpenRouterAccountBalanceTool(cache_ttl=0)
    with respx.mock(assert_all_called=True) as mock:
        route = _mock(mock)
        assert "OpenRouter credits remaining: $376.54 USD" in tool.run()

    assert route.calls.last.request.headers["authorization"] == f"Bearer {KEY}"


# --- degraded modes -----------------------------------------------------------


def test_no_key_is_a_clean_unavailable_not_a_crash():
    tool = OpenRouterAccountBalanceTool(cache_ttl=0)  # nothing configured
    with respx.mock(assert_all_called=False) as mock:
        route = _mock(mock)
        result = tool.run()

    assert not route.called  # no HTTP call at all
    assert result.startswith("OpenRouter account balance unavailable — ")
    assert "OPENROUTER_MANAGEMENT_KEY" in result
    assert "openrouter.ai/settings/management-keys" in result  # where the key is minted


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_names_the_credential_and_never_echoes_it(tool, status):
    """A real 401 body is `{"error":{"message":"User not found.","code":401}}` — never relayed."""
    body = {"error": {"message": "User not found.", "code": status}, "user_id": "user_secret_42"}
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(status, json=body))
        result = tool.run()

    assert result.startswith("OpenRouter account balance unavailable — ")
    assert "Management key" in result  # points at the wrong-kind-of-key mistake
    assert "OPENROUTER_MANAGEMENT_KEY" in result
    assert KEY not in result
    assert "User not found" not in result and "user_secret_42" not in result


def test_a_server_error_degrades_and_names_the_action(tool):
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(503, text="boom"))
        result = tool.run()

    assert result.startswith("OpenRouter account balance unavailable — ")
    assert "503" in result
    assert "read the credits remaining" in result  # the action that failed, named
    assert "boom" not in result  # raw bodies never relayed


def test_an_unreachable_endpoint_degrades(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(CREDITS_URL).mock(side_effect=httpx.ConnectError("no route"))
        result = tool.run()

    assert "unavailable" in result
    assert "couldn't reach" in result


def test_an_unreadable_response_degrades(tool):
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, text="not json"))
        result = tool.run()

    assert "unavailable" in result
    assert "unreadable" in result


def test_a_json_list_response_degrades(tool):
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, json=[]))
        result = tool.run()

    assert "unavailable" in result
    assert "unexpected" in result


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        pytest.param({}, "carried no data object", id="no-data-object"),
        pytest.param({"data": []}, "carried no data object", id="data-not-an-object"),
        pytest.param(
            _credits_body(total_credits=None), "no usable total_credits", id="no-total-credits"
        ),
        pytest.param(_credits_body(total_usage=None), "no usable total_usage", id="no-total-usage"),
        pytest.param(
            _credits_body(total_credits="500.0"),
            "no usable total_credits",
            id="credits-as-a-string",
        ),
        pytest.param(
            _credits_body(total_usage=True), "no usable total_usage", id="usage-as-a-bool"
        ),
        pytest.param(
            _credits_body(total_credits=None, total_usage=None),
            "no usable total_credits",
            id="empty-data",
        ),
    ],
)
def test_an_unusable_payload_is_unavailable_never_a_guess(tool, body, reason):
    """There is no second surface to fall back to, so an incomplete payload has no answer in it.

    In particular a payload carrying only `total_credits` must NOT report it: that is the credit
    the account has ever bought, not what is left of it.

    The expectation is the **whole diagnostic phrase**, not a bare field name: the tail of the
    message always names *both* fields ("the remaining credit is total_credits minus
    total_usage"), so `"total_usage" in result` is true even when it was `total_credits` that was
    missing — an assertion that passes with the ternary inverted pins nothing.
    """
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, json=body))
        result = tool.run()

    assert result.startswith("OpenRouter account balance unavailable — ")
    assert "$" not in result  # no figure of any kind is surfaced
    assert reason in result  # the reason names the field that was actually missing


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('{"data": {"total_credits": NaN, "total_usage": 0.0}}', id="nan"),
        pytest.param('{"data": {"total_credits": Infinity, "total_usage": 0.0}}', id="infinity"),
        pytest.param('{"data": {"total_credits": 100.0, "total_usage": -Infinity}}', id="-inf-use"),
        pytest.param(
            '{"data": {"total_credits": ' + "1" * 400 + ', "total_usage": 0.0}}', id="huge"
        ),
    ],
)
def test_a_number_that_is_not_a_dollar_figure_degrades_instead_of_raising(tool, raw):
    """Python's `json` accepts `NaN`/`Infinity`, and a JSON integer literal has no width limit.

    Neither survives the renderer: `nan < 0` is False (so an overdraft check says healthy) and
    both format as a *figure* — `$nan USD`, `$inf USD`. The oversized integer is worse: `float()`
    raises `OverflowError` straight out of `run()`, the one thing this tool must never do.
    """
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, content=raw, headers={"content-type": "application/json"}))
        result = tool.run()  # must not raise

    assert result.startswith("OpenRouter account balance unavailable — ")
    for garbage in ("nan", "inf", "$1,111"):
        assert garbage not in result.lower()


def test_a_bool_is_not_a_credit_figure(tool):
    """`isinstance(True, int)` is True in Python — an unguarded parse makes it $1.00 of credit."""
    body = _credits_body(total_credits=True, total_usage=0.0)
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, json=body))
        result = tool.run()

    assert "unavailable" in result
    assert "$1.00" not in result


# --- the renderer never prints a malformed figure -----------------------------


@pytest.mark.parametrize(
    ("purchased", "used"),
    [
        pytest.param(-0.001, 0.0, id="purchased-negative-sub-cent"),
        pytest.param(10.0, -0.004, id="used-negative-sub-cent"),
        pytest.param(-0.004, -0.004, id="both-negative-sub-cent"),
    ],
)
def test_a_value_that_rounds_to_negative_zero_never_prints_a_sign(tool, purchased, used):
    """`round()` makes `-0.0` out of anything in `[-0.005, 0)`, and `-0.0 < 0` is `False`.

    So a naive renderer sends it through the *positive* arm and prints `$-0.00` — the sign on the
    wrong side of the dollar, outside the uniform headline shape the live probe's regex depends
    on, and exactly the alarming near-zero "overdraft" the cent quantization exists to remove.
    """
    body = _credits_body(total_credits=purchased, total_usage=used)
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, json=body))
        result = tool.run()

    assert "$-" not in result  # never the malformed sign order
    assert "-$0.00" not in result  # nor a signed zero the other way round
    assert re.fullmatch(
        r"OpenRouter credits remaining: -?\$[\d,]+\.\d{2} USD \(as of \S+Z\)\.",
        result.splitlines()[0],
    )


def test_a_genuinely_negative_term_still_renders_in_the_uniform_shape(tool):
    # A refund would make a term negative. It is honest to show it; it must not break the shape.
    body = _credits_body(total_credits=100.0, total_usage=-25.0)
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, json=body))
        result = tool.run()

    assert result.startswith("OpenRouter credits remaining: $125.00 USD (as of ")
    assert "less the -$25.00 used to date." in result


def test_a_non_ascii_key_degrades_instead_of_raising(monkeypatch):
    """httpx ASCII-encodes a header value at *client construction*, before any request exists.

    So a key carrying a smart quote or a non-breaking space — what pasting a credential actually
    produces — raised `UnicodeEncodeError` straight out of `run()`, past the `httpx.RequestError`
    guard, which never sees it. A misconfigured credential is the case this tool most owes a soft
    answer to.
    """
    tool = OpenRouterAccountBalanceTool(management_key="sk-or-\u201cnot-ascii\u201d", cache_ttl=0)
    with respx.mock(assert_all_called=False) as mock:
        route = _mock(mock)
        result = tool.run()  # must not raise

    assert not route.called  # it never got as far as a request
    assert result.startswith("OpenRouter account balance unavailable — ")
    assert "non-ASCII" in result
    assert "OPENROUTER_MANAGEMENT_KEY" in result
    assert "not-ascii" not in result  # the key is never echoed back


def test_a_deeply_nested_body_degrades_instead_of_raising(tool):
    # `json` raises `RecursionError`, not a `ValueError`, past a certain nesting depth.
    raw = '{"data": ' + "[" * 2000 + "]" * 2000 + "}"
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(200, content=raw, headers={"content-type": "application/json"}))
        result = tool.run()  # must not raise

    assert result.startswith("OpenRouter account balance unavailable — ")


# --- caching -----------------------------------------------------------------


class _FakeClock:
    """A hand-advanced monotonic clock, so cache expiry is deterministic (no sleeping)."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_a_second_call_within_the_ttl_is_served_from_cache():
    clock = _FakeClock()
    tool = OpenRouterAccountBalanceTool(management_key=KEY, cache_ttl=30, clock=clock)
    with respx.mock(assert_all_called=True) as mock:
        route = _mock(mock)
        first = tool.run()
        clock.t += 10  # still inside the 30s window
        second = tool.run()

    assert first == second
    assert route.call_count == 1  # the second call never hit the network


def test_the_cache_expires_after_the_ttl():
    clock = _FakeClock()
    tool = OpenRouterAccountBalanceTool(management_key=KEY, cache_ttl=30, clock=clock)
    with respx.mock(assert_all_called=True) as mock:
        route = _mock(mock)
        tool.run()
        clock.t += 31  # past the window
        tool.run()

    assert route.call_count == 2  # a stale figure is re-read


def test_ttl_zero_disables_caching(tool):
    with respx.mock(assert_all_called=True) as mock:
        route = _mock(mock)
        tool.run()
        tool.run()

    assert route.call_count == 2  # cache_ttl=0 → every call re-reads


def test_a_repointed_credential_is_never_served_a_cached_figure(monkeypatch):
    """The key is resolved from the environment at *call* time, so the cache must carry it.

    Without that, a figure read for one account — with that account's `as of` stamp — is served
    as another account's for the rest of the TTL.
    """
    clock = _FakeClock()
    tool = OpenRouterAccountBalanceTool(cache_ttl=30, clock=clock)  # key from env, per call
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", KEY)
    with respx.mock(assert_all_called=True) as mock:
        route = _mock(mock)
        first = tool.run()

    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", "sk-or-mgmt-a-different-account")
    with respx.mock(assert_all_called=True) as mock:  # the clock has NOT moved
        route = _mock(mock, httpx.Response(200, json=_credits_body(total_credits=9.0)))
        second = tool.run()

    assert route.call_count == 1  # the second key re-read rather than reusing the first figure
    assert route.calls.last.request.headers["authorization"].endswith("a-different-account")
    assert "$376.54" in first
    assert "OpenRouter credits remaining: -$114.46 USD" in second


def test_an_unavailable_result_is_never_cached():
    # A transient outage must not pin "unavailable" for the whole TTL — the next call retries.
    clock = _FakeClock()
    tool = OpenRouterAccountBalanceTool(management_key=KEY, cache_ttl=30, clock=clock)
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock, httpx.Response(500))
        assert "unavailable" in tool.run()
    with respx.mock(assert_all_called=True) as mock:
        _mock(mock)  # the outage clears; the clock has not moved
        assert "OpenRouter credits remaining: $376.54 USD" in tool.run()


# --- it is a safe, locked-profile tool ---------------------------------------


def test_loads_under_the_locked_default_profile():
    # A plain read-only tool: no policy capability, so it registers under the shipped safe policy.
    registry = ToolRegistry(Policy.locked())
    registry.register(OpenRouterAccountBalanceTool())
    assert "openrouter_account_balance" in registry
    assert OpenRouterAccountBalanceTool().parameters == {"type": "object", "properties": {}}
