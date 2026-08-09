"""The `polymarket_paper` instrument: the fences, the budget, the ledger (issue #347).

No test here touches Polymarket. A catch-all respx route stands in for both public hosts, so
the *real* `PolymarketData` — its constant hosts, its query building, its parsing — runs
against fabricated bodies shaped exactly like the live ones. That matters for the safety
assertions: "every request is a GET to one of two hosts" is only worth anything if the code
under test is the code that makes the requests.

Test data is fabricated throughout, per this repo's convention: the market is Nova Digital's,
the ids are well-formed but invented, and no real market, wallet, or account appears.

`test_polymarket_fills.py` carries the fill model, settlement and scorecard; this file carries
the tool surface, the §2.5 non-goals, the §2.2 budget, the §2.6 envelope and the §A3 sweep.
"""

from __future__ import annotations

import ast
import contextlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import respx

from basecradle_harness import PolymarketData, PolymarketPaperTool
from basecradle_harness._polymarket import ACTIONS
from basecradle_harness._polymarket_data import ALLOWED_HOSTS, CLOB_BASE, GAMMA_BASE
from basecradle_harness._polymarket_ledger import (
    CALL,
    HARD_CALLS_PER_DAY,
    HARD_ORDERS_PER_DAY,
    MAX_LIST_LIMIT,
    current_epoch,
)

# --- fabricated upstream ---------------------------------------------------------

MARKET_ID = "900001"
CONDITION_ID = "0x" + "9f3c" * 16
YES_TOKEN = "48231907745503102928374650192837465012938475610293847561029384756102"
NO_TOKEN = "71904523867120394857612093847561029384756102938475610293847561029384"
EVENT_ID = "700042"
QUESTION = "Will Nova Digital ship Harness v1 before 2027?"


def gamma_market(**over):
    """One Gamma `/markets` row, shaped as the live API returns it (JSON-string lists included)."""
    row = {
        "id": MARKET_ID,
        "conditionId": CONDITION_ID,
        "question": QUESTION,
        "slug": "will-nova-digital-ship-harness-v1-before-2027",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.42", "0.58"]),
        "clobTokenIds": json.dumps([YES_TOKEN, NO_TOKEN]),
        "volume24hr": 128400.5,
        "liquidity": 54210.25,
        "endDate": "2026-12-31T00:00:00Z",
        "acceptingOrders": True,
        "active": True,
        "closed": False,
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5,
        "events": [{"id": EVENT_ID, "slug": "harness-v1", "title": "Harness v1"}],
        "tags": [{"id": "7", "label": "Technology", "slug": "technology"}],
    }
    row.update(over)
    return row


def clob_market(**over):
    """One CLOB `/markets/{condition_id}` body — tradability, fees, and the winner flags."""
    body = {
        "condition_id": CONDITION_ID,
        "question": QUESTION,
        "market_slug": "will-nova-digital-ship-harness-v1-before-2027",
        "enable_order_book": True,
        "active": True,
        "closed": False,
        "accepting_orders": True,
        "minimum_order_size": 5,
        "minimum_tick_size": 0.01,
        "maker_base_fee": 0,
        "taker_base_fee": 0,
        "tokens": [
            {"token_id": YES_TOKEN, "outcome": "Yes", "price": 0.42, "winner": False},
            {"token_id": NO_TOKEN, "outcome": "No", "price": 0.58, "winner": False},
        ],
        "tags": ["Technology"],
    }
    body.update(over)
    return body


def book(bids=(), asks=(), last=None, token=YES_TOKEN):
    """A CLOB `/book` body. Levels are given best-first here and deliberately sent scrambled.

    The live API returns asks worst-first; sending them out of order is how these tests pin
    that the client re-sorts rather than trusting the wire.
    """
    return {
        "market": CONDITION_ID,
        "asset_id": token,
        "bids": [{"price": str(p), "size": str(s)} for p, s in reversed(list(bids))],
        "asks": [{"price": str(p), "size": str(s)} for p, s in reversed(list(asks))],
        "min_order_size": "5",
        "tick_size": "0.01",
        "last_trade_price": None if last is None else str(last),
    }


class Upstream:
    """A stand-in for Gamma + the public CLOB that records every request it was handed."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.gamma = {MARKET_ID: gamma_market()}
        self.clob = {CONDITION_ID: clob_market()}
        self.books = {
            YES_TOKEN: book(bids=[("0.41", "500")], asks=[("0.43", "300"), ("0.45", "1000")]),
            NO_TOKEN: book(bids=[("0.55", "500")], asks=[("0.57", "300")], token=NO_TOKEN),
        }
        self.listing = [gamma_market()]
        self.search_events = [{"id": EVENT_ID, "slug": "harness-v1", "markets": [gamma_market()]}]
        self.search_has_more = False
        self.status = {}  # path -> status code, to force an outage
        self.fail_ids: set[str] = set()  # market ids whose lookup should 503
        # Tokens whose `/book` 404s. The live CLOB does this the moment it switches a
        # market's order book off, which is every resolved market (checked 2026-08-03).
        self.no_book: set[str] = set()

    def delist(self, market_id=MARKET_ID):
        """Gamma forgets a market entirely — its ordinary lifecycle once one resolves.

        Of 400 resolved markets the CLOB still served with winner flags, 400 were gone from
        Gamma (measured against live public data, 2026-08-03). This is the normal case, not
        an edge case, and issue #390 is what it cost.
        """
        self.gamma.pop(market_id, None)
        self.listing = [row for row in self.listing if str(row.get("id")) != market_id]

    # -- more markets, for the multi-market and cap cases -----------------------
    def add_market(self, index: int, *, asks=(("0.31", "500"),)) -> str:
        """Register a fabricated market #`index` and return its market_id.

        Each gets its own event, so a test that needs *distinct event clusters* (the
        scorecard's diversity gate) gets them by construction rather than by coincidence.
        """
        market_id = str(910000 + index)  # a range of its own, never colliding with MARKET_ID
        condition_id = "0x" + f"{index:04x}" * 16
        yes = f"{index:02d}" + YES_TOKEN[2:]
        no = f"{index:02d}" + NO_TOKEN[2:]
        self.gamma[market_id] = gamma_market(
            id=market_id,
            conditionId=condition_id,
            question=f"Will fabricated proposition {index} resolve Yes?",
            slug=f"fabricated-proposition-{index}",
            clobTokenIds=json.dumps([yes, no]),
            events=[{"id": str(700000 + index), "slug": f"event-{index}"}],
        )
        self.clob[condition_id] = clob_market(
            condition_id=condition_id,
            tokens=[
                {"token_id": yes, "outcome": "Yes", "price": 0.3, "winner": False},
                {"token_id": no, "outcome": "No", "price": 0.7, "winner": False},
            ],
        )
        self.books[yes] = book(bids=[("0.29", "500")], asks=list(asks), token=yes)
        self.books[no] = book(bids=[("0.69", "500")], asks=[("0.71", "500")], token=no)
        return market_id

    def resolve(self, winner="Yes", condition_id=CONDITION_ID):
        """Mark a market resolved in public state — the only way settlement can be triggered.

        Shaped exactly as the live CLOB shapes a resolved market, which is more than the
        winner flags: it also **switches the order book off** and starts 404ing `/book` for
        both tokens (checked against live public data, 2026-08-03). Fabricating a resolved
        market that still had a book is what let issue #390's D1 hide — the gate that refused
        every book-less market sat on the shared resolution path, so nothing in the suite ever
        settled a market shaped like a real one.
        """
        market = dict(self.clob[condition_id])
        market["closed"] = True
        market["accepting_orders"] = False
        market["enable_order_book"] = False
        market["tokens"] = [
            dict(token, winner=(token["outcome"] == winner)) for token in market["tokens"]
        ]
        self.clob[condition_id] = market
        self.no_book |= {str(token["token_id"]) for token in market["tokens"]}

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        host = request.url.host
        forced = self.status.get(path)
        if forced:
            return httpx.Response(forced, json={"error": "forced"})

        if host == "gamma-api.polymarket.com":
            if path == "/markets":
                # `market_summary` reads one market through the *collection* with an `id`
                # filter, because the single-market endpoint omits `events` (see
                # `PolymarketData.market_summary`). The double mirrors that.
                wanted = request.url.params.get("id")
                if wanted:
                    if wanted in self.fail_ids:
                        return httpx.Response(503, json={"error": "upstream"})
                    row = self.gamma.get(wanted)
                    return httpx.Response(200, json=[row] if row else [])
                return httpx.Response(200, json=self.listing)
            if path == "/public-search":
                return httpx.Response(
                    200,
                    json={
                        "events": self.search_events,
                        "pagination": {"hasMore": self.search_has_more, "totalResults": 1},
                    },
                )
            if path.startswith("/tags/slug/"):
                return httpx.Response(200, json={"id": "7", "slug": path.rsplit("/", 1)[-1]})
        if host == "clob.polymarket.com":
            if path.startswith("/markets/"):
                key = path.rsplit("/", 1)[-1]
                if key not in self.clob:
                    return httpx.Response(404, json={"error": "not found"})
                return httpx.Response(200, json=self.clob[key])
            if path == "/book":
                token = request.url.params.get("token_id", "")
                if token in self.no_book:
                    return httpx.Response(404, json={"error": "no orderbook exists"})
                return httpx.Response(200, json=self.books.get(token, book(token=token)))
        return httpx.Response(404, json={"error": f"unrouted {host}{path}"})


@contextlib.contextmanager
def upstream():
    """Route every httpx request through a fabricated Polymarket."""
    fake = Upstream()
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=fake.handle)
        yield fake


class Clock:
    """A UTC clock a test can advance — §2.4's authority, made deterministic."""

    def __init__(self, moment: datetime | None = None):
        self.value = moment or datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs):
        self.value += timedelta(**kwargs)
        return self.value


def make_tool(tmp_path: Path, clock: Clock | None = None) -> PolymarketPaperTool:
    """A tool wired to a temp home and an uncached data client (so a test can move the book)."""
    return PolymarketPaperTool(
        root=tmp_path, data=PolymarketData(cache_ttl=0), now=clock or Clock()
    )


def call(tool: PolymarketPaperTool, action: str, **kwargs) -> dict:
    """Run one operation and parse its JSON body."""
    return json.loads(tool.run(action=action, **kwargs))


def forecast(tool, p="0.6", market_id=MARKET_ID, outcome="Yes"):
    return call(
        tool,
        "log_forecast",
        market_id=market_id,
        outcome=outcome,
        p=p,
        rationale_ref="mem://calibration/note-1",
    )


_order_seq = iter(range(1, 10_000))


def buy(tool, *, shares="10", order_type="market", price=None, market_id=MARKET_ID, outcome="Yes"):
    return call(
        tool,
        "place_order",
        market_id=market_id,
        outcome=outcome,
        side="buy",
        size_shares=shares,
        order_type=order_type,
        limit_price=price,
        client_order_id=f"coid-{next(_order_seq):04d}",
    )


# --- §2.3: the operation set is closed ---------------------------------------------


def test_the_action_enum_is_exactly_the_normative_operation_set(tmp_path):
    tool = make_tool(tmp_path)
    assert tuple(tool.parameters["properties"]["action"]["enum"]) == ACTIONS
    assert set(ACTIONS) == {
        "list_markets",
        "get_market",
        "get_positions",
        "place_order",
        "cancel_order",
        "get_orders",
        "get_fills",
        "get_pnl",
        "get_scorecard",
        "log_forecast",
    }


def test_every_action_has_a_handler_and_nothing_else_is_reachable(tmp_path):
    tool = make_tool(tmp_path)
    for action in ACTIONS:
        assert callable(getattr(tool, f"_op_{action}"))
    handlers = {name[len("_op_") :] for name in dir(tool) if name.startswith("_op_")}
    assert handlers == set(ACTIONS)


# --- §2.1 and §2.5: the non-goals, enforced in code ---------------------------------


def test_no_parameter_can_carry_a_url(tmp_path):
    """§2.1: the agent never passes a URL, and never receives a generic HTTP primitive."""
    schema = make_tool(tmp_path).parameters
    forbidden = ("url", "uri", "endpoint", "host", "href", "address", "domain", "link")
    for name, spec in schema["properties"].items():
        assert not any(word in name.casefold() for word in forbidden), name
        assert "http://" not in spec.get("description", "")
        assert "https://" not in spec.get("description", "")
    blob = json.dumps(schema).casefold()
    assert "http" not in blob


def test_no_operation_offers_a_transfer_withdraw_or_wallet_path(tmp_path):
    """§2.5: no money movement and no key material — there is nothing in the surface to call."""
    tool = make_tool(tmp_path)
    blob = json.dumps(tool.parameters).casefold()
    for word in (
        "transfer",
        "withdraw",
        "deposit",
        "bridge",
        "wallet",
        "private_key",
        "seed_phrase",
        "signature",
        "usdc",
        "screenshot",
        "cookie",
        "browser",
    ):
        assert word not in blob, word
    for action in ACTIONS:
        assert not any(
            bad in action for bad in ("transfer", "withdraw", "deposit", "bridge", "fund")
        )


def test_no_operation_lets_the_agent_supply_a_price_fee_pnl_or_resolution(tmp_path):
    """The ledger boundary: the agent may read those numbers and may never define them."""
    props = set(make_tool(tmp_path).parameters["properties"])
    # `limit_price` and `p` are the agent's *own* inputs — the price it is willing to pay and
    # the probability it is being scored on. Everything valuation-side is absent.
    assert (
        props
        & {
            "price",
            "mid",
            "fee",
            "fee_bps",
            "pnl",
            "realized_pnl",
            "cash",
            "cash_usd",
            "equity",
            "balance",
            "bankroll",
            "payout",
            "resolution",
            "winner",
            "brier",
            "mtm_price",
        }
        == set()
    )
    assert "limit_price" in props and "p" in props


def test_the_data_client_only_ever_issues_gets_to_the_two_public_hosts(tmp_path):
    """§2.1 + §2.5, as a property of the traffic rather than of the docstring."""
    with upstream() as fake:
        tool = make_tool(tmp_path)
        call(tool, "list_markets", query="harness")
        call(tool, "get_market", market_id=MARKET_ID)
        forecast(tool)
        buy(tool)
        call(tool, "get_positions")
        assert fake.requests
        for request in fake.requests:
            assert request.method == "GET", request.url
            assert request.url.host in ALLOWED_HOSTS, request.url
            assert request.url.scheme == "https"
            assert "authorization" not in {k.lower() for k in request.headers}
            assert "cookie" not in {k.lower() for k in request.headers}


def test_an_agent_supplied_id_cannot_steer_the_path(tmp_path):
    """§2.1 covers the whole destination, not only its origin.

    A market id and a tag slug are the two agent strings that become **path segments**. A
    traversal in one stays on the Polymarket host, which is exactly what makes it easy to wave
    off — but it is no longer the request this code believes it is making.
    """
    with upstream() as fake:
        tool = make_tool(tmp_path)
        for hostile in ("../tags/slug/politics", "900001/../../markets", "9/0/0"):
            body = call(tool, "get_market", market_id=hostile)
            assert body["error"] == "not_found", hostile
        call(tool, "list_markets", tag="../../markets")
    for request in fake.requests:
        assert ".." not in request.url.path, request.url
        assert request.url.host in ALLOWED_HOSTS


def test_a_non_finite_number_is_refused_rather_than_carried_into_the_arithmetic(tmp_path):
    """`Decimal("NaN")` parses and then raises on the first comparison, frames away."""
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        for bad in ("NaN", "Infinity", "-Infinity"):
            body = buy(tool, shares=bad)
            assert body["error"] == "invalid_params", bad
            assert "finite" in body["message"]
        assert forecast(tool, p="NaN")["error"] == "invalid_params"


def test_the_list_projections_are_bounded(tmp_path):
    """A months-old epoch must not be able to answer one call with its whole history."""
    from basecradle_harness._polymarket import MAX_FILLS_LIMIT

    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        page = call(tool, "get_fills", limit=10_000)
    assert len(page["fills"]) <= MAX_FILLS_LIMIT


def test_the_two_hosts_are_constants_the_agent_cannot_reach(tmp_path):
    assert GAMMA_BASE == "https://gamma-api.polymarket.com"
    assert CLOB_BASE == "https://clob.polymarket.com"
    assert ALLOWED_HOSTS == {"gamma-api.polymarket.com", "clob.polymarket.com"}
    # No env var, and no parameter, can move them: the only override is a constructor kwarg,
    # which nothing on the agent's path ever supplies.
    schema = json.dumps(make_tool(tmp_path).parameters)
    assert "base" not in schema and "gamma" not in schema.casefold()


# --- §2.6: the error envelope ------------------------------------------------------


def test_an_unknown_action_is_invalid_params_and_still_costs_a_call(tmp_path):
    tool = make_tool(tmp_path)
    body = json.loads(tool.run(action="withdraw_everything"))
    assert body == {
        "ok": False,
        "error": "invalid_params",
        "message": body["message"],
        "budgets": body["budgets"],
    }
    assert "unknown action" in body["message"]
    assert body["budgets"]["calls_remaining_day"] == HARD_CALLS_PER_DAY - 1


def test_the_error_envelope_has_exactly_the_normative_keys(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        body = call(tool, "get_market", market_id="404404")
    assert set(body) == {"ok", "error", "message", "budgets"}
    assert body["ok"] is False
    assert body["error"] == "not_found"


def test_an_upstream_outage_is_not_reported_as_not_found(tmp_path):
    """The one code added beyond §2.6's list, and the reason it exists."""
    with upstream() as fake:
        fake.fail_ids.add(MARKET_ID)
        tool = make_tool(tmp_path)
        body = call(tool, "get_market", market_id=MARKET_ID)
    assert body["error"] == "upstream_unavailable"
    assert "not_found" not in body["error"]


def test_every_response_carries_budgets_and_an_as_of(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        for action, kwargs in (
            ("list_markets", {}),
            ("get_market", {"market_id": MARKET_ID}),
            ("get_positions", {}),
            ("get_orders", {}),
            ("get_fills", {}),
            ("get_pnl", {}),
            ("get_scorecard", {}),
        ):
            body = call(tool, action, **kwargs)
            assert body["ok"] is True, body
            assert set(body["budgets"]) >= {"calls_remaining_day", "orders_remaining_day"}
            assert body["as_of"].endswith("Z")


# --- §2.2: the burn ceiling ---------------------------------------------------------


def test_the_hard_call_ceiling_returns_rate_limited_and_resets_at_utc_midnight(tmp_path):
    clock = Clock()
    tool = make_tool(tmp_path, clock)
    epoch = current_epoch(tmp_path, now=clock)
    for _ in range(HARD_CALLS_PER_DAY):
        epoch.append(CALL, {"op": "get_pnl"})

    body = call(tool, "get_pnl")
    assert body["error"] == "rate_limited"
    assert body["budgets"]["calls_remaining_day"] == 0
    assert "00:00 UTC" in body["message"]

    clock.advance(days=1)
    assert call(tool, "get_pnl")["ok"] is True


def test_the_order_ceiling_is_separate_and_leaves_reads_working(tmp_path):
    clock = Clock()
    with upstream():
        tool = make_tool(tmp_path, clock)
        epoch = current_epoch(tmp_path, now=clock)
        for _ in range(HARD_ORDERS_PER_DAY):
            epoch.append(CALL, {"op": "place_order"})
        forecast(tool)
        body = buy(tool)
        assert body["error"] == "rate_limited"
        assert body["budgets"]["orders_remaining_day"] == 0
        assert call(tool, "get_pnl")["ok"] is True


def test_a_rate_limited_call_is_not_itself_charged(tmp_path):
    clock = Clock()
    tool = make_tool(tmp_path, clock)
    epoch = current_epoch(tmp_path, now=clock)
    for _ in range(HARD_CALLS_PER_DAY):
        epoch.append(CALL, {"op": "get_pnl"})
    before = len(epoch.rows())
    call(tool, "get_pnl")
    assert len(epoch.rows()) == before


def test_the_page_size_is_clamped_to_the_ceiling(tmp_path):
    with upstream():
        body = call(make_tool(tmp_path), "list_markets", limit=500)
    assert body["ok"] is True
    assert str(MAX_LIST_LIMIT) in body["note"]


# --- §A2: a position requires a forecast ---------------------------------------------


def test_place_order_without_a_forecast_is_forecast_required(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        body = buy(tool)
    assert body["error"] == "forecast_required"
    assert "log_forecast" in body["message"]


def test_a_logged_forecast_unlocks_the_position(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        locked = forecast(tool, p="0.62")
        assert locked["ok"] is True and locked["p"] == 0.62
        assert locked["locked_at"].endswith("Z")
        assert buy(tool)["ok"] is True


def test_one_forecast_covers_repeated_adds_on_the_same_key(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        assert buy(tool, shares="10")["ok"] is True
        assert buy(tool, shares="10")["ok"] is True


def test_a_forecast_on_one_outcome_does_not_unlock_the_other(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool, outcome="Yes")
        assert buy(tool, outcome="No")["error"] == "forecast_required"


def test_selling_and_cancelling_need_no_forecast(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        # §A2 gates a *buy*; reducing or cancelling never consults a forecast.
        sold = call(
            tool,
            "place_order",
            market_id=MARKET_ID,
            outcome="Yes",
            side="sell",
            size_shares="5",
            order_type="market",
            client_order_id="coid-sell-1",
        )
        assert sold["ok"] is True and sold["status"] == "filled"


def test_log_forecast_refuses_a_probability_outside_the_open_interval(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        for bad in ("0", "1", "1.5", "-0.2"):
            body = forecast(tool, p=bad)
            assert body["error"] == "invalid_params", bad


# --- §2.4: idempotency ----------------------------------------------------------------


def test_a_duplicate_client_order_id_returns_the_original_result(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        first = call(
            tool,
            "place_order",
            market_id=MARKET_ID,
            outcome="Yes",
            side="buy",
            size_shares="10",
            order_type="market",
            client_order_id="coid-same",
        )
        again = call(
            tool,
            "place_order",
            market_id=MARKET_ID,
            outcome="Yes",
            side="buy",
            size_shares="10",
            order_type="market",
            client_order_id="coid-same",
        )
    assert first["order_id"] == again["order_id"]
    assert again["duplicate_of_client_order_id"] == "coid-same"
    positions = json.loads(
        PolymarketPaperTool(root=tmp_path, data=PolymarketData(cache_ttl=0)).run(
            action="get_positions"
        )
    )
    assert len(positions["positions"]) == 1
    assert positions["positions"][0]["shares"] == 10.0  # not 20 — the second submit was a no-op


# --- the ledger boundary ----------------------------------------------------------------


def test_every_row_carries_the_normative_envelope_and_its_chain_links(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    rows = current_epoch(tmp_path).rows()
    assert rows
    for row in rows:
        assert set(row) == {
            "epoch_id",
            "ts",
            "type",
            "payload",
            "schema_version",
            "prev",
            "hash",
        }
        assert row["schema_version"] == 2
        assert row["ts"].endswith("Z")


def test_the_ledger_is_append_only(tmp_path):
    """Existing bytes are never rewritten — a later operation can only extend the file."""
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        ledger = current_epoch(tmp_path).path
        before = ledger.read_bytes()
        buy(tool)
        call(tool, "get_pnl")
        after = ledger.read_bytes()
    assert after.startswith(before)
    assert len(after) > len(before)


def test_the_epoch_freezes_its_terms_in_its_first_row(tmp_path):
    """§A2's attribution choice and the v1 caps are recorded, not just coded."""
    make_tool(tmp_path).run(action="get_pnl")
    first = current_epoch(tmp_path).rows()[0]
    assert first["type"] == "epoch_open"
    payload = first["payload"]
    assert payload["brier_attribution"] == "position_open"
    assert payload["bankroll_usd"] == "10000"
    assert payload["caps"]["max_order_notional_usd"] == "500"
    assert payload["caps"]["max_market_exposure_usd"] == "2000"
    assert payload["caps"]["max_open_positions"] == 20
    assert payload["resting_recheck"] == "hourly_sweep_plus_on_touch"


def test_the_agent_cannot_top_up_its_own_bankroll(tmp_path):
    """Nothing in the surface credits cash, and the ledger's only funding row is epoch_open."""
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        rows = current_epoch(tmp_path).rows()
    funding = [r for r in rows if r["type"] == "epoch_open"]
    assert len(funding) == 1
    assert json.loads(tool.run(action="get_pnl"))["cash_usd"] < 10000


def test_the_ledger_lives_under_the_agent_home_not_in_memory(tmp_path):
    make_tool(tmp_path).run(action="get_pnl")
    assert (tmp_path / "polymarket").is_dir()
    ledger = next((tmp_path / "polymarket").glob("epoch-*/ledger.jsonl"))
    assert ledger.exists()


def test_a_torn_final_row_is_skipped_and_never_repaired_in_place(tmp_path):
    tool = make_tool(tmp_path)
    tool.run(action="get_pnl")
    epoch = current_epoch(tmp_path)
    with open(epoch.path, "a", encoding="utf-8") as handle:
        handle.write('{"epoch_id": "x", "ts": "2026')  # a kill mid-append
    torn = epoch.path.read_bytes()
    assert json.loads(tool.run(action="get_pnl"))["ok"] is True
    assert epoch.path.read_bytes().startswith(torn)  # the damaged bytes are still there


# --- §A3: the sweep wakes no one ----------------------------------------------------------


def _imported_modules(path: Path) -> set[str]:
    """Every module name the file imports, at any nesting depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


SOURCE = Path(__file__).parent.parent / "src" / "basecradle_harness"


def test_the_sweep_module_cannot_reach_a_model_or_the_platform():
    """§A3 as a structural fact: there is no model path in the sweep's import graph."""
    allowed = {
        "__future__",
        "argparse",
        "json",  # the read-only probe's output format (issue #353) — stdlib, no I/O of its own
        "logging",
        "os",
        "sys",
        "dataclasses",
        "decimal",
        "pathlib",
        "typing",
        "basecradle_harness._polymarket_data",
        "basecradle_harness._polymarket_ledger",
        # A single module-level string constant, so the probe can stamp the contract version it
        # is answering under. It imports nothing itself — the allow-list stays a real guard.
        "basecradle_harness._version",
    }
    assert _imported_modules(SOURCE / "_polymarket_engine.py") <= allowed
    # And the version module really is inert: a leaf that pulls in a provider would smuggle the
    # whole model path in behind a name that reads as harmless.
    assert _imported_modules(SOURCE / "_version.py") == set()


def test_neither_the_ledger_nor_the_data_client_reaches_the_platform():
    forbidden = {
        "basecradle",
        "openai",
        "basecradle_harness._engine",
        "basecradle_harness._provider",
        "basecradle_harness._wake",
        "basecradle_harness._basecradle",
        "basecradle_harness._messages",
        "basecradle_harness._session",
        "basecradle_harness._platform",
    }
    for module in ("_polymarket_engine.py", "_polymarket_ledger.py", "_polymarket_data.py"):
        assert not (_imported_modules(SOURCE / module) & forbidden), module


def test_the_sweep_job_writes_rows_and_makes_no_other_call(tmp_path):
    """Behavioural half: a real sweep settles a market, and every request it made was a read."""
    from basecradle_harness._polymarket_engine import main

    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        fake.resolve(winner="Yes")
        fake.requests.clear()
        assert main(["--home", str(tmp_path)]) == 0
        assert fake.requests
        assert all(r.method == "GET" for r in fake.requests)
        assert {r.url.host for r in fake.requests} <= ALLOWED_HOSTS

    rows = current_epoch(tmp_path).rows()
    assert [r for r in rows if r["type"] == "settlement"]
    assert [r for r in rows if r["type"] == "brier_score"]
    # The sweep charges no call budget: it is the operator's job, not the agent's.
    assert not [r for r in rows if r["type"] == "call" and r["payload"]["op"] == "sweep"]


def test_the_sweep_is_a_no_op_when_there_is_no_epoch(tmp_path, capsys):
    from basecradle_harness._polymarket_engine import main

    assert main(["--home", str(tmp_path)]) == 0
    assert "nothing to sweep" in capsys.readouterr().out
    # A cron run never opens a trading record. (It does take the store lock, so the store dir
    # may exist — what must not exist is an epoch, which is a bankroll and a scorecard.)
    assert list((tmp_path / "polymarket").glob("epoch-*")) == []


def test_the_wake_and_the_sweep_do_not_double_fill_a_resting_order(tmp_path):
    """The two writers are separate processes, and both fill resting orders.

    Unlocked, a wake and the hourly sweep that read the same state a moment apart would each
    fill the same resting order and the log would record a position twice the size the agent
    ever asked for — permanently, because nothing here can un-append a row. This drives the
    two paths back to back over one crossed book, which is the interleaving that costs.
    """
    from basecradle_harness._polymarket_engine import main

    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10", order_type="limit", price="0.30")
        fake.books[YES_TOKEN] = book(bids=[("0.26", "500")], asks=[("0.28", "500")])

        main(["--home", str(tmp_path)])  # the cron job fills it
        main(["--home", str(tmp_path)])  # and again, a second later
        orders = call(tool, "get_orders", status="all")["orders"]  # and the wake's own re-check

    filled = sum(f["shares"] for order in orders for f in order["fills"])
    assert filled == 10.0  # exactly the size that was ordered, never twice
    assert call(tool, "get_positions")["positions"][0]["shares"] == 10.0


def test_the_store_lock_is_taken_by_both_entry_points():
    """A structural pin: the two writers each take it once, and `sweep` never takes it itself.

    `sweep()` runs inside the tool's lock, so a lock *there* would be a second exclusive flock
    from the same process — a deadlock, not a no-op. Keep the boundary at the entry points.
    """
    tool_src = (SOURCE / "_polymarket.py").read_text()
    engine_src = (SOURCE / "_polymarket_engine.py").read_text()
    assert "with store_lock(" in tool_src
    assert "with store_lock(" in engine_src
    body = engine_src.split("def sweep(", 1)[1].split("\ndef ", 1)[0]
    assert "store_lock" not in body.split('"""', 2)[-1]


def test_the_live_state_and_a_fresh_fold_agree_on_the_equity_curve(tmp_path):
    """A live mutation must move the curve exactly as a replay of the same rows does."""
    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="1000")
        fake.books[YES_TOKEN] = book(bids=[("0.04", "500")], asks=[("0.06", "500")])
        live = call(tool, "get_pnl")

    replayed = current_epoch(tmp_path).state()
    assert live["max_drawdown_pct"] == float(replayed.max_drawdown_pct.quantize(Decimal("0.01")))
    assert live["max_drawdown_pct"] > 0  # the drawdown shows on the call that caused it


def test_the_operator_can_freeze_and_unfreeze_from_the_job(tmp_path):
    from basecradle_harness._polymarket_engine import main

    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        main(["--home", str(tmp_path), "--freeze", "under review"])
        body = buy(tool)
        assert body["error"] == "frozen"
        assert "under review" in body["message"]
        main(["--home", str(tmp_path), "--unfreeze"])
        assert buy(tool)["ok"] is True


# --- reads ------------------------------------------------------------------------------


def test_list_markets_returns_the_normative_summary_fields(tmp_path):
    with upstream():
        body = call(make_tool(tmp_path), "list_markets")
    market = body["markets"][0]
    assert set(market) == {
        "market_id",
        "condition_id",
        "question",
        "slug",
        "outcomes",
        "mid_prices",
        "volume_24h",
        "liquidity",
        "end_date",
        "accepting_orders",
        "tick_size",
        "min_order_size",
        "tags",
        "event_id",
    }
    assert market["outcomes"] == ["Yes", "No"]
    assert market["mid_prices"] == [0.42, 0.58]
    assert market["tags"] == ["technology"]


def test_a_query_goes_through_public_search(tmp_path):
    with upstream() as fake:
        body = call(make_tool(tmp_path), "list_markets", query="harness")
    assert body["markets"][0]["market_id"] == MARKET_ID
    assert any(r.url.path == "/public-search" for r in fake.requests)


def test_a_tag_filter_resolves_the_slug_before_listing(tmp_path):
    with upstream() as fake:
        call(make_tool(tmp_path), "list_markets", tag="technology")
    paths = [r.url.path for r in fake.requests]
    assert "/tags/slug/technology" in paths
    listing = next(r for r in fake.requests if r.url.path == "/markets")
    assert listing.url.params["tag_id"] == "7"


def test_get_market_truncates_the_book_to_fifteen_levels_a_side(tmp_path):
    deep = [(f"0.{50 - i:02d}", "100") for i in range(20)]
    with upstream() as fake:
        fake.books[YES_TOKEN] = book(
            bids=deep, asks=[(f"0.{51 + i:02d}", "100") for i in range(20)]
        )
        body = call(make_tool(tmp_path), "get_market", market_id=MARKET_ID)
    yes = next(b for b in body["book"] if b["outcome"] == "Yes")
    assert len(yes["bids"]) == 15 and len(yes["asks"]) == 15
    assert yes["bids"][0][0] > yes["bids"][1][0]  # best first
    assert yes["asks"][0][0] < yes["asks"][1][0]


def test_get_market_reports_fees_tick_and_resolution_state(tmp_path):
    with upstream():
        body = call(make_tool(tmp_path), "get_market", market_id=MARKET_ID)
    market = body["market"]
    assert market["tick_size"] == 0.01
    assert market["min_order_size"] == 5.0
    assert market["fees"] == {"taker_bps": 0.0, "maker_bps": 0.0, "source": "market"}
    assert market["resolved"] is False and market["winning_outcomes"] == []


def test_get_fills_pages(tmp_path):
    with upstream() as fake:
        fake.books[YES_TOKEN] = book(bids=[("0.41", "500")], asks=[("0.43", "6"), ("0.45", "50")])
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")  # walks two ask levels -> two fills
        page = call(tool, "get_fills", limit=1)
    assert len(page["fills"]) == 1
    assert page["next_cursor"] == "1"
    assert page["total"] == 2


def test_get_orders_filters_by_status(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")  # market order -> closed immediately
        resting = buy(tool, shares="10", order_type="limit", price="0.30")
        assert resting["status"] == "open"
        assert len(call(tool, "get_orders", status="open")["orders"]) == 1
        assert len(call(tool, "get_orders", status="closed")["orders"]) == 1
        assert len(call(tool, "get_orders", status="all")["orders"]) == 2


def test_get_orders_rejects_an_unknown_status(tmp_path):
    with upstream():
        body = call(make_tool(tmp_path), "get_orders", status="pending")
    assert body["error"] == "invalid_params"


# --- the tool as a Harness citizen ---------------------------------------------------------


def test_the_plugin_ships_as_a_powerful_opt_in_default():
    from basecradle_harness._install import plugin_opts_in

    source = (SOURCE / "_defaults" / "tools" / "polymarket_paper.py").read_text()
    assert plugin_opts_in(source)


def test_the_tool_registers_under_the_locked_policy(tmp_path):
    from basecradle_harness import Harness, MemoryTool, OpenAIProvider

    agent = Harness(
        OpenAIProvider(model="gpt-5.4-mini", api_key="sk-test-000"),
        tools=[MemoryTool(), PolymarketPaperTool(root=tmp_path)],
    )
    assert "polymarket_paper" in agent.tools


def test_it_declares_no_shell_or_exec_capability(tmp_path):
    from basecradle_harness import SHELL

    assert SHELL not in PolymarketPaperTool(root=tmp_path).requires


def test_the_store_falls_back_to_an_operator_path_when_unbound(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "agent-home"))
    tool = PolymarketPaperTool(data=PolymarketData(cache_ttl=0))
    assert json.loads(tool.run(action="get_pnl"))["ok"] is True
    assert (tmp_path / "agent-home" / "polymarket").is_dir()


def test_a_bad_number_is_a_sentence_not_a_traceback(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        body = call(
            tool,
            "place_order",
            market_id=MARKET_ID,
            outcome="Yes",
            side="buy",
            size_shares="ten",
            order_type="market",
            client_order_id="coid-bad",
        )
    assert body["error"] == "invalid_params"
    assert "size_shares" in body["message"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outcome": "Yes", "side": "buy", "size_shares": "10", "order_type": "market"},
        {"market_id": MARKET_ID, "side": "buy", "size_shares": "10", "order_type": "market"},
        {"market_id": MARKET_ID, "outcome": "Yes", "size_shares": "10", "order_type": "market"},
    ],
)
def test_a_missing_required_field_is_invalid_params(tmp_path, kwargs):
    with upstream():
        body = call(make_tool(tmp_path), "place_order", client_order_id="coid-x", **kwargs)
    assert body["error"] == "invalid_params"


def test_the_data_client_caches_within_its_ttl(tmp_path):
    """A read-heavy op must not re-ask the CLOB once per position, every call."""
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4])
    data = PolymarketData(cache_ttl=60.0, now=lambda: next(ticks))
    with upstream() as fake:
        data.book(YES_TOKEN, "Yes")
        data.book(YES_TOKEN, "Yes")
        assert len([r for r in fake.requests if r.url.path == "/book"]) == 1


def test_decimal_prices_survive_the_round_trip(tmp_path):
    """A 0.001 tick must not become 0.30000000000000004 in a row that can never be corrected."""
    with upstream() as fake:
        fake.books[YES_TOKEN] = book(bids=[("0.11", "500")], asks=[("0.333", "500")])
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
    fill = next(r for r in current_epoch(tmp_path).rows() if r["type"] == "fill")
    assert fill["payload"]["price"] == "0.333"
    assert Decimal(fill["payload"]["notional"]) == Decimal("3.33")


# --- the hash chain: tamper-evidence, and refusal on a break -----------------------------


def test_the_rows_form_a_chain_from_genesis(tmp_path):
    from basecradle_harness._polymarket_ledger import GENESIS_PREV, row_hash

    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    rows = current_epoch(tmp_path).rows()
    assert rows[0]["prev"] == GENESIS_PREV
    previous = GENESIS_PREV
    for row in rows:
        assert row["prev"] == previous
        assert row["hash"] == row_hash(row)
        previous = row["hash"]
    assert current_epoch(tmp_path).head == rows[-1]["hash"]


def test_an_intact_chain_verifies(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    chain = current_epoch(tmp_path).verify()
    assert chain.ok is True
    assert chain.rows == len(current_epoch(tmp_path).rows())
    assert chain.broken_at is None


def tamper(tmp_path, index, mutate):
    """Rewrite one row in place — what an agent with a shell would do."""
    ledger = current_epoch(tmp_path).path
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    mutate(rows[index])
    ledger.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def test_editing_a_row_breaks_the_chain(tmp_path):
    """The whole point: a retroactive edit is detectable by replay."""
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)

    def inflate(row):
        row["payload"]["p"] = "0.99"

    forecast_index = next(
        i for i, r in enumerate(current_epoch(tmp_path).rows()) if r["type"] == "forecast"
    )
    tamper(tmp_path, forecast_index, inflate)
    chain = current_epoch(tmp_path).verify()
    assert chain.ok is False
    assert chain.broken_at == forecast_index
    assert "does not match its own hash" in chain.reason


def test_deleting_a_row_breaks_the_chain(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    ledger = current_epoch(tmp_path).path
    lines = [line for line in ledger.read_text().splitlines() if line.strip()]
    del lines[2]  # a row an agent would rather nobody replayed
    ledger.write_text("\n".join(lines) + "\n")
    chain = current_epoch(tmp_path).verify()
    assert chain.ok is False
    assert "altered, removed or reordered" in chain.reason


def test_a_row_lifted_from_another_epoch_breaks_the_chain(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    tamper(tmp_path, 1, lambda row: row.__setitem__("epoch_id", "epoch-somewhere-else"))
    chain = current_epoch(tmp_path).verify()
    assert chain.ok is False


def test_unchained_rows_from_before_the_chain_are_refused(tmp_path):
    """A chain cannot vouch for rows written before it existed, so it must not pretend to."""
    with upstream():
        make_tool(tmp_path).run(action="get_pnl")
    tamper(tmp_path, 0, lambda row: (row.pop("hash"), row.pop("prev")))
    chain = current_epoch(tmp_path).verify()
    assert chain.ok is False
    assert "predates the chained ledger" in chain.reason


@pytest.mark.parametrize(
    "action,kwargs",
    [
        ("get_pnl", {}),
        ("get_scorecard", {}),
        ("get_positions", {}),
        ("get_orders", {}),
        ("get_fills", {}),
        ("list_markets", {}),
        ("get_market", {"market_id": MARKET_ID}),
        ("log_forecast", {"market_id": MARKET_ID, "outcome": "Yes", "p": "0.5"}),
    ],
)
def test_a_broken_chain_returns_no_numbers_from_any_operation(tmp_path, action, kwargs):
    """The requirement, and the one failure this instrument cannot survive if softened.

    A scoreboard that reports numbers off a record it cannot vouch for — even with a warning
    attached — reads as a working scoreboard, and the governance layer's tampering trigger has
    no other detector. So the answer is an error and *no numbers at all*.
    """
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
        tamper(tmp_path, 1, lambda row: row["payload"].__setitem__("p", "0.99"))
        body = call(tool, action, **kwargs)

    assert body["ok"] is False
    assert body["error"] == "ledger_tampered"
    assert set(body) == {"ok", "error", "message", "budgets"}
    # Not one number from the record leaks out — not a price, not a count, not a budget.
    for leaked in ("cash_usd", "equity_usd", "realized_pnl", "brier", "positions", "markets"):
        assert leaked not in body
    assert body["budgets"] == {}


def test_a_broken_chain_refuses_writes_too(tmp_path):
    """A write onto an unverifiable chain buries the break under legitimate-looking rows."""
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
        before = len(current_epoch(tmp_path).rows())
        tamper(tmp_path, 1, lambda row: row["payload"].__setitem__("p", "0.99"))
        assert buy(tool)["error"] == "ledger_tampered"

    assert len(current_epoch(tmp_path).rows()) == before  # not even a `call` row was added


def test_the_sweep_refuses_to_extend_a_broken_chain(tmp_path, capsys):
    """The sweep runs with nobody watching, so it is the worst place to bury a break."""
    from basecradle_harness._polymarket_engine import main

    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
        tamper(tmp_path, 1, lambda row: row["payload"].__setitem__("p", "0.99"))
        before = len(current_epoch(tmp_path).rows())
        assert main(["--home", str(tmp_path)]) == 1  # non-zero: a cron job must say so

    assert "LEDGER CHAIN BROKEN" in capsys.readouterr().out
    assert len(current_epoch(tmp_path).rows()) == before


def test_the_verify_flag_reports_the_head_and_writes_nothing(tmp_path, capsys):
    from basecradle_harness._polymarket_engine import main

    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    epoch = current_epoch(tmp_path)
    before = epoch.path.read_bytes()

    assert main(["--home", str(tmp_path), "--verify"]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and epoch.head in out
    assert epoch.path.read_bytes() == before

    tamper(tmp_path, 1, lambda row: row["payload"].__setitem__("p", "0.99"))
    assert main(["--home", str(tmp_path), "--verify"]) == 1
    assert "BROKEN" in capsys.readouterr().out


def test_the_scorecard_publishes_the_chain_head_for_an_external_verifier(tmp_path):
    """The head plus the row count is what catches a tamperer who re-hashes the file forward."""
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
        pnl = call(tool, "get_pnl")
        # get_scorecard last: every call appends its own `call` row, so the head advances
        # with each one and only the final operation's head matches the file's.
        card = call(tool, "get_scorecard")

    epoch = current_epoch(tmp_path)
    assert card["chain_verified"] is True
    assert card["chain_head"] == epoch.head
    assert card["chain_rows"] == len(epoch.rows())
    assert len(pnl["chain_head"]) == 64 and pnl["chain_head"] != card["chain_head"]


def test_every_row_is_emitted_off_box_in_full(tmp_path, caplog):
    """The on-box file is a spool; the record is the copy journald ships out from under the UID.

    Full rows, not just heads: at the §2.2 ceiling that is a few hundred KB a day, and it buys
    a copy complete enough to *reconstruct* the ledger rather than only to notice it moved.
    """
    with upstream(), caplog.at_level("INFO", logger="basecradle_harness"):
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)

    emitted = [
        json.loads(rec.message.split(" ", 1)[1])
        for rec in caplog.records
        if rec.message.startswith("polymarket_ledger_row ")
    ]
    on_disk = current_epoch(tmp_path).rows()
    assert emitted == on_disk  # every row, in order, byte-for-byte equivalent


def test_a_rewritten_chain_still_fails_against_the_pinned_head(tmp_path):
    """A tamperer who re-hashes the whole file forward produces a *valid* chain, so the
    on-box verdict alone cannot catch them — the off-box head is what does."""
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    pinned_head = current_epoch(tmp_path).head
    pinned_rows = len(current_epoch(tmp_path).rows())

    # Rewrite the forecast and re-chain every row after it, exactly as an attacker would.
    from basecradle_harness._polymarket_ledger import GENESIS_PREV, row_hash

    ledger = current_epoch(tmp_path).path
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    rows[1]["payload"]["p"] = "0.99"
    previous = GENESIS_PREV
    for row in rows:
        row["prev"] = previous
        row["hash"] = row_hash(row)
        previous = row["hash"]
    ledger.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    assert current_epoch(tmp_path).verify().ok is True  # internally consistent...
    assert current_epoch(tmp_path).head != pinned_head  # ...but not the record that was shipped
    assert len(current_epoch(tmp_path).rows()) == pinned_rows


# --- the read-only epoch probe (issue #353) ----------------------------------
#
# The freeze is a live operational lever used as a safety control on an armed adversarial
# persona — and until this it was reachable only through a tool call inside a wake (which no
# monitor can make) or by reading the ledger on the box (which needs a shell nobody has). So
# `polymarket_paper` being *armed* was git-tracked and drift-audited, while being *frozen* was
# neither declared nor readable: a write-only control on a security boundary.


def _probe(home, *extra):
    """Run `--verify --json` and return the parsed report plus the exit code."""
    import io

    from basecradle_harness._polymarket_engine import main

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(["--home", str(home), "--verify", "--json", *extra])
    return json.loads(buffer.getvalue()), code


def test_the_probe_reports_the_epochs_freeze_state(tmp_path):
    from basecradle_harness._polymarket_engine import main

    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    epoch = current_epoch(tmp_path)

    report, code = _probe(tmp_path)

    assert code == 0
    assert report["chain_ok"] is True
    assert report["epoch"]["epoch_id"] == epoch.epoch_id
    assert report["epoch"]["frozen"] is False and report["epoch"]["frozen_reason"] == ""
    assert report["epoch"]["head"] == epoch.head
    assert report["epoch"]["rows"] == len(epoch.rows())
    assert report["epoch"]["chain_ok"] is True

    assert main(["--home", str(tmp_path), "--freeze", "verifying off-box log shipping"]) == 0
    report, code = _probe(tmp_path)

    assert code == 0  # a freeze is a state, never an error
    assert report["epoch"]["frozen"] is True
    assert report["epoch"]["frozen_reason"] == "verifying off-box log shipping"

    assert main(["--home", str(tmp_path), "--unfreeze"]) == 0
    report, _ = _probe(tmp_path)
    assert report["epoch"]["frozen"] is False  # the lift is visible too, which is the point


def test_the_probe_distinguishes_no_epoch_from_an_unfrozen_one(tmp_path):
    """A freshly provisioned agent has no epoch at all. Reporting that as `frozen: false` would
    make the audit say a *nonexistent* instrument is running unfrozen — the two are different
    facts about a box and an axis built on them must be able to tell them apart."""
    report, code = _probe(tmp_path)

    assert code == 0
    assert report["epoch"] is None and report["epochs"] == []
    assert report["chain_ok"] is True  # nothing is broken; there is simply nothing


def test_the_probe_refuses_to_answer_frozen_off_a_broken_chain(tmp_path):
    """The freeze is folded out of exactly the rows whose integrity just failed. A tamperer who
    removed the `freeze` row would otherwise be reported as `frozen: false` — the one wrong
    answer that matters, on a control whose whole purpose is to stop trading. `null` is the only
    honest state, and `chain_ok` sits beside it saying why."""
    from basecradle_harness._polymarket_engine import main

    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    assert main(["--home", str(tmp_path), "--freeze", "emergency halt"]) == 0
    assert _probe(tmp_path)[0]["epoch"]["frozen"] is True  # control: it really was frozen

    tamper(tmp_path, 1, lambda row: row["payload"].__setitem__("p", "0.99"))
    report, code = _probe(tmp_path)

    assert code == 1  # a monitor's cheapest signal
    assert report["chain_ok"] is False and report["epoch"]["chain_ok"] is False
    assert report["epoch"]["frozen"] is None  # never `false`
    assert report["epoch"]["frozen_reason"] is None
    assert report["epoch"]["broken_at"] == 1 and report["epoch"]["reason"]
    # rows/head are the *verified prefix* — how far the record vouches for itself.
    assert report["epoch"]["rows"] == 1


def test_a_truncated_tail_is_not_detectable_on_box_and_the_pin_is_what_catches_it(tmp_path):
    """The limit of the refusal above, pinned rather than assumed — because an axis built on
    `chain_ok` alone has a blind spot exactly where a tamperer would aim.

    A chain catches an edit or a removal *inside* the log: the next row's `prev` stops matching.
    It cannot catch a **truncated tail** — lopping the final rows off leaves a shorter chain
    that verifies perfectly, so a trailing `freeze` deleted from the end reads `chain_ok: true`
    with `frozen: false`, honestly describing a log that is itself a lie. That is not a defect
    here (no on-box check can help: the harness runs as the agent's own UID, so any expected-
    length marker is equally writable) — it is why `rows` and `head` are published, as the pin
    against the off-box row copy. This test exists so nobody reads the `frozen: null` refusal as
    covering more than it does, and so a later "fix" that makes the tail case merely *look*
    handled fails here.
    """
    from basecradle_harness._polymarket_engine import main

    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    assert main(["--home", str(tmp_path), "--freeze", "emergency halt"]) == 0
    epoch = current_epoch(tmp_path)
    intact = _probe(tmp_path)[0]["epoch"]
    assert intact["frozen"] is True and intact["chain_ok"] is True

    # Truncate the tail: drop the trailing `freeze` row, and nothing else.
    rows = epoch.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[-1])["type"] == "freeze"
    epoch.path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    report, code = _probe(tmp_path)

    # The chain still verifies — this is the honest, documented limit.
    assert code == 0 and report["chain_ok"] is True
    assert report["epoch"]["frozen"] is False  # the log now says un-frozen, and it verifies
    # ...and the pin is what gives it away: both moved, against the off-box copy of the row.
    assert report["epoch"]["rows"] == intact["rows"] - 1
    assert report["epoch"]["head"] != intact["head"]


def test_the_probe_writes_nothing_at_all(tmp_path):
    """Including on an agent that has never traded: taking the store lock would create
    `<home>/polymarket/` and a `.lock`, so an hourly fleet monitor would litter a paper-trading
    store onto every box it audited — the audit changing the thing it audits."""

    from basecradle_harness._polymarket_engine import main

    def tree(root):
        return {path: path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}

    assert _probe(tmp_path)[1] == 0
    assert list(tmp_path.iterdir()) == []  # not even the store dir, nor its .lock

    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    before = tree(tmp_path)
    assert before  # the control: there is now something that *could* have been written to

    for _ in range(3):  # safe to run repeatedly, which is what a monitor does
        assert _probe(tmp_path)[1] == 0
        assert _probe(tmp_path, "--head-at", "3")[1] == 0  # the prefix read is read-only too
        assert main(["--home", str(tmp_path), "--verify"]) == 0  # the text mode too

    assert tree(tmp_path) == before  # every byte under the home, ledger and lock alike


def test_the_probe_reports_every_epoch_and_keeps_the_current_one_named(tmp_path):
    from basecradle_harness._polymarket_engine import main

    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    first = current_epoch(tmp_path).epoch_id
    assert main(["--home", str(tmp_path), "--new-epoch"]) == 0
    second = current_epoch(tmp_path).epoch_id
    assert second != first

    report, code = _probe(tmp_path, "--all-epochs")

    assert code == 0
    assert [entry["epoch_id"] for entry in report["epochs"]] == [first, second]
    # `epoch` stays the *current* one either way, so a monitor's axis reads the same field
    # whether or not history was asked for.
    assert report["epoch"]["epoch_id"] == second
    assert _probe(tmp_path)[0]["epoch"]["epoch_id"] == second


def test_json_is_refused_without_verify_because_every_other_mode_writes(tmp_path):
    """`--json` is the monitor's spelling of `--verify`; binding them is what keeps "safe to run
    repeatedly" a property of the flag rather than of the caller's care."""
    from basecradle_harness._polymarket_engine import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--home", str(tmp_path), "--json"])
    assert exit_info.value.code == 2  # argparse's usage error
    assert list(tmp_path.iterdir()) == []  # and it wrote nothing on the way out


# --- the prefix read an off-box witness needs (issue #395) -------------------
#
# `head` answers "what is the head *now*", which goes stale the moment the ledger grows: a
# witness holding `(K, H_K)` from an hour ago has nothing left to compare it against, so
# rows-above-the-witness was audited as honest growth with nothing checked. Every row's hash
# commits to the whole prefix before it, so the head *at* K is still a fact the box can state —
# and a truncate-and-refill cannot reproduce it.


def rechain(home):
    """Re-`prev` and re-`hash` every row on disk — what a tamperer does to make a file verify."""
    from basecradle_harness._polymarket_ledger import GENESIS_PREV, row_hash

    ledger = current_epoch(home).path
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    previous = GENESIS_PREV
    for row in rows:
        row["prev"] = previous
        row["hash"] = row_hash(row)
        previous = row["hash"]
    ledger.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows


def test_head_at_the_witnessed_count_is_that_witnesss_head(tmp_path):
    """The contract the NOC runner binds to: `--head-at <witness.rows>` yields `witness.head`.

    A row count, never a zero-based index — the witness records `rows` and `head` together, and
    a surface that made the caller convert between the two would be a subtraction somebody
    eventually gets wrong in the one place nobody re-derives it.
    """
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    epoch = current_epoch(tmp_path)
    witnessed_rows, witnessed_head = len(epoch.rows()), epoch.head

    report, code = _probe(tmp_path, "--head-at", str(witnessed_rows))

    assert code == 0
    assert report["epoch"]["head_at_rows"] == witnessed_rows  # echoed: the answer self-describes
    assert report["epoch"]["head_at"] == witnessed_head == report["epoch"]["head"]
    assert report["epoch"]["head_at_reason"] == ""


def test_head_at_answers_the_witnesss_question_after_the_ledger_has_grown(tmp_path):
    """The whole point. Once the log has grown past K, `head` no longer answers anything the
    witness asked — but the head *at* K still does, over a prefix that is now interior."""
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
        witnessed_rows = len(current_epoch(tmp_path).rows())
        witnessed_head = current_epoch(tmp_path).head
        call(tool, "get_pnl")  # every call appends its own row: the ledger grows honestly
        call(tool, "get_orders")

    report, code = _probe(tmp_path, "--head-at", str(witnessed_rows))

    assert code == 0
    assert report["epoch"]["rows"] > witnessed_rows  # grown...
    assert report["epoch"]["head"] != witnessed_head  # ...so the live head has moved on
    assert report["epoch"]["head_at"] == witnessed_head  # ...and the prefix is still intact


def test_head_at_catches_a_truncate_and_refill_that_reads_as_pure_growth(tmp_path):
    """The attack the third arm of the off-box compare could not see (basecradle-noc#458).

    Truncate *below* the witnessed count, refill *past* it, re-chain the file: the result
    verifies perfectly, holds more rows than the witness recorded, and so presents to an audit
    that only pins `(rows, head)` as ordinary growth. The prefix hash is what gives it away —
    the refilled rows cannot reproduce `H_K`, because `H_K` commits to every row before it.
    """
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    epoch = current_epoch(tmp_path)
    witnessed_rows, witnessed_head = len(epoch.rows()), epoch.head
    assert witnessed_rows > 3  # the control: there is a prefix here worth truncating into

    rows = epoch.rows()
    kept = rows[: witnessed_rows - 2]  # truncate below the witness...
    for number in range(4):  # ...and refill past it with rows nobody witnessed
        forged = json.loads(json.dumps(rows[-1]))
        forged["ts"] = f"2026-08-09T0{number}:00:00Z"
        kept.append(forged)
    epoch.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    rechain(tmp_path)

    report, code = _probe(tmp_path, "--head-at", str(witnessed_rows))

    assert code == 0 and report["chain_ok"] is True  # internally consistent: the blind spot
    assert report["epoch"]["rows"] > witnessed_rows  # and it presents as growth
    assert report["epoch"]["head_at"] != witnessed_head  # ...but the prefix is not the one pinned
    assert report["epoch"]["head_at"]  # a hash *was* asserted — the caller does the comparing
    assert report["epoch"]["head_at_reason"] == ""


def test_head_at_asserts_no_hash_past_a_break(tmp_path):
    """A prefix that does not verify has no head, so the probe states the break and no hash.

    Asserting one would be the probe vouching for rows the chain just refused to vouch for —
    the same refusal `frozen: null` makes, applied to the one field a witness compares.
    """
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    total = len(current_epoch(tmp_path).rows())
    tamper(tmp_path, 1, lambda row: row["payload"].__setitem__("p", "0.99"))

    report, code = _probe(tmp_path, "--head-at", str(total))

    assert code == 1  # the broken chain, exactly as today
    assert report["epoch"]["chain_ok"] is False and report["epoch"]["broken_at"] == 1
    assert report["epoch"]["head_at"] is None
    assert "does not verify" in report["epoch"]["head_at_reason"]

    # ...but the part that *did* verify still answers: row 1 is the break, so a prefix of 1 holds.
    verified, _ = _probe(tmp_path, "--head-at", "1")
    assert verified["epoch"]["head_at"] == verified["epoch"]["head"]  # the verified prefix's head
    assert verified["epoch"]["head_at_reason"] == ""


def test_head_at_beyond_the_log_is_an_answer_not_an_error(tmp_path):
    """A ledger shorter than the witness is the caller's truncation signal, and the box has no
    opinion about it — from here it is simply a smaller number. So it exits 0 and says so
    plainly, rather than being laundered into the broken-chain verdict it is not."""
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    total = len(current_epoch(tmp_path).rows())

    report, code = _probe(tmp_path, "--head-at", str(total + 5))

    assert code == 0 and report["chain_ok"] is True  # an answer crosses as 0
    assert report["epoch"]["head_at"] is None
    assert report["epoch"]["head_at_rows"] == total + 5
    assert f"holds {total} verified row(s)" in report["epoch"]["head_at_reason"]


def test_head_at_zero_is_genesis_and_no_epoch_is_still_no_epoch(tmp_path):
    """The empty prefix verifies trivially and its head is what the first row chains from — the
    honest answer to a witness that pinned an epoch before it had written anything."""
    from basecradle_harness._polymarket_ledger import GENESIS_PREV

    empty, code = _probe(tmp_path, "--head-at", "0")
    assert code == 0 and empty["epoch"] is None  # no epoch stays a different fact from no rows

    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)

    report, code = _probe(tmp_path, "--head-at", "0")
    assert code == 0 and report["epoch"]["head_at"] == GENESIS_PREV


def test_head_at_is_absent_unless_it_was_asked_for(tmp_path):
    """Backward compatible, and *absent* rather than `null`: a runner that forgot the flag must
    not read as a ledger that had no answer. The keys appear only when a caller asked."""
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)

    report, code = _probe(tmp_path)

    assert code == 0
    assert "head_at" not in report["epoch"] and "head_at_rows" not in report["epoch"]
    assert "head_at_reason" not in report["epoch"]
    assert report["epoch"]["head"]  # everything it always reported is untouched


def test_head_at_reports_per_epoch_and_stays_legible_without_json(tmp_path, capsys):
    from basecradle_harness._polymarket_engine import main

    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool)
    first = current_epoch(tmp_path).epoch_id
    assert main(["--home", str(tmp_path), "--new-epoch"]) == 0
    second = current_epoch(tmp_path).epoch_id
    capsys.readouterr()

    report, code = _probe(tmp_path, "--all-epochs", "--head-at", "2")

    # Each epoch answers for itself: the same count, against its own chain.
    assert code == 0
    assert [entry["epoch_id"] for entry in report["epochs"]] == [first, second]
    assert report["epochs"][0]["head_at"] and report["epochs"][0]["head_at_reason"] == ""
    assert report["epochs"][1]["head_at"] is None  # a fresh one-row epoch has no prefix of 2

    # The human line appends, never splices: an off-box reader parsing the existing pairs keeps
    # working, and a fresh one gets the answer at the end.
    assert main(["--home", str(tmp_path), "--verify", "--all-epochs", "--head-at", "2"]) == 0
    out = capsys.readouterr().out
    assert f"{first}: OK rows={report['epochs'][0]['rows']} " in out
    assert f"head_at[2]={report['epochs'][0]['head_at']}" in out
    assert "head_at[2]=none" in out  # the second epoch, said plainly rather than omitted
    assert report["epochs"][1]["head_at_reason"] in out  # ...with the why on the line below


def test_head_at_is_refused_without_verify_and_refuses_a_negative_count(tmp_path):
    """Bound to `--verify` for the reason `--json` is — every other mode of this command writes.
    A negative count is the *caller's* mistake, and answering `null` would launder a runner bug
    into a ledger finding."""
    from basecradle_harness._polymarket_engine import main

    for argv in (["--head-at", "3"], ["--verify", "--head-at", "-1"]):
        with pytest.raises(SystemExit) as exit_info:
            main(["--home", str(tmp_path), *argv])
        assert exit_info.value.code == 2  # argparse's usage error
    assert list(tmp_path.iterdir()) == []  # and it wrote nothing on the way out
