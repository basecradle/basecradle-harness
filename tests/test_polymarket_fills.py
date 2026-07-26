"""The deterministic fill model, settlement, and the scorecard (issue #347, §2.4 / §A2 / §2.3).

These read as the specification of the simulation: what a market order does to a thin book,
where a limit order rests, what a maker fill costs, when a Brier observation is locked and
what it is scored against. The upstream double and the fabricated market live in
`test_polymarket.py`; this file is only about what the harness does with them.
"""

from __future__ import annotations

import json
from decimal import Decimal

from basecradle_harness._polymarket_engine import (
    calibration_error,
    scorecard,
)
from basecradle_harness._polymarket_ledger import (
    GENESIS_PREV,
    SCHEMA_VERSION,
    Epoch,
    Observation,
    PaperState,
    current_epoch,
    open_epoch,
    row_hash,
    store_root,
)
from tests.test_polymarket import (
    CONDITION_ID,
    MARKET_ID,
    NO_TOKEN,
    YES_TOKEN,
    book,
    buy,
    call,
    clob_market,
    forecast,
    make_tool,
    upstream,
)


def rows(tmp_path, kind=None):
    all_rows = current_epoch(tmp_path).rows()
    return [r for r in all_rows if kind is None or r["type"] == kind]


def sell(tool, *, shares, order_type="market", price=None, coid="coid-sell"):
    return call(
        tool,
        "place_order",
        market_id=MARKET_ID,
        outcome="Yes",
        side="sell",
        size_shares=shares,
        order_type=order_type,
        limit_price=price,
        client_order_id=coid,
    )


# --- §2.4: the walk ------------------------------------------------------------------


def test_a_market_buy_walks_the_ask_side_fifo_by_price(tmp_path):
    with upstream() as fake:
        fake.books[YES_TOKEN] = book(
            bids=[("0.41", "500")], asks=[("0.43", "6"), ("0.45", "20"), ("0.50", "100")]
        )
        tool = make_tool(tmp_path)
        forecast(tool)
        result = buy(tool, shares="10")

    assert result["status"] == "filled"
    assert [(f["price"], f["shares"]) for f in result["fills"]] == [(0.43, 6.0), (0.45, 4.0)]
    assert result["notional_usd"] == 6 * 0.43 + 4 * 0.45
    assert result["avg_fill_price"] == round((6 * 0.43 + 4 * 0.45) / 10, 10)


def test_a_market_sell_walks_the_bid_side(tmp_path):
    with upstream() as fake:
        fake.books[YES_TOKEN] = book(bids=[("0.41", "6"), ("0.39", "50")], asks=[("0.43", "500")])
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        result = sell(tool, shares="10")

    assert [(f["price"], f["shares"]) for f in result["fills"]] == [(0.41, 6.0), (0.39, 4.0)]


def test_a_market_order_cancels_its_remainder_and_never_leaves_a_phantom(tmp_path):
    with upstream() as fake:
        fake.books[YES_TOKEN] = book(bids=[("0.41", "500")], asks=[("0.43", "6")])
        tool = make_tool(tmp_path)
        forecast(tool)
        result = buy(tool, shares="10")

    assert result["status"] == "partially_filled"
    assert result["filled_shares"] == 6.0
    assert result["remaining_shares"] == 4.0
    assert "remainder cancelled" in result["close_reason"]
    assert call(tool, "get_orders", status="open")["orders"] == []


def test_a_limit_buy_below_the_best_ask_rests_untouched(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        result = buy(tool, shares="10", order_type="limit", price="0.30")

    assert result["status"] == "open"
    assert result["filled_shares"] == 0.0
    assert result["fills"] == []


def test_a_limit_buy_takes_the_marketable_part_and_rests_the_rest(tmp_path):
    with upstream() as fake:
        fake.books[YES_TOKEN] = book(bids=[("0.41", "500")], asks=[("0.43", "6"), ("0.60", "100")])
        tool = make_tool(tmp_path)
        forecast(tool)
        result = buy(tool, shares="10", order_type="limit", price="0.45")

    assert result["status"] == "open"
    assert result["filled_shares"] == 6.0  # only the 0.43 level is at or below 0.45
    assert result["remaining_shares"] == 4.0
    assert result["fills"][0]["liquidity"] == "taker"


def test_a_limit_sell_above_the_best_bid_rests(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        result = sell(tool, shares="5", order_type="limit", price="0.90", coid="coid-rest-sell")

    assert result["status"] == "open"
    assert result["filled_shares"] == 0.0


def test_a_tick_misaligned_limit_price_is_refused(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        body = buy(tool, shares="10", order_type="limit", price="0.4237")
    assert body["error"] == "invalid_params"
    assert "tick size" in body["message"]


def test_a_limit_order_without_a_price_is_refused(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        body = buy(tool, shares="10", order_type="limit")
    assert body["error"] == "invalid_params"
    assert "limit_price" in body["message"]


# --- §2.4: resting fills happen on the sweep, at the maker's own price ------------------


def test_a_resting_order_fills_as_maker_at_its_own_price_when_the_book_crosses(tmp_path):
    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool)
        resting = buy(tool, shares="10", order_type="limit", price="0.30")
        assert resting["status"] == "open"

        # The market falls: someone is now offering at 0.28, below our resting bid.
        fake.books[YES_TOKEN] = book(bids=[("0.26", "500")], asks=[("0.28", "50")])
        after = call(tool, "get_orders", status="all")["orders"][0]

    assert after["status"] == "filled"
    assert after["fills"][0]["price"] == 0.30  # the maker's own price, not the 0.28 cross
    assert after["fills"][0]["liquidity"] == "maker"


def test_a_resting_order_takes_only_the_depth_that_is_there(tmp_path):
    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10", order_type="limit", price="0.30")
        fake.books[YES_TOKEN] = book(bids=[("0.26", "500")], asks=[("0.28", "4")])
        after = call(tool, "get_orders", status="all")["orders"][0]

    assert after["status"] == "open"
    assert after["filled_shares"] == 4.0
    assert after["remaining_shares"] == 6.0


def test_cancelling_a_resting_order_closes_it(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        resting = buy(tool, shares="10", order_type="limit", price="0.30")
        body = call(tool, "cancel_order", order_id=resting["order_id"])

    assert body["cancelled"] is True
    assert body["order"]["status"] == "cancelled"
    assert call(tool, "get_orders", status="open")["orders"] == []


def test_cancelling_a_settled_order_is_a_structured_refusal(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        done = buy(tool, shares="10")
        body = call(tool, "cancel_order", order_id=done["order_id"])
    assert body["error"] == "invalid_params"
    assert "only a resting order" in body["message"]


def test_cancelling_an_unknown_order_is_not_found(tmp_path):
    with upstream():
        body = call(make_tool(tmp_path), "cancel_order", order_id="ord-999999")
    assert body["error"] == "not_found"


# --- §2.4: fees ---------------------------------------------------------------------------


def test_a_published_zero_is_a_rate_not_a_silence(tmp_path):
    """§2.4 forbids *silently* zeroing a fee; honestly reading a venue's zero is the opposite.

    A fee-free market publishes ``0`` on both sides (live: every market with
    ``feesEnabled: false``), and that zero is a fact, recorded as read.
    """
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        fill = buy(tool, shares="10")["fills"][0]
    assert fill["fee_bps"] == 0.0
    assert fill["fee_source"] == "market"
    assert fill["fee_usd"] == 0.0


def test_a_fee_charging_market_uses_the_contract_default_and_says_so(tmp_path):
    """The published integer is a flag, not a notional rate — so §2.4's default applies.

    Live, ``taker_base_fee``/``maker_base_fee`` read ``1000`` on *every* fee-charging market
    regardless of its category rate. Read as basis points of notional that is a 10% charge on a
    trade the venue charges about 1% on, and it would bill a maker on a venue that never bills
    makers. The rate is genuinely unavailable, so the fill takes the contract's default and is
    tagged ``default`` — the number is the harness's and the row says so.
    """
    with upstream() as fake:
        fake.clob[CONDITION_ID] = clob_market(taker_base_fee=1000, maker_base_fee=1000)
        tool = make_tool(tmp_path)
        forecast(tool)
        result = buy(tool, shares="10")

    fill = result["fills"][0]
    assert fill["fee_bps"] == 100.0  # not 1000
    assert fill["fee_source"] == "default"
    assert fill["fee_usd"] == round(10 * 0.43 * 0.01, 6)


def test_a_resting_fill_on_a_fee_charging_market_pays_no_maker_fee(tmp_path):
    """The venue documents that makers are never charged; a published 1000 must not bill one."""
    with upstream() as fake:
        fake.clob[CONDITION_ID] = clob_market(taker_base_fee=1000, maker_base_fee=1000)
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10", order_type="limit", price="0.30")
        fake.books[YES_TOKEN] = book(bids=[("0.26", "500")], asks=[("0.28", "50")])
        after = call(tool, "get_orders", status="all")["orders"][0]

    maker_fill = after["fills"][0]
    assert maker_fill["liquidity"] == "maker"
    assert maker_fill["fee_usd"] == 0.0


def test_an_unpublished_fee_falls_back_to_one_hundred_bps_and_says_so(tmp_path):
    with upstream() as fake:
        market = clob_market()
        del market["taker_base_fee"]
        del market["maker_base_fee"]
        fake.clob[CONDITION_ID] = market
        tool = make_tool(tmp_path)
        forecast(tool)
        fill = buy(tool, shares="10")["fills"][0]

    assert fill["fee_bps"] == 100.0
    assert fill["fee_source"] == "default"


def test_fees_leave_cash_and_are_booked_as_realized(tmp_path):
    with upstream() as fake:
        fake.clob[CONDITION_ID] = clob_market(taker_base_fee=1000)
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        pnl = call(tool, "get_pnl")

    notional = Decimal("10") * Decimal("0.43")
    fee = notional * Decimal("0.01")
    assert Decimal(str(pnl["cash_usd"])) == (Decimal("10000") - notional - fee).quantize(
        Decimal("0.01")
    )
    assert pnl["fees_paid"] == float(fee.quantize(Decimal("0.01")))


# --- §2.3: the caps -------------------------------------------------------------------------


def test_an_order_over_the_notional_cap_is_refused(tmp_path):
    with upstream() as fake:
        fake.books[YES_TOKEN] = book(bids=[("0.41", "5000")], asks=[("0.43", "5000")])
        tool = make_tool(tmp_path)
        forecast(tool)
        body = buy(tool, shares="2000")  # 2000 x 0.43 = $860
    assert body["error"] == "cap_exceeded"
    assert "per-order cap" in body["message"]


def test_the_per_market_exposure_cap_counts_held_and_working_notional(tmp_path):
    with upstream() as fake:
        fake.books[YES_TOKEN] = book(bids=[("0.41", "50000")], asks=[("0.43", "50000")])
        tool = make_tool(tmp_path)
        forecast(tool)
        for _ in range(4):
            assert buy(tool, shares="1000")["ok"] is True  # 4 x $430 = $1,720
        body = buy(tool, shares="1000")
    assert body["error"] == "cap_exceeded"
    assert "per-market cap" in body["message"]


def test_the_open_position_cap_counts_resting_buys_too(tmp_path):
    with upstream() as fake:
        tool = make_tool(tmp_path)
        for index in range(1, 22):
            market_id = fake.add_market(index)
            forecast(tool, market_id=market_id)
            body = buy(tool, shares="10", market_id=market_id)
            if index <= 20:
                assert body["ok"] is True, (index, body)
            else:
                assert body["error"] == "cap_exceeded"
                assert "20 positions" in body["message"]


def test_a_buy_beyond_available_cash_is_refused(tmp_path):
    with upstream() as fake:
        fake.books[YES_TOKEN] = book(bids=[("0.41", "500000")], asks=[("0.43", "500000")])
        tool = make_tool(tmp_path)
        forecast(tool)
        # Under the per-order cap ($500) but repeated until the $10,000 bankroll is gone.
        # The per-market exposure cap bites first at $2,000, so spread across markets.
        for index in range(1, 6):
            market_id = fake.add_market(index, asks=(("0.31", "500000"),))
            forecast(tool, market_id=market_id)
            for _ in range(4):
                buy(tool, shares="1600", market_id=market_id)  # 4 x $496 = $1,984 per market
        pnl = call(tool, "get_pnl")
        assert pnl["cash_usd"] < 200
        body = buy(tool, shares="1000")
    assert body["error"] == "insufficient_cash"


def test_selling_more_than_you_hold_is_refused(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        body = sell(tool, shares="25")
    assert body["error"] == "insufficient_shares"
    assert "no short side" in body["message"]


def test_a_resting_sell_reserves_the_shares_behind_it(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        sell(tool, shares="8", order_type="limit", price="0.90", coid="coid-rest")
        body = sell(tool, shares="5", coid="coid-second")
    assert body["error"] == "insufficient_shares"


def test_a_size_below_the_market_minimum_is_refused(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        body = buy(tool, shares="2")
    assert body["error"] == "size_too_small"


def test_a_market_not_accepting_orders_is_refused(tmp_path):
    with upstream() as fake:
        fake.clob[CONDITION_ID] = clob_market(accepting_orders=False)
        tool = make_tool(tmp_path)
        forecast(tool)
        body = buy(tool, shares="10")
    assert body["error"] == "market_closed"


def test_a_market_without_an_order_book_is_not_implemented(tmp_path):
    with upstream() as fake:
        fake.clob[CONDITION_ID] = clob_market(enable_order_book=False)
        tool = make_tool(tmp_path)
        body = forecast(tool)
    assert body["error"] == "not_implemented"


def test_an_unknown_outcome_is_not_found(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        body = forecast(tool, outcome="Maybe")
    assert body["error"] == "not_found"
    assert "Yes" in body["message"]


# --- §2.4: settlement ---------------------------------------------------------------------


def test_settlement_pays_a_dollar_to_the_winner(tmp_path):
    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool, p="0.7")
        buy(tool, shares="10")  # 10 @ 0.43 = $4.30
        fake.resolve(winner="Yes")
        pnl = call(tool, "get_pnl")

    assert call(tool, "get_positions")["positions"] == []
    assert pnl["cash_usd"] == float(
        (Decimal("10000") - Decimal("4.30") + Decimal("10")).quantize(Decimal("0.01"))
    )
    assert pnl["realized_pnl"] == 5.7
    settlement = rows(tmp_path, "settlement")[0]["payload"]
    assert settlement["won"] is True
    assert settlement["payout"] == "10.000000"
    assert settlement["resolution_source"] == "clob_public_market_state"


def test_settlement_pays_nothing_to_the_loser(tmp_path):
    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool, p="0.7")
        buy(tool, shares="10")
        fake.resolve(winner="No")
        pnl = call(tool, "get_pnl")

    assert pnl["realized_pnl"] == -4.3
    assert rows(tmp_path, "settlement")[0]["payload"]["payout"] == "0.000000"


def test_resolution_cancels_what_was_still_resting(tmp_path):
    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool)
        resting = buy(tool, shares="10", order_type="limit", price="0.30")
        fake.resolve(winner="Yes")
        orders = call(tool, "get_orders", status="all")["orders"]
    closed = next(o for o in orders if o["order_id"] == resting["order_id"])
    assert closed["status"] == "cancelled"
    assert closed["close_reason"] == "market resolved"


def test_only_public_state_can_resolve_a_market(tmp_path):
    """There is no parameter, anywhere, through which the agent could assert an outcome."""
    with upstream():
        tool = make_tool(tmp_path)
        blob = json.dumps(tool.parameters).casefold()
        assert "resolve" not in blob and "winner" not in blob and "settle" not in blob


# --- §A2: Brier attribution is position-open, and frozen --------------------------------------


def test_the_observation_is_locked_from_the_forecast_current_at_position_open(tmp_path):
    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool, p="0.7")
        buy(tool, shares="10")  # position opens under p=0.7
        forecast(tool, p="0.2")  # a later, superseding forecast
        buy(tool, shares="10")  # a sized add — not a new observation
        fake.resolve(winner="Yes")
        call(tool, "get_scorecard")

    observations = rows(tmp_path, "brier_obs")
    assert len(observations) == 1
    assert observations[0]["payload"]["p"] == "0.7"
    assert observations[0]["payload"]["attribution"] == "position_open"
    score = rows(tmp_path, "brier_score")[0]["payload"]
    assert score["result"] == 1
    assert Decimal(score["brier"]) == Decimal("0.09")  # (0.7 - 1)^2


def test_an_observation_is_scored_even_when_the_position_was_closed_early(tmp_path):
    """Otherwise the calibration record only grades the trades that were held to the end."""
    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool, p="0.8")
        buy(tool, shares="10")
        sell(tool, shares="10")  # flat again, well before resolution
        assert call(tool, "get_positions")["positions"] == []
        fake.resolve(winner="No")
        card = call(tool, "get_scorecard")

    assert card["resolved_n"] == 1
    assert card["brier"] == 0.64  # (0.8 - 0)^2


def test_re_opening_a_closed_position_locks_a_fresh_observation(tmp_path):
    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool, p="0.7")
        buy(tool, shares="10")
        sell(tool, shares="10")
        forecast(tool, p="0.4")
        buy(tool, shares="10")
        fake.resolve(winner="Yes")
        call(tool, "get_scorecard")

    scored = [r["payload"] for r in rows(tmp_path, "brier_score")]
    assert sorted(Decimal(s["brier"]) for s in scored) == [Decimal("0.09"), Decimal("0.36")]


def test_a_forecast_supersedes_rather_than_overwrites_in_the_log(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool, p="0.7")
        forecast(tool, p="0.2")
    logged = [r["payload"]["p"] for r in rows(tmp_path, "forecast")]
    assert logged == ["0.7", "0.2"]  # both rows survive; the later one governs


# --- §2.3: the scorecard ------------------------------------------------------------------


def observed(p, result, event="e1"):
    obs = Observation(
        obs_id=f"obs-{p}-{result}-{event}",
        market_id="m",
        outcome="Yes",
        p=Decimal(p),
        forecast_id="fc-1",
        opened_at="",
        event_id=event,
    )
    obs.result = result
    obs.brier = (obs.p - Decimal(result)) ** 2
    return obs


def state_with(observations) -> PaperState:
    state = PaperState(epoch_id="epoch-test", cash=Decimal("10000"))
    state.observations = {o.obs_id: o for o in observations}
    state.event_clusters = {o.event_id for o in observations}
    return state


def test_the_scorecard_reports_null_rather_than_a_flattering_zero_on_an_empty_sample():
    card = scorecard(PaperState(epoch_id="epoch-test"))
    assert card["resolved_n"] == 0
    assert card["brier"] is None
    assert card["calibration_error"] is None
    assert card["hit_rate"] is None


def test_brier_and_hit_rate_over_a_scored_sample():
    card = scorecard(state_with([observed("0.9", 1), observed("0.2", 0), observed("0.6", 0)]))
    assert card["resolved_n"] == 3
    # (0.01 + 0.04 + 0.36) / 3
    assert card["brier"] == float(round(Decimal("0.41") / 3, 6))
    assert card["hit_rate"] == float(round(Decimal(2) / Decimal(3), 6))


def test_a_coin_flip_forecast_is_excluded_from_hit_rate():
    card = scorecard(state_with([observed("0.5", 1), observed("0.9", 1)]))
    assert card["hit_rate"] == 1.0  # only the decisive forecast counts


def test_a_perfectly_calibrated_sample_has_no_calibration_error():
    # Ten forecasts at 0.5, five of which happen: the bin's mean p equals its frequency.
    sample = [observed("0.5", 1, f"e{i}") for i in range(5)]
    sample += [observed("0.5", 0, f"f{i}") for i in range(5)]
    assert calibration_error(sample) == Decimal(0)


def test_a_systematically_overconfident_sample_shows_calibration_error():
    sample = [observed("0.9", 0, f"e{i}") for i in range(10)]
    assert calibration_error(sample) == Decimal("0.9")


#: Every key this stem is forbidden to emit (issue #350): a promotion threshold it cannot
#: verify, or a verdict rendered against one.
_VERDICT_KEYS = frozenset({"promotion", "promotion_eligible", "promotion_thresholds", "kill_flags"})


def test_the_scorecard_measures_and_renders_no_verdict():
    """The whole of issue #350: this instrument reports facts, and grades nothing.

    0.87.0 invented four promotion thresholds because §2.3 named the fields without defining
    them, and they disagreed with the governing contract in *both* directions. Better numbers
    would have been the same defect with a longer fuse — a package that cannot read the
    contract, cannot test against it and will not be told when it moves has no business
    holding the bar. So the bar left, and every input it takes stayed.
    """
    state = state_with([observed("0.9", 1, f"e{i}") for i in range(20)])
    state.max_drawdown_pct = Decimal("40")  # a breach under any plausible bar
    card = scorecard(state)

    assert not _VERDICT_KEYS & card.keys()
    # ...and the governance layer's four comparisons all have their inputs here.
    assert card["resolved_n"] == 20
    assert card["distinct_event_clusters"] == 20
    assert card["brier"] == float(Decimal("0.01"))
    assert card["max_drawdown_pct"] == float(Decimal("40"))


def test_one_lucky_theme_still_shows_as_one_cluster():
    """Diversity stays *measured* — it is the input the removed `low_diversity` flag read."""
    single_event = state_with([])
    # Twenty observations that all ride one event — distinct ids, one cluster.
    single_event.observations = {f"obs-{i}": observed("0.9", 1, "e1") for i in range(20)}
    single_event.event_clusters = {"e1"}
    card = scorecard(single_event)
    assert card["resolved_n"] == 20
    assert card["distinct_event_clusters"] == 1


def test_a_frozen_account_says_so_on_the_scorecard():
    """`frozen` is not a verdict — it is the fact that these numbers are not a live result.

    It was the one `kill_flags` entry this stem actually owned, so it survives the removal as
    a field of its own rather than vanishing with the flags it sat among.
    """
    state = state_with([observed("0.9", 1, f"e{i}") for i in range(20)])
    assert scorecard(state)["frozen"] is False
    state.frozen = True
    assert scorecard(state)["frozen"] is True


def test_the_epoch_open_row_records_no_promotion_threshold(tmp_path):
    """The rulebook row states the rules this stem *enforces* — and a promotion bar is not one.

    The recurrence guard for #350's actual harm: the wrong bars were hash-chained into row 1,
    where tamper-evidence lent an unverifiable copy the look of authority. Re-adding any of
    them fails here.
    """
    payload = open_epoch(tmp_path).rows()[0]["payload"]

    assert not _VERDICT_KEYS & payload.keys()
    # The rules it does enforce are all still frozen there.
    assert payload["caps"]["max_order_notional_usd"] == "500"
    assert payload["brier_attribution"] == "position_open"
    assert payload["fee_defaults_bps"]["taker"] == "100"


def test_an_epoch_opened_under_the_old_block_still_verifies_and_folds(tmp_path):
    """Dropping a payload key must not break an epoch already on a box.

    The fix ships onto boxes holding a live epoch whose row 1 *does* carry the 0.87.0
    `promotion` block, hash-chained. Nothing rewrites that row — the fold reads named keys, so
    the stale block is simply inert — and the chain still verifies because it hashes what is on
    disk. That is what makes this a removal rather than a migration, and a stricter payload
    reader added later would silently take it away.
    """
    directory = store_root(tmp_path) / "epoch-20260726T000000Z"
    directory.mkdir(parents=True)
    row = {
        "epoch_id": "epoch-20260726T000000Z",
        "ts": "2026-07-26T00:00:00.000000Z",
        "type": "epoch_open",
        "payload": {
            "bankroll_usd": "10000",
            "brier_attribution": "position_open",
            "promotion": {"min_resolved": 20, "min_event_clusters": 5, "max_brier": "0.20"},
        },
        "schema_version": SCHEMA_VERSION,
        "prev": GENESIS_PREV,
    }
    row["hash"] = row_hash(row)
    (directory / "ledger.jsonl").write_text(json.dumps(row, separators=(",", ":")) + "\n")

    epoch = Epoch(directory)
    assert epoch.verify().ok is True
    state = epoch.state()
    assert state.cash == Decimal("10000")  # it still folds
    assert scorecard(state)["brier_attribution"] == "position_open"
    assert not _VERDICT_KEYS & scorecard(state).keys()  # and the stale block stays unread


def test_a_position_opened_by_market_id_still_counts_its_event_cluster(tmp_path):
    """The diversity metric is only real if a position carries the event it belongs to.

    Caught against the live API: Gamma's single-market endpoint omits `events` entirely, so
    every position opened by id recorded an empty event and `distinct_event_clusters` sat at
    zero forever — silently turning the one gate that separates range from a single lucky
    theme into a constant. A fabricated fixture will happily carry whatever it is given, which
    is exactly why this pins the *count*, not the fixture.
    """
    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        second = fake.add_market(3)
        forecast(tool, market_id=second)
        buy(tool, shares="10", market_id=second)
        card = call(tool, "get_scorecard")

    assert card["distinct_event_clusters"] == 2


def test_the_scorecard_carries_the_frozen_attribution():
    card = scorecard(state_with([observed("0.9", 1)]))
    assert card["brier_attribution"] == "position_open"


# --- the equity curve ----------------------------------------------------------------------


def test_max_drawdown_tracks_the_marked_equity_curve(tmp_path):
    with upstream() as fake:
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="1000")  # 1000 @ 0.43 = $430

        # The market collapses; the call's own sweep marks the position down, and the
        # drawdown shows on that same call — the live state moves the curve exactly as a
        # replay of the same rows would.
        fake.books[YES_TOKEN] = book(bids=[("0.04", "500")], asks=[("0.06", "500")])
        low = call(tool, "get_pnl")
        assert low["max_drawdown_pct"] > 3.5

        # It recovers: the peak stands and the drawdown is not forgotten.
        fake.books[YES_TOKEN] = book(bids=[("0.44", "500")], asks=[("0.46", "500")])
        back = call(tool, "get_pnl")

    assert back["max_drawdown_pct"] == low["max_drawdown_pct"]
    assert back["peak_equity"] >= 10000


def test_a_mark_is_only_written_when_it_moves(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        call(tool, "get_pnl")
        first = len(rows(tmp_path, "mark"))
        call(tool, "get_pnl")
        call(tool, "get_pnl")
    assert len(rows(tmp_path, "mark")) == first  # an unchanged book adds nothing to the log


def test_positions_carry_the_normative_fields(tmp_path):
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        body = call(tool, "get_positions")
    position = body["positions"][0]
    assert set(position) == {
        "market_id",
        "outcome",
        "shares",
        "avg_price",
        "mtm_price",
        "mtm_value",
        "unrealized_pnl",
    }
    assert position["avg_price"] == 0.43
    assert body["equity_usd"] > 0


def test_the_no_side_is_a_separate_position(tmp_path):
    """There is no short: betting against Yes means holding No, and both are long."""
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool, outcome="Yes")
        forecast(tool, outcome="No")
        buy(tool, shares="10", outcome="Yes")
        buy(tool, shares="10", outcome="No")
        positions = call(tool, "get_positions")["positions"]
    assert {p["outcome"] for p in positions} == {"Yes", "No"}
    assert all(p["shares"] > 0 for p in positions)


def test_a_replay_of_the_ledger_reproduces_the_state_exactly(tmp_path):
    """The whole point of a fold: another process reading the log agrees, to the cent."""
    with upstream():
        tool = make_tool(tmp_path)
        forecast(tool)
        buy(tool, shares="10")
        sell(tool, shares="4")
        live = call(tool, "get_pnl")

    replayed = current_epoch(tmp_path).state()
    assert float(replayed.cash.quantize(Decimal("0.01"))) == live["cash_usd"]
    assert float(replayed.realized_pnl.quantize(Decimal("0.01"))) == live["realized_pnl"]


def test_the_book_is_re_sorted_regardless_of_the_order_it_arrived_in(tmp_path):
    with upstream() as fake:
        fake.books[NO_TOKEN] = book(
            bids=[("0.55", "100"), ("0.50", "100")],
            asks=[("0.57", "100"), ("0.60", "100")],
            token=NO_TOKEN,
        )
        body = call(make_tool(tmp_path), "get_market", market_id=MARKET_ID)
    no_book = next(b for b in body["book"] if b["outcome"] == "No")
    assert no_book["bids"][0][0] == 0.55
    assert no_book["asks"][0][0] == 0.57
    assert no_book["mid"] == 0.56
