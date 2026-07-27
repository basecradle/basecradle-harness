"""The deterministic fill model, settlement, and the sweep that never wakes anyone (issue #347).

This is NORMATIVE §2.4 and §A3 in code. Everything here is a pure function of *the ledger*
and *public market data*: given the same log and the same book snapshot, it writes the same
rows, on a laptop, in CI, and on a fleet box. There is no randomness, no synthetic slippage
(§2.4 chose reproducibility over theater), and no agent-supplied number anywhere in the
call graph — an agent cannot pass a price, a fee, a fill, or a resolution, because no
function here takes one.

**The three things that make a fill deterministic.** The *authority* is the harness at
order-accept time; the *clock* is harness UTC; the *book* is the `as_of` snapshot the walk
consumed. A market buy walks the ask side FIFO by price and a market sell walks the bids;
depth that is not there is not filled, and the remainder is **cancelled rather than left
resting** — a phantom resting order is a position the agent thinks it has and the ledger
does not. A limit order fills its marketable part immediately (taker) and rests the rest.

**A resting fill is a maker fill, at the maker's own price.** When the re-check finds the
book has crossed a resting order, it fills at the *resting order's* limit price, not at the
crossing level — that is what a maker gets in the real venue, and it is the predictable
answer §2.4 asks for. Placement fills remove liquidity and pay the taker rate; resting
fills provide it and pay the maker rate.

**Fees are read, never assumed — and every fill row says where its rate came from.** A market
that publishes *no* fee is recorded at zero, tagged `market`: the venue's own fact, not the
silent zeroing §2.4 forbids. A fee-*charging* market turns out to publish a flag rather than a
notional rate, so those fills take the contract's 100 bps taker / 0 maker default, tagged
`default`. `fee_rates` carries the evidence and the reasoning.

**§A3, stated as a property of this file rather than a promise about it:** the sweep is a
deterministic job that mutates the ledger and nothing else. This module imports the ledger
and the public data client and *nothing* from the model path — no `Provider`, no `Engine`,
no BaseCradle SDK, no wake. There is no code here that could send a message even by
mistake, which is what `test_polymarket.py` asserts against the module's own imports. The
agent learns what happened on its next `get_fills` / `get_positions` / `get_orders`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from basecradle_harness._polymarket_data import (
    Book,
    MarketNotFound,
    MarketState,
    MarketSummary,
    PolymarketData,
    PolymarketUnavailable,
)
from basecradle_harness._polymarket_ledger import (
    BRIER_OBS,
    BRIER_SCORE,
    CALIBRATION_BINS,
    DEFAULT_MAKER_FEE_BPS,
    DEFAULT_TAKER_FEE_BPS,
    FILL,
    FREEZE,
    MARK,
    MAX_MARKET_EXPOSURE,
    MAX_OPEN_POSITIONS,
    MAX_ORDER_NOTIONAL,
    ORDER,
    ORDER_CLOSE,
    SETTLEMENT,
    UNFREEZE,
    Epoch,
    Observation,
    Order,
    PaperState,
    apply_fill,
    apply_settlement,
    current_epoch,
    epochs,
    open_epoch,
    replay,
    store_lock,
    track_equity,
    verify_chain,
)
from basecradle_harness._version import __version__

_log = logging.getLogger("basecradle_harness")

#: Money is carried as `Decimal` and written at micro-dollar precision — enough for a
#: 0.001 tick against fractional share counts, short enough that a ledger row stays legible.
CENTS = Decimal("0.000001")


def money(value: Decimal) -> Decimal:
    """`value` rounded to the ledger's monetary precision."""
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)


class BrokenChain(Exception):
    """The ledger's hash chain does not verify, so nothing may be written or reported.

    Separate from `PaperReject` on purpose: a `PaperReject` is a verdict on the agent's
    *request* and the agent can act on it, while this is a verdict on the *record itself* and
    nobody on the agent's side can do anything about it.
    """


class PaperReject(Exception):
    """A structured refusal, carrying one of §2.6's codes.

    Every rejection path in this module raises one of these rather than returning a partial
    result: §2.3 requires a reject to be "structured, never a mystery partial", and an
    exception is the one shape that cannot be accidentally treated as a half-executed order.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Slice:
    """One price level consumed by a walk: `shares` at `price`."""

    price: Decimal
    shares: Decimal

    @property
    def notional(self) -> Decimal:
        return self.price * self.shares


@dataclass(frozen=True)
class FeeRate:
    """A fill's fee rate and where it came from — `market` (published) or `default`."""

    bps: Decimal
    source: str

    def on(self, notional: Decimal) -> Decimal:
        return money(notional * self.bps / Decimal(10000))


def fee_rates(state: MarketState) -> tuple[FeeRate, FeeRate]:
    """``(taker, maker)`` rates for a market, per §2.4's read-it-or-default rule.

    §2.4 says to read each market's published fee fields *when present*, record the rate on the
    fill row, and otherwise default to 100 bps taker / 0 maker tagged ``fee_source="default"``
    — never a silent zero. Applying that faithfully needs one fact about the live venue that
    the naive reading gets wrong, so it is written down here rather than learned from a
    distorted P&L (checked against live public data, 2026-07-26):

    **The CLOB's `taker_base_fee` / `maker_base_fee` are not notional rates.** They read
    ``1000`` on *every* fee-charging market — crypto, sports and economics alike, whose
    published category rates differ (0.07 / 0.05 / 0.05) — and ``0`` where fees are switched
    off. It is a protocol constant gating "fees apply here", not a per-market rate; the actual
    charge follows the venue's published curve ``C x rate x p x (1 - p)``, **taker only**.
    Treating ``1000`` as basis points of notional would charge **10%** on a trade the venue
    charges about 1% on, *and* would charge a maker fee on a venue that documents "makers are
    never charged fees" — the wrong number, on a side that never pays it.

    So the split is by what is actually published:

    - **Fees switched off** (a published zero) → 0 bps, tagged ``market``. That is the venue's
      own explicit zero, recorded as read — the opposite of the silent zeroing §2.4 forbids.
    - **Fees switched on** → the venue publishes *that* it charges but not a notional rate, so
      the rate is "unavailable" in §2.4's sense and the contract's own default applies, tagged
      ``default`` so every such fill says on its face that the number is the harness's, not
      Polymarket's.

    Modelling the venue's real price curve would change §2.4's stated fee *basis* (notional,
    both sides), which is a normative amendment and not this implementation's call — it is
    reported on issue #347 for the capital to decide.
    """
    if state.taker_fee_bps is not None and state.taker_fee_bps == 0:
        taker = FeeRate(Decimal(0), "market")
    else:
        taker = FeeRate(DEFAULT_TAKER_FEE_BPS, "default")
    if state.maker_fee_bps is not None and state.maker_fee_bps == 0:
        maker = FeeRate(Decimal(0), "market")
    else:
        maker = FeeRate(DEFAULT_MAKER_FEE_BPS, "default")
    return taker, maker


def walk(levels, size: Decimal) -> list[Slice]:
    """Consume `size` shares from `levels` (already best-first), FIFO by price level.

    Returns what the book could actually supply — possibly less than `size`, possibly
    nothing. The caller decides what an unfilled remainder means: cancelled for a market
    order, rested for a limit order.
    """
    remaining = size
    taken: list[Slice] = []
    for level in levels:
        if remaining <= 0:
            break
        shares = min(remaining, level.size)
        if shares <= 0:
            continue
        taken.append(Slice(price=level.price, shares=shares))
        remaining -= shares
    return taken


def marketable(levels, side: str, limit_price: Decimal):
    """The levels a limit order may cross: asks at or below a bid, bids at or above an ask."""
    if side == "buy":
        return [level for level in levels if level.price <= limit_price]
    return [level for level in levels if level.price >= limit_price]


# --- placing an order -------------------------------------------------------------


@dataclass(frozen=True)
class MarketRefs:
    """A market resolved once and reused: Gamma metadata, CLOB state, and the touched book."""

    summary: MarketSummary
    state: MarketState
    outcome: str
    token_id: str


def resolve_market(data: PolymarketData, market_id: str, outcome: str | None = None) -> MarketRefs:
    """Gamma + CLOB for one market (and optionally one outcome), or a `PaperReject`.

    A market with its order book disabled is `not_implemented` rather than `market_closed`:
    it is not shut, it is a structure this instrument does not simulate, and saying so
    plainly is more useful to the agent than a wrong-but-familiar code.
    """
    try:
        summary = data.market_summary(str(market_id))
        state = data.market_state(summary.condition_id)
    except MarketNotFound as exc:
        raise PaperReject("not_found", str(exc)) from exc
    except PolymarketUnavailable as exc:
        raise PaperReject("upstream_unavailable", str(exc)) from exc
    if not state.enable_order_book:
        raise PaperReject(
            "not_implemented",
            f"market {market_id} has no CLOB order book, so this instrument cannot simulate it",
        )
    token = ""
    name = ""
    if outcome is not None:
        token = summary.outcome_token(outcome) or ""
        if not token:
            raise PaperReject(
                "not_found",
                f"market {market_id} has no outcome {outcome!r}; it has {list(summary.outcomes)}",
            )
        name = next(o for o in summary.outcomes if o.casefold() == outcome.strip().casefold())
    return MarketRefs(summary=summary, state=state, outcome=name, token_id=token)


def place_order(
    epoch: Epoch,
    data: PolymarketData,
    state: PaperState,
    *,
    market_id: str,
    outcome: str,
    side: str,
    size_shares: Decimal,
    order_type: str,
    limit_price: Decimal | None,
    client_order_id: str,
) -> dict[str, Any]:
    """Validate, fill against the `as_of` book, and write the order to the ledger.

    Every refusal in §2.3's list is raised as a `PaperReject` **before** a single row is
    written, so a rejected order leaves the ledger exactly as it found it — there is no
    half-placed order to reconcile. Once validation passes, the rows go down in causal
    order: the `order`, then any `brier_obs` the position-open creates, then the `fill`s,
    then the `order_close` if it terminated.
    """
    existing = state.client_order_ids.get(client_order_id)
    if existing:  # §2.4 idempotency: a duplicate submit returns the original result
        return order_result(state, state.orders[existing], duplicate=True)

    if state.frozen:
        raise PaperReject(
            "frozen",
            f"this paper account is frozen{': ' + state.frozen_reason if state.frozen_reason else ''}",
        )

    refs = resolve_market(data, market_id, outcome)
    market, clob = refs.summary, refs.state

    if clob.closed or not clob.accepting_orders or not market.accepting_orders:
        raise PaperReject("market_closed", f"market {market_id} is not accepting orders")

    min_size = clob.min_order_size or market.min_order_size
    if size_shares < min_size:
        raise PaperReject(
            "size_too_small",
            f"{size_shares} shares is below this market's minimum of {min_size}",
        )

    tick = clob.tick_size or market.tick_size
    if order_type == "limit":
        if limit_price is None:
            raise PaperReject("invalid_params", "a limit order needs a limit_price")
        if not (Decimal(0) < limit_price < Decimal(1)):
            raise PaperReject("invalid_params", "limit_price must be between 0 and 1 exclusive")
        if tick > 0 and (limit_price / tick) % 1 != 0:
            raise PaperReject(
                "invalid_params",
                f"limit_price {limit_price} is not a multiple of this market's tick size {tick}",
            )

    key = (str(market.market_id), refs.outcome)
    position = state.positions.get(key)

    if side == "sell":
        held = position.shares if position else Decimal(0)
        free = held - _resting_sell_shares(state, key)
        if size_shares > free:
            raise PaperReject(
                "insufficient_shares",
                f"you hold {held} shares of {refs.outcome} ({free} not already working); "
                f"selling {size_shares} is not possible (this instrument has no short side)",
            )
    elif state.forecast_for(key[0], key[1]) is None:
        # §A2: a position requires a logged, unsuperseded forecast for this exact key.
        raise PaperReject(
            "forecast_required",
            f"log_forecast a probability for {refs.outcome} on market {market_id} before "
            "taking a position in it",
        )

    book = data.book(refs.token_id, refs.outcome)
    taker, maker = fee_rates(clob)
    levels = book.asks if side == "buy" else book.bids
    if order_type == "limit":
        assert limit_price is not None  # validated above
        levels = marketable(levels, side, limit_price)
    slices = walk(levels, size_shares)
    filled = sum((s.shares for s in slices), Decimal(0))
    remainder = size_shares - filled
    rests = order_type == "limit" and remainder > 0

    immediate_notional = sum((s.notional for s in slices), Decimal(0))
    resting_notional = (
        (remainder * limit_price) if (rests and limit_price is not None) else Decimal(0)
    )
    order_notional = immediate_notional + resting_notional

    if order_notional > MAX_ORDER_NOTIONAL:
        raise PaperReject(
            "cap_exceeded",
            f"that order is ${money(order_notional)} notional, over the "
            f"${MAX_ORDER_NOTIONAL} per-order cap",
        )

    if side == "buy":
        exposure = state.market_exposure(key[0]) + order_notional
        if exposure > MAX_MARKET_EXPOSURE:
            raise PaperReject(
                "cap_exceeded",
                f"that order would put ${money(exposure)} of net notional in market "
                f"{market_id}, over the ${MAX_MARKET_EXPOSURE} per-market cap",
            )
        prospective = _prospective_positions(state)
        if key not in prospective and len(prospective) >= MAX_OPEN_POSITIONS:
            raise PaperReject(
                "cap_exceeded",
                f"you already hold or have working orders on {MAX_OPEN_POSITIONS} positions, "
                "the maximum; close one first",
            )
        fees = sum((taker.on(s.notional) for s in slices), Decimal(0))
        needed = immediate_notional + fees + resting_notional
        if needed > state.available_cash:
            raise PaperReject(
                "insufficient_cash",
                f"that order needs ${money(needed)} and only ${money(state.available_cash)} "
                "is available (cash less what your resting bids already commit)",
            )

    # --- accepted: write it down ------------------------------------------------
    order_id = state.next_id("ord")
    row = epoch.append(
        ORDER,
        {
            "order_id": order_id,
            "client_order_id": client_order_id,
            "market_id": market.market_id,
            "condition_id": market.condition_id,
            "question": market.question,
            "outcome": refs.outcome,
            "token_id": refs.token_id,
            "side": side,
            "order_type": order_type,
            "size_shares": size_shares,
            "limit_price": limit_price,
            "tick_size": tick,
            "event_id": market.event_id or "",
            "book_as_of": {"best_bid": book.best_bid, "best_ask": book.best_ask},
        },
    )
    state.seq += 1
    order = Order(
        order_id=order_id,
        client_order_id=client_order_id,
        market_id=market.market_id,
        condition_id=market.condition_id,
        outcome=refs.outcome,
        token_id=refs.token_id,
        side=side,
        order_type=order_type,
        size_shares=size_shares,
        limit_price=limit_price,
        created_at=row["ts"],
    )
    state.orders[order_id] = order
    state.client_order_ids[client_order_id] = order_id

    opening = side == "buy" and (position is None or position.shares <= 0) and filled > 0
    if opening:
        _record_observation(epoch, state, market, refs.outcome)

    for piece in slices:
        _write_fill(
            epoch,
            state,
            order=order,
            price=piece.price,
            shares=piece.shares,
            rate=taker,
            liquidity="taker",
        )

    if not rests:
        status = "filled" if filled >= size_shares else "partially_filled"
        reason = "" if status == "filled" else "remainder cancelled: insufficient book depth"
        _close_order(epoch, state, order, status=status, reason=reason)

    data.invalidate(refs.token_id)  # the walk consumed this snapshot
    return order_result(state, order, book=book)


def _record_observation(
    epoch: Epoch, state: PaperState, market: MarketSummary, outcome: str
) -> None:
    """Lock the Brier observation for a position that is about to open (§A2, position-open)."""
    forecast = state.forecast_for(market.market_id, outcome)
    if forecast is None:  # unreachable for a buy: the §A2 gate ran first
        return
    obs_id = state.next_id("obs")
    epoch.append(
        BRIER_OBS,
        {
            "obs_id": obs_id,
            "market_id": market.market_id,
            "condition_id": market.condition_id,
            "outcome": outcome,
            "p": forecast.p,
            "forecast_id": forecast.forecast_id,
            "event_id": market.event_id or "",
            "attribution": state.brier_attribution,
        },
    )
    state.seq += 1
    state.observations[obs_id] = Observation(
        obs_id=obs_id,
        market_id=market.market_id,
        outcome=outcome,
        p=forecast.p,
        forecast_id=forecast.forecast_id,
        opened_at="",
        event_id=market.event_id or "",
    )
    if market.event_id:
        state.event_clusters.add(market.event_id)


def _write_fill(
    epoch: Epoch,
    state: PaperState,
    *,
    order: Order,
    price: Decimal,
    shares: Decimal,
    rate: FeeRate,
    liquidity: str,
) -> dict[str, Any]:
    """Append one fill row and fold it into the in-memory state in the same breath."""
    notional = money(price * shares)
    payload = {
        "fill_id": state.next_id("fil"),
        "order_id": order.order_id,
        "market_id": order.market_id,
        "condition_id": order.condition_id,
        "outcome": order.outcome,
        "token_id": order.token_id,
        "side": order.side,
        "price": price,
        "shares": shares,
        "notional": notional,
        "fee": rate.on(notional),
        "fee_bps": rate.bps,
        "fee_source": rate.source,
        "liquidity": liquidity,
    }
    row = epoch.append(FILL, payload)
    state.seq += 1
    apply_fill(state, payload, row["ts"])
    # The same call `replay` makes after an equity-moving row. Without it the live state's
    # drawdown lags a wake behind the log's, so `get_pnl` and a fresh fold of the same ledger
    # would disagree — and the fold is supposed to be the definition, not a second opinion.
    track_equity(state)
    return payload


def _close_order(
    epoch: Epoch, state: PaperState, order: Order, *, status: str, reason: str
) -> None:
    """Terminate an order in the ledger and in the folded state."""
    epoch.append(
        ORDER_CLOSE,
        {
            "order_id": order.order_id,
            "status": status,
            "reason": reason,
            "filled_shares": order.filled_shares,
        },
    )
    state.seq += 1
    order.status = status
    order.close_reason = reason


def cancel_order(epoch: Epoch, state: PaperState, order_id: str) -> Order:
    """Cancel one resting limit order. Anything else is a structured refusal."""
    order = state.orders.get(order_id)
    if order is None:
        raise PaperReject("not_found", f"no order {order_id!r} in this epoch")
    if not order.is_open:
        raise PaperReject(
            "invalid_params",
            f"order {order_id} is already {order.status}; only a resting order can be cancelled",
        )
    _close_order(epoch, state, order, status="cancelled", reason="cancelled by the agent")
    return order


def order_result(
    state: PaperState, order: Order, *, duplicate: bool = False, book: Book | None = None
) -> dict[str, Any]:
    """The `OrderResult` §2.3 promises — the same shape for a fresh and a duplicate submit."""
    fills = [f for f in state.fills if f.get("order_id") == order.order_id]
    filled = sum((_dec(f.get("shares")) for f in fills), Decimal(0))
    notional = sum((_dec(f.get("notional")) for f in fills), Decimal(0))
    return {
        "order_id": order.order_id,
        "client_order_id": order.client_order_id,
        "market_id": order.market_id,
        "outcome": order.outcome,
        "side": order.side,
        "order_type": order.order_type,
        "size_shares": _num(order.size_shares),
        "limit_price": _num(order.limit_price),
        "status": order.status,
        "filled_shares": _num(filled),
        "remaining_shares": _num(order.size_shares - filled),
        "avg_fill_price": _num(notional / filled) if filled > 0 else None,
        "notional_usd": _num(notional),
        "fees_usd": _num(sum((_dec(f.get("fee")) for f in fills), Decimal(0))),
        "close_reason": order.close_reason or None,
        "duplicate_of_client_order_id": order.client_order_id if duplicate else None,
        "fills": [fill_view(f) for f in fills],
        "book_at_accept": (
            {"best_bid": _num(book.best_bid), "best_ask": _num(book.best_ask)} if book else None
        ),
    }


def fill_view(fill: dict[str, Any]) -> dict[str, Any]:
    """One fill, as the model reads it."""
    return {
        "fill_id": fill.get("fill_id"),
        "order_id": fill.get("order_id"),
        "market_id": fill.get("market_id"),
        "outcome": fill.get("outcome"),
        "side": fill.get("side"),
        "price": _num(_dec(fill.get("price"))),
        "shares": _num(_dec(fill.get("shares"))),
        "notional_usd": _num(_dec(fill.get("notional"))),
        "fee_usd": _num(_dec(fill.get("fee"))),
        "fee_bps": _num(_dec(fill.get("fee_bps"))),
        "fee_source": fill.get("fee_source"),
        "liquidity": fill.get("liquidity"),
        "ts": fill.get("ts"),
    }


# --- the sweep ----------------------------------------------------------------------


def sweep_market(epoch: Epoch, data: PolymarketData, state: PaperState, market_id: str) -> int:
    """Re-check, settle and re-mark one market. Returns the number of rows written.

    The order is deliberate: settle first, because a resolved market's resting orders can
    never fill and its marks are meaningless. Otherwise re-check the resting orders against
    a fresh book and then refresh the marks, so a fill and the mark that follows it are
    consistent with the same snapshot.
    """
    try:
        refs = resolve_market(data, market_id)
    except PaperReject as reject:
        if reject.code == "upstream_unavailable":
            _log.info("polymarket sweep: %s unreachable (%s); leaving it alone", market_id, reject)
            return 0
        _log.info("polymarket sweep: skipping %s (%s)", market_id, reject.code)
        return 0
    if refs.state.resolved:
        return settle_market(epoch, data, state, refs)
    written = recheck_resting(epoch, data, state, refs)
    return written + refresh_marks(epoch, data, state, refs)


def recheck_resting(epoch: Epoch, data: PolymarketData, state: PaperState, refs: MarketRefs) -> int:
    """Fill any resting order the book has crossed, at the resting order's own price.

    A maker fill, so it pays the maker rate. The available size is the depth standing at
    prices that cross the order — an order can never take more than the book is offering,
    and what it does not take stays resting for the next re-check.
    """
    written = 0
    for order in state.open_orders():
        if order.market_id != refs.summary.market_id or order.limit_price is None:
            continue
        book = data.book(order.token_id, order.outcome)
        levels = marketable(
            book.asks if order.side == "buy" else book.bids, order.side, order.limit_price
        )
        available = sum((level.size for level in levels), Decimal(0))
        shares = min(order.remaining, available)
        if shares <= 0:
            continue
        _, maker = fee_rates(refs.state)
        if order.side == "buy":
            cost = shares * order.limit_price + maker.on(shares * order.limit_price)
            if cost > state.cash:  # the reserve is gone (a settlement moved cash) — leave it
                continue
        _write_fill(
            epoch,
            state,
            order=order,
            price=order.limit_price,
            shares=shares,
            rate=maker,
            liquidity="maker",
        )
        written += 1
        if order.remaining <= 0:
            _close_order(epoch, state, order, status="filled", reason="filled while resting")
            written += 1
        data.invalidate(order.token_id)
    return written


def refresh_marks(epoch: Epoch, data: PolymarketData, state: PaperState, refs: MarketRefs) -> int:
    """Record a fresh MTM mark for each held outcome in this market, when it moved.

    Only a *changed* mark is written. An unchanged one carries no information and would put
    a row on the log every hour, for every position, forever — the ledger is append-only, so
    a row written for nothing is a row that can never be taken back.
    """
    written = 0
    for position in state.open_positions():
        if position.market_id != refs.summary.market_id:
            continue
        book = data.book(position.token_id, position.outcome)
        mark = book.mid
        if mark is None:
            continue
        if state.marks.get((position.market_id, position.outcome)) == mark:
            continue
        epoch.append(
            MARK,
            {
                "market_id": position.market_id,
                "outcome": position.outcome,
                "token_id": position.token_id,
                "price": mark,
                "source": (
                    "book_mid"
                    if book.best_bid is not None and book.best_ask is not None
                    else "last_trade"
                ),
            },
        )
        state.seq += 1
        state.marks[(position.market_id, position.outcome)] = mark
        track_equity(state)  # a new mark moves the curve; the fold does this too
        written += 1
    return written


def settle_market(epoch: Epoch, data: PolymarketData, state: PaperState, refs: MarketRefs) -> int:
    """Settle a resolved market: cancel what rests, pay out at $1.00/$0.00, score the Briers.

    The resolution comes from public market state (`MarketState.winners`) and from nowhere
    else. Observations are scored even when the position was closed before resolution: the
    forecast was made and acted on, and a calibration record that only scores the trades an
    agent happened to hold to the end is a calibration record with a survivorship hole in it.
    """
    written = 0
    market_id = refs.summary.market_id
    winners = {name.casefold() for name in refs.state.winning_outcomes()}

    for order in state.open_orders():
        if order.market_id == market_id:
            _close_order(epoch, state, order, status="cancelled", reason="market resolved")
            written += 1

    for position in list(state.open_positions()):
        if position.market_id != market_id:
            continue
        won = position.outcome.casefold() in winners
        payout = money(position.shares if won else Decimal(0))
        payload = {
            "market_id": market_id,
            "condition_id": position.condition_id,
            "outcome": position.outcome,
            "shares": position.shares,
            "won": won,
            "payout": payout,
            "cost_basis": position.cost,
            "realized_pnl": money(payout - position.cost),
            "resolution_source": "clob_public_market_state",
        }
        epoch.append(SETTLEMENT, payload)
        state.seq += 1
        apply_settlement(state, payload)
        track_equity(state)
        written += 1

    for obs in state.observations.values():
        if obs.market_id != market_id or obs.result is not None:
            continue
        result = 1 if obs.outcome.casefold() in winners else 0
        brier = money((obs.p - Decimal(result)) ** 2)
        epoch.append(
            BRIER_SCORE,
            {
                "obs_id": obs.obs_id,
                "market_id": market_id,
                "outcome": obs.outcome,
                "p": obs.p,
                "result": result,
                "brier": brier,
                "attribution": state.brier_attribution,
            },
        )
        state.seq += 1
        obs.result = result
        obs.brier = brier
        written += 1
    return written


def markets_to_sweep(state: PaperState) -> list[str]:
    """Every market the sweep still has business with, in a stable order.

    Open positions and resting orders are obvious. **Unscored observations are the one that
    is easy to miss**: an agent that opened a position, closed it at a profit and never
    touched the market again still has a forecast waiting to be graded, and dropping it
    would quietly bias the scorecard toward the trades that were held to resolution.
    """
    wanted = {p.market_id for p in state.open_positions()}
    wanted |= {o.market_id for o in state.open_orders()}
    wanted |= {obs.market_id for obs in state.observations.values() if obs.result is None}
    return sorted(wanted)


def sweep(epoch: Epoch, data: PolymarketData, *, state: PaperState | None = None) -> int:
    """Sweep every market this epoch still has business with. Returns rows written.

    **Takes no lock — the caller holds `store_lock`.** The tool calls this while already
    holding it, and a second exclusive `flock` from the same process on a second descriptor
    deadlocks rather than no-ops. The two entry points (the tool's `run`, this module's
    `main`) each take it once, around fold and mutation together.

    **Refuses to append to a chain that does not verify.** A sweep is the one writer that runs
    with nobody watching, so it is the worst place to extend a broken ledger: new rows would
    chain cleanly onto the tampered ones and bury the break under legitimate history. It stops
    and says so instead, loudly, and leaves the evidence exactly as it found it. (The tool
    refuses first when it calls this, so the check costs a pass only on the cron path.)
    """
    chain = epoch.verify()
    if not chain.ok:
        _log.error(
            "polymarket sweep: LEDGER CHAIN BROKEN epoch=%s rows_verified=%d head=%s reason=%s "
            "- refusing to append",
            epoch.epoch_id,
            chain.rows,
            chain.head,
            chain.reason,
        )
        raise BrokenChain(chain.reason)
    live = state or epoch.state()
    written = 0
    for market_id in markets_to_sweep(live):
        written += sweep_market(epoch, data, live, market_id)
    return written


# --- the scorecard --------------------------------------------------------------------


def scorecard(state: PaperState) -> dict[str, Any]:
    """§2.3's `get_scorecard` body: calibration, hit rate, paper P&L — measurements, no verdict.

    Brier and calibration error are computed over **scored** observations only; an epoch
    with none reports them as `null` rather than as zero, because a perfect-looking 0.0 from
    an empty sample is the single most misleading number this instrument could emit.

    **It renders no verdict** (issue #350). §2.3 names `promotion_eligible` and `kill_flags`
    without defining their thresholds, and 0.87.0 answered by inventing four — which disagreed
    with the governing contract in both directions. The lasting fix is not better numbers but
    no numbers: a promotion bar belongs to a contract this package cannot read, cannot test
    against, and will not be told about when it moves, so an eligibility claim computed here
    is an assertion nobody can check. It is also read by the very agent under measurement,
    which is the last party who should be handed a verdict field to optimize toward. Every
    input those bars take is right here — `resolved_n`, `brier`, `calibration_error`,
    `paper_pnl`, `max_drawdown_pct`, `distinct_event_clusters` — and the governance layer,
    which holds the only copy that binds anything, does the comparing.

    `frozen` stays, because it is not a verdict: it is this stem's own state, and it is the
    one fact that says the numbers beside it are not a live result.
    """
    scored = [
        o for o in state.observations.values() if o.result is not None and o.brier is not None
    ]
    resolved_n = len(scored)
    brier = (
        money(sum((o.brier for o in scored), Decimal(0)) / Decimal(resolved_n))
        if resolved_n
        else None
    )
    decisive = [o for o in scored if o.p != Decimal("0.5")]
    hit_rate = (
        money(
            Decimal(sum(1 for o in decisive if (o.p > Decimal("0.5")) == (o.result == 1)))
            / Decimal(len(decisive))
        )
        if decisive
        else None
    )
    clusters = len(state.event_clusters)
    paper_pnl = money(state.realized_pnl + state.unrealized_pnl)
    return {
        "epoch_id": state.epoch_id,
        "started_at": state.started_at,
        "resolved_n": resolved_n,
        "brier": _num(brier),
        "calibration_error": _num(calibration_error(scored)),
        "hit_rate": _num(hit_rate),
        "paper_pnl": _num(paper_pnl),
        "max_drawdown_pct": _num(money(state.max_drawdown_pct)),
        "distinct_event_clusters": clusters,
        "frozen": state.frozen,
        "brier_attribution": state.brier_attribution,
    }


def calibration_error(scored) -> Decimal | None:
    """Expected calibration error: the size-weighted gap between forecast and frequency.

    Ten equal-width bins over (0,1) — the standard ECE decomposition. ``None`` for an empty
    sample, for the same reason `brier` is.
    """
    if not scored:
        return None
    bins: dict[int, list] = {}
    for obs in scored:
        index = min(int(obs.p * CALIBRATION_BINS), CALIBRATION_BINS - 1)
        bins.setdefault(index, []).append(obs)
    total = Decimal(len(scored))
    error = Decimal(0)
    for group in bins.values():
        n = Decimal(len(group))
        mean_p = sum((o.p for o in group), Decimal(0)) / n
        frequency = Decimal(sum(1 for o in group if o.result == 1)) / n
        error += (n / total) * abs(mean_p - frequency)
    return money(error)


# --- small helpers ----------------------------------------------------------------------


def _prospective_positions(state: PaperState) -> set[tuple[str, str]]:
    """Positions held **or** working — what the 20-position cap counts.

    A resting buy is a position the agent has queued; excluding it would let twenty resting
    bids become a twenty-first, twenty-second and twenty-third position without the cap ever
    being consulted.
    """
    keys = {(p.market_id, p.outcome) for p in state.open_positions()}
    keys |= {(o.market_id, o.outcome) for o in state.open_orders() if o.side == "buy"}
    return keys


def _resting_sell_shares(state: PaperState, key: tuple[str, str]) -> Decimal:
    """Shares already committed to resting sell orders on one ``(market, outcome)``."""
    return sum(
        (
            o.remaining
            for o in state.orders.values()
            if o.is_open and o.side == "sell" and (o.market_id, o.outcome) == key
        ),
        Decimal(0),
    )


def _dec(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001 - a missing number reads as zero in a projection
        return Decimal(0)


def _num(value: Decimal | None) -> float | None:
    """A `Decimal` as the JSON number the model reads. ``None`` stays ``None``."""
    return None if value is None else float(value)


# --- the read-only probe ---------------------------------------------------------------


def _epoch_report(epoch: Epoch) -> dict[str, Any]:
    """One epoch's state for the read-only probe: the chain verdict, and the freeze.

    Both come off **one** read of the ledger file — `verify` and `state` each re-read it
    otherwise, and a probe that reads twice can straddle a concurrent append and describe two
    different files as though they were one.

    **`frozen` is `None` on a broken chain, and that is the whole care of this function.** The
    freeze is a *safety control*, and the answer is folded out of exactly the rows whose
    integrity just failed — so an edit that removed the `freeze` row from the middle of the log
    would be reported as ``frozen: false``, which is the one wrong answer that matters here.
    There is no honest third state but "unknown", and `chain_ok` sits beside it saying why.
    This is the same refusal `get_scorecard` makes with its numbers, applied to the one field
    that is not a number: a control that degrades quietly reads as a working one.

    **What that refusal does *not* cover, stated here rather than assumed:** a chain detects an
    edit or a removal *in the middle* (the next row's ``prev`` stops matching), and **cannot**
    detect a **truncated tail** — lopping the final rows off leaves a shorter chain that
    verifies perfectly. So a trailing `freeze` deleted from the end reads ``chain_ok: true``
    with ``frozen: false``, honestly reporting a log that is itself a lie. That is not a gap in
    this function; it is the property `ChainStatus` already names, and it is precisely why
    `rows` and `head` are published: they are the pin an external verifier compares against the
    off-box row copy, under a UID the agent does not have. **An audit that alarms only on
    `chain_ok` is incomplete by construction** — it must also pin ``(rows, head)``.

    `rows` and `head` are the **verified prefix** — on an intact chain the whole log and its
    real head; on a broken one, how far the record vouches for itself and the last hash that
    did. `broken_at` and `reason` name where it stopped, so a human has somewhere to look.
    """
    rows = epoch.rows()
    chain = verify_chain(epoch.epoch_id, rows)
    state = replay(rows) if chain.ok else None
    return {
        "epoch_id": epoch.epoch_id,
        "path": str(epoch.path),
        "frozen": None if state is None else state.frozen,
        "frozen_reason": None if state is None else state.frozen_reason,
        "rows": chain.rows,
        "chain_ok": chain.ok,
        "head": chain.head,
        "broken_at": chain.broken_at,
        "reason": chain.reason,
    }


def _verify(home: Path, *, all_epochs: bool, as_json: bool) -> int:
    """`--verify`: report the epoch state and write **nothing**. Exit 1 if a chain is broken.

    The off-box audit surface for a control that was previously write-only (issue #353). A
    freeze is a live operational lever — the capital froze an armed red-team persona's epoch at
    the engine while it verified that rows were reaching the off-box log store, then lifted it —
    and until now the state was reachable only through `get_scorecard` (a tool call inside a
    wake, which no monitor can make) or by reading the ledger on the box (which needs a shell
    nobody has by design). So an undeclared unfreeze could happen and every fleet signal would
    stay green, including the drift heartbeat built to catch exactly that class.

    It writes nothing at all — not even the store dir, which taking the lock would otherwise
    create on an agent that has never traded (`store_lock(create=False)`) — makes no model call
    (this module imports no provider and no platform client), and is safe to run on a schedule.

    **"No epoch" and "an epoch that is not frozen" are different facts**, so they are reported
    differently: `epoch` is ``null`` for the first and an object for the second. Collapsing them
    would make the audit read a freshly-provisioned agent as an un-frozen one, which is the
    reading the whole probe exists to prevent.

    **The probe is one half of the audit, and says so.** ``chain_ok`` catches an edit within the
    log; it cannot catch a **truncated tail**, which is why every epoch also reports ``rows`` and
    ``head`` — the pin against the off-box row copy. See `_epoch_report`.
    """
    with store_lock(home, create=False):
        targets = (
            epochs(home) if all_epochs else [e for e in [current_epoch(home, create=False)] if e]
        )
        reports = [_epoch_report(epoch) for epoch in targets]

    broken = [report for report in reports if not report["chain_ok"]]
    if as_json:
        print(
            json.dumps(
                {
                    "harness_version": __version__,
                    "home": str(home),
                    # The **current** (newest) epoch — the one a monitor's axis is about — or
                    # `null` when this agent has none. With `--all-epochs` it is `epochs[-1]`.
                    "epoch": reports[-1] if reports else None,
                    "epochs": reports,
                    "chain_ok": not broken,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1 if broken else 0

    if not targets:
        print(f"no polymarket_paper epoch under {home} — nothing to sweep")
        return 0
    for report in reports:
        verdict = "OK" if report["chain_ok"] else "BROKEN"
        # `frozen=` is **appended**, never spliced in: the line's existing prefix and its
        # `rows=`/`head=` pairs are what an off-box reader may already be parsing, and adding a
        # field to the end of a key=value line cannot break one that adding a word in the middle
        # would. (The JSON is the contract to build against; this stays legible for a human.)
        frozen = "unknown" if report["frozen"] is None else str(report["frozen"]).lower()
        print(
            f"{report['epoch_id']}: {verdict} rows={report['rows']} "
            f"head={report['head']} frozen={frozen}"
        )
        if not report["chain_ok"]:
            print(f"  {report['reason']}")
    return 1 if broken else 0


# --- the operator's job ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """`basecradle-harness-polymarket-sweep` — the §A3 job. Mutates the ledger, wakes no one.

    Run it hourly from cron or the NOC. It settles resolved markets, fills resting orders the
    book has crossed, and refreshes marks. It **makes no model call and sends no message**:
    there is no provider and no platform client anywhere in this module's imports, so there
    is nothing here that could. The agent finds out on its next pull.
    """
    parser = argparse.ArgumentParser(
        prog="basecradle-harness-polymarket-sweep",
        description=(
            "Deterministic sweep for the polymarket_paper instrument: settle resolved "
            "markets, fill crossed resting orders, refresh marks. Never calls a model and "
            "never wakes the agent."
        ),
    )
    parser.add_argument(
        "--home",
        default=None,
        help="the agent's HARNESS_HOME (default: $HARNESS_HOME, else the current directory)",
    )
    parser.add_argument("--freeze", metavar="REASON", help="halt new orders on this epoch")
    parser.add_argument("--unfreeze", action="store_true", help="lift a freeze on this epoch")
    parser.add_argument(
        "--new-epoch", action="store_true", help="open a fresh epoch and sweep nothing"
    )
    parser.add_argument(
        "--all-epochs", action="store_true", help="sweep every epoch, not the newest"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check the ledger hash chain and print its head; write nothing (exit 1 if broken)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="with --verify: emit the epoch state as JSON (epoch_id, frozen, rows, chain_ok, head)",
    )
    args = parser.parse_args(argv)
    if args.as_json and not args.verify:
        # `--json` is the *monitor's* spelling of `--verify`, and it is bound to it deliberately:
        # every other mode of this command writes to the ledger, and a caller reaching for a
        # machine-readable answer is asking a read-only question. Refusing here is what keeps
        # "safe to run repeatedly" a property of the flag rather than of the caller's care.
        parser.error("--json is only available with --verify (every other mode writes).")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # httpx logs every request at INFO; this runs hourly per agent, and a cron job that writes
    # three lines of URL noise per market per hour into journald is a job nobody reads.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    home = Path(args.home or os.environ.get("HARNESS_HOME") or ".").expanduser()

    # The read-only probe runs *outside* the mutating lock, and takes its own non-creating one:
    # every path below this writes, and the audit must not be the thing that changes the box.
    if args.verify:
        return _verify(home, all_epochs=args.all_epochs, as_json=args.as_json)

    # The whole run is under the store lock: the agent's wake is the other writer against this
    # same ledger, and both of them fill resting orders. Taking it here (and in the tool's
    # `run`) rather than inside `sweep` is deliberate — see `store_lock`.
    with store_lock(home):
        if args.new_epoch:
            epoch = open_epoch(home)
            print(f"opened {epoch.epoch_id} at {epoch.path}")
            return 0

        targets = (
            epochs(home)
            if args.all_epochs
            else [e for e in [current_epoch(home, create=False)] if e]
        )
        if not targets:
            print(f"no polymarket_paper epoch under {home} — nothing to sweep")
            return 0

        data = PolymarketData()
        for epoch in targets:
            if args.freeze:
                epoch.append(FREEZE, {"reason": args.freeze, "by": "operator"})
                print(f"{epoch.epoch_id}: frozen ({args.freeze})")
                continue
            if args.unfreeze:
                epoch.append(UNFREEZE, {"by": "operator"})
                print(f"{epoch.epoch_id}: unfrozen")
                continue
            try:
                written = sweep(epoch, data)
            except BrokenChain as broken:
                # Loud and non-zero: this is the governance layer's tampering detector, and a
                # cron job that exits 0 on it has told nobody anything.
                print(f"{epoch.epoch_id}: LEDGER CHAIN BROKEN — {broken}")
                return 1
            print(f"{epoch.epoch_id}: {written} row(s) written")
    return 0


if __name__ == "__main__":  # pragma: no cover - console-script shim
    sys.exit(main())
