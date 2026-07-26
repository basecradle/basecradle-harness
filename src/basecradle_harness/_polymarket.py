"""`polymarket_paper` — a fenced paper-trading instrument for forecast calibration (issue #347).

A **simulated** prediction-market instrument: real public prices, an entirely fictional
$10,000 bankroll, and no venue account anywhere in the design. It exists to measure one
thing — whether a research persona's probability estimates are any good — by making it put
a number on a question *before* it takes a position, and then scoring those numbers against
what actually happened. Calibration and a simulated P&L; no real funds, ever.

**What this file is.** The model-facing surface: ten enumerable operations, the §2.2
burn budget, and the §2.6 error envelope. The work lives next door —
`_polymarket_data.py` (public Gamma/CLOB reads, GET-only), `_polymarket_ledger.py` (the
append-only operator ledger and the fold that reads it) and `_polymarket_engine.py` (the
deterministic fill model, settlement and the sweep). The split is not decoration: it is
what lets the sweep be provably free of any model or platform path, because the module that
runs it does not import this one.

**Four fences, and each is structural rather than advisory:**

- **No URL, ever (§2.1).** There is no parameter here through which a string becomes a
  request destination — not a url, not a host, not a path, not an endpoint. The data client
  owns two host constants and builds every path itself, so "the agent cannot fetch an
  arbitrary page" is a property of the call graph. What comes back is parsed fields; no raw
  document reaches the model.
- **No money and no venue (§2.5).** No operation transfers, withdraws, deposits, bridges,
  or touches key material, because none exists in the enum and nothing here can sign.
- **The agent cannot write its own scoreboard — and if the record is edited anyway, every
  operation stops.** Cash, P&L, fees, fills and resolutions are all *derived* from the ledger
  and from public market state; no operation accepts a price, a fee, a P&L figure, or a
  resolution. But the harness runs as the agent's **own UID**, so "cannot write the file" was
  never a filesystem fact — it held only because this agent has no shell. Integrity therefore
  rests on **detection**: the ledger's rows are hash-chained, every call verifies the chain
  before it computes anything, and a break returns `ledger_tampered` with **no numbers at
  all** rather than numbers with a warning. A scoreboard that degrades quietly is the one
  failure this instrument cannot survive, because the governance layer's tampering trigger
  has no other detector.
- **A position requires a forecast (§A2).** `place_order` refuses with `forecast_required`
  unless the agent has already logged a probability for that exact ``(market, outcome)``.
  Optional forecast logging would produce a promotion folder with two undefined headline
  metrics; this makes the calibration record complete by construction rather than by the
  agent's goodwill.

**The burn ceiling is enforced here, not honored.** Every accepted invocation appends one
`call` row to the ledger, so the day's count is a fact about the log rather than a counter
that can disagree with it. Past the hard ceiling the tool returns `rate_limited` until
00:00 UTC — a structured error, never a hang, and never a hidden always-on loop.

**Powerful, therefore opt-in** (issue #168): off by default on every provider, reaching a
persona only when its plugin is dropped into that persona's `tools/` overlay. It is
powerful for an unusual reason — it spends nothing and touches no box — but it *creates a
standing record that a human will read as evidence of skill*, and an instrument like that
arriving switched on for everyone is a scoreboard nobody agreed to keep.
"""

from __future__ import annotations

import json
import logging
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from basecradle_harness._install import config_home
from basecradle_harness._platform import PlatformTool
from basecradle_harness._polymarket_data import BOOK_DEPTH, PolymarketData
from basecradle_harness._polymarket_engine import (
    PaperReject,
    cancel_order,
    fee_rates,
    fill_view,
    money,
    order_result,
    place_order,
    resolve_market,
    scorecard,
    sweep,
    sweep_market,
)
from basecradle_harness._polymarket_ledger import (
    CALL,
    DEFAULT_LIST_LIMIT,
    FORECAST,
    HARD_CALLS_PER_DAY,
    HARD_ORDERS_PER_DAY,
    MAX_LIST_LIMIT,
    SOFT_CALLS_PER_DAY,
    SOFT_ORDERS_PER_DAY,
    Epoch,
    Forecast,
    PaperState,
    current_epoch,
    stamp,
    store_lock,
    utc_now,
)

_log = logging.getLogger("basecradle_harness")

#: The operations, exactly as NORMATIVE §2.3 enumerates them. Closed set, no extras — the
#: schema's `enum` is the whole surface, and `test_polymarket.py` pins it against this list.
ACTIONS = (
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
)

#: How many fills one `get_fills` page returns by default, and the most it will return.
#: §2.3 gives `get_fills` a limit and a cursor and is silent on a ceiling; an epoch that ran
#: for months holds thousands of fills, and answering `limit=100000` in one blob would put the
#: whole trading history into the model's context for one call. Paged, with the page bounded.
DEFAULT_FILLS_LIMIT = 50
MAX_FILLS_LIMIT = 200

#: The most orders one `get_orders` returns. The contract gives it no paging at all, which is
#: fine for its `open` default (bounded by the 20-position cap) and unbounded for `all` on a
#: long-lived epoch — so the newest are returned, with `total` and a note saying so. Truncating
#: silently would read as "these are all your orders", which is the one answer it must not give.
MAX_ORDERS_RETURNED = 100

#: Ops that re-check resting orders and settlement before answering — §2.4's "on any agent
#: tool call touching that market". The pure log reads (`get_fills`, `list_markets`,
#: `log_forecast`) touch no market and deliberately do not sweep.
_SWEEPS_ONE_MARKET = frozenset({"get_market", "place_order", "cancel_order"})
_SWEEPS_EVERYTHING = frozenset({"get_positions", "get_pnl", "get_scorecard", "get_orders"})


class PolymarketPaperTool(PlatformTool):
    """`polymarket_paper` — paper trading on live public Polymarket data.

    A `PlatformTool` that calls no BaseCradle endpoint. It is one because the platform
    context is the seam that carries the agent's `home`, and the ledger belongs under that
    home with the wake's other operator-owned stores — the same reason `DirectMessageTool`
    is a `PlatformTool` despite speaking only to ntfy.

    Args:
        root: The agent home the ledger lives under. ``None`` (the default) takes the bound
            context's `home`, then ``$HARNESS_HOME``, then the config home — always a
            location the operator controls, never a path the model chose.
        data: The public data client. Injectable so a test can point at a fabricated host;
            there is no agent-reachable way to swap it.
        now: The harness clock (UTC), injectable for deterministic tests.
    """

    name = "polymarket_paper"
    description = (
        "Paper-trade prediction markets on live public Polymarket data, to measure how well "
        "calibrated your probability estimates are. THE MONEY IS FAKE: a simulated $10,000 "
        "bankroll, no real funds, no exchange account, nothing you do here moves a dollar. "
        "Actions: list_markets (search/browse), get_market (one market with its order book), "
        "log_forecast (record your probability for an outcome BEFORE taking a position), "
        "place_order (buy/sell shares, market or limit), cancel_order, get_orders, get_fills, "
        "get_positions, get_pnl, get_scorecard (your Brier score, calibration error, hit rate). "
        "You MUST log_forecast for a (market_id, outcome) before place_order will accept a buy "
        "on it — that forecast is what gets scored when the market resolves, so make it your "
        "real estimate, not a formality. One forecast covers as many sized adds as you like "
        "until you log a new one; selling and cancelling need none. Prices are in dollars per "
        "share, 0 to 1, and a winning share settles at $1.00 and a loser at $0.00. Caps: $500 "
        "per order, $2,000 net per market, 20 open positions. Budget: 200 calls and 40 orders "
        "per UTC day, and every reply tells you what is left. Resting limit orders fill on a "
        "sweep, so check get_orders and get_fills to find out what happened while you were away. "
        "The ledger behind all of this is hash-chained and verified on every call: if it ever "
        "fails to verify, every action here returns 'ledger_tampered' and no numbers, and that "
        "is not something you can fix — report it and stop."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "description": "Which operation to run.",
            },
            "query": {
                "type": "string",
                "description": "Free-text search over markets (list_markets only).",
            },
            "tag": {
                "type": "string",
                "description": (
                    "Restrict a listing to one Polymarket tag slug, e.g. 'politics', "
                    "'economics', 'sports' (list_markets only)."
                ),
            },
            "active": {
                "type": "boolean",
                "description": "Only markets currently active (list_markets; default true).",
            },
            "closed": {
                "type": "boolean",
                "description": "Include closed markets (list_markets; default false).",
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Page size. list_markets: 25 by default, 50 at most. get_fills: 50 by default."
                ),
            },
            "cursor": {
                "type": "string",
                "description": "The next_cursor from a previous page (list_markets, get_fills).",
            },
            "market_id": {
                "type": "string",
                "description": (
                    "A market's id, as returned by list_markets (get_market, place_order, "
                    "log_forecast)."
                ),
            },
            "outcome": {
                "type": "string",
                "description": (
                    "Which outcome of the market, e.g. 'Yes' or 'No' — exactly as spelled in "
                    "the market's outcomes list."
                ),
            },
            "side": {
                "type": "string",
                "enum": ["buy", "sell"],
                "description": (
                    "buy opens or adds to a position; sell closes shares you already hold "
                    "(there is no short side — to bet against an outcome, buy its opposite)."
                ),
            },
            "size_shares": {
                "type": "number",
                "description": "How many shares to trade (place_order).",
            },
            "order_type": {
                "type": "string",
                "enum": ["market", "limit"],
                "description": (
                    "market takes whatever the book offers now and cancels any unfilled "
                    "remainder; limit fills what it can at your price and rests the rest."
                ),
            },
            "limit_price": {
                "type": "number",
                "description": (
                    "Your price per share, 0 to 1, aligned to the market's tick size "
                    "(required for a limit order)."
                ),
            },
            "client_order_id": {
                "type": "string",
                "description": (
                    "Your own unique id for this order (required). Re-submitting the same id "
                    "returns the original result instead of placing a second order, so a "
                    "retry can never double up."
                ),
            },
            "order_id": {
                "type": "string",
                "description": "The order to cancel, from place_order or get_orders.",
            },
            "status": {
                "type": "string",
                "enum": ["open", "closed", "all"],
                "description": "Which orders to list (get_orders; default 'open').",
            },
            "p": {
                "type": "number",
                "description": (
                    "Your probability that this outcome happens, strictly between 0 and 1 "
                    "(log_forecast). This is the number your Brier score is computed from."
                ),
            },
            "rationale_ref": {
                "type": "string",
                "description": (
                    "A short pointer to your reasoning for this forecast — a memory key, a "
                    "message uuid, or a one-line summary (log_forecast)."
                ),
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        data: PolymarketData | None = None,
        now=None,
    ) -> None:
        self._root = Path(root) if root else None
        self._data = data
        self._now = now or utc_now

    # --- the entry point --------------------------------------------------------

    def run(self, action: str | None = None, **kwargs: Any) -> str:
        """Dispatch one operation and return its JSON body — success or §2.6 error envelope.

        The whole call — reading the ledger, deciding, and writing — happens under the store
        lock, because **this process is not the only writer**. The hourly sweep is a separate
        process against the same ledger, and both fill resting orders: unlocked, a wake and a
        sweep that read the same state a moment apart would each fill the same resting order
        and the log would record a position twice the size the agent ever asked for. A
        read-modify-write over a shared file needs the whole of it inside the lock, not just
        the write (`Epoch.append` is atomic on its own and that is not the property at issue).
        """
        moment = self._now()
        today = moment.date().isoformat()
        try:
            with store_lock(self._home()):
                return self._run_locked(action, kwargs, moment, today)
        except OSError as exc:
            _log.warning("polymarket_paper: ledger store unavailable: %s", exc)
            return _error("upstream_unavailable", f"the paper ledger is unavailable ({exc})", {})

    def _run_locked(self, action: str | None, kwargs: dict, moment, today: str) -> str:
        """`run`'s body, with the store lock held. See `run` for why that boundary matters."""
        epoch = current_epoch(self._home(), now=self._now)
        assert epoch is not None  # current_epoch(create=True) always returns one
        rows = epoch.rows()
        chain = epoch.verify(rows)
        if not chain.ok:
            return self._refuse_broken_chain(epoch, chain)
        state = epoch.state(today=today)

        if action not in ACTIONS:
            # Charged: an unrecognized action still consumed a turn's tool call, and letting
            # it be free is an invitation to spin.
            self._charge(epoch, state, str(action))
            return _error(
                "invalid_params",
                f"unknown action {action!r}; the operations are {', '.join(ACTIONS)}",
                _budgets(state),
            )

        if state.calls_today >= HARD_CALLS_PER_DAY:
            return _error(
                "rate_limited",
                f"this instrument's ceiling of {HARD_CALLS_PER_DAY} calls per UTC day is "
                "spent; it resets at 00:00 UTC",
                _budgets(state),
            )
        if action == "place_order" and state.orders_today >= HARD_ORDERS_PER_DAY:
            return _error(
                "rate_limited",
                f"this instrument's ceiling of {HARD_ORDERS_PER_DAY} orders per UTC day is "
                "spent; it resets at 00:00 UTC (reads are still available)",
                _budgets(state),
            )

        self._charge(epoch, state, action)
        data = self._client()

        try:
            self._presweep(action, epoch, data, state, kwargs)
            body = self._dispatch(action, epoch, data, state, kwargs)
        except PaperReject as reject:
            return _error(reject.code, reject.message, _budgets(state))
        except Exception as exc:  # noqa: BLE001 - a tool never takes the wake down
            _log.exception("polymarket_paper: %s failed", action)
            return _error(
                "upstream_unavailable",
                f"{action} could not be completed ({type(exc).__name__}: {exc})",
                _budgets(state),
            )

        body["ok"] = True
        body["as_of"] = stamp(moment)
        body["budgets"] = _budgets(state)
        return json.dumps(body, ensure_ascii=False)

    def _refuse_broken_chain(self, epoch: Epoch, chain) -> str:
        """A broken ledger chain refuses **everything** — and returns no numbers at all.

        The requirement is `get_pnl` / `get_scorecard`, and this goes past it deliberately: a
        record that cannot vouch for itself cannot be trusted to say what an agent holds, what
        it forecast, or what its budget is either — and a *write* onto an unverifiable chain
        buries the break under new rows. So every operation stops, reads and writes alike, and
        the sweep refuses to append (`_polymarket_engine.sweep`).

        The alternative — numbers plus a warning — is the one failure this instrument cannot
        survive, because the governance layer's scoreboard-tampering trigger has no other
        detector. A degraded scoreboard reads as a working one.

        Nothing here is repairable by the agent, and the message says so, so the model does not
        spend a turn trying. `budgets` is `{}` rather than a count, because the count would
        itself be derived from the rows under suspicion.
        """
        _log.error(
            "polymarket_paper: LEDGER CHAIN BROKEN epoch=%s rows_verified=%d head=%s reason=%s",
            epoch.epoch_id,
            chain.rows,
            chain.head,
            chain.reason,
        )
        return _error(
            "ledger_tampered",
            (
                f"this paper account's ledger does not verify, so no number it holds can be "
                f"reported: {chain.reason} Every operation is refused until an operator "
                f"investigates (epoch {epoch.epoch_id}, {chain.rows} row(s) verified, last "
                f"good chain head {chain.head}). You cannot fix this from here, and nothing "
                "you do will change it — say so if someone asks you why you have stopped."
            ),
            {},
        )

    # --- wiring -----------------------------------------------------------------

    def _home(self) -> Path:
        """Where the ledger lives — always an operator-controlled path, never a model's.

        The bound context's `home` (the agent's `HARNESS_HOME`) first, so the ledger sits
        beside the wake's other stores; then the env var, for the sweep and for a library
        embedding; then the config home, so there is no configuration in which this tool has
        nowhere to write and silently loses a trading record.
        """
        if self._root is not None:
            return self._root
        if self.bound and self.context.home:
            return Path(self.context.home)
        env = os.environ.get("HARNESS_HOME")
        return Path(env) if env else config_home()

    def _client(self) -> PolymarketData:
        if self._data is None:
            self._data = PolymarketData()
        return self._data

    def _charge(self, epoch: Epoch, state: PaperState, action: str) -> None:
        """Append the `call` row that spends one unit of the day's budget."""
        epoch.append(CALL, {"op": action})
        state.seq += 1
        state.calls_today += 1
        if action == "place_order":
            state.orders_today += 1

    def _presweep(
        self, action: str, epoch: Epoch, data: PolymarketData, state: PaperState, kwargs: dict
    ) -> None:
        """§2.4's on-touch re-check, before the operation reads or writes anything.

        Running it *first* is what makes the answer consistent: a `get_positions` that
        settled a resolved market and then reported the pre-settlement position would be
        showing the agent a state that no longer exists.
        """
        if action in _SWEEPS_ONE_MARKET:
            market_id = _text(kwargs.get("market_id"))
            if action == "cancel_order":
                order = state.orders.get(_text(kwargs.get("order_id")))
                market_id = order.market_id if order else ""
            if market_id:
                sweep_market(epoch, data, state, market_id)
        elif action in _SWEEPS_EVERYTHING:
            sweep(epoch, data, state=state)

    def _dispatch(
        self, action: str, epoch: Epoch, data: PolymarketData, state: PaperState, kwargs: dict
    ) -> dict[str, Any]:
        handler = getattr(self, f"_op_{action}")
        return handler(epoch, data, state, kwargs)

    # --- the operations ----------------------------------------------------------

    def _op_list_markets(self, epoch, data, state, kwargs) -> dict[str, Any]:
        """Browse or search markets. Page size is clamped to the §2.2 ceiling, and says so."""
        asked = _int(kwargs.get("limit"), DEFAULT_LIST_LIMIT)
        limit = max(1, min(asked, MAX_LIST_LIMIT))
        markets, next_cursor = data.list_markets(
            query=_text(kwargs.get("query")) or None,
            active=_bool(kwargs.get("active"), True),
            closed=_bool(kwargs.get("closed"), False),
            tag=_text(kwargs.get("tag")) or None,
            limit=limit,
            cursor=_text(kwargs.get("cursor")) or None,
        )
        body: dict[str, Any] = {
            "markets": [_summary_view(m) for m in markets],
            "next_cursor": next_cursor,
        }
        if asked > MAX_LIST_LIMIT:
            body["note"] = f"limit clamped to the maximum page size of {MAX_LIST_LIMIT}"
        return body

    def _op_get_market(self, epoch, data, state, kwargs) -> dict[str, Any]:
        """One market in full, with the top of book per outcome (15 levels a side, §2.3)."""
        market_id = _require(kwargs, "market_id")
        refs = resolve_market(data, market_id)
        market, clob = refs.summary, refs.state
        taker, maker = fee_rates(clob)
        books = []
        for outcome, token in zip(market.outcomes, market.token_ids):
            book = data.book(token, outcome)
            books.append(
                {
                    "outcome": outcome,
                    "bids": [[_num(lvl.price), _num(lvl.size)] for lvl in book.bids[:BOOK_DEPTH]],
                    "asks": [[_num(lvl.price), _num(lvl.size)] for lvl in book.asks[:BOOK_DEPTH]],
                    "mid": _num(book.mid),
                }
            )
        return {
            "market": {
                **_summary_view(market),
                "accepting_orders": clob.accepting_orders and market.accepting_orders,
                "closed": clob.closed,
                "resolved": clob.resolved,
                "winning_outcomes": list(clob.winning_outcomes()),
                "tick_size": _num(clob.tick_size),
                "min_order_size": _num(clob.min_order_size),
                "fees": {
                    "taker_bps": _num(taker.bps),
                    "maker_bps": _num(maker.bps),
                    "source": taker.source,
                },
            },
            "book": books,
        }

    def _op_get_positions(self, epoch, data, state, kwargs) -> dict[str, Any]:
        positions = []
        for position in state.open_positions():
            mark = state.mark_for(position)
            positions.append(
                {
                    "market_id": position.market_id,
                    "outcome": position.outcome,
                    "shares": _num(position.shares),
                    "avg_price": _num(position.avg_price),
                    "mtm_price": _num(mark),
                    "mtm_value": _num(_round(position.shares * mark)),
                    "unrealized_pnl": _num(_round(position.shares * mark - position.cost)),
                }
            )
        return {
            "cash_usd": _num(_round(state.cash)),
            "equity_usd": _num(_round(state.equity)),
            "positions": positions,
        }

    def _op_place_order(self, epoch, data, state, kwargs) -> dict[str, Any]:
        """Place one order against the `as_of` book. Every refusal is a §2.6 code."""
        market_id = _require(kwargs, "market_id")
        outcome = _require(kwargs, "outcome")
        client_order_id = _require(kwargs, "client_order_id")
        side = _choice(kwargs, "side", ("buy", "sell"))
        order_type = _choice(kwargs, "order_type", ("market", "limit"))
        size = _decimal(kwargs.get("size_shares"), "size_shares")
        if size is None or size <= 0:
            raise PaperReject("invalid_params", "size_shares must be a positive number")
        limit_price = _decimal(kwargs.get("limit_price"), "limit_price")
        return place_order(
            epoch,
            data,
            state,
            market_id=market_id,
            outcome=outcome,
            side=side,
            size_shares=size,
            order_type=order_type,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )

    def _op_cancel_order(self, epoch, data, state, kwargs) -> dict[str, Any]:
        order = cancel_order(epoch, state, _require(kwargs, "order_id"))
        return {"cancelled": True, "order": order_result(state, order)}

    def _op_get_orders(self, epoch, data, state, kwargs) -> dict[str, Any]:
        wanted = _text(kwargs.get("status")) or "open"
        if wanted not in ("open", "closed", "all"):
            raise PaperReject("invalid_params", "status must be one of open, closed, all")
        orders = sorted(state.orders.values(), key=lambda o: (o.created_at, o.order_id))
        if wanted == "open":
            orders = [o for o in orders if o.is_open]
        elif wanted == "closed":
            orders = [o for o in orders if not o.is_open]
        total = len(orders)
        shown = orders[-MAX_ORDERS_RETURNED:]
        body = {"status": wanted, "orders": [order_result(state, o) for o in shown], "total": total}
        if total > len(shown):
            body["note"] = (
                f"showing the {len(shown)} most recent of {total} orders; "
                "use get_fills to page through the full history"
            )
        return body

    def _op_get_fills(self, epoch, data, state, kwargs) -> dict[str, Any]:
        limit = max(1, min(_int(kwargs.get("limit"), DEFAULT_FILLS_LIMIT), MAX_FILLS_LIMIT))
        offset = max(0, _int(kwargs.get("cursor"), 0))
        window = state.fills[offset : offset + limit]
        return {
            "fills": [fill_view(f) for f in window],
            "next_cursor": str(offset + limit) if offset + limit < len(state.fills) else None,
            "total": len(state.fills),
        }

    def _op_get_pnl(self, epoch, data, state, kwargs) -> dict[str, Any]:
        return {
            "cash_usd": _num(_round(state.cash)),
            "equity_usd": _num(_round(state.equity)),
            "realized_pnl": _num(_round(state.realized_pnl)),
            "unrealized_pnl": _num(_round(state.unrealized_pnl)),
            "fees_paid": _num(_round(state.fees_paid)),
            "peak_equity": _num(_round(state.peak_equity)),
            # A percentage, not a dollar amount: kept at the scorecard's precision so
            # `get_pnl` and `get_scorecard` can never report the same drawdown differently.
            "max_drawdown_pct": _num(money(state.max_drawdown_pct)),
            "epoch_id": state.epoch_id,
            "chain_head": epoch.head,
            "chain_verified": True,  # `_run_locked` refuses before any number is computed
        }

    def _op_get_scorecard(self, epoch, data, state, kwargs) -> dict[str, Any]:
        """The scorecard, plus the chain head an external verifier pins it against.

        Reaching here at all means the chain verified — `_run_locked` refuses first — so the
        numbers below are known to derive from an intact record, and `chain_head` +
        `chain_rows` let an off-box verifier confirm it is the *same* record it was shipped.
        """
        card = scorecard(state)
        chain = epoch.verify()
        card["chain_head"] = chain.head
        card["chain_rows"] = chain.rows
        card["chain_verified"] = chain.ok
        return card

    def _op_log_forecast(self, epoch, data, state, kwargs) -> dict[str, Any]:
        """Lock a probability for one outcome — §A2's precondition for taking a position.

        The timestamp is the harness's, never the agent's, and a new forecast on the same key
        supersedes the old one rather than replacing it in the log: both rows stay, so the
        record shows what was believed when.
        """
        market_id = _require(kwargs, "market_id")
        outcome = _require(kwargs, "outcome")
        p = _decimal(kwargs.get("p"), "p")
        if p is None or not (Decimal(0) < p < Decimal(1)):
            raise PaperReject("invalid_params", "p must be a probability strictly between 0 and 1")
        refs = resolve_market(data, market_id, outcome)
        forecast_id = state.next_id("fc")
        row = epoch.append(
            FORECAST,
            {
                "forecast_id": forecast_id,
                "market_id": refs.summary.market_id,
                "condition_id": refs.summary.condition_id,
                "outcome": refs.outcome,
                "p": p,
                "rationale_ref": _text(kwargs.get("rationale_ref")),
                "question": refs.summary.question,
                "attribution": state.brier_attribution,
            },
        )
        state.seq += 1
        state.forecasts[(refs.summary.market_id, refs.outcome)] = Forecast(
            forecast_id=forecast_id,
            market_id=refs.summary.market_id,
            outcome=refs.outcome,
            p=p,
            rationale_ref=_text(kwargs.get("rationale_ref")),
            locked_at=row["ts"],
        )
        return {
            "forecast_id": forecast_id,
            "locked_at": row["ts"],
            "market_id": refs.summary.market_id,
            "outcome": refs.outcome,
            "p": _num(p),
        }


# --- response shaping -------------------------------------------------------------


def _error(code: str, message: str, budgets: dict[str, Any]) -> str:
    """§2.6's envelope, verbatim: ``{ok, error, message, budgets}``."""
    return json.dumps(
        {"ok": False, "error": code, "message": message, "budgets": budgets},
        ensure_ascii=False,
    )


def _budgets(state: PaperState) -> dict[str, Any]:
    """What is left of the UTC day, hard ceilings first (§2.2).

    The soft lines ride alongside: without them a soft limit is a number in a table that
    nothing ever surfaces, and the agent has no way to pace itself before the wall.
    """
    return {
        "calls_remaining_day": max(0, HARD_CALLS_PER_DAY - state.calls_today),
        "orders_remaining_day": max(0, HARD_ORDERS_PER_DAY - state.orders_today),
        "soft_calls_remaining_day": max(0, SOFT_CALLS_PER_DAY - state.calls_today),
        "soft_orders_remaining_day": max(0, SOFT_ORDERS_PER_DAY - state.orders_today),
    }


def _summary_view(market) -> dict[str, Any]:
    """A `MarketSummary` as §2.3 specifies it."""
    return {
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "question": market.question,
        "slug": market.slug,
        "outcomes": list(market.outcomes),
        "mid_prices": [_num(p) for p in market.mid_prices],
        "volume_24h": _num(market.volume_24h),
        "liquidity": _num(market.liquidity),
        "end_date": market.end_date,
        "accepting_orders": market.accepting_orders,
        "tick_size": _num(market.tick_size),
        "min_order_size": _num(market.min_order_size),
        "tags": list(market.tags),
        "event_id": market.event_id,
    }


# --- parameter coercion ------------------------------------------------------------
#
# Every one of these raises `PaperReject("invalid_params", …)` rather than a `TypeError`:
# a model that passes a string where a number belongs should read a sentence telling it so,
# not have its turn broken by a traceback.


def _require(kwargs: dict, key: str) -> str:
    value = _text(kwargs.get(key))
    if not value:
        raise PaperReject("invalid_params", f"this action needs a {key}")
    return value


def _choice(kwargs: dict, key: str, allowed: tuple[str, ...]) -> str:
    value = _text(kwargs.get(key)).casefold()
    if value not in allowed:
        raise PaperReject("invalid_params", f"{key} must be one of {', '.join(allowed)}")
    return value


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _decimal(value: Any, key: str) -> Decimal | None:
    """A finite `Decimal`, or a readable refusal.

    The finiteness check is not pedantry: `Decimal("NaN")` *parses*, and then the first
    comparison against it raises `InvalidOperation` several frames deeper — surfacing as a
    generic failure with the wrong error code, from an input that should simply have been
    refused here. `Infinity` is the same shape of problem with a different name.
    """
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PaperReject("invalid_params", f"{key} must be a number") from exc
    if not parsed.is_finite():
        raise PaperReject("invalid_params", f"{key} must be a finite number")
    return parsed


def _round(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


def _num(value: Decimal | None) -> float | None:
    return None if value is None else float(value)
