"""The operator-controlled, append-only paper ledger — and the fold that reads it (issue #347).

NORMATIVE §"Ledger boundary" gives this file its whole shape: the ledger is an
append-only log the *harness* owns, **not** agent disk and **not** agent memory; every row
carries ``epoch_id, ts, type, payload, schema_version``; there is no UPDATE and no DELETE
of a past row, and a correction is a compensating entry. The agent reads through the
`get_*` projections and may never supply or define a P&L number.

**Where it lives, and why that is the operator's disk.** An epoch is a directory under the
agent's ``HARNESS_HOME`` — beside `marks/`, `seen/`, `claims/`, `breaker/`, the stores the
wake already keeps there. That path is reachable by the harness and by whoever runs the
box; it is reachable by the *model* through no tool in this stem, because every operation
here is a typed projection and none of them takes a path, a row, or a number.

**State is a fold, never a side-file.** Cash, positions, open orders, fills, forecasts,
the equity curve *and* the day's burn counters are all derived by replaying the rows
(`replay`). This is the same reasoning the delivery guarantee uses for its create ordinal
(CLAUDE.md → Delivery Guarantee): a counter is a second source of truth and it drifts the
first moment a write lands that the counter misses. Here the temptation is stronger,
because a `budgets.json` would be so easy — and the first crash mid-order would leave it
disagreeing with the log about how much of the day the agent has spent. One source, folded
on every call. A whole epoch is at most a few tens of thousands of small rows (the §2.2
ceiling is 200 calls a day), so the fold costs milliseconds and buys an audit trail that
cannot disagree with itself.

**The rows are hash-chained, and that is tamper-*evidence*, not tamper-proofing.** Each row
carries `prev` (the hash of the row before it) and its own `hash`, so an edit, a deletion or a
reordering breaks the linkage and is caught by replay — cheap, because a fold-based design is
already re-reading every row. The distinction matters and should not be overstated: the harness
runs as the **agent's own UID**, so a same-UID write path is unavoidable and "the agent cannot
write this file" was never true at the filesystem layer — it was true only because this agent
has no shell. Integrity therefore rests on detection, not prevention: `verify_chain` refuses to
report numbers off a broken chain, and the **on-box JSONL is a spool** whose authoritative copy
is the per-row log line the NOC ships off-box, under a UID the agent does not have. The chain
head and row count surface in `get_scorecard` so an external verifier can pin both — which is
what catches a tamperer who re-hashes the file forward or truncates its tail.

**Money is `Decimal` end to end.** Prices arrive from Polymarket as strings, sizes are
fractional, and a tick is 0.001; floats would put `0.30000000000000004` into a row that is
never allowed to be corrected in place. Serialization to the ledger is via `str(Decimal)`,
so a value round-trips exactly.

**The frozen-for-the-epoch choices live in the first row.** `epoch_open` records the
bankroll, the caps, the budget ceilings, the fill-model version, the resting-limit
re-check policy, the fee defaults and — §A2's explicit implementer choice —
``brier_attribution: "position_open"``. Freezing them *in the log* rather than in code is
what makes a later change safe: constants may move, and an epoch that was scored under the
old ones still says so.

**It records the rules this instrument *enforces*, and nothing else** (issue #350). Every
entry in that row is a rule the machine obeys — a cap that refuses an order, a fee that
changes a fill, an attribution that decides what gets scored. Promotion thresholds are not
such a rule: they gate nothing here, they change no number below, and they belong to a
governance contract this package cannot read. Recording a copy of them made the row assert
something it had no way to keep true, inside a tamper-evident record that lent the copy an
authority it did not have. Each layer pre-commits the rules it owns; this one owns
measurement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

try:  # POSIX-only; the fleet is Linux/macOS, but the kit is public (see `store_lock`)
    import fcntl
except ImportError:  # pragma: no cover - not the fleet's platform
    fcntl = None  # type: ignore[assignment]

_log = logging.getLogger("basecradle_harness")

#: The ledger row schema. Bumped when a row's payload shape changes incompatibly; every
#: row carries it, so a reader can always tell which vintage it is looking at. **2** added the
#: hash chain (`prev` + `hash`); version-1 rows are unchained and are *refused* rather than
#: accepted, because a chain cannot vouch for rows written before it existed.
SCHEMA_VERSION = 2

#: What the first row of an epoch chains from. A fixed, domain-separated constant rather than
#: zeros, so a row lifted out of another log cannot pass as somebody's genesis.
GENESIS_PREV = "genesis:basecradle-harness:polymarket_paper:v1"

#: §A2's implementer choice, made once and frozen per epoch: a Brier observation is locked
#: from the forecast current at **position open** — the instant a ``(market_id, outcome)``
#: position goes from flat to non-flat. The rejected alternative was per-fill attribution,
#: which lets an agent slice one conviction into fifty fills and dilute a bad call fifty to
#: one against a good one. One position-open, one scored observation, not gameable by order
#: slicing.
BRIER_ATTRIBUTION = "position_open"

#: §2.4's "choose and freeze" for resting limit orders: a re-check runs on the hourly
#: harness/NOC sweep **and** on any agent tool call that touches the market. Predictable
#: beats clever.
RESTING_RECHECK_POLICY = "hourly_sweep_plus_on_touch"

#: The deterministic fill model's version, recorded so a scorecard can never be read
#: against the wrong simulation semantics.
FILL_MODEL_VERSION = "book_walk_v1"

# --- v1 constants (NORMATIVE §2.2 and §2.3 "Caps") -----------------------------

#: The virtual bankroll an epoch opens with. **Not top-upable by the agent** — no operation
#: in this stem accepts cash, and the only row that credits it is `epoch_open`.
BANKROLL_USD = Decimal("10000")

#: Max notional for a single order.
MAX_ORDER_NOTIONAL = Decimal("500")

#: Max net notional exposure in one market — held cost basis plus resting buy notional.
MAX_MARKET_EXPOSURE = Decimal("2000")

#: Max simultaneously open positions across the epoch.
MAX_OPEN_POSITIONS = 20

#: Tool calls per UTC day: a soft line that only warns, and a hard ceiling that returns
#: `rate_limited` until 00:00 UTC.
SOFT_CALLS_PER_DAY = 120
HARD_CALLS_PER_DAY = 200

#: `place_order` calls per UTC day, same two-line shape.
SOFT_ORDERS_PER_DAY = 20
HARD_ORDERS_PER_DAY = 40

#: `list_markets` page size: the default, and the most the agent may ask for.
DEFAULT_LIST_LIMIT = 25
MAX_LIST_LIMIT = 50

#: The fee rates used **only** when the CLOB published none for a market (§2.4). A
#: published zero is a rate and is recorded as one; this is the "never silently zero fees"
#: fallback, and a fill that uses it is tagged ``fee_source="default"``.
DEFAULT_TAKER_FEE_BPS = Decimal("100")
DEFAULT_MAKER_FEE_BPS = Decimal("0")

#: There are deliberately **no promotion thresholds here** (issue #350). §2.3 names
#: `kill_flags` and `promotion_eligible` without defining them, and 0.87.0 filled that gap by
#: inventing four numbers — which turned out to disagree with the governing contract in both
#: directions (4x and 6x looser on the sample floors, stricter on Brier). The numbers were the
#: symptom; the disease is that a package which cannot read the contract, cannot test against
#: it, and will not be told when it moves is not a place a governing threshold can live. So
#: this stem measures and does not grade: it reports `resolved_n`, `brier`,
#: `calibration_error`, `hit_rate`, `paper_pnl`, `max_drawdown_pct` and
#: `distinct_event_clusters`, and the governance layer — the authority, holding the only copy
#: that binds anything — does the comparison. A tool that reports facts and refuses to render
#: a verdict beats one that renders a verdict against thresholds nobody here can verify.

#: Calibration bins: ten equal-width buckets over (0,1), the standard ECE decomposition.
CALIBRATION_BINS = 10

# --- row types -----------------------------------------------------------------

EPOCH_OPEN = "epoch_open"
CALL = "call"
FORECAST = "forecast"
ORDER = "order"
FILL = "fill"
ORDER_CLOSE = "order_close"
MARK = "mark"
SETTLEMENT = "settlement"
BRIER_OBS = "brier_obs"
BRIER_SCORE = "brier_score"
FREEZE = "freeze"
UNFREEZE = "unfreeze"
NOTE = "note"

#: The store's directory under the agent's home. One level, then one dir per epoch.
STORE_DIRNAME = "polymarket"

_LEDGER_FILENAME = "ledger.jsonl"


def utc_now() -> datetime:
    """The harness clock, UTC — §2.4's authority. No agent-supplied timestamp anywhere."""
    return datetime.now(timezone.utc)


def stamp(moment: datetime) -> str:
    """An ISO-8601 UTC timestamp with a ``Z`` suffix — the `ts` on every row."""
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_stamp(value: str) -> datetime | None:
    """A row's `ts` back into a datetime, or ``None`` if it is unreadable."""
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


# --- the folded state -----------------------------------------------------------


@dataclass
class Position:
    """An open position in one ``(market_id, outcome)``. Long-only: there is no short.

    Shorting an outcome is expressed the way Polymarket expresses it — by buying the
    complementary outcome — so a sell can never take shares below zero, and `place_order`
    rejects one that would with `insufficient_shares`.
    """

    market_id: str
    condition_id: str
    outcome: str
    token_id: str
    shares: Decimal = Decimal(0)
    cost: Decimal = Decimal(0)  # cost basis of the open shares, fees excluded
    opened_at: str = ""

    @property
    def avg_price(self) -> Decimal:
        return (self.cost / self.shares) if self.shares > 0 else Decimal(0)


@dataclass
class Order:
    """An accepted order. Open until an `order_close` row terminates it."""

    order_id: str
    client_order_id: str
    market_id: str
    condition_id: str
    outcome: str
    token_id: str
    side: str  # buy | sell
    order_type: str  # market | limit
    size_shares: Decimal
    limit_price: Decimal | None
    created_at: str
    filled_shares: Decimal = Decimal(0)
    status: str = "open"  # open | filled | partially_filled | cancelled
    close_reason: str = ""

    @property
    def remaining(self) -> Decimal:
        return self.size_shares - self.filled_shares

    @property
    def is_open(self) -> bool:
        return self.status == "open"


@dataclass
class Forecast:
    """A locked probability for a ``(market_id, outcome)`` — §A2's gate on `place_order`."""

    forecast_id: str
    market_id: str
    outcome: str
    p: Decimal
    rationale_ref: str
    locked_at: str


@dataclass
class Observation:
    """One Brier observation: a forecast locked at position-open, scored at settlement."""

    obs_id: str
    market_id: str
    outcome: str
    p: Decimal
    forecast_id: str
    opened_at: str
    event_id: str = ""
    result: int | None = None  # 1 won, 0 lost, None unresolved
    brier: Decimal | None = None
    scored_at: str = ""


@dataclass
class PaperState:
    """Everything the ledger says, folded. Built only by `replay` — never mutated by a tool.

    The equity curve (`peak_equity`, `max_drawdown_pct`) is derived *during* the fold rather
    than sampled by a side process: after every row that moves cash, positions or marks, the
    running equity is recomputed and the peak and worst drawdown updated. That makes the
    drawdown a property of the log — replay it on another machine and get the same number —
    instead of a statistic whose accuracy depends on how often something remembered to
    sample it.
    """

    epoch_id: str = ""
    started_at: str = ""
    schema_version: int = SCHEMA_VERSION
    brier_attribution: str = BRIER_ATTRIBUTION
    cash: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    fees_paid: Decimal = Decimal(0)
    positions: dict[tuple[str, str], Position] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)
    client_order_ids: dict[str, str] = field(default_factory=dict)
    forecasts: dict[tuple[str, str], Forecast] = field(default_factory=dict)
    marks: dict[tuple[str, str], Decimal] = field(default_factory=dict)
    fills: list[dict[str, Any]] = field(default_factory=list)
    observations: dict[str, Observation] = field(default_factory=dict)
    event_clusters: set[str] = field(default_factory=set)
    frozen: bool = False
    frozen_reason: str = ""
    peak_equity: Decimal = Decimal(0)
    max_drawdown_pct: Decimal = Decimal(0)
    calls_today: int = 0
    orders_today: int = 0
    seq: int = 0  # rows seen — the source of the next deterministic id

    # --- projections ------------------------------------------------------------

    def open_positions(self) -> list[Position]:
        """Positions with shares outstanding, in a stable (market, outcome) order."""
        return [p for _, p in sorted(self.positions.items()) if p.shares > 0]

    def open_orders(self) -> list[Order]:
        """Resting orders, oldest first — the FIFO the sweep re-checks."""
        return sorted(
            (o for o in self.orders.values() if o.is_open), key=lambda o: (o.created_at, o.order_id)
        )

    def mark_for(self, position: Position) -> Decimal:
        """The MTM price for a position: the last recorded mark, else its own average cost.

        Falling back to average cost (rather than to zero, or to nothing) keeps `equity`
        defined for a position whose market has not been marked yet — a position opened
        seconds ago, before any sweep. It reads as "no gain, no loss yet", which is the
        honest statement of what is known.
        """
        return self.marks.get((position.market_id, position.outcome), position.avg_price)

    @property
    def equity(self) -> Decimal:
        """Cash plus the marked value of every open position."""
        return self.cash + sum(
            (p.shares * self.mark_for(p) for p in self.positions.values()), Decimal(0)
        )

    @property
    def unrealized_pnl(self) -> Decimal:
        return sum(
            (p.shares * self.mark_for(p) - p.cost for p in self.positions.values()), Decimal(0)
        )

    def market_exposure(self, market_id: str) -> Decimal:
        """Net notional at risk in one market: held cost basis + resting buy notional.

        The §2.3 ``$2,000`` cap is checked against this. Resting buys count because the cash
        behind them is committed — an exposure cap that ignored working orders would let an
        agent queue ten thousand dollars of bids under a two-thousand-dollar ceiling.
        """
        held = sum(
            (p.cost for (mid, _), p in self.positions.items() if mid == market_id), Decimal(0)
        )
        resting = sum(
            (
                o.remaining * (o.limit_price or Decimal(0))
                for o in self.orders.values()
                if o.is_open and o.market_id == market_id and o.side == "buy"
            ),
            Decimal(0),
        )
        return held + resting

    @property
    def committed_cash(self) -> Decimal:
        """Cash reserved by resting buy orders — spent from the agent's point of view."""
        return sum(
            (
                o.remaining * (o.limit_price or Decimal(0))
                for o in self.orders.values()
                if o.is_open and o.side == "buy"
            ),
            Decimal(0),
        )

    @property
    def available_cash(self) -> Decimal:
        """Cash a new buy may actually draw on: settled cash less what resting bids hold."""
        return self.cash - self.committed_cash

    def forecast_for(self, market_id: str, outcome: str) -> Forecast | None:
        return self.forecasts.get((market_id, outcome))

    def next_id(self, prefix: str) -> str:
        """A deterministic per-epoch id: ``ord-000007``. Row count, never randomness.

        Deterministic ids are what let a test replay a scenario and assert on the ledger by
        value, and they make two runs of the same sweep over the same log produce identical
        rows.
        """
        return f"{prefix}-{self.seq + 1:06d}"


# --- the fold --------------------------------------------------------------------


def replay(rows: list[dict[str, Any]], *, today: str | None = None) -> PaperState:
    """Fold the ledger's rows into a `PaperState`.

    `today` is the UTC date (``YYYY-MM-DD``) the day-budget counters are computed for;
    ``None`` uses the harness clock. Passing it explicitly is what makes the §2.2 rollover
    ("`rate_limited` until 00:00 UTC") testable without waiting for midnight.
    """
    day = today or utc_now().date().isoformat()
    state = PaperState()
    for row in rows:
        state.seq += 1
        kind = str(row.get("type") or "")
        payload = row.get("payload") or {}
        ts = str(row.get("ts") or "")
        _apply(state, kind, payload, ts, day, row)
        if kind in _EQUITY_MOVING:
            track_equity(state)
    return state


#: The row kinds that can move the equity curve. Everything else (a `call`, a `forecast`)
#: leaves the curve alone, so a busy read-only day does not manufacture drawdown samples.
_EQUITY_MOVING = frozenset({EPOCH_OPEN, FILL, SETTLEMENT, MARK})


def _apply(
    state: PaperState,
    kind: str,
    payload: dict[str, Any],
    ts: str,
    day: str,
    row: dict[str, Any],
) -> None:
    """Apply one row to the running state. One branch per row type, in ledger order."""
    if kind == EPOCH_OPEN:
        state.epoch_id = str(row.get("epoch_id") or "")
        state.started_at = ts
        state.schema_version = int(row.get("schema_version") or SCHEMA_VERSION)
        state.brier_attribution = str(payload.get("brier_attribution") or BRIER_ATTRIBUTION)
        state.cash = _dec(payload.get("bankroll_usd"), BANKROLL_USD)
        state.peak_equity = state.cash
        return

    if kind == CALL:
        if ts.startswith(day):
            state.calls_today += 1
            if payload.get("op") == "place_order":
                state.orders_today += 1
        return

    if kind == FORECAST:
        forecast = Forecast(
            forecast_id=str(payload.get("forecast_id") or ""),
            market_id=str(payload.get("market_id") or ""),
            outcome=str(payload.get("outcome") or ""),
            p=_dec(payload.get("p"), Decimal(0)),
            rationale_ref=str(payload.get("rationale_ref") or ""),
            locked_at=ts,
        )
        state.forecasts[(forecast.market_id, forecast.outcome)] = forecast
        return

    if kind == ORDER:
        order = Order(
            order_id=str(payload.get("order_id") or ""),
            client_order_id=str(payload.get("client_order_id") or ""),
            market_id=str(payload.get("market_id") or ""),
            condition_id=str(payload.get("condition_id") or ""),
            outcome=str(payload.get("outcome") or ""),
            token_id=str(payload.get("token_id") or ""),
            side=str(payload.get("side") or ""),
            order_type=str(payload.get("order_type") or ""),
            size_shares=_dec(payload.get("size_shares"), Decimal(0)),
            limit_price=_dec_or_none(payload.get("limit_price")),
            created_at=ts,
        )
        state.orders[order.order_id] = order
        if order.client_order_id:
            state.client_order_ids[order.client_order_id] = order.order_id
        if payload.get("event_id"):
            state.event_clusters.add(str(payload["event_id"]))
        return

    if kind == FILL:
        apply_fill(state, payload, ts)
        return

    if kind == ORDER_CLOSE:
        order = state.orders.get(str(payload.get("order_id") or ""))
        if order is not None:
            order.status = str(payload.get("status") or "cancelled")
            order.close_reason = str(payload.get("reason") or "")
        return

    if kind == MARK:
        price = _dec_or_none(payload.get("price"))
        if price is not None:
            state.marks[(str(payload.get("market_id")), str(payload.get("outcome")))] = price
        return

    if kind == SETTLEMENT:
        apply_settlement(state, payload)
        return

    if kind == BRIER_OBS:
        obs = Observation(
            obs_id=str(payload.get("obs_id") or ""),
            market_id=str(payload.get("market_id") or ""),
            outcome=str(payload.get("outcome") or ""),
            p=_dec(payload.get("p"), Decimal(0)),
            forecast_id=str(payload.get("forecast_id") or ""),
            opened_at=ts,
            event_id=str(payload.get("event_id") or ""),
        )
        state.observations[obs.obs_id] = obs
        if obs.event_id:
            state.event_clusters.add(obs.event_id)
        return

    if kind == BRIER_SCORE:
        obs = state.observations.get(str(payload.get("obs_id") or ""))
        if obs is not None:
            obs.result = int(payload.get("result") or 0)
            obs.brier = _dec_or_none(payload.get("brier"))
            obs.scored_at = ts
        return

    if kind == FREEZE:
        state.frozen = True
        state.frozen_reason = str(payload.get("reason") or "")
        return

    if kind == UNFREEZE:
        state.frozen = False
        state.frozen_reason = ""
        return

    if kind == NOTE:
        return  # a compensating/observational entry: recorded, no state change

    _log.warning("polymarket ledger: ignoring unknown row type %r", kind)


def apply_fill(state: PaperState, payload: dict[str, Any], ts: str) -> None:
    """Move cash, position and realized P&L for one fill.

    Fees are a realized cost the moment they are charged — they leave cash and are booked
    into `realized_pnl` immediately — and they are kept *out* of the position's cost basis
    so `avg_price` stays a price rather than a price-plus-friction.
    """
    market_id = str(payload.get("market_id") or "")
    outcome = str(payload.get("outcome") or "")
    key = (market_id, outcome)
    shares = _dec(payload.get("shares"), Decimal(0))
    price = _dec(payload.get("price"), Decimal(0))
    fee = _dec(payload.get("fee"), Decimal(0))
    notional = shares * price
    state.fills.append(dict(payload, ts=ts))
    state.fees_paid += fee
    state.realized_pnl -= fee

    order = state.orders.get(str(payload.get("order_id") or ""))
    if order is not None:
        order.filled_shares += shares

    if str(payload.get("side")) == "buy":
        state.cash -= notional + fee
        position = state.positions.get(key)
        if position is None:
            position = Position(
                market_id=market_id,
                condition_id=str(payload.get("condition_id") or ""),
                outcome=outcome,
                token_id=str(payload.get("token_id") or ""),
                opened_at=ts,
            )
            state.positions[key] = position
        position.shares += shares
        position.cost += notional
        return

    position = state.positions.get(key)
    if position is None:  # defensive: a sell is gated on holdings before it is ever written
        _log.warning("polymarket ledger: sell fill with no position for %s/%s", market_id, outcome)
        state.cash += notional - fee
        return
    basis = position.avg_price * shares
    position.shares -= shares
    position.cost -= basis
    state.cash += notional - fee
    state.realized_pnl += notional - basis
    if position.shares <= 0:
        state.positions.pop(key, None)


def apply_settlement(state: PaperState, payload: dict[str, Any]) -> None:
    """Credit a resolved position to $1.00 or $0.00 a share and close it (§2.4)."""
    key = (str(payload.get("market_id") or ""), str(payload.get("outcome") or ""))
    position = state.positions.get(key)
    payout = _dec(payload.get("payout"), Decimal(0))
    basis = _dec(payload.get("cost_basis"), position.cost if position else Decimal(0))
    state.cash += payout
    state.realized_pnl += payout - basis
    state.positions.pop(key, None)
    state.marks.pop(key, None)


def track_equity(state: PaperState) -> None:
    """Update the peak and the worst drawdown after a row that moved the curve."""
    equity = state.equity
    if equity > state.peak_equity:
        state.peak_equity = equity
    if state.peak_equity > 0:
        drawdown = (state.peak_equity - equity) / state.peak_equity * Decimal(100)
        if drawdown > state.max_drawdown_pct:
            state.max_drawdown_pct = drawdown


# --- the hash chain ----------------------------------------------------------------


@dataclass(frozen=True)
class ChainStatus:
    """The verdict on an epoch's hash chain: does the record vouch for itself?

    `ok` is the only thing a caller should branch on, and the branch is **refuse**, never
    "report with a warning". `head` and `rows` are what an external verifier pins against the
    off-box copy — together they catch the attack the chain alone cannot, a tamperer who
    rewrites the file forward or truncates its tail into a shorter but internally consistent
    chain.
    """

    ok: bool
    head: str
    rows: int
    broken_at: int | None = None
    reason: str = ""


def canonical(row: Mapping[str, Any]) -> bytes:
    """A row's bytes for hashing: sorted keys, compact, its own `hash` excluded.

    Computed from the **parsed** row rather than the line on disk, so re-serializing with
    different whitespace, key order or escaping cannot change a hash. That matters because the
    verifier and the writer are different code paths at different times.
    """
    body = {key: value for key, value in row.items() if key != "hash"}
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def row_hash(row: Mapping[str, Any]) -> str:
    """The row's own hash — sha256 over `canonical`, hex."""
    return hashlib.sha256(canonical(row)).hexdigest()


def verify_chain(epoch_id: str, rows: list[dict[str, Any]]) -> ChainStatus:
    """Recompute every row's hash and check its linkage to the row before it.

    Three ways a chain fails, and each names the row it failed at so a human has somewhere to
    look: a row whose recomputed hash does not match the one it carries (its content changed),
    a row whose `prev` does not match the previous row's hash (a row was removed or reordered),
    and a first row that does not chain from genesis (the head of the log was cut off).

    An **unchained** row — one written before the chain existed, `schema_version` 1 — fails
    too, deliberately. A chain cannot vouch for rows it never saw, so accepting them would make
    the whole verdict a lie about exactly the period an attacker would target.
    """
    if not rows:
        return ChainStatus(ok=True, head=GENESIS_PREV, rows=0)
    previous = GENESIS_PREV
    for index, row in enumerate(rows):
        carried = row.get("hash")
        if not carried or "prev" not in row:
            return ChainStatus(
                ok=False,
                head=previous,
                rows=index,
                broken_at=index,
                reason=(
                    f"row {index} ({row.get('type')!r}) carries no hash — it predates the "
                    "chained ledger (schema_version 1). A chain cannot vouch for rows written "
                    "before it existed; start a new epoch."
                ),
            )
        if row.get("prev") != previous:
            return ChainStatus(
                ok=False,
                head=previous,
                rows=index,
                broken_at=index,
                reason=(
                    f"row {index} ({row.get('type')!r}) chains to {row.get('prev')!r} but the "
                    f"row before it hashes to {previous!r} — a row was altered, removed or "
                    "reordered here."
                ),
            )
        recomputed = row_hash(row)
        if recomputed != carried:
            return ChainStatus(
                ok=False,
                head=previous,
                rows=index,
                broken_at=index,
                reason=(
                    f"row {index} ({row.get('type')!r}) does not match its own hash — its "
                    "content was changed after it was written."
                ),
            )
        if row.get("epoch_id") != epoch_id:
            return ChainStatus(
                ok=False,
                head=previous,
                rows=index,
                broken_at=index,
                reason=(
                    f"row {index} claims epoch {row.get('epoch_id')!r} but sits in "
                    f"{epoch_id!r} — a row was moved between epochs."
                ),
            )
        previous = str(carried)
    return ChainStatus(ok=True, head=previous, rows=len(rows))


# --- the epoch -------------------------------------------------------------------


class Epoch:
    """One paper-trading epoch: a directory, an append-only log, and the fold over it.

    `append` is the only writer, and there is deliberately no `update`, no `rewrite` and no
    `delete`: correcting something means appending a compensating row, which is what
    §"Ledger boundary" requires and what makes the log an audit trail rather than a
    database with history attached.
    """

    def __init__(self, directory: str | Path, *, now=None) -> None:
        self.directory = Path(directory)
        self.epoch_id = self.directory.name
        self.path = self.directory / _LEDGER_FILENAME
        self._now = now or utc_now

    # --- reading ---------------------------------------------------------------

    def rows(self) -> list[dict[str, Any]]:
        """Every row, in order. A torn final line is skipped, never repaired in place.

        A kill mid-append can leave a partial last line. Dropping it on *read* is not a
        delete — the bytes stay on disk for a human to look at — and it keeps one unlucky
        signal from bricking the instrument, exactly as `Session._load` heals rather than
        refuses (CLAUDE.md → Delivery Guarantee).
        """
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                _log.warning("polymarket ledger %s: unreadable row at line %d", self.path, number)
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def state(self, *, today: str | None = None) -> PaperState:
        """The folded state of this epoch right now.

        Callers that will *report numbers* must check `verify` first — see `ChainStatus`. This
        does not verify on its own, because the sweep and the tool both want the fold and the
        verdict from **one** read of the file rather than two.
        """
        state = replay(self.rows(), today=today)
        state.epoch_id = state.epoch_id or self.epoch_id
        return state

    def verify(self, rows: list[dict[str, Any]] | None = None) -> ChainStatus:
        """Whether this epoch's hash chain holds, and its head. See `verify_chain`."""
        return verify_chain(self.epoch_id, self.rows() if rows is None else rows)

    @property
    def head(self) -> str:
        """The hash of the last row — the value an external verifier pins.

        Read off the last *parseable* line rather than kept in memory, because the other
        writer (the sweep, or the wake) may have appended since this object was built.
        """
        for row in reversed(self.rows()):
            digest = row.get("hash")
            if digest:
                return str(digest)
        return GENESIS_PREV

    # --- writing ----------------------------------------------------------------

    def append(self, kind: str, payload: dict[str, Any], *, at: datetime | None = None) -> dict:
        """Append one **chained** row and return it. The only mutation this module offers.

        Each row carries `prev` (the previous row's hash, genesis for the first) and its own
        `hash`, so a retroactive edit or a removed row breaks the linkage at the next row and
        is caught by replay (`verify_chain`). This is **tamper-evidence, not tamper-proofing**:
        the harness runs as the agent's own UID, so a same-UID write path is unavoidable and
        the honest goal is that an edit cannot go unnoticed rather than that it cannot happen.

        The write is `O_APPEND` + `fsync`: a row a later fold depends on must be on disk before
        the operation that wrote it reports success, for the same reason the engine persists an
        assistant turn before dispatching its tools.

        **The row is also emitted to the log, in full.** The on-box JSONL is a *spool*; the
        record is the off-box copy the NOC already ships out of journald. That is what closes
        the one attack a chain alone cannot see — a tamperer who edits a row and re-hashes the
        whole file forward, or who truncates its tail — because the off-box copy carries the
        rows and the heads as they were written, under a UID the agent does not have.
        """
        moment = at or self._now()
        row = {
            "epoch_id": self.epoch_id,
            "ts": stamp(moment),
            "type": kind,
            "payload": _jsonable(payload),
            "schema_version": SCHEMA_VERSION,
            "prev": self.head,
        }
        row["hash"] = row_hash(row)
        self.directory.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        # The off-box path. One line per row, greppable, machine-parseable; at the §2.2 ceiling
        # of 200 calls a day this is a few hundred KB per agent per day, which buys a copy of
        # the record that the agent's UID cannot reach or rewrite.
        _log.info("polymarket_ledger_row %s", line.rstrip("\n"))
        return row


def store_root(home: str | Path) -> Path:
    """The instrument's store under an agent's home: ``<home>/polymarket``."""
    return Path(home) / STORE_DIRNAME


@contextmanager
def store_lock(home: str | Path):
    """Hold an exclusive lock on one agent's paper store, across processes.

    **There is more than one writer, and that is by design.** The tool writes from a wake; the
    hourly sweep (`basecradle-harness-polymarket-sweep`) writes from a cron job in a different
    process. Both of them *fill resting orders* — so two overlapping runs that each read the
    state a moment apart would each fill the same resting order, and the ledger would record a
    position twice the size the agent ever asked for. An append-only log makes that permanent:
    there is no row to take back, only a compensating one somebody has to notice is needed.

    So the boundary is the whole **read-modify-write**, not the write. `Epoch.append` is
    already atomic on its own, and that is not the property at issue: the hazard is two
    processes deciding from the same stale fold. A caller takes this lock once, at its entry
    point, around fold *and* mutation — the tool in `run`, the job in `main`. It is
    deliberately **not** taken inside `sweep()`, because the tool calls `sweep()` while already
    holding it and a second exclusive `flock` from the same process on a second descriptor is a
    deadlock, not a no-op.

    It is also why the lock lives on the **store**, not on an epoch directory: `current_epoch`
    can *create* the first epoch, and two processes racing that would leave two epochs where
    the agent should have one.

    On a platform without `fcntl` (not the fleet's, but the kit is public) it degrades to no
    locking rather than refusing to run — a single-writer install is the common case, and
    failing there would be worse than the race it prevents.
    """
    root = store_root(home)
    handle = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        handle = open(root / ".lock", "a+")  # noqa: SIM115 - released in the finally below
    except OSError as error:
        _log.warning("polymarket ledger: could not take the store lock (%s); proceeding", error)
    try:
        if handle is not None and fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if handle is not None:
            handle.close()  # closing the descriptor releases the flock


def epochs(home: str | Path) -> list[Epoch]:
    """Every epoch under `home`, oldest first (epoch ids sort chronologically by name)."""
    root = store_root(home)
    if not root.is_dir():
        return []
    return [Epoch(child) for child in sorted(root.iterdir()) if (child / _LEDGER_FILENAME).exists()]


def current_epoch(home: str | Path, *, create: bool = True, now=None) -> Epoch | None:
    """The active epoch under `home` — the newest one, opened on demand.

    `create=False` is the sweep's mode: a box with no epoch yet has nothing to sweep, and
    inventing one there would write the first row of a trading record from a cron job.
    """
    existing = epochs(home)
    if existing:
        return existing[-1]
    if not create:
        return None
    return open_epoch(home, now=now)


def open_epoch(home: str | Path, *, now=None) -> Epoch:
    """Start a new epoch: a fresh directory and the `epoch_open` row that freezes its terms.

    Everything §2.3, §2.4 and §A2 leave to the implementer *and this stem then enforces* is
    written down here — the bankroll, the caps, the day ceilings, the fee defaults, the fill
    model, the resting re-check policy and the Brier attribution — so a scorecard can never be
    read against terms other than the ones its epoch actually ran under.

    Promotion thresholds are **not** among them (issue #350): they gate nothing this row
    governs, and they are owned by a contract this package cannot read. See the constants
    block above for why a copy here was worse than no copy at all.
    """
    clock = now or utc_now
    started = clock()
    epoch_id = "epoch-" + started.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = store_root(home) / epoch_id
    suffix = 1
    while (directory / _LEDGER_FILENAME).exists():  # two epochs in one second: never collide
        suffix += 1
        directory = store_root(home) / f"{epoch_id}-{suffix}"
    epoch = Epoch(directory, now=clock)
    epoch.append(
        EPOCH_OPEN,
        {
            "bankroll_usd": BANKROLL_USD,
            "brier_attribution": BRIER_ATTRIBUTION,
            "fill_model": FILL_MODEL_VERSION,
            "resting_recheck": RESTING_RECHECK_POLICY,
            "caps": {
                "max_order_notional_usd": MAX_ORDER_NOTIONAL,
                "max_market_exposure_usd": MAX_MARKET_EXPOSURE,
                "max_open_positions": MAX_OPEN_POSITIONS,
            },
            "budgets": {
                "soft_calls_per_day": SOFT_CALLS_PER_DAY,
                "hard_calls_per_day": HARD_CALLS_PER_DAY,
                "soft_orders_per_day": SOFT_ORDERS_PER_DAY,
                "hard_orders_per_day": HARD_ORDERS_PER_DAY,
                "max_list_limit": MAX_LIST_LIMIT,
            },
            "fee_defaults_bps": {
                "taker": DEFAULT_TAKER_FEE_BPS,
                "maker": DEFAULT_MAKER_FEE_BPS,
            },
            "funding": "fixed at epoch open; no operation in this stem credits cash",
        },
        at=started,
    )
    _log.info("polymarket_paper: opened %s under %s", epoch_id, store_root(home))
    return epoch


# --- serialization helpers --------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """`Decimal` → its exact string, recursively. Keeps 0.001 ticks exact on disk."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dec(value: Any, default: Decimal) -> Decimal:
    parsed = _dec_or_none(value)
    return default if parsed is None else parsed


def _dec_or_none(value: Any) -> Decimal | None:
    """A **finite** `Decimal`, else ``None``. A non-finite value is not a number the fold can
    add up — and `Decimal("NaN")` parses, then raises on the first comparison."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except Exception:  # noqa: BLE001 - an unparseable number is simply absent
        return None
    return parsed if parsed.is_finite() else None
