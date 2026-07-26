"""Public Polymarket market data — Gamma + CLOB, read-only, **GET-only** (issue #347).

The data half of the `polymarket_paper` instrument, and the place its hardest safety
clause lives. NORMATIVE §2.1 says the agent never passes a URL and never receives a
generic HTTP primitive through this stem, so the rule is enforced by *construction*
rather than by validation: this module owns two host constants, exposes exactly one
request method (`_get`), and every operation builds its own path and query from typed
arguments. There is no code path from a model-supplied string to a URL — not a base,
not a path, not a host override — so "the agent cannot fetch an arbitrary page" is a
property of the call graph, not a filter somebody has to keep correct.

**Only GET, ever — and that is load-bearing, so do not add a write verb here.** The client
has no `post`, no `put`, no session, no cookie jar and sends no `Authorization` header. That
is what makes the §2.5 non-goal "authenticated Polymarket/CLOB trading APIs" mechanically
true rather than a promise: an authenticated trade is a **signed POST**, so the forbidden
thing is not merely refused, it is *unbuildable from this file*. Adding a `_post` — even for
an innocent-looking public batch endpoint like the CLOB's `/midpoints` — quietly demotes that
property from "cannot" to "does not", and the next person to need a signature finds the
plumbing already there. If a batch read is ever worth it, add it as another GET or not at
all. `test_polymarket.py` asserts both halves against a mocked transport: every request a
GET, every host one of the two constants.

**What comes back is structured fields, never a document.** Each endpoint is parsed into
a frozen dataclass here; no raw HTML, no screenshots, no response body reaches the model.
The three sources, and why each is used:

- **Gamma `/markets`** (+ `/public-search`, `/tags/slug/{slug}`) — discovery and metadata:
  the question, slug, outcomes, the CLOB token ids, 24h volume, liquidity, end date, tags.
- **CLOB `/markets/{condition_id}`** — the *authoritative market state*: whether it is
  accepting orders, its tick and minimum size, its **published fee rates**, and — the one
  that matters most — each token's `winner` flag, which is how a resolved market is
  recognized. §2.4 requires resolution to come from public market state and never from an
  agent claim; this field is that state.
- **CLOB `/book`** — the order book the deterministic fill model walks, plus the last trade
  price the MTM mark falls back to.

**Money is `Decimal`, from the string the API sent.** Polymarket returns prices and sizes
as strings; parsing them straight into `Decimal` keeps a 0.001 tick exact through a book
walk, where a float would leave $0.30000000000000004 in a ledger row that is never allowed
to be corrected in place.

**Freshness is bounded by a small TTL cache**, not by hope. A `get_positions` on twenty
open positions would otherwise re-ask the CLOB for twenty marks every call; within
`CACHE_TTL` seconds the same read is served from memory. The cache is per-instance and
per-wake — it never persists, so it can never make a *ledger* row stale.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

_log = logging.getLogger("basecradle_harness")

#: Gamma — markets, events, metadata, search, tags. A **constant**: §2.1 forbids an
#: agent-supplied URL, so there is deliberately no way to point this elsewhere except the
#: constructor argument a test double uses.
GAMMA_BASE = "https://gamma-api.polymarket.com"

#: The public CLOB — order book, midpoint, market state, tick size. Also a constant, same
#: reason.
CLOB_BASE = "https://clob.polymarket.com"

#: Per-request timeout (seconds). A tool call must never hang a wake on a third party.
DEFAULT_TIMEOUT = 15.0

#: How long a fetched book / market state / midpoint stays reusable within one process.
#: Bounds upstream traffic on the read-heavy ops (a 20-position `get_positions`) without
#: ever touching what the ledger already recorded.
CACHE_TTL = 60.0

#: The hosts this client is allowed to reach. Not a filter the client consults — a
#: statement the tests assert against every request it made (see `test_polymarket.py`).
ALLOWED_HOSTS = frozenset({"gamma-api.polymarket.com", "clob.polymarket.com"})

#: How many book levels per side `get_market` shows the model (NORMATIVE §2.3).
BOOK_DEPTH = 15

#: What an id or slug may contain before it is spliced into a **path**.
#:
#: §2.1's rule is that no agent-supplied string becomes a request destination, and a path
#: segment is part of the destination: `market_id="../tags/slug/x"` stays on the same host but
#: it is no longer the request this code thinks it is making. The two host constants already
#: bound the blast radius to Polymarket; this bounds it to the *endpoint* as well, so the
#: property holds for the path and not merely for the origin. Anything outside the class is a
#: `MarketNotFound` — which is the truth: no market has that id.
_PATH_SAFE = re.compile(r"^[A-Za-z0-9_-]+$")


class PolymarketUnavailable(Exception):
    """Upstream did not answer, or answered with something this client will not guess at.

    Deliberately distinct from "no such market": §2.6's code list describes verdicts on the
    *agent's request*, and reporting an outage as `not_found` would have the agent reason
    from "this market does not exist" when the truth is "Polymarket is down." The tool maps
    this to `upstream_unavailable` (the one code added beyond the contract's list, declared
    on issue #347 before the build).
    """


class MarketNotFound(Exception):
    """No market on Polymarket with that id — a real answer, not an outage."""


# --- parsed, structured market data -------------------------------------------


@dataclass(frozen=True)
class Level:
    """One price level of an order book: a price in dollars and its size in shares."""

    price: Decimal
    size: Decimal


@dataclass(frozen=True)
class Book:
    """One outcome's order book, sorted best-first, as of `fetched_at`.

    Polymarket returns asks worst-first and bids best-last; both sides are re-sorted here
    so the fill model can walk them FIFO by price without re-deriving the convention. A
    side may be empty — an untraded outcome has no bids — which is why `best_bid` /
    `best_ask` / `mid` are all optional and every caller handles `None`.
    """

    outcome: str
    token_id: str
    bids: tuple[Level, ...]  # best (highest price) first
    asks: tuple[Level, ...]  # best (lowest price) first
    last_trade_price: Decimal | None = None

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        """The MTM mark per §2.4: mid of the two sides, else last trade, else nothing."""
        bid, ask = self.best_bid, self.best_ask
        if bid is not None and ask is not None:
            return (bid + ask) / Decimal(2)
        return self.last_trade_price


@dataclass(frozen=True)
class MarketSummary:
    """One row of `list_markets` — the `MarketSummary` shape NORMATIVE §2.3 specifies.

    `mid_prices` here are Gamma's published `outcomePrices`, not a book walk: a 25-row
    listing that fetched a book per outcome would be 50 upstream reads for one tool call.
    `get_market` is the operation that returns book-derived prices, and it says so.
    """

    market_id: str
    condition_id: str
    question: str
    slug: str
    outcomes: tuple[str, ...]
    token_ids: tuple[str, ...]
    mid_prices: tuple[Decimal | None, ...]
    volume_24h: Decimal
    liquidity: Decimal
    end_date: str | None
    accepting_orders: bool
    tick_size: Decimal
    min_order_size: Decimal
    tags: tuple[str, ...]
    event_id: str | None = None
    event_slug: str | None = None

    def outcome_token(self, outcome: str) -> str | None:
        """The CLOB token id for `outcome`, matched case-insensitively, or ``None``."""
        wanted = outcome.strip().casefold()
        for name, token in zip(self.outcomes, self.token_ids):
            if name.casefold() == wanted:
                return token
        return None


@dataclass(frozen=True)
class MarketState:
    """The CLOB's own view of a market: tradability, fees, and **resolution**.

    This is the §2.4 "public market state" that settlement reads. A market is resolved when
    the CLOB reports it closed *and* names a winning token; `winners` carries that verdict
    per token id. An agent has no way to assert it — no operation in this stem accepts a
    resolution, a price, or a fee.

    `maker_fee_bps` / `taker_fee_bps` are ``None`` only when the CLOB did not publish them,
    which is the sole case where the fill model falls back to its default rate and tags the
    row `fee_source=default`. A published **zero** is a published rate and is recorded as
    one — §2.4 forbids silently zeroing a fee, not honestly reading a zero.
    """

    condition_id: str
    accepting_orders: bool
    closed: bool
    enable_order_book: bool
    tick_size: Decimal
    min_order_size: Decimal
    maker_fee_bps: Decimal | None
    taker_fee_bps: Decimal | None
    winners: Mapping[str, bool]
    tokens: Mapping[str, str]  # token_id -> outcome name
    tags: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        """Whether public state says this market has settled to an outcome."""
        return self.closed and any(self.winners.values())

    def winning_outcomes(self) -> tuple[str, ...]:
        """The outcome names public state marks as winners (usually exactly one)."""
        return tuple(
            self.tokens[token]
            for token, won in self.winners.items()
            if won and token in self.tokens
        )


# --- the client ---------------------------------------------------------------


class PolymarketData:
    """Read-only public market data. One request verb, two hosts, typed results.

    Args:
        gamma_base: Gamma's root. Exists so a test can point at a fabricated host; there is
            no agent-reachable path to it and no env override, by design (§2.1).
        clob_base: The public CLOB's root, same reasoning.
        timeout: Per-request timeout in seconds.
        cache_ttl: Seconds a fetched book / state / midpoint stays reusable in-process.
        now: Injectable clock, for deterministic cache tests.
    """

    def __init__(
        self,
        *,
        gamma_base: str = GAMMA_BASE,
        clob_base: str = CLOB_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        cache_ttl: float = CACHE_TTL,
        now=None,
    ) -> None:
        self._gamma = gamma_base.rstrip("/")
        self._clob = clob_base.rstrip("/")
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._now = now or time.monotonic
        self._cache: dict[tuple[str, str], tuple[float, Any]] = {}

    # --- the one request method ------------------------------------------------

    def _get(self, base: str, path: str, params: dict[str, Any] | None = None) -> Any:
        """The **only** way this module makes a request: a GET, to one of two known hosts.

        `base` is always one of the two constants and `path` is always a literal assembled
        here, so no caller — and certainly no model — can steer the destination. Any
        transport failure or non-JSON answer becomes `PolymarketUnavailable`; a 404 is left
        to the caller, which knows whether "absent" means not-found or unavailable.
        """
        url = f"{base}{path}"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url, params=params)
        except httpx.RequestError as exc:
            raise PolymarketUnavailable(f"could not reach {_host(base)} ({exc})") from exc
        if response.status_code == 404:
            raise MarketNotFound(f"{_host(base)} has no record at {path}")
        if response.status_code >= 400:
            raise PolymarketUnavailable(f"{_host(base)} returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise PolymarketUnavailable(f"{_host(base)} returned a non-JSON body") from exc

    def _cached(self, kind: str, key: str, fetch) -> Any:
        """`fetch()`, memoized for `cache_ttl` seconds under ``(kind, key)``."""
        now = self._now()
        hit = self._cache.get((kind, key))
        if hit is not None and now - hit[0] < self._cache_ttl:
            return hit[1]
        value = fetch()
        self._cache[(kind, key)] = (now, value)
        return value

    # --- discovery -------------------------------------------------------------

    def list_markets(
        self,
        *,
        query: str | None = None,
        active: bool = True,
        closed: bool = False,
        tag: str | None = None,
        limit: int = 25,
        cursor: str | None = None,
    ) -> tuple[list[MarketSummary], str | None]:
        """Markets matching the filters, plus the cursor for the next page.

        Two Gamma endpoints, picked by whether there is a `query`: a free-text search goes
        through `/public-search` (whose results are events carrying their markets), and an
        unfiltered or tag-filtered listing goes through `/markets`, ordered by 24h volume so
        the first page is the liquid end of the market rather than an arbitrary slice.

        The cursor is an **opaque decimal string** the caller round-trips: an offset for
        `/markets`, a page number for `/public-search`. Keeping it opaque means the paging
        convention can change without the agent's stored cursor becoming a lie.
        """
        offset = _cursor_offset(cursor)
        if query:
            return self._search_markets(query, offset, limit, active, closed)
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "active": _flag(active),
            "closed": _flag(closed),
            "order": "volume24hr",
            "ascending": "false",
            "include_tag": "true",
        }
        if tag:
            tag_id = self._tag_id(tag)
            if tag_id is None:
                return [], None
            params["tag_id"] = tag_id
        rows = self._get(self._gamma, "/markets", params)
        markets = [m for m in (_summary(row) for row in _as_list(rows)) if m is not None]
        next_cursor = str(offset + limit) if len(_as_list(rows)) >= limit else None
        return markets, next_cursor

    def _search_markets(
        self, query: str, offset: int, limit: int, active: bool, closed: bool
    ) -> tuple[list[MarketSummary], str | None]:
        """Free-text search through Gamma's public search, flattened to markets.

        `/public-search` answers in *events*, each carrying its markets, so a page of events
        yields a variable number of markets; the result is flattened, filtered to the
        caller's active/closed intent (search does not filter as precisely as `/markets`),
        and truncated to `limit`. The cursor counts pages, one per `limit`-sized request.
        """
        page = offset // max(limit, 1) + 1
        body = self._get(
            self._gamma,
            "/public-search",
            {"q": query, "limit_per_type": limit, "page": page, "events_status": "active"},
        )
        events = body.get("events", []) if isinstance(body, Mapping) else []
        markets: list[MarketSummary] = []
        for event in events:
            if not isinstance(event, Mapping):
                continue
            for row in event.get("markets") or []:
                summary = _summary(row, event=event)
                if summary is None:
                    continue
                if closed is False and _truthy(row.get("closed")):
                    continue
                if active is True and row.get("active") is False:
                    continue
                markets.append(summary)
        has_more = bool(_dig(body, "pagination", "hasMore"))
        next_cursor = str(offset + limit) if has_more else None
        return markets[:limit], next_cursor

    def _tag_id(self, tag: str) -> str | None:
        """Gamma's numeric id for a tag slug (``"politics"`` → ``"2"``), or ``None``.

        A tag the agent named that Gamma does not know is an empty result, not an error —
        the listing simply has nothing in it, which is the truthful answer.
        """
        slug = tag.strip().strip("/").casefold().replace(" ", "-")
        if not slug or not _PATH_SAFE.match(slug):
            return None  # not a slug, so no tag — an empty listing, which is the truth
        try:
            body = self._get(self._gamma, f"/tags/slug/{slug}")
        except MarketNotFound:
            return None
        tag_id = body.get("id") if isinstance(body, Mapping) else None
        return str(tag_id) if tag_id is not None else None

    def market_summary(self, market_id: str) -> MarketSummary:
        """One market's Gamma metadata by its Gamma id. Raises `MarketNotFound`.

        **Fetched through the collection (`/markets?id=`), not `/markets/{id}`**, and that is
        not a stylistic choice: the single-market endpoint omits `events` entirely, so a market
        reached by id came back with no event — which silently zeroed the scorecard's
        `distinct_event_clusters`, the one metric that separates real forecasting range from
        one lucky theme. It was invisible in tests, because a fabricated market carries whatever
        the fixture says. The collection form returns the same row *with* its event and tags.

        The id is agent-supplied, so it is checked against `_PATH_SAFE` before use. It rides as
        a query parameter here rather than a path segment, which makes the check belt-and-braces
        rather than load-bearing — and a junk id gets an honest `not_found` either way.
        """
        if not _PATH_SAFE.match(str(market_id)):
            raise MarketNotFound(f"{market_id!r} is not a Polymarket market id")

        def fetch() -> MarketSummary:
            body = self._get(self._gamma, "/markets", {"id": str(market_id), "include_tag": "true"})
            row = body[0] if isinstance(body, list) and body else body
            summary = _summary(row) if isinstance(row, Mapping) else None
            if summary is None:
                raise MarketNotFound(f"Gamma has no market with id {market_id!r}")
            return summary

        return self._cached("summary", str(market_id), fetch)

    # --- tradability, fees, resolution -----------------------------------------

    def market_state(self, condition_id: str) -> MarketState:
        """The CLOB's state for a market: accepting/closed, tick, min size, fees, winners.

        The condition id comes from Gamma rather than from the agent, but it is still spliced
        into a path, so it gets the same check — a value is validated where it is *used*, not
        where it is believed to have come from.
        """
        if not _PATH_SAFE.match(str(condition_id)):
            raise MarketNotFound(f"{condition_id!r} is not a Polymarket condition id")

        def fetch() -> MarketState:
            body = self._get(self._clob, f"/markets/{condition_id}")
            if not isinstance(body, Mapping) or not body.get("condition_id"):
                raise MarketNotFound(f"the CLOB has no market {condition_id!r}")
            winners: dict[str, bool] = {}
            tokens: dict[str, str] = {}
            for token in body.get("tokens") or []:
                if not isinstance(token, Mapping):
                    continue
                token_id = str(token.get("token_id") or "")
                if not token_id:
                    continue
                winners[token_id] = bool(token.get("winner"))
                tokens[token_id] = str(token.get("outcome") or "")
            return MarketState(
                condition_id=str(body["condition_id"]),
                accepting_orders=bool(body.get("accepting_orders")),
                closed=bool(body.get("closed")),
                enable_order_book=bool(body.get("enable_order_book", True)),
                tick_size=_dec(body.get("minimum_tick_size"), Decimal("0.001")),
                min_order_size=_dec(body.get("minimum_order_size"), Decimal("1")),
                maker_fee_bps=_dec_or_none(body.get("maker_base_fee")),
                taker_fee_bps=_dec_or_none(body.get("taker_base_fee")),
                winners=winners,
                tokens=tokens,
                tags=tuple(str(t) for t in (body.get("tags") or []) if t),
            )

        return self._cached("state", str(condition_id), fetch)

    # --- the book ---------------------------------------------------------------

    def book(self, token_id: str, outcome: str = "") -> Book:
        """One outcome's order book, best-first on both sides."""

        def fetch() -> Book:
            body = self._get(self._clob, "/book", {"token_id": token_id})
            if not isinstance(body, Mapping):
                raise PolymarketUnavailable("the CLOB returned a non-object book")
            return Book(
                outcome=outcome,
                token_id=str(token_id),
                bids=_levels(body.get("bids"), reverse=True),
                asks=_levels(body.get("asks"), reverse=False),
                last_trade_price=_dec_or_none(body.get("last_trade_price")),
            )

        cached: Book = self._cached("book", str(token_id), fetch)
        # The cache is keyed by token, not by the label a caller happened to pass; re-stamp
        # the outcome name so a book fetched by the sweep reads correctly in a tool result.
        return (
            cached
            if cached.outcome == outcome
            else Book(
                outcome=outcome or cached.outcome,
                token_id=cached.token_id,
                bids=cached.bids,
                asks=cached.asks,
                last_trade_price=cached.last_trade_price,
            )
        )

    def invalidate(self, *token_ids: str) -> None:
        """Drop cached books/midpoints for `token_ids` (or everything, with no arguments).

        The fill model calls this after it walks a book: the very next read in the same wake
        should not be served a snapshot the simulation has already consumed against.
        """
        if not token_ids:
            self._cache.clear()
            return
        for token in token_ids:
            self._cache.pop(("book", str(token)), None)
            self._cache.pop(("midpoint", str(token)), None)


# --- parsing helpers -----------------------------------------------------------


def _summary(row: Any, *, event: Mapping[str, Any] | None = None) -> MarketSummary | None:
    """A `MarketSummary` from one Gamma market object, or ``None`` if it is unusable.

    Gamma sends `outcomes`, `outcomePrices` and `clobTokenIds` as *JSON strings*, not lists,
    which is the one genuinely surprising thing about the shape. A market with no CLOB token
    ids cannot be traded even on paper, so it is dropped from a listing rather than shown as
    something the agent could act on and then be refused.
    """
    if not isinstance(row, Mapping):
        return None
    outcomes = tuple(str(o) for o in _json_list(row.get("outcomes")))
    tokens = tuple(str(t) for t in _json_list(row.get("clobTokenIds")))
    market_id = str(row.get("id") or "")
    condition_id = str(row.get("conditionId") or "")
    if not market_id or not outcomes or not tokens or len(tokens) != len(outcomes):
        return None
    prices = [_dec_or_none(p) for p in _json_list(row.get("outcomePrices"))]
    prices += [None] * (len(outcomes) - len(prices))
    events = row.get("events") or ([event] if event else [])
    first_event = events[0] if events and isinstance(events[0], Mapping) else {}
    return MarketSummary(
        market_id=market_id,
        condition_id=condition_id,
        question=str(row.get("question") or ""),
        slug=str(row.get("slug") or ""),
        outcomes=outcomes,
        token_ids=tokens,
        mid_prices=tuple(prices[: len(outcomes)]),
        volume_24h=_dec(row.get("volume24hr"), Decimal(0)),
        liquidity=_dec(row.get("liquidity"), Decimal(0)),
        end_date=str(row["endDate"]) if row.get("endDate") else None,
        accepting_orders=_truthy(row.get("acceptingOrders")),
        tick_size=_dec(row.get("orderPriceMinTickSize"), Decimal("0.001")),
        min_order_size=_dec(row.get("orderMinSize"), Decimal("1")),
        tags=_tags(row),
        event_id=str(first_event.get("id")) if first_event.get("id") else None,
        event_slug=str(first_event.get("slug")) if first_event.get("slug") else None,
    )


def _tags(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Tag slugs off a Gamma market (`include_tag=true` returns objects, not strings)."""
    out: list[str] = []
    for tag in row.get("tags") or []:
        if isinstance(tag, Mapping):
            slug = tag.get("slug") or tag.get("label")
            if slug:
                out.append(str(slug))
        elif tag:
            out.append(str(tag))
    return tuple(out)


def _levels(raw: Any, *, reverse: bool) -> tuple[Level, ...]:
    """Parse and sort one side of a book; `reverse=True` for bids (highest price first)."""
    levels: list[Level] = []
    for entry in raw or []:
        if not isinstance(entry, Mapping):
            continue
        price = _dec_or_none(entry.get("price"))
        size = _dec_or_none(entry.get("size"))
        if price is None or size is None or size <= 0:
            continue
        levels.append(Level(price=price, size=size))
    levels.sort(key=lambda level: level.price, reverse=reverse)
    return tuple(levels)


def _json_list(value: Any) -> list[Any]:
    """Gamma's ``'["Yes", "No"]'`` string (or an actual list) as a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _dec(value: Any, default: Decimal) -> Decimal:
    """`value` as a `Decimal`, falling back to `default` for anything unparseable."""
    parsed = _dec_or_none(value)
    return default if parsed is None else parsed


def _dec_or_none(value: Any) -> Decimal | None:
    """`value` as a **finite** `Decimal`, or ``None`` — "the API did not publish this".

    `Decimal("NaN")` parses happily and then raises `InvalidOperation` the first time anything
    compares it, which would surface as an unrelated error several frames away. A non-finite
    number is not a price; treat it as absent.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _truthy(value: Any) -> bool:
    """Gamma sends booleans as booleans and, occasionally, as strings."""
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes"}
    return bool(value)


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _cursor_offset(cursor: str | None) -> int:
    """The offset a cursor encodes. A malformed cursor reads as the first page, not an error."""
    if not cursor:
        return 0
    try:
        return max(0, int(str(cursor).strip()))
    except ValueError:
        _log.debug("polymarket_paper: ignoring unparseable cursor %r", cursor)
        return 0


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dig(body: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(body, Mapping):
            return None
        body = body.get(key)
    return body


def _host(base: str) -> str:
    return httpx.URL(base).host
