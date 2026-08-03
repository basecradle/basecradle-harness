"""What happens when public data stops answering (issue #390): marks, settlement, force-resolve.

This file is the record of one live incident and the three defects behind it. A paper position
of 1,000 shares sat open for four days after its market had resolved against it, showing a
confident **+$772.95 unrealized gain** on a leg that was worth $0.0005, while the scorecard's
`resolved_n` stayed pinned at zero. Nothing errored. Nothing logged. The three causes were
independent, and each is pinned below:

- **The mark was inverted** because `/book`'s `last_trade_price` is *market*-scoped and was
  read as though it were the token's — so a leg with no bids took the complement's price.
- **Settlement could not run** because the resolution path went through Gamma, which *deletes*
  a market when it resolves, and because a tradability gate refused the book-less market a
  resolved market always is.
- **Nothing said so.** An unpriceable position went on presenting a number.

The fabricated market and the upstream double live in `test_polymarket.py`; the shapes here —
a one-sided book, a delisted market, a book that 404s — are the live ones, checked against
public Polymarket data on 2026-08-03.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from basecradle_harness._polymarket import ACTIONS
from basecradle_harness._polymarket_data import Book, Level
from basecradle_harness._polymarket_engine import main
from basecradle_harness._polymarket_ledger import (
    BRIER_SCORE,
    MARKET_BACK,
    MARKET_GONE,
    SETTLEMENT,
    current_epoch,
)
from tests.test_polymarket import (
    CONDITION_ID,
    MARKET_ID,
    NO_TOKEN,
    YES_TOKEN,
    book,
    buy,
    call,
    forecast,
    make_tool,
    upstream,
)

SOURCE = Path(__file__).parent.parent / "src" / "basecradle_harness"


def rows(tmp_path, kind=None):
    all_rows = current_epoch(tmp_path).rows()
    return [r for r in all_rows if kind is None or r["type"] == kind]


def levels(*pairs):
    return tuple(Level(price=Decimal(p), size=Decimal(s)) for p, s in pairs)


def held(tmp_path, fake, *, shares="10"):
    """Open the standard 10-share Yes position and hand back the tool."""
    tool = make_tool(tmp_path)
    forecast(tool)
    buy(tool, shares=shares)
    return tool


# --- D2: the mark ---------------------------------------------------------------------
#
# `/book` publishes a `last_trade_price` that reads like this token's and is the *market's*:
# both tokens of a binary market return the identical value. A deep-out-of-the-money token is
# exactly the token nobody bids on, so it was exactly the token that fell through to that
# fallback — and took the other leg's price.


def test_a_book_with_no_bids_marks_at_the_floor_not_the_last_trade():
    """The live signature: no bids, a $0.001 ask, and a market last trade of $0.999."""
    one_sided = Book(
        outcome="Yes",
        token_id=YES_TOKEN,
        bids=(),
        asks=levels(("0.001", "537959.54")),
        market_last_trade_price=Decimal("0.999"),
    )
    # A share that pays $1.00 or $0.00 cannot be bid below zero, so an empty bid side *is* a
    # floor of $0.00 — which is what the CLOB's own /midpoint returns for this exact book.
    assert one_sided.mid == Decimal("0.0005")
    assert one_sided.mid != one_sided.market_last_trade_price


def test_a_book_with_no_asks_marks_at_the_ceiling():
    """The mirror, and the same reasoning: nobody offering caps the ask at $1.00."""
    one_sided = Book(
        outcome="No",
        token_id=NO_TOKEN,
        bids=levels(("0.999", "664.47")),
        asks=(),
        market_last_trade_price=Decimal("0.999"),
    )
    assert one_sided.mid == Decimal("0.9995")


def test_a_book_empty_on_both_sides_has_no_mark():
    """Nothing to say, so it says nothing — never a stale print from a dead book."""
    empty = Book(
        outcome="Yes",
        token_id=YES_TOKEN,
        bids=(),
        asks=(),
        market_last_trade_price=Decimal("0.42"),
    )
    assert empty.mid is None


def test_a_two_sided_book_is_unchanged():
    """The healthy case is the midpoint it always was — the fix touches only a missing side."""
    both = Book(
        outcome="Yes",
        token_id=YES_TOKEN,
        bids=levels(("0.41", "500")),
        asks=levels(("0.43", "300")),
        market_last_trade_price=Decimal("0.999"),
    )
    assert both.mid == Decimal("0.42")


def test_the_mark_agrees_with_the_markets_own_mid_for_the_held_outcome(tmp_path):
    """The regression the incident asks for, at the tool surface (issue #390, D2).

    A position is opened at $0.43 and the market then moves against it to the live collapsed
    shape: the bid side empties and a $0.001 offer is all that is left, while `/book` goes on
    reporting the *market's* last trade of $0.999. `get_positions` must agree with what
    `get_market` publishes for that same outcome — and must not report a near-par mark on a
    leg the market has priced at almost nothing.
    """
    with upstream() as fake:
        tool = held(tmp_path, fake)
        fake.books[YES_TOKEN] = book(bids=[], asks=[("0.001", "537959.54")], last="0.999")

        market = call(tool, "get_market", market_id=MARKET_ID)
        positions = call(tool, "get_positions")

    published = next(b["mid"] for b in market["book"] if b["outcome"] == "Yes")
    position = positions["positions"][0]
    assert position["mtm_price"] == published == 0.0005
    assert position["unrealized_pnl"] < 0  # the leg lost, and the ledger says so
    # The inversion, stated as the thing that must never come back.
    assert position["mtm_price"] != 0.999


# --- D1: settlement when Gamma has deleted the market ----------------------------------
#
# Gamma dropping a resolved market is the ordinary lifecycle, not an edge case: of 400 resolved
# markets the CLOB still served with winner flags, 400 were gone from Gamma. The condition id
# the ledger has recorded since v1 is what still reaches the authority.


def test_a_market_gamma_has_dropped_still_settles_off_the_clob(tmp_path):
    with upstream() as fake:
        tool = held(tmp_path, fake)
        fake.resolve(winner="No")  # the position loses
        fake.delist()  # ...and Gamma forgets the market entirely, as it does
        body = call(tool, "get_pnl")

    assert body["realized_pnl"] == -4.3  # basis 10 x $0.43, paid out at $0.00
    settlements = rows(tmp_path, SETTLEMENT)
    assert len(settlements) == 1
    assert settlements[0]["payload"]["resolution_source"] == "clob_public_market_state"
    assert settlements[0]["payload"]["won"] is False
    assert not call(tool, "get_positions")["positions"]


def test_settling_a_delisted_market_scores_the_brier_and_moves_resolved_n(tmp_path):
    """`resolved_n` stuck at 0 was the headline symptom; this is the thing that unsticks it."""
    with upstream() as fake:
        tool = held(tmp_path, fake)
        fake.resolve(winner="Yes")
        fake.delist()
        card = call(tool, "get_scorecard")

    assert card["resolved_n"] == 1
    assert card["brier"] == 0.16  # (0.6 - 1)^2, against the forecast locked at position-open
    assert card["chain_verified"] is True


def test_an_observation_only_market_still_settles_after_gamma_drops_it(tmp_path):
    """A closed-out position leaves nothing but an unscored forecast — and it must still grade.

    The condition id survives on the `brier_obs` row, and `Observation` folds it for exactly
    this case. Without it, an agent that took a position and closed it before resolution would
    have its forecast silently dropped from the scorecard the moment Gamma delisted the market
    — the survivorship hole `markets_to_sweep` exists to close, reopened one layer down.
    """
    with upstream() as fake:
        tool = held(tmp_path, fake)
        call(
            tool,
            "place_order",
            market_id=MARKET_ID,
            outcome="Yes",
            side="sell",
            size_shares="10",
            order_type="market",
            client_order_id="coid-flat",
        )
        assert not call(tool, "get_positions")["positions"]
        fake.resolve(winner="Yes")
        fake.delist()
        card = call(tool, "get_scorecard")

    assert card["resolved_n"] == 1
    assert card["brier"] == 0.16
    assert len(rows(tmp_path, BRIER_SCORE)) == 1


def test_a_resolved_market_settles_even_though_its_order_book_is_off(tmp_path):
    """The second, independent blocker: a resolved market always reports no order book.

    The `enable_order_book` gate used to sit on the shared resolution path, so *every* market
    this instrument most needed to settle answered `not_implemented` and the sweep skipped it —
    with Gamma still listing the market and nothing wrong anywhere else.
    """
    with upstream() as fake:
        tool = held(tmp_path, fake)
        fake.resolve(winner="Yes")  # switches the book off and 404s /book, as the live CLOB does
        body = call(tool, "get_pnl")

    assert body["realized_pnl"] == 5.7  # 10 shares paid $1.00 against a $4.30 basis
    assert len(rows(tmp_path, SETTLEMENT)) == 1


def test_a_transient_outage_never_marks_a_market_gone(tmp_path):
    """A Polymarket outage must never read as "this market ceased to exist"."""
    with upstream() as fake:
        tool = held(tmp_path, fake)
        fake.fail_ids.add(MARKET_ID)  # Gamma 503s rather than 404s
        body = call(tool, "get_positions")

    assert not rows(tmp_path, MARKET_GONE)
    assert body["positions"][0]["priceable"] is True
    assert "unpriceable_markets" not in body


def test_a_market_gone_from_both_sources_is_flagged_once(tmp_path):
    with upstream() as fake:
        tool = held(tmp_path, fake)
        fake.delist()
        fake.clob.pop(CONDITION_ID)
        call(tool, "get_positions")
        call(tool, "get_positions")  # the flag is a transition, not a row an hour forever

    gone = rows(tmp_path, MARKET_GONE)
    assert len(gone) == 1
    assert gone[0]["payload"]["market_id"] == MARKET_ID


def test_a_market_that_comes_back_is_priceable_again(tmp_path):
    """Delisting is not always permanent, so the flag lifts by itself when the data returns."""
    with upstream() as fake:
        tool = held(tmp_path, fake)
        listed, state = fake.gamma[MARKET_ID], fake.clob[CONDITION_ID]
        fake.delist()
        fake.clob.pop(CONDITION_ID)
        assert call(tool, "get_positions")["positions"][0]["priceable"] is False

        fake.gamma[MARKET_ID], fake.clob[CONDITION_ID] = listed, state  # published again
        body = call(tool, "get_positions")

    assert len(rows(tmp_path, MARKET_BACK)) == 1
    position = body["positions"][0]
    assert position["priceable"] is True
    assert position["mtm_price"] == 0.42  # the live book, marked again
    assert "unpriceable_markets" not in body


def test_a_stale_gain_is_given_up_too_not_just_a_stale_loss(tmp_path):
    """Carrying at cost is not a clamp to zero — it drops a flattering stale mark as readily.

    The incident's own mark happened to be an invented *gain*, so it would be easy to "fix" this
    by flooring the value and call it done. The rule is the other thing: while a market is
    flagged, **no** mark of its reaches any computation, in either direction. Here the position
    is genuinely up $4.80 at its last mark and still reports no unrealized P&L once the market
    disappears — because nobody can evidence that gain any more, not because gains are suspect.
    """
    with upstream() as fake:
        tool = held(tmp_path, fake)  # Yes, 10 shares @ $0.43 = $4.30 basis
        fake.books[YES_TOKEN] = book(bids=[("0.90", "500")], asks=[("0.92", "500")])
        up = call(tool, "get_pnl")
        assert up["unrealized_pnl"] == 4.8  # marked 0.91, so $9.10 against a $4.30 basis
        assert up["equity_usd"] == 10004.8

        fake.delist()
        fake.clob.pop(CONDITION_ID)
        stranded = call(tool, "get_pnl")
        position = call(tool, "get_positions")["positions"][0]

    assert stranded["unrealized_pnl"] == 0.0
    assert stranded["equity_usd"] == 10000.0  # carried at cost: no gain, no loss
    assert position["last_known_mark"] == 0.91  # still shown, simply never computed from
    assert position["mtm_price"] is None


# --- D3: an unpriceable position says so -----------------------------------------------


def test_an_unpriceable_position_reports_no_mark_and_no_unrealized_pnl(tmp_path):
    """The incident's whole shape, end to end: a stale mark must never read as a live one."""
    with upstream() as fake:
        tool = held(tmp_path, fake)
        # The book collapses and the sweep catches one last, correct mark...
        fake.books[YES_TOKEN] = book(bids=[], asks=[("0.001", "537959.54")], last="0.999")
        assert call(tool, "get_positions")["positions"][0]["mtm_price"] == 0.0005
        # ...and then the market is deleted from both public sources.
        fake.delist()
        fake.clob.pop(CONDITION_ID)
        body = call(tool, "get_positions")
        pnl = call(tool, "get_pnl")

    position = body["positions"][0]
    assert position["priceable"] is False
    assert position["mtm_price"] is None
    assert position["mtm_value"] is None
    assert position["unrealized_pnl"] is None
    assert position["last_known_mark"] == 0.0005  # shown, never computed from
    assert "operator" in position["note"]

    # Carried at cost: equity claims neither the gain nor the loss nobody can evidence.
    assert pnl["unrealized_pnl"] == 0.0
    assert pnl["equity_usd"] == 10000.0
    assert body["unpriceable_markets"] == [MARKET_ID]
    assert pnl["unpriceable_markets"] == [MARKET_ID]


def test_the_stranded_block_clears_once_the_market_is_settled(tmp_path):
    """A `market_gone` row is history and stays in the log; a *stranded* market is not."""
    with upstream() as fake:
        tool = held(tmp_path, fake)
        fake.delist()
        fake.clob.pop(CONDITION_ID)
        assert call(tool, "get_pnl")["unpriceable_markets"] == [MARKET_ID]

    main(
        [
            "--home",
            str(tmp_path),
            "--force-resolve",
            MARKET_ID,
            "--winner",
            "No",
            "--evidence",
            "fomc-statement-2026-07-29",
            "--yes",
        ]
    )

    with upstream():
        body = call(make_tool(tmp_path), "get_pnl")
    assert "unpriceable_markets" not in body
    assert rows(tmp_path, MARKET_GONE)  # the log still remembers it happened


# --- the operator's force-resolve --------------------------------------------------------


def resolve_cli(tmp_path, *extra, market=MARKET_ID, winner="No", evidence="fomc-2026-07-29"):
    return main(
        [
            "--home",
            str(tmp_path),
            "--force-resolve",
            market,
            "--winner",
            winner,
            "--evidence",
            evidence,
            *extra,
        ]
    )


def test_force_resolve_settles_realizes_scores_and_counts(tmp_path, capsys):
    """The floor the incident asks for: settle at $1/$0, realize, score Brier, move resolved_n."""
    with upstream() as fake:
        tool = held(tmp_path, fake)
        fake.delist()
        fake.clob.pop(CONDITION_ID)
        call(tool, "get_positions")  # flags it unpriceable

    assert resolve_cli(tmp_path, "--yes", "--by", "capital") == 0

    with upstream():
        tool = make_tool(tmp_path)
        pnl = call(tool, "get_pnl")
        card = call(tool, "get_scorecard")
        positions = call(tool, "get_positions")

    assert pnl["realized_pnl"] == -4.3  # Yes was held; No won; the basis is gone
    assert not positions["positions"]
    assert card["resolved_n"] == 1
    assert card["brier"] == 0.36  # (0.6 - 0)^2 — the forecast was wrong, and it is scored so
    assert card["chain_verified"] is True


def test_force_resolve_records_who_decided_it_and_on_what_evidence(tmp_path):
    """An override has to be distinguishable from the machine's own work, forever."""
    with upstream() as fake:
        held(tmp_path, fake)
    resolve_cli(tmp_path, "--yes", "--by", "capital", evidence="fomc-statement-2026-07-29")

    settlement = rows(tmp_path, SETTLEMENT)[0]["payload"]
    assert settlement["resolution_source"] == "operator_force_resolve"
    assert settlement["resolved_by"] == "capital"
    assert settlement["resolution_evidence"] == "fomc-statement-2026-07-29"
    assert settlement["forced"] is True
    assert (
        rows(tmp_path, BRIER_SCORE)[0]["payload"]["resolution_source"] == "operator_force_resolve"
    )
    # Corrected by appending: the chain still verifies end to end.
    assert current_epoch(tmp_path).verify().ok is True


def test_force_resolve_previews_without_writing(tmp_path, capsys):
    """Irreversible in an append-only log, so the write is a second, deliberate act."""
    with upstream() as fake:
        held(tmp_path, fake)
    before = len(rows(tmp_path))

    assert resolve_cli(tmp_path) == 0
    out = capsys.readouterr().out

    assert len(rows(tmp_path)) == before  # nothing written
    assert "preview only" in out
    assert "LOST" in out and "-4.30" in out  # the operator sees the loss before committing


def test_a_winner_the_agent_never_traded_is_flagged_but_not_refused(tmp_path, capsys):
    """The live case: the position was *Yes* and the market resolved *No*.

    The ledger has never heard of the only correct answer, so a strict allow-list would refuse
    the one invocation this lever exists for. It warns instead — and the warning is the same
    one a typo earns, printed right beside the per-position WON/LOST lines that show what it
    would cost.
    """
    with upstream() as fake:
        held(tmp_path, fake)

    assert resolve_cli(tmp_path, "--yes", winner="No") == 0
    out = capsys.readouterr().out
    assert "not an outcome this epoch has traded" in out
    assert "['Yes']" in out
    assert rows(tmp_path, SETTLEMENT)[0]["payload"]["won"] is False


def test_a_misspelling_settles_the_holding_to_zero_and_says_so_first(tmp_path, capsys):
    """The one genuinely dangerous typo — a WON turned into a LOST — is visible before --yes."""
    with upstream() as fake:
        held(tmp_path, fake)

    assert resolve_cli(tmp_path, winner="Yess") == 0  # preview, no --yes
    out = capsys.readouterr().out
    assert "not an outcome this epoch has traded" in out
    assert "LOST" in out
    assert not rows(tmp_path, SETTLEMENT)  # and it wrote nothing


def test_force_resolve_requires_a_winner(tmp_path, capsys):
    with upstream() as fake:
        held(tmp_path, fake)

    assert resolve_cli(tmp_path, "--yes", winner="  ") == 2
    assert "--winner is required" in capsys.readouterr().out
    assert not rows(tmp_path, SETTLEMENT)


def test_force_resolve_requires_evidence(tmp_path, capsys):
    with upstream() as fake:
        held(tmp_path, fake)

    assert resolve_cli(tmp_path, "--yes", evidence="  ") == 2
    assert "--evidence is required" in capsys.readouterr().out
    assert not rows(tmp_path, SETTLEMENT)


def test_force_resolve_refuses_a_market_it_has_nothing_on(tmp_path, capsys):
    with upstream() as fake:
        held(tmp_path, fake)

    assert resolve_cli(tmp_path, "--yes", market="900999") == 2
    assert "nothing here to resolve" in capsys.readouterr().out


def test_force_resolve_refuses_a_market_already_settled(tmp_path, capsys):
    with upstream() as fake:
        tool = held(tmp_path, fake)
        fake.resolve(winner="Yes")
        call(tool, "get_pnl")  # settles it automatically

    assert resolve_cli(tmp_path, "--yes", winner="Yes") == 2
    assert "already been resolved" in capsys.readouterr().out
    assert len(rows(tmp_path, SETTLEMENT)) == 1  # not settled twice


def test_force_resolve_refuses_to_append_to_a_broken_chain(tmp_path, capsys):
    """The one situation where somebody most plausibly edited the ledger by hand first."""
    with upstream() as fake:
        held(tmp_path, fake)
    epoch = current_epoch(tmp_path)
    lines = epoch.path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["payload"]["p"] = "0.99"
    lines[1] = json.dumps(tampered, ensure_ascii=False, separators=(",", ":"))
    epoch.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert resolve_cli(tmp_path, "--yes") == 1
    assert "LEDGER CHAIN BROKEN" in capsys.readouterr().out


# --- the agent's surface did not grow ------------------------------------------------------


def test_force_resolve_is_not_reachable_from_the_agent_surface(tmp_path):
    """§2.5's line, mechanically: a settlement is the operator's, never the agent's.

    The instrument's whole integrity is that the agent cannot write its own scoreboard, and a
    resolution is the single most valuable thing it could write. `force_resolve` is therefore
    an operator entry point on the sweep binary and nothing else — not an action, not a
    parameter, and not something the model-facing module can even name.
    """
    tool = make_tool(tmp_path)
    assert "force_resolve" not in ACTIONS
    assert tuple(tool.parameters["properties"]["action"]["enum"]) == ACTIONS
    assert not hasattr(tool, "_op_force_resolve")

    schema = json.dumps(tool.parameters).casefold()
    for word in ("force", "resolve", "winner", "evidence", "settle"):
        assert word not in schema, word

    source = (SOURCE / "_polymarket.py").read_text(encoding="utf-8")
    for symbol in ("force_resolve", "preview_force_resolve", "market_outcomes"):
        assert symbol not in source, symbol
