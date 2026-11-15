"""What the telemetry contract says a line carries, written once, for the offline and live suites.

A live test used to restate by hand what an `llm` or `tool` line should hold, and nothing offline
ever ran that restatement (issue #904). When the contract moved a field, the offline suite and
`tests/test_telemetry_contract.py` moved with it in the same pull request, and the live copy did
not: 0.168.0 (#759) moved Vertex's location from ``endpoint=`` to ``residency=``, and
`test_google_live.py` kept asserting ``endpoint`` for 39 days, until its next live run (#901).

So each line shape a live test checks has **one** function here, and both sides call it: an adapter's
offline suite on the line it renders through respx, and the live test on the vendor's real line. A
contract change that moves a field has to edit that function or the offline suite goes red in the
pull request that moves it, and the live test inherits the change untouched.

What belongs here is what `docs/telemetry.md` says about a line: which fields it carries, which it
never carries, and the relations between them. The per-outcome rule is read off the page's own
Outcomes table, never re-typed. What only a live call can say stays in the live file: a cost equal to
the rate table's arithmetic over the usage the vendor reported, an id in the vendor's own format, an
endpoint that is a member of the model's live routing pool.

Every function returns the line's fields, so a test reads its values from the same parse.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tests.conftest import plain

DOC = Path(__file__).resolve().parent.parent / "docs" / "telemetry.md"

# --- the grammar ---------------------------------------------------------------------------------

#: A strict structured line: a head of words, then one or more ``key=value`` fields. Anything else
#: is a prose line, which carries no fields (`docs/telemetry.md` → Prose Lines).
_KV = r'[a-z_][a-z0-9_]*=(?:"[^"]*"|[^\s"]+)'
_LINE = re.compile(
    rf"^(?:PROBE )?(?P<head>[A-Za-z][A-Za-z_-]*(?: [A-Za-z][A-Za-z_-]*)*?)(?P<fields>(?: {_KV})+)$"
)
_PAIR = re.compile(rf" ({_KV})")


@dataclass
class Line:
    head: str
    fields: dict[str, str]
    message: str


def parse_line(message: str) -> Line | None:
    """`message` read as the grammar defines it, or ``None`` for a prose line. Exact: no color is
    stripped, so a caller reading a colored record strips it first (`plain`)."""
    match = _LINE.match(message)
    if match is None:
        return None
    pairs = (pair.split("=", 1) for pair in _PAIR.findall(match["fields"]))
    return Line(match["head"], {key: value.strip('"') for key, value in pairs}, message)


def fields_of(message: str) -> dict[str, str]:
    """A structured line's fields, its color stripped: the one reader a test takes values from."""
    line = parse_line(plain(message))
    assert line is not None, f"not a structured line: {message}"
    return line.fields


def _parsed(message: str, head: str) -> Line:
    line = parse_line(plain(message))
    assert line is not None and line.head == head, f"not a structured `{head}` line: {message}"
    return line


# --- the `llm` outcomes, off the page --------------------------------------------------------------

#: The fields every `llm` line carries, whatever its outcome (`docs/telemetry.md` → Outcomes).
LLM_ALWAYS = frozenset({"provider", "purpose", "model", "outcome"})

#: The outcomes of an attempt the vendor answered and billed (`docs/telemetry.md` → `llm`).
BILLED = frozenset({"ok", "refused", "fallback"})


@functools.cache
def outcomes() -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """The page's Outcomes table: each outcome's required and never-carried fields."""
    table = DOC.read_text(encoding="utf-8").split("\n#### Outcomes\n", 1)[1].split("\n### ", 1)[0]
    found: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for row in table.splitlines():
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) == 4 and cells[0].startswith("`"):
            names = [frozenset(re.findall(r"`([a-z_]+)`", cell)) for cell in cells]
            (outcome,) = names[0]
            found[outcome] = (names[2], names[3])
    return found


def _llm(
    message: str, *, provider: str, model: str | None, purpose: str, kind: str | None, outcome: str
) -> dict[str, str]:
    """Every rule the page states for an `llm` line of this outcome, purpose and kind."""
    fields = _parsed(message, "llm").fields
    required, never = outcomes()[outcome]
    assert LLM_ALWAYS | required <= set(fields), message
    assert not never & set(fields), message
    assert fields["outcome"] == outcome, message
    assert fields["provider"] == provider, message
    assert fields["purpose"] == purpose, message
    assert fields.get("kind") == kind, message  # absent on `purpose=main` (`llm` → `kind`)
    if model is not None:
        assert fields["model"] == model, message
    if outcome in BILLED:
        # A billed call these shapes describe stated its usage, so its counts are on the line, and
        # each part the page names is a part of the count it belongs to (`llm` → `tokens_*`).
        assert {"tokens_in", "tokens_out", "tokens_total"} <= set(fields), message
        tokens_in, tokens_out = int(fields["tokens_in"]), int(fields["tokens_out"])
        assert tokens_in > 0 and tokens_out > 0, message
        assert int(fields["tokens_total"]) == tokens_in + tokens_out, message
        read = int(fields.get("cached_tokens", 0))
        written = int(fields.get("cache_write_tokens", 0))
        assert read + written <= tokens_in, message
        assert int(fields.get("tokens_reasoning", 0)) <= tokens_out, message
    return fields


#: How a line's dollars were reached (`docs/telemetry.md` → `cost`, `cost_basis`, `cost_gap`):
#: ``computed`` by the harness from a table or a live price list, ``stated`` by the vendor, or
#: ``None`` for work that carries no price on this line.
Cost = Literal["computed", "stated"] | None


def _cost(fields: Mapping[str, str], cost: Cost, message: str) -> None:
    if cost is None:
        assert "cost" not in fields and "cost_basis" not in fields, message
    else:
        assert "cost" in fields, message
        assert fields.get("cost_basis") == ("computed" if cost == "computed" else None), message
    assert "cost_gap" not in fields, message  # a priced call names no gap (#878, #880)


class _Reported:
    """`REPORTED`: the field is on the line, and the vendor chose its value."""

    def __repr__(self) -> str:
        return "REPORTED"


#: A ``residency`` the vendor reports rather than the request pinning (Anthropic's
#: ``inference_geo``): on the line, with a value the vendor chose.
REPORTED: Any = _Reported()


# --- the shapes the live tests check ---------------------------------------------------------------


def assert_direct_call(
    message: str,
    *,
    provider: str,
    model: str,
    cost: Cost,
    residency: str | None = None,
    purpose: str = "main",
    kind: str | None = None,
) -> dict[str, str]:
    """A billed `ok` call made **direct** to the vendor that bills it.

    The vendor *is* the endpoint, so no ``endpoint`` (`llm` → `endpoint`). ``residency`` is where
    the vendor ran it: the location asked for, `REPORTED` when the vendor names its own, ``None``
    for a vendor with no notion of location, which carries none.
    """
    fields = _llm(message, provider=provider, model=model, purpose=purpose, kind=kind, outcome="ok")
    assert "endpoint" not in fields, message
    if residency is REPORTED:
        assert fields.get("residency"), message
    else:
        assert fields.get("residency") == residency, message
    _cost(fields, cost, message)
    return fields


def assert_routed_call(
    message: str,
    *,
    provider: str,
    model: str,
    cost: Cost = "stated",
    purpose: str = "main",
    kind: str | None = None,
) -> dict[str, str]:
    """A billed `ok` call a **router** dispatched: ``endpoint`` names the upstream that served it
    (`llm` → `endpoint`), and the router states the cost (`llm` → `cost`)."""
    fields = _llm(message, provider=provider, model=model, purpose=purpose, kind=kind, outcome="ok")
    assert fields.get("endpoint"), message
    assert "residency" not in fields, message
    _cost(fields, cost, message)
    return fields


def assert_rerank_call(
    message: str, *, provider: str, model: str, pool: int, picked: int
) -> dict[str, str]:
    """The MemPalace reranker's billed `ok` call, through its router: `purpose=memory kind=rerank`
    with the recall's ``surface``, the candidates it was shown and the ones it picked."""
    fields = assert_routed_call(
        message, provider=provider, model=model, purpose="memory", kind="rerank"
    )
    assert fields.get("surface") in {"turn0", "tool"}, message
    assert {"pool", "picked"} <= set(fields), message
    assert (int(fields["pool"]), int(fields["picked"])) == (pool, picked), message
    return fields


def assert_failed_call(
    message: str,
    *,
    provider: str,
    purpose: str,
    kind: str | None = None,
    model: str | None = None,
    reason: str | None = None,
) -> dict[str, str]:
    """A call that ended with no usable answer (`outcome=failed`): a decision line, carrying its
    ``reason`` and nothing the vendor billed. ``reason=None`` leaves its value to the caller."""
    fields = _llm(
        message, provider=provider, model=model, purpose=purpose, kind=kind, outcome="failed"
    )
    if reason is not None:
        assert fields["reason"] == reason, message
    return fields


def assert_server_tool_line(
    message: str,
    *,
    name: str,
    provider: str,
    model: str,
    joins: str | Collection[str],
    cost: Cost,
    kind: str | None = None,
) -> dict[str, str]:
    """The `tool` line for a tool the vendor ran **inside** a model call (`executor=server`).

    Its uses are always counted, it carries no ``duration`` (the call's `llm` line timed the work),
    and its ``generation_id`` joins it to that call: ``joins`` is the id, or the ids, of the `llm`
    lines it may join (`tool` → `count`, `duration`, `generation_id`). A use the vendor reports as
    failed makes the line ``outcome=error`` with the vendor's code as its ``reason``, and is not
    billed, so ``cost`` binds an `ok` line in full and an `error` line only in what it carries.
    """
    fields = _parsed(message, "tool").fields
    assert (fields.get("name"), fields.get("executor")) == (name, "server"), message
    assert (fields.get("provider"), fields.get("model")) == (provider, model), message
    assert fields.get("kind") == kind, message
    assert int(fields.get("count", 0)) >= 1, message
    assert "duration" not in fields, message
    assert fields.get("generation_id") in ({joins} if isinstance(joins, str) else set(joins)), (
        message
    )
    assert fields.get("outcome") in {"ok", "error"}, message
    assert ("reason" in fields) == (fields["outcome"] == "error"), message  # Tool Outcomes
    if fields["outcome"] == "ok" or "cost" in fields:
        _cost(fields, cost, message)
    else:
        assert "cost_basis" not in fields and "cost_gap" not in fields, message
    return fields


def assert_priced_fee(fields: Mapping[str, Any], *, provider: str, kind: str) -> Mapping[str, Any]:
    """A fee a harness tool's vendor work recorded on its meter (`_observability.tool_meter`), as
    the engine writes it onto the run's `tool` line: computed, and naming no gap (`tool` → `cost`,
    `cost_basis`, `cost_gap`; issue #880)."""
    from basecradle_harness._observability import ComputedCost

    assert (fields.get("provider"), fields.get("kind")) == (provider, kind), fields
    assert isinstance(fields.get("cost"), ComputedCost), fields
    assert fields.get("cost_gap") is None, fields
    if fields.get("count") is not None:
        assert fields["count"] >= 1, fields
    return fields
