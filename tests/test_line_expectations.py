"""The line expectations stay shared: each one is run offline and read live (issue #904).

`tests/_line_expectations.py` exists so a live test's idea of a line is the offline suite's idea,
and so a contract change that breaks it breaks CI in the pull request that makes it. That holds
only while every expectation is called from both sides. One that only a live file calls is the
defect of issue #901 again, a restatement nothing offline runs; one that no live file calls is a
shape built for a test that does not exist.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from tests import _line_expectations

TESTS = Path(__file__).resolve().parent

#: Every line shape the module holds: its public ``assert_*`` functions.
SHAPES = sorted(
    name
    for name, member in inspect.getmembers(_line_expectations, inspect.isfunction)
    if name.startswith("assert_") and member.__module__ == _line_expectations.__name__
)


def _callers(name: str) -> set[str]:
    """The test files that call `name`, read from their syntax trees, never from their text.

    This file is never counted: it calls every shape to prove what each refuses, which says nothing
    about whether an adapter suite runs it.
    """
    found = set()
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == name
            ):
                found.add(path.name)
                break
    return found


def test_the_module_holds_the_shapes_the_live_tests_check():
    assert SHAPES == [
        "assert_direct_call",
        "assert_failed_call",
        "assert_priced_fee",
        "assert_rerank_call",
        "assert_routed_call",
        "assert_server_tool_line",
    ]


def test_every_shape_is_run_offline_and_read_live():
    for name in SHAPES:
        callers = _callers(name)
        live = {caller for caller in callers if caller.endswith("_live.py")}
        assert live, (
            f"{name} is called by no live test: a shape built for a test that does not exist"
        )
        assert callers - live, f"{name} is called by no offline suite, so nothing in CI runs it"


# --- each shape refuses a line that breaks one fact the page states ---------------------------------
#
# The callers above hand these functions conforming lines, so they would stay green if a check were
# weakened away. These are the other half: one fabricated line per fact, which must be refused.

DIRECT = (
    "llm provider=google purpose=main residency=us model=gemini-3.8-flash duration=1.20s "
    "tokens_in=1200 tokens_out=100 tokens_total=1300 cached_tokens=1000 cost=0.00066 "
    "cost_basis=computed outcome=ok generation_id=resp-0123456789abcdef"
)
ROUTED = (
    "llm provider=openrouter purpose=main endpoint=StreamLake model=z-ai/glm-5.2 duration=2.50s "
    "tokens_in=1000 tokens_out=200 tokens_total=1200 cost=0.0042 outcome=ok "
    "generation_id=gen-1791030923-8abbrsSEsIk06gI224Fi"
)
RERANK = (
    "llm provider=openrouter purpose=memory kind=rerank endpoint=DeepInfra model=z-ai/glm-5.3-flash "
    "duration=1.10s tokens_in=4812 tokens_out=611 tokens_total=5423 tokens_reasoning=540 "
    "cost=0.000846 outcome=ok surface=turn0 pool=20 picked=8 generation_id=gen-rerank0001"
)
FAILED = (
    "llm provider=openrouter purpose=memory kind=rerank model=z-ai/glm-5.3-flash outcome=failed "
    "reason=config:model_not_found attempt=1/1 surface=tool pool=20"
)
SEARCH = (
    "tool name=web_search executor=server provider=openai kind=search.web model=gpt-6-sol count=2 "
    "outcome=ok cost=0.02 cost_basis=computed generation_id=resp-fake0001"
)


def _direct(line: str) -> None:
    _line_expectations.assert_direct_call(
        line, provider="google", model="gemini-3.8-flash", residency="us", cost="computed"
    )


def _routed(line: str) -> None:
    _line_expectations.assert_routed_call(line, provider="openrouter", model="z-ai/glm-5.2")


def _rerank(line: str) -> None:
    _line_expectations.assert_rerank_call(
        line, provider="openrouter", model="z-ai/glm-5.3-flash", pool=20, picked=8
    )


def _failed(line: str) -> None:
    _line_expectations.assert_failed_call(
        line,
        provider="openrouter",
        purpose="memory",
        kind="rerank",
        reason="config:model_not_found",
    )


def _search(line: str) -> None:
    _line_expectations.assert_server_tool_line(
        line,
        name="web_search",
        provider="openai",
        model="gpt-6-sol",
        kind="search.web",
        joins="resp-fake0001",
        cost="computed",
    )


CASES = {
    "direct": (_direct, DIRECT),
    "routed": (_routed, ROUTED),
    "rerank": (_rerank, RERANK),
    "failed": (_failed, FAILED),
    "search": (_search, SEARCH),
}


@pytest.mark.parametrize("shape", sorted(CASES))
def test_a_conforming_line_is_accepted(shape):
    check, line = CASES[shape]
    check(line)


@pytest.mark.parametrize(
    ("shape", "old", "new"),
    [
        ("direct", "residency=us ", "residency=global "),  # where it ran is the location asked for
        ("direct", "residency=us ", "residency=us endpoint=StreamLake "),  # direct: no endpoint
        ("direct", " cost_basis=computed", ""),  # a computed cost says so
        ("direct", " outcome=ok", " outcome=ok cost_gap=unknown_model"),  # a priced call names none
        ("direct", "tokens_total=1300", "tokens_total=1301"),  # exactly in + out
        ("direct", " tokens_total=1300", ""),  # a billed call that stated usage carries its counts
        ("direct", "cost=0.00066 ", "cost=0.00066 tokens_reasoning=101 "),  # a part of tokens_out
        ("direct", "provider=google", "provider=openai"),
        ("direct", "model=gemini-3.8-flash", "model=gemini-3.8-pro"),
        ("direct", "purpose=main", "purpose=helper"),
        ("direct", "outcome=ok", "outcome=refused"),
        ("direct", "cost=0.00066 cost_basis=computed ", ""),  # a priced call carries its cost
        ("direct", "cost=0.00066 ", ""),  # a basis beside no figure
        ("routed", "cost=0.0042 ", ""),  # the router states its cost
        ("direct", "cached_tokens=1000", "cached_tokens=1201"),  # a part of tokens_in
        ("direct", "purpose=main ", "purpose=main kind=rerank "),  # no job on the brain's own call
        ("direct", " outcome=ok", " outcome=ok reason=parse"),  # `ok` never carries a reason
        ("routed", "endpoint=StreamLake ", ""),  # a routed call names its upstream
        ("routed", " outcome=ok", " outcome=ok cost_basis=computed"),  # the router states it
        ("routed", "endpoint=StreamLake ", "endpoint=StreamLake residency=us "),
        ("rerank", "surface=turn0", "surface=elsewhere"),
        ("rerank", "picked=8", "picked=7"),
        ("rerank", "pool=20", "pool=21"),
        ("failed", "reason=config:model_not_found", "reason=config:auth"),
        ("failed", " attempt=1/1", " attempt=1/1 cost=0.01"),  # a decision line carries no cost
        ("failed", " attempt=1/1", " attempt=1/1 tokens_in=12"),
        ("search", "executor=server", "executor=harness"),
        ("search", " count=2", ""),  # a server-side tool always counts its uses
        ("search", "count=2 ", "count=2 duration=1.00s "),  # its `llm` line timed the work
        ("search", "generation_id=resp-fake0001", "generation_id=resp-other"),  # the join
        ("search", "outcome=ok", "outcome=error"),  # an error names its reason
        ("search", " cost_basis=computed", ""),
        ("search", "provider=openai", "provider=anthropic"),
        ("search", "model=gpt-6-sol", "model=gpt-6-luna"),
        ("search", "kind=search.web ", ""),
        ("search", "name=web_search", "name=web_fetch"),
        ("search", " cost_basis=computed", " cost_basis=computed cost_gap=unpriced_host"),
        # An `error` line still names nothing it should not, and still joins its call.
        ("search", "outcome=ok", "outcome=error reason=max_uses_exceeded cost_gap=unpriced_host"),
        ("search", "outcome=ok cost=0.02 cost_basis=computed", "outcome=ok cost=0.02"),
    ],
)
def test_a_line_that_breaks_one_fact_is_refused(shape, old, new):
    check, line = CASES[shape]
    assert line.count(old) == 1, old
    with pytest.raises(AssertionError):
        check(line.replace(old, new))


def test_an_error_line_or_an_unbilled_use_may_carry_no_fee():
    """A failed use is not billed, and a use is not always a billed unit (issue #904 review): the
    shape binds the fee a line carries and never demands one."""
    _search(SEARCH.replace("outcome=ok cost=0.02 cost_basis=computed", "outcome=ok"))
    _search(
        SEARCH.replace(
            "outcome=ok cost=0.02 cost_basis=computed", "outcome=error reason=max_uses_exceeded"
        )
    )


def test_a_fee_nobody_charges_apart_is_absent():
    """A router's searches: their fee is inside the call's `usage.cost`, so the line has none."""
    unbilled = SEARCH.replace(" cost=0.02 cost_basis=computed", "")
    _line_expectations.assert_server_tool_line(
        unbilled,
        name="web_search",
        provider="openai",
        model="gpt-6-sol",
        kind="search.web",
        joins="resp-fake0001",
        cost=None,
    )
    with pytest.raises(AssertionError):
        _line_expectations.assert_server_tool_line(
            SEARCH,
            name="web_search",
            provider="openai",
            model="gpt-6-sol",
            kind="search.web",
            joins="resp-fake0001",
            cost=None,
        )


def test_a_reported_residency_must_be_on_the_line():
    line = DIRECT.replace("residency=us ", "")
    with pytest.raises(AssertionError):
        _line_expectations.assert_direct_call(
            line,
            provider="google",
            model="gemini-3.8-flash",
            residency=_line_expectations.REPORTED,
            cost="computed",
        )


def test_a_priced_fee_refuses_a_figure_it_did_not_compute():
    from basecradle_harness._observability import ComputedCost

    fee = {"provider": "openai", "kind": "audio.transcribe", "count": None, "cost_gap": None}
    _line_expectations.assert_priced_fee(
        fee | {"cost": ComputedCost(0.01)}, provider="openai", kind="audio.transcribe"
    )
    with pytest.raises(AssertionError):
        _line_expectations.assert_priced_fee(
            fee | {"cost": 0.01}, provider="openai", kind="audio.transcribe"
        )
    for broken in (
        {"cost": ComputedCost(0.01), "cost_gap": "unpriced_host"},  # a priced fee names no gap
        {"cost": ComputedCost(0.01), "provider": "xai"},
        {"cost": ComputedCost(0.01), "kind": "image.generate"},
        {"cost": ComputedCost(0.01), "count": 0},
    ):
        with pytest.raises(AssertionError):
            _line_expectations.assert_priced_fee(
                fee | broken, provider="openai", kind="audio.transcribe"
            )
