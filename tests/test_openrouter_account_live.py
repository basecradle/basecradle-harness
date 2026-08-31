"""Live smoke for openrouter_account_balance against the real OpenRouter API — issue #425.

The mocked suite (`test_openrouter_account.py`) pins the arithmetic, the rendering, and every
degraded mode, but it **cannot** catch a drift in the real endpoint's path, auth, or response
shape — the very things a live account confirms. A fixture only ever replays what we already
believe. This test builds the real tool with a real Management key and hits
``openrouter.ai/api/v1/credits`` for real, so a regression to a wrong path or a changed payload
fails loudly here.

The headline is checked against an **independent oracle** — the raw ``/credits`` body, fetched
here rather than through the tool — because a test that reads the payload through the code under
test can only ever confirm that code agrees with itself. That is what makes this able to catch
the one defect the shape invites: reporting ``total_credits`` (the credit ever purchased) as the
runway, which is xAI's issue #388 one vendor over and which every "is there a dollar figure?"
assertion would happily pass.

It is an explicitly-marked **live** job (`@pytest.mark.live`), deselected from the default
offline run by ``addopts = -m 'not live'`` and skipped when no key is present. Run it
deliberately::

    OPENROUTER_MANAGEMENT_KEY=... uv run pytest -m live tests/test_openrouter_account_live.py -v -s

This is the repeatable form of the live verify: the printed figure is what gets compared against
the account's Credits page.

**Something runs this on a schedule now, and a SKIP is RED** (issue #450). Nothing did before: ``-m
'not live'`` hides this file from every default run and from CI, and it skips itself green with no
key — three states, *passed* / *skipped* / *never invoked*, and from outside the box the last two
look exactly like the first. That is this repo's own named failure shape, Green-While-Absent,
pointed at the one suite that can catch drift in ``/credits``' path, auth, or shape — the figure an
agent reads its own OpenRouter runway from. The trigger is the NOC prober (``basecradle-noc#563``,
grown from one pinned path to a **registry** of five arms in ``basecradle-noc#575``): this file is
the **``openrouter-account``** arm, and the prober clones the tip of ``main`` and runs this file's
own invocation with a **dedicated** ``OPENROUTER_MANAGEMENT_KEY`` that lives only on the NOC box —
never a copy of a running agent's runtime key, per @origin's per-consumer ruling on #441. Weekly on
green, daily on red, per arm. The verdict is read off a JUnit report rather than ``returncode``,
because pytest exits **0** when every collected test skips, so an absent key is byte-identical to a
pass at the process boundary; a run reporting ``skipped > 0`` is a failure there.

Two consequences for anyone editing this file. **The path and the marker are a cross-repo
contract**: the arm selects ``-m live tests/test_openrouter_account_live.py``, so renaming either
makes it collect zero tests, which the prober reads as *never invoked* and calls red — correct and
loud, but tell the NOC rather than leaving it to fire. And **the assertions stay ours** — the
prober runs this suite instead of re-implementing it, so what this file proves is what gets proven
on a cadence, and adding a case here needs no coordination at all.

One asymmetry, checked rather than assumed, so it does not read as a finding later:
`test_an_inference_key_is_rejected_softly_not_as_a_crash` hardcodes its own bad key to assert the
soft-failure path, so under a deliberately-bogus ``OPENROUTER_MANAGEMENT_KEY`` it correctly still
passes while the arm goes red on the other test. With **no** key both skip, which is red for the
right reason.
"""

from __future__ import annotations

import os
import re

import httpx
import pytest

from basecradle_harness import OpenRouterAccountBalanceTool
from basecradle_harness._openrouter_account import DEFAULT_BASE_URL

pytestmark = pytest.mark.live

KEY = os.environ.get("OPENROUTER_MANAGEMENT_KEY")

# e.g. "OpenRouter credits remaining: $106.54 USD (as of 2026-08-28T23:40:03Z)." — a real figure.
_REMAINING = re.compile(
    r"^OpenRouter credits remaining: (-?)\$([\d,]+\.\d{2}) USD \(as of \S+Z\)\.$", re.MULTILINE
)


def _usd(match: re.Match[str]) -> float:
    """The signed dollar amount out of a rendered figure."""
    return (-1.0 if match.group(1) else 1.0) * float(match.group(2).replace(",", ""))


def _live_credits() -> dict[str, float]:
    """The raw ``data`` off the live ``/credits`` body — an **oracle**, fetched without the tool.

    Deliberately not `OpenRouterAccountBalanceTool`'s own helpers: this is the reference the
    rendered headline is checked against, so it must not share the code being checked.
    """
    with httpx.Client(headers={"Authorization": f"Bearer {KEY}"}, timeout=30.0) as client:
        response = client.get(f"{DEFAULT_BASE_URL}/credits")
    response.raise_for_status()
    return response.json()["data"]


@pytest.mark.skipif(not KEY, reason="set OPENROUTER_MANAGEMENT_KEY to run the live balance probe")
def test_reports_the_live_credits_remaining():
    """The tool reads the real account and reports purchased-less-used, not either term."""
    tool = OpenRouterAccountBalanceTool(cache_ttl=0)  # key from env
    result = tool.run()
    print(f"\n{result}\n")  # the figure compared against the account's Credits page

    assert "unavailable" not in result, result  # a real Management key → a real figure
    assert KEY not in result  # the key never leaks into the output

    data = _live_credits()
    purchased, used = float(data["total_credits"]), float(data["total_usage"])
    print(f"live /credits: total_credits ${purchased:,.2f}, total_usage ${used:,.2f}\n")

    remaining = _REMAINING.search(result)
    assert remaining, f"no credits-remaining headline in:\n{result}"
    live_usd = _usd(remaining)

    assert live_usd == pytest.approx(round(purchased, 2) - round(used, 2), abs=0.01), (
        f"reported ${live_usd:,.2f} is not total_credits ${purchased:,.2f} less the ${used:,.2f} "
        f"used (= ${purchased - used:,.2f}):\n{result}"
    )
    # The defect the shape invites: reporting the credit ever *purchased* as the runway.
    if used > 0:
        assert live_usd < purchased, (
            f"reported ${live_usd:,.2f} is not below total_credits ${purchased:,.2f} despite "
            f"${used:,.2f} used — the purchased total is being reported as the runway:\n{result}"
        )


@pytest.mark.skipif(not KEY, reason="set OPENROUTER_MANAGEMENT_KEY to run the live balance probe")
def test_an_inference_key_is_rejected_softly_not_as_a_crash():
    """The endpoint takes a *Management* key; the ordinary API key 401s (verified 2026-08-28).

    That is the mistake an operator actually makes, so it must come back as the readable
    "needs a Management key" reason rather than an exception out of a wake.
    """
    tool = OpenRouterAccountBalanceTool(management_key="sk-or-v1-not-a-management-key", cache_ttl=0)
    result = tool.run()
    print(f"\n{result}\n")

    assert result.startswith("OpenRouter account balance unavailable — ")
    assert "Management key" in result
    assert "OPENROUTER_MANAGEMENT_KEY" in result
