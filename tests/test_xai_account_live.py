"""Live smoke for xai_account_balance against the real xAI Management API — issues #179, #384.

The mocked suite (`test_xai_account.py`) pins the parsing, the inverted-sign math, and every
degraded mode, but it **cannot** catch a drift in the real endpoints' paths, auth, or response
shapes — the very things a live account confirms. This test builds the real tool with a real
Management Key and hits `management-api.x.ai` for real, so a regression to a wrong path or a
changed payload fails loudly here.

It carries the #384 invariant in the only place that can actually observe it: **the live figure
never exceeds the posted prepaid ledger.** That is the direction of the bug — the posted ledger
settles at cycle close, so mid-cycle it is the larger, staler number — and a fixture cannot
prove it, because a fixture only replays what we already believe. If xAI ever moves the live
figure back onto the ledger's semantics, this is what turns red.

It is an explicitly-marked **live** job (`@pytest.mark.live`), deselected from the default
offline run by ``addopts = -m 'not live'`` and skipped when no key is present. Run it
deliberately::

    XAI_MANAGEMENT_KEY=... uv run pytest -m live tests/test_xai_account_live.py -v -s

This is the repeatable form of the founder live-verify: the printed figure is what gets compared
against the Console's **Credits remaining**.
"""

from __future__ import annotations

import os
import re

import pytest

from basecradle_harness import XaiAccountBalanceTool

pytestmark = pytest.mark.live

KEY = os.environ.get("XAI_MANAGEMENT_KEY")

# e.g. "xAI credits remaining: $118.47 USD (as of 2026-08-02T07:05:11Z)." — a real dollar figure.
_REMAINING = re.compile(
    r"^xAI credits remaining: (-?)\$([\d,]+\.\d{2}) USD \(as of \S+Z\)\.$", re.MULTILINE
)
_LEDGER = re.compile(r"Posted prepaid ledger: (-?)\$([\d,]+\.\d{2})")


def _usd(match: re.Match[str]) -> float:
    """The signed dollar amount out of a rendered figure."""
    return (-1.0 if match.group(1) else 1.0) * float(match.group(2).replace(",", ""))


@pytest.mark.skipif(not KEY, reason="set XAI_MANAGEMENT_KEY to run the live xAI balance probe")
def test_reports_the_live_credits_remaining_not_the_posted_ledger():
    """The tool reads the live remaining credit, team auto-discovered from the key.

    No `team_id` is passed, so this also exercises the discovery call (`/auth/management-keys/
    validation`) against the real endpoint — the path that makes `XAI_TEAM_ID` optional.
    """
    tool = XaiAccountBalanceTool(cache_ttl=0)  # key from env, team discovered from the key
    result = tool.run()
    print(f"\n{result}\n")  # the figure a founder compares against the Console

    assert "unavailable" not in result, result  # a real key + BillingRead scope → a real figure
    assert KEY not in result  # the key never leaks into the output

    remaining = _REMAINING.search(result)
    assert remaining, f"no live credits-remaining headline in:\n{result}"
    # The posted ledger is never the headline — that was the bug (#384).
    assert not result.startswith("xAI posted prepaid ledger total:"), result

    ledger = _LEDGER.search(result)
    assert ledger, f"the posted ledger should appear as labelled context in:\n{result}"
    assert "not what you have left to spend" in result  # ...and be labelled as not-the-runway

    # The invariant: this cycle's usage is unposted, so the live figure can only be at or below
    # the posted ledger. A live figure *above* it means the tool has slipped back onto ledger
    # semantics, or xAI changed what these fields mean.
    live_usd, ledger_usd = _usd(remaining), _usd(ledger)
    assert live_usd <= ledger_usd + 0.01, (
        f"live remaining ${live_usd:,.2f} exceeds the posted ledger ${ledger_usd:,.2f} — the "
        f"live figure should never be the larger of the two:\n{result}"
    )
