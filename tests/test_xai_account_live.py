"""Live smoke for xai_account_balance against the real xAI Management API — #179, #384, #388.

The mocked suite (`test_xai_account.py`) pins the parsing, the inverted-sign math, and every
degraded mode, but it **cannot** catch a drift in the real endpoints' paths, auth, or response
shapes — the very things a live account confirms. This test builds the real tool with a real
Management Key and hits `management-api.x.ai` for real, so a regression to a wrong path or a
changed payload fails loudly here.

It carries both regressions in the only place that can actually observe them, and they are the
same mistake at two depths — a bigger figure this account can show being read as the runway:

- **#384: the live figure never exceeds the posted prepaid ledger.** The ledger settles at cycle
  close, so mid-cycle it is the larger, staler number.
- **#388: the live figure is strictly below the preview's own `prepaidCredits` whenever the cycle
  has drawn any prepaid credit** — because remaining is that field *minus* the draw. This one is
  checked against an **independent oracle**: the raw preview, fetched here rather than through
  the tool. Without it this file passes on the bug it exists to catch — 0.96.0 reported $168.47
  against a Console showing $52.14 and every assertion above stayed green, because the ledger
  ($567.49) was larger still.

A fixture can prove neither, because a fixture only replays what we already believe.

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
from typing import Any

import httpx
import pytest

from basecradle_harness import XaiAccountBalanceTool
from basecradle_harness._xai_account import DEFAULT_BASE_URL

pytestmark = pytest.mark.live

KEY = os.environ.get("XAI_MANAGEMENT_KEY")

# e.g. "xAI credits remaining: $51.81 USD (as of 2026-08-02T21:14:03Z)." — a real dollar figure.
_REMAINING = re.compile(
    r"^xAI credits remaining: (-?)\$([\d,]+\.\d{2}) USD \(as of \S+Z\)\.$", re.MULTILINE
)
_LEDGER = re.compile(r"Posted prepaid ledger: (-?)\$([\d,]+\.\d{2})")


def _usd(match: re.Match[str]) -> float:
    """The signed dollar amount out of a rendered figure."""
    return (-1.0 if match.group(1) else 1.0) * float(match.group(2).replace(",", ""))


def _live_core_invoice() -> dict[str, Any]:
    """The raw `coreInvoice` off the live preview — an **oracle**, fetched without the tool.

    Deliberately not `XaiAccountBalanceTool`'s own helpers: a test that reads the payload through
    the code under test can only ever confirm that code agrees with itself. This is the reference
    the rendered headline is checked against.
    """
    with httpx.Client(headers={"Authorization": f"Bearer {KEY}"}, timeout=30.0) as client:
        team = os.environ.get("XAI_TEAM_ID")
        if not team:
            validation = client.get(f"{DEFAULT_BASE_URL}/auth/management-keys/validation")
            validation.raise_for_status()
            team = validation.json()["teamId"]
        preview = client.get(f"{DEFAULT_BASE_URL}/v1/billing/teams/{team}/postpaid/invoice/preview")
    preview.raise_for_status()
    return preview.json()["coreInvoice"]


def _credit_usd(core: dict[str, Any], field: str) -> float:
    """A credit-signed field of the live preview, in dollars (stored negative — see the module)."""
    return -int(str(core[field]["val"])) / 100.0


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


@pytest.mark.skipif(not KEY, reason="set XAI_MANAGEMENT_KEY to run the live xAI balance probe")
def test_the_live_figure_nets_the_cycles_prepaid_draw():
    """The #388 regression, against the live payload: remaining is a subtraction, not a field.

    The oracle is the raw preview, read independently of the tool. Two things are asserted, and
    the second is the one 0.96.0 would have failed: the headline equals `prepaidCredits` minus
    `prepaidCreditsUsed`, and — once the cycle has drawn anything — it is strictly *below*
    `prepaidCredits`, which is the credit the cycle draws against rather than what is left of it.
    """
    core = _live_core_invoice()
    prepaid = _credit_usd(core, "prepaidCredits")
    drawn = _credit_usd(core, "prepaidCreditsUsed")
    print(f"\nlive preview: prepaidCredits ${prepaid:,.2f}, drawn ${drawn:,.2f}\n")

    result = XaiAccountBalanceTool(cache_ttl=0).run()
    remaining = _REMAINING.search(result)
    assert remaining, f"no live credits-remaining headline in:\n{result}"

    live_usd = _usd(remaining)
    assert live_usd == pytest.approx(prepaid - drawn, abs=0.01), (
        f"live remaining ${live_usd:,.2f} is not prepaidCredits ${prepaid:,.2f} less the "
        f"${drawn:,.2f} drawn this cycle (= ${prepaid - drawn:,.2f}):\n{result}"
    )
    if drawn > 0:
        assert live_usd < prepaid, (
            f"live remaining ${live_usd:,.2f} is not below prepaidCredits ${prepaid:,.2f} despite "
            f"${drawn:,.2f} drawn this cycle — the undrawn field is being reported as the runway "
            f"(the #388 defect):\n{result}"
        )
