"""Live smoke for xai_account_balance against the real xAI Management API — issue #179.

The mocked suite (`test_xai_account.py`) pins the parsing, the inverted-sign math, and every
degraded mode, but it **cannot** catch a drift in the real endpoint's path, auth, or response
shape — the very things a live account confirms. This test builds the real tool with a real
Management Key and hits `management-api.x.ai` for real, so a regression to a wrong path or a
changed payload fails loudly here.

It is an explicitly-marked **live** job (`@pytest.mark.live`), deselected from the default
offline run by ``addopts = -m 'not live'`` and skipped when no key is present. Run it
deliberately::

    XAI_MANAGEMENT_KEY=... uv run pytest -m live tests/test_xai_account_live.py -v

This is the repeatable form of the @briggs live-verify the capital runs to close issue #179 —
the tool returning the account's real prepaid balance.
"""

from __future__ import annotations

import os
import re

import pytest

from basecradle_harness import XaiAccountBalanceTool

pytestmark = pytest.mark.live

KEY = os.environ.get("XAI_MANAGEMENT_KEY")

# e.g. "xAI prepaid credit balance: $42.50 USD (as of 2026-07-15T22:14:03Z)." — a real dollar figure.
_BALANCE = re.compile(r"xAI prepaid credit balance: -?\$[\d,]+\.\d{2} USD \(as of .+Z\)\.")


@pytest.mark.skipif(not KEY, reason="set XAI_MANAGEMENT_KEY to run the live xAI balance probe")
def test_returns_the_real_prepaid_balance():
    """The tool reads the live account's real prepaid balance, team auto-discovered from the key.

    No `team_id` is passed, so this also exercises the discovery call (`/auth/management-keys/
    validation`) against the real endpoint — the path that makes `XAI_TEAM_ID` optional.
    """
    tool = XaiAccountBalanceTool(cache_ttl=0)  # key from env, team discovered from the key
    result = tool.run()

    assert "unavailable" not in result, result  # a real key + BillingRead scope → a real figure
    assert _BALANCE.match(result), result
    assert KEY not in result  # the key never leaks into the output
