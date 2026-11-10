"""Live gate for refuseless_account_balance — one real read of the vendor's Credits API (issue #848).

The one check the offline suite **structurally cannot** make. `test_refuseless_account.py` drives
the tool against bodies *we wrote* in the shape the capital read on 2026-11-10; it cannot prove the
vendor still serves that shape, or that the figures the tool shows are the figures the vendor holds.

It is an explicitly-marked **live** job (`@pytest.mark.live`), deselected from the default run and
skipped without a credential. From a fresh clone of ``main``::

    REFUSELESS_LIVE_API_KEY=… uv run pytest -m live tests/test_refuseless_account_live.py -s -v

The read is free (the vendor charges no credit and places no hold). A SKIP here is **not** a pass:
pytest exits 0 when every test skips, so read the report. What it pins is a value, never presence:
the figures the tool shows equal the ones a client of the test's own reads from the same endpoint.
Run it on a key with no requests in flight, or ``reserved`` and ``available`` can move between the
two reads.
"""

from __future__ import annotations

import os
import re

import httpx
import pytest

from basecradle_harness import RefuselessAccountBalanceTool

pytestmark = pytest.mark.live

KEY = (os.environ.get("REFUSELESS_LIVE_API_KEY") or "").strip() or None
HOST = "https://api.refuseless.com/v1"

needs_key = pytest.mark.skipif(
    not KEY, reason="set REFUSELESS_LIVE_API_KEY to run the live refuseless balance gate"
)

_HEADLINE = re.compile(
    r"^refuseless credit available: \$([0-9,]+\.[0-9]+) USD "
    r"\(as of \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\)\.$"
)


def _figure(text: str) -> float:
    return float(text.replace(",", ""))


@needs_key
def test_the_tool_shows_the_figures_the_vendor_holds(monkeypatch):
    monkeypatch.delenv("AI_BASE_URL", raising=False)
    monkeypatch.setenv("AI_API_KEY", KEY)
    result = RefuselessAccountBalanceTool(cache_ttl=0).run()
    print(f"\n{result}")
    headline, context = result.splitlines()
    found = _HEADLINE.match(headline)
    assert found, headline

    # The vendor's own answer, read by a client of the test's own: the tool's parser is not the judge.
    response = httpx.get(
        f"{HOST}/credits", headers={"Authorization": f"Bearer {KEY}"}, timeout=15.0
    )
    response.raise_for_status()
    body = response.json()
    print(body)
    assert set(body) >= {
        "object",
        "balance",
        "reserved",
        "available",
        "key_limit",
        "unit",
        "updated_at",
    }
    assert body["object"] == "credits" and body["unit"] == "usd"
    assert _figure(found.group(1)) == pytest.approx(body["available"], abs=1e-6)
    balance, reserved = re.search(
        r"Balance \$([0-9,]+\.[0-9]+), of which \$([0-9,]+\.[0-9]+) is held", context
    ).groups()
    assert _figure(balance) == pytest.approx(body["balance"], abs=1e-6)
    assert _figure(reserved) == pytest.approx(body["reserved"], abs=1e-6)
    if body["key_limit"] is None:
        assert "This key has no spending limit of its own." in context
    else:
        limit = re.search(r"limit allows \$([0-9,]+\.[0-9]+) more", context).group(1)
        assert _figure(limit) == pytest.approx(body["key_limit"], abs=1e-6)
