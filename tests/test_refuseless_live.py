"""Live gate for the refuseless vendor — one real priced call (issue #844).

The one check the offline suite **structurally cannot** make. `test_refuseless.py` drives the real
``openai`` SDK against bodies *we wrote* in the shape the capital read off the vendor on 2026-11-09;
it cannot prove that the vendor still serves that shape, that the model a response names is a row
of the list, or that the dollars the line states are the dollars the vendor's own price implies.
This is where the vendor answers for itself.

It is an explicitly-marked **live** job (`@pytest.mark.live`), deselected from the default run and
skipped without a credential. From a fresh clone of ``main``::

    REFUSELESS_LIVE_API_KEY=… uv run --extra openai pytest -m live tests/test_refuseless_live.py -s -v

``REFUSELESS_LIVE_MODEL`` defaults to ``glm-5.3-flash-abliterated``. ``-s`` prints the ``llm`` line
and the arithmetic, for a hand check. A SKIP here is **not** a pass: pytest exits 0 when every test
skips, so read the report, not the exit code. What it pins is a value, never mere presence: the
``cost=`` equals ``prompt_tokens × input_per_m + completion_tokens × output_per_m`` over a million,
read from the vendor's list by a client of the test's own, not by the harness.
"""

from __future__ import annotations

import logging
import os
import re

import pytest

from basecradle_harness import Message
from basecradle_harness._basecradle import _config_from_env, _provider_from_config
from basecradle_harness._observability import _money

pytestmark = pytest.mark.live

KEY = (os.environ.get("REFUSELESS_LIVE_API_KEY") or "").strip() or None
MODEL = (os.environ.get("REFUSELESS_LIVE_MODEL") or "glm-5.3-flash-abliterated").strip()
HOST = "https://api.refuseless.com/v1"

needs_key = pytest.mark.skipif(
    not KEY, reason="set REFUSELESS_LIVE_API_KEY to run the live refuseless gate"
)


def _field(line: str, name: str) -> str | None:
    found = re.search(rf"(?:^| ){name}=(\S+)", line)
    return found.group(1) if found else None


@needs_key
def test_one_real_call_is_priced_from_the_vendors_own_list(monkeypatch, caplog):
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("AI_PROVIDER", "refuseless")
    monkeypatch.setenv("AI_MODEL", MODEL)
    monkeypatch.setenv("AI_API_KEY", KEY)
    for var in ("AI_SDK", "AI_SDK_SURFACE", "AI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    caplog.set_level(logging.INFO, logger="basecradle_harness")

    provider = _provider_from_config(*_config_from_env(), max_output_tokens=256)
    try:
        limit = provider.context_limit()
        provider.chat([Message.user("Reply with the single word: ready")])
    finally:
        provider.close()

    lines = [record.getMessage() for record in caplog.records]
    (llm,) = [line for line in lines if line.startswith("llm ")]
    print(f"\n{llm}")
    assert not [line for line in lines if line.startswith("cost unpriced")], lines
    assert _field(llm, "provider") == "refuseless"
    assert _field(llm, "model") == MODEL
    assert _field(llm, "cost_basis") == "computed"

    # The vendor's list, read by a client of the test's own: the harness's parser is not the judge.
    with openai.OpenAI(api_key=KEY, base_url=HOST) as client:
        rows = {item.id: item.model_dump() for item in client.models.list()}
    row = rows[MODEL]
    pricing = row["pricing"]
    assert pricing["currency"] == "USD"
    tokens_in, tokens_out = int(_field(llm, "tokens_in")), int(_field(llm, "tokens_out"))
    expected = (tokens_in * pricing["input_per_m"] + tokens_out * pricing["output_per_m"]) / 1e6
    print(
        f"{tokens_in} × ${pricing['input_per_m']}/M + {tokens_out} × ${pricing['output_per_m']}/M"
        f" = ${expected!r} (line: cost={_field(llm, 'cost')})"
    )
    assert _field(llm, "cost") == _money(expected)
    assert limit == row["context_length"]
