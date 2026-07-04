"""Live smoke for the native OpenRouter adapter (`OpenRouterProvider`) — issue #234.

The one check the mocked suite **structurally cannot** make: the adapter tests inject a fake SDK
client (respx mocks the transport), so a request the *real* OpenRouter endpoint rejects — a
model-params key the live API refuses, a wire shape that drifted on an SDK bump — still passes
them. This test builds a **real** ``openrouter`` client and hits ``openrouter.ai`` for real, so a
regression to a server-rejected wiring fails loudly here.

It is an explicitly-marked **live** job (`@pytest.mark.live`), deselected from the default offline
run by ``addopts = -m 'not live'`` and skipped when no key is present. Run it deliberately::

    OPENROUTER_API_KEY=sk-or-... uv run pytest -m live tests/test_openrouter_live.py -v

The capital re-runs it (with a valid OpenRouter key) at the release gate; this file makes that a
repeatable command rather than a one-off manual probe.
"""

from __future__ import annotations

import os

import pytest

from basecradle_harness import Message, OpenRouterProvider

pytestmark = pytest.mark.live

KEY = os.environ.get("OPENROUTER_API_KEY")


@pytest.mark.skipif(not KEY, reason="set OPENROUTER_API_KEY to run the live OpenRouter probe")
def test_native_openrouter_returns_a_reply():
    """A real turn against ``openrouter.ai`` returns non-empty assistant text (@glm-5.2's brain)."""
    provider = OpenRouterProvider(model="z-ai/glm-5.2", api_key=KEY)
    try:
        reply = provider.chat([Message.user("Reply with a single short greeting.")])
    finally:
        provider.close()

    assert reply.role == "assistant"
    assert reply.content  # a real, non-empty answer


@pytest.mark.skipif(not KEY, reason="set OPENROUTER_API_KEY to run the live OpenRouter probe")
def test_live_model_params_reach_the_endpoint():
    """A ``model_params.json``-style tuning key (``temperature``) is accepted by the live API.

    The mocked suite proves the key lands in the request body; only this proves the *real*
    endpoint accepts it rather than rejecting the request.
    """
    provider = OpenRouterProvider(model="z-ai/glm-5.2", api_key=KEY, temperature=0.2, max_tokens=32)
    try:
        reply = provider.chat([Message.user("Say hi.")])
    finally:
        provider.close()

    assert reply.role == "assistant"
    assert reply.content
