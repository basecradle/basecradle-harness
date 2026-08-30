"""Live smoke for the ``openai`` SDK adapter (`OpenAIProvider`) — issue #410.

The offline suite mocks the HTTP **transport**, which is precisely the thing ``openai`` 3.0
changed: the SDK moved to HTTPX2, and with it to the operating system's TLS trust store rather
than ``certifi``'s. A mocked transport cannot fail on a certificate it never verifies, so the
suite that intercepts the wire is structurally blind to the one class of breakage this bump
could cause on a real box — and it stayed green through the whole migration while a live call
was the only thing that could answer the question.

So this hits ``api.openai.com`` for real: one turn, on the surface @jt actually runs
(``responses``), proving the SDK path end to end — TLS handshake, request shape, parsed reply.
It is an explicitly-marked **live** job (`@pytest.mark.live`), deselected from the default
offline run by ``addopts = -m 'not live'`` and skipped when no key is present. Run it
deliberately::

    AI_API_KEY=sk-... uv run pytest -m live tests/test_openai_live.py -v

The @jt live-test key is `AI_PROVIDER_API_KEY` in the laptop's harness-test env (the older
name for the same value); export it as ``AI_API_KEY`` to run this.
"""

from __future__ import annotations

import os

import pytest

from basecradle_harness import Message, OpenAIProvider

pytestmark = pytest.mark.live

KEY = os.environ.get("AI_API_KEY")
MODEL = "gpt-5.4-mini"


@pytest.mark.skipif(not KEY, reason="set AI_API_KEY to run the live OpenAI probe")
def test_a_real_turn_reaches_openai_over_the_sdks_own_transport():
    """A real Responses turn answers, and the token count comes back from the live endpoint.

    Asserted on the **value**, not the presence: an empty reply or a ``tokens_in`` of ``None``
    would both satisfy "it didn't raise" while meaning the call never really landed.
    """
    provider = OpenAIProvider(model=MODEL, api_key=KEY, surface="responses", max_retries=0)
    try:
        reply = provider.chat([Message.user("Reply with exactly: pong")])
    finally:
        provider.close()

    assert reply.role == "assistant"
    assert "pong" in (reply.content or "").lower()
    assert provider.last_tokens_in and provider.last_tokens_in > 0


@pytest.mark.parametrize("surface", ["responses", "chat"])
@pytest.mark.skipif(not KEY, reason="set AI_API_KEY to run the live OpenAI probe")
def test_openai_accepts_the_cache_affinity_key_on_both_surfaces(surface):
    """The live half of issue #435: OpenAI really takes ``prompt_cache_key``, on both surfaces.

    The offline tests prove the field leaves the process on the real request body — respx reads
    the actual bytes — but only the endpoint can say whether it is *accepted*. That distinction is
    the whole cost of getting this wrong: an unknown top-level field is a 400 on every wake, which
    is a silent agent rather than a missed discount.

    Deliberately an **acceptance** probe, not a hit-rate one. A cache read needs a prefix over a
    thousand tokens and two calls close together, and the answer would be a measurement with a date
    on it — the standing rule is that ``cached_tokens=`` on a real agent's log line is the authority
    for that, never a test (`_caching`).
    """
    provider = OpenAIProvider(model=MODEL, api_key=KEY, surface=surface, max_retries=0)
    provider.bind_conversation("timeline:019f6e71-2a12-7b69-a204-0fec1497b9c2")
    try:
        reply = provider.chat([Message.user("Reply with exactly: pong")])
    finally:
        provider.close()

    assert "pong" in (reply.content or "").lower()
    assert provider.last_tokens_in and provider.last_tokens_in > 0
