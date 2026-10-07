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

**The key is this suite's own, and that is the point** (issue #441). It is a dedicated OpenAI key
named *Harness Live Test*, on the laptop as ``AI_PROVIDER_API_KEY`` in
``~/.config/basecradle/harness-test.env``; export it as ``AI_API_KEY`` to run this. It is **not**
@jt's key, and must never be seeded from one again: this env file used to hold a *copy* of @jt's
runtime key, so rotating his box's credential silently killed this suite. Two consumers sharing one
credential is the defect — @origin's ruling on #441 is one key per consumer, the same per-consumer
isolation the fleet's per-agent keys already follow.

**Something runs this on a schedule now, and a SKIP is RED** (issue #443). Nothing did before:
``-m 'not live'`` hides it from every default run and from CI, and it skips itself green with no
key — so the suite had three states, *passed* / *skipped* / *never invoked*, and from outside the
box the last two look exactly like the first. The dead key above went unnoticed for an unknown
period for precisely that reason. The trigger is a NOC prober (``basecradle-noc#563``): weekly on
green, daily on red, cloning the tip of ``main`` and running this file's own invocation with a
**third** dedicated key that lives only on the NOC box. Its verdict is read off a JUnit report
rather than ``returncode`` — pytest exits **0** when every collected test skips, so an absent key is
byte-identical to a pass at the process boundary — and a run reporting ``skipped > 0`` is a failure
there, per this suite's non-negotiable.

Two consequences for anyone editing this file. **The path and the marker are a cross-repo
contract**: the prober selects ``-m live tests/test_openai_live.py``, so renaming either makes it
collect zero tests, which it reads as *never invoked* and calls red — correct and loud, but tell the
NOC rather than leaving it to fire. And **the assertions stay ours** — the NOC deliberately runs
this suite instead of re-implementing it, so what this file proves is what gets proven on a cadence,
and adding a case here needs no coordination at all.

The other four live suites have the same trigger now (issue #450): the prober grew from this one
pinned path into a **registry** of five arms — ``openai`` (this file), ``xai``, ``openrouter``,
``xai-account``, ``openrouter-account`` — each with its own dedicated key, its own freshness clock,
and its own ``armed`` flag, since four keys do not arrive as one event.
"""

from __future__ import annotations

import logging
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


@pytest.mark.parametrize(("surface", "prefix"), [("responses", "resp_"), ("chat", "chatcmpl-")])
@pytest.mark.skipif(not KEY, reason="set AI_API_KEY to run the live OpenAI probe")
def test_the_live_line_names_the_call_by_openais_own_id(surface, prefix, caplog):
    """The line's ``generation_id=`` is the body's id for the call, on both surfaces (issue #634).

    By value: each surface's own prefix, which also tells it apart from the ``req_`` id OpenAI sends
    in ``x-request-id`` — a different id with a different use, never to be passed off as this one.
    On the Responses surface the id is looked up as well, through the API that stores responses by
    default, so the line is proved to name a call OpenAI can find.
    """
    provider = OpenAIProvider(model=MODEL, api_key=KEY, surface=surface, max_retries=0)
    try:
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            provider.chat([Message.user("Reply with exactly: pong")])
        line = next(
            r.getMessage() for r in caplog.records if r.getMessage().startswith("llm provider=")
        )
        generation = line.split()[-1].removeprefix("generation_id=")
        assert generation.startswith(prefix), line
        if surface == "responses":
            assert provider._client.responses.retrieve(generation).id == generation
    finally:
        provider.close()


# --- the computed cost (issue #657) ------------------------------------------------------------
#
# OpenAI states no price, so the line's `cost=` is the harness's arithmetic over `_openai_rates`.
# Only the live endpoint can say whether the usage shapes that arithmetic reads are the ones OpenAI
# actually sends — the cache-write count, the Images split, the transcription duration — so each is
# checked here against a real response, by value: the line's figure must equal the table's own
# arithmetic over what OpenAI reported, rendered the way the line renders it — and the cache-write
# count is proven non-zero on a prompt long enough to be written, so a misspelt field cannot price
# every write as ordinary input and still pass.

FLEET_MODEL = "gpt-6-sol"


def _line(caplog, head):
    return next(m for m in (r.getMessage() for r in caplog.records) if m.startswith(head))


def _field(line, name):
    for token in line.split():
        if token.startswith(f"{name}="):
            return token.split("=", 1)[1]
    return None


@pytest.mark.skipif(not KEY, reason="set AI_API_KEY to run the live OpenAI probe")
def test_a_fleet_model_turn_logs_the_tables_arithmetic_as_its_cost(caplog):
    from basecradle_harness._observability import _money, capture_llm_call
    from basecradle_harness._openai_rates import call_cost, usage_of

    provider = OpenAIProvider(model=FLEET_MODEL, api_key=KEY, surface="responses", max_retries=0)
    try:
        with capture_llm_call() as call:
            provider.chat([Message.user("Reply with exactly: pong")])
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            provider.chat([Message.user("Reply with exactly: pong")])
    finally:
        provider.close()

    usage = usage_of(call.usage)
    assert usage is not None and usage.input > 0 and usage.output > 0
    expected = call_cost(FLEET_MODEL, usage)
    assert expected is not None and _money(call.cost) == _money(expected)
    line = _line(caplog, "llm provider=")
    assert _field(line, "cost") is not None and _field(line, "cost_basis") == "computed", line


@pytest.mark.skipif(not KEY, reason="set AI_API_KEY to run the live OpenAI probe")
def test_a_web_search_logs_one_priced_line_per_search_call(caplog):
    provider = OpenAIProvider(
        model=FLEET_MODEL,
        api_key=KEY,
        surface="responses",
        max_retries=0,
        builtin_tools=["web_search"],
    )
    try:
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            provider.chat(
                [Message.user("Search the web: what is today's top headline on bbc.com? One line.")]
            )
    finally:
        provider.close()

    line = _line(caplog, "media provider=openai kind=search.web")
    count = int(_field(line, "count"))
    assert count >= 1
    assert _field(line, "cost") == f"{count * 0.01:.8f}".rstrip("0").rstrip(".")
    assert _field(line, "cost_basis") == "computed"


@pytest.mark.skipif(not KEY, reason="set AI_API_KEY to run the live OpenAI probe")
def test_an_images_response_carries_the_usage_split_its_price_reads():
    """The Images API reports image and text input apart; without that split there is no price."""
    from openai import OpenAI

    from basecradle_harness._openai_rates import image_cost

    client = OpenAI(api_key=KEY, max_retries=0)
    response = client.images.generate(
        model="gpt-image-2.5-flare", prompt="a small red dot", size="1024x1024", quality="low"
    )
    usage = response.usage
    assert usage is not None and usage.input_tokens_details.text_tokens > 0
    assert usage.output_tokens > 0
    assert image_cost("gpt-image-2.5-flare", usage.model_dump()) is not None


@pytest.mark.skipif(not KEY, reason="set AI_API_KEY to run the live OpenAI probe")
def test_a_transcription_reports_the_duration_its_price_reads():
    """gpt-transcribe is priced per minute, from the duration the response states."""
    import io
    import struct
    import wave

    from openai import OpenAI

    from basecradle_harness._openai_rates import transcription_cost

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:  # two seconds of a quiet tone, 16 kHz mono
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"".join(struct.pack("<h", (i % 40) * 50) for i in range(32_000)))
    client = OpenAI(api_key=KEY, max_retries=0)
    response = client.audio.transcriptions.create(
        model="gpt-transcribe", file=("tone.wav", buffer.getvalue(), "audio/wav")
    )
    usage = response.usage
    assert usage is not None and usage.type == "duration"
    assert usage.seconds == pytest.approx(2.0, abs=0.5)
    assert transcription_cost("gpt-transcribe", usage.model_dump()) == pytest.approx(
        usage.seconds / 60 * 0.0045
    )


@pytest.mark.skipif(not KEY, reason="set AI_API_KEY to run the live OpenAI probe")
def test_a_cache_write_is_reported_and_priced_at_the_write_rate():
    """A long, never-seen prefix is written to the cache, and OpenAI says so in the field we read.

    Unique per run (a fresh uuid leads the prompt), so no earlier run's cache can turn the write
    into a read. If OpenAI renamed ``cache_write_tokens``, `usage_of` would read 0 and this fails.
    """
    import uuid

    from basecradle_harness._observability import _money, capture_llm_call
    from basecradle_harness._openai_rates import RATES, Usage, call_cost, usage_of

    filler = " ".join(f"Clause {i}: the harness prices every call it can." for i in range(400))
    prompt = f"Run {uuid.uuid4()}. {filler}\n\nReply with exactly: pong"
    provider = OpenAIProvider(model=FLEET_MODEL, api_key=KEY, surface="responses", max_retries=0)
    try:
        with capture_llm_call() as call:
            provider.chat([Message.user(prompt)])
    finally:
        provider.close()

    usage = usage_of(call.usage)
    assert usage is not None and usage.cache_write > 0, call.usage
    assert _money(call.cost) == _money(call_cost(FLEET_MODEL, usage))
    as_plain_input = call_cost(
        FLEET_MODEL, Usage(input=usage.input, cached=usage.cached, output=usage.output)
    )
    tier = RATES[FLEET_MODEL].short
    assert call.cost - as_plain_input == pytest.approx(
        usage.cache_write * (tier.cache_write - tier.input) / 1e6
    )
