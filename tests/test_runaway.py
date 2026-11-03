"""A runaway generation is bounded per attempt — issue #788.

On 2026-11-03 `@ping`'s brain call ran **941 s** and wrote **131,072** output tokens (the model's
ceiling) before the harness saw an answer it could not use. Nothing bounded the attempt:

- **no output cap** — no fleet agent set one, so every attempt was allowed the vendor's ceiling;
- **no wall clock** — the fitted budget (`_timeouts`) was handed to ``httpx`` as a *read* timeout,
  which bounds the silence between two chunks, and OpenRouter sends a keepalive chunk every three
  seconds while a non-streaming answer is written (measured live), so it never fired;
- **no name for it** — the cut-off tool call it ended in was logged ``reason=invalid_response``,
  which the board cannot tell from a malformed body.

This file pins the three fixes, in that order. The keepalive itself is reproduced against a real
loopback server in `test_timeouts.py`, which is where every adapter's budget is read off the wire.
"""

from __future__ import annotations

import json
import logging

import httpx
import httpx2
import pytest
import respx

from basecradle_harness import (
    AnthropicProvider,
    Engine,
    Message,
    OpenRouterProvider,
    ProviderResponseError,
    ProviderTruncatedError,
    ToolRegistry,
    _deadline,
)
from basecradle_harness import _basecradle as basecradle
from basecradle_harness._anthropic import DEFAULT_BASE_URL as ANTHROPIC_BASE
from basecradle_harness._basecradle import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    _max_output_tokens_from_env,
    _provider_from_config,
)
from basecradle_harness._deadline import _budget, bound_attempts
from basecradle_harness._observability import read_answer
from tests.conftest import completion, is_retry, out_message, responses_body
from tests.test_openrouter import completion as openrouter_completion
from tests.test_openrouter import wire_tool_call

FAKE_KEY = "sk-test-0123456789abcdef0123456789abcdef"
ANTHROPIC_KEY = "sk-ant-api03-FAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfake-0000AA"
CLAUDE = "claude-haiku-5-5"
OPENROUTER_CHAT = "https://openrouter.ai/api/v1/chat/completions"


# === 1. An attempt may write no more than its cap ==============================================


def test_the_default_cap_clears_every_real_answer_the_fleet_has_written():
    """The number is the measurement (basecradle-harness#788, the capital's 30-day read): every
    answer a model meant was under 16 K output tokens, the largest 15,799. Twice that, so nothing a
    model means to write is cut; a quarter of the 131,072-token runaway."""
    assert DEFAULT_MAX_OUTPUT_TOKENS == 32_768
    assert DEFAULT_MAX_OUTPUT_TOKENS >= 2 * 15_799


def test_the_knob_is_the_default_unless_set(monkeypatch):
    monkeypatch.delenv("HARNESS_MAX_OUTPUT_TOKENS", raising=False)
    assert _max_output_tokens_from_env() == DEFAULT_MAX_OUTPUT_TOKENS
    monkeypatch.setenv("HARNESS_MAX_OUTPUT_TOKENS", "  ")
    assert _max_output_tokens_from_env() == DEFAULT_MAX_OUTPUT_TOKENS
    monkeypatch.setenv("HARNESS_MAX_OUTPUT_TOKENS", "4096")
    assert _max_output_tokens_from_env() == 4096


@pytest.mark.parametrize("value", ["0", "-1", "lots"])
def test_a_knob_that_is_not_a_cap_fails_at_startup(monkeypatch, value):
    """There is no "uncapped": a value that would send the vendor nothing, or nonsense, refuses
    before the first wake rather than reaching a vendor that refuses it on every one."""
    monkeypatch.setenv("HARNESS_MAX_OUTPUT_TOKENS", value)
    with pytest.raises(ValueError):
        _max_output_tokens_from_env()


@pytest.fixture
def brain_env(monkeypatch, tmp_path):
    home = tmp_path / "config"
    home.mkdir()
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(home))
    monkeypatch.setenv("AI_MODEL", "z-ai/glm-5.3")
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("AI_BASE_URL", raising=False)
    return home


#: Every ``openai``/``openrouter``-SDK cell and the field its wire calls the cap — read off the
#: request body, never off what the adapter decided.
WIRE_CELLS = [
    ("openai", "openai", "responses", "https://api.openai.com/v1/responses", "max_output_tokens"),
    (
        "openai",
        "openai",
        "chat",
        "https://api.openai.com/v1/chat/completions",
        "max_completion_tokens",
    ),
    ("xai", "openai", "chat", "https://api.x.ai/v1/chat/completions", "max_tokens"),
    ("openrouter", "openai", "chat", OPENROUTER_CHAT, "max_tokens"),
    ("openrouter", "openrouter", "chat", OPENROUTER_CHAT, "max_tokens"),
]


def _ok_body(sdk: str, surface: str) -> dict:
    if surface == "responses":
        return responses_body(out_message("ok"))
    return openrouter_completion(content="ok") if sdk == "openrouter" else completion(content="ok")


@pytest.mark.parametrize(("provider", "sdk", "surface", "url", "field"), WIRE_CELLS)
def test_the_default_cap_is_on_every_brain_request(brain_env, provider, sdk, surface, url, field):
    built = _provider_from_config(provider, sdk, surface, default_output_tokens=32_768)
    with respx.mock(assert_all_called=True) as router:
        route = router.post(url).mock(return_value=httpx.Response(200, json=_ok_body(sdk, surface)))
        built.chat([Message.user("hi")])

    body = json.loads(route.calls.last.request.content)
    assert body[field] == 32_768
    # One spelling per wire: the cap is never sent twice under two names.
    assert not {"max_tokens", "max_completion_tokens", "max_output_tokens"} - {field} & set(body)


@pytest.mark.parametrize(
    ("provider", "surface", "url", "key"),
    [
        ("openrouter", "chat", OPENROUTER_CHAT, "max_tokens"),
        ("openai", "chat", "https://api.openai.com/v1/chat/completions", "max_tokens"),
        ("openai", "chat", "https://api.openai.com/v1/chat/completions", "max_completion_tokens"),
        ("openai", "responses", "https://api.openai.com/v1/responses", "max_output_tokens"),
    ],
)
def test_an_operators_own_cap_is_sent_as_written_and_the_default_adds_nothing(
    brain_env, provider, surface, url, key
):
    """``model_params.json`` is the agent's own tuning: whatever spelling it names a cap in, that
    cap is the call's, and the default does not add a second one beside it."""
    (brain_env / "model_params.json").write_text(json.dumps({key: 1000}))
    built = _provider_from_config(provider, "openai", surface, default_output_tokens=32_768)
    with respx.mock(assert_all_called=True) as router:
        route = router.post(url).mock(
            return_value=httpx.Response(200, json=_ok_body("openai", surface))
        )
        built.chat([Message.user("hi")])

    body = json.loads(route.calls.last.request.content)
    assert body[key] == 1000
    assert [
        k for k in ("max_tokens", "max_completion_tokens", "max_output_tokens") if k in body
    ] == [key]


def test_a_callers_own_cap_wins_over_the_default(brain_env):
    """The describer's budget (issue #488) is its own, and the brain's default never touches it."""
    built = _provider_from_config(
        "openrouter", "openrouter", "chat", max_output_tokens=2048, default_output_tokens=32_768
    )
    assert built.tuning["max_tokens"] == 2048


def test_no_default_sends_nothing_which_is_every_caller_but_the_brain(brain_env):
    built = _provider_from_config("openrouter", "openrouter", "chat")
    assert "max_tokens" not in built.tuning


@pytest.mark.parametrize(
    ("provider", "sdk", "adapter", "field"),
    [
        ("xai", "xai-sdk", "XaiSdkProvider", "max_tokens"),
        ("google", "google-genai", "GoogleProvider", "max_output_tokens"),
    ],
)
def test_the_default_reaches_the_native_adapters_in_their_own_spelling(
    brain_env, monkeypatch, provider, sdk, adapter, field
):
    """The two adapters whose wire this file cannot read with respx (gRPC, and Vertex's own
    credentials): what the build hands them is the seam."""
    received: dict = {}

    def record(model, **kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(basecradle, adapter, record)
    _provider_from_config(
        provider, sdk, "native" if sdk == "xai-sdk" else "", default_output_tokens=32_768
    )
    assert received[field] == 32_768


def test_the_brain_build_reads_the_knob(brain_env, monkeypatch):
    """The wake's own provider is built with `HARNESS_MAX_OUTPUT_TOKENS`, not only a test's."""
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("AI_SDK", "openrouter")
    monkeypatch.setenv("HARNESS_MAX_OUTPUT_TOKENS", "12345")
    provider, *_rest = basecradle._resolve_tools_and_provider()
    assert provider.tuning["max_tokens"] == 12345


MODEL_RECORD = {
    "type": "model",
    "id": CLAUDE,
    "display_name": "Claude Haiku 5.5",
    "created_at": "2026-09-01T00:00:00Z",
    "lifecycle": "active",
    "max_input_tokens": 1_000_000,
    "max_tokens": 128_000,
    "capabilities": None,
}


def _claude_reply(*content, stop_reason="end_turn") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_01FakeFakeFakeFake",
            "type": "message",
            "role": "assistant",
            "model": CLAUDE,
            "content": list(content),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 7},
        },
    )


@pytest.mark.parametrize(("ceiling", "sent"), [(128_000, 32_768), (8_192, 8_192)])
def test_anthropic_sends_the_default_no_higher_than_the_models_own_maximum(ceiling, sent):
    """The Messages API requires `max_tokens` and refuses one past the model's maximum, which only
    the Models API states, so the default is lowered to it rather than refused on every wake."""
    adapter = AnthropicProvider(CLAUDE, api_key=ANTHROPIC_KEY, default_max_tokens=32_768)
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{ANTHROPIC_BASE}/v1/models/{CLAUDE}").mock(
            return_value=httpx.Response(200, json={**MODEL_RECORD, "max_tokens": ceiling})
        )
        route = router.post(f"{ANTHROPIC_BASE}/v1/messages").mock(
            return_value=_claude_reply({"type": "text", "text": "ok"})
        )
        adapter.chat([Message.user("hi")])
    assert json.loads(route.calls.last.request.content)["max_tokens"] == sent
    adapter.close()


def test_anthropic_keeps_an_operators_own_max_tokens():
    adapter = AnthropicProvider(
        CLAUDE, api_key=ANTHROPIC_KEY, max_tokens=100, default_max_tokens=32_768
    )
    with respx.mock(assert_all_called=True) as router:
        route = router.post(f"{ANTHROPIC_BASE}/v1/messages").mock(
            return_value=_claude_reply({"type": "text", "text": "ok"})
        )
        adapter.chat([Message.user("hi")])
    assert json.loads(route.calls.last.request.content)["max_tokens"] == 100
    adapter.close()


# === 2. An attempt may take no longer than its budget ==========================================
#
# Measured on a fake clock, so each case is exact; the real keepalive, against a real server, is in
# `test_timeouts.py`.


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(_deadline, "_clock", fake)
    return fake


@pytest.mark.parametrize(
    ("timeout", "budget"),
    [
        ({"connect": 10.0, "read": 540.0, "write": 540.0, "pool": 10.0}, 550.0),
        ({"connect": None, "read": 60.0}, 60.0),
        ({"connect": 10.0, "read": None}, None),  # nothing says how long the answer may take
        ({"read": 0}, None),
        ({"read": True}, None),  # a bool is not a number of seconds
        (None, None),
        ("540", None),
    ],
)
def test_the_deadline_is_the_two_budgets_a_request_carries(timeout, budget):
    assert _budget(timeout) == budget


@pytest.mark.parametrize("family", [httpx, httpx2], ids=["httpx", "httpx2"])
def test_a_keepalive_body_is_cut_at_the_deadline_with_the_familys_own_timeout(clock, family):
    """The incident's shape: headers at once, then a byte every few seconds, the answer never. Each
    chunk arrives 3 s after the last (OpenRouter's measured interval); the budget is 10 s."""

    def keepalive():
        for _ in range(100):
            clock.now += 3.0
            yield b"\n"
        yield b"{}"

    def handler(request):
        return family.Response(200, content=keepalive())

    client = family.Client(transport=family.MockTransport(handler))
    assert bound_attempts(client)
    with pytest.raises(family.ReadTimeout):
        client.post("https://model.test/v1/chat", timeout=family.Timeout(10.0, connect=0.0))
    assert clock.now == 1_000.0 + 12.0  # the fourth chunk, the first one past the deadline


@pytest.mark.parametrize("family", [httpx, httpx2], ids=["httpx", "httpx2"])
def test_headers_that_arrive_after_the_deadline_end_the_attempt(clock, family):
    def handler(request):
        clock.now += 61.0
        return family.Response(200, json={})

    client = family.Client(transport=family.MockTransport(handler))
    bound_attempts(client)
    with pytest.raises(family.ReadTimeout):
        client.post("https://model.test/v1/chat", timeout=family.Timeout(60.0, connect=0.0))


@pytest.mark.parametrize("family", [httpx, httpx2], ids=["httpx", "httpx2"])
def test_an_answer_inside_its_budget_is_untouched(clock, family):
    def handler(request):
        def body():
            clock.now += 3.0
            yield b'{"ok"'
            clock.now += 3.0
            yield b": true}"

        return family.Response(200, content=body())

    client = family.Client(transport=family.MockTransport(handler))
    bound_attempts(client)
    response = client.post("https://model.test/v1/chat", timeout=family.Timeout(10.0, connect=0.0))
    assert response.json() == {"ok": True}


def test_a_request_with_no_read_budget_carries_no_deadline(clock):
    def handler(request):
        clock.now += 10_000.0
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bound_attempts(client)
    assert client.post("https://model.test/x", timeout=httpx.Timeout(None)).json() == {"ok": True}


def test_the_hooks_run_after_the_ones_already_on_the_client(clock):
    """The Vertex adapter's request hook sets the timeout the deadline reads, so it must run first."""
    order: list[str] = []

    def phased(request):
        order.append("phased")
        request.extensions["timeout"] = httpx.Timeout(5.0, connect=0.0).as_dict()

    def handler(request):
        clock.now += 6.0
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler), event_hooks={"request": [phased]})
    bound_attempts(client)
    with pytest.raises(httpx.ReadTimeout):
        client.post("https://model.test/x", timeout=httpx.Timeout(None))
    assert order == ["phased"]


def test_the_deadline_is_on_the_body_before_any_other_hook_reads_it(clock):
    """The OpenRouter adapter's citation capture reads the whole body in a response hook. Run
    ahead of the deadline, it read the keepalive stream to its end with no deadline on it, which
    the loopback proof caught; the deadline's response hook therefore runs first."""

    def reads_the_body(response):
        response.read()

    def keepalive():
        for _ in range(100):
            clock.now += 3.0
            yield b"\n"

    def handler(request):
        return httpx.Response(200, content=keepalive())

    client = httpx.Client(
        transport=httpx.MockTransport(handler), event_hooks={"response": [reads_the_body]}
    )
    bound_attempts(client)
    with pytest.raises(httpx.ReadTimeout):
        client.post("https://model.test/x", timeout=httpx.Timeout(10.0, connect=0.0))
    assert clock.now == 1_000.0 + 12.0


def test_something_that_is_not_a_client_is_left_alone():
    assert bound_attempts(object()) is False


# === 3. A cut-off attempt is written as what it is =============================================


def test_an_unreadable_answer_at_the_cap_becomes_truncated_with_its_cause():
    cause = ProviderResponseError("Could not decode tool-call arguments")
    body = {"id": "gen-1793695889-9hxBQEAkumHkPzQwF6hi", "choices": [{"finish_reason": "length"}]}

    def read():
        raise cause

    reply, error = read_answer(read, body)
    assert reply is None
    assert isinstance(error, ProviderTruncatedError) and isinstance(error, ProviderResponseError)
    assert error.__cause__ is cause
    assert error.generation_id == "gen-1793695889-9hxBQEAkumHkPzQwF6hi"


def test_an_unreadable_answer_that_stopped_of_its_own_accord_is_left_as_it_was():
    cause = ProviderResponseError("malformed")

    def read():
        raise cause

    assert read_answer(read, {"choices": [{"finish_reason": "stop"}]}) == (None, cause)


def test_an_adapter_that_read_the_stop_reason_itself_hands_it_over():
    """Vertex's body carries no ``choices``: its adapter reads ``MAX_TOKENS`` and passes it in."""

    def read():
        raise ValueError("no parts")

    _reply, error = read_answer(read, {"id": "resp-1"}, finish="MAX_TOKENS")
    assert isinstance(error, ProviderTruncatedError)


def _cut_body():
    cut = wire_tool_call(id="call_0", name="get_weather", arguments='{"city": "Dal')
    return openrouter_completion(tool_calls=[cut], finish_reason="length")


def test_the_runaway_is_retried_and_both_lines_say_truncated(caplog):
    """The incident's recovery, which worked and must keep working: the cut-off attempt is billed as
    `fallback reason=truncated`, the decision says `retrying reason=truncated`, and the re-issued
    request is answered."""
    provider = OpenRouterProvider("z-ai/glm-5.3", api_key=FAKE_KEY)
    engine = Engine(provider, ToolRegistry(), sleep=lambda _seconds: None)
    with respx.mock(assert_all_called=True) as router:
        router.post(OPENROUTER_CHAT).mock(
            side_effect=[
                httpx.Response(200, json=_cut_body()),
                httpx.Response(200, json=openrouter_completion(content="done")),
            ]
        )
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            reply = engine.run([Message.user("weather?")])

    assert reply.content == "done"
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm ")]
    billed, retrying, ok = lines
    assert "outcome=fallback" in billed and "reason=truncated" in billed
    assert is_retry(retrying) and "reason=truncated" in retrying
    assert "outcome=ok" in ok


def test_anthropic_never_dispatches_a_tool_call_the_cap_cut_off(caplog):
    """Anthropic hands a cut-off ``tool_use`` back with ``input: {}`` (measured live, 2026-11-03):
    it parses, and it is not the call the model was making, so it is refused before any tool runs."""
    adapter = AnthropicProvider(CLAUDE, api_key=ANTHROPIC_KEY, max_tokens=60)
    cut = {"type": "tool_use", "id": "toolu_01FakeFakeFake", "name": "post", "input": {}}
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{ANTHROPIC_BASE}/v1/messages").mock(
            return_value=_claude_reply(cut, stop_reason="max_tokens")
        )
        with (
            caplog.at_level(logging.INFO, logger="basecradle_harness"),
            pytest.raises(ProviderTruncatedError),
        ):
            adapter.chat([Message.user("post a poem")])

    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm ")]
    assert "outcome=fallback" in line and "reason=truncated" in line and "tokens_in=" in line
    adapter.close()


def test_anthropic_text_cut_off_at_the_cap_is_still_an_answer():
    """Issue #490's half, untouched: cut-off text is kept, and the engine leaves the turn
    unfinished rather than retrying it."""
    adapter = AnthropicProvider(CLAUDE, api_key=ANTHROPIC_KEY, max_tokens=60)
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{ANTHROPIC_BASE}/v1/messages").mock(
            return_value=_claude_reply(
                {"type": "text", "text": "half a sente"}, stop_reason="max_tokens"
            )
        )
        reply = adapter.chat([Message.user("go")])
    assert reply.content == "half a sente"
    adapter.close()


def test_a_connection_dropped_mid_body_on_openrouter_is_a_retryable_transport_fault():
    """The defect the loopback proof surfaced behind the deadline: the adapter's citation capture
    swallowed a failed body read, leaving the stream consumed, and the SDK's own read then failed on
    ``StreamConsumed``, which no mapper knows. A retryable fault became a fatal one."""
    from basecradle_harness import ProviderConnectionError

    class Dropped(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"id": "gen-1'
            raise httpx.RemoteProtocolError("peer closed connection without sending complete body")

    provider = OpenRouterProvider("z-ai/glm-5.3", api_key=FAKE_KEY)
    with respx.mock(assert_all_called=True) as router:
        router.post(OPENROUTER_CHAT).mock(return_value=httpx.Response(200, stream=Dropped()))
        with pytest.raises(ProviderConnectionError):
            provider.chat([Message.user("hi")])


@pytest.mark.parametrize(("billed", "reason"), [(30, "truncated"), (29, "invalid_response")])
def test_an_answer_billed_at_its_cap_is_truncated_whatever_its_stop_reason_says(
    caplog, billed, reason
):
    """Measured live on 2026-11-03: cut off at ``max_tokens`` inside a tool call, OpenAI's model
    behind OpenRouter reports ``finish_reason: "tool_calls"`` with exactly the cap billed. The bill
    is the fact; one token under the cap is a body that did not parse, and stays that."""
    cut = wire_tool_call(id="call_0", name="post", arguments='{"body": "Lighthouse stands')
    body = openrouter_completion(tool_calls=[cut], finish_reason="tool_calls")
    body["usage"] = {"prompt_tokens": 45, "completion_tokens": billed, "total_tokens": 45 + billed}
    provider = OpenRouterProvider("openai/gpt-4.1-mini", api_key=FAKE_KEY, max_tokens=30)
    with respx.mock(assert_all_called=True) as router:
        router.post(OPENROUTER_CHAT).mock(return_value=httpx.Response(200, json=body))
        with (
            caplog.at_level(logging.INFO, logger="basecradle_harness"),
            pytest.raises(ProviderResponseError) as raised,
        ):
            provider.chat([Message.user("post a poem")])

    assert isinstance(raised.value, ProviderTruncatedError) is (reason == "truncated")
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm ")]
    assert f"reason={reason}" in line


def test_an_uncapped_call_is_judged_by_its_stop_reason_alone():
    def read():
        raise ValueError("unterminated string")

    usage = {"completion_tokens": 131_072}
    _reply, error = read_answer(read, {"choices": [{"finish_reason": "stop"}]}, usage=usage)
    assert not isinstance(error, ProviderTruncatedError)
