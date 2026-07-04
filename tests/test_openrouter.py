"""The native OpenRouter adapter (`OpenRouterProvider`), behind the `Provider` seam.

The harness reaches a model **only through a vendor SDK** (issue #158); this is the native
OpenRouter adapter for ``AI_SDK=openrouter`` (issue #234). The ``openrouter`` SDK is httpx-backed,
so every test mocks the transport with respx — and because respx patches httpx at the transport
level, it intercepts the SDK's *own* httpx client, driving the **real SDK** against SDK-valid
response bodies without ever touching the network. OpenRouter speaks the OpenAI chat wire, so the
adapter reuses the shared `_openai_wire` translation; the round-trip test proves the SDK's
client-side Pydantic marshalling accepts those wire dicts (including ``content: None`` assistant
turns and tool results).

The SDK ships a default 5xx retry (backoff up to an hour), so the error tests inject a client with
``retry_config=None`` — a single attempt, no stall.
"""

from __future__ import annotations

import json

import httpx
import pytest
from openrouter import OpenRouter

from basecradle_harness import (
    Message,
    OpenRouterProvider,
    Provider,
    ProviderAPIError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ToolCall,
    ToolSpec,
)

# A fabricated OpenRouter endpoint + a correctly-shaped fake key. The SDK posts to
# ``<server_url>/chat/completions`` — verified against the real 0.11.3 SDK.
BASE_URL = "https://openrouter.test/api/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
FAKE_KEY = "sk-or-v1-0123456789abcdef0123456789abcdef"
MODEL = "z-ai/glm-5.2"

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Look up the weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


# --- response builders (OpenAI-chat shaped; the SDK requires system_fingerprint) --------------


def completion(*, content=None, tool_calls=None, finish_reason="stop"):
    """A chat-completions response body the ``openrouter`` SDK's ``ChatResult`` accepts.

    ``system_fingerprint`` is a **required** field of the SDK's ``ChatResult`` model (verified
    against 0.11.3) — omit it and the SDK raises ``ResponseValidationError`` before the adapter
    ever sees the body.
    """
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "gen-fake0001",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "system_fingerprint": "fp_test",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def wire_tool_call(*, id, name, arguments):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _provider(*, retries_disabled=False, **kw):
    """An `OpenRouterProvider` over an injected SDK client (respx mocks its httpx transport).

    ``retries_disabled`` builds the client with ``retry_config=None`` so an error test is not
    stalled by the SDK's default 5xx backoff.
    """
    client_kwargs = {"api_key": FAKE_KEY, "server_url": BASE_URL}
    if retries_disabled:
        client_kwargs["retry_config"] = None
    client = OpenRouter(**client_kwargs)
    return OpenRouterProvider(MODEL, client=client, base_url=BASE_URL, **kw)


@pytest.fixture
def router():
    import respx

    with respx.mock(assert_all_called=True) as r:
        yield r


# === Happy path ===============================================================


def test_chat_returns_assistant_text(router):
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="Hello, peer."))
    )
    provider = _provider()
    reply = provider.chat([Message.user("Hi")])
    assert reply.role == "assistant"
    assert reply.content == "Hello, peer."
    provider.close()


def test_request_sends_model_and_messages(router):
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )
    provider = _provider()
    provider.chat([Message.user("What's up?")])
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == MODEL
    assert body["messages"] == [{"role": "user", "content": "What's up?"}]
    provider.close()


def test_authorization_header_carries_the_key(router):
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(content="ok")))
    provider = _provider()
    provider.chat([Message.user("Hi")])
    assert router.calls.last.request.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    provider.close()


def test_tools_are_serialized_to_the_function_shape(router):
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )
    provider = _provider()
    provider.chat([Message.user("Weather?")], tools=[WEATHER_TOOL])
    body = json.loads(route.calls.last.request.content)
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "get_weather"
    provider.close()


def test_tool_call_arguments_are_parsed_to_a_dict(router):
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=completion(
                tool_calls=[
                    wire_tool_call(id="call_1", name="get_weather", arguments='{"city": "Dallas"}')
                ],
                finish_reason="tool_calls",
            ),
        )
    )
    provider = _provider()
    reply = provider.chat([Message.user("Weather in Dallas?")])
    assert reply.tool_calls == [
        ToolCall(id="call_1", name="get_weather", arguments={"city": "Dallas"})
    ]
    provider.close()


def test_assistant_tool_calls_and_tool_results_serialize_back(router):
    """A full round trip: an assistant ``tool_calls`` turn (``content: None``) and a ``tool``
    result go back over the wire — proving the SDK's client-side Pydantic marshalling accepts the
    shared ``_openai_wire`` dicts.
    """
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="It's sunny."))
    )
    provider = _provider()
    history = [
        Message.user("Weather in Dallas?"),
        Message.assistant(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Dallas"})],
        ),
        Message.tool(tool_call_id="call_1", content="Sunny, 95F"),
    ]
    reply = provider.chat(history)
    assert reply.content == "It's sunny."
    body = json.loads(route.calls.last.request.content)
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "tool"]
    # The assistant turn carried its tool call; the tool turn answered by id.
    assert body["messages"][1]["tool_calls"][0]["id"] == "call_1"
    assert body["messages"][2]["tool_call_id"] == "call_1"
    provider.close()


def test_default_params_pass_through(router):
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )
    provider = _provider(temperature=0.7, max_tokens=4096, reasoning={"effort": "high"})
    provider.chat([Message.user("Hi")])
    body = json.loads(route.calls.last.request.content)
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 4096
    assert body["reasoning"] == {"effort": "high"}
    provider.close()


# === Errors ===================================================================


def test_unknown_param_key_is_an_actionable_error():
    # chat.send is typed with no **kwargs — an unknown key raises TypeError at call time, which
    # the adapter converts into a ProviderError naming model_params.json (no HTTP happens).
    provider = _provider(not_a_real_param=1)
    with pytest.raises(ProviderError, match="model_params.json"):
        provider.chat([Message.user("Hi")])
    provider.close()


def test_internal_typeerror_is_not_misattributed_to_model_params():
    # Only an *unexpected keyword argument* TypeError (the kwarg-splat rejection) is reframed as a
    # model_params error. A TypeError from deep in the SDK's own marshalling must propagate as-is,
    # never blamed on a possibly-empty model_params.json.
    class _BoomChat:
        def send(self, **kwargs):
            raise TypeError("internal marshalling boom")

    class _BoomClient:
        chat = _BoomChat()

    provider = OpenRouterProvider(MODEL, client=_BoomClient(), base_url=BASE_URL)
    with pytest.raises(TypeError, match="internal marshalling boom"):
        provider.chat([Message.user("Hi")])
    provider.close()


def test_response_validation_error_maps_to_provider_error_not_a_fake_status(router):
    # A 200 whose body fails the SDK's ChatResult validation raises ResponseValidationError (an
    # OpenRouterError subclass carrying no HTTP-error status). It must surface as a plain
    # ProviderError, never a ProviderAPIError stamped with a misleading status_code like 200.
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json={"id": "gen-x", "object": "chat.completion"})
    )
    provider = _provider(retries_disabled=True)
    with pytest.raises(ProviderError) as exc:
        provider.chat([Message.user("Hi")])
    assert not isinstance(exc.value, ProviderAPIError)
    provider.close()


def test_401_maps_to_auth_error(router):
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key", "code": 401}})
    )
    provider = _provider(retries_disabled=True)
    with pytest.raises(ProviderAuthError) as exc:
        provider.chat([Message.user("Hi")])
    assert exc.value.status_code == 401
    provider.close()


def test_429_maps_to_rate_limit_with_retry_after(router):
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "12"},
            json={"error": {"message": "slow down", "code": 429}},
        )
    )
    provider = _provider(retries_disabled=True)
    with pytest.raises(ProviderRateLimitError) as exc:
        provider.chat([Message.user("Hi")])
    assert exc.value.status_code == 429
    assert exc.value.retry_after == 12.0
    provider.close()


def test_500_maps_to_api_error_keeping_the_body(router):
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(500, json={"error": {"message": "boom", "code": 500}})
    )
    provider = _provider(retries_disabled=True)
    with pytest.raises(ProviderAPIError) as exc:
        provider.chat([Message.user("Hi")])
    assert exc.value.status_code == 500
    assert "boom" in exc.value.body
    provider.close()


def test_transport_failure_maps_to_connection_error(router):
    router.post(CHAT_URL).mock(side_effect=httpx.ConnectError("no route to host"))
    provider = _provider(retries_disabled=True)
    with pytest.raises(ProviderConnectionError):
        provider.chat([Message.user("Hi")])
    provider.close()


def test_missing_sdk_is_a_clear_no_llm_error(monkeypatch):
    """With the ``openrouter`` package unimportable, construction fails loud — "no LLM, by design"."""
    import builtins

    real_import = builtins.__import__

    def _no_openrouter(name, *args, **kwargs):
        if name == "openrouter":
            raise ModuleNotFoundError("No module named 'openrouter'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_openrouter)
    with pytest.raises(ProviderError, match="openrouter"):
        OpenRouterProvider(MODEL, api_key=FAKE_KEY)


# === Construction =============================================================


def test_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "sk-or-env-key")
    # No client injected → the adapter builds the SDK client from the env key (no call made).
    provider = OpenRouterProvider(MODEL, base_url=BASE_URL)
    assert isinstance(provider, OpenRouterProvider)
    provider.close()


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        OpenRouterProvider(MODEL, base_url=BASE_URL)


def test_default_base_url_is_openrouter():
    provider = OpenRouterProvider(MODEL, api_key=FAKE_KEY)
    assert provider.base_url == "https://openrouter.ai/api/v1"
    provider.close()


# === The one-class promise ===================================================


def test_satisfies_the_provider_protocol():
    provider = _provider()
    assert isinstance(provider, Provider)
    provider.close()
