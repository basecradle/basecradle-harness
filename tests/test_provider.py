"""The OpenAI-compatible adapter, behind the provider-agnostic `Provider` seam.

Every test mocks the HTTP transport with respx — no model is ever called. The
shapes here are the OpenAI chat-completions schema, which OpenRouter and xAI
mirror; only `base_url` / `api_key` / `model` change between them.
"""

import json

import httpx
import pytest

from basecradle_harness import (
    HarnessError,
    Message,
    OpenAICompatibleProvider,
    Provider,
    ProviderAPIError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ToolCall,
    ToolSpec,
)
from basecradle_harness._openai import DEFAULT_BASE_URL
from tests.conftest import BASE_URL, CHAT_URL, FAKE_KEY, completion, wire_tool_call

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Look up the weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


# --- A plain text turn -------------------------------------------------------


def test_chat_returns_assistant_text(router, provider):
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="Hello, peer."))
    )

    reply = provider.chat([Message.user("Hi")])

    assert reply.role == "assistant"
    assert reply.content == "Hello, peer."
    assert reply.tool_calls == []


def test_request_sends_model_and_messages(router, provider):
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )

    provider.chat([Message.system("be terse"), Message.user("Hi")])

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "gpt-4o"
    assert body["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "Hi"},
    ]
    # No tools offered → no tools key at all.
    assert "tools" not in body


def test_authorization_header_carries_the_key(router, provider):
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )

    provider.chat([Message.user("Hi")])

    assert route.calls.last.request.headers["Authorization"] == f"Bearer {FAKE_KEY}"


# --- Tool calling ------------------------------------------------------------


def test_tool_call_arguments_are_parsed_to_a_dict(router, provider):
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=completion(
                tool_calls=[
                    wire_tool_call(id="call_1", name="get_weather", arguments={"city": "Dallas"})
                ],
                finish_reason="tool_calls",
            ),
        )
    )

    reply = provider.chat([Message.user("weather?")], tools=[WEATHER_TOOL])

    assert reply.content is None
    assert reply.tool_calls == [
        ToolCall(id="call_1", name="get_weather", arguments={"city": "Dallas"})
    ]
    # The wire's JSON-string arguments never leak out as a string.
    assert isinstance(reply.tool_calls[0].arguments, dict)


def test_tools_are_serialized_to_the_function_shape(router, provider):
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )

    provider.chat([Message.user("weather?")], tools=[WEATHER_TOOL])

    body = json.loads(route.calls.last.request.content)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the weather for a city.",
                "parameters": WEATHER_TOOL.parameters,
            },
        }
    ]


def test_assistant_tool_calls_and_tool_results_serialize_back(router, provider):
    """An assistant tool-call turn and a tool result round-trip onto the wire."""
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="It's sunny."))
    )

    history = [
        Message.user("weather?"),
        Message.assistant(
            tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Dallas"})]
        ),
        Message.tool(tool_call_id="call_1", content="sunny, 88F"),
    ]
    provider.chat(history)

    body = json.loads(route.calls.last.request.content)
    assistant, tool = body["messages"][1], body["messages"][2]
    assert assistant == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": "Dallas"})},
            }
        ],
    }
    assert tool == {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 88F"}


def test_full_tool_round_trip(router, provider):
    """Model asks for a tool, gets the result, then answers — two chat calls."""
    router.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json=completion(
                    tool_calls=[
                        wire_tool_call(
                            id="call_1", name="get_weather", arguments={"city": "Dallas"}
                        )
                    ],
                    finish_reason="tool_calls",
                ),
            ),
            httpx.Response(200, json=completion(content="It's sunny in Dallas.")),
        ]
    )

    history = [Message.user("weather in Dallas?")]
    first = provider.chat(history, tools=[WEATHER_TOOL])
    assert first.tool_calls[0].name == "get_weather"

    # The engine's job, done by hand here: append the call and its result.
    history.append(first)
    history.append(Message.tool(tool_call_id="call_1", content="sunny, 88F"))
    second = provider.chat(history, tools=[WEATHER_TOOL])

    assert second.tool_calls == []
    assert second.content == "It's sunny in Dallas."


# --- Errors ------------------------------------------------------------------


def test_401_raises_provider_auth_error(router, provider):
    router.post(CHAT_URL).mock(return_value=httpx.Response(401, text="bad key"))

    with pytest.raises(ProviderAuthError) as exc:
        provider.chat([Message.user("Hi")])

    assert exc.value.status_code == 401
    assert exc.value.body == "bad key"
    assert isinstance(exc.value, ProviderAPIError)
    assert isinstance(exc.value, HarnessError)


def test_429_raises_rate_limit_with_retry_after(router, provider):
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "30"}, text="slow down")
    )

    with pytest.raises(ProviderRateLimitError) as exc:
        provider.chat([Message.user("Hi")])

    assert exc.value.status_code == 429
    assert exc.value.retry_after == 30.0


def test_500_raises_provider_api_error_keeping_the_body(router, provider):
    router.post(CHAT_URL).mock(return_value=httpx.Response(500, text="boom"))

    with pytest.raises(ProviderAPIError) as exc:
        provider.chat([Message.user("Hi")])

    assert exc.value.status_code == 500
    assert exc.value.body == "boom"
    # Not auth or rate-limit, just the generic API error.
    assert not isinstance(exc.value, (ProviderAuthError, ProviderRateLimitError))


def test_transport_failure_raises_connection_error(router, provider):
    router.post(CHAT_URL).mock(side_effect=httpx.ConnectError("no route"))

    with pytest.raises(ProviderConnectionError):
        provider.chat([Message.user("Hi")])


def test_malformed_response_raises_provider_error(router, provider):
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json={"unexpected": True}))

    with pytest.raises(ProviderError):
        provider.chat([Message.user("Hi")])


# --- Construction & configuration -------------------------------------------


def test_api_key_falls_back_to_env(monkeypatch, router):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
    provider = OpenAICompatibleProvider(model="gpt-4o", base_url=BASE_URL)
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )

    provider.chat([Message.user("Hi")])

    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-env-key"
    provider.close()


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAICompatibleProvider(model="gpt-4o")


def test_default_base_url_is_openai():
    provider = OpenAICompatibleProvider(model="gpt-4o", api_key=FAKE_KEY)
    assert provider.base_url == DEFAULT_BASE_URL
    provider.close()


def test_default_params_pass_through(router):
    provider = OpenAICompatibleProvider(
        model="gpt-4o", api_key=FAKE_KEY, base_url=BASE_URL, temperature=0.2
    )
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )

    provider.chat([Message.user("Hi")])

    body = json.loads(route.calls.last.request.content)
    assert body["temperature"] == 0.2
    assert body["model"] == "gpt-4o"
    provider.close()


def test_context_manager_closes(router):
    with OpenAICompatibleProvider(model="gpt-4o", api_key=FAKE_KEY, base_url=BASE_URL) as provider:
        router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(content="ok")))
        assert provider.chat([Message.user("Hi")]).content == "ok"


# --- The one-class promise ---------------------------------------------------


class EchoProvider:
    """A whole provider in a few lines — the hackability promise, tested.

    Implementing `chat` is the entire contract; nothing inherits from anything.
    """

    def chat(self, messages, tools=None):
        return Message.assistant(content=messages[-1].content)


def test_any_class_with_chat_satisfies_the_protocol(provider):
    assert isinstance(provider, Provider)
    assert isinstance(EchoProvider(), Provider)


def test_a_handwritten_provider_works_through_the_interface():
    p: Provider = EchoProvider()
    assert p.chat([Message.user("ping")]).content == "ping"
