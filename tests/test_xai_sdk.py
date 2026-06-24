"""The native xAI adapter (`XaiSdkProvider`), behind the provider-agnostic `Provider` seam.

The harness reaches a model **only through a vendor SDK** (issue #158); this is the native xAI
adapter for ``AI_SDK=xai-sdk`` (issue #165). The xai-sdk is gRPC, so there is no httpx transport to
respx-mock — instead each test injects a **fake client** and drives the **real** ``xai_sdk`` wire
helpers (real ``chat_pb2`` protos) into it, so the message/tool/search translation is exercised
against the genuine SDK without ever opening a socket. Response parsing runs against a duck-typed
fake `Response` (the SDK's `Response` is a thin proto reader; the adapter only touches
``.content`` / ``.tool_calls`` / ``.citations``).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from basecradle_harness import (
    ImageContent,
    Message,
    Provider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ToolCall,
    ToolSpec,
    XaiSdkProvider,
)

FAKE_KEY = "xai-test-0123456789abcdef"

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Look up the weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


# --- fakes: an injected client + a duck-typed Response -----------------------


class _FakeConversation:
    def __init__(self, response):
        self._response = response

    def sample(self):
        return self._response


class _FakeChatClient:
    """Stands in for ``client.chat``: records the create() payload, returns a canned Response."""

    def __init__(self, response):
        self._response = response
        self.captured: dict | None = None

    def create(self, **kwargs):
        self.captured = kwargs
        return _FakeConversation(self._response)


class _FakeClient:
    def __init__(self, response):
        self.chat = _FakeChatClient(response)
        self.closed = False

    def close(self):
        self.closed = True


def _response(*, content="", tool_calls=(), citations=()):
    """A duck-typed xai_sdk `Response` — only the fields the adapter reads."""
    calls = [
        SimpleNamespace(
            id=c["id"],
            function=SimpleNamespace(name=c["name"], arguments=json.dumps(c["arguments"])),
        )
        for c in tool_calls
    ]
    return SimpleNamespace(content=content, tool_calls=calls, citations=list(citations))


def _provider(response):
    return XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, client=_FakeClient(response))


def _text(msg) -> str:
    """The text content of an xai_sdk chat message proto (content is a repeated Content)."""
    return "".join(c.text for c in msg.content if c.WhichOneof("content") == "text")


_ROLE = {"user": 1, "assistant": 2, "system": 3, "tool": 5, "developer": 6}


# --- the adapter -------------------------------------------------------------


def test_satisfies_the_provider_protocol():
    assert isinstance(_provider(_response(content="hi")), Provider)


def test_chat_returns_assistant_text():
    provider = _provider(_response(content="Hello, peer."))
    reply = provider.chat([Message.user("Hi")])
    assert reply.role == "assistant"
    assert reply.content == "Hello, peer."
    assert reply.tool_calls == []


def test_sends_model_and_maps_roles_to_the_wire():
    provider = _provider(_response(content="ok"))
    provider.chat([Message.system("be terse"), Message.user("Hi")])

    captured = provider._client.chat.captured
    assert captured["model"] == "grok-4.3"
    roles = [m.role for m in captured["messages"]]
    assert roles == [_ROLE["system"], _ROLE["user"]]
    assert _text(captured["messages"][0]) == "be terse"
    assert _text(captured["messages"][1]) == "Hi"


def test_a_tool_spec_becomes_a_wire_tool_and_a_tool_call_round_trips():
    provider = _provider(
        _response(
            tool_calls=[{"id": "call_9", "name": "get_weather", "arguments": {"city": "Dallas"}}]
        )
    )
    reply = provider.chat([Message.user("weather?")], tools=[WEATHER_TOOL])

    # request: the ToolSpec became a native Tool with the right function name + schema.
    tools = provider._client.chat.captured["tools"]
    assert tools[0].function.name == "get_weather"
    # response: the SDK tool call became a harness ToolCall with decoded arguments.
    assert reply.content is None
    assert reply.tool_calls == [
        ToolCall(id="call_9", name="get_weather", arguments={"city": "Dallas"})
    ]


def test_an_assistant_tool_call_and_its_result_round_trip_in_history():
    # The engine sends the whole transcript back each turn: an assistant turn that *made* a tool
    # call, then the tool result. Both must serialize so grok sees the linkage by id.
    provider = _provider(_response(content="It's sunny in Dallas."))
    history = [
        Message.user("weather?"),
        Message.assistant(
            tool_calls=[ToolCall(id="call_9", name="get_weather", arguments={"city": "Dallas"})]
        ),
        Message.tool(tool_call_id="call_9", content="sunny, 75F"),
    ]
    provider.chat(history)

    msgs = provider._client.chat.captured["messages"]
    assert [m.role for m in msgs] == [_ROLE["user"], _ROLE["assistant"], _ROLE["tool"]]
    assistant = msgs[1]
    assert assistant.tool_calls[0].id == "call_9"
    assert assistant.tool_calls[0].function.name == "get_weather"
    assert json.loads(assistant.tool_calls[0].function.arguments) == {"city": "Dallas"}
    assert msgs[2].tool_call_id == "call_9"  # the tool result references the call


def test_vision_image_becomes_an_image_part():
    provider = _provider(_response(content="I see a cat."))
    turn = Message.user("what's this?")
    turn.images = [ImageContent(url="data:image/png;base64,AAAA", alt="cat.png")]
    provider.chat([turn])

    content = provider._client.chat.captured["messages"][0].content
    kinds = [c.WhichOneof("content") for c in content]
    assert "text" in kinds and "image_url" in kinds
    image_part = next(c for c in content if c.WhichOneof("content") == "image_url")
    assert image_part.image_url.image_url == "data:image/png;base64,AAAA"


def test_opted_in_search_builtins_become_agent_tools():
    # Issue #171: the search built-ins are xAI Agent Tools appended to the chat `tools` list (the
    # deprecated native `search_parameters` path is gone). Each is a real `chat_pb2.Tool` proto.
    provider = XaiSdkProvider(
        "grok-4.3",
        api_key=FAKE_KEY,
        client=_FakeClient(_response(content="news")),
        builtin_tools=["web_search", "x_search"],
    )
    provider.chat([Message.user("news?")])

    captured = provider._client.chat.captured
    assert "search_parameters" not in captured  # the deprecated field is never sent
    kinds = [t.WhichOneof("tool") for t in captured["tools"]]
    assert kinds == ["web_search", "x_search"]


def test_search_builtins_coexist_with_function_tools_in_one_list():
    # Function tools and search Agent Tools share the single native `tools` list (both are Tools).
    provider = XaiSdkProvider(
        "grok-4.3",
        api_key=FAKE_KEY,
        client=_FakeClient(_response(content="ok")),
        builtin_tools=["web_search"],
    )
    provider.chat([Message.user("weather then news?")], tools=[WEATHER_TOOL])

    tools = provider._client.chat.captured["tools"]
    assert tools[0].WhichOneof("tool") == "function"
    assert tools[0].function.name == "get_weather"
    assert tools[1].WhichOneof("tool") == "web_search"


def test_no_search_builtins_sends_no_search_tool():
    provider = _provider(_response(content="hi"))
    provider.chat([Message.user("hi")])
    captured = provider._client.chat.captured
    assert "search_parameters" not in captured
    assert "tools" not in captured  # no function tools and no search built-ins -> no tools at all


def test_live_search_citations_footer_the_reply():
    provider = _provider(
        _response(
            content="Here is the news.",
            citations=["https://ex.com/a", "https://ex.com/a", "https://ex.com/b"],
        )
    )
    reply = provider.chat([Message.user("news?")])
    assert "Here is the news." in reply.content
    assert "Sources:" in reply.content
    assert reply.content.count("https://ex.com/a") == 1  # deduped
    assert "https://ex.com/b" in reply.content


def test_missing_api_key_raises_without_a_client():
    with pytest.raises(ValueError, match="API key"):
        XaiSdkProvider("grok-4.3", api_key=None)


def test_close_closes_the_client():
    provider = _provider(_response(content="hi"))
    provider.close()
    assert provider._client.closed is True


# --- gRPC errors -> the harness provider hierarchy ---------------------------

import grpc  # noqa: E402 - imported here, beside the error tests it serves (ships with xai-sdk)


class _FakeRpcError(grpc.RpcError):
    """A real ``grpc.RpcError`` subclass carrying a status code + details, like the SDK raises."""

    def __init__(self, code, details="boom"):
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class _RaisingClient:
    def __init__(self, error):
        self.chat = SimpleNamespace(create=self._raise)
        self._error = error

    def _raise(self, **kwargs):
        raise self._error

    def close(self):
        pass


def _provider_raising(code):
    return XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, client=_RaisingClient(_FakeRpcError(code)))


def test_grpc_unauthenticated_maps_to_auth_error():
    with pytest.raises(ProviderAuthError):
        _provider_raising(grpc.StatusCode.UNAUTHENTICATED).chat([Message.user("hi")])


def test_grpc_resource_exhausted_maps_to_rate_limit():
    with pytest.raises(ProviderRateLimitError):
        _provider_raising(grpc.StatusCode.RESOURCE_EXHAUSTED).chat([Message.user("hi")])


def test_grpc_unavailable_maps_to_connection_error():
    with pytest.raises(ProviderConnectionError):
        _provider_raising(grpc.StatusCode.UNAVAILABLE).chat([Message.user("hi")])
