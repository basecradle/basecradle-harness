"""The OpenAI Responses adapter, behind the same `Provider` seam as the chat one.

Every test mocks the HTTP transport with respx — no model is ever called. The
shapes here are OpenAI's Responses API schema: `input` items in, an `output`
array out, the built-in `web_search` tool resolved server-side, and custom
function tools still looping through the harness.
"""

import json

import httpx
import pytest

from basecradle_harness import (
    HarnessError,
    ImageContent,
    Message,
    OpenAIResponsesProvider,
    Provider,
    ProviderAPIError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ToolCall,
    ToolSpec,
)
from basecradle_harness._responses import DEFAULT_BASE_URL
from tests.conftest import (
    BASE_URL,
    FAKE_KEY,
    RESPONSES_URL,
    out_function_call,
    out_message,
    out_web_search_call,
    responses_body,
    url_citation,
)

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


def test_chat_returns_assistant_text(router, responses_provider):
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("Hello, peer.")))
    )

    reply = responses_provider.chat([Message.user("Hi")])

    assert reply.role == "assistant"
    assert reply.content == "Hello, peer."
    assert reply.tool_calls == []


def test_request_targets_responses_with_model_and_input(router, responses_provider):
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    responses_provider.chat([Message.system("be terse"), Message.user("Hi")])

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "gpt-5.4-mini"
    # Chat's `messages` becomes Responses' `input` items — and the charter's
    # `system` role maps to Responses' first-class `developer` instruction role.
    assert body["input"] == [
        {"role": "developer", "content": "be terse"},
        {"role": "user", "content": "Hi"},
    ]


def test_system_charter_maps_to_the_developer_role(router, responses_provider):
    """A system turn (the agent's charter) lands in Responses' `developer` role."""
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    responses_provider.chat([Message.system("You are Nova."), Message.user("Hi")])

    body = json.loads(route.calls.last.request.content)
    roles = [item.get("role") for item in body["input"]]
    assert roles == ["developer", "user"]


def test_authorization_header_carries_the_key(router, responses_provider):
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    responses_provider.chat([Message.user("Hi")])

    assert route.calls.last.request.headers["Authorization"] == f"Bearer {FAKE_KEY}"


# --- The built-in web_search tool (the reason this adapter exists) -----------


def test_web_search_is_enabled_by_default(router, responses_provider):
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    responses_provider.chat([Message.user("news?")])

    body = json.loads(route.calls.last.request.content)
    assert body["tools"] == [{"type": "web_search"}]


def test_web_search_result_surfaces_text_and_citations(router, responses_provider):
    """A server-side web_search turn: its call item is ignored, its answer + sources kept."""
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200,
            json=responses_body(
                out_web_search_call(query="who won"),
                out_message(
                    "Spain won.",
                    annotations=[
                        url_citation(url="https://news.test/spain", title="Spain wins"),
                    ],
                ),
            ),
        )
    )

    reply = responses_provider.chat([Message.user("Who won the cup?")])

    # The web_search_call item is server-side noise — never a tool call for the engine.
    assert reply.tool_calls == []
    # The answer carries a Sources footer built from the url_citation annotations.
    assert reply.content == "Spain won.\n\nSources:\n- Spain wins — https://news.test/spain"


def test_citations_are_deduplicated_by_url(router, responses_provider):
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200,
            json=responses_body(
                out_message(
                    "Two sources, one repeated.",
                    annotations=[
                        url_citation(url="https://a.test", title="A"),
                        url_citation(url="https://a.test", title="A again"),
                        url_citation(url="https://b.test", title="B"),
                    ],
                )
            ),
        )
    )

    reply = responses_provider.chat([Message.user("?")])

    assert reply.content == (
        "Two sources, one repeated.\n\nSources:\n- A — https://a.test\n- B — https://b.test"
    )


def test_no_citations_means_no_sources_footer(router, responses_provider):
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("Plain answer.")))
    )

    reply = responses_provider.chat([Message.user("?")])

    assert reply.content == "Plain answer."


def test_builtin_tools_are_configurable(router):
    """The built-in set is a registration seam: name a built-in, it shows up by type."""
    provider = OpenAIResponsesProvider(
        model="gpt-5.4-mini",
        api_key=FAKE_KEY,
        base_url=BASE_URL,
        builtin_tools=["web_search", {"type": "image_generation", "size": "1024x1024"}],
    )
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    provider.chat([Message.user("draw a cat")])

    body = json.loads(route.calls.last.request.content)
    assert body["tools"] == [
        {"type": "web_search"},
        {"type": "image_generation", "size": "1024x1024"},
    ]
    provider.close()


def test_builtin_tools_can_be_disabled(router):
    provider = OpenAIResponsesProvider(
        model="gpt-5.4-mini", api_key=FAKE_KEY, base_url=BASE_URL, builtin_tools=[]
    )
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    provider.chat([Message.user("Hi")])

    body = json.loads(route.calls.last.request.content)
    # No built-ins and no function tools → no tools key at all.
    assert "tools" not in body
    provider.close()


# --- Custom function tools (still loop through the harness) -------------------


def test_function_tools_use_the_flat_responses_shape(router, responses_provider):
    """Responses flattens the function tool — no nested `function` key — alongside web_search."""
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    responses_provider.chat([Message.user("weather?")], tools=[WEATHER_TOOL])

    body = json.loads(route.calls.last.request.content)
    assert body["tools"] == [
        {"type": "web_search"},
        {
            "type": "function",
            "name": "get_weather",
            "description": "Look up the weather for a city.",
            "parameters": WEATHER_TOOL.parameters,
        },
    ]


def test_function_call_is_parsed_to_a_tool_call(router, responses_provider):
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200,
            json=responses_body(
                out_function_call(
                    call_id="call_1", name="get_weather", arguments={"city": "Dallas"}
                )
            ),
        )
    )

    reply = responses_provider.chat([Message.user("weather?")], tools=[WEATHER_TOOL])

    assert reply.content is None
    assert reply.tool_calls == [
        ToolCall(id="call_1", name="get_weather", arguments={"city": "Dallas"})
    ]
    assert isinstance(reply.tool_calls[0].arguments, dict)


def test_web_search_and_function_call_coexist_in_one_turn(router, responses_provider):
    """The hybrid: web_search resolved server-side, a function call still handed back."""
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200,
            json=responses_body(
                out_web_search_call(query="dallas weather source"),
                out_message(
                    "Let me check the live reading.",
                    annotations=[url_citation(url="https://wx.test", title="Weather")],
                ),
                out_function_call(
                    call_id="call_9", name="get_weather", arguments={"city": "Dallas"}
                ),
            ),
        )
    )

    reply = responses_provider.chat([Message.user("weather in Dallas?")], tools=[WEATHER_TOOL])

    # Text + citation from the message item...
    assert reply.content == (
        "Let me check the live reading.\n\nSources:\n- Weather — https://wx.test"
    )
    # ...and the custom function call the engine must run (web_search is NOT among them).
    assert reply.tool_calls == [
        ToolCall(id="call_9", name="get_weather", arguments={"city": "Dallas"})
    ]


def test_assistant_tool_calls_and_results_serialize_back_to_input(router, responses_provider):
    """An assistant function-call turn and its result round-trip into Responses input items."""
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("It's sunny.")))
    )

    history = [
        Message.user("weather?"),
        Message.assistant(
            tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Dallas"})]
        ),
        Message.tool(tool_call_id="call_1", content="sunny, 88F"),
    ]
    responses_provider.chat(history)

    body = json.loads(route.calls.last.request.content)
    assert body["input"] == [
        {"role": "user", "content": "weather?"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": json.dumps({"city": "Dallas"}),
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny, 88F"},
    ]


def test_assistant_text_and_tool_call_emit_two_input_items(router, responses_provider):
    """An assistant turn that both spoke and called a tool becomes a message + a function_call."""
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("done")))
    )

    history = [
        Message.assistant(
            content="Checking now.",
            tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "Dallas"})],
        ),
    ]
    responses_provider.chat(history)

    body = json.loads(route.calls.last.request.content)
    assert body["input"] == [
        {"role": "assistant", "content": "Checking now."},
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "get_weather",
            "arguments": json.dumps({"city": "Dallas"}),
        },
    ]


def test_full_tool_round_trip(router, responses_provider):
    """Model asks for a tool, gets the result, then answers — two Responses calls."""
    router.post(RESPONSES_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json=responses_body(
                    out_function_call(
                        call_id="call_1", name="get_weather", arguments={"city": "Dallas"}
                    )
                ),
            ),
            httpx.Response(200, json=responses_body(out_message("It's sunny in Dallas."))),
        ]
    )

    history = [Message.user("weather in Dallas?")]
    first = responses_provider.chat(history, tools=[WEATHER_TOOL])
    assert first.tool_calls[0].name == "get_weather"

    history.append(first)
    history.append(Message.tool(tool_call_id="call_1", content="sunny, 88F"))
    second = responses_provider.chat(history, tools=[WEATHER_TOOL])

    assert second.tool_calls == []
    assert second.content == "It's sunny in Dallas."


# --- Vision: images become input_image parts ---------------------------------


def test_a_message_with_an_image_serializes_to_input_parts(router, responses_provider):
    """A turn carrying an image becomes a parts list: input_text + input_image."""
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("A tabby cat.")))
    )

    history = [
        Message.user("What's in this image?"),
        Message(
            role="user",
            content="(Showing image: cat.png)",
            images=[ImageContent(url="data:image/png;base64,AAAA", alt="cat.png")],
        ),
    ]
    responses_provider.chat(history)

    body = json.loads(route.calls.last.request.content)
    assert body["input"] == [
        {"role": "user", "content": "What's in this image?"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "(Showing image: cat.png)"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            ],
        },
    ]


def test_an_image_only_turn_omits_the_input_text_part(router, responses_provider):
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    history = [
        Message(role="user", images=[ImageContent(url="https://img.test/a.png")]),
    ]
    responses_provider.chat(history)

    body = json.loads(route.calls.last.request.content)
    assert body["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": "https://img.test/a.png"}],
        },
    ]


def test_a_plain_text_turn_still_serializes_as_a_string(router, responses_provider):
    """No images → content stays a bare string, exactly as before (no regression)."""
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    responses_provider.chat([Message.user("plain text")])

    body = json.loads(route.calls.last.request.content)
    assert body["input"] == [{"role": "user", "content": "plain text"}]


# --- Errors ------------------------------------------------------------------


def test_401_raises_provider_auth_error(router, responses_provider):
    router.post(RESPONSES_URL).mock(return_value=httpx.Response(401, text="bad key"))

    with pytest.raises(ProviderAuthError) as exc:
        responses_provider.chat([Message.user("Hi")])

    assert exc.value.status_code == 401
    assert exc.value.body == "bad key"
    assert isinstance(exc.value, ProviderAPIError)
    assert isinstance(exc.value, HarnessError)


def test_429_raises_rate_limit_with_retry_after(router, responses_provider):
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "30"}, text="slow down")
    )

    with pytest.raises(ProviderRateLimitError) as exc:
        responses_provider.chat([Message.user("Hi")])

    assert exc.value.status_code == 429
    assert exc.value.retry_after == 30.0


def test_500_raises_provider_api_error_keeping_the_body(router, responses_provider):
    router.post(RESPONSES_URL).mock(return_value=httpx.Response(500, text="boom"))

    with pytest.raises(ProviderAPIError) as exc:
        responses_provider.chat([Message.user("Hi")])

    assert exc.value.status_code == 500
    assert exc.value.body == "boom"
    assert not isinstance(exc.value, (ProviderAuthError, ProviderRateLimitError))


def test_transport_failure_raises_connection_error(router, responses_provider):
    router.post(RESPONSES_URL).mock(side_effect=httpx.ConnectError("no route"))

    with pytest.raises(ProviderConnectionError):
        responses_provider.chat([Message.user("Hi")])


def test_malformed_response_raises_provider_error(router, responses_provider):
    router.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json={"unexpected": True}))

    with pytest.raises(ProviderError):
        responses_provider.chat([Message.user("Hi")])


def test_unparseable_tool_arguments_raise_provider_error(router, responses_provider):
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200,
            json=responses_body(
                {
                    "type": "function_call",
                    "call_id": "call_x",
                    "name": "get_weather",
                    "arguments": "{not json",
                }
            ),
        )
    )

    with pytest.raises(ProviderError):
        responses_provider.chat([Message.user("weather?")], tools=[WEATHER_TOOL])


# --- Construction & configuration -------------------------------------------


def test_api_key_falls_back_to_env(monkeypatch, router):
    monkeypatch.setenv("AI_PROVIDER_API_KEY", "sk-env-key")
    provider = OpenAIResponsesProvider(model="gpt-5.4-mini", base_url=BASE_URL)
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    provider.chat([Message.user("Hi")])

    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-env-key"
    provider.close()


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="AI_PROVIDER_API_KEY"):
        OpenAIResponsesProvider(model="gpt-5.4-mini")


def test_default_base_url_is_openai():
    provider = OpenAIResponsesProvider(model="gpt-5.4-mini", api_key=FAKE_KEY)
    assert provider.base_url == DEFAULT_BASE_URL
    provider.close()


def test_default_params_pass_through(router):
    provider = OpenAIResponsesProvider(
        model="gpt-5.4-mini", api_key=FAKE_KEY, base_url=BASE_URL, temperature=0.2
    )
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    provider.chat([Message.user("Hi")])

    body = json.loads(route.calls.last.request.content)
    assert body["temperature"] == 0.2
    assert body["model"] == "gpt-5.4-mini"
    provider.close()


def test_context_manager_closes(router):
    with OpenAIResponsesProvider(
        model="gpt-5.4-mini", api_key=FAKE_KEY, base_url=BASE_URL
    ) as provider:
        router.post(RESPONSES_URL).mock(
            return_value=httpx.Response(200, json=responses_body(out_message("ok")))
        )
        assert provider.chat([Message.user("Hi")]).content == "ok"


# --- The one-protocol promise ------------------------------------------------


def test_satisfies_the_provider_protocol(responses_provider):
    """The same `Provider` seam — the engine cannot tell the two adapters apart."""
    assert isinstance(responses_provider, Provider)
