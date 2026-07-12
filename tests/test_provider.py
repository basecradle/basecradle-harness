"""The ``openai`` SDK adapter (`OpenAIProvider`), behind the provider-agnostic `Provider` seam.

The harness reaches a model **only through a vendor SDK** (issue #158); this is that adapter
for ``AI_SDK=openai``. Every test mocks the HTTP transport with respx — and because respx
patches httpx at the transport level, it intercepts the ``openai`` SDK's *own* httpx client, so
these tests drive the **real SDK** against SDK-valid response bodies without ever touching the
network. Both surfaces are covered: ``chat`` (Chat Completions) and ``responses`` (the Responses
API, with the server-side ``web_search`` built-in and vision).
"""

import json
import logging

import httpx
import pytest

from basecradle_harness import (
    HarnessError,
    ImageContent,
    Message,
    OpenAIProvider,
    Provider,
    ProviderAPIError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ToolCall,
    ToolSpec,
)
from basecradle_harness._openai import DEFAULT_BASE_URL
from tests.conftest import (
    BASE_URL,
    CHAT_URL,
    FAKE_KEY,
    RESPONSES_URL,
    completion,
    out_function_call,
    out_message,
    out_web_search_call,
    responses_body,
    url_citation,
    wire_tool_call,
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


def _chat(**kw):
    return OpenAIProvider(
        model="gpt-4o", api_key=FAKE_KEY, base_url=BASE_URL, surface="chat", max_retries=0, **kw
    )


# === Chat surface ============================================================


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
    assert "tools" not in body  # no tools offered → no tools key


def test_authorization_header_carries_the_key(router, provider):
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(content="ok")))

    provider.chat([Message.user("Hi")])

    assert router.calls.last.request.headers["Authorization"] == f"Bearer {FAKE_KEY}"


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

    history.append(first)
    history.append(Message.tool(tool_call_id="call_1", content="sunny, 88F"))
    second = provider.chat(history, tools=[WEATHER_TOOL])

    assert second.tool_calls == []
    assert second.content == "It's sunny in Dallas."


# === Responses surface =======================================================


def test_responses_returns_assistant_text(router, responses_provider):
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("Hello, peer.")))
    )

    reply = responses_provider.chat([Message.user("Hi")])

    assert reply.content == "Hello, peer."
    assert reply.tool_calls == []


def test_responses_targets_responses_with_input_and_developer_role(router, responses_provider):
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    responses_provider.chat([Message.system("be terse"), Message.user("Hi")])

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "gpt-5.4-mini"
    # Chat's `messages` becomes Responses' `input`; `system` maps to the `developer` role.
    assert body["input"] == [
        {"role": "developer", "content": "be terse"},
        {"role": "user", "content": "Hi"},
    ]


def test_web_search_builtin_is_offered_alongside_function_tools(router):
    """A built-in (web_search) and a function tool coexist in one Responses turn."""
    provider = OpenAIProvider(
        model="gpt-5.4-mini",
        api_key=FAKE_KEY,
        base_url=BASE_URL,
        surface="responses",
        max_retries=0,
        builtin_tools=["web_search"],
    )
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    provider.chat([Message.user("news?")], tools=[WEATHER_TOOL])

    tools = json.loads(route.calls.last.request.content)["tools"]
    assert {"type": "web_search"} in tools
    # The function tool is the Responses *flat* shape (no nested `function` key).
    assert {
        "type": "function",
        "name": "get_weather",
        "description": "Look up the weather for a city.",
        "parameters": WEATHER_TOOL.parameters,
    } in tools
    provider.close()


def test_code_interpreter_builtin_carries_an_auto_container_by_default(router):
    """The ``code_interpreter`` built-in is offered with an auto container when no bridge wires one."""
    provider = OpenAIProvider(
        model="gpt-5.4-mini",
        api_key=FAKE_KEY,
        base_url=BASE_URL,
        surface="responses",
        max_retries=0,
        builtin_tools=["code_interpreter"],
    )
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    provider.chat([Message.user("compute something")])

    tools = json.loads(route.calls.last.request.content)["tools"]
    assert {"type": "code_interpreter", "container": {"type": "auto"}} in tools
    provider.close()


def test_code_interpreter_container_comes_from_the_callback_per_turn(router):
    """The Asset bridge supplies the live container per turn (a pinned id once it knows one)."""
    container_id = "cntr_pinned_42"
    provider = OpenAIProvider(
        model="gpt-5.4-mini",
        api_key=FAKE_KEY,
        base_url=BASE_URL,
        surface="responses",
        max_retries=0,
        builtin_tools=["code_interpreter"],
        code_container=lambda: container_id,
    )
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    provider.chat([Message.user("read my file")])

    tools = json.loads(route.calls.last.request.content)["tools"]
    assert {"type": "code_interpreter", "container": container_id} in tools
    provider.close()


def test_code_interpreter_call_surfaces_source_and_output_files(router, responses_provider):
    """A code_interpreter_call + container_file_citation become the Message's CodeExecutionTrace."""
    from tests.conftest import container_file_citation, out_code_interpreter_call

    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200,
            json=responses_body(
                out_code_interpreter_call(code="print(sum(range(10)))", container_id="cntr_x"),
                out_message(
                    "Done — see the chart.",
                    annotations=[
                        container_file_citation(
                            file_id="cfile_1", filename="chart.png", container_id="cntr_x"
                        )
                    ],
                ),
            ),
        )
    )

    reply = responses_provider.chat([Message.user("plot it")])

    assert reply.content == "Done — see the chart."
    trace = reply.code_execution
    assert trace is not None
    assert trace.container == "cntr_x"
    assert trace.code == ["print(sum(range(10)))"]
    assert [(f.file_id, f.filename) for f in trace.output_files] == [("cfile_1", "chart.png")]


def test_a_turn_without_code_execution_has_no_trace(router, responses_provider):
    """A plain reply (no code run) carries ``code_execution=None`` — nothing else changes."""
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("hi")))
    )

    reply = responses_provider.chat([Message.user("hello")])

    assert reply.content == "hi"
    assert reply.code_execution is None


def test_extra_body_is_sent_on_the_responses_surface(router):
    """A vendor-specific body field (xAI's ``search_parameters``) rides the SDK's ``extra_body``.

    This is how the ``openai`` SDK pointed at ``api.x.ai`` wires Live Search — the field the
    typed SDK params don't cover lands in the request body, on the Responses surface.
    """
    provider = OpenAIProvider(
        model="grok-4.3",
        api_key=FAKE_KEY,
        base_url=BASE_URL,
        surface="responses",
        max_retries=0,
        extra_body={"search_parameters": {"mode": "on", "sources": ["web", "x"]}},
    )
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("ok")))
    )

    provider.chat([Message.user("news?")])

    body = json.loads(route.calls.last.request.content)
    assert body["search_parameters"] == {"mode": "on", "sources": ["web", "x"]}
    provider.close()


def test_extra_body_is_sent_on_the_chat_surface(router):
    """``extra_body`` applies on **both** surfaces — xAI Live Search works over Chat too."""
    provider = OpenAIProvider(
        model="grok-4.3",
        api_key=FAKE_KEY,
        base_url=BASE_URL,
        surface="chat",
        max_retries=0,
        extra_body={"search_parameters": {"mode": "on", "sources": ["web"]}},
    )
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )

    provider.chat([Message.user("news?")])

    body = json.loads(route.calls.last.request.content)
    assert body["search_parameters"] == {"mode": "on", "sources": ["web"]}
    provider.close()


def test_responses_function_call_becomes_a_tool_call(router, responses_provider):
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200,
            json=responses_body(
                out_function_call(
                    call_id="call_9", name="get_weather", arguments={"city": "Dallas"}
                )
            ),
        )
    )

    reply = responses_provider.chat([Message.user("weather?")], tools=[WEATHER_TOOL])

    assert reply.content is None
    assert reply.tool_calls == [
        ToolCall(id="call_9", name="get_weather", arguments={"city": "Dallas"})
    ]


def test_web_search_call_item_is_ignored_but_citations_are_kept(router, responses_provider):
    """A server-side web_search_call is not a tool call; its citations footer the reply."""
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200,
            json=responses_body(
                out_web_search_call(),
                out_message(
                    "Here is the news.",
                    annotations=[url_citation(url="https://ex.com/a", title="A")],
                ),
            ),
        )
    )

    reply = responses_provider.chat([Message.user("news?")])

    assert reply.tool_calls == []  # web_search_call is resolved server-side, never run here
    assert "Here is the news." in reply.content
    assert "Sources:" in reply.content
    assert "https://ex.com/a" in reply.content


def test_vision_image_is_sent_as_an_input_image_part(router, responses_provider):
    route = router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("I see a cat.")))
    )

    turn = Message.user("what's this?")
    turn.images = [ImageContent(url="data:image/png;base64,AAAA", alt="cat.png")]
    responses_provider.chat([turn])

    content = json.loads(route.calls.last.request.content)["input"][0]["content"]
    assert {"type": "input_text", "text": "what's this?"} in content
    assert {"type": "input_image", "image_url": "data:image/png;base64,AAAA"} in content


# === Errors (SDK exceptions → the harness provider hierarchy) ================


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
    assert not isinstance(exc.value, (ProviderAuthError, ProviderRateLimitError))


def test_400_relays_the_providers_real_message(router, provider):
    """A 400's body carries the true cause under error.message — kept for the relay."""
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "Unsupported value: 'foo'."}})
    )

    with pytest.raises(ProviderAPIError) as exc:
        provider.chat([Message.user("Hi")])

    assert exc.value.status_code == 400
    assert "Unsupported value" in exc.value.body


def test_transport_failure_raises_connection_error(router, provider):
    router.post(CHAT_URL).mock(side_effect=httpx.ConnectError("no route"))

    with pytest.raises(ProviderConnectionError):
        provider.chat([Message.user("Hi")])


def test_malformed_response_raises_the_retryable_response_error(router, provider):
    """The SDK leniently constructs the model; an empty `choices` (the shape a truncated body
    leaves behind) surfaces as the *retryable* ProviderResponseError, so the engine re-requests it
    rather than aborting the wake (issue #259)."""
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json={"unexpected": True}))

    with pytest.raises(ProviderResponseError):
        provider.chat([Message.user("Hi")])


def test_a_non_json_body_raises_the_retryable_response_error(router, provider):
    """A 200 whose body is truncated / not JSON (the SDK cannot parse it — the observed
    EOF-mid-JSON class) surfaces as the retryable ProviderResponseError, not a permanent one."""
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "application/json"}, text="{trunc"
        )
    )

    with pytest.raises(ProviderResponseError):
        provider.chat([Message.user("Hi")])


def test_truncated_tool_call_arguments_raise_the_retryable_response_error(router, provider):
    """A well-formed envelope whose tool-call `arguments` string is cut off mid-JSON (a very common
    truncation locus — args are often the largest/last part) is the same transient class: retryable
    ProviderResponseError, not a permanent drop (issue #259)."""
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Dal'},  # truncated JSON
    }
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, json=completion(tool_calls=[tool_call], finish_reason="tool_calls")
        )
    )

    with pytest.raises(ProviderResponseError):
        provider.chat([Message.user("weather?")], tools=[WEATHER_TOOL])


def test_truncated_tool_call_arguments_on_the_responses_surface_are_retryable(
    router, responses_provider
):
    """Same truncated-args locus on the Responses surface → the retryable ProviderResponseError."""
    bad_call = {
        "id": "fc-1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"city": "Dal',  # truncated JSON
    }
    router.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json=responses_body(bad_call)))

    with pytest.raises(ProviderResponseError):
        responses_provider.chat([Message.user("weather?")], tools=[WEATHER_TOOL])


# === Construction & configuration ===========================================


def test_api_key_falls_back_to_env(monkeypatch, router):
    monkeypatch.setenv("AI_API_KEY", "sk-env-key")
    provider = OpenAIProvider(model="gpt-4o", base_url=BASE_URL, surface="chat", max_retries=0)
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(content="ok")))

    provider.chat([Message.user("Hi")])

    assert router.calls.last.request.headers["Authorization"] == "Bearer sk-env-key"
    provider.close()


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="AI_API_KEY"):
        OpenAIProvider(model="gpt-4o")


def test_unknown_surface_is_a_clear_error():
    with pytest.raises(ValueError, match="surface"):
        OpenAIProvider(model="gpt-4o", api_key=FAKE_KEY, surface="telepathy")


def test_default_base_url_is_openai():
    provider = OpenAIProvider(model="gpt-4o", api_key=FAKE_KEY)
    assert provider.base_url == DEFAULT_BASE_URL
    provider.close()


def test_default_surface_is_responses():
    provider = OpenAIProvider(model="gpt-4o", api_key=FAKE_KEY)
    assert provider.surface == "responses"
    provider.close()


def test_default_params_pass_through(router):
    provider = _chat(temperature=0.2)
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )

    provider.chat([Message.user("Hi")])

    body = json.loads(route.calls.last.request.content)
    assert body["temperature"] == 0.2
    assert body["model"] == "gpt-4o"
    provider.close()


def test_context_manager_closes(router):
    with _chat() as provider:
        router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(content="ok")))
        assert provider.chat([Message.user("Hi")]).content == "ok"


def test_missing_sdk_is_a_clear_no_llm_error(monkeypatch):
    """With the ``openai`` package unimportable, construction fails loud — "no LLM, by design"."""
    import builtins

    real_import = builtins.__import__

    def _no_openai(name, *args, **kwargs):
        if name == "openai":
            raise ModuleNotFoundError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_openai)
    with pytest.raises(ProviderError, match="openai"):
        OpenAIProvider(model="gpt-4o", api_key=FAKE_KEY)


# === The one-class promise ===================================================


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


# --- the per-call log line (issue #272) --------------------------------------


def _llm_line(caplog) -> str:
    return next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("llm "))


def test_a_responses_call_logs_one_line_with_the_provider_model_and_tokens(
    router, responses_provider, caplog
):
    body = responses_body(out_message("Hi."))
    body["usage"] = {"input_tokens": 120, "output_tokens": 8, "total_tokens": 128}
    router.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json=body))

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        responses_provider.chat([Message.user("hello")])

    line = _llm_line(caplog)
    assert "provider=openai" in line and "model=gpt-5.4-mini" in line
    assert "tokens_in=120 tokens_out=8 tokens_total=128" in line


def test_the_chat_surface_logs_the_same_line_from_its_own_usage_shape(router, provider, caplog):
    """Responses reports ``input_tokens``, Chat reports ``prompt_tokens`` — one line, either way."""
    body = completion(content="Hi.")
    body["usage"] = {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100}
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=body))

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("hello")])

    assert "tokens_in=90 tokens_out=10 tokens_total=100" in _llm_line(caplog)


def test_the_provider_label_names_the_endpoint_vendor_not_the_sdk(router, caplog):
    """One adapter serves three vendors, so grok-through-the-openai-SDK must not log as OpenAI —
    the label is what makes the line true (`_provider_from_config` passes AI_PROVIDER through)."""
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("Hi.")))
    )
    grok = OpenAIProvider(
        model="grok-4.3",
        api_key=FAKE_KEY,
        base_url=BASE_URL,
        provider="xai",
        surface="responses",
        max_retries=0,
    )

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        grok.chat([Message.user("hello")])

    assert "provider=xai" in _llm_line(caplog)


def test_the_same_adapter_reports_the_endpoint_and_cost_when_the_response_carries_them(
    router, provider, caplog
):
    """One adapter, three endpoint vendors — so whether a call *has* a serving endpoint or a cost
    is a fact about the **response**, never a vendor branch in the code. Pointed at OpenRouter
    (issue #274) the body names both, and the ``openai`` SDK keeps the fields it doesn't model, so
    the same reader that finds nothing on an OpenAI response finds them here."""
    body = completion(content="Hi.")
    body["provider"] = "Novita"
    body["usage"] = {
        "prompt_tokens": 90,
        "completion_tokens": 10,
        "total_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 64},
        "cost": 0.0445,
    }
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=body))

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("hello")])

    line = _llm_line(caplog)
    assert "endpoint=Novita" in line and "cached_tokens=64" in line and "cost=0.0445" in line


def test_an_openai_response_names_no_endpoint_or_cost_and_the_fields_are_omitted(
    router, responses_provider, caplog
):
    """OpenAI is not a router and reports no dollars — the honest line says neither, rather than
    restating `provider=openai` as an endpoint or pricing the call from a table of our own."""
    router.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=responses_body(out_message("Hi.")))
    )

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        responses_provider.chat([Message.user("hello")])

    line = _llm_line(caplog)
    assert "endpoint=" not in line and "cost=" not in line


def test_a_call_that_never_returned_logs_no_llm_line(router, provider, caplog):
    """A duration is only honest for a call that completed; the failure path is the engine's
    retry/give-up story to tell."""
    router.post(CHAT_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "no"}}))

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        with pytest.raises(ProviderAPIError):
            provider.chat([Message.user("hello")])

    assert not any(m.startswith("llm ") for m in (r.getMessage() for r in caplog.records))
