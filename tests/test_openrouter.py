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
    ProviderContextLengthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
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


# === Web search (server tool, issue #237) =====================================


def completion_with_annotations(*, content, annotations, server_tool_use=None):
    """A chat-completions body carrying nested ``url_citation`` annotations on the message.

    On the Chat Completions surface OpenRouter nests each citation under ``url_citation`` (unlike
    the flat Responses shape). ``server_tool_use`` on ``usage`` is what the vendor reports for the
    search count; both are fields the SDK's typed model does not keep, so this exercises the raw-body
    recovery, not the model_dump.
    """
    body = completion(content=content)
    body["choices"][0]["message"]["annotations"] = annotations
    if server_tool_use is not None:
        body["usage"]["server_tool_use"] = server_tool_use
    return body


def test_web_search_builtin_emits_the_openrouter_server_tool(router):
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )
    provider = _provider(builtin_tools=["web_search"])
    provider.chat([Message.user("news?")])
    tools = json.loads(route.calls.last.request.content)["tools"]
    assert tools == [{"type": "openrouter:web_search"}]  # bare object when no params configured
    provider.close()


def test_web_search_params_ride_the_parameters_block(router):
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )
    params = {"engine": "exa", "max_results": 10, "allowed_domains": ["arxiv.org"]}
    provider = _provider(builtin_tools=["web_search"], web_search_params=params)
    provider.chat([Message.user("news?")])
    tools = json.loads(route.calls.last.request.content)["tools"]
    assert tools == [{"type": "openrouter:web_search", "parameters": params}]  # verbatim
    provider.close()


def test_server_tools_lead_the_function_tools(router):
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )
    provider = _provider(builtin_tools=["web_search"])
    provider.chat([Message.user("weather + news?")], tools=[WEATHER_TOOL])
    tools = json.loads(route.calls.last.request.content)["tools"]
    assert tools[0]["type"] == "openrouter:web_search"  # the server tool leads
    assert tools[1]["type"] == "function"  # then the custom function tool
    assert tools[1]["function"]["name"] == "get_weather"
    provider.close()


def test_no_builtins_sends_no_server_tool(router):
    # The default openrouter agent (no opted-in web search) sends no tools at all — the safe,
    # benign-only baseline.
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )
    provider = _provider()
    provider.chat([Message.user("Hi")])
    assert "tools" not in json.loads(route.calls.last.request.content)
    provider.close()


def test_an_unknown_builtin_is_skipped_not_sent(router):
    # A name that is not a known OpenRouter server tool is dropped, never forwarded as an unknown
    # tool object the endpoint would reject. (In practice the resolver only feeds claimed names.)
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )
    provider = _provider(builtin_tools=["not_a_server_tool"])
    provider.chat([Message.user("Hi")])
    assert "tools" not in json.loads(route.calls.last.request.content)
    provider.close()


def test_url_citation_annotations_are_footered_as_sources(router):
    # The response-side contract: the SDK's typed model drops `annotations`, so the adapter recovers
    # them from the raw body (a response event hook on its own httpx client) and footers them as the
    # same `Sources:` block the other web-search built-ins produce. Built without an injected client
    # so the capture hook is live; respx still mocks the transport.
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=completion_with_annotations(
                content="Bitcoin is up.",
                annotations=[
                    {
                        "type": "url_citation",
                        "url_citation": {"url": "https://ex.com/btc", "title": "BTC News"},
                    }
                ],
                server_tool_use={"web_search_requests": 2},
            ),
        )
    )
    provider = OpenRouterProvider(
        MODEL, api_key=FAKE_KEY, base_url=BASE_URL, builtin_tools=["web_search"]
    )
    reply = provider.chat([Message.user("btc price?")])
    assert "Bitcoin is up." in reply.content
    assert "Sources:" in reply.content
    assert "https://ex.com/btc" in reply.content
    assert "BTC News" in reply.content
    provider.close()


def test_the_capture_hook_watches_the_sdks_own_client_on_every_call(monkeypatch):
    """The hook is on every call, web search or not (issue #274): the raw body is the only place
    the **serving endpoint** lives, and every call has one. It is attached to the httpx client the
    *SDK* built and owns — the harness constructs no client of its own to hand in."""
    monkeypatch.setenv("AI_API_KEY", "sk-or-env-key")

    for provider in (
        OpenRouterProvider(MODEL, base_url=BASE_URL),
        OpenRouterProvider(MODEL, base_url=BASE_URL, builtin_tools=["web_search"]),
    ):
        hooks = provider._client.sdk_configuration.client.event_hooks["response"]
        assert provider._capture in hooks
        provider.close()


def test_a_client_double_without_an_httpx_client_underneath_is_simply_unwatched():
    """Observability never breaks a turn: an unwatchable client leaves the capture empty, so the
    endpoint field is omitted and the reply is unaffected."""

    class Double:  # no `sdk_configuration` — nothing to hook
        pass

    provider = OpenRouterProvider(MODEL, client=Double())

    assert provider._capture.last is None


def test_a_plain_answer_without_annotations_is_unchanged(router):
    # No annotations → no footer, exactly as before web search. Guards against the recovery path
    # mangling an ordinary reply.
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="Just a plain answer."))
    )
    provider = OpenRouterProvider(
        MODEL, api_key=FAKE_KEY, base_url=BASE_URL, builtin_tools=["web_search"]
    )
    reply = provider.chat([Message.user("hi")])
    assert reply.content == "Just a plain answer."
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


def test_response_validation_error_maps_to_the_retryable_response_error(router):
    # A 200 whose body fails the SDK's ChatResult validation raises ResponseValidationError (an
    # OpenRouterError subclass carrying no HTTP-error status) — the truncated / EOF-mid-JSON class
    # (issue #259). It must surface as the *retryable* ProviderResponseError (so the engine
    # re-requests it), never a ProviderAPIError stamped with a misleading status_code like 200.
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json={"id": "gen-x", "object": "chat.completion"})
    )
    provider = _provider(retries_disabled=True)
    with pytest.raises(ProviderResponseError) as exc:
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


def test_500_maps_to_the_retryable_server_error_keeping_the_body(router):
    """A 5xx is the provider's *own* fault — mapped to the transient class the engine re-requests
    (issue #284), not the generic API error it used to be. It matters most on this adapter: the SDK's
    own retry is disabled here, so before the shared class existed a 5xx was simply fatal on
    OpenRouter while the `openai` SDK quietly retried the identical fault."""
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(500, json={"error": {"message": "boom", "code": 500}})
    )
    provider = _provider(retries_disabled=True)
    with pytest.raises(ProviderServerError) as exc:
        provider.chat([Message.user("Hi")])
    assert exc.value.status_code == 500
    assert "boom" in exc.value.body  # the body survives, so a tool can still relay the true cause
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


# --- the per-call log line (issue #272) --------------------------------------


def test_a_call_logs_the_line_with_the_openrouter_vendor_and_model(router, caplog):
    import logging

    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(content="Hi.")))

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _provider().chat([Message.user("hello")])

    line = next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("llm "))
    assert "provider=openrouter" in line and f"model={MODEL}" in line
    assert "tokens_in=1 tokens_out=2 tokens_total=3" in line  # the builder's usage block


# --- the serving endpoint, cache hit, and cost (issue #274) -------------------


def _llm_line(caplog) -> str:
    return next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("llm "))


def _routing_metadata(selected: str, *considered: str) -> dict:
    """OpenRouter's `openrouter_metadata` block: the endpoints it weighed, and the one it picked."""
    available = [{"model": MODEL, "provider": name, "selected": False} for name in considered]
    available.append({"model": MODEL, "provider": selected, "selected": True})
    return {
        "attempt": 1,
        "endpoints": {"available": available, "total": len(available)},
        "is_byok": False,
        "region": None,
        "requested": MODEL,
        "strategy": "direct",
        "summary": f"routed to {selected}",
    }


def test_the_line_names_the_upstream_that_actually_served_the_call(router, caplog):
    """`provider=openrouter` names who *dispatched* the call. OpenRouter fans one model id out to
    ~27 endpoints spanning 10× in context ceiling and 5.4× in price, so the line has to say who
    *served* it — and the answer is the endpoint the router flags `selected` in the routing metadata
    the request asked for (issue #280)."""
    import logging

    body = completion(content="Hi.")
    body["openrouter_metadata"] = _routing_metadata("StreamLake", "Novita")
    body["usage"] |= {"cost": 0.0445, "prompt_tokens_details": {"cached_tokens": 238277}}
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=body))

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _provider().chat([Message.user("hello")])

    line = _llm_line(caplog)
    assert "endpoint=StreamLake" in line
    assert "cached_tokens=238277" in line  # the cache is working, or it isn't — no more inferring
    assert "cost=0.0445" in line  # OpenRouter's own figure, not harness arithmetic


def test_the_call_asks_openrouter_which_endpoint_it_routed_to(router, caplog):
    """The metadata is **opt-in**: unasked, OpenRouter says nothing trustworthy about routing. So
    every call sends the header — without it the `endpoint=` field silently disappears."""
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="Hi."))
    )

    _provider().chat([Message.user("hello")])

    assert route.calls.last.request.headers["X-OpenRouter-Metadata"] == "enabled"


def test_the_top_level_provider_field_is_never_read(router, caplog):
    """The #280 defect, pinned at the adapter: with `openrouter:web_search` active, OpenRouter's
    undocumented top-level `provider` reports **the search tool's** upstream (`OpenAI`) — a vendor
    that serves no endpoint in `z-ai/glm-5.2`'s pool — while the routing metadata correctly names
    the model's. Reading the former logged a routing distribution that never happened.
    """
    import logging

    body = completion(content="Hi.")
    body["provider"] = "OpenAI"  # what the live wake actually returned, and must never be believed
    body["openrouter_metadata"] = _routing_metadata("StreamLake", "Novita")
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=body))

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _provider().chat([Message.user("hello")])

    line = _llm_line(caplog)
    assert "endpoint=StreamLake" in line
    assert "OpenAI" not in line


def test_a_body_that_names_no_endpoint_omits_the_field_rather_than_faking_one(router, caplog):
    import logging

    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion(content="Hi.")))

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _provider().chat([Message.user("hello")])

    line = _llm_line(caplog)
    assert "endpoint=" not in line and "cost=" not in line and "cached_tokens=" not in line


# === The context-limit capability (issue #276) ===============================
#
# The honest ceiling behind a *router*. OpenRouter's model object advertises the best case across
# every endpoint it fronts (`z-ai/glm-5.2` says 1,048,576) while the endpoints actually range
# 101,376–1,048,576 — so the number must come from the live per-endpoint data, and must count only
# the endpoints a request could actually be routed to.

ENDPOINTS_URL = f"{BASE_URL}/models/z-ai/glm-5.2/endpoints"


def _endpoint(*, name, context_length, status=0, max_prompt_tokens=None):
    """One entry of the live `/endpoints` payload, shaped as the SDK's typed model requires.

    The values mirror a real `z-ai/glm-5.2` endpoints response — including the `status: -5`,
    zero-uptime entries the live pool actually carries, which is the case the capability must not
    count (see the test below).
    """
    return {
        "name": f"{name} | z-ai/glm-5.2-20260616",
        "provider_name": name,
        "model_id": MODEL,
        "model_name": "Z.ai: GLM 5.2",
        "context_length": context_length,
        "max_completion_tokens": 131072,
        "max_prompt_tokens": max_prompt_tokens,
        "pricing": {"prompt": "0.00000042", "completion": "0.00000132", "discount": 0},
        "supported_parameters": ["tools", "tool_choice", "reasoning"],
        "quantization": "fp8",
        "tag": f"{name.lower()}/fp8",
        "status": status,
        "uptime_last_5m": 99.7,
        "uptime_last_30m": 99.0,
        "uptime_last_1d": 99.5,
        "latency_last_30m": None,
        "throughput_last_30m": None,
        "supports_implicit_caching": False,
    }


def _endpoints_body(*endpoints):
    return {
        "data": {
            "id": MODEL,
            "name": "Z.ai: GLM 5.2",
            "created": 1781631930,
            "description": "GLM 5.2.",
            "architecture": {
                "tokenizer": "Other",
                "instruct_type": None,
                "modality": "text->text",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
            "endpoints": list(endpoints),
        }
    }


def test_the_context_limit_is_the_largest_endpoint_a_request_could_route_to(router):
    router.get(ENDPOINTS_URL).mock(
        return_value=httpx.Response(
            200,
            json=_endpoints_body(
                _endpoint(name="Ambient", context_length=101_376),
                _endpoint(name="StreamLake", context_length=1_024_000),
                _endpoint(name="Novita", context_length=1_048_576),
            ),
        )
    )

    # OpenRouter filters endpoints by required context at routing time, so a large request is never
    # dispatched to the small endpoint — the wall it actually meets is the largest that can serve it.
    assert _provider().context_limit() == 1_048_576


def test_a_dead_endpoints_ceiling_is_not_a_ceiling(router):
    """`status` is not decorative: the live pool really does carry endpoints at -5, 0% uptime."""
    router.get(ENDPOINTS_URL).mock(
        return_value=httpx.Response(
            200,
            json=_endpoints_body(
                _endpoint(name="Ambient", context_length=262_144),
                _endpoint(name="AkashML", context_length=1_048_576, status=-5),
            ),
        )
    )

    # Counting the dead endpoint would report a ceiling no request can reach — and the agent would
    # then compact too late, which is the whole failure this exists to prevent.
    assert _provider().context_limit() == 262_144


def test_max_prompt_tokens_beats_context_length_where_an_endpoint_sets_it(router):
    router.get(ENDPOINTS_URL).mock(
        return_value=httpx.Response(
            200,
            json=_endpoints_body(
                _endpoint(name="Alibaba", context_length=1_048_576, max_prompt_tokens=200_000),
            ),
        )
    )

    # The tighter promise about the *prompt* is the half we are budgeting.
    assert _provider().context_limit() == 200_000


def test_an_unreachable_endpoints_api_degrades_to_no_answer(router):
    router.get(ENDPOINTS_URL).mock(side_effect=httpx.ConnectError("no route to host"))

    # `retries_disabled` for the same reason the production adapter passes `retry_config=None`: the
    # SDK's Speakeasy default backs off for up to an hour on a transport failure, which would hang
    # a wake (and, here, the test run) far past its timeout.
    # No answer → the budget falls to its conservative floor. A metadata read never breaks a wake.
    assert _provider(retries_disabled=True).context_limit() is None


# === supports_vision — the per-model vision gate the asset-wake reads (issue #228) ============
#
# OpenRouter serves a model's real ``architecture.input_modalities`` — the same authoritative field
# that told us ``z-ai/glm-5.2`` is text-only. The adapter reads it (through the real SDK's
# ``models.get``, path ``/model/{author}/{slug}``) so the wake can swap a posted image for its text
# description instead of shipping pixels a text-only endpoint 404s on.

MODEL_URL = f"{BASE_URL}/model/z-ai/glm-5.2"


def _model_body(input_modalities):
    """A `models.get` response body the real ``openrouter`` SDK's ``Model`` accepts.

    Only ``architecture.input_modalities`` is under test; the rest are the SDK's required fields,
    mirrored from a real ``z-ai/glm-5.2`` response so the test drives the actual SDK marshalling
    (respx patches its transport) rather than a hand-rolled double.
    """
    return {
        "data": {
            "id": MODEL,
            "canonical_slug": "z-ai/glm-5.2-20260616",
            "name": "Z.ai: GLM 5.2",
            "created": 1781631930,
            "description": "GLM 5.2.",
            "context_length": 1048576,
            "architecture": {
                "modality": "text->text",
                "input_modalities": list(input_modalities),
                "output_modalities": ["text"],
                "tokenizer": "Other",
                "instruct_type": None,
            },
            "pricing": {"prompt": "0.00000091", "completion": "0.00000286"},
            "top_provider": {
                "context_length": 1024000,
                "max_completion_tokens": 128000,
                "is_moderated": False,
            },
            "per_request_limits": None,
            "supported_parameters": ["tools", "reasoning"],
            "default_parameters": {"temperature": 1, "top_p": 0.95},
            "supported_voices": None,
            "links": {"details": "/api/v1/models/z-ai/glm-5.2-20260616/endpoints"},
        }
    }


def test_supports_vision_is_true_for_a_model_that_takes_image_input(router):
    router.get(MODEL_URL).mock(
        return_value=httpx.Response(200, json=_model_body(["image", "text", "file"]))
    )
    assert _provider().supports_vision() is True


def test_supports_vision_is_false_for_a_text_only_model(router):
    """The reachable case: ``z-ai/glm-5.2`` is genuinely ``input_modalities: ["text"]``."""
    router.get(MODEL_URL).mock(return_value=httpx.Response(200, json=_model_body(["text"])))
    assert _provider().supports_vision() is False


def test_supports_vision_is_none_when_the_metadata_is_unreadable(router):
    """A metadata read never breaks a wake — an unreachable models API is *unknown*, not "no vision".

    ``None`` is what makes the wake's gate fail *open* (`model_sees_images`), so a hiccup shows the
    image rather than silently withholding it from a vision-capable agent.
    """
    router.get(MODEL_URL).mock(side_effect=httpx.ConnectError("no route to host"))
    assert _provider(retries_disabled=True).supports_vision() is None


def test_supports_vision_is_memoized_after_a_definite_answer(router):
    """A model's modality doesn't change mid-process, so a known answer is read from OpenRouter once."""
    route = router.get(MODEL_URL).mock(return_value=httpx.Response(200, json=_model_body(["text"])))
    provider = _provider()
    assert provider.supports_vision() is False
    assert provider.supports_vision() is False
    assert route.call_count == 1  # the second call is served from the memo, not a second HTTP read


def test_supports_vision_retries_after_an_inconclusive_read(router):
    """An inconclusive read leaves the memo ``None`` and retries — a one-time hiccup never sticks.

    This is why the memo is keyed on ``None`` rather than a sentinel: a transient failure must not
    permanently disable the gate in the router's long-lived per-agent process.
    """
    route = router.get(MODEL_URL).mock(
        side_effect=[
            httpx.Response(503, json={"error": {"message": "upstream", "code": 503}}),
            httpx.Response(200, json=_model_body(["text"])),
        ]
    )
    provider = _provider(retries_disabled=True)
    assert provider.supports_vision() is None  # first read failed → unknown
    assert provider.supports_vision() is False  # retried, and this one answered
    assert route.call_count == 2


def test_supports_vision_strips_the_routing_variant(router):
    """``z-ai/glm-5.2:free`` selects routing, not a different model — the models API keys on the bare
    slug and would 404 on the suffix, silently reading a real model as unknown (as `context_limit`)."""
    router.get(MODEL_URL).mock(return_value=httpx.Response(200, json=_model_body(["text"])))
    client = OpenRouter(api_key=FAKE_KEY, server_url=BASE_URL)
    provider = OpenRouterProvider("z-ai/glm-5.2:free", client=client, base_url=BASE_URL)
    assert provider.supports_vision() is False  # the ``:free`` suffix came off before the lookup


def test_supports_vision_is_none_for_a_malformed_model_id():
    """A model id without the ``author/slug`` shape can't be looked up — unknown, and no HTTP call."""
    client = OpenRouter(api_key=FAKE_KEY, server_url=BASE_URL)
    provider = OpenRouterProvider("just-a-name", client=client, base_url=BASE_URL)
    assert provider.supports_vision() is None


def test_an_over_length_400_maps_to_the_context_length_error(router):
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "message": "This endpoint's maximum context length is 101376 tokens.",
                    "code": 400,
                }
            },
        )
    )
    provider = _provider(retries_disabled=True)

    with pytest.raises(ProviderContextLengthError) as exc:
        provider.chat([Message.user("Hi")])

    # Deterministic, not transient: it is classed apart so the session compacts and retries once,
    # rather than re-sending the same doomed request on every wake forever.
    assert exc.value.status_code == 400
    assert isinstance(exc.value, ProviderAPIError)
    provider.close()


def test_an_ordinary_400_is_not_mistaken_for_the_wall(router):
    router.post(CHAT_URL).mock(
        return_value=httpx.Response(
            400, json={"error": {"message": "Invalid tool schema", "code": 400}}
        )
    )
    provider = _provider(retries_disabled=True)

    with pytest.raises(ProviderAPIError) as exc:
        provider.chat([Message.user("Hi")])

    assert not isinstance(exc.value, ProviderContextLengthError)
    provider.close()
