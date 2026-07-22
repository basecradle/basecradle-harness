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
from xai_sdk.chat import chat_pb2  # the real SDK enum the adapter tags server-side calls with

from basecradle_harness import (
    ImageContent,
    Message,
    Provider,
    ProviderAuthError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderContextLengthError,
    ProviderError,
    ProviderPayloadTooLargeError,
    ProviderRateLimitError,
    ProviderResponseError,
    ToolCall,
    ToolSpec,
    XaiSdkProvider,
)

_TYPE = chat_pb2.ToolCallType

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
    """A duck-typed xai_sdk `Response` — only the fields the adapter reads.

    A tool_call dict may carry an optional ``"type"`` (a ``chat_pb2.ToolCallType`` int); omitted,
    the call carries no ``type`` attribute at all — the unset/legacy shape the adapter treats as a
    client-side call.
    """
    calls = []
    for c in tool_calls:
        call = SimpleNamespace(
            id=c["id"],
            function=SimpleNamespace(name=c["name"], arguments=json.dumps(c["arguments"])),
        )
        if "type" in c:
            call.type = c["type"]
        calls.append(call)
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


def test_server_side_tool_calls_are_not_surfaced_for_dispatch():
    # Issue #183: grok runs Live Search server-side inside one gRPC turn, then surfaces every tool
    # call it made — the already-executed server-side ones — in Response.tool_calls, each tagged by
    # a ToolCallType. Re-dispatching those to the harness function registry bounces "no tool named
    # web_search" and the model confabulates. The grounded answer + citations are the real output;
    # the server-side tool calls must be dropped, never surfaced. (x_semantic_search is x_search's
    # internal X sub-operation — the exact name grok "guessed" in the live forensics.)
    provider = _provider(
        _response(
            content="One recent AI headline …",
            tool_calls=[
                {
                    "id": "c1",
                    "name": "web_search",
                    "arguments": {},
                    "type": _TYPE.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
                },
                {
                    "id": "c2",
                    "name": "x_semantic_search",
                    "arguments": {},
                    "type": _TYPE.TOOL_CALL_TYPE_X_SEARCH_TOOL,
                },
            ],
            citations=["https://ex.com/a"],
        )
    )
    reply = provider.chat([Message.user("news?")])

    assert reply.tool_calls == []  # nothing bounces to the function dispatcher
    assert "One recent AI headline" in reply.content
    assert "Sources:" in reply.content  # the grounded answer survives intact


def test_a_client_call_survives_among_server_side_calls():
    # Mixed turn: grok ran web_search server-side *and* wants a real client function tool. Only the
    # client-side call is the harness's to run — the server-side one is dropped (issue #183).
    provider = _provider(
        _response(
            tool_calls=[
                {
                    "id": "s1",
                    "name": "web_search",
                    "arguments": {},
                    "type": _TYPE.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
                },
                {
                    "id": "f1",
                    "name": "get_weather",
                    "arguments": {"city": "Dallas"},
                    "type": _TYPE.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                },
            ]
        )
    )
    reply = provider.chat([Message.user("weather?")], tools=[WEATHER_TOOL])

    assert reply.tool_calls == [ToolCall(id="f1", name="get_weather", arguments={"city": "Dallas"})]


def test_code_execution_server_side_call_is_not_surfaced():
    # Same #183 contract for the code_execution built-in: grok runs Python in xAI's sandbox
    # server-side; that call must not reach the harness function registry.
    provider = _provider(
        _response(
            content="42",
            tool_calls=[
                {
                    "id": "x1",
                    "name": "code_execution",
                    "arguments": {},
                    "type": _TYPE.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
                }
            ],
        )
    )
    reply = provider.chat([Message.user("compute 6*7")])
    assert reply.tool_calls == []
    assert reply.content == "42"


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


def test_code_execution_builtin_becomes_an_agent_tool():
    # Issue #172: the code_execution built-in is an xAI Agent Tool on the chat `tools` list, the
    # same shape as the search built-ins. grok runs Python server-side; the harness never does.
    provider = XaiSdkProvider(
        "grok-4.3",
        api_key=FAKE_KEY,
        client=_FakeClient(_response(content="42")),
        builtin_tools=["code_execution"],
    )
    provider.chat([Message.user("sum of squares 1..100?")])

    kinds = [t.WhichOneof("tool") for t in provider._client.chat.captured["tools"]]
    assert kinds == ["code_execution"]


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


def test_grpc_internal_maps_to_the_retryable_response_error():
    # INTERNAL / DATA_LOSS are gRPC's broken/undecodable-payload codes — the native analogue of a
    # truncated JSON body (issue #259). They map to the retryable ProviderResponseError so the
    # engine re-requests, not the generic (non-retried) ProviderError.
    for code in (grpc.StatusCode.INTERNAL, grpc.StatusCode.DATA_LOSS):
        with pytest.raises(ProviderResponseError):
            _provider_raising(code).chat([Message.user("hi")])


def test_an_unclassified_grpc_error_stays_a_plain_provider_error():
    # A code that is neither transient-response nor connection/auth/rate-limit/too-large/out-of-funds
    # stays a plain ProviderError (not retried, not reported) — so the response-retry never fires on
    # an unrelated fault. A non-context INVALID_ARGUMENT (a fixable malformed request) is exactly this
    # case: it propagates rather than being reported, so the peer's message stays re-drivable (#336).
    with pytest.raises(ProviderError) as exc:
        _provider_raising(grpc.StatusCode.INVALID_ARGUMENT).chat([Message.user("hi")])
    assert not isinstance(exc.value, ProviderResponseError)
    assert type(exc.value) is ProviderError  # a plain provider error, not a reported subclass


def test_grpc_resource_exhausted_reads_the_detail_not_just_the_code():
    """The @briggs incident's root fix (issue #336): RESOURCE_EXHAUSTED is overloaded across three
    faults with three remedies, so the *detail* decides — a bare one stays a rate limit, a
    message-too-large is a permanent payload error, a credit-exhaustion is the billing class."""
    too_large = _FakeRpcError(
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        details="CLIENT: Sent message larger than max (25470493 vs. 20971520)",
    )
    with pytest.raises(ProviderPayloadTooLargeError) as exc:
        XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, client=_RaisingClient(too_large)).chat(
            [Message.user("hi")]
        )
    assert exc.value.status_code == 413
    assert "Sent message larger than max" in str(exc.value)

    out_of_funds = _FakeRpcError(
        grpc.StatusCode.RESOURCE_EXHAUSTED, details="insufficient credit on this account"
    )
    with pytest.raises(ProviderBillingError) as exc:
        XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, client=_RaisingClient(out_of_funds)).chat(
            [Message.user("hi")]
        )
    assert exc.value.status_code == 402

    # A bare RESOURCE_EXHAUSTED (no billing/too-large wording) is still a transient rate limit — the
    # safe fall-through, so a genuine rate limit is never mis-reported as a permanent outage.
    with pytest.raises(ProviderRateLimitError):
        _provider_raising(grpc.StatusCode.RESOURCE_EXHAUSTED).chat([Message.user("hi")])


# --- the per-call log line (issue #272) --------------------------------------


def test_a_native_call_logs_the_line_reading_usage_off_the_proto(caplog):
    """The gRPC response carries usage as proto *attributes*, not dict keys — the shared reader
    handles both, so the native path logs exactly what the HTTP adapters do."""
    import logging

    response = _response(content="Hi.")
    response.usage = SimpleNamespace(prompt_tokens=42, completion_tokens=6, total_tokens=48)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _provider(response).chat([Message.user("hello")])

    line = next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("llm "))
    assert "provider=xai" in line and "model=grok-4.3" in line
    assert "tokens_in=42 tokens_out=6 tokens_total=48" in line


# --- the cached count, the cost, and the absent endpoint (issue #274) ---------


def _llm_line(caplog) -> str:
    return next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("llm "))


def test_the_cached_count_is_read_off_the_real_usage_proto(caplog):
    """xAI spells the cache hit ``cached_prompt_text_tokens`` and reports it flat on the proto, not
    under a details block like the HTTP wires — driven through the **real** ``SamplingUsage`` here,
    because the field name is exactly what the shared reader has to get right."""
    import logging

    from xai_sdk.proto import usage_pb2

    response = _response(content="Hi.")
    response.usage = usage_pb2.SamplingUsage(
        prompt_tokens=42,
        completion_tokens=6,
        total_tokens=48,
        cached_prompt_text_tokens=32,
    )

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _provider(response).chat([Message.user("hello")])

    line = _llm_line(caplog)
    assert "tokens_in=42 tokens_out=6 tokens_total=48 cached_tokens=32" in line
    # The native SDK reaches xAI directly: the vendor *is* the endpoint, so there is no upstream to
    # name and the field is omitted rather than restating `provider=xai`.
    assert "endpoint=" not in line


def test_the_cost_is_the_sdks_own_dollar_figure_not_harness_arithmetic(caplog):
    """xAI reports the charge in *ticks* (1e-10 USD); ``Response.cost_usd`` is the SDK's own
    accessor for the converted dollars, so the adapter passes it through and never owns the
    constant. Rendered fixed-point — ``4.45e-05`` is not money a log reader can grep."""
    import logging

    response = _response(content="Hi.")
    response.cost_usd = 4.45e-05

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _provider(response).chat([Message.user("hello")])

    assert "cost=0.0000445" in _llm_line(caplog)


def test_an_unreported_cost_logs_nothing_rather_than_a_fabricated_zero(caplog):
    """The SDK distinguishes "xAI named no cost" (``None``) from "the call was free" — so must the
    line. This is also the shape of an SDK too old to carry the property at all."""
    import logging

    response = _response(content="Hi.")
    response.cost_usd = None

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        reply = _provider(response).chat([Message.user("hello")])

    assert reply.content == "Hi."
    assert "cost=" not in _llm_line(caplog)


# === The context-limit capability + the wall (issue #276) ====================
#
# The cleanest of the three adapters: xAI's own model metadata carries the number
# (`LanguageModel.max_prompt_length`), so there is nothing to infer and no table to rot.


class _FakeModelsClient:
    """Stands in for ``client.models``: returns a canned language-model description."""

    def __init__(self, model):
        self._model = model
        self.asked = []

    def get_language_model(self, name):
        self.asked.append(name)
        if isinstance(self._model, Exception):
            raise self._model
        return self._model


def _provider_with_models(model):
    client = _FakeClient(_response(content="hi"))
    client.models = _FakeModelsClient(model)
    return XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, client=client)


def test_the_context_limit_comes_from_xais_own_model_metadata():
    provider = _provider_with_models(SimpleNamespace(max_prompt_length=2_000_000))

    assert provider.context_limit() == 2_000_000
    assert provider._client.models.asked == ["grok-4.3"]


def test_a_model_that_reports_no_length_yields_no_answer():
    provider = _provider_with_models(SimpleNamespace(max_prompt_length=0))

    # No answer → the budget falls to its conservative floor, rather than trusting a zero.
    assert provider.context_limit() is None


def test_an_unreachable_models_endpoint_degrades_to_no_answer():
    provider = _provider_with_models(RuntimeError("grpc: deadline exceeded"))

    # A metadata read must never break a wake.
    assert provider.context_limit() is None


def test_the_last_reported_input_tokens_are_remembered_for_the_context_budget():
    response = _response(content="hi")
    # `prompt_tokens` is the whole prompt (xAI's proto also carries the text-only subset,
    # `prompt_text_tokens`) — the budget must trigger on the total, images included.
    response.usage = SimpleNamespace(
        prompt_tokens=41_000, prompt_text_tokens=38_000, completion_tokens=12
    )
    provider = XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, client=_FakeClient(response))

    assert provider.last_tokens_in is None  # nothing to report before the first call
    provider.chat([Message.user("hi")])

    # The same usage read that writes the log line feeds the compaction decision — the trigger is
    # the provider's *own* count, exact and free, never a client-side estimate.
    assert provider.last_tokens_in == 41_000


def test_an_over_length_invalid_argument_maps_to_the_context_length_error():
    error = _FakeRpcError(
        grpc.StatusCode.INVALID_ARGUMENT,
        details="prompt is too long: 300000 tokens > 256000 maximum context length",
    )
    provider = XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, client=_RaisingClient(error))

    with pytest.raises(ProviderContextLengthError):
        provider.chat([Message.user("hi")])


def test_an_ordinary_invalid_argument_is_not_mistaken_for_the_wall():
    provider = _provider_raising(grpc.StatusCode.INVALID_ARGUMENT)

    with pytest.raises(ProviderError) as exc:
        provider.chat([Message.user("hi")])

    # A non-context INVALID_ARGUMENT stays a plain provider error and propagates — never the context
    # wall, and never a reported class (a fixable malformed request, not permanent-for-content, #336).
    assert not isinstance(exc.value, ProviderContextLengthError)
