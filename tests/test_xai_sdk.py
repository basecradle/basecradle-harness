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
    ProviderToolSchemaError,
    ToolCall,
    ToolSpec,
    XaiSdkProvider,
)
from basecradle_harness._xai_sdk import CONVERSATION_METADATA_KEY
from tests.conftest import mail_tool

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
    """Stands in for ``client.chat``: records the create() payload, returns a canned Response.

    Takes one response, or several — the last one repeats, so a single-response client answers
    every call with it (what almost every test wants) while a test that needs to watch state
    *change across calls* can hand over a sequence.
    """

    def __init__(self, *responses):
        self._responses = list(responses)
        self.captured: dict | None = None
        self.calls = 0

    def create(self, **kwargs):
        self.captured = kwargs
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return _FakeConversation(response)


class _FakeClient:
    def __init__(self, *responses):
        self.chat = _FakeChatClient(*responses)
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


def test_a_video_turn_raises_rather_than_reaching_the_wire_without_one(monkeypatch):
    """This adapter declares no `supports_video`, so the fail-closed gate never routes a clip here.

    It raises rather than dropping the payload, for the same reason the Responses surface does: a
    silently-dropped video leaves the model reading a caption for something it never received —
    the exact defect the vision gate ended (issues #316, #471).
    """
    from basecradle_harness import ProviderError, VideoContent

    provider = _provider(_response(content="never reached"))
    turn = Message.user("watch this")
    turn.videos = [VideoContent(url="data:video/mp4;base64,AAAA", alt="clip.mp4")]

    with pytest.raises(ProviderError, match="native xai-sdk surface"):
        provider.chat([turn])


def test_the_adapter_declares_no_video_capability_so_the_gate_stays_closed():
    provider = _provider(_response(content="ok"))

    assert not hasattr(provider, "supports_video")


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


# --- cache affinity: the conversation_id (issues #431, #433) -----------------
#
# These pin the ``conversation_id`` the adapter puts on ``chat.create``, which — as issue #433
# established — is **telemetry**: `xai-sdk` 1.19.0 spends it on the OpenTelemetry span attribute
# ``gen_ai.conversation.id`` and puts it on no request proto. It is still the right value for that
# attribute, and it is still harness-owned (a static one in ``model_params.json`` would name every
# session on the box the same thing), so the behavior is pinned here. What actually earns the
# per-server cache is the ``x-grok-conv-id`` gRPC metadata, and that is proven where a fake client
# structurally cannot prove it: against a real gRPC server, in `test_xai_sdk_wire.py`.


def test_no_conversation_id_until_one_is_bound():
    """Unbound, the field is **omitted** — never a fabricated id, which would read as a brand-new
    conversation to xAI on every call and turn a lucky hit into a guaranteed miss."""
    provider = _provider(_response(content="ok"))
    provider.chat([Message.user("Hi")])

    assert "conversation_id" not in provider._client.chat.captured


def test_a_bound_conversation_rides_every_create():
    """Consecutive calls carry the same id, so a session's spans group under one conversation
    instead of scattering. (The *routing* half of that is the metadata — `test_xai_sdk_wire.py`.)"""
    provider = _provider(_response(content="ok"))
    provider.bind_conversation("timeline:019f6e71-2a12-7b69-a204-0fec1497b9c2")

    provider.chat([Message.user("Hi")])
    first = provider._client.chat.captured["conversation_id"]
    provider.chat([Message.user("Again")])

    assert first == "timeline:019f6e71-2a12-7b69-a204-0fec1497b9c2"
    assert provider._client.chat.captured["conversation_id"] == first


@pytest.mark.parametrize("conversation", [None, ""])
def test_clearing_the_binding_omits_the_field_again(conversation):
    """`None` (and an empty string, which is not an id either) means omit — the contract the
    `Provider` capability states, so a caller with no session never invents one."""
    provider = _provider(_response(content="ok"))
    provider.bind_conversation("timeline:x")
    provider.bind_conversation(conversation)

    provider.chat([Message.user("Hi")])

    assert "conversation_id" not in provider._client.chat.captured


def test_the_bound_conversation_wins_over_a_static_model_params_value():
    """Harness-owned, like `model`/`messages`/`tools`. A single `conversation_id` left in
    `model_params.json` would pin *every* session on the box to one server — the anti-pattern this
    exists to prevent — so the session's own key overrides it."""
    provider = XaiSdkProvider(
        "grok-4.3",
        api_key=FAKE_KEY,
        client=_FakeClient(_response(content="ok")),
        conversation_id="one-id-for-everything",
    )
    provider.bind_conversation("timeline:x")

    provider.chat([Message.user("Hi")])

    assert provider._client.chat.captured["conversation_id"] == "timeline:x"


def test_a_constructor_value_still_rides_when_nothing_is_bound():
    """The mirror: a library caller driving an `Engine` with no `Session` binds nothing, and the
    value they passed is left exactly as they wrote it — the adapter fabricates nothing either way."""
    provider = XaiSdkProvider(
        "grok-4.3",
        api_key=FAKE_KEY,
        client=_FakeClient(_response(content="ok")),
        conversation_id="chosen-by-hand",
    )

    provider.chat([Message.user("Hi")])

    assert provider._client.chat.captured["conversation_id"] == "chosen-by-hand"


# --- the client rebuild that puts the key on the wire (issue #433) -----------
#
# `xai_sdk` fixes a client's gRPC metadata at construction and its stub calls take no per-call
# ``metadata=``, so switching the affinity key means switching the client. These pin the *cost* of
# that mechanism (it must not rebuild per turn, and it must not leak the channel it replaces) and
# its *seam* (an injected client is never replaced). That the rebuilt client's metadata actually
# reaches a server is `test_xai_sdk_wire.py`'s job — a fake cannot answer it.


class _RecordingClients:
    """Stands in for ``xai_sdk.Client``: records every construction, hands back a fake client."""

    def __init__(self, response):
        self._response = response
        self.built: list[dict] = []
        self.clients: list[_FakeClient] = []

    def __call__(self, **kwargs):
        self.built.append(kwargs)
        client = _FakeClient(self._response)
        self.clients.append(client)
        return client

    @property
    def keys(self) -> list[str | None]:
        """The ``x-grok-conv-id`` each construction asked for, ``None`` where none was set."""
        return [dict(kw.get("metadata") or ()).get(CONVERSATION_METADATA_KEY) for kw in self.built]


@pytest.fixture
def rebuilding(monkeypatch):
    """A real `XaiSdkProvider` that builds its own (fake) clients, so rebuilds are countable."""
    import xai_sdk

    from basecradle_harness import _xai_sdk

    clients = _RecordingClients(_response(content="ok"))
    monkeypatch.setattr(
        _xai_sdk,
        "require_xai_sdk",
        lambda: SimpleNamespace(Client=clients, chat=xai_sdk.chat, tools=xai_sdk.tools),
    )
    return XaiSdkProvider("grok-4.3", api_key=FAKE_KEY), clients


def test_the_client_is_rebuilt_only_when_the_conversation_changes(rebuilding):
    """Every turn of a session rebinds the same id (`Session._drive` does it unconditionally), so
    keying the rebuild on the *value* is what keeps a per-session cost from becoming a per-turn one."""
    provider, clients = rebuilding
    provider.bind_conversation("timeline:one")
    provider.chat([Message.user("Hi")])
    provider.bind_conversation("timeline:one")
    provider.chat([Message.user("Again")])
    provider.bind_conversation("timeline:one")
    provider.chat([Message.user("And again")])

    assert clients.keys == [None, "timeline:one"]  # the constructor's, then one rebuild


def test_each_session_gets_a_client_carrying_its_own_key(rebuilding):
    """One adapter serves every session. A stale key would herd a second timeline's prefix onto the
    first one's server — the anti-pattern, not the fix — so a switch has to take."""
    provider, clients = rebuilding
    provider.bind_conversation("timeline:one")
    provider.chat([Message.user("Hi")])
    provider.bind_conversation("timeline:two")
    provider.chat([Message.user("Hi")])

    assert clients.keys == [None, "timeline:one", "timeline:two"]


def test_the_replaced_client_is_closed_so_a_session_switch_leaks_no_channel(rebuilding):
    """A gRPC channel holds sockets. An agent cycling timelines would accumulate one per switch for
    the life of the process, which is a slow leak nothing would ever raise on."""
    provider, clients = rebuilding
    provider.bind_conversation("timeline:one")
    provider.chat([Message.user("Hi")])
    provider.bind_conversation("timeline:two")
    provider.chat([Message.user("Hi")])

    assert [c.closed for c in clients.clients] == [True, True, False]  # only the live one is open


def test_an_injected_client_is_never_replaced():
    """The seam the whole offline suite rests on, and the contract for a library caller who supplies
    their own client: there is nothing to rebuild it from, and replacing it would throw away
    whatever they configured it with. Such a caller runs unbound — no affinity, and no surprise."""
    client = _FakeClient(_response(content="ok"))
    provider = XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, client=client)
    provider.bind_conversation("timeline:one")

    provider.chat([Message.user("Hi")])

    assert provider._client is client


def test_a_conversation_grpc_could_not_carry_is_refused_at_bind_time(caplog, rebuilding):
    """Affinity is an optimization and must never cost a wake. grpc rejects a non-ASCII metadata
    value at **call** time, inside the model call, so an unusable key is dropped on the way in —
    leaving the adapter exactly as well off as it was before any of this: unbound, and lucky."""
    provider, clients = rebuilding
    with caplog.at_level("WARNING", logger="basecradle_harness"):
        provider.bind_conversation("timeline:ｕｎｉｃｏｄｅ")
    provider.chat([Message.user("Hi")])

    assert clients.keys == [None]  # no rebuild, and nothing grpc would have thrown on
    assert "cache affinity" in caplog.text


def test_a_failed_rebuild_costs_the_hit_and_not_the_wake(caplog, rebuilding):
    """The rebuild is the one piece of the affinity path that runs *inside* `chat`, so it sits
    outside both guards that cover the rest of it: `_caching.bind_conversation`'s blanket
    try/except wraps only the bind, and `_mapped_errors` classifies only gRPC faults. Unguarded, a
    construction failure would kill the wake for an optimization."""
    provider, _clients = rebuilding
    original = provider._client

    def refuse(**kwargs):
        raise OSError("too many open files")

    provider._xai.Client = refuse
    provider.bind_conversation("timeline:one")

    with caplog.at_level("WARNING", logger="basecradle_harness"):
        reply = provider.chat([Message.user("Hi")])

    assert reply.content == "ok"  # answered, on the client it already had
    assert provider._client is original
    assert "continuing unbound" in caplog.text


def test_a_failed_rebuild_is_not_reattempted_every_turn(caplog, rebuilding):
    """And it gives up on *that conversation*, once. A session rebinds the same id before every
    turn, so retrying would repeat the identical failure — and the identical warning — for the life
    of the process. A different conversation still gets its own fresh attempt."""
    provider, _ = rebuilding
    attempts: list[dict] = []

    def refuse(**kwargs):
        attempts.append(kwargs)
        raise OSError("too many open files")

    provider._xai.Client = refuse
    with caplog.at_level("WARNING", logger="basecradle_harness"):
        for _ in range(3):
            provider.bind_conversation("timeline:one")  # what `Session._drive` does every turn
            provider.chat([Message.user("Hi")])
        provider.bind_conversation("timeline:two")
        provider.chat([Message.user("Hi")])

    assert len(attempts) == 2  # one per conversation, not one per turn
    assert caplog.text.count("continuing unbound") == 2


def test_a_client_that_refuses_to_close_does_not_strand_its_replacement(rebuilding):
    """Releasing the old channel is *cleanup*, and cleanup must not cost the wake or the switch: a
    channel that refuses to close is a leaked socket, and letting that escape would make it a dead
    wake instead. (The publish-then-close order in `_bound_client` is the belt to this suspenders —
    it keeps even a non-`Exception` failure from stranding the replacement unreferenced with its
    own channel open, which would be the same leak reached through the other door.)"""
    provider, clients = rebuilding
    provider.bind_conversation("timeline:one")
    provider.chat([Message.user("Hi")])
    live = provider._client

    def refuse():
        raise RuntimeError("channel has a call in flight")

    live.close = refuse
    provider.bind_conversation("timeline:two")
    provider.chat([Message.user("Hi")])

    assert provider._client is clients.clients[-1]  # the replacement is in place, not stranded
    assert clients.keys == [None, "timeline:one", "timeline:two"]


def test_an_unusable_key_warns_once_per_key_not_once_per_turn(caplog, rebuilding):
    """Same reasoning as the failed rebuild: the de-duplication lives with the *value*, because the
    caller rebinds unconditionally. A per-turn warning is a log nobody can read."""
    provider, _ = rebuilding
    with caplog.at_level("WARNING", logger="basecradle_harness"):
        for _ in range(3):
            provider.bind_conversation("timeline:ｕｎｉｃｏｄｅ")
        provider.bind_conversation("timeline:ｏｔｈｅｒ")

    assert caplog.text.count("cache affinity") == 2


# === issue #488: the finish reason reaches the caller that has to judge the answer ===


def test_the_native_response_finish_reason_is_recorded_for_a_capturing_caller():
    """This SDK names its proto enum, so ``REASON_MAX_LEN`` is xAI's word for "out of room".

    The shared reader knows it alongside the two chat-wire spellings, which is what lets one
    describer work identically whatever brain stack the agent runs (issue #488).
    """
    from basecradle_harness._observability import capture_llm_call

    response = _response(content="First frame: a poster")
    response.finish_reason = "REASON_MAX_LEN"
    provider = _provider(response)

    with capture_llm_call() as call:
        provider.chat([Message.user("describe it")])

    assert call.finish_reason == "REASON_MAX_LEN"


def test_the_last_finish_reason_is_remembered_for_the_delivery_guarantee():
    """Issue #490: the same read, kept on the adapter where the **engine** can reach it."""
    response = _response(content="Half a sen")
    response.finish_reason = "REASON_MAX_LEN"
    provider = _provider(response)

    assert provider.last_finish_reason is None  # nothing to report before the first call
    provider.chat([Message.user("write me an essay")])

    assert provider.last_finish_reason == "REASON_MAX_LEN"


def test_a_later_call_clears_it_rather_than_keeping_the_last_one_it_had():
    """**Every call overwrites it, including a call that reports nothing.**

    The natural-looking "don't clobber a good value with nothing" edit
    (``self.last_finish_reason = reason or self.last_finish_reason``) is the dangerous one: one
    truncated turn would then make *every* later turn read as truncated, so no item would ever
    commit, the mark would never advance, and the agent would resume the same timeline forever
    while answering nobody. It is asserted here because a single-call test cannot see it.
    """
    truncated_reply = _response(content="Half a sen")
    truncated_reply.finish_reason = "REASON_MAX_LEN"
    silent_reply = _response(content="ok")  # no `finish_reason` attribute at all
    provider = XaiSdkProvider(
        "grok-4.3", api_key=FAKE_KEY, client=_FakeClient(truncated_reply, silent_reply)
    )

    provider.chat([Message.user("write me an essay")])
    assert provider.last_finish_reason == "REASON_MAX_LEN"

    provider.chat([Message.user("hi")])
    assert provider.last_finish_reason is None


def test_a_response_that_names_no_finish_reason_records_none():
    """An SDK too old to carry the property says nothing, and nothing is what is recorded."""
    from basecradle_harness._observability import capture_llm_call

    provider = _provider(_response(content="ok"))

    with capture_llm_call() as call:
        provider.chat([Message.user("hi")])

    assert call.finish_reason is None


# --- refused tool schemas: fail per tool, never per wake (issue #496) ---------
#
# One MCP tool whose schema xAI's validator will not take used to kill *every* wake of the agent
# that loaded it (`@briggs`, grok-4.6): the router re-drove it, `posted=0` every time, and the peer
# was answered never. The schema in question is not invented here — it is the real
# `mcp-mail-server@2.0.2` `send_email` shape, read out of the published tarball
# (`tests/data/mcp_mail_server_2_0_2_tools.json`).


def _mail_send_email_spec() -> ToolSpec:
    tool = mail_tool("send_email")
    # Namespaced exactly as `_mcp.mcp_tool_name` does, so the name in these tests is the name that
    # appeared in xAI's own error text.
    return ToolSpec(
        name="workmail__send_email",
        description=tool["description"],
        parameters=tool["inputSchema"],
    )


def _wire_tool(provider, name):
    return next(
        t
        for t in provider._client.chat.captured["tools"]
        if t.WhichOneof("tool") == "function" and t.function.name == name
    )


def test_the_real_mail_server_schema_reaches_the_wire_with_an_object_root():
    spec = _mail_send_email_spec()
    provider = _provider(_response(content="sent"))

    provider.chat([Message.user("email John")], tools=[spec])

    sent = json.loads(_wire_tool(provider, "workmail__send_email").function.parameters)
    assert sent["type"] == "object"
    assert not any(k in sent for k in ("anyOf", "oneOf", "allOf"))
    assert sent["properties"]["signature"].get("anyOf") is None
    # Nothing the model can call with was lost on the way.
    assert set(sent["properties"]) == set(spec.parameters["properties"])
    assert sent["required"] == ["to", "subject"]


def test_the_constraint_the_schema_lost_is_told_to_the_model_in_the_description():
    provider = _provider(_response(content="sent"))

    provider.chat([Message.user("email John")], tools=[_mail_send_email_spec()])

    described = _wire_tool(provider, "workmail__send_email").function.description
    assert "At least one of: (text) or (html)." in described
    assert "`signature`: At least one of: (text) or (html)." in described
    assert described.startswith("Send a new email via SMTP")  # the server's own words come first


def test_a_well_formed_schema_is_offered_exactly_as_it_arrived():
    """The regression bar: an ordinary tool's offer is byte-identical to the pre-#496 one."""
    provider = _provider(_response(content="ok"))

    provider.chat([Message.user("weather?")], tools=[WEATHER_TOOL])

    sent = _wire_tool(provider, "get_weather")
    assert json.loads(sent.function.parameters) == WEATHER_TOOL.parameters
    assert sent.function.description == WEATHER_TOOL.description


def test_an_unrepresentable_schema_drops_that_tool_and_the_wake_proceeds(caplog):
    broken = ToolSpec(
        name="broken", description="Takes a bare string.", parameters={"type": "string"}
    )
    provider = _provider(_response(content="ok"))

    with caplog.at_level("WARNING", logger="basecradle_harness"):
        reply = provider.chat([Message.user("hi")], tools=[broken, WEATHER_TOOL])

    assert reply.content == "ok"  # the wake survived
    names = [t.function.name for t in provider._client.chat.captured["tools"]]
    assert names == ["get_weather"]  # every other tool is untouched
    assert "broken" in caplog.text and "not an object" in caplog.text


def test_a_dropped_tool_is_reported_once_per_adapter_not_once_per_turn(caplog):
    broken = ToolSpec(name="broken", description="x", parameters={"type": "string"})
    provider = _provider(_response(content="ok"))

    with caplog.at_level("WARNING", logger="basecradle_harness"):
        provider.chat([Message.user("hi")], tools=[broken])
        provider.chat([Message.user("again")], tools=[broken])

    assert caplog.text.count("Not offering tool") == 1


def test_every_function_tool_refused_sends_no_tools_key_at_all():
    broken = ToolSpec(name="broken", description="x", parameters={"type": "string"})
    provider = _provider(_response(content="ok"))

    provider.chat([Message.user("hi")], tools=[broken])

    assert "tools" not in provider._client.chat.captured


class _RefusingOnceClient:
    """Refuses the first ``create`` by naming one tool's schema, then behaves normally.

    xAI's live rejection is the only authority on its own validator, so this is the path that
    matters most: a schema the harness thought fine, refused by name, must cost that tool and not
    the wake.
    """

    def __init__(self, response, details):
        self.chat = SimpleNamespace(create=self._create)
        self._response = response
        self._details = details
        self.calls: list[dict] = []

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise _FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT, details=self._details)
        return _FakeConversation(self._response)

    def close(self):
        pass


#: What the **live** xAI endpoint actually sends for this refusal, captured from `api.x.ai` while
#: building the fix (`tests/test_xai_sdk_live.py` re-proves it on the NOC prober's cadence). It is
#: deliberately this string and not the one the issue report quoted — that one carried a
#: `[invalid_client_tool_schema]` code the live endpoint does not send, and a matcher written to it
#: alone would have been dead against every real refusal. `test_faults` pins both shapes.
_XAI_REFUSAL = (
    "workmail__send_email: tool parameter root must be an object type "
    "(root schema is an anyOf/oneOf union with a non-object branch)"
)


def test_a_tool_xai_refuses_by_name_is_dropped_and_the_turn_is_re_issued(caplog):
    client = _RefusingOnceClient(_response(content="ok"), _XAI_REFUSAL)
    provider = XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, client=client)
    spec = _mail_send_email_spec()

    with caplog.at_level("WARNING", logger="basecradle_harness"):
        reply = provider.chat([Message.user("email John")], tools=[spec, WEATHER_TOOL])

    assert reply.content == "ok"
    assert len(client.calls) == 2
    first = [
        t.function.name for t in client.calls[0]["tools"] if t.WhichOneof("tool") == "function"
    ]
    second = [
        t.function.name for t in client.calls[1]["tools"] if t.WhichOneof("tool") == "function"
    ]
    assert first == ["workmail__send_email", "get_weather"]
    assert second == ["get_weather"]  # only the refused one left
    # The WARNING quotes the vendor, which is the authority on its own validator.
    assert "workmail__send_email" in caplog.text
    assert "tool parameter root must be an object type" in caplog.text


def test_a_tool_xai_refuses_stays_dropped_on_the_next_turn():
    client = _RefusingOnceClient(_response(content="ok"), _XAI_REFUSAL)
    provider = XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, client=client)
    spec = _mail_send_email_spec()

    provider.chat([Message.user("email John")], tools=[spec, WEATHER_TOOL])
    provider.chat([Message.user("again")], tools=[spec, WEATHER_TOOL])

    last = [
        t.function.name for t in client.calls[-1]["tools"] if t.WhichOneof("tool") == "function"
    ]
    assert last == ["get_weather"]
    assert len(client.calls) == 3  # one retry on the first turn, one clean call on the second


class _AlwaysRefusingClient(_RefusingOnceClient):
    def _create(self, **kwargs):
        self.calls.append(kwargs)
        raise _FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT, details=self._details)


def test_a_refusal_that_names_no_offered_tool_propagates_rather_than_looping():
    """The residual case, and the right floor: the wake fails **visibly** and the peer's message
    stays re-drivable, exactly as it did before the per-tool drop existed."""
    client = _AlwaysRefusingClient(
        _response(content="ok"),
        "Failed to start sampling: [invalid_client_tool_schema] someone_elses_tool: nope",
    )
    provider = XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, client=client)

    with pytest.raises(ProviderToolSchemaError):
        provider.chat([Message.user("hi")], tools=[WEATHER_TOOL])

    assert len(client.calls) == 1  # never retried


def test_a_vendor_that_keeps_refusing_the_same_tool_gives_up_instead_of_looping():
    client = _AlwaysRefusingClient(_response(content="ok"), _XAI_REFUSAL)
    provider = XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, client=client)

    with pytest.raises(ProviderToolSchemaError):
        provider.chat([Message.user("email John")], tools=[_mail_send_email_spec()])

    # One call, one drop, one re-issue — and then the same name comes back, so it stops.
    assert len(client.calls) == 2


def test_an_invalid_argument_that_is_not_a_tool_schema_refusal_is_unchanged():
    """The class stays narrow: a generic malformed request is still a plain, propagating
    ProviderError (a fixable harness/config defect, never permanent-for-content — issue #336)."""
    with pytest.raises(ProviderError) as exc:
        _provider_raising(grpc.StatusCode.INVALID_ARGUMENT).chat([Message.user("hi")])

    assert type(exc.value) is ProviderError


def test_a_tool_whose_schema_cannot_be_prepared_at_all_still_only_costs_that_tool(caplog):
    """The invariant is about the *tool list*, not about one anticipated fault.

    A schema the SDK's own tool builder chokes on (here: not JSON-serializable) must degrade the
    same way an unrepresentable root does — one tool, one WARNING naming it and the error, and a
    wake that keeps going.
    """
    unserializable = ToolSpec(
        name="weird", description="x", parameters={"type": "object", "properties": {"a": object()}}
    )
    provider = _provider(_response(content="ok"))

    with caplog.at_level("WARNING", logger="basecradle_harness"):
        reply = provider.chat([Message.user("hi")], tools=[unserializable, WEATHER_TOOL])

    assert reply.content == "ok"
    assert [t.function.name for t in provider._client.chat.captured["tools"]] == ["get_weather"]
    assert "weird" in caplog.text
