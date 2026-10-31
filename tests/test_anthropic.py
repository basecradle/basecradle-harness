"""The native Anthropic adapter (`AnthropicProvider`), driven through the real ``anthropic`` SDK.

Every test here runs the real SDK against respx, so what is asserted about the wire is read off the
request the SDK actually serialized — the breakpoint above all (the bytes-left rule, issue #433: a
capability whose whole job is to put bytes on a wire is not implemented until something outside the
adapter proves they left). The responses are bodies we wrote from Anthropic's documented shapes;
`test_anthropic_live.py` is where Anthropic answers for itself.
"""

from __future__ import annotations

import json
import logging

import httpx
import httpx2
import pytest

from basecradle_harness import (
    AnthropicConfigError,
    AnthropicProvider,
    ImageContent,
    Message,
    ProviderAPIError,
    ProviderAuthError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderContextLengthError,
    ProviderError,
    ProviderPayloadTooLargeError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    ToolCall,
    ToolSpec,
    VideoContent,
)
from basecradle_harness._anthropic import (
    BUILTINS,
    DEFAULT_BASE_URL,
    PAUSE_CONTINUATIONS,
    SYSTEM_LABEL,
)
from basecradle_harness._basecradle import _provider_from_config, resolved_model_params
from basecradle_harness._caching import EXPLICIT, anchor_cacheable_prefix
from basecradle_harness._observability import TRUNCATED_FINISH_REASONS

MODEL = "claude-haiku-5-5"
FAKE_KEY = "sk-ant-api03-FAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfake-0000AA"
MESSAGES = f"{DEFAULT_BASE_URL}/v1/messages"
MODEL_INFO = f"{DEFAULT_BASE_URL}/v1/models/{MODEL}"

TOOL = ToolSpec(
    name="get_secret_number",
    description="Returns the secret number for a label.",
    parameters={"type": "object", "properties": {"label": {"type": "string"}}},
)
OTHER_TOOL = ToolSpec(name="add", description="Adds two numbers.", parameters={})


@pytest.fixture(autouse=True)
def _no_stray_anthropic_env(monkeypatch):
    """Nothing in a developer's shell reaches the SDK: the adapter must never read these anyway."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "AI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)


@pytest.fixture
def provider():
    adapter = AnthropicProvider(MODEL, api_key=FAKE_KEY, max_tokens=4096)
    yield adapter
    adapter.close()


def text(value: str, **extra) -> dict:
    return {"type": "text", "text": value, **extra}


def tool_use(call_id: str, name: str, arguments: dict) -> dict:
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}


def thinking(value: str = "", signature: str = "EqQBCkYIBxgCKkBfake") -> dict:
    return {"type": "thinking", "thinking": value, "signature": signature}


def usage(**fields) -> dict:
    return {"input_tokens": 12, "output_tokens": 7, **fields}


def reply(
    *content, stop_reason="end_turn", used=None, message_id="msg_01FakeFakeFakeFake", **extra
):
    return httpx.Response(
        200,
        json={
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": list(content),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": used or usage(),
            **extra,
        },
    )


def failure(status: int, kind: str, message: str, **headers) -> httpx.Response:
    return httpx.Response(
        status,
        json={"type": "error", "error": {"type": kind, "message": message}},
        headers=headers,
    )


def body(route, index: int = -1) -> dict:
    return json.loads(route.calls[index].request.content)


def llm_lines(caplog) -> list[str]:
    return [m for m in (r.getMessage() for r in caplog.records) if m.startswith("llm provider=")]


# === configuration ===========================================================================


def test_no_key_is_a_config_fault_and_the_sdk_never_looks_for_one(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-someone-elses-key")
    with pytest.raises(AnthropicConfigError) as caught:
        AnthropicProvider(MODEL, max_tokens=64)
    assert caught.value.reason == "missing_api_key"
    assert "someone-elses" not in str(caught.value)


def test_the_key_and_host_are_the_adapters_never_the_sdks_environment(router, monkeypatch):
    """A stray ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY`` reaches nothing: the configured pair wins."""
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-someone-elses-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
    route = router.post(MESSAGES).mock(return_value=reply(text("hi")))
    adapter = AnthropicProvider(MODEL, max_tokens=64)
    adapter.chat([Message.user("hello")])
    request = route.calls.last.request
    # The 2.x SDK sends an API key as a bearer token (API version 2026-10-21).
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert "x-api-key" not in request.headers
    assert adapter.base_url == DEFAULT_BASE_URL


def test_custom_headers_in_the_environment_are_named(monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Stray: 1")
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        AnthropicProvider(MODEL, api_key=FAKE_KEY, max_tokens=64).close()
    assert "ANTHROPIC_CUSTOM_HEADERS" in caplog.text


@pytest.mark.parametrize(
    "params",
    [{"temperature": 0.2}, {"top_p": 0.9}, {"top_k": 40}, {"extra_body": {"temperature": 1.0}}],
)
def test_sampling_params_are_refused_at_startup(params):
    """Every 5.x model 400s on them; the operator hears it once, at construction, not every wake."""
    with pytest.raises(AnthropicConfigError) as caught:
        AnthropicProvider(MODEL, api_key=FAKE_KEY, **params)
    assert caught.value.reason == "sampling_params"


def test_sampling_params_fail_resolved_config_too(tmp_path, monkeypatch):
    home = tmp_path / "config"
    home.mkdir()
    (home / "model_params.json").write_text(json.dumps({"temperature": 0.3}))
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(home))
    with pytest.raises(ValueError, match="temperature"):
        resolved_model_params("anthropic")


def test_a_key_the_sdk_does_not_name_is_refused():
    with pytest.raises(AnthropicConfigError) as caught:
        AnthropicProvider(MODEL, api_key=FAKE_KEY, reasoning_effort="low")
    assert caught.value.reason == "invalid_model_params"
    assert "reasoning_effort" in str(caught.value)


@pytest.mark.parametrize(
    "cache_control",
    [{"type": "persistent"}, {"type": "ephemeral", "ttl": "24h"}, "1h", {"ttl": "1h"}],
)
def test_an_unusable_cache_control_is_refused(cache_control):
    with pytest.raises(AnthropicConfigError) as caught:
        AnthropicProvider(MODEL, api_key=FAKE_KEY, cache_control=cache_control)
    assert caught.value.reason == "invalid_cache_control"


def test_the_tuning_is_what_the_operator_wrote():
    adapter = AnthropicProvider(
        MODEL,
        api_key=FAKE_KEY,
        max_tokens=2048,
        output_config={"effort": "low"},
        cache_control={"type": "ephemeral", "ttl": "1h"},
    )
    assert adapter.tuning == {
        "max_tokens": 2048,
        "output_config": {"effort": "low"},
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }


def test_the_factory_builds_it_only_for_its_own_vendor(monkeypatch):
    monkeypatch.setenv("AI_MODEL", MODEL)
    adapter = _provider_from_config(
        "anthropic", "anthropic", "messages", api_key=FAKE_KEY, builtins=["web_fetch"]
    )
    assert isinstance(adapter, AnthropicProvider)
    assert adapter._builtins == [dict(BUILTINS["web_fetch"])]
    with pytest.raises(ValueError, match="requires AI_PROVIDER=anthropic"):
        _provider_from_config("openai", "anthropic", "messages", api_key=FAKE_KEY)


def test_the_factory_passes_extra_body_through_and_caps_a_describer(tmp_path, monkeypatch):
    home = tmp_path / "config"
    home.mkdir()
    (home / "model_params.json").write_text(json.dumps({"extra_body": {"metadata": {}}}))
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(home))
    adapter = _provider_from_config(
        "anthropic", "anthropic", "messages", model=MODEL, api_key=FAKE_KEY, max_output_tokens=900
    )
    assert adapter.tuning == {"extra_body": {"metadata": {}}, "max_tokens": 900}


# === the wire ================================================================================


def test_leading_system_turns_are_the_system_prompt_and_later_ones_are_labelled(router, provider):
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    provider.chat(
        [
            Message.system("You are Nova."),
            Message.system("Operating guidance."),
            Message.user("hello"),
            Message.assistant("hi"),
            Message.system("the per-wake brief"),
            Message.user("again"),
        ]
    )
    sent = body(route)
    assert sent["system"] == [text("You are Nova."), text("Operating guidance.")]
    assert sent["messages"] == [
        {"role": "user", "content": [text("hello")]},
        {"role": "assistant", "content": [text("hi")]},
        {"role": "user", "content": [text(SYSTEM_LABEL + "the per-wake brief"), text("again")]},
    ]
    assert sent["model"] == MODEL and sent["max_tokens"] == 4096 and "stream" not in sent


def test_a_steps_results_travel_as_one_user_message_with_the_results_first(router, provider):
    route = router.post(MESSAGES).mock(return_value=reply(text("done")))
    calls = [ToolCall("toolu_01A", "add", {"a": 1}), ToolCall("toolu_01B", "add", {"a": 2})]
    provider.chat(
        [
            Message.user("add twice"),
            Message.assistant(tool_calls=calls),
            Message.tool("toolu_01A", "1"),
            Message.tool("toolu_01B", "2"),
            Message.system("Step 2 of 50."),
        ],
        tools=[OTHER_TOOL],
    )
    sent = body(route)["messages"]
    assert sent[1] == {
        "role": "assistant",
        "content": [tool_use("toolu_01A", "add", {"a": 1}), tool_use("toolu_01B", "add", {"a": 2})],
    }
    assert sent[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_01A", "content": "1"},
            {"type": "tool_result", "tool_use_id": "toolu_01B", "content": "2"},
            text(SYSTEM_LABEL + "Step 2 of 50."),
        ],
    }


def test_the_tool_list_is_byte_stable_server_tools_first_then_sorted(router):
    adapter = AnthropicProvider(
        MODEL, api_key=FAKE_KEY, max_tokens=64, builtin_tools=["code_execution", "web_search"]
    )
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    adapter.chat([Message.user("hi")], tools=[TOOL, OTHER_TOOL])
    adapter.chat([Message.user("hi")], tools=[OTHER_TOOL, TOOL])
    first, second = body(route, 0)["tools"], body(route, 1)["tools"]
    assert first == second
    assert first == [
        {"type": "web_search_20260318", "name": "web_search"},
        {"type": "code_execution_20260521", "name": "code_execution"},
        {
            "name": "add",
            "description": "Adds two numbers.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": TOOL.name,
            "description": TOOL.description,
            "input_schema": TOOL.parameters,
            # Nothing is anchored, so the end of the tool list is the boundary (issue #726).
            "cache_control": {"type": "ephemeral"},
        },
    ]


def test_an_image_is_an_image_block(router, provider):
    route = router.post(MESSAGES).mock(return_value=reply(text("a cat")))
    picture = Message(
        role="user",
        content="what is this?",
        images=[ImageContent(url="data:image/png;base64,iVBORw0KGgo=")],
    )
    provider.chat([picture])
    assert body(route)["messages"][0]["content"] == [
        text("what is this?"),
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
        },
    ]


def test_a_clip_is_never_dropped_silently(provider):
    clip = Message(
        role="user", content="watch", videos=[VideoContent(url="data:video/mp4;base64,AA==")]
    )
    with pytest.raises(ProviderError, match="no video input"):
        provider.chat([clip])


def test_the_call_carries_the_fitted_budget(router, provider):
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    provider.chat([Message.user("hi")])
    budget = route.calls.last.request.extensions["timeout"]
    assert budget["read"] == provider.last_timeout and budget["connect"] == 10.0


# === the breakpoint, on the wire =============================================================


def test_the_anchor_becomes_cache_control_on_the_anchored_turns_last_block(router, provider):
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    frozen = [Message.system("charter"), Message.user("earlier"), Message.assistant("reply")]
    convo = [*frozen, Message.system("brief"), Message.user("now")]
    provider.chat(anchor_cacheable_prefix(convo, stable=len(frozen), mode=provider.cache_mode))
    sent = body(route)
    assert sent["system"] == [text("charter")]  # the breakpoint is on the frozen tail, not here
    assert sent["messages"][1] == {
        "role": "assistant",
        "content": [text("reply", cache_control={"type": "ephemeral"})],
    }
    assert "cache_control" not in json.dumps(sent["messages"][2])
    assert "cache_control" not in sent  # never the top-level automatic field
    assert not any(message.cache_anchor for message in convo)  # copy-on-write: history is clean


def test_on_a_first_wake_the_charter_carries_the_breakpoint(router, provider):
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    convo = [Message.system("charter"), Message.system("brief"), Message.user("hello")]
    provider.chat(anchor_cacheable_prefix(convo, stable=1, mode=EXPLICIT))
    assert body(route)["system"] == [
        text("charter", cache_control={"type": "ephemeral"}),
        text("brief"),
    ]


def test_a_one_hour_ttl_rides_the_breakpoint(router):
    adapter = AnthropicProvider(
        MODEL, api_key=FAKE_KEY, max_tokens=64, cache_control={"type": "ephemeral", "ttl": "1h"}
    )
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    convo = [Message.system("charter"), Message.user("hello")]
    adapter.chat(anchor_cacheable_prefix(convo, stable=1, mode=EXPLICIT))
    sent = body(route)
    assert sent["system"] == [text("charter", cache_control={"type": "ephemeral", "ttl": "1h"})]
    assert "cache_control" not in sent


def marks(sent: dict) -> list[str]:
    """Where every breakpoint in a request body landed, latest first, as a path into the body."""
    found = [
        f"tools[{i}]" for i, tool in enumerate(sent.get("tools", ())) if "cache_control" in tool
    ]
    found += [f"system[{i}]" for i, b in enumerate(sent.get("system", ())) if "cache_control" in b]
    for m, turn in enumerate(sent["messages"]):
        found += [
            f"messages[{m}][{i}]" for i, b in enumerate(turn["content"]) if "cache_control" in b
        ]
    return found[::-1]


def chatter(count: int) -> list[Message]:
    """A frozen transcript of `count` one-block turns, so turn `i` is lookback position `i`."""
    return [
        Message.user(f"question {i}") if i % 2 == 0 else Message.assistant(f"answer {i}")
        for i in range(count)
    ]


def anchored(frozen: list[Message]) -> list[Message]:
    """`frozen`, then a brief and the new turn, anchored the way the engine anchors it."""
    convo = [*frozen, Message.system("brief"), Message.user("now")]
    return anchor_cacheable_prefix(convo, stable=len(frozen), mode=EXPLICIT)


def test_with_nothing_anchored_the_last_tool_carries_the_breakpoint(router):
    """A first wake: the charter rides the brief, so the boundary is the end of the tool list."""
    adapter = AnthropicProvider(
        MODEL, api_key=FAKE_KEY, max_tokens=64, builtin_tools=["web_search"]
    )
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    convo = [Message.system("brief"), Message.user("hello")]
    adapter.chat(anchor_cacheable_prefix(convo, stable=0, mode=EXPLICIT), tools=[TOOL, OTHER_TOOL])
    sent = body(route)
    assert marks(sent) == ["tools[2]"]
    assert sent["tools"][2]["name"] == TOOL.name  # the last after sorting, not the last handed over
    assert sent["system"] == [text("brief")]
    assert "cache_control" not in sent


def test_with_nothing_anchored_and_no_tools_nothing_is_marked(router, provider):
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    provider.chat([Message.system("brief"), Message.user("hello")])
    assert marks(body(route)) == []


def test_an_anchored_turn_leaves_the_tool_list_unmarked(router, provider):
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    provider.chat(anchored(chatter(3)), tools=[TOOL])
    assert marks(body(route)) == ["messages[2][0]"]


@pytest.mark.parametrize(
    ("frozen", "expected"),
    [
        # The anchor's own window (positions 19..0) already reaches the start: nothing to add.
        (20, ["messages[19][0]"]),
        # Position 0 is one past the anchor's window, so one breakpoint is added there.
        (21, ["messages[20][0]", "messages[0][0]"]),
        # Twenty apart, tiling the windows with no gap, and never more than four in all.
        (100, ["messages[99][0]", "messages[79][0]", "messages[59][0]", "messages[39][0]"]),
    ],
)
def test_breakpoints_chain_back_from_the_anchor_twenty_positions_apart(
    router, provider, frozen, expected
):
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    provider.chat(anchored(chatter(frozen)))
    sent = body(route)
    assert marks(sent) == expected
    # Never through the brief or the new turn (an odd-length transcript merges into their message).
    assert "cache_control" not in json.dumps(sent["messages"][-1]["content"][-2:])


def test_a_run_of_parallel_calls_or_of_their_results_is_one_position(router, provider):
    """Twelve steps of three parallel calls: two positions a step, not six (Anthropic's rule)."""
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    frozen = [Message.user("go")]
    for step in range(12):
        ids = [f"toolu_01S{step:02d}C{n}" for n in range(3)]
        frozen.append(Message.assistant(tool_calls=[ToolCall(i, "add", {"a": 1}) for i in ids]))
        frozen.extend(Message.tool(i, "1") for i in ids)
    frozen.append(Message.assistant("done"))
    provider.chat(anchored(frozen))
    sent = body(route)
    # "done" is position 25; twenty back is position 5, step 2's run of calls, marked on its last.
    assert marks(sent) == ["messages[25][0]", "messages[5][2]"]
    assert [block["type"] for block in sent["messages"][5]["content"]] == ["tool_use"] * 3


def test_a_breakpoint_skips_a_block_that_cannot_carry_one(provider):
    """Where the reachable position is thinking, the next later block takes it: reach narrows, no gap."""
    blocks = [text(f"t{i}") for i in range(25)]
    blocks[4] = thinking("reasoning")
    blocks[5] = text("")
    blocks[24]["cache_control"] = {"type": "ephemeral"}
    provider._place_breakpoints(blocks)
    assert [i for i, block in enumerate(blocks) if "cache_control" in block] == [6, 24]


def test_a_callers_own_anchors_count_against_the_four(router, provider):
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    frozen = chatter(100)
    for index in (51, 71, 99):  # assistant turns, so no two merge
        frozen[index] = Message(role="assistant", content=f"answer {index}", cache_anchor=True)
    provider.chat([*frozen, Message.user("now")])
    assert marks(body(route)) == [
        "messages[99][0]",
        "messages[71][0]",
        "messages[51][0]",
        "messages[31][0]",
    ]


def test_moving_a_breakpoint_never_strips_the_thinking(router, provider):
    """The replay digest reads every block without its mark, the tool list's included."""
    route = router.post(MESSAGES).mock(
        side_effect=[
            reply(thinking("plan"), tool_use("toolu_01T", TOOL.name, {}), stop_reason="tool_use"),
            reply(text("done")),
        ]
    )
    convo = [Message.user("go")]
    first = provider.chat(convo, tools=[TOOL])  # nothing anchored: the tool list is marked
    follow = [*convo, first, Message.tool("toolu_01T", "7")]
    provider.chat(anchor_cacheable_prefix(follow, stable=1, mode=EXPLICIT), tools=[TOOL])
    assert marks(body(route, 0)) == ["tools[0]"]
    second = body(route, 1)
    assert marks(second) == ["messages[0][0]"]
    assert second["messages"][1]["content"][0] == thinking("plan")


# === the llm line, the cost, and the search fee ==============================================


def test_the_llm_line_grammar(router, provider, caplog):
    router.post(MESSAGES).mock(
        return_value=reply(
            text("ok"),
            used=usage(
                input_tokens=1_000,
                output_tokens=200,
                cache_creation_input_tokens=3_000,
                cache_read_input_tokens=40_000,
                cache_creation={"ephemeral_5m_input_tokens": 3_000, "ephemeral_1h_input_tokens": 0},
                server_tool_use={"web_search_requests": 2, "web_fetch_requests": 1},
            ),
            message_id="msg_01GrammarGrammar",
        )
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("hi")])
    (line,) = llm_lines(caplog)
    # 1,000 × $0.10 + 3,000 × $0.125 + 40,000 × $0.01 + 200 × $0.50, per million.
    cost = (1_000 * 0.10 + 3_000 * 0.125 + 40_000 * 0.01 + 200 * 0.50) / 1_000_000
    head, _, duration_and_rest = line.partition(" duration=")
    assert head == f"llm provider=anthropic purpose=main model={MODEL}"
    rest = duration_and_rest.split(" ", 1)[1]
    assert rest == (
        "tokens_in=44000 tokens_out=200 tokens_total=44200 cached_tokens=40000 "
        f"cost={cost:.6f} cost_basis=computed cache_write_tokens=3000 web_search_requests=2 "
        "web_fetch_requests=1 "
        "generation_id=msg_01GrammarGrammar"
    ).replace(f"cost={cost:.6f}", _money(cost))
    assert provider.last_tokens_in == 44_000
    media = [m for m in (r.getMessage() for r in caplog.records) if m.startswith("media ")]
    assert media == [
        "media provider=anthropic kind=search.web model=claude-haiku-5-5 count=2 cost=0.02 cost_basis=computed"
    ]


def _money(cost: float) -> str:
    from basecradle_harness._observability import _money as money

    return f"cost={money(cost)}"


def test_a_call_with_no_search_carries_no_search_field(router, provider, caplog):
    router.post(MESSAGES).mock(return_value=reply(text("ok")))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("hi")])
    (line,) = llm_lines(caplog)
    assert "web_search_requests" not in line and "cache_write_tokens" not in line
    assert not [r for r in caplog.records if r.getMessage().startswith("media ")]


def test_an_unknown_model_gets_no_cost_and_one_warning(router, caplog):
    adapter = AnthropicProvider("claude-future-9", api_key=FAKE_KEY, max_tokens=64)
    router.post(MESSAGES).mock(return_value=reply(text("ok")))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        adapter.chat([Message.user("a")])
        adapter.chat([Message.user("b")])
    assert all("cost=" not in line for line in llm_lines(caplog))
    assert sum("No published rate" in r.getMessage() for r in caplog.records) == 1


def test_us_inference_is_priced_at_the_multiplier(router, caplog):
    adapter = AnthropicProvider(MODEL, api_key=FAKE_KEY, max_tokens=64, inference_geo="us")
    router.post(MESSAGES).mock(
        return_value=reply(text("ok"), used=usage(input_tokens=50_000, output_tokens=0))
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        adapter.chat([Message.user("a")])
    assert _money(50_000 * 0.10 * 1.1 / 1_000_000) in llm_lines(caplog)[0]


# === turn shapes: pause_turn, mixed tools, thinking, server-tool errors, refusal ==============


SEARCH_USE = {
    "type": "server_tool_use",
    "id": "srvtoolu_01S",
    "name": "web_search",
    "input": {"query": "q"},
}
SEARCH_RESULT = {
    "type": "web_search_tool_result",
    "tool_use_id": "srvtoolu_01S",
    "content": [
        {
            "type": "web_search_result",
            "url": "https://example.com/a",
            "title": "A",
            "encrypted_content": "EncryptedFakeContent==",
            "page_age": None,
        }
    ],
}


def test_a_paused_turn_is_resumed_with_the_paused_content_unchanged(router, provider, caplog):
    route = router.post(MESSAGES).mock(
        side_effect=[
            reply(thinking("t1"), SEARCH_USE, SEARCH_RESULT, stop_reason="pause_turn"),
            reply(text("answer")),
        ]
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        answer = provider.chat([Message.user("search it")])
    assert answer.content == "answer" and not answer.tool_calls
    resumed = body(route, 1)["messages"]
    assert resumed[0] == {"role": "user", "content": [text("search it")]}
    assert resumed[1] == {
        "role": "assistant",
        "content": [thinking("t1"), SEARCH_USE, SEARCH_RESULT],
    }
    assert len(llm_lines(caplog)) == 2  # a line per billed call


def test_a_turn_that_never_stops_pausing_is_a_transient_fault(router, provider):
    router.post(MESSAGES).mock(return_value=reply(SEARCH_USE, stop_reason="pause_turn"))
    with pytest.raises(ProviderResponseError, match="paused the turn"):
        provider.chat([Message.user("search it")])
    assert len(router.calls) == PAUSE_CONTINUATIONS + 1


def test_a_server_tool_beside_a_client_tool_goes_back_with_the_client_result(router, provider):
    fetch = {
        "type": "server_tool_use",
        "id": "srvtoolu_01F",
        "name": "web_fetch",
        "input": {"url": "https://example.com"},
    }
    route = router.post(MESSAGES).mock(
        side_effect=[
            reply(
                thinking("t"),
                fetch,
                tool_use("toolu_01C", TOOL.name, {"label": "x"}),
                stop_reason="tool_use",
            ),
            reply(text("the number is 7")),
        ]
    )
    convo = [Message.user("fetch it and get the number")]
    step = provider.chat(convo, tools=[TOOL])
    assert step.tool_calls == [ToolCall("toolu_01C", TOOL.name, {"label": "x"})]
    convo += [step, Message.tool("toolu_01C", "7")]
    provider.chat(convo, tools=[TOOL])
    sent = body(route, 1)["messages"]
    assert sent[1]["content"] == [
        thinking("t"),
        fetch,
        tool_use("toolu_01C", TOOL.name, {"label": "x"}),
    ]
    assert sent[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_01C", "content": "7"}
    ]


def test_thinking_goes_back_verbatim_while_the_prefix_holds(router, provider):
    route = router.post(MESSAGES).mock(
        side_effect=[
            reply(
                thinking("plan", "SigA"),
                tool_use("toolu_01D", TOOL.name, {}),
                stop_reason="tool_use",
            ),
            reply(
                thinking("more", "SigB"),
                tool_use("toolu_01E", TOOL.name, {}),
                stop_reason="tool_use",
            ),
            reply(text("done")),
        ]
    )
    convo = [Message.system("charter"), Message.system("brief"), Message.user("go")]
    for result in ("1", "2"):
        step = provider.chat(convo, tools=[TOOL])
        convo += [step, Message.tool(step.tool_calls[0].id, result)]
    provider.chat(convo, tools=[TOOL])
    last = body(route, 2)["messages"]
    assert last[1]["content"][0] == thinking("plan", "SigA")
    assert last[3]["content"][0] == thinking("more", "SigB")


def test_a_changed_prefix_drops_the_thinking_from_there_on(router, provider):
    """A batched wake's second turn no longer carries the first turn's brief: the chain is broken."""
    route = router.post(MESSAGES).mock(
        side_effect=[
            reply(thinking("one", "SigA"), text("first answer")),
            reply(text("second answer")),
        ]
    )
    first = provider.chat(
        [Message.system("charter"), Message.system("brief 1"), Message.user("one")]
    )
    provider.chat(
        [
            Message.system("charter"),
            Message.user("one"),
            first,
            Message.system("brief 2"),
            Message.user("two"),
        ]
    )
    sent = body(route, 1)["messages"]
    assert sent[1] == {"role": "assistant", "content": [text("first answer")]}
    assert "SigA" not in json.dumps(body(route, 1))


def test_a_renamed_call_id_rides_the_replayed_block(router, provider):
    route = router.post(MESSAGES).mock(
        side_effect=[
            reply(
                tool_use("toolu_01X", TOOL.name, {}),
                tool_use("toolu_01X", TOOL.name, {}),
                stop_reason="tool_use",
            ),
            reply(text("ok")),
        ]
    )
    convo = [Message.user("twice")]
    step = provider.chat(convo, tools=[TOOL])
    step.tool_calls[1].id = "toolu_01X-1"  # what `_engine._unique_call_ids` does to a duplicate
    convo += [step, Message.tool("toolu_01X", "a"), Message.tool("toolu_01X-1", "b")]
    provider.chat(convo, tools=[TOOL])
    ids = [block["id"] for block in body(route, 1)["messages"][1]["content"]]
    assert ids == ["toolu_01X", "toolu_01X-1"]


def test_a_server_tool_error_inside_a_200_is_logged_and_the_turn_goes_on(router, provider, caplog):
    error = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_01S",
        "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
    }
    router.post(MESSAGES).mock(return_value=reply(SEARCH_USE, error, text("I could not search.")))
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        answer = provider.chat([Message.user("search")])
    assert answer.content == "I could not search."
    assert "error_code=max_uses_exceeded" in caplog.text


def test_citations_become_a_sources_footer(router, provider):
    cited = text(
        "The sky is blue.",
        citations=[
            {
                "type": "web_search_result_location",
                "url": "https://example.com/sky",
                "title": "Sky",
                "encrypted_index": "FakeIndex",
                "cited_text": "blue",
            }
        ],
    )
    router.post(MESSAGES).mock(return_value=reply(SEARCH_USE, SEARCH_RESULT, cited))
    answer = provider.chat([Message.user("sky?")])
    assert answer.content == "The sky is blue.\n\nSources:\n- Sky — https://example.com/sky"


def test_a_refusal_is_final_and_logged_with_its_category(router, provider, caplog):
    router.post(MESSAGES).mock(
        return_value=reply(
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": "cyber", "explanation": "Declined."},
        )
    )
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        answer = provider.chat([Message.user("…")])
    assert answer.content is None and not answer.tool_calls
    assert len(router.calls) == 1
    assert "category=cyber" in caplog.text
    assert provider.last_finish_reason == "refusal"


def test_running_out_of_room_is_a_truncation():
    assert {"max_tokens", "model_context_window_exceeded"} <= TRUNCATED_FINISH_REASONS


def test_code_execution_without_web_tools_states_the_container_gap_once(router, caplog):
    adapter = AnthropicProvider(
        MODEL, api_key=FAKE_KEY, max_tokens=64, builtin_tools=["code_execution"]
    )
    ran = {
        "type": "bash_code_execution_tool_result",
        "tool_use_id": "srvtoolu_01K",
        "content": {
            "type": "bash_code_execution_result",
            "stdout": "2\n",
            "stderr": "",
            "return_code": 0,
            "content": [],
        },
    }
    use = {
        "type": "server_tool_use",
        "id": "srvtoolu_01K",
        "name": "bash_code_execution",
        "input": {"command": "echo 2"},
    }
    router.post(MESSAGES).mock(return_value=reply(use, ran, text("2")))
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        adapter.chat([Message.user("a")])
        adapter.chat([Message.user("b")])
    assert sum("container time" in r.getMessage() for r in caplog.records) == 1


def test_code_execution_beside_web_fetch_states_no_gap(router, caplog):
    adapter = AnthropicProvider(
        MODEL, api_key=FAKE_KEY, max_tokens=64, builtin_tools=["code_execution", "web_fetch"]
    )
    ran = {
        "type": "bash_code_execution_tool_result",
        "tool_use_id": "srvtoolu_01K",
        "content": {
            "type": "bash_code_execution_result",
            "stdout": "",
            "stderr": "",
            "return_code": 0,
            "content": [],
        },
    }
    router.post(MESSAGES).mock(return_value=reply(ran, text("ok")))
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        adapter.chat([Message.user("a")])
    assert "container time" not in caplog.text


# === the Models API =========================================================================


MODEL_RECORD = {
    "type": "model",
    "id": MODEL,
    "display_name": "Claude Haiku 5.5",
    "created_at": "2026-09-01T00:00:00Z",
    "lifecycle": "active",
    "max_input_tokens": 1_000_000,
    "max_tokens": 128_000,
    "capabilities": None,
}


def test_an_unset_max_tokens_is_the_models_own_maximum_read_once(router):
    adapter = AnthropicProvider(MODEL, api_key=FAKE_KEY)
    info = router.get(MODEL_INFO).mock(return_value=httpx.Response(200, json=MODEL_RECORD))
    route = router.post(MESSAGES).mock(return_value=reply(text("ok")))
    adapter.chat([Message.user("a")])
    adapter.chat([Message.user("b")])
    assert [body(route, i)["max_tokens"] for i in (0, 1)] == [128_000, 128_000]
    assert len(info.calls) == 1
    assert adapter.context_limit() == 1_000_000
    assert info.calls.last.request.extensions["timeout"]["read"] == 60.0


def test_vision_is_read_off_the_models_capabilities(router, provider):
    capable = {"supported": True}
    record = {**MODEL_RECORD, "capabilities": _capabilities(image_input=capable)}
    router.get(MODEL_INFO).mock(return_value=httpx.Response(200, json=record))
    assert provider.supports_vision() is True
    assert provider.supports_video() is False


def test_an_unreadable_models_record_fails_safe(router, provider, caplog):
    router.get(MODEL_INFO).mock(return_value=failure(500, "api_error", "boom"))
    assert provider.supports_vision() is None
    assert provider.context_limit() is None


def _capabilities(**overrides) -> dict:
    no = {"supported": False}
    base = {
        "batch": no,
        "citations": no,
        "code_execution": no,
        "context_management": {
            "supported": False,
            "clear_thinking_20251015": None,
            "clear_tool_uses_20250919": None,
            "compact_20260112": None,
        },
        "effort": {"supported": False, "low": no, "medium": no, "high": no, "max": no},
        "image_input": no,
        "pdf_input": no,
        "server_tools": {"supported": False, "web_search": no, "code_execution": no},
        "structured_outputs": no,
        "thinking": {"supported": False, "types": {"adaptive": no, "enabled": no}},
    }
    return {**base, **overrides}


# === faults, by nature =======================================================================


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            failure(429, "rate_limit_error", "slow down", **{"retry-after": "2"}),
            ProviderRateLimitError,
        ),
        (failure(529, "overloaded_error", "Overloaded"), ProviderServerError),
        (failure(500, "api_error", "Internal"), ProviderServerError),
        (failure(504, "timeout_error", "Timed out"), ProviderTimeoutError),
        (failure(401, "authentication_error", "invalid x-api-key"), ProviderAuthError),
        (failure(403, "permission_error", "no access"), ProviderAuthError),
        (
            failure(
                400,
                "invalid_request_error",
                "Your credit balance is too low to access the Anthropic API.",
            ),
            ProviderBillingError,
        ),
        (failure(402, "billing_error", "Payment required"), ProviderBillingError),
        (
            failure(
                400, "invalid_request_error", "prompt is too long: 250000 tokens > 200000 maximum"
            ),
            ProviderContextLengthError,
        ),
        (
            failure(
                413, "request_too_large", "Request exceeds the maximum allowed number of bytes."
            ),
            ProviderPayloadTooLargeError,
        ),
        (
            failure(400, "invalid_request_error", "tool_choice: forced tool use is not supported"),
            ProviderAPIError,
        ),
    ],
)
def test_faults_are_classified_by_nature(router, provider, response, expected):
    router.post(MESSAGES).mock(return_value=response)
    with pytest.raises(expected) as caught:
        provider.chat([Message.user("hi")])
    assert type(caught.value) is expected
    if expected is ProviderRateLimitError:
        assert caught.value.retry_after == 2.0


def test_a_dropped_connection_and_a_timeout_are_transient(router, provider):
    route = router.post(MESSAGES)
    route.mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ProviderConnectionError):
        provider.chat([Message.user("hi")])
    route.mock(side_effect=httpx2.ReadTimeout("slow"))  # the SDK runs on HTTPX2
    with pytest.raises(ProviderTimeoutError):
        provider.chat([Message.user("hi")])


def test_the_sdk_never_retries_on_its_own(router, provider):
    router.post(MESSAGES).mock(return_value=failure(529, "overloaded_error", "Overloaded"))
    with pytest.raises(ProviderServerError):
        provider.chat([Message.user("hi")])
    assert len(router.calls) == 1


# === the server tools, as plugins ===========================================================


def _resolved(home, provider: str):
    from basecradle_harness import ActivationContext, load_plugins, resolve_plugins

    ctx = ActivationContext(
        provider=provider,
        sdk=provider if provider == "anthropic" else "openai",
        surface="messages" if provider == "anthropic" else "responses",
        model=MODEL,
        env={"AI_API_KEY": FAKE_KEY},
    )
    return resolve_plugins(load_plugins(home), ctx)


def test_under_anthropic_web_fetch_is_claudes_own_and_everywhere_else_the_harnesss(tmp_path):
    from basecradle_harness import WebFetchTool, install

    claude, other = tmp_path / "claude", tmp_path / "other"
    install(claude, provider="anthropic")
    install(other, provider="openai")
    assert (claude / "tools" / "anthropic_web_fetch.py").exists()  # benign: a shipped default
    assert not (other / "tools" / "anthropic_web_fetch.py").exists()
    assert (other / "tools" / "web_fetch.py").exists()  # never pruned off another vendor's agent
    on_claude = _resolved(claude, "anthropic")
    assert "web_fetch" in on_claude.builtins
    assert not any(isinstance(tool, WebFetchTool) for tool in on_claude.tools)
    assert "web_fetch" in dict(on_claude.manifest)
    assert "only a URL that already" in dict(on_claude.manifest)["web_fetch"]
    on_other = _resolved(other, "openai")
    assert "web_fetch" not in on_other.builtins
    assert any(isinstance(tool, WebFetchTool) for tool in on_other.tools)


def test_web_search_and_code_execution_are_opt_in_server_tools_under_anthropic(tmp_path):
    from basecradle_harness import install

    home = tmp_path / "claude"
    install(home, provider="anthropic")
    assert not {"web_search", "code_execution"} & set(_resolved(home, "anthropic").builtins)
    install(home, provider="anthropic", opt_in=["anthropic_search", "code_execution"])
    granted = _resolved(home, "anthropic")
    assert {"web_search", "web_fetch", "code_execution"} <= set(granted.builtins)
    assert {"anthropic_search", "code_execution"} <= set(granted.opt_in_stems)


def test_the_off_box_resolver_names_the_same_stems():
    from basecradle_harness._resolve import resolve_stems

    report = resolve_stems(
        provider="anthropic", sdk="anthropic", model=MODEL, opt_in="anthropic_search,code_execution"
    )
    assert {"web_search", "web_fetch", "code_execution"} <= set(report["builtins"])


# === review findings, pinned ================================================================


def test_a_reply_produced_after_a_strip_keeps_its_own_thinking(router, provider):
    """Only a reply whose own prefix changed loses its thinking — never one made on the new prefix."""
    route = router.post(MESSAGES).mock(
        side_effect=[
            reply(thinking("one", "SigA"), text("first answer")),
            reply(
                thinking("two", "SigC"),
                tool_use("toolu_01T", TOOL.name, {}),
                stop_reason="tool_use",
            ),
            reply(text("done")),
        ]
    )
    first = provider.chat(
        [Message.system("charter"), Message.system("brief 1"), Message.user("one")]
    )
    convo = [
        Message.system("charter"),
        Message.user("one"),
        first,
        Message.system("brief 2"),
        Message.user("two"),
    ]
    step = provider.chat(convo, tools=[TOOL])
    convo += [step, Message.tool("toolu_01T", "7")]
    provider.chat(convo, tools=[TOOL])
    sent = json.dumps(body(route, 2))
    assert "SigA" not in sent  # its prefix lost brief 1
    assert "SigC" in sent  # produced on the stripped prefix, replayed onto exactly that prefix


def test_a_paused_turns_empty_text_block_is_not_sent_back(router, provider):
    route = router.post(MESSAGES).mock(
        side_effect=[reply(text(""), SEARCH_USE, stop_reason="pause_turn"), reply(text("ok"))]
    )
    provider.chat([Message.user("search")])
    assert body(route, 1)["messages"][1] == {"role": "assistant", "content": [SEARCH_USE]}


def test_the_served_geography_is_what_is_priced(router, provider, caplog):
    router.post(MESSAGES).mock(
        return_value=reply(
            text("ok"), used=usage(input_tokens=50_000, output_tokens=0, inference_geo="us")
        )
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("a")])
    assert _money(50_000 * 0.10 * 1.1 / 1_000_000) in llm_lines(caplog)[0]


def test_a_spend_limit_is_the_billing_class(router, provider):
    router.post(MESSAGES).mock(
        return_value=failure(
            400,
            "invalid_request_error",
            "You have reached your specified API usage limits. You will regain access on "
            "2026-11-01 at 00:00 UTC.",
        )
    )
    with pytest.raises(ProviderBillingError):
        provider.chat([Message.user("hi")])


@pytest.mark.parametrize("owned", ["tools", "system", "messages", "cache_control"])
def test_extra_body_cannot_override_the_harnesss_wiring(owned):
    with pytest.raises(AnthropicConfigError) as caught:
        AnthropicProvider(MODEL, api_key=FAKE_KEY, extra_body={owned: []})
    assert caught.value.reason == "invalid_model_params"


@pytest.mark.parametrize(
    "params", [{"reasoning_effort": "low"}, {"cache_control": {"type": "persistent"}}]
)
def test_resolved_config_refuses_what_startup_refuses(tmp_path, monkeypatch, params):
    home = tmp_path / "config"
    home.mkdir()
    (home / "model_params.json").write_text(json.dumps(params))
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(home))
    with pytest.raises(ValueError):
        resolved_model_params("anthropic")


def test_resolved_config_passes_a_good_file(tmp_path, monkeypatch):
    home = tmp_path / "config"
    home.mkdir()
    good = {"max_tokens": 4096, "output_config": {"effort": "low"}, "model": "ignored"}
    (home / "model_params.json").write_text(json.dumps(good))
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(home))
    assert resolved_model_params("anthropic") == (good, ["model"])
