"""A reasoning brain gets its own reasoning back with its tool results (issue #839).

OpenRouter asks for it: *"when you post tool results, including the original reasoning ensures the
model can continue its reasoning from where it left off."* Everything here is read off the request
body the real SDK put on the (mocked) wire, because this capability's whole job is to put bytes
there: an adapter that merely decided to send them has proved nothing (issue #433).
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx
from openrouter import OpenRouter

from basecradle_harness import (
    Engine,
    Message,
    OpenAIProvider,
    OpenRouterProvider,
    Session,
    Tool,
    ToolCall,
    ToolRegistry,
)
from basecradle_harness._openai_wire import chat_messages_to_wire

BASE_URL = "https://openrouter.test/api/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
FAKE_KEY = "sk-or-v1-0123456789abcdef0123456789abcdef"
MODEL = "moonshotai/kimi-k3.1-pro"

#: One of each block type OpenRouter documents, in the order the model wrote them, with non-Latin
#: text, because the copy must be exact and the ordering is the vendor's stated requirement.
BLOCKS = [
    {
        "type": "reasoning.summary",
        "summary": "Check the weather first.",
        "format": "xai-responses-v1",
        "index": 0,
    },
    {"type": "reasoning.text", "text": "東京の天気を見る", "format": "unknown", "index": 1},
    {
        "type": "reasoning.encrypted",
        "data": "ZW5jcnlwdGVk",
        "id": "rs_0193",
        "format": "openai-responses-v1",
        "index": 2,
    },
]
LATER_BLOCKS = [{"type": "reasoning.text", "text": "Now the forecast.", "index": 0}]


class EchoTool(Tool):
    name = "echo"
    description = "Echo."

    def run(self, **kwargs):
        return "echoed"


def call(call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "echo", "arguments": "{}"},
    }


def completion(*, content=None, tool_calls=None, **reasoning):
    """A recorded-shape OpenRouter completion, with whichever reasoning fields it is given."""
    message = {"role": "assistant", "content": content, **reasoning}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "gen-fake0001",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "system_fingerprint": "fp_test",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def native():
    client = OpenRouter(api_key=FAKE_KEY, server_url=BASE_URL, retry_config=None)
    return OpenRouterProvider(MODEL, client=client, base_url=BASE_URL)


def through_openai_sdk(vendor="openrouter"):
    return OpenAIProvider(
        model=MODEL,
        api_key=FAKE_KEY,
        base_url=BASE_URL,
        provider=vendor,
        surface="chat",
        max_retries=0,
    )


@pytest.fixture
def router():
    with respx.mock(assert_all_called=True) as r:
        yield r


def answer(router, *bodies):
    return router.post(CHAT_URL).mock(side_effect=[httpx.Response(200, json=b) for b in bodies])


def assistant_turns(call):
    """The assistant messages of one recorded request, as the wire carried them."""
    return [m for m in json.loads(call.request.content)["messages"] if m["role"] == "assistant"]


def run_loop(router, provider, *first_reply_fields):
    """One tool round by hand: the model calls, the result goes back, and the next request is read."""
    route = answer(
        router,
        completion(tool_calls=[call()], **dict(first_reply_fields)),
        completion(content="Done."),
    )
    messages = [Message.user("What is the weather?")]
    reply = provider.chat(messages)
    messages += [reply, Message.tool("call_1", "sunny")]
    provider.chat(messages)
    return assistant_turns(route.calls[1])


@pytest.mark.parametrize("build", [native, through_openai_sdk], ids=["native", "openai_sdk"])
def test_the_blocks_go_back_exactly_as_the_model_wrote_them(router, build):
    [turn] = run_loop(router, build(), ("reasoning_details", BLOCKS), ("reasoning", "Plain."))

    assert turn["reasoning_details"] == BLOCKS
    assert [block["index"] for block in turn["reasoning_details"]] == [0, 1, 2]


@pytest.mark.parametrize("build", [native, through_openai_sdk], ids=["native", "openai_sdk"])
def test_the_blocks_are_preferred_and_the_string_is_not_sent_beside_them(router, build):
    [turn] = run_loop(router, build(), ("reasoning_details", BLOCKS), ("reasoning", "Plain."))

    assert "reasoning" not in turn


@pytest.mark.parametrize("build", [native, through_openai_sdk], ids=["native", "openai_sdk"])
def test_a_reasoning_string_alone_goes_back_as_the_string(router, build):
    [turn] = run_loop(router, build(), ("reasoning", "東京の天気を見る"))

    assert turn["reasoning"] == "東京の天気を見る"
    assert "reasoning_details" not in turn


@pytest.mark.parametrize("build", [native, through_openai_sdk], ids=["native", "openai_sdk"])
def test_a_model_that_reasons_about_nothing_sends_nothing_back(router, build):
    [turn] = run_loop(router, build())

    assert "reasoning" not in turn and "reasoning_details" not in turn


def test_a_final_answer_keeps_no_reasoning(router):
    answer(router, completion(content="Sunny.", reasoning_details=BLOCKS))

    assert native().chat([Message.user("Weather?")]).reasoning is None


@pytest.mark.parametrize(
    "vendor, details_key_survives",
    [("openrouter", True), ("xai", False), ("acme", False)],
)
def test_only_openrouter_is_sent_the_fields_on_the_openai_sdk(router, vendor, details_key_survives):
    turns = run_loop(router, through_openai_sdk(vendor), ("reasoning_details", BLOCKS))

    assert ("reasoning_details" in turns[0]) is details_key_survives


@pytest.mark.parametrize(
    "unsendable",
    [
        pytest.param(
            [{"type": "reasoning.future_kind", "payload": "x", "index": 0}], id="unknown_type"
        ),
        pytest.param(
            [{"type": "reasoning.text", "text": "t", "index": 0, "new_field": "x"}],
            id="unknown_field",
        ),
    ],
)
def test_blocks_the_sdk_cannot_carry_unchanged_are_not_altered_or_fatal(router, caplog, unsendable):
    """The SDK refuses an unknown type at ``chat.send`` and drops an unknown field: a wake survives
    both, and what would have reached the wire altered does not reach it at all."""
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        [turn] = run_loop(
            router, native(), ("reasoning_details", unsendable), ("reasoning", "Plain.")
        )

    assert "reasoning_details" not in turn
    assert turn["reasoning"] == "Plain."
    assert "cannot send" in caplog.text


def test_each_round_of_a_loop_keeps_its_own_blocks_and_none_of_them_reach_the_disk(
    router, tmp_path
):
    """The real engine, three calls: nothing between the adapter and the wire loses the blocks, and
    nothing writes them to the transcript."""
    route = answer(
        router,
        completion(tool_calls=[call("call_1")], reasoning_details=BLOCKS),
        completion(tool_calls=[call("call_2")], reasoning_details=LATER_BLOCKS),
        completion(content="Done."),
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    path = tmp_path / "session.json"

    assert Session("default", Engine(native(), registry), path=path).send("Weather?") == "Done."

    second, third = assistant_turns(route.calls[1]), assistant_turns(route.calls[2])
    assert [t["reasoning_details"] for t in second] == [BLOCKS]
    assert [t["reasoning_details"] for t in third] == [BLOCKS, LATER_BLOCKS]
    # Stable across the steps of the loop, which is what keeps the prefix cacheable.
    assert third[0] == second[0]
    on_disk = path.read_text()
    assert "reasoning" not in on_disk and "東京" not in on_disk


def _a_turn_with_blocks(call_id):
    return Message(
        role="assistant",
        tool_calls=[ToolCall(id=call_id, name="echo")],
        reasoning={"reasoning_details": BLOCKS},
    )


def test_a_new_turn_of_the_conversation_ends_the_loop_the_blocks_belonged_to():
    wire = chat_messages_to_wire(
        [
            Message.user("First."),
            _a_turn_with_blocks("c1"),
            Message.tool("c1", "ok"),
            Message.assistant("Answered."),
            Message.user("Second."),
            _a_turn_with_blocks("c2"),
            Message.tool("c2", "ok"),
        ]
    )

    assert "reasoning_details" not in wire[1]
    assert wire[5]["reasoning_details"] == BLOCKS


def test_an_injected_turn_is_the_loops_own_work_and_does_not_end_it():
    wire = chat_messages_to_wire(
        [
            Message.user("Look."),
            _a_turn_with_blocks("c1"),
            Message.tool("c1", "ok"),
            Message(role="user", content="[image]", injected=True),
        ]
    )

    assert wire[1]["reasoning_details"] == BLOCKS
