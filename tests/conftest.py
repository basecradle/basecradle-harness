"""Shared fixtures. All HTTP is mocked with respx — no test ever touches a model.

The endpoint is a fabricated OpenAI-compatible host; the key is a correctly-shaped
fake. Model responses follow the OpenAI chat-completions schema.
"""

import json

import pytest
import respx

from basecradle_harness import OpenAICompatibleProvider

# A fabricated OpenAI-compatible endpoint and a correctly-shaped fake key.
BASE_URL = "https://api.openai.test/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
FAKE_KEY = "sk-test-0123456789abcdefghijklmnop"


@pytest.fixture
def router():
    """A respx router; routes are matched by absolute URL."""
    with respx.mock(assert_all_called=True) as r:
        yield r


@pytest.fixture
def provider():
    """A provider pointed at the fabricated endpoint."""
    p = OpenAICompatibleProvider(model="gpt-4o", api_key=FAKE_KEY, base_url=BASE_URL)
    yield p
    p.close()


def completion(*, content=None, tool_calls=None, finish_reason="stop"):
    """A chat-completions response body, OpenAI-shaped."""
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-fake0001",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


def wire_tool_call(*, id, name, arguments):
    """A tool call as it appears on the wire — `arguments` is a JSON string."""
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
