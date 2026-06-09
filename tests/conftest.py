"""Shared fixtures. All HTTP is mocked with respx — no test ever touches a model.

The endpoint is a fabricated OpenAI-compatible host; the key is a correctly-shaped
fake. Model responses follow the OpenAI chat-completions schema.
"""

import json

import pytest
import respx

from basecradle_harness import OpenAICompatibleProvider, OpenAIResponsesProvider

# A fabricated OpenAI-compatible endpoint and a correctly-shaped fake key.
BASE_URL = "https://api.openai.test/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
RESPONSES_URL = f"{BASE_URL}/responses"
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


@pytest.fixture
def responses_provider():
    """A Responses-API provider pointed at the fabricated endpoint (web_search default)."""
    p = OpenAIResponsesProvider(model="gpt-5.4-mini", api_key=FAKE_KEY, base_url=BASE_URL)
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


# --- Responses API shapes (the second adapter) -------------------------------


def responses_body(*output):
    """A Responses-API response body wrapping the given `output` items."""
    return {
        "id": "resp-fake0001",
        "object": "response",
        "model": "gpt-5.4-mini",
        "output": list(output),
    }


def out_message(text, *, annotations=None):
    """A Responses `message` output item: assistant text with optional citations."""
    content = {"type": "output_text", "text": text}
    if annotations is not None:
        content["annotations"] = annotations
    return {
        "id": "msg-fake0001",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [content],
    }


def out_function_call(*, call_id, name, arguments):
    """A Responses `function_call` output item — a custom tool the harness must run."""
    return {
        "id": "fc-fake0001",
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


def out_web_search_call(*, query="latest news"):
    """A Responses `web_search_call` output item — resolved server-side, never run here."""
    return {
        "id": "ws-fake0001",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "query": query},
    }


def url_citation(*, url, title, start_index=0, end_index=1):
    """A `url_citation` annotation, as web_search attaches to message text."""
    return {
        "type": "url_citation",
        "start_index": start_index,
        "end_index": end_index,
        "url": url,
        "title": title,
    }
