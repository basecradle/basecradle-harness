"""Shared fixtures. All HTTP is mocked with respx — no test ever touches a model.

The endpoint is a fabricated OpenAI-compatible host; the key is a correctly-shaped fake. respx
mocks httpx at the transport level, so it intercepts the ``openai`` SDK's own httpx client the
same way it intercepts the harness's hand-rolled httpx — the SDK adapter is tested against
real, SDK-valid response bodies without any network. Model responses follow the OpenAI
chat-completions / Responses schemas.
"""

import json

import pytest
import respx

from basecradle_harness import OpenAIProvider, OpenAIResponsesProvider

# A fabricated OpenAI-compatible endpoint and a correctly-shaped fake key.
BASE_URL = "https://api.openai.test/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
RESPONSES_URL = f"{BASE_URL}/responses"
FAKE_KEY = "sk-test-0123456789abcdefghijklmnop"


@pytest.fixture(autouse=True)
def _isolated_config_home(tmp_path_factory, monkeypatch):
    """Point the config home at an empty temp dir so no test reads the real ``$HOME``.

    The charter is now sourced from files under ``$BASECRADLE_CONFIG_HOME`` (default
    ``$HOME/.config/basecradle``). Without this, a dev/CI box that has ever run
    ``basecradle-harness-install`` would leak its real charter into every ``from_env``
    test. Pinning the var to a fresh, empty dir makes the whole suite hermetic; a test
    that exercises the config home overrides it (or passes an explicit ``home=``).
    """
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(tmp_path_factory.mktemp("config-home")))


@pytest.fixture
def router():
    """A respx router; routes are matched by absolute URL."""
    with respx.mock(assert_all_called=True) as r:
        yield r


@pytest.fixture
def provider():
    """The openai-SDK adapter on its **chat** surface, pointed at the fabricated endpoint.

    ``max_retries=0`` keeps error tests single-shot and deterministic (the SDK otherwise
    retries 429/5xx with backoff).
    """
    p = OpenAIProvider(
        model="gpt-4o", api_key=FAKE_KEY, base_url=BASE_URL, surface="chat", max_retries=0
    )
    yield p
    p.close()


@pytest.fixture
def responses_provider():
    """The openai-SDK adapter on its **responses** surface (web_search-capable)."""
    p = OpenAIProvider(
        model="gpt-5.4-mini",
        api_key=FAKE_KEY,
        base_url=BASE_URL,
        surface="responses",
        max_retries=0,
    )
    yield p
    p.close()


@pytest.fixture
def xai_responses_provider():
    """The xAI **interim httpx** Responses adapter, pointed at the fabricated endpoint.

    This is the death-row hand-rolled path the ``xai`` profile still uses until the native
    ``xai-sdk`` adapter lands (issue #158, Q3) — exercised here to keep it honest.
    """
    p = OpenAIResponsesProvider(model="grok-4.3", api_key=FAKE_KEY, base_url=BASE_URL)
    yield p
    p.close()


def completion(*, content=None, tool_calls=None, finish_reason="stop"):
    """A chat-completions response body, OpenAI-shaped (SDK-valid)."""
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-fake0001",
        "object": "chat.completion",
        "created": 0,
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


# --- Responses API shapes (the Responses surface) ----------------------------


def responses_body(*output):
    """A Responses-API response body wrapping the given `output` items (SDK-valid).

    Carries the fields the ``openai`` SDK's ``Response`` model needs (``created_at``,
    ``parallel_tool_calls``, ``tool_choice``, ``tools``) so the SDK adapter validates it; the
    extra keys are harmless to the xAI httpx adapter, which parses the raw JSON directly.
    """
    return {
        "id": "resp-fake0001",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5.4-mini",
        "output": list(output),
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def out_message(text, *, annotations=None):
    """A Responses `message` output item: assistant text with optional citations.

    ``annotations`` is always present (default ``[]``) — the SDK's ``output_text`` content
    part requires the field.
    """
    content = {"type": "output_text", "text": text, "annotations": annotations or []}
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
