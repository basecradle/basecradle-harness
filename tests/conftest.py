"""Shared fixtures. All HTTP is mocked with respx — no test ever touches a model.

The endpoint is a fabricated OpenAI-compatible host; the key is a correctly-shaped fake. respx
mocks the HTTP client at the transport level, so it intercepts each SDK's own client without
any network — the SDK adapter is tested against real, SDK-valid response bodies. Model
responses follow the OpenAI chat-completions / Responses schemas.
"""

import json
import re

import pytest
import respx
import respx.mocks
from respx.mocks import HTTPCoreMocker

from basecradle_harness import OpenAIProvider

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """A log line as a human reads it — the ANSI verdict colors stripped back off (issue #414).

    The colors are wrapped around **whole tokens**, so almost every existing assertion (a substring
    search for ``outcome=ok``, for ``wake failed``) matches the colored bytes untouched. The ones
    that need this are the *anchored* ones — ``startswith("wake end")`` — which is exactly the class
    of consumer the token-integrity rule cannot protect. Tests that assert on the color itself read
    the raw record instead; this is for the ones that only ever cared what the line *said*.
    """
    return _ANSI.sub("", text)


# A fabricated OpenAI-compatible endpoint and a correctly-shaped fake key.
BASE_URL = "https://api.openai.test/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
RESPONSES_URL = f"{BASE_URL}/responses"
FAKE_KEY = "sk-test-0123456789abcdefghijklmnop"


class _HTTPCoreBothMocker(HTTPCoreMocker):
    """respx's transport mocker, patching **both** HTTPX families this suite drives.

    ``openai>=3`` moved its client to HTTPX2 — a separate distribution on ``httpcore2``, not a
    new major of ``httpx`` — while everything else the harness speaks HTTP with is still legacy
    ``httpx``: the BaseCradle SDK, the OpenRouter SDK, and the harness's own calls (web_fetch,
    the Grok tools, the ntfy push, asset downloads). A respx router patches **one** family, so a
    default router would silently stop intercepting the model endpoint the moment the SDK moved
    — which is exactly how this arrived: every route still registered, none of them called.

    Both families are patched by one mocker rather than by two routers because the *same* test
    routinely mocks both at once (every wake test mocks the platform **and** the model), and a
    split would make ``assert_all_called`` answer for only half the routes. respx converts each
    intercepted request to an ``httpx.Request`` and hands back an ``httpcore.Response``; HTTPX2
    reads both structurally, so the legacy objects the routes are written against keep working
    on either transport — which is why the suite needed no per-test change.

    Installed as respx's default mocker below, so the ~220 bare ``respx.mock(...)`` call sites
    (and the `router` fixture) get it without naming it — subclass-plus-``DEFAULT_MOCKER`` is the
    extension respx's own maintainer points forked-transport users at (respx#316), and the same
    shape his ``pytest-httpx2`` plugin ships; this one covers both families in a single mocker
    rather than the plugin's httpcore2-only one, which is the part the mixed suite needs.
    """

    name = "httpcore+httpcore2"
    targets = [
        *HTTPCoreMocker.targets,
        "httpcore2._sync.connection.HTTPConnection",
        "httpcore2._sync.connection_pool.ConnectionPool",
        "httpcore2._sync.http_proxy.HTTPProxy",
        "httpcore2._async.connection.AsyncHTTPConnection",
        "httpcore2._async.connection_pool.AsyncConnectionPool",
        "httpcore2._async.http_proxy.AsyncHTTPProxy",
    ]


respx.mocks.DEFAULT_MOCKER = _HTTPCoreBothMocker.name


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


def out_code_interpreter_call(*, code, container_id="cntr_fake0001"):
    """A Responses `code_interpreter_call` output item — code run server-side (issue #172)."""
    return {
        "id": "ci-fake0001",
        "type": "code_interpreter_call",
        "status": "completed",
        "container_id": container_id,
        "code": code,
        "outputs": [],
    }


def container_file_citation(*, file_id, filename, container_id="cntr_fake0001"):
    """A `container_file_citation` annotation — a file the Code Interpreter produced."""
    return {
        "type": "container_file_citation",
        "container_id": container_id,
        "file_id": file_id,
        "filename": filename,
        "start_index": 0,
        "end_index": 1,
    }
