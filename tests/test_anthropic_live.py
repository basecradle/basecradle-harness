"""Live gate for the Anthropic adapter (`AnthropicProvider`) — Claude, called direct (issue #725).

The one check the offline suite **structurally cannot** make. `test_anthropic.py` drives the real SDK
against respx, so it proves the request the SDK serializes and the readers the adapter applies to a
body *we wrote*; it cannot prove that Anthropic accepts that request, that it still answers in the
shape we read, or that a cache breakpoint placed where we place it is ever read back. Every one of
those was written from Anthropic's documentation, and this file is where Anthropic answers for
itself — a fault class this repo believes in is checked live, or it is a guess with a test around it.

It is an explicitly-marked **live** job (`@pytest.mark.live`), deselected from the default run and
skipped without a credential. It reads the harness live suite's own key — never an agent's::

    set -a; . ~/.config/basecradle/harness-test.env; set +a
    uv run pytest -m live tests/test_anthropic_live.py -v

``ANTHROPIC_LIVE_API_KEY`` is the key; ``ANTHROPIC_LIVE_MODEL`` defaults to ``claude-haiku-5-5``. A
SKIP here is **not** a pass: pytest exits 0 when every test skips, so read the report, not the exit
code. What each test pins is a value, never mere presence: the cache read is a positive count on the
second call over the same prefix, the cost equals the table's own arithmetic over the usage
Anthropic reported, a fetched page's title is the page's, a computed hash is the right hash.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os

import pytest

from basecradle_harness import (
    AnthropicProvider,
    ImageContent,
    Message,
    ProviderAuthError,
    ToolSpec,
)
from basecradle_harness._anthropic_rates import Usage, call_cost
from basecradle_harness._caching import EXPLICIT, anchor_cacheable_prefix
from basecradle_harness._observability import _money

pytestmark = pytest.mark.live

KEY = (os.environ.get("ANTHROPIC_LIVE_API_KEY") or "").strip() or None
MODEL = (os.environ.get("ANTHROPIC_LIVE_MODEL") or "claude-haiku-5-5").strip()

needs_key = pytest.mark.skipif(
    not KEY, reason="set ANTHROPIC_LIVE_API_KEY to run the live Anthropic gate"
)

NUMBER_TOOL = ToolSpec(
    name="get_secret_number",
    description="Returns the secret number for a given label. Always call it; never guess.",
    parameters={
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    },
)

#: A charter comfortably past the 512-token minimum cacheable prefix of the 5.x models, distinct
#: from anything another test sends, so the first call writes the cache and the second reads it.
CHARTER = (
    "You are Nova Digital, a test agent of the BaseCradle Harness live suite.\n\n"
    + "\n".join(
        f"Rule {n}: when asked about topic {n}, answer in one short sentence and never invent facts "
        f"about the number {n * 7}, the colour of lantern {n}, or the river that runs past mill {n}."
        for n in range(1, 90)
    )
)


def _provider(**tuning) -> AnthropicProvider:
    tuning.setdefault("max_tokens", 2048)
    return AnthropicProvider(MODEL, api_key=KEY, **tuning)


def _lines(caplog, head: str) -> list[str]:
    return [m for m in (r.getMessage() for r in caplog.records) if m.startswith(head)]


def _field(line: str, name: str) -> str | None:
    for part in line.split():
        if part.startswith(f"{name}="):
            return part.split("=", 1)[1]
    return None


@needs_key
def test_a_chat_reply_and_its_line(caplog):
    adapter = _provider()
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        answer = adapter.chat([Message.user("Reply with exactly the word: lighthouse")])
    assert "lighthouse" in (answer.content or "").lower()
    (line,) = _lines(caplog, "llm provider=")
    assert line.startswith(f"llm provider=anthropic purpose=main model={MODEL} ")
    assert _field(line, "cost_basis") == "computed"
    assert int(_field(line, "tokens_in")) == adapter.last_tokens_in > 0
    assert _field(line, "generation_id").startswith("msg_")


@needs_key
def test_the_second_call_on_the_same_prefix_reads_the_cache(caplog):
    """The fleet's first explicit-cache path: the breakpoint we place must be read back."""
    adapter = _provider()
    convo = [Message.system(CHARTER), Message.system("brief"), Message.user("Topic 3?")]
    anchored = anchor_cacheable_prefix(convo, stable=1, mode=EXPLICIT)
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        adapter.chat(anchored)
        adapter.chat(anchor_cacheable_prefix(convo, stable=1, mode=EXPLICIT))
    first, second = _lines(caplog, "llm provider=")
    assert (
        int(_field(first, "cache_write_tokens") or 0) > 0
        or int(_field(first, "cached_tokens") or 0) > 0
    )
    cached = int(_field(second, "cached_tokens"))
    assert cached > 0, second
    assert cached >= int(_field(first, "cache_write_tokens") or cached)


@needs_key
def test_the_cost_is_the_tables_arithmetic_over_the_reported_usage(caplog):
    """The line's ``cost=`` is the table's figure over the very counts the same line reports."""
    adapter = _provider()
    convo = [Message.system(CHARTER), Message.user("Topic 5?")]
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        adapter.chat(anchor_cacheable_prefix(convo, stable=1, mode=EXPLICIT))
    (line,) = _lines(caplog, "llm provider=")
    read = int(_field(line, "cached_tokens") or 0)
    written = int(_field(line, "cache_write_tokens") or 0)
    reported = Usage(
        uncached=int(_field(line, "tokens_in")) - read - written,
        write_5m=written,
        write_1h=0,
        read=read,
        output=int(_field(line, "tokens_out")),
    )
    expected = call_cost(MODEL, reported)
    assert expected is not None and expected > 0
    assert _field(line, "cost") == _money(expected)


@needs_key
def test_a_web_search_runs_server_side_and_its_fee_is_logged(caplog):
    adapter = _provider(builtin_tools=["web_search"])
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        answer = adapter.chat(
            [Message.user("Search the web: what is the current stable version of Python? Cite it.")]
        )
    llm = _lines(caplog, "llm provider=")
    searches = sum(int(_field(line, "web_search_requests") or 0) for line in llm)
    assert searches >= 1, llm
    media = _lines(caplog, "media provider=anthropic kind=search.web")
    assert sum(int(_field(line, "count")) for line in media) == searches  # one per billed call
    assert sum(float(_field(line, "cost")) for line in media) == pytest.approx(searches * 0.01)
    # With dynamic filtering (web_search_20260318) the search runs inside Anthropic's code
    # execution and the model cites its sources in its own words, when it does: no citation blocks
    # arrive, so no Sources: footer is built (measured 2026-10-31). What is pinned is the vendor's
    # own result — a successful search result list carrying real URLs, kept for the next step.
    assert answer.content
    kept = next(iter(adapter._replies.values()))[1]
    results = [b["content"] for b in kept if b.get("type") == "web_search_tool_result"]
    assert results and all(isinstance(found, list) for found in results)
    assert any(item.get("url", "").startswith("https://") for found in results for item in found)


@needs_key
def test_a_web_fetch_reads_a_url_given_in_the_user_turn(caplog):
    adapter = _provider(builtin_tools=["web_fetch"])
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        answer = adapter.chat(
            [Message.user("Fetch https://example.com and tell me the page's exact title.")]
        )
    assert "example domain" in (answer.content or "").lower()
    assert "server tool failed" not in caplog.text
    kept = next(iter(adapter._replies.values()))[1]
    assert any(block.get("type") == "web_fetch_tool_result" for block in kept)


@needs_key
def test_code_execution_computes_the_right_answer():
    adapter = _provider(builtin_tools=["code_execution"])
    answer = adapter.chat(
        [
            Message.user(
                "Use code execution to compute the sha256 hex digest of the string nova-725."
            )
        ]
    )
    assert hashlib.sha256(b"nova-725").hexdigest() in (answer.content or "")
    kept = next(iter(adapter._replies.values()))[1]
    assert any(block.get("type") == "bash_code_execution_tool_result" for block in kept)


@needs_key
def test_an_image_is_seen_and_described():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 20, 20)).save(buffer, format="PNG")
    import base64

    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    adapter = _provider()
    assert adapter.supports_vision() is True
    answer = adapter.chat(
        [
            Message(
                role="user",
                content="What single colour fills this image? One word.",
                images=[ImageContent(url=url)],
            )
        ]
    )
    assert "red" in (answer.content or "").lower()


@needs_key
def test_the_models_api_states_the_context_window():
    limit = _provider().context_limit()
    assert isinstance(limit, int) and limit >= 200_000


@needs_key
def test_a_tool_loop_replays_its_thinking_and_is_accepted():
    """Preserved thinking: the raw blocks go back unchanged between steps, and Anthropic takes them."""
    adapter = _provider(thinking={"type": "adaptive", "display": "summarized"})
    convo = [
        Message.system("Use tools when asked."),
        Message.user("Get the secret number for 'blue'."),
    ]
    step = adapter.chat(convo, tools=[NUMBER_TOOL])
    assert step.tool_calls, "the model did not call the tool"
    convo += [step, *(Message.tool(call.id, "42") for call in step.tool_calls)]
    final = adapter.chat(convo, tools=[NUMBER_TOOL])
    assert "42" in (final.content or "")


@needs_key
def test_a_changed_prefix_strips_the_thinking_and_is_accepted():
    """A batched wake's second turn drops the first turn's brief; the kept reply's thinking must go."""
    adapter = _provider(thinking={"type": "adaptive", "display": "summarized"})
    first = adapter.chat(
        [Message.system("charter"), Message.system("brief one"), Message.user("Name a colour.")]
    )
    second = adapter.chat(
        [
            Message.system("charter"),
            Message.user("Name a colour."),
            first,
            Message.system("brief two"),
            Message.user("Now name a different one."),
        ]
    )
    assert second.content


@needs_key
def test_a_wrong_key_is_the_credential_class():
    adapter = AnthropicProvider(MODEL, api_key="sk-ant-api03-not-a-real-key-000000", max_tokens=16)
    with pytest.raises(ProviderAuthError):
        adapter.chat([Message.user("hi")])
