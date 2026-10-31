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
import uuid

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


#: The input tokens at the end of a request that no breakpoint covers — the turn's closing tokens,
#: 4 on ``claude-haiku-5-5`` when measured (2026-10-31); the bound leaves room for a model that
#: counts a few more, and is far below the brief a missed tail would leave uncached.
CLOSING_TOKENS = 16


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
    """The fleet's first explicit-cache path: the anchor we place must be read back.

    Each call changes the brief and the turn after the anchored charter, so the tail entry one call
    writes can never serve the next (issue #729) and the only entry they share is the anchor's. The
    second and third calls each read exactly that entry; with the anchor unmarked they read nothing.
    """
    nonce = uuid.uuid4().hex  # so no earlier run's tail entry matches a brief
    adapter = _provider()
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        for n in (3, 4, 5):
            convo = [
                Message.system(CHARTER),
                Message.system(f"[{nonce}] brief {n}"),
                Message.user(f"Topic {n}?"),
            ]
            adapter.chat(anchor_cacheable_prefix(convo, stable=1, mode=EXPLICIT))
    first, second, third = _lines(caplog, "llm provider=")
    assert sum(_cache(first)) > 0, first
    anchor = _cache(second)[0]
    assert 0 < anchor < sum(_cache(first)), second
    assert _cache(third)[0] == anchor, third


def _cache(line: str) -> tuple[int, int]:
    """A line's ``(cached_tokens, cache_write_tokens)``, an absent field read as zero."""
    return int(_field(line, "cached_tokens") or 0), int(_field(line, "cache_write_tokens") or 0)


@needs_key
def test_with_nothing_anchored_the_tool_list_is_read_back(caplog):
    """A first wake (issue #726): no frozen turn, so the anchor's breakpoint is the tool list's.

    Each call sends a different brief, so the tail entry one call writes (#729) cannot serve the
    next and the only entry they share is the tool list's. The second and third calls each read
    exactly that entry; with the tool list unmarked they read nothing.
    """
    nonce = uuid.uuid4().hex
    tools = [
        ToolSpec(
            name=f"lookup_topic_{n}",
            description=f"[{nonce}] " + CHARTER.splitlines()[n + 1] * 3,
            parameters={"type": "object", "properties": {"topic": {"type": "string"}}},
        )
        for n in range(1, 13)
    ]
    adapter = _provider()
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        for n in range(3):
            brief = Message.system(f"[{nonce}] brief {n}")
            convo = [brief, Message.user("Reply with exactly the word: ok")]
            adapter.chat(anchor_cacheable_prefix(convo, stable=0, mode=EXPLICIT), tools=tools)
    first, second, third = _lines(caplog, "llm provider=")
    assert sum(_cache(first)) > 0, first
    entry = _cache(second)[0]
    assert 0 < entry < sum(_cache(first)), second
    assert _cache(third)[0] == entry, third


@needs_key
def test_a_long_previous_wake_is_found_through_the_lookback_chain(caplog):
    """A wake 44 positions past the last entry: only the chained breakpoints can reach it (#726).

    The first request ends on its anchor, position 10, so its one breakpoint writes its whole prompt
    there (no block after the anchor can carry a tail, #729). The second anchors position 54, whose
    own window (54..35) cannot see 10; its chain adds 34 and 14, and 14's window (14..0) holds the
    entry. So a read equal to the first request's whole prompt is the chain, and nothing else,
    finding it: the second request's tail (position 56) and anchor windows see no entry.
    """
    nonce = uuid.uuid4().hex
    head = Message.user(f"[{nonce}]\n{CHARTER}\nQuestion 0.")
    turns = [head] + [
        Message.assistant(f"Answer {i}.") if i % 2 else Message.user(f"Question {i}.")
        for i in range(1, 55)
    ]
    adapter = _provider()
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        first = turns[:11]  # ends on a user turn, the anchor
        adapter.chat(anchor_cacheable_prefix(first, stable=len(first), mode=EXPLICIT))
        frozen = turns[:55]
        convo = [*frozen, Message.system("brief"), Message.user("Reply with the word: ok")]
        adapter.chat(anchor_cacheable_prefix(convo, stable=len(frozen), mode=EXPLICIT))
    first, second = _lines(caplog, "llm provider=")
    read, written = _cache(first)
    assert written > 0, first
    assert _cache(second)[0] == read + written, second


@needs_key
def test_the_next_step_of_a_wake_reads_the_brief_through_the_tail(caplog):
    """Within one wake (issue #729): step 2 re-sends step 1's whole request and reads all of it.

    A first wake, so nothing is anchored and the one tool is under the cacheable minimum: before the
    tail breakpoint, step 2 read nothing at all and re-sent the brief in full. Step 1's tail sits on
    the new user turn; step 2's sits on the tool result two positions later, inside its window. So
    step 2 reads exactly what step 1 wrote, and that write is step 1's whole prompt — tools, brief
    and turn — but for the few tokens that close a request, which Anthropic counts as input and no
    breakpoint can cover (4 on ``claude-haiku-5-5``, measured 2026-10-31).
    """
    brief = Message.system(f"[{uuid.uuid4().hex}]\n{CHARTER}")
    convo = [brief, Message.user("Get the secret number for 'blue', then tell me what it is.")]
    adapter = _provider()
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        step = adapter.chat(anchor_cacheable_prefix(convo, stable=0, mode=EXPLICIT), [NUMBER_TOOL])
        assert step.tool_calls, "the model did not call the tool"
        convo += [step, *(Message.tool(call.id, "42") for call in step.tool_calls)]
        adapter.chat(anchor_cacheable_prefix(convo, stable=0, mode=EXPLICIT), [NUMBER_TOOL])
    first, second = _lines(caplog, "llm provider=")
    read, written = _cache(first)
    assert written > 0, first
    assert _cache(second)[0] == read + written, second
    assert int(_field(first, "tokens_in")) - (read + written) <= CLOSING_TOKENS, first


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


POST_TOOL = ToolSpec(
    name="post_message",
    description="Post a short message to the user's timeline.",
    parameters={
        "type": "object",
        "properties": {"body": {"type": "string"}},
        "required": ["body"],
    },
)


@needs_key
def test_a_mixed_server_and_client_group_is_continued_with_the_step_note_aboard():
    """Issue #735: a server tool and a client tool in one group, then a step with a trailing note.

    The engine always follows a step's tool results with its step-counter note, a trailing text
    block in the same user message, which Anthropic answers with a 400 naming the server tool's
    missing result. The adapter folds it into the last result. A model that does not split the
    group this time cannot test it, so the attempt is retried a few times and skipped, not passed,
    if no mixed reply is ever drawn.
    """
    ask = Message.user(
        "In ONE response, call BOTH tools in parallel right away: web_fetch https://example.com "
        "AND post_message with body 'Fetching now.'. Do not wait for one before the other."
    )
    for _ in range(4):
        adapter = _provider(builtin_tools=["web_fetch"])
        step = adapter.chat([ask], tools=[POST_TOOL])
        kept = adapter._replies[id(step)][1]
        uses = {b.get("id") for b in kept if b.get("type") == "server_tool_use"}
        answered = {b.get("tool_use_id") for b in kept if b.get("type", "").endswith("_result")}
        if step.tool_calls and uses - answered:
            break
    else:
        pytest.skip("the model never drew a mixed server+client group; nothing was tested")
    convo = [ask, step]
    convo += [Message.tool(call.id, "posted") for call in step.tool_calls]
    convo.append(Message.system("Current Time: 2026-10-31 12:00:00 UTC\nStep 2 of 24."))
    answer = adapter.chat(convo, tools=[POST_TOOL])  # a 400 here is the defect
    assert "example domain" in (answer.content or "").lower() or answer.tool_calls
