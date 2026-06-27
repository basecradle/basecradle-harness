"""Live smoke for the native xAI Live Search (Agent Tools API) — issue #171.

This is the one check the mocked-client suite **structurally cannot** make. The native adapter
tests inject a fake ``xai_sdk.Client``, so a request the *real* gRPC endpoint rejects still passes
them — exactly how the deprecated ``SearchParameters`` path (now ``UNIMPLEMENTED: Live search is
deprecated``) slipped through. This test builds a **real** client and hits ``api.x.ai`` for real,
so a regression to a server-rejected wiring fails loudly here.

It is an explicitly-marked **live** job (`@pytest.mark.live`), deselected from the default offline
run by ``addopts = -m 'not live'`` and skipped when no key is present. Run it deliberately::

    XAI_API_KEY=xai-... uv run pytest -m live tests/test_xai_sdk_live.py -v

The capital re-runs it (with a valid grok key) at the v0.37.0 release gate; this file makes that a
repeatable command rather than a one-off manual probe.
"""

from __future__ import annotations

import os

import pytest

from basecradle_harness import Message, ToolSpec, XaiSdkProvider

pytestmark = pytest.mark.live

KEY = os.environ.get("XAI_API_KEY")


@pytest.mark.skipif(not KEY, reason="set XAI_API_KEY to run the live xAI Agent Tools probe")
def test_native_live_search_returns_a_grounded_answer_with_citations():
    """The capital's reproduction (#171), as a runnable test against the real grok endpoint.

    With ``web_search`` / ``x_search`` opted in, grok runs the search server-side via the Agent
    Tools API and returns a sourced answer — no ``UNIMPLEMENTED: Live search is deprecated``, and a
    ``Sources:`` citation footer the adapter appends from ``Response.citations``.
    """
    provider = XaiSdkProvider(
        model="grok-4.3",
        api_key=KEY,
        builtin_tools=["web_search", "x_search"],
    )
    try:
        reply = provider.chat([Message.user("Name a recent AI headline with a source URL.")])
    finally:
        provider.close()

    assert reply.role == "assistant"
    assert reply.content  # a real, non-empty grounded answer
    assert "Sources:" in reply.content  # Live Search returned citations


@pytest.mark.skipif(not KEY, reason="set XAI_API_KEY to run the live xAI Agent Tools probe")
def test_live_search_works_alongside_function_tools_without_bouncing():
    """Orion's exact runtime condition (#183): search built-ins **and** function tools in one turn.

    The #171 live test above offers search built-ins *only*. Orion (xai-sdk/native, a full BaseCradle
    tool-set) hit the real bug only with **both** present: grok ran the search server-side, then
    surfaced its server-side ``web_search`` / ``x_search`` tool calls in ``Response.tool_calls``
    mixed with any function call — the adapter re-dispatched them, the harness bounced
    ``Error: no tool named 'web_search'``, and the model confabulated a result. With the
    `_is_client_side` filter the server-side calls are dropped, so a research turn returns the
    grounded answer with **no** spurious surfaced tool call. This is the check the mocked suite and
    the search-only live test both structurally miss.
    """
    a_function_tool = ToolSpec(
        name="post_message",
        description="Post a message to the timeline.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
    )
    provider = XaiSdkProvider(
        model="grok-4.3",
        api_key=KEY,
        builtin_tools=["web_search", "x_search"],
    )
    try:
        reply = provider.chat(
            [Message.user("Research one recent AI headline and give me the source URL.")],
            tools=[a_function_tool],
        )
    finally:
        provider.close()

    assert reply.role == "assistant"
    # No server-side search call leaks back as a function call to bounce. grok may legitimately ask
    # for the one *function* tool, but never a search built-in (web_search/x_search) or an X
    # sub-op (x_semantic_search/x_keyword_search) — those are the #183 bounce.
    bounced = {"web_search", "x_search", "x_semantic_search", "x_keyword_search"}
    assert not any(call.name in bounced for call in reply.tool_calls)
    assert reply.content  # a real grounded answer, not a confabulated empty turn
