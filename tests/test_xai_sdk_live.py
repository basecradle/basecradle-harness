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

**Something runs this on a schedule now, and a SKIP is RED** (issue #450). Nothing did before: ``-m
'not live'`` hides this file from every default run and from CI, and it skips itself green with no
key — three states, *passed* / *skipped* / *never invoked*, and from outside the box the last two
look exactly like the first. That is this repo's own named failure shape, Green-While-Absent,
pointed at the one suite that can say whether the real gRPC endpoint still accepts the wiring this
adapter sends. The trigger is the NOC prober (``basecradle-noc#563``, grown from one pinned path to
a **registry** of five arms in ``basecradle-noc#575``): this file is the **``xai``** arm, and the
prober clones the tip of ``main`` and runs this file's own invocation with a **dedicated**
``XAI_API_KEY`` that lives only on the NOC box — never a copy of a running agent's runtime key, per
@origin's per-consumer ruling on #441. Weekly on green, daily on red, per arm. The verdict is read
off a JUnit report rather than ``returncode``, because pytest exits **0** when every collected test
skips, so an absent key is byte-identical to a pass at the process boundary; a run reporting
``skipped > 0`` is a failure there.

Two consequences for anyone editing this file. **The path and the marker are a cross-repo
contract**: the arm selects ``-m live tests/test_xai_sdk_live.py``, so renaming either makes it
collect zero tests, which the prober reads as *never invoked* and calls red — correct and loud, but
tell the NOC rather than leaving it to fire. And **the assertions stay ours** — the prober runs
this suite instead of re-implementing it, so what this file proves is what gets proven on a
cadence, and adding a case here needs no coordination at all.
"""

from __future__ import annotations

import logging
import os
import re
import uuid

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


@pytest.mark.skipif(not KEY, reason="set XAI_API_KEY to run the live xAI cache-affinity probe")
def test_a_bound_conversation_earns_the_per_server_cache_hit():
    """The #431/#433 fix against the real fleet: does ``x-grok-conv-id`` actually buy the discount?

    The last of three checks, each answering what the one below it cannot. The mocked-client tests
    prove the adapter *decided* on a key; `test_xai_sdk_wire.py` proves the key **left the process**
    (a real gRPC server reads it back off the connection) — the step whose absence let 0.110.0 ship
    a key the SDK swallowed into a telemetry attribute; and only the live endpoint can say whether
    xAI **routes** on it, the same gap the deprecated ``SearchParameters`` path fell through.

    The shape is deliberate. **One arm, not an A/B**: the unbound control is inherently lucky —
    a scattered call can land on a warm server anyway (that is exactly what @briggs's 0.2–18% was),
    so asserting a control *miss* would be a flaky test asserting the absence of luck. What is not
    luck is the bound arm: with the routing key held constant, call 2 must find the prefix call 1
    left. So this asserts a **value** — most of the prompt came back cached — rather than the mere
    presence of a counter, and it uses the one authority the design recognizes for the question
    (`_caching`): the ``cached_tokens=`` the endpoint itself reported, off the per-call log line.

    The prefix is arm-private (a fresh uuid in the body) so no earlier run of this test can warm it,
    and it is well past the ~512-token granularity xAI appears to cache at.
    """
    tag = uuid.uuid4().hex
    body = "\n".join(
        f"{tag} note {i}: the quick brown fox jumps over the lazy dog." for i in range(1200)
    )
    messages = [
        Message.system("You are a terse assistant."),
        Message.user(f"{body}\n\nReply with the single word: ok"),
    ]
    provider = XaiSdkProvider(model="grok-4.3", api_key=KEY)
    provider.bind_conversation(f"timeline:{uuid.uuid4()}")

    logger = logging.getLogger("basecradle_harness")
    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        provider.chat(messages)  # cold: writes the prefix onto whichever server serves it
        provider.chat(messages)  # warm: the same key must route back to that server
    finally:
        logger.setLevel(previous)
        logger.removeHandler(handler)
        provider.close()

    llm = [line for line in records if line.startswith("llm ")]
    assert len(llm) == 2, llm
    cached = int(re.search(r"cached_tokens=(\d+)", llm[1]).group(1))
    tokens_in = int(re.search(r"tokens_in=(\d+)", llm[1]).group(1))
    # Not "> 0" — a scattered call gets that by luck. Affinity means most of the prefix comes back.
    assert cached > tokens_in * 0.5, f"cached={cached} of tokens_in={tokens_in}: {llm[1]}"
