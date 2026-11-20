"""Live gate for the refuseless vendor — a real priced call on each surface (issues #844, #847).

The one check the offline suite **structurally cannot** make. `test_refuseless.py` drives the real
``openai`` SDK against bodies *we wrote* in the shape the capital read off the vendor on 2026-11-09;
it cannot prove that the vendor still serves that shape, that the model a response names is a row
of the list, or that the dollars the line states are the dollars the vendor's own price implies.
This is where the vendor answers for itself.

It is an explicitly-marked **live** job (`@pytest.mark.live`), deselected from the default run and
skipped without a credential. From a fresh clone of ``main``::

    REFUSELESS_LIVE_API_KEY=… uv run --extra openai pytest -m live tests/test_refuseless_live.py -s -v

``REFUSELESS_LIVE_MODEL`` defaults to ``glm-5.3-flash-abliterated``. ``-s`` prints the ``llm`` line
and the arithmetic, for a hand check. A SKIP here is **not** a pass: pytest exits 0 when every test
skips, so read the report, not the exit code. What it pins is a value, never mere presence: the
``cost=`` equals ``prompt_tokens × input_per_m + completion_tokens × output_per_m`` over a million,
read from the vendor's list by a client of the test's own, not by the harness. The Responses arm
(issue #847) prices its stated cached split too: ``(input − cached) × input_per_m + cached ×
cached_input_per_m + output × output_per_m``. It sends one long prefix twice so a cache hit is
possible; a hit is not required, because the vendor's cache is not the harness's to promise, and the
arithmetic is checked against whatever count the vendor stated. A third test replays what a real wake
sends on that surface and the first two do not: a charter (the ``developer`` role), a function tool,
and a ``function_call`` / ``function_call_output`` pair, scripted rather than asked for so the model's
choice cannot skip it.
"""

from __future__ import annotations

import logging
import os

import pytest

from basecradle_harness import Engine, Message, Tool, ToolCall, ToolRegistry, ToolSpec
from basecradle_harness._basecradle import (
    _config_from_env,
    _provider_from_config,
    system_turns_from_env,
)
from basecradle_harness._exceptions import ProviderAPIError
from basecradle_harness._observability import _money
from tests._line_expectations import assert_direct_call

pytestmark = pytest.mark.live

KEY = (os.environ.get("REFUSELESS_LIVE_API_KEY") or "").strip() or None
MODEL = (os.environ.get("REFUSELESS_LIVE_MODEL") or "glm-5.3-flash-abliterated").strip()
HOST = "https://api.refuseless.com/v1"

needs_key = pytest.mark.skipif(
    not KEY, reason="set REFUSELESS_LIVE_API_KEY to run the live refuseless gate"
)


@needs_key
def test_one_real_call_is_priced_from_the_vendors_own_list(monkeypatch, caplog):
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("AI_PROVIDER", "refuseless")
    monkeypatch.setenv("AI_MODEL", MODEL)
    monkeypatch.setenv("AI_API_KEY", KEY)
    for var in ("AI_SDK", "AI_SDK_SURFACE", "AI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    caplog.set_level(logging.INFO, logger="basecradle_harness")

    provider = _provider_from_config(*_config_from_env(), max_output_tokens=256)
    try:
        limit = provider.context_limit()
        provider.chat([Message.user("Reply with the single word: ready")])
    finally:
        provider.close()

    lines = [record.getMessage() for record in caplog.records]
    (llm,) = [line for line in lines if line.startswith("llm ")]
    print(f"\n{llm}")
    assert not [line for line in lines if line.startswith("cost unpriced")], lines
    # The contract's shape of a priced direct call (`tests/_line_expectations.py`, issue #904).
    fields = assert_direct_call(llm, provider="refuseless", model=MODEL, cost="computed")

    # The vendor's list, read by a client of the test's own: the harness's parser is not the judge.
    with openai.OpenAI(api_key=KEY, base_url=HOST) as client:
        rows = {item.id: item.model_dump() for item in client.models.list()}
    row = rows[MODEL]
    pricing = row["pricing"]
    assert pricing["currency"] == "USD"
    tokens_in, tokens_out = int(fields["tokens_in"]), int(fields["tokens_out"])
    expected = (tokens_in * pricing["input_per_m"] + tokens_out * pricing["output_per_m"]) / 1e6
    print(
        f"{tokens_in} × ${pricing['input_per_m']}/M + {tokens_out} × ${pricing['output_per_m']}/M"
        f" = ${expected!r} (line: cost={fields['cost']})"
    )
    assert fields["cost"] == _money(expected)
    assert limit == row["context_length"]


@needs_key
def test_a_real_responses_call_is_priced_with_its_cached_split(monkeypatch, caplog):
    """Issue #847: ``AI_SDK_SURFACE=responses`` reaches the vendor and its usage block is priced."""
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("AI_PROVIDER", "refuseless")
    monkeypatch.setenv("AI_MODEL", MODEL)
    monkeypatch.setenv("AI_API_KEY", KEY)
    monkeypatch.setenv("AI_SDK_SURFACE", "responses")
    for var in ("AI_SDK", "AI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    caplog.set_level(logging.INFO, logger="basecradle_harness")

    # A prefix long enough for a prefix cache to hold, sent twice; the second call may read it.
    prefix = " ".join(f"Line {n}: the quick brown fox jumps over the lazy dog." for n in range(200))
    provider = _provider_from_config(*_config_from_env(), max_output_tokens=256)
    try:
        assert provider.surface == "responses"
        for _ in range(2):
            reply = provider.chat([Message.user(f"{prefix}\n\nReply with the single word: ready")])
            assert reply.content
    finally:
        provider.close()

    lines = [record.getMessage() for record in caplog.records]
    calls = [line for line in lines if line.startswith("llm ")]
    assert len(calls) == 2
    assert not [line for line in lines if line.startswith("cost unpriced")], lines

    with openai.OpenAI(api_key=KEY, base_url=HOST) as client:
        rows = {item.id: item.model_dump() for item in client.models.list()}
    pricing = rows[MODEL]["pricing"]
    assert pricing["currency"] == "USD"
    for llm in calls:
        print(f"\n{llm}")
        fields = assert_direct_call(llm, provider="refuseless", model=MODEL, cost="computed")
        generation = fields.get("generation_id", "")
        assert generation.startswith("resp"), llm  # the Responses shape, not a chat completion
        tokens_in, tokens_out = int(fields["tokens_in"]), int(fields["tokens_out"])
        cached = int(fields.get("cached_tokens", 0))
        expected = (
            (tokens_in - cached) * pricing["input_per_m"]
            + cached * pricing["cached_input_per_m"]
            + tokens_out * pricing["output_per_m"]
        ) / 1e6
        print(
            f"({tokens_in} − {cached}) × ${pricing['input_per_m']}/M + {cached} × "
            f"${pricing['cached_input_per_m']}/M + {tokens_out} × ${pricing['output_per_m']}/M"
            f" = ${expected!r} (line: cost={fields['cost']})"
        )
        assert fields["cost"] == _money(expected)


@needs_key
def test_a_real_responses_call_takes_a_charter_a_tool_and_a_tool_round_trip(monkeypatch, caplog):
    """Issue #847: the Responses wire a wake sends — developer role, function tool, call items."""
    pytest.importorskip("openai")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("AI_PROVIDER", "refuseless")
    monkeypatch.setenv("AI_MODEL", MODEL)
    monkeypatch.setenv("AI_API_KEY", KEY)
    monkeypatch.setenv("AI_SDK_SURFACE", "responses")
    for var in ("AI_SDK", "AI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    caplog.set_level(logging.INFO, logger="basecradle_harness")

    weather = ToolSpec(
        name="get_weather",
        description="The current weather in a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    history = [
        Message.system("You are a terse assistant. Answer in one short sentence."),
        Message.user("What is the weather in Dallas?"),
        Message.assistant(
            tool_calls=[ToolCall(id="call_0001", name="get_weather", arguments={"city": "Dallas"})]
        ),
        Message.tool("call_0001", '{"city": "Dallas", "sky": "clear", "temperature_f": 71}'),
    ]
    provider = _provider_from_config(*_config_from_env(), max_output_tokens=256)
    try:
        reply = provider.chat(history, tools=[weather])
    finally:
        provider.close()

    print(f"\nreply: {reply.content!r} tool_calls={reply.tool_calls!r}")
    # The vendor read the tool result: the answer names it, or the model called the tool again.
    assert reply.content or reply.tool_calls
    if reply.content:
        assert "71" in reply.content or "clear" in reply.content.lower(), reply.content
    lines = [record.getMessage() for record in caplog.records]
    (llm,) = [line for line in lines if line.startswith("llm ")]
    print(llm)
    fields = assert_direct_call(llm, provider="refuseless", model=MODEL, cost="computed")
    assert fields.get("generation_id", "").startswith("resp"), llm
    assert not [line for line in lines if line.startswith("cost unpriced")], lines


#: The model whose vendor refuses a system turn no user turn follows (issue #945), and the setting
#: that declares it. A separate variable from ``REFUSELESS_LIVE_MODEL``, because the arms above run
#: on any refuseless model and these are about one.
ORDER_MODEL = (os.environ.get("REFUSELESS_LIVE_ORDER_MODEL") or "kimi-k3.1-pro-abliterated").strip()


class _Echo(Tool):
    """The tool in play: the model is asked to call it, and its result must reach the answer."""

    name = "get_weather"
    description = "The current weather in a city."
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    def run(self, **kwargs):
        return '{"city": "Dallas", "sky": "clear", "temperature_f": 71}'


def _order_env(monkeypatch, surface: str, system_turns: str | None) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("AI_PROVIDER", "refuseless")
    monkeypatch.setenv("AI_MODEL", ORDER_MODEL)
    monkeypatch.setenv("AI_API_KEY", KEY)
    monkeypatch.setenv("AI_SDK_SURFACE", surface)
    if system_turns is None:
        monkeypatch.delenv("AI_SYSTEM_TURNS", raising=False)
    else:
        monkeypatch.setenv("AI_SYSTEM_TURNS", system_turns)
    for var in ("AI_SDK", "AI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def _wake_order() -> list[Message]:
    """The order a wake sends: an earlier wake's tool turn (its step notes in place), the brief, the
    newest message. The engine appends the step note last on every step."""
    return [
        Message.user("What is the weather in Austin?"),
        Message.system("Current Time: 2026-11-19 18:00:00 UTC\n\nStep 1 of 24."),
        Message.assistant(
            tool_calls=[ToolCall(id="call_0001", name="get_weather", arguments={"city": "Austin"})]
        ),
        Message.tool("call_0001", '{"city": "Austin", "sky": "rain", "temperature_f": 58}'),
        Message.system("Current Time: 2026-11-19 18:00:05 UTC\n\nStep 2 of 24."),
        Message.assistant("Austin: rain, 58°F."),
        Message.system("You are a terse assistant. Use your tools; answer in one short sentence."),
        Message.user("Now call get_weather for Dallas and tell me what it says."),
    ]


@needs_key
@pytest.mark.parametrize("surface", ["responses", "chat"])
def test_the_wake_order_is_refused_without_the_declaration(monkeypatch, surface):
    """The control: the vendor still refuses the shape, so the declaration is what the fix is.

    If this goes red because the vendor now answers, the vendor changed, not the harness: the
    declaration can then be retired from the agents that carry it.
    """
    pytest.importorskip("openai")
    _order_env(monkeypatch, surface, None)
    provider, sdk, surface = _config_from_env()
    provider = _provider_from_config(
        provider, sdk, surface, max_output_tokens=512, system_turns=system_turns_from_env(sdk)
    )
    try:
        assert provider.system_turns == "anywhere"
        registry = ToolRegistry()
        registry.register(_Echo())
        with pytest.raises(ProviderAPIError) as refused:
            provider.chat(
                [*_wake_order(), Message.system("Current Time: now\n\nStep 1 of 24.")],
                tools=registry.specs(),
            )
    finally:
        provider.close()
    print(f"\n{refused.value}")
    assert refused.value.status_code == 400


@needs_key
@pytest.mark.parametrize("surface", ["responses", "chat"])
def test_a_wake_with_a_tool_in_play_completes_when_declared(monkeypatch, caplog, surface):
    """Issue #945's definition of done: the real order, through the engine, answered 200."""
    pytest.importorskip("openai")
    _order_env(monkeypatch, surface, "before_user")
    caplog.set_level(logging.INFO, logger="basecradle_harness")
    provider, sdk, surface = _config_from_env()
    # Built as the wake builds its brain (`_resolve_tools_and_provider`): the setting is the brain's.
    provider = _provider_from_config(
        provider,
        sdk,
        surface,
        max_output_tokens=4096,
        system_turns=system_turns_from_env(sdk),
    )
    registry = ToolRegistry()
    registry.register(_Echo())
    try:
        assert provider.system_turns == "before_user"
        reply = Engine(provider, registry, max_steps=4).run(_wake_order())
    finally:
        provider.close()

    print(f"\nreply: {reply.content!r}")
    lines = [record.getMessage() for record in caplog.records]
    calls = [line for line in lines if line.startswith("llm ")]
    for line in calls:
        print(line)
    tools = [line for line in lines if line.startswith("tool ")]
    print("\n".join(tools))
    # Every attempt answered, and the tool ran before the final reply: at least two calls.
    assert len(calls) >= 2, calls
    for line in calls:
        assert_direct_call(line, provider="refuseless", model=ORDER_MODEL, cost="computed")
    assert any("get_weather" in line for line in tools), tools
    assert reply.content and ("71" in reply.content or "clear" in reply.content.lower())
