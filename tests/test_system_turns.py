"""A model that refuses a system turn no user turn follows gets one it accepts (issue #945).

refuseless serves ``kimi-k3.1-pro-abliterated`` so that a request whose system-role message is not
followed by a user turn is a 400, and every wake ends its input with one (the per-step note). The
operator declares it with ``AI_SYSTEM_TURNS=before_user``; the adapter then sends each such system
turn as a user turn marked ``[System]``, and keeps the system role wherever a user turn follows.

What is pinned here: the rule over the orders the capital measured, the request bytes on both of
the openai SDK's surfaces through a real engine tool loop (respx at the transport, so the real SDK
serializes them), the wire left exactly as it was when nothing is declared, and the setting's
validation in every place a wake or ``--resolved-config`` reads it.
"""

from __future__ import annotations

import json
import random

import httpx
import pytest
import respx

from basecradle_harness import Engine, Message, Tool, ToolRegistry
from basecradle_harness import _basecradle as basecradle
from basecradle_harness._basecradle import (
    _config_from_env,
    _provider_from_config,
    system_turns_from_env,
)
from basecradle_harness._openai import OpenAIProvider
from basecradle_harness._openai_wire import chat_message_to_wire, message_to_input
from basecradle_harness._system_turns import (
    ANYWHERE,
    BEFORE_USER,
    SYSTEM_LABEL,
    before_user,
)

from .conftest import completion, out_function_call, out_message, responses_body

VENDOR = "refuseless"
HOST = "https://api.refuseless.com/v1"
MODEL = "kimi-k3.1-pro-abliterated"
FAKE_KEY = "abl_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL"
ROLE = {"system": "S", "user": "U", "assistant": "A", "tool": "T"}


def _shape(messages: list[Message]) -> str:
    """A conversation as one letter per turn, the way the capital's table reads."""
    return "".join(ROLE[m.role] for m in messages)


def _turns(shape: str) -> list[Message]:
    """A conversation from that shorthand: ``S`` system, ``U`` user, ``A`` assistant, ``T`` tool."""
    built = {
        "S": lambda n: Message.system(f"note {n}"),
        "U": lambda n: Message.user(f"user {n}"),
        "A": lambda n: Message.assistant(f"assistant {n}"),
        "T": lambda n: Message.tool(f"call_{n:04d}", f"result {n}"),
    }
    return [built[letter](n) for n, letter in enumerate(shape)]


# --- the rule ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sent", "placed"),
    [
        # The orders the vendor answered 200: nothing to change, and nothing is changed.
        ("SU", "SU"),
        ("SSU", "SSU"),
        ("USU", "USU"),
        # The orders it answered 400: every system turn no user turn follows travels as a user turn.
        ("US", "UU"),
        ("SUS", "SUU"),
        ("USUS", "USUU"),  # the real order of a wake's first step: user, brief, user, step note
        ("SUAS", "SUAU"),
        # A wake's later step: the earlier step note now sits before the assistant turn it preceded.
        ("USUSATS", "USUUATU"),
    ],
)
def test_the_measured_orders(sent, placed):
    assert _shape(before_user(_turns(sent))) == placed


def test_a_relabelled_turn_says_it_is_the_harnesss_and_keeps_its_text():
    note = Message.system("Current Time: 2026-11-20 01:30:00 UTC\n\nStep 1 of 24.")
    (placed,) = before_user([note])
    assert placed.role == "user"
    assert placed.content == SYSTEM_LABEL + note.content


def test_a_kept_system_turn_and_every_other_turn_are_the_same_objects():
    turns = _turns("USUSAT")
    placed = before_user(turns)
    for index in (0, 1, 2, 4, 5):
        assert placed[index] is turns[index]
    assert placed[3] is not turns[3]


def test_the_conversation_handed_in_is_never_changed():
    """A copy at the wire: the transcript, compaction and every recovery reader see system turns."""
    turns = _turns("USUS")
    before = [(m.role, m.content) for m in turns]
    before_user(turns)
    assert [(m.role, m.content) for m in turns] == before


def test_every_other_field_of_a_relabelled_turn_is_kept():
    note = Message.system("note")
    note.cache_anchor = True
    note.items = ["019ea47c-957b-70ef-9852-76211bd24c0a"]
    (placed,) = before_user([note])
    assert placed.cache_anchor is True
    assert placed.items == note.items


def test_over_any_conversation_no_system_turn_is_left_without_a_user_turn_after_it():
    """The one property the vendor needs, checked over orders nobody measured too."""
    rng = random.Random(945)
    for _ in range(2000):
        shape = "".join(rng.choice("SUAT") for _ in range(rng.randint(0, 12)))
        placed = _shape(before_user(_turns(shape)))
        for index, role in enumerate(placed):
            if role == "S":
                assert placed[index:].lstrip("S")[:1] == "U", (shape, placed)
        # Only system turns change, and only into user turns.
        for sent, kept in zip(shape, placed, strict=True):
            assert sent == kept or (sent, kept) == ("S", "U"), (shape, placed)
        assert before_user(before_user(_turns(shape))) == before_user(_turns(shape))


def test_a_step_note_renders_the_same_on_every_later_step_of_the_loop():
    """Once a call has followed it, a relabelled note never changes, so a cached prefix holds."""
    first = before_user(_turns("USUS"))
    later = before_user(_turns("USUSATSATS"))
    assert later[: len(first)] == first


# --- the wire ---------------------------------------------------------------------------------


class EchoTool(Tool):
    name = "echo"
    description = "Echo."

    def run(self, **kwargs):
        return "echoed"


def _models() -> httpx.Response:
    row = {"id": MODEL, "object": "model", "created": 0, "owned_by": VENDOR}
    return httpx.Response(200, json={"object": "list", "data": [row]})


def _conversation() -> list[Message]:
    """What a wake hands the engine: an old exchange, the brief, then the newest message."""
    return [
        Message.user("Hello."),
        Message.system("You are Nova. (The per-wake brief.)"),
        Message.user("Please call echo, then tell me what it said."),
    ]


def _loop(surface: str, system_turns: str) -> list[dict]:
    """Run one real two-step tool loop on a refuseless adapter and return each request's body."""
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{HOST}/models").mock(return_value=_models())
        if surface == "responses":
            call = out_function_call(call_id="call_0001", name="echo", arguments={})
            replies = [responses_body(call), responses_body(out_message("It said echoed."))]
            route = router.post(f"{HOST}/responses")
        else:
            tool_call = {
                "id": "call_0001",
                "type": "function",
                "function": {"name": "echo", "arguments": "{}"},
            }
            replies = [
                completion(tool_calls=[tool_call], finish_reason="tool_calls"),
                completion(content="It said echoed."),
            ]
            route = router.post(f"{HOST}/chat/completions")
        route.mock(side_effect=[httpx.Response(200, json=body) for body in replies])
        adapter = OpenAIProvider(
            MODEL,
            api_key=FAKE_KEY,
            base_url=HOST,
            provider=VENDOR,
            surface=surface,
            system_turns=system_turns,
        )
        registry = ToolRegistry()
        registry.register(EchoTool())
        try:
            reply = Engine(adapter, registry).run(_conversation())
        finally:
            adapter.close()
        assert reply.content == "It said echoed."
        return [json.loads(request.content) for request, _ in route.calls]


def _roles(surface: str, body: dict) -> str:
    """A request's turns in the shorthand, read off the bytes that were sent."""
    if surface == "responses":
        kinds = {"function_call": "A", "function_call_output": "T"}
        letters = {"developer": "S", "user": "U", "assistant": "A"}
        return "".join(
            kinds.get(i.get("type"), letters.get(i.get("role"), "?")) for i in body["input"]
        )
    return "".join(ROLE[m["role"]] for m in body["messages"])


def _last_text(surface: str, body: dict) -> str:
    return (
        body["input"][-1]["content"] if surface == "responses" else body["messages"][-1]["content"]
    )


@pytest.mark.parametrize("surface", ["responses", "chat"])
def test_before_user_sends_the_step_notes_as_marked_user_turns(surface):
    first, second = _loop(surface, BEFORE_USER)
    assert _roles(surface, first) == "USUU"
    assert _roles(surface, second) == "USUUATU"
    for body in (first, second):
        assert _last_text(surface, body).startswith(SYSTEM_LABEL + "Current Time: ")
    # The brief is followed by a user turn, so it keeps the system role the model reads as authority.
    brief = second["input"][1] if surface == "responses" else second["messages"][1]
    assert brief["role"] == ("developer" if surface == "responses" else "system")
    assert brief["content"] == "You are Nova. (The per-wake brief.)"


@pytest.mark.parametrize("surface", ["responses", "chat"])
def test_nothing_declared_leaves_the_wire_exactly_as_it_was(surface):
    """Every request byte for byte what the adapter's own renderer makes of the transcript."""
    first, second = _loop(surface, ANYWHERE)
    assert _roles(surface, first) == "USUS"
    assert _roles(surface, second) == "USUSATS"
    transcript = _conversation()
    render = (
        (lambda ms: [i for m in ms for i in message_to_input(m)])
        if surface == "responses"
        else (lambda ms: [chat_message_to_wire(m) for m in ms])
    )
    key = "input" if surface == "responses" else "messages"
    sent = first[key]
    expected = render([*transcript, Message.system(_last_text(surface, first))])
    assert sent == expected


def test_the_adapter_refuses_a_placement_it_does_not_know():
    with pytest.raises(ValueError, match="system_turns"):
        OpenAIProvider(MODEL, api_key=FAKE_KEY, base_url=HOST, system_turns="trailing")


# --- the setting ------------------------------------------------------------------------------


@pytest.fixture
def kimi_env(monkeypatch, tmp_path):
    home = tmp_path / "config"
    home.mkdir()
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(home))
    monkeypatch.setenv("AI_PROVIDER", VENDOR)
    monkeypatch.setenv("AI_MODEL", MODEL)
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    for var in ("AI_SDK", "AI_SDK_SURFACE", "AI_BASE_URL", "AI_SYSTEM_TURNS"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_or_blank_is_anywhere(kimi_env, raw):
    if raw is not None:
        kimi_env.setenv("AI_SYSTEM_TURNS", raw)
    assert system_turns_from_env("openai") == ANYWHERE


def test_before_user_is_read_trimmed_and_in_any_case(kimi_env):
    kimi_env.setenv("AI_SYSTEM_TURNS", "  Before_User ")
    assert system_turns_from_env("openai") == BEFORE_USER


def test_an_unknown_value_fails_the_wake_at_startup(kimi_env):
    kimi_env.setenv("AI_SYSTEM_TURNS", "trailing_user")
    with pytest.raises(ValueError, match="AI_SYSTEM_TURNS must be one of anywhere, before_user"):
        _config_from_env()


@pytest.mark.parametrize(
    ("provider", "sdk", "reason"),
    [
        ("openrouter", "openrouter", "applied by the openai SDK's adapter only"),
        ("xai", "xai-sdk", "applied by the openai SDK's adapter only"),
        ("anthropic", "anthropic", "has no effect"),
        ("google", "google-genai", "has no effect"),
    ],
)
def test_before_user_on_an_sdk_that_does_not_apply_it_is_refused(kimi_env, provider, sdk, reason):
    """Never read as taken when the adapter would ignore it."""
    kimi_env.setenv("AI_PROVIDER", provider)
    kimi_env.setenv("AI_SDK", sdk)
    kimi_env.setenv("AI_SYSTEM_TURNS", BEFORE_USER)
    with pytest.raises(ValueError, match=reason):
        _config_from_env()
    with pytest.raises(ValueError, match=reason):
        _provider_from_config(provider, sdk, "native", system_turns=BEFORE_USER)


def test_the_brain_build_reads_the_setting(kimi_env):
    kimi_env.setenv("AI_SYSTEM_TURNS", BEFORE_USER)
    provider, *_rest = basecradle._resolve_tools_and_provider()
    try:
        assert provider.system_turns == BEFORE_USER
    finally:
        provider.close()


@pytest.mark.parametrize("surface", ["chat", "responses"])
def test_every_openai_sdk_cell_honours_it(kimi_env, surface):
    for provider in ("openai", "xai", VENDOR, "example-gateway"):
        kimi_env.setenv("AI_BASE_URL", "https://llm.example.com/v1")
        built = _provider_from_config(provider, "openai", surface, system_turns=BEFORE_USER)
        try:
            assert built.system_turns == BEFORE_USER, provider
        finally:
            built.close()
    built = _provider_from_config("openrouter", "openai", "chat", system_turns=BEFORE_USER)
    try:
        assert built.system_turns == BEFORE_USER
    finally:
        built.close()


def test_a_second_model_on_the_brains_stack_never_inherits_it(kimi_env):
    """The describer builds through the same seam with no `system_turns`: a different model."""
    kimi_env.setenv("AI_SYSTEM_TURNS", BEFORE_USER)
    built = _provider_from_config(VENDOR, "openai", "chat", model="another-model")
    try:
        assert built.system_turns == ANYWHERE
    finally:
        built.close()


def test_resolved_config_reports_it(kimi_env):
    from basecradle_harness._wake import resolved_config

    assert resolved_config()["ai_system_turns"] == ANYWHERE
    kimi_env.setenv("AI_SYSTEM_TURNS", BEFORE_USER)
    assert resolved_config()["ai_system_turns"] == BEFORE_USER
    kimi_env.setenv("AI_SYSTEM_TURNS", "sometimes")
    with pytest.raises(ValueError, match="AI_SYSTEM_TURNS"):
        resolved_config()
