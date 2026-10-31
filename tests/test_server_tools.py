"""The record of server-side tool activity (issue #727): kept, bounded, and shown on later calls.

The defect, in one wake pair: a wake ran Anthropic's server-side ``web_fetch`` and truthfully posted
the page's title; the next wake saw a post with no fetch before it and publicly retracted the true
post. A server tool's blocks are vendor state the transcript never keeps, so the harness had hidden
the evidence of the agent's own action from it. These tests pin the fix's four constraints — its own
field (never content, never mined), never a `ToolCall`, bounded and attributed, and rendered beside
the turn on every later call — plus the reproduction, offline, through the real Anthropic SDK.

The cast is the fixed fiction: Nova Digital (`nova`, AI) is the agent; John Doe (`john`).
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from basecradle_harness import AnthropicProvider, Engine, Harness, Message, ToolCall, ToolRegistry
from basecradle_harness._anthropic import DEFAULT_BASE_URL, SYSTEM_LABEL
from basecradle_harness._attribution import attribute
from basecradle_harness._context import _chars, _render, message_chars, request_chars
from basecradle_harness._idempotency import creates
from basecradle_harness._messages import ServerToolRecord, ServerToolUse
from basecradle_harness._server_tools import (
    HEADER,
    INPUT_CAP,
    MAX_LISTED,
    NAME_CAP,
    NOTE_CAP,
    UNLISTED_CAP,
    bound,
    chars,
    record,
    render,
    summarize,
    with_records,
)
from basecradle_harness._session import heal_interrupted_calls
from basecradle_harness._tools import Tool

FETCHED = "https://example.com"


def fetched(url: str = FETCHED, text: str = "Exact page title: Example Domain.") -> Message:
    """An assistant turn whose model call ran a server-side fetch, as an adapter reports it."""
    reply = Message.assistant(content=text)
    reply.server_tools = record([ServerToolUse("web_fetch", url)])
    return reply


# === the record: its own field, persisted ======================================================


def test_the_record_round_trips_through_the_transcript_format():
    turn = fetched()
    turn.server_tools = ServerToolRecord(uses=[ServerToolUse("web_fetch", FETCHED)], unlisted=3)
    data = turn.to_dict()
    assert data["server_tools"] == {
        "uses": [{"name": "web_fetch", "input": FETCHED}],
        "unlisted": 3,
    }
    assert Message.from_dict(json.loads(json.dumps(data))).server_tools == turn.server_tools


def test_a_turn_without_one_persists_exactly_as_before():
    assert "server_tools" not in Message.assistant(content="hi").to_dict()
    assert Message.from_dict({"role": "assistant", "content": "hi"}).server_tools is None


def test_a_malformed_persisted_record_reads_as_none_rather_than_raising():
    for junk in ("x", 7, {"uses": "x"}, {"uses": [], "unlisted": True}, {"uses": [3]}):
        data = {"role": "assistant", "content": "hi", "server_tools": junk}
        assert Message.from_dict(data).server_tools is None


def test_the_record_is_never_the_turns_content():
    """The Mining Boundary: a mining provider files `content` as the model's own words."""
    turn = fetched()
    assert FETCHED not in (turn.content or "")
    assert turn.tool_calls == []


# === the bound ================================================================================


def test_names_and_inputs_are_capped_and_flattened_onto_one_line():
    found = bound(
        ServerToolRecord(uses=[ServerToolUse("n" * 500, "print(1)\n\n   print(2)\n" + "x" * 900)])
    )
    use = found.uses[0]
    assert len(use.name) == NAME_CAP and use.name.endswith("…")
    assert len(use.input) == INPUT_CAP and use.input.endswith("…")
    assert use.input.startswith("print(1) print(2) x")


def test_uses_past_the_listed_cap_are_counted_not_dropped():
    found = record(ServerToolUse("web_search", f"q{n}") for n in range(MAX_LISTED + 5))
    assert len(found.uses) == MAX_LISTED and found.unlisted == 5
    assert "and 5 more server-side tool use(s)" in render(found)


def test_the_bound_is_a_fixed_point():
    """Applied at arrival and again to what is read back from disk, it never changes the record."""
    found = record(ServerToolUse("w" * 99, "i" * 999) for _ in range(MAX_LISTED * 2))
    assert bound(found) == found
    assert (
        bound(
            Message.from_dict(Message(role="assistant", server_tools=found).to_dict()).server_tools
        )
        == found
    )


def test_an_empty_record_is_no_record():
    assert record([]) is None
    assert bound(ServerToolRecord()) is None
    assert chars(Message(role="assistant", server_tools=ServerToolRecord())) == 0


def test_the_worst_case_note_is_exactly_the_stated_cap():
    """`NOTE_CAP` is an exact bound, derived from the functions that write the note."""
    worst = ServerToolRecord(
        uses=[ServerToolUse("n" * 900, "i" * 900) for _ in range(MAX_LISTED)],
        unlisted=10 * UNLISTED_CAP,
    )
    assert len(render(worst)) == NOTE_CAP
    assert "999+" in render(worst)
    for unlisted in (0, 1, UNLISTED_CAP):
        assert len(render(ServerToolRecord(uses=worst.uses, unlisted=unlisted))) <= NOTE_CAP


def test_a_bound_is_measured_in_characters_not_escaped_bytes():
    found = record([ServerToolUse("web_search", "東京の天気" * 100)])
    assert len(found.uses[0].input) == INPUT_CAP  # characters, as the model reads them


def test_summarize_reads_what_a_tool_ran_on():
    assert summarize({"url": FETCHED}) == FETCHED
    assert summarize({"query": "dallas weather"}) == "dallas weather"
    assert summarize({"queries": ["a", "b"]}) == "a, b"
    assert summarize({"code": "print(1)"}) == "print(1)"
    assert summarize({"limit": 3}) == '{"limit": 3}'
    assert summarize(None) == summarize({}) == ""
    assert summarize("as is") == "as is"


# === the note, beside the turn ================================================================


def test_the_note_names_the_tool_and_its_input_and_says_what_it_is():
    note = render(fetched().server_tools)
    assert note.startswith(HEADER)
    assert f"\n- web_fetch: {FETCHED}" in note


def test_the_note_follows_the_turn_and_its_tool_results_never_splits_a_call():
    turn = fetched()
    turn.tool_calls = [ToolCall(id="c1", name="echo", arguments={})]
    later = Message.user("next")
    history = [Message.user("go"), turn, Message.tool("c1", "echoed"), later]
    shown = with_records(history)
    assert [m.role for m in shown] == ["user", "assistant", "tool", "system", "user"]
    assert shown[3].content == render(turn.server_tools)
    assert shown[1] is turn and shown[4] is later  # the same objects, never copies
    assert history == [history[0], turn, history[2], later]  # the transcript is untouched


def test_a_trailing_turn_gets_its_note_and_a_list_with_none_is_returned_as_is():
    history = [Message.user("go"), fetched()]
    assert with_records(history)[-1].role == "system"
    plain = [Message.user("go"), Message.assistant(content="hi")]
    assert with_records(plain) is plain


def test_the_record_is_not_a_tool_call_to_the_recovery():
    """No create counted, nothing healed: the recovery reads `ToolCall`s, and this is not one."""
    turn = fetched()
    history = [Message.user("go"), turn]
    assert heal_interrupted_calls(history) == 0
    assert list(creates(history)) == []


# === the engine: bounded on arrival, shown on the next call, never written into the transcript ==


class _Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen: list[list[Message]] = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)


class _Echo(Tool):
    name = "echo"
    description = "Echo."

    def run(self, **kwargs):
        return "echoed"


def test_the_next_step_sees_the_note_and_the_transcript_keeps_only_the_record():
    first = Message.assistant(tool_calls=[ToolCall(id="c1", name="echo", arguments={})])
    first.server_tools = ServerToolRecord(
        uses=[ServerToolUse("web_fetch", FETCHED + "/" + "p" * 900)]
    )
    model = _Scripted(first, Message.assistant(content="done"))
    registry = ToolRegistry()
    registry.register(_Echo())
    messages = [Message.user("fetch it, then echo")]

    Engine(model, registry).run(messages)

    step_two = model.seen[1]
    notes = [m for m in step_two if m.role == "system" and m.content.startswith(HEADER)]
    assert len(notes) == 1
    at = step_two.index(notes[0])
    assert step_two[at - 1].role == "tool"  # after the turn's result, never between call and result
    assert not any(m.role == "system" and (m.content or "").startswith(HEADER) for m in messages)
    # Bounded at arrival: what the disk keeps is what the model read.
    assert len(first.server_tools.uses[0].input) == INPUT_CAP
    assert first.server_tools.uses[0].input in notes[0].content


def test_a_later_wake_is_shown_what_an_earlier_wake_fetched(tmp_path):
    """The issue's reproduction, at the library seam: a restart replays the record as a note."""
    Harness(_Scripted(fetched()), home=tmp_path).send("fetch it", source="timeline:t1")

    second = _Scripted(Message.assistant(content="ok"))
    Harness(second, home=tmp_path).send("anything else?", source="timeline:t1")

    shown = second.seen[0]
    turn = next(i for i, m in enumerate(shown) if m.role == "assistant")
    assert shown[turn].server_tools.uses == [ServerToolUse("web_fetch", FETCHED)]
    assert shown[turn + 1].role == "system" and f"web_fetch: {FETCHED}" in shown[turn + 1].content


# === the arithmetic: transcript cost, request size, attribution ===============================


def test_the_record_is_its_own_attribution_section_and_the_sections_still_sum():
    turn = fetched()
    convo = [Message.system("charter"), Message.user("go"), turn]
    fields = attribute(convo)
    assert fields["history_server_tools"] == chars(turn) == len(render(turn.server_tools))
    assert fields["history_assistant"] == message_chars(turn)
    sections = sum(v for k, v in fields.items() if k.startswith("history_"))
    assert sections == fields["history"]
    assert (
        fields["total"] == fields["tools"] + fields["brief"] + fields["history"] + fields["images"]
    )


def test_a_section_with_nothing_in_it_still_reports_its_zero():
    assert attribute([Message.user("go")])["history_server_tools"] == 0


def test_the_transcript_cost_counts_the_note_once_and_the_request_counts_it_once():
    turn = fetched()
    history = [Message.user("go"), turn]
    assert _chars(history) == message_chars(history[0]) + message_chars(turn) + chars(turn)
    # The request already carries the note as a turn of its own; it is not counted twice.
    assert request_chars(with_records(history)) == _chars(history)


def test_the_summarizer_is_shown_the_record_so_a_compaction_keeps_the_fetch():
    rendered = _render([Message.user("go"), fetched()])
    assert f"web_fetch: {FETCHED}" in rendered


# === through the real Anthropic SDK: two wakes, the second shown the first's fetch ============


MESSAGES = f"{DEFAULT_BASE_URL}/v1/messages"
FAKE_KEY = "sk-ant-api03-FAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfake-0000AA"


def _reply(*content):
    return httpx.Response(
        200,
        json={
            "id": "msg_01FakeFakeFakeFake",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": list(content),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 7},
        },
    )


FETCH_USE = {
    "type": "server_tool_use",
    "id": "srvtoolu_01F",
    "name": "web_fetch",
    "input": {"url": FETCHED},
}
FETCH_RESULT = {
    "type": "web_fetch_tool_result",
    "tool_use_id": "srvtoolu_01F",
    "content": {
        "type": "web_fetch_result",
        "url": FETCHED,
        "retrieved_at": "2026-10-31T09:00:00Z",
        "content": {
            "type": "document",
            "title": "Example Domain",
            "citations": None,
            "source": {"type": "text", "media_type": "text/plain", "data": "Example Domain"},
        },
    },
}


@pytest.fixture
def router():
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as r:
        yield r


def test_an_anthropic_fetch_is_on_the_next_wakes_request(router, tmp_path):
    route = router.post(MESSAGES).mock(
        side_effect=[
            _reply(FETCH_USE, FETCH_RESULT, {"type": "text", "text": "Title: Example Domain."}),
            _reply({"type": "text", "text": "Nothing to correct."}),
        ]
    )
    for prompt in ("fetch it and post its title", "anything else?"):
        adapter = AnthropicProvider("claude-haiku-5-5", api_key=FAKE_KEY, max_tokens=64)
        Harness(adapter, home=tmp_path).send(prompt, source="timeline:t1")
        adapter.close()

    (transcript,) = tmp_path.rglob("sessions/*.json")
    persisted = [t for t in json.loads(transcript.read_text()) if "server_tools" in t]
    assert [t["server_tools"] for t in persisted] == [
        {"uses": [{"name": "web_fetch", "input": FETCHED}]}
    ]

    second = json.loads(route.calls[1].request.content)["messages"]
    assistant = next(i for i, m in enumerate(second) if m["role"] == "assistant")
    assert (
        second[assistant]["content"][0]["text"] == "Title: Example Domain."
    )  # text and calls only
    note = second[assistant + 1]["content"][0]["text"]
    assert note.startswith(SYSTEM_LABEL + HEADER)
    assert f"web_fetch: {FETCHED}" in note
