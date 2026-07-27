"""The context-attribution line: what the model is about to read, section by section (issue #369).

The properties worth pinning are not "a line was logged" but the three claims the line makes:
its sections **add up**, they are measured off the **assembled payload** rather than re-derived,
and each section is the thing its name says. The first is what a reader computes shares from; the
second is what makes those shares true; the third is what makes them useful.
"""

import json
import logging
from datetime import datetime, timezone

import pytest

from basecradle_harness import (
    Engine,
    ImageContent,
    Message,
    Session,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolSpec,
)
from basecradle_harness._attribution import attribute, log_context_attribution
from basecradle_harness._brief import brief_parts, brief_section_sizes, join_brief
from basecradle_harness._context import TOOL_RESULT_CAP
from basecradle_harness._engine import _step_note, is_step_note
from tests.test_session import ScriptedProvider, calls_tool, text

# Aware UTC, matching what `Engine` actually feeds `_step_note` (`datetime.now(timezone.utc)`) —
# the note labels itself UTC, so the fixture should be the same kind of value the note renders.
NOW = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)


class Weather(Tool):
    name = "weather"
    description = "Report the weather for a city."
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "The city."}},
        "required": ["city"],
    }

    def run(self, city: str = "Dallas") -> str:
        return f"Clear in {city}."


class Firehose(Tool):
    """A tool whose result is far past `TOOL_RESULT_CAP`, so persistence has to elide it."""

    name = "firehose"
    description = "Return more than anyone asked for."

    def run(self) -> str:
        return "x" * (TOOL_RESULT_CAP * 4)


def spec_chars(spec: ToolSpec) -> int:
    return len(spec.name) + len(spec.description) + len(json.dumps(spec.parameters))


# --- the sections add up ------------------------------------------------------


def test_every_section_sums_to_the_reported_total():
    """The line's whole contract: a share computed off ``total`` is a real share.

    Sum the three top-level sections plus images and you land exactly on ``total``. A section
    that quietly did not count (or double-counted) would leave a reader dividing by a number
    that describes a payload nobody assembled.
    """
    brief = Message.system("You are Nova Digital.")
    convo = [
        Message.user("hello"),
        Message.assistant(content="thinking", tool_calls=[ToolCall(id="c1", name="weather")]),
        Message.tool(tool_call_id="c1", content="Clear in Dallas."),
        brief,
        Message.user("and now?"),
    ]
    fields = attribute(convo, tools=[Weather().to_spec()], brief=brief)
    assert (
        fields["tools"] + fields["brief"] + fields["history"] + fields["images"] == fields["total"]
    )


def test_the_transcript_sections_sum_to_the_transcript():
    convo = [
        Message.system("You are Nova Digital."),
        Message.system("[Earlier conversation compacted: 4 messages replaced] my notes"),
        Message.system(_step_note(1, 24, NOW)),
        Message.user("hello"),
        Message.assistant(content="hi"),
        Message.tool(tool_call_id="c1", content="a result"),
    ]
    fields = attribute(convo)
    sections = ("charter", "summary", "steps", "user", "assistant", "tool")
    assert sum(fields[f"history_{name}"] for name in (*sections, "other")) == fields["history"]
    assert all(fields[f"history_{name}"] > 0 for name in sections)
    assert fields["history_other"] == 0  # nothing here wears a role the vocabulary lacks


def test_a_role_outside_the_vocabulary_lands_in_other_rather_than_losing_the_line():
    """The partition may not have a hole in it, or the sections stop summing to `total`.

    A hand-edited transcript can carry anything (`Message.from_dict` takes `role` verbatim), and
    the two failure modes this avoids are both worse than a catch-all: a `KeyError` throws the
    whole line away over one odd message, and a silent skip leaves a `total` nobody can check.
    """
    rogue = Message(role="observer", content="x" * 40)  # type: ignore[arg-type]
    fields = attribute([rogue, Message.user("hi")])
    assert fields["history_other"] == 40
    assert fields["history"] == 40 + len("hi") == fields["total"]


def test_the_brief_parts_sum_to_the_brief():
    """`brief_section_sizes` charges each part its joining separator, so the parts partition it.

    Two characters per part is nothing; a reader who checks and finds the parts *don't* sum has
    no way to tell a rounding convention from a section the line forgot to report.
    """
    parts = brief_parts(
        now="Current Time: 2026-07-26 12:00:00 UTC (+00:00, Sunday)",
        budget="Step budget: 24 steps.",
        initialize="Operate like this.",
        manifest="Your active tools right now:\n- weather",
        dashboard="# Dashboard",
        memory="You met John Doe on Tuesday.",
        system_prompt="You are Nova Digital.",
    )
    composed = join_brief(parts)
    sizes = brief_section_sizes(parts)
    assert sum(sizes.values()) == len(composed)

    brief = Message.system(composed)
    fields = attribute([brief, Message.user("hi")], brief=brief, brief_sections=sizes)
    assert sum(fields[f"brief_{name}"] for name in sizes) == fields["brief"]


def test_an_absent_brief_part_is_not_reported_as_a_zero():
    """The brief's parts are a set the config decides, so a part nobody composed names nothing.

    Distinct from the transcript's roles, which are a fixed partition and *are* reported at
    zero — "this agent's transcript holds no tool output at all" is a fact, where "there is no
    memory section" is just an operator who runs no memory provider.
    """
    parts = brief_parts(
        initialize="Operate like this.",
        manifest=None,
        dashboard=None,
        system_prompt="You are Nova Digital.",
    )
    brief = Message.system(join_brief(parts))
    fields = attribute(
        [brief, Message.user("hi")], brief=brief, brief_sections=brief_section_sizes(parts)
    )
    assert "brief_initialize" in fields and "brief_system_prompt" in fields
    assert "brief_memory" not in fields and "brief_dashboard" not in fields
    assert fields["history_tool"] == 0  # a fixed section still reports its zero


# --- each section is the thing its name says ----------------------------------


def test_the_brief_is_attributed_to_the_brief_not_to_the_transcript():
    """Identified by **identity**, so the splice point is stated once — where the splice happens."""
    brief = Message.system(
        "Current Time: 2026-07-26 12:00:00 UTC (+00:00, Sunday)\nstanding context"
    )
    fields = attribute([Message.user("hello"), brief, Message.user("and now?")], brief=brief)
    assert fields["brief"] == len(brief.content)
    assert fields["history_charter"] == 0
    assert fields["history_steps"] == 0  # the brief's clock is not the engine's step ledger
    assert fields["messages"] == 2  # the brief is not a turn of the conversation


def test_step_notes_are_their_own_section_not_the_charter():
    """24 notes per model call, forever — a growth curve the charter's does not resemble."""
    charter = Message.system("You are Nova Digital.")
    notes = [Message.system(_step_note(step, 24, NOW)) for step in (1, 2, 22)]
    fields = attribute([charter, *notes, Message.user("hi")])
    assert fields["history_charter"] == len(charter.content)
    assert fields["history_steps"] == sum(len(note.content) for note in notes)


def test_is_step_note_recognizes_what_step_note_writes_and_nothing_else():
    """The writer/reader pairing, pinned — both note forms, and the brief's look-alike anchor."""
    assert is_step_note(Message.system(_step_note(1, 24, NOW)))  # terse
    assert is_step_note(Message.system(_step_note(24, 24, NOW)))  # escalated
    # The brief's own time anchor opens identically and must never be mistaken for one.
    anchor = "Current Time: 2026-07-26 12:00:00 UTC (+00:00, Sunday)\nThis clock is UTC."
    assert not is_step_note(Message.system(anchor))
    assert not is_step_note(Message.user(_step_note(1, 24, NOW)))  # role still matters


def test_a_compaction_summary_is_its_own_section():
    summary = Message.system("[Earlier conversation compacted: 40 messages replaced] my notes")
    fields = attribute([summary, Message.user("hi")])
    assert fields["history_summary"] == len(summary.content)
    assert fields["history_charter"] == 0


def test_tool_call_arguments_are_charged_to_the_assistant_turn():
    """The same unit the compaction proof uses (`_context.message_chars`) — text *and* arguments.

    A call's arguments are persisted context (`TOOL_ARGS_CAP` exists because of it), so an
    attribution that counted only ``content`` would report a tool-heavy agent's assistant turns
    as nearly free.
    """
    call = ToolCall(id="c1", name="weather", arguments={"city": "Dallas"})
    fields = attribute([Message.assistant(content="checking", tool_calls=[call])])
    assert fields["history_assistant"] == (
        len("checking") + len("weather") + len(json.dumps({"city": "Dallas"}))
    )


def test_sizes_are_measured_in_characters_the_model_reads_not_bytes_the_disk_escapes():
    """The `_json_size` lesson (issue #301): a Japanese character costs one, not six."""
    english = Message.assistant(
        tool_calls=[ToolCall(id="c1", name="t", arguments={"body": "aaaaa"})]
    )
    japanese = Message.assistant(
        tool_calls=[ToolCall(id="c1", name="t", arguments={"body": "あああああ"})]
    )
    assert attribute([english])["total"] == attribute([japanese])["total"]


def test_tool_schemas_are_their_own_section():
    spec = Weather().to_spec()
    fields = attribute([Message.user("hi")], tools=[spec, spec])
    assert fields["tools"] == 2 * spec_chars(spec)
    assert fields["tools_count"] == 2


def test_image_payload_is_its_own_section_never_folded_into_the_turn():
    """base64 is enormous in characters and is not billed in text tokens — so it is never text."""
    pixels = ImageContent(url="data:image/png;base64," + ("A" * 4000), alt="photo.png")
    fields = attribute([Message(role="user", content="look at this", images=[pixels])])
    assert fields["history_user"] == len("look at this")
    assert fields["images"] == len(pixels.url) + len("photo.png")
    assert fields["total"] == fields["history"] + fields["images"]


# --- it never costs a turn ----------------------------------------------------


def test_a_measurement_failure_warns_and_returns_rather_than_raising(caplog):
    """Observability never breaks a turn — and a failure here is a real defect, so it is loud."""
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        log_context_attribution([object()])  # not a `Message` at all
    assert "Could not attribute the assembled context" in caplog.text


# --- measured off the assembled payload, end to end ---------------------------


def build(tmp_path, *replies, tools=None):
    registry = ToolRegistry()
    # `None` (not `()`) is the "use the default" sentinel, so `tools=()` still means "no tools".
    for tool in (Weather(),) if tools is None else tools:
        registry.register(tool)
    return Session(
        "timeline:0198e3f1-0000-7000-8000-000000000001",
        Engine(ScriptedProvider(*replies), registry),
        path=tmp_path / "session.json",
    )


def lines(caplog) -> list[dict[str, str]]:
    """Every emitted attribution line, parsed back into fields."""
    head = "context attribution "
    bodies = [r.getMessage() for r in caplog.records if r.getMessage().startswith(head)]
    return [dict(f.split("=", 1) for f in body[len(head) :].split(" ")) for body in bodies]


def sizes(line: dict[str, str]) -> dict[str, int]:
    return {key: int(value) for key, value in line.items() if value.lstrip("-").isdigit()}


def test_a_real_send_reports_the_payload_it_assembled(tmp_path, caplog):
    """The end-to-end claim: what the line says is what `Session._drive` handed the engine.

    Everything is present and accounted for on a live turn — the brief and its parts, the
    registry's tool schema, the transcript — and it still sums.
    """
    parts = brief_parts(
        now="Current Time: 2026-07-26 12:00:00 UTC (+00:00, Sunday)",
        initialize="Operate like this.",
        manifest="Your active tools right now:\n- weather",
        dashboard=None,
        memory="You met John Doe on Tuesday.",
        system_prompt="You are Nova Digital.",
    )
    session = build(tmp_path, text("done"))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        session.send(
            "what's the weather?",
            brief=join_brief(parts),
            brief_sections=brief_section_sizes(parts),
        )
    fields = sizes(lines(caplog)[0])
    assert fields["brief"] == len(join_brief(parts))
    assert fields["brief_memory"] == len("You met John Doe on Tuesday.") + 2  # + its separator
    assert fields["tools"] == spec_chars(Weather().to_spec())
    assert fields["tools_count"] == 1
    assert fields["history_user"] == len("what's the weather?")
    assert (
        fields["tools"] + fields["brief"] + fields["history"] + fields["images"] == fields["total"]
    )


def test_the_line_names_its_unit_and_its_source(tmp_path, caplog):
    """A number with no unit is not a measurement — and one with no subject is not one either."""
    session = build(tmp_path, text("done"))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        session.send("hello")
    line = lines(caplog)[0]
    assert line["unit"] == "chars"
    assert line["source"] == "timeline:0198e3f1-0000-7000-8000-000000000001"


def test_one_line_per_assembled_turn(tmp_path, caplog):
    """A wake runs several turns; each assembles its own payload, so each reports its own."""
    session = build(tmp_path, text("first done"), text("second done"))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        session.send("first")
        session.send("second")
    assert len(lines(caplog)) == 2


def test_the_transcript_is_measured_as_it_now_stands_not_as_it_arrived(tmp_path, caplog):
    """A capped tool result is reported at what it *costs*, which is the point of measuring at all.

    An oversized result is elided on persistence (`TOOL_RESULT_CAP`), so the next turn's payload
    carries only the excerpt. An attribution that reported the original size would be naming a
    cost nobody is paying any more — and the whole reason this line exists is to say what the
    *next* call will be billed for.
    """
    session = build(
        tmp_path,
        calls_tool("c1", "firehose"),
        text("done"),
        text("second done"),
        tools=(Firehose(),),
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        session.send("go")
        session.send("and now?")
    first, second = (sizes(line) for line in lines(caplog))
    assert first["history_tool"] == 0  # nothing had run yet
    assert 0 < second["history_tool"] < TOOL_RESULT_CAP  # the excerpt, not the 16 KB firehose


def test_a_session_with_no_brief_reports_no_brief(tmp_path, caplog):
    """The plain library path: no brief, no brief sections, and the arithmetic still closes."""
    session = build(tmp_path, text("done"))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        session.send("hello")
    line = lines(caplog)[0]
    fields = sizes(line)
    assert fields["brief"] == 0
    assert not [key for key in line if key.startswith("brief_")]
    assert fields["tools"] + fields["history"] + fields["images"] == fields["total"]


@pytest.mark.parametrize(
    "part", ["now", "budget", "initialize", "manifest", "defects", "safety", "dashboard", "memory"]
)
def test_each_composed_brief_part_reaches_the_line_under_its_own_name(tmp_path, caplog, part):
    """The trigger's actual question, part by part — charter vs. manifest vs. dashboard vs. memory.

    Every seam `compose_brief` has, so a part added to the brief without a name on this line is a
    part the attribution silently folds into ``brief`` and nobody can see.
    """
    absent = dict.fromkeys(("initialize", "manifest", "dashboard", "system_prompt"))
    parts = brief_parts(**{**absent, part: "content"})
    session = build(tmp_path, text("done"))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        session.send("hi", brief=join_brief(parts), brief_sections=brief_section_sizes(parts))
    line = lines(caplog)[0]
    assert line[f"brief_{part}"] == str(len("content"))
