"""What the model is about to read, section by section — one line per assembled turn.

An agent's input-token count is the single largest recurring cost it has, and until now nothing
in the fleet could say **what composes it**. The per-call line reports the total the provider
charged for (``llm … tokens_in=``) and the compaction line reports when the total crossed a
threshold, but a standing agent sitting at ~494 K input tokens per call left a question no log
could answer: how much of that is the charter, the tool schemas, the timeline history, recalled
memory, or the per-wake brief? (basecradle-noc#388: @glm-5.2, three days, every wake, and an
intended-vs-defect ruling blocked on attribution — issue #369.)

This module answers it, and the shape of the answer is the whole design:

**It is measured off the assembled payload, never re-derived.** The list handed here is the list
`Session._drive` is about to hand `Engine.run`, and the specs are the registry's own
(`ToolRegistry.specs`) — so the line reports what *was* assembled, not what a second pass thinks
should have been. That is the `overlay_tool_stems` principle applied to context: report what
loaded. Nothing here reads a config, a prompt file, or a plugin manifest to reconstruct a number
it could instead have counted.

**It reports measurements and renders no verdict.** No section is labelled bloat, nothing is
compared against a budget, and no threshold lives here. A section that looks wrong is a question
for whoever reads it; the harness's job is to make the question answerable. (The one adjacent
number that *is* a judgment — the compaction threshold — is `_context`'s, and it is stated on its
own line.)

**The unit is characters, and it is named on the line.** Tokens would be the natural unit and the
harness will not fabricate them: a client-side count needs a tokenizer per model, GLM publishes
none, and this repo's standing rule is that a number it cannot get honestly is a number it does
not print (`_context`: "a client-side count could not even be honest, let alone free"). So the
line carries characters — exact, free, and in the *same* unit the compaction arithmetic already
uses (`_context.message_chars`, ``ensure_ascii=False``: what a character costs the model does not
depend on the script it is written in). Converting is a division the reader can do and the
harness cannot: the ``llm`` line that follows this one within milliseconds carries the provider's
own ``tokens_in`` for very nearly this payload, so the two together give a **measured**
chars→tokens ratio per agent, per model, per wake, rather than an assumed one.

*Very nearly*, and the gap is named rather than glossed: the engine appends its own step-counter
note immediately before that first call (tens of characters — the engine's, not the assembly's),
and the adapter wraps everything in a vendor envelope on the wire. Both make ``tokens_in``
fractionally larger than these characters imply, so a ratio computed from the pair is a slight
over-estimate of chars-per-token and errs in the safe direction, exactly as
`_context.maybe_compact`'s does and for the same reason.

**The sections add up.** ``total`` is the sum of every section reported, exactly, which is what
lets a reader trust a share they compute from it. Three consequences follow, and each is a
decision: the brief's parts are charged their joining separators (`_brief.brief_section_sizes`);
image payload is its own section rather than folded into the turn that carries it (base64 is
enormous in characters and is *not* billed in text tokens — folding it in would inflate the one
section a reader is most likely to misread); and what this measures is model-visible content, not
wire bytes — a provider's own JSON envelope (role keys, tool-call wrappers, the vendor's tool
schema shape) is the adapter's, varies per surface, and is not the agent's context to account for.

**What it is not: a per-call line.** It fires once per *assembled turn*, before the first model
call of that turn, because that is where the sections are distinguishable — a turn's later steps
append the model's own output and its tools' results to the tail, which the step ledger and the
per-call ``tokens_in`` already track. A wake that runs several turns logs several lines, which is
correct: each one assembled its own payload.

The cost is one pass over the transcript per turn — the same walk `_context` already makes when it
compacts, against a payload the harness is about to spend seconds and cents sending to a model. It
is why the line is per *turn* and not per *step*: paying it 24 times to re-measure a prefix that
did not change would be a real cost for no information.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from basecradle_harness._context import is_summary, message_chars
from basecradle_harness._engine import is_step_note
from basecradle_harness._messages import Message, ToolSpec
from basecradle_harness._observability import kv

_log = logging.getLogger("basecradle_harness")

#: The unit every size on the line is measured in, stated on the line itself. Characters as the
#: **model** reads them, never bytes as the disk escapes them — the `_session._json_size` lesson
#: (issue #301): ``ensure_ascii=True`` expands one Japanese character into six, and a "size" that
#: does that reports a Japanese-speaking agent's context as six times an English one's.
UNIT = "chars"

#: The transcript's sections, in the order they are reported. A fixed partition, so the line's
#: schema is stable for a machine reader: each is emitted every turn, including at ``0``, because
#: "this agent's transcript holds no tool output at all" is a fact worth logging (the same reason
#: `_observability.kv` keeps a zero and drops a ``None``).
#:
#: ``other`` is the catch-all, and it is not decoration. The line's whole contract is that its
#: sections **sum to** ``total``, so the partition may not have a hole in it — a turn wearing a
#: role outside the vocabulary (a hand-edited transcript, a future role) has to land *somewhere*
#: or the sum silently stops closing, and the alternative — letting the lookup raise — throws away
#: the whole line over one odd message. It should read ``0`` forever; if it ever does not, that is
#: itself the finding.
_SECTIONS = ("charter", "summary", "steps", "user", "assistant", "tool", "other")


def log_context_attribution(
    convo: Sequence[Message],
    *,
    tools: Sequence[ToolSpec] | None = None,
    brief: Message | None = None,
    brief_sections: Mapping[str, int] | None = None,
    source: str | None = None,
) -> None:
    """Emit the one attribution line for the payload in `convo` — and never break the turn.

    Args:
        convo: The message list about to be handed to the engine, exactly as assembled.
        tools: The tool schemas the model will be offered — the registry's own `ToolSpec` list,
            which is where an MCP server's tools live too (they register as ordinary tools, so
            they are counted here without this module knowing MCP exists).
        brief: The ephemeral per-wake brief's turn *object*, identified by identity rather than
            by position or by a marker in its text. Position would be a second statement of the
            splice `_session._exchange` already makes, and it would be silently wrong the day
            that splice moves; a text marker would be a string in the model's context that exists
            only for a log line.
        brief_sections: What each named part of the brief contributed
            (`_brief.brief_section_sizes`), or ``None`` for a caller that composed no brief or
            does not track its parts (the library API). Reported as ``brief_<part>=``.
        source: The session's channel key, so a line can be tied to the conversation it measured.

    A failure here is a WARNING and nothing more. Observability never breaks a turn — and this one
    is pure arithmetic over data structures, so a failure means a real defect worth seeing rather
    than a condition worth tolerating quietly.
    """
    try:
        fields = attribute(
            convo, tools=tools, brief=brief, brief_sections=brief_sections, source=source
        )
    except Exception:  # noqa: BLE001 - a measurement must never cost a turn
        _log.warning("Could not attribute the assembled context; continuing.", exc_info=True)
        return
    _log.info("context attribution %s", kv(**fields))


def attribute(
    convo: Sequence[Message],
    *,
    tools: Sequence[ToolSpec] | None = None,
    brief: Message | None = None,
    brief_sections: Mapping[str, int] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """The line's fields, in the order they are rendered. See `log_context_attribution`.

    Split out from the emitter so the partition is testable as data rather than as a log string:
    the property that matters — every section sums to ``total`` — is an assertion about this
    mapping, and a test that has to parse a rendered line to make it would be pinning the
    formatter instead of the arithmetic.
    """
    history = [message for message in convo if message is not brief]
    text = dict.fromkeys(_SECTIONS, 0)
    images = 0
    for message in history:
        text[_section(message)] += message_chars(message)
        images += _image_chars(message)
    if brief is not None:
        images += _image_chars(brief)
    brief_chars = message_chars(brief) if brief is not None else 0
    tool_chars = sum(_spec_chars(spec) for spec in tools or ())
    history_chars = sum(text.values())
    fields: dict[str, Any] = {
        "unit": UNIT,
        "source": source,
        "total": tool_chars + brief_chars + history_chars + images,
        "messages": len(history),
        "tools": tool_chars,
        "tools_count": len(tools or ()),
        "brief": brief_chars,
    }
    # Only the parts that were actually composed — `_brief.brief_parts` drops an absent or blank
    # one, so a section that names nothing never appears. That asymmetry with `_SECTIONS` is
    # deliberate: the transcript's roles are a fixed partition worth a stable schema, while the
    # brief's parts are a set the operator's config decides.
    for name, size in (brief_sections or {}).items():
        fields[f"brief_{name}"] = size
    fields["history"] = history_chars
    for name in _SECTIONS:
        fields[f"history_{name}"] = text[name]
    fields["images"] = images
    return fields


def _section(message: Message) -> str:
    """Which transcript section a turn belongs to.

    Three of the four roles answer for themselves. `system` does not, and the split matters: a
    transcript's system turns are the agent's **charter** (standing context, seeded once), the
    **summaries** compaction left behind (conversation, cumulative), and the engine's **step**
    ledger (one note per model call, forever) — three things with completely different growth
    behavior that a single ``system`` bucket would report as one number.

    Both non-role tests are the *writers'* own (`_context.is_summary`, `_engine.is_step_note`), not
    a pattern this module invents; see those functions for why a marker and its reader live
    together. Everything else a system turn can be — the seeded charter, a `_drive` failure note —
    lands in ``charter``, which is the honest name for "standing context the harness put here".

    The per-wake brief never reaches here: `attribute` removes it by identity first, so it is
    reported as ``brief`` and its parts, never as a transcript section.

    A role outside the four the vocabulary defines lands in ``other`` rather than raising — see
    `_SECTIONS` for why the partition must not have a hole in it.
    """
    if message.role != "system":
        return message.role if message.role in _SECTIONS else "other"
    if is_summary(message):
        return "summary"
    return "steps" if is_step_note(message) else "charter"


def _image_chars(message: Message) -> int:
    """A turn's image payload, in characters of the reference the provider is handed.

    Its own section, never folded into the turn that carries it. An inlined asset arrives as a
    ``data:`` URL, so a single photo can outweigh an entire transcript in characters while costing
    a fraction of it in tokens — the vendor prices an image by its own rules, not by the length of
    the base64. Counted rather than dropped because it *is* on the wire and a turn that carries one
    is genuinely expensive; kept separate so it can never be mistaken for text.

    It is also transient by construction: the engine evicts a shown image after the turn that
    showed it, so this is nonzero only on the turn that presents one (`_engine._evict_images`).
    """
    return sum(len(image.url) + len(image.alt or "") for image in message.images)


def _spec_chars(spec: ToolSpec) -> int:
    """One tool schema's cost — its name, its description, and its JSON-Schema parameters.

    The three fields the model is actually shown. What an adapter *wraps* them in on the wire is
    the adapter's (and differs by surface), so it is deliberately not counted here: this is the
    agent's context, not the vendor's envelope.
    """
    return (
        len(spec.name)
        + len(spec.description)
        + len(json.dumps(spec.parameters, ensure_ascii=False, default=str))
    )
