"""The harness's own elision markers — rendered here, and recognized here (issue #576).

When the transcript cap shortens something on its way to the disk (`_session`), it leaves a marker in
the gap: head, `[... elided from 2413 chars — … ...]`, tail. Every later wake replays that transcript
to the model, so the model reads its own past speech in that shape. At 2026-09-29 01:09Z @briggs,
writing a long reply, **reproduced it**: a head, the marker with a size in it, a tail. The harness
posted exactly what he wrote, and the platform then held a fragment whose own text said "the full
value was sent when the call ran". There never was a full value.

**A marker in the model's context is harness text the model can copy**, like any other text it reads.
The cap was not what went wrong. It runs on a copy made for the disk. The live dispatch runs the calls
the provider adapter has just parsed out of the model's response. The one path that runs arguments
read back from disk, the recovery re-issuing an interrupted create, runs only arguments the cap kept
whole (`_session._replayable`). `test_elision.py` pins the first two; `test_resume.py` pins the third.
What was missing is the other half of "the harness never files its own words as the agent's" (CLAUDE.md
→ The Mining Boundary): the agent's call must not carry the harness's words out either. So the engine
refuses to run a call whose arguments contain one of these markers (`Engine._run_tool`, via
`archive_marker`), and tells the model what the marker is.

**Rendered and recognized from the same templates**, never spelled twice: `_session` renders through
`result_marker`, `argument_marker` and `gone`; `_context` derives the excerpt opening it trims
identifiers at from `EXCERPT_OPENING`; the guard compiles its regex from the templates themselves. A
matcher that drifted from the renderer would be a guard that silently stopped matching, which is the
#438 lesson (a mined rendering that drifted from the shown one) in another place. The wordings the
harness no longer writes stay recognized too (`_LEGACY`), with their provenance: the cap is a fixed
point, so an excerpt capped in 0.70.0 is re-saved byte for byte on every wake since, and a model can
still be reading it today.

**Only markers that stand in for content the reader no longer has belong here.** Harness text that is
*true when quoted* (a shell tool's truncation note, the INTERRUPTED result, a step note) is not an
excerpt of anything, and refusing it would refuse legitimate forwarding.

Matching is **exact on the prose, loose only where copying is**: every word and every mark of
punctuation must be the harness's own, a number may be any run of digits (grouped with commas or
not), and a run of spaces may be any whitespace (a model that line-wraps a long paragraph wraps the
marker with it). A message that *talks about* elision or excerpts does not match. A message that
quotes a marker **verbatim** does, the harness's own docs included, and the error says to reword it:
a refused quote costs a step, where a missed copy puts a false claim on a timeline.
"""

from __future__ import annotations

import re
from typing import Any

#: How every marker opens. An excerpt marker also sits behind a blank line (`EXCERPT_OPENING`).
_OPEN = "[... "

#: The marker `_session._elide` leaves in a capped tool **result**, since issue #275 (0.62.0).
_RESULT = (
    _OPEN
    + "{cut} chars elided of {total} — this is an archived excerpt; the full result was shown "
    "when the tool ran. Re-run it if you need it in full. ...]"
)

#: The marker `_session._elide_argument` leaves in a capped tool-call **argument**, since issue #301
#: (0.70.0). This is the one @briggs copied (issue #576).
_ARGUMENT = (
    _OPEN
    + "elided from {size} chars — this argument is an archived excerpt; the full value was sent "
    "when the call ran. ...]"
)

#: The floor (`gone`): all that is left when a share has no room for an excerpt. Since
#: issue #304 (0.72.0).
_GONE = _OPEN + "{size} chars elided ...]"

#: Wordings no source writes any more, and every transcript persisted while they were current can
#: still hold. Both are 0.70.0–0.71.x (issue #301) and both were replaced by `_GONE` in issue #304: the
#: first stood in for a non-string argument, the second for a call too big to bound by cutting its
#: values (`_session._arguments_stub`). Never remove one while a transcript could still carry it.
_LEGACY = (
    (
        "[... {size} chars elided — this argument was archived out of the transcript; the full value "
        "was sent when the call ran. ...]"
    ),
    (
        "[... the {count} arguments of this call ({size} chars) were archived out of the transcript; "
        "they were sent in full when the call ran. ...]"
    ),
)

#: What text looks like where an excerpt marker begins: the blank line that sets it off, then its
#: opening. `_context` stops harvesting an identifier that runs into one, because the identifier was
#: cut short there.
EXCERPT_OPENING = "\n\n" + _OPEN

#: What a number in a marker may look like once a model has copied it: the harness writes bare
#: digits, and a model may group them.
_NUMBER = r"\d[\d,]*"


def result_marker(cut: int, total: int) -> str:
    """What stands in for the elided middle of a tool result, blank lines included."""
    return "\n\n" + _RESULT.format(cut=cut, total=total) + "\n\n"


def argument_marker(size: int) -> str:
    """What stands in for the elided middle of one tool-call argument, blank lines included."""
    return "\n\n" + _ARGUMENT.format(size=size) + "\n\n"


def gone(size: int) -> str:
    """What an elision says when there is no room even for an excerpt: how much there was, and nothing.

    **The floor of the whole cap, and the one place the bound stops being hard** — so it is worth
    saying exactly what it is. A tool result cannot be *dropped* (its call would dangle, and a dangling
    `tool_call_id` is malformed permanently) and neither can a call's arguments (`create_kind` reads
    them, and a create the recovery cannot count is a message posted twice). A thing that cannot be
    dropped must be allowed to say that it is gone — so a step that fans out wider than its budget has
    characters for pays one of these per call, and the total creeps past `TOOL_RESULT_CAP` at a fan-out
    of ~140 (or `TOOL_ARGS_CAP` at ~50).

    That residue is bounded by **what the model emitted, never by what its tools returned** — one
    short record per call it chose to make, of the same order as the `id`+`name` envelope the transcript
    must keep for that call anyway, and bounded the same way (the provider's max-output-tokens). It is
    the excerpt *markers* that are chatty, and deliberately: they accompany content worth reading. Here
    there is none, and their prose ("re-run it if you need it in full") would cost five times the fact
    it decorates — per call, on the one shape where every call is already down to its last few dozen
    characters.
    """
    return _GONE.format(size=size)


def _pattern(template: str) -> str:
    """`template` as a regex: its prose exact, each placeholder a number, each space any whitespace."""
    prose = re.split(r"\{\w+\}", template)
    return _NUMBER.join(r"\s+".join(map(re.escape, piece.split(" "))) for piece in prose)


_MARKER = re.compile("|".join(map(_pattern, (_ARGUMENT, _RESULT, _GONE, *_LEGACY))))


def archive_marker(arguments: Any) -> str | None:
    """The first harness elision marker anywhere in a call's `arguments`, or ``None``.

    Searches every string a model can put in a call: the values, and the values inside any list or
    object among them. Keys are not searched; a copied `_session._arguments_stub` carries its marker in
    a value, and an unexpected key already fails the call on its own. The walk is iterative, so no
    nesting depth a provider can deliver makes it raise.
    """
    pending = [arguments]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            found = _MARKER.search(value)
            if found:
                return found.group(0)
        elif isinstance(value, dict):
            pending.extend(reversed(list(value.values())))
        elif isinstance(value, (list, tuple)):
            pending.extend(reversed(value))
    return None


def is_marker(value: Any) -> bool:
    """Is `value` one of the harness's elision markers, **whole** — nothing before it, nothing after?

    What `_session` asks of a call it has already reduced to its stub, so that it is never reduced
    again (see `_session._cap_arguments`). Whole, because a value that merely *contains* a marker is an
    excerpt, and an excerpt is still something the cap may cut.
    """
    return isinstance(value, str) and _MARKER.fullmatch(value) is not None


def refusal(marker: str, *, reissue: bool = False) -> str:
    """What the model is told instead of a result, when its call carried `marker` (see the module).

    It says what the marker is, what did and did not happen, and the ways forward, without choosing
    between them. **What happened depends on the seam**, and saying the wrong one costs a peer a
    duplicate. A live call that is refused was never sent. A call the recovery is re-issuing
    (`reissue`) was *already attempted* by a wake that died before recording the outcome, so
    whether it reached anyone is unknown. It is not re-issued, and the model is told to check before
    it sends anything again, exactly as `_session.INTERRUPTED` tells it for a call nothing can
    re-run.

    The marker is quoted only up to its dash, and never with its closing `...]`, so the error cannot
    become one more copy of the text it warns about: a model that quotes the error back is not
    refused for it.
    """
    flat = " ".join(marker.split())
    opening, dash, _ = flat.partition(" — ")
    shown = f"{opening} — …]" if dash else flat.removesuffix("...]") + "…]"
    what = (
        "The harness writes that marker, never you: it marks where one of your earlier calls or tool "
        "results was shortened when it was saved, and the text either side of it is an excerpt, not "
        "the whole."
    )
    if reissue:
        return (
            f"Error: this call was not re-issued. Its arguments contain an elision marker (“{shown}”). "
            f"{what} The wake that first made this call was killed before its result was recorded, so "
            f"whether it reached anyone is unknown, and it has not been sent again. If it matters, "
            f"check for yourself before sending anything. If you do send it, write the complete "
            f"content out."
        )
    return (
        f"Error: this call was not run, and nothing was sent. Its arguments contain an elision "
        f"marker (“{shown}”). {what} If you meant to send this, write the complete content out and "
        f"call again; if it repeats something you posted earlier, the full text is on the timeline. "
        f"If you are quoting the marker on purpose, reword it."
    )
