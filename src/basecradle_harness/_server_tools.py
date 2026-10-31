"""The record of server-side tool activity: what ran inside a model call, kept and shown (#727).

A **server-side** tool — Anthropic's ``web_fetch``, OpenAI's ``web_search_call``, xAI's agent tools,
Gemini's ``code_execution`` and ``url_context`` — runs inside the vendor's model call. Its blocks are
opaque vendor state, kept by an adapter for one wake at most and never persisted, so the transcript a
later wake replays holds the turn's text and its *client* calls and nothing else. The defect that
opened this module was a peer publicly retracting a true statement: a wake fetched a page and posted
its title, and the next wake, seeing a post with no fetch before it, read its own claim as a
fabrication and corrected it. The harness had hidden the evidence of its own action from it.

So each adapter that can see its vendor's server-tool activity reports it on the reply as a
`ServerToolRecord` (one `ServerToolUse` per use: the tool's name and one line of what it was given),
the engine bounds it where every reply enters the loop (`bound`, in `Engine._chat`), the transcript
persists it on the assistant turn, and every later call shows it to the model as one short system
note placed immediately after the turn and its tool results (`with_records`). Four constraints shape
it, and each rules out a form that looks simpler:

- **It is never the turn's `content`** (the Mining Boundary). A mining provider files the model's
  own words; harness text appended to the assistant's content would be filed as something the agent
  said. The record has its own field, the note is its own system turn, and `_wake._observe` is handed
  neither (`test_mining.py` carries a sentinel for it).
- **It is never a `ToolCall`** (the Delivery Guarantee). The recovery reads *"a tool call absent from
  the transcript never ran"*, heals an unanswered call as "outcome unknown", and counts create-shaped
  calls into an idempotency ordinal. A recorded server call shaped as a client call would read as one
  the harness dispatched and might need re-issuing.
- **It is bounded, and the bound is stated** (Context Discipline). At most `MAX_LISTED` uses per turn,
  each name cut to `NAME_CAP` and each input to `INPUT_CAP` characters (collapsed onto one line),
  the rest counted in ``unlisted``; so one turn's note is never longer than `NOTE_CAP` characters, and
  `_context.persisted_step_cap` counts it, because it persists per step and is replayed forever. The
  bound is applied once, at arrival, so the live turn and the persisted turn are the same record —
  a fixed point, which re-saving can never grind smaller. It is measured on the ``context
  attribution`` line as its own section (``history_server_tools``).
- **It is shown by the engine, for every vendor, and written by the adapter.** No vendor branch reads
  it: an adapter that cannot see its vendor's server-tool activity simply reports none, and the
  engine renders whatever was reported. The note is spliced into the *request* (copy-on-write), never
  into the history, so compaction's cut rules, the recovery's transcript walks and the persisted
  shape are untouched; it sits after the turn's tool results, so a call is never separated from its
  result, and it is composed from persisted data alone, so it is byte-stable across wakes and costs
  the prefix cache nothing.

A truncated input ends in ``…``. That is not one of `_elision`'s markers: it stands in for the tail
of a URL, a query or a line of code that was never conversation the agent could forward, and a model
that copied it back would be copying an ellipsis, not presenting an excerpt as whole.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from basecradle_harness._messages import Message, ServerToolRecord, ServerToolUse

#: The most uses one turn's record lists by name; any beyond are counted, not listed.
MAX_LISTED = 4

#: The longest tool name kept, in characters. Vendor tool names are short; the cap only exists so
#: the bound is a property of this module and not of a vendor's naming.
NAME_CAP = 40

#: The longest input kept, in characters (the URL, the query, the code), after whitespace is
#: collapsed onto one line. Enough for an ordinary URL or query whole; code is an opening excerpt.
#:
#: These three caps are sized by the compaction threshold's proof, not by taste: `NOTE_CAP` is a
#: term of `_context.persisted_step_cap`, and the 128 K context floor must still clear one turn of
#: `DEFAULT_MAX_STEPS` worst-case steps (`_context` → the threshold inequality). Raise one and check
#: `test_context`'s floor guarantee before anything else.
INPUT_CAP = 160

#: The largest ``unlisted`` count rendered as a number; a larger one renders as ``999+``. A count of
#: vendor calls in one turn will never reach it — it exists so `NOTE_CAP` is an exact bound.
UNLISTED_CAP = 999

#: The note's opening — the sentence that says what the record is and why it exists.
HEADER = (
    "Server-side tools ran during your turn above. The provider ran them inside that model call, so "
    "their results are not kept in this conversation; this record is the evidence that they ran:"
)

_ELLIPSIS = "…"
_WHITESPACE = re.compile(r"\s+")

#: The keys an adapter's input mapping is summarized by, in order of preference: what a reader most
#: needs to know a fetch, a search or a run happened on.
_INPUT_KEYS = ("url", "query", "queries", "pattern", "code", "command", "input")


def _line(use: ServerToolUse) -> str:
    return f"\n- {use.name}: {use.input}" if use.input else f"\n- {use.name}"


def _more(unlisted: int) -> str:
    count = str(unlisted) if unlisted <= UNLISTED_CAP else f"{UNLISTED_CAP}+"
    return f"\n- and {count} more server-side tool use(s), not listed."


#: The longest note `render` can produce from a bounded record, in characters as the model reads
#: them: the header, `MAX_LISTED` lines at their caps, and the widest "more" line. Derived from the
#: same functions the note is written with, so the bound cannot drift from the text.
NOTE_CAP = (
    len(HEADER)
    + MAX_LISTED * len(_line(ServerToolUse("n" * NAME_CAP, "i" * INPUT_CAP)))
    + len(_more(UNLISTED_CAP + 1))
)


def summarize(value: Any) -> str:
    """One line saying what a server tool was given — for an adapter building a `ServerToolUse`.

    A string is taken as it is. A mapping is read by the first of `_INPUT_KEYS` it carries (a list
    under one is joined with ``, ``); a mapping with none of them, or anything else, is its compact
    JSON. ``None`` or an empty value is ``""``: a use with no input still names the tool. The result
    is bounded later, by `bound`; this only decides *what* to say.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in _INPUT_KEYS:
            found = value.get(key)
            if isinstance(found, str) and found:
                return found
            if isinstance(found, Sequence) and not isinstance(found, (str, bytes)) and found:
                return ", ".join(str(item) for item in found)
        if not value:
            return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def record(uses: Iterable[ServerToolUse]) -> ServerToolRecord | None:
    """A bounded record of `uses`, or ``None`` when there were none — the adapters' constructor."""
    return bound(ServerToolRecord(uses=list(uses)))


def bound(found: ServerToolRecord | None) -> ServerToolRecord | None:
    """`found`, bounded: at most `MAX_LISTED` uses, each name and input capped, the rest counted.

    Idempotent — a bounded record is returned equal to itself, so applying it at arrival and again
    to a record read back from disk never changes what the model reads. A record with nothing in it
    is ``None``: an empty note is no evidence of anything.
    """
    if found is None:
        return None
    uses = [_capped(use) for use in found.uses]
    unlisted = max(0, found.unlisted) + max(0, len(uses) - MAX_LISTED)
    uses = uses[:MAX_LISTED]
    if not uses and not unlisted:
        return None
    return ServerToolRecord(uses=uses, unlisted=unlisted)


def _capped(use: ServerToolUse) -> ServerToolUse:
    name = _cut(_one_line(use.name), NAME_CAP) or "server_tool"
    return ServerToolUse(name=name, input=_cut(_one_line(use.input), INPUT_CAP))


def _one_line(text: str) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()


def _cut(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[: cap - len(_ELLIPSIS)] + _ELLIPSIS


def render(found: ServerToolRecord) -> str:
    """The system note that shows a turn's record to the model — never longer than `NOTE_CAP`."""
    found = bound(found) or ServerToolRecord()
    note = HEADER + "".join(_line(use) for use in found.uses)
    return note + _more(found.unlisted) if found.unlisted else note


def chars(message: Message) -> int:
    """What a turn's record costs the model on replay, in characters — ``0`` for a turn without one.

    The note is its own system turn on the request (`with_records`), so this is its whole length, in
    the unit `_context.message_chars` measures everything else in.
    """
    found = bound(message.server_tools) if message.role == "assistant" else None
    return len(render(found)) if found is not None else 0


def with_records(messages: list[Message]) -> list[Message]:
    """`messages` with each assistant turn's record shown as a system note after the turn.

    The note goes after the turn's own tool results — never between a call and its result, which
    some vendors refuse — and before whatever follows them. The turns themselves are the same
    objects, never copies: an adapter that keeps per-reply state finds its replies by identity. A
    list with no record in it is returned as it is.
    """
    if not any(chars(message) for message in messages):
        return messages
    shown: list[Message] = []
    pending: str | None = None
    for message in messages:
        if pending is not None and message.role != "tool":
            shown.append(Message.system(pending))
            pending = None
        shown.append(message)
        if message.role == "assistant" and chars(message):
            pending = render(message.server_tools)  # type: ignore[arg-type]
    if pending is not None:
        shown.append(Message.system(pending))
    return shown
