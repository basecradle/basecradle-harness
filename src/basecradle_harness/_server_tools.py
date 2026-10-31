"""The record of server-side tool activity: what ran inside a model call, kept and shown (#727).

A **server-side** tool — Anthropic's ``web_fetch``, OpenAI's ``web_search_call``, xAI's agent tools,
Gemini's ``code_execution`` and ``url_context`` — runs inside the vendor's model call. Its blocks are
opaque vendor state, kept by an adapter for one wake at most and never persisted, so the transcript a
later wake replays holds the turn's text and its *client* calls and nothing else. The defect that
opened this module was a peer publicly retracting a true statement: a wake fetched a page and posted
its title, and the next wake, seeing a post with no fetch before it, read its own claim as a
fabrication and corrected it. The harness had hidden the evidence of its own action from it.

So each adapter that can see its vendor's server-tool activity reports it on the reply as a
`ServerToolRecord` (one `ServerToolUse` per use: the tool's name, one line of what it was given,
and the vendor's own reason when it reported the use as failed — a fetch that never read the page
must not stand as evidence that it did, which would invert the defect this module fixes),
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
  each name cut to `NAME_CAP`, each failure to `FAILURE_CAP` and each *quoted* input to `INPUT_CAP`
  characters (collapsed onto one line), the rest counted in ``unlisted``; so one turn's note is never
  longer than `NOTE_CAP` characters, and `_context.persisted_step_cap` counts it, because it persists
  per step and is replayed forever — once more for the step-capped reserve summary, which withholds
  function tools but cannot stop a server built-in from running inside its call.
- **An input is quoted.** A URL or a query may carry text a peer chose, and the note rides a system
  turn; JSON-quoting each input (control characters removed first) makes it unmistakably data and
  means a newline in it can never open a line of the note's own. The
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

#: The longest failure reason kept, in characters — the vendor's own word for why a use failed
#: (``url_not_accessible``, ``URL_RETRIEVAL_STATUS_ERROR``), or ``failed`` when it gives none.
FAILURE_CAP = 40

#: The largest ``unlisted`` count rendered as a number; a larger one renders as ``999+``. A count of
#: vendor calls in one turn will never reach it — it exists so `NOTE_CAP` is an exact bound.
UNLISTED_CAP = 999

#: The note's opening — what the record is, and why it exists. It says *called*, never *ran*: in a
#: group that mixes a server tool with a client call the server tool runs only once the client's
#: results go back, and a call the vendor reported as failed is listed with its failure, not as work
#: that succeeded. It says the results are not kept *beyond the wake*: inside the wake that made the
#: calls an adapter may still be sending the vendor's own blocks back (Anthropic does). And it says
#: what the record alone cannot: the text written after a server tool was written with its result
#: in view. Without that sentence a careful model, asked to check its claims a wake later, conceded
#: the fetch ran and then disowned the title it had read from it, live on 2026-10-31.
HEADER = (
    "Server-side tools were called during your turn above. The provider runs them itself, inside "
    "its own model calls, and whatever you wrote after one was written with its result in view; "
    "this conversation does not keep the results themselves beyond the wake they ran in. This list "
    "is the lasting record of those calls (each input quoted, a failure the provider reported "
    "marked):"
)

_ELLIPSIS = "…"
_WHITESPACE = re.compile(r"\s+")
#: Control characters other than whitespace — removed, so a quoted input escapes nothing but ``"``
#: and ``\`` and its quoted length is predictable.
_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")

#: The keys an adapter's input mapping is summarized by, in order of preference: what a reader most
#: needs to know a fetch, a search or a run happened on.
_INPUT_KEYS = ("url", "query", "queries", "pattern", "code", "command", "input")


def _line(use: ServerToolUse) -> str:
    line = f"\n- {use.name}: {_quoted(use.input)}" if use.input else f"\n- {use.name}"
    return f"{line} (failed: {use.failed})" if use.failed else line


def _quoted(text: str) -> str:
    """`text` as a JSON string — unmistakably data, never a sentence of the note's own."""
    return json.dumps(text, ensure_ascii=False)


def _more(unlisted: int) -> str:
    count = str(unlisted) if unlisted <= UNLISTED_CAP else f"{UNLISTED_CAP}+"
    return f"\n- and {count} more server-side tool use(s), not listed."


#: The longest note `render` can produce from a bounded record, in characters as the model reads
#: them: the header, `MAX_LISTED` lines at their caps (the input's cap is on its *quoted* form), and
#: the widest "more" line. Derived from the same functions the note is written with, so the bound
#: cannot drift from the text.
NOTE_CAP = (
    len(HEADER)
    + MAX_LISTED
    * (len(_line(ServerToolUse("n" * NAME_CAP, "", "f" * FAILURE_CAP))) + 2 + INPUT_CAP + 2)
    + len(_more(UNLISTED_CAP + 1))
)


def summarize(value: Any) -> str:
    """One line saying what a server tool was given — for an adapter building a `ServerToolUse`.

    A string is taken as it is. A mapping is read by the first of `_INPUT_KEYS` it carries (a list
    under one is joined with ``, ``), looking one level into a nested mapping when it carries none
    itself — live, Anthropic's ``web_fetch`` arrived once as ``{"params": {"url": …}}``. A mapping
    with none of them, or anything else, is its compact JSON. ``None`` or an empty value is ``""``:
    a use with no input still names the tool. The result is bounded later, by `bound`; this only
    decides *what* to say.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        found = _keyed(value)
        if found is None:
            nested = (_keyed(v) for v in value.values() if isinstance(v, Mapping))
            found = next((f for f in nested if f is not None), None)
        if found is not None:
            return found
        if not value:
            return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _keyed(value: Mapping[str, Any]) -> str | None:
    """The first of `_INPUT_KEYS` `value` carries, as one string — ``None`` when it carries none."""
    for key in _INPUT_KEYS:
        found = value.get(key)
        if isinstance(found, str) and found:
            return found
        if isinstance(found, Sequence) and not isinstance(found, (str, bytes)) and found:
            return ", ".join(str(item) for item in found)
    return None


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
    failed = _cut(_one_line(use.failed), FAILURE_CAP)
    return ServerToolUse(name=name, input=_cut_quoted(_one_line(use.input)), failed=failed)


def _one_line(text: str) -> str:
    return _WHITESPACE.sub(" ", _CONTROL.sub("", text or "")).strip()


def _cut_quoted(text: str) -> str:
    """`text`, cut so its *quoted* form (less the two quotes) is at most `INPUT_CAP` characters.

    Measured on what the model reads: a ``"`` or ``\\`` costs two once quoted. Already within the
    cap, it is returned unchanged, which is what keeps `bound` a fixed point.
    """
    if len(_quoted(text)) - 2 <= INPUT_CAP:
        return text
    cut = text[: INPUT_CAP - len(_ELLIPSIS)]
    while len(_quoted(cut + _ELLIPSIS)) - 2 > INPUT_CAP:
        cut = cut[:-1]
    return cut + _ELLIPSIS


def _cut(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[: cap - len(_ELLIPSIS)] + _ELLIPSIS


def mark_failed(message: Message, index: int, reason: str) -> bool:
    """Mark the `index`-th use on `message`'s record as failed, bounded — ``True`` if it was listed.

    For a failure the vendor reports *after* the reply that recorded the use: on Anthropic a mixed
    server-and-client group runs its server tool only on the next call, so the error result arrives
    in a later reply than the ``server_tool_use`` (issue #737). A use past `MAX_LISTED` is counted,
    not listed, and has no line to mark.
    """
    found = message.server_tools
    if found is None or not 0 <= index < len(found.uses):
        return False
    use = found.uses[index]
    failed = _cut(_one_line(reason), FAILURE_CAP) or "failed"
    found.uses[index] = ServerToolUse(name=use.name, input=use.input, failed=failed)
    return True


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


def with_records(
    messages: list[Message], shown: dict[int, tuple[Message, str]] | None = None
) -> list[Message]:
    """`messages` with each assistant turn's record shown as a system note after the turn.

    The note goes after the turn's own tool results — never between a call and its result, which
    some vendors refuse — and before whatever follows them. The turns themselves are the same
    objects, never copies: an adapter that keeps per-reply state finds its replies by identity. A
    list with no record in it is returned as it is.

    `shown` remembers, by turn identity, the note each turn was first shown with, and that note is
    sent again for as long as the memo lives (the engine keeps it for its wake). A record can
    change after it was first shown — a failure Anthropic reports in a later reply is marked on it
    (issue #737) — and re-rendering would change bytes already sent mid-wake: every later reply's
    preserved thinking would be stripped (its prefix changed) and the cache tail rewritten, to tell
    the model what the raw result in front of it already says. The changed record is persisted at
    once and shown from the next wake. Entries for turns no longer in `messages` are dropped.
    """
    if not any(chars(message) for message in messages):
        return messages
    result: list[Message] = []
    pending: str | None = None
    present: set[int] = set()
    for message in messages:
        if pending is not None and message.role != "tool":
            result.append(Message.system(pending))
            pending = None
        result.append(message)
        if message.role == "assistant" and chars(message):
            pending = _note_for(message, shown)
            present.add(id(message))
    if pending is not None:
        result.append(Message.system(pending))
    if shown is not None:
        for key in [key for key in shown if key not in present]:
            del shown[key]
    return result


def _note_for(message: Message, shown: dict[int, tuple[Message, str]] | None) -> str:
    """The note `message` was first shown with, if `shown` holds it, else rendered (and held)."""
    held = shown.get(id(message)) if shown is not None else None
    if held is not None and held[0] is message:
        return held[1]
    note = render(message.server_tools)  # type: ignore[arg-type]
    if shown is not None:
        shown[id(message)] = (message, note)
    return note
