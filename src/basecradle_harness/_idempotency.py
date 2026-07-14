"""Deterministic `Idempotency-Key`s for the four platform creates (issue #297).

A wake killed by a signal can die *between* the platform POST and the write that would have
recorded it. The recovery then has to re-issue that create without knowing whether it landed —
and the only thing that makes that safe is a key the platform can recognize: replay the same key
and it returns the original record instead of making a second one (`basecradle` 0.6.0;
keys are scoped per timeline + author, and *the same key with a different body returns the first
record*, which is exactly the property the recovery needs, because a resumed model is not
deterministic and will not rewrite the same body).

So the key must be **derivable twice** — once by the wake that died, once by the wake that
recovers it — from nothing but the timeline, the item, and the transcript. Three ingredients:

    key = uuid5(NAMESPACE, f"{timeline}:{anchor}:{kind}:{ordinal}")

- **`timeline`** — the timeline the wake fired on. Not the *target* of the post (a cross-timeline
  reply targets another timeline): this is one ingredient of a name, not an address.
- **`anchor`** — the item the turn is answering. For a batched message turn that is the **oldest**
  message in the batch, which is the one thing about a batch that a recovery cannot change: a
  rebuild or a re-drive only ever *appends* newer messages to it.
- **`kind`** — which of the four creates. Scoping by kind is what lets a re-run that posts a
  message *and* uploads an asset dedupe both, in whatever order it happens to do them.
- **`ordinal`** — the nth create of that kind in this turn.

**The ordinal is read off the transcript, never off a counter, and that is the whole trick.**

    ordinal = 1 + (create calls of this kind in this turn that already have a result)

At the instant a tool call executes, the create-shaped calls with results are exactly the ones
*earlier in call order* — the engine appends each result before it dispatches the next call — so
counting answered calls at runtime and counting positions at replay time are the same number
(`completed`, `ordinal_of`). A **counter** would be a second source of truth, and it would drift
from the first one the moment a create-shaped call was recorded in the transcript without ever
reaching the tool's create branch (the model passes an unknown kwarg, `run(**arguments)` raises
`TypeError`, the engine feeds the error back as the result). The counter would say 1; the
transcript would say 2; the recovery would re-issue under a key the dead wake never used, and the
platform — never having seen it — would post the message a second time. There is no counter.

This is only possible because the transcript is now persisted *incrementally* (issue #297, piece
A): at mint time the assistant turn carrying the call is already in the turn's work.

**A create issued outside one of these four tool calls carries no key** — a generated image's
upload, the code bridge's harvest of a run's output files. That is deliberate on both counts: they
are never re-issued (a resume refuses to re-run a non-idempotent effect, because no platform key
can un-spend money at fal.ai), and they consume no ordinal, so runtime and replay keep agreeing.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from basecradle_harness._messages import Message, ToolCall

if TYPE_CHECKING:  # pragma: no cover - a type-only edge, and it keeps `_session` out of the cycle
    from basecradle_harness._session import Session

#: The uuid5 namespace every harness key is minted under. Arbitrary, fixed, and **contractual**:
#: changing it silently re-mints every key, so a recovery would stop matching the dead wake's
#: post and would duplicate it. It is a constant, not a knob.
NAMESPACE = uuid.UUID("9f2b0c4e-6a1d-4f7e-9c3b-8d5a2e1f0b74")

#: The four content creates the platform keys, by the tool call that issues them. The `action`
#: shape is what makes a create *recognizable in the transcript* — which is the whole reason the
#: ordinal can be reconstructed after a crash without re-running anything.
MESSAGE = "message"
ASSET = "asset"
TASK = "task"
WEBHOOK_ENDPOINT = "webhook_endpoint"

CREATE_CALLS: dict[tuple[str, str], str] = {
    ("messages", "create"): MESSAGE,
    ("assets", "create"): ASSET,
    ("tasks", "create"): TASK,
    ("webhook_endpoints", "create"): WEBHOOK_ENDPOINT,
}


def create_kind(call: ToolCall) -> str | None:
    """Which of the four creates this tool call issues, or ``None`` if it is not one.

    Read from the call's *recorded* name and arguments, so a transcript loaded back after a crash
    answers the question exactly as the live call did.
    """
    action = call.arguments.get("action")
    if not isinstance(action, str):
        return None
    return CREATE_CALLS.get((call.name, action))


def key(*, timeline: str, anchor: str, kind: str, ordinal: int) -> str:
    """The `Idempotency-Key` for the nth create of `kind` in the turn answering `anchor`."""
    return str(uuid.uuid5(NAMESPACE, f"{timeline}:{anchor}:{kind}:{ordinal}"))


@dataclass(frozen=True)
class Create:
    """One platform create a turn issued: the call, its kind, its ordinal, and its result (if any).

    `result` is the `tool` message answering the call — ``None`` only for a call still in flight
    (the one currently executing). A killed wake's un-recorded call comes back with the healed
    "outcome unknown" placeholder as its result, which is how the recovery finds it.
    """

    call: ToolCall
    kind: str
    ordinal: int
    result: Message | None


def creates(work: list[Message]) -> list[Create]:
    """Every platform create in this turn's work, in call order, with its ordinal and its result.

    **This is the one walk, and everything reads its ordinal from here** — the live mint while the
    turn runs, and the recovery re-issuing an interrupted call after the turn dies. The two numbers
    must be equal, and the only way to be sure of that is for there to be one function producing
    them. Two functions that obviously agree is how they stop agreeing.

    **A call is paired with a result from its own assistant turn's run, never by a global id
    lookup.** A tool-call id is the provider's own string and nothing normalizes it: a model that
    numbers its calls per response (`call_0`, `call_1` — what an OpenRouter-fronted model emits)
    reuses the same ids on every turn. Matching them across the whole transcript would pair a call
    with the *previous* turn's result — declaring an interrupted call answered (so it is never
    healed, and the provider 400s on the transcript forever) and counting the currently-executing
    call as already-answered (so the live ordinal runs one ahead of the recovery's).
    """
    found: list[Create] = []
    counts: dict[str, int] = {}
    for index, turn in enumerate(work):
        if turn.role != "assistant" or not turn.tool_calls:
            continue
        run = _run(work, index)
        for call in turn.tool_calls:
            kind = create_kind(call)
            if kind is None:
                continue
            counts[kind] = counts.get(kind, 0) + 1
            result = next((r for r in run if r.role == "tool" and r.tool_call_id == call.id), None)
            found.append(Create(call=call, kind=kind, ordinal=counts[kind], result=result))
    return found


def _run(work: list[Message], assistant: int) -> list[Message]:
    """The turns answering `work[assistant]`'s calls — everything up to the next assistant turn.

    A step note or an injected image turn may sit among them; neither ends the run.
    """
    end = assistant + 1
    while end < len(work) and work[end].role != "assistant":
        end += 1
    return work[assistant + 1 : end]


def completed(work: list[Message], kind: str) -> int:
    """Create calls of `kind` that already have a result — the ordinal of the one about to run.

    "Already has a result" and "earlier in call order" are the same set at the moment a call runs:
    the engine appends each result before it dispatches the next call. It is the *former* that is
    checkable without knowing which call is currently executing — which the tool, reasonably, does
    not.
    """
    return sum(1 for c in creates(work) if c.kind == kind and c.result is not None)


def interrupted(work: list[Message], marker: str) -> list[Create]:
    """Every platform create this turn issued whose result is the healed "outcome unknown" marker.

    The recovery's work list: these — and only these — may be safely re-executed, because the key
    the dead wake minted for each is reproducible from its ordinal here.
    """
    return [c for c in creates(work) if c.result is not None and c.result.content == marker]


class IdempotencyKeys:
    """The live minter a create tool reaches through its `PlatformContext` (issue #297).

    One object, bound into the platform tools once (like the `SpeechLedger`), with its contents
    cycled per turn: the hosting agent calls `begin` before each model call, naming the item the
    turn is answering. Outside a turn — and in the library and poll paths, which bind no minter at
    all — `mint` returns ``None``, the SDK sends no `Idempotency-Key` header, and every create
    behaves exactly as it did before this existed.
    """

    def __init__(self) -> None:
        self._session: Session | None = None
        self._timeline: str | None = None
        self._anchor: str | None = None
        #: Set only while a resume re-issues one specific interrupted call: the `(kind, ordinal)`
        #: that call was *originally* minted under. Without it the re-issue would take the next
        #: free ordinal, mint a key the platform has never seen, and post the message twice —
        #: the exact duplicate the key exists to prevent.
        self._reissue: tuple[str, int] | None = None

    def begin(self, session: Session, *, timeline: str, anchor: str) -> None:
        """Arm the minter for the turn that is about to run against `anchor`."""
        self._session = session
        self._timeline = timeline
        self._anchor = anchor
        self._reissue = None

    def clear(self) -> None:
        """Disarm. A create outside a turn (there are none today) mints no key."""
        self._session = None
        self._timeline = None
        self._anchor = None
        self._reissue = None

    @contextmanager
    def reissue(self, kind: str, ordinal: int) -> Iterator[None]:
        """Mint `(kind, ordinal)` for the duration — a resume re-running one interrupted call."""
        previous = self._reissue
        self._reissue = (kind, ordinal)
        try:
            yield
        finally:
            self._reissue = previous

    def mint(self, kind: str) -> str | None:
        """The key for the create about to be issued, or ``None`` when no turn is armed."""
        if self._session is None or self._timeline is None or self._anchor is None:
            return None
        if self._reissue is not None and self._reissue[0] == kind:
            ordinal = self._reissue[1]
        else:
            ordinal = completed(self._session.turn_work, kind) + 1
        return key(timeline=self._timeline, anchor=self._anchor, kind=kind, ordinal=ordinal)
