"""A `Session`: one conversation thread on one input channel.

The unified-identity model, in one small file. An agent is a single
identity-and-memory locus, but it is addressed over many channels — a GitHub PR
thread, a BaseCradle timeline, any future input the router learns to forward.
Each channel is a *different conversation*, not one merged transcript; yet all of
them must share what the agent *knows* and *is*. (This is law: the BaseCradle
constitution, "Sovereignty and Governance" → identity is *unified* — "what
converges is memory and charter, not conversation.")

A `Session` is that one conversation: its own `history`, keyed by a `source`
string the caller chooses (`"github:pr-123"`, `"timeline:<uuid>"`, `"default"`).
What it does *not* own is the agent's brain, hands, or memory — those live on the
shared `Engine` (provider + tool registry, including the memory tool) it runs
against. So two sessions of the same agent keep separate transcripts but write to
and read from the *same* durable memory and start from the *same* charter. That
is "channels share memory, not conversation."

If given a `path`, a session persists its transcript there on every turn and
loads it on construction — so a past session's reasoning is readable after a
restart, the durable half of cross-session answerability. With no `path`, a
session is in-memory only (the default; transcripts of *live* sessions are still
readable from the one running instance via `Harness.transcript`).

**What persists is bounded, on purpose (issue #275).** The whole transcript is
replayed to the model on every turn, so anything written into it is paid for again
on every future turn, forever. Two disciplines keep that bill honest, and both live
here:

- **Ephemeral context never persists.** A caller may hand `send` a `brief` — standing
  context recomposed fresh for *this* call (the wake's operating brief: current time,
  step budget, live dashboard). It is spliced into the message list the provider sees
  and is *never* appended to `history`. Persisting it would store a stale copy per
  wake — dozens of obsolete "current" times and spent step budgets — that the model
  then reads as context and pays for on every later turn.
- **Tool results are capped.** The model sees a tool's full output on the turn it ran;
  what *persists* is head + tail around an elision marker naming the original size
  (see `TOOL_RESULT_CAP`). This is the same cost discipline the engine already applies
  to a viewed image — seen once, never re-billed — extended to the text that a single
  mailbox listing or wide file read would otherwise tax every future wake with.

**Position is load-bearing.** The provider's prefix cache only pays out on a
byte-stable prefix, so volatile content goes at the *tail*: the frozen history first,
then the brief, then the newest user turn. Moving the brief to the head of the list
("system prompts go first") would change the prefix on every request and silently
destroy caching — an invariant stated in this repo's CLAUDE.md → Context Discipline.
"""

from __future__ import annotations

import json
from pathlib import Path

from basecradle_harness._engine import Engine
from basecradle_harness._messages import ImageContent, Message

#: The most characters of one tool result that persist into the transcript. Above it, the
#: result is elided to head + tail around a marker naming the original size (`_elide`). The
#: model still saw the *whole* result on the turn the tool ran — this bounds only what every
#: *future* turn re-reads and re-pays for. 4 KB is what the live containment prune used on
#: @glm-5.2's transcripts (issue #275): comfortably more than a normal tool answer, far below
#: the 142 KB mailbox dumps that drove one agent's context to 754 K input tokens per call.
TOOL_RESULT_CAP = 4096

#: How the elided result is split around the marker: enough head to keep the shape and the
#: first rows of a listing, and a short tail so a result whose payload lands at the end (a
#: summary line, a closing error) is not lost. Their sum is well under `TOOL_RESULT_CAP`, so
#: eliding always shrinks — a result between the two sizes is left whole rather than "elided"
#: into something longer than it started.
_ELISION_HEAD = 2048
_ELISION_TAIL = 512


class Session:
    """One channel's conversation, run against the agent's shared engine.

    Args:
        source: The channel/thread key this conversation belongs to. Free-form;
            the caller's convention (e.g. `"github:pr-123"`). It is the identity
            of the *conversation*, never of the agent.
        engine: The agent's shared loop — provider plus tool registry. Shared
            across every session of the agent, which is how separate transcripts
            still converge on one memory.
        system_prompt: The agent's charter, seeded as the first turn of a *new*
            session. A session reloaded from disk keeps its stored charter and is
            not reseeded.
        path: Where to persist this session's transcript. `None` (the default)
            keeps it in memory only. A path enables across-restart durability;
            its parent directory is created on first write.
    """

    def __init__(
        self,
        source: str,
        engine: Engine,
        *,
        system_prompt: str | None = None,
        path: str | Path | None = None,
    ) -> None:
        self.source = source
        self.engine = engine
        self.path = Path(path) if path is not None else None
        self.history: list[Message] = self._load()
        if not self.history and system_prompt:
            self.history.append(Message.system(system_prompt))

    def send(
        self,
        text: str,
        *,
        images: list[ImageContent] | None = None,
        brief: str | None = None,
    ) -> str:
        """Send one user message, run the loop to a text reply, persist the turn.

        The exchange the model produced — its assistant turns, every tool result, the
        engine's step ledger — is appended to `history` (and saved if this session has a
        path), so memory of *this* conversation carries into its next `send`, while the
        agent's durable memory, shared through the engine, carries across every
        conversation.

        `images` places pictures *in front of* the model on this turn (vision), so a
        peer's posted file is perceived directly rather than only described — the asset
        wake uses this. Once the model has answered, the pixels are **evicted** from the
        turn (the text stays as a breadcrumb): the same cost discipline the engine
        applies to a viewed image, so a presented picture is never re-sent (or re-billed)
        on a later turn, nor persisted as base64 into the transcript on disk.

        `brief` is **ephemeral** standing context for this call only — the wake's operating
        brief, recomposed fresh each wake. It is spliced in immediately before the newest
        user turn (stable history first, volatile brief last, so the provider's prefix cache
        stays hot) and never enters `history`: it is a snapshot of a *moment*, and a persisted
        copy is a stale duplicate the model would re-read — and re-pay for — on every later
        turn (issue #275).
        """
        turn = Message(role="user", content=text, images=list(images) if images else [])
        self.history.append(turn)
        # What the provider sees: the durable transcript, then the ephemeral brief, then this
        # turn. The engine appends its work onto this list, so `adopted` marks where that work
        # begins — everything past it is the conversation and belongs in `history`; the brief,
        # sitting before it, never is.
        convo = list(self.history)
        if brief:
            convo.insert(len(convo) - 1, Message.system(brief))
        adopted = len(convo)
        try:
            reply = self.engine.run(convo)
        except Exception as exc:
            # A failed run — most pointedly the engine's reserve-summary fallback erroring —
            # still did work worth keeping (the counter notes, every assistant tool-call turn,
            # every tool result). Mark the tail so a later reader knows the turn failed; the
            # `finally` then adopts and persists that partial transcript rather than discarding
            # the whole ledger the way a save-only-on-success path did (issue #244).
            convo.append(Message.system(_failure_note(exc)))
            raise
        finally:
            # However the loop ends — success or failure — adopt the engine's work (never the
            # brief, which sits before `adopted`) and bound what persists. Evict the pixels, so
            # a presented picture is never re-sent (or persisted as base64) on a later turn; the
            # text caption stays as the breadcrumb. Cap the tool results for the same reason: the
            # model has read them, and what persists must not tax every future turn.
            self.history.extend(convo[adopted:])
            turn.images = []
            _cap_tool_results(self.history)
            self._save()
        return reply.content or ""

    def note(self, text: str) -> None:
        """Record an out-of-band system note in the transcript — no model call.

        For a fact the conversation should carry that the model did not produce: most
        pointedly, a reply that could *not* be delivered because the platform refused the
        post (a locked timeline). Recording it keeps the agent's own transcript honest and
        gives the next turn the context, while costing nothing — no provider request, no
        tokens. Persisted like any turn when the session has a path.
        """
        self.history.append(Message.system(text))
        self._save()

    # --- transcript persistence: load on construct, save on every turn --------

    def _load(self) -> list[Message]:
        if self.path is None or not self.path.exists():
            return []
        history = [Message.from_dict(d) for d in json.loads(self.path.read_text())]
        # Cap on the way in as well as on the way out, so a transcript written *before* the
        # cap existed heals on the next wake instead of taxing the agent forever. Idempotent:
        # an already-capped result is under the cap and passes through untouched.
        _cap_tool_results(history)
        return history

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([m.to_dict() for m in self.history], indent=2))


def _cap_tool_results(history: list[Message]) -> None:
    """Elide any over-long tool result in place, so the transcript stays bounded (issue #275).

    Only `tool` turns are touched, and only their `content` — the `tool_call_id` that pairs a
    result to the call it answers is untouched, and no message is ever dropped. A dropped tool
    turn would leave a dangling assistant tool-call and make the next wake's transcript
    malformed; an edited one is always well-formed.

    The model already read the full result on the turn the tool ran. This governs only what
    every *future* turn re-reads: one mailbox listing or wide file read is otherwise a
    permanent tax on the life of the timeline.
    """
    for turn in history:
        if turn.role == "tool" and turn.content:
            turn.content = _elide(turn.content)


def _elide(text: str) -> str:
    """`text` unchanged, or its head and tail around a marker naming what was cut.

    The marker states the original size, so the model (and a human reading the transcript)
    knows it is looking at an excerpt and how much is missing — a silent truncation would let
    the model reason on a partial listing as though it were whole. It names the remedy
    *conditionally* ("if you need it in full"), never as an instruction: a marker that reads
    like a directive would invite the model to re-run tools it has no present use for.
    """
    if len(text) <= TOOL_RESULT_CAP:
        return text
    cut = len(text) - _ELISION_HEAD - _ELISION_TAIL
    marker = (
        f"\n\n[... {cut} chars elided of {len(text)} — this is an archived excerpt; the full "
        f"result was shown when the tool ran. Re-run it if you need it in full. ...]\n\n"
    )
    return text[:_ELISION_HEAD] + marker + text[-_ELISION_TAIL:]


def _failure_note(exc: BaseException) -> str:
    """The trailing marker for a partial transcript persisted after `engine.run` raised.

    Names the exception type and message so a later reader — or the operator diagnosing a
    step-cap — can tell the turn failed and why, rather than the transcript ending on a
    dangling tool call with no explanation (issue #244).
    """
    return f"[turn failed: {type(exc).__name__} — {exc}]"
