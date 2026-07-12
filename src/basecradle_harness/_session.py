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
- **The conversation itself is bounded.** Capping each turn only slows the growth; it does not
  stop it, and a long-lived agent would still walk into its model's context ceiling — where the
  provider 400s deterministically and every later wake rebuilds the same doomed request. So a
  session given a `compactor` (`basecradle_harness._context`) watches the provider's *own*
  reported usage and, past half the ceiling, replaces everything but a recent window with a
  model-written summary (issue #276). That is the third discipline, and the invariant behind all
  three: **nothing replayed per wake may be unbounded.**

**Position is load-bearing.** The provider's prefix cache only pays out on a
byte-stable prefix, so volatile content goes at the *tail*: the frozen history first,
then the brief, then the newest user turn. Moving the brief to the head of the list
("system prompts go first") would change the prefix on every request and silently
destroy caching — an invariant stated in this repo's CLAUDE.md → Context Discipline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from basecradle_harness._context import Compactor
from basecradle_harness._engine import Engine
from basecradle_harness._exceptions import ProviderContextLengthError
from basecradle_harness._messages import ImageContent, Message

_log = logging.getLogger("basecradle_harness")

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
        compactor: The context budget's rewriter (`basecradle_harness._context`). Given one,
            the session bounds its own transcript: after a settled turn it asks whether the
            provider's own reported usage crossed the compaction threshold, and if so replaces
            everything but a recent window with a model-written summary. `None` (the default)
            leaves the transcript to grow — the pre-#276 behavior, still correct for a
            short-lived or hand-managed session.
    """

    def __init__(
        self,
        source: str,
        engine: Engine,
        *,
        system_prompt: str | None = None,
        path: str | Path | None = None,
        compactor: Compactor | None = None,
    ) -> None:
        self.source = source
        self.engine = engine
        self.path = Path(path) if path is not None else None
        self.compactor = compactor
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

        Two things bound the transcript around this call, and both are the same discipline —
        *nothing replayed per wake may be unbounded* (issue #276). **Before** it: an
        over-length failure (`ProviderContextLengthError` — the transcript has outgrown the
        model's context window) is caught, the transcript is compacted hard, and the turn is
        re-run **once**, so an agent already at the wall self-heals instead of failing
        identically on every wake until a human edits its session file. **After** it: if the
        provider's own reported usage for this turn crossed the compaction threshold, the
        transcript is compacted for the *next* one.
        """
        pixels = list(images) if images else []
        turn = Message(role="user", content=text, images=list(pixels))
        self.history.append(turn)
        mark = len(self.history)  # everything past here is this turn's work — what a retry rewinds
        try:
            reply = self._exchange(turn, brief)
        except ProviderContextLengthError as error:
            if not self._recover_from_overflow(mark, turn, pixels):
                raise  # nothing could be compacted: the same request would fail the same way
            _log.warning("Retrying the turn once against the compacted transcript: %s", error)
            reply = self._exchange(turn, brief)
        self._compact_if_needed()
        return reply.content or ""

    def _exchange(self, turn: Message, brief: str | None) -> Message:
        """One engine run against the current history, adopting and persisting whatever it produced.

        `turn` must be the last message in `history` — the brief is spliced in just ahead of it, so
        the volatile content sits at the tail and the cacheable prefix stays byte-stable.
        """
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
        return reply

    def _recover_from_overflow(self, mark: int, turn: Message, pixels: list[ImageContent]) -> bool:
        """Rewind the failed turn, compact hard, and report whether a retry is worth making.

        Rewinding drops the failed run's residue so `turn` is once more the tail of `history`, which
        is what the retry (and the brief's splice position) require. The pixels are restored with it:
        an image a peer posted must still be *seen* on the retry, not silently dropped by the rescue.

        **A retry is only safe when the failed run ran no tools.** The overflow usually strikes the
        run's *first* call — the transcript was already too long before the model did anything — and
        then the residue is inert (a step-counter note, the failure marker) and re-running the turn
        repeats nothing. But a run can also cross the ceiling *mid-flight*, after tool calls have
        already executed; rewinding there would erase the record that they ran, and the re-run would
        very likely call them again — posting the same message twice, creating the same task twice.
        The `ClaimStore` makes each *item* exactly-once, but nothing makes a *turn* replay-safe. So a
        run that got as far as executing a tool is **not** retried: its work stays in the transcript,
        the error propagates (the wake degrades as it always did), and the compaction below still
        shrinks the transcript so the *next* wake comes in under the ceiling. Self-healing, one wake
        later, and never at the price of a duplicated side effect.

        Returns False when there is no compactor, when the run already fired tools, or when
        compaction could not free anything — the caller then lets the original error propagate.
        """
        if self.compactor is None:
            return False
        ran = sum(1 for message in self.history[mark:] if message.role == "tool")
        if ran:
            # Tools already ran. Keep their record, compact anyway (we *know* the transcript is over
            # the ceiling — the provider just said so, which no reported usage would tell us, since
            # the call that would have reported it never completed), and do not re-run the turn.
            _log.warning(
                "Context overflow after %d tool call(s) had already run: the turn is NOT retried "
                "(a re-run could repeat their side effects). Compacting for the next wake.",
                ran,
            )
            if self.compactor.emergency_compact(self.history):
                self._save()
            return False
        del self.history[mark:]
        turn.images = list(pixels)
        if not self.compactor.emergency_compact(self.history):
            return False
        self._save()
        return True

    def _compact_if_needed(self) -> None:
        """Bound the transcript for the *next* turn, on what the provider said about this one.

        Read the usage **here**, immediately after the run that produced it: one provider instance is
        shared by every session of an agent, so `last_tokens_in` means "the last call anyone made" —
        unambiguous on this single-threaded path, where the last call is always this session's own.

        A compaction failure is never a turn failure: the reply is already in hand and the peer is
        owed it. The worst case is a transcript that stays too long and tries again next turn.
        """
        if self.compactor is None:
            return
        try:
            if self.compactor.maybe_compact(self.history):
                self._save()
        except Exception as exc:  # noqa: BLE001 - a failed compaction degrades; it never breaks a turn
            _log.warning("Context compaction failed; the transcript is unchanged: %s", exc)

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
