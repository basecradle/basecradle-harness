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

**What persists is bounded, on purpose (issues #275, #301).** The whole transcript is
replayed to the model on every turn, so anything written into it is paid for again
on every future turn, forever. Four disciplines keep that bill honest, and every one
of them passes through this file:

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
  mailbox listing or wide file read would otherwise tax every future wake with. The cap
  is *enforced* here but **defined in `_context`**, because the compaction threshold's
  safety proof is computed from it: it is a load-bearing input to that arithmetic, not a
  neighbor of it, and the two must never drift apart in separate files.
- **A tool call's arguments are capped too** (`TOOL_ARGS_CAP`, issue #301) — the half of a call
  nobody bounded. Capping the *result* and writing the *arguments* whole left an `assets create`
  carrying a 200 KB document in the transcript forever, re-sent to the model on every wake for the
  life of the timeline. **With one exception, and it is the whole design:** the recovery re-issues
  an interrupted platform create *from exactly these arguments* (`_wake._reissue_interrupted_creates`),
  so eliding them naively would re-post the peer's message with its body cut out. So an **interrupted
  create keeps its arguments whole** — its healed "outcome unknown" result is the flag — and the cap
  falls on it the moment the resume replaces that marker with a real result. Everything the recovery
  will never replay is bounded from the first save.
- **Both caps are per *step*, not per call** (issue #304). A model may emit several tool calls in one
  assistant turn, and the engine's step budget bounds the model's *calls*, never the tools it
  dispatched — so a per-call cap let a step's persisted growth scale with a fan-out nothing bounds,
  and the compaction proof (`_context.worst_case_turn_tokens`) understated the worst case by that
  factor. A step's results share one `TOOL_RESULT_CAP` and its calls' arguments share one
  `TOOL_ARGS_CAP`, water-filled (`_fill`), so the small ones survive whole and only the fat ones pay.
  A lone call — the ordinary shape — gets the whole budget and is untouched by any of this.
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
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from basecradle_harness._caching import anchor_cacheable_prefix, cache_mode
from basecradle_harness._context import TOOL_ARGS_CAP, TOOL_RESULT_CAP, Compactor
from basecradle_harness._engine import Engine
from basecradle_harness._exceptions import ProviderContextLengthError
from basecradle_harness._idempotency import create_kind
from basecradle_harness._messages import ImageContent, Message, ToolCall

_log = logging.getLogger("basecradle_harness")

#: What `_fill` allocates: a tool result (a `str`), a call's arguments (a `dict`), or one argument's
#: value (anything a model emitted). It needs only to measure an item and to cut one down to a size.
_T = TypeVar("_T")
_Size = Callable[[_T], int]
_Elide = Callable[[_T, int], _T]

#: The result a killed wake's unanswered tool call is given on load (`heal_interrupted_calls`).
#: It is written for the model that will read it, and it says the two things that are true and
#: nothing else: the outcome is unknown, and nobody has re-run it. It does not instruct — under the
#: Unspoken Channel the harness informs and never forces, and an agent that wants to know what
#: happened can simply go and look.
INTERRUPTED = (
    "[Interrupted: this wake was killed while this call was running, before its result could be "
    "recorded. Its outcome is unknown — it may have completed, completed partially, or never "
    "started. It has NOT been re-run, because re-running it could repeat something that already "
    "happened. If it matters here, check for yourself and decide what to do.]"
)

#: How the elided result is split around the marker: enough head to keep the shape and the
#: first rows of a listing, and a short tail so a result whose payload lands at the end (a
#: summary line, a closing error) is not lost. Their sum is well under `TOOL_RESULT_CAP`, so
#: eliding always shrinks — a result between the two sizes is left whole rather than "elided"
#: into something longer than it started.
_ELISION_HEAD = 2048
_ELISION_TAIL = 512

#: How a cut argument is split: the head takes whatever its share of the cap allows (the opening of
#: the document it was posting, the first lines of the body it wrote) and the tail takes a fifth of it
#: — capped here, because a tail is a coda, not half the excerpt — so a value whose point lands at the
#: end is not lost. How much room it gets depends on how many siblings it is sharing the call with,
#: and how many calls its step is sharing the budget with (`_fit`, `_calls_payload`).
_ARG_ELISION_TAIL = 128

#: Below this much room, an "excerpt" is a few words torn out of context — worse than the marker alone,
#: which at least reports honestly that the value is gone. Shared by both caps: a result's share of its
#: step can be squeezed to the same point an argument's can (`_elide`, `_elide_argument`).
_MIN_EXCERPT = 64

#: The longest `action` the stub will carry back. No `CREATE_CALLS` action is longer than a word, so a
#: longer one cannot be a create and dropping it leaves `create_kind` answering ``None`` either way —
#: which is the property that keeps the idempotency ordinal identical across a crash.
_ARG_ACTION_MAX = 128

#: How many times the cap may halve its budget and re-measure before giving up on the fair share and
#: falling back to the stub. Two would do for every shape we can construct; four is slack, and the loop
#: is bounded either way — see `_cap_arguments` for why the fit has to be *measured* at all.
_ARG_FIT_ATTEMPTS = 4


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
        #: This session's private staging token for the atomic save (`_save`). Per *instance*, not
        #: per process: two `Harness` instances over one home hold two `Session`s on the same path
        #: in the same process, and they must not share a temp file.
        self._token = uuid4().hex[:8]
        #: The index in `history` of the user turn of the exchange in progress (or the last one
        #: that ran). With `_convo`/`_adopted` it answers `turn_work` — what *this* turn has
        #: produced — which is what the idempotency minter counts its ordinal off (`_idempotency`).
        #: Recomputed after any compaction, because a compaction rewrites `history` in place and
        #: shorter, and every index into it stops meaning what it meant.
        #:
        #: **It is the `Message` itself, not its index**, and that is the point: a compaction
        #: rewrites `history` in place and shorter, so an index into it silently starts naming a
        #: different turn — and here that would mean counting the wrong turn's creates and minting
        #: a key the platform has never seen, which is a duplicate post. A compaction *moves*
        #: objects; it does not copy them, so identity survives exactly what an index does not.
        self._turn: Message | None = None
        #: The engine's live message list while a turn is in flight, and the offset past which its
        #: work begins. `None` between turns.
        self._convo: list[Message] | None = None
        self._adopted = 0
        #: Where in `history` the engine's live work will be spliced (the end of the turn it is
        #: work *for*), and where the last run's work actually landed. The rescue rewinds by
        #: `_span`, never by `history[-n:]` — on a resume the work is not at the tail.
        self._at = 0
        self._span: tuple[int, int] = (0, 0)
        self.history: list[Message] = self._load()
        if not self.history and system_prompt:
            self.history.append(Message.system(system_prompt))

    def send(
        self,
        text: str,
        *,
        images: list[ImageContent] | None = None,
        brief: str | None = None,
        items: list[str] | None = None,
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

        `items` are the uuids of the platform items this turn is carrying (the messages a wake
        batched into it). They persist *on* the turn, and they are how a later wake knows — after
        a crash, from the transcript alone — that this message was put in front of the model
        (issue #297). Nothing in the send path reads them; the recovery does.
        """
        pixels = list(images) if images else []
        turn = Message(
            role="user", content=text, images=list(pixels), items=list(items) if items else []
        )
        self.history.append(turn)
        self._turn = turn
        # **The peer's message reaches disk before the model is ever called** (issue #297), and a
        # failure here is allowed to take the turn down with it. That is the safe direction and the
        # only one: nothing has run, nothing has posted, and the wake's claim on the message stays
        # in-flight — so the next wake re-drives it cleanly. Proceeding on a failed save would mean
        # running the model with no record that we ever read the message, which is the state this
        # whole issue exists to make impossible.
        self._save()
        try:
            reply = self._exchange(turn, brief)
        except ProviderContextLengthError as error:
            if not self._recover_from_overflow(turn=turn, pixels=pixels):
                raise  # nothing could be compacted: the same request would fail the same way
            _log.warning("Retrying the turn once against the compacted transcript: %s", error)
            reply = self._exchange(turn, brief)
        self._compact_if_needed()
        return reply.content or ""

    def resume(self, turn: Message, *, brief: str | None = None) -> str:
        """Finish the interrupted turn `turn` — **no new user turn** (issue #297).

        A wake killed mid-tool-chain leaves a real turn on disk: the peer's message, the calls the
        model made, and the results of the ones that finished. That turn does not need re-driving
        (which would re-fire its tools) or abandoning (which would drop the peer). It needs
        *continuing* — so this replays the partial transcript to the model and lets it finish what
        it started. Zero tools re-fire, because their results are already there.

        The transcript it runs against is well-formed by construction: `_load` has already given
        every unanswered call a result (`heal_interrupted_calls`), so there is no dangling
        `tool_call_id` for a provider to reject. The caller decides what those results *say* — a
        platform create is re-issued under its original key before we get here; a non-idempotent
        effect keeps the honest "outcome unknown" placeholder (`_wake._resume_orphan`).

        `turn` is the user `Message` itself — handed over by the recovery classifier, which already
        found it. Locating it twice would be two chances to disagree, and an *index* would be worse
        than either: the compaction this resume may itself trigger rewrites `history` shorter, and
        every index into it then names some other turn.
        """
        if _index_of(self.history, turn) is None:
            raise ValueError("the turn to resume is not in this transcript")
        self._turn = turn
        try:
            reply = self._continue(turn, brief)
        except ProviderContextLengthError as error:
            # A compaction that *drops the turn we are finishing* is worse than the overflow: the
            # ordinal would then be counted off an empty work list, the continuation's first create
            # would re-mint ordinal 1 — the key the dead wake already used — and the platform would
            # hand back the original record while the model is told its new message posted. So the
            # rescue must both succeed *and* leave the turn standing; otherwise the error goes up,
            # and the recovery decides what to do with a turn it cannot finish.
            if not self._recover_from_overflow() or _index_of(self.history, turn) is None:
                raise
            _log.warning(
                "Retrying the resumed turn once against the compacted transcript: %s", error
            )
            reply = self._continue(turn, brief)
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
        # The frozen transcript: everything the *previous* wake already sent, byte for byte. It is
        # `history` minus the turn just appended — and it is both the cacheable prefix and the
        # splice point for the brief, which is not a coincidence: they are the same
        # stable/volatile boundary, named once here.
        stable = len(self.history) - 1
        if brief:
            convo.insert(len(convo) - 1, Message.system(brief))
        return self._drive(convo, stable=stable, turn=turn, at=len(self.history))

    def _continue(self, turn: Message, brief: str | None) -> Message:
        """`_exchange` for a resume: replay the transcript **as it stood when the turn was cut off**.

        Two things here are not obvious, and both were bugs before they were code.

        **The model is shown `history[:at]`, not the whole transcript.** `at` is the end of *this*
        turn's work — and there can be conversation *after* it, because a resume can fail and the
        wake goes on to answer newer messages, leaving an older turn unfinished behind a newer one.
        Handing the model that later exchange and asking it to finish the earlier turn is asking it
        to continue a sentence it can see somebody else already finished. It sees the conversation up
        to the moment it was interrupted, which is what "finish the turn you started" means.

        **The continuation is spliced back in at `at`, not appended to the end** (`_drive`). Append it
        and the narration of an *old* turn lands inside a *newer* turn's work — where the recovery
        classifier reads it as the newer turn's own terminal text, commits a message that was never
        answered, and lets the mark sail past it. That is a silent drop, produced by the machinery
        built to prevent silent drops.

        When the resumed turn is the last one (the common case) `at == len(history)` and both of
        these collapse into the obvious thing.
        """
        at = _turn_end(self.history, turn)
        convo = list(self.history[:at])
        if brief:
            convo.append(Message.system(brief))
        return self._drive(convo, stable=at, turn=None, at=at)

    def _drive(
        self, convo: list[Message], *, stable: int, turn: Message | None, at: int
    ) -> Message:
        """Run the engine over `convo`, adopting and persisting whatever it produced.

        Shared by `_exchange` (a fresh turn) and `_continue` (a resumed one); `turn` is the new
        user turn when there is one, and ``None`` when the turn was already in the transcript.

        `at` is **where the work belongs in `history`** — the end of the turn it is work *for*. For a
        fresh turn that is the tail, and the splice below is an append. For a resumed one it is the
        end of that turn's existing work, which may be nowhere near the tail: appending there would
        file an old turn's narration under a newer turn, and the recovery would read it as the newer
        turn's own (see `_continue`).
        """
        # Tell an explicit-cache provider (Anthropic) where the stable prefix ends; a provider that
        # caches automatically — every one that ships — is unaffected and this is a no-op. The
        # anchor is stamped on a copy, so it never reaches `history` (see `_caching`).
        convo = anchor_cacheable_prefix(convo, stable=stable, mode=cache_mode(self.engine.provider))
        adopted = len(convo)
        # Publish the in-flight work, so the progress hook can persist it *in the right place* and
        # the idempotency minter can count its own ordinal off it (`turn_work`). Cleared in the
        # `finally`.
        self._convo, self._adopted, self._at = convo, adopted, at
        failing = False  # is an exception already on its way out through the `finally`?
        try:
            reply = self.engine.run(convo, on_progress=self._progress_hook())
        except Exception as exc:
            # A failed run — most pointedly the engine's reserve-summary fallback erroring —
            # still did work worth keeping (the counter notes, every assistant tool-call turn,
            # every tool result). Mark the tail so a later reader knows the turn failed; the
            # `finally` then adopts and persists that partial transcript rather than discarding
            # the whole ledger the way a save-only-on-success path did (issue #244).
            failing = True
            convo.append(Message.system(_failure_note(exc)))
            raise
        finally:
            # However the loop ends — success or failure — adopt the engine's work (never the
            # brief, which sits before `adopted`) into the turn it belongs to, and bound what
            # persists. Evict the pixels, so a presented picture is never re-sent on a later turn;
            # the text caption stays as the breadcrumb. Cap the tool results for the same reason:
            # the model has read them, and what persists must not tax every future turn.
            work = convo[adopted:]
            # A run that *raised* can end mid-tool-chain, so its work carries a call with no
            # result. Heal it **before** it joins `history`, not only on the next load: the wake
            # may well survive this failure (the recovery catches, and goes on to answer other
            # messages on the same `Session`), and every later provider call in it would 400 on a
            # dangling `tool_call_id` that only a restart would repair.
            heal_interrupted_calls(work)
            self.history[at:at] = work
            #: Where the work this run produced ended up, so the over-length rescue can rewind
            #: exactly it — `history[-n:]` would be the wrong messages on a resume.
            self._span = (at, len(work))
            if turn is not None:
                turn.images = []
            _cap_tool_results(self.history)
            self._convo = None  # the work is adopted; nothing is in flight
            self._persist(masking=failing)
        return reply

    @property
    def turn_work(self) -> list[Message]:
        """Everything the current turn has produced — what is already in `history`, plus what is
        still in flight in the engine's list.

        The one reader is the idempotency minter (`_idempotency`), and the *union* is what it
        needs: on a fresh turn the persisted half is empty and the work is all in flight; on a
        **resumed** turn the dead wake's calls are in `history` and only the continuation is live.
        Counting one half would mint the wrong ordinal, and the wrong ordinal is a duplicate post.

        **Bounded by `turn_work`, the same function the recovery uses**, and that is not a
        convenience — it is the invariant. The minter's ordinal is computed twice, once by the wake
        that dies and once by the wake that recovers it, and the two numbers have to be equal. They
        are equal only if both are counted over the *same* window, so there is exactly one function
        that says where a turn's work ends, and both halves call it. Counting the live half to the
        end of `history` instead would sweep in a *later* turn's creates whenever anything sat past
        the resumed one — two dead wakes in a row, or a `note()` — and the resumed turn's next
        create would mint a key nobody has ever seen.
        """
        if self._turn is None:
            return []
        live = self._convo[self._adopted :] if self._convo is not None else []
        return turn_work(self.history, self._turn) + live

    def rollback(self, mark: int) -> None:
        """Drop everything past `mark` **and rewrite the file** — the staleness guard's undo.

        `_generate_settled` rolls a stale (tool-free, so speechless) build out of the transcript
        and regenerates. That used to be a purely in-memory edit, because nothing had been written
        yet. Incremental persistence changed the premise: the build it is discarding is *already on
        disk*, so an in-memory-only rollback would leave it there — and a wake killed in the window
        before the replacement build persisted would come back to a transcript carrying a turn that
        was deliberately unmade (issue #297).
        """
        del self.history[mark:]
        self._save()

    def excise(self, turn: Message) -> None:
        """Cut `turn` and its work out of the transcript, and rewrite the file (issue #297).

        The recovery's undo for a turn that reached the model and produced *nothing* — the provider
        was down, the box was killed inside the call. Before incremental persistence such a turn
        left no trace and there was nothing to clean up. Now it is on disk, and the message it
        carried is about to be re-driven into a fresh turn — so leaving it would stack a second
        copy of the peer's message into the transcript, and a third on the next failure, forever.

        The caller must have established that the turn issued **no tool calls**; that is what makes
        this safe, and it is not a thing this method can check for itself, because the emptiness it
        relies on is the *recovery's* conclusion, not a property of the list.
        """
        start = _index_of(self.history, turn)
        if start is None:
            return
        end = start + 1 + len(turn_work(self.history, turn))
        del self.history[start:end]
        if self._turn is turn:
            self._turn = None
        self._save()

    def working_on(self, turn: Message) -> None:
        """Name the turn a caller is about to act on, before `resume` gets there.

        The recovery re-issues an interrupted create *before* it hands the transcript back to the
        model, and the idempotency minter counts that create's ordinal off this turn's work — so
        the turn has to be named first. Unnamed, `turn_work` reports no work at all, and every
        create in the turn mints ordinal 1: the key the dead wake used for its *first* post, handed
        to a call that may be its third.
        """
        self._turn = turn

    def persist(self) -> None:
        """Write the transcript as it now stands — for a caller that edited `history` in place.

        The recovery uses it after re-issuing an interrupted platform create, so the real result
        replaces the "outcome unknown" placeholder on disk *before* the model is asked to continue
        the turn. Without it, a wake killed between the re-issue and the resume would come back and
        re-issue again — harmless (the key still dedupes), but it would be luck rather than design.
        """
        self._save()

    def _progress_hook(self) -> Callable[[], None] | None:
        """The engine's per-append persist callback — ``None`` for an in-memory session."""
        return self._persist_progress if self.path is not None else None

    def _persist_progress(self) -> None:
        """Write the turn *in progress*: the frozen transcript with the work so far spliced in.

        The engine's work lives on its own list until the turn ends (`_drive` adopts it in a
        `finally`), so what is on disk mid-turn is the union — and it must be, because the whole
        point is that a wake killed *right now* leaves a transcript that says what it had done.

        It goes in **at `_at`**, not at the end. For a fresh turn those are the same place. For a
        resumed one they are not, and appending would write an old turn's work under a newer turn —
        which the recovery would then read as the newer turn's, and commit a message nobody answered.
        """
        live = self._convo[self._adopted :] if self._convo else []
        at = self._at
        self._save(self.history[:at] + live + self.history[at:])

    def _recover_from_overflow(
        self,
        *,
        turn: Message | None = None,
        pixels: list[ImageContent] | None = None,
    ) -> bool:
        """Rewind the failed run, compact hard, and report whether a retry is worth making.

        Rewinding drops the failed run's residue so `turn` is once more the tail of `history`, which
        is what the retry (and the brief's splice position) require. The pixels are restored with it:
        an image a peer posted must still be *seen* on the retry, not silently dropped by the rescue.

        A **resumed** turn passes no `turn` and no `pixels` (issue #297): the turn it is finishing is
        already in the transcript, below `mark`, so there is nothing to re-append and nothing to
        restore. Everything else is identical — including, most of all, the rule below, which is the
        same rule for the same reason: `mark` bounds *this run's* work, so what it asks is "did the
        attempt I am about to repeat already fire a tool?", and a resume's own continuation is
        judged exactly as a fresh turn's is. The dead wake's tools sit below `mark`, are never
        re-run, and are never counted.

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
        at, count = self._span
        span = self.history[at : at + count]
        # "Did this run already do something?" — and a `tool` turn is not the only evidence of it.
        # A server-side built-in (hosted code execution, web search) resolves *in-call* and leaves
        # no tool turn at all; what it leaves is the code bridge's **injected** turn naming the
        # Assets it harvested. Counting only `tool` turns would rewind a run that had executed the
        # model's Python in a vendor sandbox and uploaded its output, and then re-run it.
        ran = sum(1 for message in span if message.role == "tool" or message.injected)
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
        del self.history[at : at + count]
        if turn is not None:
            turn.images = list(pixels or [])
        if not self.compactor.emergency_compact(self.history):
            return False  # deliberately no save: an `OSError` here would mask the error we re-raise
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
        # A transcript written by a wake that was *killed* can end mid-tool-chain, with a call
        # issued and its result never recorded. Give every such call a result before anyone can
        # send this list to a provider — see `heal_interrupted_calls` for why that is not
        # hygiene but survival.
        healed = heal_interrupted_calls(history)
        if healed:
            _log.warning(
                "Loaded a transcript with %d interrupted tool call(s) — a wake was killed while "
                "they were running. Each is marked with its outcome unknown; none has been re-run.",
                healed,
            )
        return history

    def _persist(self, *, masking: bool) -> None:
        """`_save`, from a `finally` that may already be carrying an exception out.

        An exception raised inside a `finally` **replaces** the one propagating through it — and the
        one propagating through *this* one is load-bearing. `send` catches
        `ProviderContextLengthError` to compact the transcript and retry: the self-heal that keeps
        an agent at its context ceiling from failing identically on every wake, forever (issue
        #276). A save that failed while that exception was in flight would swallow it, the retry
        would never run, and the agent would die of an unrelated `OSError` with its transcript still
        over the wall.

        The atomic write is what makes that reachable, which is why the guard arrives alongside it:
        staging a temp needs the old transcript and the new one on disk at once (~2× the space,
        where a truncating write needed 1×), and `fsync` is exactly where a filesystem reports the
        deferred write errors a buffered `close()` used to swallow. **A near-full box is precisely
        where an over-long transcript turns up.**

        So when the turn is already failing, a failed save is logged and stood down: the transcript
        is lost, which the high-water mark makes survivable (the item is re-driven — issue #285),
        while the masked exception would not be. When nothing is in flight, a failed save *is* the
        failure and it propagates untouched.

        `masking` is passed in rather than sniffed from `sys.exc_info()`, and that is not a matter
        of taste: `sys.exc_info()` reports the exception *currently being handled*, so inside an
        `except OSError` it is the `OSError` itself — never `None`. The sniffed form therefore
        swallows **every** save failure, including the ones that are the only failure. It did, and
        the tests caught it.
        """
        try:
            self._save()
        except OSError as error:
            if not masking:
                raise  # nothing to mask — this failed save *is* the turn's failure
            _log.error(
                "Could not persist the transcript of a turn that was already failing: %s", error
            )

    def _save(self, messages: list[Message] | None = None) -> None:
        """Write the transcript **atomically** — a killed wake can never leave a torn one.

        `messages` defaults to `history`; the progress hook passes the *in-flight* union instead
        (`_persist_progress`), because mid-turn the engine's work is not in `history` yet.

        The obvious form, and the one this replaces, is `path.write_text(...)`: truncate, then
        write. A signal landing between those two — `SIGKILL`, the default disposition of
        `SIGTERM`, the OOM killer, a box reset — leaves a **half-written file**, and a half-written
        transcript is not a degraded transcript. It is invalid JSON, so `_load` raises on it; the
        wake dies on load, and so does the next one, and every one after that. The agent is bricked
        on that timeline and its memory of the conversation is gone, until a human deletes the file.
        The window is small (one write per turn) and it is open on every turn of every agent.

        Write-to-temp + `fsync` + `os.replace` (atomic on POSIX) closes it: a crash can leave the
        *previous* transcript intact or the *new* one complete, never a splice of the two. The
        `fsync` is what makes that true against a power loss rather than only against a signal —
        without it `os.replace` can publish a file whose bytes never reached the platter.

        This is not a new idea in this codebase; it is an old one that skipped a file.
        `ClaimStore._write` (`_wake.py`) already does exactly this, for exactly this reason, and
        says so. The claim file is a few bytes of bookkeeping. The session is the agent's mind, and
        it was the one being written non-atomically. (Issue #297, finding 4.)

        **The temp is per-`Session`, not per-process** (`_temp`), because a pid does not identify a
        writer: two `Harness` instances over one home hold two `Session` objects on the *same* path
        in the *same* process, and a shared temp name would let them tear each other's temp — which
        `os.replace` would then publish, defeating the whole point. `ClaimStore` keys its temp on the
        wake id for the same reason; a pid was the weaker guard.

        **A leaked temp holds the whole transcript, so it is swept, not merely ignored.** On success
        the temp *becomes* the transcript (that is what `os.replace` does) and on an exception the
        `finally` removes it — but a process killed inside this window leaves it behind, containing
        the entire conversation. `_cleanup.enumerate_artifacts` therefore purges `sessions/*.json.*
        .tmp` alongside the transcript itself: a timeline deleted on the platform must not survive
        on the box as a temp file the sweep never looked at. (Issue #297, and it is exactly the trap
        `ClaimStore` avoids by keeping its temps inside a directory the sweep removes wholesale.)

        **The bound, stated rather than implied:** the *directory* entry is not fsynced, so a power
        loss immediately after `os.replace` may leave the previous transcript in place. That costs
        the newest turn — it does not tear the file, which is the invariant this method exists to
        hold. Losing a turn is recoverable (the mark never advanced, so the item is re-driven);
        an unparseable transcript is not.
        """
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload(self.history if messages is None else messages)
        temp = self._temp(self.path)
        try:
            with open(temp, "w") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())  # the bytes are on disk *before* the name points at them
            os.replace(temp, self.path)
        finally:
            # Success renamed it away; a failure must not leave a stray copy of the transcript.
            temp.unlink(missing_ok=True)

    def _temp(self, path: Path) -> Path:
        """This session's private staging path — `<name>.json.<pid>-<token>.tmp`.

        Dot-free between the `.json` and the `.tmp` so the cleanup sweep can recognize it by shape
        (`_cleanup._SESSION_TEMP`). Unique per process *and* per `Session` instance; see `_save`.
        """
        return path.with_suffix(f"{path.suffix}.{os.getpid()}-{self._token}.tmp")


def _payload(messages: list[Message]) -> str:
    """The JSON a transcript persists as — **bounded on the way out, never in place**.

    Persisting once per turn could bound the *live* list, because by then the model had finished
    with it: `_drive`'s `finally` evicts the pixels and caps the tool results, and what it mutates
    is what it then writes. A save that lands **mid-turn** (issue #297) cannot do that — the model
    is still looking at the image, and still reading the full tool result it got back one step ago.
    Bounding the live list there would reach into the conversation the engine is holding and quietly
    take things out of it.

    So the two jobs separate: `history` stays whole for as long as the turn needs it, and *this*
    decides what reaches the disk. Three disciplines, all of them here (Context Discipline: a viewed
    image is seen once and never re-billed; a tool result persists head-and-tail around an elision
    marker; a tool call's arguments are bounded the same way) — none of them has to be paid for by
    mutating the turn in flight.

    **Never mutate `message.tool_calls`.** `to_dict` hands back the *live* `arguments` dict by
    reference, so a cap written in place would reach into the call the engine is about to dispatch —
    and, worse, into the arguments a resume replays a create from. The capped copy goes into the
    payload and nowhere else.
    """
    capped = _capped_results(messages)
    payload: list[dict] = []
    for index, message in enumerate(messages):
        data = message.to_dict()
        data.pop("images", None)  # a viewed image is seen once; base64 never lands in a transcript
        if index in capped:
            data["content"] = capped[index]
        if message.tool_calls:
            data["tool_calls"] = _calls_payload(message.tool_calls, _results(messages, index))
        payload.append(data)
    return json.dumps(payload, indent=2)


def _calls_payload(calls: list[ToolCall], results: dict[str, Message]) -> list[dict]:
    """The `tool_calls` a turn persists — **the step's calls share one argument budget** (issue #304).

    The same unit change the results got, for the same reason: `max_steps` bounds the model's calls,
    not the tools it dispatched, so granting each call of a fanned-out step its own `TOOL_ARGS_CAP`
    would leave a step's persisted arguments scaling with a fan-out nothing bounds. They are
    water-filled (`_fill`), so a lone call still gets the whole budget, and a step whose calls are all
    ordinary keeps every one of them byte for byte.

    **A replayable call is outside the pool, not merely exempt from the cap.** Its arguments persist
    whole because the recovery re-issues the create from them (`_replayable`), and charging that
    against the step's budget would starve its siblings to buy nothing: the 200 KB body is in the
    transcript either way. The exception stays exactly as bounded as issue #301 left it — one dead
    turn's creates, until the re-issue writes a real result over the marker.
    """
    bounded = [
        index for index, call in enumerate(calls) if not _replayable(call, results.get(call.id))
    ]
    fitted = _fill(
        [calls[index].arguments for index in bounded],
        TOOL_ARGS_CAP,
        size=_json_size,
        elide=_cap_arguments,
    )
    capped = dict(zip(bounded, fitted))
    return [
        {"id": call.id, "name": call.name, "arguments": capped.get(index, call.arguments)}
        for index, call in enumerate(calls)
    ]


def _results(messages: list[Message], assistant: int) -> dict[str, Message]:
    """The results answering `messages[assistant]`'s calls — **from its own run, never globally.**

    A `tool_call_id` is the provider's own string and nothing normalizes it: a model that numbers its
    calls per response (`call_0`, `call_1` — what an OpenRouter-fronted model emits, and the fleet's
    primary agent is one) reuses the same ids on every turn. A global lookup would hand this turn's
    call the *previous* turn's result — declaring an interrupted create answered, capping the
    arguments the recovery is about to replay it from, and posting the peer a message with its body
    cut out. It is the same trap `heal_interrupted_calls` and `_idempotency.creates` each avoid, in
    the same way, for the same reason.
    """
    end = _run_end(messages, assistant)
    return {
        result.tool_call_id: result
        for result in messages[assistant + 1 : end]
        if result.role == "tool" and result.tool_call_id is not None
    }


def _replayable(call: ToolCall, result: Message | None) -> bool:
    """Might the recovery still re-issue this call **from these very arguments**? Then keep them whole.

    The exception that the whole argument cap turns on (issue #301). `_wake._reissue_interrupted_creates`
    re-runs an interrupted **platform create** under the deterministic key its dead wake minted — and
    it re-runs it from the arguments *on disk*. Elide those and the recovery posts the peer's message
    with its body cut out; if the original POST never landed, the elided body is what the timeline
    keeps. That is worse than the cost it saves, which is exactly why #297 left the arguments alone.

    Two states qualify, and no others:

    - **No result yet** — the call is in flight (this is the pre-dispatch save, the one the delivery
      guarantee rests on). A wake killed here leaves the call unanswered, and the next load heals it
      into the case below.
    - **The healed "outcome unknown" marker** — a wake *was* killed here, and the resume has not
      re-issued it yet. The marker is the flag, and it is a *durable* one: the arguments stay whole
      across as many failed wakes as it takes, and the cap falls the moment the re-issue writes a real
      result over it.

    **Only creates.** Every other interrupted call is surfaced to the model and never re-run — no
    idempotency key can un-spend money at fal.ai — so its arguments are dead weight the instant it is
    interrupted, and are bounded from the first save. That is what keeps the exception from becoming
    the loophole: what persists whole is exactly what is load-bearing, and nothing else.
    """
    if create_kind(call) is None:
        return False
    return result is None or result.content == INTERRUPTED


def turn_work(history: list[Message], turn: Message) -> list[Message]:
    """What one turn produced: everything after `turn`, up to the next **real** user turn.

    **There is exactly one of these, and both readers call it** — the idempotency minter, counting
    a turn's creates while it runs, and the recovery classifier, counting them again after a crash
    (`_wake._turn_work`). The two numbers must agree, and the only way to be sure they agree is for
    there to be one definition of where a turn ends. Two functions that "obviously" mean the same
    thing is how the ordinal drifts, and a drifted ordinal is a duplicate post.

    "Real" is the load-bearing word. The engine and the code bridge each inject `user` turns of
    their own — an image for the model to look at, a note naming the Assets a code run produced.
    They wear the role because it is the only one that content may ride on, but they are *this
    turn's work*, not a new turn of the conversation, and `injected` says so. Stopping at one would
    cut a turn in half: the narration behind it would vanish, and a turn that finished would read
    as one that was interrupted — which is a re-run of everything it already did.

    An unknown `turn` (compacted away, or never in this history) yields nothing, which is the
    honest answer rather than a wrong one.
    """
    start = _index_of(history, turn)
    if start is None:
        return []
    work: list[Message] = []
    for message in history[start + 1 :]:
        if message.role == "user" and not message.injected:
            break
        work.append(message)
    return work


def _turn_end(history: list[Message], turn: Message) -> int:
    """Where `turn`'s work ends — the index its continuation must be spliced at.

    The tail of `history` when the turn is the last one (the ordinary case), and somewhere in the
    middle when it is not: a resume that failed leaves an older turn unfinished *behind* a newer
    one, and an old turn's narration filed under a newer turn is a message committed that nobody
    ever answered.
    """
    start = _index_of(history, turn)
    if start is None:
        return len(history)
    return start + 1 + len(turn_work(history, turn))


def _index_of(history: list[Message], turn: Message) -> int | None:
    """Where `turn` sits in `history`, **by identity** — never by equality.

    `list.index` compares with `==`, and `Message` is a dataclass: two turns carrying the same text
    are equal, so a repeated question would find the wrong one. Identity is also what survives a
    compaction — it rewrites the list, moving objects rather than copying them.
    """
    for index, message in enumerate(history):
        if message is turn:
            return index
    return None


def heal_interrupted_calls(history: list[Message]) -> int:
    """Give every unanswered tool call a result. Returns how many were healed (issue #297).

    **This is not tidiness; it is the difference between a recoverable agent and a bricked one.**
    Persisting a turn incrementally means a wake can be killed between two tool calls of a single
    assistant turn, leaving the call on disk and its result not — a **dangling `tool_call_id`**,
    which is malformed *permanently*: the next wake loads it, sends it, the provider 400s, and so
    does every wake after that, until a human deletes the file. Incremental persistence done
    naively is therefore strictly worse than the bug it fixes — it trades an occasional double post
    for an agent that can never speak on that timeline again.

    Healing on load closes it, and closes it in the one place that cannot be forgotten: *every*
    reader of a transcript goes through `_load`. The synthesized result says exactly what is true —
    the outcome is unknown, and nothing has been re-run — because that is the only honest thing to
    say, and because the model reading it is perfectly capable of going and looking. A resume
    replaces this text for the one class of call it *can* safely re-issue (a platform create, under
    the key the dead wake used); everything else keeps it.
    """
    healed = 0
    index = 0
    while index < len(history):
        turn = history[index]
        if turn.role != "assistant" or not turn.tool_calls:
            index += 1
            continue
        # **Answered-ness is scoped to the assistant turn that issued the call, never to the whole
        # transcript.** A tool-call id is the *provider's* string and nothing normalizes it: a model
        # that numbers its calls per response (`call_0`, `call_1` — the shape an OpenRouter-fronted
        # model emits) reuses the same id on every turn. A global set of answered ids would then see
        # the *previous* turn's result and call this turn's identical id answered — leaving a real
        # dangling call unhealed, which is the permanent brick this function exists to prevent.
        after = _run_end(history, index)
        answered = {r.tool_call_id for r in history[index + 1 : after] if r.role == "tool"}
        missing = [call for call in turn.tool_calls if call.id not in answered]
        for offset, call in enumerate(missing):
            history.insert(after + offset, Message.tool(tool_call_id=call.id, content=INTERRUPTED))
        healed += len(missing)
        index = after + len(missing)
    return healed


def _run_end(history: list[Message], assistant: int) -> int:
    """Where the tool run answering `history[assistant]` ends — the next assistant turn, or the end.

    The results of one assistant turn's calls sit between it and the next assistant turn (a step
    note or an injected image turn may sit among them, and neither ends the run).
    """
    index = assistant + 1
    while index < len(history) and history[index].role != "assistant":
        index += 1
    return index


def _cap_tool_results(history: list[Message]) -> None:
    """Elide the over-long tool results in place, so the transcript stays bounded (issue #275).

    Only `tool` turns are touched, and only their `content` — the `tool_call_id` that pairs a
    result to the call it answers is untouched, and no message is ever dropped. A dropped tool
    turn would leave a dangling assistant tool-call and make the next wake's transcript
    malformed; an edited one is always well-formed.

    The model already read the full result on the turn the tool ran. This governs only what
    every *future* turn re-reads: one mailbox listing or wide file read is otherwise a
    permanent tax on the life of the timeline.

    It reads its answer from `_capped_results`, the same function `_payload` writes the disk from, so
    the live transcript and the file can never disagree about what a step's results cost.
    """
    for index, content in _capped_results(history).items():
        history[index].content = content


def _capped_results(messages: list[Message]) -> dict[int, str]:
    """Each tool result's persisted content, by index — **one step's results share one cap** (#304).

    The unit is the step, not the call, and that is the whole fix. A model may emit several tool calls
    in one assistant turn, and `max_steps` bounds the model's *calls*, never the tools dispatched — so
    a per-call cap let a step's persisted growth scale with a fan-out nothing bounds, and the
    compaction threshold's safety proof (`_context.worst_case_turn_tokens`) understated the worst case
    by exactly that factor. Sharing the budget makes the proof's arithmetic true instead of nearly true.

    It costs the ordinary step nothing. A lone call gets the whole `TOOL_RESULT_CAP` and is elided
    exactly as it always was, byte for byte; and because the share is *water-filled* (`_fill`), a step
    that fans out into ten small results keeps all ten of them whole. Only a fan-out that is also
    **fat** pays — which is precisely the shape that has to be bounded.

    **An interrupted result is outside the pool — never charged, never elided — and that is not a
    nicety.** `INTERRUPTED` is a *sentinel the recovery reads*, by exact match: `_replayable` keeps an
    interrupted create's arguments whole because of it, and `_idempotency.interrupted` finds the calls
    to re-issue by it. A shared budget is the first thing that could ever cut it — at a fan-out of ~14
    a share falls below its ~300 characters — and eliding it would be silent and severe: the create's
    arguments would be capped and then re-posted with the peer's message body cut out, and the recovery
    would no longer recognize the call as one to re-issue at all. It is the mirror of the exception
    `_replayable` already makes on the arguments side, and for the same reason: **the recovery's
    evidence is never bounded away.** It is bounded in count (one dead turn's calls) and in duration
    (the resume writes a real result over it, and that result is capped like any other).

    A `tool` turn answering no assistant turn cannot occur in a transcript this harness wrote, but a
    transcript is a file and a file can be anything: one is capped on its own rather than waved through
    the single gate that exists to bound it.
    """
    capped: dict[int, str] = {}
    for index, message in enumerate(messages):
        if message.role != "assistant" or not message.tool_calls:
            continue
        step = [
            i
            for i in range(index + 1, _run_end(messages, index))
            if messages[i].role == "tool"
            and messages[i].content
            and not _is_interrupted(messages[i])
        ]
        results = [messages[i].content or "" for i in step]
        capped.update(zip(step, _fill(results, TOOL_RESULT_CAP, size=len, elide=_elide)))
    for index, message in enumerate(messages):
        if index in capped or message.role != "tool" or not message.content:
            continue
        if not _is_interrupted(message):
            capped[index] = _elide(message.content, TOOL_RESULT_CAP)
    return capped


def _is_interrupted(message: Message) -> bool:
    """Is this the healed "outcome unknown" result a killed wake left behind (`INTERRUPTED`)?

    The recovery reads it by exact match, so nothing may rewrite it — see `_capped_results`.
    """
    return message.content == INTERRUPTED


def _fill(items: list[_T], budget: int, *, size: _Size[_T], elide: _Elide[_T]) -> list[_T]:
    """Water-filling: hand every item an equal share of `budget`, smallest first.

    The one allocator behind all three caps — a step's results, a step's calls, and one call's
    arguments — because they are the same problem three times: several things, one budget, and no
    reason for the small ones to pay for the big one.

    Take the items in *ascending* size and give each an equal share of what is left. An item that fits
    under its share is kept byte for byte and rolls its surplus over to the ones above it; only an item
    over its share is cut, and only to the share it actually got.

    Two properties follow, and they are why this is worth a function rather than a loop of `min()`:

    - **If the whole set fits the budget, every item is kept whole.** The smallest of the remaining
      items is never larger than their mean, and its share *is* their mean — so it is kept, and by
      induction so is every item after it. This is what makes a wide fan-out of *small* results free,
      and it is what makes the cap a **fixed point**: a set that has already been capped fits, so
      re-saving it every turn for the life of the timeline never grinds an excerpt into an excerpt of
      an excerpt.
    - **It always makes progress.** The obvious alternative — cut the biggest item until the set fits —
      has a cliff: an excerpt costs a marker, so an item can be *too small to be worth eliding and
      still too big to keep*, and a set made of several of those can never be brought under the budget
      at all (issue #301's `tasks create` with three 700-character fields, which fell through to the
      total-loss stub). Water-filling takes the room from where the room actually is.
    """
    kept = list(items)
    room = budget
    for spent, index in enumerate(sorted(range(len(items)), key=lambda i: size(items[i]))):
        share = room // (len(items) - spent)
        item = items[index]
        kept[index] = item if size(item) <= share else elide(item, share)
        room -= size(kept[index])
    return kept


def _elide(text: str, budget: int) -> str:
    """`text` unchanged, or its head and tail around a marker naming what was cut.

    `budget` is this result's **share of its step** (`_capped_results`) — the whole `TOOL_RESULT_CAP`
    when the step made a single call, which is the ordinary case and where this is byte for byte what
    it has always been. A step that fans out shrinks the excerpt proportionally rather than granting
    each of its calls a full cap.

    The marker states the original size, so the model (and a human reading the transcript)
    knows it is looking at an excerpt and how much is missing — a silent truncation would let
    the model reason on a partial listing as though it were whole. It names the remedy
    *conditionally* ("if you need it in full"), never as an instruction: a marker that reads
    like a directive would invite the model to re-run tools it has no present use for.
    """
    if len(text) <= max(budget, _MIN_EXCERPT):
        # Under its share, or too small for cutting to be worth the marker that says so. The second
        # is also what makes this a **fixed point**: `_gone` is well under `_MIN_EXCERPT`, so a value
        # already at the floor is never elided again — no marker of a marker, each naming a size that
        # is no longer true, on every save for the life of the timeline.
        return text
    # Size the marker against the widest cut it could ever name — the whole text. The real cut is
    # smaller, so the real marker is never longer than this one, and the excerpt below therefore
    # cannot overrun the share. (The marker's length depends on the numbers printed in it, which
    # depend on the excerpt, which depends on the marker: measuring the worst case breaks the circle.)
    room = budget - len(_result_marker(len(text), len(text)))
    if room < _MIN_EXCERPT:
        return _gone(len(text))  # no room for an excerpt worth reading; the floor is all that fits
    room = min(room, _ELISION_HEAD + _ELISION_TAIL)  # a share may shrink the excerpt, never grow it
    tail = room * _ELISION_TAIL // (_ELISION_HEAD + _ELISION_TAIL)
    head = room - tail
    cut = len(text) - head - tail
    return text[:head] + _result_marker(cut, len(text)) + (text[-tail:] if tail else "")


def _gone(size: int) -> str:
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
    return f"[... {size} chars elided ...]"


def _result_marker(cut: int, total: int) -> str:
    """What stands in for the elided middle of a tool result."""
    return (
        f"\n\n[... {cut} chars elided of {total} — this is an archived excerpt; the full "
        f"result was shown when the tool ran. Re-run it if you need it in full. ...]\n\n"
    )


def _cap_arguments(arguments: dict[str, Any], budget: int) -> dict[str, Any]:
    """`arguments`, bounded to `budget` — a **new dict**, never an edit of the live one.

    `budget` is this call's **share of its step's** `TOOL_ARGS_CAP` (`_calls_payload`) — the whole cap
    when the step made one call, which is the ordinary case.

    **Capping may never change what `create_kind` reads off a call**, and that is a correctness
    requirement, not a nicety: the idempotency ordinal is "the nth create of this kind in this turn",
    counted once by the live mint over the in-memory transcript and again by the recovery over the
    *reloaded* one (`_idempotency.creates`). The two counts must be identical. If capping could hide an
    `action`, a create would vanish from the recovery's count, the next key would be minted one short of
    the one the platform has already seen — and a key the platform has never seen is a message posted
    twice. Two things make it safe: an `action` is a handful of characters, so it always fits inside its
    share below and is never cut; and the stub re-states it verbatim. An `action` long enough to be
    dropped is one no `CREATE_CALLS` entry could match anyway, so `create_kind` answers ``None`` before
    the cap and ``None`` after it.

    **The fit is measured, never computed**, and that is why this loop exists rather than one pass of
    arithmetic: how many characters a string costs once serialized depends on the string (a quote or a
    backslash escapes to two), so a budget that is exactly right on prose can be exceeded by the same
    length of JSON or source code. Halving and re-measuring converges in one or two passes and cannot
    lie about the result; the alternative — assuming a worst-case escape factor — would cut every
    ordinary argument to a fraction of the budget it is actually entitled to.
    """
    if _json_size(arguments) <= budget:
        return arguments
    attempt = budget
    for _ in range(_ARG_FIT_ATTEMPTS):
        capped = _fit(arguments, attempt)
        if _json_size(capped) <= budget:
            return capped
        attempt //= 2
    # Not the size of any one argument but the *number* of them — hundreds of fields, or names longer
    # than their values, so there is no share left to give anybody. Rare to the point of pathological,
    # but "rare" is not a bound, and an unbounded fallback would be the very defect this exists to
    # close, hiding behind a shape nobody expected.
    return _arguments_stub(arguments)


def _fit(arguments: dict[str, Any], budget: int) -> dict[str, Any]:
    """Every argument gets a **fair share** of `budget`, so the small ones survive whole.

    Water-filling (`_fill`), and it is the whole reason a capped call is still worth reading: a
    `messages create` with a 200 KB body still persists its `action`, its `timeline` and its `subject`
    in full, and spends the entire remaining budget on an excerpt of the body. The keys are charged
    first — they are not optional, and a share handed out of money that was already spent is not a share.
    """
    room = budget - _json_size(dict.fromkeys(arguments, ""))  # what the keys themselves cost
    names = list(arguments)  # the model reads them in the order it wrote them
    values = _fill(
        [arguments[name] for name in names], room, size=_json_size, elide=_elide_argument
    )
    return dict(zip(names, values))


def _elide_argument(value: Any, budget: int) -> Any:
    """One argument, cut to `budget` — or left alone, if cutting it would not make it smaller.

    A string keeps a head and a tail around a marker naming what was lost, so the model still reads the
    opening of what it wrote. Anything else (a large list or object) is replaced outright: there is no
    honest head-and-tail of a structure, and a truncated one would read as complete.

    **The shrink check is a guard, not a flourish.** The marker costs ~120 characters, so "eliding" a
    200-character value would *grow* it — and a cap that can make a transcript bigger is not a cap.
    Comparing the two sizes states that in the one place it cannot be forgotten. It is also what keeps
    the cap a **fixed point** when a value is already down at the marker's own size: without it, a
    transcript re-saved every turn would grind a marker into a marker of a marker, each one naming a
    size that is no longer true.
    """
    size = _json_size(value)
    if size <= _MIN_EXCERPT:
        # Too small for cutting to be worth the marker that says so — and the fixed point that keeps a
        # floor marker (`_gone`, well under this) from being ground into a shorter one on every save.
        return value
    if not isinstance(value, str):
        # A structure has no honest head-and-tail, so there is nothing to excerpt: it is the floor or
        # it is whole.
        elided: Any = _gone(size)
        return elided if _json_size(elided) < size else value

    marker = (
        f"\n\n[... elided from {len(value)} chars — this argument is an archived excerpt; the full "
        f"value was sent when the call ran. ...]\n\n"
    )
    room = budget - _json_size(marker)
    if room < _MIN_EXCERPT:
        elided = _gone(size)  # no room for an excerpt worth reading; the floor is all that fits
        return elided if _json_size(elided) < size else value
    tail = min(room // 5, _ARG_ELISION_TAIL)
    head = room - tail
    elided = value[:head] + marker + (value[-tail:] if tail else "")
    return elided if _json_size(elided) < size else value


def _arguments_stub(arguments: dict[str, Any]) -> dict[str, Any]:
    """The backstop for a call too big to bound by cutting its values — the hard end of the bound.

    Keeps `action`, and only `action`, because `create_kind` reads it and the idempotency ordinal must
    be the same number computed off the live transcript and the reloaded one (see `_cap_arguments`). A
    long `action` is dropped with the rest: no create the platform keys has one, so `create_kind`
    answers ``None`` either way and the count still agrees.
    """
    stub: dict[str, Any] = {}
    action = arguments.get("action")
    if isinstance(action, str) and len(action) <= _ARG_ACTION_MAX:
        stub["action"] = action
    stub["[elided]"] = _gone(_json_size(arguments))
    return stub


def _json_size(value: Any) -> int:
    """What `value` costs the model — in characters, **the way the model reads them, not the disk**.

    `ensure_ascii=False`, and it is load-bearing rather than cosmetic. The default (`True`) escapes
    every non-Latin character to a six-character `\\uXXXX`, so measuring with it prices one Japanese
    character at six and a modest 500-character message body at 3,000 — six times the cap. The cap is a
    bound on **context**, and context is billed in tokens, which are computed from the decoded string:
    a character is a character, whatever script it is written in. Measuring the escaped form made the
    cap bite ~6× harder on every non-Latin script — so a peer answered in Japanese kept nothing but its
    `action` (issue #301 review), while the same message in English persisted whole. That is not a cost
    bug; it is an agent that cannot remember what it said because of the language it said it in, on a
    platform whose whole premise is that its peers are equals. The disk file still stores the escapes;
    what it costs on disk is not what this bounds.

    `default=str` because a tool call's arguments come off a provider's wire and may carry anything a
    model emitted; a sizing helper that raised on an exotic value would take the whole save down.
    """
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _failure_note(exc: BaseException) -> str:
    """The trailing marker for a partial transcript persisted after `engine.run` raised.

    Names the exception type and message so a later reader — or the operator diagnosing a
    step-cap — can tell the turn failed and why, rather than the transcript ending on a
    dangling tool call with no explanation (issue #244).
    """
    return f"[turn failed: {type(exc).__name__} — {exc}]"
