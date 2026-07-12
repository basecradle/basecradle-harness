"""The context budget: bound the transcript, so a long-lived agent never hits the wall.

The harness replays a session's **entire** persisted transcript to the model on every wake, and
until now nothing bounded it. A continuous agent therefore grew monotonically toward its model's
context ceiling — where the provider returns a deterministic 400 and *every* subsequent wake
rebuilds the same over-long request and fails identically. That agent is bricked on that timeline
until a human edits its session file by hand (@glm-5.2 came within ~25% of that wall in three days,
issue #276). This module is the structural fix, and it has two halves:

- **`ContextBudget`** — what the ceiling *is*, and whether we are too close to it.
- **`Compactor`** — the rewrite that pulls us back: keep a recent window verbatim, replace
  everything older with one model-written summary.

Four properties are load-bearing, and each is a decision, not an accident:

**The trigger is the provider's own reported usage — never a client-side count.** Every provider
returns exact input-token usage per response and every adapter already logs it (``tokens_in=``);
each one now also *remembers* it (`last_tokens_in`). So the harness asks a question it can answer
exactly, for free, on any provider: *how big was the last call, really?* Counting tokens locally
would need a tokenizer per model — and GLM publishes none, so a client-side count could not even be
honest, let alone free.

**The limit resolves env → adapter → floor, never from a table.** A static model→limit table cannot
express routed reality (one OpenRouter model id fans out to endpoints spanning 10× in ceiling) and
it rots silently: a stale row means compacting at the wrong threshold, or never. So each adapter
answers however it honestly can (`context_limit`), ``None`` when it cannot, and the operator's
`HARNESS_MAX_CONTEXT_TOKENS` always wins.

**Compaction fires at half the ceiling, never near it** — headroom for the reply, for the
summarization call itself, and for estimate error. That threshold is only safe because *no single
turn can leap over it*: the persisted growth of one turn is bounded by `TOOL_RESULT_CAP` (4 KB) ×
`DEFAULT_MAX_STEPS` (24) ≈ 30 K tokens, which cannot cross from under-half to over-full on any
budget at or above the floor. **The tool-result cap (issue #275) is a prerequisite of this file**,
not a neighbor of it: relax it and the 50% threshold stops being a safe distance.

**A cut may land only immediately before a `user` turn.** This is the correctness constraint the
whole rewrite turns on. Tool results follow the assistant turn that called them, so cutting
mid-chain would strand a `tool` message whose `tool_call_id` refers to a dropped assistant call —
malformed on strict providers, and malformed *forever*, breaking every later wake. That is a worse
failure than the bloat this fixes, so when no safe cut exists the compaction **declines** rather
than producing a transcript it cannot prove is well-formed.

Cache interplay, eyes open (`CLAUDE.md` → Context Discipline): each compaction rewrites the prefix
and invalidates the provider's prompt cache **once**. That is accepted and bounded — we retain ~20%
of the budget and fire at 50%, so the context must roughly double before the next compaction, and
the new prefix is byte-stable from the moment it is written. Compaction happens *inside* the stable
prefix; it never moves the volatile tail (the per-wake brief stays spliced immediately before the
newest user turn), so the caching invariant is untouched.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from basecradle_harness._messages import Message
from basecradle_harness._observability import kv
from basecradle_harness._provider import Provider

_log = logging.getLogger("basecradle_harness")

#: The ceiling assumed when the operator names none and the adapter cannot answer one. It is a
#: *floor on plausible ceilings*, not a guess at the real one: every model the fleet runs, and
#: everything current from the majors, is at or above 128 K, so compacting against this number is
#: safe (early, slightly lossy) rather than wrong. It is what an OpenAI-direct agent lands on today,
#: because OpenAI's models API states no context window and the harness will not pretend to know one.
#:
#: **The one case it does not cover:** a deliberately small-context deployment (a local model, a
#: budget endpoint) whose real window is *below* 128 K. There the floor sits above the ceiling and
#: compaction would never fire before the wall — so for such a model `HARNESS_MAX_CONTEXT_TOKENS`
#: is **not optional**.
DEFAULT_CONTEXT_LIMIT = 128_000

#: Compact once the last call's input crossed this fraction of the budget. Half, deliberately: the
#: remaining headroom absorbs the reply, the summarization call, and the error in the token estimate
#: — see the module docstring for why one turn cannot jump the gap.
COMPACT_AT = 0.5

#: How much of the budget the retained tail may occupy after a compaction. The gap between this and
#: `COMPACT_AT` is what makes compaction *rare*: the live context must roughly double before the
#: next one fires, so the prompt cache re-warms and pays for itself many times over in between.
KEEP_FRACTION = 0.2

#: The cap on what is fed to the summarization call, as a fraction of the budget. On the normal path
#: this never bites — the trigger fires at 50%, so the dropped region is at most ~half the budget by
#: construction, and the summarize call (dropped region + a short instruction + a short reply) fits
#: with room to spare. **Keep that arithmetic in view before raising `COMPACT_AT`:** the summarize
#: call is itself a model call against the same ceiling, and a trigger set too high would make the
#: rescue call the thing that overflows. The cap exists for the *emergency* path (`emergency_compact`),
#: where the transcript is already past the ceiling and the region to summarize is unbounded.
SUMMARY_INPUT_FRACTION = 0.5

#: The chars-per-token assumed on the emergency path, where there is no successful call to calibrate
#: against (the request never completed). Deliberately pessimistic: real text runs ~4 chars/token, so
#: assuming 2 keeps roughly half of what the calibrated ratio would — the right instinct when the
#: transcript has already proven it is over the ceiling.
PESSIMISTIC_CHARS_PER_TOKEN = 2.0

#: The most of an over-ceiling transcript the emergency path may retain, as a fraction of what is
#: actually there. The provider has just *refused* this transcript, so any conclusion our token
#: arithmetic reaches about it is already known to be wrong — including, dangerously, "it fits."
#: Capping the tail against the transcript's real size guarantees the rescue always makes progress
#: instead of declining and leaving the agent bricked. The provider's word beats our estimate.
EMERGENCY_KEEP_RATIO = 0.25

#: The opening of the system turn a compaction leaves behind. It is a *marker*, not decoration:
#: `_prelude_end` reads it to tell a previous summary (compacted conversation, which the next
#: compaction must fold in) apart from the agent's charter (standing context, never summarized).
_SUMMARY_MARKER = "[Earlier conversation compacted"

#: The over-length 400 an endpoint returns when the request exceeded the model's context window.
#: Every vendor spells it differently and none of them give it a machine-readable code, so this is
#: the one heuristic in the file — and it **fails safe**: a phrasing we don't recognize is simply not
#: recognized, and the harness behaves exactly as it did before this module existed.
#:
#: **Every alternative names tokens, the context, or the prompt.** A bare "exceeds the maximum" is
#: deliberately *not* here: the openai error mapper this feeds is shared with the image and audio
#: tools (`sdk_error_context`), where "exceeds the maximum size" is an ordinary file-too-big 400 —
#: and a rescue that fires on the wrong 400 would compact a transcript that was never too long. A
#: false negative costs one un-rescued wake; a false positive silently eats conversation.
_OVERFLOW_PHRASES = re.compile(
    r"context[ _-]?(?:length|window|limit)"
    r"|maximum context"
    r"|too many (?:input |prompt )?tokens"
    r"|(?:prompt|input|message[s]?) (?:is |are )?too long"
    r"|exceeds? the (?:model'?s? )?(?:maximum |max )?context"
    r"|reduce the length of the (?:messages|prompt|input)",
    re.IGNORECASE,
)

#: What the summarizer is asked for. Written as notes-to-self, and **work-first on purpose**: the
#: memory seam's `observe` hook captures only user+assistant text, so tool-driven work leaves no
#: durable trace unless something writes it down (issue #276, requirement 7). Compaction is where
#: that work would otherwise be lost, so this is where it is recorded — and the resulting summary is
#: also handed to durable memory (`Compactor.on_summary`). Raw tool output is *not* preserved: the
#: point is a record of what was done, not a second copy of the bytes we are dropping.
_SUMMARIZE_INSTRUCTION = """You are compacting your own conversation transcript to stay inside your context window. \
The excerpt below is about to be deleted and replaced by what you write now. Write dense, factual \
notes to your future self, in the first person, under these three headings:

1. WORK DONE — the actions you actually took, the tools you used, and what came of them: artifacts \
produced (asset uuids, URLs, file paths, task uuids), things posted, things changed, things that \
failed. Tool results are deleted along with the excerpt, so an action you do not write down here \
leaves no trace that it ever happened.
2. WHAT WAS SAID — the substance of the conversation: who said what, what was decided, what was \
promised.
3. OPEN THREADS — what is unfinished, what you owe someone, and what you meant to do next.

Preserve identifiers (uuids, URLs, handles, numbers) verbatim — they are unrecoverable once the \
excerpt is gone. Do not speculate, do not pad, and do not address anyone: these are your own notes, \
not a message. Everything you leave out is forgotten."""


def is_context_overflow(text: str) -> bool:
    """Does this provider error text say the request exceeded the model's context window?

    Called by each adapter's error mapper on an over-length-shaped status error, so the
    `ProviderContextLengthError` class is raised provider-agnostically — classified by the *nature
    of the fault*, exactly as the truncated-response class is (issue #259). See `_OVERFLOW_PHRASES`
    for why this is a phrase match and why that is safe.
    """
    return bool(text) and bool(_OVERFLOW_PHRASES.search(text))


@dataclass(frozen=True)
class Limit:
    """The resolved context ceiling and where it came from (for the log line)."""

    tokens: int
    source: str  # "env" | "adapter" | "default"


def provider_tokens_in(provider: Provider) -> int | None:
    """The input-token count the provider reported for its **most recent** call, if it reported one.

    A capability read, guarded — a third-party adapter that never records it simply never triggers
    compaction, rather than crashing a wake.

    **Read it immediately after the run that produced it.** One provider instance is shared by every
    session of an agent (one engine, many channels), so this attribute is "the last call *anyone*
    made". That is unambiguous where it is used — `Session.send` reads it the moment its own
    `engine.run` returns, on a single thread, so the last call is always its own — and the test suite
    pins that assumption so a future change that pools or parallelizes providers cannot silently
    attribute one session's usage to another's transcript.
    """
    value = getattr(provider, "last_tokens_in", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


class ContextBudget:
    """The agent's context ceiling, resolved once and lazily, and the compaction threshold from it.

    Resolution order (issue #276, requirement 2):

    1. **`HARNESS_MAX_CONTEXT_TOKENS`** — the operator's override. Always wins; it is the 2 a.m.
       escape hatch, and the only correct answer for a model whose window is below the floor or
       whose routing an operator has pinned. ``0`` disables compaction outright.
    2. **`provider.context_limit()`** — the adapter capability. Each adapter answers however it
       honestly can (xAI reads its SDK's ``max_prompt_length``; OpenRouter reads the live per-endpoint
       ceilings) and returns ``None`` when it cannot (OpenAI states no context window anywhere).
    3. **`DEFAULT_CONTEXT_LIMIT`** — the conservative floor.

    Args:
        provider: The model adapter. Only its optional `context_limit` capability is used.
        override: `HARNESS_MAX_CONTEXT_TOKENS`, or ``None`` when unset. ``0`` disables compaction.
    """

    def __init__(self, provider: Provider, *, override: int | None = None) -> None:
        self._provider = provider
        self._override = override
        self._limit: Limit | None = None

    @property
    def enabled(self) -> bool:
        """False only when the operator explicitly set the budget to ``0`` (compaction off).

        Honored on **every** path, including the over-length rescue: ``0`` means "I manage this
        agent's context myself," and an escape hatch that quietly rewrites the operator's transcript
        anyway — at exactly the moment they would least expect it — is not an escape hatch. The cost
        of taking them at their word is stated where they set it: with compaction off, an agent that
        outgrows its ceiling stays bricked until they intervene.
        """
        return self._override != 0

    def limit(self) -> Limit:
        """The resolved ceiling — computed once per process, then cached.

        The adapter lookup is a live API call (an SDK metadata request), so it is made **lazily,
        at most once, and never fatally**: any failure — network, auth, an SDK shape we did not
        expect — degrades to the conservative floor. A wake must never break over a metadata read.
        """
        if self._limit is not None:
            return self._limit
        # `if self._override` would read a deliberate 0 as "unset" and fall through to the adapter,
        # reporting a ceiling the operator explicitly opted out of. Nothing should ask for a limit
        # while compaction is disabled (`enabled` gates every caller), but a budget that answers
        # dishonestly when misused is a trap, so the check is explicit.
        if self._override is not None and not self.enabled:
            return Limit(0, "env")
        if self._override:
            self._limit = Limit(self._override, "env")
        else:
            self._limit = Limit(*(self._from_adapter() or (DEFAULT_CONTEXT_LIMIT, "default")))
        _log.info(
            "context limit %s",
            kv(limit=self._limit.tokens, source=self._limit.source, compact_at=self.threshold()),
        )
        return self._limit

    def _from_adapter(self) -> tuple[int, str] | None:
        """The adapter's own answer, or ``None`` if it has none (or could not get one)."""
        capability = getattr(self._provider, "context_limit", None)
        if not callable(capability):
            return None
        try:
            value = capability()
        except Exception as exc:  # noqa: BLE001 - a metadata read must never break a wake
            _log.warning("Could not read the model's context limit from the provider: %s", exc)
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return value, "adapter"

    def threshold(self) -> int:
        """The input-token count above which the next turn must compact."""
        return int(self.limit().tokens * COMPACT_AT)

    def should_compact(self, tokens_in: int | None) -> bool:
        """Did the last call cross the threshold?

        The cheap guard first: with no operator override, a call under half the *floor* cannot have
        crossed half of any ceiling at or above the floor — so it needs no answer, and the adapter's
        metadata call is never made. A quiet agent therefore pays nothing at all for this feature:
        no extra API call, ever.
        """
        if not self.enabled or not tokens_in:
            return False
        if self._override is None and tokens_in < DEFAULT_CONTEXT_LIMIT * COMPACT_AT:
            return False
        return tokens_in > self.threshold()


class Compactor:
    """Rewrites a transcript in place: recent window verbatim, everything older summarized.

    Args:
        provider: The model. Used for the summarization call (`tools=None` — the summarizer needs
            no tools) and for reading back the usage it reported (`provider_tokens_in`).
        budget: The resolved ceiling and threshold (`ContextBudget`).
        on_summary: Called with each summary the moment it is written, so the summary reaches
            **durable memory** and the work it records outlives the turns being dropped (issue #276,
            requirement 7). Wired by the wake to the agent's bound memory provider; ``None`` in the
            plain library API, where compaction still works and only the memory write is absent.
            It is guarded — a memory failure logs and never blocks the compaction.
    """

    def __init__(
        self,
        provider: Provider,
        budget: ContextBudget,
        *,
        on_summary: Callable[[str], None] | None = None,
    ) -> None:
        self.provider = provider
        self.budget = budget
        self.on_summary = on_summary

    def maybe_compact(self, history: list[Message]) -> bool:
        """Compact `history` if the provider's last call crossed the threshold. Returns whether it did.

        The trigger is the *previous* call's reported usage, so this runs after a turn has settled —
        the number is in hand, no state has to survive the process, and the compacted transcript on
        disk is itself the record of the decision.
        """
        tokens_in = provider_tokens_in(self.provider)
        if not self.budget.should_compact(tokens_in):
            return False
        assert tokens_in is not None  # should_compact is False for None
        # Calibrate chars→tokens against the call that just happened, so no tokenizer is needed and
        # a model that publishes none (GLM) is served exactly like one that does. The ratio errs
        # conservative *by construction*: `tokens_in` also covers the ephemeral brief and the tool
        # schemas, which are not in `history`, so the measured chars-per-token comes out low and the
        # retained tail is sized smaller than the truth, never larger.
        chars_per_token = _chars(history) / tokens_in
        return self._compact(history, chars_per_token=chars_per_token, tokens_in=tokens_in)

    def emergency_compact(self, history: list[Message]) -> bool:
        """Compact a transcript that is **already past the ceiling**, after an over-length 400.

        Prevention cannot help an agent that is already at the wall: its request fails before it can
        report any usage, so there is nothing to calibrate against and no successful call to trigger
        on — every wake rebuilds the same doomed request and dies the same way. This is the path that
        unbricks it, and it is why `ProviderContextLengthError` exists: compact hard on pessimistic
        assumptions (`PESSIMISTIC_CHARS_PER_TOKEN`), and let the caller retry the turn once.

        Returns whether the transcript was actually rewritten — ``False`` means the caller must let
        the original error propagate rather than retry a request that would fail identically. An
        operator who disabled compaction (`HARNESS_MAX_CONTEXT_TOKENS=0`) gets ``False`` here too:
        the rescue is compaction, and "off" means off (see `ContextBudget.enabled`).
        """
        if not self.budget.enabled:
            _log.warning(
                "Context overflow, but compaction is disabled (HARNESS_MAX_CONTEXT_TOKENS=0): the "
                "transcript is left exactly as it is and the error stands."
            )
            return False
        _log.warning(
            "context overflow: the last request exceeded the model's context window — compacting "
            "the transcript on pessimistic assumptions and retrying the turn once."
        )
        return self._compact(
            history, chars_per_token=PESSIMISTIC_CHARS_PER_TOKEN, tokens_in=None, emergency=True
        )

    def _compact(
        self,
        history: list[Message],
        *,
        chars_per_token: float,
        tokens_in: int | None,
        emergency: bool = False,
    ) -> bool:
        limit = self.budget.limit()
        keep_chars = int(limit.tokens * KEEP_FRACTION * chars_per_token)
        if emergency:
            # **The provider's word beats our estimate.** It just refused this transcript as too
            # long, so whatever our chars→tokens arithmetic concludes about it is *wrong* — and if
            # the arithmetic says "it already fits" the rescue would decline and the agent would
            # stay bricked, which is the one outcome this path exists to prevent. So the retained
            # tail is also capped as a fraction of what is actually there: a compaction that has to
            # happen always makes real progress.
            keep_chars = min(keep_chars, int(_chars(history) * EMERGENCY_KEEP_RATIO))
        head = _prelude_end(history)
        cut = _cut_index(history, head, keep_chars)
        if cut is None:
            # No safe boundary: the whole transcript is one unbroken chain (or is already only the
            # newest turn). Declining is the *correct* failure — a cut we cannot prove is well-formed
            # would strand a tool result from its call and break every future wake, which is worse
            # than the bloat. WARNING, because the agent keeps working and nothing else looks wrong.
            _log.warning(
                "Context compaction declined: no safe cut point in the transcript (%d messages). "
                "The transcript is unchanged.",
                len(history),
            )
            return False
        dropped = history[head:cut]
        before_chars = _chars(history)
        summary = self._summarize(dropped, limit=limit, chars_per_token=chars_per_token)
        if summary is None:
            return (
                False  # the summarize call failed; _summarize logged it. Leave the transcript be.
            )
        history[head:cut] = [Message.system(_summary_note(len(dropped), summary))]
        self._remember(summary)
        _log.info(
            "context compact %s",
            kv(
                tokens_in=tokens_in,
                limit=limit.tokens,
                source=limit.source,
                threshold=self.budget.threshold(),
                messages=f"{len(dropped) + len(history) - 1}→{len(history)}",
                chars=f"{before_chars}→{_chars(history)}",
                summarized=len(dropped),
                emergency=("yes" if emergency else None),
            ),
        )
        return True

    def _summarize(
        self, dropped: Sequence[Message], *, limit: Limit, chars_per_token: float
    ) -> str | None:
        """One model call: the dropped region in, notes-to-self out. ``None`` if the call failed.

        The input is the region **rendered as text**, not replayed as messages — a replay would carry
        assistant tool-calls whose results we are dropping, and a request with a dangling
        `tool_call_id` is exactly the malformed shape this module exists to avoid producing. Rendering
        also lets the region be trimmed to fit without breaking anything.

        **The summarize call is itself a model call against the same ceiling**, so its input is
        bounded (`SUMMARY_INPUT_FRACTION`) — a rescue that overflows rescues nothing. When the region
        does not fit, the **carried summary is never what gets cut**: it is the cumulative record of
        *everything* before it, so dropping it to make room would forget the entire past to preserve
        the recent past — precisely backwards. The oldest un-summarized conversation is trimmed
        instead, and loudly, because that material is genuinely lost.

        Tools are withheld (`tools=None`): summarizing is reading, not acting.
        """
        # The previous compaction's summary rides first and whole; the fresh region fills what's left.
        carried = _render([m for m in dropped if _is_summary(m)])
        excerpt = _render([m for m in dropped if not _is_summary(m)])
        allowance = int(limit.tokens * SUMMARY_INPUT_FRACTION * chars_per_token)
        room = max(0, allowance - len(carried))
        if len(excerpt) > room:
            # Reached when the transcript overshot the threshold badly before this fired — a wake
            # whose turn added a lot, or the first compaction of a transcript that grew large before
            # this existed. Say plainly what is lost: a silent trim would let the model summarize a
            # fragment as though it were the whole.
            cut = len(excerpt) - room
            _log.warning(
                "Context compaction: %d characters of the oldest conversation did not fit in one "
                "summarization call and are dropped unsummarized.",
                cut,
            )
            excerpt = (
                f"[{cut} characters of the oldest conversation in this excerpt could not fit in a "
                f"single summarization call and are gone, unsummarized.]\n\n" + excerpt[cut:]
            )
        region = "\n\n".join(part for part in (carried, excerpt) if part)
        try:
            reply = self.provider.chat(
                [
                    Message.system(_SUMMARIZE_INSTRUCTION),
                    Message.user(f"The excerpt to compact (oldest first):\n\n{region}"),
                ],
                tools=None,
            )
        except Exception as exc:  # noqa: BLE001 - a failed compaction degrades; it never breaks a wake
            _log.warning("Context compaction failed: the summarization call errored: %s", exc)
            return None
        summary = (reply.content or "").strip()
        if not summary:
            _log.warning("Context compaction failed: the summarization call produced no text.")
            return None
        return summary

    def _remember(self, summary: str) -> None:
        """Hand the summary to durable memory — guarded, and never a reason to abort a compaction."""
        if self.on_summary is None:
            return
        try:
            self.on_summary(summary)
        except Exception as exc:  # noqa: BLE001 - memory is best-effort; the transcript still compacts
            _log.warning("Could not write the compaction summary to memory: %s", exc)


def _prelude_end(history: list[Message]) -> int:
    """Where the conversation starts: past the leading system turns that are the agent's *charter*.

    The charter is standing context, not conversation, so it is never summarized away. (Under a
    router the charter rides the ephemeral brief and the transcript has no leading system turn at
    all — then this is simply 0.)

    **A previous summary is emphatically not part of the prelude.** It is compacted *conversation*,
    so it sits in the region the next compaction drops and gets folded into the new summary — which
    is what makes summaries cumulative rather than a pile that grows one entry per compaction,
    forever, at the head of the transcript. Skipping it here would rebuild the very unbounded prefix
    this module exists to prevent.
    """
    index = 0
    while (
        index < len(history) and history[index].role == "system" and not _is_summary(history[index])
    ):
        index += 1
    return index


def _is_summary(message: Message) -> bool:
    """Is this the system turn a previous compaction left behind? (See `_prelude_end`.)"""
    return message.role == "system" and (message.content or "").startswith(_SUMMARY_MARKER)


def _cut_index(history: list[Message], head: int, keep_chars: int) -> int | None:
    """The index the retained tail begins at: the **earliest safe cut** whose tail fits `keep_chars`.

    Safe means one thing: the tail must begin at a `user` turn. A `tool` result only ever follows the
    assistant turn that called it, so a tail that starts at a user turn can never open with a tool
    result whose call was dropped — no dangling `tool_call_id`, on any provider, ever. Cutting
    anywhere else risks exactly that, and a malformed transcript poisons *every* later wake.

    Walking backward and keeping the earliest affordable boundary retains as much real conversation
    as the budget allows. If even the newest user turn overruns `keep_chars` it is kept anyway — the
    current turn is not optional — and if there is nothing before the cut to drop, ``None`` says so
    and the caller declines.
    """
    boundaries = [i for i in range(head, len(history)) if history[i].role == "user"]
    if not boundaries:
        return None
    chosen = boundaries[-1]  # the floor: the newest user turn always survives
    tail = 0
    for index in reversed(range(head, len(history))):
        tail += _size(history[index])
        if history[index].role == "user" and tail <= keep_chars:
            chosen = index
    return chosen if chosen > head else None


def _render(messages: Sequence[Message]) -> str:
    """The dropped region as plain text for the summarizer — roles named, tool calls named.

    The tool *names* ride along (``assistant → called: web_search``) because the summary is required
    to record the work, and a bare assistant turn often does not say which tool it drove.
    """
    blocks = []
    for message in messages:
        header = message.role
        if message.tool_calls:
            header += " → called: " + ", ".join(call.name for call in message.tool_calls)
        blocks.append(f"### {header}\n{(message.content or '').strip()}")
    return "\n\n".join(blocks)


def _summary_note(dropped: int, summary: str) -> str:
    """The single system turn that replaces the region — labelled, so the model knows what it is."""
    return (
        f"{_SUMMARY_MARKER}: {dropped} messages replaced by these notes, so this conversation stays "
        f"inside the model's context window. The detail is gone; what follows is what was kept.]"
        f"\n\n{summary}"
    )


def _size(message: Message) -> int:
    """One message's cost in characters — its text, plus the tool calls it carried.

    Images are not counted: they are evicted after the turn that showed them (`Engine`), so they are
    never part of what a *later* wake replays, which is the only thing this file governs.
    """
    total = len(message.content or "")
    for call in message.tool_calls:
        total += len(call.name) + len(json.dumps(call.arguments, default=str))
    return total


def _chars(messages: Sequence[Message]) -> int:
    """The whole transcript's cost in characters — the quantity the token estimate scales."""
    return sum(_size(message) for message in messages)
