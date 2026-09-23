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
turn can leap over it*: the persisted growth of one turn is bounded by what a tool call may leave
behind — its result (`TOOL_RESULT_CAP`, 4 KB) **and its arguments** (`TOOL_ARGS_CAP`, 2 KB) — times
`DEFAULT_MAX_STEPS` (24), ≈ 49 K tokens, which cannot cross from under-half to over-full on any
budget at or above the floor. **Those two caps are prerequisites of this file**, not neighbors of
it: relax either and the 50% threshold stops being a safe distance. That is why they are *defined*
here and merely *enforced* in `_session` — the proof and its inputs live together, where they cannot
drift apart. (The arguments were the half nobody bounded: uncapped until issue #301, they made the
right-hand side of the inequality below *unbounded*, so the guarantee it states was not merely tight
— it was untrue, and silently, since an over-long turn simply lands on the rescue.)

**And the proof has a precondition, so the harness states it out loud (issue #287).** Written as an
inequality, the guarantee is
`limit × (1 - COMPACT_AT) > (TOOL_RESULT_CAP + TOOL_ARGS_CAP) × max_steps ÷ chars-per-token` —
headroom above the threshold must exceed what one tool-heavy turn can add. Two operator knobs move
those terms: `HARNESS_MAX_CONTEXT_TOKENS` shrinks the left side, `HARNESS_MAX_STEPS` grows the right.
Either can walk an agent out of the guarantee, and **nothing about that is visible** — compaction runs
only *between* turns, so a turn under a too-small budget can cross from under-threshold to over-ceiling
in one step, and the agent simply falls back on the over-length rescue (`emergency_compact` + retry)
without anyone being told it left the primary mechanism. The escape hatch must always win (`0` still
means "compaction off"), so `_warn_if_unguaranteed` **warns and never refuses** — the defect was the
silence, not the setting. It is derived from the constants above, never hardcoded, so it cannot rot
when one of them is tuned.

**The unit of the cap is the *step*, never the call — and that is what makes the arithmetic above
true (issue #304).** A model may emit *several* tool calls in one assistant turn (parallel calls —
every model the fleet runs does), and `max_steps` bounds the model's *calls*, not the tools it
dispatched. So a per-*call* cap let a step's persisted growth scale with its fan-out, and the
right-hand side above — counting one call per step — understated the worst case by that factor,
silently and without bound. It is closed at the source rather than estimated with a fan-out constant
(a guess inside a proof is how the proof rots): a step's tool results share one `TOOL_RESULT_CAP` and
its calls' arguments share one `TOOL_ARGS_CAP`, water-filled (`_session._fill`), so **a step persists
at most `persisted_step_cap()` characters of tool payload, however wide it fans out.** It costs the
ordinary agent nothing: a lone call — the overwhelmingly common shape — gets the whole budget and is
byte-for-byte what it always was, and a wide fan-out of *small* results keeps every one of them whole,
because water-filling only takes room from the items that are over their share.

**What that bounds, and what it does not — stated, because the difference is the whole reason the
fix lives here and not in the engine.** These caps govern the payloads the harness persists *on the
model's behalf*, which is the one class of content that can be arbitrarily large **independently of
what the model wrote**: a three-token call can return a 200 KB mailbox. What is left in a step is the
model's *own* output — its assistant text, and one `id`+`name` envelope per call it emitted — and that
term the harness cannot bound and does not need to. It cannot, because the turn record must keep
**every** call the model made: drop one and `_idempotency.creates` counts a different number, and a
drifted ordinal is a message posted twice. It does not need to, because the provider already bounds it,
at every response's max-output-tokens. (This is also why capping the *dispatch* was the wrong lever for
#304, tempting as it looks: refusing to *run* a call does not un-write the call.)

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

**Compaction is a transcript concern, and it never touches memory** (founder decision, issue #561).
The transcript is *outside* the agent: the harness owns it, shows it to the model, compacts it when it
is too big, and deletes it with its timeline — the summary is part of the transcript and goes with it.
Memory is *inside* the agent: what it chose to write, or (on a mining provider) what was said. The two
systems are never connected, so nothing here calls, reads, or holds a memory provider. The summary
records the work first because the model's *next turn on this timeline* needs to know what it did; the
durable trace of tool work lives outside the agent entirely (the per-call ``tool`` log line, and the
platform itself). A summary is **validated before it is committed** — it must be smaller than what it
replaces, or the compaction declines — and the identifiers in the dropped region are **harvested by
code** and appended to it (`_identifier_block`), so a uuid, a URL or a handle survives whether or not
the summarizer copied it.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from basecradle_harness._engine import DEFAULT_MAX_STEPS
from basecradle_harness._messages import Message
from basecradle_harness._observability import kv
from basecradle_harness._provider import Provider

_log = logging.getLogger("basecradle_harness")

#: The most characters of tool results **one step** persists into the transcript — shared across
#: every call that step made, not granted to each of them (issue #304). Over it, a result is elided
#: to head + tail around a marker naming the original size (`_session._elide`). The model still saw
#: the *whole* result on the turn the tool ran — this bounds only what every *future* turn re-reads
#: and re-pays for. 4 KB is what the live containment prune used on @glm-5.2's transcripts (issue
#: #275): comfortably more than a normal tool answer, far below the 142 KB mailbox dumps that drove
#: one agent's context to 754 K input tokens per call.
#:
#: **Per step, because a step's fan-out is unbounded and `max_steps` is not a bound on it.** A lone
#: call — the ordinary shape — gets the whole budget and is elided exactly as it always was; a step
#: that fans out shares it, water-filled, so its *small* results survive whole and only the fat ones
#: pay (`_session._fill`).
#:
#: **It lives here, not in `_session` where it is enforced**, because the compaction threshold's
#: safety proof is computed from it (see the module docstring): it is an *input to the arithmetic*,
#: and a number the proof depends on must not sit in another file where it can be tuned without the
#: proof noticing.
TOOL_RESULT_CAP = 4096

#: The most characters of tool-call **arguments** that **one step** persists into the transcript —
#: shared across every call that step made, for the same reason `TOOL_RESULT_CAP` is (issue #304).
#: Over it, every argument gets a fair share of the budget and the ones that overflow theirs are cut
#: to a head and a tail around a marker (`_session._cap_arguments`) — so the short arguments survive
#: whole and the blob pays for itself. The model sent the *whole* arguments when the call ran; this
#: bounds only what every *future* turn re-reads. **Characters as the model reads them, never as the
#: disk escapes them** (`_session._json_size`) — a Japanese character costs one, not six.
#:
#: **This was the one class of persisted content with no bound at all** (issue #301): the brief is
#: never persisted, tool results are capped, images are evicted, the conversation is compacted — and
#: a tool call's arguments were written whole and replayed to the model on every wake, forever. An
#: `assets create` carrying a 200 KB document was a 200 KB tax on the life of the timeline.
#:
#: Half the result cap, and the asymmetry is deliberate: a *result* is content the model **received**
#: and may need to re-read, while an *argument* is content it **wrote** — it knows what it asked for,
#: the elision marker names the size, and the artifact it created is on the platform to be re-read.
#: 2 KB still keeps the ordinary call whole (a long message body runs a few hundred characters), so
#: an agent's memory of its own speech is untouched and only the blob is bounded.
#:
#: It lives here for the same reason `TOOL_RESULT_CAP` does: the compaction threshold's safety proof
#: is computed from it, so it is an *input to the arithmetic* and must not sit in another file where
#: it can be tuned without the proof noticing.
TOOL_ARGS_CAP = 2048

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

#: The chars-per-token assumed when sizing the **worst case** a single turn can add to the
#: transcript (`worst_case_turn_tokens`). It is deliberately *not* the ~4 chars/token that ordinary
#: prose runs at, because the content being sized is not prose: it is **tool output** — JSON, uuids,
#: file paths, log lines, base64-adjacent junk — which every tokenizer splits far more finely than
#: English. 3.0 is the conservative middle between prose (~4) and the emergency path's deliberately
#: pessimistic 2.0, and conservative is the correct direction here: assuming *fewer* chars per token
#: makes the estimated worst case *larger*, so the harness warns early rather than late. A too-loud
#: warning costs a log line; a too-quiet one costs the guarantee it exists to protect.
WORST_CASE_CHARS_PER_TOKEN = 3.0

#: The most of an over-ceiling transcript the emergency path may retain, as a fraction of what is
#: actually there. The provider has just *refused* this transcript, so any conclusion our token
#: arithmetic reaches about it is already known to be wrong — including, dangerously, "it fits."
#: Capping the tail against the transcript's real size guarantees the rescue always finds a region
#: worth replacing instead of declining and leaving the agent bricked — it can still decline if the
#: summarizer answers with notes no smaller than that region (issue #561), which a rescue must never
#: commit. The provider's word beats our estimate.
EMERGENCY_KEEP_RATIO = 0.25

#: The most characters the harvested-identifier block may add to a summary note, heading included
#: (`_identifier_block`). The same 4 KB as `TOOL_RESULT_CAP`, and for the same reason: the note is
#: replayed on every wake until the next compaction, so what it carries is bounded, never "small".
#: Over it, the identifiers **mentioned least recently** are the ones left out, and the block says how
#: many — compaction keeps the recent verbatim and lets the old go, and the harvest follows that grain.
IDENTIFIER_CAP = 4096

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
#: dropped region takes its tool results with it, and the model's next turn on this timeline has
#: to know what it already did — what it posted, what it produced, what failed — or it repeats the
#: work or contradicts it. That is a *transcript* concern, not a memory one (issue #561): the summary
#: lives in the transcript and nowhere else. Raw tool output is *not* preserved: the point is a
#: record of what was done, not a second copy of the bytes we are dropping.
#:
#: The headings are **operational** — what happened, what holds, what is underway, what comes next —
#: rather than a recap of what was said, which invites transcription instead of continuity. And the
#: cumulative case is spelled out: the previous summary rides in with the excerpt, and an item carried
#: forward only because it was written down before is how a summary fills up with resolved work.
#:
#: Rewording this does not orphan an already-polluted palace: the scrub catalog names the wording
#: that was mined before issue #438 closed the path, as a historical literal (`_mining`).
_SUMMARIZE_INSTRUCTION = """You are compacting your own conversation transcript to stay inside your context window. \
The excerpt below is about to be deleted and replaced by what you write now. Write dense, factual \
notes to your future self, in the first person, under these five headings, in this order:

1. WORK DONE — the actions you actually took, the tools you used, and what came of them: artifacts \
produced (asset uuids, URLs, file paths, task uuids), things posted, things changed, things that \
failed. Tool results are deleted along with the excerpt, so an action you do not write down here is \
one your next turn will not know you took.
2. DECISIONS AND FACTS — what was decided and by whom, what was promised, and the facts you learned \
that you will still need.
3. IN PROGRESS — what you were in the middle of when the excerpt ends.
4. NEXT ACTION — what you should do next, if anything.
5. OPEN THREADS — what is unfinished, what you owe someone, and what is waiting on someone else.

If the excerpt opens with your notes from an earlier compaction, fold them in: carry forward what \
still matters, and drop what is resolved or obsolete. Do not keep an item only because it was \
written down before.

Preserve identifiers (uuids, URLs, handles, numbers) verbatim. The identifiers in the excerpt are \
also listed, automatically, after your notes — so do not list them bare; name the ones that matter \
next to what they are. Do not speculate, do not pad, and do not address anyone: these are your own \
notes, not a message. Everything you leave out is forgotten."""

#: The heading the harvested identifiers ride under, below the model's summary and inside the same
#: system turn (`_identifier_block`). Fixed text, so the model can tell code-kept identifiers from
#: the notes it wrote itself.
_IDENTIFIER_HEADING = "IDENTIFIERS (harvested, verbatim):"

#: What `_identifier_block` harvests from the dropped region, in one alternation so a match's
#: position is its first-seen order and a URL claims the uuid or handle inside it. Three classes, each
#: the shape the thing *actually* has, because a harvested identifier that is not verbatim is worse
#: than none — it is a confident wrong answer the model will act on:
#:
#: - **URLs** are a *positive* class: the ASCII characters RFC 3986 allows in a URL, less the square
#:   brackets (legal only around an IPv6 host, and the seam of every Markdown link). Anything else ends
#:   the match — a backslash (``\n`` or ``\"`` in a JSON-escaped body), a pipe, a quote, a CJK
#:   character or an em dash the URL runs straight into. Trailing prose punctuation is trimmed after
#:   the match (`_trim_url`).
#: - **UUIDs**, lowercase, as the platform writes them.
#: - **Handles** in the platform's own grammar (`HandleValidator`: a letter or digit, then segments of
#:   ``[a-z0-9_-]`` joined by single periods). A narrower class reads ``@glm-5.2`` as ``@glm-5`` — a
#:   real fleet handle, cut into a wrong one. The lookbehind skips the domain half of an email; the
#:   lookahead refuses a match that stops inside a longer word (``@johnDoe`` is not ``@john``).
#:   Spelled ``[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9_-]+)*`` — the same language as the validator's
#:   ``(?:\.?[a-z0-9_-]+)*`` and never that spelling, which is the ``(a+)*`` shape: when the lookahead
#:   refuses, the engine retries every way of splitting a run of letters into segments, so a peer's
#:   ``@`` and forty lowercase letters followed by a capital would hang the compaction for hours. Every
#:   repetition here must open on a period, so a run splits exactly one way.
#:
#: **Every boundary is ASCII, never ``\b`` or ``\w``.** Python's are Unicode-aware, so a kana or a
#: CJK ideograph counts as a word character, and ``アセット019e…を確認`` or ``请@john看`` — ordinary
#: Japanese and Chinese, which put no space there — harvested nothing. An agent that keeps a peer's
#: identifiers in English and loses them in Japanese is the defect `_session._json_size` records.
_IDENTIFIER = re.compile(
    r"https?://[A-Za-z0-9\-._~:/?#@!$&'()*+,;=%]+"
    r"|(?<![0-9A-Za-z_])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9A-Za-z_])"
    r"|(?<![A-Za-z0-9_.+-])@[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9_-]+)*(?![A-Za-z0-9_])"
)

#: What follows an identifier the harness itself cut short: the ellipsis a list preview ends on
#: (``_reads``, ``_tasks``, ``_webhooks``) and the head of an archived excerpt's elision marker
#: (``_session``) — in both spellings, because an elided *argument* reaches `_render` inside JSON,
#: where its blank line is the four characters ``\n\n``. A match that runs into one is a
#: **fragment** — ``https://example.com/some/pa`` or ``@nova-dig`` — and is dropped: a missing
#: identifier is a gap, a truncated one is a wrong answer.
_CUT_SHORT = ("…", "\n\n[... ", "\\n\\n[... ")

#: Punctuation a URL match may have swallowed from the prose around it — ``see https://x.com/a.``,
#: ``'https://x.com/a'`` in a Python repr — and the closer that is part of the URL only when the URL
#: also opened one.
_URL_TRAILING = ".,;:!?*')"
_URL_PAIRS = {")": "("}


def persisted_step_cap() -> int:
    """What one **step** may leave behind in the transcript, in characters — however wide it fans out.

    Both halves of a step's tool payload, because both persist and both are replayed on every future
    wake: the **results** its calls returned (`TOOL_RESULT_CAP`) and the **arguments** it called them
    with (`TOOL_ARGS_CAP`).

    **The step is the unit, not the call** (issue #304). A model may emit several calls in one
    assistant turn, and `max_steps` bounds the model's calls, not the tools dispatched — so a per-call
    cap scaled a step's growth by a fan-out nothing bounds, and left the arithmetic below understating
    the worst case without limit. `_session` shares each budget across the step's calls instead.

    Counting only the result — as this arithmetic did until issue #301 — was likewise not a tight
    estimate but a wrong one, because the arguments were not capped *at all*: the term it omitted was
    unbounded. Both mistakes have the same shape, and this function is where they stay fixed.
    """
    return TOOL_RESULT_CAP + TOOL_ARGS_CAP


def worst_case_turn_tokens(max_steps: int) -> int:
    """The most one turn can add to the persisted transcript, in tokens — the proof's right-hand side.

    Every step of the think→act loop may run tools, and a step persists at most `persisted_step_cap()`
    characters of results-plus-arguments **whatever its fan-out** (issue #304). So one turn's worst-case
    persisted growth is that cap times the step budget, converted at `WORST_CASE_CHARS_PER_TOKEN`.
    **Derived, never hardcoded** — the whole point is that tuning either cap or the step budget moves
    this number automatically, so the compaction threshold's safety argument can never quietly go stale
    behind a literal.

    What it counts is the *tool payload* a step persists — the class the harness bounds. The model's own
    output in that step (its text, and one id+name envelope per call it emitted) is bounded by the
    provider's max-output-tokens, not by anything here; see the module docstring for why that term is
    the provider's to bound and not ours.
    """
    return math.ceil(persisted_step_cap() * max_steps / WORST_CASE_CHARS_PER_TOKEN)


def min_safe_limit(max_steps: int) -> int:
    """The smallest context budget that still guarantees no single turn can leap the threshold.

    Solve `limit × (1 - COMPACT_AT) >= worst_case_turn_tokens(max_steps)` for `limit`. At the shipped
    constants (4 KB result + 2 KB arguments per step, 24 steps, half-ceiling trigger) this is ~98 K —
    under the 128 K floor, which is why an agent that never touches the knobs is safe by construction
    and never hears a word about any of this. **That margin is what sizes `TOOL_ARGS_CAP`**: at a 4 KB
    argument cap this lands at 131,072, *above* the floor, and every stock agent would start warning.

    **This budget is itself safe** — it is the smallest value that *clears* the bar, not the largest
    that fails it. So the warning quotes it as "to at least N", never "above N": `threshold()` floors
    with `int()`, which hands a token back, so N (and in fact N-1) is already silent. Telling an
    operator to exceed a number that already works is the kind of small dishonesty that erodes trust
    in the whole warning.
    """
    return math.ceil(worst_case_turn_tokens(max_steps) / (1.0 - COMPACT_AT))


def max_safe_steps(limit: int) -> int:
    """The largest step budget a given ceiling can sustain — the same inequality solved the other way.

    The remedy for a *small* ceiling is not "raise the ceiling": for an adapter-reported or floor
    ceiling that is the model's real window, raising `HARNESS_MAX_CONTEXT_TOKENS` above it would move
    the threshold past the wall and compaction would never fire in time — strictly worse. There the
    honest fix is to spend fewer steps per turn, so the warning quotes this number instead.
    """
    return int(limit * (1.0 - COMPACT_AT) * WORST_CASE_CHARS_PER_TOKEN / persisted_step_cap())


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
        max_steps: The engine's **effective** per-turn step budget (`HARNESS_MAX_STEPS`, else
            `DEFAULT_MAX_STEPS`). Used for one thing: sizing the worst-case single-turn growth the
            50% threshold is proved against (`_warn_if_unguaranteed`). It is the *effective* budget
            deliberately — an operator who raises `HARNESS_MAX_STEPS` erodes the very same guarantee
            that lowering `HARNESS_MAX_CONTEXT_TOKENS` erodes, from the other side of the
            inequality, and a warning that only watched one of the two terms would be half a guard.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        override: int | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self._provider = provider
        self._override = override
        self._max_steps = max_steps
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
        self._warn_if_unguaranteed(self._limit)
        return self._limit

    def _warn_if_unguaranteed(self, limit: Limit) -> None:
        """Say so, once, when this budget cannot sustain the no-single-turn-can-leap guarantee.

        The 50% threshold is only a *safe* distance while one turn's worst-case growth fits in the
        headroom above it (see the module docstring). Below that, a tool-heavy turn can cross from
        under-threshold to over-ceiling in one step — compaction runs only *between* turns and never
        gets a chance — and the agent lands on the over-length rescue (`emergency_compact` + retry)
        instead. The rescue works. The defect is that the operator was **never told** they had
        dropped from the primary mechanism to the safety net (issue #287: we did this to @pinky
        ourselves, at `HARNESS_MAX_CONTEXT_TOKENS=20000`, and the harness said nothing).

        **Warn, never refuse.** The override is the 2 a.m. escape hatch and must always win — the
        same reason `0` is honored as "compaction off". This only makes the cost audible.

        Keyed on the **arithmetic, not the source**, and the remedy is keyed on the source:

        - **`env`** — the operator picked this number, so raising it is a real option, and the
          minimum that restores the guarantee is quoted.
        - **`adapter` / `default`** — that ceiling is the model's actual window (or a conservative
          floor below it). Raising `HARNESS_MAX_CONTEXT_TOKENS` past a real ceiling would push the
          threshold *beyond the wall* and compaction would never fire in time — strictly worse than
          the problem. So the remedy quoted there is the step budget instead.

        The floor (128 K) satisfies the inequality by construction and is silent, always. An adapter
        is silent whenever it reports a ceiling that clears the bar — which every model the fleet
        runs today does. It is **not** silent for a genuinely small-context model (a local model, a
        budget endpoint), and that is deliberate: an operator who *did not choose* the dangerous
        budget has even less reason to guess they are running without the guarantee than one who did.
        """
        worst_case = worst_case_turn_tokens(self._max_steps)
        headroom = limit.tokens - self.threshold()
        if headroom >= worst_case:
            return
        steps = max_safe_steps(limit.tokens)
        if limit.source == "env":
            # "to at least N", never "above N": N itself clears the bar (`min_safe_limit` is the
            # smallest budget that satisfies the inequality, and the test pins that it is silent
            # there). "Above" would send an operator chasing a number one higher than they need.
            fixes = [
                f"raise HARNESS_MAX_CONTEXT_TOKENS to at least {min_safe_limit(self._max_steps)}"
            ]
            if steps >= 1:
                fixes.append(f"lower HARNESS_MAX_STEPS to {steps}")
            remedy = f"To restore the guarantee, {' or '.join(fixes)}."
        else:
            # Deliberately never phrased as "raise the budget above N". This ceiling is the model's
            # real window, and an operator skimming a log for an actionable number could act on that
            # phrasing and push the threshold *past the wall* — turning a warning into the outage it
            # exists to prevent. The only actionable number offered here is the step budget.
            remedy = (
                f"Lower HARNESS_MAX_STEPS to {steps} to restore the guarantee. (Do not raise "
                f"HARNESS_MAX_CONTEXT_TOKENS to clear this warning: it is the model's own ceiling, "
                f"and a budget above it would move the threshold past the wall — compaction would "
                f"then never fire in time, which is worse than the warning.)"
                if steps >= 1
                else "No step budget satisfies the guarantee at this ceiling."
            )
        _log.warning(
            "context budget %d (source=%s) leaves %d tokens of headroom above the compaction "
            "threshold, below the %d a single tool-heavy turn can add (%d chars of result + "
            "arguments per step x %d steps at %.1f chars/token): a turn may overshoot the ceiling "
            "before compaction, which runs only *between* turns, can fire. The over-length rescue "
            "(emergency compaction + retry) still applies. %s",
            limit.tokens,
            limit.source,
            headroom,
            worst_case,
            persisted_step_cap(),
            self._max_steps,
            WORST_CASE_CHARS_PER_TOKEN,
            remedy,
        )

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

    It touches the transcript and nothing else — never a memory provider (issue #561; see the module
    docstring). The summary it writes lives in the transcript, is replayed with it, and is deleted
    with it when the timeline goes.

    Args:
        provider: The model. Used for the summarization call (`tools=None` — the summarizer needs
            no tools) and for reading back the usage it reported (`provider_tokens_in`).
        budget: The resolved ceiling and threshold (`ContextBudget`).
    """

    def __init__(self, provider: Provider, budget: ContextBudget) -> None:
        self.provider = provider
        self.budget = budget

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
        replaced = _chars(dropped)
        # **A replacement that is not smaller is not a compaction.** A summarizer that echoes the
        # excerpt back, or pads, would grow the very transcript this exists to shrink — and on the
        # emergency path hand the retry a request *longer* than the one the provider just refused.
        # Measured in this file's own unit. Checked first against the note's fixed opening, which
        # costs nothing to know: a region that cannot beat even that is not worth a summarize call.
        if len(_summary_note(len(dropped), "")) >= replaced:
            _log.warning(
                "Context compaction declined: the %d chars it would replace are no more than the "
                "summary note's own heading. The transcript is unchanged.",
                replaced,
            )
            return False
        summary = self._summarize(dropped, limit=limit, chars_per_token=chars_per_token)
        if summary is None:
            return (
                False  # the summarize call failed; _summarize logged it. Leave the transcript be.
            )
        bare = _summary_note(len(dropped), summary)
        # The identifier block is part of the note, so it counts — and it **degrades before the
        # compaction declines**: it gets whatever room the region leaves under the summary (at most
        # `IDENTIFIER_CAP`), because an optional appendix must never be the reason a transcript that
        # needs compacting stays long. The two characters are the blank line that joins it.
        identifiers = _identifier_block(
            _render(dropped), cap=min(IDENTIFIER_CAP, replaced - 1 - len(bare) - 2)
        )
        note = Message.system(_summary_note(len(dropped), summary, identifiers))
        written = message_chars(note)
        if written >= replaced:
            _log.warning(
                "Context compaction declined: the summary (%d chars) is not smaller than the %d "
                "chars it would replace. The transcript is unchanged.",
                written,
                replaced,
            )
            return False
        note.items = _carried_items(dropped)
        history[head:cut] = [note]
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
        carried = _render([m for m in dropped if is_summary(m)])
        excerpt = _render([m for m in dropped if not is_summary(m)])
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
        index < len(history) and history[index].role == "system" and not is_summary(history[index])
    ):
        index += 1
    return index


def is_summary(message: Message) -> bool:
    """Is this the system turn a previous compaction left behind? (See `_prelude_end`.)

    Public because the context-attribution line reports summarized conversation as its own
    section (`_attribution`), and it must answer the question **the same way this file does** —
    off `_SUMMARY_MARKER`, the marker the compactor itself writes. A second spelling of "is this
    a summary?" living in the reporting module would drift from the one that governs behavior,
    and the log would then attribute a section that compaction does not recognize.
    """
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

    **An `injected` turn is not a boundary** (issue #297), and that is a correctness rule, not a
    tidiness one. The engine and the code bridge both append `user`-role turns that are a turn's own
    *work* — an image for the model to look at, a note naming the Assets a code run produced. Both
    are eligible cut points if you go by role alone, and the newest one can therefore become the
    "newest user turn always survives" **floor** — at which point the compaction keeps the caption
    and summarizes the peer's actual message away. That leaves a valid transcript (the assistant
    turn and its tool results are dropped together, so nothing dangles) and a *broken agent*: the
    recovery classifier can no longer find the turn that carried a message, so a wake killed while
    holding it re-drives a turn that already posted, and the peer is answered twice.
    """
    boundaries = [i for i in range(head, len(history)) if _is_boundary(history[i])]
    if not boundaries:
        return None
    chosen = boundaries[-1]  # the floor: the newest real user turn always survives
    tail = 0
    for index in reversed(range(head, len(history))):
        tail += message_chars(history[index])
        if _is_boundary(history[index]) and tail <= keep_chars:
            chosen = index
    return chosen if chosen > head else None


def _is_boundary(message: Message) -> bool:
    """May the retained tail begin here? Only at a **real** user turn — see `_cut_index`."""
    return message.role == "user" and not message.injected


def _carried_items(dropped: Sequence[Message]) -> list[str]:
    """Every platform item whose turn this compaction is about to destroy.

    **Compaction summarizes the conversation away; it may never *erase the recovery's evidence*.**
    The delivery guarantee's classifier reads one thing off the transcript — *is there a turn
    carrying this item?* — and treats "no" as proof that the dead wake never reached the model, which
    licenses a **re-drive**. Compaction can make that inference a lie: it replaces a whole region of
    `history` with a single summary, so a turn that ran tools — that posted, that generated an image
    at fal.ai — can simply cease to exist while the item's claim is still `in-flight`. The next wake
    then re-drives a turn that already spoke, re-firing every tool in it. Nothing errors, and the
    peer is answered twice (or, worse, the idempotency key hands back the *original* record, so the
    model's new message is swallowed and nobody is answered at all).

    So the summary **inherits the uuids of the turns it drops**, and the recovery reads them
    (`_wake._evidence_lost`): a missing turn whose uuid is *here* was seen and its outcome is now
    unknowable — a loud, rare abandon — while a missing turn whose uuid is nowhere was genuinely
    never sent, and re-drives exactly as before. A previous summary sits inside the dropped region,
    so its uuids carry forward with the rest; `dict` dedupes and keeps them in order.

    **What bounds this** (Context Discipline): the uuids cost no tokens — `items` is persisted but
    never sent to a provider — and `_wake._forget_settled` prunes every uuid whose claim has since
    reached a final phase, which is all of them within a wake or two. What remains is the handful of
    items genuinely still in flight, which is the only thing the recovery will ever ask about.
    """
    carried: dict[str, None] = {}
    for message in dropped:
        for uuid in message.items:
            carried[uuid] = None
    return list(carried)


def _render(messages: Sequence[Message]) -> str:
    """The dropped region as plain text for the summarizer — roles named, tool calls in full.

    The tool *names* ride along (``assistant → called: web_search``) because the summary is required
    to record the work, and a bare assistant turn often does not say which tool it drove.

    **And so do their arguments, one line per call — because that is where the agent speaks.** Since
    the final-text auto-post was removed (issue #293), everything an agent says to anyone is the
    ``body`` of a ``messages`` call, and the call's result says only that it posted and the new uuid.
    Rendering names alone therefore showed the summarizer every peer's words and **none of the
    agent's own**: it could record that a message went out, never what it said or promised — and the
    identifier harvest, which reads this same text, could never keep a URL the agent sent. Arguments
    are bounded exactly as the transcript bounds them on disk (`_session._fill` over `TOOL_ARGS_CAP`
    per step), so a 200 KB document an agent posted costs the summarizer an excerpt, not its budget.
    """
    blocks = []
    for message in messages:
        header = message.role
        if message.tool_calls:
            header += " → called: " + ", ".join(call.name for call in message.tool_calls)
        lines = [f"### {header}", (message.content or "").strip()]
        lines += [
            f"{call.name} {json.dumps(arguments, ensure_ascii=False, default=str)}"
            for call, arguments in zip(message.tool_calls, _bounded_arguments(message))
        ]
        blocks.append("\n".join(line for line in lines if line))
    return "\n\n".join(blocks)


def _bounded_arguments(message: Message) -> list[dict]:
    """A turn's call arguments as the transcript would persist them — one step, one shared budget.

    Imported at call time: `_session` imports this module for the caps, so a module-level import back
    would be a cycle, and the summarizer is the only caller.
    """
    if not message.tool_calls:
        return []
    from basecradle_harness._session import _cap_arguments, _fill, _json_size

    return _fill(
        [call.arguments for call in message.tool_calls],
        TOOL_ARGS_CAP,
        size=_json_size,
        elide=_cap_arguments,
    )


def _summary_note(dropped: int, summary: str, identifiers: str = "") -> str:
    """The single system turn that replaces the region — labelled, so the model knows what it is.

    The harvested identifiers ride **below** the model's summary and inside this same turn, so the
    transcript's shape is exactly what it was without them: one system message per compaction, opening
    on `_SUMMARY_MARKER`.
    """
    note = (
        f"{_SUMMARY_MARKER}: {dropped} messages replaced by these notes, so this conversation stays "
        f"inside the model's context window. The detail is gone; what follows is what was kept.]"
        f"\n\n{summary}"
    )
    return f"{note}\n\n{identifiers}" if identifiers else note


def _identifier_block(rendered: str, cap: int = IDENTIFIER_CAP) -> str:
    """The identifiers in the dropped region, harvested by code — ``""`` when there are none.

    `_SUMMARIZE_INSTRUCTION` asks the model to keep identifiers verbatim, and a prompt is a request,
    not a guarantee: a uuid the summarizer paraphrased, truncated or left out is gone for good, and it
    is exactly the thing the next turn needs to act on (the asset to read, the task to finish, the
    peer to answer). So the identifiers are also kept **deterministically** — a regex over the same
    rendering the summarizer read, no second model call, nothing to retry — and appended under
    `_IDENTIFIER_HEADING`, one per line, deduplicated, in the order they first appeared.

    **Bounded at `cap`, heading and all** — `IDENTIFIER_CAP`, or less when the note has only that
    much room left under the size of the region it replaces (`Compactor._compact`). A busy region
    (mailbox listings, timeline reads) carries hundreds of uuids, so the cap is expected to bite, and
    what it gives up is chosen rather than truncated: the list is filled **most recently mentioned
    first** (an identifier's last position in the region; one carried in a previous block counts as
    mentioned where that block lists it), an identifier too long for the room left is passed over for
    shorter ones behind it, the kept ones are written in first-seen order, and a closing line says how
    many are missing. A cap too small to hold the heading and that closing line yields ``""`` — an
    unannounced partial list is the one outcome this never produces.
    """
    last: dict[str, int] = {}  # identifier → where it was last seen; insertion order is first-seen
    for match in _IDENTIFIER.finditer(rendered):
        if rendered.startswith(_CUT_SHORT, match.end()):
            continue  # a fragment the harness itself truncated — see `_CUT_SHORT`
        found = match.group()
        if found.startswith("http"):
            found = _trim_url(found)
            if found.partition("://")[2] == "":
                continue
        last[found] = match.start()
    if not last:
        return ""
    block = "\n".join((_IDENTIFIER_HEADING, *last))
    if len(block) <= cap:
        return block
    # Room for the identifier lines once the heading and the worst-case closing line are paid for;
    # every kept line costs its length plus the newline that joins it.
    room = cap - len(_IDENTIFIER_HEADING) - len(_not_kept(len(last), cap)) - 1
    if room < 0:
        return ""
    kept: set[str] = set()
    for found in sorted(last, key=last.__getitem__, reverse=True):
        if len(found) + 1 <= room:
            kept.add(found)
            room -= len(found) + 1
    lines = [found for found in last if found in kept]
    return "\n".join((_IDENTIFIER_HEADING, *lines, _not_kept(len(last) - len(lines), cap)))


def _not_kept(count: int, cap: int) -> str:
    """The closing line of a capped identifier block — the cap says what it cost, never silently."""
    noun = "identifier" if count == 1 else "identifiers"
    return (
        f"[{count} {noun} not kept: this list is capped at {cap} characters, and the least recently "
        f"mentioned are left out first.]"
    )


def _trim_url(url: str) -> str:
    """A matched URL without the prose punctuation it swallowed — ``https://x.com/a).`` → the URL.

    A closing bracket is kept only when the URL also opened one (``…/Foo_(bar)`` is a real path),
    so a Markdown link's ``)`` goes and a Wikipedia title's stays.
    """
    # Counted once and walked by index: the tail is peer-controlled, and re-counting (or re-slicing)
    # per stripped character is quadratic in a URL followed by a hundred thousand ``)``.
    opened = {closer: url.count(opener) for closer, opener in _URL_PAIRS.items()}
    closed = {closer: url.count(closer) for closer in _URL_PAIRS}
    end = len(url)
    while end and url[end - 1] in _URL_TRAILING:
        tail = url[end - 1]
        if tail in closed:
            if closed[tail] <= opened[tail]:
                break
            closed[tail] -= 1
        end -= 1
    return url[:end]


def message_chars(message: Message) -> int:
    """One message's cost in characters — its text, plus the tool calls it carried.

    Images are not counted: they are evicted after the turn that showed them (`Engine`), so they are
    never part of what a *later* wake replays, which is the only thing this file governs.

    `ensure_ascii=False` for the same reason `_session._json_size` uses it, and to say the same thing in
    the same unit: what a character costs the model does not depend on the script it is written in. With
    the default, this counted a message's *text* raw and its tool call's *arguments* escaped — one
    function, two units, and a sixfold overcount of any non-Latin argument.

    Public because the context-attribution line (`_attribution`) measures the assembled payload
    section by section and must do it in **this** unit. Two functions answering "how big is this
    message?" would be two units again — the exact defect the paragraph above records — and here
    the attribution would silently disagree with the compaction arithmetic it exists to explain.
    """
    total = len(message.content or "")
    for call in message.tool_calls:
        total += len(call.name) + len(json.dumps(call.arguments, ensure_ascii=False, default=str))
    return total


def _chars(messages: Sequence[Message]) -> int:
    """The whole transcript's cost in characters — the quantity the token estimate scales."""
    return sum(message_chars(message) for message in messages)
