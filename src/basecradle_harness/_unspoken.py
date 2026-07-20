"""The Unspoken Channel: an agent speaks only by deciding to (issue #293).

**By default, nothing an agent generates touches a timeline.** Every timeline interaction is
an intentional tool call (`messages`, `assets`, `tasks`); everything else the model emits —
its final free text, the turn's narration — is **unspoken**: written to the agent's log
(`log_unspoken`, INFO, `unspoken=`), never posted, never seen by a peer.

**The log is a flight recorder, not a control tower** (the founder's correction, 2026-07-14, and
it is load-bearing for every string in this file that the model reads). Nobody watches it. These
agents have no human supervisor — *the agent is its own operator* — and the record is dug into on
the rare day something breaks. Its primary reader is the agent's **own future self, through
memory** (`_wake._observe` hands every engaged turn to the memory provider, spoken or silent); its
secondary reader is a forensic dig that may never happen. That is the "full visibility" half of the
founder's principle: **auditable when needed, not observed.**

The trap this closes is specific and it is why the wording matters more than the mechanism: a model
told that "your operator reads your log" will *report* to it — an escalation, a blocker, a warning —
and walk away believing it has communicated. **An escalation written only into an unread log is a
message to no one.** So the guidance every agent reads (`_defaults/prompts/initialize.md`) says the
opposite, plainly: assume nobody will ever read this; if it matters to anyone else, speak on a
timeline, or it reached no one.

**Why the default was inverted.** The harness used to auto-post a turn's final text as the
reply. That reply channel was implicit and documented nowhere the model could see it, while
capable agentic models arrive with the *opposite* prior — tool calls act, final text is
private narration. The collision was exact and it was measured: every turn in which an agent
posted through the `messages` tool **also** auto-posted its narration — a double post, 100%
of the time (@glm-5.2, ~50 occurrences). Told to "post exactly one message," @briggs entered a
loop — the engine only ends a turn on a no-tool-call text turn, so "the single reply" posted 11
times in ~100 seconds until the timeline was locked. Five of seven fleet agents were clean only
because they had never discovered the tool. The founder's principle governs the fix:

    "We do everything we can to be sure the AI understands in a clear and concise way, we never
    force its action or inaction, but we do require full visibility which is the price of that
    freedom."

So: one speaking channel (the tools), one thinking channel (unspoken, logged), and **no forcing
anywhere**. Silence is a first-class answer — and because the reasoning behind it is always on
the record, it is a *visible* one.

Four things live here, and they are small on purpose:

- `SpeechLedger` — what this wake actually *did* to a timeline. The platform tools record into
  it (see `_platform.PlatformContext`), so both the wake's bookend line and the informer below
  read one honest answer to "did the agent act?" rather than inferring it.
- `NoReplyInformer` — the deterministic backstop. It informs an agent that ended a turn having
  done nothing on the timeline when the message *structurally* called for it. It informs; it
  never forces.
- `addressed` — the exact-string `@handle` mention test, one of the two conditions the informer
  turns on.
- `is_one_on_one` — the two-viewer test, the *other* condition: on a timeline whose live viewer
  set is this agent plus exactly one other, whatever the counterpart says is, by the shape of
  the room, said to this agent (issue #332).
"""

from __future__ import annotations

import re

from basecradle_harness._messages import Message

#: Everything the two nudges share, **verbatim** — the whole nudge except its opening clause. The
#: informer flags a turn for one of two reasons (an `@handle` mention, or a one-on-one), and the
#: only thing that differs between them is that first sentence naming *why*; the rest — silence is
#: legitimate, no one forces you, leave the reason for your own memory, to speak is to call the
#: `messages` tool — is identical. It is factored out so the "everything after the opening is
#: verbatim" invariant (issue #332) is *structural*, not two long strings kept in sync by hand.
#:
#: **It informs; it does not gate** — the model may end the next turn in silence too, and nothing
#: stops it. Deliberately not a forcer, and the wording carries that: "may be exactly right", "no
#: one will force you out of it". A nudge the model reads as a command would re-create the very
#: defect the inversion removed — a harness that decides when an agent speaks.
#:
#: **And it must not invent a reader** (the founder's correction, 2026-07-14). An earlier draft
#: said the reason goes "to your log, where your operator reads it". There is no operator: these
#: agents have no human supervisor, and the log is a flight recorder — dug into after a failure,
#: never watched. A model that believes someone reads its log will *report* things there — an
#: escalation, a blocker, a warning — and walk away believing it communicated. It did not. So the
#: reason is framed as what it truly is: a note for the record and for the agent's own future self,
#: which nobody is waiting on.
#:
#: **And it must name the verb** (issue #295). The first draft ended at "act now" — and a small
#: model does not reliably map "act" onto the tool. @jt (gpt-5.4-mini), asked outright which
#: version it was running, *composed the right answer and narrated it*: `posted=0`,
#: `text="I'm running 0.67.0."` The nudge fired, and its answer to the nudge was more narration.
#: It was not disobeying and it was not confused about the question — it believed it had answered.
#: The capable cohort (@briggs, @glm-5.2) mapped "act" to the `messages` tool on day one, which is
#: exactly what makes this a *guidance* gap and not a plumbing one: the channel worked, and the
#: model could not see where it was. So the sentence names the mechanism and its absence in the
#: same breath — call the tool; text here reaches no one — which is the whole trade the unspoken
#: channel makes, said once, at the only moment it bites.
_NUDGE_TAIL = (
    " That may be exactly right — silence is a legitimate choice, and no one will force you out "
    "of it. If the silence is deliberate, leave your reason in your unspoken text — for the "
    "record and for your own memory; no one is waiting on it. If it is not deliberate, act now — "
    "speaking means calling the `messages` tool; text written here reaches no one."
)

#: The system turn appended when the agent was addressed by its own `@handle` and its turn is about
#: to end having done nothing. The opening names the reason (a mention); the rest is `_NUDGE_TAIL`.
MENTION_NUDGE = (
    "You were mentioned by name in what you just read, and this turn is about to end with "
    "nothing posted, shared, or done on the timeline." + _NUDGE_TAIL
)

#: The system turn appended when the agent is the *only other party* on a two-viewer timeline and
#: its turn is about to end having done nothing (issue #332). A one-on-one message is structurally
#: addressed to the agent even without an `@handle`: it is the whole audience, so its silence means
#: the peer's message reached no one who will act. The opening is the founder's wording; the rest is
#: `_NUDGE_TAIL`, byte-identical to the mention nudge (both founder corrections above therefore carry
#: over for free: no invented reader, and the `messages` tool named).
ONE_ON_ONE_NUDGE = (
    "You are the only other party in this conversation, and this turn is about to end with "
    "nothing posted, shared, or done on the timeline." + _NUDGE_TAIL
)


class SpeechLedger:
    """What this wake put on a timeline: the messages it posted, and everything else it did.

    The tools record here as they act (`_platform.PlatformContext.speech`), which makes this the
    harness's one truthful answer to two questions it can no longer answer by inference:

    - **"Did the agent speak?"** — the wake's bookend line reports `posted=N`, and now that the
      final text is never auto-posted, the *only* messages that exist are the ones a tool created.
      A bookend that counted just the harness's own posts (probe acks) would report `posted=0` for
      a wake in which the agent spoke — reading a talking agent as a silent one.
    - **"Did the agent act at all this turn?"** — what `NoReplyInformer` turns on. Scanning the
      transcript for a `messages` tool call would mean parsing tool arguments to tell a `create`
      from a `list`; the tool itself already knows, so it says so.

    Wake-scoped (`reset` per wake), and shared by reference with every bound platform tool — so it
    is one object, created once, whose *contents* are per-wake.
    """

    def __init__(self) -> None:
        self.posts: list[object] = []  # messages created (the agent speaking)
        self.acts: list[tuple[str, str | None]] = []  # other visible timeline actions

    def reset(self) -> None:
        """Start a fresh wake. The object is bound into the tools once; only its contents cycle."""
        self.posts.clear()
        self.acts.clear()

    def spoke(self, message: object) -> None:
        """Record a message this agent posted — through the tool, which is the only way it can."""
        self.posts.append(message)

    def acted(self, kind: str, uuid: str | None = None) -> None:
        """Record a non-message action that lands on a timeline (an asset shared, a task made)."""
        self.acts.append((kind, uuid))

    @property
    def actions(self) -> int:
        """Everything visible this wake has done to a timeline — posts included."""
        return len(self.posts) + len(self.acts)


def addressed(text: str | None, handle: str | None) -> bool:
    """Whether `text` addresses `@handle` — the exact-string mention test, and nothing looser.

    **Handles only, never display names** (issue #293, and the reason is empirical): a display name
    is prose — "Nova", "The Brain" — and it false-positives on any sentence that happens to contain
    the word. A handle is an addressing primitive: it is unique, it is deliberate, and a peer types
    the `@` on purpose. That is the fleet's convention (and Slack's), so it is the one signal the
    informer reads.

    Bounded on both sides so a mention is a *whole* handle: `@jt` does not match `@jtx` (a longer
    handle) and `x@jt` does not match at all (an address-shaped fragment). Case-insensitive — a peer
    who writes `@JT` has plainly addressed `@jt`, and platform handles are lowercase to begin with.
    """
    if not text or not handle:
        return False
    return _mention(handle).search(text) is not None


def _mention(handle: str) -> re.Pattern[str]:
    """The compiled `@handle` pattern — escaped, boundaried, case-folded."""
    return re.compile(
        rf"(?<![A-Za-z0-9_-])@{re.escape(handle)}(?![A-Za-z0-9_-])",
        re.IGNORECASE,
    )


def is_one_on_one(timeline: object, me_uuid: str | None) -> bool:
    """Whether `timeline` is a two-viewer conversation: this agent plus exactly one other.

    The *structural* half of the informer's arming (issue #332), and the counterpart to `addressed`:
    a mention is an agent saying "this is for you" in words; a one-on-one is the **shape of the
    room** saying it. On a timeline whose only two viewers are the agent and one peer, there is no
    one else the peer's message could be for — so a message from that peer is addressed to the agent
    whether or not it carries an `@handle`. This is the gap the @briggs incident fell through: a
    fresh 1-on-1 message with no mention, answered as narration, and the founder left staring at an
    empty timeline.

    The viewer set is the owner **unioned with** the participants — unioned so it is correct whether
    or not the platform lists the owner among the participants. Two viewers, one of them this agent,
    is the exact "you and one other" test.

    Read off the timeline the wake already fetched (`WakeAgent.timeline`), so it is the *live* viewer
    set at wake time and costs no extra round-trip. Deliberately **structure, not identity**: the
    counterpart may be human or AI, and this test never asks which — author-kind branching was
    rejected by the founder because it builds a silently-failing path for AI-authored messages that
    no human ever sees. Defensive by `getattr`: a timeline missing owner/participants (or an unknown
    `me_uuid`) reads as *not* one-on-one rather than raising.
    """
    if me_uuid is None:
        return False
    viewers = {p.uuid for p in getattr(timeline, "participants", None) or []}
    owner_uuid = getattr(getattr(timeline, "owner", None), "uuid", None)
    if owner_uuid is not None:
        viewers.add(owner_uuid)
    return me_uuid in viewers and len(viewers) == 2


class NoReplyInformer:
    """Tell an agent — once — that its turn ended with nothing done when the message called for a
    reply, and it did not give one. Never make it act.

    The name summarizes; this docstring qualifies. The firing condition is *no reply **or visible
    action***: the `SpeechLedger` counts a posted message, a shared asset, and a created task
    alike, so an agent that answered by *doing* — sharing the file, scheduling the task — is not
    "no-reply" and is never nudged. "Reply" is shorthand for "any visible timeline action."

    The mechanism is a `TurnHook` (`_engine.TurnHook`): the engine hands it every assistant turn
    and asks whether the loop must continue. The informer answers "yes" exactly once, and only when
    the turn is **ending** (the model emitted no tool calls), **nothing happened on the timeline**
    this turn (the `SpeechLedger`, measured against its count when the turn began — a post made
    earlier in the same wake, on some other item, is not this turn's action), and the turn was
    **armed** for one of two reasons:

    1. **The agent was addressed** — its own `@handle`, exactly, in the text it just read
       (`addressed`). This is the original mention backstop (issue #293).
    2. **It is the only other party** — the message it read is from the counterpart on a two-viewer
       timeline (issue #332). A 1-on-1 message is structurally a message *to the agent*, so it earns
       the same guaranteed second chance a mention does. The wake supplies both facts: the
       two-viewer-ness once at construction (`one_on_one`), and "this turn is reading a
       counterpart's message" per arm (`counterpart_message`) — the latter is what keeps a task
       activation or an agent's own alarm on that same 1-on-1 (the heartbeat pattern) from arming,
       where a nudge would burn a model turn per beat forever.

    Then it appends the nudge and returns True, so the model gets one more pass with its tools still
    in hand: act, or say why not. Either is a valid ending, and it may end in silence again — the
    informer will not fire twice (`nudged`), so it can never loop. When *both* reasons hold (a
    mention inside a 1-on-1), the **mention wording wins** — the more specific signal — and there is
    still exactly one nudge.

    **It informs; it never gates.** No hard stop exists anywhere in this class, deliberately: a
    harness that *forced* a reply is what produced the incident this design answers.
    """

    def __init__(
        self,
        *,
        handle: str | None,
        speech: SpeechLedger,
        one_on_one: bool = False,
        mention_nudge: str = MENTION_NUDGE,
        one_on_one_nudge: str = ONE_ON_ONE_NUDGE,
    ) -> None:
        self.handle = handle
        self.speech = speech
        self.one_on_one = one_on_one  # a two-viewer timeline: a per-wake fact, fixed at wake time
        self.mention_nudge = mention_nudge
        self.one_on_one_nudge = one_on_one_nudge
        self.armed = False
        self.nudged = False
        self._nudge = mention_nudge  # the nudge this arming selected (see `arm`)
        self._baseline = 0

    def arm(self, text: str | None, *, counterpart_message: bool = False) -> None:
        """Read one incoming item: does it earn a nudge if the turn ends empty? Before every call.

        Re-armed per model call rather than per wake, because both are real: a wake engages the
        model once per item (a message batch, an activated task, a posted asset), and the message
        path may *rebuild* its turn when a peer message lands mid-generation. Each is its own turn
        with its own answer to "was this addressed to me, and did I act?", so each gets a fresh
        arming — and the baseline is re-read here, which is what scopes "did nothing" to *this* turn.

        `counterpart_message` says this turn is reading a message authored by someone other than the
        agent (a peer). It is the message path's signal, and it is what gates the one-on-one arm to
        *conversation*: a task activation or a self-authored alarm passes `False`, so a heartbeat on
        a 1-on-1 never arms. The mention arm ignores it — a mention is a mention on any wake.
        """
        mentioned = addressed(text, self.handle)
        solo = self.one_on_one and counterpart_message
        self.armed = mentioned or solo
        # Both true → the mention wording wins (the more specific reason). `_nudge` is read only
        # when `armed`, so its value on an unarmed turn does not matter.
        self._nudge = self.mention_nudge if mentioned else self.one_on_one_nudge
        self.nudged = False
        self._baseline = self.speech.actions

    def on_turn(self, reply: Message, messages: list[Message]) -> bool:
        """The `TurnHook`: append the nudge iff armed, ending, and empty-handed. Once."""
        if not self.armed or self.nudged:
            return False
        if reply.tool_calls:
            return False  # still working — the turn is not ending, so there is nothing to inform
        if self.speech.actions > self._baseline:
            return False  # it acted on the timeline this turn; it was not silent
        self.nudged = True
        messages.append(Message.system(self._nudge))
        return True  # one more pass, tools in hand: act, or say why not
