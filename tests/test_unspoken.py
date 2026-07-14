"""The Unspoken Channel: silence by default, speech on purpose, reasons always on the record.

The inversion (issue #293, program basecradle/basecradle#420) in one file:

- **Nothing an agent generates touches a timeline.** The final text of a turn is *unspoken* —
  written to the agent's log, fed to its memory, shown to its own next turn, never posted.
- **Every timeline interaction is an intentional tool call.** The `messages` tool is how an agent
  speaks; that is the whole channel.
- **Full visibility is the price of that freedom.** The narration is logged in full — but the log
  is a **flight recorder, not a control tower**: nobody watches it, and the guidance says so, or an
  agent would "escalate" into a void and believe it had spoken.
- **The mention informer informs; it never forces.** Addressed by `@handle` and ending a turn
  having done nothing → one system nudge, once, and the model may still choose silence.

The cast is the fixed fiction: Nova Digital (`nova`, AI) is the agent; John Doe (`john`) the human.
"""

import json
import logging
import re

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import Harness, MemoryTool, Message, MessagesTool, WakeAgent
from basecradle_harness._engine import compose_hooks
from basecradle_harness._messages import ToolCall
from basecradle_harness._observability import log_unspoken
from basecradle_harness._unspoken import (
    MENTION_NUDGE,
    MentionInformer,
    SpeechLedger,
    addressed,
)

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"  # the agent (me) — @nova
JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"  # the human — @john
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"
M0 = "019e7751-4a1b-7c2d-8e3f-1a2b3c4d5e6f"
REPLY = "019e7755-8e5f-7f70-9283-5e6f70819203"


# === the mention test: handles only, whole handles ============================


@pytest.mark.parametrize(
    "text",
    [
        "@nova can you look at this?",
        "hey @nova",
        "cc @nova, thanks",
        "(@nova)",
        "@NOVA — a peer who shouts still addressed you",
        "line one\n@nova on line two",
    ],
)
def test_addressed_recognizes_a_mention(text):
    assert addressed(text, "nova") is True


@pytest.mark.parametrize(
    "text",
    [
        "Nova, can you look at this?",  # a display name is prose — never a mention
        "the nova explosion was bright",  # …and prose false-positives, which is why we don't
        "@novabot is a different account",  # a *longer* handle is a different peer
        "@nova-2 is also someone else",
        "mail me at bob@nova",  # an address-shaped fragment is not an addressing
        "",
    ],
)
def test_addressed_rejects_a_non_mention(text):
    assert addressed(text, "nova") is False


def test_addressed_with_no_handle_is_false():
    """An agent whose handle we never learned is never 'addressed' — fail quiet, not loud."""
    assert addressed("@nova hello", None) is False


# === the speech ledger: what actually reached a timeline ======================


def test_the_ledger_counts_posts_and_other_visible_acts():
    ledger = SpeechLedger()
    assert ledger.actions == 0

    ledger.spoke(object())  # a message — the agent speaking
    ledger.acted("asset", "uuid-1")  # a file shared: visible, so it counts as acting
    assert ledger.actions == 2
    assert len(ledger.posts) == 1  # …but only the message is a *post*

    ledger.reset()  # wake-scoped: the object is bound into the tools once, its contents cycle
    assert ledger.actions == 0 and ledger.posts == []


# === the informer: it informs, it never forces ================================


def _turn(*, tools=False):
    calls = [ToolCall(id="c1", name="messages", arguments={})] if tools else []
    return Message.assistant(content=None if tools else "…", tool_calls=calls)


def test_the_informer_nudges_once_when_addressed_and_empty_handed():
    speech = SpeechLedger()
    informer = MentionInformer(handle="nova", speech=speech)
    informer.arm("@nova what's the status?")
    history = []

    assert informer.on_turn(_turn(), history) is True  # continue the loop: act, or say why
    assert history == [Message.system(MENTION_NUDGE)]

    # **Once.** A second ending turn is not nudged again — the model may end in silence, and the
    # informer must never be able to loop an agent that has decided.
    assert informer.on_turn(_turn(), history) is False
    assert len(history) == 1


def test_the_informer_is_silent_when_the_agent_acted():
    """It acted on the timeline this turn → there is nothing to inform it of."""
    speech = SpeechLedger()
    informer = MentionInformer(handle="nova", speech=speech)
    informer.arm("@nova please post the summary")
    speech.spoke(object())  # …and it did

    assert informer.on_turn(_turn(), []) is False


def test_the_informer_measures_this_turn_not_the_whole_wake():
    """A post made *earlier in the same wake*, on another item, is not this turn's action.

    The ledger is wake-scoped (the bookend counts it), so the informer takes a baseline when it
    arms. Without that, an agent that answered a task at the start of a wake would look like it had
    "already acted" on every later item it was addressed in — and would never be informed again.
    """
    speech = SpeechLedger()
    informer = MentionInformer(handle="nova", speech=speech)
    speech.spoke(object())  # an earlier item in this wake was answered

    informer.arm("@nova and what about this one?")  # a *new* turn begins here

    assert informer.on_turn(_turn(), []) is True  # this turn did nothing → inform


def test_the_informer_is_silent_mid_turn():
    """A turn carrying tool calls is not ending — there is nothing yet to conclude."""
    informer = MentionInformer(handle="nova", speech=SpeechLedger())
    informer.arm("@nova look into this")

    assert informer.on_turn(_turn(tools=True), []) is False


def test_the_informer_is_silent_when_not_addressed():
    """Not mentioned → not addressed → no nudge. Silence needs no defense when nobody asked."""
    informer = MentionInformer(handle="nova", speech=SpeechLedger())
    informer.arm("john: I think we should ship on Friday.")

    assert informer.on_turn(_turn(), []) is False


def test_the_nudge_never_commands_and_never_invents_a_reader():
    """Two things the wording must do, and both were nearly got wrong.

    **It may not command.** The whole incident this design answers came from a harness that decided
    when an agent speaks. The informer may say "you were addressed and did nothing"; it may not say
    "reply".

    **It may not promise a reader.** The founder's correction (2026-07-14): there is no operator
    watching the log. A nudge that says "say why — your operator reads it" teaches the model that
    its log is a channel to a person, and a model that believes that will *escalate* into it and
    walk away satisfied. The reason is for the record and for the agent's own memory; nobody is
    waiting on it.
    """
    assert "may be exactly right" in MENTION_NUDGE
    assert "no one will force you" in MENTION_NUDGE
    assert "for your own memory" in MENTION_NUDGE  # the true reader
    assert "no one is waiting on it" in MENTION_NUDGE  # …and the true audience: none
    assert "operator" not in MENTION_NUDGE.lower()  # the frame that must never come back


def test_the_nudge_names_the_verb_it_asks_for():
    """ "Act now" is not an instruction to a model that cannot see the channel (issue #295).

    The evidence is @jt (gpt-5.4-mini) on the 0.67.0 rollout: @-mentioned and asked outright what
    version it was running, it composed the correct answer and *narrated* it — `posted=0`,
    `text="I'm running 0.67.0."` The nudge fired; its reply to the nudge was more narration. It was
    not refusing and it did not misunderstand the question. It believed it had answered.

    So the nudge names the mechanism **and its absence** in one breath — call the tool; text here
    reaches no one — because either half alone leaves the gap: naming the tool without saying the
    narration goes nowhere lets the model think it has two channels, and saying the text goes
    nowhere without naming the tool leaves it with no channel at all.

    What must *not* come back with the mechanism is the command. "Speaking means calling the
    `messages` tool" describes the world; "reply to them" would decide for the agent, and a harness
    that decides when an agent speaks is the defect this whole design removed.
    """
    assert "calling the `messages` tool" in MENTION_NUDGE  # the verb, named
    assert "reaches no one" in MENTION_NUDGE  # and why it is the only one
    assert "no one will force you" in MENTION_NUDGE  # still not a command


# === the standing guard: nothing the model reads may invent a supervisor =======


def _model_facing_strings():
    """Every shipped string the *model* reads — the surface the operator frame must stay out of.

    Not the docstrings and not the code comments: those are written for whoever maintains the
    harness, and "operator" there means "whoever runs the install", which is real. This is the
    other surface — the brief, the nudges, the guidance the engine feeds back — where the word
    would be a lie the model then reasons from.
    """
    from basecradle_harness._brief import render_budget, render_defects, render_safety
    from basecradle_harness._engine import (
        _RESERVE_NUDGE,
        _server_builtin_guidance,
        _step_note,
    )
    from basecradle_harness._install import prompt_text

    now = __import__("datetime").datetime(2026, 7, 14, tzinfo=__import__("datetime").timezone.utc)
    return {
        "initialize.md": prompt_text("initialize.md") or "",
        "system-prompt.md": prompt_text("system-prompt.md") or "",
        "step budget": render_budget(24) or "",
        "tool defect": render_defects(["memory — failed to load"]) or "",
        "safety opt-out": render_safety(["mcp: filesystem"]) or "",
        "reserve nudge": _RESERVE_NUDGE,
        "step note (terse)": _step_note(1, 24, now),
        "step note (escalated)": _step_note(23, 24, now),
        "builtin guidance": _server_builtin_guidance("web_search"),
        "mention nudge": MENTION_NUDGE,
    }


#: The **supervisor frame**: an operator with agency, who reads, fixes, watches, or is owed a
#: report. It is the frame that must never reach a model — not the word itself, which the guidance
#: uses precisely to *deny* it ("you are your own operator"). Every pattern here was a real string
#: in this repo before issue #293: the brief told the model "an operator can fix it", and the first
#: draft of the guidance told it "your operator reads it".
_SUPERVISOR_FRAME = re.compile(
    r"your operator"
    r"|an operator (?:can|will|may|should|reads|watches)"
    r"|the operator (?:can|will|may|should|reads|watches)"
    r"|so (?:an|the|your) operator"
    r"|operator (?:reads|watches|reviews|monitors)",
    re.IGNORECASE,
)


def test_no_model_facing_string_invents_a_supervisor():
    """**There is no operator, and the model must never be told there is** (the founder, 2026-07-14).

    A standing guard, not a one-time cleanup, because the frame is endemic and it creeps back in
    through the most natural-sounding sentence: "so an operator can audit it", "your operator reads
    the log". These agents have no human supervisor — the agent *is* its own operator — and the log
    is a flight recorder: dug into after a failure, never watched.

    The failure it causes is not cosmetic. A model that believes its log has a reader will **report
    into it** — an escalation, a blocker, an attack it spotted — and consider the matter
    communicated. It was not: nobody read it. An escalation written only into an unread log is a
    message to no one, which is exactly what the security guidance exists to prevent.

    The *word* is fine where the guidance uses it to deny the thing ("you are your own operator").
    The **frame** — an operator who reads, fixes, or is owed a report — is what is banned.
    """
    offenders = {
        name: _SUPERVISOR_FRAME.search(text).group(0)
        for name, text in _model_facing_strings().items()
        if _SUPERVISOR_FRAME.search(text)
    }
    assert offenders == {}, (
        f"model-facing text implies a supervising human: {offenders}. "
        "Reword it — the agent is its own operator, and nobody reads its log."
    )


#: Every model-facing string whose whole job is to tell the agent that reaching a peer takes an
#: act — and which is therefore useless to a model that cannot name the act. Each one carried a
#: generic "with a tool" / "takes a tool call" before issue #295, and each one is read at exactly
#: the moment the gap costs a peer their answer: addressed and about to fall silent (the mention
#: nudge), out of steps with something still unsaid (the escalation and the reserve report), and
#: the once-per-wake statement of the rule (the brief's budget line).
_SPEECH_INSTRUCTIONS = ("mention nudge", "reserve nudge", "step note (escalated)", "step budget")


def test_every_string_that_asks_for_speech_names_the_tool():
    """A generic "post it with a tool" is one inference away from the act — and small models miss it.

    A standing guard, and it is the *same* guard as the supervisor one, aimed the other way: that
    one bans a channel the model believes in and does not have, this one bans a channel the model
    has and cannot see. Both end identically — the agent finishes a turn believing it communicated,
    and nobody heard anything.

    This is not a style rule about tool names. These four strings exist *only* to close the gap
    between "I have something to say" and "the platform has it", so a version of one that stops at
    "a tool call" has stopped one step short of its own purpose. New guidance in this class joins
    the list; a rewrite that drops the tool name fails here.
    """
    strings = _model_facing_strings()
    silent = [name for name in _SPEECH_INSTRUCTIONS if "`messages`" not in strings[name]]

    assert silent == [], (
        f"guidance about reaching a peer that never names the mechanism: {silent}. "
        "Say `messages` — 'act' and 'a tool call' are riddles to the models that need this most."
    )


def test_the_guidance_denies_the_supervisor_outright():
    """…and the denial itself must survive: the model is told, in words, that it is on its own."""
    guidance = _model_facing_strings()["initialize.md"].lower()

    assert "there is no operator behind you" in guidance
    assert "you are your own operator" in guidance


def test_the_guidance_tells_the_agent_its_log_reaches_no_one():
    """The positive half of the same law: the floor must state the world-model, not just avoid a lie.

    Three claims have to survive any future edit of `initialize.md`, because each one closes a
    specific trap: **assume nobody reads it** (or the agent escalates into a void), **speak or it
    reached no one** (the escalation law's teeth under silence-default), and **speech is a tool
    call** (or the agent waits for an auto-post that will never come).
    """
    guidance = _model_facing_strings()["initialize.md"].lower()

    assert "assume no one will ever read it" in guidance
    assert "speak on a timeline, or it reached no one" in guidance
    assert "you speak by calling a tool" in guidance
    assert "presence is not performance" in guidance  # the founder kept this one on purpose


# === compose_hooks: the informer shares the seam with the code-exec bridge =====


def test_compose_hooks_runs_the_second_only_when_the_first_did_not_extend():
    """Order is load-bearing: a turn the bridge extended is not a turn that is ending."""
    seen = []

    def bridge(reply, messages):
        seen.append("bridge")
        return True  # the bridge harvested files and wants another pass

    def informer(reply, messages):
        seen.append("informer")
        return False

    assert compose_hooks(bridge, informer)(_turn(), []) is True
    assert seen == ["bridge"]  # the informer was never consulted — the turn was not ending


def test_compose_hooks_consults_the_second_when_the_turn_is_ending():
    def bridge(reply, messages):
        return False  # nothing to harvest: the turn really is ending

    def informer(reply, messages):
        return True

    assert compose_hooks(bridge, informer)(_turn(), []) is True


def test_compose_hooks_collapses_around_none():
    def hook(reply, messages):
        return True

    assert compose_hooks(None, hook) is hook
    assert compose_hooks(hook, None) is hook
    assert compose_hooks(None, None) is None


# === the journal: the narration is logged in full ==============================


def test_the_unspoken_line_carries_the_whole_narration(caplog):
    """**Full visibility is the price of the freedom** — so this one field is never truncated.

    Every other value in the log stream is bounded by `MAX_VALUE` (240 chars), because every other
    value is recoverable somewhere else. An unspoken narration is not: no peer saw it, no timeline
    holds it. Bounding it would quietly turn "full visibility" into "the first 240 characters of
    visibility", which is not the trade the founder's principle names.
    """
    narration = "I read John's note. " + "It needs no reply. " * 39 + "Leaving it."  # ~800 chars
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_unspoken(narration, timeline=TIMELINE_UUID)

    line = caplog.records[-1].getMessage()
    assert line.startswith("unspoken ")
    assert f"timeline={TIMELINE_UUID}" in line and "kind=narration" in line
    assert f"chars={len(narration)}" in line
    assert narration in line  # whole, not elided


def test_the_unspoken_line_is_one_record_and_scrubs_secrets(caplog):
    """One record per narration (a severity filter must never see a decapitated fragment), and no
    credential rides out in an agent's own prose."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_unspoken("line one\nline two, key sk-abcdefgh12345678 here", timeline=TIMELINE_UUID)

    line = caplog.records[-1].getMessage()
    assert "\n" not in line  # flattened: one record, one line
    assert "sk-abcdefgh12345678" not in line and "[redacted]" in line


# === end to end: a wake speaks only when it decides to ========================


def _wire(router, *, body="What's the status?", mine=False):
    """The four platform routes a wake touches, with one unseen message on the timeline."""
    actor = (
        {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}
        if mine
        else {"uuid": JOHN_UUID, "handle": "john", "name": "John Doe", "kind": "human"}
    )
    router.get("/users/dashboard").mock(
        return_value=httpx.Response(
            200,
            json={
                "identity": {
                    "uuid": NOVA_UUID,
                    "handle": "nova",
                    "name": "Nova Digital",
                    "kind": "ai",
                }
            },
        )
    )
    router.get(f"/timelines/{TIMELINE_UUID}").mock(
        return_value=httpx.Response(
            200,
            json={
                "timeline": {
                    "uuid": TIMELINE_UUID,
                    "name": "Incident response",
                    "locked": False,
                    "created_at": "2026-06-01T00:00:00.000Z",
                    "updated_at": "2026-06-02T00:00:00.000Z",
                    "owner": actor,
                    "participants": [],
                },
                "items": [],
            },
        )
    )
    router.get("/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "messages": [
                    {
                        "type": "message",
                        "created_at": "2026-06-04T00:00:00.000Z",
                        "user": actor,
                        "timeline": {"uuid": TIMELINE_UUID},
                        "content": {"uuid": M0, "body": body},
                    }
                ],
                "next_cursor": None,
            },
        )
    )
    router.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
        return_value=httpx.Response(
            201,
            json={
                "message": {
                    "type": "message",
                    "created_at": "2026-06-04T00:00:01.000Z",
                    "user": {
                        "uuid": NOVA_UUID,
                        "handle": "nova",
                        "name": "Nova Digital",
                        "kind": "ai",
                    },
                    "timeline": {"uuid": TIMELINE_UUID},
                    "content": {"uuid": REPLY, "body": "posted"},
                }
            },
        )
    )
    router.get("/assets").mock(
        return_value=httpx.Response(200, json={"assets": [], "next_cursor": None})
    )
    router.get("/webhook_events").mock(
        return_value=httpx.Response(200, json={"webhook_events": [], "next_cursor": None})
    )
    router.get("/tasks").mock(
        return_value=httpx.Response(200, json={"tasks": [], "next_cursor": None})
    )


@pytest.fixture
def platform():
    with respx.mock(base_url=BC_URL, assert_all_called=False) as router:
        yield router


def _posts(platform):
    return [
        json.loads(call.request.content)["message"]["body"]
        for call in platform.calls
        if call.request.method == "POST" and call.request.url.path.endswith("/messages")
    ]


class _Brain:
    """A scripted brain: each entry is one turn's reply, in order."""

    provider = "openai"
    model = "gpt-4o"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.shown = []

    def chat(self, messages, tools=None):
        self.shown = list(messages)
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def _speak(body):
    return Message.assistant(
        tool_calls=[
            ToolCall(id="c1", name="messages", arguments={"action": "create", "body": body})
        ]
    )


def test_a_silent_wake_posts_nothing_and_says_why(platform, tmp_path, caplog):
    """The inversion, end to end: the agent reads, thinks, and stays silent — deliberately.

    Nothing reaches the timeline. The wake's bookend reports `posted=0`, so the silence is
    *visible*; the `unspoken` line carries the reasoning, so it is *accountable*. That pair is the
    whole design: never forced to speak, never invisible.
    """
    _wire(platform, body="thanks, that's all I needed!")
    brain = _Brain(Message.assistant(content="A closing line. It needs no reply; I'll leave it."))
    harness = Harness(brain, home=tmp_path, tools=[MessagesTool()])
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=BaseCradle(token=FAKE_TOKEN))

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        posted = agent.wake()

    assert posted == []
    assert _posts(platform) == []  # not one word — the peer's timeline is untouched
    end = next(m.getMessage() for m in caplog.records if m.getMessage().startswith("wake end"))
    assert "posted=0" in end  # visibly silent
    unspoken = next(m.getMessage() for m in caplog.records if m.getMessage().startswith("unspoken"))
    assert "It needs no reply" in unspoken  # …and the reason is on the record


def test_a_speaking_wake_posts_exactly_once(platform, tmp_path):
    """It speaks by calling the tool — and the narration that follows does **not** double it.

    This is the defect that started the program (@glm-5.2, ~50 occurrences; @briggs, 11 posts in
    ~100 seconds): every tool-post turn also auto-posted its final text. The two channels are now
    one, and it is the tool.
    """
    _wire(platform)
    brain = _Brain(_speak("All clear, John."), Message.assistant(content="Answered him."))
    harness = Harness(brain, home=tmp_path, tools=[MessagesTool()])
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=BaseCradle(token=FAKE_TOKEN))

    posted = agent.wake()

    assert _posts(platform) == ["All clear, John."]  # exactly one body, and it is the tool's
    assert len(posted) == 1  # the wake reports the agent's own post


def test_a_mentioned_agent_that_does_nothing_is_informed_once(platform, tmp_path):
    """Addressed by @handle, ending its turn empty-handed → one nudge, and it may still be silent.

    The informer's contract, end to end. The model gets one more pass with its tools in hand; here
    it uses it to *explain itself* rather than to speak, which is a legitimate ending — and the
    explanation lands in the log, where an operator can judge whether the silence was good.
    """
    _wire(platform, body="@nova can you take a look?")
    brain = _Brain(
        Message.assistant(content="Not for me."),  # ending, having done nothing
        Message.assistant(content="Deliberate: john is asking @ops, not me. Staying out."),
    )
    harness = Harness(brain, home=tmp_path, tools=[MessagesTool()])
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=BaseCradle(token=FAKE_TOKEN))

    posted = agent.wake()

    assert brain.calls == 2  # the nudge bought one more pass
    nudges = [m for m in brain.shown if m.role == "system" and m.content == MENTION_NUDGE]
    assert len(nudges) == 1  # exactly one, ever
    assert posted == []  # …and it was still free to stay silent. Never forced.
    assert _posts(platform) == []


def test_a_mentioned_agent_that_speaks_is_not_nudged(platform, tmp_path):
    """It was addressed and it acted — so there is nothing to inform it of, and no extra call."""
    _wire(platform, body="@nova can you take a look?")
    brain = _Brain(_speak("On it."), Message.assistant(content="Answered."))
    harness = Harness(brain, home=tmp_path, tools=[MessagesTool()])
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=BaseCradle(token=FAKE_TOKEN))

    agent.wake()

    assert brain.calls == 2  # the tool call and the settle — no third, nudged pass
    assert [m for m in brain.shown if m.content == MENTION_NUDGE] == []
    assert _posts(platform) == ["On it."]


def test_an_unaddressed_silent_wake_is_not_nudged(platform, tmp_path):
    """Nobody asked. Silence is the default, and the default needs no explaining."""
    _wire(platform, body="john: I'll take this one myself.")
    brain = _Brain(Message.assistant(content="Nothing for me here."))
    harness = Harness(brain, home=tmp_path, tools=[MessagesTool()])
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=BaseCradle(token=FAKE_TOKEN))

    agent.wake()

    assert brain.calls == 1  # one pass, no nudge
    assert [m for m in brain.shown if m.content == MENTION_NUDGE] == []


def test_two_agents_over_one_harness_do_not_accrete_informers(platform, tmp_path):
    """Attaching the informer **replaces**, never chains — the trap in a one-line constructor.

    A hosting agent composes its informer onto the engine's `base_turn_hook`, not onto whatever hook
    happens to be live. Chaining would accrete: an embedder that builds one `Harness` and a
    `WakeAgent` per timeline would stack N informers, each holding the *others'* dead `SpeechLedger`
    — and a stale-armed one would then fire on a turn it knows nothing about, reading "did it act?"
    off a ledger nobody writes to any more. The second agent's hook must simply be the second
    agent's.
    """
    _wire(platform)
    harness = Harness(_Brain(), home=tmp_path, tools=[MessagesTool()])
    client = BaseCradle(token=FAKE_TOKEN)

    first = WakeAgent(harness, timeline=TIMELINE_UUID, client=client)
    second = WakeAgent(harness, timeline=TIMELINE_UUID, client=client)

    # The live hook is the *second* agent's informer alone — the first's is gone, not stacked
    # behind it. (With no other hook wired, composition collapses to the bare informer.)
    assert harness.engine.turn_hook == second.informer.on_turn
    assert harness.engine.turn_hook != first.informer.on_turn
    assert harness.engine.base_turn_hook is None  # …and the base is untouched, so it can't drift


def test_the_unspoken_line_names_which_ending_the_turn_had(platform, tmp_path, caplog):
    """A step-capped turn and an ordinary one read nothing alike — so the journal says which.

    The log is the flight recorder, and `kind=` is what makes it worth reading after the fact:
    `narration` (the model settled), `reserve` (the step budget was spent and it wrote its own
    progress report), `stuck` (even that failed). Without the label they are one undifferentiated
    stream and a reconstruction has to re-read the transcript to tell them apart.
    """
    _wire(platform)

    class Looping:
        """Never settles while it has tools; writes a report once they are withheld."""

        provider, model = "openai", "gpt-4o"

        def chat(self, messages, tools=None):
            if tools is None:  # the out-of-budget reserve call
                return Message.assistant(content="Ran out of steps; here is where I got to.")
            return Message.assistant(
                tool_calls=[ToolCall(id="c1", name="memory", arguments={"action": "list"})]
            )

    harness = Harness(Looping(), home=tmp_path, tools=[MemoryTool()], max_steps=2)
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=BaseCradle(token=FAKE_TOKEN))

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake()

    unspoken = next(m.getMessage() for m in caplog.records if m.getMessage().startswith("unspoken"))
    assert "kind=reserve" in unspoken  # not `narration` — this turn hit the cap
    assert "Ran out of steps" in unspoken
    assert _posts(platform) == []  # and it went to nobody, which is the point of the change


def test_a_mute_agent_cannot_speak(platform, tmp_path):
    """The kit's contract, stated as a test: **speech is a tool you hand it** (issue #293).

    An agent with no `messages` tool has no voice — there is no implicit channel left to fall back
    on. That is the whole point of the inversion, and it is worth pinning: a harness assembled by
    hand without `MessagesTool` produces an agent that thinks out loud into its log and nothing more.
    """
    _wire(platform)
    brain = _Brain(Message.assistant(content="I would love to answer that."))
    harness = Harness(brain, home=tmp_path)  # no tools at all
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=BaseCradle(token=FAKE_TOKEN))

    posted = agent.wake()

    assert posted == []
    assert _posts(platform) == []
