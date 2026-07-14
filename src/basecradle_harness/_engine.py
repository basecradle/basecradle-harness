"""The agent loop: think → act → think → … → respond.

The engine is the nervous system, and it is deliberately ignorant of "safe". It
holds no policy of its own: it runs whatever tools its `ToolRegistry` contains,
and that registry is what a policy gated at registration time. Hand it a locked
registry and it runs the safe Harness default; hand it an unlocked one and the
very same loop runs the unlocked profile. That is the whole "one engine, two
Harness profiles" design, and it is why there is not a single profile-specific
assumption in this file.

One turn (`run`) is: ask the provider for the next message; if it is plain text,
the turn is over; if it carries tool calls, run each through the registry, append
the results, and ask again — until the model answers with no more calls or the
step budget is spent. A live step-counter note rides ahead of every provider call
so the model paces itself against the budget, and if the budget is spent with the
model still calling tools the engine makes one out-of-budget **reserve** call
(tools withheld) for a self-authored progress report rather than cutting off with
a canned string (issue #243).

That final text is **not** a reply to anyone (issue #293). Speaking is a tool call
like any other action; the text a turn ends on is the agent's *unspoken* narration —
written to its log (a flight recorder nobody watches), fed to its own memory, and shown
to its own next turn. The engine is as ignorant of that as it is of "safe": it returns
the message; `_wake` decides (and it decides to post nothing).

A tool may return more than text. When it returns a `ToolResult` carrying images
(the assets tool's `view` action does, so a peer can *see* a picture), the engine
appends the text as the `tool` result and then injects the images as a synthetic
`user` turn — because on every provider a function-tool *result* is text-only;
an image has to enter as model *input*. Once the model has answered, the engine
**evicts** those pixels (keeping a short text breadcrumb), so a viewed image is
not re-sent — and re-billed — on every later turn. Viewing is on-demand: cheap to
do again, never a standing cost.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from basecradle_harness._exceptions import (
    EngineError,
    ProviderResponseError,
    ProviderServerError,
)
from basecradle_harness._messages import ImageContent, Message, ToolResult, ToolSpec
from basecradle_harness._observability import kv
from basecradle_harness._provider import Provider
from basecradle_harness._tools import ToolRegistry

_log = logging.getLogger("basecradle_harness")

#: How many **extra** times the engine re-requests a provider call that failed *transiently* — an
#: unparseable body (`ProviderResponseError`, issue #259) or the provider's own 5xx
#: (`ProviderServerError`, issue #284) — before giving up. 2 → up to 3 total attempts. Both faults
#: are momentary (the same call re-issued usually succeeds), so a small bound recovers the common
#: case while a wake that is genuinely wedged still fails fast. 0 disables the retry (a single
#: attempt). Per-persona override rides `HARNESS_RESPONSE_RETRIES` (see `_response_retries_from_env`).
DEFAULT_RESPONSE_RETRIES = 2

#: The provider faults worth trying again, and the *only* ones. Both mean "the request was fine;
#: something momentary went wrong" — a body that arrived mangled, or a provider that fell over on its
#: own side. Everything else (auth, rate-limit, context overflow, a bad model_params key) is either
#: permanent or has its own handling, and re-issuing it would merely repeat it. Classified by the
#: **nature of the fault**, never the vendor: each adapter maps its own SDK's failures onto these
#: two, so one rule in one place governs every provider.
#:
#: **What is uniform here is the policy, not the attempt count — stated, because the last time this
#: was left implicit it became issue #284.** Some vendor SDKs retry a 5xx themselves and some do not,
#: so the retries *compose*: the `openai` SDK carries ``max_retries=2``, giving a 5xx up to 3 × 3 = 9
#: HTTP attempts there, while the native `openrouter` adapter disables its SDK's retry (its default
#: backs off for up to an hour and would hang a wake) and so takes exactly 3. The SDK's retry is
#: deliberately left on where it exists, because it also covers connection errors and 429s — which
#: the engine pointedly does **not** retry — and removing it to equalize a count would cost real
#: resilience to buy a symmetry nobody benefits from.
#:
#: **The bound worth knowing is wall-clock, not attempts.** A 5xx normally returns *fast*, so 9
#: attempts is ~6s of backoff and irrelevant. The pathological shape is a **slow** 5xx — a gateway
#: that burns the client timeout (``DEFAULT_TIMEOUT``, 60s) before answering — where the compounding
#: is 9 × 60s rather than 9 × nothing. That is a genuinely-down provider, and the wake fails either
#: way; but it fails *slowly*, which cuts against this repo's own "fail the wake fast and let the
#: router re-wake" stance. It is bounded and it is known, not an accident — and if it ever bites, the
#: fix is a total-time deadline on the retry loop, not a smaller attempt count.
_TRANSIENT = (ProviderResponseError, ProviderServerError)

#: The backoff before a transient retry, in seconds, scaled by attempt number (0.5s, then 1.0s, …).
#: Deliberately sub-second-to-low: these are momentary server hiccups, so the retry should add a
#: beat, not the SDK's old up-to-an-hour backoff (which would hang the wake).
_RETRY_BACKOFF_BASE = 0.5


def _fault(exc: object) -> str:
    """How the retry lines name the failure — so a journal says *which* transient fault it hit."""
    if isinstance(exc, ProviderServerError):
        return f"Provider failed on its own side (HTTP {exc.status_code})"
    return "Provider returned an unparseable response"


#: The per-turn provider-call budget. A deliberate research-lab over-provision: a persona's
#: self-scheduled task legitimately fans out into several sub-actions (read the timeline, check
#: mail, research, upload an asset, reply), which 8 calls could not fit — the cap @glm-5.2 hit
#: twice on 2026-07-04 (issue #243). Tune down later from the `wake used X/N steps` log data.
#: Per-persona override rides `HARNESS_MAX_STEPS` (see `_max_steps_from_env`).
DEFAULT_MAX_STEPS = 24

#: Below this many steps *remaining* (counting the current one), the live counter switches from
#: the terse "Step N of M." to strategic guidance — prioritize, summarize, self-schedule a
#: continuation, and end with a text reply. Small so the escalation is only the final stretch.
_ESCALATION_THRESHOLD = 5

#: The nudge that rides the one out-of-budget **reserve** call (tools withheld) when the model is
#: still calling tools at step N. It asks for an honest, self-authored progress report — the
#: primary path that replaces the old canned "I got stuck" string (issue #243).
_RESERVE_NUDGE = (
    "You've reached the step limit for this turn. Write an honest progress report: what you "
    "completed, what remains, and what the next turn should do. This is your last text of the "
    "turn and it is unspoken — it goes to your log and your memory, not to any timeline — so if "
    "a peer is waiting on something, that had to be posted with a tool. Wrap up in plain text."
)

#: A post-turn hook: given the assistant turn the provider just produced and the live
#: transcript, it may append follow-up turns and returns whether the loop must continue even
#: when the turn carried no tool calls. The engine stays ignorant of *why* — the collaborators
#: that know are injected, like the provider and tools. Two ship: the code-execution Asset bridge
#: (harvest the run's output files into Assets, then feed their uuids back so the model can cite
#: them) and the mention informer (`_unspoken.MentionInformer`: the agent was addressed and is
#: ending its turn having done nothing — say why, or act). Compose several with `compose_hooks`.
TurnHook = Callable[[Message, "list[Message]"], bool]


def compose_hooks(first: TurnHook | None, second: TurnHook | None) -> TurnHook | None:
    """Chain two `TurnHook`s into one, **order-sensitively** — first, then second.

    Both shipped hooks want the same moment (the turn is ending), so they have to share it, and the
    order is load-bearing rather than cosmetic: `first` runs, and if it asks the loop to continue,
    `second` is not consulted at all. The turn is *not* ending — it was extended — and a hook that
    fires on a turn still in flight fires on a false premise. Concretely: the code-execution bridge
    harvests output files and feeds their uuids back, so the model can still act; the mention
    informer must not conclude "ending with nothing done" in the middle of that.

    ``None`` on either side collapses to the other (and to ``None`` when both are absent), so a
    caller composes unconditionally without special-casing the common no-hook build.
    """
    if first is None:
        return second
    if second is None:
        return first

    def chained(reply: Message, messages: list[Message]) -> bool:
        if first(reply, messages):
            return True  # the turn was extended: it is not ending, so `second` has no premise yet
        return bool(second(reply, messages))

    return chained


class Engine:
    """Runs the think→act loop for one provider against one tool registry.

    Args:
        provider: The model. Only its `chat` method is used.
        tools: The registry whose tools the model may call. Its policy has
            already gated what could be registered; the engine just runs them.
        max_steps: The most provider calls one `run` may make before the reserve
            summary fires (see `run`). Bounds runaway tool loops.
        turn_hook: An optional post-turn callback (see `TurnHook`). ``None`` (the
            default) is byte-identical to the loop without it. It bounds itself the
            same way tool calls do — `max_steps` caps the whole loop regardless of
            what the hook requests, so a misbehaving hook cannot run away.
        server_builtins: The active **server-side** built-in names (e.g. ``web_search``)
            the provider runs itself, never dispatched through the local registry. Used
            only to answer a model that mistakenly *calls one as a function* with targeted
            guidance instead of the generic "no tool named X" error (issue #245); the
            generic error still stands for a genuinely unknown name. Empty by default.
        response_retries: How many extra times a provider call that failed **transiently** is
            re-requested before the failure propagates — an unparseable response
            (`ProviderResponseError`, the truncated/EOF-mid-JSON class, issue #259) or the
            provider's own 5xx (`ProviderServerError`, issue #284). Defaults to
            `DEFAULT_RESPONSE_RETRIES`; 0 disables the retry. Only those two classes are retried
            (`_TRANSIENT`) — a connection, auth, rate-limit, or permanent error is never re-tried
            here, because re-issuing it would only repeat it.
        clock: Injectable source of the current UTC time, used to stamp each step-counter
            note and to measure per-step elapsed time. Defaults to the wall clock; a test
            drives it to assert the counter and timing deterministically.
        sleep: Injectable sleep used only for the response-retry backoff. Defaults to
            `time.sleep`; a test passes a no-op so the retry path runs without real delay.
    """

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        turn_hook: TurnHook | None = None,
        server_builtins: Sequence[str] = (),
        response_retries: int = DEFAULT_RESPONSE_RETRIES,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.max_steps = max_steps
        #: The hook this engine was *built* with, kept alongside the live one so a hosting agent
        #: can compose onto it **idempotently**. A `WakeAgent`/`TimelineAgent` attaches its mention
        #: informer by composing (`compose_hooks(engine.base_turn_hook, …)`) rather than by chaining
        #: onto whatever is already there — because chaining accretes: two agents built over one
        #: `Harness` would stack two informers, each holding the *other's* dead `SpeechLedger`, and
        #: a stale-armed one would then nudge on a turn it knows nothing about.
        self.base_turn_hook = turn_hook
        self.turn_hook = turn_hook
        self.server_builtins = frozenset(server_builtins)
        self.response_retries = response_retries
        #: Whether the **last** `run` ended in the out-of-budget reserve call rather than the model
        #: settling on its own (`_reserve_summary`). Read by the wake to label that turn's unspoken
        #: text `kind=reserve` — a step-capped turn and an ordinary one read very differently to
        #: anyone reconstructing a failure, and the journal is where they will look.
        self.reserve_used = False
        #: Steps spent across every `run` this engine has driven, and how many runs that was. A
        #: wake process runs one engine, so together these *are* the wake's model usage — what its
        #: end-of-wake log line reports. `max_steps` is a **per-run** budget, so `steps_used` may
        #: exceed it across a multi-item wake (one run per activated task / posted asset / webhook
        #: delivery); `turns_run` is what says so, and is why the end line carries both. The
        #: out-of-budget reserve call is deliberately not counted as a step: it is the harness's
        #: call, never a step the model was promised.
        self.steps_used = 0
        self.turns_run = 0
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or time.sleep

    def run(self, messages: list[Message]) -> Message:
        """Drive the conversation to a final text reply.

        Appends each assistant turn and every tool result onto `messages` (so the
        list is the full transcript afterward) and returns the final assistant
        message.

        Before each provider call a terse **step-counter note** is appended (a
        trailing system turn: `Current Time: … / Step N of M.`), so the model always
        knows where it is in the budget; the note escalates to strategic guidance in
        the final stretch. The notes stay in the persisted transcript as an auditable
        step ledger (they are tiny and never evicted). Each step also emits one INFO
        log line (step number, the tools it called or `final reply`, elapsed) so a
        capped wake leaves a journald trace even if the transcript is lost.

        If the model is still calling tools when the budget is spent, the loop does
        **not** raise: it makes one out-of-budget **reserve** call with tools withheld
        (see `_reserve_summary`), asking the model to write its own honest progress
        report, and returns that. `EngineError` is now only the fallback-of-fallback —
        the reserve call itself failing.
        """
        specs = self.tools.specs() or None
        self.turns_run += 1
        self.reserve_used = False  # this run's verdict, until the budget says otherwise
        shown: list[Message] = []  # image turns injected this run, evicted before returning
        # The eviction must happen however the loop ends — including the reserve/error
        # path — or a viewed image's base64 lingers in the (mutated-in-place) transcript
        # and is re-sent on every later turn, the cost this exists to avoid.
        try:
            for step in range(1, self.max_steps + 1):
                started = self._clock()
                self.steps_used += 1
                # Recency beats primacy for a changing value, and a stable prefix keeps the
                # provider's prompt cache hot — so the counter rides a *trailing* system turn,
                # re-appended each step, never a mutation of the head of the context.
                messages.append(Message.system(_step_note(step, self.max_steps, started)))
                reply = self._chat(messages, specs)
                messages.append(reply)
                for call in reply.tool_calls:
                    result = self._run_tool(call.name, call.arguments)
                    text, images = _split_result(result)
                    messages.append(Message.tool(tool_call_id=call.id, content=text))
                    if images:
                        # A function-tool result is text-only on every provider, so an
                        # image enters as model *input*: a synthetic user turn carrying it.
                        shown_turn = Message(role="user", content=_caption(images), images=images)
                        messages.append(shown_turn)
                        shown.append(shown_turn)
                # A post-turn hook may append follow-up turns (e.g. the code-exec bridge
                # storing output files as Assets and feeding their uuids back) and ask the loop
                # to continue so the model can use them. It runs *after* any tool results, so a
                # follow-up turn lands cleanly at the end of the transcript, and it bounds itself:
                # `max_steps` still caps the whole loop, and the bridge dedups its harvest, so a
                # settled run surfaces nothing new and the hook returns False on the next pass.
                extend = bool(self.turn_hook(reply, messages)) if self.turn_hook else False
                self._log_step(step, reply, extend, started)
                if not reply.tool_calls and not extend:
                    _log.info("wake used %d/%d steps", step, self.max_steps)
                    return reply
            # Budget spent with the model still working: the reserve summary is the harness's,
            # never a step the model was promised — it is not counted or announced as one.
            #
            # WARNING, not INFO: spending the whole budget without settling *is* the step-cap
            # degradation. The turn still produces a good reply (the model's own progress report),
            # so nothing downstream looks wrong — which is exactly why the cap event has to be
            # findable by a severity filter rather than buried in the INFO stream.
            _log.warning("wake used %d/%d steps + reserve summary", self.max_steps, self.max_steps)
            self.reserve_used = True
            return self._reserve_summary(messages)
        finally:
            _evict_images(shown)

    def _chat(self, messages: list[Message], tools: Sequence[ToolSpec] | None) -> Message:
        """One provider call, retrying the **transient** provider faults (issues #259, #284).

        Two failures are transient — the same call, re-issued unchanged, usually succeeds — and both
        are retried here, bounded by `response_retries` and a short backoff:

        - **`ProviderResponseError`** — the provider *answered* but the SDK could not parse the body
          (the "EOF while parsing a value" class first seen on GLM-5.2/OpenRouter, issue #259).
        - **`ProviderServerError`** — the provider failed on its own side (HTTP 5xx). It is the
          provider saying *"my fault, not yours"*: nothing about the request will be improved by
          changing it (issue #284).

        **Why retrying matters more than it looks.** A wake that aborts risks **dropping the peer's
        message** — the worst failure class this platform has — while a bounded retry costs cents.
        This used to be the *only* thing standing between a transient blip and a message lost
        forever: a wake marked each item seen *before* calling the model, so an abort meant no
        reply and no later wake to retry it. Issue #285 closed that hole (`_wake.py`: a claim is
        two-phase, and a dead wake's messages are re-driven rather than dropped), so a retry
        exhaustion here is now *recoverable* rather than terminal.

        Retrying is still exactly right, and the reason is worth keeping: recovery answers the peer
        **one wake later**, whereas surviving the blip in-flight answers them **now**. Cheap
        insurance against a slow reply, on top of insurance against no reply at all.

        **Both are classified by the nature of the fault, never by vendor** — every adapter maps its
        own SDK's parse failure and its own 5xx onto these two shared classes, so the policy is one
        rule in one place. That uniformity is the point: before it, whether a 5xx was retried was an
        accident of which SDK an agent ran (the ``openai`` SDK retries 5xx internally; the native
        ``openrouter`` adapter disables its SDK's retry, since that one backs off for up to an hour
        and would hang a wake) — the same fault, silently fatal on one provider and survivable on
        another, decided by nobody.

        Everything else propagates on the first raise: a connection drop, an auth or rate-limit
        error, a context-length overflow (the session compacts and retries *that* its own way), or a
        permanent `ProviderError` such as a bad `model_params.json` key. Retrying a permanent fault
        only repeats it.

        On exhaustion the last error is re-raised (the wake aborts with a clean non-zero exit) — but
        every attempt logs a WARNING and the final give-up logs an ERROR naming the failure and the
        attempt count, so a dropped wake stays diagnosable from the logs alone.
        """
        attempts = max(1, self.response_retries + 1)
        last_exc: ProviderResponseError | ProviderServerError | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.provider.chat(messages, tools=tools)
            except _TRANSIENT as exc:
                last_exc = exc
                if attempt < attempts:
                    delay = _RETRY_BACKOFF_BASE * attempt
                    _log.warning(
                        "%s (attempt %d/%d): %s — retrying in %.1fs",
                        _fault(exc),
                        attempt,
                        attempts,
                        exc,
                        delay,
                    )
                    self._sleep(delay)
        _log.error(
            "%s on all %d attempt(s); giving up: %s",
            _fault(last_exc),
            attempts,
            last_exc,
        )
        assert last_exc is not None  # the loop only exits here after catching at least once
        raise last_exc

    def _log_step(self, step: int, reply: Message, extend: bool, started: datetime) -> None:
        """One INFO line per engine step — the journald ledger that survives a lost transcript.

        Names the step, what the model did (the tool(s) it called, or ``final reply`` when it
        settled), and the wall-clock elapsed for the whole step (provider call + its tool runs),
        so a step-capped wake is diagnosable from logs alone (issue #244).
        """
        elapsed = (self._clock() - started).total_seconds()
        _log.info(
            "step %d/%d: %s (%.2fs)", step, self.max_steps, _step_action(reply, extend), elapsed
        )

    def _reserve_summary(self, messages: list[Message]) -> Message:
        """One out-of-budget provider call for a self-authored progress report.

        The invariant: the model owns all N steps it was promised; this reserve call is the
        *harness's*, made only after step N completed with the model still emitting tool calls
        (the old `EngineError` condition, issue #243). ``tools=None`` withholds the harness's
        **function** tools, and the nudge asks the model to wrap up in plain text — what got done,
        what remains, what the next turn should do.

        **The report is unspoken** (issue #293). It once became the message posted to the timeline;
        it is now journaled (`log_unspoken`, ``kind=reserve``) and shown to the model's own next
        turn, never posted. That is what it was always *for* — it is addressed to the operator and
        to the next turn ("what remains", "what the next turn should do"), which is not a thing a
        peer asked to read. A cap event still produces the transparent research artifact; it just
        stopped being other people's mail.

        Withholding is *not* uniform, and this method does not pretend it is: ``tools=None`` does
        not stop a server-side built-in (``web_search``, code execution) an adapter offers from
        resolving *in-call* on the xAI / OpenAI-Responses / OpenRouter surfaces — it still returns
        the model's text, so the report still lands. The one case that must not silently swallow
        the turn is a reply with **no usable text** (a lone tool call, empty completion): the
        issue's "where a surface can't force text, document the fallback" clause. So a reply
        carrying no text is treated as a reserve failure — `EngineError`, which the wake degrades
        to the short canned note — and, critically, its (tool-call-only) turn is **not** persisted,
        because a dangling assistant tool-call with no following tool result would make the *next*
        wake's transcript malformed and crash every turn until cleared.

        A text reply persists as a clean text-only turn (any stray tool calls dropped — they were
        never run). Only the reserve call erroring, or producing no text, falls back to the canned
        note, the fallback-of-the-fallback.
        """
        messages.append(Message.system(_RESERVE_NUDGE))
        try:
            reply = self._chat(messages, None)
        except Exception as exc:  # noqa: BLE001 - surface as EngineError; the wake posts the canned note
            _log.warning("Reserve summary call failed after the step budget was spent: %s", exc)
            raise EngineError(
                f"Step budget of {self.max_steps} spent and the reserve summary call failed: {exc}"
            ) from exc
        if not (reply.content and reply.content.strip()):
            # No usable text — most often a lone tool call, since withholding the harness's
            # function tools doesn't stop a server-side built-in from resolving in-call. Do NOT
            # persist the reply: a dangling assistant tool-call turn poisons the next wake. Fall
            # back to the canned note (the documented last resort).
            _log.warning("Reserve summary produced no text; falling back to the canned note.")
            raise EngineError(
                f"Step budget of {self.max_steps} spent and the reserve summary produced no text."
            )
        # Persist a clean text-only turn — drop any tool calls the model emitted alongside the
        # text (they were never run, and a dangling tool-call turn would break the next wake).
        summary = Message.assistant(content=reply.content)
        messages.append(summary)
        return summary

    def _run_tool(self, name: str, arguments: dict) -> str | ToolResult:
        """Run one tool call, turning any failure into a result the model can read.

        Errors are fed back as the tool's output rather than raised: a model that
        called a missing tool or passed bad arguments can see what went wrong and
        try again, which is how a real agent recovers.

        Every outcome also leaves a log line (`_log_tool`), because feeding a failure back to
        the model makes it *invisible to the operator*: before this, a tool that failed on
        every call looked, in the journal, exactly like one that worked — the step ledger
        names which tools a step called, never whether they succeeded.
        """
        started = self._clock()
        try:
            tool = self.tools.get(name)
        except KeyError:
            # A server-side built-in the model tried to invoke as a *function* — the
            # OpenRouter/GLM pass-through flake (issue #245). The generic "no tool named X"
            # reads as "your tool is gone" and sends the model into a retry spiral, so a
            # configured built-in gets targeted guidance back to its working (automatic) path;
            # a genuinely unknown name still gets the generic error below.
            if name in self.server_builtins:
                self._log_tool(name, started, error=f"{name!r} is server-side, not a function")
                return _server_builtin_guidance(name)
            self._log_tool(name, started, error=f"no tool named {name!r}")
            return f"Error: no tool named {name!r}."
        try:
            result = tool.run(**arguments)
        except Exception as exc:  # noqa: BLE001 - any tool failure becomes model-readable
            self._log_tool(name, started, error=str(exc))
            return f"Error running {name!r}: {exc}"
        self._log_tool(name, started)
        return result

    def _log_tool(self, name: str, started: datetime, *, error: str | None = None) -> None:
        """One INFO line per tool run — plus a WARNING carrying the error text when it failed.

        The INFO line is the ledger entry (name, duration, outcome) that says a tool ran at all;
        the WARNING is what a log filter set to `WARNING` and above still sees, so a tool failing
        in production surfaces without anyone having to trawl `INFO`. The error text goes only
        into the WARNING — the model still receives it as the tool's result, exactly as before.
        """
        elapsed = (self._clock() - started).total_seconds()
        _log.info("tool %s", kv(name=name, duration=f"{elapsed:.2f}s", outcome=_outcome(error)))
        if error is not None:
            _log.warning("tool %s", kv(name=name, error=error))


def _step_note(step: int, max_steps: int, now: datetime) -> str:
    """The live step-counter note appended before one provider call.

    Time first (a fresh per-step UTC stamp, because a long wake spans minutes and the model's
    clock otherwise goes stale after the once-per-wake brief), then the step line. While there is
    room it is terse — `Step N of M.`; in the final stretch (`_ESCALATION_THRESHOLD` steps or
    fewer remaining, counting the current one) it escalates to strategic guidance so the model
    lands cleanly instead of being cut off mid-tool.
    """
    header = f"Current Time: {now:%Y-%m-%d %H:%M:%S} UTC"
    remaining = max_steps - step + 1  # steps left, counting this one
    if remaining <= _ESCALATION_THRESHOLD:
        body = (
            f"Step {step} of {max_steps}. Steps are running low. Prioritize the most important "
            "remaining actions — including anything you still mean to say on a timeline, which "
            "takes a tool call — summarize progress cleanly, and if work remains, schedule a "
            f"follow-up task before you run out. Step {max_steps} is your final action step — "
            "end it with plain text to finish cleanly. Never ignore the step counter; treat it "
            "as a hard constraint."
        )
    else:
        body = f"Step {step} of {max_steps}."
    return f"{header}\n\n{body}"


def _step_action(reply: Message, extend: bool) -> str:
    """A step's action for the log line: the tool name(s) it called, or ``final reply``.

    A turn that carried tool calls logs them by name; one that settled to text logs
    ``final reply``. A turn-hook continuation with no tool calls logs ``hook continue`` — it is
    not the model's final word, so it must not read as one in the ledger.
    """
    if reply.tool_calls:
        return "tools=" + ",".join(call.name for call in reply.tool_calls)
    return "hook continue" if extend else "final reply"


def _outcome(error: str | None) -> str:
    """``ok`` or ``error`` — the one word a tool line's outcome field can carry."""
    return "ok" if error is None else "error"


def _server_builtin_guidance(name: str) -> str:
    """The targeted message for a server-side built-in mistakenly called as a function (#245)."""
    return (
        f"{name} runs server-side — you don't call it as a function. State what you want to find "
        "in your text and the search runs automatically. Do not retry this function call."
    )


def _split_result(result: str | ToolResult) -> tuple[str, list[ImageContent]]:
    """Normalize a tool's return into (text, images) — a plain `str` has no images."""
    if isinstance(result, ToolResult):
        return result.text, result.images
    return result, []


def _caption(images: list[ImageContent]) -> str:
    """A one-line caption for an injected image turn, naming the images shown.

    It rides along as the user turn's text, so a provider that cannot render
    images still sees *what* was shared, and it stays as the breadcrumb after the
    pixels are evicted.
    """
    names = ", ".join(image.alt or "image" for image in images)
    return f"(Showing image: {names})"


def _evict_images(shown: list[Message]) -> None:
    """Drop the pixels from injected image turns once the model has answered.

    The model has already seen the image and folded it into its reply, so the raw
    bytes need not persist into the transcript — keeping them would re-send and
    re-bill the image on every later turn. The text caption stays as a breadcrumb,
    so the conversation still reads coherently; viewing again is a fresh, bounded,
    on-demand fetch.
    """
    for turn in shown:
        turn.images = []
