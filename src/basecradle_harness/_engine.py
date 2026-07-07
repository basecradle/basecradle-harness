"""The agent loop: think → act → think → … → respond.

The engine is the nervous system, and it is deliberately ignorant of "safe". It
holds no policy of its own: it runs whatever tools its `ToolRegistry` contains,
and that registry is what a policy gated at registration time. Hand it a locked
registry and it runs the safe Harness default; hand it an unlocked one and the
very same loop runs the unlocked profile. That is the whole "one engine, two
Harness profiles" design, and it is why there is not a single profile-specific
assumption in this file.

One turn (`run`) is: ask the provider for the next message; if it is plain text,
that is the reply; if it carries tool calls, run each through the registry,
append the results, and ask again — until the model answers with no more calls
or the step budget is spent. A live step-counter note rides ahead of every
provider call so the model paces itself against the budget, and if the budget is
spent with the model still calling tools the engine makes one out-of-budget
**reserve** call (tools withheld) for a self-authored progress report rather than
cutting off with a canned string (issue #243).

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
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from basecradle_harness._exceptions import EngineError
from basecradle_harness._messages import ImageContent, Message, ToolResult
from basecradle_harness._provider import Provider
from basecradle_harness._tools import ToolRegistry

_log = logging.getLogger("basecradle_harness")

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
    "completed, what remains, and what the next turn should do. This is your final message for "
    "this turn — wrap up now in plain text."
)

#: A post-turn hook: given the assistant turn the provider just produced and the live
#: transcript, it may append follow-up turns and returns whether the loop must continue even
#: when the turn carried no tool calls. The engine stays ignorant of *why* — the one collaborator
#: that knows (the code-execution Asset bridge: harvest the run's output files into Assets, then
#: feed their uuids back so the model can cite them) is injected, like the provider and tools.
TurnHook = Callable[[Message, "list[Message]"], bool]


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
        clock: Injectable source of the current UTC time, used to stamp each step-counter
            note and to measure per-step elapsed time. Defaults to the wall clock; a test
            drives it to assert the counter and timing deterministically.
    """

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        turn_hook: TurnHook | None = None,
        server_builtins: Sequence[str] = (),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.max_steps = max_steps
        self.turn_hook = turn_hook
        self.server_builtins = frozenset(server_builtins)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

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
        shown: list[Message] = []  # image turns injected this run, evicted before returning
        # The eviction must happen however the loop ends — including the reserve/error
        # path — or a viewed image's base64 lingers in the (mutated-in-place) transcript
        # and is re-sent on every later turn, the cost this exists to avoid.
        try:
            for step in range(1, self.max_steps + 1):
                started = self._clock()
                # Recency beats primacy for a changing value, and a stable prefix keeps the
                # provider's prompt cache hot — so the counter rides a *trailing* system turn,
                # re-appended each step, never a mutation of the head of the context.
                messages.append(Message.system(_step_note(step, self.max_steps, started)))
                reply = self.provider.chat(messages, tools=specs)
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
            _log.info("wake used %d/%d steps + reserve summary", self.max_steps, self.max_steps)
            return self._reserve_summary(messages)
        finally:
            _evict_images(shown)

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
        what remains, what the next turn should do. Its reply becomes the message posted to the
        timeline, so a cap event produces a transparent research artifact instead of a canned shrug.

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
            reply = self.provider.chat(messages, tools=None)
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
        """
        try:
            tool = self.tools.get(name)
        except KeyError:
            # A server-side built-in the model tried to invoke as a *function* — the
            # OpenRouter/GLM pass-through flake (issue #245). The generic "no tool named X"
            # reads as "your tool is gone" and sends the model into a retry spiral, so a
            # configured built-in gets targeted guidance back to its working (automatic) path;
            # a genuinely unknown name still gets the generic error below.
            if name in self.server_builtins:
                return _server_builtin_guidance(name)
            return f"Error: no tool named {name!r}."
        try:
            return tool.run(**arguments)
        except Exception as exc:  # noqa: BLE001 - any tool failure becomes model-readable
            return f"Error running {name!r}: {exc}"


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
            "remaining actions, summarize progress cleanly, and if work remains, schedule a "
            f"follow-up task before you run out. Step {max_steps} is your final action step — "
            "end it with a text reply to finish cleanly. Never ignore the step counter; treat "
            "it as a hard constraint."
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


def _server_builtin_guidance(name: str) -> str:
    """The targeted message for a server-side built-in mistakenly called as a function (#245)."""
    return (
        f"{name} runs server-side — you don't call it as a function. State what you want to find "
        "in your reply text and the search runs automatically. Do not retry this function call."
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
