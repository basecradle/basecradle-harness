"""The brain check: would this brain, configured this way, accept a real wake's request? (issue #947)

On 2026-11-20 an agent moved to a new vendor and model failed its first real wake with the vendor's
400 (issue #945). A hand-made call had passed beforehand, because it was not shaped like a wake: it
had no step note after the conversation. Nothing could send a wake-shaped request without a platform
event — `basecradle-harness-wake` needs a timeline, and ``--resolved-config`` calls no model — so a
change of brain was proven only by the wake that broke on it.

`basecradle-harness-brain-check` sends that request and nothing else. Each property has a broken
form, and the form is why it is stated:

- **The configuration is read as a wake reads it** — the ``AI_*`` variables, ``model_params.json``
  and the tool set a wake resolves — through the same functions (`_config_from_env`,
  `_resolve_tools`, `brain_from_config`). ``--model-params`` names a candidate file, read by the
  reader a wake uses, so a change is checked before it is written into the agent's home.
- **The request is built by the engine, never copied.** A `Harness` built as a wake builds one,
  an in-memory `Session` on it, and the brief composed by the wake's own composition
  (`_wake.wake_brief_parts`): the charter, the brief before a fixed user turn, the step note after
  it, every tool definition the resolved set exposes. A request assembled here would be a second
  model of the wake's, and it would stop matching the day the first one changed, which is the
  failure this check exists to catch. The two parts only a platform or a memory read can supply,
  the live dashboard and the recalled memory, are absent, as they are on a wake whose dashboard
  fetch failed and whose provider recalled nothing; both ride inside the brief, so the order of
  roles a vendor judges is the wake's.
- **At most two rounds.** When the model calls a tool, the call is answered with a fixed result
  (`STAND_IN_RESULT`) and the engine sends its second step, so the order of a later step is tested
  too — the order #945's fix had to cover. A turn hook ends the run after a second reply that still
  calls tools (`_Rounds`), before a third request can be sent; it never ends a run the model ended.
- **It writes nothing and touches nothing.** No platform client is built, no transcript or claim is
  written (the session has no path), no memory is read or written, and no tool runs: every resolved
  tool is offered through a `StandIn` that carries its definition and returns the fixed result. The
  config-home reconcile a wake runs first is skipped, because it writes; the overlay is read as it
  stands, as ``--resolved-config`` reads it. MCP servers start only so their tool lists can be read,
  and stop with the process.
- **One JSON verdict on stdout, and the exit code says the same thing**: ``ok`` (0) with each
  call's usage; ``refused`` (70) with the vendor's HTTP status and its error text, scrubbed as every
  logged value is; ``config_error`` (64); ``failed`` (69) when no answer came at all (a timeout, a
  dropped connection, a response nobody could read), which is not a refusal and not a
  configuration fault, and which reporting as either would be a false verdict. No key and no prompt
  text is ever in it. The journal lines a wake writes (``llm``, ``tool``, the step ledger) go to
  stderr as they would on a wake.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from typing import Any

from basecradle_harness._basecradle import (
    _config_from_env,
    _configure_logging,
    _max_steps_from_env,
    _onboard_from_env,
    _profile_from_env,
    _resolve_tools,
    _response_retries_from_env,
    brain_from_config,
    system_turns_from_env,
)
from basecradle_harness._brief import brief_section_sizes, join_brief
from basecradle_harness._exceptions import (
    HarnessError,
    ProviderAPIError,
    ProviderConnectionError,
    ProviderResponseError,
    ProviderToolSchemaError,
)
from basecradle_harness._harness import Harness
from basecradle_harness._install import charter_from_env
from basecradle_harness._messages import Message, ToolSpec
from basecradle_harness._observability import FAILED, RETRYING, scrub, tally_llm_calls
from basecradle_harness._plugins import ResolvedTools
from basecradle_harness._provider import Provider
from basecradle_harness._tools import Tool
from basecradle_harness._wake import wake_brief_parts

_log = logging.getLogger("basecradle_harness")

OK = "ok"
REFUSED = "refused"
CONFIG_ERROR = "config_error"
FAILED_VERDICT = "failed"

#: The exit code of each verdict: sysexits' ``EX_USAGE`` for a configuration the harness refuses,
#: ``EX_UNAVAILABLE`` for a call that got no answer, and ``EX_SOFTWARE`` for a vendor's refusal.
EXIT_CODES = {OK: 0, CONFIG_ERROR: 64, FAILED_VERDICT: 69, REFUSED: 70}

#: The most model calls one check makes: the first step, and the second when the first called a tool.
ROUNDS = 2

#: The session the check's turn runs in. Only an adapter that routes by conversation reads it.
SOURCE = "brain-check"

#: The fixed user turn: the newest message a wake's first step carries. It asks for a reply through
#: the messages tool because a reply is a tool call on a wake (the Unspoken Channel), and a tool call
#: is what takes the check to its second step, whose order #945's fix had to cover. An agent with no
#: messages tool answers in text, and the check ends after one round.
USER_TURN = (
    "Hello. This message checks that you can be reached. "
    "Please reply to it with your messages tool, in one short sentence."
)

#: What a tool call made during the check returns. No tool runs.
STAND_IN_RESULT = "Done."

#: What a configuration fault raises while the brain is built or before a request is sent. A plain
#: `ProviderError` is one: an adapter raises it for a request its SDK would not send.
_CONFIG_FAULTS = (HarnessError, ValueError, KeyError, ImportError)

#: What a call that got no usable answer raises: a timeout or a dropped connection
#: (`ProviderConnectionError`), or a response nobody could read (`ProviderResponseError`).
_NO_ANSWER = (ProviderConnectionError, ProviderResponseError)


class StandIn(Tool):
    """A resolved tool, offered by its own definition, that answers every call with a fixed result.

    The definition is the real tool's, read through it (`to_spec`), so the request offers exactly
    what a wake's does; its capabilities and its load refusal are the real tool's too, so the policy
    admits the same set. Only `run` differs, and that is the point: a check that ran the real tool
    would post, spend or write.
    """

    def __init__(self, real: Tool) -> None:
        self.real = real
        self.name = real.name
        self.description = real.description
        self.parameters = real.parameters
        self.requires = real.requires
        self.executor = real.executor

    def to_spec(self) -> ToolSpec:
        return self.real.to_spec()

    def load_refusal(self) -> str | None:
        return self.real.load_refusal()

    def run(self, **kwargs: Any) -> str:
        return STAND_IN_RESULT


class _RoundsSpent(Exception):
    """Raised by `_Rounds` to end the run once `ROUNDS` replies have come back."""


class _Rounds:
    """The engine's turn hook for the check: counts replies, and ends the run after the last round.

    The engine calls it after each step's tool calls are answered. It returns ``False`` (extend
    nothing) until `ROUNDS` replies have come back; a reply after that which still calls tools
    would send a third request, so it raises instead. A reply that ends the turn ends it as it
    would on a wake.
    """

    def __init__(self) -> None:
        self.replies = 0
        self.tool_called = False

    def __call__(self, reply: Message, messages: list[Message]) -> bool:
        self.replies += 1
        self.tool_called = self.tool_called or bool(reply.tool_calls)
        if self.replies >= ROUNDS and reply.tool_calls:
            raise _RoundsSpent
        return False


def check(model_params: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Run the check and return its verdict (see the module docstring for what each one means)."""
    report: dict[str, Any] = {}
    try:
        provider_name, sdk, surface = _config_from_env()
        report.update(
            provider=provider_name,
            sdk=sdk,
            surface=surface,
            model=os.environ.get("AI_MODEL") or None,
            system_turns=system_turns_from_env(sdk),
            model_params=None if model_params is None else str(model_params),
        )
        resolved, _memory = _resolve_tools(provider_name, sdk, surface)
        provider, _bridge = brain_from_config(
            provider_name, sdk, surface, resolved, params_file=model_params
        )
    except _CONFIG_FAULTS as error:
        return _verdict(CONFIG_ERROR, report, error=_text(error))
    try:
        return _send(provider, resolved, report)
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()


def _send(provider: Provider, resolved: ResolvedTools, report: dict[str, Any]) -> dict[str, Any]:
    rounds = _Rounds()
    try:
        _, policy = _profile_from_env()
        # Built as `WakeAgent.from_env` builds its `Harness`, with three differences, each one of
        # the check's: stand-in tools, no home (nothing persists), and its own turn hook in place
        # of the code-execution bridge's (which creates Assets) — and no compactor, so nothing but
        # the one turn can call the model.
        harness = Harness(
            provider,
            system_prompt=charter_from_env(),
            tools=[StandIn(tool) for tool in resolved.tools],
            policy=policy,
            max_steps=_max_steps_from_env(),
            response_retries=_response_retries_from_env(),
            server_builtins=resolved.builtins,
            withheld_tools=resolved.withheld,
            turn_hook=rounds,
        )
        brief, sections = _brief(harness, resolved) if _onboard_from_env() else (None, None)
    except _CONFIG_FAULTS as error:
        return _verdict(CONFIG_ERROR, report, error=_text(error))
    report["tools_offered"] = len(harness.tools.specs())
    with tally_llm_calls() as calls:
        try:
            harness.session(SOURCE).send(USER_TURN, brief=brief, brief_sections=sections)
        except _RoundsSpent:
            pass
        except (ProviderAPIError, ProviderToolSchemaError) as error:
            return _verdict(
                REFUSED,
                report,
                rounds,
                calls,
                status=getattr(error, "status_code", None),
                error=_text(error),
            )
        except _NO_ANSWER as error:
            return _verdict(FAILED_VERDICT, report, rounds, calls, error=_text(error))
        except _CONFIG_FAULTS as error:
            # A fault the build could not see, raised by the adapter as it shaped the request: a
            # `model_params.json` key the SDK does not take is the common one (OpenRouter's typed
            # SDK raises a plain `ProviderError` naming the file, a `HarnessError`).
            return _verdict(CONFIG_ERROR, report, rounds, calls, error=_text(error))
    return _verdict(OK, report, rounds, calls)


def _brief(harness: Harness, resolved: ResolvedTools) -> tuple[str | None, dict[str, int] | None]:
    """The wake's brief for this agent, without the dashboard or recall a wake would fetch.

    A failure costs the brief, as it does on a wake (`WakeAgent._compose_brief`), so the check sends
    what that wake would send.
    """
    try:
        parts = wake_brief_parts(
            harness,
            manifest=resolved.manifest,
            defects=resolved.broken,
            safety=resolved.notices,
            mcp_about=resolved.mcp_about,
            dashboard=None,
            memory=None,
        )
        return join_brief(parts), brief_section_sizes(parts)
    except Exception:  # noqa: BLE001 - the brief never breaks a wake, so it never breaks the check
        _log.warning(
            "Failed to compose the persistent brief; proceeding without it.", exc_info=True
        )
        return None, None


def _verdict(
    verdict: str,
    report: dict[str, Any],
    rounds: _Rounds | None = None,
    calls: Sequence[dict[str, Any]] = (),
    **fields: Any,
) -> dict[str, Any]:
    out = {"verdict": verdict, **report, **fields}
    if rounds is not None:
        out["rounds"] = rounds.replies
        out["tool_called"] = rounds.tool_called
        out["calls"] = list(calls)
        out["usage"] = _usage(calls)
    return out


def _usage(calls: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The billed calls' token counts summed, and their cost when every one of them was priced.

    A billed call is one the vendor answered: a ``retrying`` or ``failed`` line carries no usage.
    """
    billed = [call for call in calls if call.get("outcome") not in (RETRYING, FAILED)]
    totals: dict[str, Any] = {"calls": len(billed)}
    for call in billed:
        for key, value in call.items():
            if key.startswith(("tokens_", "cached_", "cache_")) and isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    costs = [call.get("cost") for call in billed]
    if billed and all(isinstance(cost, float) for cost in costs):
        totals["cost"] = round(sum(costs), 10)
    return totals


def _text(error: BaseException) -> str:
    """An error's text for the verdict: scrubbed as every logged value is, and never empty."""
    return scrub(str(error)) or type(error).__name__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="basecradle-harness-brain-check",
        description=(
            "Send one wake-shaped request to the configured brain and print a JSON verdict: "
            "ok (exit 0), refused (70), config_error (64) or failed (69). Writes nothing; "
            "calls no platform API, reads no memory and runs no tool."
        ),
    )
    parser.add_argument(
        "--model-params",
        metavar="PATH",
        help=(
            "read the brain's tuning from this model_params.json instead of the config home's, "
            "to check a candidate before it is installed."
        ),
    )
    args = parser.parse_args(argv)
    _configure_logging()
    verdict = check(model_params=args.model_params)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    sys.stdout.flush()
    return EXIT_CODES[verdict["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
