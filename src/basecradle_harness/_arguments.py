"""Checking a tool call's arguments against the tool before it runs (issue #779).

A model's tool-call arguments arrive as the model wrote them: the adapters decode the JSON into an
object (a non-object reads as no arguments, `_openai_wire.tool_arguments`) and nothing else, so a
key the tool never declared — `instructions_text`, `task_id`, `instructions
devam` — reaches the dispatch untouched. Handed straight to `run(**arguments)`, it raised Python's
own ``TypeError`` (``TasksTool.run() got an unexpected keyword argument 'task_id'``), and that text
became the tool's result. It names a Python method, never the parameters the tool *does* take, so a
model reading it has nothing to correct its next call with — a live agent repeated the same broken
call six times in an hour.

So the engine asks this module first, and a call that cannot bind comes back as a refusal that says
what was wrong and what the tool accepts, with the nearest name when one is close.

**The authority is `run()`'s own signature, never the schema.** The question is exactly "would this
call raise before the tool's first line", and only the signature answers it: a tool that takes
``**kwargs`` (every MCP proxy, whose server validates its own arguments) is never refused for a name
it was not written to expect; a third-party overlay tool whose schema disagrees with its `run()` is
judged by what it can actually be called with. The signature read is the callable's *own*, never the
one a decorator's `functools.wraps` points at: a wrapper that supplies an argument itself takes calls
the inner function would refuse, and refusing those would invent a ``TypeError``. The schema decides
only which names the model is *shown*, so a keyword `run()` takes but the schema never advertised
stays out of the message — and a *required* one the model was never shown is the tool's own defect,
not the call's, so it is left to raise and be reported as the tool's ``exception``.

This checks the call's **shape** — an object, every name known, every required name present — and
nothing about the values. Types and enums stay each tool's own business, as they were: the shape is
the one failure a tool's body never gets the chance to explain.
"""

from __future__ import annotations

import difflib
import inspect
from collections.abc import Mapping
from typing import Any

#: How alike a misspelling must be to its parameter for the refusal to suggest it: `difflib`'s ratio,
#: where `instructions_text` → `instructions` scores 0.83 and `task_id` → `uuid` scores 0.36. The
#: library's own default, chosen because a wrong suggestion costs more than none.
SUGGEST_CUTOFF = 0.6

#: The parameter kinds a call's named arguments can bind to.
_NAMED = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def problem(tool: Any, arguments: Any) -> str | None:
    """What stops `tool.run(**arguments)` from binding, in words for the model — or ``None``.

    ``None`` means the call binds and the tool runs. A run with no signature to read (a builtin, a
    C callable) is never refused: this module exists to explain a ``TypeError``, never to invent one.
    """
    try:
        parameters = inspect.signature(tool.run, follow_wrapped=False).parameters.values()
    except (TypeError, ValueError):
        return None
    named = {p.name: p for p in parameters if p.kind in _NAMED}
    shown = _shown(tool, named)
    if not isinstance(arguments, Mapping):
        return (
            f"its arguments must be a JSON object of named parameters, and this call sent "
            f"{_kind(arguments)}. {_accepts(shown)}"
        )
    faults = []
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        faults += [
            f"it has no parameter {key!r}{_suggestion(key, shown)}"
            for key in arguments
            if key not in named
        ]
    missing = [
        name
        for name, p in named.items()
        if p.default is inspect.Parameter.empty and name not in arguments
    ]
    if any(name not in shown for name in missing) and not faults:
        return None  # a required name the model was never shown: the tool's defect, not the call's
    if missing := [name for name in missing if name in shown]:
        names = ", ".join(repr(name) for name in missing)
        faults.append(f"it needs {names}, which this call did not pass")
    if not faults:
        return None
    return f"{'; '.join(faults)}. {_accepts(shown)}"


def refusal(name: str, fault: str, *, reissue: bool = False) -> str:
    """What the model is told instead of a result, when its call to `name` did not bind.

    **What happened depends on the seam**, as for the elision refusal (`_elision.refusal`). A live
    call refused here never ran, so nothing was sent. A call the recovery is re-issuing was first
    made by a wake that died before recording its outcome; the same arguments cannot have bound
    then either, unless the tool's parameters changed across an upgrade in between — so the model is
    told the honest thing, that whether it reached anyone is unknown, rather than a "nothing was
    sent" that is true only almost always.
    """
    if reissue:
        return (
            f"Error: this call to {name!r} was not re-issued — {fault} The wake that first made this "
            f"call was killed before its result was recorded, so whether it reached anyone is "
            f"unknown, and it has not been sent again. If it matters, check for yourself before "
            f"sending anything."
        )
    return (
        f"Error: {name!r} was not run, and nothing was sent — {fault} Correct the call before "
        f"making it again."
    )


def _shown(tool: Any, named: Mapping[str, inspect.Parameter]) -> list[str]:
    """The parameter names the model may be told about: the schema's, in its order, that `run` takes.

    Never `run`'s own names beyond those: a name the schema does not advertise is one the model was
    not offered, and the refusal does not offer it either.
    """
    schema = getattr(tool, "parameters", None)
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    advertised = list(properties) if isinstance(properties, Mapping) else []
    return [key for key in advertised if key in named]


def _accepts(shown: list[str]) -> str:
    """The sentence naming what the tool takes."""
    if not shown:
        return "It takes no parameters."
    return f"Its parameters are: {', '.join(shown)}."


def _suggestion(key: str, shown: list[str]) -> str:
    """`` (did you mean 'x'?)`` for the closest advertised name, or nothing when none is close."""
    if not isinstance(key, str):
        return ""
    close = difflib.get_close_matches(key, shown, n=1, cutoff=SUGGEST_CUTOFF)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _kind(value: Any) -> str:
    """A JSON-flavoured name for what the call sent instead of an object."""
    if isinstance(value, str):
        return "a string"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, (list, tuple)):
        return "a list"
    if value is None:
        return "nothing"
    return f"a {type(value).__name__}"
