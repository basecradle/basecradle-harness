"""A tool call whose arguments cannot bind is refused in words the model can act on (issue #779).

The live failure: a model called `tasks` with `instructions_text` (and, on another agent, `task_id`
and `instructions devam`). The engine handed those straight to `run(**arguments)`, and Python's own
`TypeError` — ``TasksTool.run() got an unexpected keyword argument 'instructions_text'`` — became the
result. It names a method, never the parameters the tool takes, and the model repeated the same call
six times. These pin what replaced it: the call never reaches `run()`, the model is told what was
wrong and what the tool accepts, and the line says ``reason=invalid_arguments``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil

import pytest

import basecradle_harness
from basecradle_harness import Engine, Message, Tool, ToolCall, ToolRegistry
from basecradle_harness._arguments import problem, refusal
from basecradle_harness._tasks import TasksTool


class ScriptedProvider:
    """Replays a fixed list of assistant turns, one per call."""

    def __init__(self, *replies: Message) -> None:
        self._replies = list(replies)

    def chat(self, messages, tools=None):
        return self._replies.pop(0)


class Schedule(Tool):
    """The `tasks` tool's shape — the same parameters, the same required `action` — and a run log."""

    name = "schedule"
    description = "Schedule a task."
    parameters = TasksTool.parameters

    def __init__(self) -> None:
        self.ran: list[dict] = []

    def run(self, action, uuid=None, instructions=None, activate_at=None, timeline=None):
        self.ran.append({"action": action, "instructions": instructions})
        return "Scheduled a task."


class Proxy(Tool):
    """An MCP-proxy-shaped tool: `run(**kwargs)`, its server validates its own arguments."""

    name = "proxy"
    description = "Forward anything."

    def run(self, **kwargs):
        return f"forwarded {sorted(kwargs)}"


class Clumsy(Tool):
    """A tool whose own body raises `TypeError` — a bug inside it, not a bad call."""

    name = "clumsy"
    description = "Add."
    parameters = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}

    def run(self, n):
        return n + "1"


class Hidden(Tool):
    """A tool whose `run` takes a keyword its schema never shows the model."""

    name = "hidden"
    description = "Look."
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}}

    def run(self, query=None, _internal=None):
        return "looked"


def _run(tool: Tool, *calls: dict) -> list[Message]:
    """Drive one turn that makes each call in order, then ends; return the turn's history."""
    replies = [
        Message.assistant(tool_calls=[ToolCall(id=f"c{n}", name=tool.name, arguments=arguments)])
        for n, arguments in enumerate(calls, 1)
    ]
    registry = ToolRegistry()
    registry.register(tool)
    history = [Message.user("go")]
    Engine(ScriptedProvider(*replies, Message.assistant(content="done")), registry).run(history)
    return history


def _results(history: list[Message]) -> list[str]:
    return [m.content for m in history if m.role == "tool"]


# --- the live failures, through the engine ----------------------------------------------------


def test_a_misspelled_argument_is_refused_with_its_near_match_and_never_reaches_run(caplog):
    tool = Schedule()
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        history = _run(
            tool,
            {"action": "create", "instructions_text": "@nova check in", "activate_at": "+1h"},
            {"action": "create", "instructions": "@nova check in", "activate_at": "+1h"},
        )

    first, second = _results(history)
    assert first == (
        "Error: 'schedule' was not run, and nothing was sent — it has no parameter "
        "'instructions_text' (did you mean 'instructions'?). Its parameters are: action, uuid, "
        "instructions, activate_at, timeline. Correct the call before making it again."
    )
    assert "TypeError" not in first and "unexpected keyword" not in first
    # The refused call never ran; the corrected one did, exactly once.
    assert tool.ran == [{"action": "create", "instructions": "@nova check in"}]
    assert second == "Scheduled a task."

    lines = [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "INFO" and r.getMessage().startswith("tool ")
    ]
    assert "outcome=error" in lines[0] and lines[0].endswith(" reason=invalid_arguments")
    assert "outcome=ok" in lines[1]
    (warning,) = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warning == (
        "tool name=schedule error=\"it has no parameter 'instructions_text' (did you mean "
        "'instructions'?). Its parameters are: action, uuid, instructions, activate_at, timeline.\""
    )


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        # @rowan, 2026-11-02: no near match, so no guess — just the names it does take.
        (
            {"action": "read", "task_id": "0190a1b2-c3d4-7e5f-8a6b-7c8d9e0f1a2b"},
            "no parameter 'task_id'. Its",
        ),
        # The other agent's five attempts: a stray word glued onto the key.
        ({"action": "create", "instructions devam": "x"}, "(did you mean 'instructions'?)"),
        ({}, "it needs 'action', which this call did not pass."),
    ],
)
def test_each_live_shape_is_refused_by_name(arguments, expected):
    tool = Schedule()
    (result,) = _results(_run(tool, arguments))
    assert expected in result
    assert "Its parameters are: action, uuid, instructions, activate_at, timeline." in result
    assert tool.ran == []


def test_the_real_tasks_tool_is_judged_by_its_own_signature():
    tasks = TasksTool.__new__(TasksTool)  # no platform context: `problem` never calls `run`
    fault = problem(tasks, {"action": "create", "instructions_text": "x", "activate_at": "+1h"})
    assert fault is not None and "(did you mean 'instructions'?)" in fault
    assert problem(tasks, {"action": "list"}) is None


@pytest.mark.parametrize(
    ("sent", "kind"), [(None, "nothing"), (["create"], "a list"), ("go", "a string")]
)
def test_arguments_that_are_not_an_object_are_refused_not_unpacked(sent, kind):
    fault = problem(Schedule(), sent)
    assert fault is not None
    assert fault.startswith(
        f"its arguments must be a JSON object of named parameters, and this call sent {kind}."
    )


# --- what it must not refuse ------------------------------------------------------------------


def test_a_tool_that_takes_any_keyword_is_never_refused():
    (result,) = _results(_run(Proxy(), {"anything": 1, "at_all": 2}))
    assert result == "forwarded ['anything', 'at_all']"


def test_a_type_error_inside_a_tools_own_body_is_still_reported_as_the_tools_failure(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        (result,) = _results(_run(Clumsy(), {"n": 1}))
    assert result.startswith("Error running 'clumsy': ")
    assert any("reason=exception" in r.getMessage() for r in caplog.records)


def test_only_the_names_the_schema_advertises_are_shown():
    fault = problem(Hidden(), {"qeury": "cats"})
    assert (
        fault == "it has no parameter 'qeury' (did you mean 'query'?). Its parameters are: query."
    )
    assert "_internal" not in fault


def test_several_faults_are_reported_together():
    fault = problem(Schedule(), {"task_id": "x"})
    assert fault == (
        "it has no parameter 'task_id'; it needs 'action', which this call did not pass. "
        "Its parameters are: action, uuid, instructions, activate_at, timeline."
    )


def test_a_tool_with_no_parameters_is_never_told_to_use_parameter_names():
    class Balance(Tool):
        name = "balance"
        description = "Read the balance."

        def run(self):
            return "$4.00"

    fault = problem(Balance(), {"account": "main"})
    assert fault == "it has no parameter 'account'. It takes no parameters."
    assert refusal("balance", fault) == (
        "Error: 'balance' was not run, and nothing was sent — it has no parameter 'account'. "
        "It takes no parameters. Correct the call before making it again."
    )


def test_a_keyword_only_parameter_is_judged_like_any_other():
    class Find(Tool):
        name = "find"
        description = "Find."
        parameters = {"type": "object", "properties": {"query": {}, "limit": {}}}

        def run(self, *, query, limit=10):
            return "found"

    assert problem(Find(), {"query": "cats"}) is None
    assert problem(Find(), {"limit": 3}) == (
        "it needs 'query', which this call did not pass. Its parameters are: query, limit."
    )


def test_a_decorated_run_is_judged_by_the_wrapper_the_engine_calls_never_the_inner_function():
    """A wrapper that supplies `client` itself binds `{"q": ...}`; the inner `run` alone would not."""
    import functools

    def with_client(run):
        @functools.wraps(run)
        def wrapper(self, **kwargs):
            return run(self, "the-client", **kwargs)

        return wrapper

    class Search(Tool):
        name = "search"
        description = "Search."
        parameters = {"type": "object", "properties": {"q": {}}}

        @with_client
        def run(self, client, q=None):
            return f"{client}: {q}"

    assert problem(Search(), {"q": "cats"}) is None
    (result,) = _results(_run(Search(), {"q": "cats"}))
    assert result == "the-client: cats"


def test_a_required_name_the_schema_hides_is_the_tools_defect_and_raises_as_one(caplog):
    """The model was never shown it, so it cannot be told to pass it: the tool's own `exception`."""

    class Broken(Tool):
        name = "broken"
        description = "Broken."
        parameters = {"type": "object", "properties": {"q": {}}}

        def run(self, secret, q=None):
            return "ran"

    assert problem(Broken(), {"q": "x"}) is None
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        (result,) = _results(_run(Broken(), {"q": "x"}))
    assert result.startswith("Error running 'broken': ")
    assert any(r.getMessage().endswith("reason=exception") for r in caplog.records)
    # Beside a fault the model can fix, only the names it was shown are named.
    assert problem(Broken(), {"z": 1}) == "it has no parameter 'z'. Its parameters are: q."


# --- the recovery's seam ----------------------------------------------------------------------


def test_a_reissued_call_is_never_told_nothing_was_sent():
    """A dead wake's create may have bound under an older signature, so its outcome is unknown."""
    registry = ToolRegistry()
    tool = Schedule()
    registry.register(tool)
    engine = Engine(ScriptedProvider(), registry)

    text = engine.run_tool("schedule", {"action": "create", "task_id": "x"})

    assert text.startswith("Error: this call to 'schedule' was not re-issued — it has no parameter")
    assert "whether it reached anyone is unknown" in text
    assert "nothing was sent" not in text
    assert tool.ran == []
    assert refusal("schedule", "f.").startswith(
        "Error: 'schedule' was not run, and nothing was sent"
    )


# --- the schema half: every shipped tool advertises exactly what its run() takes ---------------


def _shipped_tools() -> list[type[Tool]]:
    found: dict[str, type[Tool]] = {}
    for module in pkgutil.iter_modules(basecradle_harness.__path__):
        loaded = importlib.import_module(f"basecradle_harness.{module.name}")
        for _, cls in inspect.getmembers(loaded, inspect.isclass):
            if _is_tool(cls) and not inspect.isabstract(cls) and cls.__module__ == loaded.__name__:
                found[f"{cls.__module__}.{cls.__qualname__}"] = cls
    return [found[key] for key in sorted(found)]


def _is_tool(cls: object) -> bool:
    """`issubclass`, for anything `inspect.isclass` admits — a generic alias like `list[str]` too."""
    try:
        return issubclass(cls, Tool)  # type: ignore[arg-type]
    except TypeError:
        return False


def _schema_and_signature(cls: type[Tool]) -> tuple[dict, inspect.Signature]:
    if cls.__name__ == "AssetsTool":  # its schema is built per instance, from the actions it offers
        from basecradle_harness import _assets

        return _assets._parameters(_assets._ALL_ACTIONS), inspect.signature(cls.run)
    return inspect.getattr_static(cls, "parameters"), inspect.signature(cls.run)


def test_the_scan_finds_the_shipped_tools():
    names = {cls.__name__ for cls in _shipped_tools()}
    assert {"TasksTool", "AssetsTool", "MessagesTool", "TimelinesTool"} <= names


@pytest.mark.parametrize("cls", _shipped_tools(), ids=lambda cls: cls.__name__)
def test_every_shipped_tool_advertises_exactly_the_parameters_its_run_takes(cls):
    """A schema naming a key `run()` refuses is a call the model is told to make and cannot."""
    schema, signature = _schema_and_signature(cls)
    parameters = list(signature.parameters.values())[1:]  # past `self`
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        pytest.skip(f"{cls.__name__} forwards its arguments (`**kwargs`)")
    accepted = {p.name for p in parameters}
    needed = {p.name for p in parameters if p.default is inspect.Parameter.empty}
    assert set(schema.get("properties", {})) == accepted
    assert needed <= set(schema.get("required", []))
