"""A tool that answers the model with its failure has still failed (issue #822).

Most tool failures reach the engine as a plain string, because feeding the failure back is how a
model recovers. So the `tool` line's ``outcome`` is decided by a structural verdict (`tool_failed`,
an MCP ``isError``), and the *Tool Outcomes* chart is only as true as these tests are.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx
from basecradle import BaseCradleError

import basecradle_harness
from basecradle_harness import Engine, Policy, Tool, ToolRegistry, tool_failed
from basecradle_harness._delete import DeleteTool
from basecradle_harness._exceptions import ProviderError, ProviderRateLimitError
from basecradle_harness._mcp import McpTool
from basecradle_harness._mempalace import MemPalaceMemoryProvider
from basecradle_harness._observability import metered
from basecradle_harness._openrouter_account import OpenRouterAccountBalanceTool
from basecradle_harness._platform import PlatformContext
from basecradle_harness._reads import MessagesTool
from basecradle_harness._shell import ShellTool
from basecradle_harness._system_prompt import SystemPromptEditTool
from basecradle_harness._webfetch import WebFetchTool

TIMELINE = "0191f3a4-7c11-7b5e-9a42-1d2e3f4a5b6c"


class _Scripted(Tool):
    name = "scripted"
    description = "Runs the function it was built with."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, body) -> None:
        self._body = body

    def run(self) -> str:
        return self._body()


def _lines(caplog) -> tuple[str, list[str]]:
    """The run's one INFO `tool` line, and any WARNING companions."""
    messages = [r for r in caplog.records if r.getMessage().startswith("tool ")]
    (line,) = [r.getMessage() for r in messages if r.levelname == "INFO"]
    return line, [r.getMessage() for r in messages if r.levelname == "WARNING"]


def _run(tool: Tool, caplog, **arguments) -> tuple[object, str, list[str]]:
    registry = ToolRegistry(policy=Policy.unlocked())
    registry.register(tool)
    with caplog.at_level(logging.INFO):
        result = Engine(provider=None, tools=registry)._run_tool(tool.name, arguments)  # type: ignore[arg-type]
    return (result, *_lines(caplog))


def test_a_marked_failure_is_an_error_and_the_model_reads_the_same_text(caplog):
    text = "Error: 'weather' needs a city."
    result, line, companions = _run(
        _Scripted(lambda: tool_failed(text, reason="invalid_arguments")), caplog
    )
    assert result == text
    assert "outcome=error" in line and line.endswith(" reason=invalid_arguments")
    assert companions == ["tool name=scripted error=\"'weather' needs a city.\""]


def test_the_word_error_in_a_result_is_never_the_verdict(caplog):
    _, line, companions = _run(_Scripted(lambda: "Error: a page that says this."), caplog)
    assert "outcome=ok" in line and "reason=" not in line
    assert companions == []


def test_a_vendor_fault_the_meter_recorded_outranks_the_tools_own_mark(caplog):
    def body() -> str:
        try:
            with metered(provider="xai", kind="image.generate", model="m"):
                raise ProviderRateLimitError("slow down", status_code=429)
        except ProviderError:
            return tool_failed("Error generating image: slow down", reason="provider_error")

    _, line, _ = _run(_Scripted(body), caplog)
    assert line.endswith(" reason=rate_limited") and "provider=xai" in line


def test_a_mark_is_a_no_op_outside_a_tool_run():
    assert tool_failed("Error: nothing listens.", reason="invalid_arguments") == (
        "Error: nothing listens."
    )


@pytest.mark.parametrize(
    ("result", "outcome"),
    [
        ({"content": [{"type": "text", "text": "page crashed"}], "isError": True}, "error"),
        ({"content": [{"type": "text", "text": "Error: the page says so"}]}, "ok"),
    ],
)
def test_an_mcp_server_verdict_is_the_field_never_the_text(caplog, result, outcome):
    tool = McpTool(
        server="browser",
        remote_name="click",
        description="",
        parameters={},
        client=SimpleNamespace(workdir=None, call_tool=lambda name, args: result),
    )
    _, line, _ = _run(tool, caplog)
    assert f"outcome={outcome}" in line and "executor=mcp" in line
    assert ("reason=tool_error" in line) == (outcome == "error")


def _messages(client: object) -> MessagesTool:
    tool = MessagesTool()
    tool.bind(PlatformContext(client=client, timeline=TIMELINE))
    return tool


def test_a_shipped_tool_tells_a_bad_call_from_a_platform_refusal(caplog):
    _, line, _ = _run(_messages(SimpleNamespace()), caplog, action="read")
    assert line.endswith(" reason=invalid_arguments")
    caplog.clear()

    def refuse(**_filters: object) -> None:
        raise BaseCradleError("You are not a viewer of this timeline.")

    client = SimpleNamespace(messages=SimpleNamespace(filter=refuse))
    result, line, companions = _run(_messages(client), caplog, action="list")
    assert isinstance(result, str) and "not a viewer" in result
    assert line.endswith(" reason=platform_error")
    assert len(companions) == 1 and "not a viewer" in companions[0]


def test_a_nonzero_exit_is_the_commands_answer_and_a_kill_is_a_failure(caplog):
    shell = ShellTool(default_timeout=1, max_timeout=1)
    _, line, _ = _run(shell, caplog, command="exit 3")
    assert "outcome=ok" in line
    caplog.clear()
    _, line, companions = _run(shell, caplog, command="sleep 5")
    assert line.endswith(" reason=timeout")
    assert "ran past its 1s timeout" in companions[0]


def _timeline(uuid: str) -> SimpleNamespace:
    return SimpleNamespace(name="Test", uuid=uuid, items=[])


@pytest.mark.parametrize(("confirm", "marked"), [(None, False), ("not-the-uuid", True)])
def test_a_confirm_gate_preview_is_ok_and_a_wrong_confirm_is_a_bad_call(caplog, confirm, marked):
    tool = DeleteTool()
    client = SimpleNamespace(timelines=SimpleNamespace(get=_timeline))
    tool.bind(PlatformContext(client=client, timeline=TIMELINE))
    _, line, _ = _run(tool, caplog, confirm=confirm)
    assert ("reason=invalid_arguments" in line) == marked
    assert ("outcome=ok" in line) == (not marked)


@pytest.mark.parametrize(("confirm", "marked"), [(None, False), ("not-the-token", True)])
def test_the_system_prompt_gate_marks_only_a_wrong_token(
    caplog, tmp_path, monkeypatch, confirm, marked
):
    from basecradle_harness._install import install

    root = tmp_path / "config"
    install(root)
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(root))
    _, line, _ = _run(SystemPromptEditTool(), caplog, content="new", confirm=confirm)
    assert ("reason=invalid_arguments" in line) == marked
    assert ("outcome=ok" in line) == (not marked)


def test_a_fetched_error_status_is_a_failed_fetch(caplog):
    with respx.mock() as mock:
        mock.route(host="example.com").mock(return_value=httpx.Response(500))
        _, line, _ = _run(WebFetchTool(), caplog, url="https://example.com/")
    assert line.endswith(" reason=fetch_failed")


def test_a_rejected_management_key_is_a_config_fault(caplog, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", "sk-or-v1-fabricatedmanagementkey0000")
    with respx.mock() as mock:
        mock.route(host="openrouter.ai").mock(return_value=httpx.Response(401))
        _, line, _ = _run(OpenRouterAccountBalanceTool(), caplog)
    assert line.endswith(" reason=config:auth")


def test_a_failed_memory_search_is_not_a_clean_no_matches(caplog, tmp_path, fake_mempalace):
    _, searcher = fake_mempalace
    palace = tmp_path / "palace"
    palace.mkdir()
    (tool,) = MemPalaceMemoryProvider(palace).tools()

    searcher.result = {"error": "Search error: john lives in dallas", "results": []}
    result, line, companions = _run(tool, caplog, query="where does john live")
    assert result == "No memories match 'where does john live'."
    assert line.endswith(" reason=memory_error")
    assert "john lives" not in companions[0]  # upstream's text can carry a peer's words
    caplog.clear()

    searcher.result = {"results": []}
    _, line, _ = _run(tool, caplog, query="where does john live")
    assert "outcome=ok" in line and "reason=" not in line


#: The engine's own refusals: it writes their `tool` line itself, with their slug.
_WRITES_ITS_OWN_LINE = {"_engine.py", "_arguments.py", "_elision.py"}
_FAILURE_WORDS = ("Error", "Couldn't", "Cannot", "Refused")


def _literal_starts(node: ast.expr) -> list[str]:
    """The literal text each way this expression can begin, through a conditional or a `+`."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr) and node.values:
        return _literal_starts(node.values[0])
    if isinstance(node, ast.IfExp):
        return _literal_starts(node.body) + _literal_starts(node.orelse)
    if isinstance(node, ast.BinOp):
        return _literal_starts(node.left)
    return []


def test_no_tool_hands_a_failure_to_the_model_unmarked():
    """Absence emits no signal: a new failure-as-text return would read `ok` with nothing to see.

    A floor, not a proof: it reads how a failure sentence begins in a `return`, so a failure worded
    some other way, or built up in a variable first, is the reviewer's to catch.
    """
    package = Path(basecradle_harness.__file__).parent
    unmarked = []
    for path in sorted(package.rglob("*.py")):
        if path.name in _WRITES_ITS_OWN_LINE:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Return)
                and node.value is not None
                and any(s.startswith(_FAILURE_WORDS) for s in _literal_starts(node.value))
            ):
                unmarked.append(f"{path.name}:{node.lineno}")
    assert unmarked == []
