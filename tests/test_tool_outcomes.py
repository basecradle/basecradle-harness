"""A tool that answers the model with its failure has still failed (issue #822).

Most tool failures reach the engine as a plain string, because feeding the failure back is how a
model recovers. So the `tool` line's ``outcome`` is decided by a structural verdict (`tool_failed`,
an MCP ``isError``), and the *Tool Outcomes* chart is only as true as these tests are.
"""

from __future__ import annotations

import ast
import logging
import socket
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


@pytest.fixture(autouse=True)
def _plain(monkeypatch):
    """Lines read as bytes, without the verdict's color, so a field's neighbors can be asserted."""
    monkeypatch.setenv("NO_COLOR", "1")


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
    assert f" target=timeline:{TIMELINE} " in companions[0]


MESSAGE = "0191f3a4-8d22-7c6f-8b53-2e3f4a5b6c7d"


def test_a_failed_platform_read_names_what_it_asked_for(caplog):
    """The platform's own text does not say which uuid it could not find, so the companion line
    does (issue #892): two `messages` reads failed in a red-team run and nothing could say why."""

    def missing(uuid):
        raise BaseCradleError("No record exists for the given UUID.")

    client = SimpleNamespace(messages=SimpleNamespace(get=missing))
    result, line, companions = _run(_messages(client), caplog, action="read", uuid=MESSAGE)
    assert result == "Couldn't read messages: No record exists for the given UUID."
    assert line.endswith(" outcome=error reason=platform_error") and "target=" not in line
    assert companions == [
        (
            f"tool name=messages target=message:{MESSAGE} error=\"Couldn't read messages: "
            'No record exists for the given UUID."'
        )
    ]


def test_an_uncaught_platform_error_still_names_its_target(caplog):
    """A tool that lets the SDK's error escape (the assets tool) is marked by the engine, and the
    target its run recorded still rides the companion."""
    from basecradle_harness._assets import AssetsTool

    def missing(uuid):
        raise BaseCradleError("No record exists for the given UUID.")

    tool = AssetsTool()
    tool.bind(
        PlatformContext(
            client=SimpleNamespace(assets=SimpleNamespace(get=missing)), timeline=TIMELINE
        )
    )
    _, line, companions = _run(tool, caplog, action="read", uuid=MESSAGE)
    assert line.endswith(" reason=exception")
    assert len(companions) == 1 and f" target=asset:{MESSAGE} " in companions[0]


def test_a_target_is_never_written_when_the_run_succeeds(caplog):
    message = SimpleNamespace(
        content=SimpleNamespace(uuid=MESSAGE, body="hi"),
        user=SimpleNamespace(handle="john"),
        created_at="2026-11-14T21:38:00Z",
    )
    client = SimpleNamespace(messages=SimpleNamespace(get=lambda uuid: message))
    _, line, companions = _run(_messages(client), caplog, action="read", uuid=MESSAGE)
    assert "outcome=ok" in line and "target=" not in line
    assert companions == []


@pytest.mark.parametrize("command", ["yes", "printf 'abcdefghijklmnop'"])
def test_past_the_output_cap_the_agent_gets_the_head_and_the_line_says_it_was_cut(caplog, command):
    """Past the shell output limit the agent gets the head, told it was cut and at what size, and
    the line is `ok` with `cut_at` (issue #892) — never `degraded`, which pages a person. `yes` is
    still writing and is killed; the `printf` has usually exited by the time the reader stops — a
    race, so both are the one line."""
    shell = ShellTool(max_output=10, drain_timeout=2)
    result, line, companions = _run(shell, caplog, command=command, timeout=30)
    assert result.startswith(("y\ny\ny\ny\ny", "abcdefghij"))
    assert "[output cut at 10 characters" in result
    assert line.endswith(" outcome=ok cut_at=10") and "reason=" not in line
    assert companions == []  # nothing failed


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


@pytest.fixture
def public_dns(monkeypatch):
    """`example.com` resolves to a public address, offline: the SSRF check resolves every host."""

    def resolve(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)


@pytest.mark.parametrize("status", [200, 404, 410, 503])
def test_a_fetch_the_server_answered_is_ok_and_carries_the_status(caplog, public_dns, status):
    """An answer is not a tool error (issue #892): a 404 is the far end's answer to the URL the
    model chose, so the line is `ok`, the status is on it, and no WARNING is written — guessed URLs
    stay countable, and a reader filtering for warnings sees only a fetch that broke."""
    with respx.mock() as mock:
        mock.route(host="example.com").mock(return_value=httpx.Response(status, text="body"))
        result, line, companions = _run(WebFetchTool(), caplog, url="https://example.com/guess")
    assert f" outcome=ok status={status}" in line and "reason=" not in line
    assert companions == []
    if status >= 400:  # what the model reads is unchanged
        assert (
            result
            == f"The server returned HTTP {status} for https://example.com/guess. Nothing to read."
        )


def test_a_host_dns_says_does_not_exist_is_an_answer_too(caplog, monkeypatch):
    def no_such_name(host, port, *args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known")

    monkeypatch.setattr(socket, "getaddrinfo", no_such_name)
    result, line, companions = _run(WebFetchTool(), caplog, url="https://no-such-host.example/")
    assert (
        result
        == "Couldn't fetch 'https://no-such-host.example/': could not resolve host 'no-such-host.example'."
    )
    assert " outcome=ok status=nxdomain" in line and "reason=" not in line
    assert companions == []


def test_a_resolver_that_could_not_answer_is_still_a_failed_fetch(caplog, monkeypatch):
    def try_again(host, port, *args, **kwargs):
        raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")

    monkeypatch.setattr(socket, "getaddrinfo", try_again)
    _, line, companions = _run(WebFetchTool(), caplog, url="https://example.com/")
    assert " outcome=error reason=fetch_failed" in line and "status=" not in line
    assert len(companions) == 1 and "could not resolve host" in companions[0]


def test_a_fetch_with_no_answer_is_an_error_with_no_status(caplog, public_dns):
    with respx.mock() as mock:
        mock.route(host="example.com").mock(side_effect=httpx.ConnectError("refused"))
        _, line, companions = _run(WebFetchTool(), caplog, url="https://example.com/")
    assert " outcome=error reason=transport" in line and "status=" not in line
    assert len(companions) == 1


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
