"""No credential reaches the journal through a prose line (issue #857).

`kv` scrubs every value it renders, but a prose line interpolates its own arguments
(``_log.warning("…: %s", exc)``), and an exception's text is foreign: a vendor's 401 body can echo
the key it rejected. So the scrub's floor is the handler every harness entrypoint logs through
(`_observability.log_handler`, whose `ScrubbingFormatter` scrubs each rendered record), and these
tests prove it three ways, the way `test_mining.py` proves the mining boundary:

* **the formatter itself** — a sentinel key in a prose argument, a traceback and another library's
  line is redacted, and a clean line renders byte for byte as before;
* **nothing goes around it** — every entrypoint configures logging through `log_handler`, no logger
  stops propagating to it, and every entrypoint's direct stderr print of an error scrubs it;
* **a sentinel wake of every prose site** the issue named: each site's exception path is driven
  with an error whose text holds fabricated keys, and the journal the site wrote holds none of them.
"""

from __future__ import annotations

import ast
import io
import logging
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from basecradle_harness import _assets, _caching, _engine, _mcp
from basecradle_harness._anthropic import AnthropicProvider
from basecradle_harness._basecradle import _configure_logging
from basecradle_harness._context import Compactor, ContextBudget, Limit
from basecradle_harness._engine import Engine
from basecradle_harness._exceptions import (
    EngineError,
    ProviderContextLengthError,
    ProviderError,
)
from basecradle_harness._messages import Message
from basecradle_harness._observability import (
    LOG_FORMAT,
    ScrubbingFormatter,
    install_log_handler,
    kv,
    log_handler,
)
from basecradle_harness._openai import ModelCatalog, OpenAIProvider, list_owner
from basecradle_harness._openrouter import OpenRouterProvider
from basecradle_harness._session import Session
from basecradle_harness._tools import ToolRegistry
from basecradle_harness._xai_sdk import XaiSdkProvider

SRC = Path(__file__).resolve().parent.parent / "src" / "basecradle_harness"

#: A refuseless key (``abl_`` plus 48 alphanumerics), fabricated.
REFUSELESS_KEY = "abl_" + "Kq7Zt2Wm9Rx4Lp6Vn3Hs8Dj5Fb1Gc0Ye7Ua2Ti9Ow4Qe6Nr3"
#: A key with no shape of its own, sent as an ``x-api-key`` header value (issue #861); fabricated.
HEADER_KEY = "wk9f8e7d6c5b4a3f2e1d0c9b8a"
#: The body of an OpenAI-style key, fabricated.
OPENAI_KEY = "sk-proj-Ab3Cd5Ef7Gh9Ij1Kl3Mn5Op7"
#: What a vendor's 401 says when it echoes what it rejected — the realistic carrier.
ERROR_TEXT = (
    f"Error code: 401 - Invalid API key: {REFUSELESS_KEY} "
    f"(sent as x-api-key: {HEADER_KEY}; Authorization: Bearer {OPENAI_KEY})"
)
SECRETS = (REFUSELESS_KEY[4:], HEADER_KEY, OPENAI_KEY[3:])
#: A key with no shape a rule knows, read only where its label is: ``API_KEY=…``; fabricated.
HEADER_VALUE = HEADER_KEY


class Leak(Exception):
    """An exception carrying the sentinel text, raised at every site below."""

    def __init__(self) -> None:
        super().__init__(ERROR_TEXT)


@pytest.fixture
def journal():
    """The text a harness entrypoint's handler writes, read back after the site under test ran."""
    stream = io.StringIO()
    handler = log_handler(stream=stream)
    logger = logging.getLogger("basecradle_harness")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield stream.getvalue
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def _assert_clean(written: str, phrase: str) -> None:
    """The site wrote its line, and the line holds no part of any sentinel key."""
    assert phrase in written, written
    assert "[redacted]" in written, written
    for secret in SECRETS:
        assert secret not in written, written


# --- the formatter --------------------------------------------------------------------------------


def _record(msg: str, *args, exc_info=None, name: str = "basecradle_harness") -> logging.LogRecord:
    return logging.LogRecord(name, logging.WARNING, __file__, 1, msg, args, exc_info)


def test_a_prose_argument_is_scrubbed_once_rendered():
    rendered = ScrubbingFormatter(LOG_FORMAT).format(_record("Could not read it: %s", Leak()))
    assert rendered.startswith("WARNING Could not read it: Error code: 401 - Invalid API key: ")
    assert not any(secret in rendered for secret in SECRETS), rendered


def test_a_traceback_is_scrubbed_with_the_message():
    try:
        raise Leak()
    except Leak:
        record = _record("A wake failed", exc_info=sys.exc_info())
    rendered = ScrubbingFormatter(LOG_FORMAT).format(record)
    assert "Traceback (most recent call last)" in rendered
    assert not any(secret in rendered for secret in SECRETS), rendered


def test_another_librarys_line_is_scrubbed_too():
    """The floor is the handler, so it covers every logger under it, not only the harness's."""
    record = _record("HTTP Request: GET https://h.example.com/?api_key=%s", "Zx9", name="httpx")
    assert (
        ScrubbingFormatter(LOG_FORMAT).format(record)
        == "WARNING HTTP Request: GET https://h.example.com/[redacted]"
    )


@pytest.mark.parametrize(
    "msg",
    [
        "wake start timeline=0190a8b2-7c3d-7e4f-8a9b-0c1d2e3f4a5b delivery=d-1",
        (
            "llm provider=openai purpose=main model=gpt-5.2 duration=3.41s tokens_in=764942 "
            "cached_tokens=238277 outcome=\x1b[32mok\x1b[0m"
        ),
        'unspoken chars=12 text="the task-specification failed"',
    ],
)
def test_a_clean_line_renders_byte_for_byte_as_before(msg):
    record = _record(msg)
    assert ScrubbingFormatter(LOG_FORMAT).format(record) == logging.Formatter(LOG_FORMAT).format(
        record
    )


def test_the_handler_writes_through_the_formatter_to_its_stream():
    stream = io.StringIO()
    handler = log_handler(stream=stream)
    assert isinstance(handler.formatter, ScrubbingFormatter)
    handler.handle(_record("rejected: %s", Leak()))
    assert not any(secret in stream.getvalue() for secret in SECRETS)


# --- a kv line keeps its shape ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"error": "GET https://h.example.com/api/v1/heartbeat/", "outcome": "error"},
        {"error": "no route /heartbeat/", "delivery": "d-1"},
        {"error": "GET https://h.example.com/v1/models?token=", "outcome": "error"},
        {"error": "the request was GET /v1?api_key=", "timeline": "0190a8b2"},
        {"outcome": "error", "error": "GET https://h.example.com/api/v1/heartbeat/"},
    ],
)
def test_a_second_scrub_never_reaches_across_a_kv_values_closing_quote(fields):
    """A value ending in a rule's literal holds no secret; on the whole line, the rule would take the
    closing quote and leave the line unparseable (the review of #857)."""
    line = "wake failed " + kv(**fields)
    assert ScrubbingFormatter("%(message)s").format(_record(line)) == line


# --- nothing goes around it -----------------------------------------------------------------------


def _entrypoints() -> list[tuple[str, str]]:
    scripts = tomllib.loads((SRC.parent.parent / "pyproject.toml").read_text())["project"][
        "scripts"
    ]
    return [tuple(target.split(":")) for target in scripts.values()]


#: The calls that configure logging through the scrubbing handler, any one of which an entrypoint
#: makes before it does anything else that could log.
_CONFIGURES = {"_configure_logging", "install_log_handler", "logging.basicConfig"}


@pytest.mark.parametrize("target", _entrypoints(), ids=lambda target: ":".join(target))
def test_every_console_script_configures_the_scrubbing_handler(target):
    """A command that configures nothing writes a WARNING through `logging.lastResort`, unscrubbed."""
    module, function = target
    path = SRC / f"{module.rsplit('.', 1)[1]}.py"
    (node,) = [
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == function
    ]
    calls = {ast.unparse(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)}
    assert calls & _CONFIGURES, f"{module}:{function} never configures logging"


def _isolated_root(monkeypatch) -> logging.Logger:
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", root.level)
    return root


def test_the_wake_and_cleanup_entrypoints_install_the_scrubbing_handler(monkeypatch):
    root = _isolated_root(monkeypatch)
    _configure_logging()
    (handler,) = root.handlers
    assert isinstance(handler.formatter, ScrubbingFormatter)
    assert handler.formatter._fmt == LOG_FORMAT


def test_the_other_entrypoints_keep_python_s_default_output_scrubbed(monkeypatch):
    """`install_log_handler` stands in for `logging.lastResort`: ``WARNING`` and up, bare message."""
    root = _isolated_root(monkeypatch)
    install_log_handler()
    (handler,) = root.handlers
    assert isinstance(handler.formatter, ScrubbingFormatter)
    assert handler.formatter._fmt == "%(message)s"
    assert root.level == logging.WARNING


def test_an_embedding_applications_logging_wins(monkeypatch):
    root = _isolated_root(monkeypatch)
    own = logging.NullHandler()
    root.handlers.append(own)
    install_log_handler()
    _configure_logging()
    assert root.handlers == [own]


def _calls(name: str):
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == name:
                yield path.name, node


def test_every_basic_config_goes_through_the_scrubbing_handler():
    """A ``basicConfig`` with a ``format=`` and no handler installs a formatter that does not scrub."""
    sites = list(_calls("logging.basicConfig"))
    assert {name for name, _ in sites} == {
        "_basecradle.py",
        "_scrub.py",
        "_palace_check.py",
        "_observability.py",
    }
    for name, call in sites:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert "format" not in keywords and "stream" not in keywords, name
        handlers = keywords["handlers"]
        assert isinstance(handlers, ast.List) and len(handlers.elts) == 1, name
        assert ast.unparse(handlers.elts[0].func) == "log_handler", name


#: Text that configures logging, or reaches around a handler, by any spelling but the one above.
_BYPASSES = (
    ".addHandler(",
    ".setFormatter(",
    ".handlers.append(",
    ".handlers.extend(",
    "dictConfig",
    "fileConfig",
    ".propagate",
    '"propagate"',
    "'propagate'",
    "from logging import",
    "traceback.print_exc(",
    "traceback.print_exception(",
    "captureWarnings",
)


@pytest.mark.parametrize("bypass", _BYPASSES)
def test_no_module_reaches_around_the_handler(bypass):
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "_observability.py" and bypass in (".setFormatter(",):
            continue
        assert bypass not in path.read_text(encoding="utf-8"), (path.name, bypass)


def test_every_harness_logger_propagates_to_the_handler():
    for name, call in _calls("logging.getLogger"):
        if not call.args or name == "_basecradle.py" and ast.unparse(call.args[0]) == "name":
            continue  # the root logger, and `_quiet_transport_logs` naming the HTTP clients
        (arg,) = call.args
        assert isinstance(arg, ast.Constant), name
        assert arg.value == "basecradle_harness" or arg.value.startswith("basecradle_harness."), (
            name,
            arg.value,
        )


def _bound_names(tree: ast.AST) -> set[str]:
    return {h.name for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler) and h.name}


#: Calls whose result is an exception's text: a traceback rendered to a string.
_FOREIGN_CALLS = {"traceback.format_exc", "traceback.format_exception", "traceback.format_tb"}


def _texts_of(node: ast.AST, errors: set[str]) -> tuple[int, int]:
    """How many times `node` renders an exception's text, and how many of those are scrubbed.

    ``type(error).__name__`` names the class, not its text, so it renders nothing foreign."""
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "type":
        return 0, 0
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "scrub":
        rendered, _ = _texts_of(ast.Tuple(elts=node.args, ctx=ast.Load()), errors)
        rendered = max(rendered, 1)
        return rendered, rendered
    if isinstance(node, ast.Call) and ast.unparse(node.func) in _FOREIGN_CALLS:
        return 1, 0
    if isinstance(node, ast.Name):
        return (1, 0) if node.id in errors else (0, 0)
    rendered = scrubbed = 0
    for child in ast.iter_child_nodes(node):
        r, s = _texts_of(child, errors)
        rendered, scrubbed = rendered + r, scrubbed + s
    return rendered, scrubbed


#: The calls that write straight to a stream, past every handler: on a box, stderr is the journal.
_DIRECT_WRITES = {"print", "sys.stderr.write", "sys.stdout.write", "warnings.warn"}


def test_every_direct_write_of_an_error_scrubs_it():
    """An entrypoint's last word goes to stderr directly. Data flow is not followed: an error's text
    assigned to a name first and written later is the reviewer's to catch."""
    found = 0
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        errors = _bound_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in _DIRECT_WRITES:
                rendered, scrubbed = _texts_of(ast.Tuple(elts=node.args, ctx=ast.Load()), errors)
                assert rendered == scrubbed, (path.name, node.lineno, ast.unparse(node))
                found += scrubbed
    # wake (two), cleanup, resolve, claims (its reason and its traceback), log grammar, both palace
    # tools
    assert found >= 9


# --- a sentinel wake of every prose site the issue named ------------------------------------------


_REFUSELESS = "https://api.refuseless.com/v1"
_REFUSELESS_OWNER = list_owner(_REFUSELESS, "sk-rl-0123456789abcdef0123456789abcdef")


def test_the_refuseless_model_list(journal):
    client = SimpleNamespace(models=SimpleNamespace(list=lambda **_: (_ for _ in ()).throw(Leak())))
    assert ModelCatalog().rows(client, _REFUSELESS, _REFUSELESS_OWNER) is None
    _assert_clean(journal(), "Could not read the model list from https://api.refuseless.com/v1")


def test_the_refuseless_model_list_is_scrubbed_for_a_library_callers_handler(caplog):
    """(b): the one site scrubbed at its argument too, so a handler the harness did not install
    still renders no key (the refuseless agent's first wake reads this list)."""
    caplog.set_level(logging.WARNING, logger="basecradle_harness")
    client = SimpleNamespace(models=SimpleNamespace(list=lambda **_: (_ for _ in ()).throw(Leak())))
    ModelCatalog().rows(client, _REFUSELESS, _REFUSELESS_OWNER)
    (record,) = [r for r in caplog.records if "model list" in r.getMessage()]
    assert not any(secret in record.getMessage() for secret in SECRETS), record.getMessage()


def _raise(*_args, **_kwargs):
    raise Leak()


def test_the_openai_context_limit(journal, monkeypatch):
    adapter = OpenAIProvider("gpt-5.2", api_key="sk-test", base_url="https://api.openai.com/v1")
    monkeypatch.setattr(adapter._client.models, "retrieve", _raise)
    assert adapter.context_limit() is None
    _assert_clean(journal(), "Could not read gpt-5.2's context limit from the provider")


def test_the_openrouter_context_limit(journal):
    client = SimpleNamespace(endpoints=SimpleNamespace(list=_raise))
    adapter = OpenRouterProvider("z-ai/glm-5.2", client=client, base_url="https://or.example.com")
    assert adapter.context_limit() is None
    _assert_clean(journal(), "Could not read z-ai/glm-5.2's context limit from OpenRouter")


def test_the_anthropic_models_record(journal, monkeypatch):
    adapter = AnthropicProvider("claude-sonnet-5-5", client=SimpleNamespace())

    def fail():
        raise ProviderError(ERROR_TEXT)

    monkeypatch.setattr(adapter, "_model_info", fail)
    assert adapter._info_or_none() is None
    _assert_clean(journal(), "Could not read the Models API record for claude-sonnet-5-5")


def test_the_xai_context_limit(journal):
    client = SimpleNamespace(models=SimpleNamespace(get_language_model=_raise))
    adapter = XaiSdkProvider("grok-4.3", client=client)
    assert adapter.context_limit() is None
    _assert_clean(journal(), "Could not read grok-4.3's context limit from xAI")


def test_the_vision_capability(journal):
    assert _assets.model_sees_images(SimpleNamespace(supports_vision=_raise)) is True
    _assert_clean(journal(), "Could not read the model's vision capability")


def test_the_video_capability(journal):
    assert _assets.model_sees_video(SimpleNamespace(supports_video=_raise)) is False
    _assert_clean(journal(), "Could not read the model's video capability")


def test_the_context_budgets_adapter_read(journal):
    budget = ContextBudget(SimpleNamespace(context_limit=_raise))
    assert budget._from_adapter() is None
    _assert_clean(journal(), "Could not read the model's context limit from the provider")


def test_the_compaction_summary_call(journal):
    provider = SimpleNamespace(chat=_raise)
    compactor = Compactor(provider, ContextBudget(provider, override=100_000))
    summary = compactor._summarize(
        [Message.user("hello")], limit=Limit(tokens=100_000, source="env"), chars_per_token=4.0
    )
    assert summary is None
    _assert_clean(journal(), "Context compaction failed: the summarization call errored")


def test_the_reserve_summary_call(journal, monkeypatch):
    engine = Engine(SimpleNamespace(), ToolRegistry())
    monkeypatch.setattr(engine, "_chat", _raise)
    with pytest.raises(EngineError):
        engine._reserve_summary([Message.user("hello")])
    _assert_clean(journal(), "Reserve summary call failed after the step budget was spent")


def test_a_failed_progress_persist(journal):
    _engine._progress(_raise)
    _assert_clean(journal(), "Could not persist a turn in progress")


def _session() -> Session:
    return Session(
        "timeline:0190a8b2-7c3d-7e4f-8a9b-0c1d2e3f4a5b", Engine(SimpleNamespace(), ToolRegistry())
    )


def test_the_turn_retried_after_an_overflow(journal, monkeypatch):
    session = _session()
    replies = iter(
        [ProviderContextLengthError(ERROR_TEXT, status_code=400), SimpleNamespace(content="done")]
    )

    def exchange(*_args, **_kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(session, "_exchange", exchange)
    monkeypatch.setattr(session, "_recover_from_overflow", lambda **_: True)
    assert session.send("hello") == "done"
    _assert_clean(journal(), "Retrying the turn once against the compacted transcript")


def test_a_failed_compaction_between_turns(journal):
    session = _session()
    session.compactor = SimpleNamespace(maybe_compact=_raise)
    session._compact_if_needed()
    _assert_clean(journal(), "Context compaction failed; the transcript is unchanged")


def test_the_cache_affinity_bind(journal):
    _caching.bind_conversation(SimpleNamespace(bind_conversation=_raise), "session-1")
    _assert_clean(journal(), "Could not bind the conversation id for cache affinity")


def test_a_rejected_mcp_config(journal, monkeypatch, tmp_path):
    (tmp_path / "mcp").mkdir()
    (tmp_path / "mcp" / "search.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(_mcp, "_parse_config", _raise)
    configs, rejected = _mcp.load_mcp_configs_report(tmp_path)
    assert configs == [] and [stem for stem, _ in rejected] == ["search"]
    _assert_clean(journal(), "Skipping MCP server config search.json")
