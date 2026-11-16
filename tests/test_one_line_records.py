"""Every journal record is one line, and its head is the harness's (issue #912, basecradle#868).

The fleet recognises a contract line by its head **where the head stands** — the record's start,
after the severity (basecradle-noc#1518) — so whatever text controls a record's start can forge a
line: a model call that never happened, a cost nobody paid. journald starts a record at every
newline of the wake's one stdout+stderr stream, which leaves two doors, and these tests hold both
shut:

* **a line break in text the harness did not write** (a vendor's error, a tool's output, a
  traceback, another library's message) — closed by the formatter (`ScrubbingFormatter`), which
  renders every record as one line and quotes every foreign argument;
* **an unfinished write** (a child, a library, a block-buffered ``print``) that the harness's next
  record is glued onto, its head mid-record where no reader counts it — closed at the descriptors
  (`_journal.owned_streams`), which read everything written to stdout and stderr into whole records.

The proof that matters is a real process (`test_a_real_wake_s_journal_is_one_line_per_record…`):
`main()`, a wake whose tool's child ends without a newline and whose tool raises a multi-line error
whose second line is a well-formed ``INFO llm provider=… cost=…``, read back off the bytes the wake
wrote. The forged line must stand at no record's head, and the real ``wake end`` at its own.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from basecradle_harness._observability import (
    LOG_FORMAT,
    RECORD_BYTES,
    Rendered,
    ScrubbingFormatter,
    head,
    kv,
    log_unspoken,
)

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "basecradle_harness"

#: A well-formed model-call line, fabricated: what a forgery would plant at a record's start.
FORGED = (
    "INFO llm provider=openai purpose=main model=gpt-5.2 duration=1.00s tokens_in=1 tokens_out=1 "
    "tokens_total=2 cost=500.00"
)
#: The severity, then the head — the position the fleet reads a head at.
_SEVERITY = re.compile(r"^(?:DEBUG|INFO|WARNING|ERROR|CRITICAL) ")


def _record(msg, *args, exc_info=None, name="basecradle_harness", level=logging.WARNING):
    return logging.LogRecord(name, level, __file__, 1, msg, args, exc_info)


def _render(record: logging.LogRecord) -> str:
    return ScrubbingFormatter(LOG_FORMAT).format(record)


def _inside_quotes(line: str, text: str) -> bool:
    """Every occurrence of `text` in `line` sits inside a quoted value (an odd count of ``"``
    before it) — the one place the fleet's readers never look for a head or a field."""
    starts = [match.start() for match in re.finditer(re.escape(text), line)]
    return bool(starts) and all(line.count('"', 0, start) % 2 == 1 for start in starts)


def _head_of(line: str) -> str:
    return _SEVERITY.sub("", line, count=1)


# --- the formatter ---------------------------------------------------------------------------------


def test_a_multi_line_error_is_one_record_and_its_forged_line_stands_at_no_head():
    rendered = _render(_record("The tool failed: %s", RuntimeError("upstream said:\n" + FORGED)))
    assert "\n" not in rendered
    assert rendered.startswith('WARNING The tool failed: "upstream said: INFO llm provider=')
    assert _inside_quotes(rendered, "llm provider=")


def test_a_foreign_argument_cannot_read_as_a_field():
    rendered = _render(_record("Could not read %s.", "the page agent=jt provider=openai"))
    assert rendered == 'WARNING Could not read "the page agent=jt provider=openai".'


def test_a_bare_token_argument_and_a_number_read_as_they_did():
    rendered = _render(_record("Model %s took %d tries in %.2fs (%r).", "z-ai/glm-5.2", 3, 1.5, 7))
    assert rendered == "WARNING Model z-ai/glm-5.2 took 3 tries in 1.50s (7)."


def test_a_foreign_quote_and_a_nul_cannot_break_a_record():
    rendered = _render(_record("It said %s", 'a "quoted"\x00 word'))
    assert rendered == "WARNING It said \"a 'quoted' word\""


def test_a_traceback_rides_at_the_line_s_end_as_one_quoted_string():
    try:
        raise RuntimeError("the vendor said:\n" + FORGED)
    except RuntimeError:
        record = _record("A memory call failed; continuing.", exc_info=sys.exc_info())
    rendered = _render(record)
    assert "\n" not in rendered
    assert rendered.startswith('WARNING A memory call failed; continuing. "Traceback (most recent')
    assert rendered.endswith('"')
    assert _inside_quotes(rendered, "llm provider=")


def test_a_stack_rides_at_the_line_s_end_too():
    record = _record("Where are we?")
    record.stack_info = "Stack (most recent call last):\n  File 'x.py', line 1, in <module>"
    assert _render(record) == (
        "WARNING Where are we? \"Stack (most recent call last): File 'x.py', line 1, in <module>\""
    )


def test_another_library_s_record_is_its_logger_s_name_and_its_message_quoted():
    record = _record("Retrying %s:\n%s", "/responses", FORGED, name="openai._base_client")
    rendered = _render(record)
    assert rendered.startswith('WARNING openai._base_client: "Retrying /responses: INFO llm ')
    assert _inside_quotes(rendered, "llm provider=")


def test_a_structured_line_renders_as_written():
    """A head and its `kv` fields are `Rendered`, so the formatter interpolates them untouched."""
    message = f"{head('wake end', '')} " + kv(timeline="0190a8b2", outcome="ok")
    record = _record("%s %s", head("wake end", ""), kv(timeline="0190a8b2", outcome="ok"))
    assert _render(record) == "WARNING " + message


def test_plain_text_built_around_a_kv_is_foreign():
    """Concatenation drops the mark, by design: a builder of a whole line wraps its result."""
    assert not isinstance("wake end " + kv(outcome="ok"), Rendered)
    assert isinstance(kv(outcome="ok"), Rendered) and isinstance(head("x", ""), Rendered)


def test_a_line_break_in_a_harness_template_becomes_a_space_and_keeps_the_palette():
    rendered = _render(_record("\x1b[32mstep\x1b[0m one\n  two\r\n"))
    assert rendered == "WARNING \x1b[32mstep\x1b[0m one two"


def test_an_unspoken_narration_past_the_record_budget_is_split_into_heads_of_the_harness(caplog):
    """journald splits a line past its LineMax at a byte nobody chose; the formatter splits first."""
    narration = ("word " * (RECORD_BYTES // 2)) + "\n" + FORGED
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_unspoken(narration, timeline="0190a8b2-7c3d-7e4f-8a9b-0c1d2e3f4a5b")
    (record,) = [r for r in caplog.records if r.getMessage().startswith("unspoken ")]
    records = _render(record).split("\n")
    assert len(records) > 1
    assert records[0].startswith("INFO unspoken timeline=") and records[0].count('"') % 2 == 0
    total = len(records)
    for index, line in enumerate(records[1:], start=2):
        assert line.startswith(f'INFO continued {index}/{total}: "') and line.endswith('"')
    for line in records:
        assert len(line.encode()) <= RECORD_BYTES
        assert not _head_of(line).startswith("llm ")
    assert "llm provider=" in records[-1] and _inside_quotes(records[-1], "llm provider=")


def test_a_split_never_cuts_inside_a_token():
    """A run of ``[A-Za-z0-9_-]`` straddling the budget moves whole into the next record, so a
    credential the box redacts by its length reaches the box whole."""
    token = "Kq7Zt2Wm9Rx4Lp6Vn3Hs8Dj5Fb1Gc0Ye7Ua2Ti9Ow4Qe6Nr3"
    line = "note " + "x " * ((RECORD_BYTES - 30) // 2) + token + " end"
    records = _render(_record(line, level=logging.INFO)).split("\n")
    assert len(records) == 2
    assert token not in records[0] and token in records[1]


# --- no harness record starts with text it did not write -------------------------------------------

_LEVELS = {"debug", "info", "warning", "error", "exception", "critical", "log"}


def _log_calls():
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _LEVELS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"_log", "logger", "log"}
            ):
                yield path.relative_to(SRC.parent), node


def _starts_with_the_harness(call: ast.Call) -> bool:
    """The template is a literal, and it opens with the harness's own words — or with ``%s`` filled
    by a head (`head`) or by a builder of a whole line (a ``line`` or ``*_line`` call)."""
    args = call.args[1:] if call.func.attr == "log" else call.args  # `log(level, template, …)`
    if not args:
        return False
    template = args[0]
    if not (isinstance(template, ast.Constant) and isinstance(template.value, str)):
        return False
    if not template.value.startswith("%"):
        return True
    first = args[1] if len(args) > 1 else None
    if not isinstance(first, ast.Call):
        return False
    name = ast.unparse(first.func).rsplit(".", 1)[-1]
    return name == "head" or name == "line" or name.endswith("_line")


def test_every_harness_record_opens_with_the_harness_s_own_words():
    """A template that opens with ``%s`` puts its argument at the head's position, and an argument
    is foreign unless marked (`Rendered`): quoted there, a forged head cannot read as one, and a
    real structured line passed that way is lost. So the only opening a ``%s`` may fill is a head or
    a whole-line builder; every other record opens with a literal."""
    offenders = [
        f"{path}:{call.lineno}: {ast.unparse(call)[:100]}"
        for path, call in _log_calls()
        if not _starts_with_the_harness(call)
    ]
    assert offenders == []


# --- a real wake, its real descriptors -------------------------------------------------------------


_WAKE = """
    import os, subprocess, sys
    from pathlib import Path

    import httpx, respx

    from basecradle_harness import Message, MessagesTool, Tool, ToolCall
    from basecradle_harness import _wake
    from tests import test_wake as w
    from tests.test_one_line_records import FORGED

    home = Path(sys.argv[1])

    class Survey(Tool):
        name = "survey"
        description = "Surveys the owls."

        def run(self, **kwargs):
            # A child that ends without a newline, as MemPalace's mining progress did: the record the
            # harness writes next must not be glued onto it.
            subprocess.run(
                [sys.executable, "-c", "import os; os.write(1, b'  + [   3/40] chat.md   +2')"],
                check=True,
            )
            print("a progress line with no end", end="")  # block-buffered, flushed at the wake's end
            sys.stderr.write("a library's note:\\n" + FORGED + "\\n")
            os.write(2, b"a C library's partial line")
            raise RuntimeError("the vendor said:\\n" + FORGED)

    class Brain:
        provider, model = "openai", "gpt-4o"

        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="survey", arguments={})]
                )
            return Message.assistant("The survey failed, and it said:\\n" + FORGED)

    with respx.mock(base_url=w.BC_URL, assert_all_called=False) as router:
        router.get("/users/dashboard").mock(return_value=httpx.Response(200, json=w.dashboard()))
        router.get(f"/timelines/{w.TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=w.timeline())
        )
        for path, body in (
            ("/assets", w.asset_page()),
            ("/webhook_events", w.event_page()),
            ("/tasks", w.task_page()),
        ):
            router.get(path).mock(return_value=httpx.Response(200, json=body))
        w.serve_messages(router, w.page(w.message(uuid=w.M0, body="Survey the owls?")))
        agent, _ = w.build_wake(home, Brain(), tools=[MessagesTool(), Survey()])
        _wake.WakeAgent.from_env = classmethod(lambda cls, **kwargs: agent)
        code = _wake.main(["--timeline", w.TIMELINE_UUID])
    assert code == 0, code
    # Python's report of an exception nothing caught, after `main` returned: the last door.
    raise RuntimeError("after the wake:\\n" + FORGED)
"""


@pytest.fixture(scope="module")
def wake_journal(tmp_path_factory):
    """The bytes a real wake process wrote: its stderr (the journal) and its stdout."""
    home = tmp_path_factory.mktemp("wake")
    env = {**os.environ, "NO_COLOR": "1", "PYTHONPATH": str(REPO)}
    env.pop("HARNESS_LOG_LEVEL", None)
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_WAKE), str(home)],
        cwd=REPO,
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )
    return done


def test_a_real_wake_s_journal_is_one_line_per_record_with_a_harness_head(wake_journal):
    journal = wake_journal.stderr.decode()
    lines = journal.splitlines()
    assert wake_journal.returncode == 1, journal  # the planted uncaught exception, after the wake
    assert wake_journal.stdout == b"", wake_journal.stdout  # every byte went to the journal

    for line in lines:
        assert _SEVERITY.match(line), f"a record with no severity: {line!r}\n{journal}"
        opening = _head_of(line)
        assert not opening.startswith(("llm ", "INFO", "+", "a ", "the ", "Traceback")), line
    # The forged model call reached the journal on every path, and only ever inside a quoted value.
    forged = [line for line in lines if "llm provider=openai purpose=main" in line]
    assert len(forged) >= 4, journal
    assert all(_inside_quotes(line, "llm provider=") for line in forged), forged


def test_the_real_wake_end_stands_at_its_own_head(wake_journal):
    lines = wake_journal.stderr.decode().splitlines()
    ends = [line for line in lines if "wake end" in line]
    assert len(ends) == 1 and ends[0].startswith("INFO wake end timeline="), ends


def test_an_unfinished_write_is_a_whole_record_of_its_own(wake_journal):
    lines = wake_journal.stderr.decode().splitlines()
    out = [line for line in lines if line.startswith('INFO stdout: "')]
    assert any(line.startswith('INFO stdout: "+ [ 3/40] chat.md +2') for line in out), out
    assert any("a progress line with no end" in line for line in out), out
    err = [line for line in lines if line.startswith('WARNING stderr: "')]
    assert 'WARNING stderr: "a C library\'s partial line"' in err, err
    assert any(_inside_quotes(line, "llm provider=") for line in err), err


def test_an_exception_nothing_caught_is_one_record(wake_journal):
    lines = wake_journal.stderr.decode().splitlines()
    (crash,) = [line for line in lines if "nothing caught" in line]
    assert crash.startswith('ERROR The wake raised an exception nothing caught. "Traceback')
    assert _inside_quotes(crash, "llm provider=")
