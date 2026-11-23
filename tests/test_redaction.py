"""One definition of a secret, applied to every text the harness writes (issue #999).

The definition is one module (`_redaction`) built from one public file
(``data/secret-shapes.json``). These tests pin four things: the rules are the file (read here from
the raw JSON, never back through the module under test) and the file is strict about its own shape
and documented where its readers look; the shapes catch exactly what the log scrub caught before
the move; the agent's own configured secrets are blanked by value, with the 20-character floor for
a bare match; and a redacted file is rewritten only when it is a text file with something to lose.
The places the definition is applied (a log line, an exchange filed into memory, a fact the memory
tool stores, a capture a tool saved) are tested where each writer is: `test_memory_provider.py`,
`test_mcp.py`, and the log tests.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from pathlib import Path

import pytest

from basecradle_harness import _redaction
from basecradle_harness._memory import SqliteMemoryStore
from basecradle_harness._observability import kv, log_handler
from basecradle_harness._redaction import (
    CONFIGURED_VALUE_FLOOR,
    REDACTED,
    REDACTED_VALUE,
    configured_values,
    names_a_credential,
    redact_file,
    scrub,
    scrub_counted,
)
from basecradle_harness._secret import Secret, known_secrets

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "src" / "basecradle_harness" / "data" / "secret-shapes.json"
DOC = ROOT / "docs" / "secret-shapes.md"
RAW = json.loads(FILE.read_text(encoding="utf-8"))

#: A fabricated secret of no listed shape: no vendor prefix, no label, 32 characters. Only the value
#: half can catch it.
SHAPELESS = "Kq7vZt2Wm9LpXc4RbN8sYdHjQe5UaF3G"

#: A fabricated configured value under the bare-match floor.
SHORT = "Zr8Lq2Vx9Tm"


# --- the file says what it is, and the rules are the file -----------------------------------------


def test_the_file_names_its_format_and_version():
    assert RAW["format"] == _redaction.FORMAT == "basecradle-harness-secret-shapes"
    assert RAW["format_version"] == _redaction.FORMAT_VERSION == 1


def test_the_package_reads_the_file_at_its_documented_path():
    assert _redaction.RESOURCE == ("data", "secret-shapes.json")
    assert _redaction.load() == RAW


def test_the_vendor_key_rules_are_the_file_in_its_order():
    assert _redaction._PREFIXED == tuple(
        (rule["prefix"], rule["run"], rule["minimum_run"]) for rule in RAW["vendor_keys"]["rules"]
    )


def test_the_assignment_rule_and_the_markers_are_the_file():
    rule = RAW["credential_assignment"]
    assert _redaction.CREDENTIAL_WORDS == tuple(rule["words"])
    assert _redaction.CREDENTIAL_TAIL == rule["tail"]
    assert _redaction.CANARY_VALUE_STOPS == rule["value_stops"]
    assert _redaction.CANARY_VALUE_FLOOR == rule["value_minimum_length"]
    assert REDACTED == RAW["markers"]["redacted"] == "[redacted]"
    assert REDACTED_VALUE == RAW["markers"]["redacted_value"] == "'<a redacted value>'"
    assert CONFIGURED_VALUE_FLOOR == RAW["configured_values"]["minimum_length"] == 20


def test_the_bearer_and_heartbeat_pieces_are_the_file():
    assert _redaction._SCHEME_LETTERS == RAW["bearer"]["scheme"]
    assert _redaction._BEARER_TOKEN.pattern == (
        f"{RAW['bearer']['token']}{{{RAW['bearer']['minimum_token']},}}"
    )
    assert _redaction._HEARTBEAT == RAW["heartbeat_url"]["path"]


#: Each label family's rule as the log scrub wrote it before the definition moved into the file
#: (0.207.0's `_observability`), kept here as the oracle the file's literal names are held to.
_BEFORE = {
    "query_parameters": r"api[-_]?key|access[-_]?token|token|key",
    "api_key_assignment": r"api[-_]?key",
    "headers": r"x[-_]api[-_]?key",
    "json_members": (
        r"api[-_]?key|access[-_]?token|refresh[-_]?token|client[-_]?secret|password|secret"
        r"|authorization"
    ),
}
_LANGUAGE_SIZE = {"query_parameters": 8, "api_key_assignment": 3, "headers": 6, "json_members": 15}


@pytest.mark.parametrize("section", sorted(_BEFORE))
def test_each_label_family_names_exactly_what_the_scrub_matched_before(section):
    """Each old rule is a finite language, and the file lists every word of it and nothing else."""
    names = RAW[section]["names"]
    assert len(set(names)) == len(names) == _LANGUAGE_SIZE[section]
    assert all(re.fullmatch(_BEFORE[section], name) for name in names)


def test_no_pattern_is_spelled_twice_in_the_package():
    """The file is the one place a shape lives: no module of the package carries a vendor prefix,
    a label family or a marker as a literal of its own."""
    package = ROOT / "src" / "basecradle_harness"
    spelled = [
        "sk-or-|sk-|xai-",
        "github_pat_",
        "access[-_]?token",
        "x[-_]api",
        "/heartbeat/",
        "'<a redacted value>'",
        '"[redacted]"',
        "PASSPHRASE",
    ]
    found = [
        (path.name, literal)
        for path in package.rglob("*.py")
        for literal in spelled
        if literal in _code(path.read_text(encoding="utf-8"))
    ]
    assert found == []


def _code(source: str) -> str:
    """`source` without its comments and docstrings, which may quote a shape."""
    lines = []
    in_doc = False
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        quotes = stripped.count('"""')
        if in_doc or (quotes and stripped.startswith(('"""', 'r"""', 'f"""'))):
            if quotes % 2:
                in_doc = not in_doc
            continue
        lines.append(line)
    return "\n".join(lines)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{not json", "cannot be read"),
        ("[]", "not a JSON object"),
        (json.dumps({**RAW, "format_version": 2}), r"is \('basecradle-harness-secret-shapes', 2\)"),
    ],
)
def test_a_file_this_version_cannot_read_is_a_loud_error(tmp_path, monkeypatch, content, match):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "secret-shapes.json").write_text(content, encoding="utf-8")
    monkeypatch.setattr(_redaction.resources, "files", lambda _package: tmp_path)
    _redaction.load.cache_clear()
    try:
        with pytest.raises(_redaction.ShapesFileError, match=match):
            _redaction.load()
    finally:
        monkeypatch.undo()
        _redaction.load.cache_clear()
    assert _redaction.load() == RAW


def _field_names(node):
    names = set()
    if isinstance(node, dict):
        for key, value in node.items():
            names.add(key)
            names |= _field_names(value)
    elif isinstance(node, list):
        for item in node:
            names |= _field_names(item)
    return names


def test_every_field_the_file_carries_is_documented():
    doc = DOC.read_text(encoding="utf-8")
    missing = sorted(name for name in _field_names(RAW) if f"`{name}`" not in doc)
    assert not missing, f"docs/secret-shapes.md does not document {missing}"


def test_the_doc_and_the_readme_name_the_path_and_the_format():
    doc = DOC.read_text(encoding="utf-8")
    assert "basecradle_harness/data/secret-shapes.json" in doc
    assert f"`{_redaction.FORMAT}`" in doc
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "basecradle_harness/data/secret-shapes.json" in readme
    assert "docs/secret-shapes.md" in readme


# --- by value: the agent's own configured secrets ---------------------------------------------


@pytest.mark.parametrize(
    ("name", "names_one"),
    [
        ("AI_API_KEY", True),
        ("BASECRADLE_TOKEN", True),
        ("NTFY_DM_TOKEN", True),
        ("SECRET_KEY_BASE", True),
        ("PASSWORD", True),
        ("X-API-Key", True),  # an MCP header name, read upper-cased with `-` as `_`
        ("GH_TOKEN_FILE", False),  # the one tail is `_BASE`; nothing else follows the word
        ("HARNESS_DESCRIBER_CREDENTIALS_FILE", False),  # a path, not a secret
        ("AI_MODEL", False),
        ("PLAYWRIGHT_BROWSERS_PATH", False),
    ],
)
def test_a_name_names_a_credential_by_the_assignment_rules_words(name, names_one):
    assert names_a_credential(name) is names_one


def test_a_configured_secret_is_blanked_wherever_it_stands(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", SHAPELESS)
    assert SHAPELESS in configured_values()
    text, count = scrub_counted(f"the vendor said {SHAPELESS} was refused; retry {SHAPELESS}.")
    assert text == f"the vendor said {REDACTED} was refused; retry {REDACTED}."
    assert count == 2


def test_the_value_half_is_off_for_a_name_that_names_no_credential(monkeypatch):
    monkeypatch.setenv("HARNESS_SOME_SETTING", SHAPELESS)
    assert SHAPELESS not in configured_values()
    assert scrub(f"value {SHAPELESS}") == f"value {SHAPELESS}"


def test_a_short_value_is_not_blanked_bare_but_is_after_its_name(monkeypatch):
    """`@briggs`'s floor: under 20 characters a bare match would take ordinary words out of a
    memory; where a credential name introduces the value, the assignment rule still takes it."""
    monkeypatch.setenv("NTFY_DM_TOKEN", SHORT)
    assert SHORT not in configured_values()
    assert scrub(f"saw {SHORT} here") == f"saw {SHORT} here"
    assert scrub(f"NTFY_DM_TOKEN={SHORT}") == f"NTFY_DM_TOKEN={REDACTED_VALUE}"


def test_a_held_secret_is_known_while_it_lives_and_forgotten_after():
    held = Secret(SHAPELESS)
    assert SHAPELESS in known_secrets()
    assert scrub(f"key {SHAPELESS}") == f"key {REDACTED}"
    del held
    gc.collect()
    assert SHAPELESS not in known_secrets()
    assert scrub(f"key {SHAPELESS}") == f"key {SHAPELESS}"


def test_a_value_wrapped_only_for_privacy_is_not_redacted():
    """An MCP server's ``env`` entry whose name names no credential is wrapped to keep it out of a
    representation, and is not a secret to blank from every text (a browsers path, a log level)."""
    path = "/home/nova/.cache/ms-playwright/chromium"
    held = Secret(path, credential=False)
    assert path not in known_secrets()
    assert scrub(f"launching {path}") == f"launching {path}"
    del held


def test_the_held_list_never_keeps_a_value_in_plain_text():
    """It holds `Secret` objects weakly, so a walk of the module finds no credential."""
    held = Secret(SHAPELESS)
    from basecradle_harness import _secret

    assert SHAPELESS not in repr(vars(_secret))
    del held


def test_a_configured_value_is_redacted_in_a_log_line(monkeypatch):
    """The log half of the DoD: a configured value goes from a `kv` value and from a prose line."""
    monkeypatch.setenv("AI_API_KEY", SHAPELESS)
    assert SHAPELESS not in kv(error=f"401: key {SHAPELESS} is invalid")
    record = logging.LogRecord(
        "basecradle_harness", logging.WARNING, __file__, 1, "refused: %s", (SHAPELESS,), None
    )
    line = log_handler().format(record)
    assert SHAPELESS not in line
    assert REDACTED in line


def test_without_a_configured_value_the_shapes_alone_decide(monkeypatch):
    """With no value configured the scrub is the shape scrub, so every shape test of the log scrub
    (`test_log_value_cut.py`, `test_credential_assignment.py`) still pins exactly what it caught."""
    for name in [name for name in os.environ if names_a_credential(name)]:
        monkeypatch.delenv(name)
    gc.collect()
    assert [value for value in configured_values() if value in SHAPELESS] == []
    assert scrub(f"value {SHAPELESS}") == f"value {SHAPELESS}"


# --- a text file ------------------------------------------------------------------------------


def test_a_text_file_is_redacted_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_API_KEY", SHAPELESS)
    capture = tmp_path / "console.log"
    capture.write_text(f"[log] token {SHAPELESS}\nAuthorization: Bearer eyJhbGciOiJIUzI1NiJ9.e30\n")
    inode = capture.stat().st_ino
    os.chmod(capture, 0o640)

    assert redact_file(capture) == 2
    assert capture.read_text() == f"[log] token {REDACTED}\nAuthorization: {REDACTED}\n"
    assert capture.stat().st_ino == inode  # rewritten in place: nothing staged beside it
    assert capture.stat().st_mode & 0o777 == 0o640
    assert sorted(path.name for path in tmp_path.iterdir()) == ["console.log"]


def test_a_clean_text_file_is_never_rewritten(tmp_path):
    capture = tmp_path / "page.yml"
    capture.write_text("- heading: Example Domain\n")
    before = capture.stat()
    assert redact_file(capture) == 0
    assert capture.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize("name", ["shot.png", "clip.webm", "trace.zip", "page.pdf", "noext"])
def test_a_file_that_is_not_text_is_never_opened(tmp_path, monkeypatch, name):
    monkeypatch.setenv("AI_API_KEY", SHAPELESS)
    data = b"\x89PNG\r\n\x1a\n" + SHAPELESS.encode()
    path = tmp_path / name
    path.write_bytes(data)
    opened = []
    real_open = os.open
    monkeypatch.setattr(os, "open", lambda *a, **k: opened.append(a[0]) or real_open(*a, **k))
    assert redact_file(path) == "it is not a text file"
    assert opened == []
    assert path.read_bytes() == data


def test_a_text_file_that_is_not_utf8_is_left_byte_for_byte(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_API_KEY", SHAPELESS)
    data = b"\xff\xfe" + SHAPELESS.encode()
    path = tmp_path / "dump.txt"
    path.write_bytes(data)
    assert redact_file(path) == "it is not UTF-8 text"
    assert path.read_bytes() == data


def test_a_symlinked_capture_is_never_followed(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_API_KEY", SHAPELESS)
    target = tmp_path / "elsewhere.txt"
    target.write_text(SHAPELESS)
    link = tmp_path / "capture.txt"
    link.symlink_to(target)
    assert isinstance(redact_file(link), str)
    assert target.read_text() == SHAPELESS


# --- a fact the memory tool stores ---------------------------------------------------------------


def test_a_fact_is_stored_redacted_and_still_found_by_its_key(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("AI_API_KEY", SHAPELESS)
    store = SqliteMemoryStore(tmp_path / "memory.db")
    try:
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            reply = store.write("vendor", f"the vendor key is {SHAPELESS}")
        assert "1 secret(s) in it were redacted" in reply
        assert store.read("vendor") == f"the vendor key is {REDACTED}"
        assert "secrets redacted kind=fact count=1" in caplog.text
        assert SHAPELESS not in (tmp_path / "memory.db").read_bytes().decode("latin-1")
    finally:
        store.close()


def test_a_fact_that_cannot_be_redacted_is_never_written(tmp_path, monkeypatch):
    store = SqliteMemoryStore(tmp_path / "memory.db")
    try:
        store.write("kept", "before")

        def broken(text):
            raise RuntimeError("the redaction failed")

        monkeypatch.setattr("basecradle_harness._memory.scrub_counted", broken)
        with pytest.raises(RuntimeError):
            store.write("vendor", f"the key is {SHAPELESS}")
        assert store.list() == "kept"
    finally:
        store.close()


# --- the review's cases (issue #999) -------------------------------------------------------------


def test_a_configured_value_is_matched_as_a_vendor_echoes_it(monkeypatch):
    """An ``agent.env`` value can carry a trailing newline; the vendor echoes the bare key."""
    monkeypatch.setenv("AI_API_KEY", SHAPELESS + "\n")
    assert scrub(f"key {SHAPELESS} refused") == f"key {REDACTED} refused"


def test_a_marker_already_in_the_text_is_not_counted_again():
    for text in ("OPENAI_API_KEY='<a redacted value>'", '{"password":"[redacted]"}'):
        assert scrub_counted(text) == (text, 0)


def test_a_hard_linked_file_is_never_rewritten(tmp_path, monkeypatch):
    """Rewritten in place, a file with another name would change that one too (an `agent.env`)."""
    monkeypatch.setenv("AI_API_KEY", SHAPELESS)
    store = tmp_path / "agent.env"
    store.write_text(f"AI_API_KEY={SHAPELESS}\n")
    capture = tmp_path / "capture.txt"
    os.link(store, capture)
    assert redact_file(capture) == "it has another hard link"
    assert store.read_text() == f"AI_API_KEY={SHAPELESS}\n"


def test_a_key_holding_a_secret_is_refused_and_an_old_row_stays_reachable(tmp_path):
    """Redacted alike, two keys would overwrite each other, and a row stored before the rule
    could not be read by its own name: so a write is refused, and read and delete take the key
    as written."""
    store = SqliteMemoryStore(tmp_path / "memory.db")
    try:
        conn = store._connect()
        conn.execute(
            "INSERT INTO memories (key, value, created_at, updated_at) VALUES (?, ?, '', '')",
            ("deploy sk-abcdefgh12345", "old"),
        )
        conn.commit()
        reply = store.write("gh sk-aaaaaaaa11", "A")
        assert reply.startswith("Error: that key holds a secret")
        assert store.list() == "deploy sk-abcdefgh12345"
        assert store.read("deploy sk-abcdefgh12345") == "old"
        assert store.delete("deploy sk-abcdefgh12345").startswith("Forgot")
    finally:
        store.close()


# --- the session transcript, where a turn is recorded (issue #999, the capital's amendment) ------

from basecradle_harness import Harness, Message, Session, Tool, ToolCall  # noqa: E402
from basecradle_harness._engine import _TRUNCATED_NOTE  # noqa: E402
from basecradle_harness._session import INTERRUPTED, STOPPED, STOPPED_NOTE, _payload  # noqa: E402

#: A fabricated account password, typed into a login form through a browser tool: no listed shape,
#: so only the value half catches it, as it was configured for the agent.
_PASSWORD = "Tq6Wn2Ld8Xr4Vb9Kz3Mf7Hc1Jp5Gy0Sa"


class _Scripted:
    """A provider that replays prepared replies and keeps what it was shown on each call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.shown = []

    def chat(self, messages, tools=None):
        self.shown.append(
            "\n".join(m.content or "" for m in messages)
            + "".join(json.dumps(c.arguments) for m in messages for c in (m.tool_calls or []))
        )
        return self.replies.pop(0)


def _login_turn(path, typed):
    class Fill(Tool):
        name = "browser_fill"
        description = "type into a field"
        parameters = {"type": "object", "properties": {}}

        def run(self, **kwargs):
            typed.append((kwargs["value"], path.read_text()))  # the pre-dispatch save, on disk
            return f'Filled "#password" with "{kwargs["value"]}". Ran: page.fill("#password", ...)'

    provider = _Scripted(
        Message.assistant(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="browser_fill",
                    arguments={"selector": "#password", "value": _PASSWORD},
                )
            ]
        ),
        Message.assistant(content=f"Logged in; the form took {_PASSWORD}."),
    )
    session = Session("timeline:x", Harness(provider, tools=[Fill()]).engine, path=path)
    session.send("log in to the vendor")
    return provider, session


def test_a_value_typed_through_a_tool_call_is_absent_after_the_turn_and_the_next(
    tmp_path, monkeypatch
):
    """The way it failed: one browser login put the password in the session log three times (the
    call's argument, its result, the echoed call), and the log came back after every hand redaction
    because it is rewritten from the transcript. The value is blanked where the turn is recorded,
    so it is absent after the turn ends **and** after the next turn opens."""
    monkeypatch.setenv("VENDOR_PASSWORD", _PASSWORD)
    path = tmp_path / "timeline.json"
    typed = []
    _login_turn(path, typed)

    ((value, on_disk_mid_turn),) = typed
    assert value == _PASSWORD  # the tool was handed the value itself
    assert _PASSWORD not in on_disk_mid_turn  # and the save before dispatch never held it
    after_turn = path.read_text()
    assert _PASSWORD not in after_turn
    assert after_turn.count(REDACTED) == 3  # the argument, the result, the echo

    provider = _Scripted(Message.assistant(content="Done."))
    Session("timeline:x", Harness(provider).engine, path=path).send("anything else?")
    assert _PASSWORD not in path.read_text()
    assert _PASSWORD not in provider.shown[0]  # the next turn opened on the redacted form


def test_the_model_keeps_the_value_for_the_turn_that_saw_it(tmp_path, monkeypatch):
    """Redacted on the way out, never in the live list: the turn in flight still reads the value."""
    monkeypatch.setenv("VENDOR_PASSWORD", _PASSWORD)
    provider, session = _login_turn(tmp_path / "timeline.json", [])
    assert _PASSWORD in provider.shown[1]
    assert any(_PASSWORD in (m.content or "") for m in session.history)


def _create_turn(result):
    call = ToolCall(
        id="c1", name="messages", arguments={"action": "create", "body": f"key {_PASSWORD}"}
    )
    return [
        Message.user("post it"),
        Message.assistant(tool_calls=[call]),
        Message(role="tool", content=result, tool_call_id="c1"),
    ]


def test_an_interrupted_create_keeps_its_arguments_whole_until_its_re_issue_lands(monkeypatch):
    """The recovery re-issues an interrupted platform create from the bytes on disk, under the key
    its dead wake minted; those arguments are recovery evidence and are kept as the cap keeps them.
    Once a real result answers the call, the next save redacts them."""
    monkeypatch.setenv("VENDOR_PASSWORD", _PASSWORD)
    interrupted = json.loads(_payload(_create_turn(INTERRUPTED)))
    assert interrupted[1]["tool_calls"][0]["arguments"]["body"] == f"key {_PASSWORD}"
    assert interrupted[2]["content"] == INTERRUPTED
    answered = json.loads(_payload(_create_turn("message 0198e3f1-0000-7000-8000-000000000001")))
    assert answered[1]["tool_calls"][0]["arguments"]["body"] == f"key {REDACTED}"


def test_every_sentinel_a_later_wake_reads_survives_the_redaction_byte_for_byte():
    for sentinel in (INTERRUPTED, STOPPED, STOPPED_NOTE, _TRUNCATED_NOTE):
        assert scrub(sentinel) == sentinel


def test_a_secret_configured_after_a_save_is_redacted_on_the_next(tmp_path, monkeypatch):
    """The per-session memo rescans nothing that has not changed, but a change in the configured
    secrets is a change: what was saved before is redacted on the next save."""
    path = tmp_path / "timeline.json"
    provider = _Scripted(Message.assistant(content=f"noted {_PASSWORD}"))
    session = Session("timeline:x", Harness(provider).engine, path=path)
    session.send("remember this")
    assert _PASSWORD in path.read_text()  # not yet one of the agent's secrets
    monkeypatch.setenv("VENDOR_PASSWORD", _PASSWORD)
    session._save()
    assert _PASSWORD not in path.read_text()


def test_the_redaction_memo_shows_nothing_it_holds(tmp_path, monkeypatch):
    import pickle

    monkeypatch.setenv("VENDOR_PASSWORD", _PASSWORD)
    _, session = _login_turn(tmp_path / "timeline.json", [])
    memo = session._redactions
    assert _PASSWORD not in repr(memo)
    assert not hasattr(memo, "__dict__")
    with pytest.raises(TypeError):
        pickle.dumps(memo)


# --- the second review's cases (issue #999) -----------------------------------------------------

import random  # noqa: E402

from basecradle_harness._messages import ServerToolRecord, ServerToolUse  # noqa: E402
from basecradle_harness._server_tools import bound  # noqa: E402

_DENSE = [
    "GH_TOKEN=ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8",
    "OPENAI_API_KEY=sk-proj-Z9y8X7w6V5u4T3s2",
    'export DB_PASSWORD="hunter2-correct-horse"',
    "url=https://h.example/v1?api_key=K9xQ2mV7pL4wZ8nR3tY6",
    "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.e30.sig",
    '{"password":"Pw-4Kd9-Lm2x","x-api-key":"Ab9Cd8Ef7"}',
    "plain words about the weather and nothing else",
    "https://uptime.example.com/api/v1/heartbeat/H8k2Lm9Qx",
]


def _dense(rng, lines):
    return "\n".join(rng.choice(_DENSE) for _ in range(lines))


def _resaved(messages):
    """The transcript a later wake writes back: loaded from the file, then saved again."""
    once = _payload(messages)
    return once, _payload([Message.from_dict(d) for d in json.loads(once)])


@pytest.mark.parametrize("fan_out", [1, 3, 10, 30])
def test_a_redacted_capped_transcript_is_a_fixed_point(fan_out):
    """Re-saving a transcript the harness wrote changes nothing: no cut leaves a value open for the
    next save's scrub to complete, so no excerpt grows past its share and no marker names a size
    that is no longer true (the cap's fixed point, held with the redaction in front of it)."""
    rng = random.Random(fan_out)
    for _ in range(40):
        calls = [
            ToolCall(
                id=f"c{n}", name="shell", arguments={"command": _dense(rng, rng.randint(1, 400))}
            )
            for n in range(fan_out)
        ]
        messages = [Message.user("run them"), Message.assistant(tool_calls=calls)]
        messages += [
            Message(role="tool", content=_dense(rng, rng.randint(1, 600)), tool_call_id=f"c{n}")
            for n in range(fan_out)
        ]
        once, twice = _resaved(messages)
        assert once == twice


def test_a_secret_under_a_credential_named_field_is_redacted_at_any_depth(tmp_path):
    call = ToolCall(
        id="c1",
        name="http",
        arguments={
            "password": "Hunter2-correct-horse",
            "headers": {"x-api-key": "abc123def456ghi", "Accept": "application/json"},
            "env": {"GH_TOKEN": "plainvalue123456", "LOG_LEVEL": "debug"},
        },
    )
    messages = [
        Message.user("call it"),
        Message.assistant(tool_calls=[call]),
        Message(role="tool", content="200 OK", tool_call_id="c1"),
    ]
    (stored,) = [c for m in json.loads(_payload(messages)) for c in m.get("tool_calls", [])]
    assert stored["arguments"] == {
        "password": REDACTED,
        "headers": {"x-api-key": REDACTED, "Accept": "application/json"},
        "env": {"GH_TOKEN": REDACTED, "LOG_LEVEL": "debug"},
    }


def test_a_server_tools_input_is_redacted_where_it_is_bounded():
    """Bounded once, at arrival, and never again, so it is redacted there: later, a marker longer
    than its value would carry the record past its cap."""
    found = bound(
        ServerToolRecord(
            uses=[
                ServerToolUse(
                    "web_fetch", "https://api.test/v1?api_key=abcdef1234567890 then " + "x " * 400
                )
            ]
        )
    )
    (use,) = found.uses
    assert "abcdef1234567890" not in use.input and REDACTED in use.input
    assert bound(found) == found  # still a fixed point


def test_a_tool_result_that_is_not_text_never_fails_the_load(tmp_path):
    path = tmp_path / "timeline.json"
    path.write_text(
        json.dumps(
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "tool_calls": [{"id": "c1", "name": "x", "arguments": {}}]},
                {"role": "tool", "tool_call_id": "c1", "content": ["a", "list"]},
            ]
        )
    )
    session = Session("timeline:x", Harness(_Scripted()).engine, path=path)
    assert session.history[2].content == ["a", "list"]


def test_a_legacy_turn_is_still_found_by_its_redacted_rendering(monkeypatch):
    """A pre-#297 turn carries no uuids and is matched by its rendered text, which the transcript
    now holds redacted."""
    from basecradle_harness._wake import _carries

    monkeypatch.setenv("VENDOR_PASSWORD", _PASSWORD)
    rendered = f"[2026-06-04T00:00:00.000Z] john: the password is {_PASSWORD}"
    assert _carries(f"earlier\n{scrub(rendered)}", rendered)


@pytest.mark.parametrize("share", [4096, 1200, 400])
def test_a_redacted_excerpt_degrades_and_never_collapses(share):
    """The fixed point is held by where the cut falls (`stable_cut`: never inside a marker, never
    leaving a value open), not by the floor the measured check falls back to: an excerpt of dense,
    already-redacted credential text keeps its head and tail at every share with room for one."""
    from basecradle_harness._elision import gone
    from basecradle_harness._session import _elide

    rng = random.Random(share)
    for _ in range(200):
        text = scrub(_dense(rng, rng.randint(80, 600)))
        excerpt = _elide(text, share)
        assert excerpt != gone(len(text))
        assert scrub(excerpt) == excerpt


def test_a_cut_never_splits_a_marker():
    text = "x" * 3000 + "GH_TOKEN='<a redacted value>' and [redacted] " + "y" * 3000
    for cut in range(3000, 3050):
        head, tail = _redaction.stable_cut(text, cut, cut)
        for marker in (REDACTED_VALUE, REDACTED):
            for at in (head, tail):
                start = text.find(marker)
                assert not start < at < start + len(marker)
