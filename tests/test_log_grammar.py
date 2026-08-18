"""The log-grammar probe (basecradle-noc#509) — the pins that keep a founder's page alive.

Every test here exists because the thing it pins fails **silently**: the agent keeps working, no
log line moves, and the only witness is an alarm that has quietly stopped hearing. The two lines
under proof appear only when a vendor account runs out of money, so nothing in production
exercises them and no behaviour test covers them.
"""

from __future__ import annotations

import re
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest

from basecradle_harness import _log_grammar, _verify
from basecradle_harness._log_grammar import (
    BILLING_BLOCKED,
    COLUMNS,
    EX_TEMPFAIL,
    IDENTIFIER,
    REASON,
    SCRIPT,
    TTL_HOURS,
    Unprovable,
    emit,
    lines,
    main,
    record,
)
from basecradle_harness._observability import LOG_FORMAT
from basecradle_harness._report import (
    PROBE_SOURCE,
    billing_onset_line,
    billing_repeat_line,
)
from basecradle_harness._verify import claims

#: The NOC's `billing_blocked` column, clause by clause, exactly as `observability/ai-box.json`
#: spells it (basecradle-noc#501 re-pointed both to cross the colour gap with `.*`). The column ORs
#: them, but they are pinned separately here on purpose: the extraction guard asks only whether the
#: column extracted *anything*, so one working clause would green a column whose other clause has
#: gone deaf.
ONSET_CLAUSE = r"wake reported_failure.*kind=billing"
REPEAT_CLAUSE = r"wake billing_blocked"

#: The fleet's `source` label column, and the block-list predicate the alarm charts carry. A probe
#: line must satisfy the first and be excluded by the second; a production line must do neither.
SOURCE_LABEL = r"source=([A-Za-z0-9_-]+)"

#: The witness parents a synthetic line must not move — `provider` is the declared parent for the
#: `llm_missing_tokens` and `tool_cost` witnesses, so a probe line carrying it would let the guard
#: call two healthy columns deaf in an hour when no real `llm` line arrived at all.
FORBIDDEN_ON_A_SYNTHETIC = (
    r"provider=([A-Za-z0-9._-]+)",  # parent of two witnesses
    r" llm provider=",  # llm_calls / llm_missing_tokens / llm_cost / endpoint
    r"stage=([a-z_]+)",  # stage label, wake_duration_s, wake_failed
    r"outcome=([a-z]+)",  # outcome label, wake_failed, tool_errors
    r" tool name=.*outcome=",  # tool_calls / tool_errors
    r" unspoken timeline=",  # the unspoken family
    r"cost=([0-9.]+)",  # llm_cost / tool_cost
    r"tokens_(?:in|out)=[0-9]+",  # the token metrics
    r"model=([A-Za-z0-9._/:-]+)",  # model label
    r"event=breaker_tripped|CIRCUIT BREAKER TRIPPED|Wake breaker TRIPPED",  # the router's needle
)


def probe_lines(agent: str = "jt") -> list[str]:
    return lines(BILLING_BLOCKED, agent=agent)


# --- the grammar itself -------------------------------------------------------------------


def test_the_probe_emits_both_clauses_as_separate_lines():
    """One line per clause, and each matches exactly one of them.

    ``billing_blocked`` is two OR'd clauses and the guard's rule is *"did the column extract
    anything at all"* — so a probe emitting only the onset would keep the column green forever
    while the debounce clause rotted. The clauses are different heads, in different branches,
    painted different colours; a rename will plausibly hit one and not both.
    """
    onset, repeat = probe_lines()

    assert re.search(ONSET_CLAUSE, onset)
    assert not re.search(REPEAT_CLAUSE, onset)
    assert re.search(REPEAT_CLAUSE, repeat)
    assert not re.search(ONSET_CLAUSE, repeat)


def test_the_probe_renders_through_the_production_renderers():
    """The synthetic is built by the *same functions* the real failure path calls.

    This is the whole design in one assertion (basecradle-noc#509): two spellings would let the
    probe keep proving a grammar production no longer writes — the alarm dark, the ledger green.
    """
    onset, repeat = probe_lines(agent="nova")

    assert onset == billing_onset_line(reason=REASON, source=PROBE_SOURCE, agent="nova")
    assert repeat == billing_repeat_line(reason=REASON, source=PROBE_SOURCE, agent="nova")


def test_the_production_lines_still_match_both_clauses():
    """The real lines — the ones an out-of-funds outage actually writes — against the live regexes.

    Pinned here as well as on the probe, because the probe proving a grammar the production path
    has drifted away from is precisely the failure this whole instrument is built to prevent.
    """
    onset = billing_onset_line(reason="out_of_funds", provider="xai", timeline="T", delivery="D")
    repeat = billing_repeat_line(reason="out_of_funds", provider="xai", timeline="T", delivery="D")

    assert re.search(ONSET_CLAUSE, onset)
    assert re.search(REPEAT_CLAUSE, repeat)


def test_a_production_line_carries_no_source_stamp():
    """The alarm's predicate is a **block-list** (`!= 'probe'`), so a real failure must carry no
    ``source=`` at all — a stamp leaking onto the production path would filter a genuine
    out-of-funds event out of the founder-named page and nothing would ever say so."""
    for line in (
        billing_onset_line(reason="out_of_funds", provider="xai", timeline="T", delivery="D"),
        billing_repeat_line(reason="out_of_funds", provider="xai", timeline="T", delivery="D"),
    ):
        assert not re.search(SOURCE_LABEL, line), line


def test_every_synthetic_line_carries_the_probe_stamp():
    """And the mirror: **every** line the probe can emit is stamped, unconditionally.

    The stamp is not a parameter — `lines` passes it itself, so there is no "quiet" mode to get
    wrong. An unstamped synthetic is a page to a human's phone.
    """
    for column in COLUMNS:
        for line in lines(column, agent="jt"):
            assert re.search(SOURCE_LABEL, line).group(1) == PROBE_SOURCE, line


def test_a_synthetic_line_moves_no_other_columns_witness_parent():
    """The contamination audit, as a test.

    A monitor that manufactures false positives in the instrument beside it is worse than the gap
    it closes. The rule for anything added here later: **carry no field that is a witness parent
    for another column**, and keep every value a bare token — `kv` quotes a value holding a space,
    but the fleet's label extractors are naive regexes over the whole message, so a
    ``reason="… provider=x …"`` would populate ``provider`` from inside the quotes.
    """
    for line in probe_lines():
        for pattern in FORBIDDEN_ON_A_SYNTHETIC:
            assert not re.search(pattern, line), f"{pattern!r} matched {line!r}"
        assert '"' not in line  # no quoted value, so no extractor can read inside one


def test_a_synthetic_line_populates_exactly_the_three_intended_labels():
    """`billing_blocked` (the proof), `source` (the discriminator) and `agent` (the alarm's series)
    — and, once Vector prefixes the identifier, `level`. Nothing else."""
    for line in probe_lines(agent="jt"):
        shipped = f"[{IDENTIFIER}] " + record(line)  # what Vector's `ai_scrub` transform ships
        assert re.search(r" (CRITICAL|ERROR|WARNING|INFO|DEBUG) ", shipped).group(1) == "INFO"
        assert re.search(r"agent=([A-Za-z0-9._-]+)", shipped).group(1) == "jt"
        assert re.search(SOURCE_LABEL, shipped).group(1) == PROBE_SOURCE


def test_the_shipped_record_wears_the_production_log_envelope():
    """Severity survives only as a text token inside the message (nothing on an agent box sets a
    syslog priority), so a synthetic rendered without the envelope would have no severity at all."""
    assert record("hello") == LOG_FORMAT % {"levelname": "INFO", "message": "hello"}
    assert record("hello", level="ERROR").startswith("ERROR ")


def test_error_lines_is_not_contaminated_because_the_identifier_is_this_repos_own():
    """`error_lines` is identifier-scoped to ``basecradle-router`` and ``basecradle-wake-*`` and
    powers *Server Errors*, one of only two charts the fleet's spec records as deliberately
    carrying no filter. Under this repo's own identifier the probe contributes nothing to it — at
    **any** severity, which is what makes the property structural rather than a promise about
    log levels."""
    assert IDENTIFIER != "basecradle-router"
    assert not IDENTIFIER.startswith("basecradle-wake-")


# --- emission -----------------------------------------------------------------------------


@pytest.fixture
def journal(monkeypatch):
    """A real AF_UNIX datagram socket standing in for journald, so the native protocol is exercised
    rather than mocked. Bound under a short path — `sun_path` is ~104 bytes on darwin."""
    # `/tmp` rather than pytest's `tmp_path`: `sun_path` is ~104 bytes and darwin's per-test temp
    # dirs blow through it.
    directory = Path(tempfile.mkdtemp(dir="/tmp"))
    path = directory / "socket"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(str(path))
    server.settimeout(0.2)
    monkeypatch.setattr(_log_grammar, "_JOURNAL_SOCKET", str(path))
    try:
        yield server
    finally:
        server.close()
        path.unlink(missing_ok=True)
        directory.rmdir()


def _parse_journal_datagram(raw: bytes) -> dict[str, str]:
    """Parse a datagram the way journald does: ``KEY=value`` up to a newline, or ``KEY`` + newline
    + a little-endian 64-bit length + exactly that many raw bytes + a newline."""
    fields: dict[str, str] = {}
    offset = 0
    while offset < len(raw):
        end = raw.index(b"\n", offset)
        head = raw[offset:end]
        if b"=" in head:
            key, _, value = head.partition(b"=")
            fields[key.decode()] = value.decode()
            offset = end + 1
            continue
        size = int.from_bytes(raw[end + 1 : end + 9], "little")
        start = end + 9
        fields[head.decode()] = raw[start : start + size].decode()
        offset = start + size + 1
    return fields


def test_emit_speaks_the_journald_native_protocol(journal):
    """Written straight to journald's own socket rather than through ``systemd-cat``, so the probe
    depends on the journal existing rather than on a binary being on a ``PATH`` that ``env -i``
    may not carry."""
    messages = [record(line) for line in probe_lines()]

    emit(messages)

    for expected in messages:
        fields = _parse_journal_datagram(journal.recv(65536))
        assert fields["SYSLOG_IDENTIFIER"] == IDENTIFIER
        assert fields["PRIORITY"] == "6"  # INFO
        assert fields["MESSAGE"] == expected


def test_a_newline_in_a_value_cannot_forge_a_journal_field(journal):
    """The short ``KEY=value`` form is newline-terminated, so a value carrying a newline would make
    journald read the rest as *more fields* — a message appending its own ``PRIORITY=`` or
    ``SYSLOG_IDENTIFIER=`` and landing wherever it liked. Nothing here can write one today (`kv`
    flattens every value); this pins the encoding that keeps it true after the change that has not
    happened yet."""
    hostile = "INFO wake billing_blocked\nSYSLOG_IDENTIFIER=basecradle-router"

    emit([hostile])

    fields = _parse_journal_datagram(journal.recv(65536))

    # Parsed as journald parses it: the smuggled text stays *inside* the length-framed payload,
    # so exactly three fields arrive and the identifier is the module's own.
    assert fields == {
        "MESSAGE": hostile,
        "PRIORITY": "6",
        "SYSLOG_IDENTIFIER": IDENTIFIER,
    }


def test_emit_is_unprovable_when_there_is_no_journal(monkeypatch, tmp_path):
    """No journald is *"we never got to ask"*, not *"the answer is no"* — the difference between a
    red row that names a broken box and one that names a broken monitor."""
    monkeypatch.setattr(_log_grammar, "_JOURNAL_SOCKET", str(tmp_path / "absent"))

    with pytest.raises(Unprovable):
        emit(["INFO hello"])


# --- the CLI's three answers ----------------------------------------------------------------


def _fake_journalctl(monkeypatch, *, stdout="", returncode=0, error=None):
    def run(argv, **kwargs):
        assert argv[0] == "journalctl"
        assert "--identifier" in argv and IDENTIFIER in argv
        if error is not None:
            raise error
        return subprocess.CompletedProcess(argv, returncode, stdout, "boom")

    monkeypatch.setattr(_log_grammar.subprocess, "run", run)


def test_main_exits_zero_when_the_lines_are_readable_back(journal, monkeypatch, capsys):
    """The claim this upgrades is *"rendered"* → *"in the journal"*: the difference between a write
    call returning and a line actually existing for Vector to ship."""
    written: list[str] = []

    def run(argv, **kwargs):
        while True:
            try:
                written.append(journal.recv(65536).decode())
            except (TimeoutError, OSError):
                break
        return subprocess.CompletedProcess(argv, 0, "\n".join(written), "")

    monkeypatch.setattr(_log_grammar.subprocess, "run", run)
    monkeypatch.setattr(_log_grammar, "agent_slug", lambda *a, **k: "jt")

    assert main([BILLING_BLOCKED]) == 0
    assert "proven: billing_blocked" in capsys.readouterr().out


def test_main_fails_when_the_lines_never_appear(journal, monkeypatch, capsys):
    """journalctl ran and kept not finding them: *we asked, and the answer is no.* A real finding
    about this box — recorded FAIL, never the softer `unprovable`."""
    monkeypatch.setattr(_log_grammar, "_READBACK_TIMEOUT_S", 0.0)
    _fake_journalctl(monkeypatch, stdout="something else entirely")

    assert main([BILLING_BLOCKED]) == 1
    assert "FAILED" in capsys.readouterr().err


def test_main_is_unprovable_when_journalctl_cannot_run(journal, monkeypatch, capsys):
    """A box with no ``journalctl`` established nothing. Landing that as FAIL would say *the
    capability is broken* about a run that never asked — both are red, but the ledger row would
    misdescribe the fleet."""
    _fake_journalctl(monkeypatch, error=FileNotFoundError("journalctl"))

    assert main([BILLING_BLOCKED]) == EX_TEMPFAIL
    assert "UNPROVABLE" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["billing_blocked", "extra"], ["breaker_tripped"]])
def test_a_question_this_build_cannot_answer_is_unprovable_not_failed(argv, capsys):
    """A manifest naming a column this build does not exercise is a question we never got to ask.
    ``breaker_tripped`` is the router's — a sibling claim in the same namespace, deliberately not
    answerable here."""
    assert main(argv) == EX_TEMPFAIL
    assert "UNPROVABLE" in capsys.readouterr().err


# --- the ledger row -------------------------------------------------------------------------


def test_the_claim_row_is_rare_with_a_ttl_and_names_its_own_script(tmp_path):
    """`class: rare` is the contract's teeth here. Every other row this package emits is
    `dependency` — *present after a converge*, re-proven by the converge floor. This one asks
    something the floor structurally cannot: *do the bytes a founder-named alarm matches still
    exist?*, about a line that appears only when an account runs out of money. Silence is its
    normal state, so its proof is a forced exercise on a TTL — and a `rare` claim with no
    `ttl_hours` could never go stale, which would make one success green forever.
    """
    rows = {c["claim"]: c for c in claims(tmp_path)["claims"]}
    row = rows[f"log-grammar:{BILLING_BLOCKED}"]

    assert row["class"] == "rare"
    assert row["ttl_hours"] == TTL_HOURS and TTL_HOURS  # required, and not zero
    assert row["prove"]["kind"] == "probe"
    assert row["prove"]["cmd"].endswith(f"{SCRIPT} {BILLING_BLOCKED}")
    assert row["evidence"] == f"journal:{IDENTIFIER}"


def test_the_claim_row_is_emitted_even_by_a_box_with_no_config_home(tmp_path):
    """A claim that disappears when it stops being true makes the ledger agree with the box
    precisely when the box is wrong. An agent that cannot emit its billing grammar must have a row
    to be red about."""
    rows = {c["claim"] for c in claims(tmp_path / "never-installed")["claims"]}

    assert f"log-grammar:{BILLING_BLOCKED}" in rows


def test_the_probe_command_resolves_beside_the_interpreter(tmp_path, monkeypatch):
    """Absolute, because the wrapper requires the first token to be one — and because a converge
    has neither the agent's ``PATH`` nor its shell. Resolved beside the running interpreter, which
    is where a venv puts its console scripts."""
    interpreter = tmp_path / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    (tmp_path / "bin" / SCRIPT).touch()
    monkeypatch.setattr(_verify.sys, "executable", str(interpreter))

    assert _verify._script(SCRIPT) == str(tmp_path / "bin" / SCRIPT)
    assert _verify._script("basecradle-harness-absent") == "basecradle-harness-absent"


def test_the_probe_command_is_inert_argv_the_wrapper_will_accept(tmp_path):
    """The NOC never hands a ``cmd`` to a shell: it whitespace-splits it to an argv vector, holds
    every token to an allow-list charset and requires the first to be an absolute path. A ``cmd``
    needing a quote, a glob or a substitution is **refused with a named reason** — a probe quietly
    not run is indistinguishable from one that passed."""
    allowed = re.compile(r"^[A-Za-z0-9_@%+=:,./-]+$")

    for row in claims(tmp_path)["claims"]:
        tokens = row["prove"]["cmd"].split()
        assert all(allowed.match(token) for token in tokens), row["claim"]
