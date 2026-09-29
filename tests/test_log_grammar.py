"""The log-grammar probe (basecradle-noc#509) — the pins that keep a needle alarm hearing.

Every test here exists because the thing it pins fails **silently**: the agent keeps working, no
log line moves, and the only witness is an alarm that has quietly stopped hearing. The lines under
proof appear only on a failure path — a vendor account out of money (`billing_blocked`, the
founder's page) or a wake-loop breaker tripping (`breaker_tripped`, issue #592) — so nothing in
production exercises them and no behaviour test covers the bytes an alarm reads.
"""

from __future__ import annotations

import re
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest

from basecradle_harness import _log_grammar, _verify
from basecradle_harness._breaker import breaker_reset_line, breaker_tripped_line
from basecradle_harness._log_grammar import (
    BILLING_BLOCKED,
    BREAKER_TRIPPED,
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
from basecradle_harness._observability import LOG_FORMAT, RED, YELLOW
from basecradle_harness._report import (
    PROBE_SOURCE,
    PROBE_TOKEN,
    billing_onset_line,
    billing_repeat_line,
    probe_prefix,
)
from basecradle_harness._verify import claims
from tests.conftest import plain

#: The NOC's `billing_blocked` column, clause by clause, exactly as `observability/ai-box.json`
#: spells it (basecradle-noc#501 re-pointed both to cross the colour gap with `.*`). The column ORs
#: them, but they are pinned separately here on purpose: the extraction guard asks only whether the
#: column extracted *anything*, so one working clause would green a column whose other clause has
#: gone deaf.
ONSET_CLAUSE = r"wake reported_failure.*kind=billing"
REPEAT_CLAUSE = r"wake billing_blocked"

#: The NOC's `breaker_tripped` column, whole, exactly as `observability/ai-box.json` spells it —
#: three clauses over two layers' runaway backstops, powering *Circuit Breaker Tripped*. The router
#: owns the first two and proves them with its own probe; this package owns the third.
BREAKER_NEEDLE = r"event=breaker_tripped|CIRCUIT BREAKER TRIPPED|Wake breaker TRIPPED"
HARNESS_BREAKER_CLAUSE = r"Wake breaker TRIPPED"

#: Every column's clauses, so a synthetic can be held to landing on its own column and no other.
NEEDLES = {
    BILLING_BLOCKED: (ONSET_CLAUSE, REPEAT_CLAUSE),
    BREAKER_TRIPPED: (BREAKER_NEEDLE,),
}

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
    r"duration=([0-9.]+)s",  # wake_duration_s / llm_duration_s and the per-purpose durations
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


def test_every_probe_line_leads_with_the_probe_token():
    """The token is for the person, where the stamp is for the machine (issue #593).

    On 2026-09-29 the founder read the probe's red ``wake reported_failure`` in a Live Tail as a
    real billing block on @briggs mid-task. ``source=probe`` was on the line — at the end, where the
    eye does not go — so every synthetic line now *starts* with ``PROBE``, envelope and all.
    """
    for column in COLUMNS:
        for line in lines(column, agent="jt"):
            assert line.startswith(f"{PROBE_TOKEN} "), line
            assert record(line).startswith(f"INFO {PROBE_TOKEN} "), line


def test_a_real_line_never_wears_the_probe_token():
    """The mirror, and the one that pages: a genuine out-of-funds line, or a genuine trip, reading
    ``PROBE`` would be a real failure a human waves away."""
    for line in (
        billing_onset_line(reason="out_of_funds", provider="xai", timeline="T", delivery="D"),
        billing_repeat_line(reason="out_of_funds", provider="xai", timeline="T", delivery="D"),
        _real_trip(),
        breaker_reset_line(timeline="T", held=60.0),
    ):
        assert PROBE_TOKEN not in line, line


def test_the_token_and_the_stamp_are_one_switch():
    """Rendered from the same value, so they can never disagree (the capital's amended ruling): a
    line leads with ``PROBE`` exactly when it carries ``source=probe``."""
    for source in (None, "", PROBE_SOURCE, "basecradle", "github"):
        for line in (
            billing_onset_line(reason="r", source=source),
            billing_repeat_line(reason="r", source=source),
        ):
            stamped = re.search(SOURCE_LABEL, line)
            assert line.startswith(f"{PROBE_TOKEN} ") == bool(
                stamped and stamped.group(1) == PROBE_SOURCE
            ), line
    assert probe_prefix(PROBE_SOURCE) == "PROBE " and probe_prefix(None) == ""


def test_the_token_never_touches_the_bytes_under_proof():
    """The token precedes the grammar and nothing else moves: the head is still painted whole, in
    its own colour, and the rest of the line is byte-for-byte what it was before issue #593."""
    onset, repeat = probe_lines(agent="jt")

    assert onset.startswith(f"PROBE {RED}wake reported_failure")
    assert repeat.startswith(f"PROBE {YELLOW}wake billing_blocked")
    assert plain(onset) == (
        "PROBE wake reported_failure kind=billing reason=log_grammar_probe source=probe agent=jt"
    )
    assert plain(repeat) == (
        "PROBE wake billing_blocked reason=log_grammar_probe source=probe agent=jt"
    )


@pytest.mark.parametrize("column", COLUMNS)
def test_a_synthetic_line_moves_no_other_columns_witness_parent(column):
    """The contamination audit, as a test.

    A monitor that manufactures false positives in the instrument beside it is worse than the gap
    it closes. The rule for anything added here later: **carry no field that is a witness parent
    for another column**, and keep every value a bare token — `kv` quotes a value holding a space,
    but the fleet's label extractors are naive regexes over the whole message, so a
    ``reason="… provider=x …"`` would populate ``provider`` from inside the quotes.
    """
    for line in lines(column, agent="jt"):
        for pattern in FORBIDDEN_ON_A_SYNTHETIC:
            assert not re.search(pattern, line), f"{pattern!r} matched {line!r}"
        assert '"' not in line  # no quoted value, so no extractor can read inside one


@pytest.mark.parametrize("column", COLUMNS)
def test_a_synthetic_line_lands_on_its_own_column_and_no_other(column):
    """Each probe proves one column. A line that also matched a *neighbour's* clause would feed an
    alarm it was never meant to reach — and would let that column read ``populating`` off a probe
    that says nothing about its own spelling."""
    for line in lines(column, agent="jt"):
        assert any(re.search(clause, line) for clause in NEEDLES[column]), line
        for other, clauses in NEEDLES.items():
            if other != column:
                assert not any(re.search(clause, line) for clause in clauses), (other, line)


@pytest.mark.parametrize("column", COLUMNS)
def test_a_synthetic_line_populates_exactly_the_three_intended_labels(column):
    """The column (the proof), `source` (the discriminator) and `agent` (the alarm's series) — and,
    once Vector prefixes the identifier, `level`. Nothing else."""
    for line in lines(column, agent="jt"):
        shipped = f"[{IDENTIFIER}] " + record(line)  # what Vector's `ai_scrub` transform ships
        assert re.search(r" (CRITICAL|ERROR|WARNING|INFO|DEBUG) ", shipped).group(1) == "INFO"
        assert re.search(r"agent=([A-Za-z0-9._-]+)", shipped).group(1) == "jt"
        assert re.search(SOURCE_LABEL, shipped).group(1) == PROBE_SOURCE


# --- the breaker clause (issue #592) ----------------------------------------------------------


def _real_trip() -> str:
    return breaker_tripped_line(
        timeline="019e7750-66ee-7f53-829f-13a8a710b6da",
        count=11,
        threshold=10,
        window=60.0,
        cooldown=60.0,
        hold=60.0,
    )


def test_the_breaker_probe_is_the_production_trip_line_with_only_the_probe_fields():
    """Rendered by the function the real trip is rendered by — and carrying nothing else, because
    every production field describes one real trip and a synthetic has none to report. A
    ``timeline=`` or ``count=`` on a probe would be a made-up fact in the fleet's journal."""
    (line,) = lines(BREAKER_TRIPPED, agent="nova")

    assert line == breaker_tripped_line(source=PROBE_SOURCE, agent="nova")
    assert plain(line) == "PROBE Wake breaker TRIPPED source=probe agent=nova"
    assert line.startswith(f"PROBE {YELLOW}Wake breaker TRIPPED")  # the head, painted whole


def test_the_production_trip_line_matches_the_harness_clause_colored_and_plain(monkeypatch):
    """The real trip, against the live needle, both ways it can reach the journal. The head is
    painted as one span (the token-integrity rule), so the consumed literal stays contiguous in
    color; ``NO_COLOR`` must not change the match either."""
    assert re.search(HARNESS_BREAKER_CLAUSE, _real_trip())
    monkeypatch.setenv("NO_COLOR", "1")
    assert re.search(HARNESS_BREAKER_CLAUSE, _real_trip())
    assert _real_trip() == (
        "Wake breaker TRIPPED timeline=019e7750-66ee-7f53-829f-13a8a710b6da count=11 "
        "threshold=10 window=60s cooldown=60s hold=60.00s"
    )


def test_a_real_trip_carries_no_source_stamp_and_feeds_no_other_column():
    """The alert's predicate is a block-list (``!= 'probe'``), so a real trip must carry no
    ``source=`` — a stamp leaking onto the production path would filter every genuine runaway out of
    *Circuit Breaker Tripped*. And its fields must be ones no column extracts: a ``duration=`` here
    would land a breaker hold in the wake-duration series."""
    for line in (_real_trip(), breaker_reset_line(timeline="T", held=60.0)):
        assert not re.search(SOURCE_LABEL, line), line
        for pattern in FORBIDDEN_ON_A_SYNTHETIC:
            assert not re.search(pattern, line), f"{pattern!r} matched {line!r}"


def test_the_reset_line_matches_no_alarm():
    """The reset is good news, and it must never read as a trip: a clause that matched it would
    page on every recovery."""
    line = breaker_reset_line(timeline="T", held=12.5)

    assert plain(line) == "Wake breaker RESET timeline=T held=12.50s"
    assert not any(re.search(c, line) for clauses in NEEDLES.values() for c in clauses)


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


@pytest.mark.parametrize("column", COLUMNS)
def test_main_exits_zero_when_the_lines_are_readable_back(journal, monkeypatch, capsys, column):
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

    assert main([column]) == 0
    clauses = len(lines(column, agent="jt"))
    noun = "clause line" if clauses == 1 else "clause lines"
    assert f"proven: {column} ({clauses} {noun})" in capsys.readouterr().out


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


@pytest.mark.parametrize("argv", [[], ["billing_blocked", "extra"], ["wake_failed"]])
def test_a_question_this_build_cannot_answer_is_unprovable_not_failed(argv, capsys, monkeypatch):
    """A manifest naming a column this build does not exercise is a question we never got to ask.
    ``wake_failed`` is a real fleet column this package feeds but has no probe for. `emit` is
    stubbed, so the refusal is proven to come from the question and not from a box with no
    journald (which would also answer 75, and did — for the wrong reason — before issue #592)."""
    monkeypatch.setattr(_log_grammar, "emit", lambda messages: None)
    assert main(argv) == EX_TEMPFAIL
    assert "UNPROVABLE" in capsys.readouterr().err


# --- the ledger row -------------------------------------------------------------------------


@pytest.mark.parametrize("column", COLUMNS)
def test_the_claim_row_is_rare_with_a_ttl_and_names_its_own_script(tmp_path, column):
    """`class: rare` is the contract's teeth here. Every other row this package emits is
    `dependency` — *present after a converge*, re-proven by the converge floor. This one asks
    something the floor structurally cannot: *do the bytes a founder-named alarm matches still
    exist?*, about a line that appears only when an account runs out of money. Silence is its
    normal state, so its proof is a forced exercise on a TTL — and a `rare` claim with no
    `ttl_hours` could never go stale, which would make one success green forever.
    """
    rows = {c["claim"]: c for c in claims(tmp_path)["claims"]}
    row = rows[f"log-grammar:{column}"]

    assert row["class"] == "rare"
    assert row["ttl_hours"] == TTL_HOURS and TTL_HOURS  # required, and not zero
    assert row["prove"]["kind"] == "probe"
    assert row["prove"]["cmd"].endswith(f"{SCRIPT} {column}")
    assert row["evidence"] == f"journal:{IDENTIFIER}"


def test_the_claim_row_is_emitted_even_by_a_box_with_no_config_home(tmp_path):
    """A claim that disappears when it stops being true makes the ledger agree with the box
    precisely when the box is wrong. An agent that cannot emit its billing grammar must have a row
    to be red about."""
    rows = {c["claim"] for c in claims(tmp_path / "never-installed")["claims"]}

    assert {f"log-grammar:{column}" for column in COLUMNS} <= rows


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
