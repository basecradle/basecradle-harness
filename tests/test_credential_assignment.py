"""No log line the harness writes carries a credential-named assignment (issue #918, basecradle#877).

The NOC's log-store canary pages on a credential-named ``KEY=VALUE`` whatever the value is, and a
tool failure whose ``error=`` quotes ``export SOME_TOKEN=<value>`` reached the line as written:
the scrub knew an ``api_key=`` assignment, a bearer token, four query parameters and an
``x-api-key`` header, and no other assignment. So `_observability.scrub` — and with it `kv`, the
`ScrubbingFormatter` every entrypoint's handler renders through, `last_word`, and the provider
error relayed by `_report.verbatim` — redacts the value of every credential-named assignment.

These tests hold the scrub's definition of a credential's name to the canary's own pattern
(``conftest.NOC_CREDENTIAL_ASSIGNMENT``), use that pattern as the oracle every scrubbed text is
checked against, prove the fleet's own grammar is untouched, and pin the router's marker, so both
writers on a box leave the same bytes where a value stood (basecradle-router#491).
"""

from __future__ import annotations

import io
import logging
import random
import re
import time

import pytest

from basecradle_harness._engine import Engine
from basecradle_harness._observability import (
    _BREAK_CHARS,
    _SCHEME_GLUE,
    CANARY_VALUE_FLOOR,
    CANARY_VALUE_STOPS,
    CREDENTIAL_NAME,
    CREDENTIAL_TAIL,
    CREDENTIAL_WORDS,
    MAX_VALUE,
    REDACTED_VALUE,
    _flatten,
    bounded,
    kv,
    last_word,
    log_handler,
    redact_credentials,
    scrub,
    scrub_line,
)
from basecradle_harness._report import verbatim
from basecradle_harness._tools import Tool, ToolRegistry
from tests.conftest import NOC_CREDENTIAL_ASSIGNMENT
from tests.test_telemetry_contract import DOC, parse_contract

#: A fabricated GitHub App installation token: ``ghs_`` and 36 characters, the shape a bot mints.
FAKE_TOKEN = "ghs_FABRICATEDnovaDigitalToken0000000000"

#: A credential of a shape no prefix rule knows, fabricated: only its name can say what it is.
SHAPELESS = "wk9f8e7d6c5b4a3f2e1d0c9b8a"

#: The command a builder's mint line is, as the fleet's bot-auth skill gives it — the text the
#: canary matched on the router's side (basecradle/basecradle#875).
MINT = (
    'GH_TOKEN="$(~/Documents/claude-workspace/shared/fleet-identity/gh-app-token'
    ' basecradle-nova-ai)" || exit 1; [ -n "$GH_TOKEN" ] || exit 1; export GH_TOKEN'
)

#: A tool's refusal quoting that command, the way a shell tool's error reaches the engine.
INCIDENT = '<tool_use_error>Blocked: sleep 45 followed by: cd … export GH_TOKEN="$(gh-app-token …'


def silent(canary: re.Pattern[str], *lines: str) -> bool:
    return not any(canary.search(line) for line in lines)


# --- one definition: the canary's ---------------------------------------------------------------


def test_the_pieces_spell_the_canarys_rule_byte_for_byte():
    words, stops, tail = "|".join(CREDENTIAL_WORDS), CANARY_VALUE_STOPS, CREDENTIAL_TAIL
    spelled = f"[A-Z0-9_]*({words})({tail})?=[^{stops}]{{{CANARY_VALUE_FLOOR},}}"
    assert spelled == NOC_CREDENTIAL_ASSIGNMENT


def test_the_marker_is_the_routers_byte_for_byte():
    assert REDACTED_VALUE == "'<a redacted value>'"


def _canary_name_ends(text: str) -> set[int]:
    """Every ``=`` the canary's own name pattern ends at (overlapping starts included)."""
    name = re.compile(NOC_CREDENTIAL_ASSIGNMENT.split("=")[0] + "=")
    return {m.end() for start in range(len(text)) if (m := name.match(text, start))}


def _linear_name_ends(text: str) -> set[int]:
    return {m.end() for m in re.finditer(CREDENTIAL_NAME, text)}


@pytest.mark.parametrize(
    "text",
    [
        "GH_TOKEN=",
        "export GH_TOKEN=x",
        "NOC_FLEET_GH_TOKEN=x",
        "AKEY=x KEY=x TOKEN=x _TOKEN=x _9TOKEN=x 9ATOKEN=x xGH_TOKEN=x",
        "KEYKEY=x PASSPHRASE=x A_PASSPHRASE=x APASSWORD=x A_SECRET_KEY=x",
        "gh_token=x Gh_TOKEN=x GH_TOKEN_FILE=x GH_TOKENS=x A_KEY==x",
        "A_TOKEN=B_TOKEN=C_TOKEN=x",
        # The tail (issue #927): `_BASE` after the word, and no other.
        "SECRET_KEY_BASE=x KEY_BASE=x TOKEN_BASE=x _BASE=x BASE=x KEY_BASE_BASE=x",
        "KEY_BASEX=x KEY_BASES=x KEYBASE=x KEY__BASE=x KEY_base=x A_KEY_BASE_KEY=x",
        "$PASSWORD=x -PASSWORD=x apiKEY=x fooBAR_TOKEN=x 9_KEY=x",
    ],
)
def test_the_linear_spelling_names_exactly_the_canarys_credentials(text):
    assert _linear_name_ends(text) == _canary_name_ends(text)


def test_the_linear_spelling_agrees_with_the_canary_on_random_text():
    rng = random.Random(918)
    alphabet = ["A", "K", "E", "Y", "T", "O", "N", "S", "B", "_", "9", "a", "=", " "]
    alphabet += [*CREDENTIAL_WORDS, CREDENTIAL_TAIL, "BASE"]
    for _ in range(4000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 14)))
        assert _linear_name_ends(text) == _canary_name_ends(text), text


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        # A bare word is a name (issue #927): the canary no longer needs a capital before it.
        ("PASSWORD=hunter2hunter2", "PASSWORD='<a redacted value>'"),
        ("TOKEN=abcdefghij next", "TOKEN='<a redacted value>' next"),
        ("KEY=abcdefghij", "KEY='<a redacted value>'"),
        ("_TOKEN=abcdefghij", "_TOKEN='<a redacted value>'"),
        ("PASSPHRASE=abcdefghij", "PASSPHRASE='<a redacted value>'"),
        ("export $PASSWORD=abcdefghij", "export $PASSWORD='<a redacted value>'"),
        # A literal rule's label is read by the same definition, so a bare query word keeps it.
        ("GET /v1?TOKEN=abcdefghijk&x=1 HTTP", "GET /v1?TOKEN='<a redacted value>' HTTP"),
        ("GET /v1?KEY=abcdefghijk HTTP", "GET /v1?KEY='<a redacted value>' HTTP"),
        # `_BASE` is the one tail: the Rails master secret's name.
        ("SECRET_KEY_BASE=0123456789abcdef", "SECRET_KEY_BASE='<a redacted value>'"),
        ("KEY_BASE=abcdefghij", "KEY_BASE='<a redacted value>'"),
    ],
)
def test_a_bare_word_and_the_base_tail_are_credentials_names(text, scrubbed, canary):
    assert scrub(text) == scrubbed
    assert scrub(scrubbed) == scrubbed
    assert silent(canary, scrubbed)


@pytest.mark.parametrize(
    "text",
    [
        # No other tail, a plural, a lowercase name: what the canary does not page on is kept.
        "GH_TOKEN_FILE=/run/secrets/gh-token",
        "GH_TOKENS=abcdefghij",
        "API_KEY_ID=abcdefghij",
        "KEY_BASES=abcdefghij",
        "SECRET_KEY_BASE_DUMMY=abcdefghij",
        "gh_token=abcdefghij",
    ],
)
def test_any_other_tail_is_no_credentials_name_because_the_canarys_is_not(text, canary):
    assert scrub(text) == text
    assert silent(canary, text)


def test_two_hundred_thousand_capitals_are_scrubbed_in_linear_time():
    """The canary's own pattern is quadratic on a run of capitals in a backtracking engine, and
    Python's `re` is one. Measured at 12-16 ms each on the laptop (issue #918); quadratic, each would
    take minutes. The bound is far from both."""
    started = time.monotonic()
    for text in (
        "A" * 200_000,
        "A" * 200_000 + "=x",
        "KEY" * 66_667 + "=x",
        "_" * 200_000 + "TOKEN=x",
        "KEY_BASE" * 25_000 + "=x",
        "KEY_" * 50_000 + "BASE=x",
        "9" * 200_000 + "KEY=x",
        "A_TOKEN=" * 25_000,
        'A_TOKEN="' * 22_223,
        'API_KEY="' * 22_223,
        "?ACCESS_TOKEN=" * 14_286,
        "OPENAI_API_KEY=x-api-key: " * 7_693,
    ):
        scrub(text)
    assert time.monotonic() - started < 2.0


def test_nesting_no_interpreter_stack_could_hold_is_read_whole():
    assert scrub("A_TOKEN=" + "$(" * 100_000) == "A_TOKEN=" + REDACTED_VALUE


# --- the value, whatever it is and however it is quoted -----------------------------------------


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        (f"GH_TOKEN={FAKE_TOKEN} gh api", "GH_TOKEN='<a redacted value>' gh api"),
        ("A_TOKEN=short tail", "A_TOKEN='<a redacted value>' tail"),
        ('A_TOKEN="a b c" tail', "A_TOKEN='<a redacted value>' tail"),
        ("A_TOKEN='a b c' tail", "A_TOKEN='<a redacted value>' tail"),
        ('A_TOKEN="a \\" b" tail', "A_TOKEN='<a redacted value>' tail"),
        ("A_TOKEN=$'a\\'b c' tail", "A_TOKEN='<a redacted value>' tail"),
        ("A_TOKEN=$(gh-app-token nova) tail", "A_TOKEN='<a redacted value>' tail"),
        ('A_TOKEN="$(gh-app-token "a b" $(x y))" tail', "A_TOKEN='<a redacted value>' tail"),
        ("A_TOKEN=`gh-app-token nova` tail", "A_TOKEN='<a redacted value>' tail"),
        ("A_TOKEN=${SECRET:-a b} tail", "A_TOKEN='<a redacted value>' tail"),
        ("A_TOKEN=a\\ b tail", "A_TOKEN='<a redacted value>' tail"),
        ("A_KEY=(sk1 sk2secretsecret) x", "A_KEY='<a redacted value>' x"),
        ("A_TOKEN=\tsecretsecret", "A_TOKEN='<a redacted value>'"),
        ('A_TOKEN="never closed and the rest', "A_TOKEN='<a redacted value>'"),
        (
            "GH_TOKEN=don't do that.\n\nThe rest of the reply.",
            "GH_TOKEN='<a redacted value>'\n\nThe rest of the reply.",
        ),
        ('A_TOKEN="spans\na line" tail', "A_TOKEN='<a redacted value>'\na line\" tail"),
        (
            "x=1 A_KEY=1 y=2 B_SECRET=2",
            "x=1 A_KEY='<a redacted value>' y=2 B_SECRET='<a redacted value>'",
        ),
        (
            MINT,
            (
                "GH_TOKEN='<a redacted value>' || exit 1; [ -n \"$GH_TOKEN\" ] || exit 1;"
                " export GH_TOKEN"
            ),
        ),
        # The canary reads a value as far as its own run goes, so the value goes that far too.
        (
            "GET /v1?ACCESS_TOKEN=abcdefghijk&x=1 HTTP",
            "GET /v1?ACCESS_TOKEN='<a redacted value>' HTTP",
        ),
    ],
)
def test_the_name_stays_and_the_value_goes_whole(text, scrubbed, canary):
    assert scrub(text) == scrubbed
    assert silent(canary, scrub(text))


@pytest.mark.parametrize(
    "text", ["GH_TOKEN=", "GH_TOKEN= next", "GH_TOKEN=\nnext", "no=credential"]
)
def test_an_empty_value_has_nothing_to_hide_and_is_left_alone(text):
    assert scrub(text) == text


@pytest.mark.parametrize("brk", list(_BREAK_CHARS))
def test_the_canarys_run_ends_at_every_record_break(brk):
    assert scrub(f"A_TOKEN=\t{brk}next") == f"A_TOKEN='<a redacted value>'{brk}next"


def test_the_marker_is_no_value_to_the_canary_whatever_follows_it(canary):
    # The scrub's own `[redacted]` would not do: ten characters the canary reads as a value.
    assert canary.search("GH_TOKEN=[redacted]")
    for after in ("", "x" * 20, "\tx" * 10, ",", "}", '"'):
        assert silent(canary, "GH_TOKEN=" + REDACTED_VALUE + after)


# --- the assignment pass beside the scrub's other rules -----------------------------------------


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        # The `api_key=` rule once took `API_KEY=` with the value; the whole name stays now.
        (f"OPENAI_API_KEY=sk-proj-{SHAPELESS} next", "OPENAI_API_KEY='<a redacted value>' next"),
        (f'config API_KEY="{SHAPELESS}" set', "config API_KEY='<a redacted value>' set"),
        # A value that is a bearer header takes its token, every `Bearer` with it.
        (
            "AUTH_TOKEN=Bearer eyJhbGciOiJIUzI1NiJ9.e30.sig sent",
            "AUTH_TOKEN='<a redacted value>' sent",
        ),
        # The bearer rule takes the second `Bearer` with its token, and the assignment's value,
        # the first `Bearer`, takes the bearer rule's `[redacted]` after it as its token (#926).
        (
            "AUTH_TOKEN=Bearer Bearer eyJhbGciOiJIUzI1NiJ9 sent",
            "AUTH_TOKEN='<a redacted value>' sent",
        ),
        ("AUTH_TOKEN=KBearer eyJhbGciOiJIUzI1NiJ9 sent", "AUTH_TOKEN='<a redacted value>' sent"),
        ('AUTH_TOKEN="Bearer eyJhbGciOiJIUzI1NiJ9" sent', "AUTH_TOKEN='<a redacted value>' sent"),
        # A short word after a credential's bearer value is its token, whatever its length, as the
        # router reads it (basecradle-router#497): the value is a credential's, so it errs toward
        # redaction. A short word after a bare `Bearer` is still prose to the bearer rule.
        ("AUTH_TOKEN=Bearer missing here", "AUTH_TOKEN='<a redacted value>' here"),
        ("auth: Bearer missing here", "auth: Bearer missing here"),
        # A heartbeat URL runs first, and its name still stands with the router's marker.
        (
            "HB_TOKEN=https://uptime.example.com/api/v1/heartbeat/Ab3Cd5Ef7Gh9Ij1K ok",
            "HB_TOKEN='<a redacted value>' ok",
        ),
        # Every other rule still redacts what stands beside the assignment.
        (
            f"GH_TOKEN={FAKE_TOKEN} and {FAKE_TOKEN} and api_key={SHAPELESS}",
            "GH_TOKEN='<a redacted value>' and [redacted] and [redacted]",
        ),
    ],
)
def test_the_other_rules_never_read_the_marker_and_never_lose_a_token(text, scrubbed, canary):
    assert scrub(text) == scrubbed
    assert SHAPELESS not in scrub(text)
    assert silent(canary, scrub(text))


# --- a bearer value goes with its token, called alone (issue #926) ------------------------------

#: A fabricated JWT: the token a bearer header carries, which no shape rule matches.
FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.e30.fabricatedSig"


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        # The issue's reproductions, which shipped the token when the function was called alone.
        (f"AUTH_TOKEN=Bearer {FAKE_JWT} next=1", "AUTH_TOKEN='<a redacted value>' next=1"),
        (f"X_API_KEY=KBearer {FAKE_JWT} next=1", "X_API_KEY='<a redacted value>' next=1"),
        (f"A_TOKEN=Bearer Bearer {FAKE_JWT} next=1", "A_TOKEN='<a redacted value>' next=1"),
        (f"DB_PASSWORD=Bearer {FAKE_JWT} next=1", "DB_PASSWORD='<a redacted value>' next=1"),
        # Any case alone, any run of the two forms in any order, and a tab between them.
        (f"A_TOKEN=bearer {FAKE_JWT} next=1", "A_TOKEN='<a redacted value>' next=1"),
        (f"A_TOKEN=BEARER\t{FAKE_JWT} next=1", "A_TOKEN='<a redacted value>' next=1"),
        (f"A_TOKEN=bearer\tBEARER {FAKE_JWT} next=1", "A_TOKEN='<a redacted value>' next=1"),
        (f"A_TOKEN=Bearer KBearer {FAKE_JWT} next=1", "A_TOKEN='<a redacted value>' next=1"),
        (f"A_TOKEN=KBearer Bearer {FAKE_JWT} next=1", "A_TOKEN='<a redacted value>' next=1"),
        (f"A_TOKEN=x.Bearer Bearer {FAKE_JWT} next=1", "A_TOKEN='<a redacted value>' next=1"),
        # A short word after the scheme word is the token too, as the router reads it.
        ("X_TOKEN=Bearer abc next=1", "X_TOKEN='<a redacted value>' next=1"),
        # The token is a shell word like any value: quoted, it goes whole, after a tab too.
        ('A_TOKEN=Bearer "a b" next=1', "A_TOKEN='<a redacted value>' next=1"),
        ('A_TOKEN=Bearer\t"a b" next=1', "A_TOKEN='<a redacted value>' next=1"),
        ("A_TOKEN=Bearer\t'a b' next=1", "A_TOKEN='<a redacted value>' next=1"),
        # Glued only as HTTP spells it: a value merely ending in the letters is no header, nor is
        # a scheme word inside a quoted piece.
        ("A_TOKEN=abc-bearer next=1", "A_TOKEN='<a redacted value>' next=1"),
        ("A_TOKEN='a Bearer' next=1", "A_TOKEN='<a redacted value>' next=1"),
        ("A_TOKEN=a\\ Bearer next=1", "A_TOKEN='<a redacted value>' next=1"),
        # A scheme word with no word after it on its line is the value.
        ("A_TOKEN=Bearer", "A_TOKEN='<a redacted value>'"),
        ("A_TOKEN=Bearer ", "A_TOKEN='<a redacted value>' "),
        ("A_TOKEN=Bearer \nnext=1", "A_TOKEN='<a redacted value>' \nnext=1"),
        # A marker after a scheme word is the token, already redacted: one marker stands.
        ("A_TOKEN=Bearer '<a redacted value>' next=1", "A_TOKEN='<a redacted value>' next=1"),
    ],
)
def test_a_bearer_scheme_word_goes_with_its_token_called_alone(text, redacted, canary):
    assert redact_credentials(text) == redacted
    assert redact_credentials(redacted) == redacted
    assert FAKE_JWT not in redact_credentials(text)
    assert silent(canary, redacted)


@pytest.mark.parametrize(
    "text",
    [
        # `main`'s `scrub` shipped each token: the bearer rule wants eight token characters and no
        # quote, and the assignment pass took the scheme word alone.
        f'AUTH_TOKEN=Bearer "{FAKE_JWT}" next=1',
        f"AUTH_TOKEN=Bearer '{FAKE_JWT}' next=1",
        "AUTH_TOKEN=Bearer s3cr3t next=1",
        "AUTH_TOKEN=Bearer abc,defghijklmnop next=1",
        "AUTH_TOKEN=Bearer $(cat /run/tok) next=1",
    ],
)
def test_scrub_ships_no_token_after_a_credentials_scheme_word(text):
    assert scrub(text) == text.split("=", 1)[0] + "='<a redacted value>' next=1"


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        # The review of #926: the value's shell word runs to the tab, past a second name's `=`,
        # and the next search starts past it, so that name's value was read by nothing. Without a
        # scheme word this already shipped on `main`; through one, `main` had redacted it.
        (
            "AUTH_TOKEN=Bearer x,DB_PASSWORD=\thunter2hunter2 next=1",
            "AUTH_TOKEN='<a redacted value>' next=1",
        ),
        (
            "AUTH_TOKEN=x,DB_PASSWORD=\thunter2hunter2 next=1",
            "AUTH_TOKEN='<a redacted value>' next=1",
        ),
        ("A_TOKEN=Bearer }DB_PASSWORD=\thunter2hunter2 x", "A_TOKEN='<a redacted value>' x"),
        # A quote the second value opens runs to the end, as the shell reads it.
        ('A_TOKEN=Bearer x"DB_PASSWORD="\thunter2hunter2 x', "A_TOKEN='<a redacted value>'"),
        # A name the value does not reach past its `=` is its own, and keeps its name.
        ("A_TOKEN=x B_TOKEN=y", "A_TOKEN='<a redacted value>' B_TOKEN='<a redacted value>'"),
    ],
)
def test_a_name_inside_a_value_takes_its_value_with_it(text, redacted, canary):
    for redact in (redact_credentials, scrub):
        assert redact(text) == redacted
        assert "hunter2" not in redact(text)
    assert silent(canary, redacted)


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        # The review of #927: a name inside a name inside a value is read whole too, so a bare word
        # that became a name (`PASSWORD=`) never pushes a real one down to the canary's run alone.
        (
            "PASSWORD=x,AUTH_TOKEN=\tDB_TOKEN=Bearer s3cr3t next=1",
            "PASSWORD='<a redacted value>' next=1",
        ),
        (
            'PASSWORD=x,A_TOKEN=\tB_TOKEN=Bearer "eyJ secret" next=1',
            "PASSWORD='<a redacted value>' next=1",
        ),
        ("A_KEY=X_TOKEN=\tY_TOKEN=Bearer s3cr3t next=1", "A_KEY='<a redacted value>' next=1"),
        ("TOKEN=PASSWORD=\tAKEY=' hunter2hunter2", "TOKEN='<a redacted value>'"),
    ],
)
def test_a_name_at_any_depth_takes_its_value_with_it(text, redacted, canary):
    for redact in (redact_credentials, scrub):
        assert redact(text) == redacted
    assert silent(canary, redacted)


def test_a_value_holding_many_names_is_read_in_linear_time():
    started = time.monotonic()
    for unit in ("A_TOKEN=", "A_TOKEN=\t", "x,A_TOKEN=\t", "A_TOKEN=Bearer x,"):
        redact_credentials("A_TOKEN=" + unit * 40_000)
    assert time.monotonic() - started < 5.0


def test_the_scheme_glue_is_the_routers_byte_for_byte():
    assert _SCHEME_GLUE == "A-Za-z0-9._~+/%:=-"


def test_no_run_of_scheme_words_lets_a_token_through_called_alone(canary):
    rng = random.Random(926)
    words = ["Bearer", "bearer", "BEARER", "KBearer", "x.Bearer", "a=Bearer"]
    for _ in range(3000):
        blanks = [" ", "\t", "  "]
        run = "".join(rng.choice(words) + rng.choice(blanks) for _ in range(rng.randint(1, 5)))
        head, tail = rng.choice(["", "x ", "curl -H "]), rng.choice(["", " next=1"])
        token = rng.choice([FAKE_JWT, f'"{FAKE_JWT} x"', f"'{FAKE_JWT} x'", f"x,{FAKE_JWT}"])
        text = f"{head}A_TOKEN={run}{token}{tail}"
        once = redact_credentials(text)
        assert FAKE_JWT not in once, repr(text)
        assert redact_credentials(once) == once, repr(text)
        assert silent(canary, once)
        # A cut anywhere in the raw text keeps the assignment whole or drops it: it never keeps
        # the assignment's scheme words as if they were its whole value, with the token unseen.
        cap = rng.randint(len(head) + 9, len(text) + 2)
        cut = bounded(text, cap)
        assert cut == text or not re.search(r"(?i)bearer\s*…\Z", cut), (repr(text), cap)


def test_a_bound_never_keeps_a_scheme_word_whose_token_the_cut_hides():
    # The cut falls in the blanks after a scheme word: the token past it is unseen, so the
    # assignment goes whole rather than standing as `Bearer` alone and its `…`.
    for pad in range(MAX_VALUE - 40, MAX_VALUE + 5):
        text = "x" * pad + f" A_TOKEN=Bearer   {FAKE_JWT}"
        cut = bounded(text, MAX_VALUE)
        assert "Bearer" not in cut and FAKE_JWT not in cut, (pad, cut[-40:])


def test_a_long_run_of_scheme_words_is_read_in_linear_time():
    started = time.monotonic()
    redact_credentials("A_TOKEN=" + "Bearer " * 50_000 + "x")
    redact_credentials("A_TOKEN=" + "Bearer\t" * 50_000)
    redact_credentials("A_TOKEN=" + "xbearer" * 50_000)
    redact_credentials("A_TOKEN=" + "xBearer" * 50_000 + " x")
    bounded("A_TOKEN=" + "Bearer " * 50_000 + "x", MAX_VALUE)
    scrub("A_TOKEN=" + "Bearer " * 50_000 + "x")
    assert time.monotonic() - started < 5.0


def test_scrubbing_twice_is_scrubbing_once():
    for text in (
        MINT,
        INCIDENT,
        f"OPENAI_API_KEY={SHAPELESS}",
        "AUTH_TOKEN=Bearer eyJhbGciOiJIUzI1NiJ9",
    ):
        once = scrub(text)
        assert scrub(once) == once, text


def test_hostile_text_never_leaves_a_line_the_canary_reads(canary):
    rng = random.Random(877)
    pieces = [
        "GH_TOKEN=", "A_KEY=", "X_SECRET=", "P_PASSWORD=", "PASSPHRASE=", "OPENAI_API_KEY=",
        FAKE_TOKEN, "abcdefghij", "Bearer ", "bearer", "?token=", "&", "api_key=", "x-api-key: ",
        "https://h.example.com/heartbeat/", '"', "'", "$(", ")", "`", "${", "}", "\\", " ", "\t",
        "\n", "\r", ",", "=", "$'", REDACTED_VALUE, "[redacted]", "\x00", "…",
    ]  # fmt: skip
    for _ in range(4000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 14)))
        line = "WARNING tool " + kv(name="shell", error=text)
        # A raw text is read record by record, as the canary reads the store: no row spans a break.
        records = re.split("\r\n|[" + _BREAK_CHARS + "]", scrub(text))
        assert silent(canary, *records, line, scrub_line(line), last_word(text)), repr(text)


def test_the_marker_reads_back_as_itself_on_every_flattened_text():
    # Every harness record is flattened before it is scrubbed (see `scrub`'s docstring). The pieces
    # hold no other rule's label, since a literal rule can read its own `[redacted]` as a value.
    rng = random.Random(918)
    pieces = [
        "GH_TOKEN=", "A_KEY=", "X_SECRET=", "OPENAI_API_KEY=", FAKE_TOKEN, "abcdefghij", '"', "'",
        "$(", ")", "`", "${", "}", "\\", " ", "\t", "\n", ",", "=", "$'", REDACTED_VALUE, "…",
    ]  # fmt: skip
    for _ in range(4000):
        flat = _flatten("".join(rng.choice(pieces) for _ in range(rng.randint(1, 14))))
        assert scrub(scrub(flat)) == scrub(flat), repr(flat)


# --- the cut treats an assignment as one token ---------------------------------------------------


@pytest.mark.parametrize(
    "tail",
    [
        MINT,
        "A_TOKEN=abc and more words",
        "A_TOKEN= then words",
    ],
)
def test_a_bound_cuts_an_assignment_whole_or_not_at_all(tail, canary):
    # `ScrubbingFormatter` scrubs every line again: a cut marker, or a `…` glued to a marker or an
    # `=`, would be read as a value and rebuilt as a whole marker, past the bound and with no `…`.
    for pad in range(MAX_VALUE - 90, MAX_VALUE + 20):
        cut = bounded(scrub("x" * pad + " " + tail), MAX_VALUE)
        assert len(cut) <= MAX_VALUE
        assert scrub(cut) == cut, (pad, cut[-40:])
        assert silent(canary, cut)


@pytest.mark.parametrize(
    ("text", "cut"),
    [
        # The review of #918: an assignment the cut never reached once went with it, and so did
        # the assignments before it, down to the whole field.
        ("GH_TOKEN=abcdefghij " + "x" * 300, "GH_TOKEN='<a redacted value>' …"),
        (
            "env: FOO_KEY=aaaaaaaaaa BAR_TOKEN=bbbbbbbbbb " + "x" * 300,
            "env: FOO_KEY='<a redacted value>' BAR_TOKEN='<a redacted value>' …",
        ),
    ],
)
def test_a_bound_keeps_every_assignment_it_does_not_reach(text, cut):
    assert bounded(scrub(text), MAX_VALUE) == cut
    assert scrub(cut) == cut  # the blank after the marker keeps the `…` off it


def test_a_bound_over_many_assignments_is_linear():
    for size in (20_000, 200_000):
        text = scrub("A_KEY=x " * (size // 8)) + " " + "y" * 50
        started = time.monotonic()
        cut = bounded(text, len(text) - 10)
        assert time.monotonic() - started < 1.0, size
        assert cut.endswith("' …")


# --- the fleet's own grammar is untouched --------------------------------------------------------


def _contract_fields() -> set[str]:
    contract = parse_contract(DOC.read_text(encoding="utf-8"))
    return {
        field.rstrip("*") + ("x" if field.endswith("*") else "")
        for fields in contract.heads.values()
        for field in fields
    }


def test_no_field_of_the_contract_is_a_credentials_name():
    # The grammar is lowercase and the canary's names are capitals, so no field the fleet counts
    # (`tokens_in=`, `cached_tokens=`, `cache_write_tokens=`, `cost=`, …) can ever be redacted.
    fields = _contract_fields()
    assert {"tokens_in", "cached_tokens", "cache_write_tokens", "cost"} <= fields
    for field in sorted(fields):
        assert not re.search(CREDENTIAL_NAME, f"{field}="), field
        assert not re.search(CREDENTIAL_NAME, f"{field.upper()}=") or field.upper() != field


def test_every_field_of_the_contract_renders_and_scrubs_as_written():
    line = "INFO llm " + kv(**{field: "1234567890" for field in sorted(_contract_fields())})
    assert scrub(line) == line
    assert scrub_line(line) == line


def test_every_example_line_of_the_contract_is_left_as_written():
    examples, inside = [], False
    for text in DOC.read_text(encoding="utf-8").splitlines():
        if text.startswith("```"):
            inside = not inside
        elif inside and re.match(r"(?:INFO|WARNING|ERROR) ", text):
            examples.append(text)
    assert any("tokens_in=" in line and "cached_tokens=" in line for line in examples)
    for line in examples:
        assert scrub(line) == line, line
        assert scrub_line(line) == line, line


# --- every place a line carries foreign text -----------------------------------------------------


class _Refused(Tool):
    name = "shell"
    description = "A fabricated shell whose every run is refused with the command it was given."

    def run(self, **kwargs):
        raise RuntimeError(f"{INCIDENT}\n{MINT}\nexit 1 GH_TOKEN={FAKE_TOKEN}")


@pytest.fixture
def journal():
    """What a harness entrypoint's handler writes, read back after the code under test ran."""
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


def test_a_tool_failures_error_line_carries_the_name_and_never_the_value(journal, canary):
    registry = ToolRegistry()
    registry.register(_Refused())
    Engine(provider=None, tools=registry)._run_tool("shell", {})
    written = journal()
    (line,) = [line for line in written.splitlines() if "error=" in line]
    assert "export GH_TOKEN='<a redacted value>'" in line
    assert FAKE_TOKEN not in written and "gh-app-token" not in written
    assert silent(canary, *written.splitlines())


def test_a_prose_line_and_a_traceback_are_redacted_by_the_formatter(journal, canary):
    log = logging.getLogger("basecradle_harness.test")
    log.warning("The palace refused: %s", f"OPENAI_API_KEY={SHAPELESS} rejected")
    try:
        raise ValueError(f"env -i NOC_FLEET_GH_TOKEN={SHAPELESS} run")
    except ValueError:
        log.exception("A step failed.")
    logging.getLogger("httpx").warning("PASSWORD_KEY=%s", SHAPELESS)
    written = journal()
    assert SHAPELESS not in written
    assert "OPENAI_API_KEY='<a redacted value>'" in written
    assert "NOC_FLEET_GH_TOKEN='<a redacted value>'" in written
    assert silent(canary, *written.splitlines())


def test_a_commands_last_word_and_a_relayed_vendor_error_are_redacted(canary):
    error = RuntimeError(f"upstream said: SERVICE_SECRET={SHAPELESS}")
    for text in (last_word(error), verbatim(error)):
        assert SHAPELESS not in text
        assert "SERVICE_SECRET='<a redacted value>'" in text
        assert silent(canary, text)
