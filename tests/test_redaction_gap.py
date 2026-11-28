"""A gap between a separator and its value never keeps the value (issue #1035).

The log renderer flattens a value before it scrubs it (`_observability._flatten`): every control
character and every run of whitespace becomes one space. The assignment rule took a value only when
it followed the ``=`` directly, so ``A_TOKEN=<TAB>…`` was redacted on every path that scrubs raw
text and written in full on a log line, whose renderer had already made the tab a space. The rules
that skipped blanks skipped no control character, so the paths that scrub raw text had the mirror
image (``Bearer <ESC>eyJ…``).

So a gap (whitespace and control characters, exactly what the renderer flattens) is skipped after
every separator a value is anchored to, and wherever a rule skipped a blank. These tests hold the
issue's table on every path that writes text down, through the writer and never only through
`scrub`: the log line a field is rendered into, the prose line, the unspoken text, a command's last
word, a provider error relayed to a timeline, a fact the memory tool stores, the session transcript,
a server tool's record, a text file a tool saved, and a reply held in pieces. The exchange filed into
memory is held in `tests/test_memory_provider.py`, beside the wake that files it.

Every value here is fabricated.
"""

from __future__ import annotations

import io
import json
import logging
import time

import pytest

from basecradle_harness._memory import SqliteMemoryStore
from basecradle_harness._messages import Message, ServerToolRecord, ServerToolUse, ToolCall
from basecradle_harness._observability import (
    _flatten,
    kv,
    last_word,
    log_handler,
    log_unspoken,
)
from basecradle_harness._redaction import (
    REDACTED_VALUE,
    bounded,
    redact_file,
    scrub,
    scrub_parts,
    stable_cut,
)
from basecradle_harness._report import verbatim
from basecradle_harness._server_tools import bound
from basecradle_harness._session import _payload

#: The issue's fabricated value.
SECRET = "secretsecret12345"

#: A fabricated value of a JWT's shape, long enough for every token rule.
JWT = "eyJhbGciOiJIUzI1NiJ9.e30.fabricatedGapSig"

#: Every gap the issue names, as the single characters it names, and the few more that are one.
GAPS = {
    "none": "",
    "space": " ",
    "tab": "\t",
    "nbsp": "\xa0",
    "esc": "\x1b",
    "nul": "\x00",
    "newline": "\n",
    "crlf": "\r\n",
    "del": "\x7f",
    "nel": "\x85",
    "line-separator": " ",
    "run": " \t\x1b\r\n\xa0",
}

#: The issue's table: every text comes out as the name and the marker.
TABLE = [pytest.param(f"A_TOKEN={gap}{SECRET}", id=name) for name, gap in GAPS.items()]

#: Every other rule that anchors a value to a separator, the gap at each place it can stand.
RULES = [
    pytest.param(f"{shape.format(gap=gap, secret=JWT)}", id=f"{rule}-{name}")
    for rule, shape in {
        "nested-name": "x,DB_PASSWORD={gap}{secret}",
        "credential-label": "OPENAI_API_KEY={gap}{secret}",
        "api-key": "url api_key={gap}{secret} next",
        "api-key-quoted": 'url api_key="{gap}{secret}" next',
        "query": "GET /v1?token={gap}{secret} next",
        "header": "x-api-key:{gap}{secret} next",
        "header-before-colon": "x-api-key{gap}: {secret} next",
        "header-quoted": 'x-api-key: "{gap}{secret}" next',
        "member-separator": '{{"password":{gap}"{secret}"}}',
        "member-value": '{{"password": "{gap}{secret}"}}',
        "bearer": "Authorization: Bearer{gap} {secret} next",
        "bearer-gap-only": "Authorization: Bearer {gap}{secret} next",
        "assignment-bearer": "A_TOKEN=Bearer{gap} {secret} next",
    }.items()
    for name, gap in GAPS.items()
]


def _absent(written: str, secret: str = SECRET) -> None:
    assert secret not in written, repr(written)


# --- the rule, read raw and read flattened --------------------------------------------------------


@pytest.mark.parametrize("text", TABLE)
def test_the_tables_value_is_redacted_raw_and_flattened_alike(text, canary):
    # Scrubbed raw (memory, the transcript, a capture file) or flattened first (a log line): the
    # same bytes, the name kept and the value gone, whatever stood between them.
    assert scrub(text) == "A_TOKEN=" + REDACTED_VALUE
    assert scrub(_flatten(text)) == _flatten(scrub(text)) == "A_TOKEN=" + REDACTED_VALUE
    assert not canary.search(scrub(text))


@pytest.mark.parametrize("text", RULES)
def test_every_rule_reads_its_value_past_a_gap(text):
    for written in (scrub(text), scrub(_flatten(text))):
        _absent(written, JWT)
        assert scrub(written) == written  # a fixed point: a second scrub changes nothing


@pytest.mark.parametrize("gap", ["", " ", "\n", "\x1b"])
def test_a_colored_member_value_is_read_to_its_close(gap):
    # A raw ESC once ended a JSON string's reading, so the key after a second color code stayed.
    text = f'{{"password":{gap}"\x1b[1m\x1b[33m{JWT}\x1b[0m", "x": 1}}'
    _absent(scrub(text), JWT)
    assert scrub(text).endswith('", "x": 1}')  # the line still parses past it


@pytest.mark.parametrize(
    "text",
    [
        "GH_TOKEN=",
        "GH_TOKEN= ",
        "GH_TOKEN=\n",
        "GH_TOKEN= \r\n",
        '{"password": "  "}',
        '{"password": "\x1b"}',
    ],
)
def test_a_value_that_is_only_a_gap_has_nothing_to_hide(text):
    assert scrub(text) == text


# --- nothing redacted before is redacted less ----------------------------------------------------
#
# Each found by a differential fuzz of this change against the scrub before it: a wider reading that
# took less somewhere else.


def test_a_gap_before_a_word_the_router_reads_whole_is_the_routers_reading():
    # `\r}eyJ…` is one shell word to the router; after the gap a `}` stands, where no value starts.
    text = f"A_TOKEN=\r}}{JWT} next"
    _absent(scrub(text), JWT)


def test_a_name_the_router_reads_as_its_own_is_still_read():
    # Read past its gap, the first value opens a quote that a record break ends, and the second
    # name stands inside it; the router reads that name as its own, so it is read here too.
    text = f'x,DB_PASSWORD=  \t(  \t"x,DB_PASSWORD=\x00{JWT}'
    _absent(scrub(text), JWT)


@pytest.mark.parametrize(
    "text",
    [
        # A value read past its gap closes at the quote that opens the next member's name.
        f'"api_key": "\t$("password":"\xa0{JWT}"',
        # A separator only the gap reads makes a member whose value is the next member's name.
        f'"password":\x00\x00"api_key": "{JWT}"',
    ],
)
def test_a_member_the_reading_before_found_is_still_found(text):
    _absent(scrub(text), JWT)


def test_a_label_after_a_long_gap_inside_a_value_is_read_on():
    # The first value takes the control characters, so the label after `x,` stands further back
    # than a label's length before the blank where that value stops.
    text = "api_key=x,api_key=" + "\x1b" * 40 + " " + JWT
    _absent(scrub(text), JWT)


@pytest.mark.parametrize("after", [",", "}", "\r}", "\x00}"])
@pytest.mark.parametrize("gap", ["\x1b", "\r", "\x00", " "])
def test_a_gap_before_a_comma_or_a_brace_still_opens_the_value(gap, after):
    # Raw, `\x1b,K9…` is one shell word; flattened, it is a blank and then `,K9…`. Both are the
    # value, so the two orders write the same bytes (the review of #1035).
    text = f"A_TOKEN={gap}{after}{JWT}"
    assert scrub(text) == scrub(_flatten(text)) == "A_TOKEN=" + REDACTED_VALUE


@pytest.mark.parametrize("label", ["?token=", "url api_key=", "x-api-key: "])
@pytest.mark.parametrize("gap", ["\x1b", "\x7f", "\x00", "\x9b"])
@pytest.mark.parametrize("glue", ["%2F", ":", ","])
def test_a_glued_scheme_word_across_a_control_character_takes_the_whole_run(label, gap, glue):
    # The glued alternative once ended at the token's last character, where the run went on.
    text = f"{label}abcbearer{gap}{JWT}{glue}K9q7Zr2LmX4pV8nT"
    _absent(scrub(text), "K9q7Zr2LmX4pV8nT")


@pytest.mark.parametrize("label", ["api_key= ", "?token= ", "x-api-key: "])
def test_a_marker_after_a_label_and_a_gap_is_no_value(label):
    # `api_key= &key=K9…` scrubs to `api_key= [redacted]`; a second scrub changes nothing.
    once = scrub(f"{label}&key=K9q7Zr2LmX4pV8nT")
    _absent(once, "K9q7Zr2LmX4pV8nT")
    assert scrub(once) == once


@pytest.mark.parametrize(
    "text", ["ok DB_PASSWORD=" + " " * 40, 'ok {"api_key":"\x9b' + "\x9b" * 40]
)
def test_a_bound_on_an_empty_value_before_a_gap_is_a_fixed_point(text):
    # Cut inside the gap, a head ending in the name would have its `…` read as the value.
    head = bounded(scrub(text), 20)
    assert scrub(head) == head
    assert head in ("ok…", "ok {…")  # the name goes with its value; a brace before it stays


_KEY = "K9xQ2mZ7pL4vB8nR"


@pytest.mark.parametrize(
    "gap", ["\r\n", "\x1b ", "\x00 ", "\x9b ", "\xa0 ", "\x7f ", "\r\n\x1b[33m"]
)
@pytest.mark.parametrize("after", [",", "}", ""])
@pytest.mark.parametrize("scheme", ["Bearer", "abc-bearer"])
def test_a_scheme_word_the_shell_glues_a_gap_to_still_takes_its_token(scheme, gap, after):
    # The shell reads `Bearer<CR>` as one word, which no longer ends in the scheme word (the second
    # review of #1035); flattened, the same text is `Bearer ,K9…`.
    text = f"A_TOKEN={scheme}{gap}{after}{_KEY}"
    _absent(scrub(text), _KEY)
    _absent(scrub(_flatten(text)), _KEY)


@pytest.mark.parametrize(
    "text",
    [
        f"api_key=Kbearer\x1b,{_KEY}",
        f"?token=abcbearer\x1b[33m{_KEY}",
        f"x-api-key: abcbearer\x00}}{_KEY}",
        f"?token=abcbearer\x1bbearer\x1b{JWT}abcdef%2F{_KEY}",
    ],
)
def test_a_glued_scheme_word_in_a_literal_value_takes_the_run_after_its_gap(text):
    # Raw, the value's run takes the control character and what follows; flattened, the value would
    # end at the scheme word, so the run after its gap goes with it.
    _absent(scrub(text), _KEY)
    _absent(scrub(_flatten(text)), _KEY)


@pytest.mark.parametrize(
    "text",
    [
        f"api_key=Kbearer\x1b\t,{_KEY}",
        f"x-api-key: abcbearer\x00\t,{_KEY}",
        f"?token=.Bearer\x9b :{_KEY}",
    ],
)
def test_a_scheme_word_a_literal_run_took_a_control_character_after_still_reaches_on(text):
    # The run takes the control character after the scheme word, so the word is looked for before
    # it (the third review of #1035).
    _absent(scrub(text), _KEY)
    _absent(scrub(_flatten(text)), _KEY)


@pytest.mark.parametrize(
    "text", ['"password":"Bearer ""password":" ', '{"password": "Bearer ""password": " ']
)
def test_an_open_member_value_that_is_all_gap_goes_and_reads_back_as_itself(text):
    once = scrub(text)
    assert scrub(once) == once


def test_a_bound_on_a_member_value_that_is_all_gap_is_a_fixed_point():
    # Malformed: the first value's close is also the second member's opening quote.
    text = '"password": "\x01"password": "[redacted]"'
    assert scrub(text) == text
    head = bounded(text, 28)
    assert scrub(head) == head
    head_end, tail = stable_cut(text, 27, len(text))
    excerpt = text[:head_end] + "…" + text[tail:]
    assert scrub(excerpt) == excerpt


def test_an_opener_quote_with_a_gap_after_it_at_a_strings_close_opens_nothing():
    # The quote after the label is the string's own close; a gap after it changes nothing.
    assert scrub('{"h":"api_key=" ,"k":"v"}') == '{"h":"api_key=" ,"k":"v"}'


# --- through every writer ------------------------------------------------------------------------


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


@pytest.mark.parametrize("text", TABLE)
def test_a_fields_value_on_its_log_line(text, journal, canary):
    logging.getLogger("basecradle_harness.test").info("tool failed %s", kv(error=text))
    written = journal()
    _absent(written)
    assert "A_TOKEN=" + REDACTED_VALUE in written.replace('"', "'")
    assert not canary.search(written)


@pytest.mark.parametrize("text", TABLE)
def test_a_prose_lines_argument_and_its_traceback(text, journal, canary):
    log = logging.getLogger("basecradle_harness.test")
    log.warning("the vendor refused: %s", text)
    try:
        raise RuntimeError(text)
    except RuntimeError:
        log.exception("it raised")
    written = journal()
    _absent(written)
    assert not canary.search(written)


@pytest.mark.parametrize("text", TABLE)
def test_a_template_that_keeps_its_line_break(text, journal):
    # A harness template is its own, so it is not quoted; a break in it becomes a space.
    logging.getLogger("basecradle_harness.test").info(text)
    _absent(journal())


@pytest.mark.parametrize("text", TABLE)
def test_the_unspoken_text(text, journal, canary):
    log_unspoken(f"I will set {text} and go on.", timeline="019d0000-0000-7000-8000-000000000001")
    written = journal()
    _absent(written)
    assert not canary.search(written)


@pytest.mark.parametrize("text", TABLE)
def test_a_commands_last_word(text):
    _absent(last_word(RuntimeError(text)))


@pytest.mark.parametrize("text", TABLE)
def test_a_provider_error_relayed_to_a_timeline(text):
    _absent(verbatim(RuntimeError(f"400 Bad Request: {text}")))


@pytest.mark.parametrize("text", TABLE)
def test_a_fact_the_memory_tool_stores(text, tmp_path):
    store = SqliteMemoryStore(tmp_path / "memory.db")
    try:
        store.write("vendor", f"the vendor said {text}")
        _absent(store.read("vendor"))
        _absent((tmp_path / "memory.db").read_bytes().decode("latin-1"))
    finally:
        store.close()


@pytest.mark.parametrize("text", TABLE)
def test_the_session_transcript(text):
    call = ToolCall(id="c1", name="shell", arguments={"command": f"export {text}"})
    messages = [
        Message.user(f"use {text}"),
        Message.assistant(content=f"Setting {text} now.", tool_calls=[call]),
        Message(role="tool", content=f"ran: export {text}", tool_call_id="c1"),
    ]
    _absent(_payload(messages))
    _absent(json.loads(_payload(messages))[1]["content"])


@pytest.mark.parametrize("text", TABLE)
def test_a_server_tools_record(text):
    (use,) = bound(ServerToolRecord(uses=[ServerToolUse("web_fetch", f"curl -H {text}")])).uses
    _absent(use.input)


@pytest.mark.parametrize("text", TABLE)
def test_a_text_file_a_tool_saved(text, tmp_path):
    capture = tmp_path / "console.log"
    capture.write_text(f"[info] {text}\n", encoding="utf-8")
    redact_file(capture)
    _absent(capture.read_text(encoding="utf-8"))


@pytest.mark.parametrize("text", TABLE)
def test_a_reply_held_in_pieces_cut_inside_the_gap(text):
    # A reply cut off at its output budget, the cut falling between the `=` and the value.
    cut = len("A_TOKEN=") + (1 if text[len("A_TOKEN=")] != SECRET[0] else 0)
    _absent("".join(scrub_parts([text[:cut], text[cut:]])))


# --- the cost the rule carries --------------------------------------------------------------------


def test_a_word_of_prose_after_a_separator_and_a_gap_is_read_as_its_value():
    # The accepted cost the issue states: a word is over-redacted, a credential never kept.
    assert scrub("set PASSWORD= then restart") == f"set PASSWORD={REDACTED_VALUE} restart"


def test_a_bound_treats_a_value_after_a_gap_as_one_token():
    # The cut reaches the value, so the name goes with it; a later scrub reads no `…` as a value.
    head = bounded(f"ok A_TOKEN= {SECRET} more words", 20)
    assert head == "ok…" and scrub(head) == head


def test_every_gap_reading_is_linear():
    # Each would be quadratic if a reading rescanned a gap per character, or a member per member.
    started = time.monotonic()
    for text in (
        "A_TOKEN=" + "\x1b" * 200_000 + "x",
        "A_TOKEN= (" * 30_000,
        "A_TOKEN= " * 30_000,
        '"password":"\x01' * 30_000,
        '"password":\x00"' * 30_000,
        "api_key=" + "\x1b" * 200_000 + " " + JWT,
        "api_key=x,api_key=" + "\x1b" * 100_000 + " " + JWT,
        "x-api-key:" + "\x7f" * 200_000,
        "Bearer" + "\x00" * 200_000 + JWT,
        "?token=" + " " * 200_000,
    ):
        scrub(text)
        bounded(text, len(text) - 1)
    assert time.monotonic() - started < 10.0
