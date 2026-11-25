"""A secret a cut-off reply's cut fell inside is redacted as one text with its resume (issue #1020).

A reply cut off at its output budget and the resume that finished it are one reply in two messages.
Redacted one at a time, a secret the cut fell inside is in neither: its opening can match a shape
and be blanked, while the rest opens the continuation, where nothing recognizes it, and is written
as it stands. So every write the harness makes of a resumed reply redacts it with the text it
continues (`_redaction.scrub_parts`): the transcript (`_session._payload`), the journal
(`WakeAgent._unspoken`), the exchange handed to memory (`WakeAgent._observe`), and the MemPalace
adapter's own record of it (`_exchange_records.continued`).

The proof is a real wake per registered shape, cut at several points inside the secret, and a
search of everything it wrote for the secret's tail.
"""

from __future__ import annotations

import json
import logging

import pytest

from basecradle_harness._memory_provider import MemoryExchange
from basecradle_harness._mempalace_provider import MemPalaceProvider
from basecradle_harness._messages import Message
from basecradle_harness._redaction import (
    REDACTED,
    REDACTED_VALUE,
    scrub,
    scrub_counted,
    scrub_parts,
    scrub_parts_counted,
)
from basecradle_harness._session import _payload, reply_pieces
from tests.test_redaction import RAW
from tests.test_truncation import (
    BODY,
    _CutOff,
    _CutOffThenDown,
    _RecordingMemory,
    _transcript,
    platform,  # noqa: F401 - the fixture, by name
)
from tests.test_wake import M0, build_wake, message, page, serve_messages

#: Each shape the definition registers (``data/secret-shapes.json``), as ``(lead, secret, trail)``:
#: the text before the secret, the secret the cut falls inside, and what closes it. One key for
#: every vendor-key rule, one per label family, the heartbeat URL, and a configured value.
_RUN = "Zq7Wv2Lp9Xc4Rb8NsYdHj3Ke6Tm"
SHAPES = {
    "sk-or-": ("", "sk-or-v1-" + _RUN, ""),
    "sk-": ("", "sk-proj-" + _RUN, ""),
    "xai-": ("", "xai-" + _RUN, ""),
    "bc_[a-z]{3}_": ("", "bc_mcp_" + _RUN, ""),
    "bc_(?:uat|isk)_": ("", "bc_uat_" + _RUN, ""),
    "gh[a-z]_": ("", "ghp_" + _RUN + "Ab3Cd5Ef7Gh", ""),
    "github_pat_": ("", "github_pat_11ABCDEFG0_" + _RUN, ""),
    "AIza": ("", "AIzaSy" + _RUN, ""),
    "hf_": ("", "hf_" + _RUN, ""),
    "r8_": ("", "r8_" + _RUN, ""),
    "ste-": ("", "ste-" + _RUN, ""),
    "abl_": ("", "abl_" + _RUN, ""),
    "pplx-": ("", "pplx-" + _RUN, ""),
    "tk_": ("", "tk_" + _RUN, ""),
    "credential_assignment": ("GH_TOKEN=", _RUN, ""),
    "bearer": ("Authorization: Bearer ", _RUN, ""),
    "query_parameters": ("https://api.example.com/v1/items?token=", _RUN, ""),
    "api_key_assignment": ("api_key=", _RUN, ""),
    "headers": ("x-api-key: ", _RUN, ""),
    "json_members": ('{"password": "', _RUN, '"}'),
    "heartbeat_url": ("https://uptime.betterstack.com/api/v1/heartbeat/", _RUN, ""),
    "configured_values": ("the key is ", "Kq7vZt2Wm9LpXc4RbN8sYdHjQe5UaF3G", ""),
}
CONFIGURED = SHAPES["configured_values"][1]
CLOSE = " and that is the whole of it."


def _registered() -> set[str]:
    """Every vendor-key prefix (each alternative of a rule's top level) and every other section."""
    prefixes = {
        alternative
        for rule in RAW["vendor_keys"]["rules"]
        for alternative in (
            [rule["prefix"]] if "(" in rule["prefix"] else rule["prefix"].split("|")
        )
    }
    sections = {name for name, section in RAW.items() if isinstance(section, dict)}
    return prefixes | (sections - {"markers", "vendor_keys"})


def test_every_registered_shape_has_a_secret_here():
    """A shape added to the file without a row here would be a shape this proof never cut."""
    assert _registered() == set(SHAPES)


def _cuts(secret: str) -> list[int]:
    """Where inside the secret the cut falls: after its first character, a third and half of the way
    in, and before its last."""
    return sorted({1, len(secret) // 3, len(secret) // 2, len(secret) - 1})


CASES = [(shape, cut) for shape, (_, secret, _) in SHAPES.items() for cut in _cuts(secret)]


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("NOVA_API_KEY", CONFIGURED)


def _reply(shape: str) -> tuple[str, int, str]:
    """The reply, where its secret starts, and the secret."""
    lead, secret, trail = SHAPES[shape]
    head = f"Here it is: {lead}"
    return f"{head}{secret}{trail}{CLOSE}", len(head), secret


def _after(shape: str) -> str:
    """What the reply holds after its secret: all a continuation may keep, once it is redacted."""
    return SHAPES[shape][2] + CLOSE


def _whole(shape: str) -> str:
    """The reply as the record files it: the words before the cut, a line break, the rest."""
    redacted = scrub(_reply(shape)[0])
    return redacted[: -len(_after(shape))] + "\n" + _after(shape)


def _absent(piece: str, text: str) -> bool:
    """``piece`` of a secret is not in ``text``; a piece under four characters is too short to tell
    from the words around it, and is checked by the exact assertions instead."""
    return len(piece) < 4 or piece not in text


# --- the one redaction -------------------------------------------------------------------------


@pytest.mark.parametrize(("shape", "cut"), CASES)
def test_a_continuation_keeps_none_of_the_secret_its_cut_fell_inside(configured, shape, cut):
    reply, at, _ = _reply(shape)
    fragment, rest = reply[: at + cut], reply[at + cut :]

    shares = scrub_parts([fragment, rest])

    assert "".join(shares) == scrub(reply)  # the reply as one text…
    assert shares[1] == _after(shape)  # …so the continuation keeps nothing of the secret


@pytest.mark.parametrize(("shape", "cut"), CASES)
def test_redacted_one_at_a_time_the_tail_is_written(configured, shape, cut):
    """The defect, stated: the halves scrubbed alone keep the tail wherever the cut leaves it
    recognizable by nothing. The proof above is not vacuous for the shapes this shows."""
    reply, at, secret = _reply(shape)
    alone = scrub(reply[at + cut :])
    if cut >= len(secret) // 2:  # a short tail: no shape knows it alone
        assert alone.startswith(secret[cut:])


def test_a_piece_is_never_redacted_less_than_alone():
    """Joined, a key glued to the last word of the piece before reads as a word (`_is_word`) and
    is no span; the piece's own reading still takes it."""
    key = "sk-abcdefghijkl"  # a key alone; after a letter, a word (its run has no digit or capital)
    assert scrub(key) == REDACTED
    assert scrub("my word" + key) == "my word" + key  # one text: the scrub declines
    assert scrub_parts(["my word", key]) == ["my word", REDACTED]


def test_one_piece_is_the_scrub_exactly(configured):
    for shape in SHAPES:
        reply = _reply(shape)[0]
        assert scrub_parts_counted([reply]) == ([scrub(reply)], [scrub_counted(reply)[1]])


def test_a_redaction_is_written_once_where_it_starts_and_counted_in_each_piece():
    key = "ghp_" + _RUN + "Ab3Cd5Ef7Gh"
    shares, counts = scrub_parts_counted(["token " + key[:10], "", key[10:] + " done"])
    assert shares == ["token " + REDACTED, "", " done"] and counts == [1, 0, 1]


def test_a_value_marker_stays_with_its_name():
    shares = scrub_parts(["export GH_TOKEN=" + _RUN[:5], _RUN[5:] + " ok"])
    assert shares == ["export GH_TOKEN=" + REDACTED_VALUE, " ok"]


# --- the transcript ----------------------------------------------------------------------------


def _cut_turn(fragment: str, rest: str) -> list[Message]:
    from basecradle_harness._engine import _TRUNCATED_NOTE

    return [
        Message.user(BODY),
        Message.system("[step 1/24]"),
        Message.assistant(content=fragment),
        Message.system(_TRUNCATED_NOTE),
        Message.system("[step 1/24]"),
        Message.assistant(content=rest),
    ]


def test_the_pieces_of_a_reply_are_the_texts_either_side_of_a_truncation_note():
    from basecradle_harness._engine import _TRUNCATED_NOTE

    turn = _cut_turn("a", "b") + [Message.system(_TRUNCATED_NOTE), Message.assistant(content="c")]
    assert reply_pieces(turn) == [[2, 5, 7]]
    # A tool result or a user turn ends a reply: the text after it is a new one.
    broken = turn[:4] + [Message.tool(tool_call_id="call_1", content="x"), *turn[4:6]]
    assert reply_pieces(broken) == []
    # No note, no pieces: two replies of one turn are two texts.
    assert reply_pieces([Message.assistant(content="a"), Message.assistant(content="b")]) == []


@pytest.mark.parametrize(("shape", "cut"), CASES)
def test_the_transcript_holds_no_tail(configured, shape, cut):
    reply, at, _ = _reply(shape)
    saved = json.loads(_payload(_cut_turn(reply[: at + cut], reply[at + cut :])))
    assert saved[2]["content"] + saved[5]["content"] == scrub(reply)
    assert saved[5]["content"] == _after(shape)


# --- a real wake: everything it writes ----------------------------------------------------------


class _Watched(MemPalaceProvider):
    """The MemPalace adapter, recording each exchange the wake hands it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed: list[MemoryExchange] = []

    def observe(self, exchange):
        self.observed.append(exchange)
        super().observe(exchange)


def _written(home, caplog) -> str:
    """Every text the wake wrote down: the transcript, every journal line, every record."""
    records = home / "palace-records"
    texts = [json.dumps(_transcript(home), ensure_ascii=False)]
    texts += [record.getMessage() for record in caplog.records]
    texts += [path.read_text() for path in records.rglob("*.json")] if records.exists() else []
    return "\n".join(texts)


@pytest.mark.parametrize(("shape", "cut"), CASES)
def test_a_wake_cut_inside_a_secret_writes_none_of_its_tail(
    platform,  # noqa: F811 - the fixture
    fake_mempalace,
    configured,
    tmp_path,
    caplog,
    shape,
    cut,
):
    reply, at, secret = _reply(shape)
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    brain = _CutOff(
        Message.assistant(content=reply[: at + cut]),
        Message.assistant(content=reply[at + cut :]),
        reasons=("length", "stop"),
    )
    memory = _Watched(tmp_path / "palace", agent="nova")
    agent, _ = build_wake(tmp_path, brain, memory_provider=memory)

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        agent.wake()

    assert brain.calls == 2  # cut off, then resumed in the same wake (issue #596)
    _, resumed = memory.observed
    assert resumed.resumed and resumed.assistant == _after(shape)
    assert "".join(resumed.continues) + resumed.assistant == scrub(reply)
    transcript = _transcript(tmp_path)
    assert [m["content"] for m in transcript if m["role"] == "assistant"] == [
        *resumed.continues,
        _after(shape),
    ]
    unspoken = [r.getMessage() for r in caplog.records if "unspoken" in r.getMessage()]
    assert CLOSE.strip() in unspoken[-1] and _absent(secret[cut:], unspoken[-1])
    assert _absent(secret[cut:], _written(tmp_path, caplog))
    convo_miner, _ = fake_mempalace
    (drawer,) = convo_miner.filed  # the record, filed whole once the resume finished it
    assert _whole(shape) in drawer["stored"]
    assert _absent(secret[cut:], drawer["stored"]) and _absent(secret[:cut], drawer["stored"])


def test_memory_is_handed_the_words_before_the_cut_as_the_one_redaction_left_them(
    platform,  # noqa: F811 - the fixture
    tmp_path,
):
    reply, at, _ = _reply("gh[a-z]_")
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    brain = _CutOff(
        Message.assistant(content=reply[: at + 4]),
        Message.assistant(content=reply[at + 4 :]),
        reasons=("length", "stop"),
    )
    memory = _RecordingMemory()
    build_wake(tmp_path, brain, memory_provider=memory)[0].wake()

    cut_off, resumed = memory.observed
    # The fragment went to memory before its resume existed, redacted alone: nothing there to know.
    assert cut_off.assistant == "Here it is: ghp_"
    # Neither half alone is a key; the reply is one, and its two halves keep none of it.
    assert resumed.continues == (f"Here it is: {REDACTED}",) and resumed.assistant == CLOSE


@pytest.mark.parametrize(("shape", "cut"), CASES)
def test_a_fragment_saved_before_its_resume_takes_the_rest_of_its_key_with_it(
    configured, shape, cut
):
    """A later wake holds the fragment only as it was saved, redacted alone: a key its opening
    already matched is ``[redacted]`` there, and the rest of it opens the continuation. The marker
    ends the piece, so the run that continues it goes with it (`_through_the_cut`)."""
    reply, at, secret = _reply(shape)
    saved = scrub(reply[: at + cut])

    shares = scrub_parts([saved, reply[at + cut :]])

    # The run goes up to the next blank, so a closer glued to the key (a JSON string's) goes too.
    assert shares[1] in (_after(shape), CLOSE)
    assert _absent(secret[cut:], "".join(shares)) and _absent(secret[:cut], "".join(shares))


@pytest.mark.parametrize(("shape", "cut"), CASES)
def test_a_resume_in_a_later_wake_writes_none_of_the_tail(
    platform,  # noqa: F811 - the fixture
    fake_mempalace,
    configured,
    tmp_path,
    caplog,
    shape,
    cut,
):
    reply, at, secret = _reply(shape)
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    memory = _Watched(tmp_path / "palace", agent="nova")
    first = _CutOffThenDown(Message.assistant(content=reply[: at + cut]))
    build_wake(tmp_path, first, memory_provider=memory)[0].wake()
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    brain = _CutOff(Message.assistant(content=reply[at + cut :]), reasons=("stop",))

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        build_wake(tmp_path, brain, memory_provider=memory)[0].wake()

    assert memory.observed[-1].resumed and memory.observed[-1].assistant in (_after(shape), CLOSE)
    assert [m["content"] for m in _transcript(tmp_path) if m["role"] == "assistant"][-1] in (
        _after(shape),
        CLOSE,
    )
    assert _absent(secret[cut:], _written(tmp_path, caplog))
    convo_miner, _ = fake_mempalace
    (drawer,) = convo_miner.filed
    assert _absent(secret[cut:], drawer["stored"])


def test_a_resume_in_a_later_wake_is_redacted_with_the_fragment_the_transcript_kept(
    platform,  # noqa: F811 - the fixture
    fake_mempalace,
    tmp_path,
    caplog,
):
    """Across wakes the fragment survives only as it was saved, redacted alone. Where no shape knew
    it alone it was saved as written, and the continuation is still redacted with it."""
    key = "ghp_" + _RUN + "Ab3Cd5Ef7Gh"
    fragment, rest = f"Here it is: {key[:12]}", f"{key[12:]}{CLOSE}"
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    memory = _Watched(tmp_path / "palace", agent="nova")
    build_wake(
        tmp_path,
        _CutOffThenDown(Message.assistant(content=fragment)),
        memory_provider=memory,
    )[0].wake()
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    brain = _CutOff(Message.assistant(content=rest), reasons=("stop",))

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        build_wake(tmp_path, brain, memory_provider=memory)[0].wake()

    assert key[12:] not in _written(tmp_path, caplog)
    assert key[:12] not in json.dumps(_transcript(tmp_path))  # the next save took it with the rest
    convo_miner, _ = fake_mempalace
    (drawer,) = convo_miner.filed
    assert key[:12] not in drawer["text"] and key[12:] not in drawer["text"]


def test_a_reply_in_three_pieces_hands_memory_every_piece_redacted_with_the_rest(
    platform,  # noqa: F811 - the fixture
    fake_mempalace,
    tmp_path,
):
    """The fragment opens a key no shape knows at that length, a resume writes nothing and is cut
    off again, and the next resume writes the rest. The opening is in the first piece, so memory is
    handed every earlier piece, and the record puts each back (review of #1020)."""
    key = "ghp_" + _RUN + "Ab3Cd5Ef7Gh"
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    brain = _CutOff(
        Message.assistant(content=f"Here it is: {key[:12]}"),
        Message.assistant(content=""),
        Message.assistant(content=f"{key[12:]}{CLOSE}"),
        reasons=("length", "length", "stop"),
    )
    memory = _Watched(tmp_path / "palace", agent="nova")
    build_wake(tmp_path, brain, memory_provider=memory)[0].wake()

    assert brain.calls == 3
    finished = memory.observed[-1]
    assert finished.continues == (f"Here it is: {REDACTED}", "") and finished.assistant == CLOSE
    convo_miner, _ = fake_mempalace
    (drawer,) = convo_miner.filed
    assert f"Here it is: {REDACTED}\n{CLOSE}" in drawer["stored"]
    assert key[:12] not in drawer["stored"] and key[12:] not in drawer["stored"]
