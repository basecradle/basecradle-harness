"""The repeat rule's key and census (issue #950).

The key is pinned against the real dialogue renderers, so a reworded template fails here rather
than silently filing every repetition of a routine exchange; and against the ways a looser key
would lose what was said (the rule masks only the opening stamp of the peer's half). The census is
pinned to "the newest exchange on its timeline". The fiction: Nova Digital (`nova`, AI) is the
agent; John Doe (`john`) a peer.
"""

import pytest

from basecradle_harness import MemoryExchange, MemoryScope
from basecradle_harness import _wake as wake_module
from basecradle_harness._mempalace import _exchange_markdown, exchange_of
from basecradle_harness._twins import TWIN_RULE, census, twin_key

TASK_A = "019e7761-2222-7333-8444-555566667777"
TASK_B = "019e7761-2222-7333-8444-555566667778"
EVENT_A = "019e7762-3333-7444-8555-666677778888"
EVENT_B = "019e7762-3333-7444-8555-666677778889"
ENDPOINT = "019e7763-4444-7555-8666-777788889999"
ASSET = "019e7760-1111-7222-8333-444455556666"
TIMELINE_A = "019e7750-66ee-7f53-829f-13a8a710b6da"
TIMELINE_B = "019e7750-66ee-7f53-829f-13a8a710b6db"
EARLY = "2026-11-20T06:00:00.000Z"
LATE = "2026-11-20T07:00:00.000Z"


class _Item:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _message(body, *, at=EARLY, handle="john"):
    return _Item(created_at=at, user=_Item(handle=handle), content=_Item(body=body))


def _task(instructions, *, at=EARLY, uuid=TASK_A, due="2026-11-20T06:00:00+00:00"):
    return _Item(
        created_at=at,
        user=_Item(handle="nova"),
        content=_Item(uuid=uuid, activate_at=due, instructions=instructions),
    )


def _event(payload, *, at=EARLY, uuid=EVENT_A):
    return _Item(
        created_at=at,
        webhook_endpoint=_Item(user=_Item(handle="nova"), content=_Item(uuid=ENDPOINT)),
        content=_Item(
            uuid=uuid, content_type="application/json", payload=payload, verified_at_receipt=True
        ),
    )


def _asset(filename, *, at=EARLY):
    return _Item(
        created_at=at,
        user=_Item(handle="john"),
        content=_Item(
            uuid=ASSET,
            file=_Item(filename=filename, content_type="image/png", byte_size=1234),
        ),
    )


def _key(item, kind, reply="Nothing needs me this hour."):
    return twin_key(wake_module._dialogue_of(item, kind), reply)


# === what the harness stamped is set aside =======================================================


def test_a_scheduled_check_in_repeated_has_one_key_whatever_its_stamps():
    """The measured case: the same check-in, a new task each hour, the same routine reply."""
    first = _task("Hourly check-in: read your timelines and act on anything new.")
    again = _task(
        "Hourly check-in: read your timelines and act on anything new.",
        at=LATE,
        uuid=TASK_B,
        due="2026-11-20T07:00:00+00:00",
    )

    assert _key(first, wake_module._TASKS) == _key(again, wake_module._TASKS)


@pytest.mark.parametrize(
    ("first", "again", "kind"),
    [
        (_message("ping"), _message("ping", at=LATE), wake_module._MESSAGES),
        (_event('{"ok":true}'), _event('{"ok":true}', at=LATE, uuid=EVENT_B), wake_module._EVENTS),
        (_asset("skyline.png"), _asset("skyline.png", at=LATE), wake_module._ASSETS),
    ],
)
def test_every_renderers_opening_stamp_is_set_aside(first, again, kind):
    assert _key(first, kind) == _key(again, kind)


def test_a_stamp_written_by_the_sdk_as_a_datetime_is_set_aside_too():
    from datetime import UTC, datetime

    first = _message("ping", at=datetime(2026, 11, 20, 6, tzinfo=UTC))
    again = _message("ping", at=datetime(2026, 11, 20, 7, tzinfo=UTC))

    assert _key(first, wake_module._MESSAGES) == _key(again, wake_module._MESSAGES)


def test_a_batch_sets_aside_only_its_opening_stamp():
    """A batch's later stamps are not at the opening, so a repeated batch is filed: the safe way."""
    first = wake_module._render_batch([_message("ping"), _message("still there?")])
    same_later = wake_module._render_batch([_message("ping", at=LATE), _message("still there?")])
    moved_later = wake_module._render_batch([_message("ping"), _message("still there?", at=LATE)])

    assert twin_key(first, "Yes.") == twin_key(same_later, "Yes.")
    assert twin_key(first, "Yes.") != twin_key(moved_later, "Yes.")


# === what was said is never set aside ============================================================


def test_a_check_in_whose_reply_says_something_new_is_not_a_repeat():
    """The worry the issue names: a scheduled wake can be the one where the agent learned something."""
    check_in = _task("Hourly check-in: read your timelines and act on anything new.")

    assert _key(check_in, wake_module._TASKS) != _key(
        check_in, wake_module._TASKS, reply="John says the deploy key moved to the vault."
    )


def test_near_is_not_a_repeat():
    """Why the rule is exact: a near-twin can be the opposite fact."""
    assert twin_key("Is the deploy ready?", "The deploy is ready.") != twin_key(
        "Is the deploy ready?", "The deploy is not ready."
    )


def test_a_dated_line_in_a_reply_is_what_was_said():
    """The self-review's counterexample: a line opening with a bracketed time, written by the model."""
    check_in = _task("Hourly check-in.")

    assert _key(
        check_in, wake_module._TASKS, reply="Reminder set:\n[2026-11-21T09:00:00Z] call John"
    ) != _key(check_in, wake_module._TASKS, reply="Reminder set:\n[2026-12-01T09:00:00Z] call John")


def test_a_dated_line_inside_a_peers_message_is_what_was_said():
    first = _message("Last backups:\n[2026-11-19T06:00:00Z] ok")
    again = _message("Last backups:\n[2026-11-20T06:00:00Z] ok", at=LATE)

    assert _key(first, wake_module._MESSAGES, "Noted.") != _key(
        again, wake_module._MESSAGES, "Noted."
    )


def test_a_uuid_or_time_a_peer_wrote_is_what_was_said():
    first = _message(f"Use timeline {TIMELINE_A} from 2026-11-21T17:00:00Z.")
    other_uuid = _message(f"Use timeline {TIMELINE_B} from 2026-11-21T17:00:00Z.", at=LATE)
    other_time = _message(f"Use timeline {TIMELINE_A} from 2026-11-22T17:00:00Z.", at=LATE)

    for again in (other_uuid, other_time):
        assert _key(first, wake_module._MESSAGES, "Noted.") != _key(
            again, wake_module._MESSAGES, "Noted."
        )


def test_a_task_header_written_into_a_reply_is_what_was_said():
    header = wake_module._task_dialogue(_task("x"))
    other = wake_module._task_dialogue(_task("x", uuid=TASK_B))

    assert twin_key("ping", header) != twin_key("ping", other)


def test_who_spoke_is_kept():
    assert _key(_message("ping"), wake_module._MESSAGES) != _key(
        _message("ping", handle="nova"), wake_module._MESSAGES
    )


def test_a_different_endpoint_is_a_different_delivery():
    first = _event('{"ok":true}')
    other = _event('{"ok":true}', uuid=EVENT_B)
    other.webhook_endpoint.content.uuid = TIMELINE_B

    assert _key(first, wake_module._EVENTS) != _key(other, wake_module._EVENTS)


def test_only_the_ends_of_each_half_are_trimmed():
    """Indentation is meaning (code); case and punctuation are words ("$5" and "€5")."""
    assert twin_key("It costs $5.", "") != twin_key("It costs €5.", "")
    assert twin_key("Ping", "") != twin_key("ping", "")
    assert twin_key("x", "if a:\n    g()") != twin_key("x", "if a:\ng()")
    assert twin_key("x", "a  b") != twin_key("x", "a b")
    assert twin_key("  ping\n", "ok\n\n") == twin_key("ping", "ok")


def test_no_text_can_move_the_boundary_between_the_halves():
    assert twin_key("a b", "") != twin_key("a", "b")
    assert twin_key("a\x00b", "c") != twin_key("a", "b\x00c")
    assert twin_key("1:a", "") != twin_key("", "a")


def test_a_lone_surrogate_keys_and_never_raises():
    """A ``\\ud800`` escape in a peer's JSON decodes to text that strict UTF-8 cannot encode."""
    import json

    assert twin_key(json.loads('"hi \\ud800"'), "ok") != twin_key("hi", "ok")


def test_the_key_names_its_rule():
    """A changed rule can only ever file an exchange once more, never suppress one it keeps."""
    assert twin_key("ping", "").startswith(f"v{TWIN_RULE}:")


# === the census replays a recorded history =======================================================


def _ping(hour, reply="Here.", timeline=TIMELINE_A):
    return (timeline, f"[2026-11-20T{hour:02d}:00:00Z] john: ping", reply)


def test_a_repeat_of_the_newest_exchange_on_its_timeline_is_skipped():
    found = census([_ping(6), _ping(7), _ping(8)])

    assert (found.total, found.kept, found.skipped, found.collapsed) == (3, 1, 2, 1)


def test_a_repeat_after_something_else_was_said_is_kept():
    """Later is current: ok, down, ok keeps all three, so memory's newest record is ok."""
    history = [_ping(6, "Status ok."), _ping(7, "Status down."), _ping(8, "Status ok.")]

    assert census(history).skipped == 0


def test_the_same_words_on_another_timeline_are_kept():
    history = [_ping(6), _ping(7, timeline=TIMELINE_B), _ping(8), _ping(9, timeline=TIMELINE_B)]

    found = census(history)

    assert (found.kept, found.skipped, found.collapsed) == (2, 2, 2)


def test_an_exchange_with_no_timeline_is_always_kept():
    assert census([_ping(6, timeline=None), _ping(7, timeline=None)]).skipped == 0


@pytest.mark.parametrize(
    ("user", "assistant"),
    [
        ("[2026-11-20T06:00:00Z] john: ping", "Here."),
        ("[2026-11-20T06:00:00Z] john: two\nlines\n\nand a gap", "A reply\n> with a quote inside"),
        ("[2026-11-20T06:00:00Z] john: seeded, never answered", ""),
        ("[2026-11-20T06:00:00Z] john: ping", ">x\nr"),
        ("[2026-11-20T06:00:00Z] john: ping\n", "\n\nr\n"),
        ("", "a reply to nothing"),
    ],
)
def test_an_exchange_file_reads_back_to_the_key_it_was_filed_under(user, assistant):
    """The census reads the files the observe wrote; it must key them as the observe did."""
    exchange = MemoryExchange(user=user, assistant=assistant, scope=MemoryScope(agent="nova"))

    assert twin_key(*exchange_of(_exchange_markdown(exchange))) == twin_key(user, assistant)
