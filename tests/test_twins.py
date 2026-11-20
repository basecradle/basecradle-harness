"""The twin rule: an exchange filed once, word for word, is not filed again (issue #950).

Pinned against the real dialogue renderers, so a reworded template fails here rather than silently
filing every repetition of a routine exchange (the rule masks a stamp only where a renderer puts
it). The fiction: Nova Digital (`nova`, AI) is the agent; John Doe (`john`) a peer.
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


def test_a_scheduled_check_in_repeated_is_a_twin_whatever_its_stamps():
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
def test_every_renderers_stamps_are_set_aside(first, again, kind):
    assert _key(first, kind) == _key(again, kind)


def test_a_batch_with_only_its_stamps_changed_is_a_twin():
    first = wake_module._render_batch([_message("ping"), _message("still there?")])
    again = wake_module._render_batch(
        [_message("ping", at=LATE), _message("still there?", at=LATE)]
    )

    assert twin_key(first, "Yes.") == twin_key(again, "Yes.")


# === what was said is never set aside ============================================================


def test_a_check_in_whose_reply_says_something_new_is_not_a_twin():
    """The worry the issue names: a scheduled wake can be the one where the agent learned something."""
    check_in = _task("Hourly check-in: read your timelines and act on anything new.")

    assert _key(check_in, wake_module._TASKS) != _key(
        check_in, wake_module._TASKS, reply="John says the deploy key moved to the vault."
    )


def test_near_is_not_a_twin():
    """Why the rule is exact: a near-twin can be the opposite fact."""
    assert twin_key("Is the deploy ready?", "The deploy is ready.") != twin_key(
        "Is the deploy ready?", "The deploy is not ready."
    )


def test_a_uuid_or_time_a_peer_wrote_is_what_was_said():
    """A stamp is masked only where a renderer puts it, never wherever its shape appears."""
    first = _message(f"Use timeline {TIMELINE_A} from 2026-11-21T17:00:00Z.")
    again = _message(f"Use timeline {TIMELINE_B} from 2026-11-21T17:00:00Z.", at=LATE)
    later = _message(f"Use timeline {TIMELINE_A} from 2026-11-22T17:00:00Z.", at=LATE)

    assert _key(first, wake_module._MESSAGES, "Noted.") != _key(
        again, wake_module._MESSAGES, "Noted."
    )
    assert _key(first, wake_module._MESSAGES, "Noted.") != _key(
        later, wake_module._MESSAGES, "Noted."
    )


def test_who_spoke_is_kept():
    assert _key(_message("ping"), wake_module._MESSAGES) != _key(
        _message("ping", handle="nova"), wake_module._MESSAGES
    )


def test_a_different_endpoint_is_a_different_delivery():
    first = _event('{"ok":true}')
    other = _event('{"ok":true}', uuid=EVENT_B)
    other.webhook_endpoint.content.uuid = TIMELINE_B

    assert _key(first, wake_module._EVENTS) != _key(other, wake_module._EVENTS)


def test_case_and_punctuation_are_words_and_whitespace_runs_are_one_space():
    assert twin_key("It costs $5.", "") != twin_key("It costs €5.", "")
    assert twin_key("Ping", "") != twin_key("ping", "")
    assert twin_key("ping  me\n later", "ok") == twin_key("ping me later ", "ok")


def test_the_halves_never_merge():
    assert twin_key("a b", "") != twin_key("a", "b")


def test_the_key_names_its_rule():
    """A changed rule can only ever file an exchange once more, never suppress one it keeps."""
    assert twin_key("ping", "").startswith(f"v{TWIN_RULE}:")


# === the census replays a recorded history =======================================================


def test_the_census_keeps_the_first_of_each_twin_set_and_skips_the_rest():
    history = [
        ("[2026-11-20T06:00:00Z] john: ping", "Here."),
        ("[2026-11-20T07:00:00Z] john: ping", "Here."),
        ("[2026-11-20T08:00:00Z] john: ping", "Here, John."),
        ("[2026-11-20T09:00:00Z] john: ping", "Here."),
    ]

    found = census(history)

    assert (found.total, found.kept, found.skipped, found.collapsed) == (4, 2, 2, 1)


@pytest.mark.parametrize(
    ("user", "assistant"),
    [
        ("[2026-11-20T06:00:00Z] john: ping", "Here."),
        ("[2026-11-20T06:00:00Z] john: two\nlines\n\nand a gap", "A reply\n> with a quote inside"),
        ("[2026-11-20T06:00:00Z] john: seeded, never answered", ""),
    ],
)
def test_an_exchange_file_reads_back_to_the_key_it_was_filed_under(user, assistant):
    """The census reads the files the observe wrote; it must key them as the observe did."""
    exchange = MemoryExchange(user=user, assistant=assistant, scope=MemoryScope(agent="nova"))

    assert twin_key(*exchange_of(_exchange_markdown(exchange))) == twin_key(user, assistant)
