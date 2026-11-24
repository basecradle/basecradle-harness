"""The MemPalace adapter selected by ``mempalace`` (issue #1011), against a fake MemPalace.

`tests/test_mempalace_provider_real.py` proves what MemPalace itself then does with each call, on a
real palace. This proves how the adapter calls it: the name every drawer is filed under, the record
written before the drawers, the door rule for recurring tasks, "already filed?", the retry fitted to
the library's lock refusal, the next wake's refile, the labels, and that nothing the library owns
is ever written.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from basecradle_harness import _exchange_records, _mempalace_provider
from basecradle_harness._exchange_records import (
    NAME,
    RecordStore,
    exchange_name,
    parse_name,
    records_dir,
    task_key,
)
from basecradle_harness._memory_provider import MemoryExchange, MemoryScope
from basecradle_harness._mempalace import (
    EXCHANGE_PIECE_CHARS,
    QUOTED_LABEL,
    MemPalaceMemoryProvider,
)
from basecradle_harness._mempalace_provider import (
    ITEM_KEY,
    LOCK_RETRY_SECONDS,
    ROOM,
    TIMELINE_KEY,
    MemPalaceProvider,
    render_hits,
)
from basecradle_harness._messages import Message
from basecradle_harness._wake import _MESSAGES, _TASKS, _turn_item

AGENT = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"
TIMELINE = "019e7750-66ee-7f53-829f-13a8a710b6da"
OTHER = "019e7740-1d2e-7a3b-8c4d-5e6f7a8b9c0d"
M0 = "019e7751-4a1b-7c2d-8e3f-1a2b3c4d5e6f"
M1 = "019e7752-5b2c-7d3e-9f40-2b3c4d5e6f70"
T0 = "019e7770-5555-7eee-8fff-506172839405"
T1 = "019e7771-6666-7fff-8aaa-617283940516"
T2 = "019e7772-7777-7aaa-8bbb-728394051627"

#: Every metadata field the library's bulk miner owns, or sets for its own chunking (issue #1011).
#: The new adapter's modules may not contain one of these names at all.
MINER_OWNED = (
    "source_mtime",
    "chunk_total",
    "content_hash",
    "chunker_version",
    "normalize_version",
    "id_source_file",
    "chunk_index",
)


def _exchange(user="[2026-06-04T00:00:00.000Z] john: Where do I live?", assistant="Dallas.", **kw):
    timeline = kw.pop("timeline", TIMELINE)
    return MemoryExchange(
        user=user, assistant=assistant, scope=MemoryScope(agent=AGENT, timeline=timeline), **kw
    )


@pytest.fixture
def provider(fake_mempalace, tmp_path):
    slept: list[float] = []
    provider = MemPalaceProvider(tmp_path / "palace", agent="nova", sleep=slept.append)
    provider.slept = slept
    return provider


# --- the name, the record, and nothing else --------------------------------------------------------


def test_every_drawer_is_named_by_timeline_item_and_piece_with_our_two_keys_and_nothing_else(
    provider, fake_mempalace
):
    convo_miner, _ = fake_mempalace

    provider.observe(_exchange(item=M0))

    (drawer,) = convo_miner.filed
    assert drawer["source_file"] == f"basecradle:{TIMELINE}:{M0}:1"  # piece 1 is always written
    assert NAME.match(drawer["source_file"])
    assert drawer["extra_metadata"] == {TIMELINE_KEY: TIMELINE, ITEM_KEY: M0}
    assert drawer["wing"] == "conversations" and drawer["room"] == ROOM
    assert drawer["text"] == "> [2026-06-04T00:00:00.000Z] john: Where do I live?\nDallas.\n"


def test_nothing_is_written_inside_the_palace_and_the_record_says_it_is_filed(provider):
    provider.observe(_exchange(item=M0))

    assert not (provider.palace_path / "conversations").exists()
    record = provider.records.read(TIMELINE, M0)
    assert (
        record.filed and record.user.endswith("Where do I live?") and record.assistant == "Dallas."
    )
    assert provider.records.unfiled() == []
    assert provider.records.root == records_dir(provider.palace_path)
    assert provider.records.root.parent == provider.palace_path.parent  # beside, never inside


def test_the_record_is_private(provider):
    provider.observe(_exchange(item=M0))

    path = provider.records.path(TIMELINE, M0)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert provider.records.root.stat().st_mode & 0o777 == 0o700


def test_a_long_exchange_is_one_name_per_piece(provider, fake_mempalace):
    convo_miner, _ = fake_mempalace
    long = "\n".join(f"line {n}: " + "x" * 90 for n in range(30))

    provider.observe(_exchange(user=long, item=M0))

    names = [drawer["source_file"] for drawer in convo_miner.filed]
    assert len(names) > 1
    assert names == [exchange_name(TIMELINE, M0, n) for n in range(1, len(names) + 1)]
    assert all(len(drawer["text"]) <= EXCHANGE_PIECE_CHARS for drawer in convo_miner.filed)


def test_an_exchange_that_answered_no_item_is_named_from_its_own_record_id(
    provider, fake_mempalace
):
    convo_miner, _ = fake_mempalace

    provider.observe(_exchange(item=None))

    parsed = parse_name(convo_miner.filed[0]["source_file"])
    assert parsed.timeline == TIMELINE and parsed.piece == 1
    assert provider.records.read(TIMELINE, parsed.id).filed


def test_an_exchange_on_no_timeline_says_so_in_its_name(provider, fake_mempalace):
    convo_miner, _ = fake_mempalace

    provider.observe(_exchange(item=M0, timeline=None))

    assert convo_miner.filed[0]["source_file"] == f"basecradle:none:{M0}:1"


def test_an_empty_exchange_writes_nothing(provider, fake_mempalace):
    convo_miner, _ = fake_mempalace

    provider.observe(_exchange(user="  ", assistant="", item=M0))

    assert convo_miner.filed == [] and not provider.records.root.exists()


def test_results_are_read_by_attribute_never_by_the_dict_style(provider, monkeypatch):
    """MemPalace keeps ``result.get("ids")`` only as a migration shim scheduled for removal."""
    import sys

    probe = sys.modules["mempalace.palace"].get_collection(str(provider.palace_path))
    result_type = type(probe.get(where={"wing": "conversations"}, include=[]))

    def refuse(self, *args, **kwargs):
        raise AssertionError("the dict style was used on a MemPalace result")

    monkeypatch.setattr(result_type, "get", refuse)

    provider.observe(_exchange(item=M0))
    provider.observe(_exchange(item=M0))  # the second asks "already filed?" and reads results

    assert provider.records.read(TIMELINE, M0).filed


@pytest.mark.parametrize("module", [_mempalace_provider, _exchange_records])
def test_the_new_adapter_names_no_field_the_library_owns(module):
    """The grep is the test: not one of these names appears in the new adapter's modules."""
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert [name for name in MINER_OWNED if name in source] == []


# --- already filed? ---------------------------------------------------------------------------------


def test_a_second_filing_of_the_same_exchange_adds_nothing(provider, fake_mempalace, caplog):
    convo_miner, _ = fake_mempalace
    provider.observe(_exchange(item=M0))
    before = provider.records.path(TIMELINE, M0).read_bytes()

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.observe(_exchange(item=M0))

    assert len(convo_miner.filed) == 1
    assert provider.records.path(TIMELINE, M0).read_bytes() == before
    assert "result=skipped reason=already_filed" in caplog.text


def test_a_drawer_the_previous_adapter_filed_is_seen_as_already_filed(
    fake_mempalace, tmp_path, caplog
):
    convo_miner, _ = fake_mempalace
    palace = tmp_path / "palace"
    MemPalaceMemoryProvider(palace, agent="nova", name="mempalace-previous").observe(_exchange())
    new = MemPalaceProvider(palace, agent="nova", sleep=lambda s: None)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        new.observe(_exchange(item=M0))

    assert len(convo_miner.filed) == 1  # the previous adapter's drawer, and no second one
    assert "reason=already_filed" in caplog.text
    assert new.records.read(TIMELINE, M0) is None


def test_an_old_drawer_on_another_timeline_is_not_this_exchange(fake_mempalace, tmp_path):
    convo_miner, _ = fake_mempalace
    palace = tmp_path / "palace"
    MemPalaceMemoryProvider(palace, agent="nova", name="mempalace-previous").observe(
        _exchange(timeline=OTHER)
    )

    MemPalaceProvider(palace, agent="nova", sleep=lambda s: None).observe(_exchange(item=M0))

    assert len(convo_miner.filed) == 2


# --- the door rule for recurring tasks --------------------------------------------------------------


def _activation(task, instructions, reply, timeline=TIMELINE):
    user = (
        f"[2026-06-10T00:00:00.000Z] A task scheduled by @john has activated (task {task}).\n"
        f"{instructions}"
    )
    return _exchange(
        user=user, assistant=reply, item=task, task_instructions=instructions, timeline=timeline
    )


def test_the_first_firing_is_filed_and_a_near_copy_of_it_is_neither_filed_nor_recorded(
    provider, fake_mempalace, caplog
):
    convo_miner, _ = fake_mempalace
    provider.observe(_activation(T0, "Check the build at 14:00.", "Build 4411 is green at 14:00."))

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.observe(
            _activation(T1, "Check the build at 15:00.", "Build 4412 is green at 15:00.")
        )

    assert [parse_name(d["source_file"]).id for d in convo_miner.filed] == [T0]
    assert provider.records.read(TIMELINE, T1) is None  # not written as a record either
    assert "result=skipped reason=near_copy" in caplog.text


def test_a_reply_that_says_something_new_is_filed_and_becomes_the_one_compared_against(
    provider, fake_mempalace
):
    convo_miner, _ = fake_mempalace
    provider.observe(_activation(T0, "Check the build.", "The build is green."))
    provider.observe(
        _activation(T1, "Check the build.", "The build failed: the linker ran out of memory.")
    )
    provider.observe(
        _activation(T2, "Check the build.", "The build failed: the linker ran out of memory again.")
    )

    assert [parse_name(d["source_file"]).id for d in convo_miner.filed] == [T0, T1]


def test_the_rule_is_keyed_on_the_task_never_on_its_words(provider, fake_mempalace):
    """Different instructions, a different timeline, or no task at all: each is its own memory."""
    convo_miner, _ = fake_mempalace
    provider.observe(_activation(T0, "Check the build.", "All green."))
    provider.observe(_activation(T1, "Check the deploy.", "All green."))
    provider.observe(_activation(T2, "Check the build.", "All green.", timeline=OTHER))
    provider.observe(_exchange(user="john: routine check-in", assistant="All green.", item=M0))
    provider.observe(_exchange(user="john: routine check-in", assistant="All green.", item=M1))

    assert [parse_name(d["source_file"]).id for d in convo_miner.filed] == [T0, T1, T2, M0, M1]


def test_the_task_key_normalises_ids_times_and_numbers_and_keeps_the_timeline():
    assert task_key(TIMELINE, "Check task 7 at 14:00.") == task_key(
        TIMELINE, "Check task 9 at 9:30."
    )
    assert task_key(TIMELINE, "Check the build.") != task_key(TIMELINE, "Check the deploy.")
    assert task_key(TIMELINE, "Check the build.") != task_key(OTHER, "Check the build.")


# --- the retry, fitted to one process per event -----------------------------------------------------


def test_the_lock_refusal_is_retried_after_one_two_and_four_seconds(provider, fake_mempalace):
    convo_miner, _ = fake_mempalace
    convo_miner.busy = 3

    provider.observe(_exchange(item=M0))

    assert provider.slept == [1, 2, 4]
    assert len(convo_miner.filed) == 1 and provider.records.read(TIMELINE, M0).filed


def test_still_busy_after_the_last_retry_leaves_the_record_unfiled_and_the_next_wake_files_it_once(
    provider, fake_mempalace, caplog
):
    convo_miner, _ = fake_mempalace
    convo_miner.busy = len(LOCK_RETRY_SECONDS) + 1

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.observe(_exchange(item=M0))

    assert provider.slept == list(LOCK_RETRY_SECONDS)
    assert convo_miner.filed == []
    assert provider.records.read(TIMELINE, M0).filed is False
    assert provider.records.unfiled() == [(TIMELINE, M0)]
    assert "result=busy" in caplog.text and "attempts=6" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.begin_wake()
        provider.begin_wake()  # a second wake finds nothing left to file

    assert [d["source_file"] for d in convo_miner.filed] == [f"basecradle:{TIMELINE}:{M0}:1"]
    assert provider.records.read(TIMELINE, M0).filed and provider.records.unfiled() == []
    assert "memory op=refile provider=mempalace records=1 filed=1" in caplog.text


def test_a_refile_completes_an_exchange_whose_filing_stopped_partway(provider, fake_mempalace):
    """A piece already in the palace is never filed twice: the refile files only what is missing."""
    convo_miner, _ = fake_mempalace
    long = "\n".join(f"line {n}: " + "x" * 90 for n in range(30))
    real = convo_miner.file_conversation_exchange
    calls = {"n": 0}

    def dies_after_two(collection, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("the disk is full")
        return real(collection, **kwargs)

    convo_miner.file_conversation_exchange = dies_after_two
    provider.observe(_exchange(user=long, item=M0))
    assert len(convo_miner.filed) == 2 and provider.records.unfiled() == [(TIMELINE, M0)]

    convo_miner.file_conversation_exchange = real
    provider.begin_wake()

    names = [d["source_file"] for d in convo_miner.filed]
    assert len(names) == len(set(names))  # nothing filed twice
    assert names == [exchange_name(TIMELINE, M0, n) for n in range(1, len(names) + 1)]


def test_any_other_error_is_never_retried(provider, fake_mempalace, caplog):
    convo_miner, _ = fake_mempalace

    def refuses(collection, **kwargs):
        raise ValueError("an embedding of the wrong width")

    convo_miner.file_conversation_exchange = refuses
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.observe(_exchange(item=M0))

    assert provider.slept == []
    assert provider.records.unfiled() == [(TIMELINE, M0)]
    assert "result=failed" in caplog.text and "error=ValueError" in caplog.text


def test_a_marker_whose_record_was_never_written_is_dropped_with_its_partial(provider):
    store = provider.records
    store._touch(store._marker(TIMELINE, M0))
    store._folder(store.path(TIMELINE, M0).parent)
    partial = store.path(TIMELINE, M0).with_name(f".{M0}.json.partial")
    partial.write_bytes(b"{")

    provider.begin_wake()

    assert store.unfiled() == [] and not partial.exists()


def test_an_unreadable_record_keeps_its_marker(provider, caplog):
    store = provider.records
    store._touch(store._marker(TIMELINE, M0))
    store._folder(store.path(TIMELINE, M0).parent)
    store.path(TIMELINE, M0).write_bytes(b"not json")

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.begin_wake()

    assert store.unfiled() == [(TIMELINE, M0)]
    assert "unreadable=1" in caplog.text


# --- recall -----------------------------------------------------------------------------------------


def test_a_seeded_exchange_is_kept_out_of_this_wakes_turn0_recall(provider, fake_mempalace):
    _, searcher = fake_mempalace
    provider.palace_path.mkdir()  # the fake files nothing to disk; a real palace exists by now
    provider.observe(_exchange(item=M0, seeded=True))
    name = f"basecradle:{TIMELINE}:{M0}:1"
    searcher.result = {
        "results": [
            {"text": "seeded", "source_file": name, "source_path": name},
            {"text": "other", "source_file": "x.md", "source_path": "/p/conversations/x.md"},
        ]
    }

    block = provider.context(MemoryScope(agent=AGENT, timeline=TIMELINE, query="where"))

    assert "other" in block and "seeded" not in block
    provider.begin_wake()
    assert provider._seeded_files == set()


def _hit(source, filed_at="2026-11-23T10:00:00"):
    return {"text": "a memory", "source_file": Path(source).name, "source_path": source,
            "filed_at": filed_at}  # fmt: skip


def test_labels_read_the_timeline_from_either_adapters_name():
    new = _hit(f"basecradle:{TIMELINE}:{M0}:2")
    old = _hit(f"/home/nova/harness/mempalace/conversations/{OTHER}.{'a' * 32}.md")
    bare = _hit(f"/home/nova/harness/mempalace/conversations/{'b' * 32}.md")
    nowhere = _hit(f"basecradle:none:{M0}:1")

    rendered = render_hits([new, old, bare, nowhere], here=TIMELINE).splitlines()

    assert "· this timeline) a memory" in rendered[0]
    assert f"· another timeline, {OTHER}) a memory" in rendered[1]
    assert "timeline" not in rendered[2] and "timeline" not in rendered[3]
    for line in rendered:  # the grammar the mining strip recognises when a reply quotes it back
        assert QUOTED_LABEL.match(line)


def test_the_search_tool_labels_the_same_way(provider, fake_mempalace):
    _, searcher = fake_mempalace
    provider.palace_path.mkdir()
    name = f"basecradle:{OTHER}:{M0}:1"
    searcher.result = {"results": [{"text": "a memory", "source_file": name, "source_path": name}]}
    (tool,) = provider.tools()
    tool.bind(SimpleNamespace(timeline=TIMELINE))

    assert f"(another timeline, {OTHER}) a memory" in tool.run("anything")


# --- the item a recovered turn is named by ------------------------------------------------------------


def test_a_recovered_turn_is_named_as_the_live_one_would_be():
    item = SimpleNamespace(content=SimpleNamespace(uuid=M0))
    batch = Message.user("two messages")
    batch.items = [M0, M1]

    assert _turn_item(batch, item, _MESSAGES) == M1  # the newest it carried
    assert _turn_item(Message.user("legacy"), item, _MESSAGES) == M0  # no items recorded
    assert _turn_item(batch, SimpleNamespace(content=SimpleNamespace(uuid=T0)), _TASKS) == T0


# --- the record store --------------------------------------------------------------------------------


def test_a_record_round_trips_and_refuses_what_is_not_one(tmp_path):
    store = RecordStore(tmp_path / "records")
    record = _exchange_records.new_record(
        timeline=TIMELINE, item=M0, agent="nova", user="hi", assistant="hello"
    )
    store.write(record)

    assert store.read(TIMELINE, M0) == record
    store.path(TIMELINE, M0).write_text(json.dumps({"format": 99}))
    with pytest.raises(ValueError):
        store.read(TIMELINE, M0)


def test_a_name_is_parsed_only_in_the_exact_shape_it_is_written():
    assert parse_name(exchange_name(TIMELINE, M0, 3)).piece == 3
    for wrong in (
        f"basecradle:{TIMELINE}:{M0}",
        f"basecradle:{TIMELINE}:{M0}:0",
        f"basecradle:{TIMELINE}:{M0}:1:extra",
        f"hermes-session:{M0}",
        f"/home/nova/conversations/{TIMELINE}.{'a' * 32}.md",
        None,
    ):
        assert parse_name(wrong) is None, wrong
    assert re.fullmatch(NAME, exchange_name("none", M0, 1))


def test_a_late_refile_never_moves_the_task_pointer_back(tmp_path):
    store = RecordStore(tmp_path / "records")
    older = _exchange_records.new_record(
        timeline=TIMELINE, item=T0, agent="nova", user="u", assistant="old", task_instructions="i"
    )
    newer = _exchange_records.new_record(
        timeline=TIMELINE, item=T1, agent="nova", user="u", assistant="new", task_instructions="i"
    )
    for record in (newer, older):  # the older activation is filed last, by a refile
        store.write(record)
        store.remember_task(task_key(TIMELINE, "i"), store.mark_filed(record))

    assert store.last_task_reply(task_key(TIMELINE, "i")).assistant == "new"
