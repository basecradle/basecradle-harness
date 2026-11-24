"""The scrub and the prune read both adapters' names under ``mempalace``, and hold under the other.

Issue #1011. Under ``mempalace-previous`` the two commands that delete inside a palace hold on every
name the previous adapter does not file under — never delete, never rewrite — so an agent set back
after filing under the new adapter loses nothing. Under ``mempalace`` they read both: a new name's
unit is its exchange (its record and every piece), and an old name's is its conversation file where
it is now, in the palace or in the archive, or they hold.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from basecradle_harness import _prune, _scrub
from basecradle_harness._exchange_records import (
    RecordStore,
    exchange_text,
    new_record,
    records_dir,
)
from basecradle_harness._mempalace import _pieces
from basecradle_harness._mempalace_provider import ITEM_KEY, TIMELINE_KEY
from basecradle_harness._mining import _LEGACY_RECALL_HEADING as LEGACY_HEADING
from basecradle_harness._wake import _STUCK_NOTE
from tests.test_prune import START, TIMELINE, Store, hourly
from tests.test_scrub import FakeCollection

ITEMS = [f"019e7800-0000-7000-8000-{n:012x}" for n in range(1, 40)]
MEMORY = "> [2026-08-02] john: John lives in Dallas.\nNoted — Dallas, Texas."


def _named(collection, store: RecordStore, item: str, user: str, reply: str, *, filed=True):
    """File one exchange the way the new adapter does: its record, then a drawer per piece."""
    record = new_record(timeline=TIMELINE, item=item, agent="nova", user=user, assistant=reply)
    store.write(record)
    pieces = _pieces(exchange_text(record))
    for number, piece in enumerate(pieces, start=1):
        meta = {
            "wing": "conversations",
            "room": "conversations",
            "source_file": record.name(number),
            TIMELINE_KEY: record.timeline,
            ITEM_KEY: record.id,
        }
        collection.add(f"drawer_{item}_{number}", piece.strip(), meta)
    if filed:
        store.mark_filed(record)
    return record


# === the scrub ===================================================================================


class ScrubCollection(FakeCollection):
    def add(self, drawer_id, document, meta):
        self.drawers[drawer_id] = (document, meta)


@pytest.fixture
def palace(tmp_path):
    root = tmp_path / "mempalace"
    (root / "conversations").mkdir(parents=True)
    return root, ScrubCollection({}), RecordStore(records_dir(root))


def test_under_previous_a_new_name_and_any_name_that_is_not_a_path_is_held(palace):
    root, collection, store = palace
    _named(collection, store, ITEMS[0], LEGACY_HEADING, "")
    collection.add("hermes", LEGACY_HEADING, {"source_file": "hermes-session:abc"})
    old = root / "conversations" / "junk.md"
    old.write_text("brief", encoding="utf-8")
    collection.add("old", LEGACY_HEADING, {"source_file": str(old)})

    report = _scrub.scan(collection, root, _scrub.PreviousReader(root))
    deleted, removed = _scrub.apply(collection, report)

    assert [f.drawer_id for f in report.scrub] == ["old"]
    assert {f.drawer_id for f in report.held} == {f"drawer_{ITEMS[0]}_1", "hermes"}
    assert {f.held for f in report.held} == {_scrub._HELD_UNRECOGNISED}
    assert (deleted, removed) == (1, 1)
    assert set(collection.drawers) == {f"drawer_{ITEMS[0]}_1", "hermes"}  # never deleted
    assert store.read(TIMELINE, ITEMS[0]).filed  # and the record never touched


def test_the_library_default_reader_is_the_previous_one(palace):
    root, collection, store = palace
    _named(collection, store, ITEMS[0], LEGACY_HEADING, "")

    report = _scrub.scan(collection, root)

    assert report.scrub == [] and len(report.held) == 1


def test_under_mempalace_a_scaffolding_exchange_goes_whole_record_first(palace):
    root, collection, store = palace
    long = (LEGACY_HEADING + "\n") * 12  # long enough to be cut into pieces
    record = _named(collection, store, ITEMS[0], long, _STUCK_NOTE)
    assert len([d for d in collection.drawers if ITEMS[0] in d]) > 1

    report = _scrub.scan(collection, root, _scrub.NamedReader(root, store))
    deleted, removed = _scrub.apply(collection, report)

    assert removed == 1 and deleted == len(report.scrub) > 1
    assert collection.drawers == {}
    assert store.read(record.timeline, record.id) is None


def test_under_mempalace_one_real_piece_holds_the_whole_exchange(palace):
    root, collection, store = palace
    _named(collection, store, ITEMS[0], LEGACY_HEADING + "\n" + "x " * 500, "real words")

    report = _scrub.scan(collection, root, _scrub.NamedReader(root, store))
    _scrub.apply(collection, report)

    assert report.scrub == []
    assert store.read(TIMELINE, ITEMS[0]).filed and collection.drawers


def test_under_mempalace_an_exchange_not_yet_fully_filed_is_held(palace):
    root, collection, store = palace
    _named(collection, store, ITEMS[0], LEGACY_HEADING, "", filed=False)

    report = _scrub.scan(collection, root, _scrub.NamedReader(root, store))

    assert report.scrub == [] and report.held[0].held == _scrub._HELD_UNFILED


def test_under_mempalace_an_old_drawer_whose_file_is_missing_is_held_never_deleted(palace):
    root, collection, store = palace
    gone = root / "conversations" / "moved-away.md"
    collection.add("old", LEGACY_HEADING, {"source_file": str(gone)})

    report = _scrub.scan(collection, root, _scrub.NamedReader(root, store))
    _scrub.apply(collection, report)

    assert report.scrub == [] and report.held[0].held == _scrub._HELD_MISSING
    assert set(collection.drawers) == {"old"}


def test_under_mempalace_an_old_file_is_found_in_the_archive(palace, tmp_path):
    root, collection, store = palace
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "junk.md").write_text("brief", encoding="utf-8")
    collection.add("old", LEGACY_HEADING, {"source_file": str(root / "conversations" / "junk.md")})

    report = _scrub.scan(collection, root, _scrub.NamedReader(root, store, archive))
    deleted, removed = _scrub.apply(collection, report)

    assert (deleted, removed) == (1, 1)
    assert not (archive / "junk.md").exists()


def test_the_cli_reads_the_reader_off_the_agents_setting(palace, monkeypatch, capsys):
    root, collection, store = palace
    _named(collection, store, ITEMS[0], LEGACY_HEADING, "")
    monkeypatch.setattr(_scrub, "open_palace", lambda _palace: collection)

    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "mempalace-previous")
    assert _scrub.main(["--palace", str(root), "--apply"]) == 0
    assert collection.drawers  # held under the previous adapter

    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "mempalace")
    assert _scrub.main(["--palace", str(root), "--apply"]) == 0
    assert collection.drawers == {} and store.read(TIMELINE, ITEMS[0]) is None


# === the prune ===================================================================================


def _named_check_ins(store: Store, records: RecordStore, count: int) -> list[str]:
    for i in range(count):
        user, reply = hourly(i, START + timedelta(hours=i))
        _named(store.collection, records, ITEMS[i], user, reply)
    return ITEMS[:count]


@pytest.fixture
def prune_store(tmp_path):
    store = Store(tmp_path)
    return store, RecordStore(records_dir(store.palace))


def test_under_previous_the_prune_never_reads_or_removes_a_new_name(prune_store):
    store, records = prune_store
    _named_check_ins(store, records, 8)

    offer = _prune.offer(
        store.collection, store.palace, store.state, _prune.PreviousView(store.palace)
    )

    assert offer.groups == [] and offer.survey.candidates == []
    assert offer.survey.elsewhere == 8  # counted, never deleted
    assert len(store.collection.drawers) == 8


def test_under_mempalace_the_unit_is_the_exchange_record_and_drawers_and_restore_puts_both_back(
    prune_store,
):
    store, records = prune_store
    _named_check_ins(store, records, 8)
    view = _prune.NamedView(store.palace, records)
    before = {path: data for path, data in records.records()}
    drawers_before = dict(store.collection.drawers)

    offer = _prune.offer(store.collection, store.palace, store.state, view)
    (group,) = offer.groups
    result = _prune.apply(
        store.collection, store.palace, store.state, offer.hash, [group.id], [],
        lock=store.lock, view=view,
    )  # fmt: skip

    assert len(result.removed) == len(group.copies) == 7  # the latest pure copy is kept
    left = {path for path, _ in records.records()}
    assert len(left) == 1 and len(store.collection.drawers) == 1

    restored = _prune.restore(
        store.collection, store.palace, store.state, result.backup, lock=store.lock, view=view
    )

    assert len(restored.files) == 7
    assert {path: data for path, data in records.records()} == before
    assert store.collection.drawers == drawers_before


def test_under_mempalace_the_report_says_how_a_memory_is_deleted(prune_store):
    store, records = prune_store
    _named_check_ins(store, records, 8)

    offer = _prune.offer(
        store.collection, store.palace, store.state, _prune.NamedView(store.palace, records)
    )

    text = offer.report.read_text(encoding="utf-8")
    assert _prune.NAMED_REPORT_YES[1] in text and _prune.REPORT_YES[1] not in text
    assert "or the next rebuild files it again" in _prune.NAMED_REPORT_YES[1]


def test_under_mempalace_old_files_are_read_in_the_archive_and_held_when_found_nowhere(
    prune_store, tmp_path
):
    store, records = prune_store
    archive = tmp_path / "archive"
    archive.mkdir()
    names = [store.file(*hourly(i, START + timedelta(hours=i))) for i in range(8)]
    for name in names[:6]:  # six moved to the archive, two lost entirely
        (store.conversations / name).rename(archive / name)
    for name in names[6:]:
        (store.conversations / name).unlink()
    view = _prune.NamedView(store.palace, records, archive)

    offer = _prune.offer(store.collection, store.palace, store.state, view)

    assert offer.survey.exchanges == 6 and offer.survey.elsewhere == 2
    (group,) = offer.groups
    result = _prune.apply(
        store.collection, store.palace, store.state, offer.hash, [group.id], [],
        lock=store.lock, view=view,
    )  # fmt: skip
    assert len(result.removed) == 5
    assert len(list(archive.iterdir())) == 1
    assert len(store.collection.drawers) == 3  # the kept copy and the two whose files are lost


def test_an_exchange_that_becomes_unfiled_between_scan_and_apply_keeps_its_drawers(palace):
    """Its unit is read again at the apply; one that is no longer provable deletes nothing."""
    root, collection, store = palace
    record = _named(collection, store, ITEMS[0], LEGACY_HEADING, "")
    reader = _scrub.NamedReader(root, store)
    report = _scrub.scan(collection, root, reader)
    assert len(report.scrub) == 1
    store._touch(store._marker(record.timeline, record.id))  # a refile began in between

    assert _scrub.apply(collection, report) == (0, 0)
    assert collection.drawers and store.read(TIMELINE, ITEMS[0]) is not None
