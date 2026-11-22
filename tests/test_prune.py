"""`basecradle-harness-prune-palace`: offer an agent the removal of its routine task check-ins.

MemPalace is an optional extra and not installed in the default run, so the store is the scrub's
kind of fake: an in-memory drawer collection with the `get`/`delete` surface the tool uses, beside
real exchange files on disk written in the observe's own format (`_exchange_markdown`). The real
library is exercised by `test_prune_real.py`.

The data is fabricated and shaped like the store issue #973 counted: a month of hourly check-ins
(the activation line, a templated instruction, a templated reply, a different time each hour) beside
later tasks that all differ from each other, plus the traps the rule must not fall into.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from basecradle_harness import _prune
from basecradle_harness._memory_provider import MemoryExchange, MemoryScope
from basecradle_harness._mempalace import _exchange_markdown
from basecradle_harness._wake import _task_dialogue

TIMELINE = "019a3b2c-4d5e-7f60-8a1b-2c3d4e5f6a7b"
START = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
SCOPE = MemoryScope(agent="nova", timeline=TIMELINE)


def _uuid(n: int) -> str:
    return f"019a3b2c-4d5e-7f60-8a1b-{n:012x}"


def _stamp(when: datetime) -> str:
    return when.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def current_line(n: int, when: datetime, handle: str = "nova") -> str:
    """Today's activation line, from the harness's own renderer."""
    task = SimpleNamespace(
        created_at=_stamp(when - timedelta(minutes=5)),
        user=SimpleNamespace(handle=handle),
        content=SimpleNamespace(uuid=_uuid(n), activate_at=_stamp(when)),
    )
    return _task_dialogue(task)


def legacy_line(n: int, when: datetime) -> str:
    """The wording before 0.155.0, as mined before #438: stamped, with its directive."""
    return (
        f"[{_stamp(when - timedelta(minutes=5))}] A task you scheduled has activated and is due now "
        f"(task {_uuid(n)}, scheduled for {_stamp(when)}). Carry out its instructions:"
    )


def hourly(n: int, when: datetime) -> tuple[str, str]:
    """One hourly check-in: the legacy line, a templated instruction, a templated reply."""
    clock = when.strftime("%H:%M")
    user = (
        f"{legacy_line(n, when)}\n"
        f"Check the status board and report anything new since the last check at {clock}."
    )
    reply = f"Checked the status board at {clock}. Nothing new since the last check; {n % 4} open."
    return user, reply


LATER = [
    ("Draft the release notes for the router and post them here.", "Drafted and posted."),
    ("Ask john whether the Dallas meetup moved to Thursday.", "Asked; waiting on his answer."),
    ("Summarise yesterday's palace measurements in three lines.", "Three lines, posted above."),
    ("Rotate the staging webhook endpoint and record the new id.", "Rotated; the new id is noted."),
    ("Read the new constitution section on sovereignty.", "Read it. Two questions filed."),
    ("Water the plants on the balcony webcam check.", "The webcam shows them watered."),
    ("Send a birthday note to the design channel.", "Sent, with a small poem."),
    ("Compare the two embedding models on the probe set.", "The smaller one won on recall."),
    ("Book the review slot for the memory tool.", "Booked for Tuesday afternoon."),
    ("Close the stale experiment timelines from October.", "Closed four of them."),
    ("Check the mempalace lock directory size.", "It holds 212 files, 1.1 MB."),
    ("Write a haiku about latency for the newsletter.", "Written and sent to the editor."),
    ("Reconcile the vendor invoice against the spend log.", "They match to the cent."),
    ("Prepare questions for the founders' Friday sync.", "Five questions drafted."),
]


class FakeCollection:
    """The drawer surface the tool uses: paged `get`, `get` by ids, `delete` by id."""

    def __init__(self):
        self.drawers: dict[str, tuple[str, dict, list[float]]] = {}
        self.deleted: list[str] = []

    def add(self, drawer_id, document, meta):
        self.drawers[drawer_id] = (document, meta, [0.25, 0.5, 0.75])

    def get(self, *, limit=None, offset=0, include=None, ids=None, where=None):
        if ids is not None:
            items = [(i, self.drawers[i]) for i in ids if i in self.drawers]
        else:
            items = list(self.drawers.items())[offset : (offset + limit) if limit else None]
        page = {
            "ids": [i for i, _ in items],
            "documents": [p[0] for _, p in items],
            "metadatas": [p[1] for _, p in items],
        }
        if include and "embeddings" in include:
            page["embeddings"] = [p[2] for _, p in items]
        return page

    def delete(self, *, ids=None, where=None):
        for drawer_id in ids or []:
            self.deleted.append(drawer_id)
            self.drawers.pop(drawer_id, None)

    def upsert(self, *, ids, documents, metadatas, embeddings=None):
        for index, drawer_id in enumerate(ids):
            embedding = embeddings[index] if embeddings is not None else [0.0]
            self.drawers[drawer_id] = (documents[index], metadatas[index], embedding)


class Store:
    """A palace directory with exchange files on disk and a fake drawer collection beside it."""

    def __init__(self, root: Path):
        self.root = root
        self.palace = root / "home" / "mempalace"
        self.conversations = self.palace / "conversations"
        self.conversations.mkdir(parents=True)
        self.state = root / "home" / _prune.STATE_DIR
        self.collection = FakeCollection()
        self.count = 0

    def file(self, user: str, reply: str, *, drawer: bool = True) -> str:
        """Write one exchange as the observe does, and file it as one drawer."""
        self.count += 1
        name = f"{TIMELINE}.{self.count:032x}.md"
        path = self.conversations / name
        text = _exchange_markdown(MemoryExchange(user=user, assistant=reply, scope=SCOPE))
        path.write_text(text, encoding="utf-8")
        if drawer:
            self.collection.add(
                f"drawer_{self.count}",
                text.strip(),
                {"wing": "conversations", "room": "general", "source_file": str(path)},
            )
        return name

    def offer(self):
        return _prune.offer(self.collection, self.palace, self.state)

    def apply(self, digest, groups, remove_latest=()):
        return _prune.apply(
            self.collection, self.palace, self.state, digest, list(groups), list(remove_latest)
        )

    def names(self) -> set[str]:
        return {p.name for p in self.conversations.iterdir()}

    def restore(self, name):
        return _prune.restore(self.collection, self.palace, self.state, name)

    def snapshot(self):
        files = {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.conversations.iterdir()
        }
        return files, dict(self.collection.drawers)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


def _check_ins(store: Store, count: int, start: int = 0) -> list[str]:
    return [
        store.file(*hourly(start + i, START + timedelta(hours=start + i))) for i in range(count)
    ]


def _later(store: Store) -> list[str]:
    return [
        store.file(f"{current_line(500 + i, START + timedelta(days=40 + i))}\n{task}", reply)
        for i, (task, reply) in enumerate(LATER)
    ]


# --- the selection rule -------------------------------------------------------------------------


def test_hourly_check_ins_are_selected_in_bulk_and_no_later_task_is(store):
    """The shape the issue counted: the bulk of the check-ins grouped, none of the later tasks."""
    hours = _check_ins(store, 30)
    later = _later(store)

    offer = store.offer()

    assert len(offer.groups) == 1
    (group,) = offer.groups
    members = {m.name for m in group.members}
    assert members == set(hours)
    assert not members & set(later)
    assert len(offer.survey.candidates) == 30 + len(LATER)  # all are tasks; rule 3 decides
    assert group.latest.name == hours[-1]
    assert len(group.copies) + 1 + len(group.exceptions) == 30


def test_every_activation_wording_opens_a_candidate():
    """The current line (from `_task_dialogue`) and every pre-0.155.0 shape the harness wrote."""
    when = START
    due = _stamp(when)
    shapes = [
        current_line(1, when),
        (
            f"[{due}] A task you scheduled has activated and is due now (task {_uuid(1)}, "
            f"scheduled for {due})."
        ),
        legacy_line(1, when),
        (
            f"A task you scheduled has activated and is due now (task {_uuid(1)}, "
            f"scheduled for {due}). Carry out its instructions:"
        ),
    ]
    for line in shapes:
        assert _prune._activation(line), line
        text = _exchange_markdown(
            MemoryExchange(user=f"{line}\nDo the thing.", assistant="Done.", scope=SCOPE)
        )
        judged = _prune.candidate("x.md", Path("x.md"), text)
        assert isinstance(judged, _prune.Exchange), line
        assert judged.instruction == "Do the thing." and judged.reply == "Done."
        assert judged.due == when


def test_the_line_is_matched_at_the_opening_only(store):
    """A conversation that quotes both wordings is a conversation, however often it repeats."""
    for i in range(8):
        when = START + timedelta(hours=i)
        quoting = (
            f"[{_stamp(when)}] john: the old line was '{legacy_line(i, when)}' and the new one is "
            f"'{current_line(i, when)}' — which reads better?"
        )
        store.file(quoting, "The new one reads better.")
        store.file(f"  {legacy_line(i, when)}\nCheck the board.", "Checked the board.")

    offer = store.offer()

    assert offer.groups == []
    assert offer.survey.candidates == []
    assert offer.survey.excluded[_prune.NOT_TASK] == 16


def test_a_peer_message_anywhere_in_the_incoming_half_excludes_the_exchange(store):
    for i in range(8):
        when = START + timedelta(hours=i)
        user, reply = hourly(i, when)
        store.file(f"{user}\n[{_stamp(when)}] john: also, are you around?", reply)

    offer = store.offer()

    assert offer.survey.excluded[_prune.SECOND_SPEAKER] == 8
    assert offer.groups == []


def test_alike_replies_under_different_instructions_are_not_grouped(store):
    for i, (task, _) in enumerate(LATER):
        store.file(f"{current_line(i, START + timedelta(hours=i))}\n{task}", "Done, nothing new.")

    assert store.offer().groups == []


def test_one_directional_overlap_is_not_a_copy():
    """A short reply whose words all sit inside a longer, different one is not its copy."""
    short = _prune.words("Done, nothing new.")
    long = _prune.words(
        "Done with the invoice audit: nothing new in the ledger, but the vendor sent two "
        "corrections and I filed both against the October close."
    )
    assert short <= long
    assert _prune.jaccard(short, long) < _prune.NEAR_COPY
    assert _prune.jaccard(short, long) == _prune.jaccard(long, short)


def test_a_short_reply_is_not_grouped_with_longer_different_ones(store):
    instruction = "Check the status board and report anything new."
    for i in range(4):
        store.file(
            f"{current_line(i, START + timedelta(hours=i))}\n{instruction}", "Done, nothing new."
        )
    for i in range(4, 8):
        store.file(
            f"{current_line(i, START + timedelta(hours=i))}\n{instruction}",
            f"Done: nothing new on the board, but ticket {['red', 'blue', 'gold', 'teal'][i - 4]} "
            "moved to review and the deploy window opened early for the router team.",
        )

    assert store.offer().groups == []


def test_a_stamped_hour_does_not_push_a_true_heartbeat_under_the_threshold():
    a = _prune.words(
        "Checked at 2026-07-01T10:00:00.000Z (10:00). 3 open, 4.5% done, id " + _uuid(1)
    )
    b = _prune.words(
        "Checked at 2026-07-02T23:59:59.999Z (23:59). 7 open, 9.25% done, id " + _uuid(2)
    )
    assert a == b


def test_a_chain_of_neighbours_is_not_a_template(store):
    """Each member a near-copy of the next, the ends far apart: no anchor holds five others."""
    vocabulary = [f"word{k}" for k in range(40)]
    for i in range(12):
        reply = " ".join(vocabulary[i : i + 10])
        store.file(f"{current_line(i, START + timedelta(hours=i))}\nRun the drift probe.", reply)
    offer = store.offer()
    exchanges = offer.survey.candidates
    assert _prune.near(exchanges[0], exchanges[1])
    assert not _prune.near(exchanges[0], exchanges[3])
    assert offer.groups == []


def test_groups_of_five_or_fewer_stay(store):
    _check_ins(store, 5)
    assert store.offer().groups == []
    report = (store.state / _prune.REPORT).read_text()
    assert "Nothing is offered" in report and "How to Say Yes" not in report


def test_six_is_a_group(store):
    _check_ins(store, 6)
    (group,) = store.offer().groups
    assert len(group.members) == 6


def test_a_member_with_a_residue_word_is_an_exception_and_survives_a_yes(store):
    names = _check_ins(store, 10)
    when = START + timedelta(hours=10)
    user, reply = hourly(10, when)
    odd = store.file(user, f"{reply} Disk alarm.")  # a near-copy, carrying two words no other does

    offer = store.offer()
    (group,) = offer.groups
    assert odd in group.exceptions
    assert {"disk", "alarm"} <= set(group.exceptions[odd])
    assert group.latest.name == names[-1]  # an exception is never the kept latest

    store.apply(offer.hash, [group.id], remove_latest=[group.id])

    assert odd in store.names()
    assert all(name not in store.names() for name in names)


# --- the yes ------------------------------------------------------------------------------------


def test_a_default_yes_keeps_the_latest_pure_copy(store):
    names = _check_ins(store, 8)
    later = _later(store)
    offer = store.offer()
    (group,) = offer.groups

    result = store.apply(offer.hash, [group.id])

    assert store.names() == {names[-1], *later}
    assert sorted(result.removed) == sorted(names[:-1])
    assert result.drawers == 7
    assert {f"drawer_{i}" for i in range(1, 8)} == set(store.collection.deleted)
    assert result.skipped == []


def test_the_explicit_option_removes_the_latest_too(store):
    names = _check_ins(store, 8)
    offer = store.offer()
    (group,) = offer.groups

    store.apply(offer.hash, [group.id], remove_latest=[group.id])

    assert not set(names) & store.names()


def test_a_yes_names_only_its_own_groups(store):
    first = _check_ins(store, 6)
    for i in range(6):
        when = START + timedelta(days=3, hours=i)
        store.file(
            f"{current_line(900 + i, when)}\nPost the weather for Dallas.",
            f"Posted: Dallas, {20 + i} degrees and clear.",
        )
    offer = store.offer()
    assert len(offer.groups) == 2
    hourly_group = next(g for g in offer.groups if first[0] in {m.name for m in g.members})

    store.apply(offer.hash, [hourly_group.id])

    remaining = store.names()
    assert not set(first[:-1]) & remaining
    assert len(remaining) == 1 + 6


@pytest.mark.parametrize(
    "case",
    ["stale_hash", "unknown_group", "no_groups", "latest_not_in_yes", "no_backup", "tampered_plan"],
)
def test_a_refused_apply_changes_nothing(store, case, monkeypatch):
    _check_ins(store, 8)
    offer = store.offer()
    (group,) = offer.groups
    before_files = store.names()
    before_drawers = dict(store.collection.drawers)
    digest, groups, latest = offer.hash, [group.id], []
    if case == "stale_hash":
        digest = "0" * 64
    elif case == "unknown_group":
        groups = [group.id, "g-0000000000"]
    elif case == "no_groups":
        groups = []
    elif case == "latest_not_in_yes":
        latest = ["g-0000000000"]
    elif case == "no_backup":

        def refuse(*_args, **_kwargs):
            raise OSError("No space left on device")

        monkeypatch.setattr(_prune, "_verify_backup", refuse)
    elif case == "tampered_plan":
        record = json.loads((store.state / _prune.PLAN).read_text())
        record["plan"]["groups"][0]["remove"].append(
            {"path": str(store.conversations / "elsewhere.md"), "sha256": "0" * 64}
        )
        (store.state / _prune.PLAN).write_text(json.dumps(record))

    with pytest.raises(_prune.Refused):
        store.apply(digest, groups, latest)

    assert store.names() == before_files
    assert store.collection.drawers == before_drawers
    assert _prune._backups(store.state) == []
    assert not any((store.state / _prune.BACKUPS).glob(".*")) if case == "no_backup" else True


def test_a_plan_from_a_report_since_overwritten_is_refused(store):
    _check_ins(store, 8)
    first = store.offer()
    _check_ins(store, 1, start=8)  # one more hour: a different plan
    second = store.offer()
    assert first.hash != second.hash

    with pytest.raises(_prune.Refused, match="not the plan in the last report"):
        store.apply(first.hash, [first.groups[0].id])


def test_a_file_changed_since_the_report_is_skipped_never_removed(store):
    names = _check_ins(store, 8)
    offer = store.offer()
    (group,) = offer.groups
    changed = store.conversations / names[2]
    changed.write_text(changed.read_text() + "One more line, added after the report.\n")
    gone = store.conversations / names[3]
    gone.unlink()

    result = store.apply(offer.hash, [group.id])

    assert (names[2], _prune.CHANGED) in result.skipped
    assert (names[3], _prune.GONE) in result.skipped
    assert changed.exists()
    assert "drawer_3" in store.collection.drawers  # the changed file's drawer stays with it
    assert len(result.removed) == 5


def test_a_drawer_with_no_source_file_is_reported_and_untouched(store):
    _check_ins(store, 8)
    user, reply = hourly(99, START)
    store.collection.add("orphan", f"> {user}\n{reply}", {"wing": "conversations"})

    offer = store.offer()
    assert offer.survey.no_source == 1
    store.apply(offer.hash, [offer.groups[0].id], remove_latest=[offer.groups[0].id])

    assert "orphan" in store.collection.drawers


def test_a_file_with_a_drawer_that_is_not_from_it_is_held(store):
    names = _check_ins(store, 8)
    foreign = store.conversations / names[0]
    store.collection.add(
        "foreign",
        "A drawer about the Lisbon offsite, recorded under the wrong file.",
        {"wing": "conversations", "source_file": str(foreign)},
    )

    offer = store.offer()
    (group,) = offer.groups

    assert offer.survey.foreign == 1
    assert names[0] not in {m.name for m in group.members}
    store.apply(offer.hash, [group.id], remove_latest=[group.id])
    assert foreign.exists()
    assert "foreign" in store.collection.drawers and "drawer_1" in store.collection.drawers


def test_a_file_whose_drawers_were_filed_under_a_moved_home_is_still_its_file(store):
    """Issue #606: a recorded path under the old home names the file where it is now."""
    names = _check_ins(store, 8)
    for drawer_id, (document, meta, embedding) in list(store.collection.drawers.items()):
        moved = dict(
            meta,
            source_file=f"/home/old-home/mempalace/conversations/{Path(meta['source_file']).name}",
        )
        store.collection.drawers[drawer_id] = (document, moved, embedding)

    offer = store.offer()
    (group,) = offer.groups
    result = store.apply(offer.hash, [group.id])

    assert len(result.removed) == 7
    assert store.names() == {names[-1]}


# --- the backup ---------------------------------------------------------------------------------


def test_the_backup_holds_every_removed_file_and_drawer_and_is_discarded_on_word(store):
    names = _check_ins(store, 8)
    originals = {name: (store.conversations / name).read_bytes() for name in names}
    offer = store.offer()
    result = store.apply(offer.hash, [offer.groups[0].id])

    backup = store.state / _prune.BACKUPS / result.backup
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    for name in result.removed:
        copy = backup / "files" / name
        assert copy.read_bytes() == originals[name]
        assert stat.S_IMODE(copy.stat().st_mode) == 0o600
    drawers = [json.loads(line) for line in (backup / "drawers.jsonl").read_text().splitlines()]
    assert {d["id"] for d in drawers} == {f"drawer_{i}" for i in range(1, 8)}
    assert all(d["embedding"] == [0.25, 0.5, 0.75] for d in drawers)
    manifest = json.loads((backup / "manifest.json").read_text())
    assert manifest["plan"] == offer.hash

    assert result.backup in store.offer().backups
    assert result.backup in (store.state / _prune.REPORT).read_text()

    _prune.discard_backup(store.state, result.backup)
    assert not backup.exists()
    with pytest.raises(_prune.Refused):
        _prune.discard_backup(store.state, result.backup)
    with pytest.raises(_prune.Refused):
        _prune.discard_backup(store.state, "../../home")


def test_a_restore_after_an_apply_gives_back_the_store_exactly(store):
    _check_ins(store, 8)
    _later(store)
    before = store.snapshot()
    offer = store.offer()
    applied = store.apply(offer.hash, [offer.groups[0].id], remove_latest=[offer.groups[0].id])
    assert store.snapshot() != before

    restored = store.restore(applied.backup)

    assert store.snapshot() == before  # bytes, mtimes, documents, metadata and embeddings
    assert sorted(restored.files) == sorted(applied.removed)
    assert restored.drawers == applied.drawers and restored.skipped == 0
    assert applied.backup in _prune._backups(store.state)  # a restore never discards it


def test_a_second_restore_changes_nothing(store):
    _check_ins(store, 8)
    offer = store.offer()
    applied = store.apply(offer.hash, [offer.groups[0].id])
    store.restore(applied.backup)
    after_first = store.snapshot()

    again = store.restore(applied.backup)

    assert store.snapshot() == after_first
    assert again.files == [] and again.drawers == 0
    assert sorted(again.skipped_files) == sorted(applied.removed)
    assert len(again.skipped_drawers) == applied.drawers


def test_a_file_recreated_in_between_is_skipped_and_reported(store):
    names = _check_ins(store, 8)
    offer = store.offer()
    applied = store.apply(offer.hash, [offer.groups[0].id])
    recreated = store.conversations / names[0]
    recreated.write_text("someone put a file here\n")

    restored = store.restore(applied.backup)

    assert restored.skipped_files == [names[0]]
    assert recreated.read_text() == "someone put a file here\n"
    assert "drawer_1" in store.collection.drawers  # its absent drawer still goes back
    assert len(restored.files) == len(applied.removed) - 1


def test_a_damaged_or_foreign_backup_restores_nothing(store, tmp_path):
    _check_ins(store, 8)
    offer = store.offer()
    applied = store.apply(offer.hash, [offer.groups[0].id])
    after = store.snapshot()
    backup = store.state / _prune.BACKUPS / applied.backup
    copy = next((backup / "files").iterdir())
    original = copy.read_bytes()
    copy.write_bytes(original + b"tampered")
    with pytest.raises(_prune.Refused, match="damaged"):
        store.restore(applied.backup)
    copy.write_bytes(original)
    other = Store(tmp_path / "other")
    other.state = store.state
    with pytest.raises(_prune.Refused, match="not of this memory store"):
        other.restore(applied.backup)
    with pytest.raises(_prune.Refused):
        store.restore("../backups")
    assert store.snapshot() == after


def test_the_report_and_the_output_show_each_backup_age(store, monkeypatch, capsys):
    _check_ins(store, 8)
    offer = store.offer()
    applied = store.apply(offer.hash, [offer.groups[0].id])
    old = "20260101T000000000000Z-ffffffffffff"
    (store.state / _prune.BACKUPS / old).mkdir()

    assert _main(store, monkeypatch) == 0

    out, _ = capsys.readouterr()
    report = (store.state / _prune.REPORT).read_text()
    for text in (out, report.replace("`", "")):
        assert f"{applied.backup}: written" in text and "less than a day ago" in text
        assert f"{old}: written 2026-01-01 00:00 UTC," in text and "days ago" in text


def test_the_cli_restores_a_backup(store, monkeypatch, capsys):
    _check_ins(store, 8)
    offer = store.offer()
    applied = store.apply(offer.hash, [offer.groups[0].id])
    capsys.readouterr()

    assert _main(store, monkeypatch, "--restore", applied.backup) == 0

    out, _ = capsys.readouterr()
    assert f"restored: {len(applied.removed)} file(s), {applied.drawers} drawer(s)" in out
    assert _main(store, monkeypatch, "--restore", applied.backup) == 0
    out, _ = capsys.readouterr()
    assert f"skipped: {applied.drawers} drawer(s) ({_prune.PRESENT})" in out
    assert out.count("skipped:") == len(applied.removed) + 1  # files by name, drawers as a count


def test_an_apply_that_dies_after_its_drawers_went_is_finished_by_the_same_plan(store):
    names = _check_ins(store, 8)
    offer = store.offer()
    (group,) = offer.groups
    # The state a kill between the two deletes leaves: the drawers gone, the file still there.
    store.collection.delete(ids=["drawer_1"])

    result = store.apply(offer.hash, [group.id])

    assert names[0] in result.removed
    assert names[0] not in store.names()


def test_a_half_written_backup_is_removed_by_the_next_run(store):
    _check_ins(store, 8)
    store.offer()
    partial = store.state / _prune.BACKUPS / ".20260701T000000000000Z-abc.partial"
    partial.mkdir(parents=True)
    (partial / "drawers.jsonl").write_text("{}\n")

    store.offer()

    assert not partial.exists()


# --- the report and what is printed -------------------------------------------------------------


def test_the_report_is_private_outside_the_palace_and_names_the_kept_file(store):
    names = _check_ins(store, 8)
    offer = store.offer()

    report = store.state / _prune.REPORT
    plan = store.state / _prune.PLAN
    for path in (report, plan):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert store.palace.resolve() not in path.resolve().parents
    assert stat.S_IMODE(store.state.stat().st_mode) == 0o700
    text = report.read_text()
    assert offer.hash in text and offer.groups[0].id in text
    assert f"`{names[-1]}`" in text  # the kept file, by name
    assert "## How to Say Yes" in text
    assert "Check the status board" in text  # one real example, for the agent


def test_the_plan_binds_every_file_by_path_and_content_hash(store):
    names = _check_ins(store, 8)
    offer = store.offer()
    (group,) = offer.plan["groups"]
    entries = [*group["remove"], group["latest"], *group["exceptions"]]
    assert sorted(Path(e["path"]).name for e in entries) == sorted(names)
    for entry in entries:
        data = Path(entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    assert offer.hash == _prune.plan_hash(offer.plan)

    assert store.offer().hash == offer.hash  # an unchanged store is the same plan
    target = store.conversations / names[4]
    target.write_text(target.read_text().replace("Nothing new", "Nothing new at all"))
    assert store.offer().hash != offer.hash


@pytest.mark.parametrize(
    "inside", ["mempalace", "mempalace/state", "mempalace/conversations/state", "x/conversations"]
)
def test_the_state_directory_is_never_where_the_miner_reads(store, inside):
    with pytest.raises(_prune.Refused):
        _prune.resolve_state(str(store.root / "home" / inside), store.palace)


def test_the_state_directory_defaults_into_the_agent_home(store, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", str(store.root / "home"))
    assert _prune.resolve_state(None, store.palace) == store.root / "home" / _prune.STATE_DIR
    monkeypatch.delenv("HARNESS_HOME")
    assert _prune.resolve_state(None, store.palace) == store.palace.parent / _prune.STATE_DIR


def _main(store, monkeypatch, *argv):
    monkeypatch.setattr(_prune, "open_palace", lambda _palace: store.collection)
    return _prune.main(["--palace", str(store.palace), "--state-dir", str(store.state), *argv])


def test_the_printed_output_carries_counts_ids_and_the_hash_never_drawer_text(
    store, monkeypatch, capsys
):
    _check_ins(store, 8)
    _later(store)

    assert _main(store, monkeypatch) == 0
    out, err = capsys.readouterr()
    offer = store.offer()

    assert offer.hash in out and offer.groups[0].id in out
    for text in (*(t for pair in LATER for t in pair), "status board", "Checked"):
        assert text not in out and text not in err


def test_the_cli_carries_out_a_yes_and_reports_what_it_skipped(store, monkeypatch, capsys):
    names = _check_ins(store, 8)
    assert _main(store, monkeypatch) == 0
    offer = store.offer()
    (store.conversations / names[1]).write_text("rewritten\n")
    capsys.readouterr()

    code = _main(
        store, monkeypatch, "--apply", "--plan", offer.hash, "--groups", offer.groups[0].id
    )

    out, _ = capsys.readouterr()
    assert code == 0
    assert "removed: 6 file(s), 6 drawer(s)" in out
    assert f"skipped: {names[1]} ({_prune.CHANGED})" in out


def test_the_cli_refuses_a_stale_plan(store, monkeypatch, capsys):
    _check_ins(store, 8)
    assert _main(store, monkeypatch) == 0
    before = store.names()

    code = _main(store, monkeypatch, "--apply", "--plan", "f" * 64, "--groups", "g-0")

    assert code == 1
    assert store.names() == before


def test_apply_flags_need_apply(store, monkeypatch):
    with pytest.raises(SystemExit):
        _main(store, monkeypatch, "--groups", "g-1")


def test_a_run_as_root_or_as_another_user_is_refused(store, monkeypatch, capsys):
    _check_ins(store, 6)
    owner = store.palace.stat().st_uid
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(Path, "stat", _owned_by(0))  # a palace root owns is still refused
    assert _main(store, monkeypatch) == 1
    assert "refusing to run as root" in capsys.readouterr().err
    monkeypatch.undo()
    monkeypatch.setattr(os, "geteuid", lambda: owner + 1)
    assert _main(store, monkeypatch) == 1
    assert "not owned by this user" in capsys.readouterr().err
    assert not (store.state / _prune.REPORT).exists()


def _owned_by(uid):
    real = Path.stat

    def stat_(self, *args, **kwargs):
        result = real(self, *args, **kwargs)
        return os.stat_result((*result[:4], uid, *result[5:]))

    return stat_


def test_there_is_one_store_per_run():
    """No option names more than one store: every option takes one value, or none."""
    actions = [a for a in _prune.build_parser()._actions if a.option_strings]
    assert all(a.nargs in (None, 0) for a in actions)
    assert all(type(a).__name__ not in {"_AppendAction", "_ExtendAction"} for a in actions)
