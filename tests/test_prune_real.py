"""`basecradle-harness-prune-palace` on a real MemPalace palace (issue #973).

`test_prune.py` pins the rule and the apply against a fake collection. What a fake cannot say is
what MemPalace does: whether a drawer its miner cut (the observe's path before 0.164.0) still reads
as a piece of its file, whether a mine of the palace after a dry run files the report, whether the
store reads back right after an apply, and whether the backup really restores what went.

Marked ``mempalace`` and excluded from the default run. Run it with ``uv run --extra mempalace
pytest -m mempalace``; `conftest` gives it a throwaway ``HOME``.
"""

from __future__ import annotations

import contextlib
import io
from datetime import timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.mempalace

mempalace = pytest.importorskip("mempalace")

from mempalace import convo_miner  # noqa: E402
from mempalace import palace as mp  # noqa: E402

from basecradle_harness import _prune  # noqa: E402
from basecradle_harness._memory_provider import MemoryExchange, MemoryScope  # noqa: E402
from basecradle_harness._mempalace import _REGISTRY_ROOM, MemPalaceMemoryProvider  # noqa: E402
from basecradle_harness._scrub import open_palace  # noqa: E402
from tests.test_prune import LATER, START, TIMELINE, current_line, hourly  # noqa: E402

SCOPE = MemoryScope(agent="nova", timeline=TIMELINE)


def _quiet(call, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return call(*args, **kwargs)


def _drawers(palace: Path) -> dict[str, tuple[str, str]]:
    """Every real drawer as ``id -> (document, source file name)``."""
    collection = mp.get_collection(str(palace), create=False)
    page = collection.get(include=["documents", "metadatas"])
    return {
        drawer_id: (document, Path(meta.get("source_file", "")).name)
        for drawer_id, document, meta in zip(page["ids"], page["documents"], page["metadatas"])
        if (meta or {}).get("room") != _REGISTRY_ROOM
    }


@pytest.fixture
def palace(tmp_path):
    """Hourly check-ins filed both ways the harness ever filed them, beside later, distinct tasks.

    Eight in the pre-0.155.0 wording, written as files and filed by MemPalace's own miner (the
    observe's path before 0.164.0); ten in today's wording, filed by today's observe; and the later
    tasks, also by the observe.
    """
    palace = tmp_path / "home" / "mempalace"
    provider = MemPalaceMemoryProvider(palace, agent="nova")
    for i, (task, reply) in enumerate(LATER):
        when = START + timedelta(days=40 + i)
        _quiet(
            provider.observe,
            MemoryExchange(
                user=f"{current_line(500 + i, when)}\n{task}", assistant=reply, scope=SCOPE
            ),
        )
    conversations = palace / "conversations"
    for i in range(8):
        user, reply = hourly(i, START + timedelta(hours=i))
        quoted = "\n".join(f"> {line}" for line in user.splitlines())
        (conversations / f"{i:032x}.md").write_text(f"{quoted}\n{reply}\n", encoding="utf-8")
    _quiet(
        convo_miner.mine_convos,
        str(conversations),
        str(palace),
        wing="conversations",
        agent="nova",
        extract_mode="exchange",
    )
    for i in range(8, 18):
        when = START + timedelta(hours=i)
        _, reply = hourly(i, when)
        instruction = "Check the status board and report anything new since the last check at "
        _quiet(
            provider.observe,
            MemoryExchange(
                user=f"{current_line(i, when)}\n{instruction}{when:%H:%M}.",
                assistant=reply,
                scope=SCOPE,
            ),
        )
    return palace


def test_a_real_store_is_offered_its_check_ins_and_never_mines_the_report(palace):
    state = palace.parent / _prune.STATE_DIR
    before = _drawers(palace)

    offer = _prune.offer(open_palace(palace), palace, state)

    assert offer.survey.foreign == 0  # the miner's drawers read as pieces of their files
    (group,) = offer.groups
    assert len(group.members) == 18
    later = {name for _, name in before.values()} - {m.name for m in group.members}
    assert len(later) == len(LATER)

    # A mine of the palace after the dry run, as `mempalace mine` would run it, and one more
    # observe: neither files the report or the plan.
    _quiet(
        convo_miner.mine_convos,
        str(palace / "conversations"),
        str(palace),
        wing="conversations",
        agent="nova",
        extract_mode="exchange",
    )
    provider = MemPalaceMemoryProvider(palace, agent="nova")
    _quiet(
        provider.observe,
        MemoryExchange(user="[2026-11-01T00:00:00Z] john: hi", assistant="Hello.", scope=SCOPE),
    )
    after = _drawers(palace)
    assert len(after) == len(before) + 1
    report = (state / _prune.REPORT).read_text()
    for document, _ in after.values():
        assert _prune.REPORT_TITLE not in document
        assert offer.hash not in document
    assert _prune.REPORT_TITLE in report


def test_an_apply_on_a_real_store_reads_back_and_its_backup_restores_it(palace):
    state = palace.parent / _prune.STATE_DIR
    collection = open_palace(palace)
    before = _drawers(palace)
    offer = _prune.offer(collection, palace, state)
    (group,) = offer.groups

    result = _prune.apply(collection, palace, state, offer.hash, [group.id], [])

    after = _drawers(palace)
    removed = {m.name for m in group.copies}
    assert set(result.removed) == removed and result.skipped == []
    assert not any((palace / "conversations" / name).exists() for name in removed)
    assert not {name for _, name in after.values()} & removed
    assert group.latest.name in {name for _, name in after.values()}
    assert len(before) - len(after) == result.drawers
    hits = MemPalaceMemoryProvider(palace, agent="nova").search("vendor invoice spend log", 5)
    assert any("invoice" in hit["text"] for hit in hits)

    # The backup puts back exactly what went, and a mine afterwards files nothing again.
    restored = _prune.restore(open_palace(palace), palace, state, result.backup)
    assert sorted(restored.files) == sorted(result.removed) and restored.skipped == 0
    assert _drawers(palace) == before
    _quiet(
        convo_miner.mine_convos,
        str(palace / "conversations"),
        str(palace),
        wing="conversations",
        agent="nova",
        extract_mode="exchange",
    )
    assert _drawers(palace) == before
    again = _prune.restore(open_palace(palace), palace, state, result.backup)
    assert again.files == [] and again.drawers == 0
    assert _drawers(palace) == before
    assert _prune.offer(open_palace(palace), palace, state).hash == offer.hash
