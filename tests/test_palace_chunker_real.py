"""MemPalace's ``convo_chunker_version`` on a real palace (issue #751).

MemPalace 3.11 added ``CONVO_CHUNKER_VERSION = 2`` and reads an exchange drawer that lacks the
field as filed by an older chunker: it drops out of the miner's mined set and content-hash map,
so a ``mempalace mine`` files its file again. 3.10 and earlier never wrote the field, so every
drawer a fleet palace holds today reads that way under 3.11. Measured on palaces built under
3.10.0 and mined under 3.11.0: in place, a mine deletes such a file's drawers and files it again
through its chunker (an exchange the observe filed as 3 drawers came back as 13); after a home
move it files a second copy beside the old drawers, which it keeps.

These tests build a palace with the installed MemPalace and then **age** it: every drawer and
registry row is rewritten without ``convo_chunker_version``, which is the field 3.10 never wrote
and the only one every mined-set decision here reads differently. Two differences remain and are
deliberate: 3.11's ``source_dir_ino`` is kept (only MemPalace's ``sync`` reads it), and the
miner's drawers were cut by 3.11's chunker, not 3.10's (no assertion here depends on how a miner
drawer was cut). The real 3.10.0-to-3.11.0 upgrade was measured separately, on issue #751; this
is the form CI can repeat, with one MemPalace installed per job.

Marked ``mempalace`` and excluded from the default run. On the floor job (a MemPalace with no
chunker version) only the first test runs; the rest are skipped there, by the version they need.
"""

from __future__ import annotations

import contextlib
import io
import re
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.mempalace

mempalace = pytest.importorskip("mempalace")

from mempalace import convo_miner  # noqa: E402
from mempalace import palace as mp  # noqa: E402

from basecradle_harness import _palace_check  # noqa: E402
from basecradle_harness._memory_provider import MemoryExchange, MemoryScope  # noqa: E402
from basecradle_harness._mempalace import (  # noqa: E402
    _REGISTRY_ROOM,
    MemPalaceMemoryProvider,
    _exchange_markdown,
    chunker_version,
    filed_by_observe,
)

TIMELINE = "019a3b2c-4d5e-7f60-8a1b-2c3d4e5f6a7b"
SCOPE = MemoryScope(agent="019a3b2c-4d5e-7f60-8a1b-000000000001", timeline=TIMELINE)
VERSION = getattr(mp, "CONVO_CHUNKER_VERSION", None)

needs_version = pytest.mark.skipif(
    VERSION is None, reason="this MemPalace defines no convo_chunker_version (before 3.11)"
)

#: Two files MemPalace's miner filed, the observe's path before 0.164.0.
MINED = {
    f"{TIMELINE}-mined-shed.md": (
        "[2026-09-01T10:00:00Z] john: what colour is the shed?\nIt was blue last spring.",
        "The shed is green now.",
    ),
    f"{TIMELINE}-mined-word.md": (
        "[2026-09-02T11:00:00Z] john: remember the code word: heliotrope.",
        "Noted: heliotrope.",
    ),
}

#: Three exchanges the observe filed, one long enough to take several drawers.
OBSERVED = [
    ("[2026-10-30T09:00:00Z] john: the boat is called Marigold.", "Marigold, got it."),
    ("[2026-10-30T09:05:00Z] john: short", "ok"),
    (
        "[2026-10-30T09:10:00Z] john: the runbook",
        "\n\n".join(f"Step {n}. " + "do it " * 40 for n in range(6)),
    ),
]

QUERIES = ["what colour is the shed", "code word heliotrope", "boat Marigold", "Step 4 do it"]


def _quiet(call, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return call(*args, **kwargs)


def _mine(palace: Path, **kwargs) -> None:
    _quiet(
        convo_miner.mine_convos,
        str(palace / "conversations"),
        str(palace),
        wing="conversations",
        agent="nova",
        extract_mode="exchange",
        **kwargs,
    )


def _build(palace: Path) -> MemPalaceMemoryProvider:
    """A palace holding both kinds of drawer a fleet palace holds: the miner's, then the observe's."""
    convos = palace / "conversations"
    convos.mkdir(parents=True)
    for name, (user, reply) in MINED.items():
        exchange = MemoryExchange(user=user, assistant=reply, scope=SCOPE)
        (convos / name).write_text(_exchange_markdown(exchange), encoding="utf-8")
    _mine(palace)
    provider = MemPalaceMemoryProvider(palace, agent="nova")
    for user, reply in OBSERVED:
        _quiet(provider.observe, MemoryExchange(user=user, assistant=reply, scope=SCOPE))
    return provider


def _age(palace: Path) -> None:
    """Rewrite every row without ``convo_chunker_version``, the field 3.10 never wrote.

    Delete and add again rather than update, because an update merges metadata and cannot drop a
    key. Embeddings, documents and every other field go back as they were.
    """
    collection = mp.get_collection(str(palace))
    page = collection.get(include=["documents", "metadatas", "embeddings"])
    metas = [
        {k: v for k, v in (meta or {}).items() if k != "convo_chunker_version"}
        for meta in page["metadatas"]
    ]
    collection.delete(ids=list(page["ids"]))
    collection.add(
        ids=list(page["ids"]),
        documents=list(page["documents"]),
        metadatas=metas,
        embeddings=[list(e) for e in page["embeddings"]],
    )


def _drawers(palace: Path) -> dict[str, tuple[str, dict]]:
    collection = mp.get_collection(str(palace), create=False, read_only=True)
    page = collection.get(include=["documents", "metadatas"])
    return {
        drawer_id: (text, meta or {})
        for drawer_id, text, meta in zip(page["ids"], page["documents"], page["metadatas"])
    }


def _census(palace: Path) -> _palace_check.Census:
    return _palace_check.census(
        mp.get_collection(str(palace), create=False, read_only=True), palace
    )


def _ours(palace: Path) -> set[str]:
    return {i for i, (text, meta) in _drawers(palace).items() if filed_by_observe(i, meta, text)}


def _recall(provider: MemPalaceMemoryProvider) -> list[list[str]]:
    return [[hit.get("drawer_id") for hit in provider.search(query, 5)] for query in QUERIES]


def _report(capsys, palace: Path, *flags: str) -> str:
    capsys.readouterr()
    _palace_check.main([str(palace.parent), *flags])
    return capsys.readouterr().out


@pytest.fixture
def palace(tmp_path) -> Path:
    # `_palace_path(<HARNESS_HOME>)` is `<HARNESS_HOME>/mempalace`, which `_report` relies on.
    return tmp_path / "home" / "mempalace"


@pytest.mark.skipif(VERSION is not None, reason="this MemPalace defines a convo_chunker_version")
def test_without_a_chunker_version_nothing_is_stale_and_nothing_is_stamped(palace, capsys):
    """The floor (3.9, 3.10): the class does not exist, and the stamp is a no-op that says so."""
    _build(palace)
    before = _drawers(palace)

    out = _report(capsys, palace, "--stamp-chunker-version")

    assert chunker_version() is None
    assert "nothing to stamp: MemPalace" in out and "defines no convo_chunker_version" in out
    assert (
        "conversation files version-stale: 0 (observe-filed 0, miner-filed 0), "
        "this MemPalace defines no convo_chunker_version"
    ) in out
    assert _drawers(palace) == before


@needs_version
def test_a_fresh_observe_files_current_drawers(palace):
    """On 3.11 and later every drawer the observe files carries the installed constant.

    Both writers stamp it: 3.11's ``file_conversation_exchange`` itself, and the harness through
    ``extra_metadata``, where upstream's value wins. So this passes on either alone; that the
    harness stamps it, from the library's constant, is pinned offline in ``test_mempalace.py``.
    """
    _build(palace)

    stamped = [meta for i, (_, meta) in _drawers(palace).items() if i in _ours(palace)]

    assert chunker_version() == VERSION
    assert len(stamped) > len(OBSERVED)  # the runbook took more than one drawer
    assert {meta["convo_chunker_version"] for meta in stamped} == {VERSION}
    found = _census(palace)
    assert (found.stale_files_ours, found.stale_files_theirs) == (0, 0)


@needs_version
def test_the_observes_drawers_are_told_from_the_miners_by_their_id(palace, tmp_path):
    """The id recipe decides, never the content: identical text filed both ways is told apart."""
    _build(palace)
    moved = tmp_path / "moved" / "mempalace"
    shutil.copytree(palace, moved)
    _mine(moved)  # writes the registry rows a move leaves, so they are in the census too

    drawers = _drawers(moved)
    by_file: dict[str, set[bool]] = {}
    for drawer_id, (text, meta) in drawers.items():
        name = Path(meta["source_file"]).name
        if meta.get("room") == _REGISTRY_ROOM:
            assert not filed_by_observe(drawer_id, meta, text)
            continue
        by_file.setdefault(name, set()).add(filed_by_observe(drawer_id, meta, text))

    assert {name for name, verdict in by_file.items() if verdict == {False}} == set(MINED)
    assert sum(verdict == {True} for verdict in by_file.values()) == len(OBSERVED)
    # The observe's own text and metadata under any other id are not the observe's.
    drawer_id, (text, meta) = next(
        (i, (t, m)) for i, (t, m) in drawers.items() if filed_by_observe(i, m, t)
    )
    assert not filed_by_observe(f"drawer_conversations_{meta['room']}_{'0' * 24}", meta, text)
    # Nor is a drawer missing a field only the observe stamps.
    assert not filed_by_observe(drawer_id, {**meta, "chunk_total": None}, text)


@needs_version
def test_a_stale_registry_row_does_not_make_the_observes_file_the_miners(palace, tmp_path):
    """A home moved and mined under 3.10 holds a registry row per file at the new path.

    3.10's registry rows lack the field too, so under 3.11 they are version-stale like the drawers.
    They are MemPalace's bookkeeping, not a drawer anybody filed, so they never decide whose a
    file is: the observe's files still count as the observe's, and the stamp still applies.
    """
    _build(palace)
    moved = tmp_path / "moved" / "mempalace"
    shutil.copytree(palace, moved)
    _mine(moved)  # a registry row at the new path for every file
    _age(moved)

    found = _census(moved)

    assert found.registry == len(OBSERVED) + len(MINED)
    assert (found.stale_files_ours, found.stale_files_theirs) == (len(OBSERVED), len(MINED))


@needs_version
def test_an_aged_palace_reports_its_version_stale_files_in_place(palace, capsys):
    """In place nothing is unregistered, so before #751 the check printed no forecast at all.

    The NOC's rename gate then fell back to "not yet registered at this location: 0" and passed,
    while a mine would have re-chunked every file.
    """
    _build(palace)
    _age(palace)

    out = _report(capsys, palace)

    assert "not yet registered at this location: 0" in out
    assert (
        f"conversation files version-stale: {len(OBSERVED) + len(MINED)} "
        f"(observe-filed {len(OBSERVED)}, miner-filed {len(MINED)}), "
        f"convo_chunker_version older than MemPalace's {VERSION}"
    ) in out
    assert _palace_check.STALE_CONSEQUENCE in out
    assert (
        f"next observe (dry run): 0 files -> registry sentinel, "
        f"{len(OBSERVED) + len(MINED)} files -> mined as new drawers"
    ) in out
    reasons = [line for line in out.splitlines() if line.strip().startswith("reason:")]
    assert len(reasons) == len(OBSERVED) + len(MINED)
    assert all("version-stale" in r and "deletes its drawers here" in r for r in reasons)
    assert sum("--stamp-chunker-version makes it current" in r for r in reasons) == len(OBSERVED)
    assert sum("left to MemPalace" in r for r in reasons) == len(MINED)


@needs_version
def test_the_stamp_makes_the_observes_drawers_current_and_touches_nothing_else(palace, capsys):
    provider = _build(palace)
    _age(palace)
    aged = _drawers(palace)
    ours = _ours(palace)
    recall = _recall(provider)

    dry = _report(capsys, palace, "--stamp-chunker-version", "--dry-run")
    assert _drawers(palace) == aged  # a dry run writes nothing
    assert (
        f"stamp chunker version (dry run): drawers to stamp {len(ours)} in {len(OBSERVED)} files, "
        f"convo_chunker_version {VERSION}; miner-filed version-stale drawers left to MemPalace "
        f"{len(aged) - len(ours)}"
    ) in dry

    out = _report(capsys, palace, "--stamp-chunker-version")
    assert (
        f"stamp chunker version: WROTE to this palace: drawers stamped {len(ours)} in "
        f"{len(OBSERVED)} files"
    ) in out
    stamped = _drawers(palace)
    assert set(stamped) == set(aged)
    for drawer_id, (text, meta) in aged.items():
        expected = {**meta, "convo_chunker_version": VERSION} if drawer_id in ours else meta
        assert stamped[drawer_id] == (text, expected)
    assert (
        f"conversation files version-stale: {len(MINED)} (observe-filed 0, miner-filed "
        f"{len(MINED)})"
    ) in out
    assert _recall(provider) == recall  # metadata only: recall is unchanged

    again = _report(capsys, palace, "--stamp-chunker-version")
    assert "drawers stamped 0 in 0 files" in again
    assert _drawers(palace) == stamped

    # A real mine in place now leaves every observe drawer exactly as it was, and files only the
    # miner's files, which MemPalace re-chunks: upstream's own migration, not the harness's.
    _mine(palace)
    mined = _drawers(palace)
    assert {i: mined[i] for i in ours} == {i: stamped[i] for i in ours}
    refiled = {Path(meta["source_file"]).name for i, (_, meta) in mined.items() if i not in ours}
    assert refiled == set(MINED)


@needs_version
def test_after_a_move_the_stamped_observe_files_are_recognised_and_the_miners_are_not(
    palace, tmp_path, capsys
):
    """The rename's case: the observe's files register by content, no drawer filed for them."""
    _build(palace)
    _age(palace)
    _report(capsys, palace, "--stamp-chunker-version")
    moved = tmp_path / "moved" / "mempalace"
    shutil.copytree(palace, moved)

    out = _report(capsys, moved)

    assert (
        f"next observe (dry run): {len(OBSERVED)} files -> registry sentinel, "
        f"{len(MINED)} files -> mined as new drawers"
    ) in out
    reasons = [line for line in out.splitlines() if line.strip().startswith("reason:")]
    assert len(reasons) == len(MINED)
    assert all("beside its old drawers, which it keeps" in r for r in reasons)
    assert all("left to MemPalace" in r for r in reasons)


@needs_version
def test_a_file_stale_for_both_reasons_is_named_version_stale(palace, capsys):
    """One drawer of a set also at an old ``normalize_version``: the class is still named.

    Before the fix the reason fell through to "a reason this check cannot name" while the summary
    line counted the file as version-stale, so the two lines disagreed.
    """
    _build(palace)
    _age(palace)
    collection = mp.get_collection(str(palace))
    runbook = [
        (drawer_id, meta)
        for drawer_id, (_, meta) in _drawers(palace).items()
        if (meta.get("chunk_total") or 0) > 1
    ]
    drawer_id, meta = runbook[0]
    collection.update(ids=[drawer_id], metadatas=[{**meta, "normalize_version": 1}])

    out = _report(capsys, palace)

    reasons = [line for line in out.splitlines() if line.strip().startswith("reason:")]
    assert not any("cannot name" in r for r in reasons)
    assert sum("the observe filed it" in r for r in reasons) == len(OBSERVED)


# ---------------------------------------------------------------------------------------------
# --relocate-source-file (issue #831)
# ---------------------------------------------------------------------------------------------

LONG = f"{TIMELINE}-mined-long.md"


@pytest.fixture
def moved(tmp_path, palace) -> Path:
    """The previous ``HARNESS_HOME`` of a palace built there, moved to ``palace``, then aged.

    The shape a rename left on the fleet: every drawer recorded under the old home, the old home
    gone, and a registry row at the new path for each file a 3.10 mine recognised by content.
    Aged after the rows are written, so they read as stale the way 3.10's do. Returns the previous
    home, for ``--previous-home``.
    """
    previous = tmp_path / "old" / "harness"
    old_palace = previous / "mempalace"
    _build(old_palace)
    exchange = MemoryExchange(
        user="[2026-09-03T12:00:00Z] john: the long one",
        assistant="\n\n".join(f"Part {n}. " + "more of it " * 30 for n in range(8)),
        scope=SCOPE,
    )
    (old_palace / "conversations" / LONG).write_text(_exchange_markdown(exchange), encoding="utf-8")
    _mine(old_palace)
    palace.parent.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(previous), str(palace.parent))
    collection = mp.get_collection(str(palace))
    for name in (*MINED, LONG):
        path = str(palace / "conversations" / name)
        convo_miner._register_file(collection, path, "conversations", "nova", "exchange")
    _age(palace)
    return previous


def _relocate(capsys, palace: Path, previous: Path, *flags: str) -> str:
    return _report(
        capsys, palace, "--relocate-source-file", "--previous-home", str(previous), *flags
    )


def _mined_ids(palace: Path) -> set[str]:
    return {
        drawer_id
        for drawer_id, (_, meta) in _drawers(palace).items()
        if Path(str(meta.get("source_file"))).name in (*MINED, LONG)
        and meta.get("room") != _REGISTRY_ROOM
    }


def _planned(palace: Path) -> int:
    """The drawers MemPalace's own dry-run mine says it will file."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        convo_miner.mine_convos(
            str(palace / "conversations"),
            str(palace),
            wing="conversations",
            agent="nova",
            extract_mode="exchange",
            dry_run=True,
        )
    return sum(int(n) for n in re.findall(r"\((\d+) drawers\)", out.getvalue()))


@needs_version
def test_a_relocated_palace_is_re_mined_in_place_with_no_duplicate(moved, palace, capsys):
    """The issue's definition of done, on a real palace: drawers = planned, duplicates 0."""
    _report(capsys, palace, "--stamp-chunker-version")
    aged = _drawers(palace)
    mined = _mined_ids(palace)
    here = str((palace / "conversations").resolve())
    observed = {i for i in aged if i not in mined and aged[i][1].get("room") != _REGISTRY_ROOM}
    assert mined and observed
    recall = _recall(MemPalaceMemoryProvider(palace, agent="nova"))

    dry = _relocate(capsys, palace, moved, "--dry-run")
    assert _drawers(palace) == aged  # a dry run writes nothing
    assert (
        f"relocate source_file (dry run): drawers to relocate {len(mined)} in 3 files, "
        f"files left alone 0, drawers left at another location {len(observed)}"
    ) in dry

    out = _relocate(capsys, palace, moved)
    assert (
        f"relocate source_file: WROTE to this palace: drawers relocated {len(mined)} in 3 files, "
        f"files left alone 0, drawers left at another location {len(observed)}"
    ) in out

    relocated = _drawers(palace)
    assert set(relocated) == set(aged)  # nothing is deleted, nothing is added
    for drawer_id, (text, meta) in aged.items():
        if drawer_id in mined:
            name = Path(str(meta["source_file"])).name
            assert relocated[drawer_id] == (text, {**meta, "source_file": f"{here}/{name}"})
        else:
            assert relocated[drawer_id] == (text, meta)  # registry rows and the observe's
    assert _recall(MemPalaceMemoryProvider(palace, agent="nova")) == recall

    again = _relocate(capsys, palace, moved)
    assert "drawers relocated 0 in 0 files" in again
    assert _drawers(palace) == relocated

    planned = _planned(palace)
    assert planned
    _mine(palace)
    after = _drawers(palace)
    filed = {
        i: entry
        for i, entry in after.items()
        if Path(str(entry[1].get("source_file"))).name in (*MINED, LONG)
    }
    assert len(filed) == planned  # the mine's plan, and no drawer beside it
    assert {str(entry[1]["source_file"]).rsplit("/", 1)[0] for entry in filed.values()} == {here}
    assert not any(entry[1].get("room") == _REGISTRY_ROOM for entry in filed.values())
    assert {i: after[i] for i in observed} == {i: relocated[i] for i in observed}

    _mine(palace)  # a second mine files nothing
    assert _drawers(palace) == after
    for query_hits in _recall(MemPalaceMemoryProvider(palace, agent="nova")):
        assert query_hits


def _one_drawer(palace: Path, name: str) -> tuple[str, dict]:
    return next(
        (i, meta)
        for i, (_, meta) in _drawers(palace).items()
        if Path(str(meta.get("source_file"))).name == name and meta.get("room") != _REGISTRY_ROOM
    )


def _still_exists(palace, moved):
    path = moved / "mempalace" / "conversations" / LONG
    path.parent.mkdir(parents=True)
    path.write_text("still here", encoding="utf-8")


def _other_wing(palace, moved):
    drawer_id, meta = _one_drawer(palace, LONG)
    mp.get_collection(str(palace)).update(ids=[drawer_id], metadatas=[{**meta, "wing": "notes"}])


def _also_here(palace, moved):
    drawer_id, meta = _one_drawer(palace, LONG)
    path = str(palace / "conversations" / LONG)
    mp.get_collection(str(palace)).update(
        ids=[drawer_id], metadatas=[{**meta, "source_file": path}]
    )


def _third_location(palace, moved):
    drawer_id, meta = _one_drawer(palace, LONG)
    mp.get_collection(str(palace)).update(
        ids=[drawer_id],
        metadatas=[{**meta, "source_file": f"/somewhere/else/conversations/{LONG}"}],
    )


def _one_current(palace, moved):
    drawer_id, meta = _one_drawer(palace, LONG)
    mp.get_collection(str(palace)).update(
        ids=[drawer_id], metadatas=[{**meta, "convo_chunker_version": VERSION}]
    )


def _no_hash(palace, moved):
    rows = [
        (i, meta)
        for i, (_, meta) in _drawers(palace).items()
        if Path(str(meta.get("source_file"))).name == LONG and meta.get("room") != _REGISTRY_ROOM
    ]
    mp.get_collection(str(palace)).update(
        ids=[i for i, _ in rows], metadatas=[{**meta, "content_hash": ""} for _, meta in rows]
    )


def _changed_here(palace, moved):
    (palace / "conversations" / LONG).write_text("> something else\n\nentirely\n", encoding="utf-8")


def _gone_here(palace, moved):
    (palace / "conversations" / LONG).unlink()


@needs_version
@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (_still_exists, "its recorded path still exists, so it was not moved"),
        (_other_wing, "it has drawers in wing conversations, notes, not only 'conversations'"),
        (_also_here, "it has drawers recorded somewhere other than the previous home"),
        (_third_location, "it has drawers recorded somewhere other than the previous home"),
        (_one_current, "not every drawer is version-stale, so a mine would not purge them"),
        (_no_hash, "its drawers carry no content hash (filed before MemPalace 3.7)"),
        (_changed_here, "its content hash is not the one its drawers record"),
        (_gone_here, "no such file at this location, so a mine never reads it"),
    ],
)
def test_a_file_that_cannot_prove_it_moved_is_left_alone_whole(
    moved, palace, capsys, mutate, reason
):
    _report(capsys, palace, "--stamp-chunker-version")
    assert len([i for i in _mined_ids(palace) if LONG in _drawers(palace)[i][1]["source_file"]]) > 1
    mutate(palace, moved)
    before = _drawers(palace)

    out = _relocate(capsys, palace, moved)

    assert f"  left alone: {LONG}: {reason}" in out
    assert "drawers relocated" in out and f"in {len(MINED)} files, files left alone 1" in out
    after = _drawers(palace)
    for drawer_id, (_, meta) in before.items():
        if Path(str(meta.get("source_file"))).name == LONG:
            assert after[drawer_id] == before[drawer_id]  # the whole file, not only the odd drawer


@needs_version
def test_an_observe_filed_file_is_left_alone_because_its_id_is_derived_from_the_path(
    moved, palace, capsys
):
    """Unstamped here, so the observe's drawers are version-stale and reach the check."""
    before = _drawers(palace)
    ours = _ours(palace)
    assert ours

    out = _relocate(capsys, palace, moved)

    names = {Path(str(before[i][1]["source_file"])).name for i in ours}
    for name in names:
        assert (
            f"  left alone: {name}: the observe filed it; --stamp-chunker-version makes it current"
        ) in out
    assert _ours(palace) == ours  # the id recipe still holds of every one of them


@needs_version
def test_a_previous_home_no_drawer_names_relocates_nothing(moved, palace, tmp_path, capsys):
    before = _drawers(palace)

    out = _relocate(capsys, palace, tmp_path / "never" / "lived" / "here")

    assert "drawers relocated 0 in 0 files, files left alone 0" in out
    assert _drawers(palace) == before


@needs_version
def test_a_running_mine_refuses_the_write(moved, palace, capsys, monkeypatch):
    def held(path):
        raise mp.MineAlreadyRunning(path)

    monkeypatch.setattr(mp, "mine_palace_lock", held)
    before = _drawers(palace)

    code = _palace_check.main(
        [str(palace.parent), "--relocate-source-file", "--previous-home", str(moved)]
    )

    assert code == 1
    assert "refused, nothing written: a mine holds this palace" in capsys.readouterr().out
    monkeypatch.undo()
    assert _drawers(palace) == before


@pytest.mark.skipif(VERSION is not None, reason="this MemPalace defines a convo_chunker_version")
def test_without_a_chunker_version_there_is_nothing_to_relocate(palace, tmp_path, capsys):
    _build(palace)

    out = _relocate(capsys, palace, tmp_path / "old" / "harness")

    assert "nothing to relocate: MemPalace" in out
