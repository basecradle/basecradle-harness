"""`basecradle-harness-scrub-palace`: find harness scaffolding in a mined palace, then delete it.

MemPalace is an optional extra and not installed in the test env, so its library is faked at the
``sys.modules`` boundary exactly as `tests/test_mempalace.py` does — here the fake is a small
in-memory drawer collection with the `get`/`delete` surface upstream's backend protocol defines
(`mempalace.backends.base`).

The palace under test is **synthetic and polluted on purpose**: real mined exchanges beside
copies of several distinct scaffolding classes (the legacy recall heading, the fence, the
compaction prompt, the canned stuck note, the charter), plus the mixed chunk that must survive.
The claim these tests make is the one the founder's staging gate turns on — *apply removes only
the scaffolding, and the dry run predicted exactly what apply did*.
"""

import sys
import types

import pytest

from basecradle_harness import _scrub
from basecradle_harness._mempalace import _CLOSE_TAG, _INJECTED_HEADING, _OPEN_TAG
from basecradle_harness._mining import _LEGACY_RECALL_HEADING as LEGACY_HEADING
from basecradle_harness._mining import Verdict, classify
from basecradle_harness._wake import _COMPACTION_OBSERVE_NOTE, _STUCK_NOTE

# --- real memories, for the "provably intact" half of the gate ---------------------------------

REAL = {
    "real-1": "> [2026-08-02] john: John lives in Dallas.\nNoted — Dallas, Texas.",
    "real-2": (
        "> [2026-08-17] origin: the staging endpoint moved to eu-west\n"
        "Understood. I'll use eu-west from now on."
    ),
    # The trap: a real memory that talks *about* memory, tools and the brief. It must be kept.
    "real-3": (
        "> [2026-08-20] john: do you actually read the tool manifest in your brief?\n"
        "I do — the manifest lists my active tools, and my memory recalls past conversations."
    ),
}

# --- scaffolding, one drawer per class ---------------------------------------------------------

POLLUTED = {
    "junk-legacy": LEGACY_HEADING,
    "junk-fenced": f"{_INJECTED_HEADING}\n{_OPEN_TAG}\n{_CLOSE_TAG}",
    "junk-compaction": _COMPACTION_OBSERVE_NOTE,
    "junk-stuck": _STUCK_NOTE,
}

# Scaffolding *beside* a real memory. Reported for review, deleted by nothing.
MIXED = {"mixed-1": f"{LEGACY_HEADING}\n- John's birthday is 3 March."}


class FakeCollection:
    """The drawer collection surface the scrub uses: paged `get`, `delete` by id, `count`."""

    def __init__(self, drawers):
        #: ``id -> (document, metadata)``, insertion-ordered like a real paged scan.
        self.drawers = dict(drawers)
        self.deleted: list[list[str]] = []

    def get(self, *, limit=None, offset=0, include=None, ids=None, where=None):
        items = list(self.drawers.items())[offset : (offset + limit) if limit else None]
        return {
            "ids": [drawer_id for drawer_id, _ in items],
            "documents": [payload[0] for _, payload in items],
            "metadatas": [payload[1] for _, payload in items],
        }

    def delete(self, *, ids=None, where=None):
        self.deleted.append(list(ids or []))
        for drawer_id in ids or []:
            self.drawers.pop(drawer_id, None)

    def count(self):
        return len(self.drawers)


@pytest.fixture
def palace(tmp_path):
    """A palace directory with a `conversations/` dir holding one source file per drawer."""
    root = tmp_path / "mempalace"
    (root / "conversations").mkdir(parents=True)
    drawers = {}
    for drawer_id, text in {**REAL, **POLLUTED, **MIXED}.items():
        source = root / "conversations" / f"{drawer_id}.md"
        source.write_text(text, encoding="utf-8")
        drawers[drawer_id] = (text, {"source_file": str(source), "wing": "conversations"})
    return root, FakeCollection(drawers)


@pytest.fixture
def fake_mempalace(monkeypatch):
    """Install a fake ``mempalace.palace`` whose `get_collection` returns the collection under test."""
    module = types.ModuleType("mempalace.palace")
    module.collection = None
    module.calls = []

    def get_collection(path, **kwargs):
        module.calls.append((path, kwargs))
        return module.collection

    module.get_collection = get_collection
    parent = types.ModuleType("mempalace")
    parent.palace = module
    monkeypatch.setitem(sys.modules, "mempalace", parent)
    monkeypatch.setitem(sys.modules, "mempalace.palace", module)
    return module


# === the classification, end to end over a polluted palace =====================================


def test_a_dry_run_finds_every_scaffolding_class_and_deletes_nothing(palace):
    root, collection = palace

    report = _scrub.scan(collection, root)

    assert report.total == len(REAL) + len(POLLUTED) + len(MIXED)
    assert {f.drawer_id for f in report.scrub} == set(POLLUTED)
    assert {f.drawer_id for f in report.review} == set(MIXED)
    assert report.kept == len(REAL)
    # A dry run is a read: the palace is untouched, and the files are still on disk.
    assert collection.deleted == []
    assert set(collection.drawers) == set(REAL) | set(POLLUTED) | set(MIXED)
    assert len(list((root / "conversations").iterdir())) == report.total


def test_apply_removes_exactly_what_the_dry_run_reported(palace):
    """The staging gate's whole claim: apply's effect equals the dry run's report."""
    root, collection = palace
    report = _scrub.scan(collection, root)

    deleted, removed = _scrub.apply(collection, report)

    assert deleted == len(POLLUTED)
    assert removed == len(POLLUTED)
    assert set(collection.drawers) == set(REAL) | set(MIXED)
    # Re-scanning finds nothing left to scrub, and the real memories are still there, verbatim.
    after = _scrub.scan(collection, root)
    assert after.scrub == []
    assert after.total == len(REAL) + len(MIXED)
    for drawer_id, text in REAL.items():
        assert collection.drawers[drawer_id][0] == text


def test_a_mixed_chunk_is_never_deleted(palace):
    """A chunk holding a memory survives, whatever scaffolding is stapled to it."""
    root, collection = palace
    report = _scrub.scan(collection, root)

    _scrub.apply(collection, report)

    assert "mixed-1" in collection.drawers
    assert "birthday is 3 March" in collection.drawers["mixed-1"][0]


def test_the_source_file_goes_with_the_drawer_so_the_next_wake_cannot_re_mine_it(palace):
    """The re-pollution guard. MemPalace re-mines a file whose drawers are gone or incomplete."""
    root, collection = palace
    report = _scrub.scan(collection, root)

    _scrub.apply(collection, report)

    remaining = {path.stem for path in (root / "conversations").iterdir()}
    assert remaining == set(REAL) | set(MIXED)


def test_a_source_file_outside_the_palace_is_never_unlinked(tmp_path):
    """A path off a metadata field must not be able to delete something elsewhere on the box."""
    root = tmp_path / "mempalace"
    (root / "conversations").mkdir(parents=True)
    outsider = tmp_path / "important.md"
    outsider.write_text("not the palace's", encoding="utf-8")
    collection = FakeCollection(
        {"junk": (LEGACY_HEADING, {"source_file": str(outsider)})},
    )
    report = _scrub.scan(collection, root)

    deleted, removed = _scrub.apply(collection, report)

    assert deleted == 1  # the drawer is scaffolding and goes
    assert removed == 0  # the file is not ours to touch
    assert outsider.exists()


def test_a_drawer_with_no_source_file_is_still_deleted(tmp_path):
    """Metadata is provenance, not a precondition: the chunk's own text is what decides."""
    root = tmp_path / "mempalace"
    root.mkdir()
    collection = FakeCollection({"junk": (LEGACY_HEADING, {})})

    report = _scrub.scan(collection, root)
    deleted, removed = _scrub.apply(collection, report)

    assert (deleted, removed) == (1, 0)
    assert collection.drawers == {}


def test_paging_covers_a_palace_larger_than_one_page(monkeypatch, tmp_path):
    monkeypatch.setattr(_scrub, "_PAGE", 3)
    drawers = {f"junk-{n}": (LEGACY_HEADING, {}) for n in range(7)}
    drawers.update({f"real-{n}": (f"John said thing number {n}.", {}) for n in range(5)})
    collection = FakeCollection(drawers)

    report = _scrub.scan(collection, tmp_path)

    assert report.total == 12
    assert len(report.scrub) == 7


# === the report a capital review reads =========================================================


def test_the_report_prints_every_match_in_full_and_says_nothing_was_deleted(palace):
    root, collection = palace
    report = _scrub.scan(collection, root)

    text = _scrub.render(report, applied=False)

    assert f"palace: {root}" in text
    assert "Nothing was deleted." in text
    for drawer_id, chunk in POLLUTED.items():
        assert drawer_id in text
        for line in chunk.splitlines():
            assert line in text  # full text, never an excerpt: a review gates the apply on it
    assert "REVIEW — 1 chunk(s)" in text
    assert "mixed-1" in text
    assert "mode=dry-run" in text


def test_the_report_names_the_catalog_class_and_its_reason(palace):
    root, collection = palace

    text = _scrub.render(_scrub.scan(collection, root), applied=False)

    assert "recall-block" in text
    assert "canned-narration" in text
    assert "harness scaffolding" in text


def test_an_empty_palace_reports_clean(tmp_path):
    report = _scrub.scan(FakeCollection({}), tmp_path)

    text = _scrub.render(report, applied=False)

    assert "(none)" in text
    assert "scrub=0" in text


# === the CLI ===================================================================================


def test_main_dry_runs_by_default(palace, fake_mempalace, capsys):
    root, collection = palace
    fake_mempalace.collection = collection

    assert _scrub.main(["--palace", str(root)]) == 0

    assert collection.deleted == []
    out = capsys.readouterr().out
    assert "mode=dry-run" in out
    assert LEGACY_HEADING in out
    # Never `create=True`: a typo in --palace must not scaffold an empty palace and pass.
    assert fake_mempalace.calls[0][1]["create"] is False


def test_main_apply_deletes_and_reports_the_after_count(palace, fake_mempalace, capsys):
    root, collection = palace
    fake_mempalace.collection = collection

    assert _scrub.main(["--palace", str(root), "--apply"]) == 0

    assert set(collection.drawers) == set(REAL) | set(MIXED)
    assert "mode=apply" in capsys.readouterr().out


def test_main_exits_nonzero_on_a_palace_that_is_not_there(tmp_path, capsys):
    assert _scrub.main(["--palace", str(tmp_path / "nope")]) == 1

    assert "no palace at" in capsys.readouterr().err


def test_a_directory_that_is_not_a_palace_fails_on_one_line(tmp_path, fake_mempalace, capsys):
    """Live-caught: an un-mined directory raised `CollectionNotInitializedError` as a traceback.

    The operator's common mistake is a wrong ``--palace``, and it must read as one line naming
    the vendor's own verdict — not as a stack ending in a class from an optional extra.
    """

    class NotInitialized(RuntimeError):
        pass

    def refuse(path, **kwargs):
        raise NotInitialized(path)

    fake_mempalace.get_collection = refuse

    assert _scrub.main(["--palace", str(tmp_path)]) == 1

    err = capsys.readouterr().err
    assert "could not open the palace" in err
    assert "NotInitialized" in err
    assert "Traceback" not in err


def test_main_resolves_the_palace_the_way_a_wake_does(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    assert _scrub.resolve_palace(None) == tmp_path / "mempalace"

    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_path / "elsewhere"))
    assert _scrub.resolve_palace(None) == tmp_path / "elsewhere"

    # An explicit flag beats both, as `--palace` does for the `mempalace` CLI itself.
    assert _scrub.resolve_palace(str(tmp_path / "flagged")) == tmp_path / "flagged"


def test_classify_agrees_with_what_the_fixtures_claim():
    """The fixtures are the contract; this keeps them honest as the catalog grows."""
    for text in POLLUTED.values():
        assert classify(text)[0] is Verdict.SCRUB
    for text in REAL.values():
        assert classify(text) == (Verdict.KEEP, ())
    for text in MIXED.values():
        assert classify(text)[0] is Verdict.REVIEW
