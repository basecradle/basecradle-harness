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

The second palace (`quoting_palace`) pins the false-positive class issue #444 was filed on: **one
real conversation that quotes every catalog literal**, chunked so each quote stands alone in its
own drawer. Every one of those chunks is scaffolding by `classify` and none of them may be
deleted, because the file they were mined from also holds real dialogue. It must lose *nothing*
— not a drawer, not the file — while a file whose drawers are unanimously scaffolding still dies
whole.
"""

import sys
import types

import pytest

from basecradle_harness import _scrub
from basecradle_harness._mempalace import _CLOSE_TAG, _INJECTED_HEADING, _OPEN_TAG
from basecradle_harness._mining import _LEGACY_RECALL_HEADING as LEGACY_HEADING
from basecradle_harness._mining import MIN_UNIT_CHARS, Verdict, _normalize, catalog, classify
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


def _quoted_catalog_lines() -> dict[str, str]:
    """One distinctive line per catalog class, exactly as a person would quote it in a message.

    Derived from `catalog()` rather than spelled out here, so a class added to the catalog later
    is covered by this fixture the day it lands — the same never-re-type-the-constants discipline
    the catalog itself is built on. The ``> `` prefix is not decoration: MemPalace's
    ``extract_mode="exchange"`` files quote the **user half of every exchange** that way, so this
    is byte-for-byte how @origin's charter-draft lines sat in @briggs's palace.
    """
    lines: dict[str, str] = {}
    for entry in catalog():
        for literal in entry.literals:
            for piece in literal.splitlines():
                if len(_normalize(piece)) >= MIN_UNIT_CHARS:
                    lines[entry.name] = f"> {piece.strip()}"
                    break
            if entry.name in lines:
                break
    return lines


@pytest.fixture
def quoting_palace(tmp_path):
    """One real conversation whose drawers quote **every** catalog literal, chunk by chunk.

    The false-positive class of issue #444: a genuine exchange in which a person quoted harness
    scaffolding, chunked so each quote stands alone in its own drawer. Every quote drawer is
    `SCRUB` by `classify` and correctly so — what makes them memories is the *file*, which also
    mined real dialogue. Beside it sits a second file that is unanimously scaffolding, so the
    same fixture proves both halves of the rule.
    """
    root = tmp_path / "mempalace"
    convos = root / "conversations"
    convos.mkdir(parents=True)
    real_file = convos / "real-conversation.md"
    junk_file = convos / "pure-scaffolding.md"
    real_file.write_text("the whole exchange", encoding="utf-8")
    junk_file.write_text("brief, mined whole", encoding="utf-8")

    quotes = _quoted_catalog_lines()
    drawers = {
        f"quote-{name}": (text, {"source_file": str(real_file)}) for name, text in quotes.items()
    }
    drawers["convo-real-1"] = (
        (
            "> [2026-08-16] origin: that's the Title Case rule, verbatim, from the draft\n"
            "Understood — Title Case for every label."
        ),
        {"source_file": str(real_file)},
    )
    drawers["convo-real-2"] = (
        "> [2026-08-16] origin: does it read right to you?\nIt does. I'll apply it.",
        {"source_file": str(real_file)},
    )
    for index, text in enumerate(POLLUTED.values()):
        drawers[f"pure-{index}"] = (text, {"source_file": str(junk_file)})
    return root, FakeCollection(drawers), real_file, junk_file, {f"quote-{n}" for n in quotes}


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


def test_a_drawer_with_no_source_file_is_held_because_unanimity_cannot_be_proven(tmp_path):
    """Unanimity is a statement about a file's *other* drawers. No file, nothing to count.

    Before issue #444 this chunk was deleted on its own text alone — which is precisely the
    evidence the founder's hold ruled insufficient, since a quote-only chunk of real dialogue
    looks identical. Unproven is held; the failure direction here is always "left alone".
    """
    root = tmp_path / "mempalace"
    root.mkdir()
    collection = FakeCollection({"junk": (LEGACY_HEADING, {})})

    report = _scrub.scan(collection, root)
    deleted, removed = _scrub.apply(collection, report)

    assert report.scrub == []
    assert [f.drawer_id for f in report.held] == ["junk"]
    assert report.held[0].held == _scrub._HELD_NO_SOURCE
    assert (deleted, removed) == (0, 0)
    assert set(collection.drawers) == {"junk"}


def test_paging_covers_a_palace_larger_than_one_page(monkeypatch, tmp_path):
    """A file's drawers can straddle a page boundary, so unanimity is decided after the walk."""
    monkeypatch.setattr(_scrub, "_PAGE", 3)
    junk = tmp_path / "conversations" / "junk.md"
    drawers = {f"junk-{n}": (LEGACY_HEADING, {"source_file": str(junk)}) for n in range(7)}
    drawers.update(
        {
            f"real-{n}": (
                f"John said thing number {n}.",
                {"source_file": str(tmp_path / "conversations" / f"real-{n}.md")},
            )
            for n in range(5)
        }
    )
    collection = FakeCollection(drawers)

    report = _scrub.scan(collection, tmp_path)

    assert report.total == 12
    # All seven junk drawers came from the one file, and every drawer that file mined is
    # scaffolding — unanimous across three pages.
    assert len(report.scrub) == 7
    assert report.held == []


# === the false-positive class: a real conversation that quotes scaffolding (issue #444) =========


def test_the_quoted_catalog_lines_really_are_scaffolding_by_themselves(quoting_palace):
    """The fixture is only a fixture if `classify` genuinely condemns every one of these.

    Without this the "loses nothing" test below could pass for the wrong reason — a fixture
    whose chunks never matched the catalog at all.
    """
    _, _, _, _, quote_ids = quoting_palace

    quotes = _quoted_catalog_lines()
    assert set(quotes) == {entry.name for entry in catalog()}  # every class, no gaps
    assert {f"quote-{name}" for name in quotes} == quote_ids
    for name, text in quotes.items():
        assert classify(text) == (Verdict.SCRUB, (name,))


def test_a_real_conversation_that_quotes_the_catalog_loses_nothing_on_apply(quoting_palace):
    """The founder's condition, end to end: not a drawer, not the file.

    Every quote drawer is scaffolding on its own text and is *held* anyway, because the file it
    was mined from also mined real dialogue. The hold names the evidence — how many siblings are
    not scaffolding — so a reviewer can judge it without re-deriving it.
    """
    root, collection, real_file, _, quote_ids = quoting_palace

    report = _scrub.scan(collection, root)

    assert {f.drawer_id for f in report.held} == quote_ids
    assert {f.held for f in report.held} == {
        _scrub._HELD_SIBLINGS.format(others=2)  # convo-real-1 and convo-real-2
    }
    assert quote_ids.isdisjoint({f.drawer_id for f in report.scrub})

    before = dict(collection.drawers)
    _scrub.apply(collection, report)

    assert real_file.exists()
    for drawer_id in quote_ids | {"convo-real-1", "convo-real-2"}:
        assert collection.drawers[drawer_id] == before[drawer_id]


def test_a_unanimous_scaffolding_file_still_dies_whole(quoting_palace):
    """True pollution — a file every one of whose drawers is scaffolding — is still deleted."""
    root, collection, real_file, junk_file, quote_ids = quoting_palace
    report = _scrub.scan(collection, root)

    deleted, removed = _scrub.apply(collection, report)

    assert {f.drawer_id for f in report.scrub} == {f"pure-{n}" for n in range(len(POLLUTED))}
    assert (deleted, removed) == (len(POLLUTED), 1)
    assert not junk_file.exists()
    assert real_file.exists()
    assert set(collection.drawers) == quote_ids | {"convo-real-1", "convo-real-2"}


def test_one_non_scaffolding_sibling_holds_every_match_in_its_file(tmp_path):
    """Unanimity, at its boundary: the file goes at zero real siblings and stays at one."""
    root = tmp_path / "mempalace"
    convos = root / "conversations"
    convos.mkdir(parents=True)
    source = convos / "exchange.md"
    source.write_text("mined", encoding="utf-8")
    drawers = {
        "junk-a": (LEGACY_HEADING, {"source_file": str(source)}),
        "junk-b": (_STUCK_NOTE, {"source_file": str(source)}),
    }

    unanimous = FakeCollection(dict(drawers))
    assert len(_scrub.scan(unanimous, root).scrub) == 2

    drawers["a-memory"] = ("> john: my dog is called Rex\nNoted.", {"source_file": str(source)})
    held = FakeCollection(drawers)

    report = _scrub.scan(held, root)
    deleted, removed = _scrub.apply(held, report)

    assert report.scrub == []
    assert {f.drawer_id for f in report.held} == {"junk-a", "junk-b"}
    assert (deleted, removed) == (0, 0)
    assert source.exists()


def test_an_unclassifiable_sibling_holds_the_file_too(tmp_path):
    """An empty drawer votes for neither side, so it cannot complete a unanimity."""
    root = tmp_path / "mempalace"
    convos = root / "conversations"
    convos.mkdir(parents=True)
    source = convos / "exchange.md"
    source.write_text("mined", encoding="utf-8")
    collection = FakeCollection(
        {
            "junk": (LEGACY_HEADING, {"source_file": str(source)}),
            "blank": ("", {"source_file": str(source)}),
        }
    )

    report = _scrub.scan(collection, root)

    assert report.scrub == []
    assert [f.drawer_id for f in report.held] == ["junk"]
    assert _scrub.apply(collection, report) == (0, 0)
    assert source.exists()


def test_the_live_briggs_palace_shape_reports_zero_deletable(tmp_path):
    """The field case, at its reported dimensions (basecradle-noc#560).

    Three real conversation files backing 5, 17 and 7 drawers; the #438 catalog matched 3 quote
    lines in the first and 1 in each of the others — five chunks, all of them dialogue. The
    healthy post-fix state is scrub=0 with all five reported and nothing deletable.
    """
    root = tmp_path / "mempalace"
    convos = root / "conversations"
    convos.mkdir(parents=True)
    quotes = list(_quoted_catalog_lines().values())
    drawers = {}
    for name, total, matches in (("charter-draft", 5, 3), ("design-a", 17, 1), ("design-b", 7, 1)):
        source = convos / f"{name}.md"
        source.write_text("a real exchange", encoding="utf-8")
        for index in range(total):
            text = quotes[index % len(quotes)] if index < matches else f"> john: point {index}"
            drawers[f"{name}-{index}"] = (text, {"source_file": str(source)})
    collection = FakeCollection(drawers)

    report = _scrub.scan(collection, root)

    assert report.total == 29
    assert report.scrub == []
    assert len(report.held) == 5
    assert _scrub.apply(collection, report) == (0, 0)
    assert len(list(convos.iterdir())) == 3
    assert len(collection.drawers) == 29


def test_apply_refuses_a_blocked_file_even_when_a_report_names_it_scrubbable(tmp_path):
    """Defense in depth: the condition is checked where the tool *acts*, not only where it sorts.

    A hand-built report puts a scaffolding chunk and a real one from the same file on opposite
    sides of the rule. `scan` can never produce this — which is the point: the guarantee must not
    depend on the classifier having been right.
    """
    root = tmp_path / "mempalace"
    convos = root / "conversations"
    convos.mkdir(parents=True)
    source = convos / "exchange.md"
    source.write_text("mined", encoding="utf-8")
    collection = FakeCollection({"junk": (LEGACY_HEADING, {"source_file": str(source)})})
    report = _scrub.Report(palace=root, total=2)
    report.scrub.append(
        _scrub.Finding("junk", Verdict.SCRUB, ("recall-block",), LEGACY_HEADING, str(source))
    )
    report.review.append(
        _scrub.Finding("mixed", Verdict.REVIEW, ("recall-block",), "real", str(source))
    )

    assert _scrub.apply(collection, report) == (0, 0)
    assert source.exists()
    assert collection.deleted == []


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


def test_the_report_gives_held_findings_their_own_section_and_states_why(quoting_palace):
    """A reviewer must be able to see the false-positive class, not re-derive it from a diff."""
    root, collection, real_file, _, quote_ids = quoting_palace

    text = _scrub.render(_scrub.scan(collection, root), applied=False)

    assert f"HELD — {len(quote_ids)} chunk(s)" in text
    assert f"held: {_scrub._HELD_SIBLINGS.format(others=2)}" in text
    assert str(real_file) in text
    assert f"held={len(quote_ids)}" in text
    for quote in _quoted_catalog_lines().values():
        assert quote in text  # full text here too: a hold is judged on what it holds


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
