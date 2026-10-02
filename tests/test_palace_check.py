"""`basecradle-harness-palace-check`: prove a palace serves from where its home is now (issue #606).

MemPalace is an optional extra and not installed in the test env, so its three modules are faked at
the ``sys.modules`` boundary, the way `test_mempalace.py` and `test_scrub.py` fake them. The fakes
model what the check depends on: a paged collection, a searcher that honors ``n_results``, and a
dry-run mine that prints MemPalace's per-file lines.
"""

import sys
import types
from importlib import metadata
from pathlib import Path

import pytest

from basecradle_harness import _palace_check
from basecradle_harness._rerank import RERANK_MODEL_VAR

OLD = "/home/olduser/harness/mempalace/conversations"
NAMES = [f"{n:032x}.md" for n in range(1, 5)]


class FakeCollection:
    """A paged collection whose vector index holds every drawer not named in ``unindexed``.

    A query by text ranks drawers that open with it first, in insertion order, like the fake
    searcher; drawers named in ``far`` rank after everything else (their own query is a poor match
    for their embedding).
    """

    def __init__(self, drawers, unindexed=(), far=()):
        self.drawers = dict(drawers)
        self.unindexed = set(unindexed)
        self.far = set(far)

    def get(self, *, ids=None, limit=None, offset=0, include=None):
        if ids is not None:
            return {"ids": list(ids), "embeddings": [[float(len(i)), 1.0] for i in ids]}
        items = list(self.drawers.items())[offset : offset + limit]
        return {
            "ids": [drawer_id for drawer_id, _ in items],
            "documents": [payload[0] for _, payload in items],
            "metadatas": [payload[1] for _, payload in items],
        }

    def query(self, *, n_results, query_embeddings=None, query_texts=None, include=None):
        ids = [i for i in self.drawers if i not in self.unindexed]
        if query_texts is not None:
            ids.sort(
                key=lambda i: (
                    i in self.far,
                    not self.drawers[i][0].strip().startswith(query_texts[0]),
                )
            )
        ids = ids[:n_results]
        return {"ids": [ids], "distances": [[0.0] * len(ids)]}


@pytest.fixture
def home(tmp_path):
    """A HARNESS_HOME whose palace was mined under ``/home/olduser`` and then moved here."""
    harness_home = tmp_path / "newuser" / "harness"
    convos = harness_home / "mempalace" / "conversations"
    convos.mkdir(parents=True)
    for name in NAMES:
        (convos / name).write_text("> mined\nyes\n", encoding="utf-8")
    return harness_home


def _drawers(home, *, registered_here=0, unhashed=0):
    here = str((home / "mempalace" / "conversations").resolve())
    drawers = {}
    for index, name in enumerate(NAMES):
        meta = {
            "source_file": f"{OLD}/{name}",
            "filed_at": f"2026-09-0{index + 1}T00:00:00",
            "chunk_index": 0,
            "room": "technical",
        }
        if index >= unhashed:
            meta["content_hash"] = f"{index:064x}"
        drawers[f"drawer-{index}"] = (f"Nova told John fact number {index}.", meta)
    for index, name in enumerate(NAMES[:registered_here]):
        path = f"{here}/{name}"
        drawers[f"registry-{index}"] = (
            f"[registry] {path}",
            {"source_file": path, "room": "_registry"},
        )
    return drawers


@pytest.fixture
def fake(monkeypatch):
    """Install fake ``mempalace.palace`` / ``.searcher`` / ``.convo_miner``; return their records."""
    state = types.SimpleNamespace(
        collection=None,
        opened=[],
        searches=[],
        mines=[],
        missing=set(),
        cut=set(),
        dry_run_lines=None,
    )

    palace = types.ModuleType("mempalace.palace")

    def get_collection(path, **kwargs):
        state.opened.append((path, kwargs))
        return state.collection

    palace.get_collection = get_collection

    searcher = types.ModuleType("mempalace.searcher")

    def search_memories(query, palace_path, **kwargs):
        state.searches.append((query, kwargs))
        rows = [
            {
                "drawer_id": drawer_id,
                "text": text,
                "room": meta.get("room"),
                "similarity": 1.0,
                "matched_via": "drawer",
            }
            for drawer_id, (text, meta) in state.collection.drawers.items()
            if drawer_id not in state.missing
            # A cut drawer reaches the top only when the search keeps enough vector candidates.
            and not (drawer_id in state.cut and kwargs["n_results"] <= 10)
        ]
        # Drawers whose text opens with the query rank first, in insertion order: the first ten of
        # a set of copies take the slots, which is how MemPalace keeps exact ties in index order.
        rows.sort(key=lambda hit: not hit["text"].strip().startswith(query))
        return {"results": rows[: kwargs["n_results"]]}

    searcher.search_memories = search_memories

    convo_miner = types.ModuleType("mempalace.convo_miner")

    def mine_convos(convo_dir, palace_path, **kwargs):
        state.mines.append((convo_dir, kwargs))
        if kwargs.get("dry_run") and state.dry_run_lines is not None:
            print("\n".join(state.dry_run_lines))

    convo_miner.mine_convos = mine_convos

    normalize = types.ModuleType("mempalace.normalize")
    normalize.normalize_conversations = lambda path: [Path(path).read_text(encoding="utf-8")]

    parent = types.ModuleType("mempalace")
    modules = (
        ("palace", palace),
        ("searcher", searcher),
        ("convo_miner", convo_miner),
        ("normalize", normalize),
    )
    for name, module in modules:
        setattr(parent, name, module)
        monkeypatch.setitem(sys.modules, f"mempalace.{name}", module)
    monkeypatch.setitem(sys.modules, "mempalace", parent)
    return state


def _run(capsys, *argv):
    code = _palace_check.main([str(arg) for arg in argv])
    return code, capsys.readouterr().out.splitlines()


def test_every_probe_found_passes_and_reads_the_palace_read_only(home, fake, capsys):
    fake.collection = FakeCollection(_drawers(home))

    code, lines = _run(capsys, home)

    assert code == 0
    assert lines[0] == f"palace: {home.resolve() / 'mempalace'}"
    assert [line.split(":")[0] for line in lines if line.startswith(("PASS", "FAIL"))] == [
        "PASS"
    ] * 3
    assert lines[-2] == "probes drawn from drawers filed at another location"
    assert lines[-1] == "registry sentinels in top 10 for query 'newuser': 0"
    assert fake.opened == [
        (str(home.resolve() / "mempalace"), {"create": False, "read_only": True})
    ]


def test_a_probe_drawer_that_does_not_come_back_fails_the_check(home, fake, capsys):
    fake.collection = FakeCollection(_drawers(home))
    fake.missing = {"drawer-0"}

    code, lines = _run(capsys, home)

    assert code == 1
    at = next(i for i, line in enumerate(lines) if line.startswith("FAIL: drawer drawer-0"))
    assert lines[at + 1].startswith(f"  why: unreached; not in top {_palace_check._DEEP};")


def test_the_report_counts_what_a_moved_palace_holds(home, fake, capsys):
    fake.collection = FakeCollection(_drawers(home, registered_here=1, unhashed=2))
    fake.dry_run_lines = [
        f"  = [   1/4] {NAMES[1]}  duplicate of {NAMES[1]}",
        f"  = [   2/4] {NAMES[2]}  duplicate of {NAMES[2]}",
        f"    [DRY RUN] {NAMES[3]} -> room:technical (1 drawers)",
    ]

    _, lines = _run(capsys, home)

    assert "drawers: 5 total, 4 real, 1 registry sentinels" in lines
    assert "real drawers filed at another location: 4" in lines
    assert "real chunk-0 drawers without content_hash (pre-3.7 schema): 2" in lines
    assert "conversation files: 4, not yet registered at this location: 3" in lines
    assert (
        "next observe (dry run): 2 files -> registry sentinel, 1 files -> mined as new drawers"
        in (lines)
    )


def test_an_unreadable_dry_run_is_reported_as_unknown_not_as_nothing(home, fake, capsys):
    fake.collection = FakeCollection(_drawers(home))
    fake.dry_run_lines = ["a format this version does not know"]

    _, lines = _run(capsys, home)

    assert "next observe (dry run): unknown, MemPalace's dry-run output was not readable" in lines


def test_no_model_call_and_the_named_home_decides_the_palace(
    home, fake, capsys, monkeypatch, tmp_path
):
    """The reranker is the only model call a search can make; an inherited palace path would
    otherwise outrank the HARNESS_HOME the caller named."""
    fake.collection = FakeCollection(_drawers(home))
    monkeypatch.setenv(RERANK_MODEL_VAR, "some/model")
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_path / "another-palace"))

    code, _ = _run(capsys, home)

    assert code == 0
    assert {kwargs["n_results"] for _, kwargs in fake.searches} == {10}  # no rerank pool
    assert fake.opened[0][0] == str(home.resolve() / "mempalace")


def test_without_the_practice_flag_nothing_is_written(home, fake, capsys):
    fake.collection = FakeCollection(_drawers(home))
    before = sorted(p.name for p in (home / "mempalace" / "conversations").iterdir())

    _run(capsys, home)

    assert all(kwargs.get("dry_run") for _, kwargs in fake.mines)
    assert sorted(p.name for p in (home / "mempalace" / "conversations").iterdir()) == before


def test_practice_observe_runs_observe_once_and_says_it_wrote(home, fake, capsys):
    fake.collection = FakeCollection(_drawers(home))
    convos = home / "mempalace" / "conversations"
    before = {p.name for p in convos.iterdir()}

    code, lines = _run(capsys, "--practice-observe", home)

    written = [p for p in convos.iterdir() if p.name not in before]
    assert len(written) == 1
    assert _palace_check.PRACTICE_USER in written[0].read_text(encoding="utf-8")
    assert fake.mines[0][1].get("dry_run", False) is False  # the observe mined for real
    assert lines[1].startswith("practice observe: WROTE one exchange to this palace")
    assert lines[1].endswith("Practice copies only.")
    assert code == 0


def test_the_practice_flag_says_it_writes_in_its_help(capsys):
    with pytest.raises(SystemExit):
        _palace_check.main(["--help"])
    assert "WRITES TO THE PALACE" in capsys.readouterr().out


def test_a_missing_palace_fails_on_one_line(tmp_path, fake, capsys):
    code, lines = _run(capsys, tmp_path / "nobody" / "harness")

    assert code == 1
    assert lines[-1].startswith("FAIL: no palace at ")


def test_no_home_named_and_none_in_the_environment_fails(monkeypatch, capsys):
    monkeypatch.delenv("HARNESS_HOME", raising=False)
    assert _palace_check.main([]) == 1
    assert "name a HARNESS_HOME" in capsys.readouterr().err


def test_a_palace_mempalace_refuses_to_open_fails_on_one_line(home, fake, capsys):
    def refuse(path, **kwargs):
        raise RuntimeError("collection not initialized")

    sys.modules["mempalace.palace"].get_collection = refuse

    code, lines = _run(capsys, home)

    assert code == 1
    assert lines[-1] == "FAIL: could not open the palace: RuntimeError: collection not initialized"


def test_the_console_script_is_registered():
    scripts = {ep.name: ep.value for ep in metadata.entry_points(group="console_scripts")}
    assert scripts["basecradle-harness-palace-check"] == "basecradle_harness._palace_check:main"


# --- phase 3: a sample two palaces can be compared on, and a text-free diagnosis of each miss ---

TEXT = "John asked Nova whether the queue worker lock was released after the deploy."


def _meta(source, filed, **extra):
    return {"source_file": source, "filed_at": filed, "chunk_index": 0, "room": "general", **extra}


def _many(count, *, root=OLD, prefix="drawer", day="2026-09-01"):
    return {
        f"{prefix}-{n:03d}": (
            f"Nova told John distinct fact number {n} about the release.",
            _meta(f"{root}/{n:032x}.md", f"{day}T00:00:{n % 60:02d}"),
        )
        for n in range(count)
    }


def test_a_sample_is_chosen_by_drawer_id_wherever_the_palace_sits(home, fake, capsys, tmp_path):
    """The same drawers, filed under another home and listed in another order, are the same
    sample: the choice keys on the drawer id, which a move never changes."""
    drawers = _many(40)
    fake.collection = FakeCollection(drawers)
    _, here = _run(capsys, "--sample", 7, home)

    moved = {
        drawer_id: (text, {**meta, "source_file": meta["source_file"].replace("olduser", "x")})
        for drawer_id, (text, meta) in reversed(list(drawers.items()))
    }
    fake.collection = FakeCollection(moved)
    _, there = _run(capsys, "--sample", 7, home)

    population = [line for line in here if line.startswith("sample population:")]
    assert population == [line for line in there if line.startswith("sample population:")]
    assert population[0].startswith("sample population: 40 drawers (any filing time)")
    probed = sorted(drawers, key=lambda d: __import__("hashlib").sha256(d.encode()).hexdigest())[:7]
    queries = [query for query, _ in fake.searches if query.startswith("Nova told John")]
    assert sorted(queries[: len(queries) // 2]) == sorted(drawers[d][0] for d in probed)


def test_filed_before_holds_the_population_still_while_a_palace_grows(home, fake, capsys):
    grown = {
        **_many(30),
        **_many(5, prefix="later", day="2026-10-05"),
        "undated": (TEXT, _meta(f"{OLD}/u.md", "not a time")),
    }
    fake.collection = FakeCollection(_many(30))
    _, copy = _run(capsys, "--sample", 50, "--filed-before", "2026-10-02T00:00:00", home)
    fake.collection = FakeCollection(grown)
    _, live = _run(capsys, "--sample", 50, "--filed-before", "2026-10-02T00:00:00", home)

    assert [line.split(";")[1] for line in copy if line.startswith("sample population")] == [
        line.split(";")[1] for line in live if line.startswith("sample population")
    ]
    assert any(
        line.startswith(
            "sample population: 30 drawers (filed before 2026-10-02T00:00:00, 1 undated"
        )
        for line in live
    )


def test_a_sample_reports_only_its_misses_and_ends_with_the_summary(home, fake, capsys):
    drawers = _many(20)
    fake.collection = FakeCollection(drawers)
    fake.missing = {"drawer-003", "drawer-011"}

    code, lines = _run(capsys, "--sample", 20, home)

    assert code == 1
    assert not [line for line in lines if line.startswith("PASS")]
    assert sorted(line.split()[2] for line in lines if line.startswith("FAIL")) == sorted(
        fake.missing
    )
    assert lines[-2] == "failed by verdict: twin 0, cut 0, crowded 0, unreached 2"
    assert lines[-1] == (
        "sample summary: probes 20, passed 18, failed 2, "
        f"failing digest {_palace_check.digest(['drawer-011', 'drawer-003'])}"
    )


def test_a_clean_sample_passes(home, fake, capsys):
    fake.collection = FakeCollection(_many(12))

    code, lines = _run(capsys, "--sample", 5, home)

    assert code == 0
    assert lines[-1] == "sample summary: probes 5, passed 5, failed 0, failing digest none"


def test_a_copy_crowded_out_by_its_identical_twins_is_a_twin_miss(home, fake, capsys):
    """Twelve copies of one exchange and ten slots: the memory is recalled, under other ids."""
    twins = {
        f"twin-{n:02d}": (TEXT, _meta(f"{OLD}/{n:032x}.md", f"2026-08-29T00:00:{n:02d}"))
        for n in range(12)
    }
    fake.collection = FakeCollection(twins)

    _, lines = _run(capsys, "--sample", 12, home)

    failed = [line.split()[2] for line in lines if line.startswith("FAIL")]
    whys = sorted(line for line in lines if line.startswith("  why:"))
    assert sorted(failed) == ["twin-10", "twin-11"]  # the copies past the tenth slot
    assert [why.split(";")[:2] for why in whys] == [
        ["  why: twin", " deep rank 11/100 (similarity 1.0, via drawer)"],
        ["  why: twin", " deep rank 12/100 (similarity 1.0, via drawer)"],
    ]
    tail = "top 10: 10 identical text, 10 same query, 0 same content hash, 0 same source file"
    assert {why.split("; ", 4)[4] for why in whys} == {
        f"drawers with identical text 11 ({later} filed later), with the same query 11; {tail}"
        for later in (0, 1)
    }


def test_a_drawer_outranked_by_near_copies_is_crowded(home, fake, capsys):
    near = {
        f"near-{n:02d}": (
            f"{TEXT} And a different ending {n}.",
            _meta(f"{OLD}/n{n}.md", "2026-09-01"),
        )
        for n in range(10)
    }
    probe = {"probe": (TEXT, _meta(f"{OLD}/p.md", "2026-08-01T00:00:00"))}
    fake.collection = FakeCollection({**near, **probe})

    _, lines = _run(capsys, "--sample", 11, home)

    at = next(i for i, line in enumerate(lines) if line.startswith("FAIL: drawer probe "))
    assert lines[at + 1] == (
        "  why: crowded; deep rank 11/100 (similarity 1.0, via drawer); vector rank 11/30; "
        "vector self-query yes; "
        "drawers with identical text 0 (0 filed later), with the same query 0; "
        "top 10: 0 identical text, 0 same query, 0 same content hash, 0 same source file"
    )


def test_a_drawer_full_scoring_would_place_but_the_vector_cut_drops_is_cut(home, fake, capsys):
    """Its own query ranks it first over a wide pool, but ten drawers sit nearer to that query in
    the vector index, so a top-10 search never lets it compete."""
    drawers = _many(12)
    fake.collection = FakeCollection(drawers, far={"drawer-004"})
    fake.cut = {"drawer-004"}

    _, lines = _run(capsys, "--sample", 12, home)

    why = next(line for line in lines if line.startswith("  why:"))
    assert why.startswith(
        "  why: cut; deep rank 1/100 (similarity 1.0, via drawer); vector rank 12/30; "
    )
    assert lines[-2] == "failed by verdict: twin 0, cut 1, crowded 0, unreached 0"


def test_a_drawer_the_vector_index_does_not_hold_is_unreached(home, fake, capsys):
    fake.collection = FakeCollection(_many(5), unindexed={"drawer-002"})
    fake.missing = {"drawer-002"}

    _, lines = _run(capsys, "--sample", 5, home)

    why = next(line for line in lines if line.startswith("  why:"))
    assert why.startswith(
        "  why: unreached; not in top 100; vector rank: not in the 30 candidates; "
        "vector self-query no (not in its own top 100)"
    )


def test_what_outranked_a_miss_is_counted_by_hash_and_source(home, fake, capsys):
    shared = {"content_hash": "a" * 64}
    siblings = {
        f"sib-{n}": (f"{TEXT} part {n}", _meta(f"{OLD}/same.md", "2026-09-02", **shared))
        for n in range(10)
    }
    probe = {"probe": (TEXT, _meta(f"{OLD}/same.md", "2026-09-01", **shared))}
    fake.collection = FakeCollection({**siblings, **probe})

    _, lines = _run(capsys, "--sample", 11, home)

    why = next(line for line in lines if line.startswith("  why:"))
    assert why.endswith("10 same content hash, 10 same source file")


def test_no_drawer_text_is_ever_printed(home, fake, capsys):
    """Ids, dates, paths, counts and digests only: the report may be pasted into a public issue."""
    secret = "SENTINEL-TEXT-THAT-MUST-NOT-PRINT"
    drawers = {
        f"twin-{n:02d}": (f"{secret} {TEXT}", _meta(f"{OLD}/{n:032x}.md", "2026-08-29"))
        for n in range(12)
    }
    fake.collection = FakeCollection(drawers)
    fake.missing = {"twin-05"}

    for argv in (("--sample", 12, home), (home,)):
        _, lines = _run(capsys, *argv)
        assert not [line for line in lines if secret in line or "queue worker" in line]


def test_a_file_forecast_as_new_is_diagnosed_in_counts(home, fake, capsys):
    drawers = _drawers(home)
    name = NAMES[3]
    meta = drawers["drawer-3"][1]
    meta["normalize_version"] = 1
    fake.collection = FakeCollection(drawers)
    fake.dry_run_lines = [f"    [DRY RUN] {name} -> room:technical (1 drawers)"]

    _, lines = _run(capsys, home)

    assert (
        f"  mined as new: {name}: 1 drawers recorded under this file name (0 at this location); "
        "content_hash on 1; normalize_version 1; extract_mode -; "
        "today's content hash matches a recorded one: no"
    ) in lines


def test_filed_before_needs_a_sample_and_a_local_time(home, fake, capsys):
    with pytest.raises(SystemExit):
        _palace_check.main(["--filed-before", "2026-10-02T00:00:00", str(home)])
    assert "--filed-before applies to --sample" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _palace_check.main(["--sample", "5", "--filed-before", "2026-10-02T00:00:00+00:00"])
    assert "without a UTC offset" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _palace_check.main(["--sample", "0", str(home)])
    assert "not a positive count" in capsys.readouterr().err


def test_the_digest_is_order_free_and_says_none_for_nothing():
    assert _palace_check.digest(["b", "a"]) == _palace_check.digest(["a", "b", "a"])
    assert len(_palace_check.digest(["a"])) == 16
    assert _palace_check.digest([]) == "none"
