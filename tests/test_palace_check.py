"""`basecradle-harness-palace-check`: prove a palace serves from where its home is now (issue #606).

MemPalace is an optional extra and not installed in the test env, so its three modules are faked at
the ``sys.modules`` boundary, the way `test_mempalace.py` and `test_scrub.py` fake them. The fakes
model what the check depends on: a paged collection, a searcher that honors ``n_results``, and a
dry-run mine that prints MemPalace's per-file lines.
"""

import hashlib
import sys
import types
from importlib import metadata
from pathlib import Path

import pytest

from basecradle_harness import _palace_check
from basecradle_harness._palace_check import _ASK, _VECTOR_ASK
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
        hash_map_fails=False,
        simulate=False,
        misbehave=False,
    )

    palace = types.ModuleType("mempalace.palace")

    def get_collection(path, **kwargs):
        state.opened.append((path, kwargs))
        return state.collection

    palace.get_collection = get_collection
    palace.NORMALIZE_VERSION = 2

    def prefetch_content_hashes(collection, extract_mode=None):
        # MemPalace's rule: hashed rows at the current normalize_version in the asked extract mode,
        # keyed by (wing, hash), the first source seen for a pair kept.
        if state.hash_map_fails:
            raise RuntimeError("partial fetch")
        hashes = {}
        for _, meta in collection.drawers.values():
            mode = meta.get("extract_mode")
            if not (mode == extract_mode or (mode is None and extract_mode == "exchange")):
                continue
            if not meta.get("wing") or meta.get("normalize_version", 1) < 2:
                continue
            for content_hash in str(meta.get("content_hash") or "").split(","):
                if content_hash:
                    hashes.setdefault((meta["wing"], content_hash), meta["source_file"])
        return hashes

    palace.prefetch_content_hashes = prefetch_content_hashes

    searcher = types.ModuleType("mempalace.searcher")

    def search_memories(query, palace_path, **kwargs):
        state.searches.append((query, kwargs))
        n = kwargs["n_results"]
        # MemPalace's union search: the vector half proposes 3n and keeps the n nearest.
        kept = set(state.collection.query(query_texts=[query], n_results=3 * n)["ids"][0][:n])
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
            # A cut drawer reaches the top only when the vector half kept it.
            and not (drawer_id in state.cut and drawer_id not in kept)
        ]
        # Drawers whose text opens with the query rank first, in insertion order: the first ten of
        # a set of copies take the slots, which is how MemPalace keeps exact ties in index order.
        rows.sort(key=lambda hit: not hit["text"].strip().startswith(query))
        return {"results": rows[:n]}

    searcher.search_memories = search_memories

    convo_miner = types.ModuleType("mempalace.convo_miner")

    def mine_convos(convo_dir, palace_path, **kwargs):
        state.mines.append((convo_dir, kwargs))
        if state.simulate:
            _simulate_mine(state, Path(convo_dir), kwargs, prefetch_content_hashes)
        elif kwargs.get("dry_run") and state.dry_run_lines is not None:
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


def _simulate_mine(state, path, kwargs, prefetch_content_hashes):
    """MemPalace's two rules, for a directory or one file: a known path in any wing is skipped;
    content already filed in the mining wing under another path gets a registry row; anything
    else is filed as a drawer. Prints the per-file lines MemPalace prints."""
    drawers = state.collection.drawers
    wing, dry = kwargs["wing"], kwargs.get("dry_run")
    known = {meta.get("source_file") for _, meta in drawers.values()}
    accepted = prefetch_content_hashes(state.collection, extract_mode="exchange")
    for file in sorted(path.glob("*.md")) if path.is_dir() else [path]:
        source = str(file.resolve())
        if source in known:
            continue
        text = file.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        dup = accepted.get((wing, content_hash))
        if dup and dup != source:
            print(f"  = [   1/1] {file.name}  duplicate of {Path(dup).name}")
            if not dry:
                drawers[f"registry-{wing}-{source}"] = (
                    f"[registry] {source}",
                    {"source_file": source, "room": "_registry", "wing": wing},
                )
            if not (state.misbehave and not dry):
                continue
        if dry:
            print(f"    [DRY RUN] {file.name} -> room:general (1 drawers)")
            continue
        drawers[f"mined-{wing}-{source}"] = (
            text,
            _meta(
                source,
                "2026-10-02T00:00:00",
                wing=wing,
                content_hash=content_hash,
                extract_mode="exchange",
                normalize_version=2,
            ),
        )


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
    # The unranked ask (issue #611), never the reranker's pool.
    assert {kwargs["n_results"] for _, kwargs in fake.searches} == {_ASK}
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
        f"  why: crowded; deep rank 11/100 (similarity 1.0, via drawer); vector rank 11/{_VECTOR_ASK}; "
        "vector self-query yes; "
        "drawers with identical text 0 (0 filed later), with the same query 0; "
        "top 10: 0 identical text, 0 same query, 0 same content hash, 0 same source file"
    )


def test_a_drawer_full_scoring_would_place_but_the_vector_cut_drops_is_cut(home, fake, capsys):
    """Its own query ranks it first over a wide pool, but more drawers than a top-10 search keeps
    from its vector half sit nearer to that query, so that search never lets it compete."""
    drawers = _many(_ASK + 2)
    fake.collection = FakeCollection(drawers, far={"drawer-004"})
    fake.cut = {"drawer-004"}

    _, lines = _run(capsys, "--sample", _ASK + 2, home)

    why = next(line for line in lines if line.startswith("  why:"))
    assert why.startswith(
        f"  why: cut; deep rank 1/100 (similarity 1.0, via drawer); vector rank {_ASK + 2}/{_VECTOR_ASK}; "
    )
    assert lines[-2] == "failed by verdict: twin 0, cut 1, crowded 0, unreached 0"


def test_a_drawer_inside_the_headroom_is_not_cut(home, fake, capsys):
    """Issue #611: eleven drawers sit nearer to its query than it does, which put it past the ten a
    top-10 search used to keep from its vector half. The search now asks for more and keeps the
    first ten of the ranking, so the drawer competes on full scoring and comes back."""
    assert _ASK > 12
    fake.collection = FakeCollection(_many(12), far={"drawer-004"})
    fake.cut = {"drawer-004"}

    code, lines = _run(capsys, "--sample", 12, home)

    assert code == 0
    assert lines[-2] == "failed by verdict: twin 0, cut 0, crowded 0, unreached 0"


def test_a_drawer_the_vector_index_does_not_hold_is_unreached(home, fake, capsys):
    fake.collection = FakeCollection(_many(5), unindexed={"drawer-002"})
    fake.missing = {"drawer-002"}

    _, lines = _run(capsys, "--sample", 5, home)

    why = next(line for line in lines if line.startswith("  why:"))
    assert why.startswith(
        f"  why: unreached; not in top 100; vector rank: not in the {_VECTOR_ASK} candidates; "
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
        "wing -; content_hash on 1; normalize_version 1; extract_mode -; "
        "today's content hash matches a recorded one: no"
    ) in lines
    assert "    reason: no drawer carries today's content hash: it changed after it was filed" in (
        lines
    )


# --- phase 4: the reason a file forecast as new is not recognised, decided by MemPalace's map ---

HAND = "notes-on-the-garden-project.md"
HAND_TEXT = "> What did John plan for the garden?\nNova said tomatoes, watered twice a week.\n"


def _hand_file(home, **meta):
    """A hand-placed file on disk and its five drawers filed under the old home, ``meta`` applied."""
    (home / "mempalace" / "conversations" / HAND).write_text(HAND_TEXT, encoding="utf-8")
    content_hash = hashlib.sha256(HAND_TEXT.strip().encode("utf-8")).hexdigest()
    drawers = {}
    for chunk in range(5):
        row = _meta(
            f"{OLD}/{HAND}",
            "2026-08-30T00:00:00",
            chunk_index=chunk,
            wing="notes_on_the_garden_project.md",
            extract_mode="exchange",
            normalize_version=2,
        )
        if chunk == 0:
            row["content_hash"] = content_hash
        row.update(meta)
        drawers[f"hand-{chunk}"] = (f"Garden chunk {chunk}.", row)
    return drawers


def _reason(capsys, home):
    _, lines = _run(capsys, home)
    return [line for line in lines if line.startswith("    reason: ")]


def test_content_filed_only_in_another_wing_is_named_as_the_reason(home, fake, capsys):
    """The phase 4 file: every field matches except the wing, which the miner's map is keyed on."""
    fake.collection = FakeCollection(_hand_file(home))
    fake.dry_run_lines = [f"    [DRY RUN] {HAND} -> room:planning (5 drawers)"]

    _, lines = _run(capsys, home)

    assert (
        f"  mined as new: {HAND}: 5 drawers recorded under this file name (0 at this location); "
        "wing notes_on_the_garden_project.md; content_hash on 1; normalize_version 2; "
        "extract_mode exchange; today's content hash matches a recorded one: yes"
    ) in lines
    assert (
        "    reason: its content is filed only in wing 'notes_on_the_garden_project.md', not the "
        "observe's 'conversations'; MemPalace recognises a moved file's content within one wing "
        "but a known path in any, so a move files it again in 'conversations': a second copy of "
        "each of its drawers"
    ) in lines


def test_content_filed_in_the_observes_wing_is_recognised(home, fake, capsys):
    """If MemPalace's map holds the hash in the observe's wing, the check says so, not a guess."""
    fake.collection = FakeCollection(_hand_file(home, wing="conversations"))
    fake.dry_run_lines = [f"    [DRY RUN] {HAND} -> room:planning (5 drawers)"]

    assert _reason(capsys, home) == [
        "    reason: none: MemPalace's content-hash map recognises every conversation in it"
    ]


@pytest.mark.parametrize(
    ("meta", "reason"),
    [
        (
            {"wing": "conversations", "extract_mode": "general"},
            "its content is filed only under extract_mode general, not exchange",
        ),
        (
            {"wing": "conversations", "normalize_version": 1},
            "its content is filed only at normalize_version 1, older than MemPalace's 2",
        ),
        (
            {"content_hash": None},
            "its drawers carry no content hash (filed before MemPalace 3.7)",
        ),
    ],
)
def test_each_way_the_map_refuses_a_record_is_named(home, fake, capsys, meta, reason):
    drawers = _hand_file(home)
    drawers["hand-0"][1].update(meta)
    if drawers["hand-0"][1]["content_hash"] is None:
        del drawers["hand-0"][1]["content_hash"]
    fake.collection = FakeCollection(drawers)
    fake.dry_run_lines = [f"    [DRY RUN] {HAND} -> room:planning (5 drawers)"]

    assert _reason(capsys, home) == [f"    reason: {reason}"]


def test_a_file_never_filed_says_so(home, fake, capsys):
    (home / "mempalace" / "conversations" / HAND).write_text(HAND_TEXT, encoding="utf-8")
    fake.collection = FakeCollection(_drawers(home))
    fake.dry_run_lines = [f"    [DRY RUN] {HAND} -> room:planning (1 drawers)"]

    assert _reason(capsys, home) == ["    reason: it was never filed"]


def test_an_unreadable_hash_map_is_an_unknown_reason_not_a_guess(home, fake, capsys):
    fake.collection = FakeCollection(_hand_file(home))
    fake.dry_run_lines = [f"    [DRY RUN] {HAND} -> room:planning (5 drawers)"]
    fake.hash_map_fails = True

    assert _reason(capsys, home) == [
        "    reason: unknown, MemPalace's content-hash map could not be read"
    ]


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


# --- issue #613: --register-off-wing files a moved off-wing file as a registry row, never a drawer ---


def _moved_palace(home, fake, *, hand=None):
    """A moved palace: four observe-mined files in ``conversations`` and the hand-placed file."""
    convos = home / "mempalace" / "conversations"
    drawers = {}
    for index, name in enumerate(NAMES):
        text = f"> Fact {index}?\nNova told John fact number {index}.\n"
        (convos / name).write_text(text, encoding="utf-8")
        drawers[f"drawer-{index}"] = (
            text,
            _meta(
                f"{OLD}/{name}",
                f"2026-09-0{index + 1}T00:00:00",
                wing="conversations",
                extract_mode="exchange",
                normalize_version=2,
                content_hash=hashlib.sha256(text.strip().encode("utf-8")).hexdigest(),
            ),
        )
    drawers.update(_hand_file(home, **(hand or {})))
    fake.collection = FakeCollection(drawers)
    fake.simulate = True
    return fake.collection.drawers


def _writes(fake):
    return [(path, kw) for path, kw in fake.mines if not kw.get("dry_run")]


def test_register_off_wing_dry_run_says_what_it_would_do_and_writes_nothing(home, fake, capsys):
    drawers = _moved_palace(home, fake)
    before = dict(drawers)

    code, lines = _run(capsys, "--register-off-wing", "--dry-run", home)

    assert code == 0
    assert "register off-wing (dry run, writes nothing):" in lines
    assert f"  would register: {HAND} in wing 'notes_on_the_garden_project.md'" in lines
    assert (
        "register off-wing (dry run): files to register 1, registry rows to write 1, "
        "drawers to write 0"
    ) in lines
    assert drawers == before
    assert not _writes(fake)


def test_register_off_wing_writes_one_registry_row_in_the_drawers_wing_and_no_drawer(
    home, fake, capsys
):
    drawers = _moved_palace(home, fake)
    real_before = sum(meta.get("room") != "_registry" for _, meta in drawers.values())

    code, lines = _run(capsys, "--register-off-wing", home)

    assert code == 0
    assert "register off-wing (WRITES TO THE PALACE):" in lines
    assert f"  registered: {HAND} in wing 'notes_on_the_garden_project.md'" in lines
    assert (
        "register off-wing: WROTE to this palace: files registered 1, registry rows written 1, "
        "drawers written 0"
    ) in lines
    # MemPalace's own single-file mine, in the wing the drawers carry: never an argument's.
    ((path, kwargs),) = _writes(fake)
    assert Path(path).name == HAND
    assert kwargs["wing"] == "notes_on_the_garden_project.md"
    assert kwargs["extract_mode"] == "exchange"
    assert sum(meta.get("room") != "_registry" for _, meta in drawers.values()) == real_before
    # And the report that follows: the next observe files nothing new.
    assert (
        "next observe (dry run): 4 files -> registry sentinel, 0 files -> mined as new drawers"
        in (lines)
    )


def test_register_off_wing_twice_registers_nothing_the_second_time(home, fake, capsys):
    drawers = _moved_palace(home, fake)
    _run(capsys, "--register-off-wing", home)
    after_first = dict(drawers)
    fake.mines.clear()

    code, lines = _run(capsys, "--register-off-wing", home)

    assert code == 0
    assert (
        "register off-wing: WROTE to this palace: files registered 0, registry rows written 0, "
        "drawers written 0"
    ) in lines
    assert drawers == after_first
    assert not _writes(fake)


def test_register_off_wing_refuses_and_writes_nothing_if_a_file_would_file_drawers(
    home, fake, capsys
):
    """Drawers MemPalace's map does not accept (an older normalize_version) cannot register it."""
    drawers = _moved_palace(home, fake, hand={"normalize_version": 1})
    before = dict(drawers)

    code, lines = _run(capsys, "--register-off-wing", home)

    assert code == 1
    assert (
        f"  refused: {HAND}: mining it in wing 'notes_on_the_garden_project.md' would file drawers"
    ) in lines
    assert "  refused, nothing written" in lines
    assert drawers == before
    assert not _writes(fake)


@pytest.mark.parametrize(
    "hand",
    [
        {"wing": "conversations", "content_hash": "0" * 64},  # changed since it was filed
        {"content_hash": None},  # filed before content hashes
    ],
)
def test_register_off_wing_leaves_every_other_reason_alone(home, fake, capsys, hand):
    _moved_palace(home, fake, hand=hand)
    if hand.get("content_hash") is None:
        del fake.collection.drawers["hand-0"][1]["content_hash"]

    code, lines = _run(capsys, "--register-off-wing", home)

    assert code == 0
    assert f"  left alone: {HAND}: not the other-wing case (see its reason: line)" in lines
    assert not _writes(fake)


def test_register_off_wing_leaves_a_file_filed_in_two_other_wings_alone(home, fake, capsys):
    drawers = _moved_palace(home, fake)
    copy = dict(drawers["hand-0"][1], wing="another_wing", source_file=f"{OLD}/elsewhere.md")
    drawers["hand-copy"] = ("Garden chunk copy.", copy)

    code, lines = _run(capsys, "--register-off-wing", home)

    assert code == 0
    assert f"  left alone: {HAND}: not the other-wing case (see its reason: line)" in lines
    assert not _writes(fake)


def test_register_off_wing_fails_loudly_if_mempalace_files_a_drawer_anyway(home, fake, capsys):
    """Its dry run said one registry row; the write filed a drawer too. Never reported as done."""
    _moved_palace(home, fake)
    fake.misbehave = True

    code, lines = _run(capsys, "--register-off-wing", home)

    assert code == 1
    assert (
        "register off-wing: WROTE to this palace: files registered 1, registry rows written 1, "
        "drawers written 1"
    ) in lines
    assert any(line.startswith("FAIL: expected 1 registry rows and 0 drawers") for line in lines)


def test_dry_run_needs_register_off_wing(home, fake, capsys):
    with pytest.raises(SystemExit):
        _palace_check.main(["--dry-run", str(home)])
    assert "--dry-run applies to --register-off-wing" in capsys.readouterr().err


def test_the_register_flag_says_it_writes_in_its_help(capsys):
    with pytest.raises(SystemExit):
        _palace_check.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--register-off-wing WRITES TO THE PALACE." in help_text


def test_register_off_wing_leaves_an_unreadable_file_alone_and_names_it(
    home, fake, capsys, monkeypatch
):
    _moved_palace(home, fake)

    def unreadable(path):
        raise OSError("refused")

    monkeypatch.setattr(sys.modules["mempalace.normalize"], "normalize_conversations", unreadable)

    code, lines = _run(capsys, "--register-off-wing", home)

    assert code == 0
    assert f"  left alone: {HAND}: could not be read (OSError)" in lines
    assert not _writes(fake)
