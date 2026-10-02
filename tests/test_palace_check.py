"""`basecradle-harness-palace-check`: prove a palace serves from where its home is now (issue #606).

MemPalace is an optional extra and not installed in the test env, so its three modules are faked at
the ``sys.modules`` boundary, the way `test_mempalace.py` and `test_scrub.py` fake them. The fakes
model what the check depends on: a paged collection, a searcher that honors ``n_results``, and a
dry-run mine that prints MemPalace's per-file lines.
"""

import sys
import types
from importlib import metadata

import pytest

from basecradle_harness import _palace_check
from basecradle_harness._rerank import RERANK_MODEL_VAR

OLD = "/home/olduser/harness/mempalace/conversations"
NAMES = [f"{n:032x}.md" for n in range(1, 5)]


class FakeCollection:
    def __init__(self, drawers):
        self.drawers = dict(drawers)

    def get(self, *, limit=None, offset=0, include=None):
        items = list(self.drawers.items())[offset : offset + limit]
        return {
            "ids": [drawer_id for drawer_id, _ in items],
            "documents": [payload[0] for _, payload in items],
            "metadatas": [payload[1] for _, payload in items],
        }


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
        collection=None, opened=[], searches=[], mines=[], missing=set(), dry_run_lines=None
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
            {"drawer_id": drawer_id, "text": text, "room": meta.get("room")}
            for drawer_id, (text, meta) in state.collection.drawers.items()
            if drawer_id not in state.missing
        ]
        rows.sort(key=lambda hit: hit["text"] != query)  # the drawer the query names ranks first
        return {"results": rows[: kwargs["n_results"]]}

    searcher.search_memories = search_memories

    convo_miner = types.ModuleType("mempalace.convo_miner")

    def mine_convos(convo_dir, palace_path, **kwargs):
        state.mines.append((convo_dir, kwargs))
        if kwargs.get("dry_run") and state.dry_run_lines is not None:
            print("\n".join(state.dry_run_lines))

    convo_miner.mine_convos = mine_convos

    parent = types.ModuleType("mempalace")
    for name, module in (("palace", palace), ("searcher", searcher), ("convo_miner", convo_miner)):
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
    assert any(line.startswith("FAIL: drawer drawer-0") for line in lines)


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
