"""``--pool-diagnosis`` on real MemPalace palaces, one per kind of miss (issue #633).

Every other palace-check test fakes MemPalace at the ``sys.modules`` boundary. This one cannot: the
mode's facts are measurements of MemPalace's own search (its vector half, its lexical half and the
Chroma scan window, its merge and hybrid rank), and a fake that defined those facts would verify
nothing but itself. So each scenario builds a **real** palace with the installed MemPalace and its
real embedding model, designed so the reason its probe drawer misses the pool is known by
construction, and asserts the verdict the mode gives, the facts behind it, and that the mode's
reconstruction of the search matches the real ``search_memories`` on that palace.

Marked ``mempalace`` and excluded from the default run: it needs the ``mempalace`` extra, and the
embedding model it downloads on first use. Run it with ``uv run --extra mempalace pytest -m
mempalace``.

How the scenarios steer MemPalace, with nothing but text:

- **Copies of a head** (``H``, 60 words, longer than the 400-character query a head probe makes) are the nearest drawers to a query of ``H`` in both halves,
  so ``n`` of them put the probe drawer ``P = H + T`` (``T`` 25 other words) at rank ``n + 1`` in
  the vector half and in the lexical half alike.
- **A decoy** is ``H`` behind 300 one-character tokens. BM25 reads only ``\\w{2,}``, so lexically it
  is a copy of ``H``; the embedding model reads its first 256 word pieces, which are all filler, so
  it is far from ``H``. Decoys push ``P`` down the lexical half and leave the vector half alone.
- **Window fillers** each carry a few of ``H``'s words and are filed first, so they are the 500
  rows the Chroma FTS scan reads for a query of ``H``, and everything filed after them is outside.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import pytest

pytestmark = pytest.mark.mempalace

mempalace = pytest.importorskip("mempalace")

from basecradle_harness import _palace_check, _palace_pool  # noqa: E402
from basecradle_harness._mempalace import _REGISTRY_ROOM, MemPalaceMemoryProvider  # noqa: E402
from basecradle_harness._palace_recall import ARM_1, ARM_2, ARM_3, HEAD, Probe, fetch  # noqa: E402

SYLLABLES = [a + b for a in "bdfgklmnprstvz" for b in ("a", "e", "i", "o", "u", "ai", "or")]


def _words(rng: random.Random, n: int, vocab: list[str]) -> list[str]:
    return [rng.choice(vocab) for _ in range(n)]


def _vocab(seed: int, size: int) -> list[str]:
    rng = random.Random(seed)
    words: set[str] = set()
    while len(words) < size:
        words.add("".join(rng.choice(SYLLABLES) for _ in range(3)))
    return sorted(words)


HEAD_VOCAB = _vocab(1, 400)
TAIL_VOCAB = _vocab(2, 2000)

#: Filler is English prose, so it sits far from ``H`` (made-up words) in embedding space and the
#: copies, the decoys and ``P`` are the only drawers near a query of ``H``.
ENGLISH = [
    "the",
    "a",
    "meeting",
    "about",
    "budget",
    "report",
    "team",
    "plan",
    "review",
    "project",
    "deadline",
    "client",
    "email",
    "call",
    "notes",
    "weekly",
    "update",
    "design",
    "server",
    "deploy",
    "release",
    "ticket",
    "issue",
    "fix",
    "test",
    "build",
    "summary",
    "agenda",
    "morning",
    "afternoon",
    "office",
    "lunch",
    "coffee",
    "travel",
    "flight",
    "hotel",
    "invoice",
    "payment",
    "contract",
    "draft",
    "schedule",
    "calendar",
    "reminder",
    "follow",
    "question",
    "answer",
    "idea",
    "proposal",
    "feedback",
    "launch",
    "market",
    "customer",
    "support",
    "account",
    "login",
    "password",
    "network",
    "backup",
    "storage",
    "data",
    "chart",
    "slide",
    "document",
]

RNG = random.Random(633)
H = " ".join(_words(RNG, 60, HEAD_VOCAB))
P = H + " " + " ".join(_words(RNG, 25, TAIL_VOCAB))
DECOY = " ".join(RNG.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(300)) + " " + H


def _filler(n: int) -> str:
    rng = random.Random(10_000 + n)
    return " ".join(_words(rng, 60, ENGLISH))


def _window_filler(n: int) -> str:
    """Filler that matches a query of ``H`` in FTS through a few of its words."""
    rng = random.Random(20_000 + n)
    return " ".join(_words(rng, 55, ENGLISH) + rng.sample(H.split(), 5))


def _build(root: Path, name: str, drawers: list[tuple[str, str, dict]]) -> Path:
    """A palace of ``drawers`` as ``(id, text, extra metadata)``, filed in list order."""
    from mempalace.palace import get_collection

    palace = root / name / "mempalace"
    palace.mkdir(parents=True)
    collection = get_collection(str(palace), create=True)
    rows = []
    for index, (drawer_id, text, extra) in enumerate(drawers):
        meta = {
            "wing": "conversations",
            "room": "general",
            "source_file": f"{palace}/conversations/{drawer_id}.md",
            "chunk_index": 0,
            "filed_at": f"2026-09-{1 + index // 1440:02d}T{index // 60 % 24:02d}:{index % 60:02d}:00",
        }
        meta.update(extra)
        rows.append((drawer_id, text, meta))
    for start in range(0, len(rows), 200):
        batch = rows[start : start + 200]
        collection.add(
            ids=[row[0] for row in batch],
            documents=[row[1] for row in batch],
            metadatas=[row[2] for row in batch],
        )
    return palace


def _copies(n: int, prefix: str = "copy") -> list[tuple[str, str, dict]]:
    return [(f"{prefix}-{i:03d}", H, {}) for i in range(n)]


def _fillers(n: int) -> list[tuple[str, str, dict]]:
    return [(f"filler-{i:03d}", _filler(i), {}) for i in range(n)]


def _diagnose(palace: Path, drawer_id: str):
    """The mode's facts and verdict for one drawer of ``palace``, with arm 1's pool."""
    from mempalace.palace import get_collection

    provider = MemPalaceMemoryProvider(palace)
    collection = get_collection(str(palace), create=False, read_only=True)
    found = _palace_check.census(collection, palace)
    texts = {row[0]: row[2] for row in found.real}
    probe = Probe(HEAD, drawer_id, texts[drawer_id], _palace_check.query_of(texts[drawer_id]))
    pools = {arm.name: fetch(provider, probe, arm)[0] for arm in (ARM_1, ARM_2, ARM_3)}
    arms = {name: drawer_id in [hit["drawer_id"] for hit in pool] for name, pool in pools.items()}
    facts = _palace_pool.measure(
        provider,
        collection,
        texts,
        found.total,
        probe,
        arms,
        pools[ARM_1.name],
        self_query=_palace_check._self_query,
    )
    return facts, _palace_pool.verdict(facts)


@pytest.fixture(scope="module")
def root(tmp_path_factory, monkeypatch_module):
    return tmp_path_factory.mktemp("palaces")


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    for name in ("MEMPALACE_PALACE_PATH", "MEMPAL_PALACE_PATH", "HARNESS_MEMPALACE_RERANK_MODEL"):
        mp.delenv(name, raising=False)
    yield mp
    mp.undo()


def test_twin(root):
    """25 identical drawers for 20 slots: five miss, each with a copy of itself in the pool."""
    palace = _build(root, "twin", _copies(25, "twin") + _fillers(40))
    verdicts = {}
    for i in range(25):
        facts, word = _diagnose(palace, f"twin-{i:03d}")
        if not facts.arms["1"]:
            verdicts[f"twin-{i:03d}"] = (word, facts.identical)
    assert len(verdicts) == 5
    assert all(word == "twin" and identical == 20 for word, identical in verdicts.values())


def test_near_twin(root):
    """25 drawers that differ by one closing word past their query: five miss, beside near copies."""
    drawers = [(f"near-{i:03d}", f"{P} note{i:03d}", {}) for i in range(25)] + _fillers(40)
    palace = _build(root, "near", drawers)
    missed = []
    for i in range(25):
        facts, word = _diagnose(palace, f"near-{i:03d}")
        if not facts.arms["1"]:
            missed.append((word, facts.identical, facts.nearest[1]))
    assert len(missed) == 5
    for word, identical, nearest in missed:
        assert (word, identical) == ("near-twin", 0)
        assert nearest >= 0.95


def test_crowded(root):
    """``P`` is a candidate through the lexical half and ranks 26th: 25 copies of its head outscore it."""
    palace = _build(root, "crowded", _copies(25) + [("probe", P, {})] + _fillers(40))
    facts, word = _diagnose(palace, "probe")
    assert word == "crowded"
    assert facts.matches
    assert (facts.vector_rank, facts.vector_kept) == (26, 20)
    assert facts.lexical_rank == 26
    assert facts.real_rank == 26
    assert facts.first_ask == 40
    assert facts.arms == {"1": False, "2": False, "3": True}


def test_cut(root):
    """The vector half proposes ``P`` (31st) and keeps 20; 65 lexical equals rank it out of the 60."""
    drawers = (
        _copies(30)
        + [(f"decoy-{i:03d}", DECOY, {}) for i in range(35)]
        + [("probe", P, {})]
        + _fillers(40)
    )
    palace = _build(root, "cut", drawers)
    facts, word = _diagnose(palace, "probe")
    assert word == "cut"
    assert facts.matches
    assert (facts.vector_rank, facts.vector_proposed, facts.vector_kept) == (31, 60, 20)
    assert facts.lexical_rank is None
    assert facts.window_rank == facts.full_rank == 66


def test_unreached(root):
    """70 copies of its head put ``P`` 71st in both halves: neither proposes it."""
    palace = _build(root, "unreached", _copies(70) + [("probe", P, {})] + _fillers(20))
    facts, word = _diagnose(palace, "probe")
    assert word == "unreached"
    assert facts.matches
    assert facts.vector_rank is None
    assert facts.lexical_rank is None
    assert facts.full_rank == 71
    assert facts.first_ask == 80


def test_fts_window(root):
    """``P`` is the best lexical match outside the copies, and is filed after 520 rows the scan reads."""
    drawers = (
        [(f"window-{i:03d}", _window_filler(i), {}) for i in range(520)]
        + _copies(45)
        + [("probe", P, {})]
    )
    palace = _build(root, "window", drawers)
    facts, word = _diagnose(palace, "probe")
    assert word == "fts-window"
    assert facts.matches
    assert facts.window == 500
    assert facts.window_rank is None
    assert facts.full_rank == 46
    assert facts.scored == 566
    assert facts.vector_rank == 46
    assert facts.first_ask == 80


def test_shadowed(root):
    """``P`` shares its file and chunk with a kept copy, so the merge refuses its lexical hit."""
    copies = _copies(20)
    shared = copies[0][2] | {"source_file": "/old/home/conversations/shared.md"}
    copies[0] = (copies[0][0], copies[0][1], shared)
    drawers = copies + [("probe", P, {"source_file": shared["source_file"]})] + _fillers(40)
    palace = _build(root, "shadowed", drawers)
    facts, word = _diagnose(palace, "probe")
    assert word == "shadowed"
    assert facts.matches
    assert facts.shadowed_by == "copy-000"
    assert facts.lexical_rank == 21
    assert facts.vector_rank == 21


def test_shadowed_by_an_earlier_lexical_hit(root):
    """Two copies of ``P`` filed under one path (#606): the merge admits the first and refuses the
    second for it, though the vector half kept neither."""
    shared = {"source_file": "/old/home/conversations/twice.md"}
    drawers = _copies(25) + [("first", P, shared), ("second", P, shared)] + _fillers(40)
    palace = _build(root, "shadowed-lexical", drawers)
    facts, word = _diagnose(palace, "second")
    assert word == "shadowed"
    assert facts.matches
    assert facts.shadowed_by == "first"
    assert facts.vector_rank > facts.vector_kept
    first, first_word = _diagnose(palace, "first")
    assert (first_word, first.real_rank) == ("crowded", 26)


def test_sentinels(root):
    """200 sentinels nearer than ``P`` fill the fetch up to the widening cap of 160."""
    sentinels = [(f"sentinel-{i:03d}", H, {"room": _REGISTRY_ROOM}) for i in range(200)]
    palace = _build(root, "sentinels", sentinels + [("probe", P, {})] + _fillers(20))
    facts, word = _diagnose(palace, "probe")
    assert word == "sentinels"
    assert facts.matches
    assert (facts.fetch, facts.capped, facts.sentinels) == (160, True, 160)
    assert (facts.raw_rank, facts.real_rank, facts.sentinels_above) == (201, 1, 200)


def test_missing_lexical_row(root):
    """``P``'s full-text row is gone: no lexical match for its own text, even with no window."""
    palace = _build(root, "missing", _copies(25) + [("probe", P, {})] + _fillers(40))
    with sqlite3.connect(palace / "chroma.sqlite3") as db:
        (rowid,) = db.execute("SELECT id FROM embeddings WHERE embedding_id = 'probe'").fetchone()
        db.execute("DELETE FROM embedding_fulltext_search WHERE rowid = ?", (rowid,))
    facts, word = _diagnose(palace, "probe")
    assert word == "missing"
    assert facts.full_rank is None
    assert facts.vector_self == "yes"
    assert facts.vector_rank == 26


def test_the_mode_end_to_end(root, capsys):
    """The CLI on the crowded palace: the same verdict, and nothing but ids, counts and ranks."""
    palace = root / "crowded" / "mempalace"
    code = _palace_check.main(["--sample", "100", "--pool-diagnosis", str(palace.parent)])
    out = capsys.readouterr().out
    assert code == 1
    assert "unpooled: drawer probe (head probe); arm 1 no, arm 2 no, arm 3 yes" in out
    lines = [line for line in out.splitlines() if line.startswith("  why: ")]
    assert any(line.startswith("  why: crowded;") for line in lines)
    assert "pool diagnosis summary: head probes 66" in out
    for word in H.split() + P.split():
        assert word not in out
    assert str(root) not in out
    assert "/conversations/" not in out
