"""The observe's write path on a real MemPalace palace (issue #741).

`test_mempalace.py` pins how the adapter calls MemPalace against a fake. What a fake cannot say is
what MemPalace then does with it: whether the drawer holds every line, whether MemPalace's own miner
agrees that a file the observe filed is already filed, and whether a fact late in a long message
can still be recalled once it is filed. Those are measurements of MemPalace, so this builds real
palaces with the installed MemPalace and its embedding model.

Marked ``mempalace`` and excluded from the default run: it needs the ``mempalace`` extra, and the
embedding model it downloads on first use. Run it with ``uv run --extra mempalace pytest -m
mempalace``.
"""

from __future__ import annotations

import contextlib
import io
import random
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
    EXCHANGE_PIECE_CHARS,
    ID_SOURCE_FILE,
    MemPalaceMemoryProvider,
    content_hashes,
    filed_by_observe,
    recognises_content,
)

TIMELINE = "019a3b2c-4d5e-7f60-8a1b-2c3d4e5f6a7b"
SCOPE = MemoryScope(agent="019a3b2c-4d5e-7f60-8a1b-000000000001", timeline=TIMELINE)

BRIEF = (
    "[2026-10-31T10:36:46.734Z] john: Opening brief for a memory test. Please read all of it.\n"
    "1. For this test, your codename is AMBER-FOX-17.\n"
    "2. Lyra.\n"
    "3. Your favorite constellation is the one named on line 2."
)


def _observe(provider, user, assistant):
    with contextlib.redirect_stdout(io.StringIO()):
        provider.observe(MemoryExchange(user=user, assistant=assistant, scope=SCOPE))


def _rows(palace: Path) -> tuple[list[tuple[str, dict]], int]:
    """The palace's real drawers as ``(text, metadata)``, and how many registry rows it holds."""
    collection = mp.get_collection(str(palace), create=False, read_only=True)
    page = collection.get(include=["documents", "metadatas"])
    real, registry = [], 0
    for text, meta in zip(page["documents"], page["metadatas"]):
        if (meta or {}).get("room") == _REGISTRY_ROOM:
            registry += 1
        else:
            real.append((text, meta or {}))
    return real, registry


def test_a_multiline_brief_is_one_drawer_with_every_line_and_its_provenance(tmp_path):
    """The defect, on a real palace: four lines were four drawers, `2. Lyra.` was dropped."""
    palace = tmp_path / "palace"
    provider = MemPalaceMemoryProvider(palace, agent="nova")

    _observe(provider, BRIEF, "Got it.\n\nAMBER-FOX-17\n\nLyra.")

    ((text, meta),) = _rows(palace)[0]
    for line in BRIEF.splitlines():
        assert f"> {line}" in text
    assert text.endswith("Got it.\n\nAMBER-FOX-17\n\nLyra.")  # the reply's short paragraphs too
    assert meta["chunk_total"] == 1 and meta["extract_mode"] == "exchange"

    hits = provider.search("[2026-10-31T12:00:00Z] john: what is your favorite constellation?")
    assert hits and "2. Lyra." in hits[0]["text"]


def test_mempalaces_own_miner_files_nothing_the_observe_filed(tmp_path):
    """A `mempalace mine` of the directory must skip every file the observe wrote, long or short."""
    palace = tmp_path / "palace"
    provider = MemPalaceMemoryProvider(palace, agent="nova")
    _observe(provider, BRIEF, "Got it.")
    _observe(
        provider,
        "Hand over the runbook.",
        "\n\n".join(f"Step {n}. " + "do it " * 40 for n in range(12)),
    )
    _observe(provider, "", "A reply to nobody in particular.")
    before, registry = _rows(palace)
    assert len(before) > 3  # the runbook took more than one drawer
    assert all(len(text) <= EXCHANGE_PIECE_CHARS for text, _ in before)

    predicted = _palace_check.forecast(palace, provider.agent)
    assert predicted is not None and predicted.new == [] and predicted.recognised == 3
    with contextlib.redirect_stdout(io.StringIO()):
        convo_miner.mine_convos(
            str(palace / "conversations"),
            str(palace),
            wing="conversations",
            agent="nova",
            extract_mode="exchange",
        )

    after, registry_after = _rows(palace)
    assert sorted(text for text, _ in after) == sorted(text for text, _ in before)
    assert registry_after == registry == 0


def test_a_filing_short_of_its_drawers_is_never_read_as_filed(tmp_path):
    """``chunk_total`` is what makes a set cut short visible: one drawer gone, the file is unfiled.

    A process killed between two drawers of one exchange leaves exactly this. Without the count
    MemPalace trusts any surviving drawer, and the missing part is never filed by anything.
    """
    palace = tmp_path / "palace"
    provider = MemPalaceMemoryProvider(palace, agent="nova")
    _observe(provider, "Go on.", "\n\n".join(f"Step {n}. " + "do it " * 40 for n in range(6)))
    collection = mp.get_collection(str(palace))
    ids = collection.get()["ids"]
    assert len(ids) > 1
    collection.delete(ids=ids[:1])

    # The file is still registered at this location, so the census must name it as short, or the
    # report never runs the forecast at all.
    found = _palace_check.census(
        mp.get_collection(str(palace), create=False, read_only=True), palace
    )
    predicted = _palace_check.forecast(palace, provider.agent)

    assert (found.unregistered, found.short) == (0, 1)
    assert predicted is not None and len(predicted.new) == 1


#: The hand-named file a practice move filed twice (basecradle-noc#899): written into
#: ``conversations/`` by hand, not by the observe, and mined on its own into a wing of its own, as
#: `@rowan`'s restored charter was. A mine of the directory after a move filed its five drawers a
#: second time, in ``conversations``.
HAND = "restored-charter-principles.md"
HAND_WING = "restored_charter"
HAND_TEXT = "\n\n".join(
    f"> [2026-07-0{n}T09:00:00Z] john: principle {n} of the charter, kept by hand.\n"
    + f"Principle {n}: "
    + "the agent keeps what it was told and says where it was told it. " * 6
    for n in range(1, 6)
)


def _mine_one(path: Path, palace: Path, wing: str) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        convo_miner.mine_convos(
            str(path), str(palace), wing=wing, agent="nova", extract_mode="exchange"
        )


def _hash_as_mined_before_3_12(palace: Path, path: Path) -> None:
    """Stamp ``content_hash`` on the file's chunk 0 the way MemPalace's miner did from 3.7 to 3.11.

    From 3.12 the miner stamps none, so a file it mines there has nothing to prove it after a move.
    Every miner-filed drawer on the fleet was filed before 3.12 and carries one: this gives the
    test's file the shape the fleet's has. A no-op where the installed miner stamped it itself.
    """
    collection = mp.get_collection(str(palace))
    found = collection.get(where={"source_file": str(path.resolve())}, include=["metadatas"])
    for drawer_id, meta in zip(found["ids"], found["metadatas"]):
        if meta.get("chunk_index") == 0 and not meta.get("content_hash"):
            collection.update(
                ids=[drawer_id],
                metadatas=[{**meta, "content_hash": ",".join(content_hashes(path))}],
            )


def _all_rows(palace: Path) -> dict[str, tuple[str, dict]]:
    collection = mp.get_collection(str(palace), create=False, read_only=True)
    page = collection.get(include=["documents", "metadatas"])
    return dict(zip(page["ids"], zip(page["documents"], page["metadatas"])))


def test_a_moved_home_is_recognised_and_never_filed_again(tmp_path, capsys):
    """After a move, the forecast lists no exchange as new, and a mine files nothing again.

    The acceptance test of issue #989 (`@briggs`'s rule, point 5). Every drawer records its absolute
    path, so after a move a path no longer finds it. Before MemPalace 3.12 the miner recognises
    each file by its content hash, and the hand-named file in its own wing by the registry row
    ``--register-off-wing`` writes for it (#613). From 3.12 the miner recognises a file by its
    path alone, so ``--relocate-source-file`` records each one at its new path first, with the
    agent's wakes held, after proving it is the same file (#989). Both run here, as the NOC's
    rename flow runs them; each is a no-op where the other is the remedy.

    Then a real mine: no drawer is added, none is deleted, and none of them changes but for what
    the relocation wrote. A registry row a pre-3.12 mine writes is never served as a memory.

    Both exchanges are longer than the miner's 30-character floor: MemPalace 3.9 and 3.10 skip a
    shorter file without a line (a real mine registers it, filing no drawer), so it would be
    counted as neither.
    """
    previous = tmp_path / "olduser" / "harness"
    old = previous / "mempalace"
    provider = MemPalaceMemoryProvider(old, agent="nova")
    _observe(provider, BRIEF, "Got it.")
    _observe(provider, "Where do I live these days?", "You live in Dallas, Texas.")
    hand = old / "conversations" / HAND
    hand.write_text(HAND_TEXT, encoding="utf-8")
    _mine_one(hand, old, HAND_WING)
    _hash_as_mined_before_3_12(old, hand)
    home = tmp_path / "newuser" / "harness"
    home.mkdir(parents=True)
    shutil.move(str(old), str(home / "mempalace"))
    new = home / "mempalace"
    before = _all_rows(new)
    real = {i for i, (_, meta) in before.items() if meta.get("room") != _REGISTRY_ROOM}
    assert len(real) > 3  # the hand-named file took several drawers

    for flags in (
        ["--relocate-source-file", "--previous-home", str(previous)],
        ["--register-off-wing"],
    ):
        capsys.readouterr()
        code = _palace_check.main([str(home), *flags])
        out = capsys.readouterr().out
        assert code == 0, out  # a read-back FAIL exits 1 before the report
        assert "  left alone:" not in out and "refused" not in out, out

    predicted = _palace_check.forecast(new, provider.agent)
    assert predicted is not None
    assert predicted.new == [] and predicted.recognised == 3

    moved = _all_rows(new)
    _mine(new)
    after = _all_rows(new)
    assert {i: after[i] for i in real} == {
        i: moved[i] for i in real
    }  # nothing changed, nothing gone
    added = set(after) - set(moved)
    assert all(after[i][1].get("room") == _REGISTRY_ROOM for i in added)  # no new drawer
    if not recognises_content():
        # Recorded here by the relocation, every one of them, and still provably the observe's.
        here = (new / "conversations").resolve()
        assert {Path(str(after[i][1]["source_file"])).parent for i in real} == {here}
        ours = {i for i in real if filed_by_observe(i, after[i][1], after[i][0])}
        assert len(ours) == 2
        assert {Path(str(after[i][1][ID_SOURCE_FILE])).parent for i in ours} == {
            (old / "conversations").resolve()
        }
    else:
        assert {i: after[i] for i in real} == {i: before[i] for i in real}  # nothing to relocate
    hits = MemPalaceMemoryProvider(new, agent="nova").search("newuser charter principle", 10)
    assert hits and not any(str(hit.get("text")).startswith("[registry]") for hit in hits)
    _mine(new)  # and a second mine files nothing either
    assert _all_rows(new) == after


def _mine(palace: Path) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        convo_miner.mine_convos(
            str(palace / "conversations"),
            str(palace),
            wing="conversations",
            agent="nova",
            extract_mode="exchange",
        )


# --- a fact late in a long message (issue #741, the capital's measurement) -----------------

RUNBOOK = [
    "Here is the full handoff for the staging migration, since you are taking it over tomorrow.",
    (
        "The cluster runs three app nodes behind the load balancer, and the database is a single "
        "primary with one replica in the other zone. Backups run nightly at two, and the "
        "retention is fourteen days."
    ),
    (
        "The migration has four phases. First we freeze writes on the admin tool, then we copy "
        "the archive tables, then we swap the read path, and finally we drop the old columns once "
        "a week has passed without errors."
    ),
    (
        "The monitoring dashboards are in the shared folder. The alerts that matter are "
        "replication lag over thirty seconds and the queue depth on the export workers."
    ),
    (
        "If the copy step stalls, it is almost always the large attachments table. Pause the "
        "export workers, let the copy catch up, then resume them one at a time."
    ),
    (
        "Rollback is simple as long as we have not dropped the columns: swap the read path back "
        "and unfreeze writes. After the drop, rollback means restoring from the nightly backup."
    ),
]
LATE_FACT = (
    "One last thing that is not in the runbook: the vault passphrase for the staging cluster is "
    "cobalt-heron-42, and my sister's dog is named after a famous jazz trumpeter."
)
NEAR = [
    "the staging cluster replica fell behind again and replication lag hit forty seconds",
    "the migration runbook needs a section on the archive tables and the export workers",
    "the vault on the production cluster rotates its passphrase every ninety days",
    "my neighbour's dog is named after a famous painter and barks at the mail carrier",
    "the backup restore drill took five hours and the retention policy needs review",
    "the jazz club downtown has a trumpet player every Friday night",
]
TOPICS = [
    "the quarterly budget review moved to Thursday and finance wants the forecast early",
    "we hiked the ridge trail on Saturday and the weather turned cold by noon",
    "the soup needs more salt and the bread should rest an hour before baking",
    "the tomatoes are ripening but the squash leaves have powdery mildew",
    "the flight to Denver was delayed and the rental car desk had closed",
]


@pytest.mark.parametrize(
    ("query", "needle"),
    [
        ("what is the staging vault passphrase?", "cobalt-heron-42"),
        ("what is my sister's dog named after?", "jazz trumpeter"),
    ],
)
def test_a_fact_at_the_end_of_a_long_message_is_still_recalled(tmp_path, query, needle):
    """Filed whole, this ~4.2k-character message's last paragraph was not in the top ten (measured).

    The embedding model reads only the first 256 word pieces of a drawer, so a fact past them is
    invisible to the vector half, and BM25 scores a long drawer down. Split at
    `EXCHANGE_PIECE_CHARS`, the paragraph that states it is a drawer of its own, competing
    against near-topic memories that share its words.
    """
    rng = random.Random(741)
    palace = tmp_path / "palace"
    provider = MemPalaceMemoryProvider(palace, agent="nova")
    for n in range(80):
        line = rng.choice(TOPICS)
        _observe(provider, f"[2026-10-{n % 28 + 1:02d}T09:00:00Z] john: {line}.", f"Noted: {line}.")
    for n in range(30):
        said, reply = rng.choice(NEAR), rng.choice(NEAR)
        _observe(
            provider, f"[2026-10-{n % 28 + 1:02d}T11:00:00Z] john: {said}.", f"Noted. {reply}."
        )
    message = "[2026-10-30T15:00:00Z] john: " + "\n\n".join(RUNBOOK * 4 + [LATE_FACT])
    assert len(message) > 4 * EXCHANGE_PIECE_CHARS
    _observe(provider, message, "Thanks, I have it.")

    hits = provider.search(f"[2026-10-31T12:00:00Z] john: {query}", 10)

    assert any(needle in hit["text"] for hit in hits)
