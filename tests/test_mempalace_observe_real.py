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
    MemPalaceMemoryProvider,
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

    assert _palace_check.forecast(palace, provider.agent) is None  # nothing new, nothing moved
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

    predicted = _palace_check.forecast(palace, provider.agent)

    assert predicted is not None and len(predicted.new) == 1


def test_a_moved_home_is_recognised_by_content_and_never_filed_again(tmp_path):
    """Every drawer records its absolute path, so after a move only the content hash recognises it.

    Both exchanges are longer than the miner's 30-character floor: MemPalace 3.9 and 3.10 skip a
    shorter file without a line (a real mine registers it, filing no drawer), so it would be
    counted as neither.
    """
    old = tmp_path / "olduser" / "palace"
    provider = MemPalaceMemoryProvider(old, agent="nova")
    _observe(provider, BRIEF, "Got it.")
    _observe(provider, "Where do I live these days?", "You live in Dallas, Texas.")
    new = tmp_path / "newuser" / "palace"
    new.parent.mkdir()
    shutil.move(str(old), str(new))

    predicted = _palace_check.forecast(new, provider.agent)

    assert predicted is not None
    assert predicted.new == [] and predicted.duplicate == 2  # one registry row each, no drawer


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
