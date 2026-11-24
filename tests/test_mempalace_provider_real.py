"""The MemPalace adapter selected by ``mempalace`` (issue #1011), on real palaces.

`tests/test_mempalace_provider.py` pins how the adapter calls MemPalace against a fake. What a fake
cannot say is what MemPalace does with those calls: whether a where-query on the adapter's own keys
finds its drawers, whether union search returns every memory filed under a unique name, whether the
library's lock really refuses a filing while another process holds the palace, and whether the
previous adapter, set back over a palace the new one has filed into, still recalls everything while
its scrub and prune delete nothing.

Marked ``mempalace`` and excluded from the default run: it needs the ``mempalace`` extra and the
embedding model. Run it with ``uv run --extra mempalace pytest -m mempalace``.
"""

from __future__ import annotations

import contextlib
import io
import re
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.mempalace

mempalace = pytest.importorskip("mempalace")

from mempalace import convo_miner  # noqa: E402
from mempalace import palace as mp  # noqa: E402

from basecradle_harness import _prune, _scrub  # noqa: E402
from basecradle_harness._exchange_records import RecordStore, records_dir  # noqa: E402
from basecradle_harness._memory_provider import MemoryExchange, MemoryScope  # noqa: E402
from basecradle_harness._mempalace import _REGISTRY_ROOM, MemPalaceMemoryProvider  # noqa: E402
from basecradle_harness._mempalace_provider import (  # noqa: E402
    ITEM_KEY,
    TIMELINE_KEY,
    MemPalaceProvider,
    render_hits,
)
from basecradle_harness._mining import _LEGACY_RECALL_HEADING  # noqa: E402
from basecradle_harness._wake import WakeAgent  # noqa: E402
from tests.test_prune import START, hourly  # noqa: E402

AGENT = "019a3b2c-4d5e-7f60-8a1b-000000000001"
TIMELINE = "019a3b2c-4d5e-7f60-8a1b-2c3d4e5f6a7b"
ITEMS = [f"019e7800-0000-7000-8000-{n:012x}" for n in range(1, 60)]


def _scope(timeline=TIMELINE):
    return MemoryScope(agent=AGENT, timeline=timeline)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _observe(provider, user, assistant, **kwargs):
    _quiet(
        provider.observe, MemoryExchange(user=user, assistant=assistant, scope=_scope(), **kwargs)
    )


def _new(palace, **kwargs):
    kwargs.setdefault("sleep", lambda seconds: None)
    return MemPalaceProvider(palace, agent="nova", **kwargs)


def _previous(palace):
    return MemPalaceMemoryProvider(palace, agent="nova", name="mempalace-previous")


def _rows(palace: Path) -> list[tuple[str, str, dict]]:
    """Every real drawer as ``(id, text, metadata)``; registry rows are MemPalace's bookkeeping."""
    collection = mp.get_collection(str(palace), create=False, read_only=True)
    found = collection.get(include=["documents", "metadatas"])
    return [
        (drawer, text, meta or {})
        for drawer, text, meta in zip(found.ids, found.documents, found.metadatas, strict=True)
        if (meta or {}).get("room") != _REGISTRY_ROOM
    ]


def _names(palace: Path) -> list[str]:
    return sorted(meta["source_file"] for _, _, meta in _rows(palace))


# --- the name -------------------------------------------------------------------------------------


def test_a_drawer_is_filed_under_its_name_with_its_piece_number_and_found_by_our_keys(tmp_path):
    palace = tmp_path / "palace"
    provider = _new(palace)
    long = "\n".join(f"[2026-11-23T10:00:00Z] john: note {n}: " + "the quick brown fox " * 4
                     for n in range(20))  # fmt: skip

    _observe(provider, "[2026-11-23T10:00:00Z] john: Where do I live?", "Dallas.", item=ITEMS[0])
    _observe(provider, long, "Noted.", item=ITEMS[1])

    names = _names(palace)
    assert f"basecradle:{TIMELINE}:{ITEMS[0]}:1" in names
    pieces = [n for n in names if ITEMS[1] in n]
    assert len(pieces) > 1
    assert sorted(pieces) == sorted(f"basecradle:{TIMELINE}:{ITEMS[1]}:{n}"
                                    for n in range(1, len(pieces) + 1))  # fmt: skip
    collection = mp.get_collection(str(palace))
    found = collection.get(
        where={"$and": [{TIMELINE_KEY: TIMELINE}, {ITEM_KEY: ITEMS[1]}]}, include=["metadatas"]
    )
    assert sorted(m["source_file"] for m in found.metadatas) == sorted(pieces)
    assert not (palace / "conversations").exists()
    assert provider.records.read(TIMELINE, ITEMS[1]).filed


def test_five_keyword_only_memories_on_one_timeline_all_return_under_union_search(tmp_path):
    """A unique name per exchange: the measurement issue #1011 rests on, held on a real palace.

    Five memories carry an exact token and are far, in the embedding, from a query that also talks
    about something two hundred other memories are close to, so the only route into the pool for
    each is lexical. MemPalace's union merge admits a lexical hit only when no hit already in the
    pool carries its ``(source_file, chunk_index)``, and the live write path files every drawer as
    chunk 0. Under one shared name the first vector hit of that name shuts every keyword memory out;
    under a name per exchange each is its own key. The contrast arm files the same memories under one
    shared name.
    """
    token = "QX7-ZEPHYRINE"
    facts = [
        f"[2026-11-23T10:0{n}:00Z] john: {token} lot {n} ships to bay {n + 3}." for n in range(5)
    ]
    noise = [
        f"[2026-11-22T09:00:00Z] john: the weather in Dallas this morning, reading {n}."
        for n in range(200)
    ]
    query = f"{token} what is the weather in Dallas this morning"

    shared = tmp_path / "shared"
    collection = mp.get_collection(str(shared))
    for text in [*noise, *facts]:
        convo_miner.file_conversation_exchange(
            collection, wing="conversations", room="conversations", text=f"> {text}\nNoted.",
            source_file=f"basecradle-shared:{TIMELINE}", agent="nova",
        )  # fmt: skip
    one_name = _previous(shared).search(query, 10)

    named = tmp_path / "named"
    provider = _new(named)
    for n, text in enumerate([*noise, *facts]):
        _observe(provider, text, "Noted.", item=f"019e7900-0000-7000-8000-{n:012x}")
    hits = provider.search(query, 10)

    assert sum(token in hit["text"] for hit in hits) == 5
    assert sum(token in hit["text"] for hit in one_name) <= 1  # the contrast the name exists for


# --- already filed? ---------------------------------------------------------------------------------


def test_a_second_filing_of_the_same_exchange_adds_nothing(tmp_path):
    palace = tmp_path / "palace"
    provider = _new(palace)
    _observe(provider, "[2026-11-23T10:00:00Z] john: Where do I live?", "Dallas.", item=ITEMS[0])
    before = _rows(palace)

    _observe(provider, "[2026-11-23T10:00:00Z] john: Where do I live?", "Dallas.", item=ITEMS[0])
    _new(palace).begin_wake()

    assert _rows(palace) == before


def test_an_old_name_drawer_is_seen_by_already_filed_and_labelled_correctly_at_recall(tmp_path):
    palace = tmp_path / "palace"
    user = "[2026-11-23T10:00:00Z] john: My dog is called Rex."
    _quiet(_previous(palace).observe, MemoryExchange(user=user, assistant="Noted.", scope=_scope()))
    before = _rows(palace)
    provider = _new(palace)

    _observe(provider, user, "Noted.", item=ITEMS[0])

    assert _rows(palace) == before
    assert provider.records.read(TIMELINE, ITEMS[0]) is None
    (hit,) = provider.search("what is my dog called", 1)
    # The timeline is read off the old file's name. The filing time rides only where the installed
    # MemPalace returns `filed_at` with a hit (3.10 and later; 3.9.0, the floor, does not).
    assert re.match(r"- \((filed [^)]* · )?this timeline\) > \[", render_hits([hit], here=TIMELINE))


# --- the door rule ------------------------------------------------------------------------------------


def test_a_near_copy_task_reply_is_neither_filed_nor_recorded_while_the_first_firing_is(tmp_path):
    palace = tmp_path / "palace"
    provider = _new(palace)
    for n in range(3):
        when = START + timedelta(hours=n)
        user, reply = hourly(n, when)
        instructions = user.split("\n", 1)[1]
        _observe(provider, user, reply, item=ITEMS[n], task_instructions=instructions)

    names = _names(palace)
    assert names == [f"basecradle:{TIMELINE}:{ITEMS[0]}:1"]
    assert provider.records.read(TIMELINE, ITEMS[1]) is None
    assert provider.records.read(TIMELINE, ITEMS[2]) is None


# --- the lock ---------------------------------------------------------------------------------------

_HOLD = """
import pathlib, sys, time
from mempalace.palace import mine_palace_lock
with mine_palace_lock(sys.argv[1]):
    pathlib.Path(sys.argv[2]).write_text("held")
    while not pathlib.Path(sys.argv[3]).exists():
        time.sleep(0.05)
"""


@contextlib.contextmanager
def _held(palace: Path, tmp_path: Path):
    """Another process holding the palace's write lock, the way a ``mempalace mine`` does."""
    ready, release = tmp_path / "held", tmp_path / "release"
    holder = subprocess.Popen([sys.executable, "-c", _HOLD, str(palace), str(ready), str(release)])
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert holder.poll() is None, "the lock holder exited before it held the lock"
            assert time.monotonic() < deadline, "the lock holder never held the lock"
            time.sleep(0.05)
        yield
    finally:
        release.write_text("go")
        holder.wait(timeout=30)


def test_a_held_lock_is_retried_then_left_unfiled_and_the_next_wake_files_it_once(tmp_path):
    palace = tmp_path / "palace"
    _observe(_new(palace), "[2026-11-23T09:00:00Z] john: first.", "ok", item=ITEMS[0])  # palace up
    slept: list[float] = []
    provider = _new(palace, sleep=slept.append)

    with _held(palace, tmp_path):
        _observe(provider, "[2026-11-23T10:00:00Z] john: Remember AMBER-FOX.", "Noted.",
                 item=ITEMS[1])  # fmt: skip

    assert slept == [1, 2, 4, 8, 16]
    assert f"basecradle:{TIMELINE}:{ITEMS[1]}:1" not in _names(palace)
    assert provider.records.read(TIMELINE, ITEMS[1]).filed is False

    _new(palace).begin_wake()
    _new(palace).begin_wake()

    assert _names(palace).count(f"basecradle:{TIMELINE}:{ITEMS[1]}:1") == 1
    assert provider.records.read(TIMELINE, ITEMS[1]).filed


# --- the redaction ----------------------------------------------------------------------------------


def test_a_redaction_failure_writes_nothing(tmp_path, monkeypatch):
    palace = tmp_path / "palace"
    provider = _new(palace)

    def broken(text):
        raise RuntimeError("the redaction failed")

    monkeypatch.setattr("basecradle_harness._wake.scrub_counted", broken)
    wake = SimpleNamespace(
        memory_provider=provider, _degraded=False, timeline_uuid=TIMELINE, me_uuid=AGENT
    )

    WakeAgent._observe(
        wake, "[2026-11-23T10:00:00Z] john: my key is k-123", "Noted.", item=ITEMS[0]
    )

    assert not palace.exists() and not records_dir(palace).exists()


# --- the rollback ------------------------------------------------------------------------------------


def test_the_rollback_keeps_every_memory_and_the_previous_scrub_and_prune_delete_nothing(
    tmp_path, monkeypatch
):
    """File under `mempalace`, set back to `mempalace-previous`, then forward again.

    Set back, the previous adapter recalls what both adapters filed, and its scrub and prune hold on
    every new name: the new adapter's drawers include pure scaffolding and a group of routine
    check-ins, exactly what each command would otherwise remove. Forward again, the next exchange is
    filed once.
    """
    palace = tmp_path / "palace"
    previous = _previous(palace)
    _quiet(previous.observe, MemoryExchange(
        user="[2026-11-20T10:00:00Z] john: The staging endpoint moved to eu-west.",
        assistant="Understood.", scope=_scope(),
    ))  # fmt: skip
    new = _new(palace)
    _observe(new, "[2026-11-23T10:00:00Z] john: Your codename is AMBER-FOX-17.", "Noted.",
             item=ITEMS[0])  # fmt: skip
    _observe(new, _LEGACY_RECALL_HEADING, "", item=ITEMS[1])  # scaffolding a scrub would take
    for n in range(8):  # check-ins a prune would group; distinct tasks, so the door files each
        user, reply = hourly(n, START + timedelta(hours=n))
        _observe(new, user, reply, item=ITEMS[10 + n])
    named = [name for name in _names(palace) if name.startswith("basecradle:")]
    assert len(named) == 10
    before = _rows(palace)
    records_before = sorted(path for path, _ in RecordStore(records_dir(palace)).records())

    # Set back.
    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "mempalace-previous")
    texts = " ".join(hit["text"] for hit in previous.search("staging endpoint codename", 10))
    assert "eu-west" in texts and "AMBER-FOX-17" in texts

    assert _quiet(_scrub.main, ["--palace", str(palace), "--apply"]) == 0
    state = tmp_path / "prune-state"
    assert _quiet(_prune.main, ["--palace", str(palace), "--state-dir", str(state)]) == 0
    offer = _prune.offer(mp.get_collection(str(palace)), palace, state,
                         _prune.view_for(palace))  # fmt: skip
    assert offer.groups == []

    assert _rows(palace) == before
    assert sorted(path for path, _ in RecordStore(records_dir(palace)).records()) == records_before

    # Forward again: the next exchange is filed once, under its name.
    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "mempalace")
    forward = _new(palace)
    forward.begin_wake()
    _observe(forward, "[2026-11-24T10:00:00Z] john: The meetup moved to Thursday.", "Noted.",
             item=ITEMS[30])  # fmt: skip
    _observe(forward, "[2026-11-24T10:00:00Z] john: The meetup moved to Thursday.", "Noted.",
             item=ITEMS[30])  # fmt: skip
    assert _names(palace).count(f"basecradle:{TIMELINE}:{ITEMS[30]}:1") == 1
    assert len(_rows(palace)) == len(before) + 1
