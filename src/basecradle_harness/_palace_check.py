"""``basecradle-harness-palace-check`` — prove a MemPalace palace serves from where it is now.

Written for a home-directory rename (issue #606). The fleet renames an agent's OS user in place,
so ``$HARNESS_HOME`` moves and the palace moves with it, while every drawer keeps the absolute
``source_file`` it was filed under. The relocation is right when the palace opens from the new
home and a search returns drawers filed before the move. This command checks exactly that, and
reports what the next observe will do to the palace.

**Token-free.** No platform call and no model call: the palace is resolved from the
``HARNESS_HOME`` given (never from an inherited ``MEMPALACE_PALACE_PATH``), and the only model call
on this path, the reranker, is switched off for the process before the provider is built. The
search is the harness's own `MemPalaceMemoryProvider.search`, so what passes here is what an agent
recalls.

**Read-only by default.** The palace is opened read-only, the forecast is MemPalace's own dry-run
mine, and the searches write nothing. The one exception is the explicit, off-by-default
``--practice-observe`` mode: a practice user has no platform account and can never complete a wake,
so the registry rows a first wake at the new home writes would never exist there. That mode runs
the harness's own observe path once, with a fixed exchange and no model call, so the report that
follows reads the palace as a first wake leaves it. It **writes to the palace** and is for a
practice copy only; its help and its output both say so.

Exit 0 only when every probe drawer comes back; 1 when one does not, or when the palace cannot be
checked at all.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from basecradle_harness._memory_provider import MemoryExchange, MemoryScope, _palace_path
from basecradle_harness._mempalace import (
    _CONVERSATIONS_WING,
    _REGISTRY_ROOM,
    MemPalaceMemoryProvider,
    _import,
)
from basecradle_harness._rerank import RERANK_MODEL_VAR, SURFACE_TOOL
from basecradle_harness._version import __version__

PROG = "basecradle-harness-palace-check"

#: Cleared for this process before anything is resolved: the reranker's switch (the only model
#: call a search can make) and MemPalace's own palace-path variables, which would otherwise outrank
#: the ``HARNESS_HOME`` the caller named.
_CLEARED_ENV = (RERANK_MODEL_VAR, "MEMPALACE_PALACE_PATH", "MEMPAL_PALACE_PATH")

#: The exchange ``--practice-observe`` writes. It says what wrote it, so a reader of the practice
#: palace can never take it for something a peer said.
PRACTICE_USER = f"{PROG} --practice-observe"
PRACTICE_REPLY = f"Written by {PROG} on a practice copy of a palace, to exercise observe."

#: How much of a probe drawer's text becomes its query. The whole chunk is at most ~800 characters;
#: its head is distinctive enough to find it, and short enough to stay an ordinary query.
_QUERY_CHARS = 400

#: How deep a probe looks for its drawer, and how deep the sentinel count reads.
_TOP = 10

_PAGE = 1000


@dataclass
class Census:
    """What the palace holds, read off its metadata in one pass."""

    total: int = 0
    registry: int = 0
    #: Real drawers as ``(drawer_id, metadata, document)``.
    real: list[tuple[str, dict, str]] = field(default_factory=list)
    #: Real drawers whose ``source_file`` is not in this palace's ``conversations`` directory.
    elsewhere: list[tuple[str, dict, str]] = field(default_factory=list)
    #: Real chunk-0 drawers with no ``content_hash`` (MemPalace before 3.7). After a move, the next
    #: observe re-mines their files as new drawers instead of recognising them.
    unhashed: int = 0
    files: int = 0
    #: Conversation files on disk with no row recorded under their current path. The next observe
    #: reads every one of them.
    unregistered: int = 0


def census(collection, palace: Path) -> Census:
    """Count drawers, registry rows and files, and collect the real drawers to probe."""
    convos = palace / _CONVERSATIONS_WING
    here = convos.resolve()
    prefix = str(here) + os.sep
    out = Census()
    recorded: set[str] = set()
    offset = 0
    while True:
        page = collection.get(limit=_PAGE, offset=offset, include=["metadatas", "documents"])
        ids = list(page.get("ids") or [])
        if not ids:
            break
        metadatas = list(page.get("metadatas") or [])
        documents = list(page.get("documents") or [])
        for index, drawer_id in enumerate(ids):
            out.total += 1
            meta = (metadatas[index] if index < len(metadatas) else None) or {}
            text = (documents[index] if index < len(documents) else None) or ""
            source = str(meta.get("source_file") or "")
            if source.startswith(prefix):
                recorded.add(source)
            if meta.get("room") == _REGISTRY_ROOM:
                out.registry += 1
                continue
            row = (drawer_id, meta, text)
            out.real.append(row)
            if not source.startswith(prefix):
                out.elsewhere.append(row)
            if meta.get("chunk_index", 0) == 0 and not meta.get("content_hash"):
                out.unhashed += 1
        offset += len(ids)
    on_disk = sorted(convos.glob("*.md")) if convos.is_dir() else []
    out.files = len(on_disk)
    out.unregistered = sum(str(here / path.name) not in recorded for path in on_disk)
    return out


def forecast(palace: Path, agent: str) -> tuple[int, int] | None:
    """What the next observe will do: ``(files given a registry row, files mined as new)``.

    MemPalace's own dry-run mine of the directory, with the adapter's arguments, read off the
    per-file lines it prints. ``None`` when those lines cannot be read, so a changed upstream format
    is reported as unknown rather than as a forecast of nothing.
    """
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _import("convo_miner").mine_convos(
            str(palace / _CONVERSATIONS_WING),
            str(palace),
            wing=_CONVERSATIONS_WING,
            agent=agent,
            extract_mode="exchange",
            dry_run=True,
        )
    lines = out.getvalue().splitlines()
    duplicate = sum("duplicate of" in line for line in lines)
    new = sum("[DRY RUN]" in line and "->" in line for line in lines)
    return (duplicate, new) if duplicate or new else None


def practice_observe(provider: MemPalaceMemoryProvider) -> float:
    """Run the harness's own observe once, with a fixed exchange. Returns the seconds it took."""
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        provider.observe(
            MemoryExchange(
                user=PRACTICE_USER,
                assistant=PRACTICE_REPLY,
                scope=MemoryScope(agent=PROG, timeline="practice", query=None),
            )
        )
    return time.monotonic() - started


def probes(found: Census) -> tuple[list[tuple[str, dict, str]], bool]:
    """Three real drawers to search for, oldest, middle and newest by ``filed_at``.

    Drawn from drawers filed at another location when there are any (after a move, those are the
    ones that matter), else from all real drawers. The flag says which.
    """
    pool = list(found.elsewhere or found.real)
    pool.sort(key=lambda row: (str(row[1].get("filed_at", "")), row[0]))
    if not pool:
        return [], bool(found.elsewhere)
    picked = {row[0]: row for row in (pool[0], pool[len(pool) // 2], pool[-1])}
    return list(picked.values()), bool(found.elsewhere)


def main(argv: list[str] | None = None) -> int:
    """The ``basecradle-harness-palace-check`` entrypoint."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Prove an agent's MemPalace palace opens from the given HARNESS_HOME and returns drawers "
            "filed before a home-directory move. No platform call, no model call. Read-only unless "
            "--practice-observe is given. Exit 0 only if every probe drawer comes back."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"{PROG} {__version__}", help="print the version."
    )
    parser.add_argument(
        "harness_home",
        nargs="?",
        help="the agent's HARNESS_HOME. Defaults to $HARNESS_HOME.",
    )
    parser.add_argument(
        "--practice-observe",
        action="store_true",
        help=(
            "WRITES TO THE PALACE. For a practice copy only, never an agent's live palace. Runs the "
            "harness's own observe once, with a fixed exchange and no model call, before the "
            "report, so the report reads the palace as a first wake at this home leaves it."
        ),
    )
    args = parser.parse_args(argv)

    for name in _CLEARED_ENV:
        os.environ.pop(name, None)
    raw_home = args.harness_home or os.environ.get("HARNESS_HOME")
    if not raw_home:
        print(f"{PROG}: name a HARNESS_HOME, or set $HARNESS_HOME", file=sys.stderr)
        return 1
    home = Path(raw_home).expanduser().resolve()
    palace = _palace_path(home)
    print(f"palace: {palace}")
    if not palace.is_dir():
        print(f"FAIL: no palace at {palace}")
        return 1

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        provider = MemPalaceMemoryProvider(palace)
        if args.practice_observe:
            seconds = practice_observe(provider)
            print(
                f"practice observe: WROTE one exchange to this palace and mined it "
                f"({seconds:.1f}s). Practice copies only."
            )
        try:
            collection = _import("palace").get_collection(str(palace), create=False, read_only=True)
        except ImportError:
            raise
        except Exception as error:  # noqa: BLE001 - a vendor refusal of any class, relayed in one line
            print(f"FAIL: could not open the palace: {type(error).__name__}: {error}")
            return 1
        found = census(collection, palace)
        print(
            f"drawers: {found.total} total, {len(found.real)} real, "
            f"{found.registry} registry sentinels"
        )
        print(f"real drawers filed at another location: {len(found.elsewhere)}")
        print(f"real chunk-0 drawers without content_hash (pre-3.7 schema): {found.unhashed}")
        print(
            f"conversation files: {found.files}, "
            f"not yet registered at this location: {found.unregistered}"
        )
        if found.unregistered:
            predicted = forecast(palace, provider.agent)
            if predicted is None:
                print(
                    "next observe (dry run): unknown, MemPalace's dry-run output was not readable"
                )
            else:
                print(
                    f"next observe (dry run): {predicted[0]} files -> registry sentinel, "
                    f"{predicted[1]} files -> mined as new drawers"
                )
        rows, elsewhere = probes(found)
        if not rows:
            print("FAIL: the palace holds no real drawers to search for")
            return 1
        failed = 0
        for drawer_id, meta, text in rows:
            hits = provider.search(text.strip()[:_QUERY_CHARS], _TOP, surface=SURFACE_TOOL)
            ok = drawer_id in [hit.get("drawer_id") for hit in hits]
            failed += not ok
            print(
                f"{'PASS' if ok else 'FAIL'}: drawer {drawer_id} (filed {meta.get('filed_at')}, "
                f"source {meta.get('source_file')})"
            )
        where = "filed at another location" if elsewhere else "any location (none filed elsewhere)"
        print(f"probes drawn from drawers {where}")
        # Last, because it is the line the relocation is judged on: what recall hands the agent for
        # its own home's name. With issue #606's fix it is 0 whatever the palace holds.
        name = home.parent.name
        hits = provider.search(name, _TOP, surface=SURFACE_TOOL)
        junk = sum(str(hit.get("text", "")).startswith("[registry]") for hit in hits)
        print(f"registry sentinels in top {_TOP} for query {name!r}: {junk}")
    except ImportError as error:
        print(f"{PROG}: {error}", file=sys.stderr)
        return 1
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
