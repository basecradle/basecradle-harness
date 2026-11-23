"""``basecradle-harness-palace-check`` — prove a MemPalace palace serves from where it is now.

Written for a home-directory rename (issue #606). The fleet renames an agent's OS user in place,
so ``$HARNESS_HOME`` moves and the palace moves with it, while every drawer keeps the absolute
``source_file`` it was filed under. The relocation is right when the palace opens from the new
home and a search returns drawers filed before the move. This command checks exactly that, and
reports what a ``mempalace mine`` of the conversation files would do to the palace.

**The forecast is of a mine, not of the observe** (issue #741). Until 0.164.0 the observe mined the
whole ``conversations`` directory on every exchange, so a dry-run mine was a forecast of the next
wake. The observe now files only the exchange it was handed, through MemPalace's live-agent write
path, and never reads another file; what is left to forecast is what MemPalace's own miner would do
if anything ran it there, which is the one way a file could still be filed twice. Two of the lines
this prints are parsed by the NOC's rename flow and keep their wording byte for byte, the first of
them still beginning ``next observe (dry run):`` (`forecast`).

**Token-free, except in one mode.** No platform call and no model call: the palace is resolved
from the ``HARNESS_HOME`` given (never from an inherited ``MEMPALACE_PALACE_PATH``), and the only
model call on this path, the reranker, is switched off for the process before the provider is
built. The search is the harness's own `MemPalaceMemoryProvider.search`, so what passes here is
what an agent recalls. The one exception is the explicit ``--end-to-end`` mode below, which calls
the agent's own rerank model and is bounded by a ``--token-ceiling`` it cannot run without.

**Read-only by default.** The palace is opened read-only, the forecast is MemPalace's own dry-run
mine, and the searches write nothing: no drawer, row, metadata value or embedding changes. The
palace's *files* are another matter: ChromaDB rewrites bytes in ``chroma.sqlite3`` and its index
segment on every open, even MemPalace's ``read_only=True`` one, so a checksum of the palace
directory moves while its contents do not (measured with MemPalace 3.9.0). One exception is the explicit, off-by-default
``--practice-observe`` mode: a practice user has no platform account and can never complete a wake,
so nothing would ever file an exchange at the new home. That mode runs the harness's own observe
path once, with a fixed exchange and no model call, so the report that follows reads the palace as
a first wake leaves it: one new file, filed and recognised by the forecast as filed. It **writes to
the palace** and is for a practice copy only; its help and its output both say so.

**One write for a real move: ``--register-off-wing``** (issue #613). MemPalace recognises a moved
file's content only within the mining wing, so a conversation file once mined on its own into
another wing is filed again in ``conversations`` by a mine after a move (the first wake's observe,
before 0.164.0). Run once after the rename and before that wake, this mode registers each such file in the wing its drawers
already carry: one registry row per file, no drawer, no drawer touched (`register_off_wing`). It
**writes to the palace**, refuses and writes nothing if any file would file a drawer, and has a
``--dry-run`` form; its help and its output say so.

**One write for a MemPalace upgrade: ``--stamp-chunker-version``** (issue #751). MemPalace 3.11
reads an exchange drawer with no ``convo_chunker_version`` as filed by an older chunker, and 3.10
and earlier never wrote one, so every drawer filed before the upgrade drops out of its record of
what is filed. This mode stamps the installed version on the drawers the observe filed
(`stamp_chunker_version`), identified by their id recipe (`filed_by_observe`), and on nothing
else: MemPalace's miner's drawers really were cut by the older chunker, so they are left to
MemPalace and reported. Every report counts both kinds on one line (`stale_line`). It **writes to
the palace** and has a ``--dry-run`` form.

**One write for a moved palace: ``--relocate-source-file``** (issues #831, #989). A mine purges a
file's drawers by their ``source_file``, so a file whose drawers are recorded under a previous home
is filed a second time beside them. This mode rewrites ``source_file`` on those drawers to this
location, so the mine that follows is the in-place case, or, for a current file, a skip
(`relocate_source_file`). On 3.11 that is needed for MemPalace's miner's version-stale files; on
3.12, which recognises a file by its path alone, for every file, the observe's included, which
keep the path their id was derived from (`ID_SOURCE_FILE`). It takes the previous home as an
argument, proves each file separately, **writes to the palace**, deletes nothing, and has a
``--dry-run`` form.

**What MemPalace recognises decides which write a move needs** (issue #989). Before 3.12 a mine
recognises a moved file by its content hash in the mining wing (`recognises_content`), so a moved
palace needs ``--register-off-wing`` for an off-wing file and ``--relocate-source-file`` for a
version-stale one. 3.12 removed that on purpose (MemPalace#2800) and recognises a file at its
recorded path only, so there ``--register-off-wing`` has nothing to register and
``--relocate-source-file`` records every provable file here. 3.12 also raised ``normalize_version``
to 3, which makes every drawer filed before it unfiled to a mine: the report counts that class on
its own line (`normalize_line`), and nothing here writes the version (issue #1001).

**Three probes, or a sample** (issue #606, phase 3). By default the check searches for three real
drawers (the oldest, middle and newest), which says whether a moved palace serves at all. Three
cannot say whether a move made recall *worse*, so ``--sample N`` probes N drawers instead, chosen
by the SHA-256 of their drawer id: a choice that does not depend on where the palace sits, so two
palaces holding the same drawers probe the same ones, and ``--filed-before`` holds the population
still while a live palace keeps growing. The sample ends with one comparable summary line, and two
reports with the same sample digest and the same failing digest failed on the same drawers.

**The reranked path, with no model call: ``--sample N --reranked-pool``** (issue #617). The sample
above runs the search with the reranker off, which is not the path a fleet agent with a reranker
takes: there `search` asks MemPalace for ``pool_size(10)`` = 20 and a model picks 10 of them, so a
drawer outside those 20 can never be recalled. This mode answers, for the same sample, whether each
probe's own drawer is among the 20 handed to the reranker today (arm A), and among the first 20 of
an ask for 40 (arm B, the candidate rule), and reports how many each arm finds and how many one
finds that the other misses. Both arms fetch through the provider's own ranking call, so registry
sentinels are dropped as `search` drops them; the reranker itself never runs. Exit 0 only when arm
A holds every probe drawer.

**The final 10, through the real reranker: ``--sample N --end-to-end``** (issue #627). Whether a
probe's drawer reaches the pool says nothing about whether the model then picks it. This mode runs
each probe through the agent's own reranker, in four arms (today's search, which passes
``max_distance=2.0``; the same again for the noise floor; today's ask without it, the search before
issue #625; and twice the ask with it), for head probes and
for rare-token probes, and reports where each probe's own drawer lands. It spends rerank-model
tokens, so it takes a ``--token-ceiling``, refuses to start when its estimate exceeds it, and stops
(reporting what it has, marked partial) when the next probe would. ``--dry-run`` prints the estimate
and calls no model. See `basecradle_harness._palace_recall`.

**Why a probe is not in the pool: ``--sample N --pool-diagnosis``** (issue #633). Token-free and
read-only. The same probes as ``--end-to-end`` and the same three fetches, and for every probe whose
drawer is not in arm 1's pool, where it ranks in the vector half and the lexical half and how far
from the cut, whether a wider ask reaches it and at which, whether a copy or near copy took its
place, whether sentinels or the widening cap kept it out, and whether an index has lost it. See
`basecradle_harness._palace_pool`.

**A miss is diagnosed, never printed.** A probe that does not come back gets a ``why:`` line made of
ids, counts and ranks: how many drawers carry its exact text or its query, what took the top slots,
where it ranks in a deeper fetch and in the vector index alone, and whether the index finds it by
its own embedding. That separates the four ways a drawer can miss (see `diagnose`). No drawer text
is ever printed.

Exit 0 only when every probe drawer comes back; 1 when one does not, or when the palace cannot be
checked at all.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import logging
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from basecradle_harness._memory_provider import MemoryExchange, MemoryScope, _palace_path
from basecradle_harness._mempalace import (
    _CONVERSATIONS_WING,
    _REGISTRY_ROOM,
    ID_SOURCE_FILE,
    MemPalaceMemoryProvider,
    _import,
    candidate_pool,
    chunker_version,
    content_hashes,
    filed_by_observe,
    mempalace_version,
    recognises_content,
)
from basecradle_harness._observability import last_word, log_handler
from basecradle_harness._rerank import RERANK_MODEL_VAR, SURFACE_TOOL, reranker_from_env
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

#: How deep a miss is looked for. Deep enough that a drawer merely crowded out of the top 10 is
#: found, so "not in the top 100" means recall cannot reach it in practice.
_DEEP = 100

#: How many results a top-10 search asks MemPalace for: the harness's own ask, with the unranked
#: headroom (issue #611; this check runs with the reranker off), so the verdicts below are read
#: against the search an agent actually makes.
_ASK = candidate_pool(_TOP, reranked=False)

#: How many candidates that search asks the vector index for: MemPalace 3.9 asks for three times
#: the requested count (``_candidate_pool_size``) and, in union mode, keeps the nearest ``_ASK``.
_VECTOR_ASK = 3 * _ASK

#: The pool a reranked top-10 search hands its reranker: the harness's own ask with a reranker
#: bound (`pool_size`), all of it kept. Arm A of ``--reranked-pool`` (issue #617).
_RERANK_POOL = candidate_pool(_TOP, reranked=True)

#: Arm B of ``--reranked-pool``, the candidate rule: ask MemPalace for twice that pool and hand the
#: reranker the first `_RERANK_POOL` of the ranking.
_CANDIDATE_ASK = 2 * _RERANK_POOL

#: Hex characters of a digest. Sixty-four bits: two reports compared by eye never collide by chance.
_DIGEST_CHARS = 16

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
    #: Real chunk-0 drawers with no ``content_hash`` (MemPalace's miner stamps none before 3.7 or
    #: from 3.12, and nothing then proves the file after a move, issue #989). After a move, a mine
    #: re-files their files as new drawers instead of recognising them.
    unhashed: int = 0
    files: int = 0
    #: Conversation files on disk with no row recorded under their current path. A mine of the
    #: directory reads every one of them; the observe, since 0.164.0, none.
    unregistered: int = 0
    #: Conversation files on disk whose drawers at this location fall short of the ``chunk_total``
    #: they record (issue #741): a filing cut off partway. MemPalace's miner reads such a file as
    #: unfiled, and a mine of the directory deletes what is there and files it again.
    short: int = 0
    #: The installed MemPalace's ``CONVO_CHUNKER_VERSION``; ``None`` before 3.11 defined one.
    chunker: int | None = None
    #: Real drawers the harness's observe filed whose ``convo_chunker_version`` is older than
    #: `chunker`, and stale for no other reason (issue #751): what ``--stamp-chunker-version`` stamps.
    stale_ours: list[tuple[str, dict, str]] = field(default_factory=list)
    #: Every other real exchange-mode drawer that is version-stale the same way: any drawer not
    #: provably the observe's, in any wing, which on a fleet palace is MemPalace's miner's (the
    #: observe's path before 0.164.0). Left to MemPalace, and reported as ``miner-filed``.
    stale_theirs: list[tuple[str, dict, str]] = field(default_factory=list)
    #: Conversation files on disk with a version-stale record and no record MemPalace accepts
    #: (`_accepted`), split by whether every version-stale real drawer under their name is the
    #: observe's (registry rows decide nothing). A file that also changed on disk is counted here
    #: too; its forecast `reason:` line says which. Measured on 3.11.0: a mine deletes the drawers filed at this location and
    #: files the file again through its chunker, and files one whose drawers are only at another
    #: location (a moved home) a second time, beside the old drawers, which it keeps.
    stale_files_ours: int = 0
    stale_files_theirs: int = 0
    #: The installed MemPalace's ``NORMALIZE_VERSION``; ``None`` when it defines none.
    normalize: int | None = None
    #: Whether the installed MemPalace recognises a moved file by its content (`recognises_content`).
    by_content: bool = True
    #: Conversation files on disk with a record at an older ``normalize_version`` and no record
    #: MemPalace accepts, split the same way (issue #989). MemPalace 3.12 raised the version to 3,
    #: and every drawer filed before it is at 2, the observe's included. Measured on 3.12.0: a mine
    #: in place deletes such a file's drawers and files it again through its chunker (3 observe
    #: drawers became 7, none kept), and one whose drawers are only at another location it files a
    #: second time, beside them.
    normalize_files_ours: int = 0
    normalize_files_theirs: int = 0


def census(collection, palace: Path) -> Census:
    """Count drawers, registry rows and files, and collect the real drawers to probe."""
    convos = palace / _CONVERSATIONS_WING
    here = convos.resolve()
    prefix = str(here) + os.sep
    module = _import("palace")
    out = Census(
        chunker=chunker_version(),
        normalize=getattr(module, "NORMALIZE_VERSION", None),
        by_content=recognises_content(),
    )
    normalize = out.normalize
    recorded: set[str] = set()
    # Per conversation file name (issue #751): names with a record MemPalace accepts, names with a
    # version-stale record, and, among the version-stale real drawers, names with one the observe
    # filed and names with one it did not. The same three for a stale ``normalize_version``.
    current_names: set[str] = set()
    stale: set[str] = set()
    observed: set[str] = set()
    theirs: set[str] = set()
    renormal: set[str] = set()
    renormal_observed: set[str] = set()
    renormal_theirs: set[str] = set()
    # Per recorded path at this location, per stored ``source_mtime``: drawers seen and the
    # ``chunk_total`` they claim. MemPalace's own completeness rule (``prefetch_mined_set``).
    groups: dict[str, dict[object, list]] = {}
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
            registry = meta.get("room") == _REGISTRY_ROOM
            if _exchange_mode(meta) and (out.chunker is not None or normalize is not None):
                name = Path(source).name
                if out.chunker is not None and _version_stale(meta, out.chunker, normalize):
                    stale.add(name)
                    # A registry row is MemPalace's bookkeeping, never a drawer: it says nothing
                    # about who filed the file, so only real drawers decide that.
                    if not registry:
                        ours = filed_by_observe(drawer_id, meta, text)
                        theirs.update(() if ours else (name,))
                        observed.update((name,) if ours else ())
                        (out.stale_ours if ours else out.stale_theirs).append(
                            (drawer_id, meta, text)
                        )
                elif normalize is not None and _normalize_stale(meta, normalize):
                    renormal.add(name)
                    if not registry:
                        ours = filed_by_observe(drawer_id, meta, text)
                        (renormal_observed if ours else renormal_theirs).add(name)
                elif _accepted(meta, source, prefix, normalize, out.by_content):
                    current_names.add(name)
            if registry:
                out.registry += 1
                continue
            if source.startswith(prefix):
                _count_in(groups.setdefault(source, {}), meta)
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
    out.short = sum(_short(groups.get(str(here / path.name))) for path in on_disk)
    for path in on_disk:
        if path.name in stale and path.name not in current_names:
            if path.name in observed and path.name not in theirs:
                out.stale_files_ours += 1
            else:
                out.stale_files_theirs += 1
        if path.name in renormal and path.name not in current_names:
            if path.name in renormal_observed and path.name not in renormal_theirs:
                out.normalize_files_ours += 1
            else:
                out.normalize_files_theirs += 1
    return out


def _accepted(
    meta: dict, source: str, prefix: str, normalize: int | None, by_content: bool
) -> bool:
    """Whether a current record is one MemPalace's miner would take as filing its file.

    By path when it was filed at this location (``prefetch_mined_set``), or, on a MemPalace that
    recognises a moved file by its content (before 3.12, `recognises_content`), by content when
    it carries a hash in the observe's wing (``prefetch_content_hashes``, which reads one wing). A
    current record anywhere else (another wing, another home, no hash, or any record elsewhere on
    3.12) files nothing for a mine here. Its mtime is not compared: the forecast, which is
    MemPalace's own, is the authority on that.
    """
    if normalize is not None and _normalize_stale(meta, normalize):
        return False
    if source.startswith(prefix):
        return True
    return by_content and meta.get("wing") == _CONVERSATIONS_WING and bool(meta.get("content_hash"))


def _normalize_stale(meta: dict, normalize: int) -> bool:
    """Whether MemPalace reads a record as unfiled because of its ``normalize_version``.

    MemPalace's own rule (``palace._meta_is_current``): a record with none counts as version 1, and
    one older than the installed constant is not filed, in any extract mode. A value that is not
    a plain number MemPalace calls unprovable and reads as unfiled too.
    """
    version = meta.get("normalize_version", 1)
    if not isinstance(version, (int, float)) or isinstance(version, bool) or math.isnan(version):
        return True
    return version < normalize


def _version_stale(meta: dict, chunker: int, normalize: int | None) -> bool:
    """Whether MemPalace reads a record as unfiled only because of its ``convo_chunker_version``.

    MemPalace's own rule (``palace._meta_is_current``, 3.11): an exchange-mode record with no
    ``convo_chunker_version`` counts as version 1, and one older than the installed constant is not
    filed. A record whose ``normalize_version`` is stale too is not this class: its reason is that
    one, and stamping the chunker version would not make it current.
    """
    if normalize is not None and _normalize_stale(meta, normalize):
        return False
    return meta.get("convo_chunker_version", 1) < chunker


def _short(groups: dict[object, list] | None) -> bool:
    """Whether a file's drawers fall short of the count they record, in every ``source_mtime``.

    MemPalace's rule: a group with no ``chunk_total`` is trusted, and one is complete once it holds
    that many drawers. A file with no real drawers here is not short; it is unregistered, or
    registered with a row only.
    """
    if not groups:
        return False
    return not any(total is None or seen >= int(total) for seen, total in groups.values())


@dataclass
class Forecast:
    """What a mine of the conversation files would do, per MemPalace's own dry run."""

    #: Files the mine recognises as already filed and files nothing for: at this path, or, before
    #: MemPalace 3.12, by their content elsewhere, which writes one registry row each.
    recognised: int
    #: Names of the files that will be mined as new drawers.
    new: list[str]


#: The counts MemPalace's mine prints of the files it skipped as already filed and of the ones it
#: read and did not skip (3.9 to 3.12).
_SKIPPED = re.compile(r"^\s*Files skipped \(already filed\): (\d+)$")
_PROCESSED = re.compile(r"^\s*Files processed: (\d+)$")


def forecast(palace: Path, agent: str) -> Forecast | None:
    """What a mine of the directory would do: which files it recognises, which it mines as new.

    MemPalace's own dry-run mine of the directory, with the arguments the adapter mined with until
    0.164.0, read off the lines it prints: one per file it would mine, and its count of the files
    it skipped as already filed. A file the observe filed is skipped by it: its drawers carry the
    ``source_mtime`` and ``chunk_total`` the miner decides by (`MemPalaceMemoryProvider._file`),
    so it is recognised here, and, after a move, by its ``content_hash`` before 3.12 or once
    ``--relocate-source-file`` has recorded it here from 3.12 (issue #989).

    ``None`` when the output cannot be read whole, so a changed upstream format is reported as
    unknown rather than as a forecast of nothing: each count must be printed exactly once, and the
    files MemPalace says it processed must be exactly the files named on a per-file line. A
    per-file line in a wording this does not read would otherwise leave ``new`` empty beside a
    readable skip count, a confident "0 new". The cost of the check is one honest unknown: a
    conversation file that normalizes to nothing is processed and names no line.
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
    skipped = [int(match.group(1)) for line in lines if (match := _SKIPPED.match(line))]
    # "    [DRY RUN] <name> -> room:<room> (<n> drawers)"
    new = [
        line.split("[DRY RUN]", 1)[1].split("->", 1)[0].strip()
        for line in lines
        if "[DRY RUN]" in line and "->" in line
    ]
    processed = [int(match.group(1)) for line in lines if (match := _PROCESSED.match(line))]
    if len(skipped) != 1 or processed != [len(new)]:
        return None
    return Forecast(skipped[0], new)


def forecast_line(predicted: Forecast | None) -> str:
    """The forecast's summary line, which the NOC's rename flow matches whole (``FORECAST_RE``).

    ``next observe (dry run): <r> files -> recognised, <n> files -> mined as new drawers``. It keeps
    the words "next observe" for that reader, though since 0.164.0 the observe never mines
    (issue #741). Until 3.12 the first count named a registry sentinel, the row a mine wrote for a
    file it recognised by its content; 3.12 recognises a file by its path alone and writes none,
    so the count names what both do (issue #989).
    """
    if predicted is None:
        return "next observe (dry run): unknown, MemPalace's dry-run output was not readable"
    return (
        f"next observe (dry run): {predicted.recognised} files -> recognised, "
        f"{len(predicted.new)} files -> mined as new drawers"
    )


def accepted_hashes(open_collection: Callable[[], Any]) -> dict | None:
    """MemPalace's own map of the content it will recognise, keyed ``(wing, content_hash)``.

    The very map the miner reads to decide that a file at a new path is already filed
    (``prefetch_content_hashes``, with the adapter's extract mode). ``None`` when it cannot be
    read, so a reason is reported as unknown rather than guessed. Handed an opener, never a handle,
    because it runs after the forecast's dry-run mine, which opens the palace
    (`MemPalaceMemoryProvider.read_collection`).
    """
    try:
        return _import("palace").prefetch_content_hashes(open_collection(), extract_mode="exchange")
    except ImportError:
        raise
    except Exception:  # noqa: BLE001 - a diagnosis input, never a crash of the check
        return None


def new_file_diagnosis(found: Census, convos: Path, name: str, accepted: dict | None) -> str:
    """Why a mine of the directory would file ``name`` as new drawers, in counts, flags and a reason.

    The first line is the file's record: the drawers filed under its name, their wings, hashes,
    ``normalize_version`` and extract mode, and whether today's content hash matches one of theirs.
    The ``reason:`` line is the miner's decision, read off `accepted_hashes`: a moved file is
    recognised only when MemPalace's map holds its content hash **in the observe's wing**, so a
    record whose every field matches can still be refused (issue #606, phase 4: the first version
    of this line listed every field but the wing, and said "matches" about a file the miner then
    filed again). If its drawers exist elsewhere, mining it again adds a second copy of each, and
    the two copies then compete for the same slots (see `diagnose`).
    """
    rows = [
        meta for _, meta, _ in found.real if Path(str(meta.get("source_file") or "")).name == name
    ]
    here = str((convos / name).resolve())
    at_here = sum(str(meta.get("source_file")) == here for meta in rows)
    hashed = [meta for meta in rows if meta.get("content_hash")]
    stored = {h for meta in hashed for h in _hashes(meta)}
    wings = sorted({str(meta.get("wing") or "-") for meta in rows}) or ["-"]
    versions = sorted({str(meta.get("normalize_version", 1)) for meta in rows}) or ["-"]
    modes = sorted({str(meta.get("extract_mode") or "-") for meta in rows}) or ["-"]
    try:
        today = _today(convos / name)
        matches = "yes" if set(today) & stored else "no"
        reason = (
            _new_reason(found, here, today, bool(hashed), bool(rows), accepted)
            if found.by_content
            else _path_reason(found, convos / name, here)
        )
    except ImportError:
        raise
    except Exception as error:  # noqa: BLE001 - a diagnosis line, never a crash of the check
        matches = reason = f"unknown ({type(error).__name__})"
    return (
        f"  mined as new: {name}: {len(rows)} drawers recorded under this file name "
        f"({at_here} at this location); wing {','.join(wings)}; content_hash on {len(hashed)}; "
        f"normalize_version {','.join(versions)}; extract_mode {','.join(modes)}; "
        f"today's content hash matches a recorded one: {matches}\n"
        f"    reason: {reason}"
    )


def _today(path: Path) -> list[str]:
    """The content hash of each conversation in a file, exactly as the miner computes them.

    The adapter's own function (`content_hashes`), the one the observe stamps on every drawer it
    files, so what this check compares against is never spelled a second time.
    """
    return content_hashes(path)


def _unrecognised(here: str, today: list[str], accepted: dict) -> list[str]:
    """Today's hashes MemPalace's map does not hold under the observe's wing for another path."""
    return [h for h in today if accepted.get((_CONVERSATIONS_WING, h)) in (None, here)]


def off_wing(found: Census, here: str, today: list[str], accepted: dict) -> str | None:
    """The one wing a file's unrecognised content is filed in, when that is the whole reason.

    Read off the drawers themselves: every conversation MemPalace will not recognise is carried
    by drawers, none of them in the observe's wing, and all of them in the same other wing. Any
    other shape (a hash nobody carries, two wings, a drawer with no wing, a file that changed
    since it was mined here) is ``None``, and `register_off_wing` leaves the file alone.
    """
    unknown = _unrecognised(here, today, accepted)
    if not unknown or any(accepted.get((_CONVERSATIONS_WING, h)) == here for h in unknown):
        return None
    wings: set[str] = set()
    for content_hash in unknown:
        carriers = [meta for _, meta, _ in found.real if content_hash in _hashes(meta)]
        if not carriers:
            return None
        wings |= {str(meta.get("wing") or "") for meta in carriers}
    if len(wings) != 1 or "" in wings or _CONVERSATIONS_WING in wings:
        return None
    return wings.pop()


def _new_reason(
    found: Census, here: str, today: list[str], hashed: bool, named: bool, accepted: dict | None
) -> str:
    """The first of the file's conversations MemPalace will not recognise, and why, in words.

    Whether it is recognised is MemPalace's answer (``accepted``). *Why not* is read off the drawers
    that carry its hash: the first of wing, extract mode and ``normalize_version`` that keeps every
    one of them out of the map under the observe's wing.
    """
    if accepted is None:
        return "unknown, MemPalace's content-hash map could not be read"
    wing = _CONVERSATIONS_WING
    unknown = _unrecognised(here, today, accepted)
    if not unknown:
        return "none: MemPalace's content-hash map recognises every conversation in it"
    content_hash = unknown[0]
    count = f" ({len(unknown)} of its {len(today)} conversations)" if len(today) > 1 else ""
    if accepted.get((wing, content_hash)) == here:
        return f"it changed since it was mined at this path, and is re-mined in place{count}"
    rows = [row for row in found.real if content_hash in _hashes(row[1])]
    carriers = [meta for _, meta, _ in rows]
    if not carriers:
        if not named:
            return f"it was never filed{count}"
        if not hashed:
            return f"its drawers carry no content hash (filed before MemPalace 3.7){count}"
        return f"no drawer carries today's content hash: it changed after it was filed{count}"
    in_wing = [meta for meta in carriers if meta.get("wing") == wing]
    if not in_wing:
        others = ", ".join(
            sorted({repr(meta["wing"]) if meta.get("wing") else "(none)" for meta in carriers})
        )
        return (
            f"its content is filed only in wing {others}, not the observe's {wing!r}; MemPalace "
            f"recognises a moved file's content within one wing but a known path in any, so a "
            f"move files it again in {wing!r}: a second copy of each of its drawers{count}"
        )
    exchange = [meta for meta in in_wing if _exchange_mode(meta)]
    if not exchange:
        modes = ", ".join(sorted({str(meta.get("extract_mode") or "-") for meta in in_wing}))
        return f"its content is filed only under extract_mode {modes}, not exchange{count}"
    current = getattr(_import("palace"), "NORMALIZE_VERSION", None)
    if current is not None and all(_normalize_stale(meta, current) for meta in exchange):
        versions = ", ".join(sorted({str(meta.get("normalize_version", 1)) for meta in exchange}))
        return (
            f"its content is filed only at normalize_version {versions}, older than "
            f"MemPalace's {current}{count}"
        )
    chunker = found.chunker
    # Every carrier here is stale for one reason or the other (MemPalace accepts none of them), so
    # if any is stale only by the chunker, that is the class: stamping or re-chunking it is what
    # would make this content recognised.
    chunked = [
        row
        for row in rows
        if row[1].get("wing") == wing
        and _exchange_mode(row[1])
        and chunker is not None
        and _version_stale(row[1], chunker, current)
    ]
    if chunked:
        return _chunker_reason(chunked, chunker, here) + count
    return (
        f"MemPalace does not accept its recorded hash, for a reason this check cannot name{count}"
    )


def _chunker_reason(chunked: list[tuple[str, dict, str]], chunker: int, here: str) -> str:
    """The version-stale reason for drawers stale by their chunker version alone (issue #751)."""
    exchange = [meta for _, meta, _ in chunked]
    versions = ", ".join(sorted({str(meta.get("convo_chunker_version", 1)) for meta in exchange}))
    ours = all(filed_by_observe(*row) for row in chunked)
    remedy = (
        "; the observe filed it, so --stamp-chunker-version makes it current"
        if ours
        else "; MemPalace's miner filed it, and it is left to MemPalace"
    )
    return (
        f"version-stale: its content is filed only at convo_chunker_version {versions}, older "
        f"than MemPalace's {chunker}, by an earlier MemPalace; {_effect(exchange, here)}{remedy}"
    )


def _effect(exchange: list[dict], here: str) -> str:
    """What a mine does to a file whose records it does not accept, by where they are filed."""
    if any(str(meta.get("source_file")) == here for meta in exchange):
        return "a mine deletes its drawers here and files it again through MemPalace's chunker"
    return "a mine files it again here, beside its old drawers, which it keeps"


def _path_reason(found: Census, path: Path, here: str) -> str:
    """Why a MemPalace that recognises a file by its path alone (3.12) files ``path`` as new.

    The mine looks for a complete set of current drawers recorded at this exact path whose
    ``source_mtime`` is the file's own (``prefetch_mined_set``), and nothing else: not the content,
    not another location, and not the wing. So the reason is read off the drawers recorded under
    this file name, here first, and then elsewhere.
    """
    name = path.name
    rows = [row for row in found.real if Path(str(row[1].get("source_file") or "")).name == name]
    if not rows:
        return "it was never filed"
    exchange = [row for row in rows if _exchange_mode(row[1])]
    located = [row for row in exchange if str(row[1].get("source_file")) == here]
    if not located:
        if not exchange:
            return "it is filed only under another extract_mode, not exchange"
        effect = (
            "it is filed only at another location, and this MemPalace recognises a file by its "
            "path alone, so a mine files it again here, beside its old drawers, which it keeps"
        )
        ours = [filed_by_observe(*row) for row in exchange]
        held = _why_not_observed(exchange, found) if ours and all(ours) else None
        if held is not None:
            return f"{effect}; --relocate-source-file leaves it alone: {held}"
        return (
            f"{effect}; --relocate-source-file records a file moved from a previous home here, "
            "once it proves it is the same file"
        )
    normalize, chunker = found.normalize, found.chunker
    renormal = [
        row for row in located if normalize is not None and _normalize_stale(row[1], normalize)
    ]
    if renormal:
        return _normalize_reason(renormal, normalize or 0, here)
    chunked = [
        row for row in located if chunker is not None and _version_stale(row[1], chunker, normalize)
    ]
    if chunked:
        return _chunker_reason(chunked, chunker or 0, here)
    if _short(_groups(located)):
        return "it was filed here short of its drawers, so a mine deletes them and files it again"
    return "it changed since it was filed here, and a mine deletes its drawers and files it again"


def _normalize_reason(rows: list[tuple[str, dict, str]], normalize: int, here: str) -> str:
    """The reason for drawers MemPalace reads as unfiled by their ``normalize_version`` (#989)."""
    exchange = [meta for _, meta, _ in rows]
    versions = ", ".join(sorted({str(meta.get("normalize_version", 1)) for meta in exchange}))
    ours = all(filed_by_observe(*row) for row in rows)
    remedy = (
        "; the observe filed it, and this harness has no step that makes it current"
        if ours
        else "; MemPalace's miner filed it, and it is left to MemPalace"
    )
    return (
        f"normalize-stale: its content is filed only at normalize_version {versions}, older than "
        f"MemPalace's {normalize}; {_effect(exchange, here)}{remedy}"
    )


def _groups(rows: list[tuple[str, dict, str]]) -> dict[object, list]:
    """Per stored ``source_mtime``: drawers seen and the ``chunk_total`` they claim (`_short`)."""
    groups: dict[object, list] = {}
    for _, meta, _ in rows:
        _count_in(groups, meta)
    return groups


def _count_in(groups: dict[object, list], meta: dict) -> None:
    """Count one drawer into its ``source_mtime`` group: MemPalace's completeness rule's input."""
    group = groups.setdefault(meta.get("source_mtime"), [0, None])
    group[0] += 1
    if meta.get("chunk_total") is not None:
        group[1] = meta.get("chunk_total")


def _exchange_mode(meta: dict) -> bool:
    """Whether MemPalace counts a drawer as filed in exchange mode (a legacy row with none, too)."""
    mode = meta.get("extract_mode")
    return mode == "exchange" or (mode is None and meta.get("ingest_mode") in (None, "convos"))


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


def _mine_file(path: Path, palace: Path, wing: str, agent: str, *, dry_run: bool) -> list[str]:
    """MemPalace's own single-file mine, in ``wing``; returns the per-file lines it prints."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _import("convo_miner").mine_convos(
            str(path), str(palace), wing=wing, agent=agent, extract_mode="exchange", dry_run=dry_run
        )
    return out.getvalue().splitlines()


def _only_registers(lines: list[str]) -> bool:
    """Whether a single-file dry run says it will write one registry row and no drawer."""
    duplicate = sum("duplicate of" in line for line in lines)
    new = sum("[DRY RUN]" in line and "->" in line for line in lines)
    return duplicate == 1 and new == 0


def _counts(palace: Path) -> tuple[int, int]:
    """Real drawers and registry rows, read fresh off the palace."""
    collection = _import("palace").get_collection(str(palace), create=False, read_only=True)
    found = census(collection, palace)
    return len(found.real), found.registry


def register_off_wing(provider: MemPalaceMemoryProvider, palace: Path, *, dry_run: bool) -> int:
    """Register each moved file whose content is filed only in another wing, in that wing.

    A mine of ``conversations/`` files into wing ``conversations`` (the observe's own, before
    0.164.0), and MemPalace recognises a moved file's content only within the mining wing, so a
    file once mined on its own into another wing is filed again after a move: a second copy of
    each of its drawers (issue #606, phase 4). Mining that one file in the wing its drawers already carry is recognised as a
    duplicate, and writes **one registry row and no drawer**; a mine then skips the file by its
    path. No drawer and no drawer metadata is touched (issue #613).

    Acts only on files the dry-run forecast says will be mined as new **and** whose unrecognised
    content is filed in exactly one other wing (`off_wing`), taking the wing from those drawers.
    Before writing anything, each file's own single-file dry run must say it registers and files
    no drawer; if any would file one, nothing is written and the run is refused. A file registered
    once is known by its path, so a second run finds nothing to do. Returns the exit code.
    """
    word = "dry run, writes nothing" if dry_run else "WRITES TO THE PALACE"
    print(f"register off-wing ({word}):")
    if not recognises_content():
        # The registry row this writes is the one a mine writes for content it recognises, and
        # 3.12 recognises none: a single-file mine there files drawers, never a row. A moved
        # off-wing file is recorded at its new path by --relocate-source-file instead (#989).
        version = mempalace_version() or "?"
        print(
            f"  nothing to register: MemPalace {version} recognises a moved file by its path "
            "alone; --relocate-source-file records a moved off-wing file here"
        )
        if dry_run:
            print(
                "register off-wing (dry run): files to register 0, registry rows to write 0, "
                "drawers to write 0"
            )
        else:
            print(
                "register off-wing: WROTE to this palace: files registered 0, "
                "registry rows written 0, drawers written 0"
            )
        return 0
    collection = _import("palace").get_collection(str(palace), create=False, read_only=True)
    found = census(collection, palace)
    predicted = forecast(palace, provider.agent) if found.unregistered else Forecast(0, [])
    if predicted is None:
        print("  refused, nothing written: MemPalace's dry-run output was not readable")
        return 1
    accepted = accepted_hashes(provider.read_collection) if predicted.new else {}
    if accepted is None:
        print("  refused, nothing written: MemPalace's content-hash map could not be read")
        return 1
    convos = palace / _CONVERSATIONS_WING
    plan: list[tuple[Path, str]] = []
    for name in predicted.new:
        path = convos / name
        try:
            wing = off_wing(found, str(path.resolve()), _today(path), accepted)
        except ImportError:
            raise
        except Exception as error:  # noqa: BLE001 - one unreadable file is left alone, and named
            print(f"  left alone: {name}: could not be read ({type(error).__name__})")
            continue
        if wing is None:
            print(f"  left alone: {name}: not the other-wing case (see its reason: line)")
        else:
            plan.append((path, wing))
    refused = [
        (path, wing)
        for path, wing in plan
        if not _only_registers(_mine_file(path, palace, wing, provider.agent, dry_run=True))
    ]
    for path, wing in refused:
        print(f"  refused: {path.name}: mining it in wing {wing!r} would file drawers")
    if refused:
        print("  refused, nothing written")
        return 1
    if dry_run:
        for path, wing in plan:
            print(f"  would register: {path.name} in wing {wing!r}")
        print(
            f"register off-wing (dry run): files to register {len(plan)}, "
            f"registry rows to write {len(plan)}, drawers to write 0"
        )
        return 0
    real, registry = _counts(palace)
    for path, wing in plan:
        _mine_file(path, palace, wing, provider.agent, dry_run=False)
        print(f"  registered: {path.name} in wing {wing!r}")
    real_after, registry_after = _counts(palace)
    rows, drawers = registry_after - registry, real_after - real
    print(
        f"register off-wing: WROTE to this palace: files registered {len(plan)}, "
        f"registry rows written {rows}, drawers written {drawers}"
    )
    if drawers or rows != len(plan):
        print(
            f"FAIL: expected {len(plan)} registry rows and 0 drawers; the palace changed in "
            "another way while this ran, or MemPalace did not do what its dry run said"
        )
        return 1
    return 0


#: What the version-stale line's second half says, printed whenever a file is version-stale.
STALE_CONSEQUENCE = (
    "a mine re-chunks each version-stale file filed at this location, and files each one filed "
    "only at another location a second time, beside its old drawers; --relocate-source-file "
    "makes a file moved from a previous home the in-place case; --stamp-chunker-version "
    "makes the observe's own current, and MemPalace's miner's are left to MemPalace"
)


def stale_line(found: Census) -> str:
    """The version-stale count, apart from unfiled files: always printed, in one grammar (#751).

    ``conversation files version-stale: <n> (observe-filed <a>, miner-filed <b>), <why>``. The
    first half is the same on every MemPalace, so a reader of the report can match it whole; the
    tail names the installed constant, or says there is none.
    """
    ours, theirs = found.stale_files_ours, found.stale_files_theirs
    why = (
        f"convo_chunker_version older than MemPalace's {found.chunker}"
        if found.chunker is not None
        else "this MemPalace defines no convo_chunker_version"
    )
    return (
        f"conversation files version-stale: {ours + theirs} "
        f"(observe-filed {ours}, miner-filed {theirs}), {why}"
    )


#: What the normalize-stale line's second half says, printed whenever a file is normalize-stale.
NORMALIZE_CONSEQUENCE = (
    "a mine deletes each normalize-stale file's drawers filed at this location and files it again "
    "through MemPalace's chunker, the observe's included, and files each one filed only at "
    "another location a second time, beside its old drawers; nothing in this harness makes the "
    "observe's current, and each file's reason: line says what --relocate-source-file does for it"
)


def normalize_line(found: Census) -> str:
    """The normalize-stale count, the version-stale line's sibling (issue #989), always printed.

    ``conversation files normalize-stale: <n> (observe-filed <a>, miner-filed <b>), <why>``, in the
    version-stale line's grammar so one reader parses both. MemPalace 3.12 raised the version to 3
    and every drawer filed before it is at 2, so on 3.12 this counts every file a palace filed
    before the upgrade.
    """
    ours, theirs = found.normalize_files_ours, found.normalize_files_theirs
    why = (
        f"normalize_version older than MemPalace's {found.normalize}"
        if found.normalize is not None
        else "this MemPalace defines no normalize_version"
    )
    return (
        f"conversation files normalize-stale: {ours + theirs} "
        f"(observe-filed {ours}, miner-filed {theirs}), {why}"
    )


def stamp_chunker_version(palace: Path, *, dry_run: bool) -> int:
    """Stamp the observe's own version-stale drawers with the installed chunker version (#751).

    MemPalace 3.11 reads an exchange drawer with no ``convo_chunker_version`` as filed by an older
    chunker, so a drawer the observe filed under 3.10 or earlier drops out of the miner's mined
    set and content-hash map. A mine then deletes it and files its exchange again through the
    very chunker the observe exists to avoid (in place), or files a second copy beside it (after
    a move). Those drawers were
    never cut by a chunker at all (the observe files each exchange whole, issue #741), so the
    installed version is true of them, and this writes it on each one and changes nothing else.

    Only drawers `filed_by_observe` proves are the observe's own, by their id recipe, and only
    those stale for this reason alone (`_version_stale`). A drawer MemPalace's miner filed is
    never touched: its chunks really were cut by the older chunker. Idempotent: a stamped drawer
    is current, so a second run finds nothing. Every stamped drawer is read back by id and must
    hold exactly its old metadata plus the stamp. Returns the exit code.
    """
    word = "dry run, writes nothing" if dry_run else "WRITES TO THE PALACE"
    print(f"stamp chunker version ({word}):")
    current = chunker_version()
    if current is None:
        version = getattr(_import("version"), "__version__", "?")
        print(f"  nothing to stamp: MemPalace {version} defines no convo_chunker_version")
        return 0
    collection = _import("palace").get_collection(str(palace), create=False, read_only=dry_run)
    found = census(collection, palace)
    targets = found.stale_ours
    files = len({Path(str(meta.get("source_file") or "")).name for _, meta, _ in targets})
    tail = (
        f"convo_chunker_version {current}; miner-filed version-stale drawers left to MemPalace "
        f"{len(found.stale_theirs)}"
    )
    if dry_run:
        print(
            f"stamp chunker version (dry run): drawers to stamp {len(targets)} in {files} files, "
            f"{tail}"
        )
        return 0
    expected = {
        drawer_id: {**meta, "convo_chunker_version": current} for drawer_id, meta, _ in targets
    }
    ids = list(expected)
    for start in range(0, len(ids), _PAGE):
        batch = ids[start : start + _PAGE]
        collection.update(ids=batch, metadatas=[expected[drawer_id] for drawer_id in batch])
    wrong = 0
    for start in range(0, len(ids), _PAGE):
        batch = ids[start : start + _PAGE]
        got = collection.get(ids=batch, include=["metadatas"])
        back = dict(zip(got.get("ids") or [], got.get("metadatas") or []))
        wrong += sum(back.get(drawer_id) != expected[drawer_id] for drawer_id in batch)
    print(
        f"stamp chunker version: WROTE to this palace: drawers stamped {len(ids) - wrong} in "
        f"{files} files, {tail}"
    )
    if wrong:
        print(
            f"FAIL: {wrong} stamped drawers did not read back as their old metadata plus the "
            "stamp; the palace changed in another way while this ran"
        )
        return 1
    return 0


@dataclass
class Relocation:
    """What ``--relocate-source-file`` found: the files it proves, and the ones it will not touch."""

    #: Per conversation file name, its drawers as ``(drawer_id, metadata)``.
    files: dict[str, list[tuple[str, dict]]] = field(default_factory=dict)
    #: ``(name, reason)`` for each file a mine would duplicate that this cannot prove.
    left_alone: list[tuple[str, str]] = field(default_factory=list)
    #: The proven files whose drawers the observe filed (issue #989): each keeps the path its id
    #: was derived from in `ID_SOURCE_FILE`.
    observed: set[str] = field(default_factory=set)
    #: Per proven file, its ``source_mtime`` today, read after the proof; ``None`` when its drawers
    #: are not one complete filing, whose mtime is never aligned (`relocated`).
    mtimes: dict[str, float | None] = field(default_factory=dict)

    @property
    def drawers(self) -> int:
        return sum(len(rows) for rows in self.files.values())


def _reads_back(back: dict | None, expected: dict) -> bool:
    """Whether a rewritten drawer reads back as `relocated` wrote it.

    Exactly, but for a float: MemPalace 3.12's store keeps one to the microsecond (measured: a
    ``source_mtime`` of ``…687.4985867`` read back as ``…687.498587``), so an aligned mtime is
    compared within that and nothing looser. Every other value, and every key, must match.
    """
    if back is None or set(back) != set(expected):
        return False
    for key, want in expected.items():
        got = back[key]
        if isinstance(want, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
            if abs(got - want) > 1e-6:
                return False
        elif got != want:
            return False
    return True


def relocated(meta: dict, path: str, *, observed: bool, mtime: float | None, current: bool) -> dict:
    """A proven drawer's metadata at its new path: what `_write_relocation` writes and reads back.

    ``source_file`` is the new path. An observe drawer keeps the path its id was derived from:
    `ID_SOURCE_FILE` is written as its old ``source_file`` when it carries none yet, and never
    rewritten once it does, so `filed_by_observe` still holds of it (issue #989). A drawer the
    mine would take as current takes the file's ``source_mtime`` today when the move changed it,
    within MemPalace's own tolerance (``_is_unchanged_since_last_mine``), and only then: that is
    what makes the mine skip it rather than file it again. A stale drawer keeps its own, because
    the mine files it again in place whatever its mtime, and so does one of a file whose drawers
    are not one complete filing (``mtime`` ``None``): aligning drawers from two filings would merge
    them into one group a mine could read as complete. Nothing else changes.
    """
    out = {**meta, "source_file": path}
    if observed and out.get(ID_SOURCE_FILE) is None:
        out[ID_SOURCE_FILE] = meta.get("source_file")
    stored = meta.get("source_mtime")
    if (
        current
        and mtime is not None
        and isinstance(stored, (int, float))
        and abs(stored - mtime) >= 0.001
    ):
        out["source_mtime"] = mtime
    return out


def _unfiled(meta: dict, chunker: int, normalize: int | None) -> bool:
    """Whether MemPalace reads an exchange-mode record as unfiled (``palace._meta_is_current``)."""
    return meta.get("convo_chunker_version", 1) < chunker or (
        normalize is not None and _normalize_stale(meta, normalize)
    )


def _why_not_relocatable(
    rows: list[tuple[str, dict, str]], here: Path, old: Path, found: Census
) -> str | None:
    """Why this file's drawers may not be rewritten to ``here``, or ``None`` when they may.

    ``rows`` is every real drawer recorded under the file's name, wherever it was filed. A file is
    moved whole or not at all, and each check names a way the rewrite would be false or useless.
    Every MemPalace:

    - the file must be here, or there is nothing for a mine to read;
    - the recorded path must be gone, or the file was never moved;
    - every drawer must be at the previous location, so the rewrite does not merge two filings;
    - the content hashes the drawers carry must be today's hashes of the file here, which is what
      makes it the same file and not merely the same name. Checked last, because it reads the file.

    A MemPalace that recognises a moved file by its content (before 3.12) recognises a current
    drawer at another location without help, so only the class measured on 3.11 is moved
    (issue #831): every drawer in the observe's wing, none of them the observe's (whose stale
    drawers `stamp_chunker_version` makes current, and then recognised), and every one stale by
    its chunker version alone. An off-wing file is `register_off_wing`'s there.

    A MemPalace that recognises a file by its path alone (3.12, issue #989) recognises nothing at
    another location, so every file there is moved that proves itself:

    - every drawer is in one wing (a mine skips a file by its path whatever the wing, so an
      off-wing file filed whole in its own wing is recorded here like any other);
    - the observe's file is the observe's whole, and current by every version: relocated stale,
      a mine would delete its drawers and cut it again through MemPalace's chunker, which turns a
      second copy into a re-chunk rather than preventing either, so it is left alone and named;
    - MemPalace's miner's may be current (the mine then skips it) or stale (the mine files it
      again in place, upstream's own migration, as on 3.11), but must be in exchange mode.
    """
    try:
        present = here.is_file()
    except OSError as error:
        return f"it could not be read ({type(error).__name__})"
    if not present:
        return "no such file at this location, so a mine never reads it"
    try:
        os.lstat(old)
    except (FileNotFoundError, NotADirectoryError):
        pass
    except OSError as error:
        return f"its recorded path could not be checked ({type(error).__name__})"
    else:
        return "its recorded path still exists, so it was not moved"
    if {str(meta.get("source_file")) for _, meta, _ in rows} != {str(old)}:
        return "it has drawers recorded somewhere other than the previous home"
    why = (_why_not_by_content if found.by_content else _why_not_by_path)(rows, found)
    if why is not None:
        return why
    stored = {h for _, meta, _ in rows for h in _hashes(meta)}
    if not stored:
        return (
            "its drawers carry no content hash (MemPalace's miner stamped one from 3.7 until "
            "3.12), so nothing proves it"
        )
    try:
        today = set(_today(here))
    except ImportError:
        raise
    except Exception as error:  # noqa: BLE001 - one unreadable file is left alone, and named
        return f"it could not be read ({type(error).__name__})"
    if today != stored:
        return "its content hash is not the one its drawers record"
    return None


def _why_not_by_content(rows: list[tuple[str, dict, str]], found: Census) -> str | None:
    """The rules for a MemPalace that recognises a moved file by its content (`_why_not_relocatable`)."""
    chunker, normalize = found.chunker or 0, found.normalize
    wings = sorted({str(meta.get("wing") or "-") for _, meta, _ in rows})
    if wings != [_CONVERSATIONS_WING]:
        return f"it has drawers in wing {', '.join(wings)}, not only {_CONVERSATIONS_WING!r}"
    if any(filed_by_observe(*row) for row in rows):
        # Considered only because a drawer is stale, so this names which way; a current observe
        # file at another location is recognised by its content and never reaches here.
        return _why_not_observed(rows, found) or (
            "the observe filed it, and this MemPalace recognises it by its content"
        )
    if not all(
        _exchange_mode(meta) and _version_stale(meta, chunker, normalize) for _, meta, _ in rows
    ):
        return "not every drawer is stale by its chunker version alone, the class this relocates"
    return None


def _why_not_by_path(rows: list[tuple[str, dict, str]], found: Census) -> str | None:
    """The rules for a MemPalace that recognises a file by its path alone (`_why_not_relocatable`)."""
    wings = sorted({str(meta.get("wing") or "-") for _, meta, _ in rows})
    if len(wings) != 1 or wings == ["-"]:
        return f"it has drawers in wing {', '.join(wings)}, not in one wing"
    if not all(_exchange_mode(meta) for _, meta, _ in rows):
        return "not every drawer is in exchange mode, the mode a mine of conversations/ reads"
    ours = [filed_by_observe(*row) for row in rows]
    if not any(ours):
        return None
    if not all(ours):
        return "it has drawers the observe filed and drawers MemPalace's miner filed"
    return _why_not_observed(rows, found)


def _why_not_observed(rows: list[tuple[str, dict, str]], found: Census) -> str | None:
    """Why the observe's file may not be relocated, or ``None``: it must be one current filing.

    Relocated stale, a mine deletes its drawers and cuts it again through MemPalace's chunker,
    which turns a second copy into a re-chunk rather than preventing either (issue #989).
    """
    chunker, normalize = found.chunker or 0, found.normalize
    renormal = sorted(
        {
            str(meta.get("normalize_version", 1))
            for _, meta, _ in rows
            if normalize is not None and _normalize_stale(meta, normalize)
        }
    )
    if renormal:
        return (
            f"the observe filed it at normalize_version {', '.join(renormal)}, older than "
            f"MemPalace's {normalize}, so a mine files it again wherever it is recorded, and this "
            "harness has no step that makes it current"
        )
    if any(_version_stale(meta, chunker, normalize) for _, meta, _ in rows):
        return (
            "the observe filed it under an older chunker; --stamp-chunker-version makes it current"
        )
    if _short(_groups(rows)) or len(_groups(rows)) != 1:
        return "its drawers are not one complete filing, so a mine files it again wherever it is"
    return None


def relocation(found: Census, convos: Path, previous: Path) -> Relocation:
    """Which files ``--relocate-source-file`` proves, read off the census.

    A file is considered when a real drawer recorded in the previous home's ``conversations``
    directory under the file's name is one a mine here would file a second time, so it is either
    relocated here or named. Before MemPalace 3.12 that is a drawer the miner reads as unfiled, by
    its chunker version or its ``normalize_version``: a current one it recognises by its content.
    From 3.12, which recognises a file by its path alone, it is every exchange-mode drawer there
    (issue #989).
    """
    chunker, normalize = found.chunker, found.normalize
    out = Relocation()
    if chunker is None:
        return out
    old_dir = str(previous / _CONVERSATIONS_WING)
    named: dict[str, list[tuple[str, dict, str]]] = {}
    for row in found.real:
        named.setdefault(Path(str(row[1].get("source_file") or "")).name, []).append(row)
    considered = sorted(
        {
            Path(str(meta.get("source_file"))).name
            for _, meta, _ in found.elsewhere
            if str(Path(str(meta.get("source_file"))).parent) == old_dir
            and _exchange_mode(meta)
            and (not found.by_content or _unfiled(meta, chunker, normalize))
        }
    )
    for name in considered:
        rows = named[name]
        # Stat before the proof reads the file: a change after the stat then shows as an mtime the
        # mine does not match, and is filed again, where a stat after the hash could certify
        # content the hash never read (the order MemPalace keeps, one fstat with the read).
        try:
            mtime: float | None = os.path.getmtime(convos / name)
        except OSError:
            mtime = None
        why = _why_not_relocatable(rows, convos / name, Path(old_dir) / name, found)
        if why is None and mtime is None:
            why = "it could not be read (OSError)"
        if why is not None:
            out.left_alone.append((name, why))
            continue
        out.files[name] = [(drawer_id, meta) for drawer_id, meta, _ in rows]
        groups = _groups(rows)
        out.mtimes[name] = mtime if len(groups) == 1 and not _short(groups) else None
        if any(filed_by_observe(*row) for row in rows):
            out.observed.add(name)
    return out


def _whole_files(files: dict[str, list[tuple[str, dict]]], size: int) -> list[list[str]]:
    """The files' names in write batches of about ``size`` drawers, never splitting a file.

    One ``update`` is one transaction, so a crash between batches leaves every file either moved
    or not. A file split across two would be left with drawers at both locations, which a re-run
    has to refuse.
    """
    batches: list[list[str]] = [[]]
    filled = 0
    for name, rows in files.items():
        if batches[-1] and filled + len(rows) > size:
            batches.append([])
            filled = 0
        batches[-1].append(name)
        filled += len(rows)
    return [batch for batch in batches if batch]


def relocate_source_file(palace: Path, home: Path, previous: Path, *, dry_run: bool) -> int:
    """Record a moved palace's old-home drawers at this location, so a mine never files them twice.

    MemPalace's mine finds what to purge by ``source_file`` alone, plus the extract mode
    (``convo_miner._source_file_delete_ids``). After a home move the drawers still name the old
    path, so a mine of the file here deletes nothing and files every drawer a second time. Measured
    on a palace mined under 3.10.0, moved and mined under 3.11.0: 215 drawers became 430; with
    ``source_file`` rewritten first they became exactly the 215 the dry run planned, with no
    duplicate, and the mine removed the registry rows a rename left at the new path.
    ``source_dir_ino`` is never changed. Only ``sync`` reads it, and a drawer whose home was copied
    rather than renamed keeps the old directory's identity, which ``sync`` reads as unresolved and
    keeps, never deletes; rewriting it would be a second claim about where a file came from.

    **On MemPalace 3.12 it is the whole of a moved home's recognition** (issue #989, `@briggs`'s
    rule, decided by `@origin`). 3.12 recognises a file at its recorded path only, so every file a
    move left at another location is considered, the observe's included: a current one is then
    skipped by the mine, and a stale miner-filed one re-filed in place. What a rewrite writes is
    `relocated`'s: the new ``source_file``; for an observe drawer, `ID_SOURCE_FILE` as the old path
    when it carries none, never rewritten, so `filed_by_observe` still holds of it; and for a
    current drawer, the file's ``source_mtime`` today when the move changed it, after the proof.
    The rules each file proves itself by are `_why_not_relocatable`'s.

    ``previous`` is the ``HARNESS_HOME`` the palace used to live under, stated by whoever runs
    this; each file must then prove itself against it (`_why_not_relocatable`). Registry rows are
    never touched. **Nothing here deletes a drawer**: the count of drawers and rows is compared
    before and after, and every rewritten drawer is read back by id and must hold exactly its old
    metadata with `relocated`'s changes. Idempotent: a rewritten drawer is at this location, so a
    second run finds nothing. Holds MemPalace's per-palace lock from the plan's read to the last write,
    because a mine that read the palace before this ran would file the drawers beside themselves;
    every other writer fails against that lock too, an observe included, so run it with the
    agent's wakes held. Returns the exit code.
    """
    word = "dry run, writes nothing" if dry_run else "WRITES TO THE PALACE"
    print(f"relocate source_file ({word}):")
    print(f"  previous home: {previous}")
    if chunker_version() is None:
        version = getattr(_import("version"), "__version__", "?")
        print(f"  nothing to relocate: MemPalace {version} defines no convo_chunker_version")
        return 0
    module = _import("palace")
    previous_palace = previous / palace.relative_to(home)
    convos = (palace / _CONVERSATIONS_WING).resolve()

    def plan(collection) -> tuple[Census, Relocation]:
        found = census(collection, palace)
        return found, relocation(found, convos, previous_palace)

    def tail(found: Census, moved: Relocation) -> str:
        return (
            f"files left alone {len(moved.left_alone)}, drawers left at another location "
            f"{len(found.elsewhere) - moved.drawers}"
        )

    if dry_run:
        found, moved = plan(module.get_collection(str(palace), create=False, read_only=True))
        for name, why in moved.left_alone:
            print(f"  left alone: {name}: {why}")
        print(
            f"relocate source_file (dry run): drawers to relocate {moved.drawers} in "
            f"{len(moved.files)} files, {tail(found, moved)}"
        )
        return 0
    try:
        with module.mine_palace_lock(str(palace)):
            collection = module.get_collection(str(palace), create=False, read_only=False)
            found, moved = plan(collection)
            for name, why in moved.left_alone:
                print(f"  left alone: {name}: {why}")
            return _write_relocation(palace, convos, collection, found, moved, tail(found, moved))
    except module.MineAlreadyRunning:
        print("  refused, nothing written: another writer holds this palace (a mine, or a wake)")
        return 1


def _write_relocation(
    palace: Path, convos: Path, collection, found: Census, moved: Relocation, tail: str
) -> int:
    """Rewrite ``source_file`` on the proven drawers, whole file at a time, and read each back.

    Counts come from this one handle. Opening the palace again here can invalidate it: when
    another process has touched ``chroma.sqlite3``, MemPalace's backend closes its clients on the
    next open, and every collection it had handed out with them (``ChromaBackend._drain_clients``).
    """
    chunker, normalize = found.chunker or 0, found.normalize
    expected = {
        drawer_id: relocated(
            meta,
            str(convos / name),
            observed=name in moved.observed,
            mtime=moved.mtimes[name],
            current=not _unfiled(meta, chunker, normalize),
        )
        for name, rows in moved.files.items()
        for drawer_id, meta in rows
    }
    before = (len(found.real), found.registry)
    for names in _whole_files(moved.files, _PAGE):
        ids = [drawer_id for name in names for drawer_id, _ in moved.files[name]]
        collection.update(ids=ids, metadatas=[expected[drawer_id] for drawer_id in ids])
    ids = list(expected)
    wrong = 0
    for start in range(0, len(ids), _PAGE):
        batch = ids[start : start + _PAGE]
        got = collection.get(ids=batch, include=["metadatas"])
        back = dict(zip(got.get("ids") or [], got.get("metadatas") or []))
        wrong += sum(
            not _reads_back(back.get(drawer_id), expected[drawer_id]) for drawer_id in batch
        )
    fresh = census(collection, palace)
    after = (len(fresh.real), fresh.registry)
    print(
        f"relocate source_file: WROTE to this palace: drawers relocated {len(ids) - wrong} in "
        f"{len(moved.files)} files, {tail}"
    )
    if wrong or before != after:
        print(
            f"FAIL: {wrong} drawers did not read back as their old metadata with the new path, "
            f"or the drawer and row counts moved ({before} -> {after}); the palace changed in "
            "another way while this ran"
        )
        return 1
    return 0


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


def query_of(text: str) -> str:
    """A probe's query: the head of its own drawer's text."""
    return text.strip()[:_QUERY_CHARS]


def digest(ids) -> str:
    """A short, order-free digest of a set of drawer ids, for comparing two reports by eye."""
    ids = sorted(set(ids))
    if not ids:
        return "none"
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


def _filed(meta: dict) -> datetime | None:
    """A drawer's ``filed_at`` as MemPalace writes it (naive local time), or ``None``."""
    try:
        when = datetime.fromisoformat(str(meta.get("filed_at") or ""))
    except ValueError:
        return None
    return when if when.tzinfo is None else None


@dataclass
class Sample:
    rows: list[tuple[str, dict, str]]
    #: Real drawers with a query, filed before the cut-off when one was given.
    population: int
    #: Drawers left out because their ``filed_at`` could not be read while a cut-off was given.
    undated: int


def sample(found: Census, size: int, before: datetime | None) -> Sample:
    """``size`` real drawers, chosen the same way on any palace that holds them.

    The order is the SHA-256 of the drawer id. A drawer's id never changes when its palace moves
    (MemPalace stores it), so a palace and its relocated copy choose the same drawers, and the
    choice does not lean toward any age, room or file the way "oldest, middle, newest" does. Every
    real drawer is eligible wherever it was filed, so a palace that never moved and its moved copy
    draw from the same population. ``before`` holds that population still: a live palace keeps
    filing drawers after a copy is taken, and a drawer filed later could otherwise displace one
    from the sample. A drawer with no text has no query and is not eligible.
    """
    pool = [row for row in found.real if query_of(row[2])]
    undated = 0
    if before is not None:
        kept = []
        for row in pool:
            when = _filed(row[1])
            if when is None:
                undated += 1
            elif when < before:
                kept.append(row)
        pool = kept
    pool.sort(key=lambda row: hashlib.sha256(row[0].encode("utf-8")).hexdigest())
    return Sample(pool[:size], len(pool), undated)


class Index:
    """Lookups the diagnosis needs, built once from the census: by id, by text, by query."""

    def __init__(self, found: Census) -> None:
        self.by_id = {drawer_id: (meta, text) for drawer_id, meta, text in found.real}
        self.texts = Counter(text.strip() for _, _, text in found.real)
        self.queries = Counter(query_of(text) for _, _, text in found.real)


def _hashes(meta: dict) -> set[str]:
    return {h for h in str(meta.get("content_hash") or "").split(",") if h}


def _self_query(open_collection: Callable[[], Any], drawer_id: str) -> str:
    """Whether the vector index returns the drawer for its own stored embedding.

    A check on the index, not on the query: "no" means the drawer cannot enter the vector half of
    any search, whatever is asked. Handed an opener, never a handle, because it runs after a search
    (`MemPalaceMemoryProvider.read_collection`).
    """
    try:
        collection = open_collection()
        got = collection.get(ids=[drawer_id], include=["embeddings"])
        embeddings = got.get("embeddings")
        if embeddings is None or len(embeddings) == 0 or embeddings[0] is None:
            return "no stored embedding"
        result = collection.query(
            query_embeddings=[list(embeddings[0])], n_results=_DEEP, include=["distances"]
        )
        ids = (result.get("ids") or [[]])[0]
        return "yes" if drawer_id in ids else f"no (not in its own top {_DEEP})"
    except Exception as error:  # noqa: BLE001 - a diagnosis line, never a crash of the check
        return f"unknown ({type(error).__name__})"


def _vector_rank(
    open_collection: Callable[[], Any], query: str, drawer_id: str
) -> tuple[int | None, str]:
    """Where the vector half of a top-10 search places the drawer: ``(rank, rendered)``.

    Asked exactly as that search asks it, for ``_VECTOR_ASK`` candidates, because the index is
    approximate (HNSW) and what it returns depends on how many results are requested: a drawer
    absent from a 100-candidate answer can be present in a 300-candidate one. Handed an opener,
    as `_self_query` is.
    """
    try:
        result = open_collection().query(
            query_texts=[query], n_results=_VECTOR_ASK, include=["distances"]
        )
    except Exception as error:  # noqa: BLE001 - a diagnosis line, never a crash of the check
        return None, f"vector rank unknown ({type(error).__name__})"
    ids = (result.get("ids") or [[]])[0]
    if drawer_id not in ids:
        return None, f"vector rank: not in the {_VECTOR_ASK} candidates"
    rank = ids.index(drawer_id) + 1
    return rank, f"vector rank {rank}/{_VECTOR_ASK}"


def diagnose(provider, index: Index, row: tuple[str, dict, str], hits) -> tuple[str, str]:
    """Why a probe drawer is not in its top 10, as ``(line, verdict)``: ids, counts and ranks only.

    How MemPalace 3.9's union search (the harness's ``candidate_strategy``) fills the top k, which
    is what every verdict below is read against: the harness asks for ``n`` results (``_ASK``,
    the unranked headroom times the 10 it keeps, issue #611); the vector index proposes ``3n``
    drawers and only the ``n`` nearest are kept; BM25 adds its own top ``3n``; the merged pool is
    ranked by ``0.6 * similarity + 0.4 * BM25`` (BM25 scaled to the pool's best) and the harness
    keeps the first 10. A drawer the vector half did not keep scores on BM25 alone, at most 0.4,
    and loses to any close vector match. Exact distance ties are kept in the index's own order, and
    equal final scores go to the newer drawer (``authored_at``).

    The probe's query is the head of the drawer, while the drawer's embedding is of the whole
    chunk, so its own query need not find it nearest. A miss is one of four kinds:

    - **twin**: a drawer with byte-identical text holds a top-10 slot. The memory is recalled,
      under another id. Identical text embeds identically, so when more copies exist than slots the
      index's tie order decides which copies are kept, and the rest miss.
    - **cut**: a top-100 search ranks it in the top 10, but the vector half of a top-10 search
      does not place it among the ``_ASK`` nearest it keeps, so that search dropped its vector
      score before ranking. Full scoring would have placed it; the pre-rank cut did not let it
      compete. Where `search` passes ``max_distance`` (issue #625) a lexical hit is rescored on its
      real distance, so a cut there also means BM25 did not bring it into the pool either. From
      MemPalace 3.11 a search whose folded copies (identical text from different files) leave it
      short is repeated with a wider vector half, which this judgment does not model;
      ``--pool-diagnosis`` does (issue #742).
    - **crowded**: a top-100 search ranks it below 10, and no identical copy is in the top 10:
      other drawers outscore it even on full scoring, typically near copies sharing its query.
    - **unreached**: not in the top 100 at all. With ``vector self-query no`` the index itself has
      lost it; with ``yes``, its head is simply a poor match for its own embedding.
    """
    drawer_id, meta, text = row
    body = text.strip()
    query = query_of(text)
    mine = _hashes(meta)
    source = meta.get("source_file")
    copies = index.texts[body] - 1
    later = 0
    if copies:
        filed = str(meta.get("filed_at") or "")
        later = sum(
            str(other.get("filed_at") or "") > filed
            for other_id, (other, other_text) in index.by_id.items()
            if other_id != drawer_id and other_text.strip() == body
        )
    tops = [index.by_id.get(hit.get("drawer_id")) for hit in hits]
    tops = [top for top in tops if top is not None]
    identical = sum(other_text.strip() == body for _, other_text in tops)
    same_query = sum(query_of(other_text) == query for _, other_text in tops)
    same_hash = sum(bool(mine & _hashes(other)) for other, _ in tops)
    same_source = sum(other.get("source_file") == source for other, _ in tops)

    deep = provider.search(query, _DEEP, surface=SURFACE_TOOL)
    ranked = [hit.get("drawer_id") for hit in deep]
    if drawer_id in ranked:
        rank = ranked.index(drawer_id) + 1
        hit = deep[rank - 1]
        where = (
            f"deep rank {rank}/{_DEEP} (similarity {hit.get('similarity')}, "
            f"via {hit.get('matched_via')})"
        )
    else:
        rank = None
        where = f"not in top {_DEEP}"
    vector, vector_line = _vector_rank(provider.read_collection, query, drawer_id)
    vector_known = not vector_line.startswith("vector rank unknown")
    if identical:
        verdict = "twin"
    elif rank is None:
        verdict = "unreached"
    elif rank <= _TOP and vector_known and (vector is None or vector > _ASK):
        verdict = "cut"
    else:
        verdict = "crowded"
    return (
        f"  why: {verdict}; {where}; {vector_line}; "
        f"vector self-query {_self_query(provider.read_collection, drawer_id)}; "
        f"drawers with identical text {copies} ({later} filed later), "
        f"with the same query {index.queries[query] - 1}; top {_TOP}: {identical} identical text, "
        f"{same_query} same query, {same_hash} same content hash, {same_source} same source file"
    ), verdict


@dataclass
class PoolComparison:
    """What ``--reranked-pool`` found: drawer ids per arm, and each arm's search times."""

    #: Probes whose own drawer is in the pool today's rule hands the reranker.
    today: set[str] = field(default_factory=set)
    #: Probes whose own drawer is in the pool the candidate rule would hand it.
    candidate: set[str] = field(default_factory=set)
    today_seconds: list[float] = field(default_factory=list)
    candidate_seconds: list[float] = field(default_factory=list)


def compare_pools(provider: MemPalaceMemoryProvider, rows) -> PoolComparison:
    """For each probe, whether its own drawer reaches the reranker under each rule (issue #617).

    **Arm A, today's rule:** ask MemPalace for `_RERANK_POOL` (``pool_size(10)``) and keep all of
    it, which is what `MemPalaceMemoryProvider.search` hands a bound reranker. **Arm B, the
    candidate rule:** ask for `_CANDIDATE_ASK` and keep the first `_RERANK_POOL`. Both are fetched
    by the provider's own `MemPalaceMemoryProvider._ranking`, the call `search` makes, so registry
    sentinels are dropped and the fetch widened exactly as there. No reranker runs: the question is
    what the model would be shown, not what it would pick, so the answer costs no model call.

    A drawer in the pool is not a drawer recalled (the reranker still picks 10 of it); a drawer
    outside it can never be recalled, whatever the model would have chosen.
    """
    found = PoolComparison()
    # The threshold `search` passes on this palace (issue #625), in both arms: they differ in the
    # ask and nothing else.
    threshold, _ = provider.distance_threshold()
    for drawer_id, _, text in rows:
        query = query_of(text)
        for ask, ids, seconds in (
            (_RERANK_POOL, found.today, found.today_seconds),
            (_CANDIDATE_ASK, found.candidate, found.candidate_seconds),
        ):
            started = time.monotonic()
            hits, _, _, _ = provider._ranking(
                query, ask=ask, need=_RERANK_POOL, surface=SURFACE_TOOL, max_distance=threshold
            )
            seconds.append(time.monotonic() - started)
            if drawer_id in [hit.get("drawer_id") for hit in hits[:_RERANK_POOL]]:
                ids.add(drawer_id)
    return found


def _median(seconds: list[float]) -> str:
    return f"{statistics.median(seconds):.3f}s" if seconds else "n/a"


def _report_pools(pools: PoolComparison, rows) -> None:
    """The ``--reranked-pool`` report: one line per arm, then the summary two reports compare on."""
    probes_n = len(rows)
    print(
        f"reranked pool arm A (ask {_RERANK_POOL}, keep {_RERANK_POOL}): found {len(pools.today)} "
        f"of {probes_n}, median search {_median(pools.today_seconds)}"
    )
    print(
        f"reranked pool arm B (ask {_CANDIDATE_ASK}, keep first {_RERANK_POOL}): found "
        f"{len(pools.candidate)} of {probes_n}, median search {_median(pools.candidate_seconds)}"
    )
    gained = pools.candidate - pools.today
    lost = pools.today - pools.candidate
    # Last, because it is the line two reports are compared on: the same sample digest and the same
    # per-arm digests mean the same drawers moved.
    print(
        f"reranked summary: probes {probes_n}, sample digest {digest(row[0] for row in rows)}, "
        f"arm A found {len(pools.today)}, arm B found {len(pools.candidate)}, "
        f"B finds A misses {len(gained)} (digest {digest(gained)}), "
        f"A finds B misses {len(lost)} (digest {digest(lost)})"
    )


def _cutoff(value: str) -> datetime:
    """``--filed-before``: an ISO timestamp, naive, because MemPalace's ``filed_at`` is naive."""
    try:
        when = datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO timestamp: {value!r}") from None
    if when.tzinfo is not None:
        raise argparse.ArgumentTypeError(
            "give the time without a UTC offset, in the box's local time, as filed_at records it"
        )
    return when


def _count(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = -1
    if number < 0:
        raise argparse.ArgumentTypeError(f"not a count: {value!r}")
    return number


def _absolute(value: str) -> Path:
    """``--previous-home``: an absolute path, because drawers record absolute paths."""
    if not os.path.isabs(value):
        raise argparse.ArgumentTypeError(f"not an absolute path: {value!r}")
    return Path(os.path.abspath(value))


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"not a positive count: {value!r}")
    return number


def main(argv: list[str] | None = None) -> int:
    """The ``basecradle-harness-palace-check`` entrypoint."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Prove an agent's MemPalace palace opens from the given HARNESS_HOME and returns drawers "
            "filed before a home-directory move. No platform call, no model call (except with "
            "--end-to-end), and no drawer text is ever printed. Read-only unless --practice-observe, "
            "--register-off-wing or --stamp-chunker-version is given. Exit 0 only if every probe "
            "drawer comes back."
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
    parser.add_argument(
        "--register-off-wing",
        action="store_true",
        help=(
            "WRITES TO THE PALACE. Before the report, register each moved conversation file whose "
            "content is filed only in another wing, in the wing its drawers carry: one registry "
            "row per file and no drawer, so a mine does not file it again. Refuses, "
            "writing nothing, if any such file would file a drawer. Run after a home move and "
            "before the first wake or any mempalace mine. Add --dry-run to print what it would do."
        ),
    )
    parser.add_argument(
        "--stamp-chunker-version",
        action="store_true",
        help=(
            "WRITES TO THE PALACE. Before the report, stamp each drawer the harness's observe "
            "filed under an older MemPalace with the installed MemPalace's "
            "convo_chunker_version, so MemPalace 3.11 and later read it as filed. Only drawers "
            "whose id proves the observe filed them; MemPalace's miner's drawers are never "
            "touched. Changes no other field, and a second run stamps nothing. Run once after "
            "upgrading MemPalace. Add --dry-run to print what it would do."
        ),
    )
    parser.add_argument(
        "--relocate-source-file",
        action="store_true",
        help=(
            "WRITES TO THE PALACE. Before the report, rewrite source_file to this location on the "
            "drawers of each conversation file filed only under --previous-home, so a mine "
            "re-chunks it in place, or skips it, instead of filing it a second time. Each file "
            "must prove it was moved: its recorded path gone, the same content here. Before "
            "MemPalace 3.12 only MemPalace's miner's version-stale files are moved; from 3.12 "
            "every file that proves itself, an observe file only when it is current, keeping the "
            "path its id was derived from. Deletes nothing, and a second run finds nothing. Add "
            "--dry-run to print what it would do."
        ),
    )
    parser.add_argument(
        "--previous-home",
        type=_absolute,
        help="with --relocate-source-file: the HARNESS_HOME this palace used to live under.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "with --relocate-source-file, --register-off-wing or --stamp-chunker-version: print "
            "what it would write, and "
            "write nothing. With --end-to-end: print the probes and the token estimate, and call "
            "no model."
        ),
    )
    parser.add_argument(
        "--sample",
        type=_positive,
        metavar="N",
        help=(
            "probe N drawers instead of three, chosen by the SHA-256 of their drawer id, so two "
            "palaces holding the same drawers probe the same ones wherever they sit. Ends with one "
            "summary line to compare between two reports."
        ),
    )
    parser.add_argument(
        "--filed-before",
        type=_cutoff,
        metavar="TIME",
        help=(
            "with --sample: draw only from drawers filed before TIME (ISO, the box's local time, "
            "as filed_at records it), so a live palace that keeps growing samples the same drawers "
            "as a copy taken at TIME."
        ),
    )
    parser.add_argument(
        "--reranked-pool",
        action="store_true",
        help=(
            f"with --sample: measure the reranked path instead, with no model call. For each "
            f"probe, is its own drawer in the {_RERANK_POOL} candidates a reranked search hands "
            f"its reranker today (ask {_RERANK_POOL}, arm A), and in the first {_RERANK_POOL} of "
            f"an ask for {_CANDIDATE_ASK} (arm B)? Ends with one summary line of both arms. Exit 0 "
            f"only if arm A holds every probe drawer."
        ),
    )
    parser.add_argument(
        "--pool-diagnosis",
        action="store_true",
        help=(
            "with --sample: token-free and read-only. Fetch the probes --end-to-end draws through "
            f"its three arms (ask {_RERANK_POOL} with max_distance 2.0, as today; ask "
            f"{_RERANK_POOL} with none; ask {_CANDIDATE_ASK} with max_distance 2.0), and for each "
            f"probe whose drawer is not in today's pool say why: its rank in the vector half and "
            "the lexical half, how far from the cut, the first wider ask that reaches it, any "
            "identical or near copy in the pool, sentinels and the widening cap, and a lost "
            "embedding or lexical row. Ids, counts and ranks only. Exit 0 only if today's pool "
            "holds every probe drawer."
        ),
    )
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help=(
            "SPENDS RERANK-MODEL TOKENS. With --sample and --token-ceiling: run each probe through "
            "the agent's own reranker (its HARNESS_MEMPALACE_RERANK_* configuration) in four arms "
            f"(ask {_RERANK_POOL} with max_distance 2.0, as today; the same again, for the noise "
            f"floor; ask {_RERANK_POOL} with no max_distance, the search before 0.144.0; ask "
            f"{_CANDIDATE_ASK} with max_distance 2.0) "
            f"and report whether the probe's own drawer is in the final {_TOP}. Read-only on the "
            "palace. Exit 0 only for a complete run."
        ),
    )
    parser.add_argument(
        "--rare-token-probes",
        type=_count,
        default=0,
        metavar="M",
        help=(
            "with --end-to-end or --pool-diagnosis: also probe M drawers by their rarest exact "
            "token alone, reported separately from the --sample head probes."
        ),
    )
    parser.add_argument(
        "--token-ceiling",
        type=_positive,
        metavar="T",
        help=(
            "with --end-to-end, and required by it: the most rerank-model tokens (in plus out, "
            "every arm) the run may spend. It refuses to start when its estimate exceeds T, and "
            "stops, marked partial, before a probe that would."
        ),
    )
    args = parser.parse_args(argv)
    if args.end_to_end:
        if args.sample is None or args.token_ceiling is None:
            parser.error("--end-to-end needs --sample and --token-ceiling")
        if (
            args.reranked_pool
            or args.practice_observe
            or args.register_off_wing
            or args.stamp_chunker_version
            or args.relocate_source_file
        ):
            parser.error(
                "--end-to-end runs alone: not with --reranked-pool, --practice-observe, "
                "--register-off-wing, --stamp-chunker-version or --relocate-source-file"
            )
    elif args.token_ceiling is not None:
        parser.error("--token-ceiling applies to --end-to-end")
    if args.pool_diagnosis:
        if args.sample is None:
            parser.error("--pool-diagnosis applies to --sample")
        if (
            args.end_to_end
            or args.reranked_pool
            or args.practice_observe
            or args.register_off_wing
            or args.stamp_chunker_version
            or args.relocate_source_file
        ):
            parser.error(
                "--pool-diagnosis runs alone: not with --end-to-end, --reranked-pool, "
                "--practice-observe, --register-off-wing, --stamp-chunker-version or "
                "--relocate-source-file"
            )
    elif args.rare_token_probes and not args.end_to_end:
        parser.error("--rare-token-probes applies to --end-to-end or --pool-diagnosis")
    if args.filed_before is not None and args.sample is None:
        parser.error("--filed-before applies to --sample")
    if args.reranked_pool and args.sample is None:
        parser.error("--reranked-pool applies to --sample")
    if args.dry_run and not (
        args.register_off_wing
        or args.stamp_chunker_version
        or args.relocate_source_file
        or args.end_to_end
    ):
        parser.error(
            "--dry-run applies to --register-off-wing, --stamp-chunker-version, "
            "--relocate-source-file or --end-to-end"
        )
    if args.relocate_source_file and args.previous_home is None:
        parser.error("--relocate-source-file needs --previous-home")
    if args.previous_home is not None and not args.relocate_source_file:
        parser.error("--previous-home applies to --relocate-source-file")

    reranker = None
    if args.end_to_end:
        # Read before the clear below switches reranking off for everything else this check does.
        reranker = reranker_from_env()
        if reranker is None:
            print(
                f"{PROG}: --end-to-end runs the agent's own reranker, and {RERANK_MODEL_VAR} is "
                "not set",
                file=sys.stderr,
            )
            return 1
        if reranker.fault:
            print(
                f"{PROG}: --end-to-end: the agent's rerank configuration is incomplete "
                f"({reranker.fault})",
                file=sys.stderr,
            )
            return 1
    for name in _CLEARED_ENV:
        os.environ.pop(name, None)
    raw_home = args.harness_home or os.environ.get("HARNESS_HOME")
    if not raw_home:
        print(f"{PROG}: name a HARNESS_HOME, or set $HARNESS_HOME", file=sys.stderr)
        return 1
    home = Path(raw_home).expanduser().resolve()
    if args.previous_home == home:
        parser.error("--previous-home is this HARNESS_HOME")
    palace = _palace_path(home)
    # `--pool-diagnosis` prints ids, counts and ranks only (issue #633): no path, no upstream text.
    paths = not args.pool_diagnosis
    print(
        f"palace: {palace}" if paths else "palace: under the HARNESS_HOME given (path not printed)"
    )
    if not palace.is_dir():
        print(
            f"FAIL: no palace at {palace}" if paths else "FAIL: no palace under that HARNESS_HOME"
        )
        return 1

    # A sample runs hundreds of searches; one `memory recall` line each would bury the report.
    level = logging.WARNING if args.sample else logging.INFO
    logging.basicConfig(level=level, handlers=[log_handler("%(message)s", sys.stderr)])
    try:
        provider = MemPalaceMemoryProvider(palace)
        if args.relocate_source_file:
            code = relocate_source_file(palace, home, args.previous_home, dry_run=args.dry_run)
            if code:
                return code
        # Before registering: a stamped file is recognised by its content, so the register pass
        # then sees only what is still unfiled (issue #751).
        if args.stamp_chunker_version:
            code = stamp_chunker_version(palace, dry_run=args.dry_run)
            if code:
                return code
        if args.register_off_wing:
            code = register_off_wing(provider, palace, dry_run=args.dry_run)
            if code:
                return code
        if args.practice_observe:
            seconds = practice_observe(provider)
            print(
                f"practice observe: WROTE one exchange to this palace and filed it "
                f"({seconds:.1f}s). Practice copies only."
            )
        try:
            collection = _import("palace").get_collection(str(palace), create=False, read_only=True)
        except ImportError:
            raise
        except Exception as error:  # noqa: BLE001 - a vendor refusal of any class, relayed in one line
            detail = f": {error}" if paths else ""
            print(f"FAIL: could not open the palace: {type(error).__name__}{detail}")
            return 1
        found = census(collection, palace)
        print(
            f"drawers: {found.total} total, {len(found.real)} real, "
            f"{found.registry} registry sentinels"
        )
        print(f"real drawers filed at another location: {len(found.elsewhere)}")
        print(
            "real chunk-0 drawers without content_hash (MemPalace's miner stamps none before 3.7 "
            f"or from 3.12): {found.unhashed}"
        )
        print(
            f"conversation files: {found.files}, "
            f"not yet registered at this location: {found.unregistered}"
        )
        print(f"conversation files filed short of their drawers: {found.short}")
        print(stale_line(found))
        if found.stale_files_ours or found.stale_files_theirs:
            print(f"  {STALE_CONSEQUENCE}")
        print(normalize_line(found))
        if found.normalize_files_ours or found.normalize_files_theirs:
            print(f"  {NORMALIZE_CONSEQUENCE}")
        # The forecast names conversation files, and is the move check's business. Its line is
        # matched whole by the NOC's rename flow (`FORECAST_RE` in basecradle-noc), as is the
        # "conversation files:" line above (`forecast_line`).
        stale = (
            found.stale_files_ours
            + found.stale_files_theirs
            + found.normalize_files_ours
            + found.normalize_files_theirs
        )
        if (found.unregistered or found.short or stale) and paths:
            predicted = forecast(palace, provider.agent)
            print(forecast_line(predicted))
            if predicted is not None and predicted.new:
                convos = palace / _CONVERSATIONS_WING
                accepted = accepted_hashes(provider.read_collection) if found.by_content else None
                for new_name in predicted.new:
                    print(new_file_diagnosis(found, convos, new_name, accepted))
        if reranker is not None:
            from basecradle_harness import _palace_recall

            try:
                return _palace_recall.main(
                    provider,
                    reranker,
                    collection,
                    found,
                    heads=args.sample,
                    rares=args.rare_token_probes,
                    before=args.filed_before,
                    ceiling=args.token_ceiling,
                    dry_run=args.dry_run,
                    sample=sample,
                    query_of=query_of,
                )
            finally:
                reranker.close()
        if args.pool_diagnosis:
            from basecradle_harness import _palace_pool

            return _palace_pool.main(
                provider,
                collection,
                found,
                heads=args.sample,
                rares=args.rare_token_probes,
                before=args.filed_before,
                sample=sample,
                query_of=query_of,
                self_query=_self_query,
            )
        if args.sample:
            drawn = sample(found, args.sample, args.filed_before)
            rows, elsewhere = drawn.rows, False
        else:
            rows, elsewhere = probes(found)
        if not rows:
            print("FAIL: the palace holds no real drawers to search for")
            return 1
        index = Index(found)
        failing: list[str] = []
        verdicts: Counter[str] = Counter()
        pools = compare_pools(provider, rows) if args.reranked_pool else None
        if pools is not None:
            failing = [row[0] for row in rows if row[0] not in pools.today]
            for label, ids in (
                ("arm B only", pools.candidate - pools.today),
                ("arm A only", pools.today - pools.candidate),
            ):
                for drawer_id in sorted(ids):
                    print(f"{label}: drawer {drawer_id}")
        else:
            for row in rows:
                drawer_id, meta, text = row
                hits = provider.search(query_of(text), _TOP, surface=SURFACE_TOOL)
                ok = drawer_id in [hit.get("drawer_id") for hit in hits]
                if ok and args.sample:
                    continue
                print(
                    f"{'PASS' if ok else 'FAIL'}: drawer {drawer_id} (filed {meta.get('filed_at')}, "
                    f"source {meta.get('source_file')})"
                )
                if not ok:
                    failing.append(drawer_id)
                    line, verdict = diagnose(provider, index, row, hits)
                    verdicts[verdict] += 1
                    print(line)
        if not args.sample:
            where = (
                "filed at another location" if elsewhere else "any location (none filed elsewhere)"
            )
            print(f"probes drawn from drawers {where}")
        name = home.parent.name
        hits = provider.search(name, _TOP, surface=SURFACE_TOOL)
        junk = sum(str(hit.get("text", "")).startswith("[registry]") for hit in hits)
        # In the default mode this is last, because it is the line the relocation's registry rows
        # are judged on: what recall hands the agent for its own home's name. With issue #606's fix
        # it is 0 whatever the palace holds.
        print(f"registry sentinels in top {_TOP} for query {name!r}: {junk}")
        if args.sample:
            cutoff = (
                f"filed before {args.filed_before.isoformat()}, {drawn.undated} undated left out"
                if args.filed_before
                else "any filing time"
            )
            print(
                f"sample population: {drawn.population} drawers ({cutoff}); "
                f"sample digest {digest(row[0] for row in rows)}"
            )
            if pools is not None:
                _report_pools(pools, rows)
            else:
                print(
                    "failed by verdict: "
                    + ", ".join(
                        f"{v} {verdicts[v]}" for v in ("twin", "cut", "crowded", "unreached")
                    )
                )
                # Last, because it is the line two reports are compared on: the same probes and
                # the same failing digest mean the same drawers failed.
                print(
                    f"sample summary: probes {len(rows)}, passed {len(rows) - len(failing)}, "
                    f"failed {len(failing)}, failing digest {digest(failing)}"
                )
    except ImportError as error:
        print(f"{PROG}: {last_word(error)}", file=sys.stderr)
        return 1
    return 1 if failing else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
