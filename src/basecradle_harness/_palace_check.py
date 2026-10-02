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

**Three probes, or a sample** (issue #606, phase 3). By default the check searches for three real
drawers (the oldest, middle and newest), which says whether a moved palace serves at all. Three
cannot say whether a move made recall *worse*, so ``--sample N`` probes N drawers instead, chosen
by the SHA-256 of their drawer id: a choice that does not depend on where the palace sits, so two
palaces holding the same drawers probe the same ones, and ``--filed-before`` holds the population
still while a live palace keeps growing. The sample ends with one comparable summary line, and two
reports with the same sample digest and the same failing digest failed on the same drawers.

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
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
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

#: How deep a miss is looked for. Deep enough that a drawer merely crowded out of the top 10 is
#: found, so "not in the top 100" means recall cannot reach it in practice.
_DEEP = 100

#: How many candidates a top-10 search asks the vector index for: MemPalace 3.9 asks for three
#: times the requested count (``_candidate_pool_size``) and, in union mode, keeps the nearest ten.
_VECTOR_ASK = 3 * _TOP

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


@dataclass
class Forecast:
    """What the next observe will do to the conversation files, per MemPalace's own dry run."""

    #: Files whose content is already filed elsewhere: each gets one registry row, no drawers.
    duplicate: int
    #: Names of the files that will be mined as new drawers.
    new: list[str]


def forecast(palace: Path, agent: str) -> Forecast | None:
    """What the next observe will do: which files get a registry row and which are mined as new.

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
    # "    [DRY RUN] <name> -> room:<room> (<n> drawers)"
    new = [
        line.split("[DRY RUN]", 1)[1].split("->", 1)[0].strip()
        for line in lines
        if "[DRY RUN]" in line and "->" in line
    ]
    return Forecast(duplicate, new) if duplicate or new else None


def new_file_diagnosis(found: Census, convos: Path, name: str) -> str:
    """Why the next observe will mine ``name`` as new drawers, in counts and flags only.

    A moved file is recognised by the content hash its drawers carry, so a file forecast as new
    either has no drawers anywhere (it was never mined) or has drawers whose recorded hash the
    miner does not accept: no hash, an older ``normalize_version``, another extract mode, or a hash
    that no longer matches the file. If its drawers exist elsewhere, mining it again adds a second
    copy of each, and the two copies then compete for the same slots (see `diagnose`): a move that
    re-mines a file can turn a drawer that passed into a ``twin`` miss.
    """
    rows = [
        meta for _, meta, _ in found.real if Path(str(meta.get("source_file") or "")).name == name
    ]
    here = str((convos / name).resolve())
    at_here = sum(str(meta.get("source_file")) == here for meta in rows)
    hashed = [meta for meta in rows if meta.get("content_hash")]
    stored = {h for meta in hashed for h in str(meta["content_hash"]).split(",") if h}
    versions = sorted({str(meta.get("normalize_version", 1)) for meta in rows}) or ["-"]
    modes = sorted({str(meta.get("extract_mode") or "-") for meta in rows}) or ["-"]
    try:
        conversations = [
            c for c in _import("normalize").normalize_conversations(str(convos / name)) if c
        ]
        today = {hashlib.sha256(c.strip().encode("utf-8")).hexdigest() for c in conversations}
        matches = "yes" if today & stored else "no"
    except ImportError:
        raise
    except Exception as error:  # noqa: BLE001 - a diagnosis line, never a crash of the check
        matches = f"unknown ({type(error).__name__})"
    return (
        f"  mined as new: {name}: {len(rows)} drawers recorded under this file name "
        f"({at_here} at this location); content_hash on {len(hashed)}; "
        f"normalize_version {','.join(versions)}; extract_mode {','.join(modes)}; "
        f"today's content hash matches a recorded one: {matches}"
    )


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


def _self_query(collection, drawer_id: str) -> str:
    """Whether the vector index returns the drawer for its own stored embedding.

    A check on the index, not on the query: "no" means the drawer cannot enter the vector half of
    any search, whatever is asked.
    """
    try:
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


def _vector_rank(collection, query: str, drawer_id: str) -> tuple[int | None, str]:
    """Where the vector half of a top-10 search places the drawer: ``(rank, rendered)``.

    Asked exactly as that search asks it, for ``_VECTOR_ASK`` candidates, because the index is
    approximate (HNSW) and what it returns depends on how many results are requested: a drawer
    absent from a 100-candidate answer can be present in a 300-candidate one.
    """
    try:
        result = collection.query(query_texts=[query], n_results=_VECTOR_ASK, include=["distances"])
    except Exception as error:  # noqa: BLE001 - a diagnosis line, never a crash of the check
        return None, f"vector rank unknown ({type(error).__name__})"
    ids = (result.get("ids") or [[]])[0]
    if drawer_id not in ids:
        return None, f"vector rank: not in the {_VECTOR_ASK} candidates"
    rank = ids.index(drawer_id) + 1
    return rank, f"vector rank {rank}/{_VECTOR_ASK}"


def diagnose(
    provider, collection, index: Index, row: tuple[str, dict, str], hits
) -> tuple[str, str]:
    """Why a probe drawer is not in its top 10, as ``(line, verdict)``: ids, counts and ranks only.

    How MemPalace 3.9's union search (the harness's ``candidate_strategy``) fills the top k, which
    is what every verdict below is read against: the vector index proposes ``3k`` drawers and only
    the ``k`` nearest are kept; BM25 adds its own top ``3k``; the merged pool is ranked by
    ``0.6 * similarity + 0.4 * BM25`` (BM25 scaled to the pool's best) and cut to ``k``. A drawer
    the vector half did not keep scores on BM25 alone, at most 0.4, and loses to any close vector
    match. Exact distance ties are kept in the index's own order, and equal final scores go to the
    newer drawer (``authored_at``).

    The probe's query is the head of the drawer, while the drawer's embedding is of the whole
    chunk, so its own query need not find it nearest. A miss is one of four kinds:

    - **twin**: a drawer with byte-identical text holds a top-10 slot. The memory is recalled,
      under another id. Identical text embeds identically, so when more copies exist than slots the
      index's tie order decides which copies are kept, and the rest miss.
    - **cut**: a top-100 search ranks it in the top 10, but the vector half of a top-10 search
      does not place it among its nearest ten, so that search dropped its vector score before
      ranking. Full scoring would have placed it; the pre-rank cut did not let it compete.
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
    vector, vector_line = _vector_rank(collection, query, drawer_id)
    vector_known = not vector_line.startswith("vector rank unknown")
    if identical:
        verdict = "twin"
    elif rank is None:
        verdict = "unreached"
    elif rank <= _TOP and vector_known and (vector is None or vector > _TOP):
        verdict = "cut"
    else:
        verdict = "crowded"
    return (
        f"  why: {verdict}; {where}; {vector_line}; "
        f"vector self-query {_self_query(collection, drawer_id)}; "
        f"drawers with identical text {copies} ({later} filed later), "
        f"with the same query {index.queries[query] - 1}; top {_TOP}: {identical} identical text, "
        f"{same_query} same query, {same_hash} same content hash, {same_source} same source file"
    ), verdict


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
            "filed before a home-directory move. No platform call, no model call, and no drawer "
            "text is ever printed. Read-only unless --practice-observe is given. Exit 0 only if "
            "every probe drawer comes back."
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
    args = parser.parse_args(argv)
    if args.filed_before is not None and args.sample is None:
        parser.error("--filed-before applies to --sample")

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

    # A sample runs hundreds of searches; one `memory recall` line each would bury the report.
    level = logging.WARNING if args.sample else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)
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
                    f"next observe (dry run): {predicted.duplicate} files -> registry sentinel, "
                    f"{len(predicted.new)} files -> mined as new drawers"
                )
                convos = palace / _CONVERSATIONS_WING
                for new_name in predicted.new:
                    print(new_file_diagnosis(found, convos, new_name))
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
                line, verdict = diagnose(provider, collection, index, row, hits)
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
            print(
                "failed by verdict: "
                + ", ".join(f"{v} {verdicts[v]}" for v in ("twin", "cut", "crowded", "unreached"))
            )
            # Last, because it is the line two reports are compared on: the same probes and the
            # same failing digest mean the same drawers failed.
            print(
                f"sample summary: probes {len(rows)}, passed {len(rows) - len(failing)}, "
                f"failed {len(failing)}, failing digest {digest(failing)}"
            )
    except ImportError as error:
        print(f"{PROG}: {error}", file=sys.stderr)
        return 1
    return 1 if failing else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
