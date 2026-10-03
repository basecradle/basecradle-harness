"""``basecradle-harness-palace-check --pool-diagnosis`` — why a probe is not in the pool (issue #633).

The end-to-end run on a real palace (basecradle-noc#957) had probes whose own drawer was in **no
arm's pool**, even at an ask of 40, and its report could only say "not in the pool". This mode says
why, for the same probes: the head probes ``--sample`` draws (and the rare-token probes
``--rare-token-probes`` draws), chosen exactly as ``--end-to-end`` chooses them, through the same
three fetches (arms 1, 2 and 3; arm 1R is arm 1 again). It is **token-free and read-only**: no
reranker runs, no model is called, nothing is written. It prints counts, drawer ids, ranks and
digests: never memory text, never a query, never a path.

**How MemPalace fills the pool**, which every fact below is read against (MemPalace 3.9 and 3.10,
``search_memories`` with ``candidate_strategy="union"``, as `MemPalaceMemoryProvider._ranking`
calls it with ``n_results=F``, the fetch width):

1. **The vector half** proposes the ``3F`` nearest drawers (``_candidate_pool_limits``) and keeps
   the ``F`` nearest of them.
2. **The lexical half** asks the backend's ``lexical_search`` for ``3F`` drawers by BM25. On the
   Chroma backend that is an FTS ``MATCH`` of the query's tokens, **capped at a scan window** of
   ``max(500, 3F)`` rows taken in index order, with no ordering by relevance, and BM25 then scores
   only the rows inside the window. A query made of common words matches far more than 500 drawers,
   so the lexical half of a head probe sees only the window's drawers (the oldest in a palace that
   only grows), however well a newer drawer matches. ``scan window`` below is that cap, read off the
   backend's own default (``_lexical_search_via_sqlite``), never restated here.
3. **The merge** adds each lexical hit the vector half did not keep, in lexical order, dropping
   one that shares a ``(source_file, chunk_index)`` with a drawer already admitted (**shadowed**: a
   second copy filed under the same path, issue #606) and, under ``max_distance``, one whose stored embedding cannot be loaded
   (**dropped**, the #625 risk).
4. **The hybrid rank** scores every candidate ``0.6 × similarity + 0.4 × BM25`` (BM25 over the
   candidates, scaled to the best) and MemPalace returns the first ``F``. The harness drops registry
   sentinels from those and, while that leaves it short, fetches again at twice the width, up to
   ``8×`` the ask (`_MAX_FETCH_FACTOR`); the first 20 real drawers are the pool.

**The candidate set and the hybrid rank are reconstructed, and the reconstruction checks itself.**
MemPalace returns only the first ``F``, so where a drawer ranked below them is not something any
call says. This mode rebuilds steps 1 to 4 for the probe's query with MemPalace's own functions
(the merge, the hybrid rank, the dedupe, the stop words and weights it resolves), replicating only
the loop that turns the vector half's rows into hits, and then asks the real ``search_memories`` the
same question: every line says whether the reconstruction's first ``F`` match the search's. A probe
whose reconstruction does not match gets the verdict ``unknown``: its vector and lexical ranks are
still direct measurements, and its merge facts are not trusted. A palace with **closets** (none of
the harness's own mining writes them) boosts vector ranks in a way this mode does not model; the
count is printed and the self-check is what catches it.

**The verdicts**, first match wins; the first two are the rerank-off report's ``twin`` (issue #606)
carried to the pool, the rest name the stage that kept the drawer out:

- **twin**: a drawer with byte-identical text is in the pool. The memory reaches the reranker under
  another id.
- **near-twin**: a drawer in the pool shares at least `NEAR_TWIN` of the probe's tokens (Jaccard over
  the distinct lowercased tokens MemPalace's BM25 reads, ``\\w{2,}``). The line prints the nearest
  pool drawer and its overlap whatever the verdict, so "near" is a number, never a judgment.
- **sentinels**: a candidate that would be in the pool but for registry sentinels ranked above it,
  with the widening at its cap.
- **crowded**: a candidate, outranked: its hybrid rank among the real candidates is past 20.
- **dropped**: the lexical half proposed it and the threshold refused it, its embedding unloadable.
- **shadowed**: the lexical half proposed it and the merge refused it: a drawer already admitted
  (kept by the vector half, or a lexical hit ranked above it) holds its ``(source_file,
  chunk_index)``, or it has no source file at all, which the merge never admits.
- **missing**: an index has lost it: no stored embedding (or its own embedding does not find it)
  and the vector half did not return it, or no lexical match for its own text even with no scan
  window.
- **fts-window**: the lexical half would have proposed it (its rank with no scan window is within
  ``3F``) but it lies outside the window, and the vector half did not keep it.
- **cut**: the vector half proposed it and did not keep it, and the lexical half would not have
  proposed it even with no window.
- **unreached**: neither half proposes it: past ``3F`` in both, with no window.
- **unknown**: the reconstruction does not match the search, or a measurement failed (named by its
  exception class, never its text).

**What it cannot tell.** Why a drawer's head is a poor match for its own embedding (the model is not
inspectable from here); whether a closet boost moved a vector rank (reported as a count, caught by
the self-check, not modelled); and what the reranker would have picked had the drawer been in the
pool (that is ``--end-to-end``). The ranks describe this palace at this moment: a live palace that
files more drawers moves them.

Exit 0 only when arm 1's pool holds every probe drawer; 1 otherwise, or when it cannot run.
"""

from __future__ import annotations

import inspect
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from basecradle_harness import _mempalace, _palace_recall
from basecradle_harness._mempalace import (
    _MAX_FETCH_FACTOR,
    _REGISTRY_ROOM,
    MemPalaceMemoryProvider,
    _import,
    palace_metric,
)
from basecradle_harness._palace_recall import (
    ARM_1,
    ARM_2,
    ARM_3,
    HEAD,
    KINDS,
    SURFACE,
    TODAY_ASK,
    Probe,
    fetch,
    select,
)

#: The arms whose pools are compared: ``--end-to-end``'s fetches (1R is arm 1 fetched again).
ARMS = (ARM_1, ARM_2, ARM_3)

#: How much of a probe's distinct tokens a pool drawer must share to be a near-twin.
NEAR_TWIN = 0.8

#: The wider asks tried, in order, to find the first whose pool holds the drawer: 2× to 32× today's.
LADDER = tuple(TODAY_ASK * 2**step for step in range(1, 6))

#: How many lexical hits MemPalace's union merge asks the backend for, per result asked: the
#: literal ``n_results * 3`` in ``_merge_bm25_union_candidates`` (the same factor `lexical_drops`
#: reads). The vector half's factor is read off ``_candidate_pool_limits`` instead.
_LEXICAL_FACTOR = 3

#: The verdicts, in the order the report counts them.
VERDICTS = (
    "twin",
    "near-twin",
    "sentinels",
    "crowded",
    "dropped",
    "shadowed",
    "missing",
    "fts-window",
    "cut",
    "unreached",
    "unknown",
)

_PROGRESS_EVERY = 10


# --- measurement ---------------------------------------------------------------


@dataclass
class Facts:
    """Everything measured about one probe whose drawer is not in arm 1's pool."""

    probe: Probe
    #: Arm name -> whether its pool holds the drawer.
    arms: dict[str, bool]
    #: The width today's fetch ended at (`_ranking`), the sentinels it dropped, and whether the
    #: widening stopped at its cap.
    fetch: int = TODAY_ASK
    sentinels: int = 0
    capped: bool = False
    #: The vector half: how many it proposed, how many it keeps, and the drawer's rank (1-based)
    #: among those proposed, ``None`` when it was not proposed.
    vector_proposed: int = 0
    vector_kept: int = 0
    vector_rank: int | None = None
    #: The drawer's own embedding finds it (`_palace_check._self_query`'s answer).
    vector_self: str = "unknown"
    #: Closets in the palace, whose boost this mode does not model; ``None`` when unreadable.
    closets: int | None = 0
    #: The lexical half today: how many it asked for and the drawer's rank among them.
    lexical_asked: int = 0
    lexical_rank: int | None = None
    #: The backend's scan window, the drawer's rank inside it, and with no window: how many
    #: drawers scored and the drawer's rank. ``window`` is ``None`` when the backend has none to read.
    window: int | None = None
    window_rank: int | None = None
    scored: int | None = None
    full_rank: int | None = None
    #: The lexical half proposed it, and the threshold refused it for want of an embedding.
    dropped: bool = False
    #: The kept drawer whose ``(source_file, chunk_index)`` the merge refused it for.
    shadowed_by: str | None = None
    #: The lexical half proposed it, and the merge refused it for having no source file, which it
    #: never admits.
    sourceless: bool = False
    #: The reconstruction: candidates after the merge (sentinels included), the drawer's rank
    #: among them with and without sentinels, and how many sentinels rank above it.
    candidates: int | None = None
    raw_rank: int | None = None
    real_rank: int | None = None
    sentinels_above: int = 0
    #: ``(shared, of)``: how many of the search's first ``fetch`` the reconstruction also ranks
    #: first, and ``matches`` when the two lists are identical.
    shared: int = 0
    matches: bool = False
    #: The first wider ask whose pool holds the drawer, ``None`` when none up to the last did.
    first_ask: int | None = None
    #: Identical-text drawers in the pool, and the pool drawer sharing most of the probe's tokens.
    identical: int = 0
    nearest: tuple[str, float] | None = None
    #: A measurement that raised, by exception class.
    error: str | None = None


def _tokens(text: str) -> set[str]:
    """The distinct lowercased tokens MemPalace's BM25 reads (``\\w{2,}``)."""
    return set(_palace_recall._TOKEN.findall(text.lower()))


def overlap(a: str, b: str) -> float:
    """Jaccard overlap of two texts' distinct tokens, ``0.0`` when either has none."""
    left, right = _tokens(a), _tokens(b)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _rank(ids: list, drawer_id: str) -> int | None:
    return ids.index(drawer_id) + 1 if drawer_id in ids else None


def _lexical_backend(collection):
    """The backend object that owns the windowed lexical search, or ``None`` when there is none.

    MemPalace wraps its backend collection (``EmbeddingCollection._inner``); the Chroma backend's
    ``_lexical_search_via_sqlite`` is where the scan window lives.
    """
    for candidate in (collection, getattr(collection, "_inner", None)):
        if candidate is not None and hasattr(candidate, "_lexical_search_via_sqlite"):
            return candidate
    return None


def scan_window(backend, asked: int) -> int | None:
    """The FTS rows the backend scans for a lexical ask of ``asked``: ``max(default, asked)``."""
    try:
        default = (
            inspect.signature(backend._lexical_search_via_sqlite)
            .parameters["max_candidates"]
            .default
        )
    except (KeyError, TypeError, ValueError):
        return None
    return max(default, asked) if isinstance(default, int) else None


def _closets(palace: Path) -> int | None:
    """How many closets the palace holds: ``0`` with no closets collection, ``None`` if unreadable.

    Opened the way MemPalace's own search opens it, read-only where this MemPalace can say so (3.9
    takes no ``read_only``).
    """
    opener = _import("palace").get_closets_collection
    extra = {"read_only": True} if "read_only" in inspect.signature(opener).parameters else {}
    try:
        return opener(str(palace), create=False, **extra).count()
    except _import("backends").CollectionNotInitializedError:
        return 0
    except Exception:  # noqa: BLE001 - reported as unreadable, never as none
        return None


def _lexical_key(hit):
    """MemPalace's merge key for a lexical hit, built from its metadata as the merge builds it."""
    meta = hit.metadata or {}
    full = meta.get("source_file", "") or ""
    return _dedup_key(
        {
            "_source_file_full": full,
            "_chunk_index": meta.get("chunk_index"),
            "source_file": Path(full).name if full else "?",
        }
    )


def _dedup_key(hit: dict):
    """MemPalace's merge key for a hit: ``(source_file, chunk_index)``, else the basename."""
    full, chunk = hit.get("_source_file_full"), hit.get("_chunk_index")
    if full and chunk is not None:
        return (full, chunk)
    return hit.get("source_file")


def reconstruct(collection, query: str, n: int, threshold: float | None):
    """MemPalace's union search for ``n`` results, unranked tail kept: ``(proposed, ranked)``.

    ``proposed`` is the vector half's rows as hits, nearest first; ``ranked`` every candidate after
    the merge, in hybrid order (sentinels included, as MemPalace ranks them). Every step but the
    first is MemPalace's own function; the first replicates ``search_memories``'s loop over the
    vector rows, without the closet boost (see the module docstring).
    """
    searcher = _import("searcher")
    metric = searcher._metric_for_collection(collection)
    stop_words = searcher._resolve_stop_words(None)
    proposed_n, kept_n = searcher._candidate_pool_limits("union", n)
    rows = collection.query(
        query_texts=[query],
        n_results=proposed_n,
        include=["documents", "metadatas", "distances"],
    )
    docs = searcher._first_or_empty(rows, "documents")
    stored = searcher._aligned_query_ids(rows, len(docs))
    dates = getattr(searcher, "_result_date_fields", None)
    distance = threshold or 0.0
    proposed: list[dict] = []
    for stored_id, doc, meta, dist in zip(
        stored,
        docs,
        searcher._first_or_empty(rows, "metadatas"),
        searcher._first_or_empty(rows, "distances"),
    ):
        meta = meta or {}
        if searcher._candidate_out_of_scope(dist, meta, distance, None, None):
            continue
        source = meta.get("source_file", "") or ""
        proposed.append(
            {
                "drawer_id": searcher._result_drawer_id(meta, stored_id),
                "text": doc or "",
                "room": meta.get("room", "unknown"),
                "source_file": Path(source).name if source else "?",
                "source_path": source,
                **(
                    dates(meta)
                    if dates is not None
                    else {"authored_at": meta.get("authored_at", meta.get("filed_at", "unknown"))}
                ),
                "distance": round(dist, 4),
                "matched_via": "drawer",
                "_sort_key": dist,
                "_source_file_full": source,
                "_chunk_index": meta.get("chunk_index"),
                "_parent_drawer_id": meta.get("parent_drawer_id"),
            }
        )
    proposed.sort(key=lambda hit: hit["_sort_key"])
    hits = [dict(hit) for hit in proposed[:kept_n]]
    searcher._apply_candidate_strategy(
        "union", hits, collection, query, None, None, n, max_distance=distance
    )
    weights = getattr(searcher, "_resolve_hybrid_rank_weights", None)
    searcher._hybrid_rank(
        hits, query, *(weights() if weights else ()), metric=metric, stop_words=stop_words
    )
    return proposed, searcher._dedupe_rendered_hits(hits)


def _searched(palace: Path, query: str, n: int, threshold: float | None) -> list:
    """The ids the real ``search_memories`` returns for the same question, in order."""
    extra = {} if threshold is None else {"max_distance": threshold}
    result = _import("searcher").search_memories(
        query, str(palace), n_results=n, candidate_strategy="union", **extra
    )
    hits = result.get("results") if isinstance(result, dict) else None
    return [hit.get("drawer_id") for hit in hits or [] if isinstance(hit, dict)]


def measure(
    provider: MemPalaceMemoryProvider,
    collection,
    texts: dict[str, str],
    total: int,
    probe: Probe,
    arms: dict[str, bool],
    pool: list[dict],
    *,
    self_query: Callable[[object, str], str],
) -> Facts:
    """Every fact about why ``probe``'s drawer is not in arm 1's ``pool``. Never raises."""
    facts = Facts(probe, arms)
    try:
        _measure(provider, collection, texts, total, facts, pool, self_query)
    except Exception as error:  # noqa: BLE001 - by class alone: upstream text can quote the query
        facts.error = type(error).__name__
    return facts


def _measure(provider, collection, texts, total, facts, pool, self_query) -> None:
    probe, drawer = facts.probe, facts.probe.drawer_id
    query = probe.query
    threshold, _ = provider.distance_threshold(collection)

    # The copies already in the pool, by the census's own text (a hit's text can be rewritten).
    body = probe.text.strip()
    pooled = [hit.get("drawer_id") for hit in pool if hit.get("drawer_id") != drawer]
    facts.identical = sum(texts.get(other, "").strip() == body for other in pooled)
    near = [(other, overlap(probe.text, texts[other])) for other in pooled if other in texts]
    if near:
        facts.nearest = max(near, key=lambda pair: (pair[1], pair[0]))

    # Today's fetch, exactly as `search` makes it: its width and the sentinels it dropped.
    _, width, sentinels = provider._ranking(
        query, ask=TODAY_ASK, need=TODAY_ASK, surface=SURFACE, max_distance=threshold
    )
    facts.fetch, facts.sentinels = width, sentinels
    facts.capped = width >= TODAY_ASK * _MAX_FETCH_FACTOR

    # The vector half, the merge and the hybrid rank, for the fetch the pool came from.
    proposed, ranked = reconstruct(collection, query, width, threshold)
    kept_n = _import("searcher")._candidate_pool_limits("union", width)[1]
    facts.vector_proposed, facts.vector_kept = len(proposed), kept_n
    facts.vector_rank = _rank([hit["drawer_id"] for hit in proposed], drawer)
    facts.vector_self = self_query(collection, drawer)
    facts.closets = _closets(provider.palace_path)
    order = [hit.get("drawer_id") for hit in ranked]
    facts.candidates = len(order)
    facts.raw_rank = _rank(order, drawer)
    if facts.raw_rank is not None:
        above = ranked[: facts.raw_rank - 1]
        facts.sentinels_above = sum(hit.get("room") == _REGISTRY_ROOM for hit in above)
        facts.real_rank = facts.raw_rank - facts.sentinels_above
    searched = _searched(provider.palace_path, query, width, threshold)
    facts.shared = len(set(searched) & set(order[:width]))
    facts.matches = searched == order[: len(searched)] and len(searched) == min(width, len(order))

    # The lexical half today, its scan window, and the same search with no window.
    facts.lexical_asked = _LEXICAL_FACTOR * width
    lexical = list(
        collection.lexical_search(query=query, n_results=facts.lexical_asked, where=None).hits
    )
    lexical_ids = [hit.id for hit in lexical]
    facts.lexical_rank = _rank(lexical_ids, drawer)
    backend = _lexical_backend(collection)
    if backend is not None:
        facts.window = scan_window(backend, facts.lexical_asked)
    if facts.window is not None:
        inside = backend._lexical_search_via_sqlite(query=query, n_results=facts.window, where=None)
        everything = backend._lexical_search_via_sqlite(
            query=query, n_results=total, where=None, max_candidates=total
        )
        if inside is not None and everything is not None:
            facts.window_rank = _rank([hit.id for hit in inside], drawer)
            facts.scored = len(everything)
            facts.full_rank = _rank([hit.id for hit in everything], drawer)
        else:
            facts.window = None
    if facts.lexical_rank is not None and facts.raw_rank is None:
        # Proposed, and the merge refused it. Replayed over the whole lexical list in MemPalace's
        # order, because a key is refused for any drawer already admitted under it: a kept vector
        # hit, or a lexical hit ranked above it (`_merge_bm25_union_candidates`).
        scored = None
        if threshold is not None:
            searcher = _import("searcher")
            scored = searcher._lexical_hit_vector_distances(
                collection, query, lexical, searcher._metric_for_collection(collection)
            )
        holders = {_dedup_key(kept): kept["drawer_id"] for kept in proposed[:kept_n]}
        for hit in lexical:
            if scored is not None and (hit.id not in scored or scored[hit.id] > threshold):
                if hit.id == drawer:
                    facts.dropped = hit.id not in scored
                    break
                continue
            key = _lexical_key(hit)
            if not key or key == "?":
                if hit.id == drawer:
                    facts.sourceless = True
                    break
                continue
            if key in holders:
                if hit.id == drawer:
                    facts.shadowed_by = holders[key]
                    break
                continue
            holders[key] = hit.id

    # The first wider ask whose pool holds it, fetched exactly as `search` fetches.
    for ask in LADDER:
        hits, _, _ = provider._ranking(
            query, ask=ask, need=ask, surface=SURFACE, max_distance=threshold
        )
        if drawer in [hit.get("drawer_id") for hit in hits[:ask]]:
            facts.first_ask = ask
            break


# --- the verdict -----------------------------------------------------------------


def verdict(facts: Facts) -> str:
    """The one word for why the drawer is not in the pool; see the module docstring."""
    if facts.error is not None:
        return "unknown"
    if facts.identical:
        return "twin"
    if facts.nearest is not None and facts.nearest[1] >= NEAR_TWIN:
        return "near-twin"
    if not facts.matches:
        return "unknown"
    if facts.real_rank is not None:
        # A candidate: past the pool on merit, or within it but for sentinels ranked above it, which
        # MemPalace counted against the fetch and the harness then dropped.
        if facts.raw_rank > facts.fetch and facts.real_rank <= TODAY_ASK:
            return "sentinels"
        return "crowded"
    if facts.dropped:
        return "dropped"
    if facts.shadowed_by is not None or facts.sourceless:
        return "shadowed"
    # A rank the vector half measured for this very query outweighs its own-embedding probe, which
    # is approximate (HNSW) and reads only a fixed depth.
    vector_lost = facts.vector_self.startswith("no") and facts.vector_rank is None
    lexical_lost = facts.window is not None and facts.full_rank is None
    if vector_lost or (lexical_lost and facts.lexical_rank is None):
        return "missing"
    lexical_reach = facts.full_rank is not None and facts.full_rank <= facts.lexical_asked
    if lexical_reach and facts.window_rank is None:
        return "fts-window"
    if facts.vector_rank is not None and facts.vector_rank > facts.vector_kept:
        return "cut"
    return "unreached"


def _past(rank: int, cut: int) -> str:
    return f"{rank - cut} past the cut" if rank > cut else "within the cut"


def render(facts: Facts) -> str:
    """The ``why:`` line: ids, counts and ranks only."""
    word = verdict(facts)
    if facts.error is not None:
        return f"  why: unknown; a measurement failed ({facts.error})"
    parts = [word]
    if facts.vector_rank is None:
        vector = f"not among the {facts.vector_proposed} it proposed"
    else:
        vector = (
            f"rank {facts.vector_rank} of the {facts.vector_proposed} it proposed, keeps "
            f"{facts.vector_kept} ({_past(facts.vector_rank, facts.vector_kept)})"
        )
    closets = (
        ", closets unreadable"
        if facts.closets is None
        else f", closets {facts.closets} (boost not modelled)"
        if facts.closets
        else ""
    )
    parts.append(f"vector half: {vector}; self-query {facts.vector_self}{closets}")
    lexical = (
        f"rank {facts.lexical_rank} of the {facts.lexical_asked} it asked for"
        if facts.lexical_rank is not None
        else f"not among the {facts.lexical_asked} it asked for"
    )
    if facts.window is None:
        lexical += "; scan window not measurable on this backend"
    else:
        inside = (
            f"inside the {facts.window}-row scan window (rank {facts.window_rank})"
            if facts.window_rank is not None
            else f"outside the {facts.window}-row scan window"
        )
        full = (
            f"rank {facts.full_rank} of {facts.scored} scored with no window"
            if facts.full_rank is not None
            else f"no match for its own text among {facts.scored} scored with no window"
        )
        lexical += f"; {inside}; {full}"
    if facts.dropped:
        lexical += "; refused under max_distance, its embedding not loadable"
    if facts.shadowed_by is not None:
        lexical += f"; refused, drawer {facts.shadowed_by} already holds its file and chunk"
    if facts.sourceless:
        lexical += "; refused, it has no source file, which the merge never admits"
    parts.append(f"lexical half: {lexical}")
    if facts.real_rank is None:
        merge = f"not among the {facts.candidates} candidates"
    else:
        merge = (
            f"hybrid rank {facts.real_rank} of the real candidates "
            f"({facts.candidates} with sentinels; {_past(facts.real_rank, TODAY_ASK)})"
        )
        if facts.sentinels_above:
            merge += f", {facts.sentinels_above} sentinels above it"
    parts.append(f"merge: {merge}")
    cap = TODAY_ASK * _MAX_FETCH_FACTOR
    parts.append(
        f"sentinels dropped {facts.sentinels}, fetch {facts.fetch} (cap {cap}"
        + (", reached" if facts.capped else "")
        + ")"
    )
    parts.append(
        f"wider ask: first in the pool at ask {facts.first_ask}"
        if facts.first_ask is not None
        else f"wider ask: not in the pool at any ask up to {LADDER[-1]}"
    )
    nearest = (
        f"nearest drawer {facts.nearest[0]} shares {facts.nearest[1]:.2f} of its tokens"
        if facts.nearest is not None
        else "no pool drawer to compare"
    )
    parts.append(
        f"in the pool: {facts.identical} identical text, {nearest} (near-twin at {NEAR_TWIN:.2f})"
    )
    parts.append(
        f"reconstruction matches the search on its first {facts.fetch}"
        if facts.matches
        else f"reconstruction differs from the search ({facts.shared} of {facts.fetch} shared)"
    )
    return "  why: " + "; ".join(parts)


# --- the mode ------------------------------------------------------------------


def _digest(ids) -> str:
    return _palace_recall._digest(ids)


def main(
    provider: MemPalaceMemoryProvider,
    collection,
    found,
    *,
    heads: int,
    rares: int,
    before,
    sample,
    query_of,
    self_query,
) -> int:
    """The ``--pool-diagnosis`` mode, after the palace check has opened the palace."""
    refused = _palace_recall.refusal(provider, collection)
    if refused is not None:
        print(f"REFUSED: {refused}.")
        return 1
    chosen = select(found, heads, rares, before, sample=sample, query_of=query_of)
    probes = _palace_recall.schedule(chosen.head, chosen.rare)
    if not probes:
        print("FAIL: no probes to run")
        return 1
    print(
        f"pool diagnosis: mempalace {_mempalace.mempalace_version()}, metric "
        f"{palace_metric(collection)}; arms "
        + "; ".join(
            f"{arm.name}: ask {arm.ask}, "
            + (
                "no max_distance"
                if arm.max_distance is None
                else f"max_distance {arm.max_distance}"
            )
            for arm in ARMS
        )
        + f"; pool = the first {TODAY_ASK} real drawers; token-free, read-only"
    )
    print(
        f"pool diagnosis probes: head {len(chosen.head)} of {heads} asked, rare-token "
        f"{len(chosen.rare)} of {rares} asked; population {chosen.population} drawers"
        + (f" ({chosen.undated} undated left out)" if before is not None else "")
    )
    texts = {drawer_id: text for drawer_id, _, text in found.real}
    missing: dict[tuple[str, str], set[str]] = {
        (kind, arm.name): set() for kind in KINDS for arm in ARMS
    }
    done: dict[str, list[str]] = {kind: [] for kind in KINDS}
    verdicts: dict[str, Counter[str]] = {kind: Counter() for kind in KINDS}
    diagnosed: dict[str, set[str]] = {kind: set() for kind in KINDS}
    for index, probe in enumerate(probes, 1):
        # Each arm's pool is what it hands the reranker (`fetch`), as `--end-to-end` judges it.
        pools = {arm.name: fetch(provider, probe, arm)[0] for arm in ARMS}
        arms = {
            name: probe.drawer_id in [hit.get("drawer_id") for hit in pool]
            for name, pool in pools.items()
        }
        done[probe.kind].append(probe.drawer_id)
        for name, held in arms.items():
            if not held:
                missing[(probe.kind, name)].add(probe.drawer_id)
        if not arms[ARM_1.name]:
            facts = measure(
                provider,
                collection,
                texts,
                found.total,
                probe,
                arms,
                pools[ARM_1.name],
                self_query=self_query,
            )
            word = verdict(facts)
            verdicts[probe.kind][word] += 1
            diagnosed[probe.kind].add(probe.drawer_id)
            print(
                f"unpooled: drawer {probe.drawer_id} ({probe.kind} probe); "
                + ", ".join(f"arm {name} {'yes' if held else 'no'}" for name, held in arms.items())
            )
            print(render(facts))
        if index % _PROGRESS_EVERY == 0 or index == len(probes):
            print(f"pool diagnosis progress: {index} of {len(probes)} probes", file=sys.stderr)
    for kind in KINDS:
        if not done[kind]:
            continue
        nowhere = set.intersection(*(missing[(kind, arm.name)] for arm in ARMS))
        print(
            f"pool diagnosis {kind}: probes {len(done[kind])} (digest {_digest(done[kind])}); "
            + "; ".join(
                f"not in arm {arm.name}'s pool {len(missing[(kind, arm.name)])} "
                f"(digest {_digest(missing[(kind, arm.name)])})"
                for arm in ARMS
            )
            + f"; in no arm's pool {len(nowhere)} (digest {_digest(nowhere)})"
        )
        print(
            f"pool diagnosis {kind} by verdict: "
            + ", ".join(f"{word} {verdicts[kind][word]}" for word in VERDICTS)
        )
    failing = set().union(*diagnosed.values())
    nowhere = set().union(
        *(set.intersection(*(missing[(kind, arm.name)] for arm in ARMS)) for kind in KINDS)
    )
    # Last, because it is the line two runs are compared on.
    print(
        f"pool diagnosis summary: head probes {len(done[HEAD])} (digest {_digest(done[HEAD])}), "
        f"rare-token probes {len(done[_palace_recall.RARE])} "
        f"(digest {_digest(done[_palace_recall.RARE])}); not in arm 1's pool {len(failing)} "
        f"(digest {_digest(failing)}); in no arm's pool {len(nowhere)} (digest {_digest(nowhere)})"
    )
    return 1 if failing else 0
