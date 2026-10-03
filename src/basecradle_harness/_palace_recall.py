"""``basecradle-harness-palace-check --end-to-end`` — is the right drawer in the final 10 (issue #627).

Every recall number before this one answered *did the right drawer reach the pool?* (#611, #617).
None answered the question an agent's memory turns on: **is the right drawer among the 10 the agent
is shown, after its reranker has picked?** This mode answers it, for a sample of real drawers, by
running each probe through the agent's **own** reranker: `MemPalaceReranker.rerank` itself, built
from the agent's own ``HARNESS_MEMPALACE_RERANK_*`` configuration, with the production prompt and
the production validation. The pool it reads is fetched by `MemPalaceMemoryProvider._ranking`, the
call `MemPalaceMemoryProvider.search` makes, so arm 1 is exactly what a wake recalls today
(``tests/test_palace_recall.py`` pins that equivalence). It changes no search behaviour, and it is
read-only on the palace.

**Four arms, the same probes in each:**

====  ==============  ================  ===============  =====
Arm   Ask upstream    ``max_distance``  Reranker reads   Keeps
====  ==============  ================  ===============  =====
1     20              not passed        20               10
1R    arm 1 again
2     20              2.0               20               10
3     40              2.0               40               10
====  ==============  ================  ===============  =====

Arm 1R is arm 1 run a second time, fetch and rerank both. Its disagreement with arm 1 is the noise
floor: a difference between two arms smaller than that is not a result. Each probe runs its arms in
the order 1, 2, 3, 1R, so the repeat is as far from arm 1 as one probe allows, and a run the budget
stops early still holds the same probes in every arm.

On a cosine palace ``max_distance=2.0`` filters nothing by distance: no cosine distance exceeds it.
What it changes, on MemPalace 3.9 and later (MemPalace#1964), is how a lexical-only candidate is
scored: on its real vector distance instead of on BM25 alone (#625). Its stated risk is that a
lexical hit whose stored embedding cannot be loaded is **dropped**, so arms 2 and 3 also count those
drops. The mode refuses to run where the arms would measure something else: before MemPalace 3.9,
where a threshold disables the lexical half of the search altogether, and on a palace whose distance
metric is not cosine (a legacy Chroma ``l2`` palace measures squared distances up to 4, so 2.0 would
also cut vector candidates a wake keeps).

**Two probe kinds, reported separately:**

- **Head probes** query the head of the drawer's own text (the existing probe, `query_of`), chosen
  by the SHA-256 of the drawer id exactly as ``--sample`` chooses them.
- **Rare-token probes** query nothing but the drawer's rarest exact token, by document frequency
  over the palace's real drawers: a uuid fragment, a handle, an error string. That is the case
  upstream's vector cut is about, and a head probe does not exercise it. Tokens are read the way
  MemPalace's BM25 reads them (`_TOKEN`), at least `RARE_MIN_CHARS` long, and a drawer whose rarest
  token is carried by more than `RARE_MAX_DF` drawers is skipped (and counted). Drawn from the same
  SHA-256 order, so the two kinds largely share drawers.

**The budget is enforced in code, never trusted to the operator.** ``--token-ceiling`` is required.
Before any model call the mode estimates the whole run (`call_tokens`, over the palace's mean drawer
length, at the repo's worst-case three characters a token, plus a full `RERANK_OUTPUT_TOKENS` of
output per call) and refuses to start when the estimate exceeds the ceiling. During the run, each
probe's four pools are fetched first (that costs no tokens), the four calls are estimated from the
very requests the reranker will send, and the probe runs only if what has been spent plus that
estimate fits; once the vendor has reported input tokens, the input half of that estimate never
assumes fewer tokens per character than the run has observed. What a call spent is what the vendor
reported, tokens in plus tokens out; a call that reported nothing, or only half, is charged its
estimate (or what it did report, if more), and every attempt retried after a timeout or a dropped
connection, which the vendor may have billed, is charged one estimate more. The one probe in flight
can still run past its estimate (a model that writes more than `RERANK_OUTPUT_TOKENS`, or text that
tokenizes denser than anything seen so far); the next probe then does not start, so the overshoot is
at most that one probe's excess.

**A run that cannot finish says so.** It stops, reports what it has, marked **PARTIAL**, and exits 1
on the ceiling, on a config-class reranker fault, on a search that returns an empty pool (a
MemPalace search that fails returns nothing and logs a WARNING), on an interrupt, and on any error.
A probe counts only once all four of its arms have run, so a stopped run still holds the same
probes in every arm; an error is named by its class alone, because an upstream exception can quote
the query.

**What it prints:** counts, drawer ids and digests. Never memory text, never a query, never a path.

Exit 0 only for a complete run. A refusal, a partial run, or a palace that cannot be measured exits 1.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import metadata

from basecradle_harness._context import WORST_CASE_CHARS_PER_TOKEN
from basecradle_harness._mempalace import (
    DEFAULT_N_RESULTS,
    MemPalaceMemoryProvider,
    _import,
    candidate_pool,
)
from basecradle_harness._rerank import (
    RERANK_OUTPUT_TOKENS,
    MemPalaceReranker,
    RerankReport,
    _messages,
)

#: The ``surface=`` the measurement's rerank calls carry on their ``llm`` lines. Its own value, so a
#: dashboard that counts an agent's ``turn0`` and ``tool`` recalls never counts a measurement run.
SURFACE = "palace-check"

#: How many memories a recall keeps: what Turn-0 injects and `memory_search` returns by default.
KEEP = DEFAULT_N_RESULTS

#: What a reranked search asks MemPalace for today (`pool_size`), and the wider ask of arm 3.
TODAY_ASK = candidate_pool(KEEP, reranked=True)
WIDE_ASK = 2 * TODAY_ASK

#: The threshold arms 2 and 3 pass. Above every distance a normalized embedding can have, so it
#: filters nothing by distance; see the module docstring for what it does change.
THRESHOLD = 2.0

#: The first MemPalace that scores a lexical hit on its distance under a threshold (MemPalace#1964).
#: Before it, any threshold disables the lexical half, and arms 2 and 3 would measure that instead.
MIN_MEMPALACE = (3, 9)


@dataclass(frozen=True)
class Arm:
    name: str
    #: The ``n_results`` asked of MemPalace. The reranker reads all of it.
    ask: int
    max_distance: float | None

    def describe(self) -> str:
        distance = (
            "no max_distance" if self.max_distance is None else f"max_distance {self.max_distance}"
        )
        return f"ask {self.ask}, {distance}, reranker reads {self.ask}, keeps {KEEP}"


ARM_1 = Arm("1", TODAY_ASK, None)
ARM_1R = Arm("1R", TODAY_ASK, None)
ARM_2 = Arm("2", TODAY_ASK, THRESHOLD)
ARM_3 = Arm("3", WIDE_ASK, THRESHOLD)
#: The order a probe runs its arms in: the repeat last, as far from arm 1 as a probe allows.
RUN_ORDER = (ARM_1, ARM_2, ARM_3, ARM_1R)
#: The order the report reads them in.
REPORT_ORDER = (ARM_1, ARM_1R, ARM_2, ARM_3)

HEAD = "head"
RARE = "rare-token"
KINDS = (HEAD, RARE)

#: A token as MemPalace's BM25 reads one: ``searcher._TOKEN_RE``, matched over lowercased text.
_TOKEN = re.compile(r"\w{2,}", re.UNICODE)

#: The shortest token MemPalace's lexical candidate query keeps (Chroma's
#: ``_lexical_search_via_sqlite`` drops anything shorter), so a shorter one cannot reach the pool
#: through the lexical half at all.
RARE_MIN_CHARS = 3

#: How many drawers may carry a rare-token probe's token. One is the clean case. Up to three admits
#: a drawer filed twice (a re-mined copy, #606), whose every token is shared with its copy, while
#: still leaving room for all of them in the final 10. The report counts the probes at each value.
RARE_MAX_DF = 3

#: Tokens a chat template adds around the two rerank messages, for the estimate. Generous: the
#: estimate is a ceiling, and the vendor's own count is what is charged against the budget.
_TEMPLATE_TOKENS = 16

#: How often, in probes, a progress line goes to stderr. A full run is long, and the report is
#: only printed at the end.
_PROGRESS_EVERY = 10


# --- probes --------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    kind: str
    drawer_id: str
    #: The drawer's own text, for the identical-twin check. Never printed.
    text: str
    #: What is searched for. Never printed.
    query: str


def document_frequencies(texts: Sequence[str]) -> Counter[str]:
    """How many of `texts` carry each token, lowercased, of at least `RARE_MIN_CHARS`."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(
            {token for token in _TOKEN.findall(text.lower()) if len(token) >= RARE_MIN_CHARS}
        )
    return counts


def rare_token(text: str, frequencies: Counter[str]) -> tuple[str, int] | None:
    """A drawer's rarest token as it is written there, and how many drawers carry it.

    Rarest first, then longest (an identifier over a word), then alphabetical, so the choice is the
    same on every run. ``None`` when every token is carried by more than `RARE_MAX_DF` drawers.
    """
    best: tuple[tuple[int, int, str], str, int] | None = None
    for token in _TOKEN.findall(text):
        if len(token) < RARE_MIN_CHARS:
            continue
        carriers = frequencies[token.lower()]
        if not 0 < carriers <= RARE_MAX_DF:
            continue
        key = (carriers, -len(token), token.lower())
        if best is None or key < best[0]:
            best = (key, token, carriers)
    return None if best is None else (best[1], best[2])


@dataclass
class Selection:
    head: list[Probe]
    rare: list[Probe]
    #: Real drawers in the population, filed before the cut-off when one was given.
    population: int
    #: Drawers left out because their ``filed_at`` could not be read while a cut-off was given.
    undated: int
    #: Drawers the rare-token draw looked at and skipped: none of their tokens was rare enough.
    skipped: int
    #: How many drawers carry each rare-token probe's token: ``{1: n, 2: n, 3: n}``.
    carriers: Counter[int] = field(default_factory=Counter)


def select(found, heads: int, rares: int, before, *, sample, query_of) -> Selection:
    """The head and rare-token probes, both drawn in the SHA-256 order ``--sample`` uses.

    `sample` and `query_of` are the palace check's own (passed in, so this module never imports the
    one that imports it), so a head probe here is exactly a ``--sample`` probe.
    """
    ordered = sample(found, len(found.real), before)
    rows = ordered.rows
    head = [Probe(HEAD, drawer_id, text, query_of(text)) for drawer_id, _, text in rows[:heads]]
    frequencies = document_frequencies([text for _, _, text in found.real]) if rares else Counter()
    rare: list[Probe] = []
    skipped = 0
    carriers: Counter[int] = Counter()
    for drawer_id, _, text in rows:
        if len(rare) >= rares:
            break
        picked = rare_token(text, frequencies)
        if picked is None:
            skipped += 1
            continue
        token, count = picked
        carriers[count] += 1
        rare.append(Probe(RARE, drawer_id, text, token))
    return Selection(head, rare, ordered.population, ordered.undated, skipped, carriers)


def schedule(head: list[Probe], rare: list[Probe]) -> list[Probe]:
    """Both kinds, interleaved in proportion, so a run the budget stops still holds some of each."""
    keyed = [((i + 0.5) / len(head), 0, probe) for i, probe in enumerate(head)]
    keyed += [((i + 0.5) / len(rare), 1, probe) for i, probe in enumerate(rare)]
    return [probe for _, _, probe in sorted(keyed, key=lambda item: (item[0], item[1]))]


# --- budget --------------------------------------------------------------------


def request_chars(query: str, pool: Sequence[dict]) -> int:
    """The characters of the rerank request for `pool`, built by the reranker's own `_messages`."""
    return sum(len(message["content"]) for message in _messages(query, pool, KEEP))


def call_tokens(chars: int, tokens_per_char: float | None = None) -> int:
    """A ceiling-side estimate of one rerank call's tokens, in and out, from its request size.

    Input at `WORST_CASE_CHARS_PER_TOKEN`, or at `tokens_per_char` (what the vendor has reported so
    far this run) when that is denser, plus a full `RERANK_OUTPUT_TOKENS` of output.
    """
    tokens_in = math.ceil(chars / WORST_CASE_CHARS_PER_TOKEN)
    if tokens_per_char is not None:
        tokens_in = max(tokens_in, math.ceil(chars * tokens_per_char))
    return tokens_in + _TEMPLATE_TOKENS + RERANK_OUTPUT_TOKENS


def estimate(probes: Sequence[Probe], mean_chars: int) -> int:
    """The whole run, before any of it is fetched: every arm of every probe at the mean drawer."""
    total = 0
    for probe in probes:
        for arm in RUN_ORDER:
            pool = [{"text": "x" * mean_chars}] * arm.ask
            total += call_tokens(request_chars(probe.query, pool))
    return total


def spent_by(report: RerankReport | None, estimated: int) -> tuple[int, bool]:
    """What one call is charged against the ceiling, and whether the vendor stated all of it.

    Tokens in plus tokens out, as reported. A call that stated neither is charged its estimate, and
    one that stated only half is charged the larger of what it stated and its estimate. Each attempt
    retried after a timeout or a dropped connection is charged one estimate more, because the vendor
    may have billed an answer that never arrived. No report at all means no call was made.
    """
    if report is None:
        return 0, True
    uncertain = report.uncertain_attempts * estimated
    stated = (report.tokens_in or 0) + (report.tokens_out or 0)
    if report.tokens_in is None or report.tokens_out is None:
        return max(stated, estimated) + uncertain, False
    return stated + uncertain, True


# --- one probe through one arm ---------------------------------------------------


@dataclass
class Result:
    """One probe through one arm."""

    in_pool: bool
    in_final: bool
    #: Its own id is not in the final 10, and a drawer with byte-identical text is.
    by_twin: bool
    report: RerankReport | None
    fetch_seconds: float
    rerank_seconds: float
    #: Lexical hits a threshold dropped because their stored embedding could not be loaded, or
    #: ``None`` (no threshold in this arm, or the count could not be made).
    dropped: set[str] | None
    #: The tokens this call was charged.
    charged: int
    stated: bool


def lexical_drops(collection, query: str, fetch: int) -> set[str] | None:
    """The lexical hits MemPalace drops under a threshold for want of a stored embedding (#625).

    The union merge asks the backend for ``3 × n_results`` lexical candidates and, under a
    threshold, admits only those `searcher._lexical_hit_vector_distances` can score. This asks the
    same backend the same question with MemPalace's own helpers and returns the ids it could not
    score. ``None`` when those helpers are not there (a MemPalace this was not written against) or
    the question raised, so a count that could not be made is never reported as zero.
    """
    searcher = _import("searcher")
    distances_of = getattr(searcher, "_lexical_hit_vector_distances", None)
    metric_of = getattr(searcher, "_metric_for_collection", None)
    if distances_of is None or metric_of is None:
        return None
    try:
        hits = list(collection.lexical_search(query=query, n_results=fetch * 3, where=None).hits)
        scored = distances_of(collection, query, hits, metric_of(collection))
    except Exception:  # noqa: BLE001 - a count that could not be made is reported as unknown
        return None
    return {hit.id for hit in hits if getattr(hit, "id", None) and hit.id not in scored}


def fetch(
    provider: MemPalaceMemoryProvider, probe: Probe, arm: Arm
) -> tuple[list[dict], int, float]:
    """The pool `arm` hands the reranker for `probe`: ``(pool, fetch, seconds)``.

    Through the provider's own `_ranking`, the fetch half of `search`: with ``need`` the whole ask,
    because the reranker reads all of it, exactly as `search` asks with one bound.
    """
    started = time.monotonic()
    hits, width, _ = provider._ranking(
        probe.query, ask=arm.ask, need=arm.ask, surface=SURFACE, max_distance=arm.max_distance
    )
    return hits[: arm.ask], width, time.monotonic() - started


def judge(probe: Probe, pool: list[dict], final: list[dict]) -> tuple[bool, bool, bool]:
    """``(in_pool, in_final, by_twin)`` for `probe`'s own drawer."""
    in_pool = probe.drawer_id in [hit.get("drawer_id") for hit in pool]
    ids = [hit.get("drawer_id") for hit in final]
    in_final = probe.drawer_id in ids
    body = probe.text.strip()
    by_twin = not in_final and any(str(hit.get("text") or "").strip() == body for hit in final)
    return in_pool, in_final, by_twin


# --- the tally -----------------------------------------------------------------


@dataclass
class Tally:
    """One probe kind through one arm, over the whole run."""

    probes: int = 0
    #: Every probe drawer this arm ran, so each miss can be tagged by the stage it happened at.
    probed: set[str] = field(default_factory=set)
    in_pool: set[str] = field(default_factory=set)
    in_final: set[str] = field(default_factory=set)
    by_twin: set[str] = field(default_factory=set)
    #: ``ok``, or ``fallback:<reason>``, per call.
    outcomes: Counter[str] = field(default_factory=Counter)
    fallbacks: set[str] = field(default_factory=set)
    search_seconds: list[float] = field(default_factory=list)
    fetch_seconds: list[float] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    #: Calls whose tokens the vendor did not state; charged their estimate against the ceiling.
    unreported: int = 0
    #: Calls whose dollars the vendor did not state.
    uncosted: int = 0
    dropped: int = 0
    dropped_probes: set[str] = field(default_factory=set)
    #: Probes whose drop count could not be made.
    dropped_unknown: int = 0

    def add(self, probe: Probe, result: Result, arm: Arm) -> None:
        self.probes += 1
        drawer = probe.drawer_id
        self.probed.add(drawer)
        if result.in_pool:
            self.in_pool.add(drawer)
        if result.in_final:
            self.in_final.add(drawer)
        if result.by_twin:
            self.by_twin.add(drawer)
        report = result.report
        if report is None:
            self.outcomes["no call (empty pool)"] += 1
        elif report.outcome == "ok":
            self.outcomes["ok"] += 1
        else:
            self.outcomes[f"fallback:{report.reason}"] += 1
            self.fallbacks.add(drawer)
        self.search_seconds.append(result.fetch_seconds + result.rerank_seconds)
        self.fetch_seconds.append(result.fetch_seconds)
        if report is not None:
            self.tokens_in += report.tokens_in or 0
            self.tokens_out += report.tokens_out or 0
            if not result.stated:
                self.unreported += 1
            if report.cost is None:
                self.uncosted += 1
            else:
                self.cost += report.cost
        if arm.max_distance is not None:
            if result.dropped is None:
                self.dropped_unknown += 1
            else:
                self.dropped += len(result.dropped)
                # Only where the drop cost it the pool: a drawer the vector half still fetched
                # lost nothing by its lexical copy being refused.
                if drawer in result.dropped and not result.in_pool:
                    self.dropped_probes.add(drawer)


@dataclass
class Run:
    tallies: dict[tuple[str, str], Tally]
    #: The probes every arm ran, per kind.
    done: dict[str, list[Probe]]
    #: Tokens charged against the ceiling: what the vendor reported, or a call's estimate where
    #: it reported nothing.
    spent: int = 0
    #: Calls charged at their estimate because the vendor reported no tokens for them.
    estimated_calls: int = 0
    #: Request characters and reported input tokens, over the calls that reported both, so the
    #: estimate never assumes fewer tokens per character than the vendor has charged.
    observed_chars: int = 0
    observed_tokens_in: int = 0

    @property
    def tokens_per_char(self) -> float | None:
        return self.observed_tokens_in / self.observed_chars if self.observed_chars else None

    #: ``None`` for a complete run; otherwise why it stopped.
    stopped: str | None = None


def measure(
    provider: MemPalaceMemoryProvider,
    reranker: MemPalaceReranker,
    collection,
    probes: Sequence[Probe],
    ceiling: int,
    *,
    progress: Callable[[str], None] | None = None,
) -> Run:
    """Run every probe through every arm, inside `ceiling`; see the module docstring."""
    run = Run(
        {(kind, arm.name): Tally() for kind in KINDS for arm in RUN_ORDER}, {k: [] for k in KINDS}
    )
    for index, probe in enumerate(probes, 1):
        try:
            results, stop = _one_probe(provider, reranker, collection, probe, ceiling, run)
        except KeyboardInterrupt:
            run.stopped = "an interrupt, during a probe that is not counted"
            break
        except Exception as error:  # noqa: BLE001 - report what the run has; never lose it
            # By class alone: an upstream exception's text can quote the query.
            run.stopped = f"an error ({type(error).__name__}), during a probe that is not counted"
            break
        if results is None:
            run.stopped = stop
            break
        # Counted only now, with all four arms run, so every arm holds the same probes.
        for arm, result in results:
            run.tallies[(probe.kind, arm.name)].add(probe, result, arm)
        run.done[probe.kind].append(probe)
        if progress is not None and (index % _PROGRESS_EVERY == 0 or index == len(probes)):
            progress(
                f"end-to-end progress: {index} of {len(probes)} probes, {run.spent} tokens charged"
            )
        if stop is not None:
            run.stopped = stop
            break
    return run


def _one_probe(
    provider: MemPalaceMemoryProvider,
    reranker: MemPalaceReranker,
    collection,
    probe: Probe,
    ceiling: int,
    run: Run,
) -> tuple[list[tuple[Arm, Result]] | None, str | None]:
    """One probe through all four arms: ``(results, why the run stops after it)``.

    ``results`` is ``None`` when the probe never reached a model (the ceiling, an empty pool), so it
    is not counted. Tokens are charged to `run` as each call returns, so a probe an error cuts short
    still costs what it spent.
    """
    pools = {arm.name: fetch(provider, probe, arm) for arm in RUN_ORDER}
    empty = [arm.name for arm in RUN_ORDER if not pools[arm.name][0]]
    if empty:
        return None, (
            f"an empty pool: arm {', '.join(empty)} fetched nothing, which a palace holding "
            "drawers answers only when the search failed (see the WARNING above)"
        )
    chars = {name: request_chars(probe.query, pool) for name, (pool, _, _) in pools.items()}
    estimates = {name: call_tokens(n, run.tokens_per_char) for name, n in chars.items()}
    if run.spent + sum(estimates.values()) > ceiling:
        return None, (
            f"the token ceiling: this probe's four calls were estimated at "
            f"{sum(estimates.values())} with {run.spent} of {ceiling} charged"
        )
    results: list[tuple[Arm, Result]] = []
    config_fault = None
    for arm in RUN_ORDER:
        pool, width, fetch_seconds = pools[arm.name]
        started = time.monotonic()
        try:
            final = reranker.rerank(probe.query, pool, KEEP, surface=SURFACE)
        except BaseException:
            # An interrupt mid-request (`rerank` absorbs every `Exception` itself): the vendor may
            # still bill the call, so it is charged its estimate before the run stops.
            run.spent += estimates[arm.name]
            run.estimated_calls += 1
            raise
        rerank_seconds = time.monotonic() - started
        report = reranker.last_report
        charged, stated = spent_by(report, estimates[arm.name])
        run.spent += charged
        run.estimated_calls += not stated
        if report is not None and report.tokens_in is not None:
            run.observed_chars += chars[arm.name]
            run.observed_tokens_in += report.tokens_in
        dropped = (
            lexical_drops(collection, probe.query, width) if arm.max_distance is not None else None
        )
        in_pool, in_final, by_twin = judge(probe, pool, final)
        results.append(
            (
                arm,
                Result(
                    in_pool,
                    in_final,
                    by_twin,
                    report,
                    fetch_seconds,
                    rerank_seconds,
                    dropped,
                    charged,
                    stated,
                ),
            )
        )
        if report is not None and str(report.reason or "").startswith("config:"):
            config_fault = report.reason
    # Dead until a human acts: every further call would fall back, and measure nothing.
    return results, None if config_fault is None else (
        f"a config-class reranker fault ({config_fault})"
    )


# --- the report ----------------------------------------------------------------


def _digest(ids) -> str:
    ids = sorted(set(ids))
    if not ids:
        return "none"
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]


def _median(seconds: list[float]) -> str:
    return f"{statistics.median(seconds):.3f}s" if seconds else "n/a"


def _outcomes(counter: Counter[str]) -> str:
    ok = counter.get("ok", 0)
    fallbacks = {
        key.split(":", 1)[1]: n for key, n in counter.items() if key.startswith("fallback:")
    }
    empty = counter.get("no call (empty pool)", 0)
    text = f"rerank ok {ok}, fallback {sum(fallbacks.values())}"
    if fallbacks:
        text += " (" + ", ".join(f"{reason} {n}" for reason, n in sorted(fallbacks.items())) + ")"
    if empty:
        text += f", no call (empty pool) {empty}"
    return text


def report(run: Run, ceiling: int, estimated: int, header: str) -> None:
    """Print the per-kind, per-arm report, the pairwise comparisons, and the summary line."""
    for kind in KINDS:
        if not run.done[kind]:
            continue
        for arm in REPORT_ORDER:
            tally = run.tallies[(kind, arm.name)]
            line = (
                f"end-to-end {kind} arm {arm.name} ({arm.describe()}): probes {tally.probes}, "
                f"in pool {len(tally.in_pool)}, in final {KEEP} {len(tally.in_final)}, "
                f"{_outcomes(tally.outcomes)}, "
                f"median search {_median(tally.search_seconds)} "
                f"(fetch {_median(tally.fetch_seconds)}), "
                f"tokens in {tally.tokens_in} out {tally.tokens_out}"
                + (f" (unreported calls {tally.unreported})" if tally.unreported else "")
                + f", cost ${tally.cost:.6f}"
                + (f" (uncosted calls {tally.uncosted})" if tally.uncosted else "")
            )
            if arm.max_distance is not None:
                line += (
                    f", lexical hits dropped for a missing embedding {tally.dropped} "
                    f"(probe drawers it kept out of the pool {len(tally.dropped_probes)}"
                    + (
                        f", not countable on {tally.dropped_unknown} probes"
                        if tally.dropped_unknown
                        else ""
                    )
                    + ")"
                )
            print(line)
            # Each miss tagged by the stage it happened at: the search never handed the drawer
            # to the reranker, or the reranker was handed it and did not pick it.
            unfetched = tally.probed - tally.in_pool
            unpicked = tally.in_pool - tally.in_final
            print(
                f"end-to-end {kind} arm {arm.name} misses by stage: not in the pool "
                f"{len(unfetched)} (digest {_digest(unfetched)}), in the pool but not in the final "
                f"{KEEP} {len(unpicked)} (digest {_digest(unpicked)})"
            )
            # The palace check's `twin`: the drawer is not in the final 10, a drawer with
            # byte-identical text is. Its own line, split by whether the drawer itself was in the
            # pool (the reranker read both and took the copy) or not (only the copy was fetched).
            twin_pooled = tally.by_twin & tally.in_pool
            print(
                f"end-to-end {kind} arm {arm.name} twins (not in the final {KEEP}, an identical-text "
                f"drawer is): {len(tally.by_twin)} (digest {_digest(tally.by_twin)}); drawer itself "
                f"in the pool {len(twin_pooled)}, not in the pool {len(tally.by_twin - twin_pooled)}"
            )
        base = run.tallies[(kind, ARM_1.name)]
        for arm in (ARM_1R, ARM_2, ARM_3):
            other = run.tallies[(kind, arm.name)]
            mixed = base.fallbacks | other.fallbacks
            # Cut both ways: by the pool the reranker was handed, and by the final 10 it picked.
            for stage, mine, theirs in (
                ("pool", other.in_pool, base.in_pool),
                (f"final {KEEP}", other.in_final, base.in_final),
            ):
                gained, lost = mine - theirs, theirs - mine
                for label, ids in (("gained", gained), ("lost", lost)):
                    for drawer in sorted(ids):
                        print(
                            f"end-to-end {kind} arm {arm.name} vs 1 in {stage} {label}: "
                            f"drawer {drawer}"
                        )
                print(
                    f"end-to-end {kind} arm {arm.name} vs 1: in {stage} gained {len(gained)} "
                    f"(digest {_digest(gained)}), lost {len(lost)} (digest {_digest(lost)}), "
                    f"probes with a fallback in either arm {len(mixed)}"
                )
    finals = "; ".join(
        f"{kind} "
        + " ".join(
            f"{arm.name}={len(run.tallies[(kind, arm.name)].in_final)}" for arm in REPORT_ORDER
        )
        for kind in KINDS
        if run.done[kind]
    )
    # Beside the finals, because a fallback hands back the hybrid order: an arm whose reranker
    # fell back is measuring the search, and two runs that differ only there must not read alike.
    fallbacks = "; ".join(
        f"{kind} "
        + " ".join(
            f"{arm.name}={len(run.tallies[(kind, arm.name)].fallbacks)}" for arm in REPORT_ORDER
        )
        for kind in KINDS
        if run.done[kind]
    )
    state = "complete" if run.stopped is None else f"PARTIAL, stopped by {run.stopped}"
    # Last, because it is the line two runs are compared on.
    print(
        f"end-to-end summary: {state}; {header}; head probes {len(run.done[HEAD])} "
        f"(digest {_digest(p.drawer_id for p in run.done[HEAD])}), rare-token probes "
        f"{len(run.done[RARE])} (digest {_digest(p.drawer_id for p in run.done[RARE])}); "
        f"in final {KEEP}: {finals or 'none'}; probes with a fallback: {fallbacks or 'none'}; "
        f"tokens charged {run.spent} of ceiling {ceiling} "
        f"(estimated {estimated}; calls charged at their estimate {run.estimated_calls})"
    )


# --- the mode ------------------------------------------------------------------


def mempalace_version() -> str | None:
    """The installed MemPalace distribution's version, or ``None`` when it cannot be read."""
    try:
        return metadata.version("mempalace")
    except metadata.PackageNotFoundError:
        return None


def _at_least(version: str | None, floor: tuple[int, ...]) -> bool:
    if version is None:
        return False
    parts: list[int] = []
    for piece in version.split("."):
        digits = re.match(r"\d+", piece)
        if digits is None:
            break
        parts.append(int(digits.group(0)))
    return tuple(parts[: len(floor)]) >= floor


def main(
    provider: MemPalaceMemoryProvider,
    reranker: MemPalaceReranker,
    collection,
    found,
    *,
    heads: int,
    rares: int,
    before,
    ceiling: int,
    dry_run: bool,
    sample,
    query_of,
) -> int:
    """The ``--end-to-end`` mode, after the palace check has opened the palace. Returns the exit code."""
    version = mempalace_version()
    if not _at_least(version, MIN_MEMPALACE):
        print(
            f"REFUSED: arms 2 and 3 measure how MemPalace "
            f"{'.'.join(map(str, MIN_MEMPALACE))} and later score a lexical hit under a "
            f"max_distance (MemPalace#1964); this box has MemPalace {version or 'unknown'}. "
            f"Nothing spent."
        )
        return 1
    metric_of = getattr(_import("searcher"), "_metric_for_collection", None)
    metric = metric_of(collection) if metric_of is not None else None
    if metric != "cosine":
        print(
            f"REFUSED: max_distance {THRESHOLD} filters nothing only on a cosine palace; this "
            f"palace's distance metric is {metric or 'unreadable'}, where arms 2 and 3 would also "
            f"cut vector candidates a wake keeps. Nothing spent."
        )
        return 1
    chosen = select(found, heads, rares, before, sample=sample, query_of=query_of)
    probes = schedule(chosen.head, chosen.rare)
    texts = [text.strip() for _, _, text in found.real]
    mean_chars = math.ceil(sum(map(len, texts)) / len(texts)) if texts else 0
    estimated = estimate(probes, mean_chars)
    header = f"mempalace {version}, metric {metric}, rerank model {reranker.model}"
    print(
        f"end-to-end: {header}, providers {','.join(reranker.providers)}; arms "
        + "; ".join(f"{arm.name}: {arm.describe()}" for arm in REPORT_ORDER)
    )
    print(
        f"end-to-end probes: head {len(chosen.head)} of {heads} asked, rare-token "
        f"{len(chosen.rare)} of {rares} asked; population {chosen.population} drawers"
        + (f" ({chosen.undated} undated left out)" if before is not None else "")
        + f"; rare-token draw skipped {chosen.skipped} drawers with no token carried by at most "
        f"{RARE_MAX_DF}; rare tokens carried by "
        + (
            ", ".join(f"{n} drawer(s): {chosen.carriers[n]}" for n in sorted(chosen.carriers))
            or "none"
        )
    )
    print(
        f"end-to-end budget: estimated {estimated} rerank-model tokens across all arms (mean "
        f"drawer {mean_chars} characters, {WORST_CASE_CHARS_PER_TOKEN:g} characters a token, "
        f"{RERANK_OUTPUT_TOKENS} output tokens a call); ceiling {ceiling}"
    )
    if not probes:
        print("FAIL: no probes to run")
        return 1
    if estimated > ceiling:
        print("REFUSED: the estimate exceeds the ceiling. Nothing spent.")
        return 1
    if dry_run:
        print("end-to-end dry run: within the ceiling; no model call made")
        return 0
    run = measure(
        provider,
        reranker,
        collection,
        probes,
        ceiling,
        progress=lambda line: print(line, file=sys.stderr, flush=True),
    )
    report(run, ceiling, estimated, header)
    return 0 if run.stopped is None else 1
