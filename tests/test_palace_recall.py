"""`basecradle-harness-palace-check --end-to-end`: the right drawer in the final 10 (issue #627).

Three things about these tests are deliberate.

**The palace has a known answer for every arm and both probe kinds.** Each drawer belongs to a
class that fixes where the hybrid search ranks it with and without a threshold, whether a threshold
drops it, and whether the model passes it over. So every count the report prints is asserted as an
exact number derived from the classes, never as "something came back".

**The reranker is the production one, over the real ``openrouter`` SDK.** It is built from the
``HARNESS_MEMPALACE_RERANK_*`` environment by the palace check itself, and only its transport is
mocked (respx), the way ``test_rerank.py`` tests it. The model on the other end is a deterministic
oracle that reads the request the reranker actually sent: it picks the candidates that contain the
query, and never one marked ``IGNORED``.

**MemPalace is faked at the ``sys.modules`` boundary**, as ``test_palace_check.py`` does, because it
is an optional extra and not installed here. The fake honours ``n_results`` and ``max_distance``,
and answers the two private helpers the drop count asks.
"""

from __future__ import annotations

import json
import re
import sys
import types
from collections import Counter
from dataclasses import dataclass

import httpx
import pytest
import respx

from basecradle_harness import _mempalace, _palace_check, _palace_recall
from basecradle_harness._mempalace import MemPalaceMemoryProvider
from basecradle_harness._palace_recall import (
    ARM_1,
    RARE_MAX_DF,
    Probe,
    document_frequencies,
    rare_token,
)
from basecradle_harness._rerank import (
    RERANK_API_KEY_VAR,
    RERANK_MODEL_VAR,
    RERANK_PROVIDERS_VAR,
    MemPalaceReranker,
    reranker_from_env,
)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
FAKE_KEY = "sk-or-v1-0123456789abcdef0123456789abcdef"
MODEL = "z-ai/glm-5.3-flash"
USAGE = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100, "cost": 0.0001}


@dataclass(frozen=True)
class Spec:
    """Where the hybrid search ranks a drawer (0-based), with and without ``max_distance``."""

    plain: int | None
    threshold: int | None
    #: A threshold drops it as a lexical hit with no loadable embedding.
    dropped: bool = False
    #: The model never picks it.
    ignored: bool = False


#: class -> (how many drawers, where they rank). The expected counts below follow from this alone.
CLASSES = {
    "a": (3, Spec(0, 0)),  # found by every arm
    "b": (2, Spec(25, 25)),  # only an ask of 40 reaches it
    "c": (2, Spec(22, 5)),  # a threshold lifts it into the 20
    "d": (2, Spec(3, None, dropped=True)),  # a threshold drops it
    "e": (1, Spec(None, None)),  # never ranked
    "f": (2, Spec(15, 15, ignored=True)),  # in every pool, never picked
}


def _text(cls: str, n: int) -> str:
    marker = " IGNORED" if CLASSES[cls][1].ignored else ""
    return (
        f"Nova told John about case {cls} item {n:02d}: ticket zq{cls}{n:02d}x was closed{marker}."
    )


#: Two drawers with no rare token at all: every word they hold is held by many drawers. Found by
#: every arm on a head probe; skipped by the rare-token draw.
COMMON = {
    "drawer-s-00": "Nova told John about case ticket.",
    "drawer-s-01": "John told Nova about case ticket.",
}


def _drawers() -> dict[str, tuple[str, Spec]]:
    drawers = {
        f"drawer-{cls}-{n:02d}": (_text(cls, n), spec)
        for cls, (count, spec) in CLASSES.items()
        for n in range(count)
    }
    drawers.update({drawer: (text, Spec(0, 0)) for drawer, text in COMMON.items()})
    return drawers


#: Ranked around the probe drawer so every pool is full. Not in the palace's census, so never a
#: probe themselves.
FILLER = [
    {"drawer_id": f"filler-{n:02d}", "text": f"filler memory {n}", "room": "general"}
    for n in range(60)
]


class FakeCollection:
    #: What the palace declares. MemPalace's own ``_metric_for_collection`` is faked to agree.
    distance_metric = "cosine"

    def __init__(self, drawers, lookup):
        self.rows = [(drawer, text) for drawer, (text, _) in drawers.items()]
        self.lookup = lookup

    def get(self, *, limit=None, offset=0, include=None, ids=None):
        page = self.rows[offset : offset + (limit or len(self.rows))]
        return {
            "ids": [drawer for drawer, _ in page],
            "documents": [text for _, text in page],
            "metadatas": [{"room": "general", "filed_at": "2026-09-01T00:00:00"} for _ in page],
        }

    def lexical_search(self, *, query, n_results, where=None):
        found = self.lookup(query)
        return types.SimpleNamespace(hits=[types.SimpleNamespace(id=found)] if found else [])


@pytest.fixture
def palace(tmp_path, monkeypatch):
    """A HARNESS_HOME whose faked MemPalace serves the classes above; returns its record."""
    home = tmp_path / "harness"
    (home / "mempalace").mkdir(parents=True)
    drawers = _drawers()
    record = types.SimpleNamespace(home=home, drawers=drawers, searches=[])

    def lookup(query):
        for drawer, (text, _) in drawers.items():
            if text.startswith(query) or re.search(rf"\b{re.escape(query)}\b", text):
                return drawer
        return None

    collection = FakeCollection(drawers, lookup)
    palace_module = types.ModuleType("mempalace.palace")
    palace_module.get_collection = lambda path, **kwargs: collection

    searcher = types.ModuleType("mempalace.searcher")

    def search_memories(query, palace_path, **kwargs):
        record.searches.append(kwargs)
        found = lookup(query)
        text, spec = drawers[found]
        position = spec.threshold if kwargs.get("max_distance") else spec.plain
        ranking = list(FILLER)
        if position is not None:
            ranking.insert(position, {"drawer_id": found, "text": text, "room": "general"})
        return {"results": ranking[: kwargs["n_results"]]}

    searcher.search_memories = search_memories
    searcher._metric_for_collection = lambda col: col.distance_metric
    searcher._lexical_hit_vector_distances = lambda col, query, hits, metric: {
        hit.id: 0.5 for hit in hits if not drawers[hit.id][1].dropped
    }
    parent = types.ModuleType("mempalace")
    for name, module in (("palace", palace_module), ("searcher", searcher)):
        setattr(parent, name, module)
        monkeypatch.setitem(sys.modules, f"mempalace.{name}", module)
    monkeypatch.setitem(sys.modules, "mempalace", parent)
    monkeypatch.setattr(_mempalace, "mempalace_version", lambda: "3.9.0")
    monkeypatch.setenv(RERANK_MODEL_VAR, MODEL)
    monkeypatch.setenv(RERANK_API_KEY_VAR, FAKE_KEY)
    monkeypatch.setenv(RERANK_PROVIDERS_VAR, "deepinfra,together")
    return record


def _candidates(user: str) -> tuple[str, list[tuple[int, str]]]:
    query = user.split("QUERY:\n", 1)[1].split("\n\nCANDIDATES:\n", 1)[0]
    listing = user.split("\n\nCANDIDATES:\n", 1)[1]
    found = [
        (int(m.group(1)), m.group(2))
        for m in re.finditer(r"^\[(\d+)\] (.*)$", listing, re.MULTILINE)
    ]
    return query, found


def oracle(*, usage=None, skip=lambda query, call: False, first_only=False):
    """The model: the candidates holding the query first, then the rest in order, never IGNORED.

    `skip(query, call)` makes it pass over the query's own drawer on the `call`-th request for that
    query (0-based), which is how a test makes the repeat arm disagree. `first_only` picks just the
    first candidate holding the query, which is how a test makes it take a twin's copy.
    """
    calls: Counter[str] = Counter()

    def answer(request):
        user = json.loads(request.content)["messages"][1]["content"]
        query, candidates = _candidates(user)
        call = calls[query]
        calls[query] += 1
        usable = [(n, text) for n, text in candidates if "IGNORED" not in text]
        hit = [n for n, text in usable if query in text and not skip(query, call)]
        if first_only:
            hit = hit[:1]
        rest = [n for n, text in usable if n not in hit and query not in text]
        body = {
            "id": "gen-measure0001",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL,
            "system_fingerprint": "fp_test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"picks": (hit + rest)[:10]}),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage or USAGE,
        }
        return httpx.Response(200, json=body)

    return answer


@pytest.fixture
def router():
    with respx.mock(assert_all_called=False) as mock:
        yield mock


def _run(capsys, palace, *argv):
    code = _palace_check.main([str(palace.home), *map(str, argv)])
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


def _line(lines, prefix):
    return next(line for line in lines if line.startswith(prefix))


ALL = len(_drawers())  # every drawer a head probe
RARE_ALL = ALL - len(COMMON)  # every drawer with a rare token


# --- the known answer ----------------------------------------------------------


def test_every_arm_and_both_probe_kinds_report_the_known_answer(palace, router, capsys):
    route = router.post(CHAT_URL).mock(side_effect=oracle())
    code, lines, _ = _run(
        capsys,
        palace,
        "--sample",
        ALL,
        "--rare-token-probes",
        ALL,
        "--end-to-end",
        "--token-ceiling",
        5_000_000,
    )
    assert code == 0
    probes = ALL + RARE_ALL
    assert route.call_count == 4 * probes

    # head: a3 + c2 + f2 + s2 reach arm 1's pool (today's search passes the threshold, so c is in
    # and d dropped); f is never picked.
    expected = {
        ("head", "1"): (14, 9, 7),
        ("head", "1R"): (14, 9, 7),
        ("head", "2"): (14, 9, 7),  # no threshold: d in, c out
        ("head", "3"): (14, 11, 9),  # b reached by the ask of 40
        ("rare-token", "1"): (12, 7, 5),  # the two common drawers have no rare token
        ("rare-token", "1R"): (12, 7, 5),
        ("rare-token", "2"): (12, 7, 5),
        ("rare-token", "3"): (12, 9, 7),
    }
    for (kind, arm), (count, pool, final) in expected.items():
        line = _line(lines, f"end-to-end {kind} arm {arm} (")
        assert f"probes {count}, in pool {pool}, in final 10 {final}, " in line, line
        assert f"rerank ok {count}, fallback 0" in line
        assert f"tokens in {count * 1000} out {count * 100}" in line
        if arm == "2":
            assert "lexical hits dropped" not in line
        else:
            assert (
                "lexical hits dropped for a missing embedding 2 (probe drawers it kept out of the pool 2)"
                in line
            )

    # Every miss tagged by its stage: b (arms 1, 1R, 2), d (arms 1, 1R, 3), c (arm 2) and e were
    # never fetched; f was fetched every time and never picked.
    stages = {"1": (5, 2), "1R": (5, 2), "2": (5, 2), "3": (3, 2)}
    for kind in ("head", "rare-token"):
        for arm, (unfetched, unpicked) in stages.items():
            line = _line(lines, f"end-to-end {kind} arm {arm} misses by stage:")
            assert f"not in the pool {unfetched} (" in line, line
            assert f"in the pool but not in the final 10 {unpicked} (" in line, line
            assert (
                f"end-to-end {kind} arm {arm} twins (not in the final 10, an identical-text "
                "drawer is): 0 (digest none)" in _line(lines, f"end-to-end {kind} arm {arm} twins")
            )

    # Pairwise, cut both ways: the pool and the final 10 move together here, by construction.
    for kind in ("head", "rare-token"):
        for stage in ("pool", "final 10"):
            assert (
                f"end-to-end {kind} arm 1R vs 1: in {stage} gained 0 (digest none), lost 0"
                in _line(lines, f"end-to-end {kind} arm 1R vs 1: in {stage} ")
            )
            two = _line(lines, f"end-to-end {kind} arm 2 vs 1: in {stage} ")
            assert "gained 2" in two and "lost 2" in two
            three = _line(lines, f"end-to-end {kind} arm 3 vs 1: in {stage} ")
            assert "gained 2" in three and "lost 0" in three

    def ids(prefix):
        return {line.rsplit(" ", 1)[1] for line in lines if line.startswith(prefix)}

    assert ids("end-to-end head arm 3 vs 1 in final 10 gained:") == {"drawer-b-00", "drawer-b-01"}
    assert ids("end-to-end head arm 3 vs 1 in pool gained:") == {"drawer-b-00", "drawer-b-01"}
    # The search before #625 loses what the threshold lifts, and finds what it drops.
    assert ids("end-to-end head arm 2 vs 1 in final 10 lost:") == {"drawer-c-00", "drawer-c-01"}
    assert ids("end-to-end head arm 2 vs 1 in final 10 gained:") == {"drawer-d-00", "drawer-d-01"}

    summary = lines[-1]
    assert summary.startswith("end-to-end summary: complete; mempalace 3.9.0, metric cosine,")
    assert "in final 10: head 1=7 1R=7 2=7 3=9; rare-token 1=5 1R=5 2=5 3=7" in summary
    assert "probes with a fallback: head 1=0 1R=0 2=0 3=0; rare-token 1=0 1R=0 2=0 3=0" in summary
    assert f"tokens charged {4 * probes * 1100} of ceiling 5000000" in summary
    assert "calls charged at their estimate 0" in summary


def test_the_rare_token_draw_skips_a_drawer_with_none_and_counts_it(palace, router, capsys):
    router.post(CHAT_URL).mock(side_effect=oracle())
    _, lines, _ = _run(
        capsys,
        palace,
        "--sample",
        1,
        "--rare-token-probes",
        ALL,
        "--end-to-end",
        "--token-ceiling",
        5_000_000,
        "--dry-run",
    )
    probes = _line(lines, "end-to-end probes:")
    assert f"rare-token {RARE_ALL} of {ALL} asked" in probes
    assert f"skipped {len(COMMON)} drawers" in probes
    assert f"rare tokens carried by 1 drawer(s): {RARE_ALL}" in probes


def test_the_repeat_arm_is_a_second_real_run_and_its_disagreement_is_counted(
    palace, router, capsys
):
    """1R is the noise floor only if it is run again: a model that answers the fourth request for
    a query differently must show up as 1R losing what arm 1 found."""
    router.post(CHAT_URL).mock(side_effect=oracle(skip=lambda query, call: call == 3))
    _, lines, _ = _run(
        capsys, palace, "--sample", ALL, "--end-to-end", "--token-ceiling", 5_000_000
    )
    assert "gained 0 (digest none), lost 7" in _line(lines, "end-to-end head arm 1R vs 1: in final")
    # The pool did not move: the disagreement is the model's, and the stage tag says so.
    assert "gained 0 (digest none), lost 0" in _line(lines, "end-to-end head arm 1R vs 1: in pool")
    assert "in final 10 0, " in _line(lines, "end-to-end head arm 1R (")
    assert "in the pool but not in the final 10 9 (" in _line(
        lines, "end-to-end head arm 1R misses by stage:"
    )


def test_twin_outcomes_get_their_own_line_split_by_whether_the_drawer_was_fetched(
    palace, router, capsys, monkeypatch
):
    """The palace check's `twin`: the drawer is not in the final 10, a byte-identical drawer is.
    Two ways to get there, told apart: only the copy was fetched (t), or both were and the
    reranker took the copy (u)."""
    t_text = "Nova told John about the twin case zqtwinx exactly."
    u_text = "John told Nova about the other twin case zqotherx exactly."
    t_hit = {"drawer_id": "drawer-t-01", "text": t_text, "room": "general"}
    u_hits = [
        {"drawer_id": "drawer-u-01", "text": u_text, "room": "general"},
        {"drawer_id": "drawer-u-00", "text": u_text, "room": "general"},
    ]
    for drawer, text in (("drawer-t-00", t_text), ("drawer-t-01", t_text)):
        palace.drawers[drawer] = (text, Spec(None, None))
    for drawer in ("drawer-u-00", "drawer-u-01"):
        palace.drawers[drawer] = (u_text, Spec(None, None))
    searcher = sys.modules["mempalace.searcher"]
    original = searcher.search_memories

    def search_memories(query, palace_path, **kwargs):
        if query in (t_text, "zqtwinx"):
            return {"results": [t_hit, *FILLER][: kwargs["n_results"]]}
        if query in (u_text, "zqotherx"):
            return {"results": [*u_hits, *FILLER][: kwargs["n_results"]]}
        return original(query, palace_path, **kwargs)

    monkeypatch.setattr(searcher, "search_memories", search_memories)
    sys.modules["mempalace.palace"].get_collection("x").rows.extend(
        [
            (d, palace.drawers[d][0])
            for d in ("drawer-t-00", "drawer-t-01", "drawer-u-00", "drawer-u-01")
        ]
    )
    router.post(CHAT_URL).mock(side_effect=oracle(first_only=True))
    _, lines, _ = _run(
        capsys, palace, "--sample", ALL + 4, "--end-to-end", "--token-ceiling", 5_000_000
    )
    # t-01 and u-01 are found by id; t-00 and u-00 only through their copies.
    assert "in final 10 9, " in _line(lines, "end-to-end head arm 1 (")
    assert _line(lines, "end-to-end head arm 1 twins").endswith(
        "drawer itself in the pool 1, not in the pool 1"
    )
    assert "drawer is): 2 (digest " in _line(lines, "end-to-end head arm 1 twins")


# --- the production path, and nothing else -----------------------------------------


def test_the_mode_calls_the_production_fetch_and_the_production_rerank(
    palace, router, capsys, monkeypatch
):
    """No copy of either: every arm fetches through `_ranking` and picks through `rerank`."""
    router.post(CHAT_URL).mock(side_effect=oracle())
    fetched, reranked = [], []
    ranking, rerank = MemPalaceMemoryProvider._ranking, MemPalaceReranker.rerank

    def spy_ranking(self, query, **kwargs):
        fetched.append(
            (kwargs["ask"], kwargs["need"], kwargs.get("max_distance"), kwargs["surface"])
        )
        return ranking(self, query, **kwargs)

    def spy_rerank(self, query, hits, k, *, surface):
        reranked.append((len(hits), k, surface))
        return rerank(self, query, hits, k, surface=surface)

    monkeypatch.setattr(MemPalaceMemoryProvider, "_ranking", spy_ranking)
    monkeypatch.setattr(MemPalaceReranker, "rerank", spy_rerank)
    code, _, _ = _run(capsys, palace, "--sample", 1, "--end-to-end", "--token-ceiling", 5_000_000)
    assert code == 0
    assert fetched == [
        (20, 20, 2.0, "palace-check"),
        (20, 20, None, "palace-check"),
        (40, 40, 2.0, "palace-check"),
        (20, 20, 2.0, "palace-check"),
    ]
    assert reranked == [
        (20, 10, "palace-check"),
        (20, 10, "palace-check"),
        (40, 10, "palace-check"),
        (20, 10, "palace-check"),
    ]
    # Today's arms send exactly what `search` sends (issue #625), and arm 2 the search before it.
    assert [search.get("max_distance") for search in palace.searches] == [2.0, None, 2.0, 2.0]
    assert palace.searches[1] == {"n_results": 20, "candidate_strategy": "union"}


def test_arm_1_recalls_exactly_what_a_reranked_wake_recalls(palace, router):
    """Arm 1 is not a model of today's search, it is today's search: `search` with the same
    reranker bound returns the same drawers in the same order. Class c ranks differently with a
    threshold and without, so this also pins that arm 1 passes the one `search` passes."""
    router.post(CHAT_URL).mock(side_effect=oracle())
    reranker = reranker_from_env()
    palace_path = palace.home / "mempalace"
    for drawer, (text, _) in palace.drawers.items():
        today = MemPalaceMemoryProvider(palace_path, reranker=reranker).search(text)
        pool, _, _ = _palace_recall.fetch(
            MemPalaceMemoryProvider(palace_path, reranker=reranker),
            Probe("head", drawer, text, text),
            ARM_1,
        )
        measured = reranker.rerank(text, pool, 10, surface=_palace_recall.SURFACE)
        assert [hit["drawer_id"] for hit in measured] == [hit["drawer_id"] for hit in today], drawer


# --- the budget -------------------------------------------------------------------


def test_an_estimate_over_the_ceiling_refuses_and_spends_nothing(palace, router, capsys):
    route = router.post(CHAT_URL).mock(side_effect=oracle())
    code, lines, _ = _run(capsys, palace, "--sample", ALL, "--end-to-end", "--token-ceiling", 1000)
    assert code == 1
    assert route.call_count == 0
    assert lines[-1] == "REFUSED: the estimate exceeds the ceiling. Nothing spent."
    assert "ceiling 1000" in _line(lines, "end-to-end budget: estimated ")


def test_a_dry_run_prints_the_estimate_and_calls_no_model(palace, router, capsys):
    route = router.post(CHAT_URL).mock(side_effect=oracle())
    code, lines, _ = _run(
        capsys, palace, "--sample", ALL, "--end-to-end", "--token-ceiling", 5_000_000, "--dry-run"
    )
    assert code == 0
    assert route.call_count == 0
    assert lines[-1] == "end-to-end dry run: within the ceiling; no model call made"


def test_the_estimate_is_built_from_the_requests_the_reranker_sends():
    """Ceiling-side: three characters a token over the real request, plus a full output budget."""
    pool = [{"text": "x" * 297}] * 20
    chars = _palace_recall.request_chars("q", pool)
    assert chars > 20 * 300  # the candidates, numbered, plus the instruction and the query
    assert _palace_recall.call_tokens(chars) == -(-chars // 3) + 16 + 1024


def test_the_run_stops_before_a_probe_that_would_pass_the_ceiling(palace, router, capsys):
    """Charged by what the vendor reported: a probe that spent more than estimated stops the run
    before the next one, and the report says partial and exits nonzero."""
    heavy = {"prompt_tokens": 60_000, "completion_tokens": 100, "total_tokens": 60_100}
    route = router.post(CHAT_URL).mock(side_effect=oracle(usage=heavy))
    code, lines, _ = _run(capsys, palace, "--sample", 3, "--end-to-end", "--token-ceiling", 200_000)
    assert code == 1
    assert route.call_count == 4
    summary = lines[-1]
    assert summary.startswith("end-to-end summary: PARTIAL, stopped by the token ceiling")
    assert "head probes 1 (" in summary
    assert "tokens charged 240400 of ceiling 200000" in summary


def test_a_call_that_reports_no_tokens_is_charged_its_estimate(palace, router, capsys):
    zeros = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    router.post(CHAT_URL).mock(side_effect=oracle(usage=zeros))
    _, lines, _ = _run(capsys, palace, "--sample", 1, "--end-to-end", "--token-ceiling", 5_000_000)
    assert "(unreported calls 1)" in _line(lines, "end-to-end head arm 1 (")
    assert "(uncosted calls 1)" in _line(lines, "end-to-end head arm 1 (")
    summary = lines[-1]
    assert "calls charged at their estimate 4" in summary
    charged = int(re.search(r"tokens charged (\d+)", summary).group(1))
    assert charged > 4 * 1024  # four estimates, each with a full output budget


def test_a_config_class_fault_stops_the_run_after_the_probe(palace, router, capsys, caplog):
    """Dead until a human acts: one probe shows it, and nothing further is spent on fallbacks."""
    route = router.post(CHAT_URL).mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key", "code": 401}})
    )
    code, lines, _ = _run(
        capsys, palace, "--sample", ALL, "--end-to-end", "--token-ceiling", 5_000_000
    )
    assert code == 1
    assert route.call_count == 4
    assert "fallback 1 (config:auth 1)" in _line(lines, "end-to-end head arm 1 (")
    assert "stopped by a config-class reranker fault (config:auth)" in lines[-1]
    assert "probes with a fallback: head 1=1 1R=1 2=1 3=1;" in lines[-1]
    assert any("reason=config:auth" in record.getMessage() for record in caplog.records)


# --- refusals before anything runs --------------------------------------------------


def test_no_rerank_model_configured_refuses(palace, router, capsys, monkeypatch):
    monkeypatch.delenv(RERANK_MODEL_VAR)
    code, _, err = _run(capsys, palace, "--sample", 1, "--end-to-end", "--token-ceiling", 5_000_000)
    assert code == 1
    assert f"{RERANK_MODEL_VAR} is not set" in err


def test_an_incomplete_rerank_configuration_refuses(palace, router, capsys, monkeypatch):
    monkeypatch.delenv(RERANK_API_KEY_VAR)
    code, _, err = _run(capsys, palace, "--sample", 1, "--end-to-end", "--token-ceiling", 5_000_000)
    assert code == 1
    assert "config:missing_api_key" in err


def test_a_mempalace_before_3_9_refuses(palace, router, capsys, monkeypatch):
    """Before 3.9 a threshold disables the lexical half, so `search` passes none and arm 1 would not
    be today's search; arm 3 would measure the disabled half."""
    monkeypatch.setattr(_mempalace, "mempalace_version", lambda: "3.8.0")
    route = router.post(CHAT_URL).mock(side_effect=oracle())
    code, lines, _ = _run(
        capsys, palace, "--sample", 1, "--end-to-end", "--token-ceiling", 5_000_000
    )
    assert code == 1
    assert route.call_count == 0
    assert lines[-1].startswith("REFUSED: arms 1 and 3 measure how MemPalace 3.9 and later")


@pytest.mark.parametrize(
    "argv",
    [
        ["--sample", "5", "--end-to-end"],  # no ceiling
        ["--end-to-end", "--token-ceiling", "100"],  # no sample
        ["--sample", "5", "--end-to-end", "--token-ceiling", "100", "--reranked-pool"],
        ["--sample", "5", "--end-to-end", "--token-ceiling", "100", "--practice-observe"],
        ["--sample", "5", "--token-ceiling", "100"],  # a ceiling with nothing to bound
        ["--sample", "5", "--rare-token-probes", "3"],
    ],
)
def test_the_flags_only_combine_one_way(palace, capsys, argv):
    with pytest.raises(SystemExit) as raised:
        _palace_check.main([str(palace.home), *argv])
    assert raised.value.code == 2


def test_no_memory_text_and_no_query_is_ever_printed(palace, router, capsys):
    router.post(CHAT_URL).mock(side_effect=oracle())
    _, lines, err = _run(
        capsys,
        palace,
        "--sample",
        ALL,
        "--rare-token-probes",
        ALL,
        "--end-to-end",
        "--token-ceiling",
        5_000_000,
    )
    output = "\n".join(lines) + err
    for text, _ in palace.drawers.values():
        assert text not in output
    for token in re.findall(r"zq\w+x", " ".join(text for text, _ in palace.drawers.values())):
        assert token not in output


# --- the rare token ----------------------------------------------------------------


def test_the_rarest_token_wins_then_the_longest_then_the_alphabetical():
    texts = [
        "alpha beta shared uniq",
        "alpha beta shared longest",
        "alpha beta",
        "beta fghij abcde",
    ]
    frequencies = document_frequencies(texts)
    assert frequencies["alpha"] == 3 and frequencies["shared"] == 2
    assert rare_token("alpha beta shared", frequencies) == ("shared", 2)  # rarest
    assert rare_token(texts[1], frequencies) == ("longest", 1)  # rarest, then longest
    assert rare_token(texts[3], frequencies) == ("abcde", 1)  # then alphabetical


def test_a_token_is_read_the_way_mempalace_bm25_reads_it():
    """``\\w{2,}``, lowercased for the count; the case it is written in is what is queried."""
    frequencies = document_frequencies(["Ticket 019A-77F2 closed", "ticket closed"])
    assert frequencies["019a"] == 1 and frequencies["77f2"] == 1
    assert rare_token("Ticket 019A-77F2 closed", frequencies) == ("019A", 1)


def test_a_token_shorter_than_the_lexical_query_keeps_is_never_chosen():
    texts = ["zz common words"] + ["common words"] * RARE_MAX_DF
    frequencies = document_frequencies(texts)
    assert "zz" not in frequencies
    assert rare_token(texts[0], frequencies) is None


def test_a_drawer_whose_every_token_is_widely_held_has_no_rare_token():
    texts = ["common words here"] * (RARE_MAX_DF + 1)
    assert rare_token(texts[0], document_frequencies(texts)) is None


# --- a run that cannot finish says so, and keeps what it has ----------------------


def test_a_palace_that_is_not_cosine_refuses(palace, router, capsys, monkeypatch):
    """On a legacy l2 palace a threshold of 2.0 also cuts vector candidates a wake keeps."""
    monkeypatch.setattr(FakeCollection, "distance_metric", "l2")
    route = router.post(CHAT_URL).mock(side_effect=oracle())
    code, lines, _ = _run(capsys, palace, "--sample", 1, "--end-to-end", "--token-ceiling", 10**7)
    assert code == 1
    assert route.call_count == 0
    assert lines[-1].startswith("REFUSED: max_distance 2.0 filters nothing only on a cosine palace")
    assert "metric is l2" in lines[-1]


def test_a_search_that_returns_nothing_stops_the_run_instead_of_reading_complete(
    palace, router, capsys, monkeypatch
):
    """A failed MemPalace search returns no hits; on a palace full of drawers that is a failure,
    never a probe that simply missed."""
    searcher = sys.modules["mempalace.searcher"]
    original, seen = searcher.search_memories, []

    def failing_after_one(query, palace_path, **kwargs):
        seen.append(query)
        if len(set(seen)) > 1:
            return {"error": "Search error", "results": []}
        return original(query, palace_path, **kwargs)

    monkeypatch.setattr(searcher, "search_memories", failing_after_one)
    route = router.post(CHAT_URL).mock(side_effect=oracle())
    code, lines, _ = _run(capsys, palace, "--sample", 3, "--end-to-end", "--token-ceiling", 10**7)
    assert code == 1
    assert route.call_count == 4  # the first probe only
    assert lines[-1].startswith("end-to-end summary: PARTIAL, stopped by an empty pool: arm 1, 2,")
    assert "head probes 1 (" in lines[-1]


def test_an_error_mid_run_keeps_every_finished_probe_and_never_prints_its_text(
    palace, router, capsys, monkeypatch
):
    """Tokens already spent are not thrown away with a traceback, and the exception's text, which
    can quote the query, is named by class alone."""
    searcher = sys.modules["mempalace.searcher"]
    original, seen = searcher.search_memories, []

    def raising_on_the_second_probe(query, palace_path, **kwargs):
        seen.append(query)
        if len(set(seen)) > 1:
            raise RuntimeError(f"Search error near {query!r}")
        return original(query, palace_path, **kwargs)

    monkeypatch.setattr(searcher, "search_memories", raising_on_the_second_probe)
    router.post(CHAT_URL).mock(side_effect=oracle())
    code, lines, err = _run(capsys, palace, "--sample", 3, "--end-to-end", "--token-ceiling", 10**7)
    assert code == 1
    summary = lines[-1]
    assert (
        "PARTIAL, stopped by an error (RuntimeError), during a probe that is not counted" in summary
    )
    assert "head probes 1 (" in summary
    for arm in ("1", "1R", "2", "3"):
        assert "probes 1, " in _line(lines, f"end-to-end head arm {arm} (")
    assert "tokens charged 4400 " in summary
    output = "\n".join(lines) + err
    assert "Search error" not in output
    for text, _ in palace.drawers.values():
        assert text not in output


def test_an_interrupt_mid_call_keeps_what_finished_and_charges_the_call_in_flight(
    palace, router, capsys
):
    calls = {"n": 0}
    answer = oracle()

    def interrupted(request):
        calls["n"] += 1
        if calls["n"] == 6:  # the second probe's second arm
            raise KeyboardInterrupt
        return answer(request)

    router.post(CHAT_URL).mock(side_effect=interrupted)
    code, lines, _ = _run(capsys, palace, "--sample", 3, "--end-to-end", "--token-ceiling", 10**7)
    assert code == 1
    summary = lines[-1]
    assert "PARTIAL, stopped by an interrupt, during a probe that is not counted" in summary
    assert "head probes 1 (" in summary
    assert "calls charged at their estimate 1)" in summary
    charged = int(re.search(r"tokens charged (\d+)", summary).group(1))
    assert charged > 5 * 1100 + 1024  # five answered calls, and the one in flight at its estimate


# --- what a call is charged ----------------------------------------------------------


def test_an_attempt_retried_after_a_timeout_is_charged_an_estimate_more(
    palace, router, capsys, monkeypatch
):
    """A request that reached the vendor and never answered may have been billed for an answer
    that never arrived, so the ceiling counts it."""
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    calls = {"n": 0}
    answer = oracle()

    def first_times_out(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return answer(request)

    router.post(CHAT_URL).mock(side_effect=first_times_out)
    _, lines, _ = _run(capsys, palace, "--sample", 1, "--end-to-end", "--token-ceiling", 10**7)
    charged = int(re.search(r"tokens charged (\d+)", lines[-1]).group(1))
    assert calls["n"] == 5
    assert "rerank ok 1, fallback 0" in _line(lines, "end-to-end head arm 1 (")
    assert charged > 4 * 1100 + 1024  # four answers, and the timed-out attempt at its estimate


def test_half_reported_usage_is_charged_no_less_than_its_estimate():
    from basecradle_harness._rerank import RerankReport

    half = RerankReport(outcome="ok", reason=None, tokens_in=10)
    assert _palace_recall.spent_by(half, 5000) == (5000, False)
    full = RerankReport(outcome="ok", reason=None, tokens_in=10, tokens_out=5)
    assert _palace_recall.spent_by(full, 5000) == (15, True)
    retried = RerankReport(
        outcome="ok", reason=None, tokens_in=10, tokens_out=5, uncertain_attempts=2
    )
    assert _palace_recall.spent_by(retried, 5000) == (10_015, True)
    assert _palace_recall.spent_by(None, 5000) == (0, True)


def test_the_estimate_learns_a_denser_tokenizer_from_what_the_vendor_reports(
    palace, router, capsys
):
    """Three characters a token is a floor on the estimate, not a promise: once the vendor reports
    far more tokens per character, the gate before each probe uses that rate. Here it stops the run
    a probe earlier than the static rate would have."""
    dense = {"prompt_tokens": 20_000, "completion_tokens": 100, "total_tokens": 20_100}
    route = router.post(CHAT_URL).mock(side_effect=oracle(usage=dense))
    code, lines, _ = _run(capsys, palace, "--sample", 3, "--end-to-end", "--token-ceiling", 200_000)
    assert code == 1
    assert route.call_count == 8
    assert "PARTIAL, stopped by the token ceiling" in lines[-1]
    assert "tokens charged 160800 of ceiling 200000" in lines[-1]
    assert _palace_recall.call_tokens(3000, 10.0) == 30_000 + 16 + 1024
    assert _palace_recall.call_tokens(3000, 0.1) == 1000 + 16 + 1024  # never below the floor


def test_a_lexical_drop_is_charged_to_a_probe_only_when_it_cost_the_pool():
    tally = _palace_recall.Tally()
    probe = Probe("head", "drawer-a-00", "text", "text")
    arm = _palace_recall.ARM_3

    def result(in_pool):
        return _palace_recall.Result(
            in_pool, in_pool, False, None, 0.0, 0.0, {"drawer-a-00", "drawer-x"}, 0, True
        )

    tally.add(probe, result(in_pool=True), arm)
    assert tally.dropped == 2 and not tally.dropped_probes
    tally.add(probe, result(in_pool=False), arm)
    assert tally.dropped == 4 and tally.dropped_probes == {"drawer-a-00"}
