"""`basecradle-harness-palace-check --pool-diagnosis`: why a probe is not in the pool (issue #633).

The measurements are proven against real MemPalace palaces in ``test_palace_pool_real.py`` (marked
``mempalace``, out of the default run), because a fake that defined MemPalace's ranking would
verify nothing but itself. What is tested here, offline, is the harness's own logic: which verdict a
set of facts earns (the precedence is the design), what a ``why:`` line says and never says, and
how the palace check wires the mode in.
"""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from basecradle_harness import _palace_check, _palace_pool, _palace_recall
from basecradle_harness._palace_pool import NEAR_TWIN, VERDICTS, Facts, overlap, render, verdict
from basecradle_harness._palace_recall import HEAD, Probe

SECRET = "Nova confided vaultcode zq7781x beside quasar lighthouse"
PROBE = Probe(HEAD, "drawer-probe", SECRET, SECRET[:400])


def _facts(**changes) -> Facts:
    """A probe nobody proposed: past both halves' 60, with no window. Every fact is ordinary."""
    base = Facts(
        PROBE,
        {"1": False, "2": False, "3": False},
        fetch=20,
        vector_proposed=60,
        vector_kept=20,
        vector_rank=None,
        vector_self="yes",
        closets=0,
        lexical_asked=60,
        lexical_rank=None,
        window=500,
        window_rank=80,
        scored=900,
        full_rank=80,
        candidates=70,
        shared=20,
        matches=True,
        nearest=("drawer-other", 0.2),
    )
    return dataclasses.replace(base, **changes)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, "unreached"),
        ({"vector_rank": 31}, "cut"),
        ({"window_rank": None, "full_rank": 46}, "fts-window"),
        # Inside the window and past 60 with none: the window did not keep it out.
        ({"window_rank": 61, "full_rank": 55, "vector_rank": 31}, "cut"),
        ({"vector_self": "no stored embedding"}, "missing"),
        ({"vector_self": "no (not in its own top 100)"}, "missing"),
        ({"full_rank": None, "window_rank": None, "vector_rank": 26}, "missing"),
        ({"lexical_rank": 12, "dropped": True}, "dropped"),
        ({"lexical_rank": 21, "shadowed_by": "drawer-copy"}, "shadowed"),
        ({"lexical_rank": 21, "sourceless": True}, "shadowed"),
        # The vector half returned it for this very query: its own-embedding probe is approximate.
        ({"vector_self": "no (not in its own top 100)", "vector_rank": 31}, "cut"),
        # The lexical half proposed it today: its row is there, whatever a wider read says.
        ({"full_rank": None, "lexical_rank": 30, "vector_rank": 31}, "cut"),
        ({"raw_rank": 26, "real_rank": 26}, "crowded"),
        ({"fetch": 160, "capped": True, "raw_rank": 201, "real_rank": 1}, "sentinels"),
        # Behind sentinels, and past the pool even without them: outranked on merit.
        ({"fetch": 160, "capped": True, "raw_rank": 230, "real_rank": 30}, "crowded"),
        ({"nearest": ("drawer-near", NEAR_TWIN), "raw_rank": 26, "real_rank": 26}, "near-twin"),
        ({"identical": 3, "nearest": ("drawer-copy", 1.0)}, "twin"),
        ({"matches": False, "raw_rank": 26, "real_rank": 26}, "unknown"),
        ({"error": "TimeoutError"}, "unknown"),
    ],
)
def test_each_set_of_facts_earns_its_verdict(changes, expected):
    assert verdict(_facts(**changes)) == expected


def test_a_copy_outranks_every_mechanism_and_an_unverified_merge_trusts_none():
    # A twin's memory reaches the reranker under another id, whatever kept this id out...
    assert verdict(_facts(identical=1, matches=False, error=None)) == "twin"
    # ...but below the copies, no mechanism is named off a reconstruction the search disagrees with.
    assert verdict(_facts(matches=False, lexical_rank=21, shadowed_by="drawer-copy")) == "unknown"


def test_a_lost_lexical_row_is_missing_only_where_the_window_was_measured():
    assert verdict(_facts(window=None, full_rank=None, window_rank=None)) == "unreached"


def test_near_is_a_number_the_line_prints():
    line = render(_facts(nearest=("drawer-near", 0.7934)))
    assert "nearest drawer drawer-near shares 0.79 of its tokens (near-twin at 0.80)" in line


def test_overlap_reads_the_tokens_bm25_reads():
    assert overlap("Alpha beta, gamma!", "gamma BETA alpha") == 1.0
    assert overlap("alpha beta", "alpha delta") == pytest.approx(1 / 3)
    assert overlap("a b", "c") == 0.0


@pytest.mark.parametrize("word", VERDICTS)
def test_a_why_line_holds_ids_counts_and_ranks_and_never_the_memory(word):
    facts = {
        "twin": _facts(identical=2, nearest=("drawer-copy", 1.0)),
        "near-twin": _facts(nearest=("drawer-near", 0.9)),
        "sentinels": _facts(fetch=160, capped=True, raw_rank=201, real_rank=1, sentinels_above=200),
        "crowded": _facts(raw_rank=26, real_rank=26),
        "dropped": _facts(lexical_rank=12, dropped=True),
        "shadowed": _facts(lexical_rank=21, shadowed_by="drawer-copy"),
        "missing": _facts(vector_self="no stored embedding"),
        "fts-window": _facts(window_rank=None, full_rank=46),
        "cut": _facts(vector_rank=31),
        "unreached": _facts(),
        "unknown": _facts(error="RuntimeError"),
    }[word]
    line = render(facts)
    assert line.startswith(f"  why: {word};")
    for token in SECRET.split():
        assert token not in line.split()
    assert "zq7781x" not in line


def test_every_fact_has_its_place_on_the_line():
    line = render(
        _facts(
            vector_rank=31,
            lexical_rank=None,
            window_rank=None,
            full_rank=46,
            closets=4,
            first_ask=80,
            sentinels=3,
        )
    )
    assert line == (
        "  why: fts-window; vector half: rank 31 of the 60 it proposed, keeps 20 (11 past the "
        "cut); self-query yes, closets 4 (boost not modelled); lexical half: not among the 60 it "
        "asked for; outside the 500-row scan window; rank 46 of 900 scored with no window; merge: "
        "not among the 70 candidates; sentinels dropped 3, fetch 20 (cap 160); wider ask: first "
        "in the pool at ask 80; in the pool: 0 identical text, nearest drawer drawer-other shares "
        "0.20 of its tokens (near-twin at 0.80); reconstruction matches the search on its first 20"
    )


def test_a_failed_measurement_names_its_class_and_nothing_else():
    assert render(_facts(error="ValueError")) == (
        "  why: unknown; a measurement failed (ValueError)"
    )


def test_the_ladder_doubles_from_today_to_thirty_two_times_it():
    assert _palace_pool.LADDER == (40, 80, 160, 320, 640)
    assert _palace_pool.LADDER[0] == _palace_recall.ARM_3.ask


def test_the_scan_window_is_read_off_the_backend_default():
    class Backend:
        def _lexical_search_via_sqlite(self, *, query, n_results, where, max_candidates=500):
            return []

    assert _palace_pool.scan_window(Backend(), 60) == 500
    assert _palace_pool.scan_window(Backend(), 1920) == 1920
    assert _palace_pool._lexical_backend(Backend()) is not None

    class Wrapper:
        _inner = Backend()

    assert _palace_pool._lexical_backend(Wrapper()) is Wrapper._inner
    assert _palace_pool._lexical_backend(object()) is None


def test_a_measurement_that_raises_is_unknown_never_a_crash(monkeypatch):
    class Provider:
        def distance_threshold(self, collection=None):
            raise RuntimeError("the palace said something about zq7781x")

    facts = _palace_pool.measure(
        Provider(), None, {}, 0, PROBE, {"1": False}, [], self_query=lambda c, d: "yes"
    )
    assert facts.error == "RuntimeError"
    assert verdict(facts) == "unknown"


# --- the palace check wires it in -----------------------------------------------


@pytest.fixture
def home(tmp_path):
    harness_home = tmp_path / "harness"
    (harness_home / "mempalace").mkdir(parents=True)
    return harness_home


def test_pool_diagnosis_needs_a_sample_and_runs_alone(home, capsys):
    with pytest.raises(SystemExit):
        _palace_check.main(["--pool-diagnosis", str(home)])
    assert "--pool-diagnosis applies to --sample" in capsys.readouterr().err
    for other in (["--reranked-pool"], ["--end-to-end", "--token-ceiling", "9"]):
        with pytest.raises(SystemExit):
            _palace_check.main(["--sample", "5", "--pool-diagnosis", *other, str(home)])
        assert "--pool-diagnosis runs alone" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _palace_check.main(["--sample", "5", "--rare-token-probes", "3", str(home)])
    assert "--rare-token-probes applies to --end-to-end or --pool-diagnosis" in (
        capsys.readouterr().err
    )
    with pytest.raises(SystemExit):
        _palace_check.main(["--sample", "5", "--pool-diagnosis", "--token-ceiling", "9", str(home)])
    assert "--token-ceiling applies to --end-to-end" in capsys.readouterr().err


def test_pool_diagnosis_runs_with_the_sample_flags_and_no_reranker(home, monkeypatch):
    """The same probe flags as ``--end-to-end``, the reranker switched off, the palace read-only."""
    monkeypatch.setenv("HARNESS_MEMPALACE_RERANK_MODEL", "z-ai/glm-5.3-flash")
    opened = []
    palace = types.ModuleType("mempalace.palace")

    class Collection:
        def get(self, **kwargs):
            return {"ids": []}

    def get_collection(path, **kwargs):
        opened.append(kwargs)
        return Collection()

    palace.get_collection = get_collection
    monkeypatch.setitem(sys.modules, "mempalace", types.ModuleType("mempalace"))
    monkeypatch.setitem(sys.modules, "mempalace.palace", palace)
    seen = {}

    def fake_main(provider, collection, found, **kwargs):
        seen.update(kwargs, reranker=provider.reranker)
        return 7

    monkeypatch.setattr(_palace_pool, "main", fake_main)
    code = _palace_check.main(
        [
            "--sample",
            "150",
            "--rare-token-probes",
            "100",
            "--filed-before",
            "2026-10-02T07:00:00",
            "--pool-diagnosis",
            str(home),
        ]
    )
    assert code == 7
    assert (seen["heads"], seen["rares"]) == (150, 100)
    assert seen["before"].isoformat() == "2026-10-02T07:00:00"
    assert seen["sample"] is _palace_check.sample
    assert seen["query_of"] is _palace_check.query_of
    assert seen["self_query"] is _palace_check._self_query
    assert seen["reranker"] is None
    assert all(kwargs.get("read_only") for kwargs in opened)


def test_pool_diagnosis_refuses_where_the_arms_are_not_todays_search(monkeypatch, capsys):
    monkeypatch.setattr(_palace_recall, "refusal", lambda provider, collection: "a reason")
    code = _palace_pool.main(
        None,
        None,
        None,
        heads=5,
        rares=0,
        before=None,
        sample=None,
        query_of=None,
        self_query=None,
    )
    assert code == 1
    assert capsys.readouterr().out == "REFUSED: a reason.\n"


def test_closets_it_cannot_read_are_unreadable_never_none():
    assert "self-query yes, closets unreadable;" in render(_facts(closets=None))
    assert "closets" not in render(_facts(closets=0))


def test_a_refused_lexical_hit_says_who_holds_its_key_or_that_it_has_none():
    assert "refused, drawer drawer-copy already holds its file and chunk" in render(
        _facts(lexical_rank=21, shadowed_by="drawer-copy")
    )
    assert "refused, it has no source file, which the merge never admits" in render(
        _facts(lexical_rank=21, sourceless=True)
    )
