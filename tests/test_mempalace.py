"""The MemPalace reference adapter, against a mocked MemPalace library.

MemPalace is an optional extra and not installed in the test env, so its library is
faked at the ``sys.modules`` boundary (the adapter imports ``mempalace.convo_miner`` /
``mempalace.searcher`` lazily). These pin the contract the adapter relies on: `observe`
mines a quote-formatted exchange file, `context` retrieves and renders top-K hits, the
provider exposes one **read-only** `memory_search` tool over that same retrieval call, and a
genuinely missing package degrades to an actionable "install the extra" error.
"""

import sys
import types

import pytest

from basecradle_harness._memory_provider import MemoryExchange, MemoryScope
from basecradle_harness._mempalace import MemPalaceMemoryProvider, MemPalaceSearchTool

# The keyword arguments MemPalace's `search_memories` actually accepts. The fake rejects
# anything outside this set, so a kwarg the adapter invents (or one upstream renames) fails
# the suite here rather than raising a TypeError against the real library in production.
_SEARCH_KWARGS = {"n_results", "candidate_strategy", "max_distance"}


@pytest.fixture
def fake_mempalace(monkeypatch):
    """Install fake ``mempalace.convo_miner`` / ``mempalace.searcher`` modules.

    Returns the two fakes so a test can assert how the adapter called them. ``mine_convos``
    records its args; ``search_memories`` records the kwargs it was *passed* (not their
    defaults — the `max_distance` guard below turns on that distinction) and returns
    whatever the test stashes on it.
    """
    convo_miner = types.ModuleType("mempalace.convo_miner")
    convo_miner.calls = []

    def mine_convos(convo_dir, palace_path, **kwargs):
        convo_miner.calls.append((convo_dir, palace_path, kwargs))

    convo_miner.mine_convos = mine_convos

    searcher = types.ModuleType("mempalace.searcher")
    searcher.result = {"results": []}
    searcher.queries = []

    def search_memories(query, palace_path, **kwargs):
        unknown = set(kwargs) - _SEARCH_KWARGS
        assert not unknown, f"MemPalace's search_memories takes no {sorted(unknown)} kwarg"
        searcher.queries.append((query, palace_path, kwargs))
        return searcher.result

    searcher.search_memories = search_memories

    parent = types.ModuleType("mempalace")
    parent.convo_miner = convo_miner
    parent.searcher = searcher

    monkeypatch.setitem(sys.modules, "mempalace", parent)
    monkeypatch.setitem(sys.modules, "mempalace.convo_miner", convo_miner)
    monkeypatch.setitem(sys.modules, "mempalace.searcher", searcher)
    return convo_miner, searcher


def _scope(query=None):
    return MemoryScope(agent="agent-uuid", timeline="tl-uuid", query=query)


# --- observe: mine a quote-formatted exchange file ---------------------------


def test_observe_writes_an_exchange_file_and_mines_it(fake_mempalace, tmp_path):
    convo_miner, _ = fake_mempalace
    provider = MemPalaceMemoryProvider(tmp_path / "palace")

    provider.observe(MemoryExchange(user="Where do I live?", assistant="Dallas.", scope=_scope()))

    convo_dir = tmp_path / "palace" / "conversations"
    files = list(convo_dir.glob("*.md"))
    assert len(files) == 1
    body = files[0].read_text()
    assert body == "> Where do I live?\nDallas.\n"  # MemPalace exchange format: `>` turn + reply

    # And the directory was mined into the palace, agent-scoped, exchange mode.
    assert len(convo_miner.calls) == 1
    cdir, palace, kwargs = convo_miner.calls[0]
    assert cdir == str(convo_dir)
    assert palace == str(tmp_path / "palace")
    assert kwargs["extract_mode"] == "exchange"


def test_observe_quotes_every_line_of_a_multiline_message(fake_mempalace, tmp_path):
    provider = MemPalaceMemoryProvider(tmp_path / "palace")
    provider.observe(MemoryExchange(user="line one\nline two", assistant="ok", scope=_scope()))

    body = next((tmp_path / "palace" / "conversations").glob("*.md")).read_text()
    assert body == "> line one\n> line two\nok\n"


def test_observe_skips_a_wholly_empty_exchange(fake_mempalace, tmp_path):
    convo_miner, _ = fake_mempalace
    provider = MemPalaceMemoryProvider(tmp_path / "palace")

    provider.observe(MemoryExchange(user="   ", assistant="", scope=_scope()))

    assert convo_miner.calls == []
    assert not (tmp_path / "palace" / "conversations").exists()


# --- context: retrieve and render top-K --------------------------------------


def test_context_renders_top_k_hits_into_a_block(fake_mempalace, tmp_path):
    _, searcher = fake_mempalace
    searcher.result = {"results": [{"text": "John lives in Dallas."}, {"text": "John uses Rails."}]}
    palace = tmp_path / "palace"
    palace.mkdir()
    provider = MemPalaceMemoryProvider(palace, n_results=3)

    block = provider.context(_scope(query="where does john live"))

    assert "Relevant memories" in block
    assert "- John lives in Dallas." in block
    assert "- John uses Rails." in block
    # The query and bound were passed through to MemPalace — in exactly one search per turn
    # (retrieval is on the wake path; a second search would double the vector + FTS work).
    assert len(searcher.queries) == 1
    query, palace_path, kwargs = searcher.queries[0]
    assert (query, palace_path, kwargs["n_results"]) == ("where does john live", str(palace), 3)


def test_context_widens_the_rerank_pool_with_the_union_candidate_strategy(fake_mempalace, tmp_path):
    """Retrieval is hybrid: lexical (BM25) candidates enter the pool, not vector hits alone.

    MemPalace's default ("vector") seeds the rerank pool from the top vector hits only, so a
    chunk whose embedding sits far from the query is never reranked however strong its exact-
    token match — the miss that matters most for agent memory (handles, UUIDs, error strings).
    """
    _, searcher = fake_mempalace
    palace = tmp_path / "palace"
    palace.mkdir()

    MemPalaceMemoryProvider(palace).context(_scope(query="019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"))

    assert searcher.queries[0][2]["candidate_strategy"] == "union"


def test_context_never_sets_max_distance(fake_mempalace, tmp_path):
    """A distance filter would silently kill the union merge — so the adapter must never set one.

    Upstream's `_merge_bm25_union_candidates` opens with `if max_distance > 0.0: return`:
    BM25-only candidates carry no vector distance, so *any* nonzero threshold drops the
    lexical half of the pool and quietly reduces `candidate_strategy="union"` to a no-op.
    This is the tripwire for a future distance filter added without knowing that.
    """
    _, searcher = fake_mempalace
    palace = tmp_path / "palace"
    palace.mkdir()

    MemPalaceMemoryProvider(palace).context(_scope(query="anything"))

    assert "max_distance" not in searcher.queries[0][2]


def test_context_is_none_when_the_backend_cannot_do_lexical_search(fake_mempalace, tmp_path):
    """Graceful degradation: a backend without `lexical_search` errors, and we simply skip.

    `search_memories` answers a union request it cannot serve with an error dict carrying no
    ``results`` key. Turn-0 composition just omits the memory section — never a crash.
    """
    _, searcher = fake_mempalace
    searcher.result = {"error": "backend does not support lexical_search"}
    palace = tmp_path / "palace"
    palace.mkdir()

    assert MemPalaceMemoryProvider(palace).context(_scope(query="anything")) is None


def test_context_is_none_without_a_query(fake_mempalace, tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()
    provider = MemPalaceMemoryProvider(palace)
    assert provider.context(_scope(query=None)) is None


def test_context_is_none_before_the_palace_exists(fake_mempalace, tmp_path):
    """No palace dir yet (nothing observed) → no search, no context."""
    _, searcher = fake_mempalace
    provider = MemPalaceMemoryProvider(tmp_path / "palace")  # never created
    assert provider.context(_scope(query="anything")) is None
    assert searcher.queries == []  # short-circuited before touching MemPalace


def test_context_is_none_when_there_are_no_hits(fake_mempalace, tmp_path):
    _, searcher = fake_mempalace
    searcher.result = {"results": []}
    palace = tmp_path / "palace"
    palace.mkdir()
    provider = MemPalaceMemoryProvider(palace)
    assert provider.context(_scope(query="nothing matches")) is None


# --- the memory_search tool: deliberate recall (issue #267) -------------------
#
# `context` retrieves once per wake, against the incoming turn's text. What these pin is the
# way *back* to the palace mid-task — a read-only tool, over the same search call, so recall is
# not frozen at Turn 0 and `observe` stays the palace's only writer.


def _search_tool(palace, **kwargs):
    (tool,) = MemPalaceMemoryProvider(palace, **kwargs).tools()
    return tool


def test_provider_supplies_one_read_only_search_tool(tmp_path):
    """One tool, and it is search-only: no write/delete surface for the model to reach for.

    The read-only shape is the whole reason there is no concurrent-writer question — `observe`
    remains the sole writer — so a future write action added here would be a real regression,
    not a feature. That is what this asserts.
    """
    tools = MemPalaceMemoryProvider(tmp_path / "palace").tools()

    assert [type(tool) for tool in tools] == [MemPalaceSearchTool]
    assert tools[0].name == "memory_search"
    # The schema exposes a query and a bound — and nothing that writes.
    assert set(tools[0].parameters["properties"]) == {"query", "n_results"}
    assert tools[0].parameters["required"] == ["query"]
    assert tools[0].requires == frozenset()  # a pure tool: loads under the locked policy


def test_search_tool_recalls_through_the_same_union_search_as_context(fake_mempalace, tmp_path):
    """The tool's retrieval *is* `context`'s: same in-process call, same union pool, so what the
    agent can reach by asking is what the palace would have injected — only with its own query."""
    _, searcher = fake_mempalace
    searcher.result = {"results": [{"text": "The staging endpoint is api.staging.example.com."}]}
    palace = tmp_path / "palace"
    palace.mkdir()

    answer = _search_tool(palace).run(query="that endpoint we discussed in March")

    assert "Memories matching" in answer
    assert "- The staging endpoint is api.staging.example.com." in answer
    query, palace_path, kwargs = searcher.queries[0]
    assert (query, palace_path) == ("that endpoint we discussed in March", str(palace))
    assert kwargs["candidate_strategy"] == "union"  # inherited from #266 — never vector-only
    assert "max_distance" not in kwargs  # which would silently kill the union pool
    assert kwargs["n_results"] == 5  # the provider's default when the model names no count


def test_search_tool_clamps_a_model_chosen_bound(fake_mempalace, tmp_path):
    """The schema's minimum/maximum are advisory to the model, so the bound is *enforced* here.

    A model that asks for 10,000 memories would otherwise flood the very context window the
    recall is meant to serve; one that asks for 0 (or sends a string) would get nothing or an
    upstream error. A malformed argument costs a tool call, never the wake.
    """
    _, searcher = fake_mempalace
    palace = tmp_path / "palace"
    palace.mkdir()
    tool = _search_tool(palace)

    tool.run(query="anything", n_results=10_000)
    tool.run(query="anything", n_results=0)
    tool.run(query="anything", n_results="3")  # a model can send a string
    tool.run(query="anything", n_results=8)

    assert [kwargs["n_results"] for _q, _p, kwargs in searcher.queries] == [20, 1, 5, 8]


def test_search_tool_reports_a_miss_so_the_model_can_refine(fake_mempalace, tmp_path):
    """A miss says so plainly (the SQLite memory tool's phrasing), rather than returning silence
    the model would read as "I know nothing" — it can narrow the query and ask again."""
    _, searcher = fake_mempalace
    searcher.result = {"results": []}
    palace = tmp_path / "palace"
    palace.mkdir()

    assert (
        _search_tool(palace).run(query="nothing matches") == "No memories match 'nothing matches'."
    )


def test_search_tool_needs_a_query(fake_mempalace, tmp_path):
    _, searcher = fake_mempalace
    palace = tmp_path / "palace"
    palace.mkdir()

    assert "needs a query" in _search_tool(palace).run(query="   ")
    assert searcher.queries == []  # never reached MemPalace


def test_search_tool_before_the_palace_exists_reports_a_miss(fake_mempalace, tmp_path):
    """Nothing observed yet → no palace dir → a clean miss, short-circuited before MemPalace."""
    _, searcher = fake_mempalace

    assert "No memories match" in _search_tool(tmp_path / "palace").run(query="anything")
    assert searcher.queries == []


def test_search_tool_degrades_when_the_backend_cannot_do_lexical_search(fake_mempalace, tmp_path):
    """A backend that can't serve the union request answers with an error dict and no `results`
    key — which reads as a miss, exactly as it does for `context`. Memory degrades; nothing raises.
    """
    _, searcher = fake_mempalace
    searcher.result = {"error": "backend does not support lexical_search"}
    palace = tmp_path / "palace"
    palace.mkdir()

    assert "No memories match" in _search_tool(palace).run(query="anything")


# --- shape + the missing-extra error -----------------------------------------


def test_missing_mempalace_degrades_to_an_actionable_error(tmp_path, monkeypatch):
    """With the extra not installed, both surfaces surface a clear "install it" ImportError.

    The tool's raise is safe: the engine turns any tool failure into model-readable text
    (`_run_tool`), so a missing extra costs the call, never the wake — and what the model then
    reads names the extra to install rather than a raw "No module named" trace.
    """
    # Ensure no fake is present and the real package is absent.
    for name in ("mempalace", "mempalace.convo_miner", "mempalace.searcher"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    palace = tmp_path / "palace"
    palace.mkdir()  # exists, so search reaches the lazy import rather than short-circuiting
    provider = MemPalaceMemoryProvider(palace)

    with pytest.raises(ImportError, match=r"basecradle-harness\[mempalace\]"):
        provider.observe(MemoryExchange(user="hi", assistant="ok", scope=_scope()))
    with pytest.raises(ImportError, match=r"basecradle-harness\[mempalace\]"):
        provider.tools()[0].run(query="anything")
