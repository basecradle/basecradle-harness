"""The MemPalace reference adapter, against a mocked MemPalace library.

MemPalace is an optional extra and not installed in the test env, so its library is
faked at the ``sys.modules`` boundary (the adapter imports ``mempalace.convo_miner`` /
``mempalace.searcher`` lazily). These pin the contract the adapter relies on: `observe`
mines a quote-formatted exchange file, `context` retrieves and renders top-K hits, the
provider exposes one **read-only** `memory_search` tool over that same retrieval call, and a
genuinely missing package degrades to an actionable "install the extra" error.
"""

import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

from basecradle_harness._basecradle import _publish_palace_binding, _resolve_tools
from basecradle_harness._memory_provider import (
    MemoryExchange,
    MemoryScope,
    SqliteMemoryProvider,
    _palace_path,
    memory_provider_from_env,
)
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


# === One palace, reachable from the CLI too (issue #409) =====================
#
# The adapter binds a per-agent palace under `$HARNESS_HOME`; MemPalace's own `mempalace` CLI
# defaults to `~/.mempalace/palace`. Nothing joined the two, so on a provisioned agent a bare
# `mempalace status` reported an empty palace it had never used while the live one — 2,488 drawers
# on @briggs — sat one directory away. What follows pins the join: the harness **publishes** the
# path it just bound into the file every CLI command reads when given no `--palace`.
#
# The publication is a *projection* of the binding, never an input to it. Nothing here reads that
# file back into the adapter, which is what keeps it from being the second source of truth a
# hand-written config would be: it cannot redirect the agent's mind, and the next bind rewrites it.


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated `$HOME`, so publishing can never touch the developer's real `~/.mempalace`.

    `Path.home()` resolves through `HOME` on POSIX, which is what the publisher uses — and what
    upstream's `MempalaceConfig` uses for the same directory, so pinning it here moves both halves
    together, exactly as a real agent's OS user does.
    """
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setenv("HOME", str(fake))
    return fake


def _cli_default_palace(home: Path, *, palace_flag: str | None = None) -> str:
    """The palace a `mempalace` command would operate on — upstream's own resolution order.

    A faithful transcription of `mempalace.config.MempalaceConfig.palace_path` (MemPalace 3.x):
    an explicit ``--palace`` wins, then ``MEMPALACE_PALACE_PATH`` / ``MEMPAL_PALACE_PATH``, then
    ``~/.mempalace/config.json``'s ``palace_path`` (expanduser'd), then ``~/.mempalace/palace``.
    Every CLI command resolves through exactly this one property — ``status``, ``search``,
    ``sync``, ``mine``, ``repair-status``, ``migrate``, ``wake-up`` alike — which is why pointing
    it is enough to point all of them.

    Transcribed rather than imported because MemPalace is an optional extra this suite never
    installs (it would drag ChromaDB and an ONNX runtime into CI for one property). A transcription
    can drift from the package it describes, so it is checked against the **real** package by a
    live smoke before a change here ships: install the extra, publish, and run bare ``mempalace
    status`` / ``search`` against a palace the adapter itself mined.
    """
    if palace_flag is not None:
        return os.path.abspath(os.path.expanduser(palace_flag))
    env = os.environ.get("MEMPALACE_PALACE_PATH") or os.environ.get("MEMPAL_PALACE_PATH")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    config_file = home / ".mempalace" / "config.json"
    try:
        stored = json.loads(config_file.read_text(encoding="utf-8")).get("palace_path")
    except (OSError, json.JSONDecodeError, AttributeError):
        stored = None
    return os.path.expanduser(stored or str(home / ".mempalace" / "palace"))


def _bind(monkeypatch, agent_home: Path) -> MemPalaceMemoryProvider:
    """Bind the MemPalace provider the way a wake does — from the environment, nothing passed."""
    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "mempalace")
    monkeypatch.setenv("HARNESS_HOME", str(agent_home))
    return memory_provider_from_env()


# --- the join itself ----------------------------------------------------------


def test_publishing_points_the_bare_cli_at_the_live_palace(home, monkeypatch, tmp_path):
    """The bug, inverted: with no flags and no env, the CLI now resolves the agent's own palace.

    Before the publication the CLI answered ``~/.mempalace/palace`` — a directory the agent had
    never written — and said so ("No palace found at …"). The assertion is the acceptance check
    on @briggs's box, off-box.
    """
    provider = _bind(monkeypatch, tmp_path / "agent")

    _publish_palace_binding(provider)

    assert _cli_default_palace(home) == str(provider.palace_path)
    assert _cli_default_palace(home) != str(home / ".mempalace" / "palace")


def test_the_published_value_is_exactly_what_the_adapter_bound(home, monkeypatch, tmp_path):
    """The drift test the design owes: a published path that is not `_palace_path(HARNESS_HOME)`
    is the very split this closes, wearing a fix's clothes."""
    agent_home = tmp_path / "agent"
    provider = _bind(monkeypatch, agent_home)

    _publish_palace_binding(provider)

    published = json.loads((home / ".mempalace" / "config.json").read_text())["palace_path"]
    assert published == str(_palace_path(agent_home)) == str(provider.palace_path)


def test_the_publication_follows_a_moved_harness_home(home, monkeypatch, tmp_path):
    """It is written from the *live* binding on every bind, so it cannot go stale.

    This is what separates it from the hand-written `config.json` the issue warned about: a second
    source of truth is right until `HARNESS_HOME` moves and the file does not. Here the next bind
    rewrites it, and the CLI follows the agent rather than a memory of where it used to live.
    """
    _publish_palace_binding(_bind(monkeypatch, tmp_path / "first"))
    assert _cli_default_palace(home) == str(tmp_path / "first" / "mempalace")

    _publish_palace_binding(_bind(monkeypatch, tmp_path / "second"))

    assert _cli_default_palace(home) == str(tmp_path / "second" / "mempalace")


def test_the_cli_and_the_adapter_are_one_mind_not_two(fake_mempalace, home, monkeypatch, tmp_path):
    """What `observe` mines into is exactly what a bare CLI command now reads.

    The acceptance check "after a wake observes a new exchange, bare `mempalace status` sees the
    new drawers", reduced to the thing that makes it true off-box: one directory, reached by both
    halves. A second `chroma.sqlite3` anywhere is the failure this forbids.
    """
    convo_miner, _ = fake_mempalace
    provider = _bind(monkeypatch, tmp_path / "agent")
    _publish_palace_binding(provider)

    provider.observe(MemoryExchange(user="Where do I live?", assistant="Dallas.", scope=_scope()))

    (_convo_dir, mined_palace, _kwargs) = convo_miner.calls[0]
    assert mined_palace == _cli_default_palace(home)
    assert (Path(mined_palace) / "conversations").is_dir()  # the palace the exchange landed in


# --- what it must not disturb -------------------------------------------------


def test_publishing_never_creates_a_palace_at_the_upstream_default(home, monkeypatch, tmp_path):
    """It writes one small JSON file — it never runs `init`/`mine`, so no second palace appears.

    Two palaces would be strictly worse than the bug: the CLI and the harness would each be right
    about a different mind, and neither would say so.
    """
    _publish_palace_binding(_bind(monkeypatch, tmp_path / "agent"))

    assert not (home / ".mempalace" / "palace").exists()
    assert list((home / ".mempalace").iterdir()) == [home / ".mempalace" / "config.json"]


def test_a_non_mempalace_agent_publishes_nothing(home, tmp_path):
    """A SQLite-provider agent — and any non-harness MemPalace user — keeps upstream's defaults.

    Nothing is written at all, so a machine that is not a MemPalace-provider harness agent still
    gets the upstream empty-default behavior it had before this existed.
    """
    _publish_palace_binding(SqliteMemoryProvider(tmp_path / "memory.db"))

    assert not (home / ".mempalace").exists()
    assert _cli_default_palace(home) == str(home / ".mempalace" / "palace")


def test_the_read_only_introspection_path_publishes_nothing(home, monkeypatch, tmp_path):
    """`_resolve_tools` is shared with `--resolved-config`, which must not write anywhere.

    The publication lives one level up, in `_resolve_tools_and_provider` — the "an agent is being
    built to act" seam — so introspecting a live agent over SSH stays side-effect-free.
    """
    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "mempalace")
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "agent"))

    _resolve_tools("openai", "openai", "responses")

    assert not (home / ".mempalace").exists()


def test_the_operators_other_settings_survive_the_publication(home, monkeypatch, tmp_path):
    """`config.json` is upstream's file, and some of its keys are load-bearing for the *data*.

    `embedding_model` in particular: ChromaDB refuses reads when the persisted embedding function
    stops matching, so clobbering it would break the very palace this is pointing at.
    """
    config_file = home / ".mempalace" / "config.json"
    config_file.parent.mkdir()
    config_file.write_text(
        json.dumps(
            {
                "palace_path": "/somewhere/stale",
                "embedding_model": "embeddinggemma",
                "backend": "chroma",
                "hooks": {"auto_save": False},
            }
        ),
        encoding="utf-8",
    )

    _publish_palace_binding(_bind(monkeypatch, tmp_path / "agent"))

    written = json.loads(config_file.read_text())
    assert written["palace_path"] == str(tmp_path / "agent" / "mempalace")
    assert written["embedding_model"] == "embeddinggemma"
    assert written["backend"] == "chroma"
    assert written["hooks"] == {"auto_save": False}


@pytest.mark.parametrize("body", ["{not json at all", '["a", "list"]'])
def test_an_unreadable_config_is_left_completely_alone(home, monkeypatch, tmp_path, body):
    """A file only its author can fix is not ours to overwrite.

    Upstream ignores such a file too, so the CLI is already on its default and the operator has a
    hand-edit to repair — silently replacing it would destroy their work to fix a symptom.
    """
    config_file = home / ".mempalace" / "config.json"
    config_file.parent.mkdir()
    config_file.write_text(body, encoding="utf-8")

    _publish_palace_binding(_bind(monkeypatch, tmp_path / "agent"))

    assert config_file.read_text() == body


def test_an_already_correct_binding_is_not_rewritten(home, monkeypatch, tmp_path):
    """Publishing runs on every bind, so the steady state must be a pure read."""
    from basecradle_harness._mempalace import publish_palace_binding

    provider = _bind(monkeypatch, tmp_path / "agent")
    assert publish_palace_binding(provider.palace_path) is not None  # first bind writes

    assert publish_palace_binding(provider.palace_path) is None  # every bind after does not


def test_an_operators_tilde_written_path_already_counts_as_correct(home, monkeypatch, tmp_path):
    """Upstream expands `~` when it reads the value, so the comparison expands it too — otherwise
    an operator's own `~/…` binding is rewritten to an absolute one on every single wake."""
    from basecradle_harness._mempalace import publish_palace_binding

    config_file = home / ".mempalace" / "config.json"
    config_file.parent.mkdir()
    config_file.write_text(json.dumps({"palace_path": "~/agent/mempalace"}), encoding="utf-8")

    assert publish_palace_binding(home / "agent" / "mempalace") is None
    assert json.loads(config_file.read_text())["palace_path"] == "~/agent/mempalace"


def test_the_published_file_is_owner_only_and_leaves_no_temp_behind(home, monkeypatch, tmp_path):
    """A `config.json` can carry an embeddings API key, so the published copy is never briefly
    world-readable — `os.open` with the mode, not a chmod after the fact.

    Asserted as "no group or other bits" rather than an exact mode, because the umask masks both
    the `mkdir` and the `os.open`: what must hold on any box is that nobody else can read it.
    """
    _publish_palace_binding(_bind(monkeypatch, tmp_path / "agent"))

    config_dir = home / ".mempalace"
    assert stat.S_IMODE((config_dir / "config.json").stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(config_dir.stat().st_mode) & 0o077 == 0
    assert [p.name for p in config_dir.iterdir()] == ["config.json"]  # no stray temp file


def test_a_real_filesystem_obstruction_is_survived(home, monkeypatch, tmp_path):
    """`~/.mempalace` occupied by a regular file — a genuine OS failure, not a stubbed raise.

    The read fails with `NotADirectoryError`, which reads as "unusable" and stops the publication
    before it can write anything, so the operator's file survives and the bind carries on.
    """
    (home / ".mempalace").write_text("not a directory", encoding="utf-8")

    _publish_palace_binding(_bind(monkeypatch, tmp_path / "agent"))  # must not raise

    assert (home / ".mempalace").read_text() == "not a directory"


def test_a_publication_failure_never_breaks_the_bind(home, monkeypatch, tmp_path, caplog):
    """It is an operator convenience: a read-only home or a full disk costs the convenience, never
    the wake. Its absence is not silent either — the CLI names the path it looked at."""
    monkeypatch.setattr(
        "basecradle_harness._mempalace.publish_palace_binding",
        lambda _path: (_ for _ in ()).throw(PermissionError("read-only home")),
    )

    _publish_palace_binding(_bind(monkeypatch, tmp_path / "agent"))  # must not raise

    assert "mempalace CLI config" in caplog.text


# --- precedence: the publication is the *default*, and only that ---------------


def test_an_explicit_palace_flag_still_wins(home, monkeypatch, tmp_path):
    """`--palace` outranks everything, upstream and here — the publication only moves the default."""
    _publish_palace_binding(_bind(monkeypatch, tmp_path / "agent"))

    assert _cli_default_palace(home, palace_flag="/tmp/other") == "/tmp/other"


def test_mempalaces_own_env_var_wins_for_both_halves(home, monkeypatch, tmp_path):
    """`MEMPALACE_PALACE_PATH` outranks the published file for the CLI — so the adapter reads it
    too, or the two would be looking at different minds again with the publication unable to say so.

    This is the trap the issue named: setting that var without teaching the adapter just opens the
    split in the other direction. Reading the same var, in upstream's own order, closes it.
    """
    elsewhere = tmp_path / "elsewhere"
    _publish_palace_binding(_bind(monkeypatch, tmp_path / "agent"))
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(elsewhere))

    provider = memory_provider_from_env()

    assert provider.palace_path == elsewhere
    assert _cli_default_palace(home) == str(elsewhere)


def test_a_host_with_no_harness_palace_keeps_the_upstream_empty_default(home):
    """Importing or installing the extra creates nothing: with no publication, the CLI still
    resolves `~/.mempalace/palace` and still reports it empty."""
    assert _cli_default_palace(home) == str(home / ".mempalace" / "palace")
    assert not (home / ".mempalace").exists()
