"""The MemPalace reference adapter, against a mocked MemPalace library.

MemPalace is an optional extra and not installed in the test env, so its library is
faked at the ``sys.modules`` boundary (the adapter imports ``mempalace.convo_miner`` /
``mempalace.searcher`` lazily). These pin the contract the adapter relies on: `observe`
mines a quote-formatted exchange file, `context` retrieves and renders top-K hits, the
provider exposes no model-facing tool, and a genuinely missing package degrades to an
actionable "install the extra" error.
"""

import sys
import types

import pytest

from basecradle_harness._memory_provider import MemoryExchange, MemoryScope
from basecradle_harness._mempalace import MemPalaceMemoryProvider


@pytest.fixture
def fake_mempalace(monkeypatch):
    """Install fake ``mempalace.convo_miner`` / ``mempalace.searcher`` modules.

    Returns the two fakes so a test can assert how the adapter called them. ``mine_convos``
    records its args; ``search_memories`` returns whatever the test stashes on it.
    """
    convo_miner = types.ModuleType("mempalace.convo_miner")
    convo_miner.calls = []

    def mine_convos(convo_dir, palace_path, **kwargs):
        convo_miner.calls.append((convo_dir, palace_path, kwargs))

    convo_miner.mine_convos = mine_convos

    searcher = types.ModuleType("mempalace.searcher")
    searcher.result = {"results": []}
    searcher.queries = []

    def search_memories(query, palace_path, n_results=5):
        searcher.queries.append((query, palace_path, n_results))
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
    # The query and bound were passed through to MemPalace.
    assert searcher.queries == [("where does john live", str(palace), 3)]


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


# --- shape + the missing-extra error -----------------------------------------


def test_provider_supplies_no_model_facing_tool(tmp_path):
    """Memory is automatic here — a MemPalace agent runs with BaseCradle-only tools."""
    assert MemPalaceMemoryProvider(tmp_path / "palace").tools() == []


def test_missing_mempalace_degrades_to_an_actionable_error(tmp_path, monkeypatch):
    """With the extra not installed, observe surfaces a clear "install it" ImportError."""
    # Ensure no fake is present and the real package is absent.
    for name in ("mempalace", "mempalace.convo_miner", "mempalace.searcher"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    provider = MemPalaceMemoryProvider(tmp_path / "palace")

    with pytest.raises(ImportError, match=r"basecradle-harness\[mempalace\]"):
        provider.observe(MemoryExchange(user="hi", assistant="ok", scope=_scope()))
