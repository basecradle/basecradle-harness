"""The MemPalace reference adapter: a real, middleware-style `MemoryProvider`.

[MemPalace](https://github.com/mempalace/mempalace) is a local-first, well-benchmarked
open-source AI memory system — ChromaDB (vectors) + SQLite (knowledge graph), all on
the host, no API key. This adapter wraps it as a `MemoryProvider` to prove the Group 4
seam end-to-end: it lights up the two middleware hooks the default SQLite provider
leaves dark —

- **`observe(exchange)`** feeds each completed exchange into MemPalace, so the agent's
  memory grows automatically from the conversation rather than only from explicit
  ``memory write`` calls.
- **`context(scope)`** retrieves the top-K relevant chunks for the turn and returns them
  as a prompt-ready block injected at Turn 0 — MemPalace's "auto-inject relevant memory
  before the model runs."

It supplies **no model-facing tool** (`tools()` is empty): memory here is automatic, so
a MemPalace agent runs with BaseCradle-only tools and never sees a ``memory`` action.

**Agent-scoped, cross-timeline.** One palace lives under the agent's home, and retrieval
is *not* filtered by timeline — so a fact learned on one timeline is recalled on another.
That is the whole point of the proof (the capital's "Memory Prince" demonstration).

**Library API, not MCP.** This uses MemPalace's in-process Python functions
(`mempalace.convo_miner.mine_convos` to store, `mempalace.searcher.search_memories` to
retrieve) — *not* its MCP tools, which are a later group (Group 5). MemPalace is an
**optional extra** (``pip install basecradle-harness[mempalace]``) so the base package
stays light; the import is lazy and a clear error names the extra when it is missing.

**Retrieval model.** `context` searches with ``candidate_strategy="union"``, so the hybrid
rerank pool is seeded from the top *lexical* (BM25) hits as well as the top vector hits —
not vectors alone (MemPalace's default). Agent memory turns on exact tokens (handles,
UUIDs, error strings, project names) that embeddings rank poorly, which is precisely the
recall gap union closes. See `_CANDIDATE_STRATEGY`.

**Storage model.** MemPalace mines *files*: `observe` writes each exchange as a tiny
quote-formatted markdown file under ``<palace>/conversations/`` and mines that directory
(MemPalace skips already-mined files, so re-mining the dir only processes the new one).
Known bound: one small file accrues per exchange — acceptable for a reference adapter; a
production deployment would compact or rotate them.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from basecradle_harness._memory_provider import MemoryExchange, MemoryProvider, MemoryScope

_log = logging.getLogger("basecradle_harness")

# How many relevant chunks `context` retrieves to inject at Turn 0. Bounded so a large
# palace can't flood the model's context window.
DEFAULT_N_RESULTS = 5

# How the hybrid (vector + BM25) rerank pool is built. MemPalace's default, "vector",
# seeds the pool from the top vector hits *alone* — so a chunk whose embedding sits far
# from the query never gets reranked, however strong its lexical signal (upstream's own
# docstring names the failure). Agent memory is made of exactly those: handles, UUIDs,
# error strings, project names — exact tokens embeddings rank poorly. "union" additionally
# pulls the top lexical (FTS BM25) candidates into the pool and merges them, for the cost
# of one extra local FTS query per retrieval. The ChromaDB backend every palace uses
# implements the `lexical_search` capability union needs; a backend that doesn't degrades
# gracefully (`search_memories` returns an error dict with no "results" key, which
# `context` already reads as "no hits").
_CANDIDATE_STRATEGY = "union"

# The wing the mined exchanges are filed under. A single wing per agent keeps every
# timeline's conversation in one searchable space, so retrieval spans them all
# (cross-timeline recall); the agent identity already partitions palaces by home.
_CONVERSATIONS_WING = "conversations"

_MISSING = (
    "MemPalace is not installed. Install the optional extra to use the MemPalace memory "
    "provider:  pip install basecradle-harness[mempalace]"
)


class MemPalaceMemoryProvider(MemoryProvider):
    """A `MemoryProvider` backed by MemPalace's local library API (observe + context).

    Args:
        palace_path: The palace directory (ChromaDB + SQLite) for this agent. Created on
            first write; private to the agent's home so peers never share a mind.
        n_results: How many relevant chunks `context` retrieves and injects. Defaults to
            `DEFAULT_N_RESULTS`.
        agent: The agent label MemPalace files mined exchanges under (provenance only;
            scoping is by `palace_path`).
    """

    def __init__(
        self,
        palace_path: str | Path,
        *,
        n_results: int = DEFAULT_N_RESULTS,
        agent: str = "harness",
    ) -> None:
        self.palace_path = Path(palace_path)
        self.n_results = n_results
        self.agent = agent
        # No host-local SQLite store of our own — MemPalace is the engine. The base
        # `store` attribute stays None, which is correct for a middleware provider.

    # --- the two middleware hooks --------------------------------------------

    def observe(self, exchange: MemoryExchange) -> None:
        """Feed one completed exchange into MemPalace by mining a tiny convo file.

        Writes the exchange as a quote-formatted markdown file (MemPalace's
        ``extract_mode="exchange"`` chunks a ``>`` user turn plus the response that
        follows it into one unit) under ``<palace>/conversations/``, then mines that
        directory. MemPalace tracks already-mined files, so re-mining only ingests the
        file just written. An empty exchange (no user text *and* no reply) is skipped.
        """
        if not (exchange.user.strip() or exchange.assistant.strip()):
            return
        convo_dir = self.palace_path / "conversations"
        convo_dir.mkdir(parents=True, exist_ok=True)
        # A uuid filename so concurrent wakes never collide and a re-mine sees each
        # exchange exactly once. The body is MemPalace's exchange format: `>` user turn,
        # then the assistant response verbatim.
        path = convo_dir / f"{uuid.uuid4().hex}.md"
        path.write_text(_exchange_markdown(exchange), encoding="utf-8")

        convo_miner = _import("convo_miner")
        convo_miner.mine_convos(
            str(convo_dir),
            str(self.palace_path),
            wing=_CONVERSATIONS_WING,
            agent=self.agent,
            extract_mode="exchange",
        )

    def context(self, scope: MemoryScope) -> str | None:
        """Retrieve the top-K relevant memories for this turn as a prompt-ready block.

        Searches the whole palace (no timeline filter, so recall spans every timeline)
        for chunks relevant to ``scope.query`` — the incoming turn's text — and renders
        them as an injectable block. Returns ``None`` when there is no query, no palace
        yet, or no hit, so Turn-0 composition simply omits the section.
        """
        query = (scope.query or "").strip()
        if not query or not self.palace_path.exists():
            return None
        searcher = _import("searcher")
        # Never pass `max_distance`: upstream's union merge opens with
        # `if max_distance > 0.0: return`, so *any* distance threshold silently disables
        # the BM25 half of the pool (lexical-only candidates carry no vector distance) and
        # `candidate_strategy` above becomes a no-op. A distance filter and union recall
        # are mutually exclusive upstream; we keep the recall. Pinned by test.
        result = searcher.search_memories(
            query,
            str(self.palace_path),
            n_results=self.n_results,
            candidate_strategy=_CANDIDATE_STRATEGY,
        )
        hits = result.get("results") if isinstance(result, dict) else None
        return _render_memories(hits or [])

    # --- tools: none (automatic memory) --------------------------------------
    # `tools()` inherits the base no-op: a MemPalace agent has BaseCradle-only tools and
    # never sees a model-facing `memory` action — memory is observe/context, not a tool.


# --- helpers -----------------------------------------------------------------


def _import(submodule: str):
    """Import a MemPalace submodule lazily, with a clear "install the extra" error.

    Lazy so the base package never imports MemPalace (or its heavy ChromaDB dependency)
    unless the operator actually selected this provider. A missing package is turned into
    an actionable `ImportError` naming the extra, not a raw "No module named" trace.
    """
    try:
        return __import__(f"mempalace.{submodule}", fromlist=[submodule])
    except ImportError as error:  # noqa: TRY003 - the message is the actionable bit
        raise ImportError(_MISSING) from error


def _exchange_markdown(exchange: MemoryExchange) -> str:
    """One exchange as a MemPalace ``extract_mode="exchange"`` file: ``>`` turn + reply.

    MemPalace chunks by exchange pair — a line beginning ``>`` is the user turn and the
    lines after it (until the next ``>`` or ``---``) are the response. We quote every
    line of the user text so a multi-line message stays one turn, then write the reply
    verbatim below it.
    """
    quoted = "\n".join(f"> {line}" for line in exchange.user.splitlines() or [""])
    return f"{quoted}\n{exchange.assistant}\n"


def _render_memories(hits: list) -> str | None:
    """Render MemPalace search hits into the injectable Turn-0 block, or ``None`` if empty.

    Each hit is a dict from ``search_memories`` carrying at least ``text``; we list the
    verbatim chunks under a heading the model reads as recalled context. An empty hit set
    (or hits with no text) yields ``None`` so the brief omits the section entirely.
    """
    lines = [hit["text"].strip() for hit in hits if isinstance(hit, dict) and hit.get("text")]
    if not lines:
        return None
    body = "\n".join(f"- {line}" for line in lines)
    return "Relevant memories from past conversations (across all your timelines):\n" + body
