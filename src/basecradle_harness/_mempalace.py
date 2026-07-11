"""The MemPalace reference adapter: a real, middleware-style `MemoryProvider`.

[MemPalace](https://github.com/mempalace/mempalace) is a local-first, well-benchmarked
open-source AI memory system — ChromaDB (vectors) + SQLite (knowledge graph), all on
the host, no API key. This adapter wraps it as a `MemoryProvider` to prove the Group 4
seam end-to-end: it lights up the two middleware hooks the default SQLite provider
leaves dark, and puts one read-only tool beside them —

- **`observe(exchange)`** feeds each completed exchange into MemPalace, so the agent's
  memory grows automatically from the conversation rather than only from explicit
  ``memory write`` calls.
- **`context(scope)`** retrieves the top-K relevant chunks for the turn and returns them
  as a prompt-ready block injected at Turn 0 — MemPalace's "auto-inject relevant memory
  before the model runs."
- **`tools()`** supplies one **read-only** `memory_search` tool (issue #267) — the *deliberate*
  half beside the automatic one. `context` retrieves once per wake, against the incoming turn's
  text alone; a memory the agent needs mid-task that the Turn-0 top-K missed was unreachable for
  the rest of that wake. The tool is the way back to the palace with a refined query. It wraps
  the very same in-process `search` call `context` uses, and adds **no write surface** — `observe`
  stays the palace's sole writer, so the concurrent-writer question never arises. (Which is also
  why this is not MemPalace's own MCP server: that would pay a chromadb import on every wake, and
  its per-palace writer lease arbitrates only between MCP server processes — not against this
  adapter's library-path writes.)

**Agent-scoped, cross-timeline.** One palace lives under the agent's home, and retrieval
is *not* filtered by timeline — so a fact learned on one timeline is recalled on another.
That is the whole point of the proof (the capital's "Memory Prince" demonstration).

**Library API, not MCP.** This uses MemPalace's in-process Python functions
(`mempalace.convo_miner.mine_convos` to store, `mempalace.searcher.search_memories` to
retrieve) — *not* its MCP tools, which are a later group (Group 5). MemPalace is an
**optional extra** (``pip install basecradle-harness[mempalace]``) so the base package
stays light; the import is lazy and a clear error names the extra when it is missing.

**Retrieval model.** Both surfaces retrieve through the one `search` method, which searches with
``candidate_strategy="union"``: the hybrid rerank pool is seeded from the top *lexical* (BM25)
hits as well as the top vector hits — not vectors alone (MemPalace's default). Agent memory turns
on exact tokens (handles, UUIDs, error strings, project names) that embeddings rank poorly, which
is precisely the recall gap union closes. One call, so the automatic hook and the model-facing
tool can never drift apart on *how* the palace is searched. See `_CANDIDATE_STRATEGY`.

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
from basecradle_harness._tools import Tool

_log = logging.getLogger("basecradle_harness")

# How many relevant chunks `context` retrieves to inject at Turn 0, and the tool returns when
# the model names no count. Bounded so a large palace can't flood the model's context window.
DEFAULT_N_RESULTS = 5

# The ceiling on a *model-chosen* count (`memory_search`'s `n_results`). The Turn-0 default is
# small because it is paid on every wake; a deliberate search is paid only when the agent asks,
# so it may reach further — but not without limit, or one tool call could bury the context
# window the recall is meant to serve.
MAX_N_RESULTS = 20

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

# The heading `context` puts above the memories it injects at Turn 0 — memories the model did
# *not* ask for, so they are framed as recall rather than as an answer (the search tool, which
# answers a question the model did ask, uses its own heading).
_INJECTED_HEADING = "Relevant memories from past conversations (across all your timelines):\n"

# The wing the mined exchanges are filed under. A single wing per agent keeps every
# timeline's conversation in one searchable space, so retrieval spans them all
# (cross-timeline recall); the agent identity already partitions palaces by home.
_CONVERSATIONS_WING = "conversations"

_MISSING = (
    "MemPalace is not installed. Install the optional extra to use the MemPalace memory "
    "provider:  pip install basecradle-harness[mempalace]"
)


class MemPalaceMemoryProvider(MemoryProvider):
    """A `MemoryProvider` backed by MemPalace's local library API (observe + context + search).

    Args:
        palace_path: The palace directory (ChromaDB + SQLite) for this agent. Created on
            first write; private to the agent's home so peers never share a mind.
        n_results: How many relevant chunks `context` retrieves and injects, and the default
            for the `memory_search` tool when the model names no count. Defaults to
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
        if not query:
            return None
        hits = self.search(query)
        if not hits:
            return None
        return _INJECTED_HEADING + _render_hits(hits)

    def search(self, query: str, n_results: int | None = None) -> list[dict]:
        """The one retrieval call both memory surfaces make: relevant chunks for `query`.

        Shared by the automatic `context` hook (Turn-0 injection) and the model-facing
        `MemPalaceSearchTool` (deliberate mid-task recall), so the two can never drift apart
        on *how* the palace is searched — the union pool, the no-`max_distance` rule, and the
        bound all live here once. Returns the raw hit dicts (possibly empty); rendering is the
        caller's, because a Turn-0 block and a tool result read differently.

        Empty before the palace exists (nothing observed yet) — short-circuited without
        touching MemPalace. A backend that cannot serve the union request answers with an error
        dict carrying no ``results`` key, which reads here as no hits: memory degrades, the wake
        does not break.
        """
        if not self.palace_path.exists():
            return []
        searcher = _import("searcher")
        # Never pass `max_distance`: upstream's union merge opens with
        # `if max_distance > 0.0: return`, so *any* distance threshold silently disables
        # the BM25 half of the pool (lexical-only candidates carry no vector distance) and
        # `candidate_strategy` above becomes a no-op. A distance filter and union recall
        # are mutually exclusive upstream; we keep the recall. Pinned by test.
        result = searcher.search_memories(
            query,
            str(self.palace_path),
            n_results=self.n_results if n_results is None else n_results,
            candidate_strategy=_CANDIDATE_STRATEGY,
        )
        hits = result.get("results") if isinstance(result, dict) else None
        return [hit for hit in (hits or []) if isinstance(hit, dict) and hit.get("text")]

    # --- tools: deliberate recall on top of the automatic hooks ---------------

    def tools(self) -> list[Tool]:
        """One **read-only** search tool, so recall is not frozen at Turn 0 (issue #267).

        `context` retrieves exactly once per wake, against the incoming turn's text. A memory
        the agent needs *mid-task* — and that the Turn-0 top-K did not happen to surface — was
        simply unreachable for the rest of the wake: the model had no way back to the palace
        with a refined query ("what was that endpoint we discussed in March?"). This is that way
        back, and it is purely additive — `observe`/`context` are unchanged, so ambient memory
        still works exactly as before and the tool is the *deliberate* half beside it.

        **Read-only, on purpose.** No write and no delete surface: `observe` remains the palace's
        sole writer, so there is no concurrent-writer question to answer at all — the reason this
        is an in-process tool rather than MemPalace's own MCP server (whose per-palace writer
        lease arbitrates only between MCP server processes, not against the adapter's library
        writes, and which would pay a chromadb import on every wake besides).
        """
        return [MemPalaceSearchTool(self)]


class MemPalaceSearchTool(Tool):
    """Deliberate recall: search the palace mid-task, with a query the model chooses.

    The model-facing half of MemPalace memory (issue #267), beside the automatic half. It is a
    thin dispatcher onto `MemPalaceMemoryProvider.search` — the *same* in-process call the
    `context` hook makes (same union pool, same no-`max_distance` rule) — so what the agent can
    reach by asking is exactly what the palace would have injected, only with a query it wrote
    itself and at the moment it needs it.

    **Read-only.** Search is the whole surface: no write, no delete. `observe` stays the palace's
    only writer, which is what keeps the concurrent-writer question from existing.

    Args:
        provider: The provider whose palace to search — the tool borrows its retrieval call and
            its default bound rather than reaching into MemPalace itself, so a change to *how*
            this agent searches its palace lands in one place.
    """

    name = "memory_search"
    description = (
        "Search your long-term memory for what you know about something. Your memory is "
        "automatic — past conversations are recalled for you at the start of each turn — so "
        "reach for this when you need something that *wasn't* recalled: a detail from an older "
        "conversation, on any timeline, that the current turn didn't surface (\"what was that "
        'endpoint we discussed in March?"). Search by what the memory was about; a specific '
        "query recalls better than a vague one."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to recall, in your own words. Exact tokens (a handle, a "
                "uuid, an error string, a project name) recall well — they are searched "
                "lexically as well as semantically.",
            },
            "n_results": {
                "type": "integer",
                "description": (
                    f"How many memories to return. Defaults to {DEFAULT_N_RESULTS}; "
                    f"at most {MAX_N_RESULTS}."
                ),
                "minimum": 1,
                "maximum": MAX_N_RESULTS,
            },
        },
        "required": ["query"],
    }

    def __init__(self, provider: MemPalaceMemoryProvider) -> None:
        self.provider = provider

    def run(self, query: str, n_results: int | None = None) -> str:
        """Search the palace and render the hits for the model. Never raises on bad input.

        The bound is clamped, not trusted: a model that asks for 10,000 memories (or zero, or a
        negative count) gets a sane page rather than a flooded context window or an upstream
        error — the schema's ``minimum``/``maximum`` are advisory to the model, and this is the
        enforcement. A miss says so plainly, the way the SQLite memory tool's ``search`` does, so
        the model can refine and try again instead of reading silence as "I know nothing."
        """
        if not query or not query.strip():
            return "Error: 'memory_search' needs a query."
        hits = self.provider.search(query.strip(), _bounded(n_results, self.provider.n_results))
        if not hits:
            return f"No memories match {query!r}."
        return f"Memories matching {query!r}:\n" + _render_hits(hits)


# --- helpers -----------------------------------------------------------------


def _bounded(n_results: int | None, default: int) -> int:
    """The requested result count, clamped into ``1..MAX_N_RESULTS`` (default when unset).

    A non-integer (a model can send ``"5"``) falls back to the default rather than raising —
    a malformed argument must cost the agent a tool call, never the wake.
    """
    if not isinstance(n_results, int) or isinstance(n_results, bool):
        return default
    return max(1, min(n_results, MAX_N_RESULTS))


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


def _render_hits(hits: list[dict]) -> str:
    """The verbatim recalled chunks as a bullet list — the body both surfaces show the model.

    Each hit is a dict from ``search_memories`` carrying a ``text`` chunk (`search` has already
    dropped any that don't). Only the *heading* differs between the two callers: `context`
    announces recalled context the model did not ask for, while the search tool answers a
    question the model did ask — same memories, different framing.
    """
    return "\n".join(f"- {hit['text'].strip()}" for hit in hits)
