"""The default memory: a private, persistent SQLite store and its tool surface.

This is the template that gets mass-copied to spawn production peers, so it is a
real memory system rather than a toy: a single SQLite file, full CRUD
(`write`/`read`/`list`/`delete`), keyword recall over key *and* value via FTS5
(`search` — so an agent need not remember its own exact keys), and a forward-only
schema migration runner so an uneven rollout across a fleet of servers is safe.

**Surface vs. engine (the Group 4 split).** Memory is now a *pluggable provider*
(`basecradle_harness._memory_provider`), so the one class this module used to fuse
is split along the seam the provider draws:

- `SqliteMemoryStore` — the **engine**: the five durable ops (write/read/list/
  delete/search) over the SQLite file, returning model-readable strings. It knows
  nothing about being a tool; a provider's `observe`/`context` hooks could read and
  write it directly.
- `MemoryTool` — the **model-facing surface**: a `Tool` that dispatches the five
  actions onto a store. It owns its store by default (so the simple
  ``MemoryTool(path=…)`` still works unchanged), or shares one a provider hands it.

The default `SqliteMemoryProvider` wires `MemoryTool` over a `SqliteMemoryStore`;
its `observe`/`context` hooks are deliberate no-ops, so explicit-memory behavior is
exactly what it was before the split. A richer provider (e.g. the MemPalace adapter)
swaps the engine and lights the hooks up without touching this surface.

**Private mind, shared world.** Memory is each agent's own store under its home
(`$HARNESS_HOME/memory.db`), isolated per OS user. It never goes on the platform —
peers do not see each other's memories; they share only by talking on timelines.

Storage is one SQLite file, the boring self-contained answer: no external service,
no vector DB, nothing leaves the host. `sqlite3` is in the standard library, so
this adds no dependency. Semantic/embedding recall (the Letta/MemGPT line) is
deliberately out of scope for the default; it arrives as a *different provider*
(the pluggable seam), not as a new action bolted onto this store.

The schema is versioned with ``PRAGMA user_version`` and migrated **forward-only
and additively** on open (see `_migrate`): never drop or rename, only add. That is
what makes a multi-server rollout safe — each agent self-migrates its own DB on its
next wake, and crucially *older code still opens a newer DB*, because every change
is additive and old code simply ignores what it does not know. Retrofitting
versioning onto a version-less store across a live fleet is the silent nightmare
this discipline avoids, so it ships now, with the rebuild.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from basecradle_harness._tools import Tool

# Where the store lives when no explicit path is given: under the agent's home
# (`$HARNESS_HOME`) so it is private to that OS user, falling back to a dotdir in
# the user's home for a plain local run. The filename is `memory.db`.
DEFAULT_DIRNAME = ".basecradle_harness"
DEFAULT_FILENAME = "memory.db"

# The schema the running code targets. On open, a DB at a lower version is migrated
# up to here; a DB at a *higher* version (written by newer code) still opens, because
# every migration is additive and this code ignores what it does not use. Bump this
# and append to `_MIGRATIONS` to evolve the schema.
SCHEMA_VERSION = 1

# Forward-only, additive migrations, indexed by the version they produce. Migration
# `n` upgrades a DB from version `n-1` to `n`; it must be idempotent-safe (it only
# runs when the DB is below `n`) and must never drop or rename. The base table is
# migration 1; a later column or table is migration 2, and so on — old code keeps
# reading a DB a newer migration touched.
_MIGRATIONS: dict[int, str] = {
    1: """
        CREATE TABLE memories (
            key        TEXT NOT NULL UNIQUE,
            value      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """,
}

# How many matches `search` returns, so a broad query can't flood the model's
# context. A memory store rarely holds enough to hit this; when it bites, the reply
# says there may be more.
SEARCH_LIMIT = 20


class SqliteMemoryStore:
    """The five durable memory ops over one SQLite file — the engine behind the tool.

    Store, recall, list, delete, and search facts persisted in a single SQLite file.
    Each fact is a ``value`` under a unique ``key``, with ``created_at``/``updated_at``
    timestamps; ``search`` does keyword recall over both key and value (SQLite FTS5),
    so the agent can find a fact without remembering the exact key it used. Every method
    returns a string written for the model to read — the engine is the source of those
    messages, and `MemoryTool` is a thin dispatcher over it.

    This is the default `MemoryProvider`'s store surface
    (`basecradle_harness._memory_provider.SqliteMemoryProvider`): a provider's
    `observe`/`context` hooks may read and write the same instance the tool uses.

    Args:
        path: Where the SQLite store lives. Defaults to ``$HARNESS_HOME/memory.db``
            when ``HARNESS_HOME`` is set, else ``~/.basecradle_harness/memory.db``;
            pass a path (e.g. a temp file in tests) to point it elsewhere. The
            parent directory is created on first use. The connection is opened
            lazily on the first call and reused for the store's life.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _default_path()
        self._conn: sqlite3.Connection | None = None
        self._fts = False  # set when the connection opens, once FTS5 support is known

    # --- the five ops --------------------------------------------------------

    def write(self, key: str, value: str) -> str:
        now = _now()
        conn = self._connect()
        # Upsert: a new key is inserted with both timestamps; an existing key keeps
        # its created_at and refreshes value + updated_at. The FTS index (when
        # present) is kept in sync by triggers, so there is nothing to update here.
        conn.execute(
            """
            INSERT INTO memories (key, value, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, now, now),
        )
        conn.commit()
        return f"Remembered {key!r}."

    def read(self, key: str) -> str:
        conn = self._connect()
        row = conn.execute("SELECT value FROM memories WHERE key = ?", (key,)).fetchone()
        if row is not None:
            return row[0]
        # Tell the model the keys it *does* have, so a wrong guess (common when a
        # fresh agent recalls across a restart) can self-correct on the next call.
        keys = self._keys()
        if keys:
            return f"No memory stored under {key!r}. Stored keys: {', '.join(keys)}."
        return f"No memory stored under {key!r}. You have no memories yet."

    def list(self) -> str:
        keys = self._keys()
        if not keys:
            return "No memories stored yet."
        return ", ".join(keys)

    def delete(self, key: str) -> str:
        conn = self._connect()
        cursor = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        conn.commit()
        if cursor.rowcount == 0:
            return f"No memory stored under {key!r}; nothing to delete."
        return f"Forgot {key!r}."

    def search(self, query: str) -> str:
        conn = self._connect()
        if self._fts:
            rows = conn.execute(
                """
                SELECT key, value FROM memories_fts
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (_fts_query(query), SEARCH_LIMIT + 1),
            ).fetchall()
        else:
            # FTS5 absent in this SQLite build — fall back to a substring scan so
            # search still works, just without ranking. The store is small. Mirror
            # the FTS path's per-term OR recall (any term matches over key or value),
            # and escape LIKE's own wildcards so a query of '%' or '_' is literal,
            # not match-everything.
            terms = query.split()
            clause = " OR ".join(
                ["(key LIKE ? ESCAPE '\\' OR value LIKE ? ESCAPE '\\')"] * len(terms)
            )
            params: list[object] = []
            for term in terms:
                pattern = f"%{_escape_like(term)}%"
                params += [pattern, pattern]
            params.append(SEARCH_LIMIT + 1)
            rows = conn.execute(
                f"SELECT key, value FROM memories WHERE {clause} ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
        if not rows:
            return f"No memories match {query!r}."
        lines = [f"{key}: {value}" for key, value in rows[:SEARCH_LIMIT]]
        if len(rows) > SEARCH_LIMIT:
            lines.append(
                f"(showing the first {SEARCH_LIMIT}; there may be more — refine the query)"
            )
        return f"Memories matching {query!r}:\n" + "\n".join(lines)

    # --- storage -------------------------------------------------------------

    def _keys(self) -> list[str]:
        conn = self._connect()
        return [row[0] for row in conn.execute("SELECT key FROM memories ORDER BY key")]

    def _connect(self) -> sqlite3.Connection:
        """Open (once) the SQLite connection, migrating the schema and wiring FTS.

        Lazy so constructing the store touches no disk; cached so a process opens the
        DB once. The connection lives for the process's life — each wake is a fresh
        process with a fresh store, so there is no cross-process connection to manage.
        """
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path)
            _migrate(conn)
            self._fts = _ensure_fts(conn)
            self._conn = conn
        return self._conn

    def close(self) -> None:
        """Close the underlying connection if one was opened. Safe to call repeatedly."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class MemoryTool(Tool):
    """The model-facing memory tool: write/read/list/delete/search over a store.

    A thin dispatcher that turns the model's ``action`` into a call on a
    `SqliteMemoryStore`. It owns its store by default — so the simple
    ``MemoryTool(path=…)`` keeps working exactly as before the Group 4 split — or
    shares one a `MemoryProvider` hands it, so the provider's `observe`/`context`
    hooks and the model's explicit ops all see the same facts.

    Args:
        path: Where the SQLite store lives when this tool constructs its own. Ignored
            when ``store`` is given. Defaults to ``$HARNESS_HOME/memory.db`` (else a
            dotdir under ``$HOME``); see `SqliteMemoryStore`.
        store: An existing store to dispatch onto, shared with a provider. When set,
            ``path`` is not consulted and `close` leaves the store alone — the
            provider that owns it is responsible for closing it.
    """

    name = "memory"
    description = (
        "Your long-term memory. Use it to remember facts across the conversation and "
        "across restarts. action='write' stores value under key (overwrites if the key "
        "exists); action='read' returns the value for key; action='list' returns every "
        "key you have stored; action='delete' forgets a key; action='search' finds "
        "memories by keyword across both keys and values — use it when you remember "
        "roughly what a fact was about but not the exact key you filed it under."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write", "read", "list", "delete", "search"],
                "description": "What to do.",
            },
            "key": {
                "type": "string",
                "description": "The label to store or recall under (write, read, delete).",
            },
            "value": {
                "type": "string",
                "description": "The fact to store (write only).",
            },
            "query": {
                "type": "string",
                "description": "Keywords to recall by, matched over keys and values (search only).",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self, path: str | Path | None = None, *, store: SqliteMemoryStore | None = None
    ) -> None:
        # Own a store (the standalone default) or share one a provider passes in. `_owns`
        # gates `close`: a tool that built its store closes it; one given a shared store
        # leaves closing to the provider that owns it.
        self.store = store if store is not None else SqliteMemoryStore(path)
        self._owns = store is None

    @property
    def path(self) -> Path:
        """The store's file path — kept for callers/tests that read it off the tool."""
        return self.store.path

    def run(
        self,
        action: str,
        key: str | None = None,
        value: str | None = None,
        query: str | None = None,
    ) -> str:
        """Dispatch on `action`. Returns a message written for the model to read."""
        if action == "write":
            if not key or value is None:
                return "Error: 'write' needs both a key and a value."
            return self.store.write(key, value)
        if action == "read":
            if not key:
                return "Error: 'read' needs a key."
            return self.store.read(key)
        if action == "list":
            return self.store.list()
        if action == "delete":
            if not key:
                return "Error: 'delete' needs a key."
            return self.store.delete(key)
        if action == "search":
            if not query or not query.strip():
                return "Error: 'search' needs a query."
            return self.store.search(query)
        return (
            f"Error: unknown action {action!r}. Use 'write', 'read', 'list', 'delete', or 'search'."
        )

    def close(self) -> None:
        """Close the store if this tool owns it; a shared store is the provider's to close."""
        if self._owns:
            self.store.close()


# --- module helpers ----------------------------------------------------------


def _default_path() -> Path:
    """The store's default location: under ``$HARNESS_HOME`` when set, else a dotdir.

    Resolved at construction (the hosting agent sets the environment before building
    its tools), so the agent's memory lands in its own home and stays private to its
    OS user. An explicit ``path`` argument overrides this entirely.
    """
    home = os.environ.get("HARNESS_HOME")
    root = Path(home) if home else Path.home() / DEFAULT_DIRNAME
    return root / DEFAULT_FILENAME


def _now() -> str:
    """An ISO 8601 UTC timestamp for created_at / updated_at."""
    return datetime.now(timezone.utc).isoformat()


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring the DB's schema up to `SCHEMA_VERSION`, forward-only and additively.

    Reads the DB's own ``PRAGMA user_version`` and applies every migration above it,
    in order, advancing the version after each. A DB already at or above the target
    is left untouched — including one written by *newer* code (a higher version):
    because every migration is additive, this code opens it and simply ignores the
    schema it does not use. That is what makes an uneven fleet rollout safe.
    """
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for target in range(version + 1, SCHEMA_VERSION + 1):
        conn.executescript(_MIGRATIONS[target])
        conn.execute(f"PRAGMA user_version = {target}")
    conn.commit()


def _ensure_fts(conn: sqlite3.Connection) -> bool:
    """Create the FTS5 index over key + value and keep it synced; report if it exists.

    The FTS index is a derived, rebuildable view of `memories`, not part of the
    versioned schema — so it is (re)created idempotently on every open rather than
    through a migration, which keeps the version counter about the durable table
    alone. When this SQLite build lacks FTS5, search degrades to a substring scan
    (see `_search`) and this returns ``False`` instead of failing the whole tool.

    Triggers mirror every insert/update/delete on `memories` into the index, so the
    write path never has to think about it. When the index is created fresh, it is
    rebuilt from any rows already present (the case where FTS5 became available, or
    a future migration added rows, after the index last existed).
    """
    if not _fts5_available(conn):
        return False
    existed = (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memories_fts'"
        ).fetchone()
        is not None
    )
    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(key, value, content='memories', content_rowid='rowid');

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts (rowid, key, value)
            VALUES (new.rowid, new.key, new.value);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts (memories_fts, rowid, key, value)
            VALUES ('delete', old.rowid, old.key, old.value);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts (memories_fts, rowid, key, value)
            VALUES ('delete', old.rowid, old.key, old.value);
            INSERT INTO memories_fts (rowid, key, value)
            VALUES (new.rowid, new.key, new.value);
        END;
        """
    )
    if not existed:
        # Index any rows that predate the index (FTS5 just became available, or the
        # base table already held data). A no-op on a fresh, empty DB.
        conn.execute("INSERT INTO memories_fts (memories_fts) VALUES ('rebuild')")
    conn.commit()
    return True


def _fts5_available(conn: sqlite3.Connection) -> bool:
    """Whether this SQLite build can create an FTS5 virtual table.

    FTS5 ships in standard SQLite builds (and CPython's bundled sqlite on macOS and
    Linux), but it is a compile-time option, so a stray build could lack it. Probe
    once with a throwaway temp table rather than assume.
    """
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp._fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE temp._fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def _fts_query(query: str) -> str:
    """Turn free text into a forgiving FTS5 MATCH expression.

    Each whitespace-separated term is wrapped as a quoted phrase (doubling any inner
    quote, per FTS5 escaping) so user text can never be read as FTS operator syntax,
    and the terms are OR-joined so any one match recalls the row — recall over
    precision, which is what "I half-remember this" wants. An empty query is guarded
    before this is ever called.
    """
    terms = query.split()
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _escape_like(term: str) -> str:
    """Escape a term for a LIKE pattern: ``\\`` is the escape, so ``%`` and ``_``
    (LIKE's wildcards) become literal. Backslash is escaped first so the others'
    escapes are not doubled. Used only by `search`'s FTS5-absent fallback.
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
