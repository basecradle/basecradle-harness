"""The pluggable memory seam: a `MemoryProvider` with four optional surfaces.

The capital's homework on the leading memory systems (Mem0/Zep/MemPalace/Letta)
found they are **middleware**, not a key-value box: they *observe* the conversation
to auto-extract durable facts, and *inject* prompt-ready context before the model
runs — not just ``write(key, value)``. The shipped default (a `MemoryTool` fused to a
SQLite file) had no seam for that. This module is the seam.

A `MemoryProvider` declares four surfaces, **each optional**:

1. **tools** (`tools()`) — the model-facing memory ops. The default returns the
   `MemoryTool` over the store; an automatic-only provider returns ``[]``.
2. **store** — the durable engine (write/read/list/delete/search). The default is
   `SqliteMemoryStore`: host-local, private — *"private mind, shared world."*
3. **observe** (`observe(exchange)`) — a wake-loop hook fired **after each
   exchange**, so a provider can auto-capture/extract what was just said.
4. **context** (`context(scope)`) — a hook fired during **Turn-0 composition** (the
   persistent-brief seam), returning prompt-ready memory to inject before the model.

`observe`/`context` default to **no-ops**, so a provider that only wants explicit,
tool-driven memory (the default `SqliteMemoryProvider`) implements nothing extra and
behaves exactly as memory did before this seam existed — the regression bar @jt is
held to.

**Scope is the agent, not the timeline.** Memory is the agent's *one private mind*
that spans **all** its timelines — that is what makes cross-timeline recall possible
(learn in timeline A, recall in timeline B). So `MemoryScope.agent` is the durable
identity key; `timeline` rides along as metadata a provider *may* record but must not
partition on. `query` carries the turn's incoming text so a relevance-ranked provider
(MemPalace) has something to retrieve *against* at Turn-0 time.

Provider selection is config-driven (`memory_provider_from_env`): ``sqlite`` (the
default), ``mempalace`` (the optional reference adapter), or a dotted
``module:Class`` path to any custom `MemoryProvider`. One provider per agent.
"""

from __future__ import annotations

import importlib
import os
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

from basecradle_harness._memory import MemoryTool, SqliteMemoryStore, _default_path
from basecradle_harness._tools import Tool


@dataclass(frozen=True)
class MemoryScope:
    """Who the memory belongs to, plus the context a retrieval hook ranks against.

    Args:
        agent: The durable agent-identity key memory is scoped to. Memory is the
            agent's private mind across **every** timeline it speaks on, so this — not
            the timeline — is the partition key; it is what makes cross-timeline recall
            work.
        timeline: The timeline this turn is on, as *metadata*. A provider may record
            it (provenance) but must not scope retrieval to it, or cross-timeline recall
            breaks.
        query: The incoming turn's text, supplied at Turn-0 composition so a
            relevance-ranked `context` hook has something to retrieve against. ``None``
            when there is no salient query (e.g. a provider that returns recent memory
            unconditionally).
    """

    agent: str
    timeline: str | None = None
    query: str | None = None


@dataclass(frozen=True)
class MemoryExchange:
    """One completed exchange handed to `observe` after the agent has replied.

    Args:
        user: The text the model read this turn (a peer's message, an activated task's
            instructions, a perceived asset) — what prompted the reply.
        assistant: The agent's reply.
        scope: Whose memory this exchange belongs to (see `MemoryScope`); ``query`` is
            unset here — observation captures what *happened*, it does not retrieve.
    """

    user: str
    assistant: str
    scope: MemoryScope


class MemoryProvider(ABC):
    """The pluggable memory backend: tools + store + two optional middleware hooks.

    Subclass and implement only what differs. `tools` and `store` describe the
    explicit, model-driven surface; `observe` and `context` are the automatic
    middleware surface and **default to no-ops**, so a tool-only provider (the default
    SQLite one) is a two-line subclass and behaves exactly as memory always did.

    A hook must be safe to call on every wake: the engine guards them so a raising
    hook degrades (a failed `observe` is logged, a failed `context` simply injects
    nothing) and **never breaks the wake** — but a provider should still avoid raising
    on the common path.
    """

    #: The durable engine, when the provider has one. The default SQLite provider sets
    #: it; a provider with no host-local store (a pure cloud middleware) may leave it None.
    store: object | None = None

    def tools(self) -> list[Tool]:
        """The model-facing memory tools to register. Default: none (automatic-only)."""
        return []

    def observe(self, exchange: MemoryExchange) -> None:
        """Capture a completed exchange. Default: no-op (explicit-memory only)."""

    def context(self, scope: MemoryScope) -> str | None:
        """Prompt-ready memory to inject at Turn 0, or ``None``. Default: no-op."""
        return None

    def close(self) -> None:
        """Release any resources (e.g. close the store). Default: no-op."""


class SqliteMemoryProvider(MemoryProvider):
    """The default provider: the `MemoryTool` over one private `SqliteMemoryStore`.

    Implements **tools + store** only; `observe`/`context` stay no-ops, so an agent on
    this provider has exactly the explicit, write-it-yourself memory it had before the
    pluggable seam — the behavior-preserving default @jt keeps. Semantic auto-capture
    and auto-injection are a *different provider* (the MemPalace adapter), not a change
    to this one.

    Args:
        path: Where the SQLite store lives. Defaults to the standard per-agent location
            (``$HARNESS_HOME/memory.db``); see `SqliteMemoryStore`. The tool and any
            future hook share this one store instance.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.store = SqliteMemoryStore(path)

    def tools(self) -> list[Tool]:
        # One MemoryTool sharing the provider's store, so the model's explicit ops and the
        # store live in one place. The provider owns the store's lifecycle (see `close`).
        return [MemoryTool(store=self.store)]

    def close(self) -> None:
        self.store.close()


# The config var selecting the provider, and the built-in aliases it understands. Any
# other value is treated as a dotted import path to a custom `MemoryProvider` subclass.
_PROVIDER_VAR = "HARNESS_MEMORY_PROVIDER"
_SQLITE = "sqlite"
_MEMPALACE = "mempalace"


def memory_provider_from_env(home: str | os.PathLike[str] | None = None) -> MemoryProvider:
    """Build the agent's one memory provider from ``HARNESS_MEMORY_PROVIDER``.

    The selector, case-insensitive:

    - unset or ``sqlite`` → `SqliteMemoryProvider` (the default — host-local, private).
    - ``mempalace`` → the optional reference adapter
      (`basecradle_harness._mempalace.MemPalaceMemoryProvider`), which requires the
      ``mempalace`` extra (``pip install basecradle-harness[mempalace]``). A clear
      error names the extra if it is not installed.
    - anything else → a dotted ``module:Class`` (or ``module.Class``) path to a custom
      `MemoryProvider` subclass, imported and instantiated with no arguments.

    ``home`` overrides where a host-local store lives (defaults to the per-agent
    location); it is passed to the built-in providers and ignored by a custom class
    that takes no args.
    """
    raw = (os.environ.get(_PROVIDER_VAR) or _SQLITE).strip()
    selector = raw.lower()
    if selector == _SQLITE:
        return SqliteMemoryProvider(_store_path(home))
    if selector == _MEMPALACE:
        from basecradle_harness._mempalace import MemPalaceMemoryProvider

        return MemPalaceMemoryProvider(palace_path=_palace_path(home))
    return _load_custom_provider(raw)


def _store_path(home: str | os.PathLike[str] | None) -> Path | None:
    """The SQLite store path for a given home override, or ``None`` for the default."""
    if home is None:
        return None
    return Path(home) / "memory.db"


def _palace_path(home: str | os.PathLike[str] | None) -> Path:
    """The MemPalace palace directory: under the agent's home, beside the SQLite default.

    Falls back to the same per-agent root the SQLite store uses (``$HARNESS_HOME`` else a
    dotdir under ``$HOME``), so a MemPalace agent's memory is private to its OS user the
    same way the default's is.
    """
    root = Path(home) if home is not None else _default_path().parent
    return root / "mempalace"


def _load_custom_provider(path: str) -> MemoryProvider:
    """Import and instantiate a custom `MemoryProvider` from a ``module:Class`` path.

    Accepts ``pkg.mod:Class`` (preferred, unambiguous) or ``pkg.mod.Class``. The target
    must be a `MemoryProvider` subclass and is instantiated with no arguments — a custom
    provider configures itself from its own environment. Any failure is a clear error
    naming the bad value, never a silent fall-through to the default.
    """
    module_name, _, attr = path.partition(":")
    if not attr:
        module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(
            f"{_PROVIDER_VAR}={path!r} is not a known provider or a 'module:Class' path. "
            f"Use 'sqlite', 'mempalace', or a dotted import path."
        )
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, attr)
    except (ImportError, AttributeError) as error:
        raise ValueError(
            f"Could not import memory provider {path!r} ({_PROVIDER_VAR}): {error}"
        ) from error
    if not (isinstance(cls, type) and issubclass(cls, MemoryProvider)):
        raise TypeError(f"{path!r} is not a MemoryProvider subclass.")
    return cls()
