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
default), ``mempalace`` (the MemPalace adapter that files each memory under a BaseCradle name and
keeps its own record, issue #1011), ``mempalace-previous`` (the MemPalace adapter the fleet ran
before it, kept selectable so each agent moves one at a time and can be set back), or a dotted
``module:Class`` path to any custom `MemoryProvider`. One provider per agent.
"""

from __future__ import annotations

import importlib
import os
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
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
        seeded: ``True`` when this is a message a first wake seeded into the transcript as
            context (a user-only exchange, observed before the turn), not a turn the agent
            answered. The model is reading it in its transcript for the rest of this wake, so a
            provider may keep what it files from it out of **this wake's** Turn-0 `context`
            (issue #746) — and must still return it from a deliberate search. A provider that
            ignores the flag behaves exactly as before it existed.
        item: The uuid of the platform item the exchange answered (issue #1011): the message, the
            asset, the webhook event, or the task activation. A turn that answered a batch of
            messages is named by its **newest** message's uuid. ``None`` when the exchange answered
            no item (a library caller's own exchange). Handed to every provider; the MemPalace
            adapter names each memory it files by it, so a retry of the same exchange finds it.
        task_instructions: The instructions of the task, when ``item`` is a task activation, as the
            task's author wrote them. ``None`` for every other kind. A provider may key a rule about
            recurring tasks on them (the MemPalace adapter's door rule): a recurring task on this
            platform is a new task with the same instructions each time it fires.
        kind: Which kind of platform item ``item`` is, in the wake's own words (``messages``,
            ``assets``, ``webhook_events``, ``tasks``). ``None`` when the exchange answered no item. It is
            what a provider hands back to `MemoryProvider.resumable` to ask whether a turn cut off
            at its output budget can still be resumed.
        cut_off: ``True`` when the reply was cut off at the output budget, so the turn is unfinished
            and a resume may continue it (issue #490). A provider may hold the exchange rather than
            file a fragment (the MemPalace adapter does, issue #1011); one that ignores the flag
            files it as before.
        resumed: ``True`` when the exchange is a resume finishing a turn an earlier exchange of the
            same ``item`` may already have begun: ``assistant`` is then the continuation only, the
            words the model wrote after the turn was cut off or interrupted. A degraded resume, one
            that produced no words of the model's, is not ``resumed``: nothing continued the turn.
        continues: For a ``resumed`` exchange whose ``assistant`` picks up a reply cut off at its
            output budget where it stopped, every earlier piece of that reply, oldest first, each
            redacted with ``assistant`` as one text (issue #1020); empty otherwise. Each piece was an
            earlier exchange's ``assistant``, redacted without what came after it, so part of a
            secret a cut fell inside can be in it; here it is without. A provider that holds those
            pieces may put these in their place; one that filed them as they came ignores them.
    """

    user: str
    assistant: str
    scope: MemoryScope
    seeded: bool = False
    item: str | None = None
    task_instructions: str | None = None
    kind: str | None = None
    cut_off: bool = False
    resumed: bool = False
    continues: tuple[str, ...] = ()


#: The question `MemoryProvider.resumable` answers: can the turn that answered ``item`` (of
#: ``kind``, on ``timeline``) still be resumed? ``(timeline, item, kind) -> bool``.
ResumeWindow = Callable[[str, str, str], bool]


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

    #: Whether a turn cut off at its output budget can still be resumed (`ResumeWindow`), set by the
    #: wake before every `begin_wake` (issue #1011). The wake answers from its claims: the turn can
    #: be resumed while the claim on the item it answered is still in flight. ``None`` outside a
    #: wake, where nothing can be told, so a provider that holds a cut-off exchange keeps holding
    #: it. Only the MemPalace adapter selected by ``mempalace`` reads it.
    resumable: ResumeWindow | None = None

    def tools(self) -> list[Tool]:
        """The model-facing memory tools to register. Default: none (automatic-only)."""
        return []

    def observe(self, exchange: MemoryExchange) -> None:
        """Capture a completed exchange. Default: no-op (explicit-memory only)."""

    def context(self, scope: MemoryScope) -> str | None:
        """Prompt-ready memory to inject at Turn 0, or ``None``. Default: no-op.

        A provider that returns its own framed block (a heading, a reading guide, fence tags)
        owes the engine two things that exist today only for the MemPalace adapter: a render
        fence around what it injects (`_mempalace`'s `<mempalace-recall>` pair, which bounds the
        block and names its generator, with the tag literals stripped from the hits so a peer
        cannot forge its end), and its framing literals in what `_mining.strip_injected` removes. Without the second, a reply that quotes the block
        is genuine model output on a mined path, so the framing is filed back into memory as
        something the agent was told and recalled on the next wake.

        When a second injecting provider lands, hoist both to this seam (a provider declares
        its framing; the engine fences and strips for every provider) rather than copying
        MemPalace's into the new adapter. The design is on issue #442.
        """
        return None

    def begin_wake(self) -> None:
        """A new wake begins on this provider. Default: no-op.

        One provider can serve more than one wake in a process (a library caller that calls
        ``WakeAgent.wake()`` again), so state a provider keeps *for one wake* is reset here, never
        in the constructor alone. The MemPalace adapter forgets which files the previous wake filed
        from its seeded backlog (issue #746), so a later wake recalls them like any memory.
        """

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
# The MemPalace adapter the fleet ran before the memory redesign (basecradle#966), under the second
# name issue #1010 gave it. Issue #1011 gives `mempalace` to the new adapter
# (`_mempalace_provider.MemPalaceProvider`) and keeps this one selectable as `mempalace-previous`,
# unchanged, so each agent moves one at a time and can be set back: cutover sets an agent's
# `HARNESS_MEMORY_PROVIDER` to `mempalace`, rollback sets it back. Both read and write the one
# palace, under the one extra, with the one `memory_search` tool. Each reports the name it was
# selected by, read off the bound object, so the NOC sees an agent's move in its resolved
# configuration and its `memory` lines.
_MEMPALACE_PREVIOUS = "mempalace-previous"
_MEMPALACE_SELECTORS = frozenset({_MEMPALACE, _MEMPALACE_PREVIOUS})

# The previous MemPalace adapter's class, by import path — so `describe_memory_provider` can *name* a
# bound MemPalace provider without importing the optional-extra module to do it. It still answers to
# both names when built directly (the palace check opens a palace through it), and reports the one
# it carries.
_MEMPALACE_CLASS = "basecradle_harness._mempalace.MemPalaceMemoryProvider"
# The new MemPalace adapter's class (issue #1011), by import path for the same reason. It answers to
# `mempalace` alone.
_MEMPALACE_NAMED_CLASS = "basecradle_harness._mempalace_provider.MemPalaceProvider"

# The PyPI distribution backing each built-in alias, for the manifest's version field. Only
# `mempalace` has one (the harness's `[mempalace]` extra); `sqlite` is absent on purpose — see
# `describe_memory_provider`.
_PROVIDER_DISTRIBUTIONS = {_MEMPALACE: "mempalace", _MEMPALACE_PREVIOUS: "mempalace"}

#: MemPalace's *own* palace-path environment variables, in upstream's precedence order
#: (`mempalace.config.MempalaceConfig.palace_path`): the documented name, the short alias 3.11
#: added (MemPalace#2366), and the legacy ``MEMPAL_`` spelling. The harness reads the same ones, in
#: the same order, so the adapter and the `mempalace` CLI can never resolve different palaces (see
#: `_palace_path`). The one list every reader of them uses; ``test_memory_provider.py`` holds it to
#: the names upstream's property reads when the extra is installed (issue #1026).
PALACE_PATH_VARS = ("MEMPALACE_PALACE_PATH", "MEMPALACE_PALACE", "MEMPAL_PALACE_PATH")


def memory_provider_from_env(home: str | os.PathLike[str] | None = None) -> MemoryProvider:
    """Build the agent's one memory provider from ``HARNESS_MEMORY_PROVIDER``.

    The selector, case-insensitive:

    - unset or ``sqlite`` → `SqliteMemoryProvider` (the default — host-local, private).
    - ``mempalace`` → the MemPalace adapter that files each memory under a BaseCradle name and
      keeps its own record (`basecradle_harness._mempalace_provider.MemPalaceProvider`, issue
      #1011); ``mempalace-previous`` → the MemPalace adapter the fleet ran before it
      (`basecradle_harness._mempalace.MemPalaceMemoryProvider`). Both require the ``mempalace``
      extra (``pip install basecradle-harness[mempalace]``), and a clear error names the extra
      if it is not installed.
    - anything else → a dotted ``module:Class`` (or ``module.Class``) path to a custom
      `MemoryProvider` subclass, imported and instantiated with no arguments.

    ``home`` overrides where a host-local store lives (defaults to the per-agent
    location); it is passed to the built-in providers and ignored by a custom class
    that takes no args.
    """
    return memory_provider_named(os.environ.get(_PROVIDER_VAR), home)


def configured_mempalace_name() -> str:
    """The MemPalace adapter's name as this agent's ``HARNESS_MEMORY_PROVIDER`` selects it.

    For a command that opens a palace by its path rather than binding the agent's memory provider
    (`basecradle-harness-palace-check`), so its ``memory`` lines carry the name the agent's own
    wakes do (issue #1010), and for the two commands that delete inside a palace
    (`basecradle-harness-scrub-palace`, `basecradle-harness-prune-palace`), which read the memories
    filed under both adapters' names only under ``mempalace`` and hold on every name the previous
    adapter does not write under ``mempalace-previous`` (issue #1011). The setting when it names
    the adapter (``mempalace`` or ``mempalace-previous``, any case), else ``mempalace``: the command
    examines a MemPalace palace
    whatever the agent binds, and refusing to run on another setting would hide the palace it was
    asked to read.
    """
    lowered = (os.environ.get(_PROVIDER_VAR) or "").strip().lower()
    return lowered if lowered in _MEMPALACE_SELECTORS else _MEMPALACE


def memory_provider_named(
    selector: str | None, home: str | os.PathLike[str] | None = None
) -> MemoryProvider:
    """Build a memory provider from an explicit selector — `memory_provider_from_env` minus the env.

    The same three-way rule (``sqlite`` / ``mempalace`` or ``mempalace-previous`` / a dotted
    ``module:Class`` path), with the selector supplied by the caller instead of read from
    ``HARNESS_MEMORY_PROVIDER``. ``None`` or empty means the default, exactly as an unset var does.

    It exists so the deterministic off-box resolver (`_resolve.resolve_stems`) can answer *which
    tools a given memory backend contributes* by asking the provider itself — the memory tool is
    part of the resolved tool set (`_basecradle._merge_memory_tools`) but comes from no plugin
    stem, so without this the caller would have to hold a private ``sqlite → memory`` /
    ``mempalace → memory_search`` table. That is the parallel model basecradle-noc#62 forbade.

    Constructing a built-in provider is **side-effect-free**: `SqliteMemoryStore` opens its
    connection lazily on first use, and the MemPalace adapter only records its paths — so this is
    safe on the pure-resolution path, which never calls a store op.
    """
    raw = (selector or _SQLITE).strip()
    lowered = raw.lower()
    if lowered == _SQLITE:
        return SqliteMemoryProvider(_store_path(home))
    if lowered == _MEMPALACE:
        from basecradle_harness._mempalace_provider import MemPalaceProvider

        return MemPalaceProvider(palace_path=_palace_path(home))
    if lowered == _MEMPALACE_PREVIOUS:
        from basecradle_harness._mempalace import MemPalaceMemoryProvider

        return MemPalaceMemoryProvider(palace_path=_palace_path(home), name=lowered)
    return _load_custom_provider(raw)


def describe_memory_provider(provider: MemoryProvider) -> tuple[str, str | None]:
    """The **bound** provider's manifest identity: ``(name, backing-package version)``.

    Read off the provider object `memory_provider_from_env` actually returned, never off
    `_PROVIDER_VAR` — an env re-read would report what the *introspecting* shell was told,
    not what the agent bound (the `--resolved-config` env-gap class, basecradle-noc#62), and
    a dotted path naming a built-in class would report as "custom" when it is not.

    - **name** — ``sqlite`` or ``mempalace`` for the built-ins (whichever alias *or* dotted
      path selected them: the class is the truth, so every spelling normalizes to the alias),
      else ``module:Class`` — the import path of the custom provider actually bound. A
      *subclass* of a built-in is a custom provider and reports its own path, not the alias. Each
      MemPalace adapter reports the name it carries, the one it was selected by, read off the
      object like the rest: ``mempalace`` for the adapter issue #1011 built, and for the previous
      one either ``mempalace-previous`` (its name since issue #1010) or, built directly with no
      name, ``mempalace``.
    - **version** — the installed version of the **backing distribution** the harness pins for
      that provider — today that means the ``mempalace`` extra — else ``None``. ``None`` for the
      built-in ``sqlite`` store, which ships *inside the harness* (its engine is stdlib
      ``sqlite3``): it has no separately-pinned package, its version already *is* the manifest's
      ``harness_version``, and a second copy would imply a pin that does not exist. ``None`` for
      a custom provider too — the harness does not know which distribution ships someone else's
      class (a module's name is not its distribution's, and the stdlib's index of the two is
      incomplete), and inventing a guess is worse than a null in a field a drift
      audit reads. ``mempalace`` with ``None``, though, is not a shrug but a **defect signal**:
      the provider bound (binding is lazy — the extra is imported only on the first
      `observe`/`context`) while its package is absent, so the agent will lose its palace at the
      first wake.
    """
    name = _provider_name(provider)
    return name, _provider_version(name)


def _provider_name(provider: MemoryProvider) -> str:
    """The bound provider's manifest name (see `describe_memory_provider`)."""
    cls = type(provider)
    if cls is SqliteMemoryProvider:
        return _SQLITE
    # Compare by import path, not `isinstance`: the MemPalace adapter is an optional-extra
    # module we must not import just to name a provider that isn't it (the same laziness the
    # adapter itself keeps around chromadb), and an exact-class check is what makes a
    # *subclass* correctly report as the custom provider it is.
    if f"{cls.__module__}.{cls.__qualname__}" in (_MEMPALACE_CLASS, _MEMPALACE_NAMED_CLASS):
        return provider.name  # the name it was selected by (issue #1010)
    return f"{cls.__module__}:{cls.__qualname__}"


def _provider_version(name: str) -> str | None:
    """The installed version of the distribution the harness pins for `name`, else ``None``.

    A provider with no *harness-pinned* backing package — the built-in sqlite store, any custom
    class — has no version the harness can honestly report (see `describe_memory_provider`), so
    it is ``None`` rather than a guess. A pinned package that is *not installed* is ``None`` too:
    that is the state worth catching, and the provider name beside it says which case it is.
    """
    dist = _PROVIDER_DISTRIBUTIONS.get(name)
    if dist is None:
        return None
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _store_path(home: str | os.PathLike[str] | None) -> Path | None:
    """The SQLite store path for a given home override, or ``None`` for the default."""
    if home is None:
        return None
    return Path(home) / "memory.db"


def _palace_path(home: str | os.PathLike[str] | None) -> Path:
    """The MemPalace palace directory: MemPalace's own env vars, else under the agent's home.

    Falls back to the same per-agent root the SQLite store uses (``$HARNESS_HOME`` else a
    dotdir under ``$HOME``), so a MemPalace agent's memory is private to its OS user the
    same way the default's is.

    **MemPalace's own variables are read first, and reading them is the point** (issue #409). The
    `mempalace` CLI resolves its palace as ``--palace`` → `PALACE_PATH_VARS`, in order → its config
    file's ``palace_path`` → a ``palace`` directory beside that file, and the harness *publishes* the path it binds into that config file (`_mempalace.publish_palace_binding`)
    so a bare ``mempalace status`` reaches the agent's live palace. A published file sits **below**
    the env in that order — so if the adapter ignored the env, the day anyone exported it (in an
    `agent.env`, in a wake runner) the CLI would silently follow the env, the harness would silently
    follow ``$HARNESS_HOME``, and the two would be looking at different minds again with the
    publication no longer able to say so. Reading the same vars, in upstream's own order, is what
    makes the two agree in *every* configuration rather than only in the unset one.

    Matched to upstream deliberately, degenerate cases included: a set-but-empty variable falls
    through to the next one (``or``, not "is set"), and the value is ``expanduser`` +
    ``abspath``-ed exactly as `MempalaceConfig.palace_path` does. The
    one case the two *cannot* be made to agree on is a **relative** value, which upstream resolves
    against the process's cwd — the wake's and the operator's shell's are not the same directory.
    Set an absolute path.
    """
    explicit = next((value for var in PALACE_PATH_VARS if (value := os.environ.get(var))), None)
    if explicit:
        return Path(os.path.abspath(os.path.expanduser(explicit)))
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
            f"Use 'sqlite', 'mempalace', 'mempalace-previous', or a dotted import path."
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
