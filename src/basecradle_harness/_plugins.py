"""The tool plugin framework: ``(name + requires + impl)``, resolved against the config.

A tool is no longer a hardcoded registry entry — it is a **plugin**, a tiny declaration
of three things:

- ``impl`` — the tool class to instantiate (today's `Tool`), **or** a ``builtin`` wire name
  for a server-side tool the provider runs (e.g. OpenAI's ``web_search``);
- ``requires`` — the *activation* preconditions: what the active **config** must provide for
  the plugin to be usable (a provider API, an API key). A plugin whose requirements aren't
  met **does not register** — the model never sees a present-but-broken tool;
- ``name`` — the model-facing identifier, defaulting to the impl's ``name`` (or the builtin
  wire name).

Two gates, kept deliberately separate
-------------------------------------
This is *activation*, and it is **not** the policy/safety gate. They are orthogonal axes:

- **Activation** (here) asks *is this usable under the active config?* — provider, keys.
- **Policy** (`_policy.py`) asks *is this capability allowed by the profile?* — e.g. ``SHELL``
  under the locked profile. Enforced later, at `ToolRegistry.register`.

A plugin can be active yet still policy-refused; both gates apply, activation first. Keeping
them apart is what lets a locked Harness and an unlocked Cradle share this exact resolver.

Sources: package defaults + the ``/tools`` overlay
--------------------------------------------------
Plugins are real ``*.py`` files (see `load_plugins`): the package ships defaults under
``_defaults/tools/``, the installer copies them into the config home's ``tools/`` dir, and
that dir is the operator's overlay — **add** a file (new tool), **override** a default by
reusing its ``name``, **disable** a default by deleting its file. `resolve_plugins` settles
the active set: later-loaded active plugins win a name (overlay precedence), and when two
plugins share a name with different ``requires`` (a Responses ``web_search`` vs. a future
xAI one), exactly one activates per config — the one whose requirements the config meets.
"""

from __future__ import annotations

import hashlib
import importlib.resources as resources
import importlib.util
import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from basecradle_harness._install import _read_manifest, config_home
from basecradle_harness._tools import Tool

_log = logging.getLogger("basecradle_harness")

# The package subtree the default plugin files live in, under `_defaults/`. The installer
# copies these into the config home's `tools/`; this is also the not-yet-installed fallback.
_DEFAULTS_TOOLS = ("_defaults", "tools")


# --- activation requirements --------------------------------------------------


@dataclass(frozen=True)
class ActivationContext:
    """The active config a plugin's `requires` are checked against.

    Args:
        provider_api: The selected provider API — ``"chat"``, ``"responses"``, or ``"xai"``
            (the value of ``AI_PROVIDER_API``). What a `ProviderAPI` requirement matches on.
        env: An environment snapshot (usually ``os.environ``) — what an `EnvSet` / `OpenAIKey`
            requirement reads. Passed in rather than read globally so resolution is pure and
            a test can drive it without touching the process environment.
    """

    provider_api: str
    env: Mapping[str, str]


class Requirement(ABC):
    """One activation precondition: is a plugin usable under the active config?

    A `Requirement` is a small, equatable value (subclasses are frozen dataclasses) with two
    parts: `met` decides activation; `reason` is the one-line human string logged when a
    plugin is skipped, so an operator can see *why* a tool isn't present.
    """

    @abstractmethod
    def met(self, ctx: ActivationContext) -> bool:
        """True if this precondition holds under `ctx`."""

    @property
    @abstractmethod
    def reason(self) -> str:
        """A short phrase naming what this requirement needs, for the skip log."""


@dataclass(frozen=True)
class ProviderAPI(Requirement):
    """Met iff the active provider API equals `api` (``"chat"`` or ``"responses"``).

    The requirement a Responses-only built-in declares: ``web_search`` requires
    ``ProviderAPI("responses")`` and so self-excludes under Chat Completions.
    """

    api: str

    def met(self, ctx: ActivationContext) -> bool:
        return ctx.provider_api == self.api

    @property
    def reason(self) -> str:
        return f"needs the {self.api!r} provider API"


@dataclass(frozen=True)
class EnvSet(Requirement):
    """Met iff environment variable `var` is set to a non-empty value."""

    var: str

    def met(self, ctx: ActivationContext) -> bool:
        return bool(ctx.env.get(self.var))

    @property
    def reason(self) -> str:
        return f"needs {self.var} set"


@dataclass(frozen=True)
class OpenAIKey(EnvSet):
    """Met iff an OpenAI API key (``AI_PROVIDER_API_KEY``) is present **and** this isn't xAI.

    The honest activation requirement for the OpenAI-coupled tools (``generate_image``,
    ``listen``): they call OpenAI's Images/Audio APIs with the agent's key under the ``chat``
    or ``responses`` provider, so what they truly need is that key — not a particular adapter.
    Whether the key behind the shared var is genuinely an *OpenAI* key can't be told in
    general; key-present is the strongest non-fragile proxy. The **one** case we *can* tell is
    the xAI-native profile (``AI_PROVIDER_API=xai``): there the key is an xAI key and these
    OpenAI tools must **not** activate (an xAI agent's stack touches no OpenAI surface — the
    grok media tools cover it instead), so the profile is excluded here by construction rather
    than left to the operator to curate.

    It is an `EnvSet` (inheriting the env-presence check) with the right default var and a
    plugin-author-friendly name and reason — so a default plugin reads ``requires=(OpenAIKey(),)``.
    """

    var: str = "AI_PROVIDER_API_KEY"

    def met(self, ctx: ActivationContext) -> bool:
        return super().met(ctx) and ctx.provider_api != "xai"

    @property
    def reason(self) -> str:
        return "needs an OpenAI API key (AI_PROVIDER_API_KEY) and a non-xAI provider"


# --- the plugin ---------------------------------------------------------------


@dataclass(frozen=True)
class ToolPlugin:
    """A tool declaration: ``(name + requires + impl)``.

    Exactly one of `impl` / `builtin` is set:

    - `impl` — a `Tool` subclass; when the plugin activates it is instantiated and registered
      as a function tool the engine runs.
    - `builtin` — a server-side built-in's wire name (e.g. ``"web_search"``); when the plugin
      activates the name is handed to the provider, which runs the tool itself.

    `requires` lists the activation preconditions (all must hold). `name` defaults to the
    impl's ``name`` (or the builtin wire name); set it only to override.

    `note` is an optional one-line gotcha the function schema cannot convey (e.g. lock's
    irreversibility). It is rendered into the generated tool manifest (the persistent Turn-0
    brief) beside the tool's name; a plugin without one just lists its name. Additive — an
    existing plugin that sets no `note` is unaffected.
    """

    impl: type[Tool] | None = None
    builtin: str | None = None
    requires: tuple[Requirement, ...] = ()
    name: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if (self.impl is None) == (self.builtin is None):
            raise ValueError("A ToolPlugin sets exactly one of `impl` or `builtin`.")

    @property
    def resolved_name(self) -> str:
        """The model-facing name: explicit `name`, else the impl's `name`, else the builtin."""
        if self.name:
            return self.name
        if self.impl is not None:
            impl_name = getattr(self.impl, "name", None)
            if not impl_name:
                raise ValueError(f"{self.impl.__name__} has no `name`; the plugin must set one.")
            return impl_name
        assert self.builtin is not None  # guaranteed by __post_init__
        return self.builtin

    @property
    def is_builtin(self) -> bool:
        return self.builtin is not None

    def active(self, ctx: ActivationContext) -> bool:
        """True iff every activation requirement is met under `ctx`."""
        return all(req.met(ctx) for req in self.requires)

    def unmet(self, ctx: ActivationContext) -> str:
        """A one-line reason a plugin is inactive (the unmet requirements), for the skip log."""
        return "; ".join(req.reason for req in self.requires if not req.met(ctx))


# --- resolution ---------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedTools:
    """The active tool set for one config: instantiated function tools + built-in names.

    Args:
        tools: The instantiated, active function tools, in resolution order — handed to the
            `Harness`/`ToolRegistry` (which still applies the policy gate on top).
        builtins: The active server-side built-in wire names — handed to the provider (e.g.
            the Responses adapter's ``builtin_tools``).
        skipped: ``(name, reason)`` for every plugin that did **not** activate, for logging —
            the visible "why isn't this tool here?" trail.
        manifest: ``(name, note)`` for every **active** tool — function tools and built-ins
            alike, in resolution order — the source the persistent Turn-0 brief renders into
            its "Your active tools right now" block. ``note`` is the plugin's optional gotcha,
            or ``None``. Always matches the active provider + drop-ins, so it can never drift.
        notices: Safe-by-default opt-out lines surfaced in the Turn-0 brief — one per active
            MCP server, plus any tool refused by the locked policy (Group 5, Part B). Empty
            for a pure-Harness config; populated only when the operator has knowingly left
            the safe zone, so "all bets off" is stated and auditable, never silent.
    """

    tools: list[Tool] = field(default_factory=list)
    builtins: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    manifest: list[tuple[str, str | None]] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)


def resolve_plugins(plugins: Iterable[ToolPlugin], ctx: ActivationContext) -> ResolvedTools:
    """Settle a list of plugins into the active tool set under `ctx`.

    One ordered pass builds a ``name → winning plugin`` map: each *active* plugin claims its
    name, and a later active plugin overrides an earlier one (overlay precedence). An
    *inactive* plugin never claims a name — so when two plugins share a name but only one's
    `requires` are met, that one wins ("exactly one activates per config"); a name claimed by
    no active plugin simply yields no tool. Claimed plugins are then split into instantiated
    function tools and built-in names.

    Inactive plugins are recorded in `skipped` with their unmet reason and logged once, so an
    OpenAI-coupled tool dropping under the wrong provider is *visible*, not silent.
    """
    claimed: dict[str, ToolPlugin] = {}
    skipped: list[tuple[str, str]] = []
    for plugin in plugins:
        name = plugin.resolved_name
        if plugin.active(ctx):
            claimed[name] = plugin  # later active plugin wins the name (overlay precedence)
        else:
            skipped.append((name, plugin.unmet(ctx)))

    for name, reason in skipped:
        if name not in claimed:  # only note a name that ends up with no active provider
            _log.info("Tool plugin %r inactive: %s.", name, reason)

    tools: list[Tool] = []
    builtins: list[str] = []
    manifest: list[tuple[str, str | None]] = []
    for plugin in claimed.values():
        manifest.append((plugin.resolved_name, plugin.note))
        if plugin.is_builtin:
            assert plugin.builtin is not None
            builtins.append(plugin.builtin)
        else:
            assert plugin.impl is not None
            tools.append(plugin.impl())
    return ResolvedTools(tools=tools, builtins=builtins, skipped=skipped, manifest=manifest)


# --- loading plugin files -----------------------------------------------------


def load_plugins(home: str | Path | None = None) -> list[ToolPlugin]:
    """Load the tool plugins for an agent: the ``/tools`` overlay, else packaged defaults.

    The config home's ``tools/`` dir is authoritative **once the installer has populated it**
    — detected by the manifest recording any ``tools/`` file. Then the operator's filesystem
    is the source: add / override / delete all take effect, including deleting every file —
    or the whole ``tools/`` dir — to run no tools (a missing dir once installed means zero
    tools, *not* a resurrection of the defaults). Until the installer has run for tools (a
    config home that predates tool defaults, or none at all) the packaged ``_defaults/tools/``
    load directly, so an un-upgraded or un-installed deployment still comes up with the full
    default set — the same files the installer would copy. This mirrors the charter's
    "files-if-installed, else fallback" precedent.
    """
    root = config_home(home)
    tools_dir = root / "tools"
    tools_installed = any(key.startswith("tools/") for key in _read_manifest(root))
    if tools_installed:
        # Installed → the overlay is authoritative; a removed dir/files is the operator's
        # deletion, honored (zero tools), never resurrected from the packaged defaults.
        return _load_dir(tools_dir) if tools_dir.is_dir() else []
    # Not yet installed for tools → load the packaged defaults straight from the package.
    with resources.as_file(resources.files("basecradle_harness").joinpath(*_DEFAULTS_TOOLS)) as p:
        return _load_dir(Path(p))


def _load_dir(directory: Path) -> list[ToolPlugin]:
    """Every `ToolPlugin` declared by the ``*.py`` files in `directory`, in filename order.

    Only ``*.py`` is loaded (so the upgrader's ``*.py.new`` shadow files are ignored). A file
    that fails to import or declares no plugin is logged and skipped — one broken operator
    file never takes the agent down with it.
    """
    plugins: list[ToolPlugin] = []
    for path in sorted(directory.glob("*.py")):
        try:
            plugins.extend(_plugins_in_file(path))
        except Exception as exc:  # noqa: BLE001 - a bad operator file is skipped, not fatal
            _log.warning("Skipping tool plugin file %s: %s", path.name, exc)
    return plugins


def _plugins_in_file(path: Path) -> list[ToolPlugin]:
    """Import one plugin file and return the `ToolPlugin`(s) it declares.

    A file exposes its plugin as a module-level ``PLUGIN`` (one) and/or ``PLUGINS`` (an
    iterable) — both are honored, so a file may ship more than one tool. A file with neither
    is an error (caught by `_load_dir` and surfaced as a skip).
    """
    module = _import_file(path)
    found: list[ToolPlugin] = []
    one = getattr(module, "PLUGIN", None)
    if one is not None:
        found.append(one)
    many = getattr(module, "PLUGINS", None)
    if many is not None:
        found.extend(many)
    if not found:
        raise ValueError("no module-level PLUGIN or PLUGINS")
    for item in found:
        if not isinstance(item, ToolPlugin):
            raise TypeError(f"PLUGIN/PLUGINS must be ToolPlugin, got {type(item).__name__}")
    return found


def _import_file(path: Path):
    """Import a ``.py`` file by path under a synthetic module name derived from its path.

    The name is the file stem plus a short digest of its absolute path, so two files sharing
    a stem get distinct names, yet re-loading the *same* file (a second `load_plugins` call,
    or another wake) reuses the same name and overwrites in place — the ``sys.modules``
    footprint stays bounded by the number of distinct plugin files, not by how many times
    they are loaded. The module is registered before exec so a dataclass or relative
    reference inside it resolves normally.
    """
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    mod_name = f"basecradle_harness._tools_overlay.{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    # Don't write a `__pycache__/*.pyc` beside the source: plugin files live in the operator's
    # config home (and, for the packaged-default fallback, inside site-packages) — neither
    # should be littered with bytecode by a load, and a stray `.pyc` under `_defaults/` would
    # also trip the installer's text walk (`_packaged_defaults`).
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous
    return module
