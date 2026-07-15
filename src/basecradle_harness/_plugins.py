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
them apart is what lets a locked and an unlocked Harness profile share this exact resolver.

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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from basecradle_harness._install import _read_manifest, config_home, plugin_relevant_to
from basecradle_harness._tools import Tool

if TYPE_CHECKING:  # type-only: avoids importing the MCP subsystem at plugin-load time
    from basecradle_harness._mcp import McpImageStore

_log = logging.getLogger("basecradle_harness")

# The package subtree the default plugin files live in, under `_defaults/`. The installer
# copies these into the config home's `tools/`; this is also the not-yet-installed fallback.
_DEFAULTS_TOOLS = ("_defaults", "tools")


# --- activation requirements --------------------------------------------------


@dataclass(frozen=True)
class ActivationContext:
    """The active config a plugin's `requires` are checked against.

    The config model is three independent axes (issue #158): the **provider** (whose endpoint
    + key), the **vendor SDK** the harness goes through, and the **model**. A plugin's
    requirements gate on the ``(provider, sdk[, model])`` pairing — a server-side built-in or a
    provider-coupled tool loads only under the pairing it actually works with.

    Args:
        provider: ``AI_PROVIDER`` — the vendor whose endpoint/key the agent uses
            (``"openai"`` | ``"xai"`` | ``"openrouter"``). What a `Vendor` requirement matches.
        sdk: ``AI_SDK`` — the PyPI package the harness imports to reach the model
            (``"openai"`` | ``"xai-sdk"`` | …). Reserved for SDK-specific gating.
        surface: The OpenAI adapter's internal wire surface — ``"responses"`` (default) or
            ``"chat"``. What an `OpenAISurface` requirement matches (the ``web_search`` built-in
            needs ``responses``).
        model: ``AI_MODEL`` — the model id, for the rare model-specific gate.
        env: An environment snapshot (usually ``os.environ``) — what an `EnvSet` / `OpenAIKey`
            requirement reads. Passed in rather than read globally so resolution is pure and a
            test can drive it without touching the process environment.
    """

    provider: str
    sdk: str
    surface: str
    model: str
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
class Vendor(Requirement):
    """Met iff the active provider equals `name` (``"openai"`` | ``"xai"`` | ``"openrouter"``).

    The requirement a provider-coupled tool declares: the grok media tools and xAI Live Search
    require ``Vendor("xai")`` and self-exclude everywhere else, so an xAI agent's stack touches
    no OpenAI surface and an OpenAI agent never sees a grok-only tool.
    """

    name: str

    def met(self, ctx: ActivationContext) -> bool:
        return ctx.provider == self.name

    @property
    def reason(self) -> str:
        return f"needs the {self.name!r} provider (AI_PROVIDER={self.name})"


@dataclass(frozen=True)
class Sdk(Requirement):
    """Met iff the active vendor SDK equals `name` (``AI_SDK`` — the package the harness imports).

    The `Vendor`/`OpenAISurface` pair gate on *whose endpoint* and *which wire surface*; this
    gates on *which SDK adapter* is actually running — the axis `ActivationContext.sdk` was
    reserved for. It is what a built-in whose wiring lives in one specific adapter declares, so it
    stays inert on a sibling cell that reaches the same provider through a different SDK. The
    OpenRouter ``web_search`` server tool is the shipped case: its request wiring lives in the
    native `OpenRouterProvider` (``AI_SDK=openrouter``), so it requires ``Sdk("openrouter")`` on
    top of ``Vendor("openrouter")`` — under the openai-SDK-at-OpenRouter cell (chat-only, which
    ships no server-side built-ins) it self-excludes rather than activating as a present-but-inert
    tool the model would see but nothing would wire.
    """

    name: str

    def met(self, ctx: ActivationContext) -> bool:
        return ctx.sdk == self.name

    @property
    def reason(self) -> str:
        return f"needs the {self.name!r} SDK (AI_SDK={self.name})"


@dataclass(frozen=True)
class OpenAISurface(Requirement):
    """Met iff the OpenAI adapter's wire surface equals `surface` (``"responses"`` | ``"chat"``).

    The requirement OpenAI's server-side ``web_search`` built-in declares (alongside
    ``Vendor("openai")``): it exists only on the Responses surface, so it self-excludes under
    Chat Completions.
    """

    surface: str

    def met(self, ctx: ActivationContext) -> bool:
        return ctx.surface == self.surface

    @property
    def reason(self) -> str:
        return f"needs the OpenAI {self.surface!r} surface"


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
    """Met iff the provider is ``openai`` **and** an API key (``AI_API_KEY``) is present.

    The activation requirement for the OpenAI-coupled tools (``generate_image``,
    ``edit_image``, ``listen``): they call OpenAI's Images/Audio endpoints through the
    ``openai`` SDK with the agent's key, so they belong to the ``openai`` provider and need its
    key set. Under any other provider (xAI, OpenRouter) they self-exclude — an xAI agent's
    media stack is the grok tools instead, touching no OpenAI surface.

    It is an `EnvSet` (inheriting the env-presence check) on the right default var, with a
    plugin-author-friendly name and reason — so a default plugin reads ``requires=(OpenAIKey(),)``.
    """

    var: str = "AI_API_KEY"

    def met(self, ctx: ActivationContext) -> bool:
        return super().met(ctx) and ctx.provider == "openai"

    @property
    def reason(self) -> str:
        return "needs the openai provider (AI_PROVIDER=openai) and an API key (AI_API_KEY)"


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

    `opt_in` marks a **powerful/dangerous** tool — media generation (image, video, audio),
    web/X search, code execution — that must **fail closed**: it is **off by default on every
    provider** (issue #168) and activates **only** when the operator explicitly drops it into a
    persona's ``tools/`` overlay (the same "ships empty" pattern as ``mcp/``). An opt-in plugin
    is **not** auto-loaded from the packaged defaults and **not** auto-scaffolded by the
    installer; a benign/platform plugin (the default, ``opt_in=False``) keeps the normal
    shipped-default → install-then-prune behavior. This is a **capability** classification,
    **provider-agnostic** — the `requires` gate (`Vendor`/`OpenAIKey`) decides *availability*,
    never the safety default. Detected from source (AST) by `_install.plugin_opts_in` so the
    installer and loader agree without importing the plugin.
    """

    impl: type[Tool] | None = None
    builtin: str | None = None
    requires: tuple[Requirement, ...] = ()
    name: str | None = None
    note: str | None = None
    opt_in: bool = False
    stem: str | None = None
    """The source file's stem (``code_execution.py`` → ``code_execution``), stamped by the
    loader (`_plugins_in_file`) — **not** authored in the plugin file. It is the unit the
    fleet inventory keys an opt-in tool on, and is **not** the same as `resolved_name`: one
    stem can fan out to several names (``code_execution`` → the ``code_interpreter`` built-in
    **+** the ``code_attach`` tool) and a name can differ from its stem (``hear_audio`` →
    ``listen``). Reported (for active opt-in plugins) by `resolved_config` so the NOC's
    fleet-drift audit compares declared inventory stems like-for-like, holding no local
    stem→name map of its own (issue #181). ``None`` for a plugin built directly via the API
    (a test), never loaded from a file."""

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
        broken: Defect lines for **shipped-default** plugins that failed to load (issue #160)
            — distinct from `notices` (an intentional opt-out) and from `skipped` (a normal,
            expected activation miss). A broken default is a defect: the brief renders these
            under their own loud heading so a silently-disabled capability is impossible to
            miss. Empty when every shipped default loaded.
        opt_in_stems: The sorted source-file **stems** of the **active opt-in** plugins (issue
            #181) — the unit the fleet inventory keys a powerful tool on. "Active" here is the
            *activation* gate (the plugin's `requires`: provider/key/surface), which is the axis
            the inventory keys on; it is deliberately **not** narrowed by the separate safety axes
            surfaced in `notices` — the locked **policy** gate *and* the runtime-veto gate
            (`Tool.load_refusal`, e.g. ``shell`` refusing to run as root). (Most shipped opt-in tools
            declare only activation requirements; ``shell`` is the exception — opt-in **and**
            ``requires={SHELL}`` **and** root-refusing, so on a locked agent that dropped it in it
            lists its stem here — correctly, as an opted-in inventory item — while its policy refusal
            (or, as root, its runtime refusal) shows up in `notices`; on the intended unlocked,
            non-root agent every gate passes.) One stem appears once even when it
            fans out to several active names (``code_execution`` → the ``code_interpreter``
            built-in **+** the ``code_attach`` tool). `resolved_config` surfaces this so the
            NOC's fleet-drift audit compares declared-vs-active stems like-for-like, holding no
            stem→name map of its own. Empty for a config with no opt-in tool active (the safe
            default).
        mcp_images: The per-wake MCP image store (issue #318), set only when an MCP server's tools
            loaded. It rides the resolved set from `_merge_mcp_tools` to the hosting agent, which
            threads it into the `PlatformContext` so the assets ``post_image`` action can post an
            image an MCP tool returned. ``None`` for any config with no active MCP image source.
    """

    tools: list[Tool] = field(default_factory=list)
    builtins: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    manifest: list[tuple[str, str | None]] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)
    opt_in_stems: list[str] = field(default_factory=list)
    #: The per-wake MCP image store (issue #318), set only when an MCP server's tools loaded. It
    #: rides the resolved set from `_merge_mcp_tools` to the hosting agent, which threads it into
    #: the `PlatformContext` so the assets ``post_image`` action can post a returned image. ``None``
    #: for any config with no active MCP image source (the common case).
    mcp_images: McpImageStore | None = None


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
    # The active opt-in *stems* (issue #181): one per file-loaded opt-in plugin that activated,
    # deduped (a stem fanning out to several active names lists once) and sorted. A stem-less
    # opt-in plugin (built directly via the API, never loaded from a file) has no inventory key
    # to report, so it is omitted rather than guessed from its name.
    opt_in_stems = sorted(
        {plugin.stem for plugin in claimed.values() if plugin.opt_in and plugin.stem}
    )
    return ResolvedTools(
        tools=tools,
        builtins=builtins,
        skipped=skipped,
        manifest=manifest,
        opt_in_stems=opt_in_stems,
    )


# --- loading plugin files -----------------------------------------------------


@dataclass(frozen=True)
class LoadedPlugins:
    """The result of loading the tool-plugin files: the good plugins, plus broken *defaults*.

    `plugins` is the loadable set (the overlay's, else the packaged defaults'). `broken_defaults`
    is ``(filename, error)`` for every **shipped-default** plugin file that failed to import or
    declared no plugin — a *defect*, not a normal skip. The constitution forbids a silent
    swallow here ("a tool an AI uses is built whole … never a silent swallow"), so a broken
    default is surfaced loudly: logged at ``ERROR`` and rendered into the Turn-0 brief
    (`_basecradle._surface_broken_defaults`), never vanished. A broken *operator-added* file
    stays a soft skip — one bad drop-in must not take the agent down — so it is logged at
    ``WARNING`` and left out of `broken_defaults`.
    """

    plugins: list[ToolPlugin] = field(default_factory=list)
    broken_defaults: list[tuple[str, str]] = field(default_factory=list)


def load_plugins(
    home: str | Path | None = None, *, provider: str | None = None
) -> list[ToolPlugin]:
    """The loadable tool plugins for an agent — `load_plugins_report` without the defect list.

    Kept as the simple accessor most callers want; see `load_plugins_report` for the source of
    record, provider-aware loading, and the broken-default reporting.
    """
    return load_plugins_report(home, provider=provider).plugins


def load_plugins_report(
    home: str | Path | None = None, *, provider: str | None = None
) -> LoadedPlugins:
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

    ``provider`` makes the load **provider-aware** (issue #160): when named (the resolver passes
    the active ``AI_PROVIDER``), a plugin file whose source declares affinity for a *different*
    provider is **not imported** — its provider-mismatched tool would be gated off at activation
    anyway, and skipping it pre-import sidesteps the latent hazard of importing a vendor SDK the
    agent never installed (which would otherwise be the very silent-import-skip this guards). The
    relevance check is source-only (AST, no execution — `_install.plugin_relevant_to`), so a
    *broken* file is never hidden by it: it still attempts to import and surfaces as a defect.
    ``None`` (the direct API / test default) imports every file, unfiltered.

    Returns a `LoadedPlugins`: the loadable plugins, plus any **shipped-default** file that
    failed to load (a defect to surface loudly — see `LoadedPlugins`). A broken file is a
    broken *default* when its filename is one of the packaged ``_defaults/tools/`` names
    (always true on the fallback path, where every file *is* a packaged default; true on the
    overlay path only when the operator's broken file shadows a shipped default's name).
    """
    root = config_home(home)
    tools_dir = root / "tools"
    tools_installed = any(key.startswith("tools/") for key in _read_manifest(root))
    if tools_installed:
        # Installed → the overlay is authoritative; a removed dir/files is the operator's
        # deletion, honored (zero tools), never resurrected from the packaged defaults. A
        # powerful (`opt_in`) plugin *present here* is an explicit per-persona opt-in — kept.
        plugins, broken = _load_dir(tools_dir, provider) if tools_dir.is_dir() else ([], [])
    else:
        # Not yet installed for tools → load the packaged defaults straight from the package,
        # but **drop the opt-in (powerful) defaults**: they fail closed and activate only when
        # explicitly dropped into a persona's overlay (issue #168), never from the packaged
        # fallback. A default-riding persona thus resolves to benign/platform tools only.
        with resources.as_file(
            resources.files("basecradle_harness").joinpath(*_DEFAULTS_TOOLS)
        ) as p:
            plugins, broken = _load_dir(Path(p), provider)
        plugins = [plugin for plugin in plugins if not plugin.opt_in]
    return _classify_broken(plugins, broken)


def _classify_broken(plugins: list[ToolPlugin], broken: list[tuple[str, str]]) -> LoadedPlugins:
    """Split broken files into loud shipped-default defects (ERROR) and soft operator skips.

    The single classification point so the log level and the brief surfacing agree on what
    counts as a defect. A broken file whose name is a packaged default is a defect — the
    shipped tranche is broken, so it is logged at ``ERROR`` and returned in `broken_defaults`
    for the Turn-0 brief. A broken operator file is a soft ``WARNING`` skip, dropped here.
    """
    default_names = _default_tool_filenames()
    broken_defaults: list[tuple[str, str]] = []
    for name, exc in broken:
        if name in default_names:
            _log.error(
                "Shipped tool plugin %s failed to load: %s. This is a defect — the capability "
                "is disabled until it is fixed (run basecradle-harness-install to refresh a "
                "stale overlay, or repair the file).",
                name,
                exc,
            )
            broken_defaults.append((name, exc))
        else:
            _log.warning("Skipping tool plugin file %s: %s", name, exc)
    return LoadedPlugins(plugins=plugins, broken_defaults=broken_defaults)


def _default_tool_filenames() -> frozenset[str]:
    """The basenames of the packaged ``_defaults/tools/*.py`` files — the shipped-default set."""
    root = resources.files("basecradle_harness").joinpath(*_DEFAULTS_TOOLS)
    return frozenset(
        child.name for child in root.iterdir() if child.name.endswith(".py") and child.is_file()
    )


def _load_dir(
    directory: Path, provider: str | None = None
) -> tuple[list[ToolPlugin], list[tuple[str, str]]]:
    """Load the ``*.py`` plugin files in `directory`: the declared plugins, and the broken ones.

    Only ``*.py`` is loaded (so the upgrader's ``*.py.new`` shadow files are ignored). When
    `provider` is named, a file whose source declares affinity for a *different* provider is
    skipped **before import** (issue #160) — read as text and AST-checked, never executed, so a
    foreign plugin's vendor-SDK import is never triggered. A file that fails to import or declares
    no plugin is collected into the broken list (with its error) for the caller to classify and
    log — `_classify_broken` decides whether it is a loud shipped-default defect or a soft
    operator skip. Either way one broken file never takes the agent down: the loadable plugins are
    still returned.
    """
    plugins: list[ToolPlugin] = []
    broken: list[tuple[str, str]] = []
    for path in sorted(directory.glob("*.py")):
        if provider is not None and not _relevant(path, provider):
            continue  # provider-mismatched → don't even import it
        try:
            plugins.extend(_plugins_in_file(path))
        except Exception as exc:  # noqa: BLE001 - a bad file is collected, never fatal
            broken.append((path.name, str(exc)))
    return plugins, broken


def _relevant(path: Path, provider: str) -> bool:
    """Whether a plugin file is relevant to `provider`, read from source without importing it.

    Reads the file as text and defers to `_install.plugin_relevant_to` (AST, no execution). An
    unreadable file is treated as relevant — let the import attempt surface the real error rather
    than silently skipping it on an IO hiccup.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return True
    return plugin_relevant_to(source, provider)


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
    # Stamp the source-file stem onto each plugin (the file is ground truth, overriding any
    # value an author set), so resolution can report the active opt-in *stems* — the unit the
    # fleet inventory keys on — without the loader's filename being lost downstream (issue #181).
    return [replace(item, stem=path.stem) for item in found]


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
