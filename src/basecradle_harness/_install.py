"""The config home, its installer, and a conffile-style upgrader.

Everything an operator customizes lives as **real files** under a visible config home —
``<agent-home>/.config/basecradle/`` — never hidden inside ``site-packages`` as a magic
fallback the package reaches for. The package *ships* defaults; the installer *copies*
them out to the config home, where the operator can see and edit them. This module is
that copy-out, plus the upgrade discipline that lets a new release refresh pristine
defaults **without ever clobbering an operator's edits**.

The layout the installer scaffolds::

    <agent-home>/.config/basecradle/
      agent.env            # the operator's env (token, keys) — never created or touched here
      model_params.json    # optional model-call params (temperature, reasoning, …) — operator-owned, never touched here
      prompts/
        system-prompt.md   # shipped default
        initialize.md      # shipped default (starter; the richer default is a later group)
      tools/               # tool-plugin overlay (drop-in *.py) — loaded by `_plugins` (Group 2)
      mcp/                 # MCP server configs (drop-in *.json) — loaded by `_mcp` (Group 5)
      .manifest.json       # bookkeeping: the hash of every shipped default as installed

**The conffile upgrader.** Re-running the installer against a newer package is an
*upgrade*, and the same per-file reconcile (`_reconcile`) drives both: on a first run
every default is fresh and written; on a later run each shipped default is compared,
dpkg-conffile style, against the hash recorded in ``.manifest.json`` at the last install
and against the on-disk file:

- **Untouched** (on-disk matches what we installed) → replace with the new default.
- **User-edited** (on-disk differs from both the old and new default) → keep theirs,
  write the new default beside it as ``<name>.new``, and log one line.
- **User-deleted** (we installed it, it is gone now) → respect it; never resurrect.
- **User-added** (a file that is not a shipped default) → never touched, because the
  reconcile only ever walks the *shipped* default set; it never enumerates the operator's
  directory to prune or judge extras.

The operator's config dir is never clobbered; only pristine defaults refresh. This
per-agent reconcile is exactly what a fleet rollout loops over a pinned version.

**Boundary.** This module owns *where things live and how install/upgrade works* — it
scaffolds the ``tools/`` and ``mcp/`` overlay dirs and reconciles the shipped defaults,
but the loading of those overlays lives elsewhere (`_plugins` for ``tools/``, `_mcp` for
``mcp/``), as does prompt composition beyond sourcing the charter from files
(`charter_from_config`). ``mcp/`` ships **empty** (safe by default), so it has no shipped
defaults for the upgrader to reconcile and an operator-added server file is never touched.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.resources as resources
import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from basecradle_harness._version import __version__

_log = logging.getLogger("basecradle_harness")

# The directories the installer scaffolds under the config home. `prompts/` holds the
# shipped charter defaults; `tools/` and `mcp/` are created empty this group (loading
# from them is a later group), so an operator already sees where those will go.
_SCAFFOLD_DIRS = ("prompts", "tools", "mcp")

# Bookkeeping, not an operator file: the hash of every shipped default as it was last
# installed, so the upgrader can tell a pristine default from an edited one. Leading dot
# signals "tooling — do not edit"; it is never composed into anything the model reads.
_MANIFEST_NAME = ".manifest.json"

# Bookkeeping, not an operator file: the harness version that last reconciled this config
# home, so a wake can tell `pip install -U` happened and refresh the overlay before loading
# it. Leading dot signals "tooling — do not edit"; it is never composed into anything the
# model reads. A package upgrade bumps the package but does NOT touch this materialized
# config home, so without this stamp a stale `tools/` overlay (a default plugin from the
# previous version) silently outlives the upgrade — exactly the issue #160 failure.
_VERSION_NAME = ".version"

# The package subtree the shipped defaults live in. The installer copies these out to the
# config home; it is the *only* place defaults exist — there is no magic in-package
# fallback the runtime reads at rest.
_DEFAULTS_PACKAGE = "_defaults"

# HTML comments in a charter file carry operator-facing notes (how to edit it, that an
# upgrade won't clobber it) — guidance for the human, not the model. They are stripped
# before the file is composed into Turn 0, so a file can document itself without spending
# the model's context on it.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def config_home(home: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the config home: explicit arg → ``BASECRADLE_CONFIG_HOME`` → ``$HOME/.config/basecradle``.

    A single resolver so the installer (which writes here) and the runtime (which reads
    the charter here) never disagree on the path. ``BASECRADLE_CONFIG_HOME`` is the
    override an operator or a test sets to point the config home somewhere other than the
    default under the OS user's home.
    """
    if home is not None:
        return Path(home).expanduser()
    override = os.environ.get("BASECRADLE_CONFIG_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "basecradle"


# --- provider affinity (issue #160) -------------------------------------------

# The active provider when none is named — mirrors `_basecradle.DEFAULT_PROVIDER`. Inlined as a
# string (not imported) because `_basecradle` imports this module: the dependency runs one way.
_DEFAULT_PROVIDER = "openai"

# The `requires` marker calls (from `_plugins`) that declare a plugin's provider affinity in its
# source. `Vendor("x")` ties a plugin to provider ``x``; `OpenAIKey()`/`OpenAISurface()` tie it to
# ``openai``. Matched by *name in the source's AST* — so a provider-mismatched plugin is classified
# WITHOUT importing it, which is the point: importing a foreign plugin could trip an import of a
# vendor SDK the agent never installed (the very silent-import-skip this issue is about). Kept as
# bare strings to avoid importing `_plugins`, which imports this module.
_VENDOR_MARKER = "Vendor"
_OPENAI_MARKERS = ("OpenAIKey", "OpenAISurface")


def plugin_source_providers(source: str) -> frozenset[str] | None:
    """The providers a tool-plugin's *source* declares affinity for, or ``None`` if universal.

    Parses the source with `ast` and **never executes it**, so a plugin is classified without
    importing it (and without triggering any vendor-SDK import it does at module load). A
    ``Vendor("x")`` marker contributes provider ``x``; an ``OpenAIKey()``/``OpenAISurface()``
    marker contributes ``openai``. No markers → ``None`` (provider-agnostic — relevant to every
    agent). Unparseable source → ``None`` too: it is treated as universal so the *loader* still
    attempts it and surfaces the real syntax error as a defect, rather than the affinity check
    hiding a broken default.

    The markers are matched by their *plain call name* in the source — the shipped defaults all
    use ``Vendor``/``OpenAIKey``/``OpenAISurface`` directly (never aliased), which this is built
    for. An operator file that imports a marker under an alias reads as universal (no affinity
    detected) and so is loaded everywhere — a safe degrade (the resolver's real `requires` gate
    still deactivates it off its provider), never a wrong exclusion.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    providers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == _VENDOR_MARKER and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                providers.add(arg.value)
        elif name in _OPENAI_MARKERS:
            providers.add("openai")
    return frozenset(providers) or None


def plugin_relevant_to(source: str, provider: str | None) -> bool:
    """Whether a tool plugin (by its source) is relevant to the active provider.

    ``provider=None`` means "don't filter" — every plugin is relevant (the unfiltered default the
    direct `install`/`load_plugins` API and the tests use). With a provider named, a plugin is
    relevant iff it declares no provider affinity (universal) or names this provider. This is the
    single predicate both the installer (which lays down only relevant defaults) and the loader
    (which imports only relevant files) apply, so the two never disagree.
    """
    if provider is None:
        return True
    affinity = plugin_source_providers(source)
    return affinity is None or provider in affinity


def plugin_opts_in(source: str) -> bool:
    """Whether a tool-plugin's *source* declares it a powerful, opt-in tool (``opt_in=True``).

    Parses the source with `ast` and **never executes it** — the same no-import discipline as
    `plugin_source_providers`, so the installer can classify a default as opt-in without
    importing it (and the loader and installer agree on the bucket). An ``opt_in=True`` keyword
    in *any* ``ToolPlugin(...)`` call in the file marks the file's tools as powerful (issue
    #168): off by default on every provider, scaffolded/loaded only on explicit opt-in. A file
    with no such keyword (or unparseable source — treated as benign so the loader still attempts
    it and surfaces a real error) is a benign default with the normal install-then-prune
    behavior.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "opt_in" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return True
    return False


def active_provider_from_env() -> str:
    """The active ``AI_PROVIDER`` for provider-aware install/reconcile, normalized (default openai)."""
    return (os.environ.get("AI_PROVIDER") or _DEFAULT_PROVIDER).strip().lower()


# --- the shipped defaults -----------------------------------------------------


def _packaged_defaults() -> dict[str, str]:
    """Every shipped default file as ``{config-home-relative-path: text}``.

    Walks the ``_defaults`` subtree inside the installed package and returns each file's
    text keyed by its path *relative to the config home* (forward-slash separated, so the
    keys are portable and stable across platforms — they are also the manifest keys). This
    is the authoritative default set the installer copies out and the upgrader reconciles
    against; there is no other source of a default.
    """
    root = resources.files("basecradle_harness").joinpath(_DEFAULTS_PACKAGE)
    out: dict[str, str] = {}

    def walk(node, prefix: str) -> None:  # node is an importlib.resources Traversable
        for child in node.iterdir():
            # Skip Python's bytecode cache: the shipped tool-plugin defaults are real `.py`
            # files, so importing them (or a stray editable-install artifact) can leave a
            # `__pycache__/*.pyc` here — binary, not a default, and not utf-8 text.
            if child.name == "__pycache__":
                continue
            rel = f"{prefix}{child.name}"
            if child.is_dir():
                walk(child, f"{rel}/")
            else:
                out[rel] = child.read_text(encoding="utf-8")

    walk(root, "")
    return out


def _hash(text: str) -> str:
    """The sha256 of a default file's text — the manifest's unit of comparison."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- install / upgrade reports ------------------------------------------------

# Per-file outcomes of a reconcile, for the CLI summary and for tests to assert on.
INSTALLED = "installed"  # fresh: never installed before → written
REFRESHED = "refreshed"  # pristine default, shipped default changed → replaced
KEPT_EDITED = "kept-edited"  # operator edited it → kept theirs, new default left as .new
KEPT_DELETED = "kept-deleted"  # operator deleted it → respected, not resurrected
ALREADY_CURRENT = "already-current"  # on-disk already equals the new default → nothing to do
UNCHANGED = "unchanged"  # shipped default unchanged since last install → left as-is
PRUNED = (
    "pruned"  # a previously-installed default is now provider-mismatched → removed (issue #160)
)


@dataclass
class InstallReport:
    """What a reconcile did: the config home, the dirs it ensured, and per-file outcomes."""

    config_home: Path
    created_dirs: list[str] = field(default_factory=list)
    actions: dict[str, str] = field(default_factory=dict)
    new_files: list[str] = field(default_factory=list)  # the `.new` files written this run
    # Power tools (now opt-in, issue #168) a *prior* version had scaffolded into this config home
    # and that the upgrade KEPT rather than silently strip — the grandfather list, surfaced loudly.
    grandfathered: list[str] = field(default_factory=list)

    def of(self, action: str) -> list[str]:
        """The relative paths whose outcome was ``action`` (for tests and the summary)."""
        return [rel for rel, got in sorted(self.actions.items()) if got == action]

    def summary(self) -> str:
        """A short human summary for the CLI — one line of counts, then any ``.new`` notes."""
        counts: dict[str, int] = {}
        for action in self.actions.values():
            counts[action] = counts.get(action, 0) + 1
        head = f"Config home: {self.config_home}"
        tally = ", ".join(f"{action}: {n}" for action, n in sorted(counts.items())) or "no defaults"
        lines = [head, tally]
        for rel in self.new_files:
            lines.append(f"  kept your edited {rel} — new default written to {rel}.new")
        if self.grandfathered:
            # Loud, never silent: an existing config keeps power tools that are now opt-in by
            # default (issue #168). Name each so the operator sees the policy change and can
            # delete any it no longer wants. New installs get the opt-in (off) default.
            kept = ", ".join(self.grandfathered)
            lines.append(
                f"  kept (now opt-in, grandfathered from a prior install): {kept}. "
                "These powerful tools are off by default for new agents; delete a file to drop it."
            )
        return "\n".join(lines)


# --- the install / upgrade reconcile ------------------------------------------


def install(
    home: str | os.PathLike[str] | None = None,
    *,
    defaults: dict[str, str] | None = None,
    provider: str | None = None,
    opt_in: Sequence[str] = (),
) -> InstallReport:
    """Scaffold the config home and reconcile the shipped defaults — idempotent, re-runnable.

    The one entry point for both first-time install and later upgrade: a first run lays
    every default down fresh; a re-run against a newer package applies the conffile
    discipline (refresh pristine, keep edited as ``.new``, respect deletions, never touch
    operator-added files). Re-running against the *same* package is a no-op beyond ensuring
    the directories exist. This is what a fleet rollout loops over a pinned version.

    ``home`` overrides the config-home location (see `config_home`); ``defaults`` overrides
    the shipped default set (the packaged defaults by default) — the seam a test uses to
    simulate a package upgrade by reconciling a *changed* default set against an existing
    install.

    ``provider`` makes the reconcile **provider-aware** (issue #160): when named (the CLI and
    the upgrade reconcile pass ``AI_PROVIDER``), a tool-plugin default whose source declares
    affinity for a *different* provider — a grok/xAI plugin on an OpenAI agent — is **not** laid
    down (it would be clutter the resolver gates off anyway, and a latent foreign-SDK import
    hazard), and one a prior install already laid down is **pruned** if still pristine. ``None``
    (the default the direct API and tests use) lays down every default unfiltered, as before.

    **Powerful tools are opt-in (issue #168).** A tool-plugin default marked ``opt_in`` (media
    generation, web/X search, code execution) is **not** scaffolded for a fresh agent — it ships
    in the package but stays off until explicitly chosen, the same "ships empty" stance as
    ``mcp/``. Two ways it still lands: it is named in ``opt_in`` (a list of plugin file stems,
    e.g. ``["grok_generate_image"]`` — the ``--opt-in`` CLI flag), **or** a *prior* version had
    already scaffolded it into this config home, in which case it is **grandfathered** — kept,
    never silently stripped (the founder's "tools stay the same" migration rule) and reported
    **loudly** in ``InstallReport.grandfathered``. Deletions stay respected either way.
    """
    root = config_home(home)
    report = InstallReport(config_home=root)

    root.mkdir(parents=True, exist_ok=True)
    for name in _SCAFFOLD_DIRS:
        directory = root / name
        if not directory.exists():
            report.created_dirs.append(name)
        directory.mkdir(parents=True, exist_ok=True)

    shipped_all = _packaged_defaults() if defaults is None else defaults
    shipped_provider = _relevant_defaults(shipped_all, provider)
    recorded = _read_manifest(root)
    updated: dict[str, str] = dict(recorded)

    # Power tools (issue #168) are excluded from the scaffold set unless explicitly opted in or
    # grandfathered (already recorded from a prior install). Kept separate from the provider
    # filter so the provider-prune below still keys on `shipped_provider`, unchanged.
    shipped, grandfathered_rels = _opt_in_scaffold_set(shipped_provider, recorded, opt_in)

    # A typo in --opt-in (the stem-vs-name trap, e.g. "listen" for the file "hear_audio") would
    # otherwise scaffold nothing, silently — so name any opt-in that matched no powerful default.
    _warn_unmatched_opt_in(opt_in, shipped_all)

    for rel in sorted(shipped):
        action = _reconcile(root, rel, shipped[rel], recorded, updated, report)
        report.actions[rel] = action

    # A grandfathered power tool is one we kept because it was already on disk — so report only
    # those that are *actually present* after the reconcile. This catches every "not kept" case,
    # not just KEPT_DELETED: a file the operator deleted whose source did not change this release
    # reconciles as UNCHANGED (no write, still absent), and must not be reported as kept either.
    report.grandfathered = [
        rel for rel in grandfathered_rels if root.joinpath(*rel.split("/")).exists()
    ]
    if report.grandfathered:
        _log.warning(
            "Grandfathered %d power tool(s) now opt-in by default (issue #168), kept on this "
            "existing config home rather than stripped: %s. New agents get them off by default.",
            len(report.grandfathered),
            ", ".join(report.grandfathered),
        )

    if shipped_provider is not shipped_all:  # provider-aware: clean up now-mismatched defaults
        _prune_mismatched_defaults(root, shipped_all, shipped_provider, recorded, updated, report)

    _write_manifest(root, updated)
    # Stamp the harness version that produced this config home, so a later wake can detect a
    # `pip install -U` (running version ≠ stamped version) and reconcile the overlay before
    # loading it. Written last, after the defaults are reconciled, so a crash mid-reconcile
    # leaves the stamp behind rather than claiming an upgrade that did not complete.
    _write_installed_version(root)
    return report


def _relevant_defaults(shipped: dict[str, str], provider: str | None) -> dict[str, str]:
    """The shipped defaults relevant to `provider` — drops provider-mismatched ``tools/*.py``.

    Only tool-plugin files carry provider affinity; ``prompts/`` and ``mcp/`` defaults are always
    relevant. With ``provider=None`` the input is returned unchanged (the same object, so the
    caller can cheaply tell no filtering happened and skip the prune pass).
    """
    if provider is None:
        return shipped
    return {
        rel: text
        for rel, text in shipped.items()
        if not _is_tool_plugin(rel) or plugin_relevant_to(text, provider)
    }


def _is_tool_plugin(rel: str) -> bool:
    """Whether a config-home-relative path is a tool-plugin file (the only provider-affine kind)."""
    return rel.startswith("tools/") and rel.endswith(".py")


def _power_tool_stems(shipped: dict[str, str]) -> set[str]:
    """The file stems of every powerful (`opt_in`) tool-plugin default in `shipped`."""
    return {
        rel[len("tools/") : -len(".py")]
        for rel, text in shipped.items()
        if _is_tool_plugin(rel) and plugin_opts_in(text)
    }


def _warn_unmatched_opt_in(opt_in: Sequence[str], shipped_all: dict[str, str]) -> None:
    """Warn (loudly, never silently) for any ``--opt-in`` name that matches no powerful default.

    Catches the stem-vs-name trap (issue #168): ``--opt-in listen`` names the *tool* but the file
    stem is ``hear_audio``, so it would otherwise scaffold nothing with no diagnostic — the
    operator thinks they granted a tool that is silently absent. A name that matches a power tool
    which is merely provider-mismatched is *not* flagged here (it is a real tool, just unavailable
    for this provider); only a name matching no power tool at all is a likely typo.
    """
    known = _power_tool_stems(shipped_all)
    unknown = sorted({name.removesuffix(".py") for name in opt_in} - known)
    if unknown:
        _log.warning(
            "--opt-in named no powerful tool default and scaffolded nothing for: %s. The known "
            "opt-in tools are: %s. (Pass the plugin *file stem*, e.g. 'hear_audio', not 'listen'.)",
            ", ".join(unknown),
            ", ".join(sorted(known)),
        )


def _opt_in_scaffold_set(
    shipped: dict[str, str], recorded: dict[str, str], opt_in: Sequence[str]
) -> tuple[dict[str, str], list[str]]:
    """Filter the scaffold set for the opt-in policy (issue #168); return it + the grandfathered.

    A powerful tool-plugin default (``opt_in=True`` in its source) is **excluded** from the
    scaffold set — it ships in the package but stays off for a fresh agent — **unless**:

    - it is named in ``opt_in`` (by its file stem, e.g. ``"grok_generate_image"``), an explicit
      operator choice (the ``--opt-in`` flag), **or**
    - it is already **recorded** (a prior install scaffolded it) — *grandfathered*, kept rather
      than silently stripped (the founder's "tools stay the same" rule). Its ``rel`` is returned
      in the second element so the caller can report it loudly.

    Benign defaults (and all non-tool defaults) pass through untouched, keeping the normal
    install-then-prune behavior. A grandfathered/opted-in default re-enters the reconcile, so a
    respected deletion still wins (the caller drops a `KEPT_DELETED` one from the loud report).
    """
    opt_in_stems = {name.removesuffix(".py") for name in opt_in}
    result: dict[str, str] = {}
    grandfathered: list[str] = []
    for rel, text in shipped.items():
        if not (_is_tool_plugin(rel) and plugin_opts_in(text)):
            result[rel] = text  # benign default / non-tool file → unchanged
            continue
        stem = rel[len("tools/") : -len(".py")]
        if stem in opt_in_stems:
            result[rel] = text  # explicit operator opt-in
        elif rel in recorded:
            result[rel] = text  # grandfathered: a prior install already laid it down
            grandfathered.append(rel)
        # else: powerful + neither opted-in nor previously installed → not scaffolded
    return result, grandfathered


def _prune_mismatched_defaults(
    root: Path,
    shipped_all: dict[str, str],
    shipped: dict[str, str],
    recorded: dict[str, str],
    updated: dict[str, str],
    report: InstallReport,
) -> None:
    """Remove a previously-installed tool default that is now provider-mismatched (issue #160).

    The @jt symptom: an earlier, provider-blind install copied the grok/xAI plugins into an
    OpenAI agent's overlay. A provider-aware reconcile cleans that up — but *only* a default we
    own and the operator has not touched: a tool default that we recorded (`recorded`), that is a
    real shipped default (`shipped_all`) yet relevant to a *different* provider (absent from the
    filtered `shipped`), and whose on-disk bytes still match what we installed (pristine). An
    operator-edited copy is left alone (their edit wins, exactly as the conffile rule keeps it);
    a file already gone just has its stale manifest entry dropped. Nothing the operator added is
    ever touched — the walk is over *recorded shipped defaults*, never the operator's directory.
    """
    for rel in list(recorded):
        if not _is_tool_plugin(rel) or rel in shipped or rel not in shipped_all:
            continue  # not a tool default we're now filtering out → leave it
        target = root.joinpath(*rel.split("/"))
        if not target.exists():
            updated.pop(rel, None)  # already gone → drop the stale manifest entry
            report.actions[rel] = KEPT_DELETED
            continue
        try:
            pristine = _hash(target.read_text(encoding="utf-8")) == recorded.get(rel)
        except OSError:
            # Unreadable (permissions, a concurrent delete after the exists() check): leave it
            # rather than delete-without-checking. Pruning is a best-effort de-clutter, so the
            # safe failure is to keep the file and its manifest entry, never to remove blindly.
            continue
        if pristine:
            target.unlink()  # pristine, ours, now mismatched → safe to remove
            updated.pop(rel, None)
            report.actions[rel] = PRUNED
        # else: the operator edited this mismatched default → keep theirs, keep tracking it.


def reconcile_on_upgrade(
    home: str | os.PathLike[str] | None = None,
    *,
    defaults: dict[str, str] | None = None,
    provider: str | None = None,
) -> InstallReport | None:
    """Reconcile a materialized config home after a package upgrade — the wake-time auto-install.

    ``pip install -U basecradle-harness`` upgrades the *package* but never touches the
    operator's *materialized* config home, so a `tools/` overlay copied out by the previous
    version outlives the upgrade — and a default plugin that the new version changed (or whose
    imports the new version removed) silently goes stale, disabling a capability on a green-CI
    deploy (issue #160). This is the automatic fix: a wake calls it before loading the overlay,
    and it re-runs the conffile reconcile (`install`) exactly when the running harness version
    differs from the one stamped at the last install.

    It acts **only on an already-installed config home** — one whose manifest records a prior
    install. A never-installed deployment (like @jt, which runs off the packaged-default
    fallback in `_plugins.load_plugins`) has nothing materialized to go stale: its tools load
    straight from the freshly-upgraded package, so there is nothing to reconcile and this is a
    no-op (it must **not** auto-create a config home and flip that agent onto the overlay path).
    A config home installed by a harness predating this stamp has no ``.version`` file, which
    reads as "unknown" and so reconciles once on the first upgrade, stamping it going forward.

    The reconcile is **provider-aware** (issue #160): it filters and prunes tool-plugin defaults
    by the active ``AI_PROVIDER`` (``provider``, defaulting to the env), so a grok/xAI default an
    earlier provider-blind install left in an OpenAI agent's overlay is cleaned up on the upgrade.

    Returns the `InstallReport` when it reconciled, or ``None`` when it was a no-op (not
    installed, or already at the running version). ``defaults`` is forwarded to `install` — the
    same test seam, to simulate a package whose shipped defaults changed.
    """
    root = config_home(home)
    if not _read_manifest(root):
        return None  # never installed → packaged-default fallback path; nothing materialized
    if installed_version(root) == __version__:
        return None  # config home already produced by the running version → overlay is current
    report = install(
        home,
        defaults=defaults,
        provider=provider if provider is not None else active_provider_from_env(),
    )
    _log.info(
        "Config home %s reconciled to basecradle-harness %s after an upgrade: %s",
        root,
        __version__,
        report.summary().replace("\n", " | "),
    )
    return report


def _reconcile(
    root: Path,
    rel: str,
    default_text: str,
    recorded: dict[str, str],
    updated: dict[str, str],
    report: InstallReport,
) -> str:
    """Reconcile one shipped default against the manifest and the on-disk file (dpkg conffile).

    Compares three hashes — the new shipped default (``new``), the one recorded at the last
    install (``was``), and the file as it sits on disk (``disk``) — and resolves the one
    correct action, never clobbering an operator's edit:

    - shipped default unchanged since last install (``new == was``) → leave whatever the
      operator has; there is no new default to offer, so an edit is theirs to keep.
    - file absent → fresh install if we never recorded it; otherwise the operator deleted a
      file we manage, so respect the deletion and do **not** resurrect it.
    - on-disk equals the new default → already current, nothing to write.
    - on-disk equals what we last installed → pristine, so replace it with the new default.
    - otherwise → the operator edited it: keep theirs, write the new default beside it as
      ``<name>.new``, and note it.

    ``updated`` is advanced to the new default's hash in every branch (the manifest always
    records the *current* shipped default for a path), so a subsequent same-version run is a
    clean no-op and a later genuine change re-evaluates from an accurate baseline.
    """
    target = root.joinpath(*rel.split("/"))
    new = _hash(default_text)
    was = recorded.get(rel)
    updated[rel] = new  # the manifest tracks the current shipped default for this path

    if was == new:
        return UNCHANGED  # default unchanged since last install → operator's copy is theirs

    if not target.exists():
        if was is None:
            _write(target, default_text)
            return INSTALLED  # never installed → lay it down fresh
        return KEPT_DELETED  # operator deleted a file we manage → respect it, don't resurrect

    disk = _hash(target.read_text(encoding="utf-8"))
    if disk == new:
        return ALREADY_CURRENT  # operator's copy already equals the new default
    if disk == was:
        _write(target, default_text)
        return REFRESHED  # pristine default, shipped default changed → replace it

    # Operator-edited: keep theirs, drop the new default beside it for them to merge.
    new_path = target.parent / f"{target.name}.new"
    _write(new_path, default_text)
    report.new_files.append(rel)
    return KEPT_EDITED


def _write(path: Path, text: str) -> None:
    """Write a config file, creating parent dirs — the installer's one write primitive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- the manifest -------------------------------------------------------------


def _manifest_path(root: Path) -> Path:
    return root / _MANIFEST_NAME


def _read_manifest(root: Path) -> dict[str, str]:
    """The hashes recorded at the last install, or ``{}`` on a first install / unreadable file.

    A missing or corrupt manifest reads as empty, so a first install proceeds (every default
    is fresh) and a damaged one degrades to re-laying defaults rather than crashing the
    rollout — the worst case is a pristine default re-written, never an operator edit lost
    (an edited file still differs from the default and is kept as ``.new``).
    """
    path = _manifest_path(root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def _write_manifest(root: Path, manifest: dict[str, str]) -> None:
    """Persist the per-file default hashes, sorted for a stable, diff-friendly file."""
    body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _write(_manifest_path(root), body)


# --- the version stamp --------------------------------------------------------


def _version_path(root: Path) -> Path:
    return root / _VERSION_NAME


def installed_version(root: str | os.PathLike[str] | None = None) -> str | None:
    """The harness version that last reconciled this config home, or ``None`` if unknown.

    ``None`` for a never-installed home (no stamp) or one installed by a harness predating
    the stamp (the file is absent) — both read as "needs reconciling on the next upgrade
    check". A whitespace-only file also reads as ``None``. Kept separate from the manifest so
    the version lives in its own one-line file, not mixed in among the per-path default hashes.
    """
    path = _version_path(config_home(root))
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _write_installed_version(root: Path) -> None:
    """Stamp the running harness version as the one that produced this config home."""
    _write(_version_path(root), __version__ + "\n")


# --- sourcing the charter from files ------------------------------------------

# The charter files, in the order they compose into Turn 0: the standing system prompt
# first, then the operating guidance. Both are shipped defaults the installer wrote.
_CHARTER_FILES = ("prompts/system-prompt.md", "prompts/initialize.md")


def charter_from_config(home: str | os.PathLike[str] | None = None) -> str | None:
    """Compose the operator charter from the config-home prompt files, or ``None`` if absent.

    Turn 0's operator charter is sourced from real files now — ``prompts/system-prompt.md``
    then ``prompts/initialize.md`` — not an env var. Each present, non-empty file
    contributes its text (HTML comments, which are operator-facing notes, stripped first);
    they join in order.

    The return distinguishes *absent* from *present-but-empty*, which the env-var fallback
    (`charter_from_env`) depends on:

    - **No charter file exists** → ``None``. The config home was never installed, so a
      caller may fall back to the legacy env var without fabricating a charter.
    - **A file exists but they compose to empty** (an operator who deliberately blanked the
      charter — all whitespace or only HTML comments) → ``""``. The config home *is*
      installed, so the files win: an empty charter is honored, never silently replaced by a
      stale ``HARNESS_SYSTEM_PROMPT``.

    Onboarding (the Dashboard orientation) still composes on top of this exactly as before —
    sourcing changes, composition does not.
    """
    root = config_home(home)
    parts: list[str] = []
    found = False
    for rel in _CHARTER_FILES:
        path = root.joinpath(*rel.split("/"))
        if not path.exists():
            continue
        found = True
        text = _strip_html_comments(path.read_text(encoding="utf-8")).strip()
        if text:
            parts.append(text)
    if not found:
        return None  # no charter files at all → not installed; let the caller fall back
    # Files exist: the config home is installed, so the files win even when they compose to
    # empty (an operator who deliberately blanks the charter). Return "" — *present but
    # empty* — not None, so the env-var fallback is not resurrected behind their back.
    return "\n\n".join(parts)


def charter_from_env(home: str | os.PathLike[str] | None = None) -> str | None:
    """The operator charter: the config-home files if installed, else the legacy env var.

    Files are the source of record now (`charter_from_config`); ``HARNESS_SYSTEM_PROMPT`` is
    retained only as a fallback for a deployment that has not yet run the installer, so the
    migration is lossless. Once the config home exists, the files win — the env var is no
    longer consulted.
    """
    from_files = charter_from_config(home)
    if from_files is not None:
        return from_files
    return os.environ.get("HARNESS_SYSTEM_PROMPT")


def _strip_html_comments(text: str) -> str:
    """Drop ``<!-- … -->`` blocks: operator notes that should not reach the model's context."""
    return _HTML_COMMENT_RE.sub("", text)


# --- sourcing one prompt at a time (the persistent brief) ---------------------

# The prompt files the brief composes individually (`initialize.md`, `system-prompt.md`).
# Unlike `charter_from_config`, which joins both into one charter, the persistent Turn-0
# brief interleaves the manifest and the live dashboard *between* them, so it needs each
# prompt's text on its own. These two helpers source one file with the same
# "config-home-if-installed, else the packaged default" precedent `load_plugins` uses.


def _prompts_installed(root: Path) -> bool:
    """Whether the config home's ``prompts/`` defaults have been installed (manifest-recorded).

    The same signal `load_plugins` uses for ``tools/``: once the installer has written a
    ``prompts/`` default, the config home is authoritative for prompts — an operator's edit,
    or *deletion*, wins, and the packaged default is no longer consulted. Until then (a
    never-installed home, or one predating these defaults) the packaged defaults load
    directly, so an un-migrated agent like @jt still composes a full brief.
    """
    return any(key.startswith("prompts/") for key in _read_manifest(root))


def _packaged_prompt(name: str) -> str | None:
    """The packaged default text for ``prompts/<name>``, or ``None`` if not shipped."""
    res = resources.files("basecradle_harness").joinpath(_DEFAULTS_PACKAGE, "prompts", name)
    if not res.is_file():
        return None
    return res.read_text(encoding="utf-8")


def prompt_text(name: str, home: str | os.PathLike[str] | None = None) -> str | None:
    """One shipped prompt's text: the config-home file if installed, else the packaged default.

    HTML comments (operator notes) are stripped; the result is whitespace-trimmed. Returns
    ``None`` when the file is absent — once prompts are installed an operator *deletion* is
    honored (no resurrection from the package); before that the packaged default is the
    source. This is the brief's accessor for the provider-independent ``initialize.md``.
    """
    root = config_home(home)
    if _prompts_installed(root):
        path = root / "prompts" / name
        raw = path.read_text(encoding="utf-8") if path.exists() else None
    else:
        raw = _packaged_prompt(name)
    if raw is None:
        return None
    return _strip_html_comments(raw).strip() or None


def system_prompt_text(home: str | os.PathLike[str] | None = None) -> str | None:
    """The personality charter for the brief: ``system-prompt.md``, with the legacy fallback.

    Resolution mirrors `charter_from_env`'s "files win once installed, else the legacy env
    var" precedent, scoped to the personality slot:

    - **Prompts installed** → the config-home ``system-prompt.md`` (an operator deletion
      honored as ``None``); the env var is no longer consulted.
    - **Not installed** → ``HARNESS_SYSTEM_PROMPT`` if set (the un-migrated agent like @jt
      keeps its env personality), else the packaged default.
    """
    root = config_home(home)
    if _prompts_installed(root):
        path = root / "prompts" / "system-prompt.md"
        if not path.exists():
            return None  # operator deleted it → honored, no resurrection
        return _strip_html_comments(path.read_text(encoding="utf-8")).strip() or None
    env = os.environ.get("HARNESS_SYSTEM_PROMPT")
    if env:
        return env  # un-migrated agent: its env personality is the legacy charter
    packaged = _packaged_prompt("system-prompt.md")
    return _strip_html_comments(packaged).strip() or None if packaged else None


# --- the CLI ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """The ``basecradle-harness-install`` entrypoint: scaffold/upgrade the config home, then exit.

    Idempotent and re-runnable: the first run installs the shipped defaults, a later run
    against a newer package upgrades them conffile-style (refresh pristine, keep edited as
    ``.new``, respect deletions, never touch operator files). Prints a short summary and the
    config-home path so a rollout can log what changed. Exit 0 on success.

    Provider-aware by default (issue #160): only the tool-plugin defaults relevant to the
    agent's ``AI_PROVIDER`` are laid down (a grok/xAI plugin is not copied into an OpenAI
    agent's overlay), and a now-mismatched one a prior install left behind is pruned.
    ``--provider`` overrides the env; ``--all-providers`` disables the filter and lays down
    every default (the old provider-blind behavior).
    """
    parser = argparse.ArgumentParser(
        prog="basecradle-harness-install",
        description=(
            "Scaffold (or upgrade) the BaseCradle Harness config home: prompts/, tools/, "
            "mcp/, and the shipped charter defaults. Idempotent — safe to re-run on every "
            "upgrade; operator edits are kept, never clobbered."
        ),
    )
    parser.add_argument(
        "--config-home",
        default=None,
        help=(
            "the config home to scaffold (default: $BASECRADLE_CONFIG_HOME, else "
            "$HOME/.config/basecradle)."
        ),
    )
    parser.add_argument(
        "--provider",
        default=None,
        help=(
            "lay down only the tool defaults relevant to this provider "
            "(default: $AI_PROVIDER, else 'openai'). Use --all-providers to disable filtering."
        ),
    )
    parser.add_argument(
        "--all-providers",
        action="store_true",
        help="lay down every tool default regardless of provider (the provider-blind behavior).",
    )
    parser.add_argument(
        "--opt-in",
        default="",
        metavar="NAMES",
        help=(
            "comma-separated plugin file stems to scaffold despite being powerful/opt-in tools "
            "(issue #168) — e.g. --opt-in generate_image,web_search. Powerful tools (media "
            "generation, web/X search, code execution) are off by default for a new agent; this "
            "is how you grant one. Already-installed ones are kept (grandfathered) regardless."
        ),
    )
    args = parser.parse_args(argv)

    provider = None if args.all_providers else (args.provider or active_provider_from_env())
    opt_in = [name.strip() for name in args.opt_in.split(",") if name.strip()]
    report = install(args.config_home, provider=provider, opt_in=opt_in)
    print(report.summary())
    return 0
