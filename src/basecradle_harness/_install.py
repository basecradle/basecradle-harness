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
      prompts/
        system-prompt.md   # shipped default
        initialize.md      # shipped default (starter; the richer default is a later group)
      tools/               # empty this group (loading from it is a later group)
      mcp/                 # empty this group (loading from it is a later group)
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

**Boundary (this group).** Tool defaults stay package-registered; the ``tools/`` and
``mcp/`` dirs are created but loading from them is a later group, as is the persistent
Turn 0 and the generated tool manifest. This module owns *where things live and how
install/upgrade works* — not the tool system or prompt composition beyond sourcing the
charter from files (`charter_from_config`).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources as resources
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# The directories the installer scaffolds under the config home. `prompts/` holds the
# shipped charter defaults; `tools/` and `mcp/` are created empty this group (loading
# from them is a later group), so an operator already sees where those will go.
_SCAFFOLD_DIRS = ("prompts", "tools", "mcp")

# Bookkeeping, not an operator file: the hash of every shipped default as it was last
# installed, so the upgrader can tell a pristine default from an edited one. Leading dot
# signals "tooling — do not edit"; it is never composed into anything the model reads.
_MANIFEST_NAME = ".manifest.json"

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


@dataclass
class InstallReport:
    """What a reconcile did: the config home, the dirs it ensured, and per-file outcomes."""

    config_home: Path
    created_dirs: list[str] = field(default_factory=list)
    actions: dict[str, str] = field(default_factory=dict)
    new_files: list[str] = field(default_factory=list)  # the `.new` files written this run

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
        return "\n".join(lines)


# --- the install / upgrade reconcile ------------------------------------------


def install(
    home: str | os.PathLike[str] | None = None,
    *,
    defaults: dict[str, str] | None = None,
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
    """
    root = config_home(home)
    report = InstallReport(config_home=root)

    root.mkdir(parents=True, exist_ok=True)
    for name in _SCAFFOLD_DIRS:
        directory = root / name
        if not directory.exists():
            report.created_dirs.append(name)
        directory.mkdir(parents=True, exist_ok=True)

    shipped = _packaged_defaults() if defaults is None else defaults
    recorded = _read_manifest(root)
    updated: dict[str, str] = dict(recorded)

    for rel in sorted(shipped):
        action = _reconcile(root, rel, shipped[rel], recorded, updated, report)
        report.actions[rel] = action

    _write_manifest(root, updated)
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
    args = parser.parse_args(argv)

    report = install(args.config_home)
    print(report.summary())
    return 0
