"""Only the harness's own commands on the PATH of what it spawns: its environment leaves it.

A harness agent is installed into a virtualenv, and that environment's ``bin`` directory holds
three kinds of thing: the harness's own console scripts (``basecradle-harness-wake``,
``basecradle-harness-verify``, …, and ``mempalace`` when the extra is installed), the
environment's own interpreter and installer (``python``, ``python3``, ``pip``), and every console
script a dependency brings (``httpx``, ``uvicorn``, ``tqdm``, ``pygmentize``, …).

The history, because the contract changed knowingly
--------------------------------------------------
Issue #409 found that a wake, launched by absolute path, activates nothing, so the agent's `shell`
tool answered ``mempalace: command not found``. The fix put the environment's whole ``bin`` on the
child's ``PATH`` under the contract *reachable, not first*. Measured on a fleet server on
2026-11-21 (issue #969), that made three things true that an agent's ``~/software`` law forbids:
``python3``, ``pip`` and ``pip3`` resolved to the harness's own environment, so a plain
``pip install`` installed **into the harness** (four agents had packages there the harness never
brought); ``~/.local/bin`` sat ahead of that directory, so a link there could replace a harness
command; and every dependency's script shadowed a tool an agent installed for itself.

The founder's decision (`@origin`, 2026-11-21, issue #969) is the contract now: **only the
harness's own commands are on an agent's PATH, from a directory that holds nothing else, placed
first.** Its requirements, each with a tempting broken form:

- **Remove, do not cover.** The environment's ``bin`` is taken **off** the ``PATH`` (`spawn_env`,
  `path_preamble`). Putting a curated directory in front of it and leaving the old entry in place
  still resolves ``python3`` to the harness.
- **The allowlist is the harness's own console scripts plus ``mempalace``** (`own_commands`) — read
  from this distribution's installed metadata, so there is no second list to go stale. Not the
  interpreter, not ``pip``, not a dependency's scripts, and not ``mempalace-mcp`` or
  ``mempalace-light-mcp``: their shebangs are absolute and they are started by path.
- **Derived, never configured.** The links point into the directory ``sys.executable`` sits in, read
  at spawn. No env var, no config key, nothing for provisioning to mirror: a moved or rebuilt venv
  is followed by the next wake because it *is* the running interpreter's location.
- **First, and after the profile.** The `shell` tool runs ``/bin/bash -lc``, a login shell, and
  some ``/etc/profile``s assign ``PATH`` outright (Debian's does; macOS's ``path_helper`` rebuilds
  and reorders it). So `spawn_env` is only half of it: `path_preamble` is one line of shell placed
  ahead of the model's command, which runs **after** the profile, removes the environment's
  ``bin`` from whatever the profile built, and puts the commands directory in front. The order the
  command sees is: the commands directory, then what the profile built (``~/.local/bin`` first on
  a fleet server), then the system.
- **Every subprocess, not only the shell**, and on every machine, not only a fleet server.

The commands directory
----------------------
A directory of symlinks, one per allowlisted name present in the environment's ``bin``, made by this
process with `tempfile.mkdtemp` (mode ``0700``) the first time something is spawned, and named
``basecradle-harness-commands-<pid>-<random>``. A symlink is enough because a console script's
interpreter is named by its own shebang (absolute, or ``realpath``-relative in a relocatable venv),
never by the ``PATH`` it runs under. It is per process because it can be: a wake is one process, and
a directory nothing else writes to is one that holds nothing else.

**Whatever creates, cleans up.** The process removes its own directory at exit (`atexit`). A process
that dies without running `atexit` — killed, or stopped, since `_stop.die` ends a stopped wake by
its signal — leaves it behind, so the next process to make one removes its predecessors: an entry
in the same place, by this exact name, owned by this user, untouched for `STRANDED_AFTER` **and**
whose stamped pid is dead (`_sweep_stranded`). Age is never the reason by itself.

**Losing it costs reachability, never the removal.** If the directory cannot be made (a read-only
temp directory, a path that cannot be a ``PATH`` entry), the environment's ``bin`` still leaves the
``PATH`` and one WARNING says the harness's commands are not reachable by name this wake. The
removal is the property the decision is about; the reach is the convenience #409 asked for.

On an interpreter that is not in a virtual environment (a system ``python3``, where
``sys.prefix == sys.base_prefix``) its ``bin`` is the system's own, so nothing is removed: there is
no separate environment to take away, and removing ``/usr/bin`` would break the box.

Its one residual, named: a background process the agent starts from a `shell` call keeps the
directory on its ``PATH`` after the wake that made it has exited and removed it, so in that process
the harness's commands are reachable only by absolute path.
"""

from __future__ import annotations

import atexit
import functools
import importlib.metadata
import logging
import os
import re
import shlex
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

_log = logging.getLogger("basecradle_harness")

_PATH = "PATH"

#: The distribution whose console scripts are the harness's own commands.
_DISTRIBUTION = "basecradle-harness"

#: The one command from outside this distribution on the allowlist (issue #969): the palace's own
#: CLI, which a bare ``mempalace status`` must reach (issue #409).
MEMPALACE = "mempalace"

#: The commands directory's name, ahead of the pid and `tempfile.mkdtemp`'s random suffix.
_PREFIX = "basecradle-harness-commands-"
_STAMPED = re.compile(rf"^{re.escape(_PREFIX)}(?P<pid>\d+)-[^{re.escape(os.sep)}]+$")

#: How long another process's commands directory must sit untouched, with its pid dead, before it
#: is called stranded and removed. A live wake re-links into its own directory on every spawn, and
#: its pid is alive besides; an hour is far past any doubt that two clocks or pid namespaces can
#: raise. The same reasoning, and the same hour, as the orphan sweep's ``STRANDED_AFTER``.
STRANDED_AFTER = 60 * 60

#: Where the commands directory is made: `tempfile.gettempdir` when ``None``. A test points it at
#: its own ``tmp_path``.
_PARENT: str | None = None

_lock = threading.Lock()

#: The directory this process made, and the environment ``bin`` its links point into.
_made: tuple[str, str] | None = None

#: Every directory this process made, removed at exit; with the pid that made them, so a forked
#: child's exit never removes its parent's.
_owned: list[tuple[int, str]] = []

_warned = False


def interpreter_bin_dir() -> str | None:
    """The directory holding this interpreter's console scripts, or ``None`` if there is none.

    ``Path(sys.executable).parent`` — a venv's ``bin/`` when the harness runs from one, and the
    system ``bin`` when it does not. ``None`` in the three cases where there is nothing honest to
    use: an interpreter that reports no ``sys.executable`` at all (an embedded/frozen host), a
    directory that does not exist, and a directory whose name contains ``os.pathsep``, which
    *cannot* be represented as a ``PATH`` entry and would silently split into two wrong ones.
    """
    executable = sys.executable
    if not executable:
        return None
    bin_dir = str(Path(executable).parent)
    if not bin_dir or os.pathsep in bin_dir:
        return None
    return bin_dir if os.path.isdir(bin_dir) else None


def _in_virtual_environment() -> bool:
    """True when this interpreter runs from a virtual environment of its own."""
    return sys.prefix != sys.base_prefix


@functools.cache
def own_commands() -> tuple[str, ...]:
    """The allowlist: this distribution's console scripts, plus `MEMPALACE`, sorted.

    Read from the installed distribution's metadata, the record the installer wrote the scripts
    from, so it cannot disagree with what is in ``bin``. A harness running without its metadata
    (vendored, not installed) has no console scripts to name and contributes only `MEMPALACE`.
    """
    try:
        points = importlib.metadata.distribution(_DISTRIBUTION).entry_points
    except importlib.metadata.PackageNotFoundError:
        names: set[str] = set()
    else:
        names = {point.name for point in points if point.group == "console_scripts"}
    return tuple(sorted(names | {MEMPALACE}))


def _process_alive(pid: int) -> bool:
    """Is a process with this pid running? One another user owns is running."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OverflowError, ValueError):
        return False  # a number no process can hold
    return True


def _parent() -> str:
    return _PARENT or tempfile.gettempdir()


def _sweep_stranded(parent: str) -> None:
    """Remove the commands directories processes that are gone left in `parent` (see the module).

    Only an entry this harness names, owned by this user, a real directory (never a link), untouched
    for `STRANDED_AFTER`, whose stamped pid is dead. Best-effort: a directory that cannot be listed
    or removed is left, and the next process to make one tries again.
    """
    now = time.time()
    try:
        entries = list(os.scandir(parent))
    except OSError:
        return
    for entry in entries:
        match = _STAMPED.match(entry.name)
        if match is None or int(match["pid"]) == os.getpid():
            continue
        try:
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if not entry.is_dir(follow_symlinks=False):
            continue
        if hasattr(os, "geteuid") and stat.st_uid != os.geteuid():
            continue
        if now - stat.st_mtime < STRANDED_AFTER or _process_alive(int(match["pid"])):
            continue
        shutil.rmtree(entry.path, ignore_errors=True)


def _link(path: str, target: str) -> None:
    """Make `path` a symlink to `target`, atomically, unless it already is one."""
    try:
        if os.readlink(path) == target:
            return
    except OSError:
        pass
    staged = f"{path}.{os.getpid()}.tmp"
    try:
        os.unlink(staged)
    except FileNotFoundError:
        pass
    os.symlink(target, staged)
    os.replace(staged, path)


def _remove_owned() -> None:
    """At exit: remove every commands directory this process made."""
    for pid, path in _owned:
        if pid == os.getpid():
            shutil.rmtree(path, ignore_errors=True)


atexit.register(_remove_owned)


def _reset() -> None:
    """Forget the directory this process made, for a test that points `_PARENT` somewhere new."""
    global _made, _warned
    with _lock:
        _made = None
        _warned = False


def commands_dir() -> str | None:
    """The directory holding links to the harness's own commands, made on first use, or ``None``.

    ``None`` when there is no interpreter directory, when none of `own_commands` is in it, or when
    the directory cannot be made — that last case logged once at WARNING. Re-links on every call,
    so a command the environment gained after the directory was made (``mempalace`` installed
    under a running wake) is reachable at the next spawn, and a directory removed under the process
    is made again.
    """
    global _made, _warned
    bin_dir = interpreter_bin_dir()
    if bin_dir is None:
        return None
    links = {
        name: os.path.join(bin_dir, name)
        for name in own_commands()
        if os.path.isfile(os.path.join(bin_dir, name))
    }
    if not links:
        return None
    with _lock:
        try:
            made = _made[0] if _made is not None and _made[1] == bin_dir else None
            if made is None or not os.path.isdir(made):
                parent = _parent()
                _sweep_stranded(parent)
                made = tempfile.mkdtemp(prefix=f"{_PREFIX}{os.getpid()}-", dir=parent)
                if os.pathsep in made:
                    os.rmdir(made)
                    raise OSError(f"{made!r} cannot be a PATH entry")
                _owned.append((os.getpid(), made))
                _made = (made, bin_dir)
            for name, target in links.items():
                _link(os.path.join(made, name), target)
        except OSError as error:
            if not _warned:
                _warned = True
                _log.warning(
                    "harness commands are not on the PATH of what this process spawns, "
                    "because their directory could not be made: %s",
                    error,
                )
            return None
    return made


def _dropped(entries: Iterable[str]) -> set[str]:
    """The ``PATH`` entries that name this interpreter's environment ``bin``, as literal strings.

    Empty outside a virtual environment (see the module). Inside one: the directory as derived and
    as resolved, each with and without a trailing separator, plus any absolute entry of `entries`
    that is the same directory under another spelling (a symlinked alias).
    """
    bin_dir = interpreter_bin_dir()
    if bin_dir is None or not _in_virtual_environment():
        return set()
    forms = {bin_dir, os.path.realpath(bin_dir)}
    forms |= {form.rstrip(os.sep) + os.sep for form in forms}
    for entry in entries:
        if entry and os.path.isabs(entry) and entry not in forms:
            try:
                if os.path.samefile(entry, bin_dir):
                    forms.add(entry)
            except OSError:
                continue
    return forms


def spawn_env(env: Mapping[str, str] | None = None, *, commands: bool = True) -> dict[str, str]:
    """A copy of `env` (default: this process's) with the harness's ``PATH`` contract applied.

    The interpreter's environment ``bin`` is removed from ``PATH`` in every spelling `_dropped`
    finds, and, when `commands` is true and there is one, `commands_dir` goes first. Every other
    entry keeps its place, empty ones included. Never mutates its argument.

    A missing ``PATH`` stays missing when there is nothing to add (the child resolves against
    ``os.defpath`` as it would have), and becomes the commands directory plus ``os.defpath`` when
    there is, so ``/bin`` and ``/usr/bin`` are not taken from the child. The leading empty entry
    POSIX ``os.defpath`` carries (*the current directory*) is dropped rather than passed on.

    `commands` false is for a child that runs nothing of the harness's (``journalctl``): the
    environment still leaves its ``PATH``, and no directory is made for it.
    """
    result = dict(os.environ if env is None else env)
    current = result.get(_PATH)
    if current is None:
        entries = os.defpath.lstrip(os.pathsep).split(os.pathsep)
    else:
        entries = current.split(os.pathsep) if current else []
    first = commands_dir() if commands else None
    drop = _dropped(entries) | ({first} if first else set())
    kept = [entry for entry in entries if entry not in drop]
    if current is None and first is None:
        return result
    result[_PATH] = os.pathsep.join([first, *kept] if first else kept)
    return result


def path_preamble(env: Mapping[str, str] | None = None) -> str:
    """One line of POSIX shell applying the same contract as `spawn_env`, after the profile ran.

    Placed ahead of a model-authored command so it runs *after* the login shell has sourced the
    profile — the half that survives a profile which assigns ``PATH`` outright. It walks ``PATH``
    entry by entry, drops every spelling of the environment ``bin`` that `_dropped` finds in `env`
    (default: this process's) and the commands directory itself, keeps the rest in order (empty
    entries included), and puts the commands directory in front. Empty string when there is
    nothing to remove and nothing to add, so a caller splices it in only when it says something.

    Every dropped spelling is written into the ``case`` pattern **quoted** (`shlex.quote`), and the
    commands directory is matched through a quoted expansion, so a glob character in a path is
    matched literally and can never turn the test into a wildcard. ``${PATH-}`` keeps a profile
    running under ``set -u`` working, an empty ``PATH`` gains no empty entry (which a shell reads as
    the current directory), and every variable the line uses is unset afterwards, so the model's
    command sees the shell it would have seen anyway.
    """
    source = os.environ if env is None else env
    current = source.get(_PATH)
    first = commands_dir()
    drop = _dropped(current.split(os.pathsep) if current else [])
    patterns = (['"$__bc_c"'] if first else []) + [shlex.quote(form) for form in sorted(drop)]
    if not patterns:
        return ""
    assign = 'PATH="$__bc_c$__bc_p"' if first else 'PATH="${__bc_p#:}"'
    return (
        f"__bc_c={shlex.quote(first or '')}; __bc_p=; __bc_r=${{PATH-}}; "
        'case "$__bc_r" in "") ;; *) __bc_r="$__bc_r:"; '
        'while [ -n "$__bc_r" ]; do __bc_e="${__bc_r%%:*}"; __bc_r="${__bc_r#*:}"; '
        f'case "$__bc_e" in {"|".join(patterns)}) ;; *) __bc_p="$__bc_p:$__bc_e" ;; esac; '
        "done ;; esac; "
        f"{assign}; export PATH; unset __bc_c __bc_p __bc_r __bc_e"
    )
