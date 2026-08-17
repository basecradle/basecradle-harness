"""Put the harness's own console scripts on the PATH of the subprocesses it spawns.

A harness agent is installed into a virtualenv, and every console script that comes with it —
``basecradle-harness-wake``, ``basecradle-harness-verify``, ``basecradle-harness-claims``, and
the entry points its extras bring (``mempalace`` foremost) — lands in that venv's ``bin``
directory. Nothing puts that directory on the agent's own ``PATH``: a wake is launched by
absolute path (the router runs ``/home/<agent>/venv/bin/basecradle-harness-wake``), which needs
no activation and therefore never performs one. Every subprocess the agent spawns inherits that
un-activated environment.

The consequence is the failure this module exists to remove (issue #409, found by the live
acceptance run rather than by any test). On a MemPalace-provider agent the harness publishes its
palace binding into ``~/.mempalace/config.json`` precisely so that a **bare** ``mempalace status``
reads the live palace — and then the agent's own `shell` tool answered ``mempalace: command not
found``, because ``/home/<agent>/venv/bin`` was not on the PATH it spawned with. The documented
CLI worked only for someone who already knew the private venv path and typed it every time: the
same knowing-the-private-path defect #409 exists to remove, one layer up.

Derived, never configured
-------------------------
The directory is read off ``sys.executable`` **at spawn time**, so it is whatever venv this
process is actually running from. There is no env var, no config key, and nothing for provisioning
to mirror: move or rebuild the venv and the next wake derives the new location, because it *is*
the running interpreter's location. That is the same property that makes the palace publication
safe — a projection of the binding, never an input to it — and it is why the alternatives were
rejected. A symlink into ``/usr/local/bin`` is *shared*, so it collapses per-agent isolation the
moment a second agent lands on the box (the same isolation the palace lives under ``$HARNESS_HOME``
to preserve); a ``PATH`` written into ``agent.env`` or a wrapper script is a second source of truth
that goes stale the first time a venv moves, silently, with nothing able to say so.

It only ever **adds**, and only when the directory is not already there: an operator who activated
the venv, or who ordered ``PATH`` deliberately, is left exactly as they are. On an interpreter that
is not in a venv at all (a system ``python3``) the directory is ``/usr/bin``, which is already on
every sane PATH — so this is a no-op for a non-venv install rather than a reordering of it.

Two mechanisms for the shell, because one is not enough
------------------------------------------------------
The `shell` tool runs its command through ``/bin/bash -lc`` — a **login** shell, deliberately, so
the profile is sourced and the agent's terminal matches a human's. That is what makes an inherited
PATH insufficient on its own: a login shell sources ``/etc/profile`` *before* running the command,
and some distributions' ``/etc/profile`` **assigns** ``PATH`` outright rather than appending to it
(Debian's does; macOS's ``path_helper`` rebuilds and reorders it). An inherited prepend is
discarded there — invisibly, with the tool working perfectly and only the console script missing.

So both halves are used, and they cover different things:

- `with_interpreter_bin` prepends the directory to the child's inherited environment. This is what
  keeps the venv **first** when a profile merely appends to what it was given, and it is the whole
  fix for a subprocess spawned without a shell at all (an MCP stdio server, whose ``command`` is
  resolved against the ``PATH`` it is handed).
- `path_preamble` is one line of shell placed ahead of the model's command, so it runs *after* the
  profile has had its say. This is the half that holds **by construction**, on any POSIX box,
  whatever the profile does.

**The contract is *reachable*, not *first*.** Both halves are idempotent and match the directory
the same way — a literal, exact ``PATH`` entry — so a profile that kept the inherited prepend
costs no duplicate, and one that kept it *and moved it* is left as it is (macOS's ``path_helper``
does exactly this: measured, the directory came back present at position 12 of 20). Prepending
unconditionally would guarantee the lead at the price of a duplicate on every well-behaved box,
and re-ordering a ``PATH`` the box deliberately arranged is not this module's business. On the
common case — a profile that leaves ``PATH`` alone — the inherited prepend leaves it first anyway.
"""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path

_PATH = "PATH"


def interpreter_bin_dir() -> str | None:
    """The directory holding this interpreter's console scripts, or ``None`` if there is none.

    ``Path(sys.executable).parent`` — a venv's ``bin/`` when the harness runs from one, and the
    system ``bin`` when it does not. ``None`` in the three cases where there is nothing honest to
    add: an interpreter that reports no ``sys.executable`` at all (an embedded/frozen host), a
    directory that does not exist, and — the one worth naming — a directory whose name contains
    ``os.pathsep``, which simply *cannot* be represented as a ``PATH`` entry. Prepending it would
    not fail; it would silently split into two wrong entries.
    """
    executable = sys.executable
    if not executable:
        return None
    bin_dir = str(Path(executable).parent)
    if not bin_dir or os.pathsep in bin_dir:
        return None
    return bin_dir if os.path.isdir(bin_dir) else None


def with_interpreter_bin(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of `env` (default: this process's) with `interpreter_bin_dir` first on ``PATH``.

    Never mutates its argument, never removes an entry, and never reorders one that is already
    there: a ``PATH`` that already names the directory is returned untouched, so an activated venv
    or a deliberately-ordered ``PATH`` is not second-guessed. The match is exact string equality
    against the split entries — the same test `path_preamble` makes in shell, so the two agree.

    A missing ``PATH`` is filled with the directory plus ``os.defpath``, not with the directory
    alone: an env with no ``PATH`` resolves against ``os.defpath`` today, and replacing that with a
    one-entry ``PATH`` would take ``/bin`` and ``/usr/bin`` away from the child. The leading empty
    entry POSIX ``os.defpath`` carries (which means *the current directory*) is dropped rather than
    passed on.
    """
    result = dict(os.environ if env is None else env)
    bin_dir = interpreter_bin_dir()
    if bin_dir is None:
        return result
    current = result.get(_PATH)
    if current is None:
        result[_PATH] = os.pathsep.join([bin_dir, os.defpath.lstrip(os.pathsep)])
    elif bin_dir not in current.split(os.pathsep):
        result[_PATH] = os.pathsep.join([bin_dir, current]) if current else bin_dir
    return result


def path_preamble() -> str:
    """One line of POSIX shell that puts `interpreter_bin_dir` on ``PATH``, idempotently.

    Placed ahead of a model-authored command so it runs *after* the login shell has sourced the
    profile — the half of the fix that survives a profile which assigns ``PATH`` outright (see the
    module docstring). Empty string when there is nothing to add, so a caller splices it in only
    when it says something.

    The directory is held in a shell variable rather than written into the ``case`` pattern
    directly: a quoted expansion is matched **literally** in a pattern, so a path containing a glob
    character (``*``, ``?``, ``[``) cannot turn the presence test into a wildcard match. The
    variable is unset afterwards, so the model's command sees the shell it would have seen anyway.
    ``${PATH:+:$PATH}`` is what keeps an empty ``PATH`` from gaining a trailing empty entry, which
    a shell reads as the current directory.
    """
    bin_dir = interpreter_bin_dir()
    if bin_dir is None:
        return ""
    return (
        f"__bc_bin={shlex.quote(bin_dir)}; "
        'case ":$PATH:" in *":$__bc_bin:"*) ;; '
        '*) PATH="$__bc_bin${PATH:+:$PATH}"; export PATH ;; esac; '
        "unset __bc_bin"
    )
