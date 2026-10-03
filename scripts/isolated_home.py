#!/usr/bin/env python3
"""Run one command under a throwaway ``$HOME``, so a run off the fleet leaves the real home alone.

MemPalace writes outside the palace it is given. Every write to a palace (an observe, a mine, a
collection ``add``) takes MemPalace's ``mine_palace_lock``, which keeps one lock file per palace,
``mine_palace_<hash of the palace path>.lock``, plus a ``.last_reap`` marker, in
``$HOME/.mempalace/locks``, and never removes it. Upstream offers no setting for that directory: it
is ``os.path.expanduser("~")``, read on every call. On a fleet box that is right, because ``~`` is
the agent's own home and the lock guards its one palace for the life of the account. On a laptop
every synthetic palace has a fresh temporary path, so every run that writes one adds a lock file
to the developer's home that nothing will ever remove (issue #630). A local wake writes there too:
it publishes its palace path into ``~/.mempalace/config.json`` (issue #409), which on a laptop is
the developer's own file.

The fix is the premise the harness already runs on: an agent runs in its own home. This wrapper
makes a temporary directory, runs the command with ``HOME`` pointed at it, and removes it when the
command ends, so every write to ``~`` that MemPalace or the harness makes lands there and goes with
it. Use it for every run that writes a palace off the fleet, a synthetic palace check and a local
wake alike::

    uv run scripts/isolated_home.py basecradle-harness-palace-check --practice-observe <home>
    uv run scripts/isolated_home.py basecradle-harness-wake --timeline <uuid>

``uv run`` goes first, deliberately: uv keeps its caches and its Python installs under ``$HOME``,
and under a fresh one it would fetch all of them again.

**One thing is shared, and it is a download cache.** MemPalace's default embedding model (ChromaDB's
all-MiniLM-L6-v2, about 80 MB) is downloaded to ``~/.cache/chroma`` on first use, and a fresh home
would download it on every run. So when the real home already holds that cache, the temporary
home links to it (`SHARED_CACHES`): a link, never a copy, and only to a directory that already
exists, so the wrapper creates nothing in the real home. Removing the temporary home removes the
link and never what it points to, because ``shutil.rmtree`` unlinks a symlink rather than
following it.

Every other variable passes through unchanged, so one that names a path explicitly
(``HARNESS_HOME``, ``BASECRADLE_CONFIG_HOME``, ``MEMPALACE_PALACE_PATH``) still wins over the
temporary ``~``. The exit status is the command's own, and a command killed by a signal exits
``128 + signal``, as a shell reports it. The test suite runs every test marked ``mempalace`` under
the same `isolated_home` (``tests/conftest.py``).

Usage:
    python scripts/isolated_home.py <command> [args...]
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

#: Download caches the temporary home links to when the real home holds them. Only caches: a
#: model a run would otherwise fetch again, never state a run writes.
SHARED_CACHES = (Path(".cache") / "chroma",)


@contextlib.contextmanager
def isolated_home(real_home: Path | None = None) -> Iterator[Path]:
    """A temporary home that links `SHARED_CACHES` from ``real_home``, removed on exit."""
    real = Path.home() if real_home is None else real_home
    with tempfile.TemporaryDirectory(prefix="harness-home-") as raw:
        home = Path(raw)
        for cache in SHARED_CACHES:
            source = real / cache
            if source.is_dir():
                link = home / cache
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(source, target_is_directory=True)
        yield home


def main(argv: list[str] | None = None) -> int:
    command = sys.argv[1:] if argv is None else argv
    if not command:
        print(__doc__.rsplit("Usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    with isolated_home() as home:
        try:
            status = subprocess.run(
                command, env={**os.environ, "HOME": str(home)}, check=False
            ).returncode
        except FileNotFoundError:
            print(f"isolated_home: command not found: {command[0]}", file=sys.stderr)
            return 127
        except KeyboardInterrupt:
            return 130
    return 128 - status if status < 0 else status


if __name__ == "__main__":
    sys.exit(main())
