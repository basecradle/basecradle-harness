"""A real palace write leaves its MemPalace lock in the throwaway home, never the real one (issue #630).

The conftest isolation (`_isolated_mempalace_home`) rests on one fact about MemPalace: it keeps its
per-palace lock in ``$HOME/.mempalace/locks``, read from ``HOME`` on every call. If upstream ever
moves that directory (to a config dir, or a path resolved once at import), the isolation stops
working without a sound, and this test is what says so. It drives the harness's own observe, the
write every wake makes.

Marked ``mempalace`` and excluded from the default run: it needs the ``mempalace`` extra and the
embedding model it downloads on first use. Run it with ``uv run --extra mempalace pytest -m
mempalace``.
"""

from __future__ import annotations

import os
import pwd
from pathlib import Path

import pytest

pytestmark = pytest.mark.mempalace

pytest.importorskip("mempalace")

from basecradle_harness._memory_provider import MemoryExchange, MemoryScope  # noqa: E402
from basecradle_harness._mempalace import MemPalaceMemoryProvider  # noqa: E402


def test_an_observe_locks_in_the_throwaway_home(tmp_path, _mempalace_home):
    assert Path(os.environ["HOME"]) == _mempalace_home
    locks = _mempalace_home / ".mempalace" / "locks"
    before = set(locks.glob("mine_palace_*.lock"))

    MemPalaceMemoryProvider(tmp_path / "palace").observe(
        MemoryExchange(
            user="John Doe says his birthday is May 4.",
            assistant="Noted, John.",
            scope=MemoryScope(agent="nova", timeline="t1", query=None),
        )
    )

    created = set(locks.glob("mine_palace_*.lock")) - before
    assert len(created) == 1
    # The account's own home, which `HOME` no longer names: the lock is not there.
    real = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".mempalace" / "locks"
    assert not (real / created.pop().name).exists()
