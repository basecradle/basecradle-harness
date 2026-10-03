"""`scripts/isolated_home.py`: a run off the fleet leaves the real home as it found it (issue #630).

Every test here hands the wrapper a fabricated "real" home under ``tmp_path`` (``HOME`` is pinned
to it), so nothing reads or writes the developer's own. The real MemPalace half, that a palace
write lands its lock in the throwaway home, is ``test_isolated_home_real.py``.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

from tests.conftest import isolated_home

CHROMA = Path(".cache") / "chroma"


def _tree(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


@pytest.fixture
def real(tmp_path, monkeypatch):
    """A fabricated real home, standing in for the developer's."""
    home = tmp_path / "real"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_what_a_run_writes_to_home_goes_with_it(real, tmp_path):
    # The shape MemPalace writes on every palace write.
    record = tmp_path / "seen"
    code = (
        "import os, pathlib\n"
        "locks = pathlib.Path.home() / '.mempalace' / 'locks'\n"
        "locks.mkdir(parents=True)\n"
        "(locks / 'mine_palace_0123456789abcdef.lock').touch()\n"
        "(locks / '.last_reap').touch()\n"
        f"pathlib.Path({str(record)!r}).write_text(os.environ['HOME'])\n"
    )

    assert isolated_home.main(_python(code)) == 0

    used = Path(record.read_text())
    assert used != real
    assert not used.exists()
    assert _tree(real) == []


def test_the_shared_model_cache_is_linked_and_survives(real, tmp_path):
    model = real / CHROMA / "onnx_models" / "all-MiniLM-L6-v2" / "model.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")
    before = _tree(real)
    record = tmp_path / "seen"
    code = (
        "import pathlib\n"
        "cache = pathlib.Path.home() / '.cache' / 'chroma'\n"
        f"pathlib.Path({str(record)!r}).write_text(f'{{cache.is_symlink()}} '"
        " + (cache / 'onnx_models/all-MiniLM-L6-v2/model.onnx').read_text())\n"
    )

    assert isolated_home.main(_python(code)) == 0

    assert record.read_text() == "True weights"
    # Removing the throwaway home unlinked the link and left what it pointed to.
    assert _tree(real) == before
    assert model.read_bytes() == b"weights"


def test_no_cache_in_the_real_home_creates_none(real):
    with isolated_home.isolated_home() as home:
        assert _tree(home) == []
    assert _tree(real) == []


def test_only_the_named_caches_are_shared(real):
    (real / ".cache" / "uv").mkdir(parents=True)
    (real / CHROMA).mkdir()
    (real / ".mempalace" / "locks").mkdir(parents=True)

    with isolated_home.isolated_home() as home:
        assert _tree(home) == [".cache", str(CHROMA)]
        assert (home / CHROMA).resolve() == (real / CHROMA).resolve()


def test_the_rest_of_the_environment_passes_through(real, tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", "/srv/nova/harness")
    record = tmp_path / "seen"
    code = (
        f"import os, pathlib; pathlib.Path({str(record)!r}).write_text(os.environ['HARNESS_HOME'])"
    )

    assert isolated_home.main(_python(code)) == 0
    assert record.read_text() == "/srv/nova/harness"


def test_the_exit_status_is_the_commands(real):
    assert isolated_home.main(_python("import sys; sys.exit(3)")) == 3


def test_a_command_killed_by_a_signal_exits_as_a_shell_reports_it(real):
    code = f"import os; os.kill(os.getpid(), {int(signal.SIGTERM)})"
    assert isolated_home.main(_python(code)) == 128 + signal.SIGTERM


def test_a_missing_command_exits_127_and_still_cleans_up(real, capsys):
    assert isolated_home.main(["harness-no-such-command-630"]) == 127
    assert "command not found" in capsys.readouterr().err
    assert _tree(real) == []


def test_no_command_is_a_usage_error(real, capsys):
    assert isolated_home.main([]) == 2
    assert "isolated_home.py <command>" in capsys.readouterr().err


def test_mempalace_tests_run_in_a_throwaway_home(request):
    """The conftest isolation is keyed on the marker, so an unmarked test keeps its own ``HOME``."""
    assert request.node.get_closest_marker("mempalace") is None
    assert "harness-home-" not in os.environ["HOME"]
