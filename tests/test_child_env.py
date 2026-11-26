"""The suite, and every child process it spawns, import the code under test, never an install.

In CI the installed harness is this checkout, so a test importing the wrong one still passes; the
gap shows only when a suite runs from one checkout against another install (issue #1016). These pin
both halves: the suite's own import (``pythonpath`` in ``pyproject.toml``) and `child_env`, the one
helper every test spawns a Python child through.
"""

import os
import subprocess
import sys
from pathlib import Path

import basecradle_harness
from tests.conftest import REPO, SRC_ROOT, child_env

PACKAGE = SRC_ROOT / "basecradle_harness"


def _imported(env: dict[str, str]) -> Path:
    """Where a child spawned under `env`, outside the checkout, finds the harness and `tests`."""
    probe = "import basecradle_harness, tests; print(basecradle_harness.__file__)"
    done = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        cwd="/",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return Path(done.stdout.strip()).resolve()


def test_the_suite_imports_this_checkouts_harness():
    assert Path(basecradle_harness.__file__).resolve().is_relative_to(PACKAGE)


def test_a_child_imports_this_checkouts_harness_first():
    env = child_env()

    assert env["PYTHONPATH"].split(os.pathsep)[:2] == [str(SRC_ROOT), str(REPO)]
    assert _imported(env).is_relative_to(PACKAGE)


def test_the_checkout_comes_before_an_inherited_path(monkeypatch, tmp_path):
    # A harness of another version first on the inherited path: the checkout still shadows it.
    other = tmp_path / "basecradle_harness"
    other.mkdir()
    (other / "__init__.py").write_text("raise RuntimeError('the other harness was imported')\n")
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}{os.pathsep}")

    env = child_env()

    # The stray separator's empty entry is dropped: it would put the child's cwd on its path.
    assert env["PYTHONPATH"].split(os.pathsep) == [str(SRC_ROOT), str(REPO), str(tmp_path)]
    assert _imported(env).is_relative_to(PACKAGE)


def test_extra_variables_ride_beside_the_inherited_environment(monkeypatch):
    monkeypatch.setenv("HARNESS_EXAMPLE", "inherited")

    env = child_env(READY_FD="7")

    assert (env["READY_FD"], env["NO_COLOR"], env["HARNESS_EXAMPLE"]) == ("7", "1", "inherited")
