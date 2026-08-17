"""The harness's own console scripts are reachable from the subprocesses it spawns.

The defect this pins (issue #409, found by the live acceptance run): a wake is launched by
absolute path, so nothing ever activates the agent's venv, so the agent's `shell` tool answered
``mempalace: command not found`` for a CLI sitting beside its own entry points. The fix derives
the directory from ``sys.executable`` at spawn time — never a config value, never a symlink into
a shared ``/usr/local/bin`` — and puts it on ``PATH`` two ways, because a login shell sources a
profile that may assign ``PATH`` outright and discard an inherited prepend.

The shell tests here run real ``/bin/bash``, exactly as the tool does, because the half of the
fix that matters most is one the environment can silently undo: a mocked shell would prove only
that the string was assembled, not that it survives a profile.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from basecradle_harness._venv import interpreter_bin_dir, path_preamble, with_interpreter_bin


def _bash(script: str, env: dict[str, str] | None = None) -> str:
    """Run `script` through a real ``/bin/bash`` and return its stdout, stripped."""
    done = subprocess.run(
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    return done.stdout.strip()


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """Point `sys.executable` at an interpreter inside a temp directory; return that directory."""
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python3"))
    return str(bin_dir)


# --- Deriving the directory --------------------------------------------------


def test_the_directory_is_the_running_interpreters_own():
    """Derived, never configured: it is wherever this interpreter lives, right now."""
    assert interpreter_bin_dir() == str(Path(sys.executable).parent)


def test_a_temp_interpreter_is_followed_without_any_configuration(fake_bin):
    assert interpreter_bin_dir() == fake_bin


def test_no_directory_without_a_sys_executable(monkeypatch):
    """An embedded/frozen host reports no interpreter path — there is nothing honest to add."""
    monkeypatch.setattr(sys, "executable", "")
    assert interpreter_bin_dir() is None


def test_no_directory_when_it_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "gone" / "python3"))
    assert interpreter_bin_dir() is None


def test_no_directory_when_its_name_cannot_be_a_path_entry(tmp_path, monkeypatch):
    """A directory containing the path separator cannot be one entry — it would split into two."""
    weird = tmp_path / f"we{os.pathsep}ird"
    weird.mkdir()
    monkeypatch.setattr(sys, "executable", str(weird / "python3"))
    assert interpreter_bin_dir() is None


# --- The inherited environment -----------------------------------------------


def test_the_directory_goes_first_on_path(fake_bin):
    env = with_interpreter_bin({"PATH": "/usr/bin:/bin"})
    assert env["PATH"] == f"{fake_bin}{os.pathsep}/usr/bin{os.pathsep}/bin"


def test_the_rest_of_the_environment_rides_along(fake_bin):
    env = with_interpreter_bin({"PATH": "/usr/bin", "HARNESS_HOME": "/home/nova/harness"})
    assert env["HARNESS_HOME"] == "/home/nova/harness"


def test_a_path_that_already_names_it_is_left_exactly_alone(fake_bin):
    """An activated venv, or a deliberately-ordered PATH, is not second-guessed or duplicated."""
    already = f"/usr/bin{os.pathsep}{fake_bin}{os.pathsep}/bin"
    assert with_interpreter_bin({"PATH": already})["PATH"] == already


def test_a_prefix_match_is_not_a_match(fake_bin):
    """Entries are compared whole: `/venv/bin-old` does not stand in for `/venv/bin`."""
    env = with_interpreter_bin({"PATH": f"{fake_bin}-old"})
    assert env["PATH"] == f"{fake_bin}{os.pathsep}{fake_bin}-old"


def test_the_callers_environment_is_never_mutated(fake_bin):
    original = {"PATH": "/usr/bin"}
    with_interpreter_bin(original)
    assert original == {"PATH": "/usr/bin"}


def test_a_missing_path_keeps_the_default_search_path(fake_bin):
    """Replacing an absent PATH with a one-entry one would take /bin and /usr/bin off the child."""
    env = with_interpreter_bin({"HOME": "/home/nova"})
    entries = env["PATH"].split(os.pathsep)
    assert entries[0] == fake_bin
    assert entries[1:] == os.defpath.lstrip(os.pathsep).split(os.pathsep)
    assert "" not in entries  # the leading empty entry means *the current directory*


def test_an_empty_path_becomes_just_the_directory(fake_bin):
    assert with_interpreter_bin({"PATH": ""})["PATH"] == fake_bin


def test_it_defaults_to_this_processs_own_environment(fake_bin, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HARNESS_HOME", "/home/nova/harness")
    env = with_interpreter_bin()
    assert env["PATH"] == f"{fake_bin}{os.pathsep}/usr/bin"
    assert env["HARNESS_HOME"] == "/home/nova/harness"


def test_nothing_changes_when_there_is_no_directory_to_add(monkeypatch):
    monkeypatch.setattr(sys, "executable", "")
    assert with_interpreter_bin({"PATH": "/usr/bin"}) == {"PATH": "/usr/bin"}


# --- The shell prelude -------------------------------------------------------


def test_the_prelude_survives_a_profile_that_reassigns_path(fake_bin):
    """The whole reason the prelude exists: /etc/profile assigns PATH on some distributions."""
    after = _bash(f'PATH=/usr/bin:/bin\n{path_preamble()}\necho "$PATH"')
    assert after.split(os.pathsep)[0] == fake_bin


def test_the_prelude_runs_after_a_real_login_shell_sourced_its_profile(fake_bin):
    """``-lc`` is what the shell tool actually runs, and the profile is sourced before the command.

    Driven with a ``PATH`` that does *not* name the directory, so whatever this box's profile does
    to ``PATH`` — Debian assigns it, macOS's ``path_helper`` rebuilds it — the prelude still wins.
    """
    env = {"PATH": f"/usr/bin{os.pathsep}/bin", "HOME": os.environ.get("HOME", "/tmp")}
    done = subprocess.run(
        ["/bin/bash", "-lc", f'{path_preamble()}\necho "$PATH"'],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert done.stdout.strip().split(os.pathsep)[0] == fake_bin


def test_the_prelude_keeps_what_the_profile_left(fake_bin):
    after = _bash(f'PATH=/usr/bin:/bin\n{path_preamble()}\necho "$PATH"')
    assert after.split(os.pathsep)[1:] == ["/usr/bin", "/bin"]


def test_the_prelude_does_not_duplicate_an_entry_that_survived(fake_bin):
    """The inherited prepend and the prelude agree on the match, so a kind profile costs nothing."""
    env = with_interpreter_bin({"PATH": "/usr/bin"})
    after = _bash(f'{path_preamble()}\necho "$PATH"', env=env)
    assert after.split(os.pathsep).count(fake_bin) == 1


def test_the_prelude_leaves_no_variable_behind_for_the_command(fake_bin):
    assert _bash(f'{path_preamble()}\necho "[${{__bc_bin-unset}}]"') == "[unset]"


def test_the_prelude_adds_no_empty_entry_to_an_empty_path(fake_bin):
    """An empty PATH entry is *the current directory* — a working directory on the search path."""
    assert _bash(f'PATH=\n{path_preamble()}\necho "$PATH"') == fake_bin


def test_a_glob_character_in_the_directory_is_matched_literally(tmp_path, monkeypatch):
    """Held in a shell variable, so a `?` in the path cannot turn the presence test into a match."""
    bin_dir = tmp_path / "a?c"
    bin_dir.mkdir()
    decoy = tmp_path / "abc"
    decoy.mkdir()
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python3"))
    after = _bash(f'PATH={decoy}\n{path_preamble()}\necho "$PATH"')
    assert after == f"{bin_dir}{os.pathsep}{decoy}"


def test_a_command_placed_there_resolves_by_bare_name(fake_bin):
    """The acceptance shape, in miniature: a script that exists *only* in the venv bin resolves."""
    script = Path(fake_bin) / "nova-cli"
    script.write_text("#!/bin/sh\necho hello\n")
    script.chmod(0o755)
    assert _bash(f"{path_preamble()}\ncommand -v nova-cli", env={"PATH": "/usr/bin:/bin"}) == str(
        script
    )


def test_there_is_no_prelude_when_there_is_nothing_to_add(monkeypatch):
    """An empty string, so a caller splices it in only when it says something."""
    monkeypatch.setattr(sys, "executable", "")
    assert path_preamble() == ""
