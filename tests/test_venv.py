"""Only the harness's own commands on the PATH of what it spawns; its environment leaves it.

The contract (issue #969, `@origin`, 2026-11-21), which replaced #409's "reachable, not first":
only the harness's own console scripts plus ``mempalace`` are on the PATH, from a directory that
holds nothing else, placed first, and the environment the harness runs from — its ``python3``, its
``pip``, every dependency's script — is taken off. Derived from the running interpreter at spawn,
never configured.

The shell tests run real ``/bin/bash``, exactly as the tool does, because the half of the fix that
matters most is one a profile can silently undo: a mocked shell would prove only that a string was
assembled, not that it survives a profile.
"""

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from basecradle_harness import _venv
from basecradle_harness._venv import (
    MEMPALACE,
    commands_dir,
    interpreter_bin_dir,
    own_commands,
    path_preamble,
    spawn_env,
)

#: What a harness environment's ``bin`` holds besides the harness's own commands.
_NOT_OURS = ("python", "python3", "pip", "pip3", "httpx", "pygmentize", "mempalace-mcp")


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


def _script(path: Path, text: str) -> Path:
    path.write_text(f"#!/bin/sh\necho {text}\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def venv(tmp_path, monkeypatch):
    """A harness environment: every allowlisted command, plus what is not ours, in one ``bin``."""
    root = tmp_path / "venv"
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    for name in (*own_commands(), *_NOT_OURS):
        _script(bin_dir / name, f"{name}-from-the-harness")
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python3"))
    monkeypatch.setattr(sys, "prefix", str(root))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "system"))
    return bin_dir


def _entries(path: str) -> list[str]:
    return path.split(os.pathsep)


# --- Deriving the interpreter's directory ------------------------------------


def test_the_directory_is_the_running_interpreters_own():
    """Derived, never configured: it is wherever this interpreter lives, right now."""
    assert interpreter_bin_dir() == str(Path(sys.executable).parent)


def test_a_temp_interpreter_is_followed_without_any_configuration(venv):
    assert interpreter_bin_dir() == str(venv)


def test_no_directory_without_a_sys_executable(monkeypatch):
    """An embedded/frozen host reports no interpreter path — there is nothing honest to use."""
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


# --- The allowlist -------------------------------------------------------------


def test_the_allowlist_is_the_distributions_console_scripts_plus_mempalace():
    """Read from the installed metadata — the record the scripts were written from, so no list."""
    from importlib.metadata import distribution

    scripts = {
        point.name
        for point in distribution("basecradle-harness").entry_points
        if point.group == "console_scripts"
    }
    assert set(own_commands()) == scripts | {MEMPALACE}
    assert "basecradle-harness-wake" in own_commands()


def test_nothing_of_the_environments_own_is_on_the_allowlist():
    """Not the interpreter, not pip, no dependency's script, not the MCP servers started by path."""
    assert not set(own_commands()) & {*_NOT_OURS, "mempalace-light-mcp"}


# --- The commands directory ----------------------------------------------------


def test_the_directory_holds_links_to_the_allowlist_and_nothing_else(venv):
    made = Path(commands_dir())
    assert sorted(p.name for p in made.iterdir()) == sorted(own_commands())
    for link in made.iterdir():
        assert os.readlink(link) == str(venv / link.name)


def test_the_directory_is_private_to_this_user(venv):
    assert Path(commands_dir()).stat().st_mode & 0o777 == 0o700


def test_a_name_the_environment_lacks_is_not_linked(venv):
    """``mempalace`` is linked only where the extra put it in ``bin``."""
    (venv / MEMPALACE).unlink()
    assert MEMPALACE not in os.listdir(commands_dir())


def test_a_command_gained_under_a_running_process_is_linked_at_the_next_spawn(venv):
    (venv / MEMPALACE).unlink()
    first = commands_dir()
    _script(venv / MEMPALACE, "late")
    assert commands_dir() == first
    assert MEMPALACE in os.listdir(first)


def test_one_directory_per_process(venv):
    assert commands_dir() == commands_dir()


def test_a_directory_removed_under_the_process_is_made_again(venv):
    import shutil

    shutil.rmtree(commands_dir())
    again = commands_dir()
    assert again is not None and sorted(os.listdir(again)) == sorted(own_commands())


def test_no_directory_when_the_environment_holds_none_of_our_commands(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "python3", "x")
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python3"))
    assert commands_dir() is None


def test_a_directory_that_cannot_be_made_is_one_warning_and_no_directory(venv, monkeypatch, caplog):
    monkeypatch.setattr(_venv, "_PARENT", str(venv.parent / "missing" / "parent"))
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        assert commands_dir() is None
        assert commands_dir() is None
    warnings = [
        r for r in caplog.records if "harness commands are not all on the PATH" in r.message
    ]
    assert len(warnings) == 1


def test_losing_the_directory_never_costs_the_removal(venv, monkeypatch):
    """The removal is the property the decision is about; the reach is the convenience."""
    monkeypatch.setattr(_venv, "_PARENT", str(venv.parent / "missing"))
    env = spawn_env({"PATH": f"{venv}{os.pathsep}/usr/bin"})
    assert env["PATH"] == "/usr/bin"


def test_one_name_that_cannot_be_linked_costs_that_name_only(venv, caplog):
    made = Path(commands_dir())
    (made / "basecradle-harness-wake").unlink()
    (made / "basecradle-harness-wake").mkdir()  # nothing can replace a directory with a link
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        assert commands_dir() == str(made)
    assert os.readlink(made / "basecradle-harness-verify") == str(
        venv / "basecradle-harness-verify"
    )
    assert any("basecradle-harness-wake could not be linked" in r.message for r in caplog.records)


def test_a_live_process_touches_its_directory_on_every_spawn(venv):
    """So its age tells the sweep whether a process uses it, not only its pid."""
    made = commands_dir()
    stamp = time.time() - _venv.STRANDED_AFTER * 2
    os.utime(made, (stamp, stamp))
    commands_dir()
    assert time.time() - os.stat(made).st_mtime < 60


def test_this_process_removes_its_own_directory_at_exit(venv):
    made = commands_dir()
    _venv._remove_owned()
    assert not os.path.exists(made)


def test_a_forked_childs_exit_never_removes_its_parents_directory(venv, monkeypatch):
    made = commands_dir()
    real = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: real + 1)
    _venv._remove_owned()
    assert os.path.isdir(made)


# --- Stranded directories: whatever creates, cleans up ---------------------------


def _stranded(parent: Path, pid: int, *, age: float) -> Path:
    path = parent / f"basecradle-harness-commands-{pid}-abc123"
    path.mkdir()
    (path / "basecradle-harness-wake").symlink_to("/nowhere")
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    return path


def _dead_pid() -> int:
    done = subprocess.run(["/bin/sh", "-c", "echo $$"], capture_output=True, text=True, check=True)
    return int(done.stdout)


def test_a_dead_processs_old_directory_is_removed_by_the_next_one(venv):
    parent = Path(_venv._PARENT)
    gone = _stranded(parent, _dead_pid(), age=_venv.STRANDED_AFTER + 60)
    commands_dir()
    assert not gone.exists()


def test_a_live_processs_directory_is_kept_however_old(venv):
    parent = Path(_venv._PARENT)
    live = subprocess.Popen(["/bin/sleep", "30"])
    try:
        kept = _stranded(parent, live.pid, age=_venv.STRANDED_AFTER * 24)
        commands_dir()
        assert kept.exists()
    finally:
        live.kill()
        live.wait()


def test_age_is_never_the_reason_by_itself_nor_a_dead_pid(venv):
    """A dead pid's directory younger than the threshold is kept: both must say the writer is gone."""
    parent = Path(_venv._PARENT)
    young = _stranded(parent, _dead_pid(), age=60)
    commands_dir()
    assert young.exists()


def test_another_users_directory_is_never_removed(venv, monkeypatch):
    parent = Path(_venv._PARENT)
    theirs = _stranded(parent, _dead_pid(), age=_venv.STRANDED_AFTER * 2)
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(theirs).st_uid + 1)
    commands_dir()
    assert theirs.exists()


def test_only_our_own_name_is_ever_removed(venv):
    parent = Path(_venv._PARENT)
    theirs = parent / f"basecradle-harness-commandsX-{_dead_pid()}-abc"
    theirs.mkdir()
    stamp = time.time() - _venv.STRANDED_AFTER * 2
    os.utime(theirs, (stamp, stamp))
    commands_dir()
    assert theirs.exists()


def test_a_link_by_our_name_is_never_followed_or_removed(venv, tmp_path):
    parent = Path(_venv._PARENT)
    target = tmp_path / "theirs"
    target.mkdir()
    link = parent / f"basecradle-harness-commands-{_dead_pid()}-abc"
    link.symlink_to(target)
    commands_dir()
    assert link.is_symlink() and target.is_dir()


# --- The environment a child is handed -------------------------------------------


def test_the_environment_bin_leaves_and_the_commands_go_first(venv):
    env = spawn_env({"PATH": f"{venv}{os.pathsep}/usr/bin{os.pathsep}/bin"})
    assert _entries(env["PATH"]) == [commands_dir(), "/usr/bin", "/bin"]


def test_every_spelling_of_the_environment_bin_leaves(venv, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(venv)
    path = os.pathsep.join([f"{venv}{os.sep}", "/usr/bin", str(alias), str(venv)])
    assert _entries(spawn_env({"PATH": path})["PATH"]) == [commands_dir(), "/usr/bin"]


def test_the_resolved_spelling_leaves_when_the_interpreter_is_reached_by_a_link(
    venv, tmp_path, monkeypatch
):
    alias = tmp_path / "linked"
    alias.symlink_to(venv.parent)
    monkeypatch.setattr(sys, "executable", str(alias / "bin" / "python3"))
    env = spawn_env({"PATH": f"{venv}{os.pathsep}/usr/bin"})
    assert _entries(env["PATH"]) == [commands_dir(), "/usr/bin"]
    after = _bash(f'PATH={venv}:/usr/bin\n{path_preamble({"PATH": "/usr/bin"})}\necho "$PATH"')
    assert _entries(after) == [commands_dir(), "/usr/bin"]


def test_a_prefix_match_is_not_a_match(venv):
    """Entries are compared whole: `/venv/bin-old` does not stand in for `/venv/bin`."""
    env = spawn_env({"PATH": f"{venv}-old"})
    assert _entries(env["PATH"]) == [commands_dir(), f"{venv}-old"]


def test_every_other_entry_keeps_its_place_empty_ones_included(venv):
    path = os.pathsep.join(["/a", "", str(venv), "/b", ""])
    assert _entries(spawn_env({"PATH": path})["PATH"]) == [commands_dir(), "/a", "", "/b", ""]


def test_the_commands_directory_is_never_duplicated(venv):
    first = commands_dir()
    env = spawn_env({"PATH": f"/usr/bin{os.pathsep}{first}"})
    assert _entries(env["PATH"]) == [first, "/usr/bin"]


def test_the_rest_of_the_environment_rides_along(venv):
    env = spawn_env({"PATH": "/usr/bin", "HARNESS_HOME": "/home/nova/harness"})
    assert env["HARNESS_HOME"] == "/home/nova/harness"


def test_the_callers_environment_is_never_mutated(venv):
    original = {"PATH": f"{venv}{os.pathsep}/usr/bin"}
    spawn_env(original)
    assert original == {"PATH": f"{venv}{os.pathsep}/usr/bin"}


def test_a_missing_path_keeps_the_default_search_path(venv):
    """Replacing an absent PATH with a one-entry one would take /bin and /usr/bin off the child."""
    entries = _entries(spawn_env({"HOME": "/home/nova"})["PATH"])
    assert entries[0] == commands_dir()
    assert entries[1:] == os.defpath.lstrip(os.pathsep).split(os.pathsep)


def test_an_empty_path_becomes_just_the_commands_directory(venv):
    assert spawn_env({"PATH": ""})["PATH"] == commands_dir()


def test_it_defaults_to_this_processs_own_environment(venv, monkeypatch):
    monkeypatch.setenv("PATH", f"{venv}{os.pathsep}/usr/bin")
    assert _entries(spawn_env()["PATH"]) == [commands_dir(), "/usr/bin"]


def test_a_child_that_runs_nothing_of_ours_loses_the_environment_and_gains_nothing(venv):
    """``journalctl``: no directory is made for it, and the harness's ``bin`` still leaves."""
    env = spawn_env({"PATH": f"{venv}{os.pathsep}/usr/bin"}, commands=False)
    assert env["PATH"] == "/usr/bin"
    assert list(Path(_venv._PARENT).iterdir()) == []


def test_a_child_that_runs_nothing_of_ours_keeps_a_missing_path_missing(venv):
    assert "PATH" not in spawn_env({"HOME": "/home/nova"}, commands=False)


def test_outside_a_virtual_environment_nothing_is_removed(venv, monkeypatch):
    """There, the interpreter's ``bin`` is the system's own; removing it would break the box."""
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    env = spawn_env({"PATH": f"{venv}{os.pathsep}/usr/bin"})
    assert _entries(env["PATH"]) == [commands_dir(), str(venv), "/usr/bin"]


def test_nothing_changes_when_there_is_nothing_to_remove_or_add(monkeypatch):
    monkeypatch.setattr(sys, "executable", "")
    assert spawn_env({"PATH": "/usr/bin"}) == {"PATH": "/usr/bin"}


# --- The shell prelude -------------------------------------------------------------


def test_the_prelude_survives_a_profile_that_reassigns_path(venv):
    """The whole reason the prelude exists: /etc/profile assigns PATH on some distributions."""
    after = _bash(f'PATH={venv}:/usr/bin:/bin\n{path_preamble()}\necho "$PATH"')
    assert _entries(after) == [commands_dir(), "/usr/bin", "/bin"]


def test_the_prelude_runs_after_a_real_login_shell_sourced_its_profile(venv, tmp_path):
    """``-lc`` is what the shell tool runs; a profile putting ``~/.local/bin`` first is a fleet box's.

    The order the command sees is the decision's: the commands directory, what the profile built,
    then the system — and the environment's ``bin``, which the profile put back, is gone again.
    """
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".bash_profile").write_text(f'PATH="$HOME/.local/bin:{venv}:$PATH"\n')
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home)}
    done = subprocess.run(
        ["/bin/bash", "-lc", f'{path_preamble(env)}\necho "$PATH"'],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    entries = _entries(done.stdout.strip())
    assert entries[:2] == [commands_dir(), str(home / ".local" / "bin")]
    assert str(venv) not in entries


def test_the_prelude_keeps_every_other_entry_in_place_empty_ones_included(venv):
    after = _bash(f'PATH=/a::{venv}:/b:\n{path_preamble()}\necho "$PATH"')
    assert _entries(after) == [commands_dir(), "/a", "", "/b", ""]


def test_the_prelude_keeps_a_leading_empty_entry_when_it_only_removes(venv, monkeypatch):
    monkeypatch.setattr(_venv, "_PARENT", str(venv.parent / "missing"))
    after = _bash(f'PATH=:{venv}:/b\n{path_preamble()}\necho "[$PATH]"')
    assert after == "[:/b]"


def test_the_prelude_and_the_environment_agree(venv):
    inherited = {"PATH": f"{venv}{os.pathsep}/usr/bin{os.pathsep}/bin"}
    env = spawn_env(inherited)
    assert _bash(f'{path_preamble(inherited)}\necho "$PATH"', env=env) == env["PATH"]


def test_the_prelude_leaves_no_variable_behind_for_the_command(venv):
    names = ("__bc_c", "__bc_b", "__bc_p", "__bc_r", "__bc_e", "__bc_k")
    script = f'{path_preamble()}\necho "[{"".join(f"${{{name}-u}}" for name in names)}]"'
    assert _bash(script) == "[uuuuuu]"


def test_a_path_the_profile_unset_becomes_the_default_search_path_under_set_u(venv):
    """As `spawn_env` treats a missing ``PATH``: ``/bin`` and ``/usr/bin`` are not taken away."""
    after = _bash(f'set -u\nunset PATH\n{path_preamble()}\necho "$PATH"')
    assert _entries(after) == [commands_dir(), *os.defpath.lstrip(os.pathsep).split(os.pathsep)]


def test_a_spelling_the_profile_adds_is_removed_too(venv, tmp_path):
    """No string computed before the profile ran names it, so the prelude compares inodes."""
    alias = tmp_path / "current"
    alias.symlink_to(venv.parent)
    dotted = f"{venv.parent}/./bin"
    after = _bash(f'PATH={alias}/bin:{dotted}:/usr/bin\n{path_preamble()}\necho "$PATH"')
    assert _entries(after) == [commands_dir(), "/usr/bin"]


def test_the_inode_comparison_never_removes_another_directory(venv, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    after = _bash(f'PATH={other}:/usr/bin\n{path_preamble()}\necho "$PATH"')
    assert _entries(after) == [commands_dir(), str(other), "/usr/bin"]


def test_the_prelude_adds_no_empty_entry_to_an_empty_path(venv):
    """An empty PATH entry is *the current directory* — a working directory on the search path."""
    assert _bash(f'PATH=\n{path_preamble()}\necho "$PATH"') == commands_dir()


def test_a_glob_character_in_a_removed_directory_is_matched_literally(tmp_path, monkeypatch):
    """Quoted in the pattern, so a `?` in the path cannot remove a directory it only resembles."""
    bin_dir = tmp_path / "a?c"
    bin_dir.mkdir()
    _script(bin_dir / "basecradle-harness-wake", "x")
    decoy = tmp_path / "abc"
    decoy.mkdir()
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python3"))
    monkeypatch.setattr(sys, "base_prefix", "/nowhere")
    after = _bash(f'PATH={decoy}:{bin_dir}\n{path_preamble({"PATH": str(bin_dir)})}\necho "$PATH"')
    assert _entries(after) == [commands_dir(), str(decoy)]


def test_there_is_no_prelude_when_there_is_nothing_to_remove_or_add(monkeypatch):
    """An empty string, so a caller splices it in only when it says something."""
    monkeypatch.setattr(sys, "executable", "")
    assert path_preamble() == ""


# --- What points an installer at the environment (issue #971) ----------------------

#: The interpreter running these tests, captured before any test points `sys.executable` elsewhere.
_REAL_PYTHON = sys.executable


@pytest.fixture
def real_venv(tmp_path, monkeypatch):
    """A real virtual environment an installer accepts, made the running harness's own."""
    root = tmp_path / "harness-venv"
    subprocess.run([_REAL_PYTHON, "-m", "venv", "--without-pip", str(root)], check=True)
    monkeypatch.setattr(sys, "executable", str(root / "bin" / "python3"))
    monkeypatch.setattr(sys, "prefix", str(root))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "system"))
    return root


def _uv() -> str:
    """The `uv` binary; the suite's own toolchain, so CI must have it (skipped only off CI)."""
    uv = shutil.which("uv")
    if uv is None:
        if os.environ.get("CI"):
            pytest.fail("uv is not on the PATH in CI; the installer tests cannot run")
        pytest.skip("uv is not on the PATH")
    return uv


def _uv_target(args: list[str], env: dict[str, str], cwd: Path) -> str:
    """Run a real, offline `uv` dry run under `env`; return everything it printed."""
    done = subprocess.run(
        [_uv(), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=60,
        check=False,
    )
    return done.stdout + done.stderr


def _named(output: str, root: Path, cwd: Path) -> bool:
    """Did `uv` say it would use (or create) the environment at `root`, in any spelling?

    A relative path uv prints is resolved against the directory uv ran in, never pytest's own.
    """
    for line in output.splitlines():
        if "environment at: " in line:
            named = cwd / line.split("environment at: ", 1)[1].strip()
            if os.path.normpath(named) == os.path.normpath(root) or (
                os.path.exists(named) and os.path.samefile(named, root)
            ):
                return True
    return False


def _bare_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """A minimal environment for `uv`: no inherited `UV_*`, no config file, its own cache, the
    system's PATH, and never a download."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(work),
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "UV_NO_CONFIG": "1",
        "UV_PYTHON_DOWNLOADS": "never",
        **extra,
    }


@pytest.mark.parametrize("variable", ["VIRTUAL_ENV", "CONDA_PREFIX", "UV_PYTHON"])
def test_a_child_cannot_uv_pip_install_into_the_harness_through_an_inherited_variable(
    real_venv, tmp_path, variable
):
    """A real `uv pip install --dry-run`: inherited as is, it targets the harness; handed on, not.

    The first half proves the vector is real (measured, not assumed), the second that it is closed.
    """
    value = str(real_venv / "bin" / "python3") if variable == "UV_PYTHON" else str(real_venv)
    inherited = _bare_env(tmp_path, **{variable: value})
    args = ["pip", "install", "--dry-run", "--offline", "--no-index", "six"]
    work = tmp_path / "work"
    assert _named(_uv_target(args, inherited, work), real_venv, work)
    child = spawn_env(inherited)
    assert variable not in child
    output = _uv_target(args, child, work)
    assert not _named(output, real_venv, work)
    assert "No virtual environment found" in output


def test_a_child_cannot_uv_sync_into_the_harness_through_uv_project_environment(
    real_venv, tmp_path
):
    project = tmp_path / "work"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "p"\nversion = "0"\nrequires-python = ">=3.11"\ndependencies = []\n'
    )
    # An interpreter of the suite's own, which is not the harness's here, so uv can always resolve
    # and the second half shows where it *would* go rather than that it could not start.
    inherited = _bare_env(tmp_path, UV_PROJECT_ENVIRONMENT=str(real_venv), UV_PYTHON=_REAL_PYTHON)
    args = ["sync", "--dry-run", "--offline"]
    assert _named(_uv_target(args, inherited, project), real_venv, project)
    child = spawn_env(inherited)
    assert child["UV_PYTHON"] == _REAL_PYTHON
    output = _uv_target(args, child, project)
    assert _named(output, project / ".venv", project)
    assert not _named(output, real_venv, project)


def test_an_operators_own_virtual_env_is_kept_and_used(real_venv, tmp_path):
    """A VIRTUAL_ENV that names anything else is the operator's own: handed on, and installed into."""
    theirs = tmp_path / "theirs"
    subprocess.run([_REAL_PYTHON, "-m", "venv", "--without-pip", str(theirs)], check=True)
    child = spawn_env(_bare_env(tmp_path, VIRTUAL_ENV=str(theirs)))
    assert child["VIRTUAL_ENV"] == str(theirs)
    args = ["pip", "install", "--dry-run", "--offline", "--no-index", "six"]
    assert _named(_uv_target(args, child, tmp_path / "work"), theirs, tmp_path / "work")


@pytest.mark.parametrize(
    "variable",
    ["VIRTUAL_ENV", "CONDA_PREFIX", "UV_PROJECT_ENVIRONMENT", "PIP_PREFIX", "PYTHONUSERBASE"],
)
def test_every_spelling_of_the_environment_root_leaves(venv, tmp_path, variable):
    """Matched by what it is: the root as derived, with a trailing separator, or by a link."""
    root = venv.parent
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    for spelling in (str(root), f"{root}/", str(alias), f"{root}/bin/.."):
        assert variable not in spawn_env({"PATH": "/usr/bin", variable: spelling})


@pytest.mark.parametrize("variable", ["UV_PYTHON", "PIP_PYTHON", "PIP_TARGET"])
def test_a_path_inside_the_environment_leaves(venv, tmp_path, variable):
    """An interpreter or a target directory inside it — even one not made yet — names it."""
    for inside in (venv / "python3", venv.parent / "lib" / "python3.13" / "site-packages"):
        assert variable not in spawn_env({"PATH": "/usr/bin", variable: str(inside)})


def test_the_environments_interpreter_counts_by_where_it_sits_not_what_it_resolves_to(
    venv, tmp_path
):
    """A venv's `bin/python` is a link to the base interpreter; resolving it would escape."""
    base = tmp_path / "base-python"
    base.write_text("")
    link = venv / "python-link"
    link.symlink_to(base)
    assert "UV_PYTHON" not in spawn_env({"PATH": "/usr/bin", "UV_PYTHON": str(link)})


def test_a_link_elsewhere_that_leads_into_the_environment_counts_by_where_it_leads(venv, tmp_path):
    """`pip --target /x/link` follows the link and writes into the harness's site-packages."""
    site = venv.parent / "lib" / "site-packages"
    site.mkdir(parents=True)
    link = tmp_path / "link-to-site"
    link.symlink_to(site)
    for value in (str(link), str(link / "not-yet")):
        assert "PIP_TARGET" not in spawn_env({"PIP_TARGET": value})


def test_a_path_that_climbs_out_of_the_environment_is_not_inside_it(venv, tmp_path):
    """`<root>/../other/bin/python` names another environment; its spelling only passes through."""
    other = tmp_path / "other" / "bin"
    other.mkdir(parents=True)
    value = f"{venv.parent}/../other/bin/python3"
    assert spawn_env({"UV_PYTHON": value})["UV_PYTHON"] == value


def test_a_root_variable_naming_a_directory_inside_the_environment_is_kept(venv):
    """`VIRTUAL_ENV` names an environment's root; a directory inside the harness's is not it."""
    nested = str(venv.parent / "lib")
    assert spawn_env({"VIRTUAL_ENV": nested})["VIRTUAL_ENV"] == nested


@pytest.mark.parametrize(
    "value",
    ["3.12", "cpython@3.13", "venv/bin/python", ""],
    ids=["version", "request", "relative", "empty"],
)
def test_a_value_that_names_no_absolute_place_is_never_judged(venv, value):
    for variable in ("VIRTUAL_ENV", "UV_PYTHON", "PIP_TARGET"):
        assert spawn_env({variable: value})[variable] == value


def test_a_prefix_of_the_environments_name_is_not_the_environment(venv, tmp_path):
    lookalike = tmp_path / "venv-other"
    lookalike.mkdir()
    child = spawn_env({"VIRTUAL_ENV": str(lookalike), "PIP_TARGET": str(lookalike / "x")})
    assert child["VIRTUAL_ENV"] == str(lookalike)
    assert child["PIP_TARGET"] == str(lookalike / "x")


def test_the_unmeasured_variables_are_left_alone(venv):
    """`PYTHONHOME`, `PYTHONPATH` and `PIP_ROOT` lead no install into it (see `_venv`)."""
    root = str(venv.parent)
    names = {"PYTHONHOME": root, "PYTHONPATH": f"{root}/lib", "PIP_ROOT": root}
    assert {name: spawn_env(names)[name] for name in names} == names


def test_outside_a_virtual_environment_no_installer_variable_is_removed(venv, monkeypatch):
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    assert spawn_env({"VIRTUAL_ENV": sys.prefix})["VIRTUAL_ENV"] == sys.prefix


def test_a_child_that_runs_nothing_of_ours_loses_the_installer_variables_too(venv):
    child = spawn_env({"VIRTUAL_ENV": str(venv.parent)}, commands=False)
    assert "VIRTUAL_ENV" not in child


# --- What the shell tool tells the model holds in both cases -----------------------


def test_without_the_commands_directory_a_harness_command_is_not_found_by_name(venv, monkeypatch):
    """What the note says of a name not found: the harness could not put it on the PATH."""
    monkeypatch.setattr(_venv, "_PARENT", str(venv.parent / "missing"))
    child = spawn_env({"PATH": f"{venv}:/usr/bin:/bin"})
    found = _bash(f"{path_preamble(child)}\ncommand -v basecradle-harness-verify", env=child)
    assert found == ""


def test_with_the_commands_directory_a_harness_command_is_found_there(venv):
    child = spawn_env({"PATH": f"{venv}:/usr/bin:/bin"})
    found = _bash(f"{path_preamble(child)}\ncommand -v basecradle-harness-verify", env=child)
    assert found == os.path.join(commands_dir(), "basecradle-harness-verify")
