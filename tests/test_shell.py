"""The shell tool: full command-line access, opt-in, unlocked-profile-only.

The headline invariant is the **double gate** — the one tool in the kit that needs
*both* safety gates to clear, so a single oversight can never arm it:

1. it declares ``requires = {SHELL}``, so the locked policy refuses it (and the
   env-path filter `_apply_safe_policy` drops it and surfaces the refusal), and
2. its plugin is ``opt_in``, so it is off by default and off the packaged fallback.

Both are asserted here, plus the tool's own behavior: it runs a command, reports the
exit code, does not raise on failure, merges stderr, kills on timeout, bounds memory on
a flooding producer, truncates huge output, and honors ``workdir``. Every command here is
harmless (``echo``, ``pwd``, ``ls``, ``exit``, ``sleep``, ``yes``, ``head``); the behavior
tests pin ``workdir`` to a temp dir so they never depend on a valid ``$HOME``.
"""

import os
import sys
import types
from importlib.resources import files
from pathlib import Path

import pytest

from basecradle_harness import (
    SHELL,
    ActivationContext,
    Policy,
    PolicyError,
    ResolvedTools,
    ShellTool,
    ToolRegistry,
    install,
    load_plugins,
    resolve_plugins,
)
from basecradle_harness._basecradle import _apply_safe_policy, _profile_from_env
from basecradle_harness._brief import render_your_home, your_home_text
from basecradle_harness._install import plugin_opts_in, plugin_source_providers
from basecradle_harness._shell import _ROOT_REFUSAL, _running_as_root, _with_path
from basecradle_harness._venv import commands_dir, own_commands, path_preamble


@pytest.fixture
def shell(tmp_path):
    """A `ShellTool` whose default workdir is a temp dir — hermetic, no `$HOME` reliance."""
    return ShellTool(workdir=str(tmp_path))


def _shipped_shell_source() -> str:
    """The packaged ``_defaults/tools/shell.py`` manifest source, read without importing it."""
    return files("basecradle_harness").joinpath("_defaults", "tools", "shell.py").read_text()


# --- The tool contract -------------------------------------------------------


def test_declares_the_shell_capability():
    assert ShellTool().requires == frozenset({SHELL})


def test_name_and_required_parameter():
    tool = ShellTool()
    assert tool.name == "shell"
    assert tool.parameters["required"] == ["command"]
    assert set(tool.parameters["properties"]) == {"command", "timeout", "workdir"}


def _shipped_shell_plugin():
    """The shipped `_defaults/tools/shell.py` PLUGIN object, loaded the way the plugin loader does.

    Through `_plugins._import_file`, which suppresses bytecode: a plain `exec_module` leaves a
    `__pycache__/*.pyc` inside the package's `_defaults/` tree, which the installer walks as text.
    """
    from importlib.resources import as_file, files

    from basecradle_harness._plugins import _import_file

    with as_file(files("basecradle_harness").joinpath("_defaults", "tools", "shell.py")) as path:
        return _import_file(path).PLUGIN


def test_the_note_is_the_text_the_handoff_specified():
    # Issue #571, verbatim: the folders are taught in one place ("Your Home"), so the note shrinks
    # to a pointer. Pinned whole, because a pointer that drifts from the section's heading points
    # at nothing.
    assert _shipped_shell_plugin().note == (
        "Full shell as your OS user — unlocked-profile only. Runs arbitrary commands, code, and "
        "network calls with no sandbox beyond your Unix permissions. Your home directory and its "
        'seven standing folders are described under "Your Home" in your instructions; each '
        "folder's README is its law. Your harness's own commands (`basecradle-harness-*`, and "
        "`mempalace` where it is installed) go first on your PATH, from a directory that holds "
        "nothing else, whenever the harness can make that directory — run them by name "
        "(`mempalace status` reads your own memory palace, if that is your backend). If one of "
        "them is not found by name, the harness could not put it on your PATH for that command. "
        "Your harness's own Python environment, if it has one, is never on your PATH: `python3`, "
        "`pip` and every other command resolve to the system's or your own."
    )


def test_the_home_steer_the_note_used_to_carry_reaches_every_shell_agent():
    # Issue #263: a shell-equipped agent should keep working files in its own home, not on a
    # shared timeline. Issue #571 moved that steer out of this note into "Your Home", so pin the
    # two facts that keep it reaching the model: the section says it, and the section is composed
    # for exactly the tool that carries this note.
    assert "Prefer these folders over timeline assets for anything not meant to be shared." in (
        your_home_text()
    )
    assert '"Your Home"' in _shipped_shell_plugin().note
    assert render_your_home([ShellTool()]).startswith("## Your Home\n")


def test_the_note_says_which_commands_are_on_its_path_and_which_environment_is_not():
    # Issue #409: putting them on PATH is worth nothing if the agent never learns they are
    # there. Issue #969: the note must also be true about what left — the harness's environment,
    # so the agent does not go looking for the harness's `pip`.
    note = _shipped_shell_plugin().note
    assert "first on your PATH" in note
    assert "mempalace" in note
    assert "never on your PATH" in note


def test_the_note_is_true_when_the_commands_directory_could_not_be_made():
    # Issue #971: when the directory cannot be made the harness's commands are not on the PATH at
    # all, so the note conditions the claim, and says what a name not found means without claiming
    # more: one link can fail while the others are made, and the directory is retried at every
    # spawn, so it speaks of that command, never of the whole wake. A model told a false thing
    # about its own tools acts on it.
    note = _shipped_shell_plugin().note
    assert "whenever the harness can make that directory" in note
    assert "could not put it on your PATH for that command" in note
    assert "come first on your PATH" not in note
    assert "this wake" not in note


# --- The policy boundary (the real, shipped tool) ----------------------------


def test_locked_policy_refuses_the_real_shell_tool():
    """The headline safety invariant, against the real tool — not a test stub."""
    with pytest.raises(PolicyError, match="shell"):
        ToolRegistry(policy=Policy.locked()).register(ShellTool())


def test_default_registry_is_locked_and_refuses_shell():
    with pytest.raises(PolicyError):
        ToolRegistry().register(ShellTool())


def test_unlocked_policy_admits_and_runs_the_real_shell_tool(tmp_path):
    """The unlocked profile: the same registry, an unlocked policy — it loads and runs."""
    registry = ToolRegistry(policy=Policy.unlocked())
    registry.register(ShellTool(workdir=str(tmp_path)))
    result = registry.run("shell", command="echo hello")
    assert "hello" in result
    assert "[exit code: 0]" in result


def test_policy_permits_is_capability_disjointness():
    assert Policy.locked().permits(ShellTool()) is False
    assert Policy.unlocked().permits(ShellTool()) is True


def test_env_path_filters_shell_under_locked_and_surfaces_the_refusal():
    """On the env-resolution path a forbidden tool self-excludes — it never crashes the wake."""
    resolved = ResolvedTools(tools=[ShellTool()], manifest=[("shell", "note")])

    filtered = _apply_safe_policy(resolved)  # defaults to the locked policy

    assert all(tool.name != "shell" for tool in filtered.tools)
    assert all(name != "shell" for name, _ in filtered.manifest)
    assert any(name == "shell" for name, _ in filtered.skipped)
    assert any("shell" in notice for notice in filtered.notices)


def test_a_policy_refused_tool_takes_its_env_dependency_with_it():
    """A refused tool is not read at run time, so its credential leaves the report too (#427).

    Without the prune, ``--resolved-config`` would name an environment variable as read by a tool
    set that does not contain the tool that reads it — sending an operator to provision a key for
    a capability the profile has refused. Pruned by name alongside the manifest, in both the wake's
    filter and `basecradle-harness-resolve`'s.
    """
    resolved = ResolvedTools(
        tools=[ShellTool()],
        manifest=[("shell", "note")],
        env_dependencies={"shell": ["SOME_KEY"], "memory": ["KEPT_KEY"]},
    )

    filtered = _apply_safe_policy(resolved)  # locked: shell is refused

    assert filtered.env_dependencies == {"memory": ["KEPT_KEY"]}


def test_env_path_keeps_shell_under_unlocked():
    resolved = ResolvedTools(tools=[ShellTool()], manifest=[("shell", "note")])

    filtered = _apply_safe_policy(resolved, Policy.unlocked())

    assert any(tool.name == "shell" for tool in filtered.tools)


# --- The deploy profile selector: HARNESS_PROFILE (issue #256) ----------------
#
# The one deploy lever that reaches `Policy.unlocked()` at wake, delivered per-agent via
# `agent.env`. Fail-closed to locked so a typo, an empty value, or an unset var never silently
# unlocks a box; the same pure decision drives both the registry and the env-resolution filter.


def test_profile_unlocked_selects_the_unlocked_policy(monkeypatch):
    monkeypatch.setenv("HARNESS_PROFILE", "unlocked")
    name, policy = _profile_from_env()
    assert name == "unlocked"
    assert policy.forbidden == frozenset()  # nothing forbidden — the shell is admitted
    assert policy.permits(ShellTool()) is True


def test_profile_unlocked_is_trimmed_and_case_insensitive(monkeypatch):
    monkeypatch.setenv("HARNESS_PROFILE", "  UnLocked  ")
    name, policy = _profile_from_env()
    assert name == "unlocked"
    assert policy.permits(ShellTool()) is True


@pytest.mark.parametrize(
    "value", [None, "", "   ", "locked", "LOCKED", "on", "unlock", "true", "1"]
)
def test_profile_fails_closed_to_locked(monkeypatch, value):
    """Unset, empty, the explicit `locked`, or any unrecognized token → the safe locked default."""
    if value is None:
        monkeypatch.delenv("HARNESS_PROFILE", raising=False)
    else:
        monkeypatch.setenv("HARNESS_PROFILE", value)
    name, policy = _profile_from_env()
    assert name == "locked"
    assert policy.forbidden == frozenset({SHELL})
    assert policy.permits(ShellTool()) is False


# --- The root backstop: refuse to run as root (issue #253) -------------------
#
# The constitution's in-process privilege guard (basecradle#404): the tool's whole safety
# model is the unprivileged OS user, so as root it refuses — fail-closed at load *and* run.
# The euid is injected via `os.geteuid` so these are deterministic regardless of the uid
# the test process actually runs as (CI and dev are both non-root, so the behavior tests
# above already cover the unprivileged case; here we drive both sides explicitly).


def test_running_as_root_reads_the_effective_uid(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    assert _running_as_root() is True
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert _running_as_root() is False


def test_running_as_root_is_false_without_geteuid(monkeypatch):
    """A host without ``os.geteuid`` (Windows) has no Unix root to detect — reads not-root."""
    monkeypatch.delattr(os, "geteuid", raising=False)
    assert _running_as_root() is False


def test_load_refusal_fires_only_as_root(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    refusal = ShellTool().load_refusal()
    assert refusal is not None
    assert "root" in refusal
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert ShellTool().load_refusal() is None


def test_register_refuses_shell_as_root_even_under_the_unlocked_profile(monkeypatch):
    """The backstop is independent of the policy gate: the unlocked profile normally admits
    shell, but at root the tool refuses to load — a root-run agent never even sees it."""
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    registry = ToolRegistry(policy=Policy.unlocked())
    with pytest.raises(PolicyError, match="root"):
        registry.register(ShellTool())
    assert "shell" not in registry


def test_run_refuses_as_root_instead_of_executing(monkeypatch, tmp_path):
    """Defense-in-depth: a tool constructed and called directly (bypassing the registry)
    still refuses at root — it returns the refusal rather than running the command."""
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    result = ShellTool(workdir=str(tmp_path)).run(command="echo hello")
    assert _ROOT_REFUSAL in result
    assert "hello" not in result
    assert "[exit code:" not in result


def test_env_path_drops_shell_as_root_and_surfaces_it(monkeypatch):
    """On the env-resolution path the root refusal self-excludes and surfaces — never crashes
    the wake — even under the unlocked profile the policy would otherwise admit it under."""
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    resolved = ResolvedTools(tools=[ShellTool()], manifest=[("shell", "note")])

    filtered = _apply_safe_policy(resolved, Policy.unlocked())

    assert all(tool.name != "shell" for tool in filtered.tools)
    assert all(name != "shell" for name, _ in filtered.manifest)
    assert any(name == "shell" for name, _ in filtered.skipped)
    assert any("shell" in notice and "root" in notice for notice in filtered.notices)


def test_load_and_run_are_normal_as_an_unprivileged_user(monkeypatch, shell):
    """The paired normal case, pinned deterministically: at a non-root euid the tool loads
    under the unlocked profile and runs commands exactly as before."""
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert shell.load_refusal() is None
    registry = ToolRegistry(policy=Policy.unlocked())
    registry.register(shell)  # the fixture's ShellTool has a temp-dir workdir
    result = registry.run("shell", command="echo hello")
    assert "hello" in result
    assert "[exit code: 0]" in result


# --- Behavior ----------------------------------------------------------------


def test_runs_a_command_and_reports_output_and_exit_code(shell):
    result = shell.run(command="echo hello world")
    assert "hello world" in result
    assert "[exit code: 0]" in result


def test_a_nonzero_exit_is_reported_not_raised(shell):
    result = shell.run(command="exit 3")
    assert "[exit code: 3]" in result


def test_stderr_is_merged_into_the_output(shell):
    result = shell.run(command="echo oops >&2")
    assert "oops" in result
    assert "[exit code: 0]" in result


def test_a_command_with_no_output_says_so(shell):
    result = shell.run(command="true")
    assert "(no output)" in result
    assert "[exit code: 0]" in result


def test_full_shell_syntax_works(shell):
    """A login shell runs pipes, redirects, and '&&' — real shell syntax, not exec of a list."""
    result = shell.run(command="echo one && echo two | tr a-z A-Z")
    assert "one" in result
    assert "TWO" in result


def test_a_timeout_kills_the_command_and_reports_it(shell):
    result = shell.run(command="sleep 5", timeout=1)
    assert "timed out after 1s" in result
    assert "[exit code:" not in result


def test_a_caller_timeout_is_clamped_to_the_hard_max(tmp_path):
    """A huge caller timeout cannot disable the guard — it is clamped to `max_timeout`."""
    tool = ShellTool(workdir=str(tmp_path), max_timeout=1)
    result = tool.run(command="sleep 5", timeout=999)
    assert "timed out after 1s" in result


def test_a_finite_command_over_the_cap_is_truncated_with_a_marker(tmp_path):
    """A command that exceeds the cap but exits on its own → truncated + real exit code."""
    tool = ShellTool(workdir=str(tmp_path), max_output=10)
    result = tool.run(command="printf 'abcdefghijklmnop'")
    assert "abcdefghij" in result
    assert "[output cut at 10 characters]" in result
    assert "[exit code: 0]" in result


def test_a_flooding_producer_is_capped_and_killed_without_hanging_or_ooming(tmp_path):
    """The OOM guard: a runaway producer is stopped near the cap, not buffered unbounded.

    `yes` never terminates and floods stdout; if the tool read it all it would OOM the
    harness. Instead the reader stops just past the cap and the command is killed — so
    this returns promptly with a bounded string (the `timeout=30` would expose a hang).
    """
    tool = ShellTool(workdir=str(tmp_path), max_output=200, drain_timeout=2)
    result = tool.run(command="yes", timeout=30)
    assert result.startswith("y\ny\n")  # the head is returned, never nothing (issue #892)
    assert "[output cut at 200 characters: the command was still writing" in result
    assert "timed out" not in result  # capped, not timed out
    assert "[exit code:" not in result  # killed, no clean exit
    assert len(result) < 400  # bounded, not gigabytes


def test_workdir_is_honored(tmp_path):
    (tmp_path / "marker.txt").write_text("hi")
    result = ShellTool().run(command="ls", workdir=str(tmp_path))
    assert "marker.txt" in result


def test_a_missing_workdir_is_a_clean_error_not_a_crash():
    result = ShellTool().run(command="pwd", workdir="/no/such/dir")
    assert "is not a directory" in result


def test_an_empty_command_is_a_clean_error():
    assert "needs a 'command'" in ShellTool().run(command="   ")


def test_binary_output_does_not_crash_the_tool(shell):
    """Undecodable bytes are replaced, not fatal — a command that emits binary still returns."""
    result = shell.run(command="head -c 16 /dev/urandom")
    assert "[exit code: 0]" in result


# --- Only the harness's own commands on PATH, first (issues #409, #969) --------
#
# The acceptance test the decision named, run through the real tool: a real `/bin/bash -lc`, a
# home whose login profile puts `~/.local/bin` first (a fleet server's shape), and an inherited
# PATH that carries the harness's environment `bin` (the shape #409's fix produced, and the shape
# `uv run` gives this suite).

_NOT_OURS = ("python", "python3", "pip", "pip3", "httpx")


@pytest.fixture
def agent(tmp_path, monkeypatch):
    """A harness environment, a fleet-shaped home, and an inherited PATH naming that environment."""
    root = tmp_path / "venv"
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    for name in (*own_commands(), *_NOT_OURS):
        script = bin_dir / name
        script.write_text(f'#!/bin/sh\necho {name}-from-the-harness "$@"\n')
        script.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python3"))
    monkeypatch.setattr(sys, "prefix", str(root))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "system"))
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    (home / ".bash_profile").write_text('PATH="$HOME/.local/bin:$PATH"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), "/usr/bin", "/bin"]))
    return types.SimpleNamespace(
        bin=bin_dir, local_bin=local_bin, tool=ShellTool(workdir=str(tmp_path))
    )


def _output(result: str) -> str:
    """The command's own output, without the tool's ``[exit code: N]`` footer."""
    return result.rsplit("[exit code:", 1)[0].strip()


def test_python3_and_pip3_are_not_the_harnesss(agent):
    for name in ("python3", "pip3", "pip", "python"):
        assert str(agent.bin) not in agent.tool.run(command=f"command -v {name} || true")


def test_mempalace_resolves_from_the_commands_directory_and_works_bare(agent):
    assert _output(agent.tool.run(command="command -v mempalace")) == str(
        Path(commands_dir()) / "mempalace"
    )
    assert _output(agent.tool.run(command="mempalace status")) == (
        "mempalace-from-the-harness status"
    )


def test_every_harness_console_script_resolves_from_the_commands_directory(agent):
    names = [name for name in own_commands() if name.startswith("basecradle-harness-")]
    assert names
    listing = agent.tool.run(command=" && ".join(f"command -v {name}" for name in names))
    assert _output(listing).splitlines() == [str(Path(commands_dir()) / name) for name in names]


def test_a_link_in_local_bin_named_after_a_harness_command_does_not_win(agent):
    impostor = agent.local_bin / "basecradle-harness-verify"
    impostor.write_text("#!/bin/sh\necho impostor\n")
    impostor.chmod(0o755)
    assert _output(agent.tool.run(command="basecradle-harness-verify")) == (
        "basecradle-harness-verify-from-the-harness"
    )


def test_a_program_in_local_bin_named_like_a_dependency_script_wins(agent):
    """The agent's own `httpx` is the agent's: a dependency's script no longer shadows it."""
    own = agent.local_bin / "httpx"
    own.write_text("#!/bin/sh\necho the-agents-own\n")
    own.chmod(0o755)
    assert _output(agent.tool.run(command="httpx")) == "the-agents-own"


def test_the_order_is_commands_then_the_profile_then_the_system(agent):
    entries = _output(agent.tool.run(command='echo "$PATH"')).split(os.pathsep)
    assert entries[:2] == [commands_dir(), str(agent.local_bin)]
    assert str(agent.bin) not in entries
    assert "/usr/bin" in entries


def test_a_new_venv_cannot_see_the_harnesss_packages(tmp_path, monkeypatch):
    """The real environment this suite runs from, on the inherited PATH: `python3 -m venv` is clean.

    `uv run` puts this suite's own ``.venv/bin`` first, which is exactly #409's shape. Before #969
    the ``python3`` there was the harness's, and a venv made from it was the harness's interpreter.
    """
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"])
    )
    made = tmp_path / "fresh"
    result = ShellTool(workdir=str(tmp_path)).run(
        command=(
            f"python3 -m venv --without-pip {made} && "
            f'{made}/bin/python -c "import basecradle_harness" 2>&1 | tail -1'
        ),
        timeout=120,
    )
    assert "ModuleNotFoundError" in result, result


def test_the_harnesss_own_entry_point_resolves_by_bare_name(tmp_path, monkeypatch):
    """The real entry point, from the real venv this suite runs in, under a stripped PATH.

    The shape of the wake the router actually launches, which activates nothing. Without the
    stripped PATH the test would pass on the developer's activated shell and prove nothing.
    """
    monkeypatch.setenv("PATH", f"/usr/bin{os.pathsep}/bin")
    wake = Path(sys.executable).with_name("basecradle-harness-wake")
    assert wake.exists(), "run the suite from the venv the package is installed into"

    result = ShellTool(workdir=str(tmp_path)).run(command="basecradle-harness-wake --version")
    assert "[exit code: 0]" in result
    assert _output(
        ShellTool(workdir=str(tmp_path)).run(command="command -v basecradle-harness-wake")
    ) == str(Path(commands_dir()) / "basecradle-harness-wake")


def test_the_prelude_precedes_the_model_command():
    """One line, spliced ahead — it must run *after* the login shell has sourced the profile."""
    assembled = _with_path("echo hello")
    assert assembled == f"{path_preamble()}\necho hello"
    assert assembled.count("\n") == 1


def test_the_command_is_untouched_when_there_is_nothing_to_remove_or_add(monkeypatch):
    monkeypatch.setattr(sys, "executable", "")
    assert _with_path("echo hello") == "echo hello"


# --- The plugin: opt-in, provider-agnostic, double-gated ---------------------


def test_shipped_shell_plugin_is_opt_in():
    """It fails closed: off by default on every provider (issue #168)."""
    assert plugin_opts_in(_shipped_shell_source()) is True


def test_shipped_shell_plugin_is_provider_agnostic():
    """A shell is an OS capability, not a provider one — no `Vendor`/`OpenAIKey` affinity."""
    assert plugin_source_providers(_shipped_shell_source()) is None


def test_shell_is_not_scaffolded_for_a_fresh_agent(tmp_path):
    install(tmp_path / "cfg")
    assert not (tmp_path / "cfg" / "tools" / "shell.py").exists()


def test_shell_is_scaffolded_only_when_explicitly_opted_in(tmp_path):
    home = tmp_path / "cfg"
    install(home, opt_in=["shell"])
    assert (home / "tools" / "shell.py").exists()


def test_opted_in_shell_activates_on_any_provider_and_lists_its_inventory_stem(tmp_path):
    """Opted in, shell activates even on OpenRouter (glm-5.2's provider) and reports its stem."""
    home = tmp_path / "cfg"
    install(home, provider="openrouter", opt_in=["shell"])

    plugins = load_plugins(home, provider="openrouter")
    ctx = ActivationContext(
        provider="openrouter", sdk="openrouter", surface="chat", model="glm-5.2", env={}
    )
    resolved = resolve_plugins(plugins, ctx)

    assert any(tool.name == "shell" for tool in resolved.tools)
    assert "shell" in resolved.opt_in_stems
