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
from importlib.resources import files

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
from basecradle_harness._install import plugin_opts_in, plugin_source_providers
from basecradle_harness._shell import _ROOT_REFUSAL, _running_as_root


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
    """The shipped `_defaults/tools/shell.py` PLUGIN object, loaded from the package data file."""
    import importlib.util
    from importlib.resources import as_file, files

    src = files("basecradle_harness").joinpath("_defaults", "tools", "shell.py")
    with as_file(src) as path:
        spec = importlib.util.spec_from_file_location("_shipped_shell_plugin", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod.PLUGIN


def test_the_note_steers_to_scratch_and_workspace_over_assets():
    # Issue #263: a shell-equipped agent should keep working files in its own home, not on a
    # shared timeline. The note the model reads points it at ~/scratch and ~/workspace and
    # away from assets — pin it so a future edit can't silently drop the steer.
    note = _shipped_shell_plugin().note
    assert "~/scratch" in note
    assert "~/workspace" in note
    assert "Prefer them over timeline assets for anything not meant to be shared." in note


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
    assert "output truncated at 10 characters" in result
    assert "[exit code: 0]" in result


def test_a_flooding_producer_is_capped_and_killed_without_hanging_or_ooming(tmp_path):
    """The OOM guard: a runaway producer is stopped near the cap, not buffered unbounded.

    `yes` never terminates and floods stdout; if the tool read it all it would OOM the
    harness. Instead the reader stops just past the cap and the command is killed — so
    this returns promptly with a bounded string (the `timeout=30` would expose a hang).
    """
    tool = ShellTool(workdir=str(tmp_path), max_output=200, drain_timeout=2)
    result = tool.run(command="yes", timeout=30)
    assert "output exceeded the 200-character limit" in result
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
