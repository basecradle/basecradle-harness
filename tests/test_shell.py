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
from basecradle_harness._basecradle import _apply_safe_policy
from basecradle_harness._install import plugin_opts_in, plugin_source_providers


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
