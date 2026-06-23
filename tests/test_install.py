"""The config home: installer + conffile upgrader + file-sourced charter (Phase 2 · Group 1).

Everything here is offline — the installer only ever touches the filesystem, so these
tests need no respx and no model. They pin the four upgrade cases the conffile discipline
must get right (refresh pristine / keep edited as ``.new`` / respect a deletion / never
touch an operator-added file), the scaffold + manifest, and the charter now being sourced
from real files instead of an env var.

The shipped defaults are reconciled through an injectable ``defaults`` seam, so an
"upgrade" is simulated by reconciling a *changed* default set against an existing install —
no second build, no package surgery.
"""

import json

from basecradle_harness import config_home, install, installed_version, reconcile_on_upgrade
from basecradle_harness._install import (
    _MANIFEST_NAME,
    _VERSION_NAME,
    ALREADY_CURRENT,
    INSTALLED,
    KEPT_DELETED,
    KEPT_EDITED,
    PRUNED,
    REFRESHED,
    UNCHANGED,
    charter_from_config,
    charter_from_env,
    main,
    plugin_relevant_to,
    plugin_source_providers,
    prompt_text,
    system_prompt_text,
)
from basecradle_harness._version import __version__

# Synthetic default sets, used to drive the upgrader's four cases deterministically: v1 is
# what we "ship" first, v2 changes one file so the reconcile has a genuine default change to
# act on. A nested path proves the relpath/dir handling.
V1 = {"prompts/system-prompt.md": "v1 charter\n", "tools/notes.md": "v1 notes\n"}
V2 = {"prompts/system-prompt.md": "v2 charter\n", "tools/notes.md": "v1 notes\n"}


# --- scaffold + manifest -----------------------------------------------------


def test_install_scaffolds_the_config_home_and_writes_shipped_defaults(tmp_path):
    home = tmp_path / "cfg"

    report = install(home)

    # The three dirs exist, created and reported.
    for name in ("prompts", "tools", "mcp"):
        assert (home / name).is_dir()
    assert sorted(report.created_dirs) == ["mcp", "prompts", "tools"]
    # The shipped charter defaults are real files, copied out (not a magic in-package fallback).
    assert (
        home / "prompts" / "system-prompt.md"
    ).read_text() == "You are a helpful peer on BaseCradle.\n"
    assert (home / "prompts" / "initialize.md").exists()
    # Every shipped default was a fresh install on a first run.
    assert set(report.of(INSTALLED)) == set(report.actions)
    assert report.config_home == home


def test_install_copies_the_shipped_tool_plugin_defaults_into_tools(tmp_path):
    # Group 2 ships the default tools as real plugin files under `_defaults/tools/`; the
    # installer copies them into the config home's `tools/` overlay (and the upgrader manages
    # them with the same conffile discipline as the prompt defaults — proven generically by
    # the four-case tests below using a `tools/` path).
    home = tmp_path / "cfg"

    install(home)

    tools = home / "tools"
    for name in ("web_fetch.py", "generate_image.py", "web_search.py"):
        assert (tools / name).exists()
    # The tool defaults are manifest-tracked, which is also the "tools are installed" signal
    # `load_plugins` keys on to treat the overlay as authoritative.
    manifest = json.loads((home / _MANIFEST_NAME).read_text())
    assert any(key.startswith("tools/") for key in manifest)


def test_install_records_every_shipped_default_hash_in_the_manifest(tmp_path):
    home = tmp_path / "cfg"
    install(home)

    manifest = json.loads((home / _MANIFEST_NAME).read_text())

    # The manifest keys are exactly the shipped default relpaths, each mapped to a hash.
    assert "prompts/system-prompt.md" in manifest
    assert "prompts/initialize.md" in manifest
    assert all(isinstance(h, str) and len(h) == 64 for h in manifest.values())


def test_install_is_idempotent(tmp_path):
    home = tmp_path / "cfg"
    install(home)
    before = (home / "prompts" / "system-prompt.md").read_text()

    report = install(home)  # re-run against the same package

    assert set(report.of(UNCHANGED)) == set(report.actions)  # nothing to do
    assert report.created_dirs == []  # dirs already there
    assert report.new_files == []  # no .new churn
    assert (home / "prompts" / "system-prompt.md").read_text() == before


# --- the four conffile upgrade cases -----------------------------------------


def test_upgrade_refreshes_an_untouched_default(tmp_path):
    """Case 1: a pristine default is replaced with the new one."""
    home = tmp_path / "cfg"
    install(home, defaults=V1)

    report = install(home, defaults=V2)

    assert report.actions["prompts/system-prompt.md"] == REFRESHED
    assert (home / "prompts" / "system-prompt.md").read_text() == "v2 charter\n"
    assert not (home / "prompts" / "system-prompt.md.new").exists()
    # The file v2 left unchanged is a no-op, not a needless rewrite.
    assert report.actions["tools/notes.md"] == UNCHANGED


def test_upgrade_keeps_an_edited_file_and_drops_the_new_default_beside_it(tmp_path):
    """Case 2: an operator edit is kept; the new default lands as ``<name>.new``."""
    home = tmp_path / "cfg"
    install(home, defaults=V1)
    edited = home / "prompts" / "system-prompt.md"
    edited.write_text("MY hand-tuned charter\n")  # operator edits it

    report = install(home, defaults=V2)

    assert report.actions["prompts/system-prompt.md"] == KEPT_EDITED
    assert edited.read_text() == "MY hand-tuned charter\n"  # theirs, untouched
    assert (home / "prompts" / "system-prompt.md.new").read_text() == "v2 charter\n"
    assert report.new_files == ["prompts/system-prompt.md"]


def test_upgrade_respects_a_deleted_file_and_does_not_resurrect_it(tmp_path):
    """Case 3: a file the operator deleted stays deleted."""
    home = tmp_path / "cfg"
    install(home, defaults=V1)
    (home / "prompts" / "system-prompt.md").unlink()  # operator deletes it

    report = install(home, defaults=V2)

    assert report.actions["prompts/system-prompt.md"] == KEPT_DELETED
    assert not (home / "prompts" / "system-prompt.md").exists()
    assert not (home / "prompts" / "system-prompt.md.new").exists()


def test_upgrade_never_touches_an_operator_added_file(tmp_path):
    """Case 4: a file that is not a shipped default is never seen, written, or recorded."""
    home = tmp_path / "cfg"
    install(home, defaults=V1)
    mine = home / "tools" / "my_tool.py"
    mine.write_text("# operator's own tool\n")

    report = install(home, defaults=V2)

    assert mine.read_text() == "# operator's own tool\n"  # untouched
    assert "tools/my_tool.py" not in report.actions  # never even visited
    manifest = json.loads((home / _MANIFEST_NAME).read_text())
    assert "tools/my_tool.py" not in manifest  # never recorded


def test_upgrade_is_a_no_op_when_the_operator_already_has_the_new_default(tmp_path):
    """A file the operator independently set to the new content needs no ``.new``."""
    home = tmp_path / "cfg"
    install(home, defaults=V1)
    (home / "prompts" / "system-prompt.md").write_text("v2 charter\n")  # matches v2 already

    report = install(home, defaults=V2)

    assert report.actions["prompts/system-prompt.md"] == ALREADY_CURRENT
    assert not (home / "prompts" / "system-prompt.md.new").exists()


def test_rerunning_the_same_version_over_an_edit_is_a_no_op(tmp_path):
    """An edit + a re-run of the *same* package must not churn out a redundant ``.new``."""
    home = tmp_path / "cfg"
    install(home, defaults=V1)
    (home / "prompts" / "system-prompt.md").write_text("MY charter\n")  # operator edits

    report = install(home, defaults=V1)  # same version again — no new default to offer

    assert report.actions["prompts/system-prompt.md"] == UNCHANGED
    assert not (home / "prompts" / "system-prompt.md.new").exists()


def test_an_edited_file_is_re_offered_on_each_genuinely_new_default_version(tmp_path):
    """The manifest records the *current* shipped default, so a later version re-offers ``.new``.

    Locks the subtle invariant behind ``updated[rel] = new`` being recorded unconditionally:
    after an edit is kept against v2, a genuinely different v3 default must still produce a
    fresh ``.new`` (the edit is never silently stranded on a stale baseline).
    """
    home = tmp_path / "cfg"
    v1 = {"prompts/system-prompt.md": "v1\n"}
    v2 = {"prompts/system-prompt.md": "v2\n"}
    v3 = {"prompts/system-prompt.md": "v3\n"}
    install(home, defaults=v1)
    edited = home / "prompts" / "system-prompt.md"
    edited.write_text("MINE\n")

    r2 = install(home, defaults=v2)
    assert r2.actions["prompts/system-prompt.md"] == KEPT_EDITED
    assert (home / "prompts" / "system-prompt.md.new").read_text() == "v2\n"

    r3 = install(home, defaults=v3)
    assert r3.actions["prompts/system-prompt.md"] == KEPT_EDITED
    assert edited.read_text() == "MINE\n"  # the edit is still kept verbatim
    assert (home / "prompts" / "system-prompt.md.new").read_text() == "v3\n"  # re-offered, fresh


# --- version stamp + upgrade reconcile (issue #160) --------------------------


def test_install_stamps_the_harness_version(tmp_path):
    home = tmp_path / "cfg"
    install(home, defaults=V1)
    assert (home / _VERSION_NAME).read_text().strip() == __version__
    assert installed_version(home) == __version__


def test_installed_version_is_none_when_never_installed(tmp_path):
    # No install has run → no stamp → unknown (reads as "needs reconciling on next check").
    assert installed_version(tmp_path / "never") is None


def test_reconcile_on_upgrade_is_a_no_op_when_not_installed(tmp_path):
    # A never-installed home runs off the packaged-default fallback — nothing materialized to
    # go stale, so the reconcile must NOT auto-create a config home and flip it onto the overlay.
    home = tmp_path / "never"
    assert reconcile_on_upgrade(home) is None
    assert not home.exists()


def test_reconcile_on_upgrade_is_a_no_op_at_the_current_version(tmp_path):
    home = tmp_path / "cfg"
    install(home, defaults=V1)
    # Same running version as the stamp → the overlay is current → nothing to do.
    assert reconcile_on_upgrade(home, defaults=V2) is None
    # V2's changed default was NOT applied — the no-op truly did nothing.
    assert (home / "prompts" / "system-prompt.md").read_text() == "v1 charter\n"


def test_reconcile_on_upgrade_refreshes_a_stale_overlay_after_a_version_bump(tmp_path):
    """The @jt fix: a pip -U (running version ≠ stamped version) reconciles the stale overlay."""
    home = tmp_path / "cfg"
    install(home, defaults=V1)
    # Simulate the config home having been produced by an older harness.
    _write_old_version(home, "0.0.0")

    report = reconcile_on_upgrade(home, defaults=V2)

    assert report is not None  # it reconciled
    # The pristine stale default was refreshed to the new one...
    assert report.actions["prompts/system-prompt.md"] == REFRESHED
    assert (home / "prompts" / "system-prompt.md").read_text() == "v2 charter\n"
    # ...and the home is re-stamped with the running version, so the next wake is a no-op.
    assert installed_version(home) == __version__


def test_reconcile_on_upgrade_runs_once_for_a_home_predating_the_stamp(tmp_path):
    """A config home installed before the stamp existed has no .version → reconcile once."""
    home = tmp_path / "cfg"
    install(home, defaults=V1)
    (home / _VERSION_NAME).unlink()  # simulate a pre-stamp install

    report = reconcile_on_upgrade(home, defaults=V2)

    assert report is not None
    assert (home / "prompts" / "system-prompt.md").read_text() == "v2 charter\n"
    assert installed_version(home) == __version__  # now stamped going forward


def _write_old_version(home, version):
    """Overwrite the version stamp to simulate a config home produced by an older harness."""
    (config_home(home) / _VERSION_NAME).write_text(version + "\n")


# --- provider affinity (issue #160, scope expansion) -------------------------

# Minimal plugin sources standing in for the shipped defaults' affinity shapes.
_XAI_SRC = "from basecradle_harness import ToolPlugin, Vendor\nPLUGIN = ToolPlugin(builtin='x', requires=(Vendor('xai'),))\n"
_OPENAI_KEY_SRC = "from basecradle_harness import GenerateImageTool, OpenAIKey, ToolPlugin\nPLUGIN = ToolPlugin(impl=GenerateImageTool, requires=(OpenAIKey(),))\n"
_OPENAI_SURFACE_SRC = "from basecradle_harness import OpenAISurface, ToolPlugin, Vendor\nPLUGIN = ToolPlugin(builtin='web_search', requires=(Vendor('openai'), OpenAISurface('responses')))\n"
_UNIVERSAL_SRC = (
    "from basecradle_harness import AssetsTool, ToolPlugin\nPLUGIN = ToolPlugin(impl=AssetsTool)\n"
)


def test_plugin_source_providers_reads_affinity_without_importing():
    assert plugin_source_providers(_XAI_SRC) == frozenset({"xai"})
    assert plugin_source_providers(_OPENAI_KEY_SRC) == frozenset({"openai"})
    assert plugin_source_providers(_OPENAI_SURFACE_SRC) == frozenset({"openai"})
    assert plugin_source_providers(_UNIVERSAL_SRC) is None  # no markers → universal


def test_plugin_source_providers_treats_broken_source_as_universal():
    # Unparseable source → None (universal), so the loader still attempts it and the broken
    # default surfaces as a defect rather than being hidden by the affinity check.
    assert plugin_source_providers("this is not valid python :(") is None


def test_plugin_relevant_to_gates_on_the_active_provider():
    assert plugin_relevant_to(_XAI_SRC, "xai")
    assert not plugin_relevant_to(_XAI_SRC, "openai")
    assert plugin_relevant_to(_OPENAI_KEY_SRC, "openai")
    assert not plugin_relevant_to(_OPENAI_KEY_SRC, "xai")
    assert plugin_relevant_to(_UNIVERSAL_SRC, "xai")  # universal → relevant everywhere
    assert plugin_relevant_to(_XAI_SRC, None)  # provider=None → no filtering


# --- provider-aware install + prune (issue #160, scope expansion) ------------

# The shipped tool defaults, by provider affinity (mirrors the real `_defaults/tools/`).
_XAI_DEFAULTS = {"grok_generate_image.py", "grok_generate_video.py", "xai_search.py"}
_OPENAI_DEFAULTS = {"generate_image.py", "edit_image.py", "hear_audio.py", "web_search.py"}


def _tool_files(home):
    return {p.name for p in (home / "tools").glob("*.py")}


def test_install_for_openai_omits_the_grok_and_xai_tool_defaults(tmp_path):
    # The @jt fix: a provider-aware install lays down no grok/xAI plugin on an OpenAI agent.
    home = tmp_path / "cfg"
    install(home, provider="openai")
    files = _tool_files(home)
    assert _XAI_DEFAULTS.isdisjoint(files)  # no grok/xai clutter
    assert _OPENAI_DEFAULTS <= files  # the OpenAI-coupled defaults are present
    assert "assets.py" in files  # universal defaults are always present


def test_install_for_xai_omits_the_openai_coupled_tool_defaults(tmp_path):
    home = tmp_path / "cfg"
    install(home, provider="xai")
    files = _tool_files(home)
    assert _XAI_DEFAULTS <= files
    assert _OPENAI_DEFAULTS.isdisjoint(files)
    assert "assets.py" in files


def test_install_unfiltered_lays_down_every_provider_default(tmp_path):
    # The direct API default (provider=None) is provider-blind — every default, as before.
    home = tmp_path / "cfg"
    install(home)
    files = _tool_files(home)
    assert _XAI_DEFAULTS <= files and _OPENAI_DEFAULTS <= files


def test_a_provider_switch_prunes_pristine_mismatched_defaults(tmp_path):
    # The @jt de-clutter: a provider-blind install left grok/xai files; a later provider-aware
    # reconcile removes them (they are ours, recorded, and pristine).
    home = tmp_path / "cfg"
    install(home)  # provider-blind: grok/xai present
    assert _XAI_DEFAULTS <= _tool_files(home)

    report = install(home, provider="openai")

    assert _XAI_DEFAULTS.isdisjoint(_tool_files(home))  # pruned off disk
    for name in _XAI_DEFAULTS:
        assert report.actions[f"tools/{name}"] == PRUNED
    # The manifest no longer tracks the pruned defaults, and the openai-coupled ones remain.
    manifest = json.loads((home / _MANIFEST_NAME).read_text())
    assert not any(name in key for key in manifest for name in _XAI_DEFAULTS)
    assert "tools/generate_image.py" in manifest


def test_a_prune_keeps_an_operator_edited_mismatched_default(tmp_path):
    # If the operator edited a now-mismatched default, their edit wins — it is NOT pruned.
    home = tmp_path / "cfg"
    install(home)
    edited = home / "tools" / "xai_search.py"
    edited.write_text("# my hand-tuned version\n" + edited.read_text())

    install(home, provider="openai")

    assert edited.exists()  # the operator's edit is kept, never pruned
    assert edited.read_text().startswith("# my hand-tuned version")


# --- config-home resolution --------------------------------------------------


def test_config_home_resolution_prefers_arg_then_env_then_default(tmp_path, monkeypatch):
    monkeypatch.delenv("BASECRADLE_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # Default: under the OS user's home.
    assert config_home() == tmp_path / "home" / ".config" / "basecradle"
    # Env override wins over the default.
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(tmp_path / "envcfg"))
    assert config_home() == tmp_path / "envcfg"
    # An explicit arg wins over everything.
    assert config_home(tmp_path / "argcfg") == tmp_path / "argcfg"


# --- charter sourcing from files ---------------------------------------------


def test_charter_from_config_is_none_when_not_installed(tmp_path):
    assert charter_from_config(tmp_path / "cfg") is None


def test_charter_from_config_composes_both_prompt_files_in_order(tmp_path):
    home = tmp_path / "cfg"
    install(home)
    (home / "prompts" / "system-prompt.md").write_text("You are Nova.\n")
    (home / "prompts" / "initialize.md").write_text("Be terse.\n")

    assert charter_from_config(home) == "You are Nova.\n\nBe terse."


def test_charter_from_config_strips_html_comments(tmp_path):
    home = tmp_path / "cfg"
    install(home)
    (home / "prompts" / "system-prompt.md").write_text(
        "<!-- operator note: edit me -->\nYou are Nova.\n"
    )
    (home / "prompts" / "initialize.md").write_text("<!-- only a note -->\n")

    # The comment is gone; an all-comment file contributes nothing.
    assert charter_from_config(home) == "You are Nova."


def test_charter_from_env_falls_back_to_the_legacy_var_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(tmp_path / "absent"))
    monkeypatch.setenv("HARNESS_SYSTEM_PROMPT", "legacy charter")

    assert charter_from_env() == "legacy charter"


def test_charter_from_env_prefers_installed_files_over_the_legacy_var(tmp_path, monkeypatch):
    home = tmp_path / "cfg"
    install(home)
    (home / "prompts" / "initialize.md").unlink()  # keep just the system prompt for a clean assert
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(home))
    monkeypatch.setenv("HARNESS_SYSTEM_PROMPT", "legacy charter")

    assert charter_from_env() == "You are a helpful peer on BaseCradle."


def test_a_deliberately_blanked_charter_wins_over_the_legacy_var(tmp_path, monkeypatch):
    """Installed-but-emptied is *present* (``""``), not *absent* — the env var stays buried.

    An operator who installs the config home and then blanks the prompt files (all
    whitespace / only HTML comments) has deliberately disabled the standing charter. That
    must be honored, never silently overridden by a stale ``HARNESS_SYSTEM_PROMPT``.
    """
    home = tmp_path / "cfg"
    install(home)
    (home / "prompts" / "system-prompt.md").write_text("<!-- intentionally blank -->\n")
    (home / "prompts" / "initialize.md").write_text("   \n")
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(home))
    monkeypatch.setenv("HARNESS_SYSTEM_PROMPT", "legacy charter")

    # present-but-empty: files win, so the charter is empty — not the legacy env var.
    assert charter_from_config(home) == ""
    assert charter_from_env() == ""


# --- sourcing one prompt at a time (the persistent brief) --------------------


def test_prompt_text_falls_back_to_the_packaged_default_when_not_installed(tmp_path):
    # No config home installed → the packaged default is the source, so an un-migrated agent
    # (like @jt) still composes a full brief. The shipped initialize.md carries the trust note.
    text = prompt_text("initialize.md", tmp_path / "absent")
    assert "Trust is directional in storage, mutual at the gate." in text
    assert "<!--" not in text  # the operator-note comment is stripped before composition


def test_prompt_text_prefers_the_installed_file_and_strips_comments(tmp_path):
    home = tmp_path / "cfg"
    install(home)
    (home / "prompts" / "initialize.md").write_text("<!-- note -->\nMy own guidance.\n")

    assert prompt_text("initialize.md", home) == "My own guidance."


def test_prompt_text_honors_a_deletion_once_installed(tmp_path):
    # Installed, then deleted → respected (None), never resurrected from the package.
    home = tmp_path / "cfg"
    install(home)
    (home / "prompts" / "initialize.md").unlink()

    assert prompt_text("initialize.md", home) is None


def test_system_prompt_text_uses_the_legacy_env_when_not_installed(tmp_path, monkeypatch):
    # @jt has no config home; its HARNESS_SYSTEM_PROMPT is the personality slot of the brief.
    monkeypatch.setenv("HARNESS_SYSTEM_PROMPT", "You are JT, a test peer.")
    assert system_prompt_text(tmp_path / "absent") == "You are JT, a test peer."


def test_system_prompt_text_defaults_to_the_packaged_personality(tmp_path, monkeypatch):
    monkeypatch.delenv("HARNESS_SYSTEM_PROMPT", raising=False)
    assert system_prompt_text(tmp_path / "absent") == "You are a helpful peer on BaseCradle."


def test_system_prompt_text_prefers_installed_files_over_the_legacy_env(tmp_path, monkeypatch):
    home = tmp_path / "cfg"
    install(home)
    (home / "prompts" / "system-prompt.md").write_text("You are Nova.\n")
    monkeypatch.setenv("HARNESS_SYSTEM_PROMPT", "legacy charter")

    assert system_prompt_text(home) == "You are Nova."


# --- the CLI -----------------------------------------------------------------


def test_cli_installs_to_the_given_config_home(tmp_path, capsys):
    code = main(["--config-home", str(tmp_path / "cfg")])

    assert code == 0
    assert (tmp_path / "cfg" / "prompts" / "system-prompt.md").exists()
    out = capsys.readouterr().out
    assert str(tmp_path / "cfg") in out


def test_cli_is_provider_aware_from_the_env_by_default(tmp_path, monkeypatch, capsys):
    # A bare install on an OpenAI agent (AI_PROVIDER=openai) lays down no grok/xAI clutter.
    monkeypatch.setenv("AI_PROVIDER", "openai")
    home = tmp_path / "cfg"
    main(["--config-home", str(home)])
    capsys.readouterr()
    assert _XAI_DEFAULTS.isdisjoint(_tool_files(home))
    assert _OPENAI_DEFAULTS <= _tool_files(home)


def test_cli_all_providers_disables_the_filter(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    home = tmp_path / "cfg"
    main(["--config-home", str(home), "--all-providers"])
    capsys.readouterr()
    assert _XAI_DEFAULTS <= _tool_files(home)  # every default, provider-blind


def test_cli_provider_flag_overrides_the_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    home = tmp_path / "cfg"
    main(["--config-home", str(home), "--provider", "xai"])
    capsys.readouterr()
    assert _XAI_DEFAULTS <= _tool_files(home)
    assert _OPENAI_DEFAULTS.isdisjoint(_tool_files(home))
