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

from basecradle_harness import config_home, install
from basecradle_harness._install import (
    _MANIFEST_NAME,
    ALREADY_CURRENT,
    INSTALLED,
    KEPT_DELETED,
    KEPT_EDITED,
    REFRESHED,
    UNCHANGED,
    charter_from_config,
    charter_from_env,
    main,
)

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


# --- the CLI -----------------------------------------------------------------


def test_cli_installs_to_the_given_config_home(tmp_path, capsys):
    code = main(["--config-home", str(tmp_path / "cfg")])

    assert code == 0
    assert (tmp_path / "cfg" / "prompts" / "system-prompt.md").exists()
    out = capsys.readouterr().out
    assert str(tmp_path / "cfg") in out
