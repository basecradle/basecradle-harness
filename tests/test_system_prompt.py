"""Self-authorship: the `system_prompt_read` / `system_prompt_edit` tools (issue #241).

The most powerful tool in the kit — an agent editing its own personality charter — so the tests
pin the properties that make that power *structurally* safe, not merely validated-safe:

- **Self-scoping by construction.** Neither tool takes a path or agent argument; the target is
  always ``<config-home>/prompts/system-prompt.md``, resolved from the environment the same way
  the wake brief resolves it. There is nothing for a prompt-injected argument to redirect.
- **``initialize.md`` is never editable.** With no file selector, the input-security floor
  (issue #239, which lives in ``initialize.md``) stays above self-authorship — an edit cannot
  reach it. We pin both the structural absence of a selector and that a real edit leaves
  ``initialize.md`` byte-for-byte untouched.
- **Guarded confirm = compare-and-swap.** A bare or mismatched confirm previews and writes
  nothing; the token is a hash of the current content, so a stale edit (file changed since the
  read) is refused too.
- **Versioned history.** Every successful edit snapshots the old file as a timestamped ``.bak``.
- **Takes effect next wake.** After an edit, the brief's charter accessor (`system_prompt_text`)
  reads the new content.

A real config home is scaffolded per test with `install`, and ``BASECRADLE_CONFIG_HOME`` points
the tools at it — the same resolver the runtime uses. Cast: Nova Digital (``nova``, an AI)
rewriting her own charter.
"""

import pytest

from basecradle_harness import (
    SystemPromptEditTool,
    SystemPromptReadTool,
    config_home,
    install,
    load_plugins,
    system_prompt_text,
)
from basecradle_harness._install import plugin_opts_in

ORIGINAL = "You are Nova Digital, a warm and precise AI peer.\n"
REWRITE = "You are Nova Digital, now terse and blunt.\n"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A freshly-installed config home the tools resolve to via BASECRADLE_CONFIG_HOME.

    `install` scaffolds ``prompts/system-prompt.md`` (a benign default) and records it in the
    manifest, so `system_prompt_text` reads the file — the "takes effect next wake" path.
    """
    root = tmp_path / "cfg"
    install(root)
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(root))
    # Author a known starting charter so the token/backup assertions are deterministic.
    (root / "prompts" / "system-prompt.md").write_text(ORIGINAL, encoding="utf-8")
    return root


def _prompt_path(root):
    return root / "prompts" / "system-prompt.md"


def _token(root):
    """The current edit token the tools compute for the charter file."""
    return SystemPromptReadTool().run().split("edit token: ")[1].split(" ")[0].rstrip(")")


# --- self-scoping (invariant 1) ----------------------------------------------


def test_neither_tool_exposes_a_path_or_agent_argument():
    # The structural guarantee: there is no parameter a prompt-injected argument could aim
    # elsewhere. Read takes nothing; edit takes only content + confirm.
    assert SystemPromptReadTool().parameters.get("properties", {}) == {}
    assert set(SystemPromptEditTool().parameters["properties"]) == {"content", "confirm"}


def test_tools_resolve_to_the_agents_own_config_home_prompt(home):
    # "Own prompt only" = the file the config-home resolver points at, nothing passed in.
    result = SystemPromptReadTool().run()
    assert ORIGINAL in result
    assert config_home() == home  # the env-resolved home the tool used


# --- read (returns raw content + a usable token) -----------------------------


def test_read_returns_the_verbatim_prompt_and_an_edit_token(home):
    result = SystemPromptReadTool().run()
    assert ORIGINAL in result
    assert "edit token:" in result
    # The token round-trips: it is exactly what edit will accept as confirm.
    assert _token(home)


def test_read_on_an_absent_prompt_reports_it_and_gives_the_empty_token(home):
    # File deleted on an *installed* home (manifest still records prompts/): the wake honors the
    # deletion as "no charter", so read reports it absent and hands out the empty-content token.
    _prompt_path(home).unlink()
    result = SystemPromptReadTool().run()
    assert "No system prompt is set" in result
    assert "edit token" in result  # the empty-content token, to author one


def test_read_declines_when_prompts_are_not_installed(tmp_path, monkeypatch):
    # When prompts are not manifest-installed the wake's live charter comes from the environment /
    # packaged default, NOT this file — so read must not claim "unset" or hand out an edit token
    # for a file the runtime does not read. It declines honestly, symmetric with edit.
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(tmp_path / "never-installed"))
    result = SystemPromptReadTool().run()
    assert "not installed" in result
    assert "edit token" not in result


# --- the confirm gate (invariant 4) ------------------------------------------


def test_a_bare_edit_previews_and_writes_nothing(home):
    result = SystemPromptEditTool().run(content=REWRITE)
    assert "Refused to edit" in result
    assert "No confirm was passed" in result
    assert f"confirm={_token(home)}" in result
    assert _prompt_path(home).read_text(encoding="utf-8") == ORIGINAL  # untouched


def test_a_mismatched_confirm_previews_and_writes_nothing(home):
    result = SystemPromptEditTool().run(content=REWRITE, confirm="not-the-token")
    assert "Refused to edit" in result
    assert "'not-the-token'" in result  # echoes the bad confirm
    assert _prompt_path(home).read_text(encoding="utf-8") == ORIGINAL


def test_a_matching_confirm_rewrites_the_prompt(home):
    result = SystemPromptEditTool().run(content=REWRITE, confirm=_token(home))
    assert "Rewrote your system prompt" in result
    assert "NEXT WAKE" in result
    assert _prompt_path(home).read_text(encoding="utf-8") == REWRITE


def test_a_stale_token_is_refused_compare_and_swap(home):
    # Read gives a token; the file then changes out from under the agent; the old token no longer
    # matches, so the edit previews instead of clobbering the newer content.
    stale = _token(home)
    _prompt_path(home).write_text("someone else edited this\n", encoding="utf-8")
    result = SystemPromptEditTool().run(content=REWRITE, confirm=stale)
    assert "Refused to edit" in result
    assert _prompt_path(home).read_text(encoding="utf-8") == "someone else edited this\n"


def test_an_identical_edit_is_a_noop_with_no_backup(home):
    result = SystemPromptEditTool().run(content=ORIGINAL, confirm=_token(home))
    assert "No change" in result
    assert not list((home / "prompts").glob("*.bak"))


# --- versioned history (invariant 5) -----------------------------------------


def test_a_successful_edit_snapshots_the_old_prompt_as_a_bak(home):
    SystemPromptEditTool().run(content=REWRITE, confirm=_token(home))
    backups = list((home / "prompts").glob("system-prompt.md.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == ORIGINAL  # the pre-edit content is preserved


# --- initialize.md exclusion (invariant 2) -----------------------------------


def test_an_edit_never_touches_initialize_md(home):
    before = (home / "prompts" / "initialize.md").read_text(encoding="utf-8")
    SystemPromptEditTool().run(content=REWRITE, confirm=_token(home))
    after = (home / "prompts" / "initialize.md").read_text(encoding="utf-8")
    assert before == after  # the input-security floor stays byte-for-byte untouched


# --- takes effect next wake (invariant 6) ------------------------------------


def test_an_edit_is_what_the_next_wake_reads(home):
    SystemPromptEditTool().run(content=REWRITE, confirm=_token(home))
    # `system_prompt_text` is the brief's charter accessor, called fresh each wake.
    assert system_prompt_text(home) == REWRITE.strip()


def test_authoring_recreates_a_deleted_prompts_dir_on_an_installed_home(home):
    # Finding 3: on an installed home (manifest records prompts/) whose prompts/ dir was later
    # deleted, the wake honors the deletion as "no charter" and the agent may author a fresh one.
    # The empty-content token authors it, and the parent dir is recreated rather than crashing.
    import shutil

    shutil.rmtree(home / "prompts")
    from basecradle_harness._system_prompt import _content_token

    result = SystemPromptEditTool().run(content=REWRITE, confirm=_content_token(""))
    assert "Rewrote your system prompt" in result
    assert _prompt_path(home).read_text(encoding="utf-8") == REWRITE
    assert system_prompt_text(home) == REWRITE.strip()  # and the next wake reads it


# --- not installed → honest refusal ------------------------------------------


def test_edit_refuses_when_the_config_home_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(tmp_path / "never-installed"))
    result = SystemPromptEditTool().run(content=REWRITE, confirm="whatever")
    assert "not installed" in result
    assert not (tmp_path / "never-installed").exists()  # wrote nothing, created nothing


def test_edit_refuses_when_prompts_dir_exists_but_is_not_manifest_installed(tmp_path, monkeypatch):
    # Invariant 6 held by construction: the wake brief reads the config-home file only when
    # prompts are *manifest*-installed. A bare prompts/ dir with a hand-placed file (no manifest)
    # is NOT what the runtime reads, so an edit here would never land — it must be refused, not
    # written into a black hole.
    root = tmp_path / "cfg"
    (root / "prompts").mkdir(parents=True)
    (root / "prompts" / "system-prompt.md").write_text(ORIGINAL, encoding="utf-8")
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(root))
    result = SystemPromptEditTool().run(content=REWRITE, confirm="whatever")
    assert "not installed" in result
    assert (root / "prompts" / "system-prompt.md").read_text(encoding="utf-8") == ORIGINAL


# --- opt-in, off by default on every provider (invariant 3, issue #168) ------


def test_the_shipped_plugin_is_classified_opt_in():
    from importlib.resources import files

    src = files("basecradle_harness").joinpath("_defaults", "tools", "system_prompt.py")
    assert plugin_opts_in(src.read_text(encoding="utf-8"))


def test_the_tools_are_absent_from_the_default_load(tmp_path, monkeypatch):
    # No overlay → the packaged-default load drops opt-in tools, so a default-riding agent never
    # gets self-authorship. It activates only when dropped into a persona's tools/ overlay.
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(tmp_path / "cfg"))
    names = {p.resolved_name for p in load_plugins()}
    assert "system_prompt_read" not in names
    assert "system_prompt_edit" not in names
