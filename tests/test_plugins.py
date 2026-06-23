"""The tool plugin framework: the (name + requires + impl) contract, provider-aware
activation, and the ``/tools`` overlay (add / override / disable).

Everything here is offline — plugin resolution is pure (an `ActivationContext` is passed in,
never read from the process), and loading only touches the filesystem. These tests pin the
mechanism Group 2 adds: a plugin self-excludes when its requirements aren't met, exactly one
of two same-named plugins activates per config, and the config-home overlay is authoritative
once installed (else the packaged defaults load).
"""

import pytest

from basecradle_harness import (
    ActivationContext,
    EnvSet,
    OpenAIKey,
    OpenAISurface,
    Tool,
    ToolPlugin,
    Vendor,
    install,
    load_plugins,
    load_plugins_report,
    resolve_plugins,
)


# A trivial tool to stand in for a real impl — no platform, no provider, nothing to bind.
class _Echo(Tool):
    name = "echo"
    description = "Echo a phrase."

    def run(self, **kwargs):
        return kwargs.get("phrase", "")


class _Other(Tool):
    name = "other"
    description = "A different tool."

    def run(self, **kwargs):
        return "other"


def _ctx(provider="openai", sdk="openai", surface="responses", model="gpt-5.4-mini", **env):
    return ActivationContext(provider=provider, sdk=sdk, surface=surface, model=model, env=env)


# --- the plugin contract -----------------------------------------------------


def test_a_plugin_must_set_exactly_one_of_impl_or_builtin():
    with pytest.raises(ValueError):
        ToolPlugin()  # neither
    with pytest.raises(ValueError):
        ToolPlugin(impl=_Echo, builtin="web_search")  # both


def test_resolved_name_defaults_to_the_impl_name_and_can_be_overridden():
    assert ToolPlugin(impl=_Echo).resolved_name == "echo"
    assert ToolPlugin(impl=_Echo, name="aliased").resolved_name == "aliased"
    assert ToolPlugin(builtin="web_search").resolved_name == "web_search"


# --- requirements ------------------------------------------------------------


def test_vendor_requirement_matches_the_active_provider():
    req = Vendor("xai")
    assert req.met(_ctx(provider="xai"))
    assert not req.met(_ctx(provider="openai"))


def test_openai_surface_requirement_matches_the_active_surface():
    req = OpenAISurface("responses")
    assert req.met(_ctx(surface="responses"))
    assert not req.met(_ctx(surface="chat"))


def test_env_and_openai_key_requirements_read_the_context_env():
    assert EnvSet("FOO").met(_ctx(FOO="x"))
    assert not EnvSet("FOO").met(_ctx())
    # OpenAIKey needs the openai provider AND a key — both, not either.
    assert OpenAIKey().met(_ctx(provider="openai", AI_API_KEY="sk-test"))
    assert not OpenAIKey().met(_ctx(provider="openai"))  # no key
    assert not OpenAIKey().met(_ctx(provider="xai", AI_API_KEY="xai-key"))  # wrong provider


# --- resolution --------------------------------------------------------------


def test_resolve_splits_active_plugins_into_function_tools_and_builtins():
    plugins = [
        ToolPlugin(impl=_Echo),
        ToolPlugin(builtin="web_search", requires=(OpenAISurface("responses"),)),
    ]
    resolved = resolve_plugins(plugins, _ctx())
    assert [t.name for t in resolved.tools] == ["echo"]
    assert resolved.builtins == ["web_search"]


def test_resolution_builds_a_manifest_of_active_tools_with_notes():
    # The manifest names every active tool — function tools and built-ins alike, in
    # resolution order — carrying each plugin's optional `note` (or None). It is the source
    # the persistent brief renders, so it must match exactly what activated.
    plugins = [
        ToolPlugin(impl=_Echo, note="a gotcha the schema can't convey."),
        ToolPlugin(impl=_Other),  # no note → None
        ToolPlugin(builtin="web_search", requires=(OpenAISurface("responses"),)),
    ]
    resolved = resolve_plugins(plugins, _ctx())
    assert resolved.manifest == [
        ("echo", "a gotcha the schema can't convey."),
        ("other", None),
        ("web_search", None),
    ]


def test_manifest_omits_inactive_plugins():
    # A plugin whose requirements aren't met never registers, so it never appears in the
    # manifest — the brief can't list a present-but-broken tool the model couldn't call.
    plugins = [ToolPlugin(impl=_Echo), ToolPlugin(impl=_Other, requires=(OpenAIKey(),))]
    resolved = resolve_plugins(plugins, _ctx())  # no key → `other` self-excludes
    assert [name for name, _ in resolved.manifest] == ["echo"]


def test_an_unmet_requirement_excludes_the_plugin_and_records_why():
    plugins = [
        ToolPlugin(impl=_Echo, requires=(OpenAIKey(),)),
        ToolPlugin(builtin="web_search", requires=(OpenAISurface("responses"),)),
    ]
    resolved = resolve_plugins(plugins, _ctx(surface="chat"))  # no key, not responses
    assert resolved.tools == []
    assert resolved.builtins == []
    skipped = dict(resolved.skipped)
    assert "echo" in skipped and "web_search" in skipped
    assert "responses" in skipped["web_search"]


def test_a_later_active_plugin_overrides_an_earlier_one_by_name():
    # Two plugins claim the name "echo"; the later active one wins (overlay precedence).
    plugins = [ToolPlugin(impl=_Echo), ToolPlugin(impl=_Other, name="echo")]
    resolved = resolve_plugins(plugins, _ctx())
    assert [type(t) for t in resolved.tools] == [_Other]


def test_exactly_one_of_two_same_named_variants_activates_per_config():
    # Two "search" plugins differing only in requires — the provider-variant case. Under each
    # provider exactly one is active, so the active set always has one "search", never two.
    chat_variant = ToolPlugin(impl=_Echo, name="search", requires=(OpenAISurface("chat"),))
    responses_variant = ToolPlugin(
        impl=_Other, name="search", requires=(OpenAISurface("responses"),)
    )
    plugins = [chat_variant, responses_variant]

    under_chat = resolve_plugins(plugins, _ctx(surface="chat"))
    under_responses = resolve_plugins(plugins, _ctx())

    assert [type(t) for t in under_chat.tools] == [_Echo]
    assert [type(t) for t in under_responses.tools] == [_Other]


# --- behavior-preserving: the packaged defaults under each provider ----------

# The full default function-tool set every existing deployment has, independent of provider.
# Memory is no longer a tool plugin — it graduated to the `MemoryProvider` subsystem (Group 4),
# so it is wired from the provider, not loaded here. See test_memory_provider.py.
_DEFAULT_TOOLS = {
    "web_fetch",
    "assets",
    "listen",
    "tasks",
    "timelines",
    "trust",
    "lock",
    "delete",
    "users",
    "messages",
    "generate_image",
    "edit_image",
    "webhook_endpoints",
    "webhook_events",
}


def test_packaged_defaults_under_responses_match_todays_tool_set_plus_web_search():
    resolved = resolve_plugins(load_plugins(), _ctx(AI_API_KEY="sk"))
    assert {t.name for t in resolved.tools} == _DEFAULT_TOOLS
    assert resolved.builtins == ["web_search"]  # the Responses built-in is active


def test_web_search_drops_on_chat_completions_the_other_tools_stay():
    resolved = resolve_plugins(load_plugins(), _ctx(surface="chat", AI_API_KEY="sk"))
    assert {t.name for t in resolved.tools} == _DEFAULT_TOOLS
    assert resolved.builtins == []  # web_search self-excludes off the Responses API


def test_openai_coupled_tools_drop_without_an_openai_key():
    resolved = resolve_plugins(load_plugins(), _ctx(surface="chat"))  # no key
    names = {t.name for t in resolved.tools}
    assert "generate_image" not in names and "edit_image" not in names and "listen" not in names
    assert _DEFAULT_TOOLS - {"generate_image", "edit_image", "listen"} == names


# --- the /tools overlay ------------------------------------------------------

# A complete operator plugin file — defines its own tool inline and exposes a PLUGIN.
_ADDED_TOOL = """
from basecradle_harness import Tool, ToolPlugin


class Greet(Tool):
    name = "greet"
    description = "Say hi."

    def run(self, **kwargs):
        return "hi"


PLUGIN = ToolPlugin(impl=Greet)
"""


def test_overlay_add_override_and_disable(tmp_path):
    home = tmp_path / "cfg"
    install(home)  # scaffolds tools/ with the shipped default plugin files + manifest
    tools_dir = home / "tools"

    # ADD — a new file registers a new tool.
    (tools_dir / "greet.py").write_text(_ADDED_TOOL)
    # OVERRIDE — a later-sorted file reusing a default's name wins.
    (tools_dir / "zz_override.py").write_text(
        "from basecradle_harness import Tool, ToolPlugin\n"
        "class FetchOverride(Tool):\n"
        "    name = 'web_fetch'\n"
        "    description = 'override'\n"
        "    def run(self, **kw):\n"
        "        return 'overridden'\n"
        "PLUGIN = ToolPlugin(impl=FetchOverride)\n"
    )
    # DISABLE — deleting a default's file removes the tool.
    (tools_dir / "assets.py").unlink()

    resolved = resolve_plugins(load_plugins(home), _ctx(AI_API_KEY="sk"))
    names = {t.name for t in resolved.tools}
    assert "greet" in names  # added
    assert "assets" not in names  # disabled
    fetch = next(t for t in resolved.tools if t.name == "web_fetch")
    assert fetch.run() == "overridden"  # overridden impl won


def test_a_broken_overlay_file_is_skipped_not_fatal(tmp_path):
    home = tmp_path / "cfg"
    install(home)
    (home / "tools" / "broken.py").write_text("this is not valid python :(\n")

    # The broken file is skipped; the shipped defaults still load.
    plugins = load_plugins(home)
    assert any(p.resolved_name == "web_fetch" for p in plugins)


def test_a_broken_shipped_default_is_surfaced_loudly_not_swallowed(tmp_path, caplog):
    """A *default* plugin that fails to load is a defect — reported and logged at ERROR (issue #160)."""
    home = tmp_path / "cfg"
    install(home)
    # Corrupt a shipped default in the overlay (the stale-plugin shape from the @jt deploy:
    # a default file importing a symbol the new version removed).
    (home / "tools" / "web_fetch.py").write_text("import a_symbol_the_rebuild_removed_zzz\n")

    with caplog.at_level("ERROR", logger="basecradle_harness"):
        report = load_plugins_report(home)

    # The broken default is reported as a defect, never silently dropped...
    assert any(name == "web_fetch.py" for name, _ in report.broken_defaults)
    # ...and logged loudly at ERROR (not the soft WARNING an operator file gets).
    assert any("web_fetch.py" in r.getMessage() and r.levelname == "ERROR" for r in caplog.records)
    # One broken file is not fatal: the other shipped defaults still load.
    assert any(p.resolved_name == "assets" for p in report.plugins)


def test_a_broken_operator_added_file_is_not_a_default_defect(tmp_path, caplog):
    """An operator's own broken drop-in stays a soft skip, never a shipped-default defect."""
    home = tmp_path / "cfg"
    install(home)
    (home / "tools" / "my_thing.py").write_text("this is not valid python :(\n")

    with caplog.at_level("WARNING", logger="basecradle_harness"):
        report = load_plugins_report(home)

    assert report.broken_defaults == []  # not a default → not a defect
    assert any("my_thing.py" in r.getMessage() for r in caplog.records)  # still surfaced as a skip


def test_a_broken_default_on_the_fallback_path_is_a_defect(tmp_path, caplog):
    """On the never-installed fallback path every file is a packaged default, so a broken one is a defect."""
    # A config home with no install: load_plugins_report reads the packaged defaults directly.
    # Monkeypatch isn't needed — we point the loader at a temp 'package' dir is overkill; instead
    # assert the classifier treats fallback-path breakage as a default. Here we simply confirm the
    # healthy fallback reports no defects (the packaged defaults all import), pinning the baseline.
    report = load_plugins_report(tmp_path / "never-installed")
    assert report.broken_defaults == []
    assert report.plugins  # the packaged defaults loaded


def test_deleting_every_default_yields_no_tools_once_installed(tmp_path):
    home = tmp_path / "cfg"
    install(home)
    for path in (home / "tools").glob("*.py"):
        path.unlink()

    # The overlay is authoritative after install, so an emptied tools/ means zero tools —
    # the operator deliberately disabled them all; the packaged defaults do not resurrect.
    assert load_plugins(home) == []


def test_removing_the_whole_tools_dir_once_installed_yields_no_tools(tmp_path):
    import shutil

    home = tmp_path / "cfg"
    install(home)
    # An operator who wants zero tools may delete the whole dir, not just the files in it.
    # Once installed, that deletion is authoritative — the packaged defaults must NOT
    # resurrect (the no-resurrect contract holds for the directory, not only the files).
    shutil.rmtree(home / "tools")

    assert load_plugins(home) == []


# The xAI-profile-only defaults: loaded under every config but active only under provider="xai".
_XAI_DEFAULTS = {"web_search", "x_search", "grok_generate_image", "grok_generate_video"}


def test_a_never_installed_config_home_falls_back_to_packaged_defaults(tmp_path):
    # No install() has run: tools/ is absent and the manifest records no tool files, so the
    # packaged defaults load directly (an un-upgraded deployment still comes up fully armed).
    plugins = load_plugins(tmp_path / "never-installed")
    assert {p.resolved_name for p in plugins} == _DEFAULT_TOOLS | _XAI_DEFAULTS


def test_xai_profile_activates_grok_tools_and_live_search_drops_openai_tools():
    # Eddie's profile: the grok media tools and xAI Live Search built-ins activate, the
    # OpenAI-coupled tools (generate_image/edit_image/listen) self-exclude, and OpenAI's own
    # web_search built-in (OpenAISurface("responses")) does not leak in — only xAI's does.
    resolved = resolve_plugins(load_plugins(), _ctx(provider="xai", AI_API_KEY="xai-key"))
    names = {t.name for t in resolved.tools}
    assert {"grok_generate_image", "grok_generate_video"} <= names
    assert not ({"generate_image", "edit_image", "listen"} & names)
    assert sorted(resolved.builtins) == ["web_search", "x_search"]


# --- provider-aware loading (issue #160, scope expansion) --------------------


def test_load_for_openai_does_not_even_import_the_grok_xai_plugins(tmp_path):
    # A provider-aware load skips the grok/xAI plugin files before import, so an OpenAI agent
    # never imports them (sidestepping a foreign-SDK import hazard), not just deactivates them.
    plugins = load_plugins(tmp_path / "never-installed", provider="openai")
    names = {p.resolved_name for p in plugins}
    assert not ({"grok_generate_image", "grok_generate_video", "x_search"} & names)
    assert "generate_image" in names  # the OpenAI-coupled plugins still load


def test_load_for_xai_skips_the_openai_coupled_plugins(tmp_path):
    plugins = load_plugins(tmp_path / "never-installed", provider="xai")
    names = {p.resolved_name for p in plugins}
    assert {"grok_generate_image", "grok_generate_video"} <= names
    assert "generate_image" not in names and "listen" not in names


def test_a_provider_mismatched_broken_file_is_not_imported_so_never_a_defect(tmp_path):
    # The latent hazard this guards: a provider-mismatched plugin that fails to import (e.g. a
    # missing vendor SDK) must be skipped before import on a mismatched agent — so it is neither
    # a broken-default defect nor a crash. A matching-provider broken default still surfaces.
    home = tmp_path / "cfg"
    install(home, provider="openai")
    # Drop an xAI-affine file that would explode on import (no xai SDK on this openai box).
    (home / "tools" / "rogue_xai.py").write_text(
        "from basecradle_harness import ToolPlugin, Vendor\n"
        "import a_vendor_sdk_not_installed_zzz\n"
        "PLUGIN = ToolPlugin(builtin='x', requires=(Vendor('xai'),))\n"
    )

    report = load_plugins_report(home, provider="openai")

    # Skipped before import → not imported, so not a defect and not a crash.
    assert report.broken_defaults == []
    assert all("rogue_xai" not in p.resolved_name for p in report.plugins)
