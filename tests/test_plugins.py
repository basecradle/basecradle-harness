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
    ProviderAPI,
    Tool,
    ToolPlugin,
    install,
    load_plugins,
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


def _ctx(api="chat", **env):
    return ActivationContext(provider_api=api, env=env)


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


def test_provider_api_requirement_matches_the_active_api():
    req = ProviderAPI("responses")
    assert req.met(_ctx(api="responses"))
    assert not req.met(_ctx(api="chat"))


def test_env_and_openai_key_requirements_read_the_context_env():
    assert EnvSet("FOO").met(_ctx(FOO="x"))
    assert not EnvSet("FOO").met(_ctx())
    assert OpenAIKey().met(_ctx(AI_PROVIDER_API_KEY="sk-test"))
    assert not OpenAIKey().met(_ctx())


# --- resolution --------------------------------------------------------------


def test_resolve_splits_active_plugins_into_function_tools_and_builtins():
    plugins = [
        ToolPlugin(impl=_Echo),
        ToolPlugin(builtin="web_search", requires=(ProviderAPI("responses"),)),
    ]
    resolved = resolve_plugins(plugins, _ctx(api="responses"))
    assert [t.name for t in resolved.tools] == ["echo"]
    assert resolved.builtins == ["web_search"]


def test_an_unmet_requirement_excludes_the_plugin_and_records_why():
    plugins = [
        ToolPlugin(impl=_Echo, requires=(OpenAIKey(),)),
        ToolPlugin(builtin="web_search", requires=(ProviderAPI("responses"),)),
    ]
    resolved = resolve_plugins(plugins, _ctx(api="chat"))  # no key, not responses
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
    chat_variant = ToolPlugin(impl=_Echo, name="search", requires=(ProviderAPI("chat"),))
    responses_variant = ToolPlugin(impl=_Other, name="search", requires=(ProviderAPI("responses"),))
    plugins = [chat_variant, responses_variant]

    under_chat = resolve_plugins(plugins, _ctx(api="chat"))
    under_responses = resolve_plugins(plugins, _ctx(api="responses"))

    assert [type(t) for t in under_chat.tools] == [_Echo]
    assert [type(t) for t in under_responses.tools] == [_Other]


# --- behavior-preserving: the packaged defaults under each provider ----------

# The full default function-tool set every existing deployment has, independent of provider.
_DEFAULT_TOOLS = {
    "memory",
    "web_fetch",
    "assets",
    "listen",
    "tasks",
    "timelines",
    "trust",
    "generate_image",
    "webhook_endpoints",
    "webhook_events",
}


def test_packaged_defaults_under_responses_match_todays_tool_set_plus_web_search():
    resolved = resolve_plugins(load_plugins(), _ctx(api="responses", AI_PROVIDER_API_KEY="sk"))
    assert {t.name for t in resolved.tools} == _DEFAULT_TOOLS
    assert resolved.builtins == ["web_search"]  # the Responses built-in is active


def test_web_search_drops_on_chat_completions_the_other_tools_stay():
    resolved = resolve_plugins(load_plugins(), _ctx(api="chat", AI_PROVIDER_API_KEY="sk"))
    assert {t.name for t in resolved.tools} == _DEFAULT_TOOLS
    assert resolved.builtins == []  # web_search self-excludes off the Responses API


def test_openai_coupled_tools_drop_without_an_openai_key():
    resolved = resolve_plugins(load_plugins(), _ctx(api="chat"))  # no key
    names = {t.name for t in resolved.tools}
    assert "generate_image" not in names and "listen" not in names
    assert _DEFAULT_TOOLS - {"generate_image", "listen"} == names


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
        "class Mem(Tool):\n"
        "    name = 'memory'\n"
        "    description = 'override'\n"
        "    def run(self, **kw):\n"
        "        return 'overridden'\n"
        "PLUGIN = ToolPlugin(impl=Mem)\n"
    )
    # DISABLE — deleting a default's file removes the tool.
    (tools_dir / "web_fetch.py").unlink()

    resolved = resolve_plugins(load_plugins(home), _ctx(AI_PROVIDER_API_KEY="sk"))
    names = {t.name for t in resolved.tools}
    assert "greet" in names  # added
    assert "web_fetch" not in names  # disabled
    memory = next(t for t in resolved.tools if t.name == "memory")
    assert memory.run() == "overridden"  # overridden impl won


def test_a_broken_overlay_file_is_skipped_not_fatal(tmp_path):
    home = tmp_path / "cfg"
    install(home)
    (home / "tools" / "broken.py").write_text("this is not valid python :(\n")

    # The broken file is skipped; the shipped defaults still load.
    plugins = load_plugins(home)
    assert any(p.resolved_name == "memory" for p in plugins)


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


def test_a_never_installed_config_home_falls_back_to_packaged_defaults(tmp_path):
    # No install() has run: tools/ is absent and the manifest records no tool files, so the
    # packaged defaults load directly (an un-upgraded deployment still comes up fully armed).
    plugins = load_plugins(tmp_path / "never-installed")
    assert {p.resolved_name for p in plugins} == _DEFAULT_TOOLS | {"web_search"}
