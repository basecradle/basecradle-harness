"""The deterministic stem → resolved-names resolver (issue #345).

The invariant every test here serves: **a stem's resolved names are computed, never transcribed.**
The near-miss that motivated it (basecradle-noc#344) was prose — a doc and a test both said the
`xai_search` stem resolved to `x_search` alone, when the file declares two ToolPlugins. As an
``exact_tools`` pin, that half-truth is a permanently-failing converge. So the load-bearing test in
this file is not any single expectation about a name: it is
`test_matches_resolved_config_on_the_box`, which pins the *answer* against the one a live wake
gives, so the two surfaces cannot drift apart the day a plugin file changes.
"""

from __future__ import annotations

import json

import pytest

from basecradle_harness._install import install
from basecradle_harness._resolve import UnknownStemError, main, resolve_stems
from basecradle_harness._wake import resolved_config

# --- the headline: one stem, two built-ins ------------------------------------


def test_xai_search_resolves_to_both_live_search_builtins():
    """The #344 case, computed: `xai_search` arms **web_search AND x_search**, not one of them.

    A pin built from either name alone asserts a tool set the box will never have (missing) or
    have more of than declared (extra) — a both-directions drift check fails forever. This is the
    single expectation the whole command exists to make impossible to get wrong by hand.
    """
    report = resolve_stems(provider="xai", sdk="xai-sdk", opt_in="xai_search")

    assert report["stems"]["xai_search"]["builtins"] == ["web_search", "x_search"]
    assert report["stems"]["xai_search"]["status"] == "active"
    assert report["builtins"] == ["web_search", "x_search"]
    assert report["opt_in_tools"] == ["xai_search"]  # the stem lists once, though it fans out


def test_only_computes_a_pruned_personas_exact_tools_pin():
    """`--only` models a deliberately tool-restricted persona, so its pin is computable too.

    @the-brain's overlay carries `messages` and nothing else; NOC#344 granted it `xai_search`. The
    resulting `exact_tools` — the union of tools and built-ins — is exactly what this returns,
    including the `memory` tool, which comes from the memory *provider* and no stem at all (the
    parallel-model hole basecradle-noc#62 refused to paper over locally).
    """
    report = resolve_stems(provider="xai", sdk="xai-sdk", only="messages,xai_search")

    assert sorted(report["tools"] + report["builtins"]) == [
        "memory",
        "messages",
        "web_search",
        "x_search",
    ]


# --- the anti-drift pin -------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "sdk", "opt_in"),
    [
        ("openai", "openai", ["generate_image", "code_execution", "hear_audio"]),
        ("openai", "openai", []),
        ("xai", "xai-sdk", ["xai_search", "grok_generate_image", "xai_account_balance"]),
        ("openrouter", "openrouter", ["openrouter_search"]),
    ],
)
def test_matches_resolved_config_on_the_box(monkeypatch, tmp_path, provider, sdk, opt_in):
    """The answer computed off-box equals the one a live agent reports — the whole contract.

    A real config home is installed for the provider with those opt-ins (exactly what the NOC's
    converge does), then `--resolved-config`'s active tool set is compared to `resolve_stems`'s.
    If the two ever diverge — a new plugin, a changed requirement, a reordered overlay — this
    fails here rather than as a converge that can never go green.
    """
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(cfg))
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    monkeypatch.setenv("AI_PROVIDER", provider)
    monkeypatch.setenv("AI_SDK", sdk)
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")  # the credential the resolver *assumes*
    monkeypatch.delenv("AI_SDK_SURFACE", raising=False)
    monkeypatch.delenv("HARNESS_MEMORY_PROVIDER", raising=False)
    monkeypatch.delenv("HARNESS_PROFILE", raising=False)
    install(cfg, provider=provider, opt_in=opt_in)

    live = resolved_config()
    computed = resolve_stems(provider=provider, sdk=sdk, opt_in=opt_in)

    assert computed["tools"] == live["tools"]
    assert computed["builtins"] == live["builtins"]
    assert computed["opt_in_tools"] == live["opt_in_tools"]


def test_matches_resolved_config_for_a_pruned_overlay(monkeypatch, tmp_path):
    """`--only` matches the box too — the case a whole-default-set answer would get wrong.

    The pruned persona is the one whose pin is easiest to compute wrongly, so it is pinned against
    a real overlay with its benign defaults deleted, exactly as a restricted persona's is.
    """
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(cfg))
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    monkeypatch.setenv("AI_PROVIDER", "xai")
    monkeypatch.setenv("AI_SDK", "xai-sdk")
    monkeypatch.delenv("AI_SDK_SURFACE", raising=False)
    monkeypatch.delenv("HARNESS_MEMORY_PROVIDER", raising=False)
    install(cfg, provider="xai", opt_in=["xai_search"])
    for path in (cfg / "tools").glob("*.py"):
        if path.stem not in {"messages", "xai_search"}:
            path.unlink()  # the operator's prune — a deletion the loader honors

    live = resolved_config()
    computed = resolve_stems(provider="xai", sdk="xai-sdk", only="messages,xai_search")

    assert computed["tools"] == live["tools"] == ["memory", "messages"]
    assert computed["builtins"] == live["builtins"] == ["web_search", "x_search"]


# --- purity -------------------------------------------------------------------


def test_reads_no_environment_and_writes_nothing(monkeypatch, tmp_path):
    """Deterministic: a hostile environment cannot move the answer, and nothing is created.

    Every axis is an argument, so the env vars a wake reads — provider, sdk, surface, profile,
    memory backend, credentials — are set here to values that would each change the result if they
    leaked in. The config home is pointed at an empty dir and must still not exist afterwards: a
    resolution that scaffolded (or reconciled) one would be a write on a read-only path.
    """
    cfg = tmp_path / "never-created"
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(cfg))
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("AI_SDK", "openrouter")
    monkeypatch.setenv("AI_SDK_SURFACE", "chat")
    monkeypatch.setenv("HARNESS_PROFILE", "unlocked")
    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "mempalace")
    monkeypatch.delenv("AI_API_KEY", raising=False)

    report = resolve_stems(provider="xai", sdk="xai-sdk", opt_in="xai_search,shell")

    assert report["ai_provider"] == "xai"
    assert report["ai_sdk_surface"] == "native"  # the xai-sdk adapter's own default, not "chat"
    assert report["active_profile"] == "locked"  # the argument's default, not HARNESS_PROFILE
    assert report["memory"]["provider"] == "sqlite"  # not HARNESS_MEMORY_PROVIDER's mempalace
    assert report["builtins"] == ["web_search", "x_search"]
    assert "shell" not in report["tools"]  # locked refused it, whatever the env said
    assert not cfg.exists()


def test_an_installed_overlay_cannot_change_the_answer(monkeypatch, tmp_path):
    """The candidate set is the *package's* defaults — an on-box overlay is never consulted.

    Determinism across machines depends on it: a GitHub Action computing a pin runs in a venv with
    no agent, while a captain may run the same command on a box whose overlay was pruned years
    ago. Both must answer for the *arguments*, not for whatever happens to be on that disk.
    """
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(cfg))
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    install(cfg, provider="openai", opt_in=[])
    for path in (cfg / "tools").glob("*.py"):
        path.unlink()  # an overlay pruned to nothing: a *wake* here would resolve no plugin tools

    report = resolve_stems(provider="openai", sdk="openai")

    assert "messages" in report["tools"]  # answered from the package, not the empty overlay


# --- the credential question --------------------------------------------------


def test_credential_gated_tools_are_assumed_present_and_say_so():
    """The default: assume the credential, name it, and mark every conditional name (req. 2).

    The caller is almost always computing a pin for a *provisioned* agent, which holds its key —
    so assuming is the useful default. What is forbidden is doing it silently: `credentials` names
    the vars assumed, and each resolved name that only exists because of one carries
    `assumes_credential`.
    """
    report = resolve_stems(provider="openai", opt_in="generate_image,hear_audio,web_search")

    assert report["credentials"] == {"mode": "assumed", "assumed": ["AI_API_KEY"]}
    assert report["stems"]["generate_image"]["assumes_credential"] == ["AI_API_KEY"]
    # `hear_audio` is the stem-vs-name trap in the same breath: its tool is called `listen`.
    assert report["stems"]["hear_audio"]["tools"] == ["listen"]
    assert report["stems"]["hear_audio"]["assumes_credential"] == ["AI_API_KEY"]
    # A powerful tool gated on the *provider*, not a credential, is unconditional — no marker.
    assert report["stems"]["web_search"]["assumes_credential"] == []
    assert {"generate_image", "listen"} <= set(report["tools"])


def test_no_assume_credentials_reports_them_inactive_with_the_reason():
    """The other half of the choice, equally explicit: absent → inactive, with the unmet reason.

    Never a silent omission — the tool appears in `stems` and in the flat `skipped` trail saying
    exactly which requirement failed, so a caller can tell "this agent lacks the key" from "this
    tool does not exist here".
    """
    report = resolve_stems(provider="openai", opt_in="generate_image", assume_credentials=False)

    assert report["credentials"] == {"mode": "absent", "assumed": []}
    entry = report["stems"]["generate_image"]
    assert entry["status"] == "inactive"
    assert "AI_API_KEY" in entry["reason"]
    assert "generate_image" not in report["tools"]
    assert {"name": "generate_image", "stem": "generate_image", "reason": entry["reason"]} in (
        report["skipped"]
    )


# --- fan-out, naming, and the gates -------------------------------------------


def test_code_execution_fans_out_to_a_builtin_and_a_tool_under_openai():
    """One stem, two resolved names, and the built-in's **wire** name differs from its own.

    `code_execution` is the stem; on OpenAI it arms the `code_interpreter` built-in (the wire name
    a pin holds — *not* the plugin's model-facing `code_execution`) plus the `code_attach` tool.
    Three names in one stem, none of them equal to it: exactly the shape prose keeps getting wrong.
    """
    report = resolve_stems(provider="openai", sdk="openai", opt_in="code_execution")

    entry = report["stems"]["code_execution"]
    assert entry["builtins"] == ["code_interpreter"]
    assert entry["tools"] == ["code_attach"]
    assert "code_interpreter" in report["builtins"]


def test_code_execution_resolves_differently_under_xai():
    """The same stem, a different provider, a different answer — which is why it must be computed.

    xAI's code execution is its own built-in and has no input-file bridge, so `code_attach`
    self-excludes with its reason rather than vanishing.
    """
    report = resolve_stems(provider="xai", sdk="xai-sdk", opt_in="code_execution")

    entry = report["stems"]["code_execution"]
    assert entry["builtins"] == ["code_execution"]
    assert entry["tools"] == []
    assert any(item["name"] == "code_attach" for item in entry["skipped"])


def test_shell_is_refused_under_locked_and_resolves_under_unlocked():
    """The profile gate, both directions — the axis a shell-class enablement turns on.

    Under `locked` the opted-in shell is refused, with the policy's own wording; under `unlocked`
    it resolves. The runtime root-veto is deliberately *not* applied (it is a property of the box),
    which is why `omitted` says so out loud.
    """
    locked = resolve_stems(opt_in="shell", profile="locked")
    unlocked = resolve_stems(opt_in="shell", profile="unlocked")

    assert "shell" not in locked["tools"]
    assert "safe-by-default policy" in locked["stems"]["shell"]["reason"]
    assert "shell" in unlocked["tools"]
    assert unlocked["stems"]["shell"]["status"] == "active"
    assert {"mcp", "runtime_veto"} == {entry["source"] for entry in locked["omitted"]}
    # `opt_in_tools` is the *inventory* axis and is deliberately not narrowed by the policy gate —
    # the same semantics `--resolved-config` reports, so the NOC compares them like-for-like.
    assert locked["opt_in_tools"] == unlocked["opt_in_tools"] == ["shell"]


def test_powerful_tools_are_off_until_granted():
    """The capability rule (issue #168) holds here too: ungranted powerful stems are excluded.

    The safe default is the answer for an agent nobody opted anything into — and the exclusion is
    stated per stem *and* in the flat `excluded_stems` list, never inferred from an absence.
    """
    report = resolve_stems(provider="xai", sdk="xai-sdk")

    assert report["builtins"] == []
    assert report["stems"]["xai_search"]["granted"] is False
    assert report["stems"]["xai_search"]["status"] == "excluded"
    assert {"stem": "xai_search", "reason": report["stems"]["xai_search"]["reason"]} in (
        report["excluded_stems"]
    )
    assert report["opt_in_tools"] == []


def test_a_provider_mismatched_stem_is_excluded_not_rejected():
    """A real tool asked for on the wrong provider is an *answer*, not an error.

    The distinction the installer draws at scaffold time, drawn here: `xai_search` under OpenAI is
    a known stem that is merely unavailable, and it reports as excluded with the reason. Only a
    name that exists nowhere is a mistake in the question.
    """
    report = resolve_stems(provider="openai", opt_in="xai_search")

    entry = report["stems"]["xai_search"]
    assert entry["status"] == "excluded"
    assert "openai" in entry["reason"]
    assert report["builtins"] == []


def test_an_unknown_sdk_is_fatal_rather_than_a_different_answer():
    """A typo'd `--sdk` must stop, not quietly resolve a *different* configuration.

    The wake can leave this to the provider build, which fails the moment it tries to reach a
    model. Nothing downstream of *this* path would ever catch it: `--sdk openroute` simply drops
    `openrouter_search` (which requires `Sdk("openrouter")`) and returns a report that is short a
    name — an unknown stem's failure wearing a different hat.
    """
    with pytest.raises(ValueError, match="Unknown sdk"):
        resolve_stems(provider="openrouter", sdk="openroute", opt_in="openrouter_search")

    # …while the correct spelling resolves the server tool, which is what makes it a real hazard.
    ok = resolve_stems(provider="openrouter", sdk="openrouter", opt_in="openrouter_search")
    assert ok["builtins"] == ["web_search"]


def test_a_broken_shipped_default_is_reported_not_silently_absent(monkeypatch):
    """A default that will not import must never read as an ordinary absence — that is a short pin.

    The wake degrades the same way (the box genuinely loses the tool), so the *answer* stays
    correct; what would be wrong is the silence. The stem reports `status: "broken"` with the load
    error, and the failure also rides the report's top-level `broken` list.
    """
    from basecradle_harness import _plugins, _resolve

    real = _plugins.load_default_plugins

    def one_file_broken(provider=None):
        loaded = real(provider=provider)
        return _plugins.LoadedPlugins(
            plugins=[p for p in loaded.plugins if p.stem != "messages"],
            broken_defaults=[("messages.py", "SyntaxError: invalid syntax")],
        )

    monkeypatch.setattr(_resolve, "load_default_plugins", one_file_broken)
    report = resolve_stems(provider="openai")

    assert report["broken"] == [
        {"stem": "messages", "file": "messages.py", "error": "SyntaxError: invalid syntax"}
    ]
    assert report["stems"]["messages"]["status"] == "broken"
    assert "SyntaxError" in report["stems"]["messages"]["reason"]
    assert "messages" not in report["tools"]  # honestly absent, and honestly explained


def test_an_unknown_stem_is_fatal_and_names_the_vocabulary():
    """A typo stops the command — it never answers a question it was not asked.

    Silently contributing nothing for `xai_serch` would produce a pin short two names, which is
    the #344 failure with extra steps. The error lists the shipped stems so the fix is immediate.
    """
    with pytest.raises(UnknownStemError) as caught:
        resolve_stems(provider="xai", opt_in="xai_serch")

    assert "xai_serch" in str(caught.value)
    assert "xai_search" in str(caught.value)  # the vocabulary, so the typo is self-correcting


def test_the_openai_surface_gates_a_builtin():
    """The surface axis is real and resolvable: OpenAI's web_search exists only on Responses."""
    responses = resolve_stems(provider="openai", sdk="openai", opt_in="web_search")
    chat = resolve_stems(provider="openai", sdk="openai", surface="chat", opt_in="web_search")

    assert responses["ai_sdk_surface"] == "responses"  # the adapter's default, unset
    assert responses["builtins"] == ["web_search"]
    assert chat["builtins"] == []
    assert "responses" in chat["stems"]["web_search"]["reason"]


# --- the CLI ------------------------------------------------------------------


def test_main_prints_stable_json_and_exits_zero(capsys):
    """The machine-consumable surface: pretty JSON on stdout, stable key order, exit 0."""
    assert main(["--provider", "xai", "--sdk", "xai-sdk", "--opt-in", "xai_search"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["builtins"] == ["web_search", "x_search"]
    assert report["harness_version"]
    assert report["requested_stems"] == ["xai_search"]


def test_main_exits_nonzero_with_no_json_on_a_bad_question(capsys):
    """A bad question fails loud and prints **nothing** to stdout — no half-answer to parse.

    An automation that reads stdout and ignores the exit code must not be handed a plausible
    object built from a question the command refused. Both failure classes are covered: an unknown
    provider and an unknown stem.
    """
    assert main(["--provider", "nope"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "nope" in captured.err

    assert main(["--opt-in", "xai_serch"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "xai_serch" in captured.err


def test_main_accepts_a_py_suffix_like_the_installer_does(capsys):
    """`--opt-in xai_search.py` works, exactly as it does for `basecradle-harness-install`."""
    assert main(["--provider", "xai", "--sdk", "xai-sdk", "--opt-in", "xai_search.py"]) == 0

    assert json.loads(capsys.readouterr().out)["builtins"] == ["web_search", "x_search"]


def test_every_shipped_stem_appears_in_the_map():
    """One call is a complete map, so a caller computing a delta never needs a second one."""
    report = resolve_stems(provider="openai")

    assert {"messages", "assets"} <= set(report["stems"])
    assert "xai_search" in report["stems"]  # present even though it is xAI-only
    assert "shell" in report["stems"]  # present even though it is unlocked-only
    assert "memory" not in report["stems"]  # not a stem — it rides the memory provider
    assert report["memory"]["tools"] == ["memory"]  # …and is reported on its own axis
    for entry in report["stems"].values():
        assert entry["status"] in {"active", "inactive", "excluded"}
