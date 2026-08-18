"""The declaration and its fail-closed prover (issue #374).

Everything here is offline — the declaration is a file, the prover reads files, and neither
touches a model or the platform. The invariants pinned are the two that decide whether this is a
prover at all: **an absence nobody declared is red**, and **an unprovable state is red**. The
complements matter just as much, because a verify that reddens on legitimate operator work is a
verify that gets switched off: an edited default is green, and a deletion a reconcile has ratified
is green.

The package "upgrade" is simulated exactly the way `test_install` simulates one — a *changed*
shipped default set, plus the running version moved under a config home that did not move with it.
No second build, no package surgery.
"""

import json
import logging

import pytest

from basecradle_harness import _install, _observability, _verify
from basecradle_harness._install import install, read_declaration
from basecradle_harness._verify import (
    DECLARATION_CONTRACT,
    VerifyReport,
    claims,
    claims_document,
    emit_main,
    main,
    verify,
)
from basecradle_harness._version import __version__

# A synthetic "package" whose defaults are a benign tool, a prompt, and one powerful tool. The
# powerful marker is read from source by the same AST classifier the installer uses, so the file
# has to look like a real plugin.
BENIGN = "from basecradle_harness._plugins import ToolPlugin\nPLUGIN = ToolPlugin(builtin='n')\n"
POWERFUL = (
    "from basecradle_harness._plugins import ToolPlugin\n"
    "PLUGIN = ToolPlugin(builtin='p', opt_in=True)\n"
)
XAI_ONLY = (
    "from basecradle_harness._plugins import ToolPlugin, Vendor\n"
    "PLUGIN = ToolPlugin(builtin='x', requires=(Vendor('xai'),), opt_in=True)\n"
)
V1 = {
    "prompts/system-prompt.md": "v1 charter\n",
    "tools/notes.py": BENIGN,
    "tools/power.py": POWERFUL,
    "tools/xai_only.py": XAI_ONLY,
}


def checks(report: VerifyReport) -> list[str]:
    """The finding slugs, which is what an alert groups on — never the prose."""
    return [finding.check for finding in report.findings]


@pytest.fixture
def converged(tmp_path, monkeypatch):
    """A config home in the state a converge leaves it: installed, current, one grant."""
    monkeypatch.setattr(_install, "_packaged_defaults", lambda: dict(V1))
    home = tmp_path / "cfg"
    install(home, provider="openai", opt_in=["power"])
    return home


def _verifying(monkeypatch, defaults=V1, version=__version__):
    """Point the prover at a given 'installed package': its default set and its version."""
    monkeypatch.setattr(_verify, "_packaged_defaults", lambda: dict(defaults))
    monkeypatch.setattr(_verify, "__version__", version)
    monkeypatch.setattr(_verify, "_distribution_version", lambda: version)


# --- the happy path -----------------------------------------------------------


def test_a_converged_config_home_verifies_green(converged, monkeypatch):
    _verifying(monkeypatch)

    report = verify(converged, provider="openai")

    assert report.ok and report.findings == []
    assert report.granted == ["power"]
    assert "tools/power.py" in report.declared_files


def test_a_green_verify_records_its_success_as_the_ledgers_evidence(converged, monkeypatch):
    _verifying(monkeypatch)

    report = verify(converged, provider="openai")

    evidence = json.loads((converged / ".verified.json").read_text())
    assert evidence["ok"] is True and evidence["opt_in"] == ["power"]
    assert evidence["checked"].endswith("+00:00")  # UTC, so the ledger can age it
    assert report.evidence == str(converged / ".verified.json")


def test_no_evidence_leaves_the_last_honest_proof_alone(converged, monkeypatch):
    _verifying(monkeypatch)

    assert verify(converged, provider="openai", record=False).ok
    assert not (converged / ".verified.json").exists()


def test_a_failing_verify_never_writes_evidence(converged, monkeypatch):
    # Only success is recorded: a red box must keep its last *honest* green, so the ledger reads
    # the age of a real proof rather than a fresh "still broken" that looks like recent activity.
    _verifying(monkeypatch)
    (converged / "tools" / "notes.py").unlink()

    assert not verify(converged, provider="openai").ok
    assert not (converged / ".verified.json").exists()


# --- what must NOT be a finding ----------------------------------------------


def test_an_operator_edit_is_not_a_finding(converged, monkeypatch):
    # It proves the declared set, never pristine-ness. An edited default is the whole point of the
    # conffile discipline, and a prover that reddens on one gets turned off.
    _verifying(monkeypatch)
    (converged / "tools" / "notes.py").write_text("# mine\n")

    assert verify(converged, provider="openai").ok


def test_a_deletion_a_reconcile_has_ratified_is_not_a_finding(converged, monkeypatch):
    # Delete a benign default, reconcile (which observes it gone and drops it from `files`), and
    # the absence is now declared — green. This is the line the declaration exists to draw.
    _verifying(monkeypatch)
    (converged / "tools" / "notes.py").unlink()
    install(converged, provider="openai")

    report = verify(converged, provider="openai")

    assert report.ok
    assert "tools/notes.py" not in report.declared_files


def test_a_grant_dormant_under_this_provider_is_not_a_finding(tmp_path, monkeypatch):
    # An xAI-only power tool granted on an openai agent is legitimately absent: the grant is kept
    # (so a provider switch restores it) and its absence is not a gap.
    monkeypatch.setattr(_install, "_packaged_defaults", lambda: dict(V1))
    home = tmp_path / "cfg"
    install(home, provider="openai", opt_in=["xai_only"])
    _verifying(monkeypatch)

    report = verify(home, provider="openai")

    assert report.ok
    assert "xai_only" in report.granted  # claimed, dormant, not lost
    assert not (home / "tools" / "xai_only.py").exists()


# --- absence nobody declared --------------------------------------------------


def test_a_stripped_benign_default_is_red(converged, monkeypatch):
    _verifying(monkeypatch)
    (converged / "tools" / "notes.py").unlink()

    report = verify(converged, provider="openai")

    assert checks(report) == ["overlay-file-missing"]
    assert report.findings[0].subject == "tools/notes.py"


def test_a_stripped_granted_power_tool_is_red_with_its_own_diagnosis(converged, monkeypatch):
    # Not merely "a file is missing": the sharper claim is that a *granted capability* is gone,
    # and the remedy names the two ways out (restore it, or revoke the grant).
    _verifying(monkeypatch)
    (converged / "tools" / "power.py").unlink()

    report = verify(converged, provider="openai")

    assert checks(report) == ["opt-in-missing"]
    assert report.findings[0].subject == "power"
    assert "--revoke-opt-in power" in report.findings[0].remedy


def test_a_stripped_prompt_is_red(converged, monkeypatch):
    # The brief is a manifest-tracked overlay file too — a stale or missing charter is exactly as
    # invisible as a missing tool, and exactly as consequential.
    _verifying(monkeypatch)
    (converged / "prompts" / "system-prompt.md").unlink()

    report = verify(converged, provider="openai")

    assert checks(report) == ["overlay-file-missing"]
    assert report.findings[0].subject == "prompts/system-prompt.md"


def test_a_grant_the_package_no_longer_ships_is_red(converged, monkeypatch):
    # The claim cannot be satisfied by any reconcile, so the only honest remedy is to withdraw it.
    _verifying(monkeypatch, defaults={k: v for k, v in V1.items() if k != "tools/power.py"})

    report = verify(converged, provider="openai")

    assert "grant-not-shipped" in checks(report)


# --- unprovable states --------------------------------------------------------


def test_a_config_home_that_was_never_installed_is_red(tmp_path, monkeypatch):
    # The packaged-default fallback is a legitimate way to run and an illegitimate thing to call
    # proven. "Nothing to check, looks fine" is the original defect wearing a prover's clothes.
    _verifying(monkeypatch)

    report = verify(tmp_path / "nothing-here")

    assert checks(report) == ["config-home-not-installed"]


def test_a_missing_declaration_is_red(converged, monkeypatch):
    _verifying(monkeypatch)
    (converged / ".declared.json").unlink()

    report = verify(converged, provider="openai")

    assert checks(report) == ["declaration-missing"]


def test_a_declaration_from_a_future_contract_is_red(converged, monkeypatch):
    # Fail closed on a shape this version does not know: reading it with today's assumptions is
    # how a prover starts confirming things that are not true.
    _verifying(monkeypatch)
    (converged / ".declared.json").write_text(
        json.dumps({"contract": DECLARATION_CONTRACT + 1, "opt_in": [], "files": []})
    )

    report = verify(converged, provider="openai")

    assert checks(report) == ["declaration-contract-unknown"]


def test_a_damaged_declaration_reads_as_undeclared_rather_than_crashing(converged, monkeypatch):
    _verifying(monkeypatch)
    (converged / ".declared.json").write_text("{not json")

    assert checks(verify(converged, provider="openai")) == ["declaration-missing"]


# --- the package / pin axis ---------------------------------------------------


def test_an_unreconciled_upgrade_is_red(converged, monkeypatch):
    # The plain `pip install -U` catch: the package moved, the materialized overlay did not.
    _verifying(monkeypatch, version="99.0.0")

    report = verify(converged, provider="openai")

    assert "config-home-stale" in checks(report)


def test_a_pin_mismatch_is_red(converged, monkeypatch):
    # A pin is the deployer's number, so it can only arrive as an argument — the one check the
    # package cannot make about itself.
    _verifying(monkeypatch)

    report = verify(converged, provider="openai", expect_version="0.0.1")

    assert "package-pin-mismatch" in checks(report)
    assert verify(converged, provider="openai", expect_version=__version__).ok


def test_a_half_applied_install_is_red(converged, monkeypatch):
    # Installed metadata and the imported package disagree: new metadata over an import path that
    # still resolves somewhere else.
    _verifying(monkeypatch)
    monkeypatch.setattr(_verify, "_distribution_version", lambda: "0.0.1")

    assert "package-version-mismatch" in checks(verify(converged, provider="openai"))


def test_a_default_the_upgrade_added_but_never_laid_down_is_red(converged, monkeypatch):
    # The capability exists in the package and not on the box — green everywhere else.
    _verifying(monkeypatch, defaults={**V1, "tools/fresh.py": BENIGN})

    report = verify(converged, provider="openai")

    assert "default-not-installed" in checks(report)
    assert "tools/fresh.py" in report.findings[0].detail


def test_an_overlay_pinned_to_an_older_default_text_is_red(converged, monkeypatch):
    # The stamp can be current and the overlay still stale (a partial reconcile), so the manifest
    # is compared against the package directly rather than trusted via the version stamp.
    _verifying(monkeypatch, defaults={**V1, "tools/notes.py": BENIGN + "# v2\n"})

    report = verify(converged, provider="openai")

    assert "overlay-stale" in checks(report)


# --- the self-ratifying prune -------------------------------------------------


def test_an_overlay_reconciled_for_another_provider_is_red(converged, monkeypatch):
    # An installer run without the agent's AI_PROVIDER filters for the default and prunes its
    # whole vendor tranche — legitimately by its own lights, updating the manifest as it goes, so
    # every later reconcile agrees and nothing is left to notice. Recording the provider the
    # reconcile filtered for is what turns that into something a converge can see.
    _verifying(monkeypatch)

    report = verify(converged, provider="xai")

    assert "provider-mismatch" in checks(report)
    assert "openai" in report.findings[0].detail and "xai" in report.findings[0].detail


def test_the_probe_reads_agent_env_when_launched_without_the_agents_environment(
    tmp_path, monkeypatch
):
    # The check that catches a mis-provided install must not itself be mis-provided: a converge
    # running the probe bare would otherwise compare an xAI agent's declaration against the
    # `openai` default and report a mismatch that is a property of the probe, not the box.
    monkeypatch.setattr(_install, "_packaged_defaults", lambda: dict(V1))
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    home = tmp_path / "cfg"
    install(home, provider="xai", opt_in=["xai_only"])
    # The last assignment wins, exactly as systemd's EnvironmentFile= resolves it — an operator who
    # switched provider by appending a line must not have the prover read the stale one.
    (home / "agent.env").write_text(
        "# the operator's env\nAI_PROVIDER=openai\nexport AI_PROVIDER='xai'\n"
    )
    _verifying(monkeypatch)

    report = verify(home)

    assert report.ok
    assert (report.active_provider, report.active_provider_source) == ("xai", "agent.env")


def test_an_assumed_provider_says_so_in_the_finding(tmp_path, monkeypatch):
    # Still red — an unprovable claim is red — but the diagnosis sends a human to the probe's
    # launch rather than to the box.
    monkeypatch.setattr(_install, "_packaged_defaults", lambda: dict(V1))
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    home = tmp_path / "cfg"
    install(home, provider="xai")
    _verifying(monkeypatch)

    report = verify(home)

    assert "provider-mismatch" in checks(report)
    assert report.active_provider_source == "default"
    assert "agent.env" in report.findings[0].detail  # names why it had to assume


def test_an_all_providers_install_matches_every_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(_install, "_packaged_defaults", lambda: dict(V1))
    home = tmp_path / "cfg"
    install(home, provider=None, opt_in=["power", "xai_only"])
    _verifying(monkeypatch)

    assert read_declaration(home)["provider"] is None
    assert verify(home, provider="xai").ok


# --- the claims emitter -------------------------------------------------------


def test_claims_carry_one_dependency_row_per_declared_capability(converged, monkeypatch):
    _verifying(monkeypatch)

    manifest = claims(converged, subject="jt")

    assert manifest["contract"] == 1
    assert manifest["subject"] == "agent:jt"
    assert manifest["component"] == "basecradle-harness"
    ids = [row["claim"] for row in manifest["claims"]]
    assert "harness-config-home" in ids and "harness-package-pin" in ids
    assert "overlay-tool:power" in ids and "overlay-tool:notes" in ids
    assert "overlay-prompt:system-prompt.md" in ids
    # Every *capability* row is `dependency` — present after a converge, re-proven by the converge
    # floor. The log-grammar rows are deliberately not (a `rare` forced exercise on a TTL, which is
    # the one thing the floor structurally cannot do); they are pinned in `test_log_grammar.py`.
    for row in manifest["claims"]:
        if row["claim"].startswith("log-grammar:"):
            continue
        assert row["class"] == "dependency"
        assert row["ttl_hours"] is None
        assert row["prove"]["kind"] == "probe"
        assert "basecradle-harness-verify" in row["prove"]["cmd"]
        assert str(converged) in row["prove"]["cmd"]  # pinned to the home the claims describe
        assert row["evidence"] == str(converged / ".verified.json")


def test_claims_are_emitted_even_when_the_box_is_failing(converged, monkeypatch):
    # A claim that disappears when it stops being true would make the ledger agree with the box
    # precisely when the box is wrong. The granted-but-missing tool is the row that must survive.
    _verifying(monkeypatch)
    (converged / "tools" / "power.py").unlink()

    ids = [row["claim"] for row in claims(converged)["claims"]]

    assert "overlay-tool:power" in ids
    assert not verify(converged, provider="openai").ok


def test_a_box_with_no_config_home_still_has_rows_to_be_red_about(tmp_path):
    ids = [row["claim"] for row in claims(tmp_path / "nothing-here")["claims"]]

    # The log-grammar row is unconditional for the same reason the other two are: an agent that
    # cannot emit the grammar a founder-named alarm matches must have a row to be red about.
    assert ids == ["harness-config-home", "harness-package-pin", "log-grammar:billing_blocked"]


def test_the_subject_slug_falls_back_to_the_env_then_the_os_user(converged, monkeypatch):
    monkeypatch.setenv("BASECRADLE_AGENT_SLUG", "glm-5.2")
    assert claims(converged)["subject"] == "agent:glm-5.2"
    assert claims(converged, subject="explicit")["subject"] == "agent:explicit"


# --- the CLI ------------------------------------------------------------------


def test_the_cli_exits_nonzero_and_diagnoses_on_stderr(converged, monkeypatch, capsys):
    _verifying(monkeypatch)
    (converged / "tools" / "power.py").unlink()

    code = main(["--config-home", str(converged), "--provider", "openai"])

    captured = capsys.readouterr()
    assert code == 1
    assert "opt-in-missing" in captured.err  # stderr, so a JSON-capturing converge still sees it
    assert captured.out == ""


def test_the_cli_exits_zero_and_says_so_when_green(converged, monkeypatch, capsys):
    _verifying(monkeypatch)

    code = main(["--config-home", str(converged), "--provider", "openai"])

    assert code == 0
    assert "OK" in capsys.readouterr().out


def test_the_cli_json_mode_keeps_the_exit_code(converged, monkeypatch, capsys):
    _verifying(monkeypatch)
    (converged / "tools" / "notes.py").unlink()

    code = main(["--config-home", str(converged), "--provider", "openai", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1 and payload["ok"] is False
    assert payload["findings"][0]["check"] == "overlay-file-missing"
    assert payload["notes"]  # the honest gaps ride in-band, never left to prose


def test_emit_claims_exits_zero_on_a_failing_box(converged, monkeypatch, capsys):
    _verifying(monkeypatch)
    (converged / "tools" / "power.py").unlink()

    code = main(["--config-home", str(converged), "--emit-claims", "--subject", "jt"])

    assert code == 0  # a declaration, not a verdict
    assert json.loads(capsys.readouterr().out)["subject"] == "agent:jt"


# --- the no-arg emitter (issue #376) ------------------------------------------
#
# The seam: the NOC's enumerated-op wrapper runs a component's emitter from its own baked map, as a
# bare bin with NO arguments, as the agent's OS user under `env -i` (HOME + a venv-first PATH). What
# it requires is exactly three things — a Contract v1 object on stdout, the invoking agent as the
# subject, and exit 0 whatever the verify verdict — so those are what these pin. The environment is
# reproduced rather than described: every test here clears the two overrides the wrapper does not
# pass, and points HOME at the box under test.


@pytest.fixture
def as_the_agent(tmp_path, monkeypatch):
    """A converged box reached the way the wrapper reaches it: ``HOME`` alone, no overrides.

    Deliberately *not* the `converged` fixture with an override bolted on. The whole no-arg
    contract is that ``$HOME/.config/basecradle`` is the answer, so a test that pointed
    ``BASECRADLE_CONFIG_HOME`` at the box would prove the emitter works in an environment the
    wrapper never gives it.
    """
    monkeypatch.setattr(_install, "_packaged_defaults", lambda: dict(V1))
    monkeypatch.delenv("BASECRADLE_CONFIG_HOME", raising=False)
    monkeypatch.delenv("BASECRADLE_AGENT_SLUG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "home" / ".config" / "basecradle"
    install(root, provider="openai", opt_in=["power"])
    assert _install.config_home() == root  # the resolution under test, not an assumption
    return root


def test_the_emitter_prints_a_contract_v1_manifest_for_the_invoking_agent(
    as_the_agent, monkeypatch, capsys
):
    _verifying(monkeypatch)
    monkeypatch.setattr(_observability.getpass, "getuser", lambda: "jt")

    code = emit_main([])

    assert code == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["contract"] == 1
    assert manifest["component"] == "basecradle-harness"
    # The wrapper refuses any manifest whose subject is not the agent it launched, and it passes no
    # --subject: the OS user *is* the answer, per the fleet's one-slug-everywhere rule.
    assert manifest["subject"] == "agent:jt"
    assert "overlay-tool:power" in [row["claim"] for row in manifest["claims"]]


def test_the_emitter_exits_zero_on_a_red_box(as_the_agent, monkeypatch, capsys):
    """The whole reason bare verify cannot serve this seam: a finding must not suppress the rows."""
    _verifying(monkeypatch)
    (as_the_agent / "tools" / "power.py").unlink()  # the strip verify reddens on
    assert not verify(as_the_agent, provider="openai", record=False).ok

    code = emit_main([])

    assert code == 0
    ids = [row["claim"] for row in json.loads(capsys.readouterr().out)["claims"]]
    assert "overlay-tool:power" in ids  # the row that must be there *to be* red


def test_the_emitter_matches_emit_claims_byte_for_byte(as_the_agent, monkeypatch, capsys):
    """One red box, two entrypoints, identical bytes — one serializer guarantees it, not discipline.

    The NOC installs whichever one it ran as the root-owned file every later probe resolves its
    `cmd` out of, so a drift between them would be a manifest that describes a box nobody emitted.
    Neither call names a subject: the *default* resolution has to agree too, or the wrapper's
    subject check would pass for one form and fail for the other on the same box.
    """
    _verifying(monkeypatch)
    (as_the_agent / "tools" / "power.py").unlink()  # the DoD's state: a red verdict

    assert emit_main([]) == 0
    from_emitter = capsys.readouterr().out
    assert main(["--config-home", str(as_the_agent), "--emit-claims"]) == 0
    from_verify = capsys.readouterr().out

    assert from_emitter == from_verify
    assert json.loads(from_emitter)["claims"]  # not two empty strings agreeing


def test_the_emitter_still_states_the_unconditional_rows_with_no_config_home(
    tmp_path, monkeypatch, capsys
):
    """A box with nothing installed emits and exits 0 — refusing would delete its own red row.

    Exiting nonzero here reads as the safer answer and is the worse one: `provision-claims` would
    install no manifest, so the ledger would hold *zero* rows for this agent and the specific,
    probeable `harness-config-home` red becomes a generic "the emitter failed" — green-while-absent
    reappearing one level up, inside the instrument built to catch it.
    """
    monkeypatch.delenv("BASECRADLE_CONFIG_HOME", raising=False)
    monkeypatch.delenv("BASECRADLE_AGENT_SLUG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "never-installed"))

    code = emit_main([])

    assert code == 0
    manifest = json.loads(capsys.readouterr().out)
    assert [row["claim"] for row in manifest["claims"]] == [
        "harness-config-home",
        "harness-package-pin",
        "log-grammar:billing_blocked",
    ]


def test_the_emitter_exits_nonzero_and_says_why_when_it_cannot_emit(monkeypatch, capsys):
    """The one nonzero case: the claims genuinely cannot be stated. The reason goes to stderr.

    The wrapper keeps a bounded head of stderr and shows it on failure, so a one-line reason is
    what turns "the component cannot state its claims" into something actionable — where an
    uncaught traceback would spend that budget on the *top* of a stack it truncates.
    """

    def unresolvable(_home=None):
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(_verify, "config_home", unresolvable)

    code = emit_main([])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""  # never half a manifest
    assert "cannot state this agent's claims" in captured.err
    assert "Could not determine home directory." in captured.err


def test_the_emitter_takes_no_arguments(as_the_agent, monkeypatch):
    """Not an omission: the wrapper validates the subject against the agent it launched, so an
    emitter that accepted a --subject would be offering to state claims about an agent it is not."""
    _verifying(monkeypatch)

    for rejected in (["--subject", "someone-else"], ["--config-home", str(as_the_agent)]):
        with pytest.raises(SystemExit) as exit_:
            emit_main(rejected)
        assert exit_.value.code == 2  # argparse refused the question, not the emitter the claims


def test_the_claims_document_is_the_bytes_that_get_written(converged, monkeypatch):
    _verifying(monkeypatch)

    document = claims_document(converged, subject="jt")

    assert json.loads(document) == claims(converged, subject="jt")
    assert document == json.dumps(claims(converged, subject="jt"), indent=2, sort_keys=True)


# --- the incident-1 regression ------------------------------------------------


def test_instance_1_regression_red_on_a_bare_upgrade_green_after_the_reconcile(
    converged, monkeypatch, caplog
):
    """From a converged state, a plain ``pip install -U`` must turn red, and a reconcile green.

    This is the program's acceptance case for this repo (issue #374, incident instance 1). The
    upgrade is faithful in the three ways that matter: the running package's version moves, its
    shipped default set gains a tool, and the opt-in the box had is gone — while the materialized
    config home, which ``pip`` never touches, stays exactly as the last converge left it.

    Before #374 every one of those was invisible: the installer's own semantics read the missing
    opt-in as a deliberate deletion and ratified it, and nothing anywhere compared the overlay
    against the package that was actually running.
    """
    upgraded = {**V1, "tools/fresh.py": BENIGN}
    monkeypatch.setattr(_install, "_packaged_defaults", lambda: dict(upgraded))
    monkeypatch.setattr(_install, "__version__", "99.0.0")
    _verifying(monkeypatch, defaults=upgraded, version="99.0.0")
    (converged / "tools" / "power.py").unlink()  # the prune

    red = verify(converged, provider="openai")

    assert not red.ok
    assert set(checks(red)) == {"config-home-stale", "opt-in-missing", "default-not-installed"}
    assert red.findings  # and each one names its own remedy
    assert all(finding.remedy for finding in red.findings)

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        report = install(converged, provider="openai")  # the converge's reconcile

    assert "tools/power.py" in report.of("restored")  # the strip is healed, and said out loud
    assert (converged / "tools" / "fresh.py").exists()  # the new default landed
    assert verify(converged, provider="openai").ok
