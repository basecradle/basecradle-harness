"""Fail-closed proof that an agent's declared capability set is actually present (issue #374).

The failure this exists to make impossible has a shape, and the shape is the point: **a system
that reads green while a capability is silently absent.** Fleet observability catches failures
that *happen*; nothing happened here. An overlay tool stripped by a plain ``pip install -U``, a
config home never reconciled after the package moved under it, an opt-in a mis-provided converge
pruned — none of these raise, none log, and none stop the agent. It simply cannot do a thing it
believes it can, and the first symptom is a peer asking for something it silently never had.

So the rule is the program's Layer 1 floor: **no capability may be claimed without something that
turns red when it is absent.** `_install` writes the claim (``.declared.json`` — the grants, the
provider filtered for, the managed files present when the reconcile finished); this module is the
prover, and ``basecradle-harness-verify`` is the command a converge runs to make a missing
capability *fail the converge* rather than survive it by luck.

Two properties are load-bearing and easy to erode:

- **It proves the declared set, never pristine-ness.** The conffile discipline's whole point is
  that an operator may edit a default and delete one; both are legitimate and neither is a
  finding. What is *not* legitimate is an absence nobody declared — a file that was there at the
  last reconcile and is gone now — and the declaration is exactly the line between the two. A
  verify that reddened on an operator's edit would be turned off within a week, and a turned-off
  prover proves nothing.
- **Unproven is red.** A config home that was never installed, a declaration that will not parse,
  a declaration written to a ``contract`` this version does not know — each is a state in which
  the claim *cannot be checked*, and every one of them exits nonzero. Reporting "nothing to
  check, looks fine" is the original defect wearing a prover's clothes.

What it deliberately does **not** prove is *activation*: a tool file can be present and still not
activate (its provider, its key, the locked policy). That is a different question with a different
authority — ``basecradle-harness-wake --resolved-config`` reads the live resolution on the box —
and folding it in here would make a missing ``AI_API_KEY`` read as a stripped overlay. The report
says so in-band (``notes``) rather than leaving it for a reader to remember.

The claims emitter is the other half: it prints this agent's rows for the fleet's
claims-vs-evidence ledger (Claims Manifest Contract v1, owned by the NOC), each pointing at this
same command as its probe. It is deliberately **independent of the verdict** — a box whose verify
is red must still emit its claims, or the ledger loses the row and the absence goes back to being
invisible, which is where this started.

It has **two** entrypoints, and the second one is the seam the NOC's collector actually uses
(issue #376). ``basecradle-harness-verify --emit-claims`` is the human/ad-hoc form; the NOC's
enumerated-op wrapper instead runs a component's emitter from a **baked** map
(``KNOWN_CLAIM_EMITTERS[basecradle-harness]=basecradle-harness-claims``) as a bare bin with **no
arguments**, as the agent's own OS user under ``env -i`` — so the emitter must be a command whose
*whole* behavior is "print this agent's manifest and exit 0". The bare ``basecradle-harness-verify``
cannot be that command: it prints a human report and exits 1 on a finding. Hence
``basecradle-harness-claims`` (`emit_main`), a thin alias that serializes through the same
`claims_document` so the two can never drift.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from basecradle_harness._install import (
    DECLARATION_CONTRACT,
    _hash,
    _is_tool_plugin,
    _packaged_defaults,
    _read_manifest,
    _relevant_defaults,
    active_provider_from_env,
    config_home,
    declared_files,
    declared_opt_in,
    installed_version,
    plugin_opts_in,
    read_declaration,
)
from basecradle_harness._log_grammar import COLUMNS as LOG_GRAMMAR_COLUMNS
from basecradle_harness._log_grammar import IDENTIFIER as LOG_GRAMMAR_IDENTIFIER
from basecradle_harness._log_grammar import SCRIPT as LOG_GRAMMAR_SCRIPT
from basecradle_harness._log_grammar import TTL_HOURS as LOG_GRAMMAR_TTL_HOURS
from basecradle_harness._observability import agent_slug
from basecradle_harness._version import __version__

_log = logging.getLogger("basecradle_harness")

#: The PyPI distribution whose installed metadata is compared against the imported version — the
#: half-applied-upgrade check. They disagree when a ``pip install -U`` wrote new metadata over an
#: import path that still resolves elsewhere (a stale editable install, a shadowed site-packages).
DISTRIBUTION = "basecradle-harness"

#: Where a *successful* verify records itself, inside the config home the agent already owns (no
#: root, nothing under ``/etc``). Only success is written, deliberately: the file's timestamp is
#: then the ledger's age-of-proof, and a box that goes red keeps its last honest green rather than
#: overwriting it with a fresher "still broken" that would read as recent activity.
_EVIDENCE_NAME = ".verified.json"

#: The claims-manifest schema this emitter speaks (Claims Manifest Contract v1). The NOC owns the
#: contract; a change to it is proposed upward, never negotiated peer-to-peer.
CLAIMS_CONTRACT = 1

#: The ``component`` every emitted claim carries — the repo answering for these capabilities.
COMPONENT = "basecradle-harness"

#: The operator's env file inside the config home. The harness never *sources* it (systemd's
#: ``EnvironmentFile=`` does, so a wake already has these in its process env) — but a converge
#: probe may be launched without it, and this prover's provider check would then compare the
#: agent's declaration against a default nobody chose. So it is read, as a fallback only, for the
#: one variable that decides which defaults an install lays down.
_AGENT_ENV_NAME = "agent.env"

#: The honest gaps, stated in-band rather than left in prose for a caller to remember — the same
#: never-silently-omit discipline `_resolve` applies to its own answer.
_NOTES = (
    (
        "Presence and currency only: a declared tool file being present does not mean it "
        "activates (its provider, its credential, and the locked policy all gate that). "
        "`basecradle-harness-wake --resolved-config` is the authority on what is live on this box."
    ),
    (
        "Operator-added files in tools/ are never checked — nothing declares them, so nothing "
        "can say whether their absence is a strip or a tidy-up."
    ),
    "mcp/ ships empty and is operator-owned, so it carries no declaration and is not verified.",
)


# --- findings -----------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One reason the declared set is not provably present.

    Args:
        check: A stable slug for the *kind* of gap (``overlay-file-missing``,
            ``config-home-stale``, …), so an alert can group on it across releases without
            matching on prose.
        subject: What the gap is about — a config-home-relative path, an opt-in stem, or
            ``"config home"`` for a whole-install finding.
        detail: The one-line diagnosis, naming the observed values rather than restating the rule.
        remedy: The command (or decision) that closes it. A finding with no actionable remedy is
            a complaint, and a complaint gets ignored.
    """

    check: str
    subject: str
    detail: str
    remedy: str

    def as_json(self) -> dict[str, str]:
        return {
            "check": self.check,
            "subject": self.subject,
            "detail": self.detail,
            "remedy": self.remedy,
        }

    def line(self) -> str:
        return f"  [{self.check}] {self.subject}: {self.detail}\n      → {self.remedy}"


@dataclass
class VerifyReport:
    """The verdict plus everything it was read off — machine-readable, and honest about its gaps."""

    config_home: Path
    harness_version: str
    distribution_version: str | None
    declared_provider: str | None
    active_provider: str
    #: Where `active_provider` came from — ``argument`` | ``environment`` | ``agent.env`` |
    #: ``default``. Reported because it changes what a provider mismatch means (see
    #: `_active_provider`), and because ``default`` is the one value that says the probe may have
    #: been launched without the agent's environment.
    active_provider_source: str = "default"
    declared_files: list[str] = field(default_factory=list)
    granted: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    evidence: str | None = None  # the success record's path, when one was written

    @property
    def ok(self) -> bool:
        """Green iff nothing came back. Unproven never reads as green — see the module docstring."""
        return not self.findings

    def as_json(self) -> dict[str, object]:
        """The stable report shape — the field contract the NOC's converge may depend on."""
        return {
            "ok": self.ok,
            "config_home": str(self.config_home),
            "harness_version": self.harness_version,
            "distribution_version": self.distribution_version,
            "declared_provider": self.declared_provider,
            "active_provider": self.active_provider,
            "active_provider_source": self.active_provider_source,
            "declared_files": list(self.declared_files),
            "opt_in": list(self.granted),
            "findings": [finding.as_json() for finding in self.findings],
            "evidence": self.evidence,
            "notes": list(_NOTES),
        }

    def summary(self) -> str:
        """A short human verdict: one headline, then one stanza per finding."""
        if self.ok:
            return (
                f"basecradle-harness-verify: OK — basecradle-harness {self.harness_version}, "
                f"{len(self.declared_files)} declared file(s), "
                f"{len(self.granted)} granted opt-in tool(s). "
                f"Config home: {self.config_home}"
            )
        head = (
            f"basecradle-harness-verify: FAILED — {len(self.findings)} finding(s). "
            f"Config home: {self.config_home}"
        )
        return "\n".join([head, *(finding.line() for finding in self.findings)])


# --- the checks ---------------------------------------------------------------


def _distribution_version() -> str | None:
    """The installed ``basecradle-harness`` distribution version, or ``None`` if not installed."""
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def _provider_from_agent_env(root: Path) -> str | None:
    """``AI_PROVIDER`` as the agent's own ``agent.env`` declares it, or ``None``.

    A deliberately minimal reader — one variable, no expansion, no sourcing, ``shell=False`` in
    spirit — because the file is the operator's and this is a *fallback*, not a config path. It
    exists for one failure it would otherwise cause: a converge that runs the probe without the
    agent's environment would compare a vendor agent's declaration against the ``openai`` default
    and report a provider mismatch that is a property of the probe, not of the box. The check that
    catches a real mis-provided install must not itself be mis-provided.
    """
    path = root / _AGENT_ENV_NAME
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    found: str | None = None
    for raw in lines:
        line = raw.strip().removeprefix("export ").strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() != "AI_PROVIDER":
            continue
        # Trailing comments are not stripped: systemd does not strip them either, so an operator
        # who wrote one has a value with a `#` in it as far as the running agent is concerned, and
        # reading it more cleverly here would make this disagree with what the wake sees. For the
        # same reason the **last** assignment wins rather than the first — that is what systemd's
        # `EnvironmentFile=` does, and an operator who switched provider by appending a line must
        # not have this prover reading the value their agent no longer runs under.
        cleaned = value.strip().strip("\"'").strip().lower()
        if cleaned:
            found = cleaned
    return found


def _active_provider(root: Path, explicit: str | None) -> tuple[str, str]:
    """The provider this agent runs, and **where that was learned** — argument, env, file, default.

    The source is reported rather than swallowed because it changes what a mismatch *means*: a
    mismatch against a provider read from the agent's own environment is a real finding about the
    box, while one against the bare default may be a finding about how the probe was launched. A
    prover that cannot say which of those it is asking a human to chase down is half a prover.
    """
    if explicit:
        return explicit.strip().lower(), "argument"
    from_env = os.environ.get("AI_PROVIDER", "").strip()
    if from_env:
        return from_env.lower(), "environment"
    from_file = _provider_from_agent_env(root)
    if from_file:
        return from_file, _AGENT_ENV_NAME
    return active_provider_from_env(), "default"


def verify(
    home: str | os.PathLike[str] | None = None,
    *,
    defaults: dict[str, str] | None = None,
    provider: str | None = None,
    expect_version: str | None = None,
    record: bool = True,
) -> VerifyReport:
    """Prove this agent's declared capability set is present and current, or say exactly what is not.

    ``home`` overrides the config-home location (see `config_home`); ``defaults`` overrides the
    shipped default set — the same test seam `install` takes, so an "upgrade" is a *changed*
    default set rather than a second build. ``provider`` overrides the active ``AI_PROVIDER``
    (which is what the declared provider is compared against). ``expect_version`` is the pin the
    caller converged to: supplied, the running package must equal it — this is the only check the
    harness cannot make on its own, because a pin is the deployer's number, not the package's.

    ``record`` writes the success evidence (see `_EVIDENCE_NAME`) when the verdict is green; pass
    ``False`` for a read-only check. A failed evidence write is a warning, never a red verdict —
    the probe already proved the thing; the ledger merely loses one age-of-proof update, and
    turning a green box red over a full disk in its config home would be a worse trade.
    """
    root = config_home(home)
    active, provider_source = _active_provider(root, provider)
    shipped_all = _packaged_defaults() if defaults is None else defaults
    findings: list[Finding] = []
    dist = _distribution_version()

    report = VerifyReport(
        config_home=root,
        harness_version=__version__,
        distribution_version=dist,
        declared_provider=None,
        active_provider=active,
        active_provider_source=provider_source,
        findings=findings,
    )

    _check_package(findings, dist, expect_version)

    manifest = _read_manifest(root)
    if not manifest:
        # Nothing was ever laid down here, so there is no declared set to prove. This is the
        # not-yet-installed fallback path (`_plugins.load_plugins`), which is a legitimate way to
        # *run* — and an illegitimate thing to call proven, which is the only claim being made.
        findings.append(
            Finding(
                check="config-home-not-installed",
                subject="config home",
                detail=(
                    f"no shipped defaults are recorded at {root} — this agent runs off the "
                    "packaged-default fallback, so it declares nothing and nothing can be proven"
                ),
                remedy="run basecradle-harness-install to materialize and declare the overlay",
            )
        )
        return report

    _check_stamp(findings, root)

    declaration = read_declaration(root)
    if not _check_declaration(findings, root, declaration):
        return report

    report.declared_provider = _declared_provider(declaration)
    report.declared_files = sorted(declared_files(declaration))
    report.granted = sorted(declared_opt_in(declaration))

    _check_provider(findings, report.declared_provider, active, provider_source)

    # The default set the *last reconcile* filtered for — read off the declaration, never
    # re-derived from the live env. If the two disagree that is itself a finding (above), and the
    # file-level checks below must still describe the install that actually happened.
    relevant = _relevant_defaults(shipped_all, report.declared_provider)
    granted = set(report.granted)

    _check_grants(findings, root, shipped_all, relevant, granted)
    _check_declared_files(findings, root, report.declared_files, granted)
    _check_currency(findings, manifest, relevant, granted)

    if report.ok and record:
        report.evidence = _record_success(root, report)
    return report


def _check_package(findings: list[Finding], dist: str | None, expect_version: str | None) -> None:
    """The package/pin axis: is the code that is *running* the code that was meant to be?"""
    if dist is None:
        findings.append(
            Finding(
                check="package-not-installed",
                subject=DISTRIBUTION,
                detail=(
                    f"the {DISTRIBUTION} distribution reports no installed metadata, so its "
                    "version cannot be confirmed from outside the import"
                ),
                remedy=f"reinstall {DISTRIBUTION} into this agent's venv",
            )
        )
    elif dist != __version__:
        findings.append(
            Finding(
                check="package-version-mismatch",
                subject=DISTRIBUTION,
                detail=(
                    f"installed metadata says {dist} but the imported package is {__version__} "
                    "— a half-applied upgrade, or an import path shadowing the installed one"
                ),
                remedy=f"reinstall {DISTRIBUTION} into this agent's venv",
            )
        )
    if expect_version and expect_version != __version__:
        findings.append(
            Finding(
                check="package-pin-mismatch",
                subject=DISTRIBUTION,
                detail=f"expected the pinned {expect_version}, running {__version__}",
                remedy=f"pip install '{DISTRIBUTION}=={expect_version}'",
            )
        )


def _check_stamp(findings: list[Finding], root: Path) -> None:
    """The config home must have been reconciled *by the running package*.

    This is the plain ``pip install -U`` catch, and it is the cheapest one: the package moved, the
    materialized overlay did not, and every file below it is a snapshot of a version that is no
    longer running. `reconcile_on_upgrade` normally closes this at the next wake — normally, which
    is precisely why it needs a prover rather than an assumption.
    """
    stamp = installed_version(root)
    if stamp != __version__:
        findings.append(
            Finding(
                check="config-home-stale",
                subject="config home",
                detail=(
                    f"last reconciled by basecradle-harness {stamp or 'an unknown version'}, "
                    f"but {__version__} is running — the overlay predates the installed package"
                ),
                remedy="run basecradle-harness-install",
            )
        )


def _check_declaration(
    findings: list[Finding], root: Path, declaration: Mapping[str, object]
) -> bool:
    """The declaration must exist and speak a contract this version understands. Else: red, stop."""
    if not declaration:
        findings.append(
            Finding(
                check="declaration-missing",
                subject="config home",
                detail=(
                    f"{root / '.declared.json'} is absent or unreadable, so this agent's "
                    "capability set is undeclared and nothing about it can be proven"
                ),
                remedy="run basecradle-harness-install to write the declaration",
            )
        )
        return False
    contract = declaration.get("contract")
    if contract != DECLARATION_CONTRACT:
        # Fail closed on a *newer* shape as firmly as on a broken one: reading it with this
        # version's assumptions is how a prover starts confirming things that are not true.
        findings.append(
            Finding(
                check="declaration-contract-unknown",
                subject="config home",
                detail=(
                    f"the declaration is contract {contract!r}; this harness understands "
                    f"{DECLARATION_CONTRACT}"
                ),
                remedy="run basecradle-harness-install under the harness that wrote it, or upgrade",
            )
        )
        return False
    return True


def _declared_provider(declaration: Mapping[str, object]) -> str | None:
    """The provider the last reconcile filtered for — ``None`` for an unfiltered install."""
    value = declaration.get("provider")
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def _check_provider(
    findings: list[Finding], declared: str | None, active: str, source: str
) -> None:
    """The overlay must have been reconciled for the provider this agent actually runs.

    The self-ratifying prune, made visible: an installer run without the agent's ``AI_PROVIDER``
    in its environment filters for the ``openai`` default and *deletes* a vendor agent's whole
    pristine tranche — legitimately, by its own lights, updating the manifest as it goes, so every
    later reconcile agrees and nothing is left to notice. Recording the provider the reconcile
    filtered for is what turns that into a mismatch a converge can see. ``None`` (an explicit
    ``--all-providers`` install) matches every agent by construction.

    The finding names **where the active provider was learned**, because the same mismatch has two
    very different causes: a real mis-provided install, or a probe launched without the agent's
    environment (``source == "default"``). Both are worth a red — an unprovable claim is red — but
    they send a human to different places.
    """
    if declared is not None and declared != active:
        assumed = (
            " (nothing named a provider — neither the environment nor the config home's "
            "agent.env — so the shipped default was assumed; run the probe with the agent's "
            "environment to rule the probe itself out)"
            if source == "default"
            else f" (read from {source})"
        )
        findings.append(
            Finding(
                check="provider-mismatch",
                subject="config home",
                detail=(
                    f"the overlay was reconciled for provider {declared!r} but this agent runs "
                    f"AI_PROVIDER={active!r}{assumed} — its {active!r} tools were filtered out, "
                    "and any it already had were pruned"
                ),
                remedy=(
                    f"run basecradle-harness-install with AI_PROVIDER={active} set "
                    f"(or --provider {active})"
                ),
            )
        )


def _check_grants(
    findings: list[Finding],
    root: Path,
    shipped_all: Mapping[str, str],
    relevant: Mapping[str, str],
    granted: set[str],
) -> None:
    """Every granted opt-in tool must be on disk — the stripped-power-tool check.

    Three ways a grant fails to be a file, and they are different diagnoses with different
    remedies: the package no longer ships the tool at all; it ships it for another provider (not a
    finding — the grant is legitimately dormant, kept so a provider switch restores it); or it is
    shipped, relevant, and simply **not there**, which is the strip.
    """
    for stem in sorted(granted):
        rel = f"tools/{stem}.py"
        if rel not in shipped_all:
            findings.append(
                Finding(
                    check="grant-not-shipped",
                    subject=stem,
                    detail=(
                        f"granted, but basecradle-harness {__version__} ships no tools/{stem}.py "
                        "— the capability is claimed and cannot exist"
                    ),
                    remedy=f"basecradle-harness-install --revoke-opt-in {stem}",
                )
            )
        elif rel not in relevant:
            continue  # granted but dormant under this provider — kept, and not a gap
        elif not root.joinpath("tools", f"{stem}.py").exists():
            findings.append(
                Finding(
                    check="opt-in-missing",
                    subject=stem,
                    detail=(
                        f"granted opt-in tool is missing from the overlay ({rel}) — it was "
                        "stripped, and this agent silently cannot do what it claims"
                    ),
                    remedy=(
                        f"run basecradle-harness-install to restore it, or "
                        f"--revoke-opt-in {stem} if the removal was intended"
                    ),
                )
            )


def _check_declared_files(
    findings: list[Finding], root: Path, files: Sequence[str], granted: set[str]
) -> None:
    """Every file the last reconcile saw present must still be present.

    This is the whole distinction the declaration buys. An operator deletion is ratified the next
    time a reconcile runs and sees it gone — it drops out of ``files`` and is never spoken of
    again. A file that was there when the reconcile finished and is gone now was removed by
    something nobody told, which is the definition of the failure this module exists for.
    Granted power tools are skipped here because `_check_grants` gives them a sharper diagnosis.
    """
    for rel in files:
        stem = rel[len("tools/") : -len(".py")] if _is_tool_plugin(rel) else None
        if stem is not None and stem in granted:
            continue
        if not root.joinpath(*rel.split("/")).exists():
            findings.append(
                Finding(
                    check="overlay-file-missing",
                    subject=rel,
                    detail=(
                        "present at the last reconcile and gone now — removed by something that "
                        "did not update the declaration"
                    ),
                    remedy=(
                        "run basecradle-harness-install: it will re-lay the file if the shipped "
                        "default changed, or ratify the deletion if you meant it"
                    ),
                )
            )


def _check_currency(
    findings: list[Finding],
    manifest: Mapping[str, str],
    relevant: Mapping[str, str],
    granted: set[str],
) -> None:
    """Every default this agent should have must be tracked, and tracked at the *current* text.

    Two gaps, kept apart because their causes are: a default the package ships and the manifest
    has never heard of (a new tool an upgrade added, with no reconcile since — the capability
    exists in the package and not on the box), and one whose recorded hash is an older release's
    (the overlay is a snapshot of a version that stopped running). The stale set is reported as
    one finding rather than one per file: they always move together, and fourteen rows saying the
    same sentence buries the ones that do not.
    """
    expected = {
        rel: text
        for rel, text in relevant.items()
        if not (_is_tool_plugin(rel) and plugin_opts_in(text))
        or rel[len("tools/") : -len(".py")] in granted
    }
    missing = sorted(rel for rel in expected if rel not in manifest)
    if missing:
        findings.append(
            Finding(
                check="default-not-installed",
                subject=f"{len(missing)} shipped default(s)",
                detail=("shipped by this package but never laid down here: " + ", ".join(missing)),
                remedy="run basecradle-harness-install",
            )
        )
    stale = sorted(
        rel for rel, text in expected.items() if rel in manifest and manifest[rel] != _hash(text)
    )
    if stale:
        findings.append(
            Finding(
                check="overlay-stale",
                subject=f"{len(stale)} tracked file(s)",
                detail=(
                    "the manifest records a different default text than this package ships, so "
                    "the overlay is a snapshot of an older release: " + ", ".join(stale)
                ),
                remedy="run basecradle-harness-install",
            )
        )


def _record_success(root: Path, report: VerifyReport) -> str | None:
    """Write the success evidence the ledger ages, or warn and carry on."""
    path = root / _EVIDENCE_NAME
    body = {
        "contract": CLAIMS_CONTRACT,
        "ok": True,
        "harness_version": report.harness_version,
        "provider": report.declared_provider,
        "declared_files": len(report.declared_files),
        "opt_in": list(report.granted),
        "checked": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as error:
        _log.warning("Verified, but could not record the success evidence at %s: %s", path, error)
        return None
    return str(path)


# --- the claims emitter -------------------------------------------------------


def _script(name: str) -> str:
    """A console script's absolute path, resolved beside the running interpreter.

    That is where a venv puts its scripts, so the resulting ``cmd`` works from a converge that has
    neither the agent's ``PATH`` nor its shell. Falls back to the bare name when the script is not
    beside this interpreter (an editable checkout, a test), which keeps the manifest readable
    rather than pointing at a path that does not exist.
    """
    script = Path(sys.executable).with_name(name)
    return str(script) if script.exists() else name


def _probe_command(root: Path) -> str:
    """The command a converge runs to prove these claims — absolute, and explicit about the home.

    Pinned to the config home *this* emitter read, so the probe can never end up proving a
    different agent's overlay than the one the claims describe.
    """
    return f"{shlex.quote(_script('basecradle-harness-verify'))} --config-home {shlex.quote(str(root))}"


def claims(
    home: str | os.PathLike[str] | None = None,
    *,
    subject: str | None = None,
) -> dict[str, object]:
    """This agent's claims for the fleet ledger — Claims Manifest Contract v1.

    One row per declared capability, each ``class: "dependency"`` (a boot/converge check, re-proven
    on change — ``ttl_hours: null``, converge-time only) and each pointing at the same probe: verify
    is holistic by design, because the gaps it finds are not independent (a stale config home makes
    every file in it stale) and N per-file probes would be N re-reads of one directory.

    It is emitted from the **declaration**, not from a resolution, and it is emitted **whatever the
    verdict is**. A red box must still produce its rows — a claim that disappears when it stops
    being true is the exact failure mode the ledger exists to catch, and it would make the
    instrument agree with the box precisely when the box is wrong.

    Two rows are unconditional, so even a box with no config home at all has something in the
    ledger to be red about: ``harness-config-home`` (installed, current, reconciled for this
    agent's provider) and ``harness-package-pin`` (the running package is the installed one).
    """
    root = config_home(home)
    declaration = read_declaration(root)
    files = declared_files(declaration)
    granted = declared_opt_in(declaration)
    probe = _probe_command(root)
    evidence = str(root / _EVIDENCE_NAME)

    def row(claim: str) -> dict[str, object]:
        return {
            "claim": claim,
            "class": "dependency",
            "prove": {"kind": "probe", "cmd": probe},
            "evidence": evidence,
            "ttl_hours": None,
        }

    ids = ["harness-config-home", "harness-package-pin"]
    # A granted tool is claimed even when its file never landed — that row is the one that must be
    # red, so deriving the set from what is *present* would drop exactly the claim that matters.
    stems = {rel[len("tools/") : -len(".py")] for rel in files if _is_tool_plugin(rel)} | granted
    ids += [f"overlay-tool:{stem}" for stem in sorted(stems)]
    ids += [f"overlay-prompt:{rel[len('prompts/') :]}" for rel in sorted(files) if _is_prompt(rel)]
    return {
        "contract": CLAIMS_CONTRACT,
        "subject": f"agent:{agent_slug(subject)}",
        "component": COMPONENT,
        "claims": [row(claim) for claim in ids] + _log_grammar_rows(),
    }


def _log_grammar_rows() -> list[dict[str, object]]:
    """The log-grammar claims — one per NOC derived column this package's grammar feeds.

    Different in class from every other row here, and deliberately so. The rest of this manifest is
    ``dependency``: *is the capability present after a converge?*, re-proven by the converge floor.
    A log-grammar claim asks something the floor structurally cannot — *do the bytes a founder-named
    alarm matches still exist?* — about a line that appears **only when a vendor account runs out of
    money**. Silence is its normal state, so its proof is a forced exercise on a TTL: `class: rare`
    (basecradle-noc#509, ratified 2026-08-18).

    Unconditional, like the two config-home rows above: an agent that cannot emit its billing
    grammar must have a row in the ledger to be red about, and deriving the set from anything
    observable here would drop exactly the claim that matters.
    """
    return [
        {
            "claim": f"log-grammar:{column}",
            "class": "rare",
            "prove": {
                "kind": "probe",
                "cmd": f"{shlex.quote(_script(LOG_GRAMMAR_SCRIPT))} {column}",
            },
            "evidence": f"journal:{LOG_GRAMMAR_IDENTIFIER}",
            "ttl_hours": LOG_GRAMMAR_TTL_HOURS,
        }
        for column in LOG_GRAMMAR_COLUMNS
    ]


def _is_prompt(rel: str) -> bool:
    """Whether a config-home-relative path is one of the shipped charter/brief prompt files."""
    return rel.startswith("prompts/")


def claims_document(
    home: str | os.PathLike[str] | None = None,
    *,
    subject: str | None = None,
) -> str:
    """The claims manifest as the exact bytes an emitter prints — serialized in **one** place.

    Two commands emit this manifest — ``basecradle-harness-claims`` and
    ``basecradle-harness-verify --emit-claims`` — and for the same box they must produce the same
    bytes, because the NOC installs one of them root-owned as the file the ledger and every later
    probe read. "They both call ``json.dumps``" is not that guarantee; it is two call sites that
    agree until one of them gains a keyword. So there is one call site, and both entrypoints write
    what it returns.

    ``ensure_ascii`` is left at its default *deliberately* here, which is the opposite of the rule
    `_session` follows: these bytes are not context to be measured, they are a file written under
    ``env -i LC_ALL=C``, where ``sys.stdout`` encodes as ASCII. An escaped manifest always survives
    that pipe; a raw one would fail on the first non-Latin character in a path.
    """
    return json.dumps(claims(home, subject=subject), indent=2, sort_keys=True)


# --- the CLI ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """The ``basecradle-harness-verify`` entrypoint: prove the declared set, or emit the claims.

    Exit **0** only when every check passed; **1** on any finding, on an unprovable state, and on a
    bad question. Nonzero is the whole product: it is what a converge reads, and it is what turns
    "the overlay was silently pruned" from a thing nobody learns into a red deploy.

    ``--emit-claims`` prints the claims manifest instead and exits 0 — the ledger's rows are a
    declaration, not a verdict, and must not vanish on a box that is failing.
    """
    parser = argparse.ArgumentParser(
        prog="basecradle-harness-verify",
        description=(
            "Prove this agent's declared capability set is actually present and current: the "
            "shipped defaults it should have, the powerful tools it was granted, the overlay's "
            "currency with the running package. Exits nonzero, with a specific diagnosis, when "
            "anything it claims is missing — including a state in which nothing can be proven."
        ),
    )
    parser.add_argument(
        "--config-home",
        default=None,
        help=(
            "the config home to verify (default: $BASECRADLE_CONFIG_HOME, else "
            "$HOME/.config/basecradle)."
        ),
    )
    parser.add_argument(
        "--provider",
        default=None,
        help=(
            "the provider this agent runs, compared against the one the overlay was reconciled "
            "for (default: $AI_PROVIDER, else 'openai')."
        ),
    )
    parser.add_argument(
        "--expect-version",
        default=None,
        metavar="VERSION",
        help=(
            "the basecradle-harness version this box was pinned to; the running package must "
            "equal it. A pin is the deployer's number, so it can only arrive as an argument."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the full report as JSON instead of the human summary (exit code unchanged).",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help=(
            "do not record the success evidence (a read-only check). The ledger ages that record, "
            "so suppress it only for an ad-hoc run you do not want counted as a proof."
        ),
    )
    parser.add_argument(
        "--emit-claims",
        action="store_true",
        help=(
            "print this agent's claims manifest (Claims Manifest Contract v1) as JSON and exit 0, "
            "instead of verifying. The NOC's converge writes it to /etc/basecradle/claims.d/."
        ),
    )
    parser.add_argument(
        "--subject",
        default=None,
        metavar="SLUG",
        help=(
            "the agent slug for the emitted claims' subject (default: $BASECRADLE_AGENT_SLUG, "
            "else the OS user — one slug across every system, per the fleet identity rule)."
        ),
    )
    args = parser.parse_args(argv)

    if args.emit_claims:
        print(claims_document(args.config_home, subject=args.subject))
        return 0

    report = verify(
        args.config_home,
        provider=args.provider,
        expect_version=args.expect_version,
        record=not args.no_evidence,
    )
    if args.json:
        print(json.dumps(report.as_json(), indent=2, sort_keys=True))
    elif report.ok:
        print(report.summary())
    else:
        # The diagnosis goes to stderr so a converge capturing stdout for the JSON still sees it,
        # and so a failure is never mistaken for output.
        print(report.summary(), file=sys.stderr)
    return 0 if report.ok else 1


def emit_main(argv: list[str] | None = None) -> int:
    """The ``basecradle-harness-claims`` entrypoint: print this agent's manifest, and nothing else.

    This is the shape the NOC's enumerated-op wrapper requires of a claims emitter (issue #376,
    consumer spec basecradle-noc#406). ``provision-claims`` looks the command up in its own **baked**
    ``KNOWN_CLAIM_EMITTERS`` map — the authorization boundary, ratified by installing the wrapper —
    and runs it as the agent's own OS user under ``env -i`` with nothing but ``HOME``, a venv-first
    ``PATH`` and ``LC_ALL=C``. Three properties follow, and none of them is decoration:

    **It takes no arguments, by design, not by omission.** It answers both questions through the
    ordinary resolvers — `config_home` and `agent_slug`, the same two a wake and an installer use,
    overrides and all — which under the wrapper's stripped environment reduce to exactly ``HOME``
    and the OS user, because the fleet sets neither ``BASECRADLE_CONFIG_HOME`` nor
    ``BASECRADLE_AGENT_SLUG``. So the manifest describes *the agent this process is*, which is what
    makes it trustworthy: the wrapper refuses any manifest whose ``subject`` is not
    ``agent:<the slug it launched>``, so an emitter offering a ``--subject`` override would be
    offering to state claims about an agent it is not. `main`'s ``--emit-claims`` keeps those
    switches because it is the ad-hoc form, run by a human who already knows which box they mean.

    **It exits 0 whatever the verify verdict**, for the reason `claims` exists at all: the rows are
    a *declaration*, not a verdict. A red box that emitted nothing would take its own claims out of
    the ledger at the exact moment the ledger is the thing that would have caught it. The verdict
    arrives separately, and later, when the NOC runs each row's probe.

    **A box with no config home therefore still emits** — the two unconditional rows (see `claims`)
    — rather than failing. This is the one place the issue's phrasing and its byte-for-byte
    requirement pull apart, and the resolution is not a preference: exiting nonzero here would make
    ``provision-claims`` install *no manifest at all*, which deletes every row for that agent from
    the ledger and turns a specific, probeable ``harness-config-home`` red into a generic "the
    emitter failed" — the green-while-absent shape reappearing one level up, inside the instrument
    built to catch it. Nonzero is reserved for genuinely *not being able to state the claims*: no
    resolvable ``HOME``, an unwritable stdout, anything unforeseen. For those the wrapper's "the
    component cannot state its claims" is exactly the right failure, and the reason is on stderr,
    which the wrapper keeps (bounded) and shows.
    """
    parser = argparse.ArgumentParser(
        prog="basecradle-harness-claims",
        description=(
            "Print this agent's claims manifest (Claims Manifest Contract v1) as JSON and exit 0. "
            "Takes no arguments: the config home comes from $HOME and the subject slug from the "
            "OS user, so the manifest always describes the agent this command runs as. Exits "
            "nonzero only when the claims cannot be stated at all."
        ),
    )
    parser.parse_args(argv)
    try:
        document = claims_document()
        # Built first, then written and flushed inside the guard: a serialization failure must not
        # leave half a manifest on stdout, and a flush that fails at interpreter shutdown instead
        # would exit nonzero with an "Exception ignored" traceback rather than a stated reason.
        sys.stdout.write(document + "\n")
        sys.stdout.flush()
    except Exception as error:  # noqa: BLE001 - any failure here is "cannot emit", which is the verdict
        # The one-line reason goes **first** and the stack after it, deliberately: the wrapper shows
        # a bounded *head* of stderr, and the head of a traceback is the top of the stack — three
        # venv paths, and never the exception that a reader needs. This way the truncated view is
        # the diagnosis and the full view (journald, or a human running it directly) is the stack.
        print(
            f"basecradle-harness-claims: cannot state this agent's claims: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 1
    return 0
