"""Deterministic stem → resolved-tool-name resolution, for automation off the box (issue #345).

The question this answers, and nothing else: **given a provider (plus sdk/surface/profile) and a
set of opt-in tool stems, which resolved tool names and built-in wire names activate?**

Why it exists
-------------
A tool's *stem* — its plugin file's name, ``xai_search`` — is not its *resolved name*. One stem can
fan out to several names (``xai_search`` → the ``web_search`` **and** ``x_search`` built-ins;
``code_execution`` → the ``code_interpreter`` built-in **and** the ``code_attach`` tool), and a name
can differ from its stem (``hear_audio`` → ``listen``). The fleet inventory declares **stems**; a
both-directions ``exact_tools`` pin asserts **names**. Every consumer that has had to bridge those
two has bridged them by *transcribing prose*, and prose has been wrong about it in more than one
repo — the near-miss in basecradle-noc#344 documented ``xai_search`` as resolving to ``x_search``
alone, which as a pin is a permanently-failing converge. **The mapping must be computed, never
transcribed.** This module is where it is computed.

It is the off-box sibling of ``basecradle-harness-wake --resolved-config``, and the split is the
point:

- ``--resolved-config`` answers *what is **this box** doing right now?* — it reads the agent's
  installed ``tools/`` overlay, its ``agent.env``, its ``mcp/`` drop-ins. Ground truth, on the box.
- ``basecradle-harness-resolve`` answers *what **would** this configuration resolve to?* — from the
  **installed package's** shipped defaults alone, with every input an explicit argument. It needs no
  agent, no config home, no credentials and no network, so a GitHub Action computing a pin can run
  it against a pinned harness version and get the same answer the box will.

Determinism is a hard requirement, so three things it deliberately does **not** do:

- it never reads the process environment for a resolution input (every axis is an argument);
- it never consults a config home (the packaged defaults are the whole candidate set);
- it never applies a **runtime** veto (`Tool.load_refusal` — e.g. ``shell`` refusing to run as
  root), because that is a property of the *box*, not of the configuration. Applying it would make
  the answer depend on the euid of whoever ran the command.

The last one is reported in the output's ``omitted`` list rather than left to the reader, alongside
the other honest gap: MCP proxy tools come from an agent's ``mcp/*.json`` drop-ins, which no stem
set can predict.

Everything else routes through the *same* functions a wake uses — `claim_plugins` /
`resolve_plugins` settle activation, `_install`'s AST classifiers decide powerful-vs-benign and
provider affinity, `_merge_memory_tools` folds in the memory provider's tools — so a second,
parallel model of "what wins" cannot exist here to drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from basecradle_harness._basecradle import (
    _PROVIDERS,
    _SDK_SURFACES,
    DEFAULT_PROVIDER,
    DEFAULT_SDK,
    _merge_memory_tools,
    _surface_for,
)
from basecradle_harness._install import (
    packaged_tool_sources,
    plugin_opts_in,
    plugin_relevant_to,
)
from basecradle_harness._memory_provider import (
    describe_memory_provider,
    memory_provider_named,
)
from basecradle_harness._plugins import (
    ActivationContext,
    EnvSet,
    ResolvedTools,
    ToolPlugin,
    claim_plugins,
    load_default_plugins,
    resolve_plugins,
)
from basecradle_harness._policy import Policy
from basecradle_harness._tools import Tool
from basecradle_harness._version import __version__

#: The two profiles `--profile` selects, mapped to the policy each one is. Unlike the deploy lever
#: (`_basecradle._profile_from_env`, which **fails closed** on an unrecognized value so a typo in an
#: `agent.env` can never unlock a box), an unrecognized value here is a hard error: this is an
#: explicit argument to a read-only question, and silently answering a *different* question than the
#: one asked is how a wrong pin gets computed.
_PROFILES: dict[str, Policy] = {"locked": Policy.locked(), "unlocked": Policy.unlocked()}

#: The placeholder value stood in for an assumed credential. Never a real secret — a `EnvSet`
#: requirement only tests for a non-empty value, so any non-empty string resolves it identically.
_ASSUMED = "<assumed>"

#: The two classes of active tool this command can never derive from a stem set, stated in-band
#: (the output's ``omitted``) rather than left in prose for a caller to remember. Requirement 2 of
#: issue #345 in general form: never silently omit, never silently include — say which assumption
#: was made.
_OMITTED = (
    {
        "source": "mcp",
        "reason": (
            "MCP proxy tools come from an agent's own mcp/*.json drop-ins, which no stem set can "
            "predict. Union them separately (the inventory's mcp_servers axis); "
            "`basecradle-harness-wake --resolved-config` reports the configured servers on the box."
        ),
    },
    {
        "source": "runtime_veto",
        "reason": (
            "A tool's runtime self-veto (Tool.load_refusal — e.g. `shell` refusing to load as "
            "root) is a property of the box, not of this configuration, so it is deliberately not "
            "applied here: the answer must not change with the euid of whoever ran the command. "
            "`--resolved-config` on the box is the authority for that gate."
        ),
    },
)


class UnknownStemError(ValueError):
    """A requested stem names no shipped tool-plugin default — a typo, and always fatal.

    Deliberately **not** the same thing as a stem that is real but inactive here (an xAI tool asked
    for under ``--provider openai``): that is an honest answer this command reports, with a reason.
    A name that exists nowhere in the package is a mistake in the *question*, and answering it
    anyway — by silently contributing nothing — is precisely how a pin ends up short a name and a
    converge fails forever. So it stops, and names the vocabulary.
    """


def _split(names: str | Sequence[str] | None) -> list[str]:
    """Normalize a comma-separated string (or a sequence) of stems into a clean list.

    Accepts the ``--opt-in a,b`` spelling the installer already uses, so an operator's muscle
    memory transfers; a ``.py`` suffix is tolerated for the same reason `_opt_in_scaffold_set`
    tolerates it.
    """
    if names is None:
        return []
    parts = names.split(",") if isinstance(names, str) else list(names)
    return [stem for part in parts if (stem := part.strip().removesuffix(".py"))]


def _plugin_credentials(plugin: ToolPlugin) -> set[str]:
    """The env vars *this* plugin's activation depends on — its resolved names' conditionality.

    Read off the `EnvSet` requirements themselves rather than a list of known credentials, so a
    plugin added later (``send_direct_message_to_origin`` gating on ``NTFY_DM_TOKEN`` was the last
    one) is handled with no change here. `OpenAIKey` is an `EnvSet` subclass, so it is covered by
    the same walk.
    """
    return {req.var for req in plugin.requires if isinstance(req, EnvSet)}


def _credential_vars(plugins: Sequence[ToolPlugin]) -> list[str]:
    """Every env var the given plugins' activation requirements read, sorted and deduped."""
    return sorted(set().union(*(_plugin_credentials(p) for p in plugins)) if plugins else set())


def _apply_policy(resolved: ResolvedTools, policy: Policy) -> tuple[ResolvedTools, dict[str, str]]:
    """Drop the tools `policy` forbids, returning the filtered set and ``{name: reason}``.

    The configuration half of `_basecradle._apply_safe_policy`, and only that half: it applies the
    profile's capability gate (what makes ``shell`` visible under ``unlocked`` and refused under
    ``locked``) while deliberately **not** applying `Tool.load_refusal`, the runtime veto — see this
    module's docstring. The reasons it produces are worded identically to the wake's, so a caller
    can match on them across both surfaces.
    """
    permitted: list[Tool] = []
    refused: dict[str, str] = {}
    for tool in resolved.tools:
        if policy.permits(tool):
            permitted.append(tool)
            continue
        blocked = ", ".join(sorted(tool.requires & policy.forbidden))
        refused[tool.name] = f"refused by the safe-by-default policy: needs {blocked}"
    if not refused:
        return resolved, {}
    filtered = replace(
        resolved,
        tools=permitted,
        skipped=resolved.skipped + sorted(refused.items()),
        manifest=[entry for entry in resolved.manifest if entry[0] not in refused],
    )
    return filtered, refused


def resolve_stems(
    *,
    provider: str = DEFAULT_PROVIDER,
    sdk: str = DEFAULT_SDK,
    surface: str | None = None,
    model: str = "",
    profile: str = "locked",
    opt_in: str | Sequence[str] | None = None,
    only: str | Sequence[str] | None = None,
    memory_provider: str | None = None,
    assume_credentials: bool = True,
) -> dict[str, object]:
    """Resolve a hypothetical configuration to its active tool names and built-ins.

    Args:
        provider: ``AI_PROVIDER`` — ``openai`` | ``xai`` | ``openrouter``. Validated; an unknown
            provider raises, exactly as a wake's `_config_from_env` would.
        sdk: ``AI_SDK`` — the vendor SDK package the harness would import.
        surface: ``AI_SDK_SURFACE``. ``None`` resolves to the SDK adapter's own ``DEFAULT_SURFACE``
            (`_surface_for`), so the common case needs no argument; a value not in that adapter's
            declared set raises.
        model: ``AI_MODEL``, for the rare model-scoped requirement. Reported back, not validated.
        profile: ``locked`` (the shipped default) or ``unlocked``. Governs the capability gate:
            under ``locked`` an opted-in ``shell`` is refused and reported with its reason; under
            ``unlocked`` it resolves.
        opt_in: The powerful stems to grant, comma-separated or a sequence — the same vocabulary
            and spelling ``basecradle-harness-install --opt-in`` takes.
        only: Restrict the candidate set to exactly these stems (benign and powerful alike; a
            powerful stem named here is thereby granted). ``None`` means the full shipped default
            set — which is what a fresh install lays down. Use it to model a **pruned** persona,
            whose operator deleted defaults from its overlay.
        memory_provider: ``HARNESS_MEMORY_PROVIDER`` — ``sqlite`` (default), ``mempalace``, or a
            ``module:Class`` path. Its tools are part of the resolved set (they are why a pin says
            ``memory``) but come from no stem, so the backend is an input here.
        assume_credentials: When ``True`` (the default), every credential an activation requirement
            reads (``AI_API_KEY``, ``NTFY_DM_TOKEN``, …) is **assumed present**, and the output says
            so — ``credentials.assumed`` names them and every conditional resolved name carries an
            ``assumes_credential`` marker. When ``False`` they are all assumed **absent** and the
            tools that need them report as inactive with their reason. It is one or the other and
            it is always stated: the one thing this must never do is silently include or silently
            omit a credential-gated tool. Assumed is the default because the caller is almost
            always computing a pin for a *provisioned* agent, which by definition holds its key.

    Returns:
        A JSON-serializable report — the field contract documented in the README under
        "A stem is not a tool name".

    Raises:
        ValueError: an unknown provider, SDK, profile, or an SDK-mismatched surface.
        UnknownStemError: a requested stem names no shipped tool-plugin default.
    """
    provider = (provider or DEFAULT_PROVIDER).strip().lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; expected one of {_PROVIDERS}.")
    sdk = (sdk or DEFAULT_SDK).strip().lower()
    if sdk not in _SDK_SURFACES:
        # The wake leaves this to the provider build, which fails with "no adapter for that SDK"
        # the moment it tries to reach a model. This path never builds a provider, so nothing
        # downstream would catch it — and a typo'd `--sdk` does not error, it silently answers a
        # *different* question (an `openroute` typo drops `openrouter_search`, and the pin computed
        # from it is short a name). Same failure class as an unknown stem, so: same hard stop.
        raise ValueError(f"Unknown sdk {sdk!r}; expected one of {tuple(_SDK_SURFACES)}.")
    resolved_surface = _surface_for(sdk, surface)
    profile = (profile or "locked").strip().lower()
    if profile not in _PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; expected one of {tuple(_PROFILES)}.")

    sources = packaged_tool_sources()
    opt_in_stems = _split(opt_in)
    only_stems = _split(only) if only is not None else None
    requested = set(opt_in_stems) | set(only_stems or ())
    if unknown := sorted(requested - set(sources)):
        raise UnknownStemError(
            f"No shipped tool plugin is named: {', '.join(unknown)}. "
            f"The shipped stems are: {', '.join(sorted(sources))}. "
            "(Pass the plugin *file stem*, e.g. 'hear_audio', not the tool name 'listen'.)"
        )

    # Per stem, the two source-level classifications the installer makes at scaffold time, applied
    # here to exactly the same files by exactly the same AST readers.
    powerful = {stem: plugin_opts_in(text) for stem, text in sources.items()}
    relevant = {stem: plugin_relevant_to(text, provider) for stem, text in sources.items()}

    # The grant rule, mirroring `_install._opt_in_scaffold_set`: a benign default is laid down for
    # every agent; a powerful one only when named. `only` narrows the benign half on top of that,
    # which is how a deliberately tool-restricted persona is modeled.
    def _granted(stem: str) -> bool:
        if only_stems is not None:
            return stem in requested
        return not powerful[stem] or stem in requested

    candidate_stems = {stem for stem in sources if _granted(stem) and relevant[stem]}

    # Load from the package only — never a config home — and keep just the granted stems. The
    # loader already dropped the provider-mismatched files before importing them.
    loaded = load_default_plugins(provider=provider)
    candidates = [p for p in loaded.plugins if p.stem in candidate_stems]

    credentials = _credential_vars(candidates) if assume_credentials else []
    env: Mapping[str, str] = {var: _ASSUMED for var in credentials}
    ctx = ActivationContext(
        provider=provider, sdk=sdk, surface=resolved_surface, model=model, env=env
    )

    # The same two calls, over the same one ordered pass: `resolve_plugins` for the settled set,
    # `claim_plugins` for the attribution (which plugin won each name).
    resolved = resolve_plugins(candidates, ctx)
    claimed, _ = claim_plugins(candidates, ctx)
    memory = memory_provider_named(memory_provider)
    memory_name, memory_version = describe_memory_provider(memory)
    memory_tools = sorted(tool.name for tool in memory.tools())
    resolved = _merge_memory_tools(resolved, memory)
    resolved, refused = _apply_policy(resolved, _PROFILES[profile])
    memory.close()  # a no-op on the built-ins (the store connection is opened lazily, never here)

    # A shipped default that failed to import contributes no names. The wake degrades the same way
    # (so a pin computed from this still matches that box) — but it must not be *invisible*, or the
    # report reads as an ordinary absence. `_classify_broken` already logged it at ERROR; this puts
    # it in the machine-readable answer too.
    broken = [
        {"stem": name.removesuffix(".py"), "file": name, "error": error}
        for name, error in sorted(loaded.broken_defaults)
    ]
    stems, skipped, excluded = _attribute(
        sources=sources,
        powerful=powerful,
        relevant=relevant,
        granted=_granted,
        candidates=candidates,
        claimed=claimed,
        refused=refused,
        broken=broken,
        ctx=ctx,
    )
    return {
        "harness_version": __version__,
        "ai_provider": provider,
        "ai_sdk": sdk,
        "ai_sdk_surface": resolved_surface,
        "ai_model": model,
        "active_profile": profile,
        "credentials": {
            "mode": "assumed" if assume_credentials else "absent",
            "assumed": credentials,
        },
        "memory": {
            "provider": memory_name,
            "provider_version": memory_version,
            "tools": memory_tools,
        },
        "requested_stems": sorted(requested),
        "tools": sorted(tool.name for tool in resolved.tools),
        "builtins": sorted(resolved.builtins),
        "opt_in_tools": list(resolved.opt_in_stems),
        "stems": stems,
        "skipped": skipped,
        "excluded_stems": excluded,
        "broken": broken,
        "omitted": [dict(entry) for entry in _OMITTED],
    }


@dataclass
class _StemReport:
    """One stem's accumulating verdict, materialized into its JSON entry by `as_json`."""

    opt_in: bool
    granted: bool
    excluded: str | None = None
    broken: str | None = None
    tools: list[str] = field(default_factory=list)
    builtins: list[str] = field(default_factory=list)
    credentials: set[str] = field(default_factory=set)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def as_json(self) -> dict[str, object]:
        """The stable per-stem shape: status + reason + the names it did and did not contribute."""
        active = bool(self.tools or self.builtins)
        if self.excluded is not None:
            status, reason = "excluded", self.excluded
        elif self.broken is not None:
            # A shipped default that will not import contributes nothing — and would otherwise
            # read as an ordinary "inactive", which is a lie: the config did not exclude it, the
            # package is defective. Left unsaid it is a **silently short pin**, the precise
            # failure this command exists to foreclose, so it gets its own status and rides the
            # report's top-level `broken` list too.
            status, reason = "broken", f"shipped default failed to load: {self.broken}"
        elif active:
            status, reason = "active", None
            # A stem can be active *and* have skips (`code_execution` under xAI activates its
            # built-in while `code_attach` self-excludes), so the skips ride the entry either way.
        else:
            reasons = dict.fromkeys(item["reason"] for item in self.skipped)
            status = "inactive"
            reason = "; ".join(reasons) or "no plugin activated"
        return {
            "opt_in": self.opt_in,
            "granted": self.granted,
            "status": status,
            "reason": reason,
            "tools": sorted(self.tools),
            "builtins": sorted(self.builtins),
            "assumes_credential": sorted(self.credentials),
            "skipped": self.skipped,
        }


def _attribute(
    *,
    sources: Mapping[str, str],
    powerful: Mapping[str, bool],
    relevant: Mapping[str, bool],
    granted: Callable[[str], bool],
    candidates: Sequence[ToolPlugin],
    claimed: Mapping[str, ToolPlugin],
    refused: Mapping[str, str],
    broken: Sequence[Mapping[str, str]],
    ctx: ActivationContext,
) -> tuple[dict[str, dict[str, object]], list[dict[str, str]], list[dict[str, str]]]:
    """Attribute every resolved (and every unresolved) name back to the stem that produced it.

    Returns ``(stems, skipped, excluded_stems)``:

    - **stems** — *every* shipped stem, so one call is a complete map and a caller computing the
      delta of adding a stem never needs a second invocation. Each entry carries its status, the
      names it contributed, the reasons it did not, and which credentials its names are
      conditional on.
    - **skipped** — the flat, name-level "why isn't this tool here?" trail, each entry attributed
      to its stem. The counterpart of ``--resolved-config``'s ``skipped``, with the attribution
      that surface has no way to carry.
    - **excluded_stems** — the stem-level exclusions decided *before* activation ran: not granted,
      or not relevant to this provider. A *name* is skipped; a *stem* is excluded — different
      things, so they are different lists.

    `broken` (the shipped defaults that would not import) is threaded in so those stems report as
    ``broken`` rather than as an ordinary ``inactive``, which would misattribute a package defect
    to the configuration.

    The walk is over the candidate plugins themselves and reads its verdicts out of `claimed` (the
    map `claim_plugins` produced), so it can only ever agree with the resolution it describes.
    """
    failed = {entry["stem"]: entry["error"] for entry in broken}
    reports: dict[str, _StemReport] = {}
    excluded: list[dict[str, str]] = []
    for stem in sorted(sources):
        is_granted = granted(stem)
        if not is_granted:
            why = (
                "powerful (opt-in) tool not granted; pass --opt-in to grant it"
                if powerful[stem]
                else "not in --only"
            )
        elif not relevant[stem]:
            why = f"not relevant to the {ctx.provider!r} provider"
        else:
            why = None
        reports[stem] = _StemReport(
            opt_in=powerful[stem],
            granted=is_granted,
            excluded=why,
            broken=failed.get(stem),
        )
        if why is not None:
            excluded.append({"stem": stem, "reason": why})

    skipped: list[dict[str, str]] = []
    for plugin in candidates:
        # Every candidate came from a granted, shipped stem, so its report always exists.
        stem = str(plugin.stem)
        report = reports[stem]
        name = plugin.resolved_name
        if not plugin.active(ctx):
            reason = plugin.unmet(ctx)
        elif claimed.get(name) is not plugin:
            # Two active plugins claimed one name; the later one won (overlay precedence). Rare
            # among the shipped defaults — but reported rather than vanished, because a name that
            # silently belongs to a different stem is the whole failure class this command closes.
            reason = f"overridden by a later plugin claiming {name!r}"
        elif name in refused:
            reason = refused[name]
        else:
            # A built-in contributes its **wire** name, which is not always its model-facing name
            # (`code_execution` is served by the `code_interpreter` built-in on OpenAI). The wire
            # name is what `--resolved-config` reports under `builtins`, so it is what a pin holds.
            if plugin.is_builtin:
                assert plugin.builtin is not None  # guaranteed by ToolPlugin.__post_init__
                report.builtins.append(plugin.builtin)
            else:
                report.tools.append(name)
            report.credentials |= _plugin_credentials(plugin)
            continue
        report.skipped.append({"name": name, "reason": reason})
        skipped.append({"name": name, "stem": stem, "reason": reason})

    skipped.sort(key=lambda item: (item["name"], item["stem"]))
    return {stem: report.as_json() for stem, report in reports.items()}, skipped, excluded


def main(argv: list[str] | None = None) -> int:
    """The ``basecradle-harness-resolve`` entrypoint: print the resolution as JSON, then exit.

    Exit 0 on success; 1 on a bad question (an unknown provider/profile, an SDK-mismatched surface,
    an unknown stem, an unloadable memory provider) with the reason on stderr and **no JSON on
    stdout** — a half-answer an automation might parse is worse than none, and a stem typo
    producing a short pin is the exact failure this command exists to prevent.
    """
    parser = argparse.ArgumentParser(
        prog="basecradle-harness-resolve",
        description=(
            "Resolve a hypothetical agent configuration to the tool names and server-side "
            "built-ins it activates, as JSON. Pure: no config home, no credentials, no network, "
            "no writes — every input is an argument, so the same arguments always give the same "
            "answer. For what a live box is actually running, use "
            "`basecradle-harness-wake --resolved-config`."
        ),
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help=f"AI_PROVIDER — one of {', '.join(_PROVIDERS)} (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--sdk",
        default=DEFAULT_SDK,
        help=f"AI_SDK — the vendor SDK package the harness would import (default: {DEFAULT_SDK}).",
    )
    parser.add_argument(
        "--surface",
        default=None,
        help=(
            "AI_SDK_SURFACE — validated against the SDK adapter's declared surfaces "
            "(default: that adapter's own DEFAULT_SURFACE)."
        ),
    )
    parser.add_argument("--model", default="", help="AI_MODEL, for a model-scoped requirement.")
    parser.add_argument(
        "--profile",
        default="locked",
        choices=sorted(_PROFILES),
        help="HARNESS_PROFILE — the policy profile gating shell-class tools (default: locked).",
    )
    parser.add_argument(
        "--opt-in",
        default="",
        metavar="NAMES",
        help=(
            "comma-separated powerful plugin file stems to grant — the same vocabulary "
            "`basecradle-harness-install --opt-in` takes, e.g. --opt-in xai_search."
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        metavar="NAMES",
        help=(
            "restrict the candidate set to exactly these stems (a powerful stem named here is "
            "thereby granted) — how you model a persona whose overlay was pruned. Omitted: the "
            "full shipped default set, which is what a fresh install lays down."
        ),
    )
    parser.add_argument(
        "--memory-provider",
        default=None,
        metavar="NAME",
        help=(
            "HARNESS_MEMORY_PROVIDER — sqlite (default), mempalace, or a module:Class path. Its "
            "tools are part of the resolved set but come from no stem, so it is an input here."
        ),
    )
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument(
        "--assume-credentials",
        dest="assume_credentials",
        action="store_true",
        default=True,
        help=(
            "assume every credential an activation requirement reads is present (the default); "
            "the report names them and marks each conditional tool with assumes_credential."
        ),
    )
    credentials.add_argument(
        "--no-assume-credentials",
        dest="assume_credentials",
        action="store_false",
        help="assume every such credential is absent; the tools needing one report as inactive.",
    )
    args = parser.parse_args(argv)

    try:
        report = resolve_stems(
            provider=args.provider,
            sdk=args.sdk,
            surface=args.surface,
            model=args.model,
            profile=args.profile,
            opt_in=args.opt_in,
            only=args.only,
            memory_provider=args.memory_provider,
            assume_credentials=args.assume_credentials,
        )
    except (ValueError, TypeError, ImportError) as error:
        # The classes a *bad question* raises — an unknown provider/profile/surface/stem
        # (`ValueError`, which `UnknownStemError` extends), and a `--memory-provider` path that
        # does not import or is not a `MemoryProvider` (`ImportError`/`TypeError`). Anything else
        # is a defect in the package, and it propagates as a traceback rather than being dressed
        # up as the caller's mistake.
        print(f"basecradle-harness-resolve: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
