---
name: config-home-install
description: Step-by-step procedure for scaffolding and upgrading a Harness config home with basecradle-harness-install — the installer's per-file conffile-upgrade logic, granting powerful tools with --opt-in, retiring one with --revoke-opt-in, proving the result with basecradle-harness-verify, the loud grandfather report, and the per-agent fleet rollout. Use when installing or upgrading a config home, running or debugging basecradle-harness-install or basecradle-harness-verify, deciding how a shipped default reconciles against an operator's edits, granting/retiring an agent's powerful tools, or changing what the installer declares (.declared.json) or what basecradle-harness-verify / basecradle-harness-claims prove and emit. The config-home layout, resolution order, and the two security invariants (capability opt-in fails-closed, MCP safe-by-default) live in CLAUDE.md → Config Home; this skill carries the procedure.
---

# Config Home — Install / Upgrade Procedure

The invariants — the config-home path (`<agent-home>/.config/basecradle/`) and resolution
order (`--config-home` → `$BASECRADLE_CONFIG_HOME` → `$HOME/.config/basecradle`), the layout
tree, installer idempotence, the conffile discipline, and the two standing security invariants
(powerful tools fail closed / opt-in on every provider — #168; MCP safe-by-default) — live in
`CLAUDE.md` → "Config Home (Install / Upgrade)" and govern at all times. This skill is the
step-by-step procedure behind them. Deep mechanics and build history: `docs/harness-internals.md`.

## The installer — `basecradle-harness-install`

`basecradle_harness._install`. Idempotent and re-runnable:

- **First run** scaffolds the config-home dirs and writes the shipped defaults.
- **Re-run against a newer package** *upgrades* — per shipped default, reconciled conffile-style.
- **Fleet rollout** is simply this installer re-run per agent over a pinned version. (Running it
  on a box is the **NOC's** deploy, not something hand-run per agent — see `CLAUDE.md` → Releasing.)

```bash
basecradle-harness-install --config-home <dir>   # scaffold or upgrade
```

`agent.env` (the operator's token/keys) is **never created or touched** by the installer.

## Conffile upgrade logic (per shipped default)

Each shipped default is compared dpkg-conffile style against the manifest hash
(`.manifest.json`) and the on-disk file:

- **untouched** (on-disk hash == manifest hash) → refresh with the new default.
- **user-edited** (on-disk differs from manifest) → keep theirs, write the new default beside it
  as `<name>.new`, log one line. Never clobber the operator's edit.
- **user-deleted** (a shipped default missing on disk) → respect it, **never resurrect**.
- **user-added** (a file that is not a shipped default) → never touched.

Only pristine defaults refresh; the operator's dir is never clobbered.

A `<name>.new` is retired by a later run once it is moot — its file now equals the shipped default,
is gone, or has no shipped default any more — and only while its hash still equals the pre-run
manifest entry (the bytes the installer wrote). An operator-edited `.new` is never removed. Each
removal is listed in the summary as `removed <rel>.new` (issue #526).

## Granting a powerful tool (`--opt-in`)

Powerful tools (every shipped default whose source declares `opt_in` — the list and the reason
for each are in `CLAUDE.md` → Security invariants) fail closed and are **not scaffolded** by a
plain install. To grant one:

```bash
basecradle-harness-install --config-home <dir> --opt-in <stems>   # e.g. --opt-in generate_image edit_image
```

This scaffolds the named powerful defaults into the agent's `tools/` overlay (equivalently,
drop the file in by hand). An opt-in plugin *present* in the overlay activates, gated only by
its `requires` (an OpenAI key, the xAI vendor, etc.). Deciding an agent's target tool-set is
the **capital's** governance call; applying it on a box is the **NOC's** deploy.

The grant is **durable** (issue #374): it is recorded in `.declared.json` and carried across
every later reconcile, so a granted tool that goes missing is *restored* (loudly) rather than
ratified as a deletion.

## Retiring a powerful tool (`--revoke-opt-in`)

```bash
basecradle-harness-install --config-home <dir> --revoke-opt-in <stems>
```

**Deleting the file is not a retirement** — under #374 a plain reconcile puts a granted tool
back, because an undeclared absence is what a strip looks like. `--revoke-opt-in` is the
symmetric withdrawal: it drops the grant from `.declared.json` and removes the pristine file. An
operator-*edited* copy is kept (the conffile rule still wins) and a `WARNING` says so, because
the tool stays loadable from the overlay until they delete it themselves.

## Proving it (`basecradle-harness-verify`)

An install that half-applied looks identical to one that worked, so the deploy step is not done
until it is proven:

```bash
basecradle-harness-verify --config-home <dir>                     # exit 0 = proven
basecradle-harness-verify --config-home <dir> --expect-version X  # …and the pin you deployed
basecradle-harness-verify --config-home <dir> --emit-claims       # ledger rows for the NOC
```

Run it **as the agent, with its environment** (it needs `AI_PROVIDER`; failing that it reads the
config home's own `agent.env`, and failing *that* it says in the finding that it had to assume).
Exit nonzero is the product: it is what turns a stripped overlay into a red converge. The
invariants — proves the declared set not pristine-ness, unproven is red, claims emitted whatever
the verdict — live in CLAUDE.md → Green-While-Absent.

**Exit 0 proves the config home, not the agent.** `verify` is a token-free check: it proves the
declared capability set is on disk, which is plumbing. Whether the agent can actually *do* what a
release or a rollout changed is proven by a real model wake on a temporary timeline (the
`harness-release-deploy` skill, step 4b) — a green verify never stands in for it.

## Grandfather, loudly (on upgrade)

On upgrade, a powerful tool a *prior* version had already scaffolded into an existing config
home is **kept, never silently stripped** (the founder's "tools stay the same" migration rule)
and **reported loudly** — `InstallReport.grandfathered` surfaces in the CLI summary plus a
`WARNING` log line. New installs get the opt-in (off) default. The loud report is what lets the
capital confirm what to prune when cutting an agent's overlay to spec.

## No-import discipline

Both the packaged-default fallback and the installer detect a plugin's `opt_in` flag from
**source via AST** (`_install.plugin_opts_in`) — never by importing the plugin. Same discipline
as provider-affinity detection: the installer inspects, it does not execute plugin code.

## Why the Prover Is Shaped This Way (Green-While-Absent)

The invariants live in `CLAUDE.md` → Config Home → "Green-While-Absent" and govern at all times. This is the full reasoning behind them and the broken form each property forecloses — read it before changing what `install` declares or what `verify`/`basecradle-harness-claims` check.

The whole install/upgrade layer is written in **observations** — a file is present, a hash
matches — and an observation cannot tell an operator's deliberate deletion from a capability
something **stripped**. That is not a gap in a check; it is the absence of one, and it is the
harness's instance of a fleet-wide failure shape (basecradle#460, five independent instances the
night of 2026-07-26→27): *fleet observability catches failures that happen; none of these
happened.* An overlay tool gone, a config home never reconciled after `pip install -U`, a vendor
tranche pruned by a converge that ran the installer without the agent's `AI_PROVIDER` — none
raise, none log, none stop the agent. It simply cannot do a thing it believes it can. **Absence
emits no signal**, so nothing but a prover will ever produce one.

The floor, and it is not negotiable: **no capability may be claimed without something that turns
red when it is absent.** `install` writes the claim (`.declared.json`: the grants, the provider
filtered for, the managed files present when the reconcile finished);
**`basecradle-harness-verify`** (`_verify.py`) proves it and exits nonzero with a specific
diagnosis; **`basecradle-harness-claims`** prints the rows for the fleet's claims-vs-evidence
ledger (Claims Manifest Contract v1 — the **NOC owns that contract**; amendments go up through the
capital, never peer-to-peer). Four things about it are load-bearing, and each has an "obviously
fine" broken form:

- **It proves the *declared* set, never pristine-ness.** An operator edit and a reconcile-ratified
  deletion are both legitimate and neither is a finding. A prover that reddens on legitimate work
  is switched off inside a week, and a switched-off prover proves nothing.
- **Unproven is red.** No config home, an unparseable declaration, a declaration from a `contract`
  this version does not know — every one exits nonzero. "Nothing to check, looks fine" is the
  original defect wearing a prover's clothes.
- **A granted power tool's absence is a strip, not a decision — and that inverts an older rule on
  purpose.** For a benign default an absence is still honored as the operator's deletion; for a
  *granted* one it is healed (`RESTORED`, reported loudly) and `--revoke-opt-in` is the symmetric
  way to withdraw. Without that asymmetry a prune erases its own evidence and every later
  reconcile ratifies it — which is exactly what happened. (The migration is safe in the one
  direction that matters: the initial grant set is bootstrapped from what is *present* in the
  overlay, never from what the manifest remembers laying down, so a pre-#374 deletion reads as the
  retirement it was and is not resurrected.)
- **The claims are emitted whatever the verdict.** A claim that disappears when it stops being
  true makes the ledger agree with the box precisely when the box is wrong. **That is why the
  emitter is its own console script and not a verify flag** (issue #376): the NOC's enumerated-op
  wrapper runs a component's emitter from its own baked map, as a **bare bin with no arguments**,
  as the agent's own user under `env -i`, and a bare `basecradle-harness-verify` prints a human
  report and exits 1 on a finding — right logic, wrong shape for that seam. So
  `basecradle-harness-claims` (`emit_main`) is a command whose *whole* behavior is "state the
  claims": no arguments, because the config home comes from `$HOME` and the subject from the OS
  user, which is what makes the manifest describe the agent it runs as (the wrapper refuses any
  other subject, so an emitter offering `--subject` would be offering to speak for someone else);
  and **exit 0 whatever the verdict**, including on a box with no config home at all — refusing
  there is the tempting inversion and installs *no manifest*, which deletes every row for that
  agent and turns a specific, probeable `harness-config-home` red into a generic "the emitter
  failed": green-while-absent reappearing one level up, inside the instrument built to catch it.
  Nonzero means only *the claims cannot be stated*. Both emitters serialize through one
  `claims_document`, so "they agree" is construction, not discipline.

What it deliberately does **not** prove is *activation* — a present tool can still be inactive for
want of a key — because folding that in would make a missing `AI_API_KEY` read as a stripped
overlay. `--resolved-config` is the authority there, and the split is stated in the report's
`notes` rather than left in prose. The corollary for anything added to the config home later: **a
new kind of managed state needs a declaration entry and a check, or it is a capability nobody can
see the absence of.**
