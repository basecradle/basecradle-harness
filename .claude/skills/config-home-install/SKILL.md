---
name: config-home-install
description: Step-by-step procedure for scaffolding and upgrading a Harness config home with basecradle-harness-install — the installer's per-file conffile-upgrade logic, granting powerful tools with --opt-in, the loud grandfather report, and the per-agent fleet rollout. Use when installing or upgrading a config home, running or debugging basecradle-harness-install, deciding how a shipped default reconciles against an operator's edits, or granting/pruning a persona's powerful tools. The config-home layout, resolution order, and the two security invariants (capability opt-in fails-closed, MCP safe-by-default) live in CLAUDE.md → Config Home; this skill carries the procedure.
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

## Granting a powerful tool (`--opt-in`)

Powerful tools (`generate_image`, `edit_image`, `hear_audio`, OpenAI `web_search`, xAI
`web_search`/`x_search`, `grok_generate_image`, `grok_generate_video`) fail closed and are
**not scaffolded** by a plain install. To grant one:

```bash
basecradle-harness-install --config-home <dir> --opt-in <stems>   # e.g. --opt-in generate_image edit_image
```

This scaffolds the named powerful defaults into the persona's `tools/` overlay (equivalently,
drop the file in by hand). An opt-in plugin *present* in the overlay activates, gated only by
its `requires` (an OpenAI key, the xAI vendor, etc.). Deciding a persona's target tool-set is
the **capital's** governance call; applying it on a box is the **NOC's** deploy.

## Grandfather, loudly (on upgrade)

On upgrade, a powerful tool a *prior* version had already scaffolded into an existing config
home is **kept, never silently stripped** (the founder's "tools stay the same" migration rule)
and **reported loudly** — `InstallReport.grandfathered` surfaces in the CLI summary plus a
`WARNING` log line. New installs get the opt-in (off) default. The loud report is what lets the
capital confirm what to prune when cutting a persona's overlay to spec.

## No-import discipline

Both the packaged-default fallback and the installer detect a plugin's `opt_in` flag from
**source via AST** (`_install.plugin_opts_in`) — never by importing the plugin. Same discipline
as provider-affinity detection: the installer inspects, it does not execute plugin code.
