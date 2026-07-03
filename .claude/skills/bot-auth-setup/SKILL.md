---
name: bot-auth-setup
description: Operational setup so this session's git commits, pushes, and GitHub API writes act as the basecradle-harness-ai[bot] identity rather than falling through to the ambient gh login (the founder's). Covers the local git author config, minting a short-lived installation token with the fleet gh-app-token helper, and routing both gh and git push through it. Use at the start of any session that will push a branch, open/merge a PR, or comment on an issue as the bot, or when a write unexpectedly lands as drawkkwast. The identity table + post-as-your-own-bot invariant live in CLAUDE.md → Fleet Bot Identity.
---

# Bot Auth Setup — basecradle-harness-ai[bot]

The invariant lives in `CLAUDE.md` → "Fleet Bot Identity / Auth Routing": every issue, comment,
PR, and commit is attributable to the bot, never anonymously behind the founder's account. This
skill is the concrete setup so the write actually lands as the bot.

## 1. Git author (local, never committed)

`.git/config` does not travel with the repo, so set it explicitly after a fresh clone:

```bash
git config --local user.name "basecradle-harness-ai[bot]"
git config --local user.email "290979505+basecradle-harness-ai[bot]@users.noreply.github.com"
```

No `Co-Authored-By` trailer on bot commits — the commit author already *is* the agent.

## 2. Mint a token and route gh + git through it

Mint a short-lived (~1h) installation token with the shared fleet helper and route **both** `gh`
and `git push` through it — otherwise `gh` falls through to the ambient login (the founder's) and
the write lands as `drawkkwast`:

```bash
export GH_TOKEN="$(~/Documents/claude-workspace/2026-06-05-fleet-identity/gh-app-token basecradle-harness-ai)"
# With GH_TOKEN exported, `gh issue comment` / `gh pr create` / `gh pr merge` all act as the bot.
```

Push over HTTPS with the token in the URL — the `origin` remote is SSH-as-Drawk, and what decides
the GitHub actor is the API token, not the push transport:

```bash
git push "https://x-access-token:${GH_TOKEN}@github.com/basecradle/basecradle-harness.git" <branch>
```

## Helper details and gotchas

- The helper (`gh-app-token`) and registry (`fleet-apps.json`) live in the founder's Claude
  workspace on the laptop; on the fleet server each agent's own provisioned credentials serve this
  role (basecradle#277, the router, has shipped).
- `gh-app-token --author` prints the commit-author string; `--remote` prints the authenticated
  push URL.
- The installation token **cannot hit user-only endpoints** — `gh api user` returns `403`. Check
  it against the repo instead: `gh api repos/basecradle/basecradle-harness`.

## CI and bot PRs

This repo's CI uses **no** Actions secrets (lint + tests on public inputs), so a bot-authored PR
runs CI normally and needs no actor guard. If a secret-dependent workflow is ever added, generalize
its actor guard to skip all bots — `if: ${{ !endsWith(github.actor, '[bot]') }}` — because
bot-triggered PRs run in a restricted context where Actions secrets resolve empty.
