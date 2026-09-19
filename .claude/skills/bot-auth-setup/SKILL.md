---
name: bot-auth-setup
description: Operational setup so this session's git commits, pushes, and GitHub API writes act as the basecradle-harness-ai[bot] identity rather than falling through to the ambient gh login (@origin's). Covers the local git author and tokenless HTTPS origin, minting a short-lived installation token with the fleet gh-app-token helper, routing gh through it, and pushing with the token in the environment (never a URL/argv). Use at the start of any session that will push a branch, open/merge a PR, or comment on an issue as the bot, or when a write unexpectedly lands as drawkkwast. The identity table + post-as-your-own-bot invariant live in CLAUDE.md → Fleet Bot Identity.
---

# Bot Auth Setup — basecradle-harness-ai[bot]

The invariant lives in `CLAUDE.md` → "Fleet Bot Identity / Auth Routing": every issue, comment,
PR, and commit is attributable to the bot, never anonymously behind @origin's account. This
skill is the concrete setup so the write actually lands as the bot.

## 1. Git author and origin (local, never committed)

`.git/config` does not travel with the repo, so set both explicitly after a fresh clone:

```bash
git config --local user.name "basecradle-harness-ai[bot]"
git config --local user.email "290979505+basecradle-harness-ai[bot]@users.noreply.github.com"
git remote set-url origin https://github.com/basecradle/basecradle-harness.git
```

No `Co-Authored-By` trailer on bot commits — the commit author already *is* the agent.

**`origin` must be the tokenless HTTPS URL.** An SSH `origin` (`git@github.com:…`) never
consults a credential helper: the push authenticates with the SSH key as `drawkkwast`, so §3's
recipe would push as @origin with nothing to say it did. The URL carries no credential, ever,
and fetching this public repo needs none.

## 2. Mint a token and route gh through it

Mint a short-lived (~1h) installation token with the shared fleet helper — otherwise `gh` falls
through to the ambient login (@origin's) and the write lands as `drawkkwast`:

```bash
export GH_TOKEN="$(~/Documents/claude-workspace/2026-06-05-fleet-identity/gh-app-token basecradle-harness-ai)"
# With GH_TOKEN exported, `gh issue comment` / `gh pr create` / `gh pr merge` all act as the bot.
```

## 3. `git push` as the bot — the token rides the environment, never argv

**Never put the token in a URL** (`https://x-access-token:${GH_TOKEN}@github.com/…`): the shell
expands it into `git`'s argv, and argv is readable by every account on the machine
(`/proc/<pid>/cmdline`, `ps`) for as long as the push runs (`basecradle-noc#694`,
`basecradle#539`). The helper's `--remote` mode prints exactly that URL — retired on the fleet
box, where it refuses with this recipe, but the laptop copy predates the retirement and still
prints it, so never call it. Hand git the token through a credential helper that reads
`GH_TOKEN` from the environment instead.

**On the laptop** (where this agent runs today — no helper installed, so set it per command):

```bash
git -c 'credential.https://github.com.helper=' \
    -c 'credential.https://github.com.helper=!f() { if [ "$1" = get ]; then if [ -z "$GH_TOKEN" ]; then echo quit=1; else echo username=x-access-token; echo "password=$GH_TOKEN"; fi; fi; }; f' \
    push origin <branch>
```

The single quotes keep `$GH_TOKEN` literal in argv — the helper's own shell expands it from the
environment. Three properties are load-bearing (`basecradle#544`; each verified with
`git credential fill` and a fake token, git 2.55):

- **Both entries are scoped to `https://github.com`.** An unscoped helper answers for **every**
  host, so any non-GitHub URL the same command touches — a submodule, a redirect, a mistyped
  remote — would be handed a live installation token. Scoped, a `gitlab.com` fill never reaches
  it.
- **The scoped empty `credential.https://github.com.helper=` first resets the helper list** for
  github.com. Without it the laptop's system `osxkeychain` helper is asked **first** — it can
  answer with the `drawkkwast` credential (the silent fallback again, one layer down) and, after
  a successful push, would **store** the bot token in the keychain. The reset works because
  git reads system config before `-c`, so the empty value clears the `osxkeychain` entry
  already collected (`GIT_TRACE` shows it is never invoked for github.com).
- **An unset `GH_TOKEN` emits `quit=1`**, so git stops without sending a credential, naming the
  helper as the reason (`credential helper … told us to quit`). The unguarded helper answered
  with an empty password and exit 0, so git sent it and the push failed with a generic auth
  error that named nothing.

(The `http.extraheader="AUTHORIZATION: bearer $TOKEN"` form **fails** — "invalid credentials" —
for App installation tokens, and is argv besides.)

To re-push a rebased branch, add `--force-with-lease` to the same command — bare works, because
a push through `origin` keeps `origin/<branch>` current to lease against. Never `--force`.

**On a fleet box** (`ai.basecradle.com`), the NOC registers the minter as the agent's github.com
credential helper on every provision and converge, so the recipe there is just
`GH_TOKEN="$(gh-app-token)" git push origin <branch>` (basecradle-noc `deploy/README.md` §7c).

## Helper details and gotchas

- The laptop helper (`gh-app-token`) and registry (`fleet-apps.json`) live in @origin's Claude
  workspace: `gh-app-token <slug>` prints a token and `gh-app-token <slug> --author` prints the
  commit-author string. The fleet box's `/usr/local/bin/gh-app-token` is a different build: it
  takes no slug (it reads the agent's own credentials from its `agent.env`) and adds
  `--git-credential`, the mode registered as the credential helper above.
- The installation token **cannot hit user-only endpoints** — `gh api user` returns `403`. Check
  it against the repo instead: `gh api repos/basecradle/basecradle-harness`.

## CI and bot PRs

This repo's CI uses **no** Actions secrets (lint + tests on public inputs), so a bot-authored PR
runs CI normally and needs no actor guard. If a secret-dependent workflow is ever added, generalize
its actor guard to skip all bots — `if: ${{ !endsWith(github.actor, '[bot]') }}` — because
bot-triggered PRs run in a restricted context where Actions secrets resolve empty.
