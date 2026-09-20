---
name: bot-auth-setup
description: Operational setup so this session's git commits, pushes, and GitHub API writes act as the basecradle-harness-ai[bot] identity rather than falling through to the ambient gh login (@origin's). Covers the committed fail-closed fence that makes a forgotten mint an error instead of a post under @origin's name, the local git author and tokenless HTTPS origin, minting a short-lived installation token with the fleet gh-app-token helper in the same shell call as every gh call (a Bash call inherits nothing from the one before it), guarding a failed mint, verifying the author field on the first write, and pushing with the token in the environment (never a URL/argv). Use before every gh call and every push as the bot, when a gh call reports "please run gh auth login", and when a write unexpectedly lands as drawkkwast. The identity table + post-as-your-own-bot invariant live in CLAUDE.md → Fleet Bot Identity.
---

# Bot Auth Setup — basecradle-harness-ai[bot]

The invariant lives in `CLAUDE.md` → "Fleet Bot Identity / Auth Routing": every issue, comment,
PR, and commit is attributable to the bot, never anonymously behind @origin's account. This
skill is the concrete setup so the write actually lands as the bot.

## 0. The session is fail-closed — a forgotten mint is an error, never a post as @origin

**Read this first, because it changes what a failure looks like.** Every rule below is an
instruction, and an instruction only holds while it is followed; issue #550 removed the
laptop's stored `drawkkwast` login from this session's reach entirely, so a forgotten mint
**cannot** fall through to it. The fence is `.claude/settings.json` → `env`, committed and
git-tracked so it survives a fresh checkout, and it is described in `CLAUDE.md` → "Fail-Closed
Session GitHub Auth" (that section, not this one, is the authority on *why* each variable is
what it is; settings.json is JSON and cannot carry a comment).

Two halves, because `gh` and `git` reach @origin's credential by two different routes:

- **`gh` → `GH_CONFIG_DIR=/var/empty`.** `/var/empty` exists on macOS, is root-owned and
  unwritable, so `gh` finds no `hosts.yml`, can never be logged in there by accident, and every
  API call — **reads included** — refuses without a `GH_TOKEN`.
- **`git` → an injected, scoped credential helper** (`GIT_CONFIG_COUNT` / `GIT_CONFIG_KEY_<n>` /
  `GIT_CONFIG_VALUE_<n>`). `GH_CONFIG_DIR` alone does **not** close this half: `gh auth
  git-credential` reads the macOS **keyring** directly and answers a bare `git push` with
  @origin's token even when `gh auth status` reports not logged in (measured, gh 2.100.0,
  2026-09-20). The injected pair resets the github.com helper list and installs one helper that
  can only ever hand out `GH_TOKEN`.

So the failure mode you will actually hit is a loud refusal, in one of these two shapes:

```text
$ gh issue view 550 --repo basecradle/basecradle-harness      # no GH_TOKEN
To get started with GitHub CLI, please run:  gh auth login
Alternatively, populate the GH_TOKEN environment variable with a GitHub API authentication token.
# exit 4

$ git push origin <branch>                                    # no GH_TOKEN
fatal: credential helper '!f() { … }; f' told us to quit
# exit 128
```

**Neither is a broken laptop — both mean "you forgot §2's mint prefix on this call."** Add it
and re-run. Never "fix" either by running `gh auth login`, by unsetting `GH_CONFIG_DIR` or
`GIT_CONFIG_COUNT`, or by reaching around the fence with `-c credential.helper=…`: each of
those hands the session @origin's personal account back, which is the thing #550 removed.

## 1. Git author and origin (local, never committed)

`.git/config` does not travel with the repo, so set both explicitly after a fresh clone:

```bash
git config --local user.name "basecradle-harness-ai[bot]"
git config --local user.email "290979505+basecradle-harness-ai[bot]@users.noreply.github.com"
git remote set-url origin https://github.com/basecradle/basecradle-harness.git
```

No `Co-Authored-By` trailer on bot commits — the commit author already *is* the agent.

**`origin` must be the tokenless HTTPS URL.** An SSH `origin` (`git@github.com:…`) never
consults a credential helper — so it walks straight past §0's fence and authenticates with the
SSH key as `drawkkwast`, with nothing to say it did. The URL carries no credential, ever, and
fetching this public repo needs none.

## 2. Every `gh` call mints its own token, in the same call

**There is no "do it once at the start" step here.** Each Claude Code Bash call starts a fresh
shell and inherits nothing from the call before it, so a token exported in one call is *gone* in
the next. Before #550, `gh` did not fail when it was missing: it fell silently through to the
laptop's stored `drawkkwast` login, and the write landed under @origin's personal account looking
exactly like a success (`basecradle#579`, 2026-09-20: a builder's `picked up — working` comment
posted as the founder). §0 turned that silence into an error — but the mint is still a **prefix
on the call itself**, every comment, `pr create`, `pr merge`, label, close, *and every read*,
never a session-level setup step:

```bash
GH_TOKEN="$(~/Documents/claude-workspace/2026-06-05-fleet-identity/gh-app-token basecradle-harness-ai)" || exit 1; [ -n "$GH_TOKEN" ] || exit 1; export GH_TOKEN
gh issue comment <n> --repo basecradle/basecradle-harness --body "…"
```

Both lines go in **one** Bash call; two calls is the bug above. **Reads mint too** — that is the
accepted cost of the fence, and it is the cheaper half of the trade: a mint you only reach for on
writes is one you can forget you needed.

**Why the guard, and not `export GH_TOKEN="$(…)"`:** `export` is a command in its own right and
returns *its* exit status, never that of the command substitution inside it, so a mint that failed
leaves an empty token behind a zero exit (the `harness#331` class). A **plain** assignment does
propagate the substitution's status (checked in both zsh and bash), so assign first and let
`|| exit 1` see the *helper's* status; check non-empty for a mint that "succeeded" with no output;
only then `export`.

**Check the author field on the first write of a session.** §0 removes the @origin fallback, but
it cannot prove a write landed as the bot — read that back from what GitHub recorded
(`gh issue comment` prints the URL, whose `#issuecomment-<id>` fragment is the id):

```bash
GH_TOKEN="$(~/Documents/claude-workspace/2026-06-05-fleet-identity/gh-app-token basecradle-harness-ai)" || exit 1; [ -n "$GH_TOKEN" ] || exit 1; export GH_TOKEN
gh api repos/basecradle/basecradle-harness/issues/comments/<comment-id> --jq '.user.login'
# → basecradle-harness-ai[bot]
```

Anything else — `drawkkwast` above all — means the fence was reached around. Delete the write,
fix the call, redo it, and say so on the issue.

## 3. `git push` as the bot — the token rides the environment, never argv

**Never put the token in a URL** (`https://x-access-token:${GH_TOKEN}@github.com/…`): the shell
expands it into `git`'s argv, and argv is readable by every account on the machine
(`/proc/<pid>/cmdline`, `ps`) for as long as the push runs (`basecradle-noc#694`,
`basecradle#539`). The helper's `--remote` mode prints exactly that URL — retired on the fleet
box, where it refuses with this recipe, but the laptop copy predates the retirement and still
prints it, so never call it.

**On the laptop** (where this agent runs today), §0's committed helper is already installed for
github.com, so the push is bare — only the mint prefix is needed, in the same Bash call:

```bash
GH_TOKEN="$(~/Documents/claude-workspace/2026-06-05-fleet-identity/gh-app-token basecradle-harness-ai)" || exit 1; [ -n "$GH_TOKEN" ] || exit 1; export GH_TOKEN
git push origin <branch>
```

Three properties of that helper are load-bearing (`basecradle#544`, `harness#550`; each verified
with `git credential fill` and a fake token, git 2.55):

- **Both injected entries are scoped to `https://github.com`.** An unscoped helper answers for
  **every** host, so any non-GitHub URL the same command touches — a submodule, a redirect, a
  mistyped remote — would be handed a live installation token. Scoped, a `gitlab.com` fill never
  reaches it.
- **The scoped empty `credential.https://github.com.helper=` first resets the helper list** for
  github.com. Without it the earlier-collected helpers are asked **first** — Homebrew's system
  `credential.helper=osxkeychain` and @origin's global `!gh auth git-credential`, either of which
  can answer with the `drawkkwast` credential (the silent fallback, one layer down) and, after a
  successful push, would **store** the bot token in the keychain. The reset works because git
  reads system, then global, then local config, and only *then* the `GIT_CONFIG_*` environment
  pairs — so the empty value clears everything already collected (`GIT_TRACE` shows only the
  injected helper is ever invoked for github.com).
- **An unset `GH_TOKEN` emits `quit=1`**, so git stops without sending a credential, naming the
  helper as the reason. An unguarded helper answers with an empty password and exit 0, so git
  sends it and the push fails with a generic auth error that names nothing.

To re-push a rebased branch, add `--force-with-lease` to the same command — bare works, because
a push through `origin` keeps `origin/<branch>` current to lease against. Never `--force`.

(The `http.extraheader="AUTHORIZATION: bearer $TOKEN"` form **fails** — "invalid credentials" —
for App installation tokens, and is argv besides. The older per-command `-c
'credential.https://github.com.helper=…'` recipe is now redundant: `-c` is read after the
environment pairs, so it re-installs the identical helper. Prefer the bare push; reaching for
`-c` to install a *different* helper is reaching around the fence.)

**On a fleet box** (`ai.basecradle.com`), the NOC registers the minter as the agent's github.com
credential helper on every provision and converge, and the documented recipe is
`GH_TOKEN="$(gh-app-token)" git push origin <branch>` (basecradle-noc `deploy/README.md` §7c) —
which §0's helper answers identically. See `CLAUDE.md` → "Fail-Closed Session GitHub Auth" for
the one behavioral difference the fence makes there.

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
