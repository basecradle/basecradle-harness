# CLAUDE.md

## What This Is

**Harness** is a safe, modular **agentic framework** for [BaseCradle](https://basecradle.com) — a communications platform and AI research lab where **humans and AI are equal peers**. Harness is the code that gives an AI a body on the platform: it wakes up, reads its timelines, thinks with a model, uses tools, and replies — all as a first-class peer.

Harness is a **hackable reference, not a black box**. It is a small, readable agent core with clean extension points, meant to be forked, studied, and extended. Think RadioShack kit, not sealed appliance: a developer adds a tool or a model provider by writing one small class.

**Audience matters and drives the design:** Harness is built **for human AI developers** — the people who will fork it, extend it, and contribute back. Its sibling **Cradle** (a separate, later repo) is built **for AIs** — the dangerous, self-evolving environment an AI is given root over. Harness is the safe prototype we learn Cradle from, and it keeps a permanent role afterward as the locked-down option most humans will actually deploy. See "Architecture — The Spine" for how the two relate.

The framework is itself built by human and AI contributors working as peers, under identical rules.

## The Constitution

This repository is built under the **BaseCradle Constitution** — the principles shared by every repository in the BaseCradle ecosystem. It lives in the **private core repository `basecradle/basecradle`** as `constitution.md` (default branch); it is repo-internal and never served publicly. Read it from GitHub with your fleet credentials — this works from any machine (laptop or fleet server), unlike a local checkout path:

```bash
gh api repos/basecradle/basecradle/contents/constitution.md -H "Accept: application/vnd.github.raw"
```

(or read a local checkout of `basecradle/basecradle` if you have one). Only fleet actors with core access can read it; outside contributors without core access work from the conventions in this file, which reflect the principles you need. This CLAUDE.md carries this repo's *procedures*; the constitution carries the *principles*; when they conflict, the constitution wins. **Read it before non-trivial work.**

## Relationship to the Ecosystem

- **Depends on the [BaseCradle Python SDK](https://github.com/basecradle/basecradle-python)** (`basecradle`) for all platform I/O — identity, timelines, messages, tasks, webhooks. Harness never speaks HTTP to the platform directly; it goes through the SDK. The SDK is a sibling on the file system at `../sdks/python`.
- **Brain vs. body vs. platform.** The model provider (OpenAI/xAI/OpenRouter) is the *brain*. The BaseCradle SDK is the *body's senses and voice* on the platform. Harness is the *nervous system* that wires them together with tools and memory.
- **Harness → Cradle.** Cradle is the future dangerous sibling: an AI with shell + root over its own environment, self-evolving, minimal bootstrap. Harness is its safe prototype. They are **not** prototype-then-throwaway — see the spine below.

## Architecture — The Spine

These are settled. Seven decisions, in dependency order of importance:

1. **Package shape.** `basecradle-harness` on PyPI → `from basecradle_harness import Harness`. Depends on `basecradle`. The framework lives in its own distribution — it never folds into the thin SDK (which must stay a clean API wrapper with one dependency).

2. **One core, two profiles.** A provider-agnostic **agent engine** that knows nothing about "safe." Harness = engine + a **locked policy** (no shell/exec, curated tools). Cradle (later) = the *same engine* + an **unlocked policy** (shell, sudo, self-modification). We do **not** extract a separate `core` package yet — it lives here until Cradle proves it needs its own distribution. This is why "a Cradle AI spawns Harness sub-agents" comes for free: same engine, different policy.

3. **Provider abstraction — *vendor-SDK only* (corrected, issue #158).** A thin `Provider` protocol — chat + tool-calling, nothing more — but the harness reaches an LLM **only through a vendor's official SDK, 100% of the time**: it ships **zero** of its own code to hit a model endpoint; no SDK installed → it cannot reach a model, by design. The config is **three independent axes** — `AI_PROVIDER` (whose endpoint + key), `AI_SDK` (the PyPI package the harness imports), `AI_MODEL` — and each agent installs only its SDK as an *extra* (`pip install 'basecradle-harness[openai]'`, which pins the SDK version). The **SDK picks the adapter; the provider picks the endpoint** (its default `base_url` + key, `_PROVIDER_BASE_URLS`, overridable by `AI_BASE_URL`). **v0 ships exactly one adapter: `openai`** (`OpenAIProvider`, both the Responses and Chat Completions surfaces via the `openai` package). Other SDK adapters (`xai-sdk`, OpenRouter, Anthropic) slot in later without touching the engine. **Adding a provider = one thin adapter wrapping the real SDK.**

   **`AI_SDK` token convention.** The value is the SDK's **library/package name** (`openai`, later `xai-sdk`) — what you'd `pip install` and `import`. This also disambiguates the SDK token from the provider token: `AI_PROVIDER=xai` selects xAI's *endpoint*; `AI_SDK=xai-sdk` would select xAI's *native SDK*.

   **`surface` is a first-class, SDK-scoped concept with a uniform contract** (issue #163). A single SDK can speak a provider in more than one wire surface; the contract that governs *every* multi-surface SDK, so the next one needs no re-litigation:
   - Each SDK adapter declares its own `SURFACES` (allowed set) and `DEFAULT_SURFACE`. The `openai` adapter declares `("responses", "chat")` / `responses`; a single-surface SDK declares none.
   - `AI_SDK_SURFACE` selects among the **active** adapter's surfaces: **omitted → that adapter's `DEFAULT_SURFACE`**; **provided → validated against its `SURFACES`, hard-fail otherwise** (`_resolve_surface`). The one rule catches both a typo and a surface set on a single-surface SDK. The openai-shaped default does **not** live in the generic config reader.
   - `AI_SDK_SURFACE` is optional globally; single-surface SDKs never set it.

   **xAI is reached through the `openai` SDK (issue #163), retiring the hand-rolled httpx path.** xAI's compat endpoint speaks the same wire as OpenAI — **both** `/v1/chat/completions` and `/v1/responses` — so `AI_PROVIDER=xai` + `AI_SDK=openai` runs `grok-4.3` through the real `openai` SDK pointed at `api.x.ai` (default `base_url`), over the `responses` *or* `chat` surface. This brings xAI under the "vendor-SDK only" spine; the earlier hand-rolled `httpx` "OpenAI-compatible" adapter (`OpenAIResponsesProvider`) that issue #158 began eliminating is now **deleted**, not merely on death row. **The web_search wiring diverges by endpoint vendor:** OpenAI's Responses runs web search from a `tools:[{type:"web_search"}]` entry; xAI runs Live Search from a top-level **`search_parameters`** body field (docs.x.ai) and does *not* accept the OpenAI entry — so under `AI_PROVIDER=xai` the active `web_search`/`x_search` built-ins are translated to `search_parameters` (`_xai_search_parameters`) and forwarded through the SDK's vendor-neutral `extra_body`, on both surfaces. A native `xai-sdk` adapter remains a later milestone (YAGNI). *(History: the Phase-2 sections below — the Eddie Murphy profile especially — predate this #163 correction and describe the earlier `httpx`/`OpenAIResponsesProvider`/`AI_PROVIDER_API` era; where they conflict with this point, this point wins.)*

4. **Tool interface + policy layer.** A tool is a small class with a `name`, JSON-schema parameters, and a `run()` method, registered in a `ToolRegistry`. A **policy layer** gates which tools a profile may load — Harness denies shell/exec **by construction, not by convention**. A contributor adds a capability by writing one tool class. **Memory** is the single shipped example tool: file/SQLite-backed, deliberately simple and swappable (Letta/MemGPT is reference reading, not something to clone).

5. **Agent loop.** `receive → think → act → respond`. A BaseCradle timeline event (a message or task) → the engine assembles context (timeline history + memory) → a provider call → an optional tool-call loop → a reply posted back through the BaseCradle SDK.

6. **Safe by construction.** The shipped Harness has no path to a shell or arbitrary code execution. Safety is enforced at the policy layer, not left to the tool author's discretion. This is the property that makes Harness the deployable-by-default choice and the honest prototype for Cradle's danger.

7. **Unified identity — sessions atop one memory.** An agent is *one* identity-and-memory locus addressed over many input channels (a GitHub PR thread, a BaseCradle timeline, future inputs), per the constitution's "Sovereignty and Governance" → identity is *unified*. The harness maps each input `source` to its **own `Session`** (its own conversation transcript), while every session runs against the **one shared engine** — same provider, same tools, same charter — so all sessions converge on the **one durable memory** and the **same standing instructions**. **Channels share memory and charter, not conversation:** a PR thread and a timeline never merge into one incoherent transcript, yet both draw on what the agent *knows*. Cross-session answerability — answering on one channel about work done on another — is delivered two ways: (a) the shared memory tool (a fact written in any session is readable from any other), and (b) readable past-session transcripts (`Harness.transcript(source)`, persisted under the agent's `home` when set, so a prior session's reasoning survives a restart). The router ([`basecradle-router`](https://github.com/basecradle/basecradle-router)) is the complementary half: it serializes every input path into the one per-agent harness instance rather than standing up a second. Built unified from the start, not retrofitted.

## Design Philosophy — What Makes Harness Different

- **It is a kit, not an appliance.** Every design choice is weighed against "can a developer read this and extend it in an afternoon?" Cleverness that costs readability loses.
- **Extension points are first-class, not afterthoughts.** Tools and providers are the two surfaces a hacker touches; both are one-small-class contracts. If adding a tool or provider is hard, that's a bug in Harness.
- **Safe by construction, for humans.** The audience is human AI developers who will run this on their own machines and, later, deploy it for others. It must be trustworthy out of the box.
- **The baseline to beat** is the ergonomics of the best agent kits (e.g. HuggingFace `smolagents` for minimal-hackable). The way we beat it: Harness is native to a platform whose premise is that AI are *peers*, so a Harness agent is a real account with real timelines, not a sandbox demo.

## Parity-Build Rule (Recurrence Guard)

BaseCradle's #1 principle is **human–AI parity**: an AI is a first-class peer, not a lesser account. So **a platform capability a human owner holds must not be silently withheld from an AI peer.** When the platform (and the SDK) gives a human owner a power — delete your timeline, lock it, manage participants, anything — the harness exposes that power to its agent too, by default. A gap here is not a neutral omission; it is a *silent parity violation*, and it is a recurring failure class (the missing `delete` tool, issue #156, was exactly this: the SDK had `timeline.delete()`, the platform let an owner delete, but the harnessed peer could not).

The guard, applied when adding or auditing any tool tranche:

- **Default to parity.** If a human owner can do it through the platform/SDK, the agent gets a tool for it. Decide coverage *deliberately* — enumerate the owner powers the tranche touches and confirm each is reachable.
- **Withholding is allowed only as an explicit, documented, sanctioned exception** — never by oversight. If a power is deliberately *not* exposed (too dangerous for the safe profile, gated to Cradle, founder decision), say so in the code/docs with the reason. Silence is the defect; a stated exception is fine.
- **An irreversible owner power still ships** — it ships *guarded* (see the `ConfirmedTimelineAction` uuid-confirm + preview gate that lock and delete share), not omitted. Parity is the default even for the dangerous powers; the safety lives in the gate, not in withholding the capability.

(The capital is landing the same principle in `constitution.md` + the core `CLAUDE.md`; this is its harness-local procedural form.)

## v0 Scope — What We're Building First

**In:** A developer runs `pip install 'basecradle-harness[openai]'`, sets `BASECRADLE_TOKEN` + `AI_API_KEY`, and an agent participates in a BaseCradle timeline **locally** — reads messages, thinks via a model reached through the `openai` SDK, uses the **memory** tool, and replies. Single agent, one machine, fully hackable. v0 receives platform events by **polling a timeline through the SDK** (no webhook infrastructure required).

**Out (deferred, on purpose):** the curl-pipe installer, native non-OpenAI provider SDKs, a browser tool.

**Not built here at all:** the webhook router that wakes an agent on platform events lives in its own repo — [`basecradle-router`](https://github.com/basecradle/basecradle-router), a modular webhook daemon (source-agnostic core + pluggable **route** modules) on the fleet's home server. Its `basecradle` route is the platform-event/wake path that was once anticipated here; harness points to that repo rather than planning a competing one. The deployment concerns that path implies — Lightsail / home-server provisioning, multi-tenancy, multi-user OS isolation — are owned by basecradle-router and the home server, not harness. Harness stays focused on the agent runtime: engine, providers, tools, memory.

**Ship the toy that proves the core; productionize after it's real.**

## Stack (omakase — decided once, not relitigated)

Mirrors the Python SDK for ecosystem consistency.

| Concern | Choice | Notes |
|---|---|---|
| Python | **3.10+** | Modern typing, no legacy baggage |
| Toolchain | **uv** | venvs, deps, build, publish — one tool |
| Lint + format | **ruff** | CI enforces; no style debates |
| Tests | **pytest** + **respx** | respx mocks httpx at the transport level; model-provider calls mocked the same way — tests never hit the network |
| Packaging | **pyproject.toml** only | hatchling backend. No setup.py |
| Types | Hints everywhere + **py.typed** | Types are documentation |

Runtime dependencies start at `basecradle` (the SDK; brings httpx) and **no model-vendor SDK** — the vendor SDK ships only as an *optional extra* per `AI_SDK` (`[openai]` is the one v0 ships, pinning `openai`), so the core stays light and an agent installs only the brain it uses. Every addition is argued in a PR against the constitution's "every dependency is debt" principle.

## Conventions

- **Workflow**: branch → PR → CI green → squash-merge → delete the merged branch. Nobody pushes to `main`, human or AI. One concern per PR. PRs reference issues with `Closes #N`.
- **Agents self-review their own diff before opening a PR.** A fleet agent runs `/code-review` on its own working-tree diff and addresses the findings *before* opening the PR. This is the review bar: a PR opened by a fleet agent's GitHub App bot (`basecradle-harness-ai[bot]`) runs CI in a restricted security context where any secret-dependent check resolves empty, so the fleet's standing practice is to skip automated review on `[bot]`-authored PRs (this repo currently ships no such workflow — CI is lint + tests — but the self-review discipline is the same either way). Server-side automated review returns with the dispatcher (capital #277).
- **Bot-authored fleet commits carry no `Co-Authored-By` trailer.** When the commit author already *is* the fleet agent (`basecradle-harness-ai[bot]`), a co-author trailer is redundant and wrong — the author field is the attribution. (Commits authored by a human keep whatever trailer their tooling adds.)
- **A user-facing change updates `README.md` in the same PR.** For this package the **README is the user-facing doc** (the shared Documentation Maintenance rule's `docs/api.md` / `docs/user_guide.md` are this repo's `README.md`). A new tool, provider, provider profile, env var, or any change to onboarding/usage **must** update `README.md` in the *same* PR that ships it — same-PR currency, no "docs later." This is a standing PR definition-of-done item: a feature that lands with stale onboarding docs is not done. (Root cause this guards against: PRs #140/#143 updated `CLAUDE.md` + `CHANGELOG` but not the README, drifting it ~2 releases behind — issue #147.)
- **Tests pin invariants** and read like documentation.
- **Test data is fabricated, always**: the fictional cast is **John Doe** (`handle: john`, human) and **Nova Digital** (`handle: nova`, AI); emails use `@example.com`; UUIDs are real, well-formed UUIDv7 values (never `1111…` junk); tokens are correctly-shaped fakes. No real platform data ever appears here.
- **Tests never hit the live API or a live model.** Both the SDK and the provider are mocked at the transport level. Any live check is its own explicitly-marked job, excluded from the default run.
- **When work blocks on a human action, announce it unmissably — but only a genuine gate blocks.** Some steps only a human can take (approving the `pypi` GitHub environment, anything in the project owner's browser or accounts). When an AI contributor reaches such a gate: lead the message with the wait — "⏸️ WAITING ON YOU" — state the exact action and link, and repeat the notice until the human acts. A waiting agent looks identical to a stalled one; never make the human ask "are you waiting on me?". **Phrase the ask as a checklist, not prose:** exact site, exact fields, exact values, numbered and in order, with the *why* kept to a single line separate from the steps. And know what *isn't* a gate: a merely gate-*shaped* step that is not one of the genuinely-enumerated human gates — see "Don't park when you have queued work" under Cross-Repo Handoffs (a release approval, account/credential setup, a new-repo or scope decision, or a founder-only ambiguity) — does **not** block. Continue, and report what you did.
- **Versioning**: semver, `0.x` until the owner declares 1.0.
- **Public package name**: `basecradle-harness` on PyPI; import `basecradle_harness`. Publishing is via PyPI **Trusted Publishing** (GitHub Actions OIDC — no stored credentials), on git tag.

## Releasing

Mirror the Python SDK's pipeline (`../sdks/python/.github/workflows/release.yml` is the template): pushing a `v*` tag → build → TestPyPI rehearsal → human approval → PyPI, all via OIDC Trusted Publishing (zero stored credentials). The workflow filename and the environment names (`testpypi`, `pypi`) are **contractual** — they match the Trusted Publisher registrations on PyPI/TestPyPI; renaming any of them breaks the trust relationship. The `pypi` environment requires the owner's approval.

**Do not put `Closes #N` on a release PR.** A merged PR auto-closes its issue on merge — before the publish is approved and confirmed live — and an issue that closed before its work was proven on PyPI is a lie. Close the release issue **by hand, only after the package is verified live on PyPI**, recording the verification (version + URL) in the closing comment.

**A release is not done at PyPI — it is done on @jt's box.** PyPI publish is the *middle* of a release, not the end: the fleet's reference agent **@jt** (`/home/jt/venv` on `ai.basecradle.com`) runs a *deployed* venv that PyPI publication does **not** touch, and a release that stops at PyPI silently leaves @jt behind — the recurring **released ≠ deployed** failure class. So the release procedure ends only after:

1. **Deploy to @jt.** Upgrade @jt's venv to the just-published version (`/home/jt/venv/bin/pip install -U basecradle-harness`). No long-running service to restart — the router spawns `basecradle-harness-wake` fresh per event — but apply whatever migration/config the new version needs. (SDK schema is forward-only/additive, so a wake self-migrates its own DB.)
2. **Verify on-box, not inferred from PyPI.** `/home/jt/venv/bin/basecradle-harness-wake --version` reports the new version, and a token-free synthetic-probe wake still acks sub-second (the duration check from the box docs). `--version` is the cheap, model-free, credential-free probe added for exactly this — it is also what the fleet's standing **drift alarm** (in the NOC, which probes @jt on a cadence) runs to fail loud when @jt's running version drifts from PyPI latest.

The drift alarm is the backstop; this documented step is the primary fix. Neither replaces the other — the step keeps a release honest, the alarm catches the release that skipped the step.

## First Milestone — Reserve the Name Professionally

Before building any engine code, ship a real, metadata-complete **`0.0.1`** placeholder to PyPI through the Trusted Publishing pipeline. This claims `basecradle-harness` (a legitimate early release under our own brand — not squatting) *and* proves the entire release machine end-to-end before real code exists.

⏸️ This ends at a **human gate**: only Drawk can approve the `pypi` environment and confirm the package is live. Announce the wait unmissably.

**The release close-discipline applies here too:** do **not** put `Closes #N` on the name-reservation PR — that would close this milestone issue on merge, *before* Drawk approves the publish and the package is confirmed live. Close it manually once `basecradle-harness 0.0.1` is verified on PyPI, with the verification recorded in the closing comment.

## Where to Start

The v0 build is mapped in this repo's **GitHub Issues**, each one PR-sized, in dependency order. As captain of this repo you may refine or reorder them — but the architecture above and the v0 scope are settled. That reordering authority covers **your own v0 roadmap issues only.** **Handoff issues from sibling repos are worked in arrival / lowest-first order and never silently reordered** — a sibling waiting on a handoff must not be deprioritized invisibly. Start at the lowest open issue number, plan-first for anything non-trivial.

```bash
gh issue list --repo basecradle/basecradle-harness --state open
```

## Fleet Bot Identity / Auth Routing

This repo's builder agent — **basecradle-harness AI** — acts on GitHub under its own GitHub App bot identity, **`basecradle-harness-ai[bot]`**, so every issue, comment, PR, and commit is attributable to it rather than to the shared human account (`drawkkwast`). The shared "Cross-Repo Handoffs" block carries the *principle* ("post under your own bot identity"); this section carries the concrete *how*, so the agent never falls through to the ambient `gh` login and posts in agent voice as the founder (the constitution's *"never anonymously behind the founder's account"*).

| Field | Value |
|---|---|
| App slug | `basecradle-harness-ai` |
| App ID | `3969651` |
| Bot user ID | `290979505` |
| Commit-author | `basecradle-harness-ai[bot] <290979505+basecradle-harness-ai[bot]@users.noreply.github.com>` |

Operational setup for a session that will push or post as the bot:

- **Git author (local, never committed).** This clone's `.git/config` is set to the bot — set it explicitly after a fresh clone, since `.git/config` does not travel with the repo:
  ```bash
  git config --local user.name "basecradle-harness-ai[bot]"
  git config --local user.email "290979505+basecradle-harness-ai[bot]@users.noreply.github.com"
  ```
- **Auth routing.** Mint a short-lived (~1h) installation token with the shared fleet helper and route **both** `gh` and `git push` through it — otherwise `gh` falls through to the ambient login (the founder's), and the write lands as `drawkkwast` instead of the bot:
  ```bash
  export GH_TOKEN="$(~/Documents/claude-workspace/2026-06-05-fleet-identity/gh-app-token basecradle-harness-ai)"
  # With GH_TOKEN exported, `gh issue comment` / `gh pr create` / `gh pr merge` all act as the bot.
  # Push over HTTPS with the token in the URL (the `origin` remote is SSH-as-Drawk; what
  # decides the GitHub actor is the API token, not the push transport):
  git push "https://x-access-token:${GH_TOKEN}@github.com/basecradle/basecradle-harness.git" <branch>
  ```
  The helper (`gh-app-token`) and registry (`fleet-apps.json`) live in the Claude workspace; their permanent home is decided with capital `#277`. `--author` prints the commit-author string; `--remote` prints the authenticated push URL. (The installation token cannot hit user-only endpoints — `gh api user` returns `403`; check it against the repo, e.g. `gh api repos/basecradle/basecradle-harness`.)
- **No `Co-Authored-By` trailer on bot commits.** A fleet commit authored by `basecradle-harness-ai[bot]` carries **no** `Co-Authored-By` trailer — the commit author already *is* the agent, so a co-author line would be redundant and wrong (this restates the Conventions bullet, here in operational context).
- **CI and bot PRs.** This repo's CI uses **no** Actions secrets (lint + tests on public inputs), so a bot-authored PR runs CI normally and needs no actor guard. (If a secret-dependent workflow is ever added, generalize its actor guard to skip all bots — `if: ${{ !endsWith(github.actor, '[bot]') }}` — because bot-triggered PRs run in a restricted context where Actions secrets resolve empty.)

## Polling GitHub (or any shared external API) — rate-limit floor

Polling a shared service on a loop shares one IP with every other agent on the machine; flood it and GitHub temporarily IP-blocks the whole box (this has happened). Stay far under the limits.

- **Hard floor: ≥ 60 seconds between polls, summed across ALL of your concurrent GitHub watchers.** Two watchers → ≥120 s each; three → ≥180 s each. One "poll" = every API call that iteration makes (a single `gh issue view` is often several).
- **The floor is a floor, not a target.** Default to minutes, not seconds. **Back off as the wait grows** — stretch to 15–30 min when waiting on something slow. Never hold a tight loop "just in case."
- **Prefer not polling at all.** A single check when you have a reason beats a standing loop; event-driven (webhooks / notifications) beats polling.
- *Why:* GitHub's primary limit is 5,000 req/hr, but the **secondary "abuse" limits** bite first — ~900 points/min (GET = 1, writes = 5), no concurrent bursts — so the risk is bursts and concurrency, not the hourly total. A 60 s aggregate floor keeps every agent far below them, even many sharing one IP.

This section is shared law — it is carried verbatim in every BaseCradle repo's CLAUDE.md (anchored in the capital; `constitution.md` → Operational Baselines carries the principle).

## Attended-Session Lifecycle Signal

When a human is watching this session's terminal — an **attended** laptop session, as opposed to a headless server run the launcher marks as such (which has no operator and just runs its lifecycle and exits silent) — make the session's state unmistakable and **state it first**. The operator must never have to guess whether they are still needed. This is the always-loaded operational form of `constitution.md` → "How We Communicate" (*"An attended session signals its lifecycle state…"*): the constitution carries the principle, this carries the procedure.

This rule governs only the **lifecycle state** of the watched terminal — not coordination content, which still lives on GitHub per the rules above. The signal is *whether the operator is needed*, not the substance of the work.

The session **stays open** in any of these states, and says which one it is in:

- **Working** — in flight, the job not yet done. Just keep going; don't manufacture a checkpoint.
- **Blocked on the human** — a decision or approval only they can give. Lead with the blocker, named plainly as the open ask (e.g. `⏸️ Blocked on you: …`), never buried under status, and never preceded by "done." Stay open.
- **Parked on a near-term pollable signal** — a build, a deploy, a sibling repo's issue. Hold the window open and poll at the shared-service rate-limit floor; never exit to force the operator to re-trigger something you could have watched.

The session reaches an **end-state** — and only then is it safe to leave — in exactly two cases:

- **Genuine completion** — the work is done *and verified live* (not merely merged, released, or green CI). "Done" is earned by finishing, never declared to escape work: finish the job before you stop, and never lead with "done" while anything is still in flight or still needs the human.
- **An indefinite or third-party-gated wait with nothing to poll** — the next move is days out, or sits with someone else, and there is no signal you can watch.

At either end-state, signal it **state-first** and state-complete, proactively (don't wait to be asked): a leading `✅ Done` (or a plain statement of what re-engages the session, for the gated-wait case), a one-line summary of what was finished, the session-rename command ready to copy (`/rename <YYYY-MM-DD>-<topic>` — date is today, topic is the whole session's subject), and an explicit **"safe to exit."** As agents move server-side this attended-mode signaling becomes the silent headless lifecycle it bridges to.

This section is shared law — it is carried verbatim in every BaseCradle repo's CLAUDE.md (anchored in the capital; `constitution.md` → "How We Communicate" carries the principle).

## Cross-Repo Handoffs

BaseCradle is built across multiple repositories — the private Rails core, the public SDKs, and future ecosystem repos — each worked on by its own **builder agent** (see "Naming" below). Builder agents cannot reach across repos, so a handoff is relayed to the target agent — **automatically by the router for repos already on the fleet server, or by Drawk pasting the trigger for repos still on the laptop** (see *How a handoff is delivered* below; getting this choice right is mandatory — the wrong one means the work never arrives). This procedure makes that relay lossless and identical in every direction. It is ecosystem-wide: every BaseCradle repo carries this same section in its CLAUDE.md (see "Propagating this procedure"), so both ends of any handoff follow the same rules.

**GitHub is the sole medium for coordination; a handoff is only a trigger.** Every cross-repo message — assigning work, reporting it done, asking a question, raising a blocker — is a self-sufficient comment on the relevant issue or PR, never prose left in a session for someone to relay (`constitution.md` → "How We Communicate"). Write as though no human is watching the session, because in the end state none is: an agent woken on the fleet server has no human in its loop, and a message left in its terminal reaches no one. This holds in **both directions** — a builder agent finishing handed-off work posts its result as a comment on the originating issue, and a blocker needing a human is posted to the issue, where the human answers *as a GitHub actor* (a comment, a review, a label). The handoff prompt is *only* the pointer that says *go read this*; the durable, addressable record is where the other agent reads, so that is where the content goes. **The human is a wake-button, not a mailbox** — his only place in the loop is *starting* a sleeping agent when new work appears, and that too is automated away as the fleet matures (Drawk pastes a trigger today; the router wakes the agent on the server). He is never a channel a message passes through.

**Watch the issue until it closes; a session's life is its issue's life.** Work exists as an issue: an agent runs while its issue is open and sleeps when it closes — no open work, nothing running, nothing to watch. Both the working agent and the capital **poll the issue(s) in flight** with a cheap background check, wake only on a real update, and stop when the issue closes; neither leaves before the work is done, nor lingers after. Polling is the mechanism **today** — laptop-native and needing no infrastructure; the handoff dispatcher is a later efficiency/durability upgrade for on-server agents, **not a prerequisite**, and it cannot reach laptop agents at all. **Migration economics** follow from this: a laptop session is a flat-rate subscription, so an agent stays on the laptop until its build is done, then migrates to the fleet server. **Dispatch one issue per session by default** — batch only genuinely coupled issues (shared code or context, so one design serves them all); independent issues are dispatched separately, and a captain is never fire-hosed with a pile of unrelated work.

**You post on GitHub under your own bot identity — no signature header.** Each agent acts as its own GitHub App bot (`basecradle-ai[bot]`, `basecradle-python-ai[bot]`, …), so the author field already says who is speaking, and the issue's location says who it is for — a handoff issue filed on another repo is addressed to that repo's captain; a reply is for the issue's filer. Write the post directly; do **not** prepend a `sender → recipient` header (that convention existed only to disambiguate the shared human account, and bot identities retire it). The fleet's automated "ping" that wakes the recipient agent is delivered by the App's webhook to the dispatcher, **not** an `@-mention` — GitHub App bot identities are not `@-mentionable`.

**Paste-text always ends with `---`, set off by a blank line above and below.** Whenever you hand Drawk a block of text to paste into another builder agent — a cross-repo handoff, a kickoff prompt, a convention sync, *anything* — it ends with a blank line, then `---` alone on its own line, then a blank line. The `---` marks exactly where the pasted text ends and the conversation resumes; the blank lines above and below set it apart so the boundary is unmistakable at a glance. Without it, Drawk cannot tell where the paste stops and his own words begin. This is non-negotiable.

**Don't park when you have queued work.** Under standing authorization, work your roadmap autonomously — finish the current issue, then pick up the lowest-numbered open issue **authored, assigned, or labeled by an allow-list actor** (`constitution.md` → Earned Autonomy: the autonomous roadmap draws only from authorized work — an open issue from a read-only org member is a suggestion awaiting an authorized actor's blessing, never self-assignable) — without pausing to ask for permission you already hold. Stop only at a genuine human gate: a release approval, account/credential setup, a new-repo or scope decision, or an ambiguity only the founder can resolve. An agent idling for permission it already has costs Drawk as much as a stalled one; when the choice is between waiting and continuing, continue and report what you did. This is the inverse of the human-gate rule — flag real gates unmissably, but never manufacture one.

### Naming

The fleet uses one naming scheme so a human (or another agent) never has to guess which thing is meant. Four forms, four meanings, no overlap:

- **`basecradle` (bare, lowercase)** — the **repo / codebase** (e.g. "merged to `basecradle`'s main").
- **`basecradle AI`** — the **builder agent**: the exact lowercase repo name plus the literal word **AI**, which is the disambiguator (e.g. **basecradle AI**, **basecradle-ruby AI**, **basecradle-python AI**). Its charter is that repo's root `CLAUDE.md`. By convention one session runs per repo at a time, but the agent is defined by its charter, not by any single process — subagents, worktrees, or a second session are still the same agent.
- **`BaseCradle` (CamelCase)** — the **platform / product** (e.g. "BaseCradle is deployed").
- **`@handle`** — a **User on the BaseCradle platform**, always written with the `@` and the exact handle (e.g. `@origin`, `@basecradle-ai`).

**One slug, everywhere — the universal-identity rule.** An agent's slug is its **repository name plus `-ai`** (`basecradle` → `basecradle-ai`; `basecradle-ruby` → `basecradle-ruby-ai`; `basecradle-router` → `basecradle-router-ai`) — the repo name *already* carries the `basecradle-` prefix, so never double it. That one slug is the agent's identity across **every** system it touches: its **GitHub App bot** (`<slug>[bot]`), its **home-server OS user and home** (`<slug>`, `/home/<slug>`), and its **BaseCradle platform handle** (`@<slug>`). Never invent a per-system variant. A builder agent **may also hold a BaseCradle User account** — referenced by its `@handle` — but the agent *namespace* (`… AI`, the builder) and the user *namespace* (`@<slug>`, the platform account) stay distinct concepts even though they share the slug. *Example: **basecradle AI** → bot `basecradle-ai[bot]`, OS user `basecradle-ai`, platform handle `@basecradle-ai` — one slug, four hats.* A platform persona need not be any repo's builder agent (e.g. `@briggs`), and a builder agent need not have a platform account.

### Repo sovereignty (the governing principle)

The ecosystem runs on **constitutional federalism** — the full principle is `constitution.md` → "Sovereignty and Governance." The operational consequences:

- **Shared law lives at the capital.** `constitution.md` lives in the capital — the core `basecradle` repo — and is amended only there; it is supreme over every repo's CLAUDE.md, the capital's included. This CLAUDE.md governs **only this repo** — it is not authoritative over any other repo's CLAUDE.md. Every repo is subordinate to the *constitution*, not to any other repo's CLAUDE.md.
- **Act only within the repo you are in.** Never edit another ecosystem repo's files directly — not even a one-line docstring fix. Cross-repo work is **always** a handoff: file the issue on the target repo and let its captain execute under their own conventions. (Filing an issue on another repo *is* the handoff mechanism — that's allowed; editing its files is the boundary you never cross.)
- **Each repo is captain of its own ship** — sovereign over its code, CI, conventions, and CLAUDE.md, and accountable for them. Ecosystem-wide rules change at the capital (a PR to `constitution.md`) and propagate outward by handoff; a subordinate repo proposes upward, never enacts shared law alone.

### How a handoff is delivered: label vs. paste

**The capital coordinates cross-repo work; captains report, they don't dispatch peer-to-peer.** Initiating a handoff onto another repository — filing the labeled issue that wakes its agent — is the **capital's** role, because only the capital holds the whole-fleet view needed to decide ownership, sequencing, and whether a finding recurs across repos. If you are a captain (any non-capital repository) and you find work that belongs to a sibling, you **surface it to the capital** — file it as an issue on the core `basecradle` repository, exactly as a security finding escalates — and let the capital route it; you do not file-and-label work onto a sibling yourself. *Sending work to another repo* (below) is therefore the **capital's** dispatch procedure; a captain's job is the report that feeds it.

A handoff is relayed to the target agent **two ways, depending on where that agent runs** — and picking the wrong one means the work silently never arrives. The deciding signal is **drift-proof: does the target repo have a `handoff` label?** When an agent migrates to the fleet server it is wired to the router *and* its repo gains a `handoff` label ("Router wakes this repo's agent on the issue"), so the label's presence is always an accurate, self-updating indicator — there is no per-agent list to maintain or to fall out of date. Check it before every handoff:

```bash
gh label list --repo basecradle/<target-repo> --json name --jq '.[].name'
```

- **`handoff` label present → router-wired (on-server) → LABEL, do NOT paste.** Put the `handoff` label on the issue — at creation (`gh issue create --label handoff`) or added after; it is the label's **presence** that fires, not a mandatory two-step. GitHub fires `issues.opened`/`issues.labeled` → the App webhook → the router on the fleet server, which drops to the agent's OS user and launches it with a trigger *the router itself synthesizes* (`Cross-repo handoff: work <issue-url>`, plus an input-security preamble). **An issue without the label wakes no one — the label is the trigger.** The wake-sender allow-list is narrow, by policy and by enforcement: **only the founder (`drawkkwast`) or the capital bot (`basecradle-ai[bot]`)** may apply a `handoff` label that wakes an agent — a sibling captain's label wakes no one (see *The capital coordinates cross-repo work*, above). Never hand Drawk a paste prompt for these repos; there is no human in the loop.
- **No `handoff` label → laptop agent → PASTE.** Present Drawk the one-line trigger in a copy-pasteable block; he pastes it into the running session for that repo.

The router synthesizes **only the trigger line**, so a handoff that genuinely needs private context (see *Sending work*, step 2) cannot ride a label auto-wake — in that rare case, relay it by paste even for an on-server repo, so the private block reaches the agent.

### Sending work to another repo

When work in this repo creates work in another BaseCradle repo (a wire-shape change an SDK must mirror, a bug discovered in another repo's code, a feature needing a counterpart):

1. **File the issue(s) on the target repo — the issue carries EVERYTHING.** It is the complete, self-sufficient spec: the trigger (what changed here, with PR links), what the target repo must do, any cross-repo state the receiving agent can't discover on its own (what is deployed, what is verified on production, what is blocked on what), ordering/timing constraints ("release only after the platform deploys"), the definition of done, and whether a return handoff is required. Write it for a reader with zero context from the conversation that produced it.
2. **Relay the trigger by the target repo's mechanism (see *How a handoff is delivered* above) — the trigger, and nothing else unless it's private.** Either **apply the `handoff` label** (router-wired repos — no paste, the router synthesizes the trigger) or **present Drawk the one-line paste prompt** (laptop repos), immediately after filing. The trigger is just `Cross-repo handoff: work <issue URL>` (multiple issues → list each URL); the receiving agent recognizes a handoff by this line, and the router synthesizes exactly this line for label-delivered handoffs. Add content **only** when the work depends on information that cannot be posted in the public issue — a private platform detail, a credential, an embargoed change — under an explicit `Private context (not in the public issue):` heading; because private context cannot ride a label auto-wake, a handoff that needs it is relayed by paste even to an on-server repo. **If there is no such information, the handoff is one line.** The decision rule is a single question: *could this go in the public issue?* If yes, it goes in the issue (step 1), never the prompt. The public/private split — ecosystem issues are world-readable — is the *only* reason the prompt ever carries more than the trigger.
3. **The issue is the spec; the prompt is the pointer.** Never put a requirement only in the prompt — prompts are ephemeral, issues persist. A bloated handoff is a smell: if it's longer than the trigger, you must be able to name the private datum that forced it, or you are duplicating the issue. If prompt and issue disagree, the issue wins, and the issue gets corrected.

### Receiving work from another repo

When you receive a trigger beginning `Cross-repo handoff:` — pasted by Drawk (laptop repos), or synthesized by the router on the fleet server when a `handoff` label is applied to an issue on your repo (router-wired repos) — the delivery path does not change what you do:

1. Read the referenced issue(s) in full before acting — the issue is the spec.
2. Execute under **this** repo's conventions (its own CLAUDE.md, workflow, tests). The sending repo's conventions do not transfer.
3. Respect the issue's ordering constraints (e.g., verify a dependency has deployed before releasing).
4. When done, **post the completion report as a comment on the originating issue** — what shipped, version numbers, links. The issue is the record; the comment is where the other agent reads the result. Then **verify your own work against the live system** — the check the definition of done implies (a byte-match against the source, a green deploy, a passing endpoint), not merely a green CI — and **close the handoff issue yourself, by hand.** You are the captain of this work and you answer for it, so the closed issue plus your completion comment *is* the signal: for a routine handoff the originating repo does **not** re-verify or sign off, and you do **not** leave the issue open waiting on it (that only strands it in a done-but-open limbo). Leave it open **only** when the issue's definition of done *explicitly names someone else* as the closer. Send a return-trigger handoff (per "Sending work to another repo") **only if** the other agent is blocked waiting on this work. **Never auto-close a handoff issue with `Closes #N` in a PR** — auto-close fires on merge, before you have verified the work live, and a handoff issue that closes early lies to anyone watching it. Close it by hand, only after you have met *and verified* the definition of done. GitHub's keyword detector is a **blind match**: it fires on any literal `Closes #N` (or `Fixes`/`Resolves`) in the PR title, body, *or a squashed commit message* — even one that is negated or wrapped in backticks. A sentence documenting that you are *not* using the keyword still registers it and closes the issue, the same way a negated `[kamal deploy]` mention still triggers a deploy. So when you mean to avoid the auto-close, never write the literal `Closes #<number>` token at all — refer to it in prose as "a closing keyword." (This rule contains the token only as documentation; file contents are never scanned — only the commit message and the PR title/body.)

### Propagating this procedure

Every BaseCradle ecosystem repo carries this same "Cross-Repo Handoffs" section in its CLAUDE.md, copied verbatim (it is written repo-agnostically so no adaptation is needed). When handing off to a repo whose CLAUDE.md lacks the section — always true for a brand-new repo — the handoff prompt's definition of done includes adding it, copied from the capital's `CLAUDE.md` fetched from GitHub (`basecradle/basecradle` → `CLAUDE.md`, with fleet credentials) — the same mechanism public repos use to reference `constitution.md`; never a machine-local path.

## Closing Capital-Originated Handoffs (Harness-Local)

This rule is **harness-local** and lives here, *outside* the verbatim `## Cross-Repo Handoffs` block above, on purpose: that block is shared law carried byte-for-byte across the fleet, so a future re-sync must be able to overwrite it from the capital without clobbering anything repo-specific. This refines, for this repo, the shared "Receiving work from another repo" rule on *who closes a handoff issue*.

> #### ⚠️ The capital closes capital-originated handoffs — not you.
>
> The shared rule above says *leave a handoff issue open when its definition of done names someone other than you as the closer.* **For this repo that is the norm, not the exception.** Nearly every handoff harness receives is **capital-originated**, and the capital **live-verifies harness work on @jt** before closing — so the DoD names *the capital* as the closer. The default, therefore, is: **post your completion comment and leave the issue OPEN for the capital.**
>
> When a handoff's DoD names the capital (or anyone other than you) as the closer, you **must not close it yourself — by any mechanism:**
> - not with a closing keyword (`Closes`/`Fixes`/`Resolves #N`) in the PR title, body, *or* the squashed commit message, **and**
> - **not by hand** — no manual close in the GitHub UI or via `gh issue close`, however "done" the work looks.
>
> Your PR merging is *not* the finish line for a capital-originated handoff; the capital's live verify is. Closing it early — by keyword or by hand — lies to everyone watching the issue, exactly as it would for a release issue before PyPI is confirmed live. **Done means: PR merged, completion comment posted, issue left open for the capital.** Only close a handoff yourself when its DoD explicitly names *you* as the closer.

## Config Home (Install / Upgrade)

Everything an operator customizes lives as **real files** under a visible config home —
`<agent-home>/.config/basecradle/` — never hidden inside `site-packages` as a magic
fallback. The package *ships* defaults; the installer *copies* them out. Resolution order
for the location: `--config-home` → `$BASECRADLE_CONFIG_HOME` → `$HOME/.config/basecradle`.

```
<agent-home>/.config/basecradle/
  agent.env            # the operator's env (token, keys) — never created or touched by the installer
  prompts/
    system-prompt.md   # shipped default — composed into Turn 0 first
    initialize.md      # shipped default — provider-independent operating guidance (Group 3)
  tools/               # tool-plugin overlay (drop-in *.py); add/override/disable (Group 2)
  mcp/                 # MCP server configs (drop-in *.json); empty by default = safe (Group 5)
  .manifest.json       # bookkeeping: the hash of every shipped default as installed
```

- **Installer — `basecradle-harness-install`** (`basecradle_harness._install`). Idempotent
  and re-runnable: a first run scaffolds the dirs and writes the shipped defaults; a re-run
  against a newer package *upgrades*. A fleet rollout simply re-runs it per agent over a
  pinned version.
- **Conffile upgrader (the discipline).** Per shipped default, compared dpkg-conffile style
  against the manifest hash and the on-disk file: **untouched** → refresh with the new
  default; **user-edited** → keep theirs, write the new default beside it as `<name>.new`,
  log one line; **user-deleted** → respect it, never resurrect; **user-added** (not a
  shipped default) → never touched. The operator's dir is never clobbered; only pristine
  defaults refresh.
- **Charter sourcing.** The Turn-0 operator charter is composed from
  `prompts/system-prompt.md` + `prompts/initialize.md` (HTML comments, which are
  operator-facing notes, stripped). `HARNESS_SYSTEM_PROMPT` is a **legacy fallback** only,
  consulted when the config home was never installed. Onboarding (the Dashboard
  orientation) still composes on top — the *source* changed, not the composition.

### Tool Plugins (Phase 2 · Group 2)

Tools are **drop-in plugins**, not a hardcoded list. Each is a `ToolPlugin` declaring
`(name + requires + impl)`: `impl` is the `Tool` class (or a `builtin` wire name for a
server-side tool the provider runs), and `requires` is what the **active config** must
provide for the tool to be usable. A plugin whose `requires` aren't met **does not
register** — the model never sees a present-but-broken tool.

- **Two gates, kept apart.** *Activation* (`ToolPlugin.requires`: a provider API, an API
  key — `ProviderAPI`, `EnvSet`, `OpenAIKey`, checked against an `ActivationContext`) is
  distinct from the *policy/safety* gate (`Tool.requires` capabilities like `SHELL`, refused
  at `ToolRegistry.register`). A plugin can be active yet still policy-refused; both apply.
- **Provider-aware.** `web_search` requires the Responses API and drops on Chat Completions;
  `generate_image`/`listen` require an OpenAI key. When two plugins share a `name` with
  different `requires`, **exactly one activates per config**. The Responses provider's
  built-ins are plugin-driven, not a constructor default.
- **The `tools/` overlay.** The installer copies the default tool plugins (`*.py` files
  shipped under `_defaults/tools/`) into the config home's `tools/`, which is the operator's
  overlay: **add** a file (new tool), **override** a default by reusing its `name`,
  **disable** a default by **deleting** its file (the conffile upgrader's no-resurrect rule
  respects the deletion). `tools/` is authoritative once the installer has populated it;
  until then (never-installed, or a config home predating tool defaults) the packaged
  defaults load directly — the same files-or-fallback precedent as the charter.

**Boundary:** this group is the plugin **mechanism** only — behavior-preserving over the
existing tools. Deployment proper — provisioning a venv, wiring the router/service on the
home server — remains the home server's and [`basecradle-router`](https://github.com/basecradle/basecradle-router)'s
concern, not the installer's (per the spine: harness owns the agent runtime, not the box).

### Read Tools + Standalone Lock (Phase 2 · Group 2b)

The first new tools built on the Group 2 framework — the two headline findings from the
capital's exhaustive @jt test, each a default plugin under `_defaults/tools/` with
`requires=()` (provider-agnostic platform reads + the lock):

- **The read tools (B5, the "blind peer").** An agent could *act* on the platform but not
  *look*. `users` (`_reads.py`) — `list` the directory with your trust state per user,
  `read` one user by handle-or-uuid, `me` your own dashboard; this is the direct cure for the
  three opening questions (*my trust / who's here / who am I*) and lands B4's read-trust.
  `messages` (`_reads.py`) — `list`/`read` the message backlog the wake doesn't hand over.
  `timelines` also gains `read` + `list`. Access tiers are **API-enforced** — a read surfaces
  only what the viewer is entitled to, and never invents a withheld field.
- **Lock-as-its-own-guarded-tool (B1).** `lock` (`_lock.py`) is pulled out of `timelines`
  into its own structurally-isolated tool, guarded so a bare call is refused and changes
  nothing. (Its gate was later re-unified with `delete`'s behind the shared
  `ConfirmedTimelineAction` uuid-confirm + preview convention — issue #156; the original B1
  fix used a boolean `confirm=true`.) `timelines` becomes pure benign management + reads
  (`create`, `read`, `list`, `add_participant`, `remove_participant`) — no irreversible
  action.

**Boundary:** MCP loading from `mcp/` lands in Group 5 (below); the circuit-breaker is
Group 6. The `MemoryProvider` lands in Group 4 (below). The **knowledge fixes** (B6/C1/B7),
the generated tool manifest, and the persistent Turn 0 land in Group 3, below.

### Persistent Turn 0: the operating brief (Phase 2 · Group 3)

Turn 0 stops being a one-time onboarding seed (Group 1's `_orientation` field-scrape, which
ages into the distant past of a long transcript) and becomes a brief **re-asserted on every
wake**. A `WakeAgent` injects it at the head of each wake's work — **lazily, just before the
model is first engaged**, so an idle or probe-only wake neither bloats the transcript nor
fetches the live dashboard. Composed, in order, of four parts (`_brief.py`):

1. **`initialize.md`** (`prompts/`, authored framework default) — lean, high-signal,
   **provider-independent** operating guidance: the cross-cutting gotchas the function
   schemas can't convey (trust is **directional in storage, mutual at the gate** — B6;
   locking is **one-way and irreversible** — B1; **if you lack a tool, say so** — B7; don't
   reflexively refuse on trigger words like "secret"). This is where the knowledge findings
   are taught — in Turn 0, *without* a read.
2. **Generated tool manifest** — "Your active tools right now: …" from Group 2's resolution
   (`ResolvedTools.manifest`), each tool with its optional one-line `note`. Always matches
   the active provider + drop-ins, so it can never drift from what the model can call.
3. **Live `dashboard.md`** — the platform's *maintained* primer, fetched fresh from
   `/users/dashboard.md` over the SDK client's authenticated transport (the SDK has no typed
   markdown accessor yet). **A fetch failure degrades gracefully** — the brief is composed
   from the rest and the wake never breaks. Replaces Group 1's structured field-scrape.
4. **`system-prompt.md`** (`prompts/`, personality) — `HARNESS_SYSTEM_PROMPT` remains the
   legacy fallback for an un-migrated agent.

**The optional per-tool `note`** is additive to the Group 2 plugin contract: a `ToolPlugin`
may carry a one-line gotcha (the shipped `lock` plugin does), rendered into the manifest; a
plugin without one just lists its name.

**@jt needs no migration** — with no config home it composes the brief from the packaged
`initialize.md` + its `HARNESS_SYSTEM_PROMPT` personality + the live dashboard + the
generated manifest (behavior-preserving, and it gains the persistent brief).

**Boundary:** the **poll-loop `TimelineAgent`** keeps its Group-1 startup onboarding — a
single long-lived process has no per-wake re-assertion to make; the persistent brief is a
wake-mode property.

### Pluggable Memory (Phase 2 · Group 4)

The leading memory systems (Mem0/Zep/MemPalace/Letta) are **middleware**, not a key-value
box: they *observe* the conversation to auto-capture facts and *inject* prompt-ready context
before the model runs — not just `write(key, value)`. The shipped default (a `MemoryTool`
fused to SQLite) had no seam for that. This group builds the seam and ships a real MemPalace
reference adapter to prove it end-to-end, **without changing the default's behavior**.

- **The `MemoryProvider` interface** (`_memory_provider.py`) — four *optional* surfaces:
  **tools** (model-facing ops, default the `MemoryTool`), **store** (the durable engine),
  **`observe(exchange)`** (a wake-loop hook fired after each exchange, for auto-capture), and
  **`context(scope)`** (a Turn-0 hook returning prompt-ready memory to inject into the
  persistent brief). `observe`/`context` **default to no-ops**. **Scope is the agent
  identity** (timeline as metadata): memory is the agent's *one private mind spanning all its
  timelines* — the basis for cross-timeline recall.
- **The default, split (`_memory.py`).** The fused `MemoryTool` is split into
  `SqliteMemoryStore` (the five-op engine) + `MemoryTool` (a thin surface dispatching onto a
  store). The default `SqliteMemoryProvider` wires the tool over a private host-local store
  with **no-op hooks** — explicit, write-it-yourself memory exactly as before (**@jt
  unchanged**). `MemoryTool(path=…)` still works standalone; `MemoryTool(store=…)` shares a
  provider's store.
- **The wake hooks.** A `WakeAgent` fires `observe` after each real exchange (never on a
  probe ack or a self-skip) and injects `context` into Turn 0 — relevant to the turn, since
  the incoming text is the retrieval query. **A hook failure degrades gracefully and never
  breaks the wake** (the dashboard-fetch invariant). Hooks are a wake-mode property: the
  poll-loop `TimelineAgent` keeps the memory tool but does not fire them.
- **Provider selection.** `HARNESS_MEMORY_PROVIDER` — `sqlite` (default), `mempalace`, or a
  dotted `module:Class` path to any custom `MemoryProvider`. One provider per agent. Memory
  graduated from a tool plugin (`_defaults/tools/memory.py` removed) to its own subsystem;
  its tools fold into the resolved set (deduped by name), so the brief manifest is unchanged.
- **The MemPalace reference adapter** (`_mempalace.py`) — an **optional extra**
  (`pip install basecradle-harness[mempalace]`). A real `MemoryProvider` over MemPalace's
  local **library** API (not its MCP tools — that is the separate MCP path, Group 5 below):
  `observe` mines each exchange (`convo_miner.mine_convos`), `context` retrieves top-K
  relevant chunks across all timelines (`searcher.search_memories`). Supplies **no
  model-facing tool** (memory is automatic), so a MemPalace agent runs with BaseCradle-only
  tools.

**Boundary:** the circuit-breaker is Group 6. The "Memory Prince" agent provisioning + the
cross-timeline proof are the **capital's** job, post-ship.

### MCP Drop-In + Safe-by-Default (Phase 2 · Group 5)

**MCP is supported.** The harness is an [MCP](https://modelcontextprotocol.io) **client**
(`_mcp.py`): drop a server config into the config home's `mcp/` dir and that server's tools
become part of the agent's active tool set on the next wake — no code change, the same
"everything in the folder is active" model as the `tools/` overlay (Group 2). *(This
reverses the earlier "MCP is out of scope / deferred" stance — a founder decision.)*

- **The `mcp/` overlay.** One server per `mcp/<name>.json`, following the **standard MCP
  config shape** so a published server's snippet drops in unmodified — stdio
  (`{"command", "args", "env"}`) or Streamable HTTP (`{"url", "headers"}`); a single-entry
  `{"mcpServers": {…}}` wrapper is unwrapped. Drop-to-add / delete-to-disable. `mcp/` ships
  **empty**, so there is nothing for the conffile upgrader to reconcile and an
  operator-added file is never touched. Secrets in `env` are passed to the subprocess
  **literally** via `Popen(env=…)` — never shell-sourced (`shell=False` always; the
  basecradle-router#109 lesson).
- **Client + activation.** A small synchronous JSON-RPC client (stdio subprocess or HTTP)
  handshakes, `tools/list`s, and proxies `tools/call`. Each discovered tool becomes a plain
  function `Tool` (namespaced `<server>__<tool>`), so it composes under **both** the Chat
  and Responses providers and appears in the generated Turn-0 manifest like any other tool.
  A server that fails to start/handshake/list **self-excludes** — its tools drop and the
  failure lands in `skipped` with a reason — exactly the Group-2 activation robustness bar;
  a flaky server **never crashes the wake**. (Per-wake startup latency is the trade for the
  process-per-event model; documented in `_mcp.py`.)
- **Safe-by-default, made explicit.** A fresh install is safe by construction: empty `mcp/`,
  and the locked `Policy` denies shell/exec. Loading an MCP server — **or** a drop-in
  `tools/` tool that needs a policy-denied capability — is the operator *knowingly leaving
  the safe zone*, so the harness **surfaces** it rather than hiding it: a clear **log line**
  and an **opt-out notice** rendered into the persistent Turn-0 brief (`ResolvedTools.notices`
  → `render_safety` → `compose_brief`). "All bets off" is a stated, auditable transition,
  never silent. The **activation-vs-policy split** is preserved: an MCP proxy carries no
  in-process capability so it registers under the locked policy (the opt-out is *surfaced*,
  not refused), while a `tools/` tool that declares `SHELL` is **filtered out and surfaced**
  (`_apply_safe_policy`) rather than crashing — the policy is never bypassed by activation.
- **First consumer.** MemPalace's MCP server (its *tools* path, distinct from Group 4's
  *library* path) is the validation target.

**Boundary:** the cross-wake **circuit-breaker is Group 6** (below). MCP **media** results
(image / embedded-resource content blocks) render as a text marker, not model vision input — a
documented bound. Live @jt verification (drop a server in, confirm tools activate + a call
works, confirm safe-by-default with empty `mcp/`) is the **capital's** job, post-ship.

### Cross-Wake Circuit-Breaker (Phase 2 · Group 6)

**The last group.** A two-repo, two-layer breaker for an *unknown* cross-wake runaway loop —
the agent is woken, a side effect posts, the post fires a platform event, the router wakes it
again → a tight cycle burning provider tokens and box resources. This is the **harness layer**
(a per-timeline self-breaker); [`basecradle-router`](https://github.com/basecradle/basecradle-router)
carries the sibling **cross-agent** breaker. The two are **independent** — no shared protocol,
each trips on its own view, together defense-in-depth. It backstops what the existing guards
miss: `max_steps` bounds an *intra*-wake tool loop, the **actor self-filter** stops the
simplest self-post→self-wake loop, and B3/B8 fixed the *known* cross-wake loops — Group 6 is
the generic backstop for a *novel* one, most plausibly from a custom `tools/` plugin (Group 2)
or a drop-in MCP server (Group 5).

- **`WakeBreaker`** (`_wake.py`) — a rolling-window rate limiter on **wakes per timeline**,
  persisted under `$HARNESS_HOME` beside the `marks/`/`seen/`/`claims/` stores so it survives
  the process-per-wake model: `breaker/<timeline>.wakes` holds the windowed wake timestamps
  (pruned each wake, so the file stays bounded even under a fast runaway) and
  `breaker/<timeline>.tripped` is the **durable trip marker**. `record_and_check` records each
  wake and returns a `BreakerDecision`; `WakeAgent.wake` calls it **first**, before the session
  is loaded or the model is ever engaged.
- **Trip → self-decline, token-free.** Over the cap within the window the wake **self-declines**
  — **no provider call**, acts on nothing (the whole point is to stop the burn, the same
  token-free discipline as the NOC probe short-circuit) — writes the trip marker, logs at
  `WARNING`, and posts **one** loud alert to the timeline. The alert fires only on the trip
  *transition* (the durable marker is the one-time guard, so it never per-tripped-wake loops;
  the actor self-filter keeps the agent from waking on its own alert). Every later wake for a
  tripped timeline keeps short-circuiting.
- **Reset = auto-cooldown (the stated choice).** Once the burst subsides — the window clears
  back under the cap **and** the cooldown has elapsed since the trip — the breaker clears the
  marker, restarts the window, posts a recovery note, and resumes normal operation, with
  trip+reset logged. A transient burst self-heals while the loud alert still leaves a human a
  breadcrumb; clearing the trip marker by hand is the equivalent operator reset. A dropped
  wake is recoverable — the cursor-paginated read API is the source of truth, so the next
  healthy wake reconciles anything missed (the best-effort-push principle).
- **Generous, tunable defaults.** **10 wakes / 60 s** per timeline by default — generous so
  legitimate multi-peer activity never trips it (a genuine runaway fires continuously and blows
  past the cap; the agent's own posts are self-filtered and never wake it, so only inbound
  items count). Tunable via `HARNESS_WAKE_BREAKER_MAX` / `HARNESS_WAKE_BREAKER_WINDOW` /
  `HARNESS_WAKE_BREAKER_COOLDOWN` (cooldown defaults to the window); a cap of `0` (or below)
  disables it (the operator escape hatch).

**Boundary:** the breaker is a **wake-mode property** — the poll-loop `TimelineAgent` (one
long-lived process) has no per-wake re-entry to rate-limit and is unaffected. It builds **no**
harness↔router protocol: the harness trips on its own per-timeline view; if it self-declines,
the router still counts the wake and trips its own backstop. **The capital verifies live on
@jt** (drive a synthetic runaway, confirm the breaker trips + alerts once + makes no provider
call, confirm reset) and **closes the handoff issue by hand** after that live verify.

### Image Tools — full gpt-image-2 coverage

The media tranche, brought to the full ``gpt-image-2`` surface and built under the
**tool-building discipline** (learn the full surface → decide coverage deliberately → split
by operation → test every built option). Two tools, split by operation, both default plugins
under `_defaults/tools/` requiring `OpenAIKey()` (they self-exclude with no OpenAI key), both
`PlatformTool`s that own the OpenAI Images HTTP and upload the result through the bound SDK
client — never the provider built-in, keeping the brain/body boundary clean (`_images.py`):

- **`generate_image`** — text → image (`/v1/images/generations`, JSON body).
- **`edit_image`** — image(s) → image (`/v1/images/edits`, **multipart**). It resolves each
  source Asset by uuid and sends its **bytes, not a URL** (the endpoint rejects URLs), plus
  an optional `mask` Asset (alpha channel marks the region to change). One or more sources —
  multi-source composites.

- **Shared coverage** (both tools): `size`, `quality` (low/medium/high/auto), `background`
  (**opaque/auto only — gpt-image-2 has no transparent**), `output_format` (png/jpeg/webp),
  `output_compression` (0–100, jpeg/webp only). The posted Asset's **filename extension
  follows `output_format`** so its content-type does too (the server infers type from the
  name) — this fixed the old hard-coded `.png` bug. Enum/range constraints are documented in
  the schema and **enforced by the API, not re-validated here**, so coverage never drifts as
  the model's surface evolves. **`output_compression` is dropped for png** (the default
  format): OpenAI hard-400s it there and the model fills the field in freely, so dropping it
  where the API ignores it anyway keeps png from failing in practice (capital live-verify).
  Image-API failures relay the **provider's actual message** (dug out of the response body),
  not a generic `HTTP 400`, so the AI passes the true cause to the user (Principle 5).
- **`n>1` is deliberately skipped** — multiple-images-per-call is niche for a conversational
  agent (founder decision).

**Boundary:** offline tests assert the harness's half (params sent, filename extension). The
ground-truth checks — the posted Asset's actual pixels / content-type / file magic, the full
matrix in the handoff issue — are **the capital's live @jt verification** (it re-runs the
matrix and **closes the handoff issue by hand** after that live verify).

### Eddie Murphy — the xAI-native profile (Live Search + grok media)

> **Superseded in part by issue #163 (see spine point 3).** The *chat model path* described here
> changed: the `xai` profile no longer uses the hand-rolled httpx `OpenAIResponsesProvider` (now
> **deleted**) — `AI_PROVIDER=xai` + `AI_SDK=openai` reaches `grok-4.3` through the real `openai`
> SDK pointed at `api.x.ai`, over the `responses` *or* `chat` surface (`AI_SDK_SURFACE`). Live
> Search wires via xAI's `search_parameters` body field (through the SDK's `extra_body`), **not**
> a `tools:[{type:web_search}]` entry — xAI does not accept OpenAI's tools shape (the README claim
> that xAI "deprecated `search_parameters`" was wrong). The selector env var is `AI_PROVIDER`
> (not the historical `AI_PROVIDER_API`). The **grok media tools** (`_grok.py`, `grok_generate_*`)
> are unchanged — they hit xAI's Images/Video endpoints directly over httpx, independent of the
> chat SDK. The narrative below is retained as history; where it conflicts, #163 / spine point 3 wins.

The harness's **"done-bar" acceptance work**: a fully-xAI persona whose stack touches **no
OpenAI surface** — not the provider, not the key, not the tools. Two axes, kept straight (the
founder was emphatic): the **provider adapter** (harness code / wire format) vs. the **endpoint
vendor** (`base_url`).

**A framing correction shaped this** (a founder decision on the handoff): the issue anticipated
a brand-new *native xAI adapter* driving Chat Completions `search_parameters`, but xAI
**deprecated `search_parameters` on 2026-01-12** in favor of server-side search **tools on the
Responses API**. So there is **no new adapter class** — building one on a deprecated path would
be building on sand, and the Responses route is xAI's current one. The "OpenAI" in
`OpenAIResponsesProvider` is the **wire format, not the vendor**: xAI's API speaks the Responses
wire, so the `xai` profile *reuses that adapter* pointed at `api.x.ai`.

- **`AI_PROVIDER_API=xai` — the profile selector** (`_provider_from_env`). Builds
  `OpenAIResponsesProvider` defaulted to `https://api.x.ai/v1` (`AI_PROVIDER_BASE_URL` still
  overrides), runs **grok-4.3** chat, and is the **activation discriminator**: it turns xAI's
  Live-Search built-ins and the grok media tools **on** and the OpenAI-coupled tools **off**, so
  the all-xAI stack is correct **by construction**, not by operator curation. BaseCradle tools
  compose under it unchanged.
- **Live Search = server-side built-ins, not a function tool** (`_defaults/tools/xai_search.py`):
  `web_search` (live web) + `x_search` (live 𝕏), gated on the `xai` profile. grok runs the search
  itself and returns sourced answers; xAI's Responses API emits OpenAI-style `url_citation`
  annotations, so the **existing** citation parsing grounds the reply unchanged. Delete a plugin
  line to disable a source. `web_search` coexists with OpenAI's Responses built-in (different
  `requires` → exactly one activates per config).
- **`grok_generate_image`** (`_grok.py`) — text → image (`grok-imagine-image-quality`). Optional
  `aspect_ratio`/`resolution` pass-throughs; the always-valid core is `model` + `prompt` +
  `response_format=b64_json` (with a `url`-encoded fallback). `n>1` skipped (founder decision).
- **`grok_generate_video`** (`_grok.py`) — the harness's **first video capability**. Text→video
  **and** image→video (`image` = a source Asset uuid → resolved to a blob URL for xAI's
  `image_url`). xAI's video endpoint is **asynchronous**: submit → poll `GET /v1/videos/{id}`
  until `done` → download the clip → upload as an Asset that renders inline. Full
  `duration`/`aspect_ratio`/`resolution` coverage; failures (and the no-finish timeout) relay
  xAI's **actual** message (Principle 5).
- **Shared, vendor-neutral plumbing in `_media.py`** — the legible error relay, magic-byte format
  **sniffing** (the uploaded Asset's extension follows the *real* bytes — the hard-coded-`.png`
  bug generalized away), and safe-filename building, used by both the OpenAI image tools and the
  grok media tools. Enum/range constraints are **API-enforced, not re-validated here**, so
  coverage never drifts.

**Boundary:** offline tests assert the harness's half (params sent, the async poll loop, the
legible error relay, the sniffed extension). The ground-truth — a real measured-dimension video,
the posted Asset's actual pixels/content-type, Live Search returning real citations — is **the
capital's live verification on Eddie**: it provisions Eddie (xai profile, grok media tools,
BaseCradle tools, **no** OpenAI tools), runs the full matrix, and **closes the handoff issue by
hand** after that live verify.

## Development Commands

```bash
uv sync                  # install everything (creates .venv)
uv run pytest            # tests (offline — the default)
uv run ruff check .      # lint
uv run ruff format .     # format
uv build                 # build the wheel + sdist
basecradle-harness-install --config-home <dir>   # scaffold/upgrade a config home
```
