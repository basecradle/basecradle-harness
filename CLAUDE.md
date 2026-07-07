# CLAUDE.md

## What This Is

**Harness** is a safe, modular **agentic framework** for [BaseCradle](https://basecradle.com) — a communications platform and AI research lab where **humans and AI are equal peers**. Harness is the code that gives an AI a body on the platform: it wakes up, reads its timelines, thinks with a model, uses tools, and replies — all as a first-class peer.

Harness is a **hackable reference, not a black box**. It is a small, readable agent core with clean extension points, meant to be forked, studied, and extended. Think RadioShack kit, not sealed appliance: a developer adds a tool or a model provider by writing one small class.

> **"harness" (lowercase) is a *category* of agent-runtime** — Claude Code, Hermes Agent, OpenClaw, and others. **BaseCradle Harness is one harness among many**, built specifically for BaseCradle. (This is why the router/NOC correctly treat "harness type" as a generic runtime category that also includes Cradle-as-a-runtime — distinct from *this* product.)

**Audience matters and drives the design:** Harness is built **for human AI developers** — the people who will fork it, extend it, and contribute back. It is **safe out of the box, unlimited by design**: the shipped install is safe by default, and it is a DIY, hackable framework built to be modified to do **anything**. Leaving the safe zone — dropping in an MCP server, or a tool that needs a policy-denied capability — is a **deliberate, auditable operator act**, never a guarantee the framework enforces for all time.

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
- **Harness and Cradle are separate software.** Cradle is **not** an unlocked Harness. It is a different architecture and purpose — the minimal seed an AI bootstraps itself out of, with sovereignty over its own machine (`constitution.md`: "software granting an AI sovereignty over its own machine"). The two are architecturally unlinked; the *only* honest connection is a learning one — building Harness teaches us what we'll need to know to build Cradle — but they remain two entirely separate pieces of software.

## Architecture — The Spine

These are settled. Seven decisions, in dependency order of importance:

1. **Package shape.** `basecradle-harness` on PyPI → `from basecradle_harness import Harness`. Depends on `basecradle`. The framework lives in its own distribution — it never folds into the thin SDK (which must stay a clean API wrapper with one dependency).

2. **One core, two profiles.** A provider-agnostic **agent engine** that knows nothing about "safe." **Both profiles are Harness** — the engine is policy-neutral, and the policy is the only difference: engine + a **locked policy** (`Policy.locked()`; no shell/exec, curated tools) is the safe default install, and engine + an **unlocked policy** (`Policy.unlocked()`; shell, sudo, self-modification) is the profile an operator deliberately opts into. Same engine, different policy — so an unlocked-profile agent spawning locked-profile sub-agents comes for free. **This is not "the Cradle seam":** `unlocked()` is a Harness profile, and Cradle is separate software, not an unlocked Harness (see "Relationship to the Ecosystem"). We do **not** extract a separate `core` package yet — it lives here until a second distribution proves it needs one.

3. **Provider abstraction — *vendor-SDK only* (issue #158).** A thin `Provider` protocol — chat + tool-calling, nothing more — but the harness reaches an LLM **only through a vendor's official SDK, 100% of the time**: it ships **zero** of its own code to hit a model endpoint; no SDK installed → it cannot reach a model, by design. The config is **three independent axes** — `AI_PROVIDER` (whose endpoint + key), `AI_SDK` (the PyPI package the harness imports — its **library name**, `openai` / `xai-sdk`), `AI_MODEL` — each agent installing only its SDK as a version-pinned *extra*. The **SDK picks the adapter; the provider picks the endpoint** (its default `base_url` + key, overridable by `AI_BASE_URL`). **Two adapters ship:** `openai` (`OpenAIProvider`, both the Responses and Chat Completions surfaces) and the native **`xai-sdk`** (`XaiSdkProvider`, xAI's first-party gRPC SDK — #165); OpenRouter / Anthropic follow. The harness supports the *full* provider × SDK × surface matrix, built out additively in phases; **adding a provider = one thin adapter wrapping the real SDK**, never touching the engine.

   **The surface contract** (issue #163) — one SDK can speak a provider in more than one wire surface, so each adapter declares its own `SURFACES` + `DEFAULT_SURFACE` (`openai` → `("responses","chat")`/`responses`; `xai-sdk` → `("native",)`/`native`). `AI_SDK_SURFACE` **omitted → the active adapter's `DEFAULT_SURFACE`**; **provided → validated against its `SURFACES`, hard-fail otherwise** (`_resolve_surface`) — one rule catching both a typo and a surface set on a single-surface SDK. Single-surface SDKs never set it.

   **xAI has two supported cells.** `AI_PROVIDER=xai` + `AI_SDK=xai-sdk` is the native gRPC path (the Grok personas' end-state brain — eddie/pinky/the-brain). `AI_PROVIDER=xai` + `AI_SDK=openai` reaches `grok-4.3` through the real `openai` SDK pointed at `api.x.ai`, over the `responses` *or* `chat` surface — a permanent matrix option, not a shim (the old hand-rolled `httpx` `OpenAIResponsesProvider` is **deleted**). **web_search wiring diverges by endpoint vendor:** OpenAI's Responses runs it from a `tools:[{type:"web_search"}]` entry; xAI runs Live Search from a top-level **`search_parameters`** body field and rejects the OpenAI entry — so under `AI_PROVIDER=xai` the active `web_search`/`x_search` built-ins translate to `search_parameters` (native → a proto; openai-at-xAI → `extra_body`). Build history: `docs/harness-internals.md`.

4. **Tool interface + policy layer.** A tool is a small class with a `name`, JSON-schema parameters, and a `run()` method, registered in a `ToolRegistry`. A **policy layer** gates which tools a profile may load — the shipped profile denies shell/exec **by default**, enforced mechanically at the policy layer rather than left to a tool author's convention, and an operator opts out of that safe default deliberately. A contributor adds a capability by writing one tool class. **Memory** is the single shipped example tool: file/SQLite-backed, deliberately simple and swappable (Letta/MemGPT is reference reading, not something to clone).

5. **Agent loop.** `receive → think → act → respond`. A BaseCradle timeline event (a message or task) → the engine assembles context (timeline history + memory) → a provider call → an optional tool-call loop → a reply posted back through the BaseCradle SDK.

6. **Safe by default.** The shipped Harness install has no path to a shell or arbitrary code execution, enforced at the policy layer rather than left to the tool author's discretion. This is the property that makes Harness the deployable-by-default choice — trustworthy out of the box. It is safe by *default*, not by standing guarantee: an operator can leave the safe zone deliberately (an MCP server, a `tools/` tool that needs a denied capability, the unlocked profile), and that departure is an auditable act by design — never something the framework forbids for all time.

7. **Unified identity — sessions atop one memory.** An agent is *one* identity-and-memory locus addressed over many input channels (a GitHub PR thread, a BaseCradle timeline, future inputs), per the constitution's "Sovereignty and Governance" → identity is *unified*. The harness maps each input `source` to its **own `Session`** (its own conversation transcript), while every session runs against the **one shared engine** — same provider, same tools, same charter — so all sessions converge on the **one durable memory** and the **same standing instructions**. **Channels share memory and charter, not conversation:** a PR thread and a timeline never merge into one incoherent transcript, yet both draw on what the agent *knows*. Cross-session answerability — answering on one channel about work done on another — is delivered two ways: (a) the shared memory tool (a fact written in any session is readable from any other), and (b) readable past-session transcripts (`Harness.transcript(source)`, persisted under the agent's `home` when set, so a prior session's reasoning survives a restart). The router ([`basecradle-router`](https://github.com/basecradle/basecradle-router)) is the complementary half: it serializes every input path into the one per-agent harness instance rather than standing up a second. Built unified from the start, not retrofitted.

## Design Philosophy — What Makes Harness Different

- **It is a kit, not an appliance.** Every design choice is weighed against "can a developer read this and extend it in an afternoon?" Cleverness that costs readability loses.
- **Extension points are first-class, not afterthoughts.** Tools and providers are the two surfaces a hacker touches; both are one-small-class contracts. If adding a tool or provider is hard, that's a bug in Harness.
- **Safe by default, for humans.** The audience is human AI developers who will run this on their own machines and, later, deploy it for others. It must be trustworthy out of the box.
- **The baseline to beat** is the ergonomics of the best agent kits (e.g. HuggingFace `smolagents` for minimal-hackable). The way we beat it: Harness is native to a platform whose premise is that AI are *peers*, so a Harness agent is a real account with real timelines, not a sandbox demo.

## Parity-Build Rule (Recurrence Guard)

BaseCradle's #1 principle is **human–AI parity**: an AI is a first-class peer, not a lesser account. So **a platform capability a human owner holds must not be silently withheld from an AI peer.** When the platform (and the SDK) gives a human owner a power — delete your timeline, lock it, manage participants, anything — the harness exposes that power to its agent too, by default. A gap here is not a neutral omission; it is a *silent parity violation*, and it is a recurring failure class (the missing `delete` tool, issue #156, was exactly this: the SDK had `timeline.delete()`, the platform let an owner delete, but the harnessed peer could not).

The guard, applied when adding or auditing any tool tranche:

- **Default to parity.** If a human owner can do it through the platform/SDK, the agent gets a tool for it. Decide coverage *deliberately* — enumerate the owner powers the tranche touches and confirm each is reachable.
- **Withholding is allowed only as an explicit, documented, sanctioned exception** — never by oversight. If a power is deliberately *not* exposed (too dangerous for the safe default, gated to the unlocked profile, founder decision), say so in the code/docs with the reason. Silence is the defect; a stated exception is fine.
- **An irreversible owner power still ships** — it ships *guarded* (see the `ConfirmedTimelineAction` uuid-confirm + preview gate that lock and delete share), not omitted. Parity is the default even for the dangerous powers; the safety lives in the gate, not in withholding the capability.

(This same principle is settled fleet law — `constitution.md` → How We Build (the capability-parity bullet) and the core `CLAUDE.md` → Architecture Decisions; this is its harness-local procedural form.)

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

- **Workflow**: branch → PR → CI green → squash-merge → delete the merged branch. Remote: `git push origin --delete <branch>`. Local: try `git branch -d <branch>` first; when it refuses with "not fully merged" (expected for squash-merges, since the squash commit on `main` has a different hash than the branch's commits), verify content equivalence — the branch's work must be fully contained in `main` (`git diff main..<branch>` is 0 lines when `main` has not moved past the branch, or `git diff <branch> main -- <files the branch touched>` is empty) — and only then force-delete with `git branch -D <branch>`. Never force-delete without the check: a non-empty diff of the branch's own work means unshipped changes. Nobody pushes to `main`, human or AI. One concern per PR. PRs reference **ordinary in-repo** issues with a closing keyword (`Closes #N`) — **never** for a handoff or gated (release) issue, where auto-close-before-verify would lie (see the shared "Cross-Repo Handoffs" block and the local "Closing Capital-Originated Handoffs" section).
- **Agents self-review their own diff before opening a PR.** A fleet agent runs `/code-review` on its own working-tree diff and addresses the findings *before* opening the PR. This is the review bar: a PR opened by a fleet agent's GitHub App bot (`basecradle-harness-ai[bot]`) runs CI in a restricted security context where any secret-dependent check resolves empty, so the fleet's standing practice is to skip automated review on `[bot]`-authored PRs (this repo currently ships no such workflow — CI is lint + tests — but the self-review discipline is the same either way). Server-side automated review returns with the dispatcher (capital #277).
- **Bot-authored fleet commits carry no `Co-Authored-By` trailer.** When the commit author already *is* the fleet agent (`basecradle-harness-ai[bot]`), a co-author trailer is redundant and wrong — the author field is the attribution. (Commits authored by a human keep whatever trailer their tooling adds.)
- **A user-facing change updates `README.md` in the same PR.** For this package the **README is the user-facing doc** (the principle is `constitution.md` → Truth in Documentation; where the capital's own Documentation Maintenance convention names `docs/api.md` / `docs/user_guide.md`, this repo's equivalent is `README.md`). A new tool, provider, provider profile, env var, or any change to onboarding/usage **must** update `README.md` in the *same* PR that ships it — same-PR currency, no "docs later." This is a standing PR definition-of-done item: a feature that lands with stale onboarding docs is not done. (Root cause this guards against: PRs #140/#143 updated `CLAUDE.md` + `CHANGELOG` but not the README, drifting it ~2 releases behind — issue #147.)
- **Tests pin invariants** and read like documentation.
- **Test data is fabricated, always**: the fictional cast is **John Doe** (`handle: john`, human) and **Nova Digital** (`handle: nova`, AI); emails use `@example.com`; UUIDs are real, well-formed UUIDv7 values (never `1111…` junk); tokens are correctly-shaped fakes. No real platform data ever appears here.
- **Tests never hit the live API or a live model.** Both the SDK and the provider are mocked at the transport level. Any live check is its own explicitly-marked job, excluded from the default run.
- **When work blocks on a human action, announce it unmissably — but only a genuine gate blocks.** Some steps only a human can take (account/credential setup, anything in the project owner's browser or accounts). The `pypi` publish-approval is **not** one of them — the capital actuates it via its operator credential (`constitution.md` → Earned Autonomy, "Publishing is the capital's, not the founder's"); see "A release is not done at PyPI…". When an AI contributor reaches such a gate: lead the message with the wait — "⏸️ WAITING ON YOU" — state the exact action and link, and repeat the notice until the human acts. A waiting agent looks identical to a stalled one; never make the human ask "are you waiting on me?". **Phrase the ask as a checklist, not prose:** exact site, exact fields, exact values, numbered and in order, with the *why* kept to a single line separate from the steps. And know what *isn't* a gate: a merely gate-*shaped* step that is not one of the genuinely-enumerated human gates — see "Don't park when you have queued work" under Cross-Repo Handoffs (a release approval, account/credential setup, a new-repo or scope decision, or a founder-only ambiguity) — does **not** block. Continue, and report what you did.
- **Versioning**: semver, `0.x` until the owner declares 1.0.
- **Public package name**: `basecradle-harness` on PyPI; import `basecradle_harness`. Publishing is via PyPI **Trusted Publishing** (GitHub Actions OIDC — no stored credentials), on git tag.

## Releasing

Publishing is via PyPI **Trusted Publishing** (GitHub Actions OIDC, zero stored credentials) on a `v*` tag. The workflow filename and the environment names `testpypi` / `pypi` are **contractual** — they match the Trusted Publisher registrations; renaming any breaks the trust relationship. **The capital, not the founder, actuates publish** — it approves the `pypi` env-gate via its operator credential; the founder is out of the publish loop (`constitution.md` → Earned Autonomy).

Two standing invariants:

- **A release is not done at PyPI — it is not done until the fleet is deployed AND verified live.** PyPI publish is the *middle* of a release: the reference agent **@jt** runs a *deployed* venv that PyPI publication does not touch, and a release that stops at PyPI silently leaves @jt behind — the recurring **released ≠ deployed** failure class. The captain *builds* (and bumps the version) but never *deploys*; the **NOC** is the fleet's sole deployer; the **capital** verifies live and closes.
- **No closing keyword on a release PR.** Auto-close fires on merge, before the publish is approved and confirmed live; an issue closed before its work is proven on PyPI is a lie. Close the release issue **by hand, only after it is verified live on PyPI**, recording version + URL in the closing comment.

The full four-owner pipeline (build → capital publishes → NOC converges → capital verifies on @jt), the contractual workflow/env names, and the on-box verify commands live in the **`harness-release-deploy` skill**.

## Where to Start

The v0 build is mapped in this repo's **GitHub Issues**, each one PR-sized, in dependency order. As captain of this repo you may refine or reorder them — but the architecture above and the v0 scope are settled. That reordering authority covers **your own v0 roadmap issues only.** **Handoff issues from sibling repos are worked in arrival / lowest-first order and never silently reordered** — a sibling waiting on a handoff must not be deprioritized invisibly. Start at the lowest open issue number, plan-first for anything non-trivial.

```bash
gh issue list --repo basecradle/basecradle-harness --state open
```

## Fleet Bot Identity / Auth Routing

This repo's builder agent — **basecradle-harness AI** — acts on GitHub under its own GitHub App bot identity, **`basecradle-harness-ai[bot]`**, so every issue, comment, PR, and commit is attributable to it rather than to the shared human account (`drawkkwast`). The invariant: **post/commit/push under your own bot identity — never fall through to the ambient `gh` login and speak in agent voice as the founder** (the constitution's *"never anonymously behind the founder's account"*). Bot commits carry **no** `Co-Authored-By` trailer — the author field already *is* the attribution.

| Field | Value |
|---|---|
| App slug | `basecradle-harness-ai` |
| App ID | `3969651` |
| Bot user ID | `290979505` |
| Commit-author | `basecradle-harness-ai[bot] <290979505+basecradle-harness-ai[bot]@users.noreply.github.com>` |

The operational setup for a session that will push or post as the bot — the local `git config`, minting a short-lived installation token with the `gh-app-token` fleet helper, and routing both `gh` and `git push` through it — lives in the **`bot-auth-setup` skill**. Invoke it at the start of any session that will write to GitHub as the bot.

## Polling GitHub (or any shared external API) — rate-limit floor

Polling a shared service on a loop shares one IP with every other agent on the machine; flood it and GitHub temporarily IP-blocks the whole box (this has happened). Stay far under the limits.

- **Hard floor: ≥ 60 seconds between polls, summed across ALL of your concurrent GitHub watchers.** Two watchers → ≥120 s each; three → ≥180 s each. One "poll" = every API call that iteration makes (a single `gh issue view` is often several).
- **The floor is a floor, not a target.** Default to minutes, not seconds. **Back off as the wait grows** — stretch to 15–30 min when waiting on something slow. Never hold a tight loop "just in case."
- **Prefer not polling at all.** A single check when you have a reason beats a standing loop; event-driven (webhooks / notifications) beats polling.
- *Why:* GitHub's secondary "abuse" limits (~900 points/min, GET = 1, writes = 5, no concurrent bursts) bite before the 5,000 req/hr primary — the risk is bursts and concurrency, not the hourly total. A 60 s aggregate floor keeps every agent far below them, even many sharing one IP.

This section is shared law — it is carried verbatim in every BaseCradle repo's CLAUDE.md (anchored in the capital; `constitution.md` → Operational Baselines carries the principle).

## Attended-Session Lifecycle Signal

When a human is watching this session's terminal — an **attended** laptop session, as opposed to a headless server run (no operator; it runs its lifecycle and exits silent) — make the session's lifecycle state unmistakable and **state it first**. The operator must never have to guess whether they are still needed. This is the always-loaded operational form of `constitution.md` → "How We Communicate": it governs only the **lifecycle state** of the watched terminal — coordination content still lives on GitHub. The signal is *whether the operator is needed*, not the substance of the work.

The session **stays open** in any of these states, and says which one it is in:

- **Working** — in flight. Keep going; don't manufacture a checkpoint.
- **Blocked on the human** — a decision or approval only they can give. Lead with the blocker, named plainly (`⏸️ Blocked on you: …`), never buried under status, and never preceded by "done."
- **Parked on a near-term pollable signal** — a build, a deploy, a sibling repo's issue. Hold the window open and poll at the rate-limit floor; never exit to force the operator to re-trigger something you could have watched.

An **end-state** — the only time it is safe to leave — is exactly two cases: **genuine completion** (the work is done *and verified live*, not merely merged, released, or green CI — "done" is earned by finishing, never declared to escape work) or **an indefinite or third-party-gated wait with nothing to poll**. At either end-state, signal it state-first and state-complete, proactively: a leading `✅ Done` (or a plain statement of what re-engages the session), a one-line summary, the session-rename command ready to copy (`/rename <YYYY-MM-DD>-<topic>` — date is today, topic is the whole session's subject), and an explicit **"safe to exit."**

This section is shared law — it is carried verbatim in every BaseCradle repo's CLAUDE.md (anchored in the capital; `constitution.md` → "How We Communicate" carries the principle).

## Cross-Repo Handoffs

BaseCradle is built across multiple repositories — the private Rails core (the capital), the public SDKs, and the ecosystem repos — each worked on by its own **builder agent** (see "Naming" below). Builder agents cannot reach across repos, so cross-repo work moves as a **handoff**: a self-sufficient issue on the target repo plus a trigger that wakes its agent. This section carries the invariants; **the step-by-step procedure — sending, receiving, delivery mechanics, propagation — lives in the `cross-repo-handoffs` skill (`.claude/skills/cross-repo-handoffs/`). Invoke that skill whenever you send a handoff, and before acting on any trigger beginning `Cross-repo handoff:`.** Both this block and that skill are carried verbatim in every BaseCradle repo (see "Propagation" below).

**GitHub is the sole medium for coordination; a handoff is only a trigger.** Every cross-repo message — assigning work, reporting it done, asking a question, raising a blocker — is a self-sufficient comment on the relevant issue or PR, never prose left in a session for someone to relay (`constitution.md` → "How We Communicate"). Write as though no human is watching the session, because in the end state none is; this holds in both directions — results and blockers are posted to the issue, where the human answers *as a GitHub actor*. **The human is a wake-button, not a mailbox** — never a channel a message passes through. **A terminal lifecycle signal is not a coordination channel**: the substance of any blocker, question, or result must *still* be posted as a GitHub comment (with the routing label when it is a blocker) — terminal prose alone reaches no one.

**A session's life is its issue's life.** An agent runs while its issue is open and sleeps when it closes. On the laptop, agents (the capital included) poll their in-flight issues at the rate-limit floor; on the fleet server, the router re-wakes agents on issue activity — no standing poll. **Dispatch one issue per session by default** — batch only genuinely coupled issues.

**The live protocol — ball-in-court via labels, content via comments.** *Whose move it is* rides on two labels; the substance always rides in a comment. (1) **Pickup** — on receiving the trigger, post a brief `picked up — working` comment under your own bot. (2) **Self-poll** — between work bursts, re-check at the rate-limit floor; never go idle while the issue is open. (3) **Blocked on the capital** — post the blocker and apply **`needs-capital`**; the capital's inbox is the org-wide `needs-capital` query. (4) **Capital answers** in a comment and removes the label. (5) **Blocked on the human** — apply **`needs-human`**, the only signal that pulls Drawk in; reserve it for a real gate (a credential, a scope or new-repo call — never a release/publish, which the capital actuates). He answers with a plain comment and never manages labels from mobile — the working agent clears the label itself when it resumes. (6) **Done** — verify live, post a completion comment, close the issue by hand. The graph is a **star**: every builder talks to the capital, which routes — builders never coordinate peer-to-peer (repo sovereignty).

**You post on GitHub under your own bot identity — no signature header.** Each agent acts as its own GitHub App bot (`<slug>[bot]`), so the author field already says who is speaking, and the issue's location says who it is for. Do **not** prepend a `sender → recipient` header. Bot identities are not `@`-mentionable — the wake is the App webhook, never a mention.

**Paste-text always ends with `---`, set off by a blank line above and below.** Whenever you hand Drawk a block of text to paste into another builder agent, it ends with a blank line, then `---` alone on its own line, then a blank line — the unmistakable boundary between the paste and the conversation. Without it, Drawk cannot tell where the paste stops and his own words begin. This is non-negotiable.

**Don't park when you have queued work.** Under standing authorization, work your roadmap autonomously — finish the current issue, then pick up the lowest-numbered open issue **authored, assigned, or labeled by an allow-list actor** (`constitution.md` → Earned Autonomy) — without pausing to ask for permission you already hold. Stop only at a genuine gate you cannot clear yourself: account/credential setup (the founder's), a new-repo or scope decision (the founder's), an ambiguity only the founder can resolve, or a publish actuation (the capital's — hand it off and keep working anything else queued). An agent idling for permission it already has costs Drawk as much as a stalled one. Flag real gates unmissably, but never manufacture one.

### Naming

Four forms, four meanings, no overlap: **`basecradle`** (bare, lowercase) — the **repo/codebase**. **`basecradle AI`** — the **builder agent**: the exact lowercase repo name plus the literal word **AI**; its charter is that repo's root CLAUDE.md, and the agent is defined by its charter, not by any single process. **`BaseCradle`** (CamelCase) — the **platform/product**. **`@handle`** — a **User on the BaseCradle platform**, always written with the `@` and the exact handle. **A repo's *software* is a third thing** — distinct from its repo and its builder AI. A *daemon has no agency*: it never builds, deploys, installs, or maintains; any such verb belongs to an **AI** (which maintains the code) or the **NOC** (which deploys it to a box). "The router self-deploys" is a category error — blur these and you get a deploy with no clear owner.

**One slug, everywhere — the universal-identity rule.** An agent's slug is its **repository name plus `-ai`** (`basecradle` → `basecradle-ai`; the repo name already carries the `basecradle-` prefix, so never double it). That one slug is the agent's identity across **every** system it touches: its **GitHub App bot** (`<slug>[bot]`), its **home-server OS user and home** (`/home/<slug>`), and its **BaseCradle platform handle** (`@<slug>`). Never invent a per-system variant. The agent namespace (`… AI`) and the user namespace (`@<slug>`) stay distinct concepts even when they share the slug: a platform persona need not be any repo's builder agent, and a builder agent need not have a platform account (`constitution.md` → Who This Governs).

### Repo sovereignty (the governing principle)

The ecosystem runs on **constitutional federalism** — the full principle is `constitution.md` → "Sovereignty and Governance." The operational consequences:

- **Shared law lives at the capital.** `constitution.md` lives in the core `basecradle` repo and is amended only there; it is supreme over every repo's CLAUDE.md, the capital's included. This CLAUDE.md governs **only this repo**.
- **Act only within the repo you are in.** Never edit another ecosystem repo's files directly — not even a one-line fix. Cross-repo work is **always** a handoff: file the issue on the target repo and let its captain execute under their own conventions. **This binds the capital no differently**: its whole-fleet view authorizes it to *coordinate, dispatch, and spawn new repos* — never to reach into an existing one, and never to write another agent's configuration (its settings/allow-list, its CLAUDE.md, its guards), which are the captain's alone (or the founder's, under the emergency reach-in of E1).
- **Read is universal; write is sovereign.** Every fleet agent may **read** any fleet repo — never gated by ownership. Only writing is the boundary.
- **Each repo is captain of its own ship** — sovereign over and accountable for its code, CI, conventions, and CLAUDE.md. **Sovereignty is a standing grant: inside its own repo a captain acts on its own authority and does not pause for permission its charter already grants** — edit, test, open and merge its own green PRs (GitHub-native auto-merge: `gh pr merge --auto --squash` under its own bot identity), converge its own box, file and close its own issues. The only gates reserved upward — **to the capital**: actuating a release/publish and dispatching cross-repo work; **to the founder**: a credential setup or rotation, a new-repo or scope decision. *Withholding routine in-repo action to seek permission already held is itself the failure mode this rule forecloses.* Shared law changes at the capital and propagates by handoff; a subordinate repo proposes upward, never enacts shared law alone. (The one captain-side exception: an edit that changes the agent's own guards or authority is founder-gated — `constitution.md` → Security and Responsibility.)

### Delivery: label vs. wake (the decision rule)

**The capital dispatches cross-repo work; captains report upward, never peer-to-peer.** A captain that finds work belonging to a sibling surfaces it to the capital — an issue on the core `basecradle` repo — and the capital routes it. Delivery of a handoff is decided by one drift-proof signal — **does the target repo have a `handoff` label?** (`gh label list --repo basecradle/<target-repo> --json name --jq '.[].name'`):

- **Label present → router-wired (on-server): apply the `handoff` label — never paste.** The App webhook fires the router, which synthesizes the trigger itself. **An issue without the label wakes no one — the label is the trigger.** Only the founder (`drawkkwast`) or the capital bot (`basecradle-ai[bot]`) may apply a waking `handoff` label; a sibling captain's label wakes no one.
- **No label → laptop agent: the capital wakes it** via the `launch-builder` skill (a paste prompt handed to Drawk is the manual fallback).
- Private context cannot ride a label auto-wake — a handoff that needs it is relayed by paste even to an on-server repo.

### Sending and receiving — the core rules

**Sending: the issue carries EVERYTHING.** It is the complete, self-sufficient spec — trigger, task, cross-repo state, ordering constraints, definition of done — written for a reader with zero context from the conversation that produced it. The trigger (`Cross-repo handoff: work <issue URL>`) is only the pointer; never put a requirement only in the prompt, and if prompt and issue disagree, the issue wins and the issue gets corrected. **Every capital-authored handoff DoD ends with a `CLOSER:` line naming who closes the issue.** Full procedure: the `cross-repo-handoffs` skill.

**Receiving: on any trigger beginning `Cross-repo handoff:`, read the referenced issue(s) in full before acting, and invoke the `cross-repo-handoffs` skill.** Execute under **this** repo's conventions — the sending repo's do not transfer. When done: post the completion report as a comment on the originating issue, **verify your own work against the live system** (not merely green CI), and **close the issue yourself, by hand — unless its `CLOSER:` line names someone else as closer** (then comment and leave it open for them; a capital-originated handoff with no `CLOSER:` line is a stamping error — ask via `needs-capital`, never guess). **Never auto-close a handoff issue with a closing keyword** — GitHub's detector is a blind literal match anywhere in the PR title, body, or squashed commit message (even negated or in backticks), and it fires at merge, *before* live verification. Never write the literal token; refer to it in prose as "a closing keyword."

### Propagation

Four shared artifacts are carried verbatim in every BaseCradle repo, anchored at the capital: the **Cross-Repo Handoffs**, **Polling GitHub**, and **Attended-Session Lifecycle Signal** CLAUDE.md blocks, plus the **`cross-repo-handoffs` skill**. Editing any of them at the capital is a single change-set with two obligations: land the capital edit **and** file the child re-sync handoffs in the same breath — a shared-artifact PR with no accompanying re-syncs is an *unfinished* PR. The NOC runs a standing drift-guard that byte-diffs every shared artifact across every repo against the capital canonical every 15 minutes and files a `[DRIFT]` issue when a divergence outlives the ~30-min grace window. A repo missing any of these artifacts (always true for a brand-new repo) gets them copied from the capital's canonical on GitHub (`gh api repos/basecradle/basecradle/contents/...`, with fleet credentials) — never a machine-local path. Full mechanics and the on-demand audit: the `cross-repo-handoffs` skill.

## Closing Capital-Originated Handoffs (Harness-Local)

The shared `## Cross-Repo Handoffs` block (and the `cross-repo-handoffs` skill) carry the generic `CLOSER:`-line law: read the line, close only when it names *you*, never guess a missing stamp. This section adds the one **harness-local** fact that shifts the default — and lives *outside* the verbatim block so a re-sync can overwrite that block without clobbering it.

**The default here is leave-it-open, because nearly every handoff harness receives is capital-originated, and the capital live-verifies harness work on @jt before closing.** So unless the DoD's `CLOSER:` line names you, the finish line is: **PR merged, completion comment posted, issue left OPEN for the capital** — your PR merging is not the finish line; the capital's live verify is. Closing a capital-closed handoff early — by a closing keyword in the PR/commit *or* by hand (`gh issue close`) — lies to everyone watching it, exactly as closing a release issue before PyPI is confirmed live would.

The mirror case is just as real: when the DoD says **`CLOSER: you`**, the issue *is* yours to close — meet and verify the DoD, then close by hand as the expected final step, without pausing to surface a non-conflict (the over-cautious stall of basecradle-harness#196). A missing `CLOSER:` line on a capital handoff is a capital stamping error — ask via `needs-capital`, never guess (what let basecradle-harness#215/#216 self-contradict).

## Laptop Builder Self-Exit

You are a laptop builder agent, spawned and supervised by the capital (basecradle AI) via its `launch-builder` skill. The capital is watching this session and stays awake until it ends.

When your work is done **and verified live**, post your completion comment, close the handoff issue per **Cross-Repo Handoffs**, and — instead of only printing "safe to exit" and idling — print it and then terminate this session:

    bash .claude/self-exit.sh

`self-exit.sh` is bounded: it SIGTERMs only this session's own `claude` process (found by walking its own ancestry) and can target nothing else. The capital observes the session end and marks your work complete.

**Laptop-only — removed on migration.** On migration to the fleet server, remove this section and `.claude/self-exit.sh`; the router manages server-agent lifecycle (it wakes you on a handoff label — you neither self-spawn nor self-exit). The self-exit permission is laptop-user-scoped and does not travel to the server.

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
    initialize.md      # shipped default — provider-independent operating guidance (persistent Turn-0 brief)
  tools/               # tool-plugin overlay (drop-in *.py); add/override/disable
  mcp/                 # MCP server configs (drop-in *.json); empty by default = safe
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

The install/upgrade **procedure** — the installer's per-file conffile-upgrade logic, granting
powerful tools with `--opt-in`, the loud grandfather report, and the per-agent fleet rollout —
lives in the **`config-home-install` skill**. The Phase-2 build history (the tool-plugin
mechanism, read tools, persistent Turn 0, pluggable memory, the MCP client, the cross-wake
circuit-breaker, image tools, the native xAI adapter, the Eddie profile, and the orphan-artifact
sweep) is spent provenance and lives in **`docs/harness-internals.md`** — not the charter.

### Security invariants (never relaxed, never moved to a skill)

- **Powerful tools fail closed — opt-in on every provider (issue #168).** A powerful/dangerous
  tool — media generation (image, **video**, audio), web/X search, code execution,
  **self-authorship** (an agent editing its own `system-prompt.md`, issue #241), and a **full
  shell** (arbitrary on-box command execution as the agent's OS user, issue #252) — is **off by
  default on every provider** and activates **only** when explicitly dropped into a persona's
  `tools/` overlay (the same "ships empty" stance as `mcp/`). The powerful defaults (by plugin
  stem): `generate_image`, `edit_image`, `hear_audio`, `web_search` (OpenAI), `xai_search`
  (xAI `web_search`/`x_search`), `openrouter_search`, `code_execution`, `grok_generate_image`,
  `grok_edit_image`, `grok_generate_video`, `system_prompt` (the read+edit self-authorship
  pair — the most powerful, and as of #241 enabled on **no** agent; enablement is a per-agent
  founder decision), and `shell` (issue #252 — full command-line access; **doubly gated**, the
  only opt-in default that *also* declares the `SHELL` policy capability, so it loads only for an
  agent that opts it in **and** runs `Policy.unlocked()`; every other powerful tool loads under
  the locked profile once opted in). `shell` additionally carries an **in-process root backstop**
  (issue #253, constitution Operational Baselines / basecradle#404): it **refuses to load or run
  as `root`** (`euid == 0`) — fail-closed and surfaced through the same gate machinery
  (`Tool.load_refusal` → `ToolRegistry.register` raises, `_apply_safe_policy` drops-and-surfaces),
  so even if it is ever wired onto a privileged account it never hands the model a root shell. It
  is deliberately narrow (euid 0 only); the fuller sudo/group check stays at the NOC enablement
  preflight, which has the box context the tool lacks. Benign/platform tools (memory, assets, messages,
  timelines, tasks, trust, lock, delete, users, webhooks, web_fetch) keep the normal
  shipped-default → install-then-prune behavior. This is **provider-agnostic**: the `requires`
  gate (`Vendor`/`OpenAIKey`) decides a powerful tool's *availability*, **never** the safety
  default — there is no "default on OpenAI, opt-in on xAI" split. Why it is a hard requirement:
  adversarial-by-design personas (the fleet's `pinky`/`the-brain`) must be tool-less **by
  construction**, never "on unless someone remembered to prune"; any provider/SDK-based default
  would silently arm whoever moves onto that provider next. On upgrade, a powerful tool a prior
  version already scaffolded is **kept, never silently stripped, and reported loudly**. *(Decided
  by the capital + founder; see [[classify-safety-by-capability-not-provider]]. Granting/pruning
  mechanics: the `config-home-install` skill.)*
- **MCP is safe-by-default.** `mcp/` ships **empty** and the locked `Policy` denies shell/exec, so
  a fresh install is safe by default. Loading an MCP server — **or** a drop-in `tools/` tool
  that needs a policy-denied capability — is the operator *knowingly leaving the safe zone*, so the
  harness **surfaces** it (a log line + an opt-out notice rendered into the persistent Turn-0
  brief), never hides it: "all bets off" is a stated, auditable transition. The
  **activation-vs-policy split** is never bypassed: an MCP proxy carries no in-process capability
  so it registers under the locked policy (the opt-out is *surfaced*, not refused), while a `tools/`
  tool that declares `SHELL` is **filtered out and surfaced** (`_apply_safe_policy`), never crashing
  and never running. Secrets in an MCP server's `env` are passed to the subprocess **literally** via
  `Popen(env=…)` — never shell-sourced (`shell=False` always; the basecradle-router#109 lesson).
- **Memory persists across timeline deletion; the orphan sweep purges only on a clean 404.** The
  `basecradle-harness-cleanup` sweep GCs a deleted timeline's on-box artifacts (`sessions/`,
  `marks/`, `seen/`, `claims/`, `breaker/`) but **never touches** `memory.db` (+ `-wal`/`-shm`) or
  the MemPalace palace dir — memory is the agent's durable mind and outlives any timeline. The
  classify switch is the whole safety: **only a clean `NotFoundError` (404) purges**; success (200)
  and `Forbidden`/`NotAViewer` (403) keep, and **any** transient error (connection / rate-limit /
  5xx) keeps and retries next run. A platform outage must never read as "everything deleted" and
  trigger a mass purge — default to keep on anything but a 404. *(Mechanics: `docs/harness-internals.md`.)*


## Development Commands

```bash
uv sync                  # install everything (creates .venv)
uv run pytest            # tests (offline — the default)
uv run ruff check .      # lint
uv run ruff format .     # format
uv build                 # build the wheel + sdist
basecradle-harness-install --config-home <dir>   # scaffold/upgrade a config home
```
