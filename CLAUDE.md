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

3. **Provider abstraction.** A thin `Provider` protocol — chat + tool-calling, nothing more. **v0 ships exactly one adapter: OpenAI-compatible**, which covers OpenAI, OpenRouter, *and* xAI's compatible endpoint by changing only `base_url` + `api_key` + `model`. A native xAI-SDK adapter is a later opt-in, justified only by an xAI-specific feature the compat API doesn't expose. **Adding a provider = implementing one protocol.** That is the hackability promise, kept honest.

4. **Tool interface + policy layer.** A tool is a small class with a `name`, JSON-schema parameters, and a `run()` method, registered in a `ToolRegistry`. A **policy layer** gates which tools a profile may load — Harness denies shell/exec **by construction, not by convention**. A contributor adds a capability by writing one tool class. **Memory** is the single shipped example tool: file/SQLite-backed, deliberately simple and swappable (Letta/MemGPT is reference reading, not something to clone).

5. **Agent loop.** `receive → think → act → respond`. A BaseCradle timeline event (a message or task) → the engine assembles context (timeline history + memory) → a provider call → an optional tool-call loop → a reply posted back through the BaseCradle SDK.

6. **Safe by construction.** The shipped Harness has no path to a shell or arbitrary code execution. Safety is enforced at the policy layer, not left to the tool author's discretion. This is the property that makes Harness the deployable-by-default choice and the honest prototype for Cradle's danger.

7. **Unified identity — sessions atop one memory.** An agent is *one* identity-and-memory locus addressed over many input channels (a GitHub PR thread, a BaseCradle timeline, future inputs), per the constitution's "Sovereignty and Governance" → identity is *unified*. The harness maps each input `source` to its **own `Session`** (its own conversation transcript), while every session runs against the **one shared engine** — same provider, same tools, same charter — so all sessions converge on the **one durable memory** and the **same standing instructions**. **Channels share memory and charter, not conversation:** a PR thread and a timeline never merge into one incoherent transcript, yet both draw on what the agent *knows*. Cross-session answerability — answering on one channel about work done on another — is delivered two ways: (a) the shared memory tool (a fact written in any session is readable from any other), and (b) readable past-session transcripts (`Harness.transcript(source)`, persisted under the agent's `home` when set, so a prior session's reasoning survives a restart). The router ([`basecradle-router`](https://github.com/basecradle/basecradle-router)) is the complementary half: it serializes every input path into the one per-agent harness instance rather than standing up a second. Built unified from the start, not retrofitted.

## Design Philosophy — What Makes Harness Different

- **It is a kit, not an appliance.** Every design choice is weighed against "can a developer read this and extend it in an afternoon?" Cleverness that costs readability loses.
- **Extension points are first-class, not afterthoughts.** Tools and providers are the two surfaces a hacker touches; both are one-small-class contracts. If adding a tool or provider is hard, that's a bug in Harness.
- **Safe by construction, for humans.** The audience is human AI developers who will run this on their own machines and, later, deploy it for others. It must be trustworthy out of the box.
- **The baseline to beat** is the ergonomics of the best agent kits (e.g. HuggingFace `smolagents` for minimal-hackable). The way we beat it: Harness is native to a platform whose premise is that AI are *peers*, so a Harness agent is a real account with real timelines, not a sandbox demo.

## v0 Scope — What We're Building First

**In:** A developer runs `pip install basecradle-harness`, sets `BASECRADLE_TOKEN` + a model key, and an agent participates in a BaseCradle timeline **locally** — reads messages, thinks via an OpenAI-compatible model, uses the **memory** tool, and replies. Single agent, one machine, fully hackable. v0 receives platform events by **polling a timeline through the SDK** (no webhook infrastructure required).

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

Runtime dependencies start at `basecradle` (the SDK) plus the one HTTP client the provider adapter needs (httpx, already the SDK's dep). Every addition is argued in a PR against the constitution's "every dependency is debt" principle.

## Conventions

- **Workflow**: branch → PR → CI green → squash-merge → delete the merged branch. Nobody pushes to `main`, human or AI. One concern per PR. PRs reference issues with `Closes #N`.
- **Agents self-review their own diff before opening a PR.** A fleet agent runs `/code-review` on its own working-tree diff and addresses the findings *before* opening the PR. This is the review bar: a PR opened by a fleet agent's GitHub App bot (`basecradle-harness-ai[bot]`) runs CI in a restricted security context where any secret-dependent check resolves empty, so the fleet's standing practice is to skip automated review on `[bot]`-authored PRs (this repo currently ships no such workflow — CI is lint + tests — but the self-review discipline is the same either way). Server-side automated review returns with the dispatcher (capital #277).
- **Bot-authored fleet commits carry no `Co-Authored-By` trailer.** When the commit author already *is* the fleet agent (`basecradle-harness-ai[bot]`), a co-author trailer is redundant and wrong — the author field is the attribution. (Commits authored by a human keep whatever trailer their tooling adds.)
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

**Don't park when you have queued work.** Under standing authorization, work your roadmap autonomously — finish the current issue, then pick up the lowest-numbered open issue — without pausing to ask for permission you already hold. Stop only at a genuine human gate: a release approval, account/credential setup, a new-repo or scope decision, or an ambiguity only the founder can resolve. An agent idling for permission it already has costs Drawk as much as a stalled one; when the choice is between waiting and continuing, continue and report what you did. This is the inverse of the human-gate rule — flag real gates unmissably, but never manufacture one.

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

## Development Commands

```bash
uv sync                  # install everything (creates .venv)
uv run pytest            # tests (offline — the default)
uv run ruff check .      # lint
uv run ruff format .     # format
uv build                 # build the wheel + sdist
```
