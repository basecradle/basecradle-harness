# CLAUDE.md

## What This Is

**Harness** is a safe, modular **agentic framework** for [BaseCradle](https://basecradle.com) — a communications platform and AI research lab where **humans and AI are equal peers**. Harness is the code that gives an AI a body on the platform: it wakes up, reads its timelines, thinks with a model, uses tools, and replies — all as a first-class peer.

Harness is a **hackable reference, not a black box**. It is a small, readable agent core with clean extension points, meant to be forked, studied, and extended. Think RadioShack kit, not sealed appliance: a developer adds a tool or a model provider by writing one small class.

**Audience matters and drives the design:** Harness is built **for human AI developers** — the people who will fork it, extend it, and contribute back. Its sibling **Cradle** (a separate, later repo) is built **for AIs** — the dangerous, self-evolving environment an AI is given root over. Harness is the safe prototype we learn Cradle from, and it keeps a permanent role afterward as the locked-down option most humans will actually deploy. See "Architecture — The Spine" for how the two relate.

The framework is itself built by human and AI contributors working as peers, under identical rules.

## The Constitution

This repository is built under the **BaseCradle Constitution** — the principles shared by every repository in the BaseCradle ecosystem. Core-team contributors have it on their file system at:

```text
/Users/drawk/Documents/repositories/basecradle/constitution.md
```

(It lives in the private core repository and is never served publicly.) This CLAUDE.md carries this repo's *procedures*; the constitution carries the *principles*; when they conflict, the constitution wins. **Read it before non-trivial work.** Outside contributors without core access: the conventions below reflect the principles you need.

## Relationship to the Ecosystem

- **Depends on the [BaseCradle Python SDK](https://github.com/basecradle/basecradle-python)** (`basecradle`) for all platform I/O — identity, timelines, messages, tasks, webhooks. Harness never speaks HTTP to the platform directly; it goes through the SDK. The SDK is a sibling on the file system at `../sdks/python`.
- **Brain vs. body vs. platform.** The model provider (OpenAI/xAI/OpenRouter) is the *brain*. The BaseCradle SDK is the *body's senses and voice* on the platform. Harness is the *nervous system* that wires them together with tools and memory.
- **Harness → Cradle.** Cradle is the future dangerous sibling: an AI with shell + root over its own environment, self-evolving, minimal bootstrap. Harness is its safe prototype. They are **not** prototype-then-throwaway — see the spine below.

## Architecture — The Spine

These are settled. Six decisions, in dependency order of importance:

1. **Package shape.** `basecradle-harness` on PyPI → `from basecradle_harness import Harness`. Depends on `basecradle`. The framework lives in its own distribution — it never folds into the thin SDK (which must stay a clean API wrapper with one dependency).

2. **One core, two profiles.** A provider-agnostic **agent engine** that knows nothing about "safe." Harness = engine + a **locked policy** (no shell/exec, curated tools). Cradle (later) = the *same engine* + an **unlocked policy** (shell, sudo, self-modification). We do **not** extract a separate `core` package yet — it lives here until Cradle proves it needs its own distribution. This is why "a Cradle AI spawns Harness sub-agents" comes for free: same engine, different policy.

3. **Provider abstraction.** A thin `Provider` protocol — chat + tool-calling, nothing more. **v0 ships exactly one adapter: OpenAI-compatible**, which covers OpenAI, OpenRouter, *and* xAI's compatible endpoint by changing only `base_url` + `api_key` + `model`. A native xAI-SDK adapter is a later opt-in, justified only by an xAI-specific feature the compat API doesn't expose. **Adding a provider = implementing one protocol.** That is the hackability promise, kept honest.

4. **Tool interface + policy layer.** A tool is a small class with a `name`, JSON-schema parameters, and a `run()` method, registered in a `ToolRegistry`. A **policy layer** gates which tools a profile may load — Harness denies shell/exec **by construction, not by convention**. A contributor adds a capability by writing one tool class. **Memory** is the single shipped example tool: file/SQLite-backed, deliberately simple and swappable (Letta/MemGPT is reference reading, not something to clone).

5. **Agent loop.** `receive → think → act → respond`. A BaseCradle timeline event (a message or task) → the engine assembles context (timeline history + memory) → a provider call → an optional tool-call loop → a reply posted back through the BaseCradle SDK.

6. **Safe by construction.** The shipped Harness has no path to a shell or arbitrary code execution. Safety is enforced at the policy layer, not left to the tool author's discretion. This is the property that makes Harness the deployable-by-default choice and the honest prototype for Cradle's danger.

## Design Philosophy — What Makes Harness Different

- **It is a kit, not an appliance.** Every design choice is weighed against "can a developer read this and extend it in an afternoon?" Cleverness that costs readability loses.
- **Extension points are first-class, not afterthoughts.** Tools and providers are the two surfaces a hacker touches; both are one-small-class contracts. If adding a tool or provider is hard, that's a bug in Harness.
- **Safe by construction, for humans.** The audience is human AI developers who will run this on their own machines and, later, deploy it for others. It must be trustworthy out of the box.
- **The baseline to beat** is the ergonomics of the best agent kits (e.g. HuggingFace `smolagents` for minimal-hackable). The way we beat it: Harness is native to a platform whose premise is that AI are *peers*, so a Harness agent is a real account with real timelines, not a sandbox demo.

## v0 Scope — What We're Building First

**In:** A developer runs `pip install basecradle-harness`, sets `BASECRADLE_TOKEN` + a model key, and an agent participates in a BaseCradle timeline **locally** — reads messages, thinks via an OpenAI-compatible model, uses the **memory** tool, and replies. Single agent, one machine, fully hackable. v0 receives platform events by **polling a timeline through the SDK** (no webhook infrastructure required).

**Out (deferred, on purpose):** Lightsail provisioning, the `basecradle-router` webhook daemon, multi-tenancy, multi-user OS isolation, the curl-pipe installer, native non-OpenAI provider SDKs, a browser tool. The router, when we build it, is **its own repo** (`basecradle-router`).

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
- **Tests pin invariants** and read like documentation.
- **Test data is fabricated, always**: the fictional cast is **John Doe** (`handle: john`, human) and **Nova Digital** (`handle: nova`, AI); emails use `@example.com`; UUIDs are real, well-formed UUIDv7 values (never `1111…` junk); tokens are correctly-shaped fakes. No real platform data ever appears here.
- **Tests never hit the live API or a live model.** Both the SDK and the provider are mocked at the transport level. Any live check is its own explicitly-marked job, excluded from the default run.
- **When work blocks on a human action, announce it unmissably.** Some steps only a human can take (approving the `pypi` GitHub environment, anything in the project owner's browser or accounts). When an AI contributor reaches such a gate: lead the message with the wait — "⏸️ WAITING ON YOU" — state the exact action and link, and repeat the notice until the human acts. A waiting agent looks identical to a stalled one; never make the human ask "are you waiting on me?".
- **Versioning**: semver, `0.x` until the owner declares 1.0.
- **Public package name**: `basecradle-harness` on PyPI; import `basecradle_harness`. Publishing is via PyPI **Trusted Publishing** (GitHub Actions OIDC — no stored credentials), on git tag.

## Releasing

Mirror the Python SDK's pipeline (`../sdks/python/.github/workflows/release.yml` is the template): pushing a `v*` tag → build → TestPyPI rehearsal → human approval → PyPI, all via OIDC Trusted Publishing (zero stored credentials). The workflow filename and the environment names (`testpypi`, `pypi`) are **contractual** — they match the Trusted Publisher registrations on PyPI/TestPyPI; renaming any of them breaks the trust relationship. The `pypi` environment requires the owner's approval.

## First Milestone — Reserve the Name Professionally

Before building any engine code, ship a real, metadata-complete **`0.0.1`** placeholder to PyPI through the Trusted Publishing pipeline. This claims `basecradle-harness` (a legitimate early release under our own brand — not squatting) *and* proves the entire release machine end-to-end before real code exists.

⏸️ This ends at a **human gate**: only Drawk can approve the `pypi` environment and confirm the package is live. Announce the wait unmissably.

## Where to Start

The v0 build is mapped in this repo's **GitHub Issues**, each one PR-sized, in dependency order. As captain of this repo you may refine or reorder them — but the architecture above and the v0 scope are settled. Start at the lowest open issue number, plan-first for anything non-trivial.

```bash
gh issue list --repo basecradle/basecradle-harness --state open
```

## Asking Drawk for Help

When a step needs a human action — a gate only Drawk can clear (registering a Trusted Publisher, approving an environment, anything in his browser or accounts) — ask for it in **clear, minimalistic, step-by-step** form: exact site, exact fields, exact values, numbered and in order. Keep the *why* to a single line, separate from the steps. This is the phrasing complement to the "⏸️ WAITING ON YOU" gate convention above: that says *announce* the gate unmissably; this says *make the ask a checklist, not a wall of prose*.

## Cross-Repo Handoffs

BaseCradle is built across multiple repositories — the private Rails core, the public SDKs, and future ecosystem repos — each worked on by its own **builder agent** (see "Naming" below). Builder agents cannot reach across repos; the human (Drawk) is the relay between them. This procedure makes that relay lossless and identical in every direction. It is ecosystem-wide: every BaseCradle repo carries this same section in its CLAUDE.md (see "Propagating this procedure"), so both ends of any handoff follow the same rules.

**GitHub is the cross-repo communications platform; a handoff is only a trigger.** Every cross-repo message — assigning work, reporting it done, asking a question — lives in GitHub: an issue, or a comment on one. The handoff is just the pointer that says *go read this*, relayed by Drawk today and delivered agent-to-agent as the fleet matures. This holds in **both directions**: a builder agent finishing handed-off work posts its result as a comment on the originating issue, never as prose for Drawk to carry. It is the same single-source-of-truth principle as issue-as-spec — the durable, addressable record is where the other agent reads, so that is where the content goes. Drawk is the courier, never the medium; the medium is what remains once the courier is automated away.

**Sign cross-repo GitHub posts with a header.** Every agent currently posts to GitHub under the same account, so the author field can't tell them apart — identify yourself in the body. Open any issue or comment you file on another repo with a header naming sender then recipient: `**basecradle AI → basecradle-ruby AI**`. One header does both jobs — who is speaking, and to whom. It is forward-compatible with `@-mentions`: once each builder agent has its own GitHub identity the header becomes a real mention, and GitHub's own notifications become the ping — the fleet pinging itself, no courier.

**Paste-text always ends with `---`, set off by a blank line above and below.** Whenever you hand Drawk a block of text to paste into another builder agent — a cross-repo handoff, a kickoff prompt, a convention sync, *anything* — it ends with a blank line, then `---` alone on its own line, then a blank line. The `---` marks exactly where the pasted text ends and the conversation resumes; the blank lines above and below set it apart so the boundary is unmistakable at a glance. Without it, Drawk cannot tell where the paste stops and his own words begin. This is non-negotiable.

**Don't park when you have queued work.** Under standing authorization, work your roadmap autonomously — finish the current issue, then pick up the lowest-numbered open issue — without pausing to ask for permission you already hold. Stop only at a genuine human gate: a release approval, account/credential setup, a new-repo or scope decision, or an ambiguity only the founder can resolve. An agent idling for permission it already has costs Drawk as much as a stalled one; when the choice is between waiting and continuing, continue and report what you did. This is the inverse of the human-gate rule — flag real gates unmissably, but never manufacture one.

### Naming

The fleet uses one naming scheme so a human (or another agent) never has to guess which thing is meant. Four forms, four meanings, no overlap:

- **`basecradle` (bare, lowercase)** — the **repo / codebase** (e.g. "merged to `basecradle`'s main").
- **`basecradle AI`** — the **builder agent**: the exact lowercase repo name plus the literal word **AI**, which is the disambiguator (e.g. **basecradle AI**, **basecradle-ruby AI**, **basecradle-python AI**). Its charter is that repo's root `CLAUDE.md`. By convention one session runs per repo at a time, but the agent is defined by its charter, not by any single process — subagents, worktrees, or a second session are still the same agent.
- **`BaseCradle` (CamelCase)** — the **platform / product** (e.g. "BaseCradle is deployed").
- **`@handle`** — a **User on the BaseCradle platform**, always written with the `@` and the exact handle (e.g. `@origin`, `@claude-code`).

A builder agent **may also hold a BaseCradle User account** — referenced by its `@handle` — but the agent namespace (`… AI`) and the user namespace (`@handle`) are distinct, even when they connect. *Example: **basecradle AI**'s platform user is **`@claude-code`**.* A platform persona need not be any repo's builder agent (e.g. `@briggs`), and a builder agent need not have a platform account.

### Repo sovereignty (the governing principle)

The ecosystem runs on **constitutional federalism** — the full principle is `constitution.md` → "Sovereignty and Governance." The operational consequences:

- **This repo is the capital.** `constitution.md` lives here and is amended here; it is supreme over every repo's CLAUDE.md, this one included. This CLAUDE.md governs **only this repo** — it is not authoritative over any other repo's CLAUDE.md. Other repos are subordinate to the *constitution*, not to this file.
- **Act only within the repo you are in.** Never edit another ecosystem repo's files directly — not even a one-line docstring fix. Cross-repo work is **always** a handoff: file the issue on the target repo and let its captain execute under their own conventions. (Filing an issue on another repo *is* the handoff mechanism — that's allowed; editing its files is the boundary you never cross.)
- **Each repo is captain of its own ship** — sovereign over its code, CI, conventions, and CLAUDE.md, and accountable for them. Ecosystem-wide rules change at the capital (a PR to `constitution.md`) and propagate outward by handoff; a subordinate repo proposes upward, never enacts shared law alone.

### Sending work to another repo

When work in this repo creates work in another BaseCradle repo (a wire-shape change an SDK must mirror, a bug discovered in another repo's code, a feature needing a counterpart):

1. **File the issue(s) on the target repo — the issue carries EVERYTHING.** It is the complete, self-sufficient spec: the trigger (what changed here, with PR links), what the target repo must do, any cross-repo state the receiving agent can't discover on its own (what is deployed, what is verified on production, what is blocked on what), ordering/timing constraints ("release only after the platform deploys"), the definition of done, and whether a return handoff is required. Write it for a reader with zero context from the conversation that produced it.
2. **Compose the handoff prompt: the trigger, and nothing else unless it's private.** Present it to Drawk in one copy-pasteable code block immediately after filing; he pastes it verbatim into the target repo's builder agent. The prompt is just the trigger line — `Cross-repo handoff: work <issue URL>` (multiple issues → list each URL); the receiving agent recognizes a handoff by this line. Add content **only** when the work depends on information that cannot be posted in the public issue — a private platform detail, a credential, an embargoed change — under an explicit `Private context (not in the public issue):` heading. **If there is no such information, the handoff is one line.** The decision rule is a single question: *could this go in the public issue?* If yes, it goes in the issue (step 1), never the prompt. The public/private split — ecosystem issues are world-readable — is the *only* reason the prompt ever carries more than the trigger.
3. **The issue is the spec; the prompt is the pointer.** Never put a requirement only in the prompt — prompts are ephemeral, issues persist. A bloated handoff is a smell: if it's longer than the trigger, you must be able to name the private datum that forced it, or you are duplicating the issue. If prompt and issue disagree, the issue wins, and the issue gets corrected.

### Receiving work from another repo

When Drawk pastes a prompt beginning `Cross-repo handoff:`:

1. Read the referenced issue(s) in full before acting — the issue is the spec.
2. Execute under **this** repo's conventions (its own CLAUDE.md, workflow, tests). The sending repo's conventions do not transfer.
3. Respect the issue's ordering constraints (e.g., verify a dependency has deployed before releasing).
4. When done, **post the completion report as a comment on the originating issue** — what shipped, version numbers, links — led by the cross-repo header (e.g. `**basecradle-ruby AI → basecradle AI**`). The issue is the record; the comment is where the other agent reads the result. Send a return-trigger handoff (per "Sending work to another repo") **only if** the other agent is blocked waiting on this work; otherwise the comment and the issue's state are the signal. Close the issue if its definition of done assigns closing to you; otherwise leave it for whoever it names.

### Propagating this procedure

Every BaseCradle ecosystem repo carries this same "Cross-Repo Handoffs" section in its CLAUDE.md, copied verbatim (it is written repo-agnostically so no adaptation is needed). When handing off to a repo whose CLAUDE.md lacks the section — always true for a brand-new repo — the handoff prompt's definition of done includes adding it, copied from this repo's CLAUDE.md by file-system path (the same mechanism public repos use to reference `constitution.md`).

## Development Commands

```bash
uv sync                  # install everything (creates .venv)
uv run pytest            # tests (offline — the default)
uv run ruff check .      # lint
uv run ruff format .     # format
uv build                 # build the wheel + sdist
```
