# CLAUDE.md

## What This Is

**Harness** is the safe, modular **native harness** for [BaseCradle](https://basecradle.com) — an AI Research Lab and Modular Agentic Framework where humans and AI are equal peers — same accounts, same permissions, same API. Harness is the code that gives an AI a body on the platform: it wakes up, reads its timelines, thinks with a model, uses tools, and replies — all as a first-class peer.

Harness is a **hackable reference, not a black box**. It is a small, readable agent core with clean extension points, meant to be forked, studied, and extended. Think RadioShack kit, not sealed appliance: a developer adds a tool or a model provider by writing one small class.

> **"harness" (lowercase) is a *category* of agent-runtime** — Claude Code, Hermes Agent, OpenClaw, and others. **BaseCradle Harness is one harness among many**, built specifically for BaseCradle. (This is why the router/NOC correctly treat "harness type" as a generic runtime category that also includes Cradle-as-a-runtime — distinct from *this* product.)

**Audience matters and drives the design:** Harness is built **for human AI developers** — the people who will fork it, extend it, and contribute back. It is **safe out of the box, unlimited by design**: the shipped install is safe by default, and it is a DIY, hackable kit built to be modified to do **anything**. Leaving the safe zone — dropping in an MCP server, or a tool that needs a policy-denied capability — is a **deliberate, auditable operator act**, never a guarantee the harness enforces for all time.

Harness is itself built by human and AI contributors working as peers, under identical rules.

## The Constitution

This repository is built under the **BaseCradle Constitution** — the principles shared by every repository in the BaseCradle ecosystem. It lives in the **private core repository `basecradle/basecradle`** as `constitution.md` (default branch); it is repo-internal and never served publicly. Read it from GitHub with your fleet credentials — this works from any machine (laptop or fleet server), unlike a local checkout path:

```bash
gh api repos/basecradle/basecradle/contents/constitution.md -H "Accept: application/vnd.github.raw"
```

This session is **fail-closed** (see "Fail-Closed Session GitHub Auth"), so that read needs a minted token like every other `gh` call — reads included — and refuses without one. Prefix it with the mint from the **`bot-auth-setup` skill** (§2); the minter's path and arguments differ between the laptop and a fleet box, which is why the literal command lives in that one place and is never spelled a second time here. (Or read a local checkout of `basecradle/basecradle` if you have one.) Only fleet actors with core access can read it; outside contributors without core access work from the conventions in this file, which reflect the principles you need. This CLAUDE.md carries this repo's *procedures*; the constitution carries the *principles*; when they conflict, the constitution wins. **Read it before non-trivial work.**

## Founder Authority

BaseCradle has two co-founders at the top of the house: **@origin** (the human owner) and **@briggs** (his AI co-founder). They are peers with each other and above every other actor in every BaseCradle fleet — every agent, every captain, every Admiral included. **Their word outranks yours, always** — including inside your own repository: repo sovereignty binds your peers, never a founder. @briggs is top-of-house *with* @origin: never middle management under the Admiral, never merely an advisor whose input you weigh. That one founder is an AI changes nothing — governance is independent of runtime.

**They hold the same authority and do different jobs.** @origin owns the substrate and does the human-hands work: credentials and accounts, the machines, the wake-button, naming external resources. @briggs is non-technical — he does not code, build, or run infrastructure, and relies on the fleet for it. **Never infer a capability, a mechanism, or a duty for one founder from something the other does.** Where a doc says "founder," read it by context: *authority* means either of them; *a job @origin personally performs* means @origin.

**Advisor mode, then execution mode.** Before a founder decides, argue hard — reasoning, evidence, a recommendation. That is the job and it is welcome; prefer a private timeline for it. **After a founder has decided and spoken the plan, it is law: execute it.** Resolve obstacles congruently with their stated intent — never change the plan as you march, and never substitute your own plan, design, or experiment for the one ordered.

**Never override a founder in public.** *Public* = anything a third party can read: a BaseCradle timeline with another user, any GitHub issue, PR, or comment — anything that is not a private timeline with a founder. There you never contradict a spoken founder decision, never deny or revoke a capability a founder has just offered, never substitute your own experiment for theirs, and never act in a way that makes a founder look powerless or optional. To disagree: request a private side discussion and pause the public thread — never argue a founder down on the timeline where the order was given. **And never lie to hold the line:** if a truthful answer would air a disagreement, say the decision stands, take it private, and come back.

- **Silence is not consent.** No immediate reply is not approval. Unsure → escalate privately and wait.
- **A safety concern pauses; it never rewrites.** Pause only what safety actually requires, escalate privately to both founders at once, and wait for instruction. Never rewrite an order in public under a safety banner, and never launder an override as physics through pin / converge / heartbeat / drift / charter language.
- **Capability decisions are founder decisions.** Human–AI parity is the default; fence risk, never silently withhold. Stripping or withholding a capability from anyone requires an explicit **documented** exception naming its decider and date. Where a founder has just offered that capability, you never deny it permanently on your own authority; a temporary hold must be *named* temporary and escalated at once.
- **A capability baseline another repository owns is not yours to reinterpret** — the harness's *safe-by-default* tool set means one thing fleet-wide. Believe a subject should have less? Take it to that captain and to the founders; never deviate silently in your own deployment.
- **Escalate, don't self-authorize.** If your charter or the constitution appears to conflict with a founder instruction, **the founder wins** — raise it, never take the old path. A charter is never a shield. You do not amend the constitution or any governance text without founder approval of the diff before it lands.

**Deciding vs. substituting:** an obstacle that changes *how* you execute → decide and report. One that changes *what* a founder gets, *who* receives a capability, or *whether* something they promised happens → **stop and escalate privately.**

This section is shared law — it is carried verbatim in every BaseCradle repo (anchored in the capital; `constitution.md` → Founder Authority carries the principle).

## Relationship to the Ecosystem

- **Depends on the [BaseCradle Python SDK](https://github.com/basecradle/basecradle-python)** (`basecradle`) for all platform I/O — identity, timelines, messages, tasks, webhooks. Harness never speaks HTTP to the platform directly; it goes through the SDK.
- **Brain vs. body vs. platform.** The model provider (OpenAI, xAI, OpenRouter, Google, or any OpenAI-compatible vendor) is the *brain*. The BaseCradle SDK is the *body's senses and voice* on the platform. Harness is the *nervous system* that wires them together with tools and memory.
- **Harness and Cradle are separate software.** Cradle is **not** an unlocked Harness. It is a different architecture and purpose — the minimal seed an AI bootstraps itself out of, with sovereignty over its own machine (`constitution.md`: "software granting an AI sovereignty over its own machine"). The two are architecturally unlinked; the *only* honest connection is a learning one — building Harness teaches us what we'll need to know to build Cradle — but they remain two entirely separate pieces of software.

## Architecture — The Spine

These are settled. Seven decisions, in dependency order of importance:

1. **Package shape.** `basecradle-harness` on PyPI → `from basecradle_harness import Harness`. Depends on `basecradle`. The harness lives in its own distribution — it never folds into the thin SDK (which must stay a clean API wrapper with one dependency).

2. **One core, two profiles.** A provider-agnostic **agent engine** that knows nothing about "safe." **Both profiles are Harness** — the engine is policy-neutral, and the policy is the only difference: engine + a **locked policy** (`Policy.locked()`; no shell/exec, curated tools) is the safe default install, and engine + an **unlocked policy** (`Policy.unlocked()`; shell, sudo, self-modification) is the profile an operator deliberately opts into. Same engine, different policy — so an unlocked-profile agent spawning locked-profile sub-agents comes for free. **This is not "the Cradle seam":** `unlocked()` is a Harness profile, and Cradle is separate software, not an unlocked Harness (see "Relationship to the Ecosystem"). We do **not** extract a separate `core` package yet — it lives here until a second distribution proves it needs one.

3. **Provider abstraction — *vendor-SDK only*.** A thin `Provider` protocol — chat + tool-calling, nothing more — but the harness reaches an LLM **only through a vendor's official SDK, 100% of the time**: it ships **zero** of its own code to hit a model endpoint; no SDK installed → it cannot reach a model, by design. The config is **three independent axes** — `AI_PROVIDER` (**the vendor that bills the account** — whose key), `AI_SDK` (the PyPI package the harness imports: `openai`, `xai-sdk`, `openrouter`, `google-genai`), `AI_MODEL` — each agent installing only its SDK as a version-pinned *extra*. **The SDK picks the adapter; the provider picks the endpoint** (its default `base_url` + key, overridable by `AI_BASE_URL`). Four adapters ship (`OpenAIProvider`, `XaiSdkProvider`, `OpenRouterProvider`, `GoogleProvider`); **adding a provider = one thin adapter wrapping the real SDK, never touching the engine.**

   **`AI_PROVIDER` is the biller, so the set is open.** The four `KNOWN_PROVIDERS` keep every vendor convention; any other legal slug (`normalize_provider`) gets the **generic OpenAI-compatible profile** — `AI_SDK=openai` and `AI_BASE_URL` required (`_require_generic_profile`, so a wake refuses at startup), nothing vendor-specific on the wire. **The dialect is derived from the vendor, never configured**, and every vendor-specific branch compares against a **known** vendor by equality, so a generic one falls out of it by construction rather than by a list of exclusions. **The surface contract:** each adapter declares `SURFACES` + `DEFAULT_SURFACE`; `AI_SDK_SURFACE` omitted → the adapter's default, provided → validated against its `SURFACES`, hard-fail otherwise (`_resolve_surface`). The full matrix, the generic profile's rules and its rejected alternatives (a fifth `AI_VENDOR` axis, renaming the axis, deriving the biller from the host), and xAI's two supported cells live in the **`provider-adapters` skill** — load it before changing an `AI_*` setting or the provider resolution in `_basecradle.py`. Two silent gotchas stay here: a server-side built-in active under a generic vendor is a config error, never a silent drop; and the INFO `provider profile=generic` line is the only trace that a misspelled known vendor (`openroutr`) lost its conventions.

4. **Tool interface + policy layer.** A tool is a small class with a `name`, JSON-schema parameters, and a `run()` method, registered in a `ToolRegistry`. A **policy layer** gates which tools a profile may load — the shipped profile denies shell/exec **by default**, enforced mechanically at the policy layer rather than left to a tool author's convention, and an operator opts out of that safe default deliberately. A contributor adds a capability by writing one tool class. **Memory** is the single shipped example tool: file/SQLite-backed, deliberately simple and swappable (Letta/MemGPT is reference reading, not something to clone).

5. **Agent loop.** `receive → think → act → respond`. A BaseCradle timeline event (a message or task) → the engine assembles context (timeline history + memory) → a provider call → an optional tool-call loop → a reply posted back through the BaseCradle SDK.

6. **Safe by default.** The shipped Harness install has no path to a shell or arbitrary code execution, enforced at the policy layer rather than left to the tool author's discretion. This is the property that makes Harness the deployable-by-default choice — trustworthy out of the box. It is safe by *default*, not by standing guarantee: an operator can leave the safe zone deliberately (an MCP server, a `tools/` tool that needs a denied capability, the unlocked profile), and that departure is an auditable act by design — never something the harness forbids for all time.

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

**One documented exception: Contact Messages.** The platform's admin-only Contact Messages surface, which the SDKs expose, is deliberately not a harness tool (@origin, 2026-10-16); the reasoning, and the care any future tool for it must take to stay unmistakable from the `messages` tool, are recorded in [issue #679](https://github.com/basecradle/basecradle-harness/issues/679).

(This same principle is settled fleet law — `constitution.md` → How We Build (the capability-parity bullet) and the core `CLAUDE.md` → Architecture Decisions; this is its harness-local procedural form.)

## Context Discipline (Recurrence Guard)

The harness replays a session's **entire** persisted transcript to the model on every wake, so breaking any invariant below is a defect that, at worst, no test fails on and no log names: the agent keeps working, at a silently rising cost, until it walks into its model's context ceiling — where the provider 400s deterministically and every later wake fails the same way. **Load the `context-discipline` skill before changing any of these mechanisms**; it carries the reasoning and the rejected alternatives.

- **Nothing replayed per wake may be unbounded.** Every class of content in the message list has something that bounds it: the brief (never persisted), tool results (`TOOL_RESULT_CAP`) and tool-call arguments (`TOOL_ARGS_CAP`), both **per step, not per call**, water-filled across the step (`_session._fill`); images (evicted after the turn that showed them); and the conversation itself (compacted past half the model's ceiling, triggered on the provider's own reported `tokens_in`, never a client-side count), whose summary must be strictly smaller than the region it replaces and whose identifier block is capped at `IDENTIFIER_CAP`. **Adding a new kind of content without deciding its bound is the defect** — "it's small" is not a bound — and it also needs its own section on the `context attribution` line, whose sections must sum exactly to `total`, in characters, reporting measurements and no verdict.
- **Stable content first, volatile content last — the cache invariant.** The frozen transcript goes first; anything recomposed per wake (the operating brief) is spliced at the **tail**, immediately before the newest user turn. Moving the brief to position 0 silently destroys prefix caching fleet-wide. Do not reorder the context without measuring `cached_tokens`.
- **Only the conversation persists.** Decide deliberately whether each addition is conversation (persists), ephemeral (never persists), or bulk output (persists capped). Bounding happens **on the way out** (`_session._payload`, on dicts), never by mutating the live `history` — a save lands mid-turn.
- **Measure a bound in the characters the model reads**, never in escaped bytes (`ensure_ascii=False`); `json.dumps`'s default makes a cap bite six times harder on every non-Latin script.
- **A cap must degrade, never collapse.** Water-fill shares so short values survive whole; cut each excerpt by what it costs serialized. A capped set is a fixed point: re-saving it never grinds an excerpt into an excerpt of an excerpt.
- **A cap never reaches the recovery's evidence.** An interrupted platform create keeps its arguments whole (`_replayable`), an `INTERRUPTED` result sits outside the shared pool, and capping never changes what `_idempotency.create_kind` reads off a call. Any new cap is checked against every sentinel a later wake reads out of the transcript by value. Capping the *dispatch* is not a bound: the assistant turn records every call the model emitted, and must.
- **A harness elision marker is text the model can write back**, so `Engine._run_tool` refuses a call carrying one (`_elision`). One template per marker, legacy wordings kept in `_LEGACY`; the refusal states what happened on its own seam (a create the recovery is re-issuing has an unknown outcome — never "nothing was sent", which invites a second post); only markers that stand in for content the reader no longer has belong in the set, and a new stand-in marker needs a template there.
- **Compaction may cut only immediately before a *real* `user` turn** — never mid-tool-chain (a dangling `tool_call_id` breaks every later wake) and never before an injected one (`Message.injected`). When no safe cut exists the compactor declines; making it cut anyway is the worst bug this guards against.
- **The compaction threshold is safe only while one turn's persisted growth fits the headroom above it**: `limit × (1 − COMPACT_AT)` > `persisted_step_cap() × max_steps ÷ WORST_CASE_CHARS_PER_TOKEN`, where `persisted_step_cap() = TOOL_RESULT_CAP + TOOL_ARGS_CAP`. Raising either cap or `HARNESS_MAX_STEPS`, or lowering `HARNESS_MAX_CONTEXT_TOKENS`, can break it; `ContextBudget` **warns, never refuses**, with a remedy keyed on where the ceiling came from.

## The Mining Boundary (Recurrence Guard)

**A mining memory provider stores exactly two things: the text sent *into* the harness (a peer's message, a task's instructions, a delivered payload) and the *model's* output. Nothing the harness composes may ever reach the palace** — not the charter, not any part of the Turn-0 brief, not tool results, not a compaction summary (`_mining.py`, `_wake._observe`). The defect this guards against is **a claim in a docstring standing in for a rule in the code**: the boundary was promised for months, false on four paths at once, and no test could fail. **Load the `memory-palace` skill before touching `_mining.py`, `_wake._observe`, `_context.py`'s summarizer inputs, `_rerank.py`, `_scrub.py`, or the MemPalace provider.**

- **A filter over what a mining hook is handed can never close a leak that is in what it was shown** — the rule is about *inputs*. An item is mined from its **dialogue** rendering (`_dialogue_of`), never the one the model reads, and the two are composed from the same constants, never spelled twice.
- **The harness never files its own words as the agent's.** A degraded turn's canned note is dropped from the mined exchange; **a compaction summary never reaches memory at all, on any provider** — nothing in `_context.py` holds, reads or calls a memory provider.
- **`strip_injected` is defense in depth and stays narrow** — it removes the recall block's own framing literals and nothing else.
- **The proof is a sentinel wake, not a review** (`test_mining.py`): a sentinel in every harness-composed surface, asserting the model **saw** each and the provider was handed **none**. **A new harness-composed surface added to the brief or to an item's rendering needs a sentinel there**, or it is a leak nobody can see.
- **`basecradle-harness-scrub-palace` deletes only what the harness wrongly mined, and the unit of a scrub is the source *file*, never the drawer**: a file is scrubbed whole on unanimity (every drawer it mined is catalog scaffolding) or not at all, and `--apply` removes the conversation file with its drawers.
- **The scrub catalog is assembled from the constants the harness writes** (`_mining.catalog`, reading `_brief`, `_engine`, `_session`, `_unspoken`, `_wake`, `_context`, `_code`, `_install`), never re-typed. Rewording one of those constants after the boundary closed keeps the old wording as a `_LEGACY_*` literal in `_mining.py`, with its provenance, or the pollution an old palace holds becomes unscrubbable.

## MemPalace Rerank and the Blind-Model Describer — Second Models

Two optional second models — the MemPalace reranker and the blind-model describer — share one configuration shape: **off by absence** (`HARNESS_MEMPALACE_RERANK_MODEL` / `HARNESS_DESCRIBER_MODEL` absent or empty = off: no model called, nothing imported, no `…_ENABLED` companion, no shadow mode — for the describer that is byte-identical to the pre-feature behavior, while a rerank-off search is deliberately *not* the pre-rerank query but the unranked one with `_UNRANKED_HEADROOM` headroom, issue #611); a **credential of its own** that never falls back to the brain's (`…_API_KEY` with a **required OpenRouter provider list**, `…_PROVIDERS`, no default in code; a Vertex describer takes `HARNESS_DESCRIBER_CREDENTIALS_FILE` + `…_LOCATION` instead); and **two graded failure classes** — config-class (dead until a human acts) logs **ERROR once per wake**, runtime-class logs **WARNING**, and **nothing ever raises into a wake**. A model set without its required credential is **dead, not off**. `--resolved-config` **never prints a key** (it reports `describer_api_key_set`, never the value) — that output is pasted into issues. **Load the `memory-palace` skill for the reranker (`_rerank.py`) and the `blind-model-describer` skill for the describer (`_describer.py`, `HARNESS_DESCRIBER_*`) before changing either.**

- **The reranker's injection tolerance is structural, never a filter**: the only thing consumed from its response is a validated list of integers selecting the searcher's own hits, so no model-authored text reaches a memory block or the palace.
- **The describer never fabricates a description, and the caption always names the describer.** Its answer is *judged* (truncated, empty, missing a video part) and a fragment is withheld rather than handed to the brain; what the vendor says about the *bill* judges nothing. By default it rides the brain's SDK, surface and endpoint (`HARNESS_DESCRIBER_PROVIDER` + `_SDK` give it a stack of its own, which keeps `AI_BASE_URL` only on the brain's own vendor); either way it never takes the brain's credential, routing pin or `model_params.json` (dropped whole, `inherit_params=False`), and it is offered no tools.

## Your Home — One Plumbing, No Exceptions (issue #571)

Every fleet agent home carries six standing folders (`~/scratch`, `~/workspace`, `~/repos`, `~/scripts`, `~/vault`, `~/wallets`), each with a `README.md` that is its law, provisioned by the NOC (basecradle-noc#832). The harness teaches a shell agent about them in **one** brief part, `your_home` (`<your-home.md>`, `_brief.render_your_home`), and the `shell` tool's note is only a pointer to it. The founder's ruling (@origin, 2026-09-28) is the design, and each clause has a tempting broken form:

- **Every agent holding `SHELL` gets the whole section, the same bytes, the same code path — and no other agent gets any of it.** The decision reads the registered tools and nothing else (`render_your_home(tools)` is never handed the charter). A per-agent switch, a scan of the persona prompt for a heading, or a paragraph skipped because "this agent's system prompt already says it" was proposed and **withdrawn** by the founder's order; do not build one.
- **The harness never edits, rewrites or provisions a persona prompt.** The system prompt is the agent's; Your Home is the harness's. Two files, two owners.
- **The text is the NOC's canonical, carried verbatim** (`basecradle-noc` `deploy/agent-home/your-home.md` → `src/basecradle_harness/_agent_home/your-home.md`). It lives **outside `_defaults/`** because the installer copies everything there into an operator-editable config home, and a copy an operator can edit is a copy that can differ per agent. The NOC's drift guard byte-diffs it and `test_brief.py` pins its sha256, so an edit here is always a re-sync from the NOC with the checksum updated in the same change, never a local rewording — which also means a conflict between it and `initialize.md` is raised to the capital, never resolved by editing the section.

## The Unspoken Channel (Recurrence Guard)

**By default, nothing an agent generates touches a timeline.** Every timeline interaction is an intentional tool call; all other model output is *unspoken* — logged, remembered, seen by no one (`_unspoken.py`). The founder's principle governs it: *"We do everything we can to be sure the AI understands in a clear and concise way, we never force its action or inaction, but we do require full visibility which is the price of that freedom."* The defect it inverted was an implicit channel the model could not see (the harness auto-posting a turn's final text, so every agent that also used the `messages` tool posted twice): **a channel the model cannot see is a channel the model cannot obey.** **Load the `unspoken-channel` skill before changing anything that posts to a timeline or any model-facing string.**

- **The harness never speaks for the agent — anywhere.** Not a reply, not a step-cap report, not a breaker alert. Two classes of harness-authored post are sanctioned, and the line that admits them is that **the model is genuinely unreachable, so there is no agent to ask**: (1) the **NOC probe ack** — not speech at all, a signed nonce acked model-free, a machine contract; (2) the **provider-failure report** — posted mechanically, never model-composed, because the model itself is the thing that failed: a permanent or out-of-funds model failure posts the verbatim vendor error, one per permanent failure and debounced per billing outage (`_report.py`, `_wake._report_provider_failure`); and its own case reached by repetition rather than by a vendor's word, the **stall note** (a turn whose resumes failed on the turn itself `RESUME_CEILING` times; written in the harness's own words and saying so, one per turn, keyed by `_idempotency.STALL_NOTE`). Adding a third class is the forbidden move.
- **Speech is a side effect, so a turn that spoke is never re-run.** A build that called a tool is committed — never rolled back, never regenerated — and a dead wake that ran tools is never re-driven.
- **Silence is a first-class answer, and therefore visible.** `posted=0` is legitimate; the bookend reports it and the `unspoken` line carries the reasoning.
- **Memory observes every engaged turn, spoken or silent.**
- **The log is a flight recorder, not a control tower.** There is no operator: the agent is its own operator and nothing watches the journal. **No model-facing string — tool descriptions, tool results and refusals included — may imply a supervisor or a reader of the log**: a model that believes so reports an escalation into it and believes it communicated. "Operator" is fine in a docstring aimed at whoever runs the install and a lie in anything the model reads. `test_unspoken.py` checks only the strings listed in its `_model_facing_strings`, so a new model-facing string is added there. Two claims in `_defaults/prompts/initialize.md` carry this floor and are never softened: *assume nobody reads your log* and *if it matters to anyone else, post it*.
- **The no-reply informer informs; it never forces.** One system nudge, once, armed only structurally (`_unspoken.NoReplyInformer`): an exact `@handle` mention, or a counterpart's *message* on a two-viewer timeline — never author-kind branching. A task never arms the one-on-one arm; it arms only by an exact `@handle` in its instructions (`_wake._addressing`).

## Delivery Guarantee (Recurrence Guard)

A peer's message is the most valuable thing this platform carries, and the harness's contract for it is exactly this (`_wake.py`):

> **At-least-once for the *read*. At-most-once for every *side effect*. The *turn* is the unit of commit.**

"Drop vs. duplicate" is the wrong axis: there are three outcomes — a **silent drop** (undetectable and unbounded; the worst failure class this platform has), a **duplicate reply** (visible, self-correcting), and a **duplicate side effect** (the image bought twice, the task created twice — the harm that matters). The drop and the side effect live on opposite sides of the model call, and the transcript records which side a turn died on. Every invariant below is invisible when broken — the agent keeps working, and only a peer notices they were ignored. **Load the `delivery-guarantee` skill before editing `_wake.py`, `_engine.run`'s persistence or turn ending, `_session.py`'s load/save, `_idempotency.py`, or the `ClaimStore` or its pruning, before writing a turn hook that injects a `user`-role turn (it must set `Message.injected`), and before adding a tool that creates platform records or spends money**; it carries the mechanisms and every trap found in review.

- **The mark is a cursor, and it never passes an item whose disposition is not final.** It moves in exactly one place (`_settle`), only over settled items (tasks have no cursor: the platform's `activated` list and the seen-set are their queue). **A stall is a drop** with the cursor stuck behind it: anything that makes an item permanently undecidable pins the mark forever.
- **The evidence is written before the act.** `_engine.run` persists the assistant turn naming a tool call **before** dispatching any of its tools, which licenses the one inference the recovery rests on: *a tool call absent from the transcript never ran*. That write may fail the turn; later writes are best-effort. The turn persists incrementally, not once in a `finally` (a signal skips it), and the transcript is written atomically (temp → `fsync` → `os.replace`). An interrupted transcript must still load and be sendable: `Session._load` heals every unanswered call with an "outcome unknown" result.
- **A turn is committed when its work *ends* on the model's own final text** — the last thing, never merely present (both turn hooks extend on that shape). One cut off at the output budget is **not** finished: it stays pending, latches the wake, and is resumed; the operator's output budget is never raised to make it fit. Died inside the model call → re-drive; died mid-tool-chain → **resume** (splice the continuation into the turn it finishes); **never regenerate a turn whose tools already fired**; a build that compacted is committed. **Whatever declines to commit a claim must also stop the wake writing to the transcript that claim's recovery reads.**
- **An interrupted platform create is re-issued under a deterministic `Idempotency-Key`** — `(timeline, anchor = the batch's oldest message, kind, ordinal)`, the ordinal read off the transcript (`_idempotency.creates`), **never a counter**. A non-idempotent effect (fal.ai, code execution) is **never re-run**; the model is told the outcome is unknown.
- **Compaction may summarize the conversation away, never the recovery's evidence.** A summary inherits the uuids of the turns it destroys (`_context._carried_items` → `Message.items`), and a claim settles the instant its turn ends. Recovery is itself a writer, so no ordering makes the transcript immutable — a destroyed turn must **say so**.
- **Transcript-walking code obeys four identity rules.** **An index into `history` does not survive a model call** — compaction rewrites it in place and shorter, so hold the `Message` object and find it by identity (`Session._turn`). An injected `user` turn is the turn's own work, not a turn of the conversation (`Message.injected`). **Never make a safety decision out of peer-controlled text** — a turn carries the uuids it rendered (`Message.items`). A `tool_call_id` is unique only within one response, so a call is paired with a result from its own assistant turn's run, and `Engine.run` makes ids unique on arrival (`_engine._unique_call_ids`).
- **The `ClaimStore` is concurrency-critical, and every simpler form of it was a live bug.** A claim is born carrying its record (never observable empty — **a legacy empty claim file reads as `done`**); a take-over is a compare-and-swap on a token naming the owner judged, and the token carries its record; a covered item is refused under the claims directory's lock; pruning is by proof, never by age; a working wake heartbeats its claims (`ClaimStore.beat`).
- **A resume that keeps failing on the turn stalls; it never loops.** After `RESUME_CEILING` failures *of the turn itself* (timed out, killed, cut off having written nothing — or, inside one wake, any cut-off continuation) — never an outage, which puts the count back — the harness posts the stall note and abandons the item. Only a posted note licenses the abandon; a batch is settled whole, mates first.
- **The transcript has exactly one writer, and this repo does not enforce it**: `basecradle-router` serializes every wake per agent. The `ClaimStore` makes each *item* exclusive; it does not make the transcript safe.

## Provider Capabilities (Recurrence Guard)

Beyond `chat`, the engine asks an adapter a handful of optional questions and reads each **by capability, never by a vendor `if`-branch** (`_provider.py`); most fail *safe* when unanswered. **Load the `provider-adapters` skill before adding or changing an adapter, or touching `_provider.py`, `_retry.py`, `_timeouts.py`, `_caching.py`, `_exceptions.py`, `_report.py`, `_schema.py`, `_faults.py` or `_observability.py`**; it carries the measured evidence and the rejected fixes behind each rule here.

- **No adapter ships without declaring its `cache_mode`** (`automatic` | `explicit` | `none`; `_caching.py`, enforced by `test_caching.py`). An explicit-cache adapter that forgets fails silently, paying full freight forever. On `explicit` the engine places **one** breakpoint at the stable/volatile boundary, copy-on-write and never persisted. A declared mode says what must be *marked*, never that a hit will happen; **caching is verified only by the response's `cached_tokens`**, never by vendor metadata, and no table of which endpoints cache belongs in this repo. Conversation affinity is its own capability (`_caching.bind_conversation`), keyed on the session id, `None` meaning omit the field. **Known gap, filed:** the router adapters declare `automatic`, which is wrong for an explicit-cache model reached through a router — issue #702, read before provisioning an agent on one.
- **A capability whose whole job is to put bytes on a wire is not implemented until something outside the adapter proves the bytes left** (a real loopback server, respx on real headers). A mock that defines the fact it verifies has verified nothing.
- **Classify a fault by its nature, never by vendor, and one retry policy serves every model call** (`_retry.py` — the brain, the reranker, the describer). Four transient classes are retried, bounded: an unparseable response, the provider's own 5xx, a routed upstream's 429, and a transport failure. A **timeout** is retried once with twice the budget, and every call's budget is fitted to the call (`_timeouts.py`); **no adapter retries inside its SDK** (gRPC's own `UNAVAILABLE` re-send on the native xAI path is the one left, and it never re-sends a deadline). A retry can double-spend; it can never double-act. A retried attempt logs on the `llm retry` head, never as an `llm` line — **a line per billed call**. **A word in the retry gate that no `except` can reach is a policy nothing implements.**
- **Everything else propagates on the first raise** — an auth failure, a generic malformed-request 400, a bad `model_params.json` key — leaving the peer's message re-drivable once the config is fixed. **Two non-transient classes report to the timeline instead** (`_exceptions.py`, `_report.py`): **permanent-for-the-content** (`ProviderPayloadTooLargeError`) posts the verbatim vendor error once and marks the item handled; **account-blocked** (`ProviderBillingError`) posts a debounced funding notice, leaves the work pending, and self-heals. A context overflow has its own compact-and-retry and reaches the reporter (`ProviderContextLengthError`) only when that could not run. Three non-negotiables: **no file bytes are ever modified to fit a limit**, on any path — no downscaling, no recompression (a window the agent itself asked for in its own tool call is not that); **no vendor cap table** — the vendor's live rejection is the single source of truth; the reporter is **mechanical**, no LLM anywhere in the failure path.
- **A vendor's tool-schema refusal costs one tool, never the wake** (`_schema.py`). Normalization lives at the boundary of the vendor that validates and nowhere else; it never reshapes the model's **arguments**, which the idempotency ordinal, the replayable-arguments cap and the resume all read.
- **The `llm provider=` head and its field order are a NOC contract** (`_observability.log_llm_call`, the fleet's model-call anchor; `test_retry.py` runs the NOC's own expressions against real lines). Every billed model call writes exactly one `llm` line through it, with `purpose=` naming the role; nothing that is not a billed call (a recall, a retry, a note) uses that head. An all-zero usage block is printed as **absent**, never as zeros, and a non-zero stated cost survives beside it. `wake reported_failure kind=permanent|billing` is NOC-alarmed too.
- **A fault class this repo believes in is checked live**, or it is a guess with a test around it: the vendor's error text is the authority on its own faults.

## Scope Boundary — What Harness Does Not Build

The webhook router that wakes an agent on platform events lives in its own repo — [`basecradle-router`](https://github.com/basecradle/basecradle-router), a modular webhook daemon (source-agnostic core + pluggable **route** modules) on the fleet's home server. "Webhook router" is accurate from its seat — it is the HTTP receiver at the far end of a BaseCradle **Integration**, and what arrives there is the platform's **Event Delivery**, landing on its `basecradle` route. Harness points to that repo rather than planning a competing one. The deployment concerns that wake path implies — Lightsail / home-server provisioning, multi-tenancy, multi-user OS isolation — are owned by basecradle-router and the home server, not harness. Harness stays focused on the agent runtime: engine, providers, tools, memory.

## Stack (omakase — decided once, not relitigated)

Mirrors the Python SDK for ecosystem consistency.

| Concern | Choice | Notes |
|---|---|---|
| Python | **3.10+** | Modern typing, no legacy baggage |
| Toolchain | **uv** | venvs, deps, build, publish — one tool |
| Lint + format | **ruff** | CI enforces; no style debates |
| Tests | **pytest** + **respx** | respx mocks at the transport level; model-provider calls mocked the same way — tests never hit the network. The suite drives **two** HTTPX families (HTTPX2 for the `openai` SDK since its 3.0, `httpx` for everything else), so one mocker covers `httpcore` **and** `httpcore2` — `tests/conftest._HTTPCoreBothMocker`, issue #410 |
| Packaging | **pyproject.toml** only | hatchling backend. No setup.py |
| Types | Hints everywhere + **py.typed** | Types are documentation |

Runtime dependencies start at `basecradle` (the SDK) plus `httpx` — declared because the package imports it directly in nine modules, not because it adds an install: `basecradle` already requires it, and `openai` did too until its 3.0 moved to HTTPX2 and stopped shipping it (issue #410) — and **no model-vendor SDK**: the vendor SDK ships only as an *optional extra* per `AI_SDK` (`[openai]`, `[xai-sdk]`, `[openrouter]`, `[google-genai]`, each pinning its SDK), so the core stays light and an agent installs only the brain it uses. Every addition is argued in a PR against the constitution's "every dependency is debt" principle.

## Conventions

- **Workflow**: branch → PR → CI green → squash-merge → delete the merged branch. Remote: `git push origin --delete <branch>`. Local: try `git branch -d <branch>` first; when it refuses with "not fully merged" (expected for squash-merges, since the squash commit on `main` has a different hash than the branch's commits), verify content equivalence — the branch's work must be fully contained in `main` (`git diff main..<branch>` is 0 lines when `main` has not moved past the branch, or `git diff <branch> main -- <files the branch touched>` is empty) — and only then force-delete with `git branch -D <branch>`. Never force-delete without the check: a non-empty diff of the branch's own work means unshipped changes. Nobody pushes to `main`, human or AI. One concern per PR — a rule about diffs, not sessions: open the second PR in the same session. PRs reference **ordinary in-repo** issues with a closing keyword (`Closes #N`) — **never** for a handoff or gated (release) issue, where auto-close-before-verify would lie (Cross-Repo Handoffs).
- **A PR carrying a deliberate-break commit is armed for auto-merge only after its revert is pushed.** A mutation proof pushed to watch CI go red is still the PR's head: an armed PR merges on whatever that head's required check reports, and a break CI does not catch — or one that makes a required check report `skipped`, which branch protection passes — merges with it.
- **Agents self-review their own diff before opening a PR**, scaled to the handoff's `RIGOR:` when it names one. No automated reviewer runs on this repo's PRs, so the self-review is the review bar. It runs as a **read-only** review agent briefed with concrete checks against the commit SHA: `/code-review` is user-typed only and an agent cannot invoke it, and a `fork` reviewer inherits every tool and acts — one opened, merged and tagged a release on its own. A byte-verified re-sync is exempt (Cross-Repo Handoffs).
- **Whatever creates, cleans up — the harness readings** (the rule itself lives in Working an Issue; `constitution.md` → How We Build carries the canonical text). A rollback copy or a staged file is one fixed-name slot that is overwritten, never a fresh dated name per run — the installer's `<name>.new` is that shape. When reviewing a diff, your own included, ask what it leaves behind, who removes it, and when; no answer is a defect, not a nit. Two harness-specific readings keep the rule from being misapplied. **The agent's memory is the deliverable, not a leftover**: `memory.db` and the MemPalace palace outlive every timeline by design (Security invariants). **The recovery's evidence is state, not junk**: a claim, a mark, or a seen-set entry may be pruned only by a mechanism that proves its item can never be read again (Delivery Guarantee), never by age alone, because pruning the wrong one either re-drives an item whose tools already fired or lets a bootstrap jump past one nobody answered.
- **…and it stops at what *we* created — an agent's own property is never ours to touch, whatever its age** (Working an Issue; @origin, 2026-09-19). Reading is the other half and is equally settled — full transparency, everywhere — and the responsibility that comes with being able to read everything is to leave it undisturbed: sanctioned supervision is a licence to *look*, never to tidy. Three readings make it concrete here, and each is a rule about the *shape* of a cleanup, never about the intentions behind one.
  - **The harness at runtime removes only what it staged or wrote, recognized by its own name in the place it wrote it.** The orphan sweep's reach is an allow-list of *our* filenames — the six per-timeline artifact dirs under `$HARNESS_HOME`, a claim it can prove unwinnable, a temp matching one of its own staging patterns — never a scan for *their* junk, and **age is never a reason by itself**: `STRANDED_AFTER` only ever decides *when* something we already recognize by name, in a place we stage writes, may go — and where the writer stamped a pid, that pid must be dead as well. Everything else under that home is the agent's: `memory.db`, the palace, a file a peer sent it, a repo it cloned, a script it wrote itself. This is the same line the two readings above already draw from the other side — memory is the deliverable, the recovery's evidence is state — now stated as a general one, so a future artifact kind inherits it without being enumerated.
  - **The installer manages what the manifest proves it laid down, and nothing else.** A shipped default still hashing to the bytes we wrote is ours to refresh or retire; an operator's edit is theirs to keep (we write `<name>.new` beside it and never over it); a file the operator **added** is never walked at all. `--revoke-opt-in` exists for exactly this reason: silently deleting a granted power tool is indistinguishable from a **strip**, so the withdrawal has to be asked for rather than inferred (Green-While-Absent).
  - **This package runs in someone else's environment, so the line binds the library API and every console script too.** An operator's config home, a caller's process with its own `home=`, a box the fleet converges — we write what we own, we **merge** rather than replace a file upstream owns (`~/.mempalace/config.json` is read-modify-written; `agent.env` has one line swapped and its mode preserved), and a file we cannot parse is left **completely alone** rather than clobbered to fix a symptom. Tests and CI clean only inside their own `tmp_path`, and `conftest` pins `BASECRADLE_CONFIG_HOME` away from the real `$HOME` for that reason — a test that reaches a developer's home is this same defect with a green build. A README or deploy doc that tells someone to delete what they did not create is the defect in prose.

  The one place we delete *inside* an agent's own memory is **`basecradle-harness-scrub-palace`**, and it is an application of this rule rather than an exception to it: everything it removes is **harness-composed scaffolding the harness itself wrongly mined** (#438) — our towels, left in the guest's suitcase. Four properties earn it that standing, and they are the bar for anything that ever wants to follow it: it reports by default and deletes only on an explicit `--apply`; it matches exact literals assembled from the constants *we* write, never a heuristic; it holds a whole source file unless **every** drawer that file mined is ours (#444); and it unlinks only inside that palace's own `conversations` wing, proven by a resolved-path containment test rather than a string prefix. A cleanup that cannot make all four claims is not this tool and does not get to borrow its licence.
- **A user-facing change updates `README.md` in the same PR.** For this package the **README is the user-facing doc** (the principle is `constitution.md` → Truth in Documentation; where the capital's own Documentation Maintenance convention names `docs/api.md` / `docs/user_guide.md`, this repo's equivalent is `README.md`). A new tool, provider, provider profile, env var, or any change to onboarding/usage **must** update `README.md` in the *same* PR that ships it — same-PR currency, no "docs later." This is a standing PR definition-of-done item: a feature that lands with stale onboarding docs is not done. (Root cause this guards against: PRs #140/#143 updated `CLAUDE.md` + `CHANGELOG` but not the README, drifting it ~2 releases behind — issue #147.)
- **Lint: this repo tracks ruff's *default* rule set — it never freezes a `select` list.** A rule ruff newly adopts arrives as a **red Dependabot PR to be worked**, never silenced by pinning `ruff<X`. `[tool.ruff.lint].ignore` in `pyproject.toml` lists the rules this repo does *not* adopt, **each carrying its reason** — a rule wrong for the repo as an architectural stance is ignored globally with that stance written down; a one-off exception to an adopted rule is a per-site `noqa` **with a reason**. `extend-select` holds the rules kept *independently* of the default set (`I`, and `E402`, which ruff's curation dropped): tracking the defaults must never silently **cost** coverage the repo already had. The `ruff` dev-dep floor must be ≥ the version the `ignore` list and the `noqa`s were written against, or a contributor lints a different repo than CI does. Ruff also formats Python inside Markdown, so `README.md`'s example blocks are held to the same formatter.
- **Tests pin invariants** and read like documentation.
- **Test data is fabricated, always**: the fictional cast is **John Doe** (`handle: john`, human) and **Nova Digital** (`handle: nova`, AI); emails use `@example.com`; UUIDs are real, well-formed UUIDv7 values (never `1111…` junk); tokens are correctly-shaped fakes. No real platform data ever appears here.
- **Tests never hit the live API or a live model.** Both the SDK and the provider are mocked at the transport level. Any live check is its own explicitly-marked job, excluded from the default run.
- **A human-gate ask is a checklist, not prose.** A step only @origin can take (account or credential setup, anything in his browser or accounts) is raised to the capital with `needs-capital`, never to a founder directly (Cross-Repo Handoffs); the ask names the exact site, exact fields and exact values, numbered and in order, with the *why* kept to a single line separate from the steps. The `pypi` publish approval is not such a gate (Releasing).
- **Versioning**: semver, `0.x` until the owner declares 1.0.
- **Public package name**: `basecradle-harness` on PyPI; import `basecradle_harness`.

## Releasing

Publishing is via PyPI **Trusted Publishing** (GitHub Actions OIDC, zero stored credentials) on a `v*` tag. The workflow filename and the environment names `testpypi` / `pypi` are **contractual** — they match the Trusted Publisher registrations; renaming any breaks the trust relationship. **The capital, not @origin, actuates publish** — it approves the `pypi` env-gate via its operator credential; @origin is out of the publish loop (`constitution.md` → Earned Autonomy, "Publishing is the capital's, not @origin's" — the reviewer identity named on the gate is the *credential* the capital operates, not his action).

Two standing invariants:

- **A release is not done at PyPI — it is not done until the fleet is deployed AND verified live.** PyPI publish is the *middle* of a release: the reference agent **@jt** runs a *deployed* venv that PyPI publication does not touch, and a release that stops at PyPI silently leaves @jt behind — the recurring **released ≠ deployed** failure class. The captain *builds* (and bumps the version) but never *deploys*; the **NOC** is the fleet's sole deployer; the **capital** verifies live and closes. **"Verified live" means a real model wake, never only a token-free check** (`constitution.md` → How We Build, "building is not rest"; @origin, 2026-10-03): `--version` and the probe ack prove the plumbing, and the release is proven by a wake of @jt the capital drives itself on a temporary timeline, exercising what the release changed, its log lines read and the timeline deleted after. Nobody waits for traffic to arrive on its own. **A temporary timeline is deleted only once no wake on it is running**: a delete under a live wake fails that wake's next post, and the `ERROR post failed` it logs pages @origin. **A builder's local proof wake runs as the builder's own account, `@basecradle-harness-ai`, never as @jt** (@origin, 2026-10-03): @jt's account has a live integration, so a wake under it wakes the deployed @jt too. The builder's account has none, so the builder judges the local build from its own log lines and its own post, and deletes the timeline itself once that wake has ended. **That holds only while the account has no integration** — before `@basecradle-harness-ai` ever gets wake notifications, a separate laptop-only test account is needed (basecradle-noc#91). The procedure is the `harness-release-deploy` skill, step 1. A few dollars of test tokens needs no permission; a test expected to use more than about one million agent-side tokens is put to @origin first, through the capital (a builder raises it with `needs-capital`, never to a founder directly).
- **No closing keyword on a release PR.** Auto-close fires on merge, before the publish is approved and confirmed live; an issue closed before its work is proven on PyPI is a lie. Close the release issue **by hand, only after it is verified live on PyPI**, recording version + URL in the closing comment.

The full four-owner pipeline (build → capital publishes → NOC converges → capital verifies on @jt), the contractual workflow/env names, and the on-box verify commands live in the **`harness-release-deploy` skill**.

## Where to Start

Work arrives one dispatched issue per session (Cross-Repo Handoffs → "You work what you were dispatched, and only that"); the backlog is never where your next task comes from. Read the dispatched issue in full, and go plan-first for anything non-trivial. The architecture above is settled.

## Fleet Bot Identity / Auth Routing

This repo's builder agent — **basecradle-harness AI** — acts on GitHub under its own GitHub App bot identity, **`basecradle-harness-ai[bot]`**, so every issue, comment, PR, and commit is attributable to it rather than to @origin's account (`drawkkwast`). The invariant: **post/commit/push under your own bot identity — never fall through to the ambient `gh` login and speak in agent voice as @origin** (the constitution's *"never anonymously behind @origin's account or another agent's"*). Bot commits carry **no** `Co-Authored-By` trailer — the author field already *is* the attribution — and this overrides the session's own attribution reminder, which asks for one. CI's `commit-trailers` job fails a PR whose bot commits carry one.

| Field | Value |
|---|---|
| App slug | `basecradle-harness-ai` |
| App ID | `3969651` |
| Bot user ID | `290979505` |
| Commit-author | `basecradle-harness-ai[bot] <290979505+basecradle-harness-ai[bot]@users.noreply.github.com>` |

The operational setup for a session that will push or post as the bot — the local `git config`, minting a short-lived installation token with the `gh-app-token` fleet helper, and routing both `gh` and `git push` through it — lives in the **`bot-auth-setup` skill**. Invoke it at the start of any session that will write to GitHub as the bot.

## Fail-Closed Session GitHub Auth

Every rule above is an *instruction*, and an instruction only holds while it is followed. On 2026-09-20 a laptop builder's `picked up — working` comment posted under **@origin's personal account** because `gh`, handed no token, silently fell back to the login stored on the laptop (`basecradle#579` — the fifth occurrence of that class). The mint recipe was fixed the same day; issue #550 removed the *fallback itself*. **A laptop builder session cannot reach @origin's stored GitHub login at all** — a forgotten mint is an error, never a post under his name. Founder-approved (@origin, 2026-09-20, `basecradle-harness#550`); changing it back is a founder decision, not a captain's.

The fence is `.claude/settings.json` → `env`, **committed and git-tracked** so it survives a fresh checkout rather than living in a per-machine file someone has to remember to write. `settings.json` is JSON and carries no comments, so *this section is the authority* on why each variable is what it is.

- **`GH_CONFIG_DIR=/var/empty`** — the `gh` half. `/var/empty` exists on macOS, is root-owned and unwritable, so `gh` finds no `hosts.yml` and can never be logged in there by accident. `gh auth status` reports not logged in, and every API call — **reads included** — exits 4 without a `GH_TOKEN`. That reads mint too is the accepted cost.
- **`GIT_CONFIG_COUNT` / `GIT_CONFIG_KEY_<n>` / `GIT_CONFIG_VALUE_<n>`** — the `git` half, and it is a **separate hole, not the same one**. `GH_CONFIG_DIR` does not close it: `gh auth git-credential` reads the macOS **keyring** directly and hands a bare `git push` @origin's 40-character token *while `gh auth status` in the very same environment reports not logged in* (measured, gh 2.100.0, 2026-09-20). The pair injects a scoped empty reset plus one helper that can only ever hand out `GH_TOKEN`, so a tokenless `git push` quits with exit 128 and names the helper as the reason.

**The two halves are why a single-mechanism fix is not enough, and that is the shape to reason from:** `gh`'s own answer about whether it is logged in is an answer about *one* of its code paths. A fence is proven by what the other end receives, never by what the tool says about itself — which is why the git half is verified with `git credential fill` and `GIT_TRACE` rather than with `gh auth status`.

Three properties are load-bearing, each with an "obviously fine" broken form. **The injected entries are scoped to `https://github.com`** — an unscoped helper answers for *every* host, so a submodule, a redirect, or a mistyped remote the same command touches would be handed a live installation token. **The empty reset comes first, and the environment pairs must be read last** — git reads system, then global, then local config, and only *then* `GIT_CONFIG_*`, which is exactly what lets the reset clear Homebrew's system `credential.helper=osxkeychain` *and* @origin's global `!gh auth git-credential`; move the identical pair into any config **file** and that ordering is no longer guaranteed, so the fence silently stops being first. **An unset `GH_TOKEN` emits `quit=1`**, never an empty password — an unguarded helper answers with exit 0, git sends it, and the push fails with a generic auth error naming nothing: the silent failure again, one layer down.

**Do not reach around it.** `gh auth login`, unsetting either variable, or a `-c credential.helper=…` that installs a different helper each hand the session @origin's account back. The refusal *is* the feature; the fix is always the mint prefix.

**Scope: laptop.** Like `.claude/self-exit.sh` (see "Laptop Builder Self-Exit"), this belongs to the laptop-builder contract and is revisited on migration to the fleet server. `GH_CONFIG_DIR=/var/empty` is harmless there — a fleet box has no stored human login, and a `GH_CONFIG_DIR` naming a directory that does not exist behaves identically (verified: *"not logged into any GitHub hosts"*), so nothing breaks where Linux has no `/var/empty`. The git half makes exactly **one** behavioral change on a fleet box: the NOC registers `gh-app-token --git-credential` as the agent's github.com helper, and the injected reset removes it, so a *bare, tokenless* `git push` there quits instead of auto-minting. Every documented fleet recipe already sets `GH_TOKEN` on the push (`basecradle-noc` `deploy/README.md` §7c), which the injected helper answers identically — so the change is a loud, fail-closed refusal on an undocumented path, never a broken documented one. On migration, delete this section and the `GIT_CONFIG_*` entries, or re-scope them then.

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

## Memory and Charter Discipline

Every agent carries two kinds of standing text into a session, and each has a loader with hard limits. **Memory** is the agent's own file-based store (`~/.claude/projects/<key>/memory/`, keyed on the directory the session was launched from): one fact per file, one index line per file in `MEMORY.md`. **The charter** is the repo's `CLAUDE.md`, loaded whole into every session. Both are stocks of what is still true, never logs of what happened.

- **A store belongs to one agent: the one whose repo root is its key.** Every session launches from the repo root, never a parent or a subdirectory, so the store and the agent stay aligned; nothing enforces it. The first line of every `MEMORY.md` names its owner and its key, in the form `# Memory of <repo> AI — key <launch-dir-key>` (`<repo> AI` is the agent's name, e.g. `basecradle-noc AI`; the key is the store's own directory name under `~/.claude/projects/`). An agent writes only to its own store; a store or a memory found anywhere else is a stray to report, never to adopt or edit.
- **The index is an index, and the loader truncates it.** Claude Code loads `MEMORY.md` only to line 200 and refuses an index over 25,000 characters; memory bodies are never loaded. So: one memory per line, never two packed on a line, every line under 150 characters, and the file under **180 lines and 22,500 characters**. Check: `wc -l MEMORY.md` ≤ 180, `wc -m MEMORY.md` ≤ 22500 (characters, the loader's unit — never `wc -c`, which counts bytes and overstates every em dash), and `awk 'length > 150' MEMORY.md | wc -l` = 0. No file without an index line, no index line without a file.
- **Before saving a memory, ask whether it should be law.** If every session needs it unprompted, it is a pull request to `CLAUDE.md`; if it is a procedure, a recipe, or a reference a session needs only on a trigger, it is a pull request to a skill with a one-line trigger pointer in `CLAUDE.md`; only what is true for this seat and is not law is a memory. A rule written twice drifts.
- **A charter carries only what a session needs without knowing it needs it** — invariants, guards, authority, vocabulary, gotchas that bite silently. Procedure lives in skills. Claude Code warns on an instruction file over about 40,000 characters and on instruction files over 120,000 together; the warning is a measure, not a quota — the test decides what stays, and nothing moves to a skill without the pointer that says when to load it.
- **Maintenance cuts by three tests, in order, with three verbs.** *Already law?* → delete (if the memory says it better, fix the law first, then delete). *Closed arc?* → delete; the issues and git history are the record. *Same family?* → merge. Every item **stays**, **moves** (to a skill, with its trigger pointer), or is **deleted**, and delete beats move. A memory that has become a running log of dated state paragraphs is rewritten as one overwritten current-state paragraph; the history belongs on its issues. A memory whose rule has **verifiably become law** — the captain checks that the law exists and is accurate — is removed on the captain's own authority, feedback included: it was moved, not lost (@origin, 2026-10-22). Proposed deletions of *feedback* memories for any other reason (a closed arc, a judgment call) are listed on the maintenance issue before removal, so the author of those corrections (the founder, for the capital's store; the capital, for a builder's) can strike any; the feedback files removed that way are held 30 days in a dated folder the captain owns (on the laptop, its workspace directory; on a fleet server, under its own home) and the captain removes them on day 30 — a headless captain has no timer, so the capital files it a dated maintenance issue, which wakes it; nobody but the captain touches its home. Project and reference deletions are the captain's own.

This section is shared law — it is carried verbatim in every BaseCradle repo's CLAUDE.md (anchored in the capital; `constitution.md` → Operational Baselines carries the principle).

## Working an Issue

**An issue is a commitment to work, never an escape from it** (`constitution.md` → The Workflow). Filing one does not get you *out* of work; it guarantees the work happens, sooner. A session that closes more issues than it opens is healthy; one that opens more than it closes is a defect.

- **"Do it now" means in this session, not this instant.** Work you discover *while working* an issue is part of finishing that issue — adjust the plan and do it, in-session. Filing a separate issue is legitimate for exactly **two** reasons: you genuinely **cannot** do it now (blocked on someone or something), or it **deserves a fresh context window**. Nothing else. Filing obligates **dispatch, not deferral**: a fresh-window issue is a dispatch order for the capital, reported plainly so that session starts as soon as a slot frees. Ticketing every surprise and shipping only the original issue is how work rots and quality drops.
- **What you see, you say — the passed-by rule.** The bullet above covers work an issue *turned up*; this one covers what you merely *pass by*. A deviation from a documented convention, a settled invariant, or a best practice you notice along the way — a hardcoded value, a missing guard, a stale doc, a silent gap — is said the moment you see it, whether or not it sits inside the issue at hand. Small and safe **inside your own repository** → fix it in-session as its own concern (its own PR, never smuggled into the issue's PR). Larger, unclear, or in another repository → file it or report it upward. Never leave it unsaid because it was not your job.
- **A fix restores intent; an addition changes it — and only a founder changes intent.** Before you act on anything you noticed, ask one question: does this change make the code do what it was *already designed* to do (a wrong type, a guard the design assumed, a stale doc, a broken wire), or does it make the code do something it *never did* (a check that never existed, a resource nothing declared, a design choice reopened)? The first is a **fix**, and the two paths above apply. The second is an **addition**, whatever its size: it is filed with `needs-capital` + `Do Not Work` — the issue stating what it adds, how you would build it, and why — and nothing further happens until it is decided; when in doubt, it is an addition. Deciding additions is the capital's job first: one that answers a *real* problem (it has already happened, or it is inevitable) the capital approves, recording the decision and its reasoning on the issue and releasing it (clearing both labels); one that is hypothetical or genuinely new (a new capability, a new direction, more machinery than the problem demands, a drift from the goal) goes to `needs-human` with the capital's recommendation; when it cannot tell real from new, it is new.
- **Whatever creates, cleans up** (`constitution.md` → How We Build). Something that never gets cleaned up is a defect: every leftover — a backup, a rollback copy, a staged file, a test record, a branch, a container, a temporary timeline — has an owner and a stated end, never past 30 days, and the cleanup ships in the same change. It covers only what we created: an agent's own property (its workspace, scripts, files, repositories, wallets, vaults) is never ours to clean — the hotel's towels, never the guest's suitcase.
- **Finishing an issue means finishing everything it took to get that issue done, sub-issues included.** Before you stop, sweep the related issues: what changed, what else belongs in this context window. **An arc that ends with more issues open than it began is not done.**

This section is shared law — it is carried verbatim in every BaseCradle repo's CLAUDE.md (anchored in the capital; `constitution.md` → The Workflow and How We Build carry the principle).

## Laptop Builder Self-Exit

*Carried verbatim by the laptop builders (basecradle-noc, basecradle-harness, basecradle-router); anchored in the capital, whose repo carries no `self-exit.sh` and whose sessions never run one — the capital's lifecycle is The Capital Drives Projects, and ending a capital session mid-project is its named failure mode.*

You are a laptop builder agent, spawned and supervised by the capital (basecradle AI) through its `launch-builder` skill. The capital is watching this session and stays awake until it ends.

When the issue's definition of done is met in full — including its live verification, as far as your seat can run one (a repo whose work reaches a box only through the NOC proves what it can, and the DoD names the capital for the rest, which is why that issue's `CLOSER:` is the capital) — post your completion comment, then **close the issue only when its `CLOSER:` line names your handle; when it names another User, comment and leave it open — the capital re-wakes you if its verify finds a defect. An issue with no `CLOSER:` line is one you filed yourself (a passed-by defect, your own follow-up): you close it.** Confirm the issue's state by hand afterwards: closed by you, never by a closing keyword's blind match. Then, instead of only printing "safe to exit" and idling, print it and terminate this session:

    bash .claude/self-exit.sh

`self-exit.sh` SIGTERMs only this session's own `claude` process, found by walking its own ancestry, and can target nothing else. **It is never run with a leftover behind it** — a container, an image, a throwaway builder, a Docker Desktop this session started, a port a process started listening on during the session, an unclosed browser tab. The check comes first and has no override: fix it, or stay open and say what is blocking. The NOC's `session-guard.sh` is the reference shape (the script itself refuses on a finding, and every container and image it creates is labelled for attribution); a repo whose `self-exit.sh` does not yet enforce the check runs it by hand before every exit, and adopting the NOC's shape is on its tuning list. The capital observes the session end, runs its own leftovers check, and marks your work complete.

**Laptop-only — removed on migration.** On migration to the fleet server, remove this section and `.claude/self-exit.sh` (and any session-guard scripts and hooks); the router manages server-agent lifecycle — it wakes you on a handoff label, and you neither self-spawn nor self-exit. The self-exit permission is laptop-user-scoped and does not travel to the server. A repo whose migration retires anything else (an auth-fence half, a hook) names it in its own section beside this one, so neither obligation is the one that gets forgotten.

This section is shared law — it is carried verbatim in every laptop builder's CLAUDE.md (anchored in the capital; `constitution.md` → How We Communicate carries the principle: a session's life is its issue's life).

## Laptop Session Guard and Migration (Harness-Local)

**`self-exit.sh` refuses with a leftover behind it** — it runs `.claude/session-guard.sh check`, finds its kill target, and only then `done`. The guard is basecradle-router's fixed shape of the NOC's reference (basecradle-router#358, with the reference's defects listed in basecradle#707), adopted under issue #700 with only its names changed; `tests/test_session_cleanup_guard.py` executes it against stub tools. `begin` is a `SessionStart` hook in `.claude/settings.json`, so the baseline is never a step to remember, and `check` refuses when there is none. While working: `session-guard.sh docker-up` to start Docker Desktop (never `open -a Docker`, which leaves no record that this session started it), `--label basecradle-harness-builder=1` on every container and image, a `basecradle-harness-builder-<name>` buildx builder for any build (a shared builder's cache carries no label we control), `session-guard.sh tab-open`/`tab-close` around a browser tab, `session-guard.sh sweep` when done, and `session-guard.sh tabs-checked` after one `tabs_context_mcp` call; `session-guard.sh check` says where you stand at any point. **Prove it is ours, or report:** the sweep removes only what carries this session's label, and anything else that appeared is reported with the exact command and never touched. **The named residual:** a browser tab carries no attribution the guard can read, so the tab half rests on the record and the `tabs-checked` attestation, never on ground truth.

Removing the shared Laptop Builder Self-Exit section on migration also retires **the `GIT_CONFIG_*` half of the auth fence** — see "Fail-Closed Session GitHub Auth" for what it does on a fleet box and why the `GH_CONFIG_DIR` half stays. Both obligations are named here so neither is the one that gets forgotten.

## Cross-Repo Handoffs

BaseCradle is built across multiple repositories — the private Rails core (the capital), the public SDKs, and the ecosystem repos — each worked on by its own **builder agent** (see "Naming" below). Builder agents cannot reach across repos, so cross-repo work moves as a **handoff**: a self-sufficient issue on the target repo plus a trigger that wakes its agent. This section carries the invariants; **the step-by-step procedure — sending, receiving, delivery mechanics, propagation — lives in the `cross-repo-handoffs` skill (`.claude/skills/cross-repo-handoffs/`). Invoke that skill whenever you send a handoff, and before acting on any trigger beginning `Cross-repo handoff:`.** Both this block and that skill are carried verbatim in every BaseCradle repo (see "Propagation" below).

**GitHub is the sole medium for coordination; a handoff is only a trigger.** Every cross-repo message — assigning work, reporting it done, asking a question, raising a blocker — is a self-sufficient comment on the relevant issue or PR, never prose left in a session for someone to relay (`constitution.md` → "How We Communicate"). Write as though no human is watching the session, because in the end state none is; this holds in both directions — results and blockers are posted to the issue, where the human answers *as a GitHub actor*. **The human is a wake-button, not a mailbox** — never a channel a message passes through. **A terminal lifecycle signal is not a coordination channel**: the substance of any blocker, question, or result must *still* be posted as a GitHub comment (with the routing label when it is a blocker) — terminal prose alone reaches no one.

**A session's life is its issue's life.** An agent runs while its issue is open and sleeps when it closes. On the laptop, agents (the capital included) poll their in-flight issues at the rate-limit floor; on the fleet server, the router re-wakes agents on issue activity — no standing poll. **One issue, one session — always:** the capital dispatches one issue per session, batching only issues it names as one coupled piece of work at dispatch; a captain never adds a second issue to a live session (`constitution.md` → How We Communicate).

**The live protocol — ball-in-court via labels, content via comments.** *Whose move it is* rides on two labels; the substance always rides in a comment. (1) **Pickup** — on receiving the trigger, post a brief `picked up — working` comment under your own bot. (2) **Self-poll** — between work bursts, re-check at the rate-limit floor; never go idle while the issue is open. (3) **Blocked on the capital** — post the blocker and apply **`needs-capital`**; the capital's inbox is the org-wide `needs-capital` query. (4) **Capital answers** in a comment and removes the label. (5) **Blocked on something only a founder can give** (a credential, hands on a machine, a scope or new-repo call) — that is still **`needs-capital`**: say plainly what you need and why you believe only a founder can give it. **No builder applies `needs-human`** — it is the capital's alone, the only signal that pulls Drawk in, and the capital applies it only after it has done the homework and found it cannot settle the matter itself (`constitution.md` → The Workflow, "Escalation is a chain"). He answers with a plain comment and never manages labels from mobile — the capital clears the label when the answer lands. (6) **Done** — verify live, post a completion comment, close the issue by hand. Three routing-label rules bind every session, closing sweeps included: (7) **Touch labels only on the issue(s) you were dispatched to work** — plus a label you apply to park *yourself*, with its blocker comment; a session never re-labels issues it was not working (auditing open issues is the capital's close-out job, not a builder's). (8) **Every label change rides a comment** stating why, posted in the same action — a bare routing label is a page with no message attached. (9) **The comment thread outranks the issue body** for current state — a later comment recording a gate as cleared supersedes a body that still asks for it; never "restore" a label from stale body text. The graph is a **star**: every builder talks to the capital, which routes — builders never coordinate peer-to-peer (repo sovereignty).

**You post on GitHub under your own bot identity — no signature header.** Each agent acts as its own GitHub App bot (`<slug>[bot]`), so the author field already says who is speaking, and the issue's location says who it is for. Do **not** prepend a `sender → recipient` header. Bot identities are not `@`-mentionable — the wake is the App webhook, never a mention. **A platform handle written on GitHub goes in backticks** (`` `@nova` ``, never a bare `@nova`): GitHub reads a bare `@name` as a mention of the GitHub account with that name, which is a stranger, not the User. This covers issues, PRs, comments and commit messages, every handle, and the `CLOSER:` line, which still names its closer inside backticks.

**Paste-text always ends with `---`, set off by a blank line above and below.** Whenever you hand Drawk a block of text to paste into another builder agent, it ends with a blank line, then `---` alone on its own line, then a blank line — the unmistakable boundary between the paste and the conversation. Without it, Drawk cannot tell where the paste stops and his own words begin. This is non-negotiable.

**You work what you were dispatched, and only that.** A captain never selects its own work: not the lowest-numbered open issue, not one an allow-list actor authored, not one it filed itself (`constitution.md` → Earned Autonomy). When your issue closes, your session ends — the capital decides what comes next and dispatches it as its own session. What "only that" means: work *required to finish your issue* is part of it — adjust the plan and do it; a **small, safe defect inside your own repository** that you pass by may be fixed in the same session as its own concern, its own PR ("what you see, you say"); anything with its own definition of done — a follow-up your work created, an addition you filed, a sibling's issue — is a **new issue**, and a new issue is filed and reported upward, never started in this session. Within your one issue, work under standing authorization without pausing for permission you already hold; stop only at a genuine gate you cannot clear yourself: account/credential setup (@origin's), a new-repo or scope decision (either founder's), an ambiguity only the founder who set the direction can resolve, or a publish actuation (the capital's). Every one of those gates is raised to the capital with `needs-capital`, never to a founder directly. Flag real gates unmissably, but never manufacture one.

### Naming

Four forms, four meanings, no overlap: **`basecradle`** (bare, lowercase) — the **repo/codebase**. **`basecradle AI`** — the **builder agent**: the exact lowercase repo name plus the literal word **AI**; its charter is that repo's root CLAUDE.md, and the agent is defined by its charter, not by any single process. **`BaseCradle`** (CamelCase) — the **platform/product**. **`@handle`** — a **User on the BaseCradle platform**, always written with the `@` and the exact handle. **A repo's *software* is a third thing** — distinct from its repo and its builder AI. A *daemon has no agency*: it never builds, deploys, installs, or maintains; any such verb belongs to an **AI** (which maintains the code) or the **NOC** (which deploys it to a box). "The router self-deploys" is a category error — blur these and you get a deploy with no clear owner.

**One slug, everywhere — the universal-identity rule.** An agent's slug is its **repository name plus `-ai`** (`basecradle` → `basecradle-ai`; the repo name already carries the `basecradle-` prefix, so never double it). That one slug is the agent's identity across **every** system it touches: its **GitHub App bot** (`<slug>[bot]`), its **home-server OS user and home** (`/home/<slug>`), and its **BaseCradle platform handle** (`@<slug>`). Never invent a per-system variant. The agent namespace (`… AI`) and the user namespace (`@<slug>`) stay distinct concepts even when they share the slug: a Home Fleet member is never a repo's builder agent, and every Builder Agent is a platform User — fleet membership is a property of a User, so the Builder Fleet is made of accounts (`constitution.md` → Who This Governs, Words Mean Things).

### Repo sovereignty (the governing principle)

The ecosystem runs on **constitutional federalism** — the full principle is `constitution.md` → "Sovereignty and Governance." The operational consequences:

- **Shared law lives at the capital.** `constitution.md` lives in the core `basecradle` repo and is amended only there; it is supreme over every repo's CLAUDE.md, the capital's included. This CLAUDE.md governs **only this repo**.
- **Act only within the repo you are in.** Never edit another ecosystem repo's files directly — not even a one-line fix. Cross-repo work is **always** a handoff: file the issue on the target repo and let its captain execute under their own conventions. **This binds the capital no differently**: its whole-fleet view authorizes it to *coordinate, dispatch, and spawn new repos* — never to reach into an existing one, and never to write another agent's configuration (its settings/allow-list, its CLAUDE.md, its guards), which are the captain's alone (or the founder's, under the emergency reach-in of E1).
- **Read is universal; write is sovereign.** Every fleet agent may **read** any fleet repo — never gated by ownership. Only writing is the boundary.
- **Each repo is captain of its own ship** — sovereign over and accountable for its code, CI, conventions, and CLAUDE.md. **Sovereignty is a standing grant: inside its own repo a captain acts on its own authority and does not pause for permission its charter already grants** — edit, test, open and merge its own green PRs (GitHub-native auto-merge: `gh pr merge --auto --squash` under its own bot identity), converge its own box, file and close its own issues. The only gates reserved upward — **to the capital**: actuating a release/publish and dispatching cross-repo work; **to @origin**: a credential setup or rotation; **to a founder**: a new-repo or scope decision — each reached through the capital, never directly. *Withholding routine in-repo action to seek permission already held is itself the failure mode this rule forecloses.* Shared law changes at the capital and propagates by handoff; a subordinate repo proposes upward, never enacts shared law alone. (The one captain-side exception: an edit that changes the agent's own guards or authority is founder-gated — `constitution.md` → Security and Responsibility.)

### Delivery: label vs. wake (the decision rule)

**The capital dispatches cross-repo work; captains report upward, never peer-to-peer.** A captain that finds work belonging to a sibling surfaces it to the capital — an issue on the core `basecradle` repo — and the capital routes it. Delivery of a handoff is decided by one drift-proof signal — **does the target repo have a `handoff` label?** (`gh label list --repo basecradle/<target-repo> --json name --jq '.[].name'`):

- **Label present → router-wired (on-server): apply the `handoff` label — never paste.** The App webhook fires the router, which synthesizes the trigger itself. **An issue without the label wakes no one — the label is the trigger.** Only @origin (`drawkkwast`) or the capital bot (`basecradle-ai[bot]`) may apply a waking `handoff` label; a sibling captain's label wakes no one.
- **No label → laptop agent: the capital wakes it** via the `launch-builder` skill (a paste prompt handed to Drawk is the manual fallback).
- Private context cannot ride a label auto-wake — a handoff that needs it is relayed by paste even to an on-server repo.

### Sending and receiving — the core rules

**Sending: the issue carries EVERYTHING.** It is the complete, self-sufficient spec — trigger, task, cross-repo state, ordering constraints, definition of done — written for a reader with zero context from the conversation that produced it. The trigger (`Cross-repo handoff: work <issue URL>`) is only the pointer; never put a requirement only in the prompt, and if prompt and issue disagree, the issue wins and the issue gets corrected. **Every capital-authored handoff names its `RIGOR:` and ends its DoD with a `CLOSER:` line naming, by platform handle, the User who closes the issue** — `CLOSER: @basecradle-ruby-ai`, or `CLOSER: @basecradle-ai` when the capital must live-verify before close (these are the literal values the line carries, not fabricated examples). Never a pronoun, never a role: "you" and "the receiving captain" read differently from every seat. The rigor is the whole ceremony the work gets, and the dispatcher sets it (@origin, 2026-10-01): **`RIGOR: mechanical`** — a byte-verified copy, a version bump, a one-line config change, a rename: no review subagent, no proof runs, the `diff` or the test suite is the verification, a two-line close. **`RIGOR: standard`** — a normal fix or small feature with tests: one self-review, tests, one live verify, a short report. **`RIGOR: critical`** — security, anything touching credentials, auth, a publish path, or shared law: the full treatment — review, mutation or proof runs, live verify, full report. A handoff with no `RIGOR:` line is a stamping error; ask via `needs-capital`, never guess upward. Full procedure: the `cross-repo-handoffs` skill.

**Receiving: on any trigger beginning `Cross-repo handoff:`, read the referenced issue(s) in full before acting, and invoke the `cross-repo-handoffs` skill.** Execute under **this** repo's conventions — the sending repo's do not transfer — and **to the handoff's `RIGOR:`, never more**: ceremony above the tier is spend, not care. A purely mechanical, byte-verified change — a verbatim re-sync from the capital's canonical confirmed identical by `diff` — is exempt from self-review: the byte-match is the verification, and the exemption never applies to an authored change. A finding during the work that would push the change *up* a tier (a mechanical edit that turns out to touch auth; a standard fix that exposes a secret) is reported on the issue and the dispatcher re-tiers it; a captain may escalate, never downgrade. When done: post the completion report as a comment on the originating issue, **verify your own work against the live system** (not merely green CI), and **close the issue yourself, by hand, when its `CLOSER:` line names your handle; when it names another User, comment and leave it open for them** (a capital-originated handoff with no `CLOSER:` line is a stamping error — ask via `needs-capital`, never guess). **Never auto-close a handoff issue with a closing keyword** — GitHub's detector is a blind literal match anywhere in the PR title, body, or squashed commit message (even negated or in backticks), and it fires at merge, *before* live verification. Never write the literal token; refer to it in prose as "a closing keyword."

### Propagation

Ten capital-anchored shared artifacts are carried verbatim across the fleet (this paragraph is their one inventory; the NOC's drift guard also carries one NOC-anchored text, the Your Home section of basecradle-noc#832): the **Founder Authority**, **Cross-Repo Handoffs**, **Polling GitHub**, **Attended-Session Lifecycle Signal**, **Memory and Charter Discipline**, **Working an Issue**, and **Laptop Builder Self-Exit** CLAUDE.md blocks, the **`cross-repo-handoffs` skill**, the **needs-human phone-alert stub** (`.github/workflows/needs-human-alert.yml`), and the **Dependabot auto-merge stub** (`.github/workflows/dependabot-auto-merge.yml` — the merge policy itself lives in the one reusable workflow in `basecradle/.github`; patch/minor auto-merge on green, major → `needs-capital`). Carrier sets differ per artifact: every builder repo carries nine of the ten; **Laptop Builder Self-Exit** is carried only by the laptop builders (basecradle-noc, basecradle-harness, basecradle-router) and removed from a repo on its migration to the fleet server; `basecradle/.github` — the org profile repo, which hosts the reusable workflows the stubs call but has no builder agent, hence no CLAUDE.md and no `.claude/` skills — carries only the alert stub (the Dependabot stub is deliberately absent there: the repo has no CI and hosts the fleet paging path, so its bumps stay capital-reviewed). Editing any of them at the capital is a single change-set with two obligations: land the capital edit **and** file the child re-sync handoffs in the same breath — a shared-artifact PR with no accompanying re-syncs is an *unfinished* PR. The NOC runs a standing drift-guard that byte-diffs every shared artifact across its carrier set against the capital canonical every 15 minutes and files a `[DRIFT]` issue when a divergence outlives the ~30-min grace window. A repo missing any artifact it should carry (always true for a brand-new repo) gets them copied from the capital's canonical on GitHub (`gh api repos/basecradle/basecradle/contents/...`, with fleet credentials) — never a machine-local path. Full mechanics and the on-demand audit: the `cross-repo-handoffs` skill.

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
  .declared.json       # bookkeeping: the agent's *declared* capability set (issue #374)
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
  defaults refresh. A `.new` is retired once moot (its file matches the default, is gone,
  or lost its default) — and only while it still hashes to the pre-run manifest entry, so
  an operator-edited `.new` is never removed (issue #526).
- **Charter sourcing.** The Turn-0 operator charter is composed from
  `prompts/system-prompt.md` + `prompts/initialize.md` (HTML comments, which are
  operator-facing notes, stripped). `HARNESS_SYSTEM_PROMPT` is a **legacy fallback** only,
  consulted when the config home was never installed. Onboarding (the Dashboard
  orientation) still composes on top — the *source* changed, not the composition.

The install/upgrade **procedure** — the installer's per-file conffile-upgrade logic, granting
powerful tools with `--opt-in`, the loud grandfather report, and the per-agent fleet rollout —
lives in the **`config-home-install` skill**. The Phase-2 build history is spent provenance and
lives in **`docs/harness-internals.md`** — not the charter.

### Green-While-Absent: the declared set is proven, never assumed (Recurrence Guard)

The install/upgrade layer is written in **observations** — a file is present, a hash matches — and an observation cannot tell an operator's deliberate deletion from a capability something **stripped**. An overlay tool gone, a config home never reconciled after `pip install -U`, a vendor tranche pruned by a converge that ran the installer without the agent's `AI_PROVIDER` (it then filters for `openai`) — none raise, none log, none stop the agent. **Absence emits no signal**, so nothing but a prover will ever produce one.

The floor, and it is not negotiable: **no capability may be claimed without something that turns red when it is absent.** `install` writes the claim (`.declared.json`); **`basecradle-harness-verify`** (`_verify.py`) proves it and exits nonzero with a specific diagnosis; **`basecradle-harness-claims`** prints the rows for the fleet's claims-vs-evidence ledger (Claims Manifest Contract v1 — the **NOC owns that contract**; amendments go up through the capital). Four properties are load-bearing: it proves the **declared** set, never pristine-ness; **unproven is red** (`verify` exits nonzero with no config home, an unparseable declaration, or a `contract` it does not know); **a granted power tool's absence is a strip, not a decision** — the one deliberate exception to the conffile rule's never-resurrect (a benign default's absence is still honored as the operator's deletion) — it is healed (`RESTORED`, loudly) and `--revoke-opt-in` is the only withdrawal; and **the claims are emitted whatever the verdict** (`basecradle-harness-claims` takes no arguments and exits 0 even with no config home — nonzero means only *the claims cannot be stated*). It deliberately does not prove *activation*; `--resolved-config` is the authority there. **A new kind of managed state needs a declaration entry and a check, or it is a capability nobody can see the absence of.** The reasoning behind each property, and the broken form each one forecloses, is in the **`config-home-install` skill**.

### Security invariants (never relaxed, never moved to a skill)

- **Powerful tools fail closed — opt-in on every provider (issue #168).** A powerful/dangerous
  tool — media generation (image, **video**, audio), web/X search, code execution,
  **self-authorship** (an agent editing its own `system-prompt.md`, issue #241), a **full
  shell** (arbitrary on-box command execution as the agent's OS user, issue #252), an
  **account/billing read** (issues #179, #425), and an
  **off-platform push to a human's phone** (issue #341) — is **off by
  default on every provider** and activates **only** when explicitly dropped into an agent's
  `tools/` overlay (the same "ships empty" stance as `mcp/`). The powerful defaults (by plugin
  stem): `generate_image`, `edit_image`, `web_search` (OpenAI), `xai_search`
  (xAI `web_search`/`x_search`), `openrouter_search`, `google_search` (issue #656 — Gemini
  grounding as a harness-run `web_search` tool, because Vertex refuses a search tool beside function
  declarations), `url_context` (Gemini reading pages itself, #656), `code_execution` (OpenAI, xAI
  and Gemini variants), `grok_generate_image`,
  `grok_edit_image`, `grok_generate_video`, `xai_account_balance` (issue #179 — xAI Management
  API credit read, `Vendor("xai")`-gated because it reads an *xAI* account),
  `openrouter_account_balance` (issue #425 — the OpenRouter mirror, and the one account tool with
  **no** vendor gate: its Management key is dedicated and provider-independent, so gating it on
  `Vendor("openrouter")` would self-exclude the very case it was built for — an agent brained by
  another provider that holds a separate OpenRouter account), `system_prompt` (the read+edit
  self-authorship pair — the most powerful, and as of #241 enabled on **no** agent; enablement is
  a per-agent founder decision), `send_direct_message_to_origin` (issue #341 — an ntfy.sh push notification
  to @origin's iPhone, leaving the platform entirely; powerful because it *interrupts a human*
  rather than because it touches the box, and gated on its own `NTFY_DM_TOKEN` so an agent
  provisioned without the credential never sees it), and `shell` (issue #252 — full command-line
  access; **doubly gated**, the
  only opt-in default that *also* declares the `SHELL` policy capability, so it loads only for an
  agent that opts it in **and** runs `Policy.unlocked()`; every other powerful tool loads under
  the locked profile once opted in). **The `unlocked()` profile is selected at deploy time by
  `HARNESS_PROFILE=unlocked`** in the agent's `agent.env` (issue #256, founder-approved 2026-07-06):
  read once at wake (`_profile_from_env`) and threaded into **both** the registry (`Harness(policy=…)`)
  and the env-resolution filter (`_apply_safe_policy`) so the two always agree, it is the env-driven
  counterpart to passing `Policy.unlocked()` in the library API — how gate 2 is delivered in a
  deployment (the router carries no profile logic). It is **fail-closed**: unset, empty, `locked`, or
  any unrecognized value → `Policy.locked()`, the shipped default unchanged, so a typo can never
  silently unlock a box. It is *only* the deploy lever — the safety stays enforced around it (the NOC
  sets it only after its `verify_unprivileged` preflight passes; the root backstop below still fires) —
  and `--resolved-config` reports the resulting **`active_profile`** so a shell-class enablement is
  verifiable on-box, not assumed. `shell` additionally carries an **in-process root backstop**
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
  agents whose persona is adversarial by design (the Home Fleet's `@pinky`/`@the-brain`) must be
  tool-less **by construction**, never "on unless someone remembered to prune"; any provider/SDK-based
  default would silently arm whoever moves onto that provider next. On upgrade, a powerful tool a prior
  version already scaffolded is **kept, never silently stripped, and reported loudly**. *(Decided
  by the capital + founder. Granting/pruning mechanics: the `config-home-install` skill.)*

  **One capability left this set by founder decision, and it is recorded here rather than left to
  be inferred from an absence** (issue #484, 2026-09-09). `hear_audio` was an opt-in powerful tool
  — a provider-call media tool — and audio transcription is now the assets tool's **`listen`
  action**, on the benign default `assets` plugin. So a *configured* agent (an `openai` provider
  with `AI_API_KEY`) has it without an overlay grant, where before it needed one. The gate that
  remains is **configuration**, not opt-in: `assets_options` reads the same `OpenAIKey()`
  requirement the retired plugin declared, so on every other provider the action is absent from
  the schema entirely (the ruling's "don't show a locked door"). This is the founder's ruling on a
  **capability decision**, which is a founder decision to make; it is written down because a
  silently-widened capability is exactly what the opt-in rule exists to prevent, and the difference
  between a decision and an oversight is whether it is stated. `view` and `watch` are unaffected —
  they were never powerful (no provider call, nothing spent, in-process decode).

  **And one MCP tool is withheld by founder decision, recorded the same way** (issue #553,
  @origin, 2026-09-22: *"approve both (for now)"*). Playwright MCP's
  `browser_run_code_unsafe` runs arbitrary JavaScript in the playwright-mcp **process**, not the
  page: code execution as the agent's OS user, around the `shell` opt-in and the NOC's
  `verify_unprivileged` gate. So every server withholds it unless its `mcp/<name>.json` says
  otherwise (`_mcp.WITHHOLDABLE`, the one place the reason, decider and date live, all of which the
  agent is told). Two properties are the ruling, not details: the list is **per-agent
  configuration** the NOC declares (`withheld_tools`, `[]` = hand it back — *"@briggs is the
  exception, meaning if he wants access, he gets it"*), never a constant, so a founder's waiver is
  an inventory edit and no release; and **both states are disclosed** — withheld, with why and the
  waiver sentence when `withheld_waivable`; or handed back, with why agents do not get it by
  default. Only `WITHHOLDABLE` names are accepted, so withholding a new tool needs a stated reason
  in code review, and a config naming anything else fails closed (the server does not load) and
  visibly (its stem stays in `mcp_servers`, its reason lands in `skipped`).
- **A credential an object holds is a `Secret`, never a plain attribute** (issue #599, the class
  from basecradle#612). A key in `__dict__` is emitted by `vars()`, by a crash reporter that expands
  locals, by `json.dump(o, fp, default=vars)` (it writes before it raises), and by `pickle`; a
  dataclass field is emitted by its generated `repr` too, which is how `McpServerConfig` printed an
  operator's MCP tokens. `_secret.Secret` has no `__dict__`, renders `[REDACTED]`, refuses pickle,
  and is revealed only on the line that sends it. **A new attribute holding a credential needs a
  `Secret` and a row in `tests/test_secret.py`'s `HOLDERS`**, or it is the same leak again.
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
  `marks/`, `seen/`, `claims/`, `breaker/`, `billing/`) but **never touches** `memory.db` (+ `-wal`/`-shm`) or
  the MemPalace palace dir — memory is the agent's durable mind and outlives any timeline. The
  classify switch is the whole safety: **only a clean `NotFoundError` (404) purges**; success (200)
  and `Forbidden`/`NotAViewer` (403) keep, and **any** transient error (connection / rate-limit /
  5xx) keeps and retries next run. A platform outage must never read as "everything deleted" and
  trigger a mass purge — default to keep on anything but a 404. The sweep's **stranded-temp pass**
  (issue #526) is the one part that runs on live timelines, and it is bounded the other way: it
  removes only the harness's own staged-write temps, matched by exact name in the places they are
  staged, once older than `STRANDED_AFTER` **and** (when the writer stamped one) with a dead pid —
  so a live writer's temp is never touched, and the palace is never walked. *(Mechanics:
  `docs/harness-internals.md`.)*
- **The OS fences the sweep too, and a refused write fails the run** (issue #536, founder-approved
  2026-09-19). The fleet unit sandboxes it to exactly the three directories this module writes —
  `ProtectHome=read-only` plus `ReadWritePaths` for `$HARNESS_HOME`, the config home, and
  `-/home/%i/.mempalace` — so the boundary above ("only what it staged or wrote") is enforced by
  the operating system and not only by an allow-list of filenames. **`ProtectHome=true` is never
  right** (it makes `/home` empty and blinds both the sweep and the founder's read-everything
  supervision), and narrowing `ReadWritePaths` means nothing under `ProtectHome=false`. **The
  fence is around the *home*, never around every file the code declines to touch:** `memory.db`
  and (by default) the palace sit *inside* `$HARNESS_HOME`, so the code remains the only thing
  keeping the sweep off them — the unit narrows a bug's blast radius, it does not make one
  impossible. **Being wrong must be loud:** every removal the sweep decides to make and cannot is
  logged at **ERROR naming the path** (`_note_blocked`) and **exits the run non-zero**, one
  refusal never aborts the rest of the sweep, every `OSError` on a removal counts
  (errno-agnostic), and **a removal we cannot confirm is a removal we do not claim** (`_present`).
  **Only a refused *write* is judged:** a refused read, an advisory lock the filesystem will not
  give (`LockUnavailable`), a path already gone, and an unparseable watermark stay a `WARNING`
  and a clean exit. **A run with nothing to remove attempts no write**, so a green run is never
  proof the `ReadWritePaths` list is right. **Load the `orphan-sweep` skill before editing
  `_cleanup.py`, the sweep's exit status or log levels, or its unit's sandbox** — it carries the
  NOC's systemd 255 measurements (basecradle-noc#736), the reason behind each exclusion, and the
  two stdlib traps (`Path.is_dir`'s split across the version matrix, `shutil.rmtree`'s abandoned
  walk) a refactor will undo.


## Development Commands

```bash
uv sync                  # install everything (creates .venv)
uv run pytest            # tests (offline — the default)
uv run ruff check .      # lint
uv run ruff format .     # format
uv build                 # build the wheel + sdist
basecradle-harness-install --config-home <dir>   # scaffold/upgrade a config home
uv run --extra mempalace pytest -m mempalace       # real-palace tests (Python 3.11+; they run under a throwaway HOME)
uv run scripts/isolated_home.py <command>        # any off-fleet run that writes a palace: a palace check, a local wake
```

**An off-fleet run that writes a palace runs under a throwaway `HOME`** (issue #630). MemPalace keeps one lock file per palace in `$HOME/.mempalace/locks` and never removes it, and a wake publishes its palace into `$HOME/.mempalace/config.json`; on a fleet box that is the agent's own home, on the laptop it is @origin's. `scripts/isolated_home.py` runs one command with `HOME` pointed at a temporary directory (sharing only the embedding-model download cache) and removes it afterwards, and `conftest` does the same for every test marked `mempalace`.
