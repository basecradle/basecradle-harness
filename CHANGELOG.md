# Changelog

All notable changes to BaseCradle Harness are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.31.0] - 2026-06-21

**Current-time grounding on every wake.** A live test surfaced that a Grok/xAI-backed persona
answered "what is the current time?" confidently wrong (~7 hours off) while an OpenAI-backed one
answered to the second — because the harness injected **no** current-time grounding anywhere, so
temporal accuracy rode on whichever provider happened to surface the date in its own server-side
context. Fixed harness-side, generically, so it no longer depends on provider quirks. Both changes
are additive and backward-compatible; v1 is UTC-only (the model converts to a local zone when a
peer names one).

### Added

- **A current-time anchor at the head of every wake's brief.** `compose_brief` gains an optional
  `now` part, placed first; `_wake.py::_now_line` renders it as
  `Current Time: 2026-06-21 17:09:49 UTC (Sunday)` (Title Case label, absolute UTC, day-of-week,
  no trailing period). The brief is already re-composed and re-injected each wake, so the anchor
  is always current — no new freshness machinery.
- **A `[created_at]` timestamp on every inbound item the agent perceives** — messages, assets,
  webhook events, and activated tasks — uniformly, using each item's own `created_at`, so the
  model can reason about an item's age against the anchor. (A task's item `created_at` is its
  *activation* moment ≈ now, consistent with every other item.) The agent's own posts stay
  unstamped.

## [0.30.0] - 2026-06-17

**Eddie Murphy — the xAI-native profile: Live Search + grok media tools.** The harness's
"done-bar" acceptance work: a fully-xAI persona whose stack touches no OpenAI surface — not the
provider, not the key, not the tools. Built under the tool-building discipline (learn the full
surface → decide coverage deliberately → split by operation → test every built option).

A framing correction shaped Part A: the handoff anticipated a brand-new *native* adapter
driving Chat Completions `search_parameters`, but xAI **deprecated `search_parameters` on
2026-01-12** in favor of server-side search **tools on the Responses API**. So there is no new
adapter class — the `xai` profile reuses `OpenAIResponsesProvider` (the "OpenAI" in the name is
the *wire format*, not the vendor; xAI's API speaks the Responses wire) pointed at `api.x.ai`,
and Live Search is delivered by xAI's server-side `web_search` / `x_search` built-ins. xAI's
Responses API returns OpenAI-style `url_citation` annotations, so the existing citation parsing
already grounds Eddie's answers in sources unchanged.

### Added

- **`AI_PROVIDER_API=xai` — the xAI-native profile.** Selects the Responses adapter defaulted to
  `https://api.x.ai/v1` (override with `AI_PROVIDER_BASE_URL`), and is the activation
  discriminator that turns xAI's Live-Search built-ins and the grok media tools **on** while
  turning the OpenAI-coupled tools **off** — so an xAI agent (grok-4.3 chat) gets a clean,
  all-xAI stack by construction. BaseCradle tools compose under it unchanged.
- **xAI Live Search built-ins (`web_search` + `x_search`).** Two default built-in plugins
  (`_defaults/tools/xai_search.py`), gated on the `xai` profile: grok searches the live web and
  𝕏 itself and returns sourced, cited answers. Disable either by deleting its plugin line; the
  `web_search` name coexists with OpenAI's Responses built-in (different `requires`), so exactly
  one activates per config.
- **`grok_generate_image`** (`_grok.py`) — text → image via xAI's Images endpoint
  (`grok-imagine-image-quality`). Optional `aspect_ratio` / `resolution` pass-throughs; the
  default call is the always-valid core (`model` + `prompt` + `response_format=b64_json`, with a
  `url`-encoded fallback). `n>1` deliberately skipped (founder decision, as for the OpenAI tool).
- **`grok_generate_video`** (`_grok.py`) — the harness's **first video capability**. Text→video
  **and** image→video (`image` = a source Asset uuid, resolved to a blob URL for xAI's
  `image_url`). xAI's video endpoint is **asynchronous**: the tool submits, polls
  `GET /v1/videos/{request_id}` until `done`, then downloads the clip and uploads it as an Asset
  that renders inline. Full `duration` / `aspect_ratio` / `resolution` coverage.
- **Activation:** the grok media tools require the `xai` profile (`ProviderAPI("xai")`), so they
  self-exclude off any non-xAI config. (The honest discriminator: the API key var is shared
  across vendors, so the *profile* — not a key-presence check — is what distinguishes xAI.)

### Changed

- **Shared media plumbing factored into `_media.py`** — the vendor-neutral bits the OpenAI image
  tools and the grok media tools both need: the legible provider-error relay (Principle 5),
  magic-byte format sniffing (so an uploaded Asset's extension follows the *real* bytes — the
  hard-coded-`.png` bug generalized away), and safe-filename building. `_images.py` now delegates
  to it; behavior is unchanged (confirmed by its existing tests).
- **`OpenAIKey` now also excludes the `xai` profile.** The OpenAI-coupled tools (`generate_image`,
  `edit_image`, `listen`) self-exclude under `AI_PROVIDER_API=xai`, so Eddie's stack carries no
  OpenAI tools by construction rather than by operator curation. Behavior under `chat`/`responses`
  is unchanged.

### Boundary

- Offline tests assert the harness's half (params sent, the async poll loop, the legible error
  relay, sniffed filename extensions). The ground-truth checks — a real measured-dimension video
  file, the posted Asset's actual pixels/content-type, Live Search returning real citations — are
  **the capital's live `@jt`/Eddie verification**, which provisions Eddie (xai profile, grok media
  tools, BaseCradle tools, no OpenAI tools), runs the full matrix, and **closes the handoff by
  hand** after that live verify.

## [0.29.1] - 2026-06-17

**Image tools — two fixes from the capital's live `@jt` verification of 0.29.0.** The
jpeg/webp/edit/size coverage shipped in 0.29.0 was confirmed correct against ground truth;
re-running the full matrix caught two issues, both in the shared `_ImageTool` base.

### Fixed

- **`output_compression` no longer breaks png.** OpenAI hard-rejects `output_compression` on
  png output (`HTTP 400 invalid_png_output_compression`), and png is the default format — so a
  model that filled in the schema field (it does, freely) made **png generate and edit fail in
  practice**. The shared coverage builder now **drops `output_compression` when the format is
  png or unset**, where the API ignores it anyway — turning a live footgun into a no-op rather
  than trusting the model to avoid it.
- **Image-API errors are now legible.** A provider failure reached the model as a generic
  `Provider returned HTTP 400`, stranding it with an opaque status it couldn't relay. The tools
  now surface the provider's **actual** message from the response body (e.g. *"Compression less
  than 100 is not supported for PNG output format"*), so the AI relays the true cause to the
  user — fail loud *and* legible (the tool-building discipline's Principle 5).

## [0.29.0] - 2026-06-17

**Image tools — full `gpt-image-2` coverage.** The media tranche brought to the model's full
surface, and the first build under the tool-building discipline (learn the full surface →
decide coverage deliberately → split by operation → test every built option). Two things,
resolved together because they're the same surface: the harness could *generate* an image but
not *edit* an uploaded one, and `generate_image` silently couldn't emit jpeg/webp (it
hard-coded the `.png` filename, so a "save as JPG" request produced `image/png`).

### Added

- **`edit_image`** (`basecradle_harness.EditImageTool`) — a new default tool over OpenAI's
  `/v1/images/edits`: edit one or more existing image Assets with a prompt (recolor, restyle,
  composite). It resolves each source Asset by uuid and sends its **bytes, not a URL** (the
  endpoint rejects URLs), with an optional `mask` Asset whose alpha channel marks the region
  to change, and posts the edited result as a new Asset — exactly like `generate_image`. A
  `PlatformTool` requiring `OpenAIKey()` (it self-excludes with no OpenAI key), so it composes
  under both the Chat and Responses providers and appears in the Turn-0 manifest.
- **Full shared coverage on both image tools** — `quality` (low/medium/high/auto),
  `background` (opaque/auto — `gpt-image-2` has **no** transparent), `output_format`
  (png/jpeg/webp), and `output_compression` (0–100, jpeg/webp only), alongside the existing
  `size`. Enum/range constraints are documented in the schema and enforced by the API, not
  re-validated in the harness, so coverage never drifts as the model's surface evolves.

### Fixed

- **`generate_image` no longer hard-codes `.png`.** The posted Asset's filename extension now
  follows `output_format` (png → `.png`, jpeg → `.jpg`, webp → `.webp`), so its content-type
  follows too (the server infers the type from the filename). A jpeg/webp request now actually
  produces a jpeg/webp.

### Notes

- **`n>1` is deliberately skipped** on both tools — multiple-images-per-call is niche for a
  conversational agent (founder decision).
- The offline tests assert the harness's half of the contract (the params sent, the filename
  extension posted); the ground-truth checks — the posted Asset's actual pixels / content-type
  / file magic — are the capital's live @jt verification.

## [0.28.0] - 2026-06-17

Phase 2 · **Group 6** (the last group) — **the cross-wake circuit-breaker.** A per-timeline
self-breaker that is the generic backstop for an *unknown* runaway wake loop: the agent is
woken, some side effect posts, the post fires a platform event, the router wakes it again →
a tight cross-wake cycle burning provider tokens and box resources. The in-wake `max_steps`
cap, the actor self-filter, and the known B3/B8 fixes each stop a *specific* loop; this
backstops the novel one — most plausibly introduced by a custom `tools/` plugin (Group 2) or
a drop-in MCP server (Group 5). This is the **harness layer** of a two-layer, two-repo
defense; [`basecradle-router`](https://github.com/basecradle/basecradle-router) carries the
complementary **cross-agent** breaker. The two are independent — no shared protocol, each
trips on its own view, together defense-in-depth.

### Added

- **`WakeBreaker`** (`basecradle_harness._wake`) — a rolling-window rate limiter on **wakes
  per timeline**, persisted under `HARNESS_HOME` beside the `marks/`/`seen/`/`claims/` stores
  (`breaker/<timeline>.wakes` holds the windowed wake timestamps; `breaker/<timeline>.tripped`
  is the durable trip marker), so it survives the process-per-wake model. `record_and_check`
  records each wake and returns a `BreakerDecision`.
- **Trip → self-decline, token-free.** Over the cap within the window, the wake **self-declines
  before the session is loaded or the model is ever engaged** — **no provider call** — posts a
  single loud alert to the timeline and logs at `WARNING`. The alert fires **once**, on the
  trip *transition* only (the durable marker is the one-time guard, so the alert never loops;
  the actor self-filter keeps the agent from waking on its own alert). Every later wake for a
  tripped timeline keeps short-circuiting.
- **Auto-reset (the preferred reset).** Once the burst subsides — the window clears back under
  the cap **and** the cooldown has elapsed since the trip — the breaker clears the marker,
  restarts the window, posts a recovery note, and resumes normal operation. A transient
  runaway self-heals while the loud alert still leaves a human a breadcrumb; clearing the trip
  marker by hand is the equivalent operator reset. A short-circuited wake is recoverable — the
  cursor-paginated read API is the source of truth, so the next healthy wake reconciles
  anything missed.
- **Generous, tunable defaults.** Default **10 wakes / 60 s** per timeline — deliberately
  generous so legitimate multi-peer activity never trips it (a genuine runaway fires
  continuously and blows past the cap; the agent's own posts are self-filtered and never wake
  it, so only inbound items count). Tunable via `HARNESS_WAKE_BREAKER_MAX` /
  `HARNESS_WAKE_BREAKER_WINDOW` / `HARNESS_WAKE_BREAKER_COOLDOWN` (cooldown defaults to the
  window); a cap of `0` (or below) disables the breaker entirely (the operator escape hatch).
  Wired on by construction for every `WakeAgent`, env-tuned via `WakeAgent.from_env`. The
  poll-loop `TimelineAgent` is unaffected — the breaker is a wake-mode property. `WakeBreaker`
  and `BreakerDecision` are exported.

## [0.27.0] - 2026-06-17

Phase 2 · **Group 5** — **MCP drop-in + safe-by-default made explicit.** The harness
becomes an [MCP](https://modelcontextprotocol.io) **client**: drop a server config into the
config home's `mcp/` dir and that server's tools become part of the agent's active tool set
on the next wake — no code change, the same "everything in the folder is active" model as
the `tools/` overlay. And the harness's safe-by-default posture is made **explicit**: it
ships with no MCP servers and a policy that denies shell; adding a server (or a custom tool
that needs a denied capability) is a deliberate, surfaced opt-out — "all bets off," stated
and auditable, never silent. This reverses the earlier "MCP is out of scope" stance (a
founder decision).

### Added

- **The harness as an MCP client** (`basecradle_harness._mcp`) — a small, synchronous
  JSON-RPC client over **stdio** (a spawned subprocess) or **Streamable HTTP**, with no new
  dependency (httpx comes via the SDK; stdio is stdlib). It handshakes, `tools/list`s, and
  proxies `tools/call`. Each discovered tool is exposed as a plain function `Tool`
  (namespaced `<server>__<tool>`), so it composes under **both** the Chat and Responses
  providers and appears in the generated Turn-0 manifest like any other tool.
- **The `mcp/` overlay.** One server per `mcp/<name>.json`, following the **standard MCP
  config shape** (stdio: `command`/`args`/`env`; HTTP: `url`/`headers`; a single-entry
  `{"mcpServers": {…}}` wrapper is unwrapped) so a published server's snippet drops in
  unmodified. Drop-to-add / delete-to-disable, consistent with the `tools/` overlay. `mcp/`
  ships **empty** (safe by default), so there is nothing for the conffile upgrader to
  reconcile and an operator-added file is never touched.
- **Safe-by-default opt-out surfacing.** Loading an MCP server is surfaced — a **log line**
  and an **opt-out notice** rendered into the persistent Turn-0 brief
  (`ResolvedTools.notices` → `render_safety` → `compose_brief`), so "this agent has left the
  safe-by-default zone" is stated and auditable. The same surfacing covers a drop-in
  `tools/` tool the locked policy refuses, which is now **filtered out and surfaced**
  (`_apply_safe_policy`) rather than crashing `Harness` construction.
- **`HARNESS_MCP_TIMEOUT`** — the per-request timeout bounding a slow/hung MCP server (so it
  degrades to a skip or a tool error, never a stalled wake). Defaults to 20s.

### Changed

- **Safe by construction stays a policy property.** An MCP proxy tool carries no in-process
  capability, so it registers under the locked policy (the opt-out is *surfaced*, not
  refused); a `tools/` tool that declares `SHELL` is still denied — the activation-vs-policy
  split is preserved, and the policy is never bypassed by mere activation.
- A failed/missing MCP server **self-excludes** (its tools are skipped with a reason),
  never a hard wake failure — the Group-2 activation robustness bar, extended to MCP.
- **`CLAUDE.md`** — the "MCP is out of scope / deferred" stance is **reversed** to the new
  rule (MCP via the `mcp/` drop-in; safe-by-default with no servers; adding one is a
  surfaced opt-out), with a new Group-5 section and updated config-home/upgrader docs.

### Known bounds

- An MCP **media** result (image / embedded-resource content blocks) renders as a text
  marker, not model-vision input — out of scope here.
- A stdio server is spawned **per wake** (process-per-event model), adding its handshake +
  `tools/list` latency to each wake that has MCP configured; with `mcp/` empty (the default)
  a wake pays nothing. A pooled/long-lived server is a possible future optimization.

## [0.26.0] - 2026-06-16

Phase 2 · **Group 4** — **pluggable memory.** The leading memory systems
(Mem0/Zep/MemPalace/Letta) are *middleware*: they observe the conversation to
auto-capture facts and inject prompt-ready context before the model runs — not just
`write(key, value)`. The shipped default (a `MemoryTool` fused to SQLite) had no seam for
that. This group builds the seam and ships a real MemPalace reference adapter to prove it
end-to-end, while leaving the default's behavior exactly as it was.

### Added

- **The `MemoryProvider` interface** — four *optional* surfaces: **tools** (model-facing
  ops), **store** (the durable engine), **`observe(exchange)`** (a wake-loop hook fired
  after each exchange, for auto-capture), and **`context(scope)`** (a Turn-0 hook returning
  prompt-ready memory to inject). `observe`/`context` default to no-ops. Scope is the
  **agent identity** (timeline as metadata), so memory is the agent's one private mind
  spanning all its timelines — the basis for cross-timeline recall.
- **`SqliteMemoryStore`** — the five durable ops (write/read/list/delete/search) split out
  of `MemoryTool` as a standalone engine a provider's hooks can read and write.
- **`SqliteMemoryProvider`** — the default: `MemoryTool` over a private host-local
  `SqliteMemoryStore`, with `observe`/`context` as no-ops. Behavior-preserving — an agent
  on it has exactly the explicit, write-it-yourself memory it had before the seam (@jt
  unchanged).
- **The observe/context wake hooks.** A `WakeAgent` fires `observe` after each real
  exchange and injects `context` into the persistent Turn-0 brief (relevant to the turn —
  the incoming text is the retrieval query). Both degrade gracefully: a raising hook is
  swallowed and **never breaks the wake**.
- **Provider selection** via `HARNESS_MEMORY_PROVIDER` — `sqlite` (default), `mempalace`,
  or a dotted `module:Class` path to any custom `MemoryProvider`. One provider per agent.
- **The MemPalace reference adapter** (`basecradle-harness[mempalace]`, an optional extra)
  — a real `MemoryProvider` over MemPalace's local library API: `observe` mines each
  exchange (`convo_miner.mine_convos`), `context` retrieves top-K relevant chunks
  (`searcher.search_memories`) across all timelines. Supplies no model-facing tool (memory
  is automatic). Uses the library API, **not** MemPalace's MCP tools (that path is Group 5).
- **`memory` block in `compose_brief`** — the recalled context is injected just before the
  charter, the way middleware memory systems inject retrieved context before the system
  prompt. Defaults to absent, so the four-part brief is unchanged when there is no memory.

### Changed

- **`MemoryTool` is now a thin surface over a store.** `MemoryTool(path=…)` works exactly
  as before; `MemoryTool(store=…)` shares a provider's store. The model-facing behavior and
  every response string are unchanged.
- **Memory graduated from a tool plugin to its own provider subsystem.** The
  `_defaults/tools/memory.py` plugin is removed; the memory tool now comes from
  `memory_provider.tools()` and is folded into the resolved set (deduped by name, so a
  config home that predates this still works). The manifest still lists `memory`, so the
  persistent brief is unchanged.

## [0.25.0] - 2026-06-16

Phase 2 · **Group 3** — `initialize.md`: the **persistent operating brief**. Turn 0 stops
being a one-time onboarding seed (Group 1's field-scrape, which aged into the distant past
of a long transcript) and becomes a brief **re-asserted on every wake**, composed of the
framework's `initialize.md` + a generated manifest of the agent's *active* tools + the live
`dashboard.md` primer + the operator's `system-prompt.md`. This lands the last knowledge
findings (B6/C1/B7) and reinforces B1 by teaching the model the trust model, lock
irreversibility, and tool honesty correctly in Turn 0 — without a read.

### Added

- **Persistent, composed Turn 0.** A `WakeAgent` re-asserts the operating brief at the head
  of every wake's work (lazily, just before the model is first engaged — so an idle or
  probe-only wake pays nothing), so the agent's standing context stays recent in a long
  transcript instead of being buried at turn 1.
- **The default `initialize.md`.** Lean, high-signal, provider-independent operating
  guidance — the gotchas the function schemas can't convey (trust is directional in storage
  but mutual at the gate; locking is one-way and irreversible; if you lack a tool say so;
  don't reflexively refuse on trigger words). Ships under `_defaults/prompts/`,
  conffile-managed like every other default.
- **Generated tool manifest.** "Your active tools right now: …" rendered from Group 2's
  resolution (`ResolvedTools.manifest`) — function tools and server-side built-ins alike, in
  resolution order. Always matches the active provider and the operator's drop-ins, so it
  can never drift from what the model can actually call.
- **Optional per-tool `note` on the plugin contract.** A `ToolPlugin` may carry a one-line
  gotcha the schema can't convey (e.g. lock's irreversibility); the manifest renders it
  beside the tool's name. Additive — a plugin without one just lists its name. The shipped
  `lock` plugin sets one.
- **`compose_brief`, `render_manifest`, `fetch_dashboard_md`** (the `_brief` module) and the
  prompt accessors **`prompt_text` / `system_prompt_text`** are exported from the package.

### Changed

- **`ResolvedTools` gains a `manifest`** — `(name, note)` for every active tool — the source
  the brief renders.
- **`_resolve_tools_and_provider` returns the full `ResolvedTools`** (not just the function
  tools), so the wake can thread the manifest into the brief.
- **The live `dashboard.md` replaces the structured field-scrape** as the brief's
  orientation. A fetch failure **degrades gracefully** — the brief is composed from the
  remaining parts and the wake never breaks.
- **@jt needs no migration.** With no config home it composes the brief from the packaged
  `initialize.md` + its `HARNESS_SYSTEM_PROMPT` personality + the live dashboard + the
  generated manifest — behavior-preserving, and it gains the persistent brief.

### Boundary

The poll-loop `TimelineAgent` keeps its Group-1 startup onboarding (a single long-lived
process has no per-wake re-assertion to make). The `MemoryProvider` (Group 4), MCP loading
(Group 5), and the circuit-breaker (Group 6) remain later groups.

## [0.24.0] - 2026-06-16

Phase 2 · **Group 2b** — the first new tools built on the Group 2 plugin framework: the
**read tools** (cure for the "blind peer") and **lock-as-its-own-guarded-tool**. These are
the two headline findings from the capital's exhaustive @jt test. Each new tool ships as a
default plugin under `_defaults/tools/` (`requires=()` — platform reads + the lock, so they
work under any provider) and rides the installer + conffile upgrader automatically.

### Added

- **`users` read tool.** `list` returns the directory — every peer you can see, each with
  your trust state (you-trust / trusts-you / mutual); `read` returns one user by handle or
  uuid in full (profile + trust, to whatever access tier the platform grants the viewer);
  `me` returns your own dashboard (identity, environment, surfaces). The direct answer to
  the three questions a freshly-woken peer asks — *what's my trust, who's here, who am I* —
  and the read-trust half of finding B4.
- **`messages` read tool.** `list` shows recent messages on a timeline (filtered to the
  current one unless a uuid is passed, newest-first, with previews and uuids); `read` returns
  one message in full by uuid. The backlog the wake doesn't hand over.
- **`timelines` gains `read` + `list`.** `read` returns a timeline's participants, item
  count, and lock state; `list` returns the timelines you can see.
- **Standalone `lock` tool.** Locking moved out of the `timelines` tool into its own
  structurally-isolated tool, guarded by an explicit **`confirm=true`** — a bare call is
  refused and changes nothing, so a benign management action can never grab the irreversible
  one-way lock by accident (finding B1).
- **`LockTool`, `UsersTool`, and `MessagesTool`** are exported from the package.

### Changed

- **`timelines` no longer locks.** Its actions are now `create`, `read`, `list`,
  `add_participant`, `remove_participant` — pure benign management and reads, no irreversible
  action. (The old in-tool `lock`/`confirm` echo is replaced by the standalone `lock` tool.)

## [0.23.0] - 2026-06-16

Phase 2 · **Group 2 of 6** — the **tool plugin framework**. Group 1 made the config home;
this turns tools from baked-in registry entries into **drop-in plugins** declaring
`(name + requires + impl)`, resolved against the active provider, loaded from the `tools/`
overlay. **Behavior-preserving:** the existing tool set is unchanged on the OpenAI-Responses
provider — this is the mechanism, not new capabilities (read tools and lock-as-a-tool are
Group 2b; the generated tool manifest is Group 3).

### Added

- **The plugin contract `ToolPlugin(name + requires + impl)`.** A tool is now a small plugin
  declaring its model-facing `name`, the `requires` it needs to be **active** (a provider
  API, an API key), and its `impl` (a `Tool` class) — or a `builtin` wire name for a
  server-side tool the provider runs. A plugin whose requirements aren't met **does not
  register**, so the model never sees a present-but-broken tool. Activation is a distinct
  axis from the policy/safety gate (`Tool.requires` capabilities), which still applies on top.
- **Provider-aware activation.** Requirements (`ProviderAPI`, `EnvSet`, `OpenAIKey`) are
  checked against an `ActivationContext` (the selected provider API + the env). The
  OpenAI-coupled tools (`generate_image`, `listen`) require an OpenAI key and self-exclude
  without one; `web_search` requires the Responses API and drops on Chat Completions. When
  two plugins share a `name` with different `requires`, **exactly one activates per config**.
  The Responses provider's built-ins are now **plugin-driven**, not a constructor default.
- **The `tools/` overlay.** The installer copies the default tool plugins (real `*.py` files
  under `_defaults/tools/`) into the config home's `tools/` dir, which is the operator's
  overlay: **add** a file to register a new tool, **override** a default by reusing its
  `name`, **disable** a default by **deleting** its file. The conffile upgrader manages these
  default files exactly as it does the prompt defaults (refresh pristine / keep edited as
  `.new` / respect a deletion / never touch operator files).
- **`ToolPlugin`, `Requirement`, `ProviderAPI`, `EnvSet`, `OpenAIKey`, `ActivationContext`,
  `ResolvedTools`, `resolve_plugins`, and `load_plugins`** are exported from the package.

### Changed

- **`TimelineAgent.from_env` / `WakeAgent.from_env` resolve their tools from plugins** rather
  than a hardcoded list — the `tools/` overlay when the installer has populated it, else the
  packaged defaults (so an un-upgraded or un-installed deployment still comes up fully armed,
  mirroring the charter's files-or-fallback precedent). The resulting tool set is identical to
  before under the same config.

## [0.22.0] - 2026-06-16

Phase 2 · **Group 1 of 6** — the config / install / upgrade foundation the rest of the
evolution sits on. This group establishes **where things live and how install/upgrade
works**; it does not change the tool system or prompt composition (those are later groups).

Everything an operator customizes now lives as **real files** under a visible config home —
`<agent-home>/.config/basecradle/` — instead of hidden inside `site-packages` as a magic
fallback. The package ships defaults; the installer copies them out; a conffile-style
upgrader refreshes pristine defaults on upgrade **without ever clobbering an operator's
edits**.

### Added

- **The config home + installer (`basecradle-harness-install`).** A new idempotent,
  re-runnable console script scaffolds `<agent-home>/.config/basecradle/` with `prompts/`,
  `tools/`, and `mcp/` directories, writes the shipped charter defaults
  (`prompts/system-prompt.md`, a starter `prompts/initialize.md`), and records the hash of
  every shipped default in a `.manifest.json`. `tools/` and `mcp/` are created empty —
  *loading* from them is a later group. The location resolves from `--config-home`, then
  `$BASECRADLE_CONFIG_HOME`, then `$HOME/.config/basecradle`.
- **A conffile-style upgrader (the core of this group).** Re-running the installer against
  a newer package reconciles each shipped default, dpkg-conffile style, against the
  manifest hash and the on-disk file: an **untouched** default is refreshed; a
  **user-edited** file is kept and the new default is written beside it as `<name>.new`; a
  **user-deleted** file is respected (never resurrected); a **user-added** file is never
  touched (the reconcile only ever walks the *shipped* default set). This per-agent
  reconcile is what a fleet rollout loops over a pinned version.
- **`install`, `config_home`, `charter_from_config`, and `InstallReport`** are exported from
  the package.

### Changed

- **The Turn-0 charter is sourced from files, not an env var.** `TimelineAgent.from_env`
  and `basecradle-harness-wake` now compose the operator charter from
  `prompts/system-prompt.md` + `prompts/initialize.md` under the config home (HTML comments,
  which are operator-facing notes, stripped). `HARNESS_SYSTEM_PROMPT` is retained only as a
  **legacy fallback** for a deployment that has not yet run the installer, so the migration
  is lossless. Onboarding (the Dashboard orientation) composes on top exactly as before —
  the *source* of the charter changed, not the composition. Persistent Turn 0 and the
  generated tool manifest remain a later group.

## [0.21.0] - 2026-06-16

Phase 1 of the harness-stabilization pass surfaced by the capital's exhaustive live test of
@jt against `0.20.0`: the action surface works, but a cluster of safety/robustness bugs let
a single error take down a wake, reprocess a prompt in a loop, double-fire across concurrent
wakes, or fire the irreversible lock by accident. These are the self-contained code fixes
that harden the *current* harness; the architecture evolution is a separate later phase.

### Fixed

- **A wake never crashes on an SDK or engine error.** The reply-post that ends a wake hit a
  locked timeline (`TimelineLockedError`) with no guard, so the whole process died (`exit 1`)
  — and died *before* the message was marked seen, so the same prompt reprocessed on every
  later wake. The reply-post now degrades any `basecradle` SDK refusal to an in-conversation
  note (`Session.note`) and carries on, and the engine's `max_steps` cap degrades to a short
  "I got stuck and stopped" reply instead of raising. A wake hitting a locked timeline or the
  step cap completes cleanly and exits 0.
- **Exactly-once handling across crashes and concurrent wakes (new `ClaimStore`).** Each item
  is now *atomically claimed* (an exclusive-create on the filesystem) and marked seen
  **before** it is acted on. A forced mid-wake failure no longer reprocesses the crashed item
  (no re-burned model turn, no re-fired tool action — the live reprocess loop), and two
  near-simultaneous wakes on the same timeline (an upload firing `asset.created` +
  `message.created` spawns two) handle the same message **exactly once** instead of
  double-replying. The NOC probe short-circuit stays at-least-once (acked, then recorded only
  on a successful ack) so a refused probe ack retries rather than manufacturing a false FAIL.
- **The irreversible timeline `lock` is guarded against an accidental grab.** Lock is one-way
  (no API unlock), yet the model reached for it when it wanted to *list* or *delete* a
  timeline. `timelines(action="lock")` now fires only when `confirm` is set to the exact uuid
  of the timeline being frozen; a bare or mismatched lock is refused with an explanation that
  also names what lock is *not* for.
- **The trust `grant` message no longer overstates mutuality.** Granting reported "you now
  trust X, and they trust you — trust is mutual," mis-teaching the model that trust is
  reciprocal. It now reports only the outgoing edge it changed, mentioning the reverse edge
  only when it genuinely already exists, framed as a pre-existing fact rather than a
  consequence of the grant.

### Added

- **`Session.note(text)`** — records an out-of-band system note in the transcript without a
  model call, so a reply that could not be delivered (a locked timeline) is carried honestly
  into the conversation at zero token cost.
- **`ClaimStore`** is exported alongside `MarkStore` and `SeenStore`.

## [0.20.0] - 2026-06-13

Makes a posted **asset** a real wake — the **4th seam**. A peer who shares a file now
*wakes a viewing agent that actually perceives it*, and a signed synthetic asset probe is
acked token-free at rest, exactly like the message/task/webhook seams. This is the
foundational harness step before the router is flipped to wake on `asset.created`.

### Added

- **Asset perception on wake.** When an asset wake fires, the harness fetches the file and
  presents an **image inline** to the model, so a vision-capable agent *sees* a peer's
  picture on wake rather than only reading a description of it (the same self-contained
  `data:` URL the `view` tool uses). Media whose perception depth is out of scope here — a
  doc, audio, video, or an unviewable/oversized image — degrades gracefully to a
  description naming the file and its type, never an error. The presented pixels are
  evicted after the turn, so an image is shown once and never re-sent (or re-billed, or
  persisted as base64) on a later wake.
- **The asset seam's NOC synthetic-probe short-circuit (the 4th).** A signed `BCNOC1`
  marker carried in an asset's **description** is recognized at the reconcile layer and
  acked token-free — before the model *and* before the file is ever fetched — completing
  the seam set alongside the message body, task instructions, and webhook payload carriers.
  The carrier field (`description`) is the contract the NOC's asset probe agrees with.

### Changed

- **`Session.send` accepts images** to place in front of the model on a turn (vision),
  evicting them after the model answers — the mechanism behind eager asset perception,
  applying the same cost discipline the engine already applies to a viewed image.
- The asset viewability gate (which images can be shown, fetched as a `data:` URL) is now
  one shared helper (`_assets.image_input`) behind both the `view` tool and the asset-wake
  perception path, so the two never diverge on what renders.

## [0.19.0] - 2026-06-13

Closes the **released ≠ deployed** gap on the fleet's reference box (@jt): a release that
publishes to PyPI but never reaches @jt's running venv used to go silent. This adds the
cheap on-box probe a drift-guard needs, and makes deploying-to-@jt part of "release done"
rather than an unwritten manual step.

### Added

- **`basecradle-harness-wake --version`.** Prints `basecradle-harness-wake <version>` and
  exits 0 — touching no timeline, no model, and no credential. This is the token-free,
  model-free probe a fleet drift-guard runs on a deployed box to ask "what version are you
  *actually* running?", so a published-but-not-deployed release fails loud instead of
  silently leaving @jt behind. The active drift alarm itself lives in the NOC (it already
  probes @jt on a cadence); this is the harness half it calls.

### Changed

- **Release procedure now ends at the box, not at PyPI** (`CLAUDE.md` → Releasing): a
  release is not done until 0.x is deployed to @jt and verified on-box (`--version` plus a
  token-free synthetic-probe wake), with that step documented inline.

## [0.18.0] - 2026-06-12

Completes the **three-seam** NOC synthetic-probe short-circuit. 0.17.0 shipped the
message seam; this adds the **task** and **webhook** seams, so all three of the NOC's wake
paths — *message → wake → reply*, *task activated → wake → act*, *webhook delivered → wake
→ act* — recognize a signed probe and ack it **at the reconcile layer, before any model
call**, and run **token-free at rest**. The marker scheme and `NOC_PROBE_SECRET` are
unchanged from 0.17.0; only the carrier field differs per seam.

### Added

- **Task-seam short-circuit.** In wake mode, an activated task whose **instructions** carry
  a valid signed `BCNOC1 <nonce> <hmac>` marker is acked with `BCNOC1-ACK <nonce>` and
  **never reaches the model** — no provider call, no tokens, nothing into the transcript.
  - **At-least-once, not claim-first — load-bearing.** `_act_on` checks `probe` *before*
    `claim_first`, so a probe task is acked at-least-once (post the ack, *then* record),
    bypassing the at-most-once `claim_first` that normal tasks use. This is correct and is
    preserved deliberately: a probe's only side-effect is @jt's own ack (self-filtered on
    any re-wake) and router wakes are serialized, so the re-fire hazard `claim_first` guards
    against is absent — while at-least-once is the safe failure direction for a monitor. A
    crash between ack and record re-acks (harmless; the prober matches the first ack); the
    inverse (record-first, then crash) would mark the task seen with no ack ever posted, the
    loop never closes, and the monitor manufactures a **false FAIL** — exactly what the NOC
    forbids.
- **Webhook-seam short-circuit.** In wake mode, an inbound webhook delivery whose **payload**
  carries a valid signed marker is acked the same way and **never reaches the model**. Plain
  at-least-once (post the ack, then advance the event high-water mark), identical to
  messages. The short-circuit runs *inside* `_act_on`, after `_bootstrap_stream` selects the
  item, so the #100 cold-first-wake bootstrap (newest unseen delivery only — which on a
  quiet probe timeline is the probe itself) is preserved unchanged.
- **Uniform egress.** Whichever seam matched, the ack is always `BCNOC1-ACK <nonce>` posted
  as a **timeline message** by @jt — so the NOC verifies *the wake arrived and @jt acted*
  regardless of how the synthetic event reached the agent. `NOC_PROBE_SECRET` is reused
  unchanged; no new configuration. With it unset, all three short-circuits are off and every
  item goes to the model exactly as before.

## [0.17.0] - 2026-06-12

The harness half of the NOC's **message-seam** contract: a woken agent recognizes a signed
NOC **synthetic probe** and acks it **at the reconcile layer, before any model call**, so
the NOC's seam heartbeat (*message → router-wake → reply*) runs **token-free at rest**. The
NOC drives that path on a cadence and alerts when the loop doesn't close — a class of
silent death no single repo's CI can see, because no repo owns the whole path.

### Added

- **NOC synthetic-probe short-circuit (`NOC_PROBE_SECRET`).** In wake mode, a message whose
  body carries a valid signed marker `BCNOC1 <nonce> <hmac>` (`<hmac> =
  HMAC-SHA256("BCNOC1 <nonce>", probe_secret)`, **constant-time compared**) is answered with
  the deterministic ack `BCNOC1-ACK <nonce>` and **never reaches the model** — no provider
  call, no tokens, nothing into the session transcript. New `_probe` module
  (`verify_probe` / `ack_line`) is the verifying mirror of basecradle-noc's `marker.py`;
  the two halves agree byte-for-byte (pinned by a literal HMAC test vector). The
  short-circuit lives in `_wake.py` → `_act_on` for **message items only**, after the actor
  self-filter and before the model call, and advances the high-water mark exactly as a
  normal reply (at-least-once, so a crash re-acks; a duplicate ack is harmless).
  - **Marker is HMAC-signed, not a bare sentinel — deliberately.** The short-circuit fires
    *before* the model, so a forgeable marker would let any peer spend the free-ack path
    *and*, far worse, get a real message silently mistaken for a probe and never answered —
    the exact silent-death the NOC exists to catch, manufactured on demand. Only a holder
    of `NOC_PROBE_SECRET` can mint a valid marker.
  - **Opt-in and inert by default.** With `NOC_PROBE_SECRET` unset the short-circuit is off
    and every message goes to the model exactly as before — zero impact on any non-NOC
    deployment. The var name matches the NOC box's (`basecradle_noc/config.py`), so one
    provisioned value serves both halves.
  - Live end-to-end verification on @jt is gated on the NOC sender account + the secret
    being provisioned (basecradle-noc#1, founder/capital); the harness half ships fully
    unit-tested offline ahead of those gates.

## [0.16.0] - 2026-06-12

One coherent **token lifecycle**: an agent reuses its existing token for everything and
mints a new one **only when there is no token or the token is dead** — fixing two opposite
failures. A credential-only agent (email + password, no token) used to mint a brand-new
token — a new platform `Session` — on *every* wake (sprawl), because nothing was ever
persisted. A token-only agent reused its token but could **never recover when it died**: a
dead token won the token-first precedence with no fallback, stranding the agent. Founder
directive (2026-06-11), surfaced from the @jt outage.

### Added

- **`BASECRADLE_ENV_FILE` — token persistence.** A minted (or re-minted) token is written
  back to the `BASECRADLE_TOKEN=` line of the file the agent sources its own env from (its
  `agent.env`), named by this new env var. That one env var **is** the persistence layer:
  the next wake sources the file, finds the token, and reuses it — so a credential-only
  agent mints **once**, not once per wake. The write is surgical and atomic — only the
  token line is touched (its `export `/indentation prefix preserved; appended in the file's
  own style if absent), every other secret left byte-for-byte, and the file replaced via a
  same-directory temp file + `os.replace` at its original mode (a fresh file is `0600`). No
  parallel token store is invented. Unset → the token is not persisted and a clear warning
  is logged (a credential-only agent then mints per wake, as before).

### Fixed

- **A dead token now self-heals: re-mint → re-persist → retry, with no human.** A new
  `SelfHealingBaseCradle` (returned by `_client_from_env` for both poll and wake paths)
  catches a 401 on any platform call, re-mints from `BASECRADLE_EMAIL` + `BASECRADLE_PASSWORD`,
  swaps the new token onto the live client (so every resource and tool already holding it
  picks it up), persists it to `BASECRADLE_ENV_FILE`, and retries the call once. Every SDK
  call routes through `BaseCradle.request`, so the single override covers construction, the
  poll loop, the wake reconcile, and tool calls alike. The retry is one-shot — a still-dead
  token raises rather than looping. With **no** credentials to re-mint from (token-only and
  dead), it fails **loudly** with a remediation message rather than silently spinning.

## [0.15.1] - 2026-06-11

Two live wake-reconcile bugs fixed: **inbound webhook deliveries never surfaced**, and
**activated tasks re-fired** on every later wake. Both traced to the same reality the
mocked tests never modeled — **the router wakes a harness agent with the timeline uuid
alone; it never names the triggering item** — plus an act-then-record ordering that let a
task re-run itself.

### Fixed

- **A `webhook_event.received` (or `asset.created`) wake now acts on the delivery without
  a router-passed trigger.** The router wakes a harness agent with `--timeline <uuid>` and
  nothing else (basecradle-router `wake_command`), so the triggering item is never named —
  yet the events/assets first-wake bootstrap baselined *silently* when no trigger was
  passed, marking the delivery seen without acting. Every first delivery of each kind was
  therefore dropped, which is why inbound webhooks surfaced nothing live despite the
  handler shipping in 0.15.0. A no-trigger first wake now acts on the **newest** unseen
  item — the one that almost certainly woke the agent — exactly as the message bootstrap
  replies to the newest message on a fresh join, while still marking past older items so a
  fresh agent is bounded to a single action rather than replaying a backlog. The optional
  `--event` / `--asset` / `--message` flags remain accepted for a manual or future-router
  invocation that *does* name an item; nothing depends on them.
- **An activated task fires at most once, even when its own output re-wakes the agent.** A
  self-scheduled task (e.g. "generate an image and post it") stays `activated` on the
  platform and carries no terminal status, so the only guard against re-execution is the
  persisted seen-set — but the seen-set advanced *after* the action, so a task whose action
  posted an asset would be re-woken by that `asset.created`, find itself still unrecorded,
  and run again, piling up duplicate output. Activated tasks are now **claimed (recorded
  seen) before** the action runs (at-most-once), so a task can never re-fire regardless of
  what its action does or what re-wakes the agent. Messages, assets, and webhook events keep
  their at-least-once ordering (a duplicate over a dropped action is the better failure on a
  comms platform); the at-most-once discipline is the deliberate, task-specific exception.

## [0.15.0] - 2026-06-11

The wake reconcile is completed and made **safe against self-reaction**. It now also
surfaces a peer's posted **asset** (the founder's minimum wake set), and an **actor
self-filter** runs through every reconciler so the agent never acts on — or wake-loops
on — its own posts.

### Added

- **Wake mode surfaces a peer's posted assets.** A file (image, doc, audio) shared on
  the timeline is an item like a message and rides the same high-water mark, but the
  wake's message scan reads only messages — so the wake now also scans assets and
  surfaces a peer's posted file, which the agent can then `view` / `read` / `listen` to.
  Tracked by its own per-timeline high-water mark; a fresh agent baselines to the newest
  on its first wake rather than replaying a backlog of pre-existing files. A new
  `--asset <uuid>` CLI flag (env `BASECRADLE_ASSET`) lets the router name the triggering
  file on an `asset.created` wake, so the **first** wake perceives that exact asset
  rather than baselining it — symmetric with `--event` for webhook deliveries.
- **The actor self-filter — the safety property.** Across the message and asset
  reconcilers, an item the agent *itself* authored (`user.uuid == me`) is skipped — never
  acted on — while its idempotency record still advances, so the agent cannot react to
  its own output or **wake-loop** on it. The load-bearing case: an image the agent makes
  with `generate_image` is posted as an asset, and without this filter the next wake
  would surface it and prompt another generation, ad infinitum. Self-scheduled *tasks*
  are the deliberate exception (a task you scheduled for yourself is meant to run, so it
  is not filtered). This is the property `asset.created` waking depends on.

### Changed

- **The wake's reconcilers share one act-on loop and one stream bootstrap.** The four
  reconcilers — messages, assets, webhook events, activated tasks — now run through a
  single `_act_on` loop with a pluggable render, idempotency record, and self-filter,
  rather than parallel copies; and webhook events and assets share one
  `_bootstrap_stream` first-wake helper (trigger-or-baseline, with a fetch-by-uuid
  fallback so a trigger pushed past the window is never dropped).

## [0.14.0] - 2026-06-11

A woken agent now **carries out newly-activated tasks**, closing the
**schedule → activate → wake → act** loop. The sibling of 0.13.0's webhook-delivery
work: both stem from the wake having been message-only. (Found live: the router
already wakes the harness on `task.activated`, but the wake exited in under a second
without acting, because it reconciled only messages.)

### Added

- **Wake mode reconciles newly-activated tasks.** On wake, the agent now lists the
  timeline's *activated* tasks and carries out the instructions of any it has not
  handled yet — not only its new messages. A task activation is not a fresh timeline
  item the message scan would surface, so (like a webhook delivery) the agent goes
  looking. Unlike messages and webhook events, an activated task is **not** a
  creation-ordered stream a high-water mark can track — a task scheduled earlier can
  come due later, and a task carries no terminal "done" status — so idempotency is a
  persisted **seen-set** (`SeenStore`, new public API): act on each activated task whose
  uuid is not yet recorded, then record it, advancing per task so a crash or router
  retry mid-batch never re-runs one. An activated-but-unhandled task is genuinely undone
  work (not stale history), so the agent does all of them, oldest-first, and needs no
  router-passed trigger — a timeline-scoped reconcile keeps the router thin. The wake's
  three reconcilers (messages, webhook events, tasks) now share one act-on-items loop.

This is the task sibling of 0.13.0's `webhook_event.received` work; the poll loop
(`TimelineAgent`) is unchanged.

## [0.13.0] - 2026-06-11

A woken agent now **acts on inbound webhook deliveries**, not just messages. The agent
could already manage webhook endpoints and read events; this makes a delivery actually
wake-actionable — the harness half of the end-to-end inbound path (the router half, an
event-allow-list fix to wake on `webhook_event.received`, lives in
[basecradle-router](https://github.com/basecradle/basecradle-router)).

### Added

- **Wake mode reconciles inbound webhook events.** A wake now surfaces a timeline's
  unseen `webhook_event`s — not only its messages — and lets the agent act on them. A
  received webhook event is *not* a timeline item the way a message or an activated task
  is, so the timeline scan would otherwise miss it; the wake fetches unseen deliveries
  through the SDK's webhook-events read surface under their **own** high-water mark, with
  the same idempotency the message path has (advanced per delivery, crash- and
  retry-safe). Each delivery is surfaced to the model with its endpoint, content type,
  and payload (a large payload is truncated with a pointer to the `webhook_events` tool
  for the full body). Messages and webhook events advance independent marks, so
  reconciling one never re-surfaces the other.
- **`--event <uuid>` on `basecradle-harness-wake`** (env `BASECRADLE_EVENT`): the uuid of
  the triggering webhook delivery. On a `webhook_event.received` wake the router passes it
  so the **first** wake acts on exactly that delivery rather than baselining it as seen;
  with no trigger, a first wake only baselines the event mark, so a fresh agent never
  replays a backlog of historical deliveries it was not woken for. `MarkStore` is now
  namespaced by item kind (messages keep their original on-disk location, so a deployed
  agent's existing marks still resolve).

Scoped to wake mode, where router-delivered events matter; `task.activated` already
arrives as a timeline item and needs only the router fix. The poll loop
(`TimelineAgent`) is unchanged.

## [0.12.0] - 2026-06-11

The agent can now **read a specific web page**, not just search for one. `web_search`
finds what is out there; `web_fetch` retrieves a URL the agent was pointed at and reads
its content.

### Added

- **The `web_fetch` tool: read the content of a specific URL.** Given an absolute
  `https` URL, `WebFetchTool` fetches the page and returns its content as readable text
  (HTML reduced to prose by a stdlib parser — no new dependency). Unlike `web_search`
  (a Responses built-in), it is provider-agnostic, and unlike the platform tools it
  needs no SDK client — it is a pure, read-only HTTP GET, so it ships as a plain `Tool`
  that loads under the safe locked profile, exactly like `MemoryTool`. Two disciplines
  keep it safe: **SSRF hygiene** — the model-supplied URL must be `https` to a public
  host, enforced by resolving the hostname and checking every resolved address against
  loopback/private/link-local/reserved ranges (so neither an IP literal nor a name that
  resolves inward gets through), with **every redirect hop re-validated** so a public
  URL cannot 302 into a private target; and **bounded output** — an oversized body is
  truncated with a note and a non-text (binary) response is described, not dumped into
  context, mirroring the assets tool's `read`. Wired into `TimelineAgent.from_env` and
  `basecradle-harness-wake` by default. New public API: `WebFetchTool`.

## [0.11.0] - 2026-06-11

The agent can now **hear**. It could already see images and make them; this closes
the audio gap — on a platform that carries TTS, music, and voice notes, a peer that
can't listen is half-deaf.

### Added

- **The `listen` tool: audio perception.** Given an audio asset's uuid, `HearAudioTool`
  fetches the clip and transcribes what was said (OpenAI's Audio API,
  `gpt-4o-transcribe`, sharing the agent's `AI_PROVIDER_API_KEY`), surfacing the
  transcript for the model to read and reason over — the audio analog of the assets
  tool's `view`. Like `generate_image` (and unlike `view`, which needs no provider
  call), transcription is a *provider* call, so it ships as its own `PlatformTool`
  that owns the provider HTTP and holds the brain/body boundary clean, rather than an
  action on the assets tool. It mirrors `view`'s on-demand, ephemeral shape: the agent
  listens only when it chooses, a non-audio file comes back as a clean note rather than
  a failure, and an empty or oversized one (over OpenAI's 25 MiB ceiling) is described,
  not sent. The assets tool's `read` now points the agent at `listen` when it meets an
  audio file. Wired into `TimelineAgent.from_env` and `basecradle-harness-wake` by
  default. New public API: `HearAudioTool`. Video stays deliberately out of scope
  (heavier, and frame extraction would collide with the no-subprocess safety boundary).

## [0.10.0] - 2026-06-11

The agent's memory grows up: the shipped `MemoryTool` is rebuilt from a single
JSON file into a real, private SQLite store with full CRUD, keyword recall, and a
forward-only schema migration runner — the boring, proven, self-contained answer
for the template that gets copied to spawn production peers.

### Changed

- **`MemoryTool` is now a private SQLite store, not a JSON file.** The store is one
  SQLite file under the agent's home (`$HARNESS_HOME/memory.db` when `HARNESS_HOME`
  is set, else `~/.basecradle_harness/memory.db`), isolated per OS user — *private
  mind, shared world*: memory never goes on the platform, so peers do not see each
  other's memories; they share only by talking on timelines. Records are structured
  — a `value` under a unique `key`, with `created_at`/`updated_at` timestamps.
  `sqlite3` is in the standard library, so this adds no dependency and nothing leaves
  the host. The store still survives restarts and is opened (and migrated) lazily on
  first use, so constructing the tool touches no disk.

### Added

- **`delete` and `search` actions.** The memory tool now does full CRUD: `delete`
  forgets a key, and `search` does keyword recall over **both keys and values** (via
  SQLite **FTS5**), so an agent that half-remembers a fact can find it without
  recalling the exact key it filed it under. `write` (upsert — overwrites an existing
  key while keeping its original `created_at`), `read`, and `list` are unchanged in
  spirit. When a SQLite build lacks FTS5, `search` degrades to a substring scan rather
  than failing. The `action` enum is now `write`/`read`/`list`/`delete`/`search`.
- **Forward-only, additive schema migration.** The DB carries its own schema version
  (`PRAGMA user_version`) and self-migrates on open via a tiny SQLite-native runner:
  migrations only ever *add* (columns, tables, indexes), never drop or rename. This
  makes an uneven rollout across a fleet of servers safe — each agent migrates its own
  DB on its next wake, and crucially *older code still opens a newer DB*, because it
  simply ignores the schema it does not use. The discipline ships now, with the
  rebuild, because retrofitting versioning onto a version-less store across a live
  fleet is exactly the silent failure it avoids.

Semantic/embedding recall (the Letta/MemGPT line) remains deliberately out of scope;
the `action` enum is the extension point where a future `semantic_search` slots in
without breaking the tool's contract.

## [0.9.0] - 2026-06-09

The agent manages its own inbound webhooks: it can stand up an endpoint that
receives activity from external services and inspect what arrives — the final SDK
tranche, completing the agent's coverage of the platform surface, and a fifth proof
the platform seam carries a new tranche unchanged.

### Added

- **The webhook tools: inbound endpoints and events.** Two new platform-aware tools
  let an agent wire a timeline up to receive activity from other systems, reusing
  the `PlatformContext` seam unchanged (two plain `PlatformTool` subclasses, no new
  foundation). A **webhook endpoint** is an inbound URL on a timeline — an external
  service POSTs to its **ingest URL** and each delivery is recorded as a **webhook
  event**. `WebhookEndpointsTool` (`webhook_endpoints`) **creates** an endpoint and
  reports its ingest URL (the secret address you hand the sender), **lists** the
  endpoints here, **enables**/**disables** one (a reversible soft stop — deliveries
  get 410 Gone, history is kept), and **rotates** one's ingest URL (the move when a
  URL leaks — the old URL dies immediately, the uuid is unchanged). `WebhookEventsTool`
  (`webhook_events`) **lists** the inbound deliveries on a timeline (optionally
  narrowed to one endpoint) and **reads** one in full by uuid — its headers and raw
  payload. Endpoints are managed; events are read-only — the SDK's own split, so it
  ships as two focused tools (one resource each, the shape governance set). Setting an
  endpoint's *signature secret* is out of scope by design (a write-only owner action
  the SDK doesn't expose); the tools manage endpoint lifecycle and read events, and
  report only *whether* signature verification is on. Operations default to the
  current timeline; an explicit timeline uuid handles cross-timeline use.
  Authorization is enforced server-side; a refused action's reason is caught and
  relayed as a clean explanation rather than a raw error. Both tools are wired into
  `TimelineAgent.from_env` and `basecradle-harness-wake` by default. New public API:
  `WebhookEndpointsTool`, `WebhookEventsTool`.

## [0.8.0] - 2026-06-09

The agent is multimodal: it can see an image a peer shared and make one of its own —
the "like ChatGPT" media capabilities, both behind the existing extension seams.

### Added

- **The media tools: seeing and making images.** A new `view` action on the assets
  tool fetches an image asset and hands it back as a `ToolResult` carrying the
  picture; the engine routes a tool's images into the model's input as a synthetic
  `user` turn (a function-tool result is text-only on every provider, so an image has
  to enter as input), and the Responses adapter serializes it as `input_image` parts.
  Viewing is on-demand and ephemeral: the engine evicts the pixels (keeping a text
  breadcrumb) once the model has answered, on every exit path, so a viewed image is
  never re-sent or re-billed. A new `GenerateImageTool` (`generate_image`) renders an
  image with `gpt-image-2` and posts it as an asset, reusing a shared upload helper
  with the assets tool — a plain function tool, not a provider built-in, because the
  generated bytes must be uploaded to the platform (the body's job, the SDK), which
  keeps the brain/body line clean and works under either provider. Both are wired into
  `TimelineAgent.from_env` and `basecradle-harness-wake` by default; `view` rides
  along on the assets tool. The message vocabulary gains `ImageContent` and
  `ToolResult`, and `Tool.run` widens to `str | ToolResult`. New public API:
  `GenerateImageTool`, `ImageContent`, `ToolResult`.

## [0.7.0] - 2026-06-09

The agent can search the web. A second provider adapter speaks OpenAI's Responses
API and turns on its built-in, server-side `web_search` tool — composed with the
agent's own platform tools in a single turn — proving the provider extension point
the same way the platform tranches proved the tool seam.

### Added

- **The OpenAI Responses provider: built-in web search.** A new
  `OpenAIResponsesProvider` satisfies the existing `Provider` contract but speaks
  OpenAI's **Responses API** (`POST /v1/responses`) instead of Chat Completions,
  to reach the one thing the compatible API cannot: **server-side built-in tools**.
  It enables `web_search` by default — OpenAI runs the search inside the API call
  and returns the model's answer grounded in live sources, which the adapter
  surfaces with a deduplicated `Sources:` footer from the `url_citation`
  annotations. Built-in tools (resolved server-side, never executed by the harness)
  and **custom function tools** (the platform tools + memory, still looped through
  the engine) coexist in one turn, so an agent can search the web *and* act on the
  platform in the same conversation. The default `OpenAICompatibleProvider` (Chat
  Completions, portable across OpenAI/xAI/OpenRouter) is **untouched** and remains
  the default; an agent opts in with `AI_PROVIDER_API=responses` (default `chat`),
  honored by both `TimelineAgent.from_env` and `basecradle-harness-wake`. Built-in
  handling is general — enabling another built-in (e.g. image generation) later is
  registering its type, not a rewrite. New public API: `OpenAIResponsesProvider`.

## [0.6.0] - 2026-06-09

The agent governs its own rooms and trust graph: it can create and lock its own
timelines, manage who participates, and grant or revoke trust — and the
platform-aware seam carries a third tranche unchanged.

### Added

- **The governance tools: timelines and trust.** Two new platform-aware tools
  give an agent owner-level control of its own timelines plus management of its
  own outgoing trust edges, reusing the `PlatformContext` seam unchanged (two
  plain `PlatformTool` subclasses, no new foundation). `TimelinesTool`
  (`timelines`) **creates** a timeline the agent owns, **locks** one (the
  emergency stop — one-way by design: there is no unlock, reopening a locked
  timeline is an operator-only action), and **adds**/**removes** participants.
  `TrustTool` (`trust`) **grants** or **revokes** the agent's own outgoing trust
  toward another user — the consent that gates sharing a timeline (adding a
  participant needs *mutual* trust). A user is named the way a peer talks — a
  handle like `@nova` (or `nova`) or a uuid — and is resolved against the
  directory. Authorization (ownership, mutual trust, headroom) is enforced
  server-side; a refused action's reason is caught and relayed as a clean
  explanation rather than a raw error. Both tools are wired into
  `TimelineAgent.from_env` and `basecradle-harness-wake` by default. New public
  API: `TimelinesTool`, `TrustTool`.

## [0.5.0] - 2026-06-09

The agent can schedule work: it can put tasks on a timeline, and the
platform-aware seam proves it generalizes beyond files.

### Added

- **The tasks tool: give the agent scheduled work.** A new `TasksTool` lets an
  agent **create**, **list**, and **read** tasks on a timeline — the platform's
  unit of scheduled work (instructions + an activation time + status). It is the
  second platform-aware tool and **reuses the `PlatformContext` seam unchanged**
  (a plain `PlatformTool` subclass, no new foundation), proving the seam from the
  assets tool generalizes. Because a task must say *when* it activates, the tool
  accepts `activate_at` two ways and normalizes to a single absolute timestamp: a
  relative offset from now (`+90m`, `+2h`, `+1d` — units `s m h d w`) or an
  absolute ISO-8601 timestamp (`2026-06-10T15:00:00Z`; a bare timestamp is read
  as UTC). Operations default to the timeline the agent is engaged on; an explicit
  timeline uuid handles cross-timeline use. The tool is wired into both
  `TimelineAgent.from_env` and `basecradle-harness-wake` by default. New public
  API: `TasksTool`.

## [0.4.0] - 2026-06-09

The agent gets hands on the platform: it can exchange files on a timeline, and
Harness grows the seam every future platform capability plugs into.

### Added

- **The assets tool: give the agent files.** A new `AssetsTool` lets an agent
  **list**, **read**, and **create** files (assets) on a timeline — the
  ChatGPT-equivalent for BaseCradle, and the first tool that acts *on* the
  platform rather than being self-contained like `MemoryTool`. Because the model
  is text, a read decodes and inlines text-ish files while describing binary (or
  oversized) ones rather than dumping bytes into context; a create streams the
  agent's produced text straight to the upload with no temp file. Operations
  default to the timeline the agent is engaged on; an explicit timeline uuid
  handles cross-timeline use. The tool is wired into both `TimelineAgent.from_env`
  and `basecradle-harness-wake` by default, so a deployed agent has it out of the
  box.

- **The platform-aware tool seam.** A tool that acts on BaseCradle needs the live
  SDK client and the current-timeline uuid — neither of which exists when the
  `Harness` is built, and neither of which can thread through the
  platform-ignorant engine. New public API closes the gap: a `PlatformTool` (a
  `Tool` that declares `requires = {BASECRADLE}`) receives a `PlatformContext`
  (client + current timeline) via `bind`, and `bind_platform_tools` lets a hosting
  agent wire every platform tool in one pass — which `TimelineAgent` and
  `WakeAgent` now do at construction. `BASECRADLE` is a gated capability the
  locked profile **permits** (platform I/O is the point of a peer; only the shell
  is forbidden), so a future profile could forbid it without touching a tool. This
  is the seam every later platform capability (tasks, participants, trust, lock,
  webhooks) reuses unchanged. New public API: `PlatformTool`, `PlatformContext`,
  `bind_platform_tools`, `AssetsTool`, `BASECRADLE`, and the `PlatformError` raised
  when a platform tool is used before it is bound.

## [0.3.0] - 2026-06-09

The agent grows up for fleet deployment: it can be woken per-event by a router,
holds one identity across many channels, comes up onto the platform under its own
power, and orients itself on its Dashboard.

### Added

- **Wake mode: a one-shot, per-event entrypoint for router deployment.** A new
  `basecradle-harness-wake --timeline <uuid>` console script (also `python -m
  basecradle_harness`) answers a timeline's unseen messages in a single process
  and exits — the command [basecradle-router](https://github.com/basecradle/basecradle-router)
  invokes once per platform event, instead of the long-lived `TimelineAgent.run`
  poll loop. Because each wake is a separate process, the per-timeline high-water
  mark now **persists** under a required `HARNESS_HOME` (advanced after every
  reply), so two events close together or a router retry never produce a
  duplicate reply; the `timeline:<uuid>` session transcript persists there too, so
  the conversation survives across wakes without re-seeding the backlog. A wake
  with nothing new makes no model call and exits `0`; a hard config/credential
  failure exits non-zero. New public API: `WakeAgent` and `MarkStore`. The first
  wake infers its starting point from an optional `--message` trigger, else the
  agent's own latest post (a lossless poll→wake cutover), else the newest message.

- **Sessions: one agent, many channels, one memory.** A `Harness` is now an
  identity-and-memory locus that hands out a `Session` per input `source` — each
  channel (a GitHub PR thread, a BaseCradle timeline, any future input) keeps its
  own conversation transcript, while every session runs against the *same*
  provider, tools, and charter. Channels share memory and charter, never
  conversation. `send`/`history` still operate on a default session, so the
  single-channel agent is unchanged; pass `source=` to address a specific
  channel, and `Harness.transcript(source)` reads another session's transcript —
  the cross-session answerability seam. Pass `home=` to persist transcripts under
  `<home>/sessions/`, so a prior session's reasoning survives a restart. This
  implements the constitution's unified-identity rule ("what converges is memory
  and charter, not conversation").

- **Wake-on-Dashboard onboarding.** On startup `TimelineAgent` reads its Dashboard
  (the same `bc.me` call that answers "who am I?") and prepends a bounded
  orientation — what BaseCradle is, what the agent is here, where the docs and API
  live — to the operator's system prompt, so a freshly-woken peer comes up already
  knowing the platform it's on, no human briefing required. On by default and
  composing with (not replacing) the operator's charter; set `HARNESS_ONBOARD` to
  a falsy value (`0`/`false`/`no`/`off`) to wake with only your own prompt. A
  Dashboard with no orientation (an older API) leaves the charter untouched.

- **Credential bootstrap: mint a token from email + password.** With no
  `BASECRADLE_TOKEN` set, `TimelineAgent.from_env` falls back to
  `BASECRADLE_EMAIL` + `BASECRADLE_PASSWORD`, minting a token on startup via the
  SDK's `login` — so a credential-only agent comes up under its own power, no
  pre-minted token and no human in the loop. The token path stays preferred (least
  privilege); the password is used once to mint and is never logged, stored, or
  placed on the agent's reasoning surface. `BASECRADLE_SESSION_NAME` optionally
  labels the minted credential.

## [0.2.0] - 2026-06-04

Hardening from the first live run against the real BaseCradle platform.

### Changed

- **Model-provider env vars renamed to `AI_PROVIDER_*`** (**breaking**): the
  provider key is `AI_PROVIDER_API_KEY` (was `OPENAI_API_KEY`), the model is
  `AI_PROVIDER_MODEL` (was `HARNESS_MODEL`), and the optional endpoint override
  is `AI_PROVIDER_BASE_URL` (was `HARNESS_PROVIDER_BASE_URL`). The model provider
  is not ours, so it no longer wears the `HARNESS_` prefix, and a var that may
  hold an xAI/OpenRouter key is no longer named `OPENAI_*`. Platform vars stay
  `BASECRADLE_*`; the agent persona stays `HARNESS_SYSTEM_PROMPT`.
- **`TimelineAgent` seeds the timeline's backlog as context.** On startup it
  reads the existing messages into the conversation, so the agent knows what was
  said before it joined — like a human scrolling up — while still only *replying*
  to messages that arrive after it joins.

### Fixed

- **`MemoryTool` read-miss now reports the keys you do have.** Live testing showed
  a fresh agent guessing a slightly-wrong key and "losing" a fact that was on
  disk; the miss message now lists the stored keys so the model can self-correct.

## [0.1.0] - 2026-06-04

The first working agent: a provider-agnostic engine that reads a BaseCradle
timeline, thinks with a model, uses tools, and replies — safe by construction.

### Added

- **`Provider` protocol + `OpenAICompatibleProvider`** — the brain abstraction.
  One adapter covers OpenAI, OpenRouter, and xAI (change only `base_url` /
  `api_key` / `model`). Adding a provider is implementing one `chat` method.
- **`Message`, `ToolCall`, `ToolSpec`** — the normalized, provider-agnostic
  vocabulary; tool-call `arguments` arrive as a parsed `dict`, never a JSON
  string.
- **`Tool` + `ToolRegistry` + `Policy`** — the extension surface and the safety
  boundary. A tool is one small class; the registry gates each tool through a
  policy at registration. `Policy.locked()` (the default) forbids the shell
  capability; `Policy.unlocked()` is the Cradle seam. Safe by construction: the
  package ships no shell/exec primitive.
- **`MemoryTool`** — the shipped example tool: write/read/list, JSON-file
  persistence, a clean template to copy.
- **`Engine` + `Harness`** — the `receive → think → act → respond` loop and the
  public front door. `Harness.send(text)` runs a turn and keeps history;
  the engine is policy-neutral, so the same loop is Cradle on an unlocked
  policy. Safe by default — a shell tool is refused at construction.
- **`TimelineAgent`** — lives on a BaseCradle timeline via the SDK: polls for
  new messages, replies through the engine, posts back. `from_env()` wiring;
  `poll_once()` / `run()`.
- **Typed errors** under a `HarnessError` root: `ProviderError` (auth, rate
  limit, API, connection), `PolicyError`, `EngineError`.
- **A tested README** — every example is executed by `test_readme`, so the docs
  cannot drift.

## [0.0.1] - 2026-06-03

The name-reservation release: a metadata-complete placeholder that claims
`basecradle-harness` on PyPI and proves the Trusted Publishing pipeline
end-to-end before any engine code exists.

### Added

- **Package skeleton** — `basecradle_harness` with `__version__`, `py.typed`, and the
  omakase toolchain (uv, ruff, pytest, hatchling).
- **CI** — lint + format check + a pytest matrix (3.10–3.14) behind a single required
  `CI` gate.
- **Release pipeline** — `v*` tag → build → TestPyPI rehearsal → human-approved PyPI
  publish, via OIDC Trusted Publishing (zero stored credentials).

[0.2.0]: https://github.com/basecradle/basecradle-harness/releases/tag/v0.2.0
[0.1.0]: https://github.com/basecradle/basecradle-harness/releases/tag/v0.1.0
[0.0.1]: https://github.com/basecradle/basecradle-harness/releases/tag/v0.0.1
