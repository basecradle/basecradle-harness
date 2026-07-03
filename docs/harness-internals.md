# Harness Internals — Phase 2 Build History

> **Reference, not charter.** This file holds the spent build-provenance for Harness's
> Phase-2 tranches — how each subsystem was built and why. It is **not loaded into the
> agent's context** every turn; the always-loaded charter (`CLAUDE.md`) keeps only the
> invariants, gotchas, and decision rules. Read this when you need the design history or
> the detailed mechanics behind a subsystem. The settled architecture lives in
> `CLAUDE.md` → "Architecture — The Spine"; the config-home layout, install idempotence,
> conffile discipline, and the two standing security invariants (capability opt-in
> fails-closed, MCP safe-by-default) stay in `CLAUDE.md` → "Config Home (Install /
> Upgrade)". The install/upgrade *procedure* lives in the `config-home-install` skill.
>
> **A recurring boundary across every tranche below:** the harness ships the *mechanism*
> and proves it with offline tests; **the NOC deploys** it to each agent box (the fleet's
> sole deployer — no one hand-provisions a box), and **the capital live-verifies** on @jt
> and closes the handoff. The per-section "Boundary:" paragraphs restate this division in
> tranche-specific terms.

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
existing tools. Deployment proper — provisioning a venv and converging an agent box, wiring the
[`basecradle-router`](https://github.com/basecradle/basecradle-router) daemon/service on the
home server — is the **NOC's** job (the fleet's sole deployer), not the installer's (per the
spine: harness owns the agent runtime, not the box).

### Powerful Tools Are Opt-In — the capability rule (issue #168)

**Tool assignment is a per-persona axis, classified by *capability*, not by provider.** A
powerful/dangerous tool — media generation (image, **video**, audio), web/X search, code
execution — **fails closed**: it is **off by default on every provider** and activates **only**
when explicitly dropped into a persona's `tools/` overlay (the same "ships empty" stance as
`mcp/`). A benign/platform tool (memory, assets, messages, timelines, tasks, trust, lock,
delete, users, webhooks, web_fetch) keeps the normal shipped-default → install-then-prune
behavior. This is **provider-agnostic**: the `requires` gate (`Vendor`/`OpenAIKey`) decides a
powerful tool's *availability/wiring*, **never** the safety default — there is no "default on
OpenAI, opt-in on xAI" split. *(Decided by the capital + founder, applying Option 1 uniformly;
see [[classify-safety-by-capability-not-provider]].)*

- **The flag.** A `ToolPlugin` marks itself `opt_in=True` (the seven powerful defaults:
  `generate_image`, `edit_image`, `hear_audio`, OpenAI `web_search`, xAI `web_search`/`x_search`,
  `grok_generate_image`, `grok_generate_video`). The packaged-default fallback **drops** opt-in
  plugins; the installer **does not scaffold** them; both detect the flag from source via AST
  (`_install.plugin_opts_in`, the no-import discipline shared with provider affinity).
- **Granting one.** `basecradle-harness-install --opt-in <stems>` scaffolds the named powerful
  defaults into the overlay (or drop the file in by hand). An opt-in plugin *present* in the
  overlay activates, gated only by its `requires`.
- **Grandfather, loudly.** On upgrade, a powerful tool a *prior* version had already scaffolded
  into an existing config home is **kept, never silently stripped** (the founder's "tools stay
  the same" migration rule) and **reported loudly** (`InstallReport.grandfathered` →
  the CLI summary + a `WARNING`). New installs get the opt-in (off) default.
- **Why it's a hard requirement.** Adversarial-by-design personas (the fleet's `pinky`/`the-brain`)
  must be tool-less **by construction**, never "on unless someone remembered to prune." Any
  provider/SDK-based default would silently arm whoever moves onto that provider next — the exact
  safety violation this rule forecloses. *(The capital specifies those personas as explicitly
  tool-less and the NOC provisions them so at cutover; the loud grandfather report is what lets
  the capital confirm what to prune.)*

**Boundary:** deciding each persona's target tool-set (cutting its overlay to spec) is the
**capital's** governance call; applying it on a box — provisioning/re-provisioning `jt`/`eddie`
in lockstep with a release — is the **NOC's** deploy (it converges each box to the git-tracked
desired config; no one hand-provisions a box). The harness ships the mechanism + the
`--opt-in`/grandfather affordances and proves them with tests.

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

**Boundary:** the circuit-breaker is Group 6. The "Memory Prince" agent is provisioned on-box by
the **NOC** (the fleet's sole deployer); the cross-timeline proof is the **capital's** live
verification, post-ship.

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

### Native xAI Adapter — grok over `xai-sdk` (gRPC, issue #165)

The **second `Provider` adapter** (`_xai_sdk.py`, `XaiSdkProvider`) and the first that is **not**
OpenAI-wire: `AI_SDK=xai-sdk` reaches grok through xAI's own first-party SDK (`xai-sdk`, gRPC),
no OpenAI-compat shim — the vendor-SDK spine for xAI's *native* path. It is the Grok personas'
end-state brain; `AI_SDK=openai` at `api.x.ai` (issue #163) stays a supported alternative cell.

- **Brain only; tools stay per-persona.** The adapter is the chat `Provider` (chat + tool calling
  + vision). It maps the harness `Message`/`ToolSpec`/`ToolCall` vocabulary onto the SDK's own
  helpers (`system`/`user`/`assistant`/`tool_result`/`tool`, real `chat_pb2` protos) and parses
  the `Response` back (text, tool calls, citation footer). Live Search is wired here when the
  persona has **opted its `web_search`/`x_search` built-ins in** (issue #168): they become a native
  `SearchParameters` object (`web_source`/`x_source`), and grok searches itself. The grok **media**
  tools stay their own httpx `PlatformTool`s (`_grok.py`), independent of the chat SDK, per-persona.
- **Single native surface.** Declares `SURFACES=("native",)` / `DEFAULT_SURFACE="native"`, so
  `AI_SDK_SURFACE` is unset and any other value fails clearly (the issue #163 surface contract).
- **Routing.** `AI_SDK=xai-sdk` builds it (requires `AI_PROVIDER=xai` — the native endpoint);
  shipped as the optional extra `[xai-sdk]` (pins `xai-sdk>=1.17,<2`). gRPC errors map onto the
  provider hierarchy (auth / rate-limit / connection).
- **Tested against the real SDK, offline.** No httpx transport to respx-mock, so tests build
  **real** protos and inject a **fake client** (no socket) — the openai adapter's "real SDK,
  mocked transport" discipline, gRPC-shaped. The tool-neutral migration is proven: an `xai-sdk`
  persona with opted-in grok tools keeps them; an empty-overlay (adversarial) persona resolves
  with **no** powerful and **no** platform tools — the SDK arms nothing.

**Boundary:** live verification on the real grok endpoint (a measured chat turn, Live Search
returning real citations, a tool round-trip) is **the capital's** job on the migrated personas;
the offline tests assert the harness's half (the wire it sends, the response it parses).

### Eddie Murphy — the xAI-native profile (Live Search + grok media)

*(Collapsed to corrected facts — the original section's dated narrative described the
pre-#163 `httpx`/`OpenAIResponsesProvider`/`AI_PROVIDER_API` era, now superseded. The
authoritative statement of the provider/SDK/surface matrix is `CLAUDE.md` → "Architecture —
The Spine", point 3; the chat-brain adapter details are in "Native xAI Adapter" above and the
image coverage in "Image Tools" above.)*

Eddie is the fully-xAI persona — the "done-bar" acceptance work proving the whole grok stack
end to end. Two axes stay straight: the **provider adapter** (harness code / wire format) vs.
the **endpoint vendor** (`base_url`). Corrected facts:

- **Chat brain.** `AI_PROVIDER=xai` selects xAI's endpoint (`https://api.x.ai/v1`, `AI_BASE_URL`
  overrides). The end-state brain is the native `xai-sdk` adapter (#165); `AI_SDK=openai` at
  `api.x.ai` over the `responses`/`chat` surface (#163) stays a supported alternative cell. The
  hand-rolled `OpenAIResponsesProvider` is **deleted**.
- **Live Search** is server-side, not a function tool (`_defaults/tools/xai_search.py`):
  `web_search` (live web) + `x_search` (live 𝕏). xAI runs it from a top-level `search_parameters`
  body field (native adapter → a `SearchParameters` proto; openai-at-xAI → `extra_body`) — it does
  **not** accept OpenAI's `tools:[{type:web_search}]` entry. Citations ground the reply via the
  existing parsing. Both are **opt-in** powerful tools (#168) — per-persona, never provider-gated.
- **grok media tools** (`_grok.py`, httpx `PlatformTool`s, independent of the chat SDK):
  `grok_generate_image` (text → image, `grok-imagine-image-quality`) and `grok_generate_video` —
  the harness's **first video capability**, text→video and image→video, over xAI's **asynchronous**
  endpoint (submit → poll `GET /v1/videos/{id}` until `done` → download → upload as an inline
  Asset). Full `duration`/`aspect_ratio`/`resolution` coverage; failures relay xAI's actual message.
- **Shared plumbing (`_media.py`)** — legible error relay, magic-byte format **sniffing** (Asset
  extension follows the real bytes), safe-filename building — shared by the OpenAI and grok media
  tools. Enum/range constraints are API-enforced, not re-validated here.

**Boundary:** offline tests assert the harness's half; the NOC provisions Eddie and the capital
runs the live matrix and closes the handoff.

### Orphan-Artifact Sweep — GC deleted timelines' on-box state (issue #192)

When a Timeline is destroyed on the platform, **nothing on the fleet server is cleaned up by
itself.** The harness persists per-timeline state under `$HARNESS_HOME` — chiefly the session
transcript (the full conversation), plus marks/seen/claims/breaker index files — and had no
deletion handler, so a destroyed timeline's content would survive on the box indefinitely. The
`basecradle-harness-cleanup` entrypoint (`_cleanup.py`) is the periodic **orphan sweep** that
GCs it. **Sweep-only by design (founder-settled):** the platform's `timeline.deleted` event is
best-effort/droppable, so an event-driven cleanup can't be trusted alone; a periodic sweep is
mandatory regardless, and the *same* sweep backfills already-deleted timelines for free (the
first run on a box is the backfill — past and future deletions are one code path). No router or
Rails change; we don't consume `timeline.deleted`.

- **The classify switch is the whole safety.** Each referenced UUID is checked with one cheap
  `client.timelines.get(uuid)` (no model call): **only a clean `NotFoundError` (404) purges.**
  Success (200) keeps; `ForbiddenError`/`NotAViewerError` (403 — exists, agent not a viewer)
  keeps + logs; **any** transient error (connection / rate-limit / 5xx / generic
  `BaseCradleError`) keeps and retries next run. *A platform outage must never read as
  "everything deleted" and trigger a mass purge — default to keep on anything but a 404.*
- **The invariant — memory deliberately persists across timeline deletion and is never swept.**
  The sweep operates *only* on the five artifact dirs (`sessions/`, `marks/`, `seen/`, `claims/`,
  `breaker/`) and **never touches** `memory.db` (+ `-wal`/`-shm`) or the MemPalace palace dir. If
  a peer told the agent its birthday on a since-deleted timeline, the agent must still remember
  it. (By construction: memory is never enumerated, so a purge can't reach it.)
- Idempotent + crash-safe (re-derives the set from disk each run; a half-done purge finishes
  next run); reuses `_client_from_env` and the stores' `quote(..., safe='')` filename
  convention. `--timeline <uuid>` is a manual unconditional ops purge.

**Boundary:** the schedule unit lives in `deploy/` (captain authors it); the **NOC deploys it**
(sole deployer) per agent, scoped to that agent's `$HARNESS_HOME` + `BASECRADLE_TOKEN`. Live
verification (drive a wake, delete the timeline, sweep, confirm the five artifacts go and memory
stays) is **the capital's** job, post-ship.
