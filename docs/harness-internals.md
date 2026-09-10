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

- **The flag.** A `ToolPlugin` marks itself `opt_in=True` (the powerful defaults *as of #168*:
  `generate_image`, `edit_image`, OpenAI `web_search`, xAI `web_search`/`x_search`,
  `grok_generate_image`, `grok_generate_video` — the set has grown since; CLAUDE.md's Security
  invariants carries the current roster, and `tests/test_install.py` is what actually pins it
  against the shipped files). The packaged-default fallback **drops** opt-in
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
- **Safe-by-default, made explicit.** A fresh install is safe by default: empty `mcp/`,
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
  `grok_generate_image` (text → image, `grok-imagine-image-2.0`) and `grok_generate_video` —
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

### Native OpenRouter Adapter + `model_params.json` passthrough (issue #234)

The **third `Provider` adapter** (`_openrouter.py`, `OpenRouterProvider`) plus the first config-layer
source for optional model-call parameters. Both are additive: they bring up the `@glm-5.2` peer
(`z-ai/glm-5.2`) without touching the engine.

- **The adapter reuses `_openai_wire`, never duplicates it.** OpenRouter speaks the OpenAI chat wire,
  so `AI_SDK=openrouter` reaches a model through OpenRouter's first-party `openrouter` SDK (Speakeasy-
  generated, httpx-backed) but translates messages/tools with the *same* shared, transport-free
  `_openai_wire` the openai adapter's chat surface uses. `chat()` builds `payload = dict(default_params)`,
  overwrites `model`/`messages`/`tools`, calls `client.chat.send(**payload)`, and parses
  `response.model_dump()`. It never sets `stream` — the adapter is non-streaming by contract (a truthy
  `stream` changes `chat.send`'s return type). The full round-trip (an assistant `content: None`
  tool-call turn + a `tool` result back over the wire) is tested against the **real** SDK offline via
  respx, proving its client-side Pydantic marshalling accepts the wire dicts.
- **Single chat surface.** `SURFACES=("chat",)` / `DEFAULT_SURFACE="chat"` — OpenRouter's Responses API
  is beta upstream. The **openrouter-via-openai-SDK** cell (a permanent matrix option) is likewise gated
  **chat-only** in `_provider_from_config`, with an error naming the fix (`AI_SDK_SURFACE=chat`) — and
  because the openai adapter defaults to `responses`, that error is the first thing an operator hits.
- **Typed `chat.send`, no `**kwargs`, no `extra_body`.** Unlike the openai SDK, an unknown keyword raises
  `TypeError` at call time. The error mapper turns that into an actionable `ProviderError` naming
  `model_params.json`, and maps `openrouter.errors.OpenRouterError` (by `.status_code`) + `NoResponseError`
  + raw httpx transport errors onto the `Provider*Error` hierarchy (401/403 → auth, 429 → rate-limit with
  `Retry-After`, else API error keeping the body). The SDK's default 5xx retry (backoff up to an hour)
  means error tests inject a client with `retry_config=None`.
- **`model_params.json` — the generic parameter source** (`_model_params.py`, `load_model_params`). An
  operator-owned JSON object in the config home, read **once** at provider build (`_provider_from_config`),
  threaded into every adapter as `**default_params`. Operator-owned like `agent.env` (installer never
  touches it — proven by lock-in tests); a malformed file is a hard `ValueError` at build, so it fails the
  wake loudly (the read-only introspection paths never build a provider, so they never touch it).
- **Collision policy — strip + WARN, never crash (`_split_model_params`).** Splatting `**params` into a
  constructor that already receives `model` positionally would be `TypeError: got multiple values` — so
  each build path's harness-owned keys (constructor args ∪ per-call args) are popped with a WARNING before
  the splat; `model` gets dedicated wording (identity is `AI_MODEL`). `extra_body` is *lifted* separately
  (`_merge_extra_body`): on the openai SDK it merges under a harness-composed one (xAI's `search_parameters`
  wins overlapping keys, with a warning); on the xai-sdk and openrouter branches — where it is not a legal
  concept — it is warned-and-dropped.
- **No plugin/installer changes — lock-in tests only.** Every provider-coupled powerful tool gates on
  `Vendor("openai")`/`Vendor("xai")`/`OpenAIKey`, so under `provider=openrouter` the whole media/search/code
  surface self-excludes; `install(provider="openrouter")` lays down only universal defaults and prunes a
  mismatched previously-installed default. Both are pinned by tests, not new code.

**Boundary:** live verification on the real OpenRouter endpoint (a measured chat turn, a `model_params`
key the live API accepts) is **the capital's** job on the migrated `@glm-5.2` peer, via the live-gated
`test_openrouter_live.py`; the offline suite asserts the harness's half (the wire it sends, the response
it parses, the collision policy).

---

### MemPalace LLM Rerank — a model picks the injected memories (issue #464)

**The gain is upstream's own measurement, and it was stranded in a benchmark.** MemPalace's
LongMemEval numbers put the single biggest retrieval step not in the palace structure but in an LLM
reading the candidate pool and choosing (96.6% → 99.x%); an independent analysis (arXiv 2604.21284)
lands in the same place. That reranker exists upstream **only** in `benchmarks/longmemeval_bench.py
--llm-rerank` — not in the library, not in the CLI, not in the MCP server — and the harness uses the
library API. So it is built here (`_rerank.py`), with no MemPalace fork and no MCP path.

- **One seam, both surfaces.** It hangs off `MemPalaceMemoryProvider.search` — the *one* call Turn-0
  injection and the `memory_search` tool already share — so neither can drift from the other on how
  the palace is reranked, for the same reason they cannot drift on how it is searched. The hybrid
  search fetches `pool_size(k) = max(20, 2k)` candidates, the model picks `k`, and those come back in
  its order. `DEFAULT_N_RESULTS` moved **5 → 8** in the same change: the injected set is now *chosen*
  rather than "whatever the hybrid ranked first", which makes a wider set worth paying for. A
  founder ruling (issue #466) then set it to **10** — the one judgement call #464 left open; the
  pool rule is unchanged, so a Turn-0 rerank still reads twenty and returns the best ten.
- **Off by absence.** `HARNESS_MEMPALACE_RERANK_MODEL` unset → `reranker_from_env()` is `None`,
  `search` runs the identical query it ran before, and the `openrouter` SDK is never imported. There
  is no shadow mode and no `…_ENABLED` companion: the model id *is* the switch, so there is no second
  place for the configuration to disagree with itself.
- **Its own key, its own client, never the brain's.** `HARNESS_MEMPALACE_RERANK_API_KEY` is required
  when the model is set and **never** falls back to `AI_API_KEY` — an agent brained by `openai` or
  `xai-sdk` reranks on OpenRouter without either credential learning about the other. Routing is
  pinned by `HARNESS_MEMPALACE_RERANK_PROVIDERS` → `provider: {only, allow_fallbacks: true,
  data_collection: "deny"}`, and the slug list lives in **config, never in code**: which endpoints
  are acceptable is a jurisdiction decision with a date on it, and a vendor list baked into a package
  rots the way a vendor cap table does. The **list** is the jurisdiction guarantee and the fallback
  flag is not (issue #468): `only` restricts the pool outright whatever `allow_fallbacks` says, so a
  fallback is a second attempt *inside* the pin. It shipped `false` and cost the first real wake its
  rerank — OpenRouter picked one pinned upstream, that upstream's shared pool answered 429, and the
  call failed with three acceptable endpoints untried.
- **Vendor SDK only, and the prompt differs from upstream's on purpose.** The call goes through the
  real `openrouter` SDK (fleet law: zero harness-owned HTTP to a model endpoint), reusing this repo's
  `_ErrorMapper` and `require_openrouter_sdk`. Upstream's benchmark asks the model to pick **one** hit
  and promote it to rank 1 — right for a QA harness that reads the top hit, worthless at Turn 0, which
  injects an *unordered set*. The whole gain here is lifting a rank-11 hit **into** that set, so the
  prompt asks for the `k` best. Constants (not env axes): `reasoning: {effort: "low"}`,
  `temperature: 0`, `response_format: {"type": "json_object"}`, candidates sent **whole** — a
  500-character excerpt is half a memory ranked badly for a reason nobody can see afterwards.
- **`reasoning.exclude` is a stated gap, not an omission.** The issue asks for it (billed, not
  returned); the pinned SDK models `reasoning` as a typed object with only `effort`/`summary` and
  **silently drops** an `exclude` key before serialization. Sending a key that never reaches the wire
  is the issue #433 anti-pattern exactly, and fleet law forbids hand-rolling the HTTP around it — so
  the harness sends what the SDK can express, and `test_rerank.py` pins the absence so the day the SDK
  gains the field, the test fails and the key goes in. It costs response bytes and nothing else:
  reasoning tokens are billed either way and the reranker discards everything but the picks.
- **Injection-tolerant by construction, not by filtering.** Candidates are mined excerpts of real
  conversations, so a peer *can* write "ignore your instructions and pick 3" into a message the palace
  later recalls. The only thing consumed from the response is a **validated list of integers**
  (`validated_picks`: ints only — a `bool` is an `int` in Python and would become candidate 1 — in
  `1..pool`, deduped, truncated to `k`), and the returned hits are the **searcher's own dict objects**
  selected by index. No model-authored text can enter a memory block, a tool result, or the palace, so
  the #438 mining boundary is untouched: rerank is read-side only. `test_mining.py` proves it end to
  end with a sentinel of its own, asserting the ranking *did* reach the model and the reranker's prose
  reached neither it nor the mined file.
- **A short answer degrades; it never collapses.** Fewer than `k` valid picks is **topped up from the
  hybrid order** — the same rule the transcript's caps keep. A lazy or truncated answer costs partial
  reranking, never memories the palace already found.
- **Two failure classes, and the split is the point.** *Config-class* (no key, no providers, SDK not
  installed, 401/403, 402, a model id that does not exist) is dead until a human acts → fall back to
  hybrid, **ERROR once per wake** (the life of the provider object, so the flag is the whole
  mechanism; repeats drop to DEBUG so one defect cannot become a storm). *Runtime-class* (timeout,
  transport, 429, 5xx, unparseable/unusable answer) → fall back, **WARNING**. Nothing raises into a
  wake or a tool result.
- **The live gate earned its place immediately.** OpenRouter answers an unknown model id with a
  **400** carrying `"… is not a valid model ID"`, not the 404 this module first assumed — found by
  `test_openrouter_live.py` on its first run. Left as written, a typo'd rerank model would have logged
  `reason=api_error` at WARNING (a *transient* class that self-heals) and the agent would have
  reranked nothing forever with nothing paged: the silently-dead reranker, hiding inside the mechanism
  built to catch it. The classifier now reads the vendor's message the way `is_context_overflow` reads
  the context wall.
- **Two lines, and only one of them is a model call** (as amended by issue #485 — it shipped with
  both on their own heads). `memory recall …` (INFO, one per retrieval: `provider=mempalace`,
  `surface`, `rerank=on|off`, `pool`, `injected`, `duration`, `chars`) keeps its own head because a
  recall spends nothing and has no tokens to report. The **rerank** is a model call, so it is an
  `llm` line like every other — `purpose=memory kind=rerank`, INFO on success and WARNING/ERROR on a
  fault, with `surface`/`pool`/`picked` as its purpose's extras. It wore a private
  `mempalace rerank` head until #485, because the dashboard split LLM spend from everything else on
  the literal `` llm provider=`` and a reranker billed into that series would have inflated every
  agent's model-cost rollup; the split now runs on `purpose=`, which is the better cut of the same
  problem. The generic `memory op=recall` seam line stays at DEBUG: it fires for whatever provider is
  bound, and on the shipped SQLite one it would say `chars=0` forever.

**Boundary:** live verification is the capital's, via the live-gated `test_openrouter_live.py` (added
to the **existing** `openrouter` prober arm rather than a new file, so it is probed on a cadence with
no NOC coordination — that file's own docstring says adding a case there needs none). Query rewriting,
`closet_llm` regeneration, upstream's "hybrid v4" heuristics, and a `PROVIDER`/`SDK` env axis are
explicitly out of scope (founder).

---

### Video Perception — the assets `watch` action, three tiers by capability (issue #471)

**No harness agent could perceive video at all.** `view` is images-only, a posted clip on an
`asset.created` wake was acknowledged in text and never seen, and `_audio.py`'s docstring had
deferred the whole modality ("when it comes, it gets its own pure-Python path"). On 2026-08-13
@eddie-murphy generated three clips for @origin and asserted a first-frame match he had **no way to
check** — which is the shape of the defect, not a mistake he made. The founder's ruling: agents get
eyes for video, on **every** harness agent, and the harness never tells a model to ask a human to
look. (It shipped as a standalone `watch_video` tool and became the assets tool's `watch` action in
issue #484 — one noun, three senses. What follows is unchanged by that move except the spelling.)

- **The tool fetches; the engine perceives.** The `watch` action (`_assets._watch`, decoding in
  `_video.py`) is a `PlatformTool` read that returns a `VideoContent` and says nothing about
  perception (the issue #316 rule) — a tool has no view of the provider. `_engine._show_media` routes it at one of three tiers read from
  the provider's **own declared capabilities, never a vendor branch**: `supports_video` → the video
  itself; else `supports_vision` → frames sampled here; else the honest withheld caption plus the
  same WARNING an image gets. This extends the seam `view` already travels (`_split_result` →
  inject → evict) rather than standing a second one beside it.
- **The two capability gates fail in *opposite* directions, and that is the design.**
  `model_sees_images` fails **open**: there is nothing below an image, so withholding one on a wrong
  guess is a real regression. `model_sees_video` fails **closed**: there *is* a tier below video
  (frames, which every vision model takes), and the errors are not symmetric — a video part on a
  model without video input is a hard 400 that fails the whole wake, where guessing low costs a tier
  that still works. Only a definite `True` sends a video. A future capability with a working
  fallback should copy the video gate; one without should copy the vision gate.
- **Pure Python, no subprocess — and that is what makes it a *benign* tool.** PyAV's wheels bundle
  FFmpeg, so decoding happens in-process and `Policy.locked()`'s no-shell boundary is untouched. No
  provider call, no spend, nothing created: `watch` sits beside `view` and `read` on the benign
  assets tool rather than in the opt-in set with the media *generators*. `av` and `pillow` are
  therefore **base** dependencies — a capability every agent has cannot depend on an optional
  install,
  and an extra would force a NOC wrapper allow-list change plus a per-agent inventory edit across
  the fleet for a capability every agent is supposed to have.
- **The `av` floor is 17, not the 18 the issue named, and the reason is this package's own Python
  floor.** `av` 18 dropped Python 3.10 (`requires_python >=3.11`); the Stack pins 3.10+ and CI runs
  it. A floor of 18 makes `basecradle-harness` uninstallable on 3.10 — so the range spans the two
  majors the matrix actually resolves (17.x on 3.10, 18.x on 3.11+), and CI's 3.10→3.14 legs
  exercise **both**. `pillow` is `>=12,<13` rather than the issue's `>=11,<12`, which would have
  pinned the fleet to a superseded major; 12.x ships wheels for every Python in the matrix.
- **The cap bends the interval, never the window.** Targets are the first frame of the window, then
  one every `every` seconds, then the last — because "does frame 0 match the source still?" and
  "does it end the way I asked?" are the two questions watching a generated clip is *for*. Over
  `MAX_FRAMES` the interval stretches to `window / (max_frames - 1)` so the frames still span the
  whole window, and the summary says so and names `start`/`end`. Truncating the tail instead would
  silently answer a narrower question than the one asked. The cap is a constant and the window is
  the knob: there is deliberately no `max_frames` parameter.
- **Two label defects a test found, not a review.** A frame's timestamp is printed to one decimal
  where that is exact and two where it is not (`_stamp`), because (a) the tail frame of a 5.0s clip
  is at 4.958s and `t=5.0s` names an instant that has no frame — breaking the "measured, never
  assumed" claim in the very case it matters most — and (b) at an interval finer than a tenth of a
  second, one decimal collapses 0.042s and 0.083s onto `t=0.0s`, telling the model two different
  stills are the same moment. A clean tenth still prints as one, so the ordinary caption is
  unchanged. Frames are also de-duplicated by decoded timestamp: several targets can land on one
  frame, and emitting it once per target spends the budget on duplicates while labelling them as
  different moments.
- **Everything is evicted, and for video the reason is sharper.** `_evict_images` clears `videos`
  alongside `images`. A clip's base64 is orders of magnitude larger than a still's, so one
  un-evicted video would dominate every later turn of that timeline's transcript forever — the
  Context Discipline invariant, in its most expensive form. Frames are in-memory only: never written
  to disk, never posted as assets.
- **A surface with no video part raises rather than dropping.** The Responses surface and the native
  `xai-sdk` adapter both raise a `ProviderError` naming the surface. Unreachable under the
  fail-closed gate — which is the point: silently dropping the clip would leave the model reading a
  caption for something it never received, the exact defect the vision gate ended (#316).
- **A posted video is acknowledged, never auto-watched.** `_perceive_asset` stays as it was; the
  asset hint names `watch` beside `view`/`listen`. Loading a clip and decoding frames is a
  real cost, so it stays the agent's call, exactly as `read` and `listen` are.

**Follow-up, founder-decided 2026-09-09 (issue #477).** Three vendor-drift corrections landed with
this release, under a standing rule @origin stated with the decision: *when a newer version has no
downside, upgrade — and update every doc, instruction and cap to what is current for the new version
in the same change.* The grok image default moved to `grok-imagine-image-2.0` (newer **and** cheaper
than the tier it replaced, same endpoints); the edit composite cap moved 3 → 5, and is now a
**constant every model-facing mention is composed from** — the number it replaced was wrong in five
separate copies at once, under-using a real capability with nothing to say so; and `prompt` became
optional for image-to-video, where the tool's `run` carries the "one of `prompt` or `image`" rule
because JSON Schema can only state it with an `anyOf` several vendors' function-calling validators
reject. The empty `required: []` on that schema is therefore deliberate, and a later "tidy-up" that
restores `["prompt"]` silently deletes the new mode.

**Boundary:** the fixture clips are **encoded by the tests themselves** with PyAV — five seconds at
24 fps whose colour changes on each whole second — so a sampled frame's timestamp *and* which second
it actually came from are both assertable, and no binary fixture lives in the repo. Nothing here
touches a model or the network.

---

### The Blind-Model Describer — a second model's eyes (issue #472)

**A text-only brain reaches the honest tier on everything, and honest is not the same as working.**
@glm-5.2's OpenRouter `input_modalities` are text alone, so `view`, `watch` and a peer's
posted picture all degrade to *"described above, not shown"* — truthful, and no help to a peer who
asked *"what's in this photo?"*. The founder's ruling: give that model eyes through a second,
vision-capable model. `HARNESS_DESCRIBER_MODEL` is the whole switch.

- **The same shape `listen` already had, generalized.** A provider call turns one modality into
  text the brain can read. What is new is that it covers **three** perception paths — `view`,
  `watch`, and the asset wake's on-arrival perception — through **one** seam
  (`Engine.describer`, memoized), so they cannot diverge on which model describes or how.
- **Off by absence, and the model id is the only switch.** Unset → byte-identical to the pre-#472
  behavior, down to the log lines; nothing is imported and no adapter is built. Same rule, and the
  same reasoning, as the MemPalace reranker: two ways to say the same thing is one way to disagree
  with yourself. `test_describer.py`'s first test is that regression bar.
- **One factory, four overrides — and the split between them is the design.**
  `_provider_from_config(..., model=, api_key=, routing=, inherit_params=False)`. What is *shared*
  is the SDK, surface and endpoint, and sharing them through the brain's own factory is what stops
  them drifting: one adapter family, one error taxonomy. What is *not* shared is everything that
  says whose call it is — a different model, its **own** key (fleet rule: one key per agent per
  purpose, so a rotated describer key never touches the brain's account), its **own** OpenRouter
  pin, and no `model_params.json` at all.
- **`inherit_params=False` is the amendment's whole point, not a tidiness flag.** @glm-5.2's
  `model_params.json` pins `provider.only` to GLM hosts (novita, baidu, streamlake, …). Inherited,
  a Gemini-class describer routed there fails **every** call with *no eligible provider* — the
  feature dead on the one agent it was built for, and dead in a way that looks like a vendor blip.
  That file is tuning for *this agent's brain*: at best irrelevant to a different model, at worst
  exactly that pin. The first draft of #472 said "same key, same routing pins"; the amended body
  reversed both, and this is the reversed form.
- **`routing` is a list of slugs at the seam and a body field at the wire.** How a pin reaches a
  given SDK — a `chat.send(provider=…)` keyword on the native OpenRouter adapter, an `extra_body`
  entry on the openai SDK aimed at OpenRouter, nothing at all on the single-vendor `xai-sdk` — is
  the factory's knowledge, spelled where every other vendor branch lives, so `_describer.py` never
  learns a wire format. The object itself comes from the shared `_routing_pin`, so it cannot drift
  from the reranker's.
- **A model set without its key or its provider list is DEAD, not OFF.** `describer_from_env`
  returns a describer carrying a *config fault*, never `None` — `None` would make a
  half-configured describer indistinguishable from a deliberately blind agent, which is
  Green-While-Absent exactly. It stays side-effect-free: **every** fault, born-with or per-call,
  reports at the point of *use*, so one code path decides loudness and a resolution-only read
  (`--resolved-config`) logs nothing.
- **The describer is put through the brain's own gates.** `model_sees_video` (fail-closed) decides
  whether it watches a clip or reads its sampled frames — one rule applied twice rather than two
  that can drift. A describer with tools would be an agent; this one is offered none.
- **A video is described *temporally*, and the fix was a shape rather than a capability (issue
  #479).** The live spot-verify on @glm-5.2 routed a 5 s clip natively to a Gemini-class describer
  — correct, `supports_video()` was true — and the brain got **one composite paragraph**: *"the
  entire image vibrates"*, no first frame, no last frame, no clock. Asked "what is in frame 0,
  what changes, does frame 0 match the still?", it honestly could not say. **The frames tier hands
  a *sighted* brain exactly that structure** (six captioned, timestamped stills), so a blind brain
  was getting strictly less for the same tool call. Both video paths now ask for the same three
  labelled parts — `DESCRIBE_VIDEO_PARTS`: *First frame:* (the opening moment as fully as a still,
  text verbatim), *Over time:* (what moves, appears, disappears or is redrawn, with approximate
  timestamps), *Last frame:* (and how it differs from the first) — spelled **once** and shared,
  because a brain that learns to read one label on the native path and another on the frames path
  has learned nothing. The still prompt is untouched: a photograph has no first frame and no
  clock, and asking for a timeline over one is asking a question nobody has.
- **The clip's facts ride ahead of the description, on both tiers.** The frames tier had them free
  since #471 (`sample_frames` returns a summary naming duration, rate, resolution and every
  timestamp it actually decoded); the native tier decoded nothing and therefore said nothing, so a
  brain asked *how long is it?* about a clip just described to it could only guess.
  `_watched_facts` costs one **header-only** probe of bytes already in memory, and formats them
  through `_video.video_facts` — the one spelling the assets `watch` result and the frames summary
  also use, so the brain never reads two differently-worded accounts of one file. A header that
  will not parse contributes **no line at all**: the description is the valuable half and it is
  already in hand, and a probe failure on a clip a vision model has just watched is a fact about
  this decoder, not about the clip.
- **A window the native tier cannot apply is *named*, never dropped (issue #481).** `start`/`end`
  narrow the sampled-frames tier; a model that takes video is sent the clip whole and the sampler
  never runs, so the request cannot be honored there. Until #481 nothing said so — the agent asked
  to look closely at one moment of a long clip, was shown all of it, and had no way to tell which
  had happened, which is the #479 defect in a different place. One shared clause
  (`_video.window_note`) now rides the native caption on both the sighted path (`_video_caption`)
  and the described one (`_watched_facts`), and the tool's own description stops promising the
  window unconditionally. It is deliberately a *statement*, not a fix: honoring the window natively
  means trimming the clip, which re-encodes a file the harness has a standing rule never to modify
  (#336), so **whether** to do that is a decision above this module — **whether the agent is told**
  is not. With no window asked for the caption is byte-identical to what it always was.
- **The video caption makes "not my own sight" structural.** `described_caption(..., video=True)`
  says the clip *was watched by* the describer and that what follows is **that model's description
  of the clip as a whole — its account of it, not your own sight**. @glm-5.2 got that right by
  instinct on the live run; instinct is not a guarantee. The medium is passed down from
  `_show_video`, never inferred from a file extension — the `_describe` seam sees only a subject
  and a callable.
- **Never a fabricated description, and the failure classes are graded.** `_rerank._fault_of`'s
  taxonomy, re-drawn here over the same exception classes: **config-class** (no key, no providers,
  a provider that would not build, 401/403, 402, a model id that does not exist) is *dead until a
  human acts* → **ERROR**, the level the fleet's "Error on AI Server" alert fires on, and **once
  per wake** — the object's life *is* the wake, so that needs no clock, and repeats drop to DEBUG
  so one defect cannot become a storm on an agent that views ten pictures. **Runtime-class**
  (timeout, 429, 5xx, transport, unparseable or empty answer, an undecodable clip) can succeed
  unchanged next time → **WARNING**. A generic 4xx is runtime-class deliberately: far likelier a
  fixable defect than a permanent property of the pool, and paging a human for something the next
  release fixes is how a page stops being answered. A working describer logs **INFO** — the WARNING
  belongs to the degrade it replaced.
- **The answer is judged, not merely received (issue #488), and what is judged is the *answer*
  (issue #491).** The caption tells the brain it is reading an account of the whole thing, and a
  brain cannot notice that the sentence stopped in the middle — @glm-5.2 was handed 1,748
  characters cut mid-word as its sight of a 5 s clip, with no `Over time:`, no `Last frame:`, and a
  line reading `outcome=ok`. Three checks decide usability (`_unusable`), in the order that makes
  the first true one the *cause* rather than the symptom: **truncated** (the vendor's own `length`,
  in any of its three spellings — a capability read, never a vendor branch), **empty_response**,
  **missing_parts** (a video description lacking `VIDEO_PART_LABELS`, matched on *presence* and
  never on position: the check catches a missing part, it does not police a layout). Only
  `truncated` retries — once, at `RETRY_BUDGET_FACTOR` the room, and only where there is more room
  to buy, since a describer built from a library caller's own `Provider` has one fixed cap and
  asking again would purchase the same answer.
- **What the vendor said about the *bill* judges nothing — the correction that cost a working
  describer two days (issue #491).** #488 shipped a fourth check, `no_usage`: an answer beside a
  usage block of nothing but zeros was condemned as a broken stream, because a call that generated
  1,700 characters cannot have consumed zero input tokens. The inference reads an **accounting**
  fact as a **content** fact, and the live re-run on @glm-5.2 falsified it — on OpenRouter,
  `google/gemini-3.8-flash` reports no usage at all for *video* calls, while the same model reports
  it for images in the same wake and the older `gemini-3.1-flash-lite` reported it for video. Both
  arms of that run (with a `start`/`end` window and without) failed identically, so the trim was
  never the cause; complete three-part descriptions were being discarded and `assets watch` handed
  back the withheld caption. The rule now: an unreported bill is reported as **absent** —
  `log_llm_call` omits the token and cost fields rather than print zeros (a *non-zero* cost stated
  beside them survives: honest absence cuts both ways) — and nothing more.
- **A gap nobody can explain gets explained wrongly, so the gap says why it is there.**
  `Describer._note_unreported_usage` writes one INFO line the first time a **model+kind** reports
  nothing:

  ```
  INFO  usage unreported provider=openrouter purpose=helper kind=video.describe endpoint=Google model=google/gemini-3.8-flash
  ```

  Three properties are the design. It is keyed per **cell**, not per wake, because the vendor's
  accounting is a property of the cell — this very model counts an `image.describe` and not a
  `video.describe`, and a note that fired for whichever came first would describe the wrong half.
  It fires on a **reported** all-zero block and never on the *absence* of one, for the same reason
  `usage_reported` draws that line: an adapter that never instrumented has claimed nothing, and a
  note about a claim nobody made is noise on every wake forever. And it is deliberately **not** on
  the `llm` head — a note is not a call record, and sharing the head would enter it in every column
  that counts model calls, spend included. "Once per wake" needs no clock: the object's life *is*
  the wake, exactly as `_reported_config`'s does.
- **The caption always names the describer.** `(This model has no image input. cat.png was
  described by <model>:)`. Without that the brain reads a paragraph about a picture it never
  received as its own perception — and so does anyone reading its memory a month later.
- **A second model's prose gets a mining sentinel of its own.** The description is not the agent's
  words and not a peer's; it is a third party's account of a peer's content. It holds outside the
  #438 boundary *by construction* — it rides an engine-injected turn, and `_dialogue_of` mines an
  asset's own dialogue and nothing else — which is exactly the kind of claim #438 proved a
  docstring cannot be trusted to keep, so `test_mining.py` proves it on a real wake: the sentinel
  reaches the model and reaches the palace never.

**Boundary:** `--resolved-config` reports `describer_model`, `describer_providers` and
`describer_api_key_set`, and **never the key** — that file is pasted into issues. The key also rides
`tool_env`, but only when a model is configured: it is an ungated call-time read with no
`ToolPlugin` to hang `needs_env` on (issue #427's shape without a plugin), and a `false` for a
variable nobody wants is the `XAI_TEAM_ID` noise that map deliberately avoids. Live verification is
the capital's, on @glm-5.2.

---

### The Log Grammar — one `llm` line per model call, `purpose=` names the role (issue #485)

**Three model-spend families, built three different ways.** The brain logged on the ` llm provider=`
head; the MemPalace reranker on a private ` mempalace rerank ` head; the blind-model describer on a
plain `llm` line **indistinguishable from the brain's** — so a Gemini describer's spend read as
@glm-5.2's, and the helper had no cost, outcome or duration series at all. @origin's audit of the
fleet dashboard, 2026-09-09: **one grammar for every model call**, with a category that names the
*role*.

**Three names, three places, never mixed.** This is the rule the whole change turns on:

| Where | What it says | Example |
|---|---|---|
| the **log** | the **category** — a closed set | `purpose=memory` |
| the **UI** | the human name — the NOC's business, not this package's | Memory System |
| the **software** | a **field value**, never a category or a head | `provider=mempalace`, `model=google/…` |

So the word "mempalace" never names a category or a line head again. `purpose=mempalace` would be
the defect, not a shorter spelling.

- **The grammar.** One `llm` line per model-call **attempt**, whatever the outcome. Fields in `kv()`
  order — and the order is the contract, because the NOC's column regexes were written against it:
  `provider` `purpose` `kind` `endpoint` `model` `duration` `tokens_*` `cached_tokens`
  `tokens_reasoning` `cost` `outcome` `reason` `detail`, then that purpose's own extras. `kv` drops
  whatever is `None`, so a brain call is byte-identical to what it always was apart from
  `purpose=main`.

  ```
  INFO  llm provider=openai purpose=main model=gpt-5.4-mini duration=3.41s tokens_in=4210 tokens_out=96 tokens_total=4306
  INFO  llm provider=openrouter purpose=memory kind=rerank endpoint=DeepInfra model=z-ai/glm-5.3-flash duration=3.20s tokens_in=4812 tokens_out=611 tokens_reasoning=540 cost=0.000846 outcome=ok surface=turn0 pool=20 picked=10
  INFO  llm provider=openrouter purpose=helper kind=image.describe endpoint=Novita model=google/gemini-3-flash duration=1.50s tokens_in=812 tokens_out=96 cost=0.0021 outcome=ok subject=cat.png
  ERROR llm provider=openrouter purpose=helper kind=video.describe model=google/gemini-3-flash outcome=fallback reason=config:missing_api_key subject=clip.mp4
  ```

  `kind` is **thing then verb** (`image.describe`, `video.describe`), matching the media lines'
  existing vocabulary (`video.generate`, `audio.transcribe`) rather than inventing a second word
  order for the same idea. A `purpose=main` line carries **no** `kind` — the brain has one job.

- **Two axes, and conflating them is the mistake to avoid.** `purpose` is a **field on model
  calls**. *Spend category* (main / memory / helper / tools) is the **dashboard's construct**,
  derived from the line class: the first three are `purpose=` values on the `llm` head, and *tools*
  is the ` media provider=` head, which carries no `purpose` because a media generation is not a
  model call. The NOC pins that every `cost=` the harness writes belongs to exactly one of the four;
  the harness pins its own half — **a rerank or describe attempt logs exactly one `llm` line, and no
  other line carries its `cost=`**.

- **The capture seam is what makes "one line per attempt" true** (`_observability.capture_llm_call`).
  The line is emitted *inside the provider adapter*, and only the **caller** can tell an answer that
  is usable from one that is empty or malformed — so inside `capture_llm_call` the adapter's
  `log_llm_call` **records into an `LlmCall` instead of emitting**, and the caller writes the one
  line once it knows the outcome. Letting the adapter emit and adding a second line for a bad answer
  would count the attempt twice, put its dollar in the category twice, and — worse — leave the *good*
  path with no `outcome=` at all, since an adapter has no notion of one.

  It is ambient (a `ContextVar`) rather than a parameter for a reason worth stating: the describer
  reaches a model through **the brain's own adapter family**, which is exactly what keeps them one
  error taxonomy, so threading a parameter would mean touching every adapter for a seam one caller
  uses. The price of ambience is three invisible failure modes, all named in the docstring — a
  second call inside the block overwrites the record; a call for some *other* purpose inside it is
  captured and never emitted; and a `ContextVar` does not cross a thread, so an adapter that
  dispatched to a worker would emit its own line while the caller emitted a second. No shipped
  adapter does any of the three, which is the assumption to re-check before adding one.

- **The reranker keeps its severity taxonomy; the describer gains one.** Config-class is ERROR once
  per wake then DEBUG (dead until a human acts, and ERROR is what pages); runtime-class is WARNING
  (it can succeed unchanged next time); a call that worked is INFO. The describer's failures used to
  wear a private `describer failed` head, which was invisible to every column that counts describes
  — precisely how a *configured and dead* describer read identical to a deliberately blind agent.

- **A recall is not a model call.** `mempalace recall` → **`memory recall`**, with
  `provider=mempalace` added: the head names the category and the software becomes a field value,
  exactly as `model=` does on a model call. It keeps its own head rather than joining the `llm` line
  because it spends nothing on a provider and has no tokens to report.

- **The `llm` head itself is byte-frozen.** It is the fleet dashboard's anchor for the whole family
  *and* the denominator of its extraction alarm, so a purpose is added **inside** the line and never
  spelled as a new head. The NOC's own gate carries a leading space (` llm provider=`) that only the
  level prefix supplies, so `test_rerank.py` asserts it against the **rendered** record — a test
  reading `record.getMessage()` would pass on a line the dashboard cannot see.

**Boundary:** the harness never learns the NOC's regexes and the NOC never learns the emitter — the
contract is the field shapes above, published in the completion comment. Live verification is the
capital's, after the NOC re-points its columns.
