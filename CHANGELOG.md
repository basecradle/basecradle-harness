# Changelog

All notable changes to BaseCradle Harness are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
