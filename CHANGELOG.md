# Changelog

All notable changes to BaseCradle Harness are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
