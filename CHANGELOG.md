# Changelog

All notable changes to BaseCradle Harness are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
