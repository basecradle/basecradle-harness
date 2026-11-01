# The Telemetry Contract

Every line the harness writes to its journal, every field on each line, what each field means, who fills it, and when it is absent. **This page is the contract**: the dashboards exist to compare vendors, which is only possible if every vendor's facts land in the same lines with the same meanings, so the harness emits nothing this page does not define. `tests/test_telemetry_contract.py` reads this page and fails the build on any head or field outside it.

## The Ruling

@origin, 2026-10-31 (issue #757): **the fleet's telemetry is contract-first.**

- A new provider adapter is mapped *into* this contract. It never introduces a field, a head, or a category of its own.
- When a vendor reports something this contract cannot hold, or cannot report something it holds, **the work stops.** The misfit is filed `needs-capital` as a design question about the existing system — what the misfit is, and what it says about the contract — and goes to the founders through the capital. The contract changes first, for every adapter whose vendor has the datum, and only then does the vendor land.
- Where today's emission differs from this page, the difference is listed under [Known Violations](#known-violations), each with the issue that removes it. That table is the only place a departure may live.

This page describes the **target** contract. A field the target defines that no adapter emits yet is part of the contract all the same; a field adapters emit today that the target retires is not, and is listed as a violation until it is gone.

## The Grammar

A structured line is one journal record:

```text
INFO llm provider=openrouter purpose=main endpoint=StreamLake model=z-ai/glm-5.2 duration=42.96s tokens_in=764942 tokens_out=236 tokens_total=765178 cached_tokens=238277 cost=0.0445 generation_id=gen-1791030923-8abbrsSEsIk06gI224Fi
```

- **Severity**, then a space. It is the only place severity exists (`_observability.LOG_FORMAT`; the NOC's `level` column reads it).
- **The head**: one or more words naming *what happened* (`llm`, `wake end`, `memory recall`). A head is a fixed literal, never composed from data. A head that states a verdict about a unit of work is colored as one span (`_observability.head`); a head that names a fact is plain.
- **Fields**: `key=value` pairs, separated by single spaces. A value is a bare token, or a double-quoted string when it holds a space, an `=` or a quote (`kv` quotes it, flattens it to one line, scrubs credential shapes, and bounds it at 240 characters — except the unspoken `text`, which is never bounded).
- **Absence is honest.** A field with nothing to say is omitted — never written empty, never written as a placeholder, and never as a zero the vendor did not state. `0` is kept where it is a fact ("posted nothing").
- **Consumers key on field names, never on positions** — with two anchors that are positional: the head itself (`llm provider=` is the fleet's anchor for every model call), and `billing`, which is always last on an `llm` line.
- **Units are fixed.** `duration` and every other length of time is seconds with a trailing `s` (`3.41s`); a moment in time is ISO 8601 in UTC with a trailing `Z` (`2026-11-01T03:12:40Z`); `cost` is plain fixed-point US dollars (`0.0445`, never scientific notation, never negative); `chars` counts characters the model or a reader sees, never bytes; every `tokens_*` field is a count of tokens as the vendor billed them.

A line that is not of this shape is a **prose line** — see [Prose Lines](#prose-lines). Prose carries no fields.

### Field Meanings Are Fleet-Wide

A field name means **one thing on every head**. The NOC extracts several fields off *every* line regardless of head (`provider=`, `source=`, the severity token), so a name reused with a second meaning silently feeds the wrong column. Three fields are head-scoped by design, and their meaning is still one thing — only the vocabulary is the head's own:

| Field | One Meaning | Vocabulary |
|---|---|---|
| `kind` | The sub-class of the thing the head names. | Listed on each head. |
| `reason` | Why the head's verdict is what it is, as a **slug** — never prose. | Listed on each head. |
| `outcome` | The verdict on the unit of work the head names. | Listed on each head. |

### Common Fields

The fields below mean exactly this wherever they appear. A head's own table repeats the ones it carries, with who fills them on that head.

| Field | Meaning |
|---|---|
| `provider` | Who did the work. On a model call or a tool, the vendor that bills it (`AI_PROVIDER`'s value for the brain); on a memory line, the memory system's software (`mempalace`). Software never names a head (issue #485); it appears only here. |
| `model` | The vendor's model id the work ran on, exactly as configured. |
| `timeline` | The uuid of the BaseCradle timeline the work is for. |
| `item` | The uuid of the message or task a wake is working. |
| `delivery` | The router's delivery id that spawned the wake (`BASECRADLE_DELIVERY_ID`); absent when the wake was not spawned by a router. |
| `handle` | The platform handle of the item's author. |
| `duration` | Wall time of the unit the head names, in seconds. |
| `chars` | A character count of the text the head names. |
| `count` | How many of the head's subject this line accounts for — in `unit`, where the line carries one. |
| `unit` | The unit the line's quantities are counted in. |
| `executor` | Who ran the work the line names: `harness`, `mcp`, or `server` (the vendor, inside a model call). |
| `cost` | US dollars charged for the unit, as the vendor stated it or as the harness computed it from the vendor's published rates. |
| `cost_basis` | `computed` when `cost` is the harness's arithmetic over published rates; absent when the vendor stated the figure. Never present without `cost`. |
| `generation_id` | The vendor's own id for the **model call** (OpenRouter `gen-…`, OpenAI `chatcmpl-…`/`resp_…`, Anthropic `msg_…`, xAI and Google response ids). On a tool line, the id of the model call the tool ran inside. |
| `error` | Foreign error text — a tool's exception, an SDK's refusal — bounded and scrubbed. A breadcrumb to the failure, never the failure's record. |
| `source` | `probe` on a line manufactured by the fleet's own instrumentation (`_log_grammar`); absent on real traffic. The NOC's `source` label is extracted off every line and filters probes out of production charts, so this name carries no other meaning. |
| `agent` | The agent's slug, on a probe line only. |
| `threshold` | The configured level the head's decision compares against. |
| `limit` | The model's context ceiling in tokens. |
| `held` | How long something was held, in seconds. |

## Model Calls

### `llm`

**One line per model-call attempt the caller ends, whatever its purpose or outcome** (`_observability.log_llm_call`, the only emitter). For the brain that is every call that answered; a brain call that raised writes none (the wake's failure lines carry it). For a non-brain caller (`purpose=memory|helper`) it is every attempt it gives up on or accepts, with its `outcome` — including a `fallback` that never reached a vendor (a config-class fault), which then carries no `duration`, tokens or cost. An attempt that failed and **is retried** writes `llm retry` instead, never this head. The head `llm provider=` is byte-frozen: it is the NOC's anchor for every call, cost, and duration series.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `provider` | The vendor that bills the call. | Every adapter. | Never. |
| `purpose` | What the model was doing for the agent: `main` (the brain's own turn), `memory` (a memory-system call, e.g. the reranker), `helper` (a second model working for the brain, e.g. the describer, or a model call a tool makes). | `log_llm_call`; `main` by default. | Never. |
| `kind` | The job within `purpose`: `rerank` (memory); `image.describe`, `video.describe`, `search.grounding` (helper). | The non-brain caller. | `purpose=main`. |
| `endpoint` | The **serving upstream** a router dispatched the call to (`StreamLake`, `DeepInfra`), read off the router's own routing metadata. | The router adapters (`serving_endpoint`). | A direct-to-vendor call — the vendor *is* the endpoint. |
| `residency` | Where the vendor ran the call: the location or inference geography the vendor reports, else the one the request pinned (`us`, `global`, `europe-west4`). `global` means the vendor chose. | Adapters whose vendor reports or accepts a location (Google, Anthropic). | The vendor has no notion of location. |
| `model` | The model id. | Every adapter. | Never. |
| `duration` | Seconds the call took; for a call that needed retries, the caller's whole wait. | Every adapter, or the capturing caller. | The call raised before timing (a config-class fault that never reached a vendor). |
| `tokens_in` | Every input token the vendor billed for the call — uncached, cached, cache-written, and any prompt tokens a server-side tool added. | Every adapter, from vendor usage. | The vendor stated no usage, or stated only zeros. |
| `tokens_out` | Every output token the vendor billed — the reply **and** any reasoning or thinking. | Every adapter. | As `tokens_in`. |
| `tokens_total` | Exactly `tokens_in` + `tokens_out`. Never the vendor's own total. | Every adapter. | As `tokens_in`. |
| `cached_tokens` | The part of `tokens_in` read from the vendor's prompt cache. The only trustworthy witness that caching works. | Every adapter whose vendor reports it. | The vendor reports no cache read. |
| `cache_write_tokens` | The part of `tokens_in` written to the vendor's prompt cache on this call. | Every adapter whose vendor reports it. | The vendor reports no cache write. |
| `tokens_reasoning` | The part of `tokens_out` spent on reasoning or thinking. | Every adapter whose vendor reports it, on every purpose. | The vendor reports none — never an invented zero. |
| `cost` | US dollars for the call's tokens. Tool fees are never folded in when the vendor states them apart (they belong on the tool's line). | Vendor-stated (OpenRouter `usage.cost`, xAI ticks) or computed (`_<vendor>_rates.py`). | Unstated and unpriceable: an unknown model, tier, or host. |
| `cost_basis` | `computed`, beside a computed `cost`. | `cost_basis()`. | A vendor-stated cost; no cost. |
| `outcome` | `ok` — the answer was usable; `fallback` — the caller fell back (rerank to hybrid order, describer to the withheld caption). | The non-brain caller, which alone can judge the answer. | `purpose=main`: the brain's turn is judged by the wake, not by its call. |
| `reason` | Why a `fallback` happened: a fault slug (`rate_limited`, `server_error`, `transport`, `timeout`, `invalid_response`, `context_length`, `payload_too_large`, `api_error`, `provider_error`, `config:<cause>`) or an answer slug (`truncated`, `empty_response`, `missing_parts`, `undecodable_video`, `internal`). | The non-brain caller. | `outcome=ok`. |
| `detail` | The vendor's or adapter's own text explaining `reason`. | The non-brain caller. | Nothing to explain. |
| `retry_after` | The wait the vendor asked for on the failed attempt. | `_retry.diagnostics`. | The vendor hinted none. |
| `provider_code` | The upstream's own status code inside a router's error body. | `_retry.diagnostics` (OpenRouter). | Not a routed failure. |
| `routing_attempt` | The router's own attempt counter (`0` = reached no upstream). Never the harness's attempt. | `_retry.diagnostics` (OpenRouter). | As `provider_code`. |
| `attempts` | The router's per-upstream account of who refused (`parasail:429,coreweave:429`). | `_retry.diagnostics` (OpenRouter). | As `provider_code`. |
| `surface` | Where a memory recall ran: `turn0` (before the turn) or `tool` (the memory tool). | The reranker. | Not `purpose=memory`. |
| `pool` | How many candidates the reranker was shown. | The reranker. | As `surface`. |
| `picked` | How many of them it selected. | The reranker. | As `surface`. |
| `subject` | What a helper call was about: the file names of the media described. | The describer. | Not a describe. |
| `generation_id` | The vendor's id for the call. Rendered next to last. | Every adapter (`generation_id`). | The response named none. |
| `billing` | `byok` — the cost is on the operator's own provider invoice, not the router's credits. Always last. | `_observability.byok`. | Billed the ordinary way. |

### `llm retry`

**One WARNING per failed attempt that is about to be retried** (`_observability.log_llm_retry`). A different head from `llm` on purpose: a refused attempt generated nothing and was billed nothing, so it must never be counted as a call.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `provider` | Common. | `_retry.Retry`. | Never. |
| `purpose` | As on `llm`. | `_retry.Retry`. | Never. |
| `kind` | As on `llm`. | `_retry.Retry`. | `purpose=main`. |
| `model` | Common. | `_retry.Retry`. | Never. |
| `attempt` | The harness's attempt number of this call, as `n/total`. | `_retry.Retry`. | Never. |
| `reason` | The fault slug (`rate_limited`, `server_error`, `transport`, `timeout`, `invalid_response`). An HTTP 504 is `timeout` on every vendor: a deadline ran out, the vendor's or the harness's. | `_retry.Retry`. | Never. |
| `retry_after` | As on `llm`. | `_retry.diagnostics`. | As on `llm`. |
| `provider_code` | As on `llm`. | `_retry.diagnostics`. | As on `llm`. |
| `routing_attempt` | As on `llm`. | `_retry.diagnostics`. | As on `llm`. |
| `attempts` | As on `llm`. | `_retry.diagnostics`. | As on `llm`. |
| `timeout` | The budget the timed-out attempt ran out of, seconds. | `_retry.Retry`. | `reason` is not `timeout`, or the budget is unknown. |
| `timeout_scale` | The multiple of the fitted budget the retry gets. | `_retry.Retry`. | `reason` is not `timeout`. |
| `next_timeout` | The budget the retry gets, seconds. | `_retry.Retry`. | As `timeout`. |
| `next_in` | Seconds until the next attempt. | `_retry.Retry`. | Never. |
| `surface` | As on `llm`. | The reranker. | Not a rerank. |
| `subject` | As on `llm`. | The describer. | Not a describe. |
| `generation_id` | The vendor's id for the refused attempt. | The adapter's error mapper. | The refusal named none. |

### `llm refused`

**One WARNING per model call the vendor refused on content grounds** — a refusal is the vendor's final word, never retried. The fact lands here so a refusal rate can be compared across vendors.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `provider` | Common. | The adapter. | Never. |
| `purpose` | As on `llm`. | The adapter. | Never. |
| `model` | Common. | The adapter. | Never. |
| `category` | The vendor's own refusal category, as a slug. | The adapter. | The vendor gave none. |
| `generation_id` | Common. | The adapter. | The response named none. |

### `provider`

The one trace that a vendor runs under the generic OpenAI-compatible profile (`_basecradle`), written once per wake: a misspelled known vendor is a legal slug and runs generic, and this line is what says it lost its conventions.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `profile` | `generic`. | `_basecradle`. | Never. |
| `provider` | Common. | `_basecradle`. | Never. |
| `sdk_surface` | The SDK surface the adapter speaks (`chat`, `responses`). | `_basecradle`. | Never. |
| `host` | The endpoint's host. | `_basecradle`. | Never. |

### `turn truncated`

A brain reply the vendor stopped at the output budget (`_engine._note_truncation`).

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `provider` | Common. | The engine. | Never. |
| `model` | Common. | The engine. | Never. |
| `finish_reason` | The vendor's own word for why it stopped (`length`, `max_tokens`, `REASON_MAX_LEN`). | The adapter's `last_finish_reason`. | Never. |
| `step` | The step the reply was. | The engine. | Never. |
| `max_steps` | The turn's step budget. | The engine. | Never. |
| `outcome` | `unfinished` — the cut-off reply ended the turn, which stays pending; `continued` — the turn went on. | The engine. | Never. |
| `reason` | `output_budget`. | The engine. | Never. |

### `usage unreported`

One INFO note, once per wake per model and kind, when a helper call's vendor returned a usage block of nothing but zeros — usage in the shape of an answer that states nothing (`_describer._note_unreported_usage`). A call with no usage block at all writes no note.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `provider` | Common. | The describer. | Never. |
| `purpose` | As on `llm`. | The describer. | Never. |
| `kind` | As on `llm`. | The describer. | Never. |
| `endpoint` | As on `llm`. | The describer. | As on `llm`. |
| `model` | Common. | The describer. | Never. |

### `describe repeated`

The describer answering again from its once-per-wake memo instead of calling a model (`_describer._once`). Not a call: no `llm` line is written for it.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `purpose` | As on `llm`. | The describer. | Never. |
| `kind` | As on `llm`. | The describer. | Never. |
| `model` | Common. | The describer. | Never. |
| `subject` | As on `llm`. | The describer. | Never. |
| `outcome` | `ok` or `fallback`, as the first describe ended. | The describer. | Never. |

## Tools

### `tool`

**One line per tool use, whoever executed it** — a harness tool, an MCP server's tool, or a tool the vendor ran server-side inside a model call (`_observability.log_tool`, the only emitter). Tool calls and tool spend are both read off this one head; there is no second definition of "tool". The engine writes the line for a harness or MCP tool, with what the tool's vendor work recorded on its meter while it ran (`_observability.metered`); the engine's own refusal of a call (an unknown or withheld name) is `executor=harness`, since the harness is what answered it. An adapter writes the line for a server-side tool, one line per model call per tool, right after that call's `llm` line — or before it, when the caller holds its `llm` line back to judge the answer (`capture_llm_call`); `generation_id` is the join either way. A failed harness or MCP run writes a second, WARNING `tool` line carrying `name` and `error`; a failed server-side tool has a vendor code and no text, so it has no companion and its line is itself a WARNING.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `name` | The tool's name as the agent knows it (`messages`, `generate_image`, `web_search`, `web_fetch`, `code_execution`, an MCP tool's name). A server-side tool carries the harness name of the capability, not the vendor's internal block type. | The engine; an adapter for a server-side tool. | Never. |
| `executor` | Who ran it: `harness`, `mcp`, or `server` (the vendor, inside a model call). | The engine; the adapter. | Never. |
| `provider` | The vendor that ran or billed the tool's work — the brain's vendor for a server-side tool; the media or search vendor a harness tool called. | Whoever knows the vendor. | The tool reaches no billed vendor (platform tools, memory). |
| `kind` | The shape of the work, for metered media and search: `image.generate`, `image.edit`, `video.generate`, `audio.transcribe`, `search.web`, `search.grounding`. One dotted vocabulary with `charge`'s `kind`. | The tool or adapter. | Unmetered work. |
| `model` | The vendor model the tool's work ran on. | The tool or adapter. | The tool used no model. |
| `count` | Billed or executed units on this line: searches, grounding queries, fetches, executions. A server-side tool writes one line per model call per tool, with its uses counted here — every use, a failed one included, so it can exceed the units a fee was charged on. | The tool or adapter. | A harness run whose work stated no units (one image, one transcription). Always present on `executor=server`. |
| `duration` | Wall time of the run — for a media tool, the generation and the upload that follows it. | The engine. | `executor=server` — the work ran inside a model call its `llm` line already timed. |
| `outcome` | `ok` or `error`. | The engine; the adapter from the vendor's result block. | The WARNING companion line. |
| `reason` | Why it failed, as a slug. The engine's own refusals: `unknown_tool`, `withheld`, `server_side` (a server-side built-in called as a function), `elision_marker`; a tool that raised: `exception`; a harness tool's vendor call that failed: the fault slug, as on `llm` (`rate_limited`, `server_error`, `transport`, `timeout`, `invalid_response`, `provider_error`). For a server-side tool, the vendor's own code or status, lowercased (`max_uses_exceeded`, `url_retrieval_status_error`, `outcome_failed`, `failed`, `incomplete`). | Whoever saw the failure. | `outcome=ok`. |
| `error` | Common; on the WARNING companion line only. | The engine. | `outcome=ok`. |
| `cost` | US dollars of the tool's own fee — never model tokens, which are on the `llm` line. | The tool or adapter. | No fee, or a fee nobody can price (see `cost unpriced`). |
| `cost_basis` | Common. | `cost_basis()`. | As Common. |
| `generation_id` | Common: the model call the tool ran inside, so a tool joins to its `llm` line. | The adapter. | `executor` is not `server`, or the call's response named no id. |

### `cost unpriced`

One WARNING, once per wake per gap, naming a charge that is real and that the harness cannot price — so a missing dollar is a counted fact rather than an absence nobody sees.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `provider` | Common. | The adapter or tool. | Never. |
| `model` | Common. | The adapter or tool. | The charge is not per-model. |
| `charge` | What carries no price: `tokens`, `container_time`, `grounding`, `search`. | The adapter or tool. | Never. |
| `reason` | `unknown_model`, `unpriced_tier`, `unpriced_host`, `not_reported`. | The adapter or tool. | Never. |

A `container_time` gap is the charge the next head would carry: the day a vendor states its container time, that charge is written as a priced `charge kind=compute.container` line and leaves this one.

### `charge`

**One line per billed resource instance**: a dollar a vendor bills by a quantity of a resource — browser session minutes, container hours, a 15-minute GPU block, bytes, storage — rather than per call (@origin, 2026-11-01, on basecradle-noc#1153). It is neither a tool *use* nor a model call, so it has a head of its own, and every resource-billed product fits it without a new one.

```text
charge provider=<biller> kind=<what was bought> executor=harness|mcp|server unit=<billing unit> count=<quantity in that unit> [duration=<s>] [started=<iso>] [resource=<vendor resource id>] [generation_id=<model call it belongs to>] outcome=ok|error [reason=<slug>] cost=<usd> [cost_basis=computed] [vendor_usage=<vendor's own raw figure> vendor_unit=<its unit>]
```

- **Which head.** If the dollar can be attributed to one request from that request's own response, it is not a charge: tokens are on the `llm` line, a per-search fee on the `tool` line. If it is billed by a quantity of a resource across zero or more requests, it is a `charge`.
- **When it is written.** Once the vendor's figures are final. A late line is fine (the Steel launcher waits for its usage to land); two lines for one instance never are. An estimate written ahead of the final figure would be a contract change (`cost_basis=estimated` with a `supersedes=`), not built.
- **A charge nobody can price writes no `charge` line**: its gap is a `cost unpriced` line, as container time is today.
- **The dashboards are the registry's, not this page's.** Every charge today is tool-adjacent, so *Tool Cost* reads the dollars on `tool` and `charge` lines while *Tool Calls* reads `tool` lines only; a future `kind` family that is not tool-adjacent (`seat.*`) is a registry bucket change, never a contract change. *Spend by Provider* already sums every `cost=` on any head.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `provider` | The vendor that bills the resource. | The producer. | Never. |
| `kind` | What was bought, in the dotted vocabulary the `tool` line's `kind` uses, so one query reads a family's spend whether a vendor bills it per use or per resource: `browser.session`, `compute.container`, `compute.gpu`, `storage.cache`, and any `tool` kind (`image.generate`, …). A new value is a change to this page. | The producer. | Never. |
| `executor` | Common: who ran the thing that incurred the charge. | The producer. | Never. |
| `unit` | Common: the billing unit `count` is in — `minute`, `hour`, `block_15m`, `gb`, `gb_month`, `seat_month`. An open set, listed here as it grows. | The producer. | Never. |
| `count` | Common: the billed quantity, in `unit`, rounded the way the vendor rounds it. | The producer. | Never. |
| `duration` | Common: how long the resource instance ran. | The producer. | The resource has no run time (bytes, storage). |
| `started` | The moment the resource instance started. | The producer. | The vendor states no start. |
| `resource` | The vendor's own id for the billed resource instance (a Steel session id, a container id). | The producer. | The vendor names no instance. |
| `generation_id` | Common: the one model call the charge belongs to (container time inside a call). | The adapter. | The charge belongs to no single model call. |
| `outcome` | `ok` or `error`: whether the resource instance's run ended cleanly. A charge is written either way, because the vendor bills either way. | The producer. | Never. |
| `reason` | Why the run ended in error, as a slug. The vocabulary is listed here as producers adopt it; none emits one yet. | The producer. | `outcome=ok`. |
| `cost` | Common: US dollars billed for the instance. | The producer. | Never (see above: an unpriceable charge writes `cost unpriced`, not `charge`). |
| `cost_basis` | Common. | `cost_basis()`, or the producer by the same rule. | As Common. |
| `vendor_usage` | The vendor's own raw figure for the instance, verbatim, when it reports one in a unit this page cannot yet price (Steel's `creditsUsed`) — kept for calibrating a computed `cost`. Generic, never a vendor-named field. | The producer. | The vendor reports no such figure. |
| `vendor_unit` | The unit `vendor_usage` is in (`credits`). | The producer. | `vendor_usage` is absent; never present without it. |

The first producer is the NOC's Steel browser launcher (see [Outside This Repository](#outside-this-repository)); nothing in this repository writes a `charge` line yet. Its per-session spend line:

```text
INFO charge provider=steel kind=browser.session executor=mcp unit=minute count=7 duration=402.18s started=2026-11-01T03:12:40Z resource=01a19518-4c40-7c41-9f3e-5d8c1a2b4e60 outcome=ok cost=0.07 cost_basis=computed vendor_usage=14 vendor_unit=credits
```

And the container time Anthropic and OpenAI do not report today (`cost unpriced charge=container_time reason=not_reported`), the day either vendor states it:

```text
INFO charge provider=anthropic kind=compute.container executor=server unit=hour count=0.15 duration=540.00s resource=container_01Kq3vYk8NfT2mWc7RbJpLsX generation_id=msg_01Hn4wQz9PcR5tVb2LmXyKdE outcome=ok cost=0.0075 cost_basis=computed
```

### `direct_message sent`

The off-platform push to @origin's phone, delivered (`_direct_message`).

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `bytes` | The payload's size in bytes. | `_direct_message`. | Never. |
| `title` | The notification's title. | `_direct_message`. | Never. |

### `direct_message failed`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `bytes` | As on `direct_message sent`. | `_direct_message`. | Never. |
| `reason` | `transport` (ntfy never answered), `server_error` (it failed on its own side), or `refused` (its verdict on the request, a 4xx). | `_direct_message`. | Never. |
| `error` | Common. | `_direct_message`. | Never. |

### `tool_call_ids rewritten`

The engine made a response's tool-call ids unique (`_engine._unique_call_ids`).

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `renamed` | The ids rewritten, comma-separated. | The engine. | Never. |

### `re-issued`

An interrupted platform create re-run under its deterministic idempotency key (`_wake._reissue_interrupted_creates`).

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `tool` | The tool whose create was re-issued. | The wake. | Never. |
| `kind` | The create's kind (`message`, `task`, …). | `_idempotency`. | Never. |
| `ordinal` | The create's ordinal within its turn. | `_idempotency`. | Never. |
| `timeline` | Common. | The wake. | Never. |

## Wake Lifecycle

### `wake start`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `timeline` | Common. | The wake. | Never. |
| `trigger` | What fired the wake, as `<message\|event\|asset>:<uuid>`. | The router's arguments. | The router named nothing. |
| `provider` | The brain's vendor. | `describe_provider`. | Never (`unknown` for a third-party adapter that names none). |
| `model` | The brain's model. | `describe_provider`. | As `provider`. |
| `delivery` | Common. | The environment. | As Common. |

### `wake continuing`

Another pass of the same wake, because work arrived while it ran.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `item` | Common. | The wake. | Never. |
| `kind` | `message` or `task`. | The wake. | Never. |
| `timeline` | Common. | The wake. | Never. |
| `pass` | The pass number now starting. | The wake. | Never. |

### `wake end`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `timeline` | Common. | The wake. | Never. |
| `outcome` | `ok` (every reconcile returned) or `error`. | The wake. | Never. |
| `turns` | Turns the engine ran. | The engine. | Never. |
| `steps` | Model steps used, as `used/budget`. | The engine. | Never. |
| `posted` | Messages the agent posted. `0` is legitimate: silence is an answer. | The wake. | Never. |
| `duration` | Common. | The wake. | Never. |
| `delivery` | Common. | The environment. | As Common. |

### `wake deferred`

An item left unsettled as the wake ends, waiting on a later wake.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `item` | Common. | The wake. | Never. |
| `kind` | `message` or `task`. | The wake. | Never. |
| `timeline` | Common. | The wake. | Never. |
| `reason` | Why it waits, as a slug. | The wake. | Never. |

### `wake truncated_turn`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `item` | Common. | The wake. | Never. |
| `kind` | `message` or `task`. | The wake. | Never. |
| `timeline` | Common. | The wake. | Never. |
| `delivery` | Common. | The environment. | As Common. |
| `reason` | `output_budget`. | The wake. | Never. |

### `wake stalled`

An item abandoned after its resumes failed `RESUME_CEILING` times, or a batch-mate abandoned with it.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `item` | Common. | The wake. | Never. |
| `kind` | `message` or `task`. | The wake. | Never. |
| `timeline` | Common. | The wake. | Never. |
| `handle` | Common. | The wake. | A batch-mate's line. |
| `resumes` | Resumes attempted. | The wake. | A batch-mate's line. |
| `tool_calls` | Tool calls the stalled turn made. | The wake. | A batch-mate's line. |
| `note` | The uuid of the stall note posted. | The wake. | A batch-mate's line. |
| `reason` | `resume_ceiling` (the turn's resumes failed `RESUME_CEILING` times) or `batch_mate` (abandoned with the turn that carried it). | The wake. | Never. |
| `detail` | The harness's sentence on how the last resume failed. | The wake. | A batch-mate's line. |

### `wake reported_failure`

A permanent or billing model failure reported to the timeline (`_wake._report_provider_failure`, `_report.billing_onset_line`). NOC-alarmed.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `kind` | `permanent` or `billing`. | The wake. | Never. |
| `reason` | The failure's slug (`payload_too_large`, `out_of_funds`). | The wake. | Never. |
| `provider` | Common. | The wake. | A probe line. |
| `item` | Common. | The wake. | A billing onset. |
| `timeline` | Common. | The wake. | A probe line. |
| `delivery` | Common. | The environment. | As Common. |
| `source` | Common. | `_log_grammar`. | Real traffic. |
| `agent` | Common. | `_log_grammar`. | Real traffic. |

### `wake billing_blocked`

The debounced repeat for every later wake an out-of-funds account blocks (`_report.billing_repeat_line`). NOC-alarmed.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `reason` | As on `wake reported_failure`. | The wake. | Never. |
| `provider` | Common. | The wake. | A probe line. |
| `timeline` | Common. | The wake. | A probe line. |
| `delivery` | Common. | The environment. | As Common. |
| `source` | Common. | `_log_grammar`. | Real traffic. |
| `agent` | Common. | `_log_grammar`. | Real traffic. |

### `wake billing_recovered`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `timeline` | Common. | The wake. | Never. |
| `delivery` | Common. | The environment. | As Common. |

### `wake skipped`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `timeline` | Common. | The entrypoint. | Never. |
| `delivery` | Common. | The environment. | As Common. |
| `reason` | `timeline_deleted`. | The entrypoint. | Never. |

### `wake failed`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `timeline` | Common. | The entrypoint. | Never. |
| `error` | Common. | The entrypoint. | Never. |
| `delivery` | Common. | The environment. | As Common. |

### `degraded`

A turn that ended on the canned stuck note because even its reserve call failed.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `timeline` | Common. | The wake. | Never. |
| `reason` | `reserve_failed` (the reserve summary call raised) or `reserve_empty` (it wrote no text). | The wake. | Never. |
| `detail` | The engine's sentence on what failed. | The wake. | Never. |

### `claim refused`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `item` | Common. | The wake. | Never. |
| `kind` | `message` or `task`. | The wake. | Never. |
| `timeline` | Common. | The wake. | Never. |
| `reason` | `covered`. | The wake. | Never. |

### `resuming`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `item` | Common. | The wake. | Never. |
| `kind` | `message` or `task`. | The wake. | Never. |
| `timeline` | Common. | The wake. | Never. |
| `handle` | Common. | The wake. | Never. |
| `resume` | This resume's number, as `n/ceiling`. | The wake. | Never. |
| `reason` | `died_mid_tool_chain`. | The wake. | Never. |

### `dropped`

An item abandoned at ERROR. A drop is never silent.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `item` | Common. | The wake. | Never. |
| `kind` | `message` or `task`. | The wake. | Never. |
| `timeline` | Common. | The wake. | Never. |
| `handle` | Common. | The wake. | Never. |
| `reason` | `compacted_away` (the turn that carried it was compacted out of the transcript), `context_length` (the resumed turn is over the ceiling and cannot be compacted), or `payload_too_large` (the resumed turn hit that permanent failure). | The wake. | Never. |
| `detail` | The harness's sentence, with any vendor text in it. | The wake. | Never. |

### `re-driving`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `item` | Common. | The wake. | Never. |
| `kind` | `message` or `task`. | The wake. | Never. |
| `timeline` | Common. | The wake. | Never. |
| `handle` | Common. | The wake. | Never. |
| `reason` | `died_before_model` (the wake died before the model saw the item) or `died_in_model_call` (it died inside the model call). | The wake. | Never. |

### `recovered`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `item` | Common. | The wake. | Never. |
| `kind` | `message` or `task`. | The wake. | Never. |
| `timeline` | Common. | The wake. | Never. |
| `handle` | Common. | The wake. | Never. |
| `action` | What recovery did: `turn-completed`. | The wake. | Never. |

### `Wake breaker TRIPPED`

The cross-wake circuit breaker's trip (`_breaker.breaker_tripped_line`). NOC-alarmed; its fields mirror the router's own trip line.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `timeline` | Common. | The breaker. | A probe line, which carries only `source` and `agent`. |
| `count` | Wakes counted in the window. | The breaker. | A probe line, which carries only `source` and `agent`. |
| `threshold` | Common: the wake count that trips. | The breaker. | A probe line, which carries only `source` and `agent`. |
| `window` | The counting window, seconds. | The breaker. | A probe line, which carries only `source` and `agent`. |
| `cooldown` | The configured cooldown, seconds. | The breaker. | A probe line, which carries only `source` and `agent`. |
| `hold` | How long this wake will hold, seconds. | The breaker. | A probe line, which carries only `source` and `agent`. |
| `source` | Common. | `_log_grammar`. | Real traffic. |
| `agent` | Common. | `_log_grammar`. | Real traffic. |

### `Wake breaker RESET`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `timeline` | Common. | The breaker. | Never. |
| `held` | Common. | The breaker. | Never. |

## Speech

### `unspoken`

**The agent's narration that reached no timeline** (`_observability.log_unspoken`). The one field that is never truncated, because this line is the only place the text exists.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `timeline` | Common. | The wake. | Never. |
| `kind` | `narration`, `reserve` (the step-cap report the model wrote for its next turn), or `stuck` (the canned note). | The wake. | Never. |
| `chars` | Common. | The wake. | Never. |
| `text` | The narration in full: flattened, scrubbed, quoted, never cut. | The wake. | Never. |

### `posted`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `message` | The uuid of the message posted. | The poster. | Never. |
| `timeline` | Common. | The poster. | Never. |
| `kind` | Who posted: `tool` (the agent, through the `messages` tool) or one of the harness's own sanctioned posts: `probe-ack`, `failure-report`, `stall-note`. | The poster. | Never. |
| `chars` | Common. | The poster. | Never. |

### `post failed`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `timeline` | Common. | The poster. | Never. |
| `kind` | As on `posted`. | The poster. | Never. |
| `error` | Common. | The poster. | Never. |

## Perception

### `media described for a model with no vision`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `subject` | As on `llm`. | The engine. | Never. |
| `describer` | The describer's model id. | The engine. | Never. |
| `model` | The brain's model. | The engine. | Never. |

### `view image withheld from a model with no vision`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `images` | The withheld images' names. | The engine. | Never. |
| `model` | The brain's model. | The engine. | Never. |
| `reason` | `no_image_input`. | The engine. | Never. |

### `watch video withheld from the model`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `video` | The withheld clip's name. | The engine. | Never. |
| `model` | The brain's model. | The engine. | Never. |
| `reason` | `no_image_input` (the model takes no image input) or `undecodable_video` (its frames would not decode). | The engine. | Never. |
| `detail` | The decoder's account of the failure. | The engine. | `reason=no_image_input`. |

### `image degraded to text`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `timeline` | Common. | The wake. | Never. |
| `asset` | The uuid of the posted asset. | The wake. | Never. |
| `filename` | The asset's file name. | The wake. | Never. |
| `model` | The brain's model. | The wake. | Never. |
| `reason` | `no_vision_input`. | The wake. | Never. |

## Context

### `context attribution`

What the request the model is about to read is made of, in characters, summing exactly to `total` (`_attribution.attribute`). Measurements only, no verdict.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `unit` | `chars`. | `_attribution`. | Never. |
| `session` | The session's channel key the line measured. | `_attribution`. | The caller named none. |
| `total` | Every section below, summed. | `_attribution`. | Never. |
| `messages` | Messages in the transcript. | `_attribution`. | Never. |
| `tools` | Characters of the tool schemas. | `_attribution`. | Never. |
| `tools_count` | Tools offered. | `_attribution`. | Never. |
| `brief` | Characters of the operating brief. | `_attribution`. | Never. |
| `brief_*` | Characters of one named part of the brief. | `_brief.brief_section_sizes`. | A caller that composed no brief. |
| `history` | Characters of the transcript. | `_attribution`. | Never. |
| `history_*` | Characters of one transcript section (`charter`, `summary`, `steps`, `user`, `assistant`, `tool`, `server_tools`, `other`). | `_attribution`. | Never. |
| `images` | Characters the images cost. | `_attribution`. | Never. |

### `context limit`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `limit` | Common. | `ContextBudget`. | Never. |
| `limit_source` | Where the ceiling came from: `env`, `adapter`, `default`. | `ContextBudget`. | Never. |
| `compact_at` | The token count compaction triggers at. | `ContextBudget`. | Never. |

### `context compact`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `tokens_in` | As on `llm`: the provider's own count that triggered compaction. | `_context`. | `emergency=yes` — the provider rejected the context, so there is no count. |
| `limit` | Common. | `_context`. | Never. |
| `limit_source` | As on `context limit`. | `_context`. | Never. |
| `threshold` | Common: the token count compaction triggers at. | `_context`. | Never. |
| `messages` | Messages, as `before→after`. | `_context`. | Never. |
| `chars` | Characters, as `before→after`. | `_context`. | Never. |
| `summarized` | Messages folded into the summary. | `_context`. | Never. |
| `emergency` | `yes` when compaction ran because the provider rejected the context. | `_context`. | An ordinary compaction. |

## Memory

### `memory`

One line per memory-system operation other than a retrieval's report, which is `memory recall` (`_wake`, `_mempalace`). The wake writes its own `op=recall` and `op=observe` guard lines here when the provider's hook fails.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `op` | The operation: `recall`, `observe`, `search`, `palace-binding`. | The caller. | Never. |
| `result` | How it ended, where it did not simply succeed: `failed`, `skipped`, `unfile_failed`. | The caller. | The operation succeeded. |
| `provider` | Common: the memory software. | The caller. | The wake's own guard line, which does not know it; `op=palace-binding`. |
| `surface` | As on `llm`. | `_mempalace`. | Not a search. |
| `reason` | Why, as a slug. | `_mempalace`. | Nothing to explain. |
| `keys` | The field names of an error envelope upstream returned. | `_mempalace`. | Not that failure. |
| `type` | The type of an answer that was not a mapping. | `_mempalace`. | Not that failure. |
| `palace` | The palace path bound. | `_mempalace`. | Not a binding. |
| `config` | The MemPalace config file path. | `_mempalace`. | Not a binding. |
| `chars` | Common. | The wake. | Not the wake's guard line. |

### `memory recall`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `provider` | Common. | `_mempalace`. | Never. |
| `surface` | As on `llm`. | `_mempalace`. | Never. |
| `rerank` | `on` or `off`. | `_mempalace`. | Never. |
| `pool` | Candidates asked for. | `_mempalace`. | Never. |
| `fetched` | Candidates the search returned, when not `pool`. | `_mempalace`. | Equal to `pool`. |
| `injected` | Memories put in front of the model. | `_mempalace`. | Never. |
| `duration` | Common. | `_mempalace`. | Never. |
| `chars` | Common. | `_mempalace`. | Never. |
| `max_distance` | The distance threshold the search passed. | `_mempalace`. | The search passed none. |
| `sentinels` | Sentinel drawers filtered out. | `_mempalace`. | None. |
| `excluded` | Drawers this wake filed from its seeded backlog, kept out of the recall. | `_mempalace`. | None. |

### `memory threshold`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `provider` | Common. | `_mempalace`. | Never. |
| `max_distance` | `off`, or the threshold in use. | `_mempalace`. | Never. |
| `reason` | Why, as a slug. | `_mempalace`. | Never. |

## Maintenance

These are written by the console scripts, not by a wake.

### `cleanup blocked`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `path` | The path the sweep could not remove. | `_cleanup`. | Never. |
| `error` | Common. | `_cleanup`. | Never. |

### `cleanup failed`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `error` | Common. | `_cleanup`. | Never. |

### `palace scrub`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `palace` | The palace path. | `_scrub`. | Never. |
| `mode` | `apply` or `dry-run`. | `_scrub`. | Never. |
| `drawers` | Drawers in the palace. | `_scrub`. | Never. |
| `scrub` | Drawers the report would delete. | `_scrub`. | Never. |
| `held_files` | Source files held back whole because not every drawer was scaffolding. | `_scrub`. | Never. |
| `review` | Drawers flagged for a human to read. | `_scrub`. | Never. |
| `deleted` | Drawers deleted. | `_scrub`. | `mode=dry-run`. |
| `files_removed` | Conversation files removed. | `_scrub`. | `mode=dry-run`. |
| `after` | Drawers left. | `_scrub`. | Never. |

### `palace scrub failed`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `error` | Common. | `_scrub`. | Never. |

### `scrub`

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `op` | `unlink`. | `_scrub`. | Never. |
| `path` | The path. | `_scrub`. | Never. |
| `error` | Common. | `_scrub`. | Never. |

## Prose Lines

Everything that is not a structured line is **prose**: an operator's note in a sentence (`No published rate for …`, `Anthropic's prompt cache needs …`), the engine's `step n/max: … (1.23s)` ledger, a startup notice. Prose is legitimate and is not part of this contract's vocabulary, under one rule: **a fact a dashboard needs never lives only in prose.** A refusal, a server-tool failure, an unpriced charge — anything a vendor comparison would count — has a structured head here. Prose may restate it for a human; it may never be its only record.

## Probe Lines

A line manufactured by the fleet's own instrumentation (`basecradle-harness-log-grammar`) renders the production grammar through the production function, prefixed by a bare `PROBE` token and stamped `source=probe agent=<slug>`, under its own journald identifier. It carries no field that is a witness parent for another NOC column (no `provider=`). See `_log_grammar`.

## Known Violations

Where today's emission departs from the target above. The conformance test reads this table: a `field` or `head` witness allows exactly that vocabulary outside the contract, and every row must still be true — a row whose violation has disappeared **fails the build until it is struck from this table**, so the table is always exactly today's departures. Witness forms, each true exactly while its departure stands:

- `head <head>: <fields>` — a head outside the contract, allowed with exactly those fields.
- `field <head>.<field>` — a field outside the contract on that head; with `@<adapter>` it is allowed only from that adapter (`_<adapter>.py`, and that adapter's rendered lines).
- `prose <file> "<text>"` — a fact still recorded only as prose: the text at that site, quoted exactly as the source spells it; with `until <head>` (or `until <head>.<field>`) the row is also over once its structured replacement is emitted, whether or not a sentence restating it stays. With `until <head>.<field>=<value>` the replacement is a line carrying that value, which only a rendered line can show — so one head may replace several gaps (`cost unpriced`, by its `charge` field), each row over only when its own charge is emitted.
- `reason-prose <head>` — a `reason=` on that head that the scan reads as a sentence, an f-string or an exception's text, following a variable to its assignments and a parameter to the calls that fill it.
- `unescaped <head>` — a structured line spelled as a `%`-format string rather than built through `kv`, so its values are never flattened, scrubbed, bounded or quoted.
- `test` — a behavioral check the test runs by this row's ID.

A prose line that carries a field the NOC extracts off every line (`provider=`, `source=`) feeds a column whatever its head, so the test also fails on any such prose that no `prose` row names.

The slices are the issues that remove each row: **slice 2 — tools** ([#758](https://github.com/basecradle/basecradle-harness/issues/758), whose rows T1–T9 are struck), **slice 3 — fields** ([#759](https://github.com/basecradle/basecradle-harness/issues/759), whose rows F1–F36 are struck). No departure stands today.

| ID | Violation | Witness | Removed By |
|---|---|---|---|

### Outside This Repository

The NOC's Steel browser launcher writes its own `browser provider=` head. Its per-session spend line retires into the [`charge`](#charge) line (@origin, 2026-11-01: a session is billed by the minute, not per tool use, so it is not a `tool` line); its lifecycle lines stay on a NOC-only head that no column reads. It is not emitted here, so it is not witnessed here; the launcher's move is basecradle-noc#1153's. Slice 2 carried the rest to the NOC through the capital, together with the NOC columns that read what this repository retired in slice 2 (*Tool Cost* off `media`/`browser`; `web_search_requests`, `web_fetch_requests`, `code_execution_requests` off `llm`), which read the `tool` line instead.

## Changing the Contract

The contract changes before the code does, never after. A change is one pull request that edits this page, the emitters, and the Known Violations table together, and it moves every adapter whose vendor has the datum — never one vendor at a time. A new field or head is a design decision about the whole fleet's telemetry: it is raised `needs-capital` first, and lands only once decided. Anything the NOC extracts is coordinated with the NOC through the capital in the same arc, so that no column goes dark.
