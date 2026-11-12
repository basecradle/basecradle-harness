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
- **Fields**: `key=value` pairs, separated by single spaces. A value is a bare token, or a double-quoted string when it holds a space, an `=` or a quote (`kv` quotes it, flattens it to one line, scrubs credential shapes, and bounds it at 240 characters, never cutting inside a token — except the unspoken `text`, which is never bounded; see [Credential Scrubbing](#credential-scrubbing)).
- **The value rule**, in order, and it is one rule for the fleet: the harness's `_observability._value` (and `_full_value`, the unspoken `text`), the router's `logfmt.log_fields` and its `ai_otel_llm` `detail=` remap render a value to the same bytes. (1) **Flatten**: every control character — C0, DEL and C1, Unicode's `Cc` exactly, ESC included — becomes a space, then every run of whitespace becomes one space, trimmed at both ends. A control character in a value is never legitimate: a newline splits a record, and an ESC quoted from a vendor's error body would paint a reader's terminal or Better Stack's rendered view with a color no emitter chose. (2) **Scrub** credential shapes, then **bound** at 240 characters (the harness only). (3) **Quote** a value holding a space, an `=` or a `"`, and turn its own `"` into `'`, so a consumer counting `"` to stay outside quoted values is never fooled; a backslash passes through untouched. The palette (`head`, a verdict pair) is applied after a value is rendered, so it is the only ESC a line carries. The shared witness is basecradle-router `tests/vector/ai_otel_llm.yaml`'s case "a detail renders by the fleet's one value rule", which `tests/test_observability.py` renders byte for byte (issue #806).
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
| `step` | The engine step the unit happened in, as `n/max` (`5/24`): where in its turn a model call was made, or a reply was cut off. |
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

### Credential Scrubbing

A value can carry text that is not the harness's (an exception, a vendor's error body), so a credential can reach a line. Two layers redact it, in order, and each owns a stated part of the job (issue #787):

| Layer | Where | Owns |
|---|---|---|
| **Source** | `_observability.scrub`, applied by `kv` to every value (and to the unspoken `text`) before the value is cut, and by `_observability.ScrubbingFormatter` to every record the handler of a harness entrypoint renders, prose lines and tracebacks included (issue #857). | **Every credential shape the harness knows of**, which is a superset of the box's: the fleet's own (`sk-`, `sk-or-`, `xai-`, `bc_uat_`, `bc_isk_`, refuseless `abl_`, Perplexity `pplx-`, an ntfy access token `tk_`), every shape `ai_scrub` carries (GitHub `gh?_` and fine-grained `github_pat_`, Google `AIza`, Hugging Face `hf_`, Replicate `r8_`, Steel `ste-`, a Better Stack heartbeat URL, an `api_key=` assignment), the bearer and key-in-query forms, and an `x-api-key` header value. It is the only layer off-fleet: a developer's install has no Vector. |
| **Box** | Vector's `ai_scrub` on each fleet server (basecradle-router `deploy/hosts/*/vector.yaml`, basecradle-noc `deploy/vector.yaml`), after the line is written. | Lines the harness never wrote (another program, a traceback Python prints for an exception nothing caught, a router line), the journald metadata no message rule reaches (`_CMDLINE`, dropped `sudo` and Vector events), and a shape the source has not yet adopted. |

The two lists cannot be one list, because one is VRL in another repository, so the rule binding them is: **a shape added to `ai_scrub` is added to the source's rules at no longer a minimum length**, and `tests/test_log_value_cut.py` holds one fabricated sample per box rule at the box's own minimum. The one stated difference is the bearer rule: the box redacts `Bearer` followed by anything, and the source needs eight characters, so a diagnosis like `Bearer token missing` keeps its words. No real bearer credential is shorter.

**A vendor key is covered only when its shape can be stated, never guessed** (issue #856). Each prefix rule is the prefix plus a minimum run under the vendor's own length, so a whole key always matches:

| Vendor | Rule | The Shape, and Whose Word It Is |
|---|---|---|
| refuseless | `abl_[A-Za-z0-9]{20,}` | `abl_` plus 48 alphanumerics, 52 characters in all, read off two real keys (@origin, 2026-11-10). |
| Perplexity | `pplx-[A-Za-z0-9]{20,}` | The `pplx-` prefix is Perplexity's documentation's; the 48 alphanumerics after it are gitleaks's `perplexity-api-key` rule. |
| ntfy | `tk_[A-Za-z0-9_-]{20,}` | `tk_` plus 29 characters, 32 in all, from ntfy's documentation; the harness holds one as `NTFY_DM_TOKEN`. ntfy generates the body from lowercase letters and digits, and its validator also admits capitals, `_` and `-` (`^tk_[-_A-Za-z0-9]{29}$`, ntfy's `user/util.go`), which is the run this rule takes. A generated token with no digit, about 1 in 12,500, is the word residual stated below when glued to a letter. |

A key with no prefix of its own has no shape a prefix rule can match, so it is covered where it travels instead. **An `x-api-key` header value is redacted by the header's name** (issue #861): Wiro's documentation gives its API key no prefix, length or character set, and sends it as that header. The rule is the first of the literal rules: `x-api-key` in any case, with `_` for either hyphen, followed by `:` or `=>` (a header dump, a Ruby hash), or by a quote, escaped or not, and then `:`, `=`, `=>` or `,` (`"x-api-key": "…"`, JSON inside a JSON string, a `('x-api-key', '…')` header tuple); the value may open with a quote, an escaped quote, a bytes literal's `b'` or bearer scheme words, and runs to the next space, quote, backslash or `&`, because a key of no stated shape may hold anything else. A bearer header glued to the value (`x-api-key: KBearer eyJ…`) is taken with it, token and all, since the value would otherwise end at the scheme word and leave its token to no rule. The unquoted `X-API-Key=…` is the `api_key=` rule's, which always took it with the wider value it allows. The header's name goes with its value, as every literal rule's label does. A bare `x-api-key` with no value is a word, kept; a colon after it is read as a value, so `x-api-key: missing` loses `missing`, and a quoted list of header names (`"x-api-key", "x-nonce"`) loses the second (the scrub errs toward redaction). Wiro's signing secret is never sent (its `x-signature` header is an HMAC keyed on the API key over the secret and a nonce), so nothing else of Wiro's travels. The harness's own route to Wiro, the generic OpenAI-compatible profile, sends the key as `Authorization: Bearer`, which the bearer rule already covers.

**A vendor prefix inside a word is a word, not a credential** (issue #794). A prefixed match (`sk-`, `sk-or-`, `xai-`, `bc_…_`, `gh?_`, `github_pat_`, `AIza`, `hf_`, `r8_`, `ste-`, `abl_`, `pplx-`, `tk_`) is kept only when **both** hold: a letter, digit or combining mark, in any script, precedes it, **and** its run after the prefix holds no digit and no uppercase letter (`_observability._is_word`). So `task-specification`, `disk-quota-exceeded` and a localized vendor sentence keep their words, and every other match is redacted. The test is deliberately not a left boundary: a key glued to a letter or digit is not a shape any vendor issues, but it is one a serialization produces (`Bearer%20sk-…`, `key%253Dsk-…`, `\nsk-…`, `\u0022sk-…`), as do an ANSI color code, a script written without spaces (`密钥sk-…`) and a header built without its space (`Bearersk-…`), and a boundary is a list of the contexts in which redaction stops. A random key of a vendor's length holds a digit or a capital with overwhelming certainty. Two residuals are stated: a glued word whose run holds a digit or a capital is still redacted (the scrub errs toward redaction), and a glued key made only of lowercase letters, `_` and `-` is kept. `tests/test_log_value_cut.py` pins both sides: the word cases survive whole, and every prefixed sample is redacted after each of 23 left neighbors. The non-prefix rules (the heartbeat URL, bearer, key-in-query, `api_key=`) are redacted wherever they match.

**A narrow run never swallows the head of a key glued after it** (issue #858). A run whose class lacks `_` or `-` (`gh?_`, `github_pat_`, `bc_uat_`, `bc_isk_`, `hf_`, `r8_`, `ste-`, `abl_`, `pplx-`) took the letters of a following key's prefix and stopped at its `_` or `-`: `hf_…aghp_ABC…` matched `hf_…agh`, and the rest of the GitHub token matched nothing, nor did all of it when the first match was kept as a word. So each such rule is two alternatives, tried in order: the run ends, as early as it can, where a whole key by its own rule begins (any rule's prefix with that rule's minimum run), and then takes everything after it in `[A-Za-z0-9_-]`, so a third key glued on goes with the second; else the run matches as it always did. It is a negative lookahead written without one, because the box's `ai_scrub` is VRL, whose regex has no lookaround. A match that took a glued key is never kept as a word, since its tail can hold a later key no word rule ever kept (`xhf_…ghp_…-sk-…`), so nothing is redacted less than before: a match only grows, and what it grows over is redacted with it. The cost, only in the glued case, is that a `-word` glued after the second key goes with it, and so does a lowercase word that held a whole key's shape. The box captures the glued alternative to know it, as `_observability._rule` captures `glued<n>`. A wide run (`sk-`, `xai-`, `bc_???_`, `AIza`, `tk_`) already took a glued key whole. The pattern, as `_observability._rule` builds it, for a narrow `run` with minimum `n`:

```
PREFIX(?:(?P<glued>RUN{n,}?(?:GLUED)[A-Za-z0-9_-]*)|RUN{n,})   (a match with glued set is never a word)
GLUED = the alternation, over every prefix rule, of PREFIX RUN{n}   (each rule's own run, exactly its minimum)
```

**The literal rules run first.** The source scrubs in three passes: the heartbeat URL, the other rules anchored on a literal of their own (an `x-api-key` header, bearer, key-in-query, `api_key=`), then the prefix rules. In one alternation the leftmost match wins, and a prefix rule's run is made of the characters `api-key` and `bearer` are, so `task-runner-api-key=abc123secret` was consumed as the run `sk-runner-api-key` and its value matched nothing. A prefix run cannot reach across the `[redacted]` the first pass leaves. Inside the second pass, an assignment whose value is a bearer header (`AUTH_API_KEY=Bearer eyJ…`) takes every `Bearer` and the token together (`Bearer Bearer eyJ…` is a token stored with its scheme and sent with another). An `api_key=` value may open with a `"` (`API_KEY="…"`, which once matched nothing), and a bearer header glued to an `api_key=`, query or `x-api-key` value (`api_key=KBearer eyJ…`) is taken with its token, because the value would otherwise end at the scheme word and leave the token to no rule; each such value is two alternatives, the first ending in a glued `bearer` and its token, written without lookaround for the box. A differential fuzz of 600,000 texts built from every prefix rule's real shapes, glued and separated by every literal rule's label, finds none the new scrub leaves more of than 0.185.0's, and none it leaves any of.

**A heartbeat URL is redacted from its scheme when it has one** (basecradle-router#398). The rule is the box's, `(?:https?://\S*)?/heartbeat/\S+`: a URL with a scheme from the scheme on, so userinfo before the host (`https://john:…@host/…/heartbeat/…`) never ships; one without (`curl uptime.example.com/api/v1/heartbeat/…`) from `/heartbeat/` on, keeping its host, which is not the secret. It runs first, as the box's does, because a rule that matched first and stopped inside the URL (a bearer run stops at the scheme's `:`) would leave the userinfo to the scheme-less arm alone. One addition to the box's bytes: a bearer scheme word ending the URL's run, not glued to a letter or digit, is kept for the bearer rule (`_observability._TRAILING_BEARER`), because taken with the URL it left its token in the next run with no rule to match it (compact JSON `…/heartbeat/K9","auth":"Bearer eyJ…`, a query value `…&access_token=Bearer eyJ…`).

Every source rule is linear in its input, because a value is scrubbed before it is cut and can be as long as a response body. The box's heartbeat pattern is not: in Python's `re` every `https?://` in a space-free run scans to the run's end before it fails, quadratic in the run. So the source matches each space-free run holding `/heartbeat/` once, from the run's start, and cuts it where the box's pattern would begin (`_observability._HEARTBEAT_RUN`, `_heartbeat`); `tests/test_log_value_cut.py` holds the two to the same bytes over random texts.

**A prose line is scrubbed whole, by the handler** (issue #857). `kv` scrubs the values it renders, but a prose line interpolates its own arguments (`_log.warning("…: %s", exc)`), and an exception's text is foreign: a vendor's 401 can echo the key it rejected. So every `basecradle-harness-*` command logs through `_observability.log_handler`, whose `ScrubbingFormatter` runs `scrub` over each record as rendered — message, traceback and stack — which covers every prose site, a future one, and another library's line (`httpx`, an SDK) alike, with nothing for a call site to remember. The wake and the cleanup install it at `HARNESS_LOG_LEVEL` in `LOG_FORMAT`; the commands that keep Python's default logging install it in that default's place (`install_log_handler`: `WARNING` and up, the bare message), so their stderr reads as it did. A clean line renders byte for byte as before. A `kv` line is scrubbed a second time, which changes nothing a value already lost, and **never across a value's closing quote**: the formatter scrubs between closing quotes (`_observability._CLOSING_QUOTE`, a `"` followed by the next `key=` or the line's end), because a quoted value ending in `/heartbeat/` or `?token=` holds no secret, and on the whole line the rule after it would take the quote and leave the line unparseable. No rule's legitimate match holds such a quote. One over-redaction is stated: a bare value ending in `bearer` beside a next field of eight or more characters goes with it, as it does on the box. An entrypoint's last word to stderr (`basecradle-harness-wake: …`, and every other command's) is scrubbed the same way. **The floor protects only a process whose handler the harness installed**: a library caller that configures its own handler gets `kv`'s scrub and the one prose site that scrubs its own argument (`_openai.ModelCatalog.rows`, the model list the first refuseless wake reads), and can install `ScrubbingFormatter` itself. A traceback Python prints for an exception nothing caught never passes a handler, and is the box's. `tests/test_prose_scrub.py` proves it: a sentinel key driven through each prose site's exception path never reaches the journal, every console script configures the scrubbing handler, every `basicConfig` goes through `log_handler`, no module attaches a handler, configures logging another way or stops a logger propagating, and every direct write of an exception's text (`print`, `sys.stderr.write`, `warnings.warn`, a formatted traceback) scrubs it. The guards read the source, not its data flow: an error's text assigned to a name and written later is a reviewer's to catch.

**A cut never falls inside a token.** A value is scrubbed whole (`_observability.scrub`), then bounded (`_observability.bounded`, 240 characters in `kv`); the cut backs off to the last character outside a `[A-Za-z0-9_-]` run, so a token is either whole, and so redactable by every rule after it, or absent. That run is the character class of every length-gated rule on the box, so a credential cut short of its rule's minimum would match no rule there and ship as a prefix. The source's own rules never see a cut text at all, because every log-bound site scrubs before it cuts. A value that is one such run past the bound is omitted. The router's `ai_otel_llm` remap cuts by the same rule (basecradle-router#385). Every site that truncates free text bound for a log line uses `bounded`: `kv`, the direct-message tool's ntfy refusal, the log-grammar probe's `journalctl` error, and the exception texts that reach `error=` (the Vertex and Anthropic inline-media refusals, the prune watermark's read error); so does the stall note's quoted vendor error, which is posted rather than logged.

**A vendor's words posted to a timeline are scrubbed by the same rules** (issue #791). The provider-failure report and the stall note quote a vendor's error text to peers, and a timeline has no box layer behind it, so `_report.verbatim` runs `scrub` over that text the moment it is dug out of the error, before the stall note's cut, and the stall note scrubs again at the post for a reason persisted on a claim before the scrub existed. What a rule matches becomes `[redacted]`, its label included (`Bearer `, `?api_key=`), and the rest of the vendor's sentence is unchanged; the rules are the log scrub's, so a timeline post carries exactly the redactions a log line would, over-matches included.

## The Palette

The fleet's journal is colored with four ANSI SGR colors and a reset (@origin, 2026-08-17), and **this section is where each tier is defined, once, for the whole fleet**: the harness, the router (its daemon lines and its `ai_otel_llm` remap) and the NOC's color corpus adopt these values, so a color means the same thing in a Live Tail whichever program wrote the line. A color wraps a **whole token** — a head, or a whole `key=value` pair — and never lands inside one, so every substring search over the plain line still matches the colored one. A token no tier names is plain: an unrecognized verdict is never guessed at a color. A non-empty `NO_COLOR` turns every harness line plain (`_observability.color_enabled`).

| Tier | Color | SGR | Means |
|---|---|---|---|
| Success | Green | `\x1b[32m` | A unit of work began or ended well, or recovered. |
| Bookend | Blue | `\x1b[34m` | The neutral close of a wake, whatever its verdict; the `outcome=` pair on the same line carries the verdict. |
| Warning | Yellow | `\x1b[33m` | In-between: a skip, a deferral, a block, a retry, a degradation. Nothing failed yet, and something did not go as asked. |
| Operator's Choice | Yellow | `\x1b[33m` | Neither success nor failure: an outcome an operator chose, such as stopping a wake (the capital, 2026-11-03, issue #799). It is never Red, because nothing failed. |
| Failure | Red | `\x1b[31m` | A unit of work failed. |

Every colored token is followed by the reset, `\x1b[0m`. The two Yellow tiers share one SGR on purpose: a reader sees *neither success nor failure* at a glance, and the tier's name says which kind.

The harness's members (`_observability._VERDICT_VALUES` for the `outcome=` pairs, `_observability.head` for the heads):

| Tier | Heads | `outcome=` Values |
|---|---|---|
| Success | `wake start`, `wake continuing`, `wake billing_recovered`, `Wake breaker RESET`, `palace scrub` (an apply that found nothing to remove) | `ok` |
| Bookend | `wake end` | — |
| Warning | `wake skipped`, `wake deferred`, `wake truncated_turn`, `wake stalled`, `wake billing_blocked`, `degraded`, `Wake breaker TRIPPED`, `palace scrub` (a dry run) | `degraded`, `retrying`, `refused`, `fallback` |
| Operator's Choice | `wake stopped` | `stopped` |
| Failure | `wake failed`, `wake reported_failure`, `post failed`, `cleanup failed`, `cleanup blocked`, `palace scrub failed`, `palace scrub` (an apply that removed what the harness mined) | `error`, `failed` |

The Operator's Choice tier's members across the fleet today: `outcome=stopped` (the router's `stage=wake` line, and the harness's `wake end`), `event=wake_stopped` (the router), and the harness's `wake stopped`. The router's members of the other tiers are its own `logfmt.PALETTE`, pinned by its `tests/test_logfmt.py`. `tests/test_telemetry_contract.py` holds the harness to this section: the SGR of every tier, and the `outcome=` vocabulary in both directions.

## Model Calls

### `llm`

**One line per model-call attempt, whatever its purpose and whatever its outcome** (`_observability.log_llm_call`, the only emitter, and `log_llm_retry`, which writes through it). A model call is one unit of work, so it has one head, and how the attempt ended is its `outcome` (issue #782, ruled 2026-11-03). A retry, a refusal, an unusable answer and a final failure are outcomes of the call, never heads of their own. There are two kinds of line. **A billed line** records what the vendor did with an attempt it answered: `ok`, `refused` or `fallback`, with the tokens and cost it billed. **A decision line** records what the harness did with an attempt that brought back no usable answer: `retrying` or `failed`, with no cost on it, because the attempt returned no usage to price. That is a fact about the line, never about the bill: a vendor may still charge an attempt that failed after it began generating (refuseless confirmed it, 2026-11-12), and such a charge never reaches an `llm` line; it shows only in the vendor's own spend counter, which [`key spend`](#key-spend) reads where the vendor states one. An attempt the vendor answered and the harness could not read writes both: its billed `fallback` line (the adapter reads the answer before it writes the line, `_observability.read_answer`), then the decision that followed, naming the same `generation_id`. The reason is `truncated` when the answer stopped at the output cap (a tool call's arguments cut off, issue #788: the vendor's stop reason says so, or the output it billed reached the call's cap) and `invalid_response` otherwise (a body that did not parse). Neither double counts, because only billed lines are calls. Every line carries the fields its outcome requires (see [Outcomes](#outcomes)), so any producer can write its model work here: a harness wake, a Claude Code builder (rendered by the router's `ai_otel_llm` remap), or a library caller. Each fills the fields it has and omits the rest. The head `llm provider=` is byte-frozen. It is the NOC's anchor for every call, cost and duration series, and the NOC counts a **call** as a line on it whose `outcome` is neither `retrying` nor `failed` (basecradle-noc#1212).

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `provider` | The vendor that bills the call. | Every adapter. | Never. |
| `purpose` | What the model was doing for the agent: `main` (the brain's own turn, and its compaction summaries), `memory` (a memory-system call, e.g. the reranker), `helper` (a second model working for the brain, e.g. the describer, a model call a tool makes, or a Claude Code builder's auxiliary call). | `log_llm_call`; `main` by default. | Never. |
| `kind` | The job within `purpose`: `rerank` (memory); `image.describe`, `video.describe`, `search.grounding` (helper). | The caller that names the job. | `purpose=main`, or a producer that names no job (a builder's auxiliary call). |
| `endpoint` | The **serving upstream** a router dispatched the call to (`StreamLake`, `DeepInfra`), read off the router's own routing metadata. | The router adapters (`serving_endpoint`). | A direct-to-vendor call — the vendor *is* the endpoint. |
| `residency` | Where the vendor ran the call: the location or inference geography the vendor reports, else the one the request pinned (`us`, `global`, `europe-west4`). `global` means the vendor chose. | Adapters whose vendor reports or accepts a location (Google, Anthropic). | The vendor has no notion of location. |
| `model` | The model id. | Every adapter. | Never. |
| `duration` | Seconds the call took; for a call that needed retries, the caller's whole wait. | Every adapter, or the capturing caller. | `outcome=retrying` (the wait is on the line that ends the call); a call that never reached a vendor (a config-class fault); or the producer cannot state the whole wait (Claude Code reports only the last attempt's time on `api_error`). |
| `tokens_in` | Every input token the vendor billed for the call — uncached, cached, cache-written, and any prompt tokens a server-side tool added. | Every adapter, from vendor usage. | The vendor stated no usage, or stated only zeros; always on `retrying` and `failed`. |
| `tokens_out` | Every output token the vendor billed — the reply **and** any reasoning or thinking. | Every adapter. | As `tokens_in`. |
| `tokens_total` | Exactly `tokens_in` + `tokens_out`. Never the vendor's own total. | Every adapter. | As `tokens_in`. |
| `cached_tokens` | The part of `tokens_in` read from the vendor's prompt cache. The only trustworthy witness that caching works. | Every adapter whose vendor reports it. | The vendor reports no cache read. |
| `cache_write_tokens` | The part of `tokens_in` written to the vendor's prompt cache on this call. | Every adapter whose vendor reports it. | The vendor reports no cache write. |
| `tokens_reasoning` | The part of `tokens_out` spent on reasoning or thinking. | Every adapter whose vendor reports it, on every purpose. | The vendor reports none — never an invented zero. |
| `cost` | US dollars for the call's tokens. Tool fees are never folded in when the vendor states them apart (they belong on the tool's line). | Vendor-stated (OpenRouter `usage.cost`, xAI ticks) or computed (`_<vendor>_rates.py`, or refuseless's live price list, `_refuseless_pricing.py`). | Unstated and unpriceable: an unknown model, tier, or host, a price list that could not be read, or a price the list does not state (see `cost unpriced`); always on `retrying` and `failed`, which returned no usage to price. |
| `cost_basis` | `computed`, beside a computed `cost`. | `cost_basis()`. | A vendor-stated cost; no cost. |
| `cost_gap` | Why a call that billed tokens carries no `cost`, in the reason its `cost unpriced` line names, on every call it applies to: `rates_unavailable`, a live price list has not been read (issue #874); `unknown_model`, the table or list holds no row for the model; `not_reported`, the row or the call's own usage does not state what the figure needs (issue #876); `unpriced_tier`, the table carries the model and not the rate band the call was served on (a Flex, Priority or batch tier, a traffic class, a prompt past a long-context threshold the model has no row for, a geography with no multiplier, a location class or billing day no row covers); `unpriced_host`, the client is aimed at a host whose price is not the table's. The `cost unpriced` line is written once per wake per gap and cannot say how many calls a gap covered, or that a second gap covered any once the first was written; each call's own line can, so the calls are counted rather than inferred: a wake with one unpriced call and a wake with all of them unpriced once read alike. Every adapter that computes a cost fills it, from a table (issue #878) or a live list. A new value is a change to this page. | The adapters that compute a cost: OpenAI on any host it is aimed at and refuseless (`_openai._cost`, `_openai._listed_cost`), Anthropic and Google (`_cost`, Google's grounded `search` line included), and the describer's line from what it captured. | A `cost`; a call that reported no usage; an adapter that computes no cost (a vendor-stated one, or none stated); always on `retrying` and `failed`. |
| `outcome` | How the attempt ended: `ok`, `refused`, `fallback`, `retrying` or `failed` (see [Outcomes](#outcomes)). | The adapter for an answer it could not read; the caller that judged the attempt; `ok` when nothing judged it. | Never. |
| `category` | The vendor's own refusal category, as a slug. | The adapter. | Not `outcome=refused`, or the vendor gave none. |
| `attempt` | The harness's attempt number of the call, as `n/total`: the attempt being retried, or the attempts a final failure made. | `_retry.Retry`; `1/1` on a compaction call, which is never retried. | Not `retrying` or `failed`; a fault found before any vendor was reached (a missing key, an unbuildable SDK), which made no attempt; a producer that does not count them. |
| `reason` | Why the attempt was not `ok`: a fault slug on `retrying` and `failed` (`rate_limited`, `server_error`, `transport`, `timeout`, `invalid_response`, `truncated` (a tool call cut off at the output cap, issue #788), `context_length`, `payload_too_large`, `tool_schema`, `api_error`, `provider_error`, `internal`, `undecodable_video`, `config:<cause>`), or an answer slug on `fallback` (`truncated`, `empty_response`, `missing_parts`, `parse`, and `invalid_response` for an answer the vendor billed that could not be read). An unreadable answer that stopped at the output cap (the vendor's stop reason says so, or the output it billed reached the call's cap) is `truncated`, never `invalid_response`, on its billed line and on the decision that follows, so a runaway is told apart from a malformed body. An HTTP 504 is `timeout` on every vendor: a deadline ran out, the vendor's or the harness's. A fault known only by its HTTP status is named by [Fault Slugs](#fault-slugs), the same words on every producer. | Whoever saw the fault, or the caller that judged the answer (`_observability.call_outcome` draws the split). | `ok` and `refused`. |
| `detail` | The vendor's or adapter's own text explaining `reason`. | The caller. | Nothing to explain. |
| `retry_after` | The wait the vendor asked for on the failed attempt. | `_retry.diagnostics`. | The vendor hinted none. |
| `provider_code` | The upstream's own status code inside a router's error body. | `_retry.diagnostics` (OpenRouter). | Not a routed failure. |
| `routing_attempt` | The router's own attempt counter (`0` = reached no upstream). Never the harness's attempt. | `_retry.diagnostics` (OpenRouter). | As `provider_code`. |
| `attempts` | The router's per-upstream account of who refused (`parasail:429,coreweave:429`). | `_retry.diagnostics` (OpenRouter). | As `provider_code`. |
| `timeout` | The budget the timed-out attempt ran out of, seconds. | `_retry.Retry`; the engine's give-up. | `reason` is not `timeout`, the budget is unknown, or the deadline was the vendor's. |
| `timeout_scale` | The multiple of the fitted budget the retry gets. | `_retry.Retry`. | Not `retrying reason=timeout`. |
| `next_timeout` | The budget the retry gets, seconds. | `_retry.Retry`. | As `timeout`, on `retrying` only. |
| `next_in` | Seconds until the next attempt. | `_retry.Retry`. | Not `retrying`. |
| `surface` | Where a memory recall ran: `turn0` (before the turn) or `tool` (the memory tool). | The reranker. | Not `purpose=memory`. |
| `pool` | How many candidates the reranker was shown. | The reranker. | As `surface`. |
| `picked` | How many of them it selected. | The reranker. | As `surface`, or no answer to pick from. |
| `subject` | What a helper call was about: the file names of the media described. | The describer. | Not a describe. |
| `timeline` | Common: the wake the call belongs to. | The wake (`_observability.in_wake`), ambiently. | The call was made outside a wake (a builder, a library caller). |
| `delivery` | Common. | The wake, ambiently. | As `timeline`, or as Common. |
| `step` | Common: the engine step the call was made in, so a wake that died at step 5 of 24 says so on the line of the call that ended it. | The engine (`_observability.at_step`), ambiently. | The call was made outside the engine's step loop (a Turn-0 recall, the reserve summary, a builder). |
| `generation_id` | The vendor's id for the call. Rendered next to last. | Every adapter (`generation_id`). | The response named none. |
| `billing` | `byok` — the cost is on the operator's own provider invoice, not the router's credits. Always last. | `_observability.byok`. | Billed the ordinary way. |

#### Outcomes

Each outcome has a strict set of fields it requires and a set it never carries, so the contract stays verifiable: `tests/test_telemetry_contract.py` renders every outcome and holds each line to its row. Every line also carries `provider`, `purpose`, `model` and `outcome`.

| Outcome | Means | Requires | Never Carries |
|---|---|---|---|
| `ok` | The vendor answered and the answer was used. A billed line. | — | `reason`, `category`, `attempt` |
| `refused` | The vendor refused the content. That is its final word, never retried, and the call was billed. A billed line. | `duration` | `reason`, `attempt` |
| `fallback` | The vendor answered (and billed) and the answer could not be used: the caller judged it unusable, or the adapter could not read it (`invalid_response`). A billed line. | `reason` | `category`, `attempt`, `next_in` |
| `retrying` | No usable answer, and the harness will re-issue the request. A decision line: no cost on it, because the attempt returned no usage to price. | `attempt`, `reason`, `next_in` | `duration`, `tokens_in`, `tokens_out`, `tokens_total`, `cached_tokens`, `cost`, `cost_basis`, `cost_gap`, `category` |
| `failed` | No usable answer, and the harness will not re-issue this request: the call ended there. A decision line: no cost on it, because the attempt returned no usage to price. A context overflow is `failed` too; the compaction that follows asks a different, smaller question as a call of its own. | `reason` | `tokens_in`, `tokens_out`, `tokens_total`, `cached_tokens`, `cost`, `cost_basis`, `cost_gap`, `category`, `next_in` |

**Severity is a rule of the line, and every producer follows it** (capital ruling, 2026-11-03, issue #786). A `failed` line is ERROR in exactly two cases:

- **A harness brain call the engine gave up on** after a transient fault exhausted its retries. It replaces the prose ERROR give-up line it retired, one for one, so the page that line raised is unchanged. This is the case the ruling names.
- **A harness second model's `config:` failure** (the reranker, the describer, search grounding), the first in a wake; a repeat in the same wake is DEBUG. A config-class fault is dead until a human acts, so it pages once. This is the second-model rule (`CLAUDE.md` → Second Models), which the ruling leaves as it stands.

Every other `failed` line is WARNING: a brain call that ended any other way (the wake's own lines carry the verdict a human acts on, `wake reported_failure` and `wake failed`), a second model's runtime-class failure, and **every failed call a Claude Code builder makes**, its auxiliary `purpose=helper` calls and their `config:` causes included, until basecradle#726's open question, whether a builder's failed call pages, is ruled. A `retrying` line is always WARNING, and so are `refused` and an unreadable answer's `fallback`. A wake that dies of a model call therefore writes two lines about two units: the call's `llm … outcome=failed timeline=… step=…`, and the wake's own `wake failed`.

The NOC's reading (basecradle-noc#1212): every series that counts, times or prices **calls** reads the billed lines, `llm provider=` lines whose `outcome` is neither `retrying` nor `failed`. The match is unanchored, because the verdict token is colored on every box. Retries and failures are series of their own, by `provider` and `purpose`.

#### Fault Slugs

**One vocabulary, every producer** (capital ruling, 2026-11-03, issue #786). A harness adapter, the router's remap of a Claude Code builder's `api_error`, and any producer that comes later name a failed attempt in the same words, the harness's, so one series counts them all. A fault a producer knows only by what the transport told it is named by this table, and by nothing else:

| What the Producer Saw | `reason` |
|---|---|
| No HTTP answer: the connection failed, or broke before an answer was read (DNS, TCP, TLS, a reset) | `transport` |
| HTTP `401` or `403` | `config:auth` |
| HTTP `402` | `config:billing` |
| HTTP `404` | `config:model_not_found` |
| HTTP `413` | `payload_too_large` |
| HTTP `429` | `rate_limited` |
| HTTP `504` | `timeout` |
| Any other HTTP `5xx` | `server_error` |
| Any other HTTP status | `api_error` |

The ruling set the transport, `401`/`403`, `429`, `504`, `5xx` and catch-all rows. The `402`, `404` and `413` rows are statuses the harness already names by the status alone, carried here so that the table is the harness's words, which is what the ruling adopts for every producer.

A producer that knows more than the status names the more specific cause, and only from the `reason` row's list. A wait that ran out of the producer's own deadline is `timeout`, not `transport` (`_retry.connection_reason`, from the error's type, else its cause). A refusal whose body the harness reads is named for what the body says: `context_length` (a `400` or `413` that says the context overflowed), `config:billing` (OpenAI's `insufficient_quota` on any status, Anthropic's `billing_error`, Google's `BILLING_DISABLED` on a `401`/`403`, or out-of-funds text), a routed refusal's `config:<cause>` (OpenRouter's region and provider-list `404`s), `tool_schema`. A word that is neither in this table nor in that list is a change to this page, never a producer's own.

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
| `step` | Common: the step the reply was, as `n/max`. | The engine. | Never. |
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
| `outcome` | As the first describe's `llm` line said: `ok`, `refused`, `fallback` or `failed`. | The describer. | Never. |

## Tools

### `tool`

**One line per tool use, whoever executed it** — a harness tool, an MCP server's tool, or a tool the vendor ran server-side inside a model call (`_observability.log_tool`, the only emitter). Tool calls and tool spend are both read off this one head; there is no second definition of "tool". The engine writes the line for a harness or MCP tool, with what the tool's vendor work recorded on its meter while it ran (`_observability.metered`); the engine's own refusal of a call (an unknown or withheld name) is `executor=harness`, since the harness is what answered it. An adapter writes the line for a server-side tool, one line per model call per tool, right after that call's `llm` line — or before it, when the caller holds its `llm` line back to judge the answer (`capture_llm_call`); `generation_id` is the join either way. A failed harness or MCP run writes a second, WARNING `tool` line carrying `name` and `error`; a failed server-side tool has a vendor code and no text, so it has no companion and its line is itself a WARNING.

**A tool that answers the model with its failure has still failed** (issue #822). Feeding a failure back is how a model recovers, so most failures reach the engine as a plain string. The tool says so through `_observability.tool_failed`, which returns the text unchanged and records the verdict on the run's meter; an MCP server says so with `isError`. The verdict is structural and never read off the result's text, because a shell command or a fetched page may begin with the word `Error`.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `name` | The tool's name as the agent knows it (`messages`, `generate_image`, `web_search`, `web_fetch`, `code_execution`, an MCP tool's name). A server-side tool carries the harness name of the capability, not the vendor's internal block type. | The engine; an adapter for a server-side tool. | Never. |
| `executor` | Who ran it: `harness`, `mcp`, or `server` (the vendor, inside a model call). | The engine; the adapter. | Never. |
| `provider` | The vendor that ran or billed the tool's work — the brain's vendor for a server-side tool; the media or search vendor a harness tool called. | Whoever knows the vendor. | The tool reaches no billed vendor (platform tools, memory). |
| `kind` | The shape of the work, for metered media and search: `image.generate`, `image.edit`, `video.generate`, `audio.transcribe`, `search.web`, `search.grounding`. One dotted vocabulary with `charge`'s `kind`. | The tool or adapter. | Unmetered work. |
| `model` | The vendor model the tool's work ran on. | The tool or adapter. | The tool used no model. |
| `count` | Billed or executed units on this line: searches, grounding queries, fetches, executions. A server-side tool writes one line per model call per tool, with its uses counted here — every use, a failed one included, so it can exceed the units a fee was charged on. | The tool or adapter. | A harness run whose work stated no units (one image, one transcription). Always present on `executor=server`. |
| `duration` | Wall time of the run — for a media tool, the generation and the upload that follows it. | The engine. | `executor=server` — the work ran inside a model call its `llm` line already timed. |
| `outcome` | `ok`, `degraded` or `error` (see [Tool Outcomes](#tool-outcomes)). No harness or adapter path writes `degraded` on this head today. | The engine; the adapter from the vendor's result block. | The WARNING companion line. |
| `reason` | Why it was not `ok`, as a slug. The engine's own refusals: `unknown_tool`, `withheld`, `server_side` (a server-side built-in called as a function), `elision_marker`, `invalid_arguments` (the call's arguments cannot bind to the tool's `run()` — an unknown, missing or misspelled name, or not an object; refused before the tool runs — or the tool itself refused what it was given: a required value for this action missing, an unknown action, a wrong confirm, an asset of the wrong kind or size); a tool that raised: `exception`; a harness tool's vendor call that failed: the fault slug, as on `llm` (`rate_limited`, `server_error`, `transport`, `timeout`, `invalid_response`, `provider_error`); `config:<cause>` for a setup fault only a human can fix, such as `config:missing_api_key`, `config:auth`, `config:running_as_root`, `config:prompts_not_installed`, `config:provider` (a tool that runs only beside one brain, built beside another), `config:base_url` (an `AI_BASE_URL` that is not a URL); `memory_error` (the memory search failed, so its "no matches" means nothing); `platform_error` (the BaseCradle platform refused or failed the call); `tool_error` (an MCP server's own `isError` verdict); `fetch_failed` (`web_fetch` refused the URL, could not complete it, or the server answered with an error status); `refused` (a push service's verdict on the request); `output_limit` (a shell command killed for passing the output cap). For a server-side tool, the vendor's own code or status, lowercased (`max_uses_exceeded`, `url_retrieval_status_error`, `outcome_failed`, `failed`, `incomplete`). | Whoever saw the failure. | `outcome=ok`. |
| `error` | Common; on the WARNING companion line only. | The engine. | `outcome=ok`. |
| `cost` | US dollars of the tool's own fee — never model tokens, which are on the `llm` line. | The tool or adapter. | No fee, or a fee nobody can price (see `cost unpriced`). |
| `cost_basis` | Common. | `cost_basis()`. | As Common. |
| `cost_gap` | Why a fee the harness computes carries no `cost`, in the reason its `cost unpriced` line names, on every use it applies to (issue #880), with the meaning it has on `llm`: `unpriced_host`, the tool's client is aimed at a host whose price is not the table's; `unknown_model`, the table holds no row for the model; `not_reported`, the vendor's answer does not state what the figure needs (an image call with no usage or no input split, or text output on a model with no text-output rate; a transcription stating neither a duration nor token counts the row prices; a grounded search that did not list the queries a per-query scheme counts). The once-per-wake line cannot count the uses a gap covered; this field does. | The tools and adapters that compute a fee: `generate_image` and `edit_image` (`_images._cost`), the assets tool's `listen` (`_audio._cost`), Google's grounded search (`_google.search`), and the server-side `web_search` fee on OpenAI and Anthropic (on any host but the vendor's own). | A `cost`; a fee stated by the vendor or by nobody (xAI's media, a router's searches); a tool that bills no fee of its own. |
| `generation_id` | Common: the model call the tool ran inside, so a tool joins to its `llm` line. | The adapter. | `executor` is not `server`, or the call's response named no id. |

#### Tool Outcomes

**One dimension, carried by two heads: `outcome` on `tool` and on `charge`, with one vocabulary** (issue #822). It is what makes Tools a sibling of Wake, Helper LLM and Memory on the fleet's *Outcomes* charts. The field keeps the name every sibling head uses; `tool_outcome` is the NOC's name for the label it extracts from it.

| Outcome | Means | `reason` |
|---|---|---|
| `ok` | The work was done. The answer may still be bad news: a shell command that exited nonzero, or a confirm gate's preview. | Absent. |
| `degraded` | The work was done and is usable, with a shortfall the line names. | Required. |
| `error` | The work was not done, or cannot be used. | Required. |

A `charge` is on this dimension because a billed resource is tool-adjacent (see [`charge`](#charge)), and the shortfall known today is one about money: Steel's stated cost and the rate card disagreeing. It is never a `tool` line, because a session is billed by the minute across zero or more tool uses (@origin, 2026-11-01). A clean `charge` is not a tool use, so a chart of tool outcomes counts every `tool` line (the WARNING companion excluded, as `tool_calls` excludes it) and a `charge` only when it is not `ok`, which keeps the chart's total equal to the tool calls plus the sessions that went wrong.

### `cost unpriced`

One WARNING, once per wake per gap, naming a charge that is real and that the harness cannot price — so a missing dollar is a counted fact rather than an absence nobody sees.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `provider` | Common. | The adapter or tool. | Never. |
| `model` | Common. | The adapter or tool. | The charge is not per-model. |
| `charge` | What carries no price: `tokens`, `container_time`, `browser_time`, `grounding`, `search`, `image`, `audio`. | The adapter, tool or producer. | Never. |
| `reason` | `unknown_model`, `unpriced_tier`, `unpriced_host`, `not_reported`, `rates_unavailable`. | The adapter, tool or producer. | Never. |

**A price read live, not from a table.** refuseless publishes its price on its models list rather than in a response, so its `cost=` is computed from that list (`cost_basis=computed`: the arithmetic is the harness's, over the rates the vendor publishes) and its gaps are worded the way every table's are. `rates_unavailable` is the one reason only a live source can have: no list has been read, so the call is not priced. It is not held for the wake (issue #874): a failed read is read again on a later call after a wait that starts at 5 s and doubles to 5 minutes (never shorter than four times the failed read took), a list that names no model counts as a failed read, the first read that answers prices every call after it, and until one does a call is priced from the list this process last read on the same host with the same key, if that is under an hour old. `unknown_model` is a model the list has no row for, and `not_reported` a price the vendor did not state for what the call billed: a row with no usable US-dollar rate (no `pricing`, another currency, a missing rate, which includes a call that states cached tokens on a row with no cached rate), a usage block missing either token count, or a cache write, for which the vendor publishes no rate. A list that **answered** is not held for the wake either when its row could not price a call (issue #876: a read made during a vendor outage answered with no price on the model's row, and the 23 calls after it went unpriced): a gap the row decides (`unknown_model`, or `not_reported` for the row's rates) reads the list again on the same backoff (never sooner than four times the last read took), the held list still pricing what it can until a read answers and a read that answers never taking away a row's price (a row it states none for keeps the held row that did), while a gap the call's own usage decides (a missing count, a cache write) is never a reason to read again. A call that reported no usage at all states no gap: it has no tokens a price is missing for. Each of these lines is written once per wake, and every call that went unpriced for one also carries its reason as `cost_gap=` on its own `llm` line, which is the count. A rate table's gaps are counted the same way (issue #878): every call an OpenAI, Anthropic or Google table cannot price carries `cost_gap=unknown_model`, `unpriced_tier`, `unpriced_host` or `not_reported` beside the one `cost unpriced` line, and a call that reported no usage states no gap there either.

**A tool's own fee** (issue #880). A fee the harness computes for a tool use is stated the same way: one line per wake per gap, and the reason on the use's own `tool` line as `cost_gap=`, which is the count. `search` is a server-side web search's per-call fee on a host whose price is not the table's (OpenAI's and Anthropic's `web_search`); `grounding` a grounded search's fee the table cannot count (Google); `image` an image made or edited through the Images API, and `audio` a recording transcribed, each named for the family of the `kind` its `tool` line carries (`image.*`, `audio.*`). A tool states no gap in prose.

A `container_time` or `browser_time` gap is the charge the next head would carry: the day a vendor states its container time, or a browser vendor's session minutes get a rate card, that charge is written as a priced `charge kind=compute.container` or `charge kind=browser.session` line and leaves this one. **The rule for the next value:** a gap in a charge billed per call or per use is named for what is billed (`tokens`, `grounding`, `search`, `image`, `audio`); a gap in a charge billed by a quantity of a resource is `<resource>_time` (or `_bytes`, `_storage` for a resource with no run time), and the sentence above names the [`charge`](#charge) `kind` that retires it. Either way it is generic in spelling, never a vendor's name or word, and a new value is a change to this page.

### `charge`

**One line per billed resource instance**: a dollar a vendor bills by a quantity of a resource — browser session minutes, container hours, a 15-minute GPU block, bytes, storage — rather than per call (@origin, 2026-11-01, on basecradle-noc#1153). It is neither a tool *use* nor a model call, so it has a head of its own, and every resource-billed product fits it without a new one.

```text
charge provider=<biller> kind=<what was bought> executor=harness|mcp|server unit=<billing unit> count=<quantity in that unit> [duration=<s>] [started=<iso>] [resource=<vendor resource id>] [generation_id=<model call it belongs to>] outcome=ok|degraded|error [reason=<slug>] cost=<usd> [cost_basis=computed] [vendor_usage=<vendor's own raw figure> vendor_unit=<its unit>]
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
| `outcome` | `ok`, `degraded` or `error` (see [Tool Outcomes](#tool-outcomes)): `error` when the resource instance's run did not end cleanly, `degraded` when it did and the line's own figure cannot be vouched for. A run that failed is `error` even when its figure is also in doubt. A charge is written either way, because the vendor bills either way. | The producer. | Never. |
| `reason` | Why the line is not `ok`, as a slug. On `error`: `crashed` (the instance died while running), `startup_failed` (it never became usable), `timeout` (a deadline ran out — the vendor's or the producer's — as on `llm`). On `degraded`: `cost_divergence` (the vendor's stated cost and the published rates disagree), `no_rate_card` (a stated cost with no published rate to check it against), `stated_cost_<why>` (the vendor would not state the cost, so `cost` is computed: `stated_cost_unreachable`, `stated_cost_http_<status>`, `stated_cost_malformed`, `stated_cost_not_usd`, `stated_cost_unavailable`). Generic in spelling, so every vendor's failures map onto the same words: Steel's `browser_crashed` → `crashed`, `startup_failed` → `startup_failed`, `creation_timeout` → `timeout`. A vendor's word that fits none is a change to this page, never written verbatim. | The producer. | `outcome=ok`. |
| `cost` | Common: US dollars billed for the instance. | The producer. | Never (see above: an unpriceable charge writes `cost unpriced`, not `charge`). |
| `cost_basis` | Common. | `cost_basis()`, or the producer by the same rule. | As Common. |
| `vendor_usage` | The vendor's own raw figure for the instance, verbatim, when it reports one in a unit this page cannot yet price (a plan's credits, say) — kept for calibrating a computed `cost`. Generic, never a vendor-named field. | The producer. | The vendor reports no such figure. |
| `vendor_unit` | The unit `vendor_usage` is in (`credits`). | The producer. | `vendor_usage` is absent; never present without it. |

The first producer is the NOC's Steel browser launcher (see [Outside This Repository](#outside-this-repository)); nothing in this repository writes a `charge` line yet. Its per-session spend line, whose `cost` is the dollar Steel states, so it carries no `cost_basis`:

```text
INFO charge provider=steel kind=browser.session executor=mcp unit=minute count=7 duration=402.18s started=2026-11-01T03:12:40Z resource=01a19518-4c40-7c41-9f3e-5d8c1a2b4e60 outcome=ok cost=0.07
```

A session Steel released as `browser_crashed`, billed for the minutes it ran:

```text
INFO charge provider=steel kind=browser.session executor=mcp unit=minute count=2 duration=71.40s started=2026-11-01T03:20:05Z resource=01a19518-5b12-7d03-8e6a-2c4f9b7d1e83 outcome=error reason=crashed cost=0.02
```

A session whose stated cost the published rates do not reproduce:

```text
INFO charge provider=steel kind=browser.session executor=mcp unit=minute count=8 duration=471.02s started=2026-11-08T14:02:11Z resource=01a19518-7e44-7a12-b3c0-6f1d8e2a9c07 outcome=degraded reason=cost_divergence cost=0.0633
```

A session on a Steel plan whose per-minute rate the launcher's table lacks writes no `charge` line, and one `cost unpriced` gap instead:

```text
WARNING cost unpriced provider=steel charge=browser_time reason=unpriced_tier
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
| `outcome` | `ok` (every reconcile returned), `error`, or `stopped` (the wake received SIGTERM — the router's `stop-wake` — and settled what it held; it matches the router's own `stage=wake outcome=stopped`, and is not a failure: the fleet's `wake_failed` reads `outcome=error` only). | The wake. | Never. |
| `signal` | The signal that stopped the wake, without its `SIG` prefix: `TERM`. | The wake. | `outcome` is not `stopped`. |
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

### `wake stopped`

An item a stopped wake held in flight, settled so no later wake resumes or re-drives it (issue #798). WARNING: deliberate and visible, never a page.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `item` | Common. | The wake. | Never. |
| `kind` | `messages`, `assets`, `webhook_events` or `tasks`. | The wake. | Never. |
| `timeline` | Common. | The wake. | Never. |
| `handle` | Common. | The wake. | `reason=signal` (the stopping wake reads the claims, not the items). |
| `signal` | As on `wake end`. | The wake. | `reason=stop_note`. |
| `reason` | `signal` (the stopping wake settled it itself) or `stop_note` (a later wake found the turn ending on the stop note, the stopping wake having been unable to write the claim). | The wake. | Never. |

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

## Accounts

### `key spend`

**What the brain's own key has been billed, as its vendor states it, once per wake** (`_key_spend`, issue #868). The fleet reconciles each agent's logged `cost=` against what its vendor billed (basecradle-noc#1365). An account's balance cannot separate one agent from another when several share an account, so the per-agent billed figure is the calling key's cumulative spend. It is read with the brain's `AI_API_KEY` from every vendor that states one: refuseless (`GET /credits` → `key_spent`, `key_spent_since`, `updated_at`) and OpenRouter (`GET /key` → `data.usage`, to the vendor's own precision). Any other brain writes no line.

The line is the last of a wake's work, so every call of the wake has settled and the next record is `wake end`. A wake that failed or was stopped before it writes none (the next wake's figure is cumulative), and a stop that lands during the read ends the wake as any stop does. A read that produced no figure is a WARNING on this same head, with `outcome=error` and its slug, never silence, so a consumer can tell "no figure this wake, because X" from "the read never ran". The read never holds or fails the wake: it is held to a ten-second deadline on the whole attempt (five to connect plus five to answer, summed by `_deadline` and checked as each part of the answer arrives, so a vendor trickling bytes overruns it by at most one idle interval), a figure longer than 32 characters is `invalid_response` rather than a value the 240-character bound would cut, and anything it did not foresee is `reason=exception`. The key, the URL and the vendor's body never reach the line.

```text
INFO key spend provider=refuseless outcome=ok spent=0.001405 as_of=2026-11-10T21:12:05Z timeline=019f6e71-2a12-7b69-a204-0fec1497b9c2
INFO key spend provider=openrouter outcome=ok spent=1.047832516 timeline=019f6e71-2a12-7b69-a204-0fec1497b9c2
WARNING key spend provider=refuseless outcome=error reason=rate_limited timeline=019f6e71-2a12-7b69-a204-0fec1497b9c2
```

Only five of the fleet's harness agents have a brain that writes this head, so the NOC cannot watch it off live traffic; the log-grammar probe renders it on every agent instead (`basecradle-harness-log-grammar key_spend`, issue #871), through the same two renderers (`_key_spend.key_spend_line`, `key_spend_error_line`), under its own identifier, at INFO, led by `PROBE` and stamped `source=probe`. Its `provider`, `reason` and figures are synthetic, and it carries no `timeline`:

```text
INFO PROBE key spend provider=log_grammar_probe outcome=ok spent=0 source=probe agent=jt
INFO PROBE key spend provider=log_grammar_probe outcome=ok spent=0 since=1970-01-01T00:00:00Z source=probe agent=jt
INFO PROBE key spend provider=log_grammar_probe outcome=error reason=log_grammar_probe source=probe agent=jt
```

A consumer of the figure reads it per agent, so it must block-list `source=probe` as the alarms do, or a probe's `spent=0` reads as the agent's reading.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `provider` | Common: the brain's vendor, whose key was read; `log_grammar_probe` on a probe line, which read no key. | `_key_spend`. | Never. |
| `outcome` | `ok` (the figure was read) or `error` (it was not). | `_key_spend`. | Never. |
| `reason` | Why there is no figure, as a slug: the Fault Slugs a balance read can meet (`config:missing_api_key`, `config:auth`, `config:base_url`, `config:billing`, `config:model_not_found`, `payload_too_large`, `rate_limited`, `timeout`, `transport`, `server_error`, `api_error`, `invalid_response`, `provider_error`), **`not_reported`** (the vendor answered and stated no figure for the key), or `exception`; `log_grammar_probe` on a probe line. | `_key_spend`. | `outcome=ok`. |
| `spent` | US dollars the key has been billed, **cumulative**, exactly as the vendor states it, in plain fixed-point. It is never `cost`: *Spend by Provider* sums every `cost=` on any head, and a cumulative figure there would be counted again on every wake. A consumer diffs it across wakes. | `_key_spend`. | `outcome=error`. |
| `since` | When the vendor's counter started, so a reset or rebase reads as a break in the series and never as negative spend. | `_key_spend`. | `outcome=error`; the vendor states `null` (refuseless: the counter runs from the key's creation) or no start at all (OpenRouter). |
| `as_of` | When the vendor read the figure (refuseless's `updated_at`). | `_key_spend`. | `outcome=error`; the vendor states no such moment (OpenRouter), in which case the record's own journal timestamp is the read. |
| `timeline` | Common. | The wake. | A probe line. |
| `delivery` | Common. | The environment. | As Common. |
| `source` | Common. | `_log_grammar`. | Real traffic. |
| `agent` | Common. | `_log_grammar`. | Real traffic. |

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

The slices are the issues that remove each row: **slice 2 — tools** ([#758](https://github.com/basecradle/basecradle-harness/issues/758), whose rows T1–T9 are struck), **slice 3 — fields** ([#759](https://github.com/basecradle/basecradle-harness/issues/759), whose rows F1–F36 are struck). Issue [#782](https://github.com/basecradle/basecradle-harness/issues/782) retired the `llm retry` and `llm refused` heads into `llm`'s `outcome`, and gave the brain's final failure a line. Issue [#796](https://github.com/basecradle/basecradle-harness/issues/796) struck S1, found when issue #786 stated the fault-slug table: a bare `402` is `config:billing` on every adapter. No departure stands today.

| ID | Violation | Witness | Removed By |
|---|---|---|---|

### Outside This Repository

The NOC's Steel browser launcher writes its own `browser provider=` head. Its per-session spend line retires into the [`charge`](#charge) line (@origin, 2026-11-01: a session is billed by the minute, not per tool use, so it is not a `tool` line); its lifecycle lines stay on a NOC-only head that no column reads. It is not emitted here, so it is not witnessed here; the launcher's move is basecradle-noc#1153's. A cost shortfall is stated on the `charge` line's own `outcome` (issue #822), so the launcher's `session.ended` companion is detail, never a second verdict a chart reads. Slice 2 carried the rest to the NOC through the capital, together with the NOC columns that read what this repository retired in slice 2 (*Tool Cost* off `media`/`browser`; `web_search_requests`, `web_fetch_requests`, `code_execution_requests` off `llm`), which read the `tool` line instead.

The Claude Code builders' model calls reach this contract through the router's `ai_otel_llm` remap (basecradle-router `deploy/vector.yaml`), which renders each `api_request` as an `llm` line. Under issue #782 an `api_error` (Claude Code's final failure, after its own retries) is rendered as `llm … outcome=failed reason=<slug>` with no tokens, and an auxiliary call as `purpose=helper` with no `kind` (basecradle-router#384). Its slug is read off the status by [Fault Slugs](#fault-slugs); the remap adopts that table's `transport`, `config:auth`, `config:billing`, `config:model_not_found` and `payload_too_large` in basecradle-router's follow-up to issue #786, and until then writes `api_error` for each. Its `duration` is absent where Claude Code cannot state the whole wait (an `api_error` after more than one attempt, or with no status), and its `failed` line is WARNING. A builder's line carries no `timeline`, `delivery` or `step`: it has no wake.

## Where a Signal Lives

@origin, 2026-11-02 (the telemetry audit, basecradle#726, ruling 2): **every meaningful signal has one of two homes.**

- **A chart, with its alert,** for the aggregate somebody watches.
- **The log stream,** for the per-event detail somebody reads to troubleshoot. The fleet server's logs are where you look when an agent breaks.

"It doesn't fit a chart" never means "drop it". It means "log it". Detail specific to one producer that does not generalize into a shared chart still belongs in the stream, keyed by agent. So nothing operationally meaningful is ever deliberately left out: a signal is charted, or it is logged.

## Changing the Contract

The contract changes before the code does, never after. A change is one pull request that edits this page, the emitters, and the Known Violations table together, and it moves every adapter whose vendor has the datum — never one vendor at a time. A new field or head is a design decision about the whole fleet's telemetry: it is raised `needs-capital` first, and lands only once decided. Anything the NOC extracts is coordinated with the NOC through the capital in the same arc, so that no column goes dark.
