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
| `provider` | Who did the work. On a model call or a tool, the vendor that bills it (`AI_PROVIDER`'s value for the brain); on a memory line, the memory system's software, by the name its adapter was selected by (`mempalace`, or `mempalace-previous` for today's adapter under its second name, issue #1010). Software never names a head (issue #485); it appears only here. Always a lowercase slug, `^[a-z0-9._-]+$`, and never a display name (see [Identifiers](#identifiers)). |
| `model` | The vendor's model id the work ran on, exactly as configured. No grammar of the harness's own (see [Identifiers](#identifiers)). |
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

### Identifiers

Three values are identifiers a consumer joins on: a price table, a label map, a NOC column. Each has a grammar the harness guarantees, or the page says it has none. An identifier is a machine token and never a display name; how a person reads one is a layer of its own (@origin, 2026-11-14, issue #889).

| Identifier | What It Is | What a Consumer May Rely On |
|---|---|---|
| `provider` | The vendor that bills the work, or the memory software (Common Fields). The brain's value is `AI_PROVIDER`; a describer's own stack's is `HARNESS_DESCRIBER_PROVIDER`; every other value is a slug in the harness's code. | Always a lowercase slug, `^[a-z0-9._-]+$`. A configured value outside it, capitals included, is refused wherever the configuration is read to run or resolve the agent (`_basecradle.normalize_provider`: a wake, `--resolved-config` and `basecradle-harness-resolve` exit nonzero, a describer is `config:invalid_provider`), never folded and never run. The installer refuses it before writing, and `basecradle-harness-verify` reports it as `provider-invalid` and compares the overlay's recorded provider as written (issue #891). Every value the harness writes itself is held to it by `tests/test_telemetry_contract.py`. |
| Provider display name | How a person reads a `provider`: the vendor's own casing (`xAI`, `OpenAI`, `OpenRouter`, `Refuseless`, `MemPalace`), never a lowercase slug and never forced Title Case. | One table, `_report._PROVIDER_LABELS`, keyed by slug. It never appears on a line. Every slug the harness can write has an entry or a stated reason it has none (`tests/test_telemetry_contract.py`'s `NO_DISPLAY_NAME`), so a new Provider lands with its name in the same change. A generic vendor (an `AI_PROVIDER` the harness does not know) has no entry, because the harness knows nothing about it, and reads as its slug. |
| `model` | The vendor's own id for the model, passed through exactly as configured (`AI_MODEL`, or a second model's own setting). The harness never rewrites it. | It is the id the vendor was sent, byte for byte, so it joins against the vendor's own model list and the harness's rate tables. **No grammar is guaranteed**: the namespace is the vendor's, and some vendors publish ids with capitals or a `:` variant suffix. Every id in the fleet's inventory today (2026-11-15) is lowercase `[a-z0-9._/-]`; that is an observation, not a rule this harness enforces. |
| A tool's `name` | The name the agent was offered the tool under: a harness tool's own name (`messages`, `web_fetch`), the harness name of a vendor's server-side capability, or an MCP tool's `<server>__<tool>`, built from the server's name in the agent's configuration (`mcp/<server>.json`, or the key of its `mcpServers` wrapper) and the tool name the server reports. | `^[A-Za-z0-9_-]{1,64}$`, the function-name grammar the model vendors require, for every tool the harness ships (tested) and every MCP tool, which is sanitized and cut to it by construction (`_mcp.mcp_tool_name`) with the server's own case kept, so it is not guaranteed lowercase. A tool an operator drops into `tools/` names itself: the harness does not check it, and a vendor refuses a name outside the grammar. On an `unknown_tool` refusal the `name` is whatever the model called, so it has no grammar there. An MCP server's name reaches a line inside a tool's `name` and in `wake start`'s `mcp_servers`, in both places sanitized the same way (a tool's `name` is then cut to 64 characters, so a server name of more than 62 is cut there and not in `mcp_servers`). |

### Credential Scrubbing

A value can carry text that is not the harness's (an exception, a vendor's error body), so a credential can reach a line. Two layers redact it, in order, and each owns a stated part of the job (issue #787):

| Layer | Where | Owns |
|---|---|---|
| **Source** | `_redaction.scrub`, applied by `kv` to every value (and to the unspoken `text`) before the value is cut, and by `_observability.ScrubbingFormatter` to every record the handler of a harness entrypoint renders, prose lines and tracebacks included (issue #857). | **Every credential shape the harness knows of**, which is a superset of the box's: the value of every credential-named assignment, by the NOC canary's own definition (below), and the fleet's own (`sk-`, `sk-or-`, `xai-`, `bc_uat_`, `bc_isk_`, refuseless `abl_`, Perplexity `pplx-`, an ntfy access token `tk_`), every shape `ai_scrub` carries (GitHub `gh?_` and fine-grained `github_pat_`, Google `AIza`, Hugging Face `hf_`, Replicate `r8_`, Steel `ste-`, a Better Stack heartbeat URL, an `api_key=` assignment), the bearer and key-in-query forms, an `x-api-key` header value, and the string value of a JSON member named on an exact list of credential names (issue #943). It is the only layer off-fleet: a developer's install has no Vector. |
| **Box** | Vector's `ai_scrub` on each fleet server (basecradle-router `deploy/hosts/*/vector.yaml`, basecradle-noc `deploy/vector.yaml`), after the line is written. | Lines the harness never wrote (another program, a traceback Python prints for an exception nothing caught, a router line), the journald metadata no message rule reaches (`_CMDLINE`, dropped `sudo` and Vector events), and a shape the source has not yet adopted. |

**The source's definition is one module and one public file** (issue #999): every rule the source reads is built from the package's `basecradle_harness/data/secret-shapes.json` (`docs/secret-shapes.md`), which the fleet's own scans can read without importing the harness, and `_redaction` is the one module that applies it, to log lines and equally to what the harness files into memory and to the text files a tool saves. Beside the shapes it carries a second half, **the agent's own configured secrets by value**: the value of every environment variable whose whole name names a credential by the assignment rule's words, and of every credential the harness holds under such a name, at least 20 characters long, blanked as `[redacted]` wherever it appears in a line (`_redaction.configured_values`). It takes its place in the same union as every other rule.

The two lists cannot be one list, because one is VRL in another repository, so the rule binding them is: **a shape added to `ai_scrub` is added to the source's rules at no longer a minimum length**, and `tests/test_log_value_cut.py` holds one fabricated sample per box rule at the box's own minimum. The one stated difference is the bearer rule's minimum: the box takes a token of any length, and the source needs eight characters, so a diagnosis like `Bearer token missing` keeps its words. No real bearer credential is shorter.

**A bearer header's token is taken whole, quoted or not, and a run of scheme words goes with the token after it** (issue #929). The rule, at the source and on the box alike:

1. The scheme word is `bearer` in any case, glued to anything before it, followed by whitespace.
2. The token is the word after that whitespace: a run of `[A-Za-z0-9._~+/-]` (eight or more at the source, any length on the box), or, when the word opens with a quote or a substitution (`'`, `"`, `` ` ``, `$'`, `$(`, `${`), the whole shell word it opens, to its closing quote or the end of its line when it never closes, or, when it opens with an escaped quote (`\"`, `\'`, escaped once or more: JSON inside a JSON string, `"Bearer \"eyJ…\""`), the rest of its word, to the next blank or record break: an escaped quote may be unbalanced or cut, so no closing one is trusted, and compact JSON after it to the next blank goes with the token. Quoted, a token is no run of token characters, so `Bearer 'eyJ…'` once matched nothing at either layer. A token in any other bracket (`Bearer (eyJ…)`, `Bearer <eyJ…>`) is read by neither layer.
3. While the word read as the token ends in `bearer`, in any case (`x.Bearer`, `a=Bearer`, `"q"bearer`), the word after the whitespace that follows it is read as the token in its place, by rule 2, and the header ends at the last token read. So `Bearer x.Bearer 'eyJ…'` loses `eyJ…`, where the source once took `x.Bearer` as the token and left the quoted one after it. A scheme word with no token after it is the end of nothing: `Bearer Bearer12 Bearer 'eyJ…'` is two headers, and both tokens go.

The source reads it in code (`_redaction._BEARER`, `_bearer_end`), reading matches leftmost-first, so a scheme word inside a token already read is that token's text. The box's VRL regex has no lookaround and runs on a linear engine, and it must now take at least the following. A fuzz of every scheme-word run, separator and singly quoted or escaped token leaves no token under it (40,000 texts, with tokens glued after a closing quote, escapes inside quotes and escaped quotes left open; the box's present `(?i)bearer\s+(?:bearer\s+)*[A-Za-z0-9._~+/\-]+=*` leaves 16,871 of 20,000 such texts):

```
(?i)bearer\s+(?:(?:'[^'\n]*'|"[^"\n]*"|`[^`\n]*`|[^\s'"`])*bearer\s+)*(?:(?:\\+["']|\$'(?:[^'\\\n]|\\.)*'?|'[^'\n]*'?|"(?:[^"\\\n]|\\.)*"?|`[^`\n]*`?|\$\([^)\n]*\)?|\$\{[^}\n]*\}?)\S*|[A-Za-z0-9._~+/\-]+=*)
```

**It reads one quoted piece or substitution, one level deep, and what is glued after it, and that is its stated limit**: a nested substitution (`Bearer $(echo $(cat) eyJ…)`), a quote inside one (`Bearer "$(a") eyJ…"`) or a second quoted piece in the same word (`Bearer 'a'b' c eyJ…'`) ends it early and leaves the rest of the token, where the source reads the whole shell word and leaves none. No regular expression nests. It is not for a backtracking engine such as Python's `re`, where its nested repetition is quadratic.

**A vendor key is covered only when its shape can be stated, never guessed** (issue #856). Each prefix rule is the prefix plus a minimum run under the vendor's own length, so a whole key always matches:

| Vendor | Rule | The Shape, and Whose Word It Is |
|---|---|---|
| refuseless | `abl_[A-Za-z0-9]{20,}` | `abl_` plus 48 alphanumerics, 52 characters in all, read off two real keys (@origin, 2026-11-10). |
| Perplexity | `pplx-[A-Za-z0-9]{20,}` | The `pplx-` prefix is Perplexity's documentation's; the 48 alphanumerics after it are gitleaks's `perplexity-api-key` rule. |
| ntfy | `tk_[A-Za-z0-9_-]{20,}` | `tk_` plus 29 characters, 32 in all, from ntfy's documentation; the harness holds one as `NTFY_DM_TOKEN`. ntfy generates the body from lowercase letters and digits, and its validator also admits capitals, `_` and `-` (`^tk_[-_A-Za-z0-9]{29}$`, ntfy's `user/util.go`), which is the run this rule takes. A generated token with no digit, about 1 in 12,500, is the word residual stated below when glued to a letter. |

A key with no prefix of its own has no shape a prefix rule can match, so it is covered where it travels instead. **An `x-api-key` header value is redacted by the header's name** (issue #861): Wiro's documentation gives its API key no prefix, length or character set, and sends it as that header. The rule is `x-api-key` in any case, with `_` for either hyphen, followed by `:` or `=>` (a header dump, a Ruby hash), or by a quote, escaped or not (at any depth, below), and then `:`, `=`, `=>` or `,` (`"x-api-key": "…"`, JSON inside a JSON string, a `('x-api-key', '…')` header tuple); the value may open with a quote, an escaped quote, a bytes literal's `b'` or bearer scheme words, and runs to the next space, quote, backslash or `&`, because a key of no stated shape may hold anything else. A bearer header glued to the value (`x-api-key: KBearer eyJ…`) is taken with it, token and all, since the value would otherwise end at the scheme word and leave its token to no rule. The unquoted `X-API-Key=…` is the `api_key=` rule's, which always took it with the wider value it allows. The header's name goes with its value, as every literal rule's label does, but for a label ending in a credential's name (below). **A value that ends in a whole header of its own is read on through it** (issue #929): in `x-api-key: "x-api-key": "K9…"` the value is `x-api-key` up to the quote, in `x-api-key: x-api-key: K9…` it is `x-api-key:`, and in `x-api-key: x-api-key:Bearer K9…` it is `x-api-key:Bearer`, so the header after it went to no rule; the span now ends where the last such header does (`_redaction._literal_spans`, the one reading every literal rule reads on with, issue #934). The box's rule is a regular expression read leftmost-first, and leaves `K9…` in both. A bare `x-api-key` with no value is a word, kept; a colon after it is read as a value, so `x-api-key: missing` loses `missing`, and a quoted list of header names (`"x-api-key", "x-nonce"`) loses the second (the scrub errs toward redaction). Wiro's signing secret is never sent (its `x-signature` header is an HMAC keyed on the API key over the secret and a nonce), so nothing else of Wiro's travels. The harness's own route to Wiro, the generic OpenAI-compatible profile, sends the key as `Authorization: Bearer`, which the bearer rule already covers.

**The header's quote is escaped at any depth, in its label and its value's opener alike** (issue #935). JSON inside two nested JSON strings writes a quote as `\\\"`, and until 0.198.3 the label and the opener each took a quote escaped once, so a header nested that deep matched neither and shipped whole (`{"log": "{\"log\": \"{\\\"x-api-key\\\": \\\"K9…\\\"}\"}"}`, and `x-api-key: \\\"K9…\\\"` in a header dump). Both now read the piece the `api_key=` opener reads (`_redaction._ESCAPE`): no backslash, or an odd run of them. After an even run the quote is a real one, the backslashes escaping each other (`{"x-api-key":"\\","k":"v"}`), and it is never taken. That reading is exact for the outermost string only: one encoding pass turns a run of `n` backslashes before a quote into `2n + 1`, so a string nested inside the line whose header value ends in an even run reaches the line as an odd run, and its real close is read as an opener. Where structure and the next string follow that close, the opener opens nothing (issue #939, below), and `{"log": "{\"h\":\"x-api-key: \\\\\",\"k\":\"v\"}"}` scrubs to itself; until then it scrubbed to `{"log": "{\"h\":\"[redacted]\"k\":\"v\"}"}`, the line parsing and the inner string losing its close. (An unquoted `api_key=` value nested so, `x\`, broke the line itself until 0.198.3, because that rule's run took the escape of the line's own closing quote: issue #937, below.) A quote escaped at every level is a run of exactly `2^d − 1` backslashes, a set no regular expression states for every depth. The value's run already stops at a backslash, so its escaped close stays as deep as it came, and every enclosing JSON string still parses: at depths 1 to 3 the innermost string reads `{"[redacted]"}`. A line that is itself the bare JSON object, unnested, reads `{"[redacted]"}` too and does not parse as JSON, because the header's name and both its quotes go with the value; that was so before #935 and leaks nothing. A differential of 1,000,000 texts built from the header's label at every escape depth, odd and even, its openers, `api_key=`, query labels, scheme words, a credential name and fabricated tokens, some nested in JSON up to three deep, was run against `main` before this change and against 0.198.2. The new scrub takes more in 239,326 and 239,515 of them. It keeps a token the older one took in 32 and 33, and every such token is one the older scrub took only by reading its marker again (`_past_marker`): the header now matched starts the span before the credential's name inside it, so the span is written as `[redacted]` and not as the marker, and nothing reads past it. In `x-api-key\\\':A_TOKEN=x-api-key\' :K9…x-api-key\\",tail`, the key goes and `tail` stays; it follows a real quote, after an even run of backslashes, and no rule reads it as a value.

**The box takes a header escaped once, and that is its stated limit.** `ai_scrub`'s header rule (basecradle-router `deploy/hosts/*/vector.yaml`, at `ec642e9`) opens its label and its value with `\\?["\x27]`, a quote escaped once at most. So on `x-api-key: \\\"K9…\\\"`, and on a header in JSON nested two deep or more, neither matches, no other box rule reads the value, and the key ships. The box keeps that one level until the router's first pass becomes one alternation (basecradle-router#516); until then the source is the only layer that takes a header escaped more than once, and a line the harness writes is scrubbed at the source first, so only a line it never wrote reaches the box with that shape.

**A vendor prefix inside a word is a word, not a credential** (issue #794). A prefixed match (`sk-`, `sk-or-`, `xai-`, `bc_…_`, `gh?_`, `github_pat_`, `AIza`, `hf_`, `r8_`, `ste-`, `abl_`, `pplx-`, `tk_`) is kept only when **both** hold: a letter, digit or combining mark, in any script, precedes it, **and** its run after the prefix holds no digit and no uppercase letter (`_redaction._is_word`). So `task-specification`, `disk-quota-exceeded` and a localized vendor sentence keep their words, and every other match is redacted. The test is deliberately not a left boundary: a key glued to a letter or digit is not a shape any vendor issues, but it is one a serialization produces (`Bearer%20sk-…`, `key%253Dsk-…`, `\nsk-…`, `\u0022sk-…`), as do an ANSI color code, a script written without spaces (`密钥sk-…`) and a header built without its space (`Bearersk-…`), and a boundary is a list of the contexts in which redaction stops. A random key of a vendor's length holds a digit or a capital with overwhelming certainty. Two residuals are stated: a glued word whose run holds a digit or a capital is still redacted (the scrub errs toward redaction), and a glued key made only of lowercase letters, `_` and `-` is kept. `tests/test_log_value_cut.py` pins both sides: the word cases survive whole, and every prefixed sample is redacted after each of 23 left neighbors. The non-prefix rules (the heartbeat URL, bearer, key-in-query, `api_key=`) are redacted wherever they match.

**A narrow run never swallows the head of a key glued after it** (issue #858). A run whose class lacks `_` or `-` (`gh?_`, `github_pat_`, `bc_uat_`, `bc_isk_`, `hf_`, `r8_`, `ste-`, `abl_`, `pplx-`) took the letters of a following key's prefix and stopped at its `_` or `-`: `hf_…aghp_ABC…` matched `hf_…agh`, and the rest of the GitHub token matched nothing, nor did all of it when the first match was kept as a word. So each such rule is two alternatives, tried in order: the run ends, as early as it can, where a whole key by its own rule begins (any rule's prefix with that rule's minimum run), and then takes everything after it in `[A-Za-z0-9_-]`, so a third key glued on goes with the second; else the run matches as it always did. It is a negative lookahead written without one, because the box's `ai_scrub` is VRL, whose regex has no lookaround. A match that took a glued key is never kept as a word, since its tail can hold a later key no word rule ever kept (`xhf_…ghp_…-sk-…`), so nothing is redacted less than before: a match only grows, and what it grows over is redacted with it. The cost, only in the glued case, is that a `-word` glued after the second key goes with it, and so does a lowercase word that held a whole key's shape. The box captures the glued alternative to know it, as `_redaction._rule` captures `glued<n>`. A wide run (`sk-`, `xai-`, `bc_???_`, `AIza`, `tk_`) already took a glued key whole. The pattern, as `_redaction._rule` builds it, for a narrow `run` with minimum `n`:

```
PREFIX(?:(?P<glued>RUN{n,}?(?:GLUED)[A-Za-z0-9_-]*)|RUN{n,})   (a match with glued set is never a word)
GLUED = the alternation, over every prefix rule, of PREFIX RUN{n}   (each rule's own run, exactly its minimum)
```

**A credential-named assignment loses its value, whatever it is** (issue #918, basecradle#877). The NOC's log-store canary pages on its `credential_assignment` rule, `[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD|PASSPHRASE)(_BASE)?=[^ ",}]{8,}` (basecradle-noc `observability/log-canary.json`), the one rule that needs no vendor's shape; widened by `@origin`'s decision on basecradle-noc#1547 (2026-11-17, issue #927), so a bare `PASSWORD=` is a name and `_BASE` is the one tail after the word (`SECRET_KEY_BASE=`, the Rails master secret's name), while `GH_TOKEN_FILE=` and `GH_TOKENS=` name nothing; a tool failure whose `error=` quoted `export GH_TOKEN="$(gh-app-token …)"` was such a line. So the source takes "names a credential" from that rule, never from a list of its own: `tests/conftest.py` holds the pattern byte for byte as `NOC_CREDENTIAL_ASSIGNMENT`, and `tests/test_credential_assignment.py` holds `_redaction.CREDENTIAL_WORDS`, `CREDENTIAL_TAIL`, `CANARY_VALUE_STOPS` and `CANARY_VALUE_FLOOR` to its spelling, holds `CREDENTIAL_NAME` to ending at exactly its `=`s, and uses it as the oracle every scrubbed text is checked against. The name stays and the value becomes `'<a redacted value>'`, the router's marker byte for byte (basecradle-router#491, `logfmt.redact_credentials`), so both writers on a box leave the same bytes; the scrub's `[redacted]` would not do, being ten characters the canary reads as a value. **The value is read as the router reads it, byte for byte** (issue #929): the harness carries basecradle-router's `logfmt._shell_word_end` and `logfmt._value_end` (basecradle-router#506 and #510, the latter merged as `a8e8f5f`, issue #934), `tests/router_reading.py` holds the router's functions verbatim, and `tests/test_credential_assignment.py` holds `_redaction.redact_credentials` and `_value_end` equal to them on fuzzed text, at every start. The value is every shell word it reaches from the character after the `=`, and the canary's own run past them, so whatever the canary would read as the value goes with it (`GET /v1?ACCESS_TOKEN=…&x=1` loses `&x=1` too). A shell word takes a quoted string, an ANSI-C `$'…'`, a `$(…)`, `${…}`, `(…)` or backquoted command whole (a quote still open at a record break ends there), and ends at a blank outside every quote. A value opening with a marker that a blank, a stop or a record break follows is that marker alone, so a value redacted twice stays one marker. From a word, the value reaches on:

1. **Across a tab** to the next word, while the word is a lead word: one the canary's run crosses whole, made of bare characters, backslash escapes, single-quoted and backquoted pieces and a closed, unnested `(…)`, never a `"`, an unclosed quote, a blank or a stop (basecradle-router#502). The canary's run crosses a tab, so a word after one is still the value's. And wherever a value would end just after a scheme word a tab glues to it (`(a)<TAB>Bearer tok`), its token goes with it, whatever stands before the tab: a value never ends between a scheme word and its token.
2. **Across blanks** to the next word, while the word ends in a bearer scheme word (issue #926, basecradle-router#497): `bearer` in any case, glued to anything before it, the scheme word of the bearer rule above (basecradle-router#510, issue #934), so a word is one when it ends in those six letters whatever comes before them (`KBearer`, `abc-bearer`, `x"q"bearer`, `(a)Bearer`). So `AUTH_TOKEN=Bearer eyJ…`, `X_API_KEY=KBearer eyJ…`, `A_TOKEN=Bearer Bearer eyJ…`, `AUTH_TOKEN=Bearer "eyJ…"` and `A_TOKEN=abc-bearer eyJ…` each lose the token, whatever its length or quoting; only a scheme word with nothing after it on its line is the value alone. Until 0.198.3 a word glued to the letters by anything but token characters (`abc-bearer` was one, by HTTP's spelling) was no scheme word, and the token after it shipped. The same end of a value, just after a scheme word, is where a tab-glued one is read (rule 1).
3. **Into a value of its own** at a credential name's `=` standing outside every quote and substitution in the word (`AUTH_TOKEN=x,DB_PASSWORD=<TAB>…`), since the next search starts past that `=`: that name's value is read by the same reading and goes with this one. **A name inside a quote is the quote's text**, so `A_TOKEN='B_TOKEN=x' next=1 other=2` keeps `next=1 other=2`, where the harness once read `B_TOKEN=` as a name, opened a quote at its value's `'` and took the rest of the line. A name the reading never reaches is its own (`A_TOKEN=Bearer }DB_PASSWORD=…` keeps `DB_PASSWORD=` and loses its value).

Each word is read once and the canary's run once from each place the value goes on past it, so the reading is linear with no budget. `_redaction.redact_credentials` is right called alone, as the router's is. An empty value is left as it is. `bounded` treats an assignment as one token: one whose value the cut would reach goes whole, name and all, one it does not reach is kept, with the blank after it so the `…` is never glued to its marker, because the formatter's second scrub would otherwise read a cut marker or a `…` glued to an `=` as a value and rebuild a whole marker past the bound. The assignments it reads are the redactor's, found as the redactor finds them, so a name inside another's value never stands for that value, and it reads the head once: `bounded("A_TOKEN=" * 40_000, cap)` at a cap one under its length took 7.3 seconds before 0.198.1 and takes about a millisecond. The canary's name pattern is quadratic in a backtracking engine on a run of name characters, and Python's `re` is one, so `CREDENTIAL_NAME` is a linear spelling, which starts only where a run of name characters starts: 200,000 capitals scrub in about 15 milliseconds. No field of this contract is touched: every name here is lowercase, and the canary's are capitals (`test_no_field_of_the_contract_is_a_credentials_name`, and every example line on this page scrubs to itself).

**Every rule reads the text as it came, and the scrub takes the union of what they take** (issue #929). The source's rules are the heartbeat URL, the rules anchored on a literal of their own (an `x-api-key` header, key-in-query, `api_key=`, a credential-named JSON member, bearer), the credential-named assignments and the prefix rules; each finds its spans on the text `scrub` was given, the spans are merged where they overlap or touch, and each merged span is written once: as the marker when it starts where a credential name's value starts, the name kept, and as `[redacted]` otherwise. No rule is ever shown another's `[redacted]`, so no rule takes less beside the others than it takes alone, and `scrub` takes everything `redact_credentials` takes. Until 0.198.1 the rules ran as four passes, each over the last one's output, and the order of the passes was a source of leaks that no ordering removes, because each pass read a text the one before had rewritten: `Bearer AB_TOKEN=secretsecret` became `[redacted]=secretsecret`, the name gone before the assignment pass could read it; the bearer rule took `Bearer x.Bearer` from `A_TOKEN=Bearer x.Bearer 'eyJ…'` and the assignment's value ended at its `[redacted]`, with the quoted token left after it; and `A_TOKEN=='\X_API_KEY=x-api-key: eyJ…` had its inner marker written inside the value's open quote, whose own quotes closed it early and left `eyJ…` outside every value. Each now ships nothing. The cases the passes were ordered for still hold: an assignment whose value is another rule's label (`GH_TOKEN=x-api-key: K…`, whose shell word is `x-api-key:`) loses the header's value with its own, and a prefix run glued before a name (`xsk-…GH_TOKEN=…`) never takes the name's value with it to no rule. In one alternation the leftmost match wins, and a prefix rule's run is made of the characters `api-key` and `bearer` are, so `task-runner-api-key=abc123secret` was once consumed as the run `sk-runner-api-key` and its value matched nothing; read apart, both rules' spans are kept. **A literal label ending at the `=` of a credential's name keeps it** and writes the marker (`OPENAI_API_KEY='<a redacted value>'`, `?ACCESS_TOKEN='<a redacted value>'`, and now `X-API-KEY='<a redacted value>'` and `?ACCESS-TOKEN='<a redacted value>'`, whose hyphenated labels end in the names `KEY=` and `TOKEN=`, and went with their values while a marker could land inside an enclosing quote); any other label (`api_key=`, `?token=`) goes with its value as `[redacted]`. A marker whose span another rule ended where a value would not end (`OPENAI_API_KEY=x-api-key: `$'sk-)`, the header's value stopping at the `'`) takes what reading the marker again would, so a marker the scrub writes reads back as itself. An `api_key=` value may open with a `"` (`API_KEY="…"`, which once matched nothing), and a bearer header glued to an `api_key=`, query or `x-api-key` value (`api_key=KBearer eyJ…`) is taken with its token; each such value is two alternatives, the first ending in a glued `bearer` and its token, written without lookaround for the box.

**The `api_key=` rule, as built** (issue #934). The label is `api_key=` in any case, with `-` or nothing for the `_`. The value may open with a `"`, or with an escaped `"` at any depth (`\"`, and `\\\"` for JSON inside a JSON string: an odd run of backslashes; after an even run the quote is a real one, the backslashes escaping each other, as in `{"h":"api_key=\\","k":"v"}`, and it is never taken as an opener), and then with bearer scheme words; it is a glued `bearer` and a token of eight or more `[A-Za-z0-9._~+/-]`, or else a run of anything but a blank, a `"` or a `&`. **A value that opened with an escaped quote ends at its escaped close**: the run still stops at the `"`, and the backslashes that escape it stay with it (`_redaction._literal_end`). So `api_key=\"K9…\" n` scrubs to `[redacted]\" n` and `{"h":"api_key=\"K9…\""}` to `{"h":"[redacted]\""}`, which still parses, where 0.198.2 took the opener's backslash alone, shipped the key and left a line that no longer parsed. **Whatever the value opened with, an odd run of backslashes before the quote it stops at is left whole** (issue #937). The run escapes that quote, and how much of it is the quote's escape and how much the value's own escaped backslashes depends on how deeply the quote is nested, which no reading of the run can tell; every JSON string the line held parses as it did only while the run stays, so it stays, and a backslash is never a key's. Until 0.198.3 a value with no escaped opener took the run, and with it the escape of the line's own closing quote: `{"log": "{\"h\":\"api_key=x\\\\\",\"k\":\"v\"}"}` scrubbed to `{"log": "{\"h\":\"[redacted]",\"k\":\"v\"}"}`, whose outer string closes early, the same on 0.198.2; it now scrubs to `{"log": "{\"h\":\"[redacted]\\\\\",\"k\":\"v\"}"}`, which parses at both depths. After an even run the quote is a real one and the backslashes, the value's own, go with it (`{"h":"api_key=x\\","k":"v"}` scrubs to `{"h":"[redacted]","k":"v"}`); the escaped opener of #934 left a backslash before such a quote and broke the line (`{"h":"api_key=\"K9\\","k":"v"}`), and no longer does. Nested, a value that is backslashes alone reaches the line as an odd run before its real close, which is read as an escaped opener; where structure and the next string follow it, it opens nothing (issue #939, below) and every string keeps its close, where until then that inner string lost it. **Every literal rule reads on through a label its value ends in** (`_redaction._literal_spans`): a value stops at a blank, a quote or an `&`, and the next value's label can stand just before that stop, so the value ends in it and the value after it went to no rule (`api_key=x,api_key="eyJ…"`, whose value is `x,api_key=` up to the quote, and `api-key="Bearer api_key="eyJ…`, each of which shipped `eyJ…`). The span now ends where that label's own reading does. A label can read further than the value it stands in at two places only, and each is searched: where the value stops (before the backslashes that escape the stop, and before the one scheme word a value may open with: `api_key=x,api_key=Bearer abc`), and just before the glued scheme word a value ended in, whose run past the token is wider than the token (`api_key=\"api_key=bearer<TAB>X-API-KEY=`). A label anywhere else in the value reads to the same stop and leaves the same odd run before it, so it ends where the value does; until #937 a label inside a value that opened with an escaped quote read to the stop backslashes and all, and its span took the escape (`api_key=\"api-key=K9…\" n` scrubbed to `[redacted]" n`, as 0.198.2 did; it now scrubs to `[redacted]\" n`). The search reaches a bounded distance before each, so the rule stays linear. The `x-api-key` header's read-on (issue #929) and the query rule are the same piece, and the query rule gains one case by it (`?token=x?token=Bearer abc` loses `abc`).

**No `api_key=` or `x-api-key` value opens at the close of the string it sits in** (issue #939, `_redaction._closes`). Both values may open with a `"`, so where the label ends a JSON string, the string's own close was read as the opener and the value ran on into the line's structure: `{"h":"api_key=","k":"v"}` scrubbed to `{"h":"[redacted]"k":"v"}`, `{"h": "x-api-key: ", "k": "v"}` to `{"h": "[redacted] "k": "v"}`, and a value read on into an inner label at such a close, `{"msg": "x,Api-Key=apikey=", "k": "v"}`, to `{"msg": "x,[redacted] "k": "v"}`. JSON follows a close with structure (a `,`, a `:`, a `}` or a `]`) at every depth of nesting, so a value opened by a `"`, escaped or not, that is structure and nothing else before any backslashes escaping the quote it stops at is no value, and the match is dropped. The first two now scrub to themselves and the third to `{"msg": "x,[redacted]", "k": "v"}`. A value that is structure and more (`api_key=",K9…"`) is still a value, and a `'` opener still opens, since a `'` closes no JSON string. **The value is read to where its run stops** (a quote, a blank or a `&`, and for a header a backslash too), so the drop applies where structure and the next string's quote follow the close, or a blank, or the end: the next member or element is a string, or the close ends the line's objects. Before a number, a literal, an object or an array, the structure runs on into that value and the opener still opens, so `["api_key=",1]`, `{"api_key=":1}` and the same nested (`{"log":"[\"api_key=\\\\\",2]"}`) still lose the string's close, as on 0.198.3. The even-run reading's limit (above) is narrowed by the same measure, not ended.

Four readings still take a string's close, and each is stated, not changed:

- **A key in a query.** The query rule's value is a run of anything but a blank or a `&`, so it takes a `"`, and `{"h":"url?api_key=K9…","k":"v"}` scrubs to `{"h":"url[redacted]`, the commonest shape a scrubbed line stops parsing in. Two alternatives were measured on an adversarial differential of 400,000 texts (keys planted after any piece, doubled CSV quotes, escapes at every depth, most nested in JSON). Stopping the run at every `"` kept 9,793 keys that 0.198.3 took. Stopping it only at a `"` that structure follows still kept 1,780. What each kept included a key in its own value position, in the line `csv.writer` writes (`GET,"https://…?token=""K9…""",200`), and the next JSON member after the string's close. A scrub that over-reads loses a field, and one that under-reads loses a secret. Nothing in the fleet parses JSON inside a harness message as a field (basecradle-noc#1583), so the rule stays, and the question reopens when such a reader exists.
- **A credential-named assignment reads its value as a shell word**, the fleet's one reading, held equal to the router's (`tests/router_reading.py`), and a shell word spans quotes. Inside a JSON string the two readings of that quote cannot both be right, and the safe one is the one that takes more, so `{"h":"A_TOKEN=x,api_key=\"K9…\"","k":"v"}` scrubs to `{"h":"A_TOKEN='<a redacted value>'`, the rest of the line with the value. No key ships.
- **A span written as the marker is read again from where it ends** (`_redaction._past_marker`), as that same shell word, so the marker reads back as itself, and where the span ends at a quote's escape or an `&` inside a JSON string, that reading crosses the close the same way.
- **The `x-api-key` label's own quote and an empty bearer header** (issue #941). The label's quote may be the close of the string the name ends, and the next member's name is then read as the value: `{"h":"see x-api-key","k":"v"}` scrubs to `{"h":"see [redacted]":"v"}`. A bearer token may open with an escaped quote that is a close: `{"log":"{\"h\":\"Bearer \",\"k\":\"v\"}"}` scrubs to `{"log":"{\"h\":\"[redacted]`. Dropping either the way `_closes` drops an opener kept 222 keys that 0.198.3 took, over the same adversarial differential with the label and the scheme words added to its pieces. A header value followed by an escaped close and a `:` (`x-api-key: K9…\": …`) is one, so both stay.

**What dropping an opener costs.** A key that 0.198.3 reached only by taking the dropped opener is kept: where a credential-named assignment's value holds an `x-api-key` header whose opener is such a close, 0.198.3 read the marker on through that opener to what followed, and the span now ends at the blank before it. `A_TOKEN=x-api-key: "]\K9…` scrubs to `A_TOKEN='<a redacted value>' "]\K9…`, where 0.198.3 took `"]\K9…`. Over the adversarial differential that is 4 keys in 400,000 texts (and 1 in 400,000 with #941's pieces added), each in this shape; an independent fuzzer built by the review, with keys planted after any piece, kept 38 in 800,000, every one in this shape too, so the rate is the generator's and the shape is the finding. A dropped opener also moves where the marker is read again, so a credential-named assignment holding a dropped `x-api-key` or `api_key=` value newly breaks 1 line of those 400,000 and 2 of the 1,000,000 below (6 in the review's 300,000), the read taking more, never less. The issue's own differential, regenerated exactly (1,000,000 texts, its seed), finds no key kept that 0.198.3 took, no line newly broken and no nested string lost. Of the 847 lines that 0.198.2 left parseable and 0.198.3 did not, 54 parse again, 784 break on a credential-named assignment's reading alone and 9 on the marker read again. (The issue's "258 through `?token=`" each held a credential name, `APIKEY=`, `X-API-KEY=` or `OPENAI_API_KEY=`, whose reading breaks the line alone.) A wider differential of 1,000,000 texts, built from every literal label, quoted and escaped openers at three depths, credential names, scheme words and JSON framing, keeps no key that 0.198.3 took and brings the JSON lines that do not parse from 305,431 to 290,395.

**The box opens an `api_key=` value at a string's close too, and that is its stated limit.** `ai_scrub`'s `api[-_]?key=` rule, which is also its only query rule (it carries none for `token=` or `key=`), lets a value open with a `"`, so `{"h":"api_key=","k":"v"}` loses its close on the box, as it did at the source until this change. A line the harness writes is scrubbed at the source first.

**The box takes neither shape, and that is its stated limit.** `ai_scrub`'s rule (basecradle-router `deploy/hosts/*/vector.yaml`) is `(?i)api[-_]?key=` and a value that may open with a `"`, never an escaped one, so on `api_key=\"K9…\"` its run takes the backslash and stops at the quote, and the key ships, as it did at the source until 0.198.3. **Its run takes the backslash before the quote it stops at the same way** (issue #937): on `{"log": "{\"h\":\"api_key=x\\\\\",\"k\":\"v\"}"}` it takes the escape of the line's own closing quote, and the line no longer parses, as it did at the source until 0.198.3. A run that never ends in a backslash, `(?:[^&\s"\\]|\\+[^&\s"\\])+`, leaves the run before the quote whole without lookaround, an even run too, which breaks nothing. It is a regular expression replaced leftmost-first in the router's first pass, a pass of rules run one after another, so `api_key=x,api_key="eyJ…"` is one match ending at the quote and `eyJ…` is matched by nothing after it. The source is the only layer that takes them until that pass becomes one alternation; a line the harness writes is scrubbed at the source first, so only a line it never wrote reaches the box with either shape.

**A credential-named JSON member loses its string value, for an exact list of names** (issue #943, `@origin`, 2026-11-19; `_redaction._MEMBER`, `_string_close`). Until 0.198.4 no rule read a JSON member, so a shapeless secret in one shipped on every version of every scrub in the fleet (`{"api_key":"K9…"}`, `{"password":"…"}`): only a secret with a vendor's prefix was caught, by the prefix rules. What the source reads, exactly:

1. **The name** is exactly `api_key`, `apikey`, `access_token`, `refresh_token`, `client_secret`, `password`, `secret` or `authorization`, in any case, with `-`, `_` or nothing between the words: `(?i:api[-_]?key|access[-_]?token|refresh[-_]?token|client[-_]?secret|password|secret|authorization)`, the whole name between its quotes, so `x_api_key`, `my_password` and `passwords` are not on it. **Never a bare `key` or `token`**, by his decision: those names are everywhere and almost never a secret, and a rule that blanks them costs more telemetry than it saves. `max_tokens` and `total_tokens` are untouched.
2. **The member** is the name's opening quote, the name, its closing quote, JSON's blanks, a `:`, blanks, and the value's opening quote, every one of the three quotes escaped by the same run: none, or an odd run of backslashes (`_ESCAPE`), so `{"log": "{\"password\": \"…\"}"}` is read at every depth. The run before the name's opening quote is taken whole, never from inside a longer one, so a quote after an even run, a real one, never opens a name. Only `"` quotes a member: a Python `repr`'s `{'password': '…'}` is no JSON member and is not read.
3. **The value** is read from its opener to its close, and only a string is: a number, `null`, `true`, an object or an array after the `:` is left alone. A string opened by a run of `e` backslashes closes at the first quote after a run of `r ≥ e` with `(r − e) mod (2e + 2) = 0`: one encoding pass turns a run of `n` before a quote into `2n + 1`, so at depth `d` (`e = 2^d − 1`) the close is a quote after `e` plus any count of the value's own escaped backslashes, and a quote inside the value is a run of `2^d` more, which never is one. The close's own `e` backslashes and its quote stay; the value's escaped backslashes before them go with it. A value that never closes runs to the first control character (U+0000 to U+001F, which no JSON string holds raw) or to the end of the text: a reading that takes more costs a field, and one that takes less a secret.
4. **It is written `[redacted]`**, the name and both of the value's quotes kept, so `{"id": 7, "password": "p\"w\\", "k": "v"}` scrubs to `{"id": 7, "password": "[redacted]", "k": "v"}`, and nested one and two deep every enclosing string still parses back to that object. An empty value is left as it is. The span joins the union like every other rule's (above), and `bounded` treats a member up to its value's opener as a literal label (issue #931): a cut inside the value or at its close goes back before the member.
5. **Each search goes on past the last value's close**, so a name inside a value is that value's text, and a closing quote never opens the next name. A close that did (`"Secret" : "\\"password":"}`, which is no JSON) was read as a name once the first scrub had taken the backslashes before it, and a second scrub then took more.

**What it costs, and what it does not cover.** A differential of 800,000 texts against `v0.198.3` (four seeds), built from every literal label, every credential name and member name on and off the list, at three escape depths, quotes and escapes, scheme words, markers, record breaks, and keys (shapeless and shaped) planted after any piece, half of them in real JSON nested up to three deep with non-string values: the new rule changes 548,154 of them and takes a key 0.198.3 left in 217,930. **It keeps 24 keys in 22 texts that 0.198.3 took, and every one is a key 0.198.3 took only by reading its marker again** (`_past_marker`; with that read turned off, 0.198.3 keeps each of them too): a credential-named assignment inside a member's value, whose marker 0.198.3 read on past the member's close to text no rule reads as a secret. Now the member's span starts the merged span before the name, so it is written `[redacted]`, never the marker, and ends at the string's close, as the line's parsing requires. `{"apikey":"\"total_tokens\":\"OPENAI_API_KEY=bearer\teyJ…x-api-key: \"key\":\"","key":{"k":"eyJ…"}}` scrubs to `{"apikey":"[redacted]","key":{"k":"eyJ…"}}`, where 0.198.3 wrote `{"apikey":"\"total_tokens\":\"OPENAI_API_KEY='<a redacted value>'` and the line no longer parsed; the `eyJ…` kept is a bare token in an object, in no credential's value and of no rule's shape. It is the class 0.198.3 accepted for the `x-api-key` label under #935 (above). Reading the marker on for every merged span that holds a credential's value instead changes how the existing rules read (it takes the `tail` that #935 keeps), and still kept one key in 200,000, so it was not done. **A second scrub can still take more, and only toward redaction**, in two shapes: where another rule's span crosses a member's close, the second scrub reads the member on to the next quote (the bearer rule's escaped token is the rest of its word, issue #941, so `"Authorization": "Bearer \"eyJ…\"", "x": 1` scrubs to `"Authorization": "[redacted] "x": 1`, and a second scrub takes the blank); and where a value's own text holds member-shaped text whose outer member another rule's span erases, which makes the inner name visible (15 in 100,000 adversarial lines of valid JSON). **The box does not read a member yet**: the same rule is in flight for the fleet's scrub block (basecradle-router#516).

A differential fuzz of 1,000,000 texts against 0.198.2 (issue #934), built from every literal label in its spellings, credential names, plain, escaped and twice-escaped quotes, scheme words glued to anything, JSON framing, markers, record breaks and fabricated keys, finds the literal rules never take a span smaller than 0.198.2's, and never leave a key 0.198.2 redacted. In 14 texts fewer characters are redacted, none of them a key: a literal span that now starts earlier swallows a credential name's value, so the union is written `[redacted]` rather than the marker, and the marker's re-read growth, which exists only so a written marker reads back as itself, no longer takes the text glued after it. The re-pin of the router's reading accounts for the rest: in 61 texts the router's reading at `a8e8f5f` ends a value earlier than at `153fb58`, because a word that now ends in a scheme word reads on to the word after it and so closes or opens a quote differently, and the scrub leaves exactly the router's bytes there. One of them leaves a fabricated shapeless key: a literal marker glued to `bearer`, whose token's quote closes at the marker's own opening quote (`X-API-KEY='<a redacted value>'bearer\t?token='… APIKEY='<a redacted value>'K9…`), and the router leaves the same.

A differential fuzz of 160,000 texts built from every literal label, credential names, scheme-word runs, quoted and escaped tokens, tabs, markers, record breaks and real key shapes (120,000 against the scrub before it, 40,000 against 0.198.0's) finds none in which 0.198.1's scrub leaves a shaped secret (a vendor key, a bearer token, an `x-api-key` header value) that the older scrub redacted, none in which it leaves a secret the router's redactor removes, and no record the canary reads. Some shapeless text (a string with no vendor's shape, standing in no credential's value by the router's reading) is kept that the old passes took, and it is stated rather than hidden: the old reading took the text after a name inside a quote, the text after a marker that a stop follows, a nested name's value read on across a record break, and text a rewritten quote happened to enclose. Each is now read as the router reads it, which leaves the same bytes. A differential fuzz of 600,000 texts built from every prefix rule's real shapes, glued and separated by every literal rule's label, found none 0.186.0's scrub leaves more of than 0.185.0's, and none it leaves any of.

**A heartbeat URL is redacted from its scheme when it has one** (basecradle-router#398). The rule is the box's, `(?:https?://\S*)?/heartbeat/\S+`: a URL with a scheme from the scheme on, so userinfo before the host (`https://john:…@host/…/heartbeat/…`) never ships; one without (`curl uptime.example.com/api/v1/heartbeat/…`) from `/heartbeat/` on, keeping its host, which is not the secret. On the box it runs first, because a rule that matched first and stopped inside the URL (a bearer run stops at the scheme's `:`) would leave the userinfo to the scheme-less arm alone; at the source every rule reads the text as it came, so no rule can stop inside it. One addition to the box's bytes: a bearer scheme word ending the URL's run, not glued to a letter or digit, is kept for the bearer rule (`_redaction._TRAILING_BEARER`), because taken with the URL it left its token in the next run with no rule to match it (compact JSON `…/heartbeat/K9","auth":"Bearer eyJ…`, a query value `…&access_token=Bearer eyJ…`).

Every source rule is linear in its input, because a value is scrubbed before it is cut and can be as long as a response body. The box's heartbeat pattern is not: in Python's `re` every `https?://` in a space-free run scans to the run's end before it fails, quadratic in the run. So the source matches each space-free run holding `/heartbeat/` once, from the run's start, and cuts it where the box's pattern would begin (`_redaction._HEARTBEAT_RUN`, `_heartbeat`); `tests/test_log_value_cut.py` holds the two to the same bytes over random texts.

**A prose line is scrubbed whole, by the handler** (issue #857). `kv` scrubs the values it renders, but a prose line interpolates its own arguments (`_log.warning("…: %s", exc)`), and an exception's text is foreign: a vendor's 401 can echo the key it rejected. So every `basecradle-harness-*` command logs through `_observability.log_handler`, whose `ScrubbingFormatter` runs `scrub` over each record as rendered — message, traceback and stack — which covers every prose site, a future one, and another library's line (`httpx`, an SDK) alike, with nothing for a call site to remember. The wake and the cleanup install it at `HARNESS_LOG_LEVEL` in `LOG_FORMAT`; the commands that keep Python's default logging install it in that default's place (`install_log_handler`: `WARNING` and up, the bare message), so their stderr reads as it did. A clean line renders byte for byte as before. A `kv` line is scrubbed a second time, which changes nothing a value already lost, and **never across a value's closing quote**: the formatter scrubs between closing quotes (`_observability.scrub_line`, the second, fourth, … `"` of the line, since every `"` a rendered record holds is the harness's own), because a quoted value ending in `/heartbeat/` or `?token=` holds no secret, and on the whole line the rule after it would take the quote and leave foreign text outside it. A traceback or stack is scrubbed apart from the message for the same reason (issue #912). One over-redaction is stated: a bare value ending in `bearer` beside a next field of eight or more characters goes with it, as it does on the box. An entrypoint's last word to stderr (`basecradle-harness-wake: …`, and every other command's) is scrubbed the same way. **The floor protects only a process whose handler the harness installed**: a library caller that configures its own handler gets `kv`'s scrub and the one prose site that scrubs its own argument (`_openai.ModelCatalog.rows`, the model list the first refuseless wake reads), and can install `ScrubbingFormatter` itself. In a wake, a traceback Python prints for an exception nothing caught passes the handler too (`_journal.route_crashes`, issue #912); in the other commands it never does, and is the box's. `tests/test_prose_scrub.py` proves it: a sentinel key driven through each prose site's exception path never reaches the journal, every console script configures the scrubbing handler, every `basicConfig` goes through `log_handler`, no module attaches a handler, configures logging another way or stops a logger propagating, and every direct write of an exception's text (`print`, `sys.stderr.write`, `warnings.warn`, a formatted traceback) scrubs it. The guards read the source, not its data flow: an error's text assigned to a name and written later is a reviewer's to catch.

**A cut never falls inside a token.** A value is scrubbed whole (`_redaction.scrub`), then bounded (`_redaction.bounded`, 240 characters in `kv`); the cut backs off to the last character outside a `[A-Za-z0-9_-]` run, so a token is either whole, and so redactable by every rule after it, or absent. That run is the character class of every length-gated rule on the box, so a credential cut short of its rule's minimum would match no rule there and ship as a prefix. The source's own rules never see a cut text at all, because every log-bound site scrubs before it cuts. A value that is one such run past the bound is omitted. **Nor does a cut leave a value open for the formatter's second scrub to read the `…` into** (issue #931): a credential-named assignment the cut reaches goes whole, name and all (see above), and so does a literal rule's label — `?token=`, `api_key=`, an `x-api-key` header, a heartbeat URL — when the head ends past it and inside what its value may take. `see ?token= tail` cut to twelve characters once read `see ?token=…`, which the second scrub wrote as `see [redacted]`, fourteen characters, past the bound and with no `…` to say it was cut; it reads `see…`. A blank between the label and the `…` would not do for every rule, since an `x-api-key` value and a bearer token are read across blanks. The router's `ai_otel_llm` remap cuts by the same rule (basecradle-router#385). Every site that truncates free text bound for a log line uses `bounded`: `kv`, the direct-message tool's ntfy refusal, the log-grammar probe's `journalctl` error, and the exception texts that reach `error=` (the Vertex and Anthropic inline-media refusals, the prune watermark's read error); so does the stall note's quoted vendor error, which is posted rather than logged.

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
| Success | `wake start`, `wake continuing`, `wake billing_recovered`, `Wake breaker RESET`, `palace scrub` (an apply that found nothing to remove), `palace prune` (a dry run that offered no group, an apply or a restore that skipped nothing, a discard) | `ok` |
| Bookend | `wake end` | — |
| Warning | `wake skipped`, `wake deferred`, `wake truncated_turn`, `wake stalled`, `wake billing_blocked`, `degraded`, `Wake breaker TRIPPED`, `palace scrub` (a dry run), `palace prune` (a dry run that offers a group, an apply or a restore that skipped something) | `degraded`, `retrying`, `refused`, `fallback`, `deferred` |
| Operator's Choice | `wake stopped` | `stopped` |
| Failure | `wake failed`, `wake reported_failure`, `post failed`, `cleanup failed`, `cleanup blocked`, `palace scrub failed`, `palace prune failed`, `palace scrub` (an apply that removed what the harness mined) | `error`, `failed` |

The Operator's Choice tier's members across the fleet today: `outcome=stopped` (the router's `stage=wake` line, and the harness's `wake end`), `event=wake_stopped` (the router), and the harness's `wake stopped`. The router's members of the other tiers are its own `logfmt.PALETTE`, pinned by its `tests/test_logfmt.py`; its Warning tier includes `outcome=deferred`, a wake that exited `75` and is scheduled to be tried again (basecradle-router#527, the router's half of harness issue #951). The harness's `wake end` says the same word for the same ending (issue #957): a wake that gives up on an outage while another attempt follows ends `outcome=deferred`, and so, since issue #962, does a wake whose exit `1` another of the router's quick tries follows (its `try=`, which the router logs as `event=wake_retry`, not as a deferral); only the last attempt or try, or a wake nothing follows, ends `outcome=error`. `tests/test_telemetry_contract.py` holds the harness to this section: the SGR of every tier, and the `outcome=` vocabulary in both directions.

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
| `cost` | US dollars for the call's tokens. Tool fees are never folded in when the vendor states them apart (they belong on the tool's line). | Vendor-stated (OpenRouter `usage.cost`, xAI ticks) or computed (`_<vendor>_rates.py` over the public prices file, [`prices.md`](prices.md), or refuseless's live price list, `_refuseless_pricing.py`). | Unstated and unpriceable: an unknown model, tier, or host, a price list that could not be read, or a price the list does not state (see `cost unpriced`); always on `retrying` and `failed`, which returned no usage to price. |
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

- **A harness brain call the engine gave up on** after a transient fault exhausted its retries. It replaces the prose ERROR give-up line it retired, one for one, so the page that line raised is unchanged. This is the case the ruling names. **Except while the router's next attempt follows** (issue #957, `_attempts.another_attempt_follows`): the give-up ends the wake on an outage, which exits `75` and is tried again, so the line is WARNING. A give-up the engine catches itself (the reserve summary's) never reaches the wake, no attempt follows it, and it stays ERROR on every attempt.
- **A harness second model's `config:` failure** (the reranker, the describer, search grounding), the first in a wake; a repeat in the same wake is DEBUG. A config-class fault is dead until a human acts, so it pages once. This is the second-model rule (`CLAUDE.md` → Second Models), which the ruling leaves as it stands.

Every other `failed` line is WARNING: a brain call that ended any other way (the wake's own lines carry the verdict a human acts on, `wake reported_failure` and `wake failed`), a second model's runtime-class failure, and **every failed call a Claude Code builder makes**, its auxiliary `purpose=helper` calls and their `config:` causes included, until basecradle#726's open question, whether a builder's failed call pages, is ruled. A `retrying` line is always WARNING, and so are `refused` and an unreadable answer's `fallback`. A wake that dies of a model call therefore writes two lines about two units: the call's `llm … outcome=failed timeline=… step=…`, and the wake's own `wake failed`. On an outage another of the router's attempts follows, the call's line is WARNING and the wake writes no `wake failed`; its `wake end … outcome=deferred` is the wake's line (issue #957). A resume's timeout on the harness's own budget is the turn's fault, not an outage: its wake exits `0`, so its give-up stays ERROR on every attempt (`Engine.resuming`).

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

**One line per tool use, whoever executed it** — a harness tool, an MCP server's tool, or a tool the vendor ran server-side inside a model call (`_observability.log_tool`, the only emitter). Tool calls and tool spend are both read off this one head; there is no second definition of "tool". The engine writes the line for a harness or MCP tool, with what the tool's vendor work recorded on its meter while it ran (`_observability.metered`); the engine's own refusal of a call (an unknown or withheld name) is `executor=harness`, since the harness is what answered it. An adapter writes the line for a server-side tool, one line per model call per tool, right after that call's `llm` line — or before it, when the caller holds its `llm` line back to judge the answer (`capture_llm_call`); `generation_id` is the join either way. A failed harness or MCP run writes a second, WARNING `tool` line carrying `name`, `target` when the run named one, and `error`; a failed server-side tool has a vendor code and no text, so it has no companion and its line is itself a WARNING.

**A tool that answers the model with its failure has still failed** (issue #822). Feeding a failure back is how a model recovers, so most failures reach the engine as a plain string. The tool says so through `_observability.tool_failed`, which returns the text unchanged and records the verdict on the run's meter; an MCP server says so with `isError`. The verdict is structural and never read off the result's text, because a shell command or a fetched page may begin with the word `Error`. A result cut at a size limit is not a failure: the line stays `ok` and carries `cut_at`, recorded through `tool_cut` (issue #892).

**An answer is not a tool error** (issue #892). A tool whose request reached the far end and got an answer did its work, whatever the answer says: a fetch answered `404` is `outcome=ok` with `status=404`, never `error`, and writes no WARNING. The tool records the answer through `tool_answered`. So a run of agents guessing URLs is counted by `status`, not hidden in a failure rate, and a reader filtering for warnings sees a broken fetch and nothing else. `error` stays for a request that got no answer (a timeout, a refused connection, a TLS failure) and one the harness itself refused.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `name` | The tool's name as the agent knows it (`messages`, `generate_image`, `web_search`, `web_fetch`, `code_execution`, an MCP tool's name). A server-side tool carries the harness name of the capability, not the vendor's internal block type. Its grammar is in [Identifiers](#identifiers). | The engine; an adapter for a server-side tool. | Never. |
| `executor` | Who ran it: `harness`, `mcp`, or `server` (the vendor, inside a model call). | The engine; the adapter. | Never. |
| `provider` | The vendor that ran or billed the tool's work — the brain's vendor for a server-side tool; the media or search vendor a harness tool called. | Whoever knows the vendor. | The tool reaches no billed vendor (platform tools, memory). |
| `kind` | The shape of the work, for metered media and search: `image.generate`, `image.edit`, `video.generate`, `audio.transcribe`, `search.web`, `search.grounding`. One dotted vocabulary with `charge`'s `kind`. | The tool or adapter. | Unmetered work. |
| `model` | The vendor model the tool's work ran on. | The tool or adapter. | The tool used no model. |
| `count` | Billed or executed units on this line: searches, grounding queries, fetches, executions. A server-side tool writes one line per model call per tool, with its uses counted here — every use, a failed one included, so it can exceed the units a fee was charged on. | The tool or adapter. | A harness run whose work stated no units (one image, one transcription). Always present on `executor=server`. |
| `duration` | Wall time of the run — for a media tool, the generation and the upload that follows it. | The engine. | `executor=server` — the work ran inside a model call its `llm` line already timed. |
| `outcome` | `ok`, `degraded` or `error` (see [Tool Outcomes](#tool-outcomes)). No harness or adapter path writes `degraded` on this head today: the fleet pages a person on it (basecradle-noc's *Tool Outcome Degraded*), so a shortfall nobody should be paged for, such as a cut output, is `ok` with the field that names it (`cut_at`). | The engine; the adapter from the vendor's result block. | The WARNING companion line. |
| `status` | What the far end answered the tool's request with, in the answering protocol's own vocabulary: an HTTP status code (`200`, `404`, `503`), or `nxdomain` when DNS answered that the host name does not exist. Written by `web_fetch` on every fetch whose answer it read, its final response's status after redirects (a body that fails mid-read is `outcome=error` with its fault slug and no `status`); a redirect to a host DNS says does not exist is `nxdomain` too, the server's broken redirect being its answer; the line is `outcome=ok` whatever the status. A name DNS says does not exist (`EAI_NONAME`, or glibc's `EAI_NODATA`) is `nxdomain`; a resolver that could not answer (`EAI_AGAIN` and the rest) is not an answer and is `outcome=error reason=fetch_failed`. One residual: macOS reports a host with no network at all as `EAI_NONAME`; the fleet's Linux reports it as `EAI_AGAIN`. | The tool, through `tool_answered`. | The request got no answer, or the tool records none (every tool but `web_fetch` today). |
| `cut_at` | The length, in characters, the result handed to the model was cut at: the agent got the head and a marker saying where it was cut, so the work was done and the line is `outcome=ok`. Written by the `shell` tool when a command's output passes its cap (30,000 characters by default), whether the command was still writing and was killed or had already exited, which is a race with the reader. | The tool, through `tool_cut`. | The result was not cut. |
| `target` | What the run asked the platform for, as `<kind>:<identifier>`, the identifier exactly as the call carried it (a uuid, a handle): `timeline`, `message`, `asset`, `task`, `user`, `webhook_endpoint`, `webhook_event`. On the WARNING companion only, beside `error`, because the platform's own text (`No record exists for the given UUID.`) does not say which one it could not find. Where a run makes more than one platform call, the last one it named before failing, which may not be the call that failed: an `edit_image` that read its source asset and then failed at the image vendor names that asset beside the vendor's error. | The platform tools, through `tool_target`, just before each call. | The INFO line; a run that names nothing (an engine refusal, a non-platform tool). |
| `reason` | Why it was not `ok`, as a slug. The engine's own refusals: `unknown_tool`, `withheld`, `server_side` (a server-side built-in called as a function), `elision_marker`, `invalid_arguments` (the call's arguments cannot bind to the tool's `run()` — an unknown, missing or misspelled name, or not an object; refused before the tool runs — or the tool itself refused what it was given: a required value for this action missing, an unknown action, a wrong confirm, an asset of the wrong kind or size); a tool that raised: `exception`; a harness tool's vendor call that failed: the fault slug, as on `llm` (`rate_limited`, `server_error`, `transport`, `timeout`, `invalid_response`, `provider_error`); `config:<cause>` for a setup fault only a human can fix, such as `config:missing_api_key`, `config:auth`, `config:running_as_root`, `config:prompts_not_installed`, `config:provider` (a tool that runs only beside one brain, built beside another), `config:base_url` (an `AI_BASE_URL` that is not a URL); `memory_error` (the memory search failed, so its "no matches" means nothing); `platform_error` (the BaseCradle platform refused or failed the call); `tool_error` (an MCP server's own `isError` verdict); `fetch_failed` (`web_fetch` refused the URL — not https, no host, a non-public address — or could not complete it: a redirect loop, a redirect with no location, a resolver that could not answer; a server's error status is an answer, `outcome=ok` with its `status`); `refused` (a push service's verdict on the request). A shell command's output past the cap is not on this list: it is `ok` with `cut_at` (issue #892), where it was once `error reason=output_limit`. For a server-side tool, the vendor's own code or status, lowercased (`max_uses_exceeded`, `url_retrieval_status_error`, `outcome_failed`, `failed`, `incomplete`). | Whoever saw the failure. | `outcome=ok`. |
| `error` | Common; on the WARNING companion line only. | The engine. | `outcome=ok`. |
| `cost` | US dollars of the tool's own fee — never model tokens, which are on the `llm` line. | The tool or adapter. | No fee, or a fee nobody can price (see `cost unpriced`). |
| `cost_basis` | Common. | `cost_basis()`. | As Common. |
| `cost_gap` | Why a fee the harness computes carries no `cost`, in the reason its `cost unpriced` line names, on every use it applies to (issue #880), with the meaning it has on `llm`: `unpriced_host`, the tool's client is aimed at a host whose price is not the table's; `unknown_model`, the table holds no row for the model; `not_reported`, the vendor's answer does not state what the figure needs (an image call with no usage or no input split, or text output on a model with no text-output rate; a transcription stating neither a duration nor token counts the row prices; a grounded search that did not list the queries a per-query scheme counts). The once-per-wake line cannot count the uses a gap covered; this field does. | The tools and adapters that compute a fee: `generate_image` and `edit_image` (`_images._cost`), the assets tool's `listen` (`_audio._cost`), Google's grounded search (`_google.search`), and the server-side `web_search` fee on OpenAI and Anthropic (on any host but the vendor's own). | A `cost`; a fee stated by the vendor or by nobody (xAI's media, a router's searches); a tool that bills no fee of its own. |
| `generation_id` | Common: the model call the tool ran inside, so a tool joins to its `llm` line. | The adapter. | `executor` is not `server`, or the call's response named no id. |

#### Tool Outcomes

**One dimension, carried by two heads: `outcome` on `tool` and on `charge`, with one vocabulary** (issue #822). It is what makes Tools a sibling of Wake, Helper LLM and Memory on the fleet's *Outcomes* charts. The field keeps the name every sibling head uses; `tool_outcome` is the NOC's name for the label it extracts from it.

| Outcome | Means | `reason` |
|---|---|---|
| `ok` | The work was done. The answer may still be bad news: a shell command that exited nonzero, a fetch the server answered `404`, a shell output cut at its cap (`cut_at`), or a confirm gate's preview. | Absent. |
| `degraded` | The work was done and is usable, with a shortfall the line names. | Required. |
| `error` | The work was not done, or cannot be used. | Required. |

A `charge` is on this dimension because a billed resource is tool-adjacent (see [`charge`](#charge)), and the shortfall known today is one about money: Steel's stated cost and the rate card disagreeing. It is never a `tool` line, because a session is billed by the time it runs across zero or more tool uses (@origin, 2026-11-01). A clean `charge` is not a tool use, so a chart of tool outcomes counts every `tool` line (the WARNING companion excluded, as `tool_calls` excludes it) and a `charge` only when it is not `ok`, which keeps the chart's total equal to the tool calls plus the sessions that went wrong.

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

A `container_time` or `browser_time` gap is the charge the next head would carry: the day a vendor states its container time, or a browser vendor's session time gets a rate card, that charge is written as a priced `charge kind=compute.container` or `charge kind=browser.session` line and leaves this one. **The rule for the next value:** a gap in a charge billed per call or per use is named for what is billed (`tokens`, `grounding`, `search`, `image`, `audio`); a gap in a charge billed by a quantity of a resource is `<resource>_time` (or `_bytes`, `_storage` for a resource with no run time), and the sentence above names the [`charge`](#charge) `kind` that retires it. Either way it is generic in spelling, never a vendor's name or word, and a new value is a change to this page.

### `charge`

**One line per billed resource instance**: a dollar a vendor bills by a quantity of a resource — browser time in the vendor's own billing unit (`second` on Steel's current plans, `minute` on some older ones), container hours, a 15-minute GPU block, bytes, storage — rather than per call (@origin, 2026-11-01, on basecradle-noc#1153). It is neither a tool *use* nor a model call, so it has a head of its own, and every resource-billed product fits it without a new one.

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
| `unit` | Common: the billing unit `count` is in — `second`, `minute`, `hour`, `block_15m`, `gb`, `gb_month`, `seat_month`. An open set, listed here as it grows. | The producer. | Never. |
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

The first producer is the NOC's Steel browser launcher (see [Outside This Repository](#outside-this-repository)); nothing in this repository writes a `charge` line yet. Its per-session spend line, whose `cost` is the dollar Steel states, so it carries no `cost_basis` (these examples predate Steel's move to per-second billing, so they are in `minute`; a session today writes `unit=second`):

```text
INFO charge provider=steel kind=browser.session executor=mcp unit=minute count=7 duration=402.18s started=2026-11-01T03:12:40Z resource=01a19518-4c40-7c41-9f3e-5d8c1a2b4e60 outcome=ok cost=0.07
```

A session Steel released as `browser_crashed`, billed for the time it ran:

```text
INFO charge provider=steel kind=browser.session executor=mcp unit=minute count=2 duration=71.40s started=2026-11-01T03:20:05Z resource=01a19518-5b12-7d03-8e6a-2c4f9b7d1e83 outcome=error reason=crashed cost=0.02
```

A session whose stated cost the published rates do not reproduce:

```text
INFO charge provider=steel kind=browser.session executor=mcp unit=minute count=8 duration=471.02s started=2026-11-08T14:02:11Z resource=01a19518-7e44-7a12-b3c0-6f1d8e2a9c07 outcome=degraded reason=cost_divergence cost=0.0633
```

A session on a Steel plan whose rate the launcher's table lacks writes no `charge` line, and one `cost unpriced` gap instead:

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
| `mcp_servers` | Every active MCP server, in load order, comma-separated, each by the sanitized name its tools' names begin with (`<server>__…`, grammar `[A-Za-z0-9_-]`, so no `,` inside one): `fetch,playwright,steel`. It is the journal's record, once per wake, that this agent runs beyond the safe-by-default tool set; until issue #892 that was a per-server WARNING on every wake, 116 of one run's 140 warnings. The tool count of each server is on an INFO prose line beside it. | The wake, from tool resolution. | No MCP server is active (`mcp/` is empty, or every server failed to load). |

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
| `outcome` | `ok` (every reconcile returned, or the wake held for its outage notice: `wake skipped reason=outage_notice`), `error`, `deferred` (the router runs this wake again after the exit it makes, so it is not a failure: an outage exits `75` and another of the router's attempts follows, `BASECRADLE_WAKE_ATTEMPT` below `BASECRADLE_WAKE_ATTEMPTS`, matching the router's own `stage=wake outcome=deferred`, issue #957; or any other fault exits `1` and another of the router's quick tries inside the attempt follows, `BASECRADLE_WAKE_TRY` below `BASECRADLE_WAKE_TRIES`, basecradle-router#533, issue #962), or `stopped` (the wake received SIGTERM — the router's `stop-wake` — and settled what it held; it matches the router's own `stage=wake outcome=stopped`, and is not a failure: the fleet's `wake_failed` reads `outcome=error` only). An outage on the last attempt, a fault on the last try, and either on a wake run without the variables its exit asks, is `error`. | The wake. | Never. |
| `error` | The fault the wake ended on: its text, or its type when it has none. | The wake. | `outcome` is not `deferred`, or the wake returned deferred (a resume absorbed the outage, and the item's own lines say why). |
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

A permanent, billing or outage model failure reported to the timeline (`_wake._report_provider_failure`, `_report.billing_onset_line`, `_wake.WakeAgent._gave_up`). NOC-alarmed on `kind=billing` only (*LLM Vendor Payment Failed*, basecradle-noc#317).

`kind=outage` (issue #951) is written once per outage per timeline, by the router's last attempt at a delivery (`BASECRADLE_WAKE_ATTEMPT` equal to `BASECRADLE_WAKE_ATTEMPTS`), and only when its notice posted: a temporary fault outlasted every in-call retry and the wake is giving up with its work pending. The same wake still writes `wake failed` when the fault propagated, and exits `75`. The marker holds the notice's own uuid (issue #957), so the wake the notice's delivery starts can tell it apart: on a timeline whose outage is announced, a wake that finds no peer item that no wake has tried (no message, asset or webhook event above the notice in uuid order, none past its kind's mark (or, with no mark yet, in the first wake's window) left unclaimed above the newest item a wake claimed in that stream, or in a stream where no listed item is claimed, and no activated task that no wake has claimed or recorded) does no model work, writes `wake skipped reason=outage_notice`, and exits `0`, so the notice starts no second chain of attempts. A marker with no uuid (written before #957), or a listing the platform will not give, attempts the work as before.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `kind` | `permanent`, `billing` or `outage`. | The wake. | Never. |
| `reason` | The failure's slug: `payload_too_large`, `context_length` or `out_of_funds`, and for an outage the brain's [fault slug](#fault-slugs) (`rate_limited`, `server_error`, `transport`, `timeout`, `invalid_response`, `truncated`). | The wake. | Never. |
| `provider` | Common. | The wake. | A probe line. |
| `item` | Common. | The wake. | A billing onset, or an outage (its notice names no item). |
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
| `timeline` | Common. | The entrypoint, or the wake. | Never. |
| `delivery` | Common. | The environment. | As Common. |
| `reason` | `timeline_deleted` (the entrypoint: the delivery's timeline is gone), or `outage_notice` (the wake: an outage was announced on this timeline and no peer item is there that no wake has tried, so the waiting work is not attempted, issue #957; see [`wake reported_failure`](#wake-reported_failure)). | The entrypoint, or the wake. | Never. |

### `wake failed`

A wake that ended on a fault. The entrypoint exits `75` (`EX_TEMPFAIL`) when that fault is a temporary provider fault that outlasted every in-call retry (an outage, issue #951), and `1` for every other. **An ending the router runs again writes no `wake failed`**, at any level: an outage another attempt follows (issue #957), and any other fault another quick try follows (issue #962). The wake was deferred, which its `wake end … outcome=deferred error=…` says, and for an outage the call's own `llm … outcome=failed` (WARNING) says why too. A wake whose setup failed writes no `wake end`, so on a try another follows it says its error in prose, at WARNING. It is never re-headed as `wake deferred`, the per-item line.

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
| `kind` | Who posted: `tool` (the agent, through the `messages` tool) or one of the harness's own sanctioned posts: `failure-report`, `stall-note`. | The poster. | Never. |
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

One line per memory-system operation other than a retrieval's report, which is `memory recall` (`_wake`, `_mempalace`). The wake writes its own `op=recall` and `op=observe` guard lines here when the provider's hook fails, and an INFO `op=observe result=skipped reason=marked` line for a marked real wake it did not hand to memory (issue #950).

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `op` | The operation: `recall`, `observe`, `search`, `palace-binding`. | The caller. | Never. |
| `result` | How it ended, where it did not simply succeed: `failed`, `skipped`, `unfile_failed`. | The caller. | The operation succeeded. |
| `provider` | Common: the memory software. | The caller. | The wake's own guard line, which does not know it; `op=palace-binding`. |
| `surface` | As on `llm`. | `_mempalace`. | Not a search. |
| `reason` | Why, as a slug: `marked` (a wake the NOC marked with a verified `BCNOC2`, not handed to memory); `redaction_failed` (an exchange the redaction could not run on, so not handed to memory: an exchange is never filed unredacted, issue #999); the MemPalace adapter's own slugs. | `_mempalace`; the wake, for `marked` and `redaction_failed`. | Nothing to explain. |
| `keys` | The field names of an error envelope upstream returned. | `_mempalace`. | Not that failure. |
| `type` | The type of an answer that was not a mapping. | `_mempalace`. | Not that failure. |
| `palace` | The palace path bound. | `_mempalace`. | Not a binding. |
| `config` | The MemPalace config file path. | `_mempalace`. | Not a binding. |
| `timeline` | Common. | The wake, on `reason=marked` and `reason=redaction_failed`. | Any other line. |
| `chars` | Common. | The wake. | Not the wake's guard line. |

### `secrets redacted`

**One INFO line for each text written outside the log that lost a secret to the redaction** (`_observability.log_redacted`, issue #999): an exchange filed into memory, a fact the memory tool stored, or a text file a tool saved. A text that lost none writes no line. It counts; it never carries the text, because the text around a marker would rebuild more than the marker removed. A redaction inside a log line is not counted here: it is the marker itself, which the fleet counts in the log store.

```text
INFO secrets redacted kind=exchange count=1 timeline=019f6e71-2a12-7b69-a204-0fec1497b9c2
INFO secrets redacted kind=fact count=2
INFO secrets redacted kind=capture count=1 path=.playwright-mcp/console-2026-11-23T01-02-03-456Z.log
```

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `kind` | Where the text was written: `exchange` (filed into memory by the wake), `fact` (stored by the memory tool), `capture` (a text file a tool saved). | The writer. | Never. |
| `count` | Common: the redactions written into the text. | The writer. | Never. |
| `timeline` | Common. | The wake, on `kind=exchange`. | Any other kind. |
| `path` | The capture's path under the MCP server's working directory. | `_mcp`, on `kind=capture`. | Any other kind. |

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

### `palace prune`

Written by `basecradle-harness-prune-palace` (issue #973), one line per run. Counts, ids and the plan hash only: never drawer text.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `palace` | The palace path. | `_prune`. | Never. |
| `mode` | `dry-run`, `apply`, `restore` or `discard`. | `_prune`. | Never. |
| `drawers` | Drawers in the palace, registry rows excluded. | `_prune`. | `mode` is not `dry-run`. |
| `exchanges` | Exchange files with drawers, read. | `_prune`. | `mode` is not `dry-run`. |
| `candidates` | Exchanges that pass the rule's first two clauses. | `_prune`. | `mode` is not `dry-run`. |
| `groups` | Groups offered. | `_prune`. | `mode` is not `dry-run`. |
| `members` | Exchanges in those groups. | `_prune`. | `mode` is not `dry-run`. |
| `copies` | Exchanges a default yes to every group would remove. | `_prune`. | `mode` is not `dry-run`. |
| `exceptions` | Members held back as exceptions. | `_prune`. | `mode` is not `dry-run`. |
| `plan` | The plan hash. | `_prune`. | `mode` is `restore` or `discard`. |
| `backups` | Backups kept from earlier applies. | `_prune`. | `mode` is not `dry-run`. |
| `accepted` | The group ids the agent accepted, which the apply carried out. | `_prune`. | `mode` is not `apply`. |
| `files_removed` | Exchange files removed. | `_prune`. | `mode` is not `apply`. |
| `drawers_deleted` | Drawers deleted with them. | `_prune`. | `mode` is not `apply`. |
| `skipped` | On an apply, planned files skipped because they no longer match the plan; on a restore, files and drawers left because they are already in the store. | `_prune`. | `mode` is `dry-run` or `discard`. |
| `files_restored` | Exchange files a restore put back. | `_prune`. | `mode` is not `restore`. |
| `drawers_restored` | Drawers a restore put back. | `_prune`. | `mode` is not `restore`. |
| `backup` | The backup's name: the one the apply wrote, or the one restored or discarded. | `_prune`. | `mode=dry-run`, or an apply that removed nothing. |

### `palace prune failed`

A refusal or an error; the run changed nothing in the store.

| Field | Meaning | Filled By | Absent When |
|---|---|---|---|
| `palace` | The palace path. | `_prune`. | It could not be resolved. |
| `error` | Common. | `_prune`. | Never. |

## Prose Lines

Everything that is not a structured line is **prose**: an operator's note in a sentence (`No published rate for …`, `Anthropic's prompt cache needs …`), the engine's `step n/max: … (1.23s)` ledger, a startup notice. Prose is legitimate and is not part of this contract's vocabulary, under one rule: **a fact a dashboard needs never lives only in prose.** A refusal, a server-tool failure, an unpriced charge — anything a vendor comparison would count — has a structured head here. Prose may restate it for a human; it may never be its only record.

### One Line, Opened by the Harness

**Every record is one line, and its start is the harness's own words** (issue #912, basecradle#868). The fleet recognises a head where the head stands — the record's start, after the severity (basecradle-noc#1518) — and journald starts a record at every newline of a wake's one stdout+stderr stream, so whatever controls a record's start could write a line indistinguishable from this contract's. The formatter every command logs through (`_observability.ScrubbingFormatter`) holds the line, and a wake holds its descriptors too (`_journal`). None of it adds a head or a field: each rendering below is prose.

| Record | How It Reads |
|---|---|
| A harness prose line with a foreign argument | The argument by the value rule, unbounded: flattened, scrubbed, and quoted when it holds a space, an `=` or a `"` (`WARNING Memory provider observe() failed: "connection reset by peer"`). Only what the harness composed is interpolated as written — a head, a `kv` rendering, a whole structured line (`_observability.Rendered`) — and a number; a bare token (a model slug, a uuid) reads as it did. |
| The step ledger | Its action quoted, because a tool name is whatever the model emitted: `step 1/24: "tools=survey" (0.03s)`. |
| A traceback or a stack | One quoted string at the line's end: `WARNING Memory provider observe() failed; continuing. "Traceback (most recent call last): … ValueError: …"`. |
| Another library's record | Its logger's name, then its whole message quoted: `WARNING openai._base_client: "Retrying request to /responses in 0.4 seconds"`. A logger name that is not one token (`[A-Za-z0-9_.:-]+`) is quoted too. |
| A line anything else wrote to a wake's stdout or stderr | Read off the descriptor into a record of its own: `INFO stdout: "…"`, `WARNING stderr: "…"`. A child process's output, a C library's, a block-buffered `print`. An unfinished line waits for its newline or the wake's end, and is then a whole record; it is never the start of another record. |
| An exception nothing caught, in a wake | `ERROR The wake raised an exception nothing caught. "Traceback …"`; in another thread, `ERROR Thread <name> raised an exception nothing caught. "…"`. A `KeyboardInterrupt` keeps Python's own report, which only a person at a terminal sees. |
| A record the handler cannot write | `ERROR A log record could not be written. "…"`, one line on the handler's own stream (`_observability.JournalHandler`), never Python's multi-line report to `sys.stderr`, which in a wake is the pipe the handler's own reader drains. |
| A record longer than 32 KiB | Split by the formatter, under journald's `LineMax` (48K by default): the first record keeps its own head, closed with a `"` when the cut fell inside a quoted value, and each continuation reads `continued 2/3: "…"` in the same severity. A cut never falls inside a `[A-Za-z0-9_-]` run unless that run is half the budget long. Measured in UTF-8 bytes, a lone surrogate counted as the `\udXXX` the stream writes for it. An unspoken narration, a long traceback and a long output line are what reach it. |

Three residuals are stated. An output line can land a few milliseconds out of order with the harness's records, because each descriptor is read on its own thread. A write after the wake (interpreter shutdown) goes out raw. It follows `wake end`, so the only record it could be glued to is a crash report written after it. A descendant that outlives the wake keeps its pipes open, so the wake waits one second in all for the readers and then exits without them. An output line takes its descriptor's level, so `HARNESS_LOG_LEVEL` governs it like any other record.

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

The NOC's Steel browser launcher writes its own `browser provider=` head. Its per-session spend line retires into the [`charge`](#charge) line (@origin, 2026-11-01: a session is billed by the time it runs, not per tool use, so it is not a `tool` line); its lifecycle lines stay on a NOC-only head that no column reads. It is not emitted here, so it is not witnessed here; the launcher's move is basecradle-noc#1153's. A cost shortfall is stated on the `charge` line's own `outcome` (issue #822), so the launcher's `session.ended` companion is detail, never a second verdict a chart reads. Slice 2 carried the rest to the NOC through the capital, together with the NOC columns that read what this repository retired in slice 2 (*Tool Cost* off `media`/`browser`; `web_search_requests`, `web_fetch_requests`, `code_execution_requests` off `llm`), which read the `tool` line instead.

The Claude Code builders' model calls reach this contract through the router's `ai_otel_llm` remap (basecradle-router `deploy/vector.yaml`), which renders each `api_request` as an `llm` line. Under issue #782 an `api_error` (Claude Code's final failure, after its own retries) is rendered as `llm … outcome=failed reason=<slug>` with no tokens, and an auxiliary call as `purpose=helper` with no `kind` (basecradle-router#384). Its slug is read off the status by [Fault Slugs](#fault-slugs); the remap adopts that table's `transport`, `config:auth`, `config:billing`, `config:model_not_found` and `payload_too_large` in basecradle-router's follow-up to issue #786, and until then writes `api_error` for each. Its `duration` is absent where Claude Code cannot state the whole wait (an `api_error` after more than one attempt, or with no status), and its `failed` line is WARNING. A builder's line carries no `timeline`, `delivery` or `step`: it has no wake.

## Where a Signal Lives

@origin, 2026-11-02 (the telemetry audit, basecradle#726, ruling 2): **every meaningful signal has one of two homes.**

- **A chart, with its alert,** for the aggregate somebody watches.
- **The log stream,** for the per-event detail somebody reads to troubleshoot. The fleet server's logs are where you look when an agent breaks.

"It doesn't fit a chart" never means "drop it". It means "log it". Detail specific to one producer that does not generalize into a shared chart still belongs in the stream, keyed by agent. So nothing operationally meaningful is ever deliberately left out: a signal is charted, or it is logged.

## Changing the Contract

The contract changes before the code does, never after. A change is one pull request that edits this page, the emitters, and the Known Violations table together, and it moves every adapter whose vendor has the datum — never one vendor at a time. A new field or head is a design decision about the whole fleet's telemetry: it is raised `needs-capital` first, and lands only once decided. Anything the NOC extracts is coordinated with the NOC through the capital in the same arc, so that no column goes dark.
