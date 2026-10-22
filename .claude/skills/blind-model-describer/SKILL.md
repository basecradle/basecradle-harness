---
name: blind-model-describer
description: The full design of the blind-model describer — a second model that describes pictures and clips for a text-only brain across view, watch and asset arrival — the HARNESS_DESCRIBER_* settings, why it rides the brain's stack by default (or a stack of its own) but never takes the brain's credential, routing pin or model_params.json, dead-versus-off, the graded failure classes, judging an answer (truncated, empty_response, missing_parts) and the single retry, why an unreported bill judges nothing, the explicit output budgets, the per-wake memo, the caption that names the describer, and its mining sentinel. Use before editing _describer.py, the perception paths (assets view/watch, the asset wake's arrival), the describer's prompt or labels, the output-budget spellings in _provider_from_config, or any HARNESS_DESCRIBER_* setting. The invariants live in CLAUDE.md → MemPalace Rerank and the Blind-Model Describer — Second Models; this skill carries the mechanisms and the reasoning.
---

# The Blind-Model Describer — The Full Design

The invariants live in `CLAUDE.md` → "MemPalace Rerank and the Blind-Model Describer — Second Models" and govern at all times. This skill carries the full design behind them. Read it in full before changing any of them.

A text-only brain (@glm-5.2, `input_modalities: ["text"]`) reaches the honest "described above, not
shown" tier on every picture and every clip. Honest is not working. **`HARNESS_DESCRIBER_MODEL`** —
a model id on the agent's own provider, or on the describer's own stack — makes the harness send the pixels to that model and hand
the brain its words, across all three perception paths (`view`, `watch`, and a peer's image on
arrival at the asset wake) through **one** seam (`Engine.describer`, memoized per wake), so they can
never diverge on which model describes or how.

| Var | Meaning |
|---|---|
| `HARNESS_DESCRIBER_MODEL` | The describer's model id. **Absent or empty = describer off** — byte-identical to the pre-#472 behavior, down to the log lines, with nothing imported and no adapter built. |
| `HARNESS_DESCRIBER_API_KEY` | A key **dedicated to describing on this agent** (fleet rule: one key per agent per purpose). Required when MODEL is set; **never** a fallback to `AI_API_KEY`. |
| `HARNESS_DESCRIBER_PROVIDERS` | Comma-separated OpenRouter slugs → `provider: {only: [...], allow_fallbacks: true, data_collection: "deny"}` — the same object, built by the same helper, as the reranker's. Required when MODEL is set on a key-based provider. **No default list in code**, for the reason `HARNESS_MEMPALACE_RERANK_PROVIDERS` has none (the `memory-palace` skill). |
| `HARNESS_DESCRIBER_PROVIDER` + `HARNESS_DESCRIBER_SDK` | *Optional, both or neither* (issue #655): the describer's **own stack** when it should not ride the brain's — e.g. `google` + `google-genai` for Gemini eyes on Vertex beside a brain on OpenRouter. Built by the same factory at that SDK's default surface (or `HARNESS_DESCRIBER_SDK_SURFACE`, read as `AI_SDK_SURFACE` is). It keeps the brain's `AI_BASE_URL` only when it names the brain's own provider (`inherit_endpoint`); another vendor gets its default endpoint. One without the other, or a surface without the stack, is `config:incomplete_stack`. |
| `HARNESS_DESCRIBER_CREDENTIALS_FILE` / `HARNESS_DESCRIBER_LOCATION` / `HARNESS_DESCRIBER_PROJECT` | A describer on **Vertex** (`google`) takes these instead of a key and a provider list: its own service-account key file and location (both required — `config:missing_credentials_file` / `config:missing_location`) and optionally its project (default: the one the key names). **Never** a fallback to the brain's `AI_CREDENTIALS_FILE` / `AI_LOCATION` / `AI_PROJECT`. |

**The describer shares the brain's stack and nothing else, and each half of that is load-bearing**
(by default — with `HARNESS_DESCRIBER_PROVIDER`/`_SDK` set it runs on a stack of its own and shares
only the factory; the credential, pin and `model_params.json` rule below holds either way). It is built by `_provider_from_config(..., model=, api_key=, routing=, inherit_params=False)` — the
brain's own factory — so the **SDK, surface and endpoint** cannot drift: one adapter family, one
error taxonomy. What it must **not** inherit is the brain's key (one key per agent per purpose, so a
rotated or compromised describer key never touches the brain's account) or its routing pin:
@glm-5.2's `model_params.json` pins `provider.only` to GLM hosts, and a Gemini-class describer
routed there fails **every** call with *no eligible provider*. `inherit_params=False` drops that
file whole, because it is tuning for **this agent's brain** — at best irrelevant to a different
model, at worst exactly that pin. (This is the reverse of the first draft of #472, which said "same
key, same pins"; the amendment is the law.)

Four invariants, each with an "obviously fine" broken form:

- **Off by absence, and the model id is the only switch** — no shadow mode, no `…_ENABLED`
  companion, for the same reason `_rerank.py` has none (the `memory-palace` skill). The regression bar is a test.
- **A model set without its key or its provider list is DEAD, not OFF** — `describer_from_env`
  returns a describer carrying a *config fault*, never `None`. `None` would make a half-configured
  describer indistinguishable from a deliberately blind agent, which is Green-While-Absent exactly.
- **Never a fabricated description, and the failure classes are graded** — `_rerank.py`'s two, in
  its words (the `memory-palace` skill). **Config-class** (no key, no providers, an incomplete own
  stack, a missing Vertex key file or location, a provider that would not build, 401/403, 402,
  a model id that does not exist) → withheld caption + **ERROR**, *once per wake* (the object's life
  **is** the wake, so this needs no clock; repeats drop to DEBUG so one defect cannot become a
  storm). **Runtime-class** (timeout, 429, 5xx, transport, unparseable or empty answer, an
  undecodable clip) → withheld caption + **WARNING**. A working describer logs **INFO**: the WARNING
  belongs to the degrade it replaced, and one on every successful description is how a real warning
  stops being read. Nothing here ever raises into a wake.
- **A fragment is a fabrication by omission, so the answer is *judged* and not merely received**
  (issue #488). The caption tells the brain it is reading an account of the whole thing, and a brain
  cannot notice that the sentence stopped in the middle — @glm-5.2 was handed 1,748 characters cut
  mid-word as its sight of a 5 s clip, with no `Over time:`, no `Last frame:`, and a line reading
  `outcome=ok`. Three checks now decide usability (`_unusable`), in the order that makes the *first*
  true one the cause rather than the symptom: **truncated** (the vendor's own `length`, in any of
  its three spellings — `_observability.finish_reason` is a capability read, never a vendor branch),
  **empty_response**, and **missing_parts** (a video description lacking `VIDEO_PART_LABELS`,
  matched on **presence** and never on position, because the check exists to catch a missing part
  and not to police a layout — a model told "plain prose" may well write the three labels inline in
  one paragraph, and discarding that would blind the agent to enforce a style nobody reads). Only
  **truncated** retries, once, at `RETRY_BUDGET_FACTOR` the room — and only where there is more room
  to buy, since a describer built from a caller's own `Provider` has one fixed cap and asking again
  would purchase the same answer. Two corollaries a later "consistency" pass will otherwise undo:
  the retry is keyed on the one fault a budget fixes, never on "something went wrong"; and the
  labels the check reads are the same tuple the prompt is **composed from**, so a reworded part
  cannot leave the check rejecting every answer the describer gives.
- **Every one of those checks reads what the vendor said about the *answer*; what it said about the
  *bill* judges nothing — and the corrected rule is the more interesting one** (issue #491; #488
  shipped the wrong version and it stood for two days). A fourth check, `no_usage`, condemned an
  answer that arrived beside a usage block of nothing but zeros, on the reasoning that a call which
  generated 1,700 characters cannot have consumed zero input tokens. That inference reads an
  accounting fact as a content fact, and the live re-run falsified it: on OpenRouter,
  `google/gemini-3.8-flash` reports **no usage at all** for *video* calls — while the same model,
  in the same wake, reports it for images, and the older `gemini-3.1-flash-lite` reported it for
  video. Complete, three-part descriptions were being thrown away, with `assets watch` returning
  the withheld caption whether or not a window was asked for. So an unreported bill is reported as
  **absent** and nothing more: `log_llm_call` already omits the token and cost fields rather than
  print zeros (a **non-zero** cost stated beside them still survives — honest absence cuts both
  ways), and `Describer._note_unreported_usage` writes one INFO note per wake per **model+kind**
  (the cell, not the agent — this model counts an `image.describe` and not a `video.describe`), so
  the hole in the helper spend series is explainable rather than mysterious. Two properties are
  load-bearing: the note fires on a **reported** all-zero block and never on the *absence* of one
  (an adapter that never instrumented has claimed nothing, and a note about a claim nobody made is
  noise on every wake forever) — and it is deliberately **not** on the `llm` head, because a note is
  not a call and must join no column that counts them. *This repo's own honest-absence rule, which
  #488 had pointed backwards: silence about the price is never evidence about the goods.*
- **The output budget is explicit, and it bounds two different things at once.** A vendor default is
  not a promise and can move under you; worse, a reasoning describer can spend the whole of it on
  thinking before writing a word. So a still asks for `IMAGE_OUTPUT_BUDGET` (2,048) and a clip for
  three times that — three parts, three times the room — spelled per SDK inside
  `_provider_from_config` (`max_tokens` / `max_completion_tokens` / `max_output_tokens`), where every
  other vendor spelling already lives, so the describer asks for *a budget* and never for a field
  name. Getting that name wrong is a 400 on the first call and a permanently blind agent, which is
  why it is parameterized by test. The second thing it bounds is **Context Discipline's first
  invariant**: a description rides an *injected turn*, which is persisted for the life of the
  timeline, so the cap on what the describer may generate is also the cap on what one picture costs
  the transcript forever. The adapters take the cap at construction, so a still, a clip and a retry
  are three caps and therefore three adapter instances — built lazily per budget from one closure, so
  the ordinary wake builds exactly one.
- **A zero is not a measurement** — `log_llm_call` omits the token fields *and* the cost when the
  vendor's usage block reports nothing but zeros. Printing them is worse than omitting them: the
  live line read `tokens_in=0 tokens_out=0 tokens_total=0 cached_tokens=0 cost=0 outcome=ok`, which
  on the dashboard is a free call that worked. The dollar goes with the counts because on every
  endpoint that states one it is read out of that same block (`_COST_FIELDS`) — it is not an
  independent claim, it is the same absence spelled `0`. Two things survive, and both directions
  matter: a genuinely free call reports real counts, so its `cost=0` is a fact; and a **non-zero**
  charge stated beside an unreported usage block survives too, because the native xAI adapter reads
  `cost_usd` off the *response* — swallowing that would lose real spend from the rollup to fix a
  line that was only ever wrong about zeros. This is the honest-absence rule `endpoint` and `cost`
  already keep, applied to the one shape that *looks* like an answer.
- **One describe per piece of media per wake, and only a transient failure is asked again**
  (issue #664). Every perception path (arrival, `view`, `watch`) reaches the one `Describer`, whose
  life is the wake, so the memo lives there (`Describer._once`), keyed by a digest of the payload it
  is shown plus a clip's window, and never by a uuid the media types do not carry. A repeat writes
  **no `llm` line**, because that line is one billed call; it writes a DEBUG `describe repeated`
  line with no cost. A `RETRYABLE_REASONS` failure is not remembered, since the vendor's bad minute
  may be over by the next perception. Every other failure is remembered as `None`, because asking
  again buys the same verdict: a config fault, an unusable answer that was already billed, or a clip
  that will not decode. Treating "failed" as one class in either direction is the regression.
- **The caption always names the describer.** Without it the brain reads a paragraph about a picture
  it never received as its own perception — and so does anyone reading its memory a month later. The
  describer is also put through the **brain's own** fail-closed `model_sees_video` gate, so a
  video-capable describer watches the clip and a vision-only one reads its frames: one rule applied
  twice, never two that can drift. It is offered no tools — a describer with tools is an agent.
- **A second model's prose gets a mining sentinel of its own.** The description is neither the
  agent's words nor a peer's; it is a third party's account of a peer's content. It sits outside the
  #438 boundary *by construction* — it rides an engine-injected turn, and `_dialogue_of` mines an
  asset's own dialogue and nothing else — which is precisely the kind of claim #438 proved a
  docstring cannot be trusted to keep, so `test_mining.py` proves it on a real wake.

`--resolved-config` reports `describer_model`, `describer_providers`, `describer_api_key_set`, the own
stack (`describer_provider`, `describer_sdk`, `describer_sdk_surface`, `describer_sdk_version`) and the
Vertex settings (`describer_location`, `describer_project`, `describer_credentials_file`,
`describer_credentials_file_state` — a verdict, never the file's contents) — **never the key
itself**, here or anywhere; that file is pasted into issues. The credential variables *also* ride
`tool_env` (`describer_credential_vars`: the key on a key-based vendor, the file and location on
Vertex), but **only when a model is configured**: it is an ungated call-time read with no plugin
to hang `needs_env` on (issue #427's shape without a `ToolPlugin`), and a `false` for a variable
nobody wants is the `XAI_TEAM_ID` noise that map deliberately avoids. With a describer configured
the map's contract holds exactly — *every `false` is an active capability that cannot do its job*.
