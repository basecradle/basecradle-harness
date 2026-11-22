# The Prices File

The harness prices every call to a vendor that returns tokens but no dollars (Anthropic, OpenAI, and Google on Vertex AI) from its own transcription of that vendor's published price page, and logs the result as `cost=… cost_basis=computed`. Those transcriptions are **public data**, shipped in every release as one JSON file. A product that wants to price an account the way the harness does reads the file. It never imports a harness module, because the harness's modules are private and change without notice (issue [#997](https://github.com/basecradle/basecradle-harness/issues/997)).

The harness reads the same file: its rate modules build their tables from it, so the file is the only place a price lives.

## Where It Is

Inside the wheel, and inside an installed package, the file is at:

```
basecradle_harness/data/prices.json
```

You can read it without importing the harness. From an installed package (an editable install included):

```python
import importlib.util
import json
from pathlib import Path

# Finds the installed package without running any of it.
spec = importlib.util.find_spec("basecradle_harness")
path = Path(spec.submodule_search_locations[0], "data", "prices.json")
prices = json.loads(path.read_text(encoding="utf-8"))
```

You can also read it straight out of the wheel PyPI publishes for a version, as a zip member of that name, with no install at all.

## What a Reader May Rely On

- **`format` is `basecradle-harness-prices`, and `format_version` is `1`.** Check both before you trust a field.
- **A field is added, but never renamed, removed or repurposed, without a `format_version` change.** A reader must ignore fields it does not know. A new field does not change the version, so it never breaks a reader that ignores it.
- **The numbers change without a format change.** When a vendor moves a price, the file is edited from the vendor's page in a reviewed PR, with the new `read_on` date, and ships in the next release. The file a release carries holds the prices the harness at that version logs.
- **Models come and go.** A model id that one release carries may be missing from a later one, and a model a table does not carry is a call the harness does not price.
- **`null` means the page states no rate there.** It never means zero. A call that needs that rate is not priced, and a reader should not price it either.

## Shape

```json
{
  "format": "basecradle-harness-prices",
  "format_version": 1,
  "documentation": "https://github.com/basecradle/basecradle-harness/blob/main/docs/prices.md",
  "currency": "USD",
  "vendors": { "anthropic": {}, "openai": {}, "google": {} }
}
```

- `documentation` is the URL of this page.
- `currency` is the currency of every price in the file.
- `vendors` holds one section per vendor, keyed by the harness's `AI_PROVIDER` slug.

Every rate in a row is in **US dollars per million tokens**, unless its field name says otherwise (`per_minute`, `…_usd_per_1000`).

### Fields Every Vendor Section Has

| Field | Meaning |
|---|---|
| `source` | Where the numbers were read: `url` (the vendor's price page), `note` (the part of the page, such as `"Standard tier"`, or `null`) and `read_on` (the day it was read, `YYYY-MM-DD`). |
| `note` | Prose on what the section carries and why. A model entry may carry its own `note` too, explaining a row a reader might otherwise "correct". |
| `model_key` | How a model id becomes a key into `models`. Apply the steps in this order: strip surrounding whitespace; when `last_path_segment` is `true`, keep only what follows the last `/`; lower-case the result; then, when `strip_suffix` is not `null`, remove the match of that regular expression. A dated snapshot is priced as the id it pins. |
| `long_prompt` | `threshold`: a call whose prompt is **over** this many tokens is priced on a model's `long` row, **every** token of it. Where models can lack a long row, `when_no_long_row` says what happens to such a call: `short_row_applies` (the model is priced flat, on its one row) or `not_priced`. |
| `models` | Model id → its rows. |

In every list of service-tier, speed or traffic names, the empty string `""` stands for a field that the request or response did not set, or set empty. Names are compared case-insensitively.

### `anthropic`

The Messages API reports a prompt in three disjoint parts: uncached input, cache writes (split by TTL) and cache reads. Their sum is the prompt length that `long_prompt` counts.

| Field | Meaning |
|---|---|
| `priced_host` | The only host the rates are the price on (`api.anthropic.com`). A call sent anywhere else is not priced. |
| `standard_service_tiers`, `standard_speeds` | The `service_tier` and `speed` values a response's usage may name for the call to be priced. Any other value is another row of the page, or no row at all. |
| `inference_geo_multipliers` | The request's `inference_geo` → the multiplier applied to the whole call (`"us"` is 1.1). A value missing from this map is not priced. |
| `web_search_usd_per_1000` | The web search fee, per 1,000 searches, counted from `usage.server_tool_use.web_search_requests`. |
| Row fields | `input` (uncached input), `write_5m` and `write_1h` (cache writes by TTL), `read` (cache reads), `output` (including thinking). `long` is `null` for a model priced flat. |

A call's cost is `(uncached × input + write_5m × write_5m + write_1h × write_1h + read × read + output × output) ÷ 1,000,000 × multiplier`, on the `long` row when the prompt is over the threshold and the model has one.

### `openai`

| Field | Meaning |
|---|---|
| `priced_host` | The only host the rates are the price on (`api.openai.com`). Data-residency and FedRAMP endpoints charge an uplift and are not priced. |
| `standard_service_tiers_served` | The `service_tier` values a **response** may name for the call to be priced as Standard. |
| `standard_service_tiers_asked` | The values a **request** may have asked for. These are consulted only when the response names no tier. |
| `web_search_usd_per_1000` | The web search fee, per 1,000 calls. Only a `web_search_call` whose action is `search` counts. |
| Row fields (`models`) | `input`, `cached`, `cache_write`, `output`. Reported input **includes** cached and cache-written tokens, and each token is priced at exactly one of the three input rates. A `cache_write` of `null` means the page charges nothing extra for a write, so a written token is priced at `input`. Reasoning tokens are output. |
| `image_models` | `image_input`, `text_input`, `image_output`, `text_output` per million tokens, from the Images API's usage, image and text priced apart. A `text_output` of `null` means a call that reports text output tokens is not priced. |
| `transcription_models` | `per_minute` (dollars per minute of audio, where the response reports a duration), and `input` / `output` per million tokens (where it reports tokens). |

A model call costs `((input − cached − written) × input + cached × cached + written × cache_write + output × output) ÷ 1,000,000`, on the `long` row above the threshold. Above the threshold, a model whose `long` is `null` is not priced.

### `google`

Google prices by model, by billing day, by endpoint class and by prompt size.

| Field | Meaning |
|---|---|
| `global_location` | The Vertex location Google prices as Global (`global`). Every other location, the `us` and `eu` multi-regions included, is non-global. |
| `billing_timezone` | The timezone of Google's billing day (`America/Los_Angeles`). A period's `start` and `end` are billing days in that zone. |
| `standard_traffic_types` | The `traffic_type` values a response may name for the call to be priced as Standard. |
| `unnamed_traffic_types` | The values that name no tier. When the response names one of these, the call is priced only if the request asked for one of the `standard_service_tiers`. |
| `standard_service_tiers` | The request `service_tier` values that ask for the standard tier. |
| `periods` | A model's rows over time, oldest first, each with an inclusive `start` and `end` (`null` is open-ended). The first period that contains the billing day applies. |
| `global`, `non_global` | A period's rows for each endpoint class, as `short` (a prompt of 200K tokens or fewer) and `long` (over 200K, which prices every token of the call, input and output alike). A class of `null` is one the page does not price for that model. Where the page states identical rows for both prompt sizes, the file states both. |
| Row fields | `input`, `cached`, `output` (including thinking). Where `audio_apart` is `true`, audio prompt tokens are priced at `audio_input` and cached audio at `audio_cached`. Where it is `false`, audio rides `input` and `cached`, and those two fields are `null`. Cached tokens are part of the prompt and are priced at the cache rate **instead of** the input rate. Tool-use prompt tokens are billed as input. |
| `grounding` | Google Search grounding, which is not a token charge. `usd_per_1000_queries` (Gemini 3: per grounding query listed in `webSearchQueries`) or `usd_per_1000_prompts` (Gemini 2.5: once per grounded prompt, however many queries it made), the other being `null`. `grounding_tokens_free` says whether the input tokens grounding feeds back are exempt from the token charge. Google bills a grounded prompt only when it returns sources. |

## What the File Cannot Tell You

The file holds every number, and every rule for choosing a row. Some things a reader still has to bring:

- **Reading a vendor's usage block.** Mapping a response's fields onto the token categories above is the vendor's API shape, not price data. This page names the fields that matter, and the vendor's API reference names the rest.
- **One gap is the harness's own choice.** When an Anthropic response reports a cache write with no TTL breakdown, the harness prices it at the TTL it asked for, which is 5 minutes. A reader pricing someone else's calls has to know what TTL those calls asked for.
- **Per-request rules need per-request usage.** A long-prompt threshold applies to one request. Usage that is already summed into buckets, such as an hourly usage report, cannot be held against it, so a reader pricing buckets uses the `short` row and reconciles against the bill.
- **Charges no response states.** Code-execution container time (Anthropic and OpenAI) is not priced anywhere, because no response reports how long a container lived. Google's free grounding allowance is not netted out, because no caller can see where the account stands against it.
- **Vendors that state their own price** (OpenRouter, xAI) and **refuseless**, whose price list is read live from its own API, are not in the file: there is nothing to transcribe.
