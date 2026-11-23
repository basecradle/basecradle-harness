# The Secret-Shapes File

The harness redacts secrets from every text it writes down: its log lines, what it files into an agent's memory, and the text files a tool saves in its capture folder (issue [#999](https://github.com/basecradle/basecradle-harness/issues/999)). What it counts as a secret is defined once, and that definition is **public data**, shipped in every release as one JSON file. The fleet's own scans and log pipeline read the file, so every reader works from the same definition. None of them imports a harness module, because the harness's modules are private and change without notice.

The harness reads the same file: its redaction module (`_redaction`) builds every rule from it, so the file is the only place a shape lives.

## Where It Is

Inside the wheel, and inside an installed package, the file is at:

```
basecradle_harness/data/secret-shapes.json
```

You can read it without importing the harness. From an installed package (an editable install included):

```python
import importlib.util
import json
from pathlib import Path

# Finds the installed package without running any of it.
spec = importlib.util.find_spec("basecradle_harness")
path = Path(spec.submodule_search_locations[0], "data", "secret-shapes.json")
shapes = json.loads(path.read_text(encoding="utf-8"))
```

You can also read it straight out of the wheel PyPI publishes for a version, as a zip member of that name, with no install at all.

## What a Reader May Rely On

- **`format` is `basecradle-harness-secret-shapes`, and `format_version` is `1`.** Check both before you trust a field.
- **A field is added, but never renamed, removed or repurposed, without a `format_version` change.** A reader must ignore fields it does not know. A new field does not change the version, so it never breaks a reader that ignores it.
- **Rules change without a format change.** A new vendor key, a new label name, or a different minimum ships in a release as an edit to the file, in a reviewed PR. The file a release carries is exactly what the harness at that version redacts.
- **Regular-expression fragments use the syntax Python's `re`, RE2 and VRL share**: character classes, alternation, `{m,n}` and `(?:…)`, never lookaround or backreferences. A label name is a literal, never a pattern.
- **The file defines what a secret is, not how a text is read for one.** How a value is found once its label is (a quoted value, an escaped quote at any depth, a JSON string's close) is each reader's own. The harness's reading is in its code, and the fleet's log pipeline has its own.

## What Neither Half Catches

The file has two halves: the agent's own configured secrets, matched by value, and the shapes, matched wherever they appear. **A key that is not one of the agent's configured secrets and has none of the listed shapes is caught by neither.** That covers an unlisted vendor's key the agent was handed in a message, read on a page, or keeps in a file the harness never loaded. This is why the agent is told that a secret put anywhere else *may* be redacted, never that it will be.

## Shape

```json
{
  "format": "basecradle-harness-secret-shapes",
  "format_version": 1,
  "documentation": "https://github.com/basecradle/basecradle-harness/blob/main/docs/secret-shapes.md",
  "note": "…",
  "markers": {},
  "configured_values": {},
  "vendor_keys": {},
  "credential_assignment": {},
  "bearer": {},
  "query_parameters": {},
  "api_key_assignment": {},
  "headers": {},
  "json_members": {},
  "heartbeat_url": {}
}
```

- `documentation` is the URL of this page.
- Every section, and the file itself, may carry a `note`: prose on what it means and why.

### `markers`

| Field | Meaning |
|---|---|
| `redacted` | What stands where a secret stood (`[redacted]`), but for a credential-named assignment's value. |
| `redacted_value` | What stands where a credential-named assignment's value stood, its name kept: `GH_TOKEN='<a redacted value>'`. It is one shell word with a space at its fourth character, so the fleet's log canary never reads it as a value and a second redaction leaves it as it is. It is the router's marker, byte for byte. |

In memory, a redacted exchange is still filed: the marker stands where the secret stood, and the rest of the exchange is kept. There is no second marker for memory, and no part, length or hash of the value is kept anywhere.

### `configured_values`

The agent's own secrets, matched **by value**, wherever they appear.

| Field | Meaning |
|---|---|
| `names` | Which rule decides that a name names a credential: `credential_assignment`, its `words` and `tail`, applied to the whole name (capitals, digits and `_` before the word, nothing after the tail). An MCP server's `env` key or header name is read upper-cased, with `-` as `_`. |
| `minimum_length` | The shortest value a bare match blanks (20, the floor the vendor-key rules use). A shorter value is still blanked where a credential name introduces it, by `credential_assignment`, but never on its own, so an ordinary word is never taken out of a memory. |

The values are the agent's environment variables whose names name a credential (`BASECRADLE_TOKEN`, `AI_API_KEY`, `NTFY_DM_TOKEN`), and every credential the harness holds under such a name. The file lists no value, ever: a reader that wants this half reads the values from the agent's own configuration.

### `vendor_keys`

`rules` is a list, tried **in order**. Each rule:

| Field | Meaning |
|---|---|
| `vendors` | Whose keys these are, in prose. |
| `prefix` | A regular-expression fragment the key starts with. |
| `run` | A character class: what follows the prefix. |
| `minimum_run` | How many characters of `run` follow the prefix, at least. |

A key is `(?:prefix)run{minimum_run,}`. A match glued to a letter, digit or combining mark before it, whose run holds no digit and no uppercase letter, is a word (`task-specification`, `disk-quota-exceeded`), not a key.

### `credential_assignment`

| Field | Meaning |
|---|---|
| `words` | The words that make a name a credential's. |
| `tail` | The one tail a name may carry after its word (`_BASE`, for `SECRET_KEY_BASE`). |
| `value_stops` | The characters the fleet's log canary ends a value at. |
| `value_minimum_length` | The canary's shortest value. |

A name is `[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD|PASSPHRASE)(_BASE)?` followed by `=`. Its value is redacted whatever it looks like, and the name is kept. This is the fleet log canary's `credential_assignment` rule, `[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD|PASSPHRASE)(_BASE)?=[^ ",}]{8,}`.

### `bearer`

| Field | Meaning |
|---|---|
| `scheme` | The scheme word, matched in any case. |
| `token` | A character class: the token's characters. |
| `minimum_token` | The shortest token. A shorter one is a word (`Bearer token missing`). |

### `query_parameters`, `api_key_assignment`, `headers`, `json_members`

Each holds `names`: literal label names, matched in any case.

| Section | Where the label stands, and what goes |
|---|---|
| `query_parameters` | After `?` or `&`, then `=`: the value, to the next `&` or whitespace. |
| `api_key_assignment` | Anywhere, then `=`: the value, to the next `&`, `"` or whitespace. |
| `headers` | Followed by `:` or `=>`, or by a quote and then `:`, `=`, `=>` or `,`: the value, to the next whitespace, quote, backslash or `&`. |
| `json_members` | As the whole name of a JSON member whose value is a string: the string's contents, its quotes kept. |

### `heartbeat_url`

| Field | Meaning |
|---|---|
| `path` | The path segment that makes a URL a Better Stack heartbeat URL. Its secret is the path, so the URL goes from its scheme (when it has one, so a userinfo before the host goes with it) or else from `path`, to the end of its run of non-whitespace. |

## Where the Harness Applies It

| Text | When | Covered |
|---|---|---|
| A log line | As it is written. | Every line the harness logs through its handler. |
| An exchange filed into memory | Before any memory provider sees it, so before it is split into drawers or written as a file. | Every exchange the wake files (`_wake.WakeAgent._observe`). An exchange the redaction cannot run on is not filed at all. |
| The session transcript (`sessions/<timeline>.json`) | On every save, before its caps cut anything: the file is rewritten from the turn on every save, so it is redacted where it is recorded, never after. | Every message's text, every tool result, every tool call's arguments (a string under a field whose name names a secret, such as `password` or an `x-api-key` header, is blanked whole at any depth), and every server-side tool's recorded input. The turn in flight keeps reading the value, and every later turn loads the redacted form. The arguments of an interrupted platform create are kept whole until its re-issue writes a result, because the recovery re-issues it from those bytes. |
| A fact the memory tool stores | Before it is written. | The SQLite store's value. A write whose key holds a secret is refused, because two keys redacted alike would overwrite each other. |
| A text file a tool saved | After the call that wrote it. | Under a stdio MCP server's working directory: every text file in `.playwright-mcp/`, and every text file the call's own arguments named, that the call wrote. Only files with a text format's suffix are opened, so an image is never read. A path that runs through a link once the call has run, and a file with another hard link, are refused. Nothing in the agent's config home, `~/.config`, `~/vault`, `~/wallets`, `~/.ssh` or `~/.gnupg` is ever read for redaction or changed. |

Each redaction outside the log is counted on one INFO line, `secrets redacted kind=<exchange|fact|capture> count=<n>` (`docs/telemetry.md`), and never carries the text.

Nothing already filed is rewritten: the rule redacts going forward.
