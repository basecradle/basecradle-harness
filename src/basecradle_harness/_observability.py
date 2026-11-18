"""The log lines that make a wake legible — the shared seam every emitter writes through.

A deployed agent's only witness is its journal. The router's wake-runner stamps each line's
**who** into journald metadata (``SYSLOG_IDENTIFIER=basecradle-wake-<agent>``) and the shipping
layer (Vector) does the presentation, so the harness's job is exactly one thing: say **what
happened**, in lean ``key=value`` text. Nothing here hand-prefixes ``[Harness]`` or an agent
name into the message — that would duplicate the identifier journald already carries.

Three emitters live here because they are cross-cutting: the per-call **LLM** line (written by
every provider adapter), the per-use **tool** line (written by the engine for a harness or MCP
tool, and by an adapter for a tool its vendor ran server-side — one head whoever executed it,
`docs/telemetry.md` → Tools), and the ``kv`` formatter they share. The other lines — the wake
bookends, the step ledger — are written where the fact is known (`_wake`, `_engine`) but format
through `kv` all the same, so one grep syntax reads the whole stream.

What is deliberately *not* logged: prompts, request bodies, response bodies, and API keys. A log
line names the shape of a call — provider, model, duration, token counts — never its content.

**Color is the one presentation choice made here** (issue #414, founder decision 2026-08-17), and
only because it is not really presentation: a wake's lifecycle and its verdict are what a human
scanning a fleet-wide Live Tail is looking for, and in a stream of uniform grey they are found by
reading rather than by seeing. So a *verdict* — a head like ``wake start`` or ``wake failed``, and
the ``outcome=`` pair — is wrapped in an ANSI color, and everything else stays exactly the bytes it
was. The whole convention rests on one rule (`head`): the escape wraps a **whole token**, never
part of one, so nothing that could grep the plain line stops matching the colored one. Set
`NO_COLOR_ENV` and every line goes out plain.

**The one deliberate exception is the agent's own voice** (`log_unspoken`, issue #293). Since the
final-text auto-post was removed, a turn's narration is *unspoken*: it reaches no timeline and no
peer, so this stream is the only place it exists. That is the trade the Unspoken Channel is built
on — *"we never force its action or inaction, but we do require full visibility, which is the price
of that freedom"* — and it only holds if the record is really complete. A truncated narration would
make the visibility partial and the trade a lie, so this one field is rendered in **full**
(flattened, scrubbed, quoted — but never cut).

It is a **flight recorder, not a control tower**: nobody watches it. The record exists so an agent's
own memory can carry what it decided and why, and so the rare failure can be reconstructed after the
fact. It is the agent's *own* words, not a provider's response body — which is why it is the one
content the "never log content" rule yields to, and why it is never bounded.

The other place foreign text enters the stream is an **error message** (a tool's exception, an
SDK refusal), and that is why `kv` *renders* values rather than interpolating them: the text is
not the harness's, so it is flattened to one line, scrubbed of credential shapes, bounded, and
quoted before it goes near a record. An error message is a breadcrumb *to* the failure — the
model still receives the full text as its tool result, and the transcript keeps it.
"""

from __future__ import annotations

import contextlib
import getpass
import heapq
import logging
import math
import numbers
import os
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

from basecradle_harness._exceptions import ProviderError, ProviderTruncatedError

_log = logging.getLogger("basecradle_harness")

#: The cap on one field's rendered length. The values that need it are the error texts — a tool's
#: exception, an SDK refusal — which can run to a whole response body. A log line is a *breadcrumb*
#: to the failure, not a copy of it: the model already received the full error as its tool result,
#: and the transcript keeps it.
MAX_VALUE = 240

#: The ANSI SGR codes the fleet's journal is colored with (founder decision, 2026-08-17). Four
#: colors and a reset, shared with the router's and the NOC's own lines so one palette reads
#: across the whole Live Tail: GREEN for a good outcome, RED for a bad one, YELLOW for the
#: in-between (a skip, a decline, a block) and for an operator's choice (a stop: neither success
#: nor failure, issue #799), BLUE for the neutral bookend that closes a wake. The tiers are defined
#: once for the fleet in `docs/telemetry.md` → The Palette, which `test_telemetry_contract.py`
#: holds this module to.
GREEN = "\x1b[32m"
BLUE = "\x1b[34m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
RESET = "\x1b[0m"

#: The cross-ecosystem opt-out (https://no-color.org): set to *anything* non-empty and every line
#: goes out in plain bytes. Honored rather than invented — an operator whose downstream cannot
#: render SGR should not have to learn a harness-specific knob for it.
#:
#: **Deliberately not gated on `isatty()`**, which is the "fix" this will attract: the whole point
#: of the color is a surface that is *never* a terminal. A deployed wake writes to stderr, systemd
#: captures it into journald, Vector ships it to Better Stack, and a human reads it there — not one
#: of those hops is a tty, so a tty gate would turn the feature off in exactly the place it was
#: built for and leave it on only where nobody was looking.
NO_COLOR_ENV = "NO_COLOR"

#: The keys whose **value** is a verdict rather than data, and the vocabulary each one colors by.
#:
#: This is the second half of the color contract and the reason it lives in `kv` rather than at a
#: call site: `outcome` is a *closed* vocabulary written by several emitters (`_wake`'s bookend,
#: the `tool` line, the `llm` line), so coloring it by value in one place means every `outcome=` token in
#: the stream reads the same, and a third emitter gets it right by construction. A value outside the
#: vocabulary is left **plain** — the same honest-absence rule `cost` and `endpoint` keep: an
#: unrecognized verdict is not guessed at a color.
#:
#: Everything else stays plain, and the exclusion is load-bearing rather than incidental:
#: `timeline=`, `delivery=`, `provider=`, `model=` are **correlation values** — data a human copies
#: out and pastes into a query — and coloring them would put escape bytes inside the very tokens the
#: line exists to hand over.
_VERDICT_VALUES: dict[str, dict[str, str]] = {
    # The `llm` line's verdicts (issue #782) join the vocabulary: a final failure reads as an error
    # does, and the three degradations (retried, refused, unusable) as the warning they are.
    "outcome": {
        "ok": GREEN,
        "error": RED,
        "failed": RED,
        # Done and usable with a named shortfall (issue #822): the launcher's `charge` line.
        "degraded": YELLOW,
        "retrying": YELLOW,
        "refused": YELLOW,
        "fallback": YELLOW,
        # A wake SIGTERM stopped (issue #798): the Operator's Choice tier (issue #799), never RED.
        "stopped": YELLOW,
    },
}

#: Credential shapes scrubbed out of any logged value — defense in depth at the *source*, so a
#: secret never enters the journal in the first place rather than relying on the shipping layer to
#: catch it later. An error text is the realistic carrier: an exception from a drop-in tool or MCP
#: server can embed the request URL (``…?api_key=…``) or an ``Authorization`` header, and a
#: provider's own 4xx body can echo the key it rejected.
#:
#: **This is the first line, and it carries every shape the box's second line does** (issue #787;
#: `docs/telemetry.md` → Credential Scrubbing). The fleet's Vector ``ai_scrub`` transform redacts the
#: journal again before it ships, but this package also runs where no Vector exists — a developer's
#: laptop, anyone's fork — so the harness never leaves a shape it knows of to a layer that may not be
#: there. The two lists cannot be one list (one is VRL in another repository), so the rule is that
#: this one is a **superset**: a shape added to ``ai_scrub`` is added here, at no longer a minimum,
#: and `tests/test_log_value_cut.py` holds one fabricated sample per box rule, each at the box's own
#: minimum. The one stated difference is the bearer rule, which needs eight characters here so a
#: diagnosis like "Bearer token missing" keeps its words; no real bearer credential is shorter.
#: Every rule is linear in the input (a value is scrubbed before it is cut, so it can be long); the
#: heartbeat rule is the one that needs care to stay so (`_HEARTBEAT_RUN`). Covered: the fleet's
#: own key shapes (OpenAI/Anthropic ``sk-``, OpenRouter ``sk-or-``, xAI ``xai-``, BaseCradle
#: ``bc_uat_``/``bc_isk_``, refuseless ``abl_``, Perplexity ``pplx-``, an ntfy access token
#: ``tk_``), the box's (GitHub ``gh?_`` and fine-grained ``github_pat_``, Google ``AIza``, Hugging
#: Face ``hf_``, Replicate ``r8_``, Steel ``ste-``, a Better Stack heartbeat URL, an unanchored
#: ``api_key=``), and the generic bearer and key-in-query forms. A vendor key is covered only when
#: its shape can be stated from the vendor's own word (issue #856): refuseless's ``abl_`` plus 48
#: alphanumerics was read off two real keys, ntfy's ``tk_`` plus 29 is its documentation's, and
#: Perplexity's ``pplx-`` prefix is its documentation's with gitleaks's 48 alphanumerics. Each of
#: the three takes a minimum run of 20, under its vendor's length, as ``AIza``, ``hf_``, ``r8_`` and
#: ``ste-`` do. ntfy generates a token from lowercase letters and digits, and its validator also
#: admits capitals, ``_`` and ``-`` (``^tk_[-_A-Za-z0-9]{29}$``, ntfy's ``user/util.go``), which is
#: the run ``tk_`` takes. A generated token with no digit, about 1 in 12,500, is the word residual
#: below when glued to a letter. A key with no prefix of its own (Wiro's) has no shape a prefix rule
#: can match; it is covered where it travels, as an ``x-api-key`` header value (`_X_API_KEY`,
#: issue #861).
#:
#: **A vendor prefix inside a word is a word, not a credential** (issue #794), and the test is
#: `_is_word`, never a left boundary. A boundary would be a list of the contexts in which
#: redaction stops, and every serialization adds one: a key glued to a letter or digit is not a
#: shape any vendor issues, but it is one a percent-encoding (``Bearer%20sk-…``), a backslash
#: escape (``\nsk-…``), an ANSI color code (``ESC[33msk-…``) or Chinese text (``密钥sk-…``)
#: produces. So a prefixed match is kept only when **both** hold: a letter, digit or combining mark
#: precedes it, **and** its run after the prefix holds no digit and no uppercase letter. A random key
#: of a vendor's length holds one of them with overwhelming certainty; ``task-specification``,
#: ``disk-quota-exceeded`` and ``risk-assessment-pending`` hold neither. The two residuals are
#: stated rather than hidden: a glued word whose run holds a digit or a capital is still redacted
#: (the scrub errs toward redaction), and a glued key made only of lowercase letters, ``_`` and
#: ``-`` is kept. The rules are spelled once, in `_PREFIXED`, as a prefix, the class of its run and
#: the run's minimum length. `_PREFIX` can strip a longer prefix than the alternative that matched
#: (``xsk-or-…`` matched as ``sk-``), and that never changes the verdict: every prefix is lowercase
#: letters, ``_`` and ``-``, which are never evidence of a key.
#:
#: **A narrow run never swallows the head of a key glued after it** (issue #858). A run whose class
#: lacks ``_`` or ``-`` once took the letters of a following key's prefix (``hf_…aghp_ABC…`` took
#: ``…agh``) and stopped at its ``_``, so the rest of the second key matched nothing, and a first
#: match kept as a word kept the second key with it. So a narrow run (`_rule`) first tries to end,
#: as early as it can, where a **whole key by its own rule** begins (`_GLUED_KEY`), and then takes
#: everything after it in the widest class, so a third key glued on is redacted with the second;
#: only when no glued key follows does it match as it always did. That is a negative lookahead
#: written without one: the box's ``ai_scrub`` is VRL, whose regex has no lookaround, and mirrors
#: this pattern. A match that took a glued key is never kept as a word (`_is_word`), so nothing
#: redacted before is redacted less: a match only grows, and what it grows over is redacted with it.
#: The cost is an over-redaction in the rare glued case: a ``-word`` glued after the second key goes
#: with it, and so does a lowercase word that held a whole key's shape. A wide run already takes a
#: glued key whole.
_WIDE = "[A-Za-z0-9_-]"
_PREFIXED = (
    (r"sk-or-|sk-|xai-|bc_[a-z]{3}_", _WIDE, 8),
    (r"bc_(?:uat|isk)_", "[A-Za-z0-9]", 1),
    (r"gh[a-z]_", "[A-Za-z0-9]", 20),
    (r"github_pat_", "[A-Za-z0-9_]", 20),
    (r"AIza", _WIDE, 20),
    (r"hf_", "[A-Za-z0-9]", 20),
    (r"r8_", "[A-Za-z0-9]", 20),
    (r"ste-", "[A-Za-z0-9]", 20),
    (r"abl_", "[A-Za-z0-9]", 20),
    (r"pplx-", "[A-Za-z0-9]", 20),
    (r"tk_", _WIDE, 20),
)
_PREFIX = re.compile("|".join(f"(?:{prefix})" for prefix, _, _ in _PREFIXED))
_GLUED_KEY = "|".join(f"(?:{prefix}){run}{{{least}}}" for prefix, run, least in _PREFIXED)


def _rule(index: int, prefix: str, run: str, least: int) -> str:
    """One prefix rule's pattern: a wide run as written, a narrow one ending where a glued key begins.

    The glued alternative is captured as ``glued<index>``, so `_is_word` can tell that a match holds
    a second key and is never a word."""
    if run == _WIDE:
        return f"(?:{prefix}){run}{{{least},}}"
    glued = f"(?P<glued{index}>{run}{{{least},}}?(?:{_GLUED_KEY}){_WIDE}*)"
    return f"(?:{prefix})(?:{glued}|{run}{{{least},}})"


_PREFIX_RULES = re.compile("|".join(_rule(i, *rule) for i, rule in enumerate(_PREFIXED)))

#: The rules anchored on a literal of their own — an ``x-api-key`` header, a key in a query, an
#: ``api_key=`` assignment, a bearer header — each redacted wherever it matches. **Every rule reads
#: the text as it came, and what the scrub takes is the union of what each rule takes** (`scrub`,
#: issue #929): no rule is ever shown another's ``[redacted]``, so no order among the rules can make
#: one take less than it takes alone. The rules once ran as passes, each over the last one's output,
#: and every leak that order made is the reason they do not: ``task-runner-api-key=…`` consumed as
#: the prefix run ``sk-runner-api-key``, its value matched by nothing; ``Bearer AB_TOKEN=secretsecret``
#: redacted to ``[redacted]=secretsecret``, whose name no later pass could see; a run of scheme words
#: split by the bearer rule (``A_TOKEN=Bearer x.Bearer 'eyJ…'``), the token left after the split; and
#: a marker written inside an enclosing value's open quote (``A_TOKEN=='\X_API_KEY=x-api-key: eyJ…``),
#: whose own quotes moved that quote's end and left the token outside every value.
#:
#: **An assignment's value may open with a ``"``, and a bearer header glued to a value goes with it**
#: (the review of #857): ``API_KEY="…"`` once matched nothing, because the value could not start at
#: the quote, and ``api_key=KBearer eyJ…`` or ``?token=KBearer eyJ…`` ended the value at the scheme
#: word and left its token to no rule. Each value is two alternatives, the first a value ending in a
#: glued ``bearer`` and its token, written without lookaround for the box's VRL.
#:
#: **An ``x-api-key`` header value is redacted by its name, never its shape** (issue #861), because
#: the vendor that sends a key there states no shape for it: Wiro's documentation gives its key no
#: prefix, length or character set. The name is ``x-api-key`` (any case, ``_`` for either hyphen)
#: followed by ``:`` or ``=>`` (a header dump, a Ruby hash), or by a quote, escaped or not, and then
#: ``:``, ``=``, ``=>`` or ``,`` (``"x-api-key": "…"``, JSON inside a JSON string, a
#: ``('x-api-key', '…')`` header tuple); the value may open with a quote, an escaped quote, a bytes
#: literal's ``b'``, or bearer scheme words, and runs to the next space, quote, backslash or ``&``,
#: since a key of no stated shape may hold anything else. **A bearer header glued to the value is taken
#: with it** (``x-api-key: KBearer eyJ…``): the value would otherwise end at the scheme word and
#: leave its token to no rule, the class `_TRAILING_BEARER` closes for a heartbeat URL. The
#: unquoted ``X-API-Key=…`` is left to the ``api_key=`` rule below, which always took it with the
#: wider value it allows. The header's name goes with its value. A bare ``x-api-key`` with no value
#: is a word.
#: Wiro's signing secret is never sent (its ``x-signature`` is an HMAC keyed on the API key), so
#: there is nothing more to match.
#:
#: **A label that ends in a credential's name keeps it** (``OPENAI_API_KEY=…``, ``?ACCESS_TOKEN=…``,
#: ``X-API-KEY=…``): where the assignment rule finds a name ending at the label's ``=``, the label
#: stays and its value becomes `REDACTED_VALUE`, as every credential-named assignment's does; any
#: other label (``api_key=``, ``?token=``) goes with its value as ``[redacted]``.
_X_API_LABEL = r"(?i:x[-_]api[-_]?key)(?:\\?[\"']\s*(?:=>|[:=,])|\s*(?::|=>))"
_X_API_OPENER = r"\s*(?:b?\\?[\"'])?"
_X_API_RUN = r"[^\s\"'&\\]"
_SCHEME_WORDS = r"(?:(?i:bearer)\s+)*"
_X_API_KEY = re.compile(
    _X_API_LABEL + _X_API_OPENER + f"(?P<value>{_SCHEME_WORDS}"
    rf"(?:{_X_API_RUN}*?(?i:bearer)\s+[A-Za-z0-9._~+/-]{{8,}}|{_X_API_RUN}+))"
)
#: How far before a header's span end a label that would read on past it can start: the longest
#: label (``x-api-key``), its separator (``=>``), and the one scheme word its value may open with
#: glued to them, the blank after it being where the span stops (`_x_api_key_spans`).
_X_API_LABEL_REACH = len("x-api-key") + len("=>") + len("bearer")
_QUERY_LABEL = r"[?&](?i:api[-_]?key|access[-_]?token|token|key)="
_QUERY_RUN = r"[^&\s]"
_QUERY = re.compile(
    f"(?P<label>{_QUERY_LABEL}){_SCHEME_WORDS}"
    rf"(?:{_QUERY_RUN}*?(?i:bearer)\s+[A-Za-z0-9._~+/-]{{8,}}|{_QUERY_RUN}+)"
)
_API_KEY_LABEL = r"(?i:api[-_]?key=)"
_API_KEY_OPENER = r"\"?"
_API_KEY_RUN = r"[^&\s\"]"
_API_KEY = re.compile(
    f"(?P<label>{_API_KEY_LABEL}){_API_KEY_OPENER}{_SCHEME_WORDS}"
    rf"(?:{_API_KEY_RUN}*?(?i:bearer)\s+[A-Za-z0-9._~+/-]{{8,}}|{_API_KEY_RUN}+)"
)

#: A bearer header: the scheme word in any case and its token, which is eight or more token
#: characters, so a diagnosis like "Bearer token missing" keeps its words, or a token that opens
#: with a quote or a substitution (issue #929), taken whole as the shell word it opens
#: (`_shell_word_end`), as a credential's value is read. Quoted, a token is no run of token
#: characters, and ``Bearer 'eyJ…'`` once matched nothing. **A run of scheme words goes with the
#: token after it** (`_bearer_end`).
_BEARER = re.compile(r"(?P<scheme>(?i:bearer)\s+)(?:[A-Za-z0-9._~+/-]{8,}|[\"'`]|\$['({]|\\+[\"'])")
_BEARER_TOKEN = re.compile(r"[A-Za-z0-9._~+/-]{8,}")
_BEARER_GAP = re.compile(r"\s+")
_TOKEN_OPENERS = ("'", '"', "`", "$'", "$(", "${")

#: A bearer token in escaped quotes (``"Bearer \\"eyJ…\\""``, JSON inside a JSON string, escaped
#: once or more): the escaped quote and the rest of its word, to the next blank or record break.
#: An escaped quote closes nothing a reader can trust (it may be unbalanced, or cut), so the word
#: is the token, whatever is glued to it, and never more than the word.
_ESCAPED_TOKEN = re.compile("\\\\+[\"'][^\\s\\x00]*")

#: A Better Stack heartbeat URL, whose secret is its path, redacted by the box's rule
#: ``(?:https?://\S*)?/heartbeat/\S+`` (basecradle-router#398): a URL with a scheme from the scheme
#: on, so userinfo before the host (``https://john:…@host/…/heartbeat/…``) never ships; one without
#: (``curl uptime.example.com/api/v1/heartbeat/…``) from ``/heartbeat/`` on, keeping its host, which
#: is not the secret. **On the box it runs first, before every other rule**: a rule that matched
#: first and stopped inside the URL (a bearer run takes ``…https`` and stops at the ``:``) would
#: leave the rest of it, userinfo included, to the scheme-less arm alone. Here every rule reads the
#: text as it came (`scrub`), so no other rule's match can stop inside the URL or hide a run's start.
#:
#: **The box's pattern is not used here, because in `re` it is quadratic**: every ``https?://`` in a
#: space-free run scans to the run's end before it fails, and ``"http://" * 8000`` took 2.9 s. So this
#: pattern matches a whole space-free run that holds ``/heartbeat/`` and something after it, and only
#: from the run's start: the lookbehind fails at once everywhere else, and each run is scanned once.
#: `_heartbeat` then cuts the run where the box's pattern would have begun its match, which is the
#: leftmost of the run's first ``/heartbeat/`` and its first scheme that ends before the run's last
#: ``/heartbeat/`` with a character after it. The box's match runs to the end of the run, as this one
#: does, so the bytes redacted are the box's — with one addition, `_TRAILING_BEARER`.
_HEARTBEAT_RUN = re.compile(r"(?<!\S)\S*/heartbeat/\S+")
_HEARTBEAT = "/heartbeat/"
_SCHEME = re.compile(r"https?://")

#: A bearer scheme word ending a heartbeat run, which `_heartbeat` keeps so the bearer rule still
#: sees it beside its token in the next run. Taken to the run's end, the word went with the URL and
#: left the token with no rule to match it: compact JSON (``{"url":"https://h/heartbeat/K9",
#: "auth":"Bearer eyJ…"}``) and a query value ending ``…&access_token=Bearer eyJ…`` shipped the
#: token, on the box as well. Not glued to a letter or digit, so a heartbeat secret that merely ends
#: in those letters is never shortened into a kept word.
_TRAILING_BEARER = re.compile(r"(?<![A-Za-z0-9])(?i:bearer)\Z")

# --- credential-named assignments (issue #918, basecradle#877) ---------------------------------
#
# The value of every assignment whose name names a credential is redacted, whatever the value looks
# like and whatever quoting surrounds it: the name stays, the value becomes `REDACTED_VALUE`. "Names
# a credential" is the NOC's log-store canary's definition, its `credential_assignment` rule
# (basecradle-noc `observability/log-canary.json`), never a list of this package's own:
#
#     [A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD|PASSPHRASE)(_BASE)?=[^ ",}]{8,}
#
# `tests/conftest.py` holds that pattern byte for byte, and the tests hold the pieces below to it:
# the words, the tail, the value's stop characters and its floor must spell it exactly, the name
# pattern must end at exactly the canary's `=`s, and the canary's own pattern is the oracle every
# scrubbed text is checked against. The router renders by the same definition (basecradle-router
# #491, `logfmt.redact_credentials`; its widened copy moves in basecradle-router#504), so both
# writers on a box leave the same bytes where a value stood. The definition was widened by `@origin`'s decision on basecradle-noc#1547 (2026-11-17,
# issue #927): a name needs no capital before its word, so a bare ``PASSWORD=`` is one, and
# ``_BASE`` is the one tail allowed after the word (``SECRET_KEY_BASE=``, the Rails master secret's
# name); no other tail, so ``GH_TOKEN_FILE=`` and ``GH_TOKENS=`` name no credential.

#: The words that make a name a credential's, in the canary's order.
CREDENTIAL_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSPHRASE")

#: The one tail a credential's name may carry after its word, as the canary's optional group.
CREDENTIAL_TAIL = "_BASE"

#: The canary reads a value as a run of anything but these, at least `CANARY_VALUE_FLOOR`
#: characters long, from the character after the ``=``.
CANARY_VALUE_STOPS = ' ",}'
CANARY_VALUE_FLOOR = 8

#: The name pattern's word and tail, as one group: a credential word and, optionally, its tail.
_WORD_AND_TAIL = "(?:" + "|".join(CREDENTIAL_WORDS) + f")(?:{CREDENTIAL_TAIL})?"

#: A credential name, up to the ``=`` it is assigned with. The canary's own name pattern,
#: ``[A-Z0-9_]*(…)(_BASE)?=``, run unanchored in a backtracking engine such as Python's `re` tries
#: every start inside a run of name characters and is quadratic in it. This spelling matches before
#: exactly the same ``=``s (pinned against the canary's pattern by a test) and is linear: it starts
#: only where a run of name characters starts, and extends lazily to a credential word and its
#: optional tail. A run holds at most one ``=``, at its end, so one start per run finds every name.
CREDENTIAL_NAME = r"(?<![A-Z0-9_])[A-Z0-9_]*?" + _WORD_AND_TAIL + "="
_CREDENTIAL_NAME = re.compile(CREDENTIAL_NAME)

#: What stands where a credential's value stood — the router's marker, byte for byte. It must read
#: as one value to the redactor and as no value to the canary: a single-quoted word, so scrubbing
#: it again leaves it as it is, with a space at its fourth character, so the canary's run after the
#: ``=`` is ``'<a``, far under its floor, whatever follows. `scrub`'s own ``[redacted]`` would not
#: do: ten characters the canary reads as a value, so ``GH_TOKEN=[redacted]`` is a finding.
REDACTED_VALUE = "'<a redacted value>'"

#: Every character journald, or Python reading a line, starts a new record or line at. A quote
#: still open at one of these ends there (`_shell_word_end`), and the canary's run stops there.
_BREAK_CHARS = "\n\r\x00\x0b\x0c\x1c\x1d\x1e\x85  "

#: The canary's value run, where it ends in every record the text can be written into: at one of
#: the canary's own stops, or at a record break, since no record holds a run across one.
_CANARY_RUN = re.compile("[^" + re.escape(CANARY_VALUE_STOPS + _BREAK_CHARS) + "]*")

#: The characters that end an unquoted shell word: the shell's blanks and its newline.
_SHELL_BLANKS = " \t\n"

#: The end of a credential name, its ``=`` included: a credential word, the one tail, the ``=``. The
#: canary's name is unanchored, so whatever comes before the word, this is a name's end.
_NAME_END = re.compile("(?:" + "|".join(CREDENTIAL_WORDS) + f")(?:{CREDENTIAL_TAIL})?=\\Z")
_NAME_END_REACH = max(len(word) for word in CREDENTIAL_WORDS) + len(CREDENTIAL_TAIL) + 1

# --- the router's reading (basecradle-router `logfmt`, issue #929) --------------------------------
#
# A credential's value is read here exactly as the router reads it (basecradle-router#506, merged as
# `153fb58`: `logfmt._shell_word_end` and `logfmt._value_end`), so both writers on a box leave the
# same bytes where a value stood; `tests/router_reading.py` carries the router's functions verbatim
# and `tests/test_credential_assignment.py` holds the two readings equal on fuzzed text. The one
# addition is `limit`, which only `bounded` passes: the reading goes no further than it, and a value
# that reaches it is reported as ending there.


def _shell_word_end(text: str, start: int, limit: int | None = None) -> tuple[int, bool]:
    """Where the shell word from `start` ends: at a blank outside every quote, or the end.

    One word as the shell reads it, so a value is taken whole whatever surrounds it: a single- or
    double-quoted string, an ANSI-C ``$'…'``, a ``$(…)``, ``${…}`` or ``(…)`` (each nesting as the
    shell's does, quotes inside included), a backquoted command, and a backslash escape (but of a
    tab outside every quote: the backslash is then the word's last character, since the canary's
    run crosses the tab and a scheme word after it is the value's, basecradle-router#502's review).
    Only a space, a tab or a newline ends an unquoted word, as in the shell, so no other character
    can cut a value short. A quote or a substitution still open at a record break ends there: it is
    prose (an apostrophe, a markdown backquote) far more often than a value that spans lines.
    Iterative, with its own stack, so no depth of nesting can exhaust the interpreter's.

    The word also ends just past a credential name's ``=`` that stands outside every quote and
    substitution in it (``x,DB_PASSWORD=…``), and then says so: the rest of the word is that name's
    value. **A name inside a quote is the quote's text** (issue #929): ``A_TOKEN='B_TOKEN=x' next=1``
    keeps ``next=1``, where reading ``B_TOKEN=`` as a name opened a quote at its value's ``'`` and
    took the rest of the line. Returns where the word ends, and whether it ends at a name's ``=``;
    a word still open at `limit` is reported as ending there.
    """
    stack: list[str] = []  # what closes each open context: ' " ` ) } and $' for ANSI-C
    i = start
    n = len(text) if limit is None else min(limit, len(text))
    while i < n:
        c = text[i]
        top = stack[-1] if stack else ""
        if top and c in _BREAK_CHARS:
            return i, False
        if top == "'":
            if c == "'":
                stack.pop()
            i += 1
        elif top == "$'":
            if c == "\\":
                i += 2
                continue
            if c == "'":
                stack.pop()
            i += 1
        elif c == "\\":
            if not top and text.startswith("\t", i + 1):
                # A backslash before a tab is the word's last character (basecradle-router#502).
                return i + 1, False
            i += 2
        elif top == "`":
            if c == "`":
                stack.pop()
            i += 1
        elif top == '"':
            if c == '"':
                stack.pop()
                i += 1
            elif text.startswith(("$(", "${"), i):
                stack.append(")" if text[i + 1] == "(" else "}")
                i += 2
            else:
                if c == "`":
                    stack.append("`")
                i += 1
        else:  # outside every quote: the word itself, or inside a ( ), $( ) or ${ }
            if not top and c in _SHELL_BLANKS:
                break
            if (
                not top
                and c == "="
                and _NAME_END.search(text, max(start, i + 1 - _NAME_END_REACH), i + 1)
            ):
                return i + 1, True
            if top and c == top:
                stack.pop()
                i += 1
            elif text.startswith("$'", i):
                stack.append("$'")
                i += 2
            elif text.startswith(("$(", "${"), i):
                stack.append(")" if text[i + 1] == "(" else "}")
                i += 2
            else:
                if c in "'\"`":
                    stack.append(c)
                elif c == "(" and top != "}":
                    stack.append(")")  # an array, a subshell, arithmetic
                i += 1
    return min(i, n), False


#: The characters a bearer header's scheme word may be glued to (``KBearer``, ``api_key=Bearer``):
#: a token's, as a header value spells one. The router's `logfmt.SCHEME_GLUE`, byte for byte
#: (basecradle-router#497), so both writers on a box read the same scheme word.
_SCHEME_GLUE = "A-Za-z0-9._~+/%:=-"

#: A run of the characters a scheme word may be glued to: one word, as `_bearer_end` reads it.
_GLUE_RUN = re.compile(f"[{_SCHEME_GLUE}]*")

#: A bearer header's scheme word, which is never a credential's whole value: the token after it is
#: (issue #926, basecradle-router#497). Alone it is the word in any case; glued to a value only as
#: HTTP spells it, so a value merely ending in the letters (``abc-bearer``) is no header.
_SCHEME_WORD = re.compile(f"(?i:bearer)|[{_SCHEME_GLUE}]*Bearer")

#: What stands between a scheme word and its token: blanks, never a newline, as the box reads them.
_SCHEME_BLANKS = re.compile("[ \t]+")

#: The characters, beyond a blank, a canary stop and a record break, that a word may not hold
#: unquoted before the tab ahead of a bearer scheme word (basecradle-router#502): a quote's opener, a
#: backslash's escape and a parenthesis, whose reading a regular expression cannot nest. The
#: router's `logfmt.LEAD_OPENERS`, byte for byte.
_LEAD_OPENERS = "'`\\("


def _run_char(but: str = "") -> str:
    """A character of the canary's run (a tab is one), and none of `but`."""
    return "[^" + re.escape(CANARY_VALUE_STOPS + _BREAK_CHARS + but) + "]"


#: A word the canary's run crosses whole, and the tabs after it: what stands before a scheme word
#: that a tab glues to a value (``x<TAB>Bearer``, ``'q'<TAB>KBearer``), so the token after that
#: scheme word goes with the value (basecradle-router#502). Its pieces each close inside the run: a
#: bare character, a backslash and the character after it, a single-quoted piece and a backquoted
#: one (in which a backslash escapes), so each piece has one reading and the word one end, the shell
#: word's. A backslash before the tab is the word's last character, as the shell reading has it
#: here (``a\<TAB>Bearer``: the canary's run crosses the tab). A word holding a quote that never
#: closes there, a ``"``, a ``(``, a blank or a stop is no lead word, and no scheme word after it is
#: read as one.
_LEAD_PIECE = (
    "(?:"
    + "|".join(
        [
            _run_char("\t" + _LEAD_OPENERS),  # a bare character: neither a tab nor an opener
            "\\\\" + _run_char("\t"),  # an escape, of a quote or a backslash included
            "'" + _run_char("'") + "*'",  # single-quoted: no escape inside
            "`(?:" + _run_char("`\\") + "|\\\\" + _run_char() + ")*`",  # backquoted
            "\\(" + _run_char("()'`\\") + "*\\)",  # a parenthesis, closed, one level, no quote
        ]
    )
    + ")"
)
_LEAD_WORD = re.compile(f"(?:{_LEAD_PIECE}+\\\\?|\\\\)\t+")

#: The tabs a value may open with: the canary's run crosses them to its first word.
_TABS = re.compile("\t*")

#: Where a marker at a value's head ends it: a blank, a canary stop or a record break after it (the
#: canary's own run ends inside the marker, at its space), so a value redacted where it is bounded
#: and again where it is rendered stays one marker.
_MARKER_ENDS = frozenset("\t" + CANARY_VALUE_STOPS + _BREAK_CHARS)

#: What a word after a tab may not start with, to be reached across it: a blank, a ``,`` or a ``}``
#: (the canary's stops, which end its run and are structure on a JSON line) and a record break, so
#: no word the value reaches this way starts in the next record. A ``"`` may: it opens the word's
#: quoted piece.
_UNREACHED = frozenset(CANARY_VALUE_STOPS.replace('"', "") + _BREAK_CHARS)


def _reached(text: str, at: int) -> bool:
    return at < len(text) and text[at] not in _UNREACHED


def _token_after_a_glued_scheme_word(text: str, floor: int, end: int) -> int | None:
    """Where the token starts, when a value read from `floor` ends at `end` just after a bearer
    scheme word a tab glues to it (``(a)<TAB>Bearer tok``: the canary's run takes the scheme word),
    with a word on its record after its blanks; else ``None``. A value never ends between a scheme
    word and its token, whatever stands before the tab (basecradle-router#502's review)."""
    tab = text.rfind("\t", floor, end)
    if tab < 0 or not _SCHEME_WORD.fullmatch(text, tab + 1, end):
        return None
    if (blanks := _SCHEME_BLANKS.match(text, end)) is None or not _reached(text, blanks.end()):
        return None
    return blanks.end()


def _run_end(text: str, at: int, stop: int) -> int:
    """Where the canary's run from `at` ends, read no further than `stop`."""
    return _CANARY_RUN.match(text, at, stop).end() if at < stop else stop


def _value_end(text: str, start: int, limit: int | None = None) -> int:
    """Where a credential's value from `start` ends: past every word the value reaches, each read
    whole as a shell word, and past the canary's run.

    A value reaches its first word, after any tabs it opens with, unless it opens with a marker
    where the canary's run would end after it, which is the whole value. From a word it reaches:

    1. **Across a tab** to the next word, while the word is a lead word (`_LEAD_WORD`): the canary's
       run crosses a tab, so a word after one is still the value's (basecradle-router#502). The next
       word must start where a run could go on (`_reached`).
    2. **Across blanks** to the next word, while the word is a bearer scheme word (`_SCHEME_WORD`),
       so a doubled scheme word goes with the token after both (issue #926), and wherever the value
       would end just after a scheme word a tab glues to it, whatever stands before the tab
       (`_token_after_a_glued_scheme_word`). A scheme word with nothing after it on its record is
       the value alone.
    3. **Into a value of its own** at a credential name's ``=`` outside every quote in the word
       (``x,DB_PASSWORD=<TAB>…``): the redactor's next search starts past that ``=``, so the name's
       value is read here, by this same reading, and goes with this one (the review of #926).

    Each word is read once, the canary's run once from each place the value goes on past it, and the
    walk stops where a word's shell reading runs past its lead word's tab: the value already reaches
    that far. So the value is linear in its length. With `limit`, the result is the router's end or
    `limit`, whichever is less: every scan stops at `limit`, and the reading returns as soon as the
    value reaches it."""
    stop = len(text) if limit is None else min(limit, len(text))
    end = run_end = _run_end(text, start, stop)
    at = start
    while end < stop:  # a value from `at`
        floor = at
        if text.startswith(REDACTED_VALUE, at):
            after = at + len(REDACTED_VALUE)
            if after == len(text) or text[after] in _MARKER_ENDS:
                return min(max(end, after), stop)
        if (tabs := _TABS.match(text, at).end()) > at:
            if not _reached(text, tabs):
                return end  # nothing after the tabs is the value's: a blank, a stop or a break
            at = tabs
        while True:  # the words it reaches
            word_end, at_a_name = _shell_word_end(text, at, stop)
            end = max(end, word_end)
            if at_a_name or end >= stop:
                break  # a credential name's `=`: its value is read as one from here
            if _SCHEME_WORD.fullmatch(text, at, word_end) and (
                blanks := _SCHEME_BLANKS.match(text, word_end)
            ):
                token = blanks.end()
                if _reached(text, token):  # a word after the blanks, on the same record
                    at = token
                    if at >= run_end:
                        run_end = _run_end(text, at, stop)
                        end = max(end, run_end)
                    continue
            lead = _LEAD_WORD.match(text, at)
            if lead is None or word_end > lead.end() or not _reached(text, lead.end()):
                # No lead word, one the shell reads past, or nothing after its tabs: the value ends,
                # unless it ends just after a scheme word a tab glues to it.
                if (token := _token_after_a_glued_scheme_word(text, floor, end)) is None:
                    return end
                at = floor = token
                if at >= run_end:
                    run_end = _run_end(text, at, stop)
                    end = max(end, run_end)
                continue
            at = lead.end()
        at = word_end
        if at >= run_end:
            run_end = _run_end(text, at, stop)
            end = max(end, run_end)
    return stop


def _credential_spans(text: str) -> Iterator[tuple[int, int]]:
    """Every credential-named assignment's value, as its start (the character after the ``=``) and
    its end (`_value_end`), in order; an empty value's two are equal. Each name is found once, and
    the next search starts where its value ends."""
    if "=" not in text:
        return
    pos = 0
    while (match := _CREDENTIAL_NAME.search(text, pos)) is not None:
        start = match.end()
        pos = _value_end(text, start)
        yield start, pos


def redact_credentials(text: str) -> str:
    """`text` with every credential-named assignment's value replaced by `REDACTED_VALUE`.

    The name stays. The value (`_value_end`) is every shell word it reaches from the character after
    the ``=`` (`_shell_word_end`), so a quoted value or a ``"$(…)"`` goes whole, and the canary's
    own run, so whatever the canary would read as the value is gone in every record the text is
    written into, a prose line that keeps a tab included. It reaches across a tab after a lead word
    (basecradle-router#502), past a bearer scheme word to its token (issue #926), and into the value
    of a credential name standing in it outside a quote (the review of #926). An empty value (a
    space, a record break or the text's end follows the ``=``) has nothing to hide and is left as it
    is. Linear in `text`, and idempotent: a marker where the canary's run would end after it is a
    whole value. The router's function of the same name, byte for byte in what it returns
    (basecradle-router#506), and one of the rules `scrub` takes the union of."""
    out: list[str] = []
    pos = 0
    for start, end in _credential_spans(text):
        out.append(text[pos:start])
        if end > start:
            out.append(REDACTED_VALUE)
        pos = end
    out.append(text[pos:])
    return "".join(out)


#: One character of the run every **length-gated** redaction rule matches — this module's
#: rules and the box's ``ai_scrub`` alike. A rule with a minimum length (``sk-…{8,}``,
#: ``AIza…{20,}``) cannot redact a credential cut below that minimum, so a cut must never fall inside
#: such a run (`bounded`). On the box, the rules that match a wider charset (bearer, a query value, a
#: heartbeat URL) have no minimum the cut can undercut, so this class is the whole of the rule there.
#: This module's own bearer rule does have one over a wider charset, which is why every log-bound
#: site scrubs **before** it cuts (`scrub`, then `bounded`): no rule here is ever shown a cut text. A
#: new box rule with a minimum length over a wider charset widens this class.
_TOKEN_CHAR = re.compile(r"[A-Za-z0-9_-]")
_TRAILING_RUN = re.compile(r"[A-Za-z0-9_-]+\Z")


def _run_start(text: str, end: int) -> int:
    """Where the ``[A-Za-z0-9_-]`` run that ends at `end` starts (`end` itself when none does),
    read backwards, so backing a cut off a run costs the run's length and no more."""
    start = end
    while start > 0 and _TOKEN_CHAR.match(text, start - 1):
        start -= 1
    return start


def bounded(text: str, cap: int, *, marker: str = "…") -> str:
    """`text` whole if it fits `cap` characters, else a head of it plus `marker` — never cut mid-token.

    **The cut never falls inside a ``[A-Za-z0-9_-]`` run** (issue #787): it backs up to the last
    character outside one, so a token is either whole — and redactable by every rule downstream — or
    absent. Truncating first and redacting later is the order a log pipeline runs in (this module
    scrubs, then cuts; the box's ``ai_scrub`` redacts after both), and a credential cut short of its
    rule's minimum length matches no rule at all and ships as a prefix. A text that is one such run
    past the bound is returned **empty**, which `kv` omits — never half a token. The router's
    ``ai_otel_llm`` remap cuts by the same rule (basecradle-router#385), so the fleet's lines are cut
    the same way whoever writes them.

    The result, marker included, is at most `cap` characters. This is the one cut for any free-text
    value bound for a log line, and such a value is `scrub`-bed first; a site that slices one by hand
    is the defect this exists to retire. Linear in the head it keeps, whatever the text holds
    (issue #929): every position is an index into `text`, and the text is sliced once.
    """
    if len(text) <= cap:
        return text
    cut = max(cap - len(marker), 0)
    if _TOKEN_CHAR.match(text, cut):  # the next character continues the head's last run
        cut = _run_start(text, cut)
    cut, value_end = _whole_values(text, cut)
    head = text[:cut]
    stripped = head.rstrip()
    if stripped != head and value_end == len(stripped):
        stripped += " "  # the blank after a value, so the marker is never glued to it
    return stripped + marker if stripped else ""


#: Each literal rule's label and, after it, everything its value may still take: the rule's own
#: pieces, with the value's run allowed to be empty and its bearer alternative left out (a token run
#: never takes a ``…``, so that alternative ends before a marker glued after it). A head ending inside
#: one of these readings, past its label, ends inside a value the rule would read the marker into
#: (`_whole_values`). The heartbeat URL's label is a space-free run up to its first ``/heartbeat/``,
#: since the box's rule reads any run holding one, and the marker, as the URL's secret.
_OPEN_LABELS = (
    re.compile(f"(?P<label>{_X_API_LABEL}){_X_API_OPENER}{_SCHEME_WORDS}{_X_API_RUN}*"),
    re.compile(f"(?P<label>{_QUERY_LABEL}){_SCHEME_WORDS}{_QUERY_RUN}*"),
    re.compile(f"(?P<label>{_API_KEY_LABEL}){_API_KEY_OPENER}{_SCHEME_WORDS}{_API_KEY_RUN}*"),
    re.compile(r"(?<!\S)(?P<label>\S*?" + _HEARTBEAT + r")\S*"),
)


def _whole_values(text: str, cut: int) -> tuple[int, int | None]:
    """`cut`, an index into `text`, backed off before every value it cuts that a later scrub reads.

    `bounded` treats a value as one token: one whose value the cut reaches goes whole, its name or
    label with it. Otherwise a later scrub (`ScrubbingFormatter` scrubs every line again) reads the
    cut marker or the ``…`` as the value and writes ``[redacted]`` or a whole marker in its place,
    past the bound when the label was short, and always with no ``…`` to say the text was cut. A
    value the cut does not reach is kept. There are two kinds:

    - **A credential-named assignment** (issue #918), reached when the cut halves its
      `REDACTED_VALUE` or its value runs to the cut's very edge, where the ``…`` would be glued to
      it. The assignments are the ones `redact_credentials` reads, found as it finds them, so a name
      inside another's value (``A_TOKEN='B_TOKEN=x y …``) is that value's text and never stands for
      it.
    - **A literal rule's label** (issue #931): ``?token=``, ``api_key=``, an ``x-api-key:`` header
      or a heartbeat URL, reached when the head, its trailing blanks gone, ends past the label and
      inside what its value may take (`_OPEN_LABELS`): ``?token= tail`` cut to ``?token=…`` scrubs
      to ``[redacted]``, ten characters for a cap of nine. A blank between the label and the marker
      would keep the label for some rules and not for others (an ``x-api-key:`` value and a scheme
      word's token are read across blanks), so a reached label goes whole, as a name does.

    Every reading goes no further than the first cut, and each search goes on from where the last
    reading ended, so each rule reads the head once, as `scrub` reads the text (issue #929: backing
    off by re-slicing the text and re-searching it, once per name, was quadratic). The cut then
    walks back over what was found, latest first, each item checked against the cut as it stands
    then; an item past the cut is already gone. A marker that opens with a blank reads less than
    these readings assume, so it can cost text that could have stayed (a label, with any token run
    glued before it), never keep a label that should go. Returns the cut and where the value of the last assignment it keeps ends, or
    ``None`` when it keeps none."""
    found: list[tuple[int, int, int]] = []  # an assignment's name start, -1 and its value end
    pos = 0
    while (name := _CREDENTIAL_NAME.search(text, pos, cut)) is not None:
        pos = _value_end(text, name.end(), cut)
        found.append((name.start(), -1, pos))
        if pos >= cut:
            break  # this value reaches the cut, so no name after it stands before the cut
    labels = (
        ((m.start(), m.end("label"), m.end()) for m in rule.finditer(text, 0, cut))
        for rule in _OPEN_LABELS
    )  # each label's start, its end, and where the reading after it ends
    items = list(heapq.merge(found, *labels))
    end = _head_end(text, cut)
    kept: int | None = None
    for start, label_end, reach in reversed(items):
        if start >= cut:
            continue  # cut away with a value after it
        if label_end < 0:
            if reach < cut:  # an assignment the cut does not reach: the latest one sets the blank
                if kept is None:
                    kept = reach
                continue
        elif not label_end <= end <= reach:
            continue  # a label the head does not end in the value of
        cut = _run_start(text, start)
        end = _head_end(text, cut)
        kept = None
    return cut, kept


def _head_end(text: str, cut: int) -> int:
    """Where the head ``text[:cut]`` ends once `str.rstrip` has taken its trailing blanks, read
    backwards, so it costs the blanks and no more."""
    while cut > 0 and text[cut - 1].isspace():
        cut -= 1
    return cut


def scrub(text: str) -> str:
    """`text` with every credential shape the scrub knows redacted.

    Five rules, each reading `text` as it came: the heartbeat URL (`_HEARTBEAT_RUN`), the literal
    rules (`_X_API_KEY`, `_QUERY`, `_API_KEY`, `_BEARER`), the value of every credential-named
    assignment (`_credential_spans`, which `redact_credentials` renders alone), and the prefix rules
    (`_PREFIXED`). **What the scrub takes is the union of what the rules take** (issue #929): the
    spans they match are merged where they overlap or touch, and each merged span is written once —
    as `REDACTED_VALUE` when it starts where a credential name's value starts, the name kept, and
    as ``[redacted]`` otherwise. A prefixed match `_is_word` reads as a word is no span.

    The union is why no rule can take less beside another than alone, which passes run one after
    another could not promise: a pass shown an earlier pass's ``[redacted]`` reads a different text,
    and every leak of that kind is recorded at `_X_API_KEY`. So ``scrub`` redacts everything
    `redact_credentials` redacts, and everything each other rule redacts on its own. A literal label
    ending at the ``=`` of a name the assignment rule found keeps it (``OPENAI_API_KEY=…``,
    ``?ACCESS_TOKEN=…``, ``X-API-KEY=…``), its value then one with the assignment's.

    **The marker reads back as itself**: a second scrub of the text `scrub` returned leaves an
    assignment's `REDACTED_VALUE` as it is, when that text has no record break, which is every text
    a harness record holds (`_flatten`, `_one_line`). It is read as a shell word, so it can grow on a
    second scrub only from what was put after it since, and then only toward redaction: a quote
    left open at a raw break, or the closing quote a ``!r`` rendering glues on (``"…X_TOKEN='<a
    redacted value>'" to Vertex`` loses `` to Vertex``). (A literal rule can still read its own
    ``[redacted]`` as a value, ``?token=&key=…``, as it could before #918.)

    Run on a whole text **before** `bounded` cuts it, so a rule is never shown a credential cut
    short of its own minimum. `kv` does both itself; a site that cuts a text for a log line before
    `kv` sees it calls this first. Linear in `text`: each rule reads it once, and the spans, each
    rule's already in order, are merged in one pass.
    """
    values: set[int] = set()  # where each credential name's value starts, empty ones included
    assignments: list[tuple[int, int]] = []
    for start, end in _credential_spans(text):
        values.add(start)
        if end > start:
            assignments.append((start, end))
    spans = heapq.merge(
        assignments,
        (_heartbeat_span(match) for match in _HEARTBEAT_RUN.finditer(text)),
        _x_api_key_spans(text),
        _label_spans(_QUERY, text, values),
        _label_spans(_API_KEY, text, values),
        _bearer_spans(text),
        (match.span() for match in _PREFIX_RULES.finditer(text) if not _is_word(match)),
    )
    out: list[str] = []
    pos = 0
    pending = next(spans, None)
    while pending is not None:
        start, end = pending
        marked = start in values
        pending = next(spans, None)
        while True:
            while pending is not None and pending[0] <= end:
                end = max(end, pending[1])
                pending = next(spans, None)
            if not marked or (grown := _past_marker(text, end)) == end:
                break
            end = grown  # what a second reading of the marker would take, and the spans it meets
        out += [text[pos:start], REDACTED_VALUE if marked else "[redacted]"]
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _past_marker(text: str, at: int) -> int:
    """Where a value ends once `scrub` has written its marker for `text` up to `at`: `at` itself when
    a marker ends a value there (`_MARKER_ENDS`), else as far as reading the marker again would go.

    A span another rule ends can end where a value would not (``OPENAI_API_KEY=x-api-key:
    `$'sk-)``: the header's value stops at the ``'``), and a marker written there is read again
    with what follows it as one shell word. So the span takes what that reading takes: the rest of
    the word the marker begins, then the value of a credential name it ends at, or the token after a
    scheme word a tab glues into it, each read as `_value_end` reads one. A value `_value_end` reads
    ends where a marker ends one, so a marker `scrub` writes reads back as itself."""
    if at == len(text) or text[at] in _MARKER_ENDS:
        return at
    word_end, at_a_name = _shell_word_end(text, at)
    if at_a_name:
        return _value_end(text, word_end)
    token = _token_after_a_glued_scheme_word(text, at, word_end)
    return word_end if token is None else max(word_end, _value_end(text, token))


def _label_spans(rule: re.Pattern[str], text: str, values: set[int]) -> Iterator[tuple[int, int]]:
    """Every match of a literal rule, its label left out when a credential's value starts after it
    (`scrub`). In order, since each starts no earlier than the match before it ends."""
    labelled = "label" in rule.groupindex
    for match in rule.finditer(text):
        label_end = match.end("label") if labelled else -1
        yield (label_end if label_end in values else match.start()), match.end()


def _x_api_key_spans(text: str) -> Iterator[tuple[int, int]]:
    """Every ``x-api-key`` header's span, read on through a header its value opens with.

    A value can end in the next header's own label (``x-api-key: "x-api-key": "K9…"``, whose value
    is ``x-api-key`` up to the quote; ``x-api-key: abc-x-api-key: K9…``, whose value ends at
    ``x-api-key:``), and the header after it would then be matched by nothing, since each search
    starts past the last match (the review of #929). So while a header whose label ends where the
    span does reaches further, it is read too, and the span ends where the last one does. Only such
    a label can: a header starting earlier in the value stops where the value stops, at the same
    space, quote, backslash or ``&``, and a bearer header that crosses one is the bearer rule's.
    The label must end where the span does, or before its separator (``:``, ``=>``) and the one
    scheme word an inner value may open with glued to it (``x-api-key: x-api-key:Bearer K9…``,
    whose value stops at the blank after ``Bearer``), so only `_X_API_LABEL_REACH` characters are
    searched for it, and the rule stays linear."""
    pos = 0
    while (match := _X_API_KEY.search(text, pos)) is not None:
        start, end, floor = match.start(), match.end(), match.start("value")
        while True:
            tail = range(max(floor, end - _X_API_LABEL_REACH), end)
            inner = next(
                (m for i in tail if (m := _X_API_KEY.match(text, i)) and m.end() > end), None
            )
            if inner is None:
                break
            end, floor = inner.end(), inner.start("value")
        yield start, end
        pos = end


def _bearer_spans(text: str) -> Iterator[tuple[int, int]]:
    """Every bearer header's span: its scheme word, any scheme words after it, and its token
    (`_bearer_end`). Matches are read leftmost-first, as the box's regular expression reads them:
    one that starts inside a span already walked is not walked again, so a scheme word inside a
    walked token (``Bearer 'a Bearer "eyJ' x"``) is that token's text, and the rule stays linear."""
    reach = -1
    for match in _BEARER.finditer(text):
        end = _bearer_end(text, match) if match.start() >= reach else match.end()
        reach = max(reach, end)
        yield match.start(), end


def _bearer_end(text: str, match: re.Match[str]) -> int:
    """Where a bearer header `match` found ends, once a run of scheme words has its token.

    The token a quote or a substitution opens is the shell word it opens (`_shell_word_end`). While
    the word the token is part of ends in the scheme word, in any case, as the rule's own scheme
    word may be glued to anything (``Bearer x.Bearer 'eyJ…'``, ``Bearer a=Bearer eyJ…``,
    ``Bearer "q"bearer 'eyJ…'``), the word after the whitespace that follows it is a token too,
    read the same way; the last token read is where the header ends (issue #929). A scheme word with
    no token after it ends nothing. Each word is read once, so the walk is linear in its length."""
    end, at = match.end(), match.end("scheme")
    while True:
        if (escaped := _ESCAPED_TOKEN.match(text, at)) is not None:
            word = escaped.end()
            end = max(end, word)
        elif text.startswith(_TOKEN_OPENERS, at):
            word = _shell_word_end(text, at)[0]
            end = max(end, word)
        else:
            if (token := _BEARER_TOKEN.match(text, at)) is not None:
                end = max(end, token.end())
            word = _GLUE_RUN.match(text, at).end()
        if word - at < len("bearer") or text[word - len("bearer") : word].lower() != "bearer":
            return end
        if (gap := _BEARER_GAP.match(text, word)) is None:
            return end
        at = gap.end()


def _heartbeat_span(match: re.Match[str]) -> tuple[int, int]:
    """The span of a space-free run holding a heartbeat URL that the box's rule would redact."""
    run = match.group()
    start = run.index(_HEARTBEAT)
    scheme = _SCHEME.search(run, 0, run.rindex(_HEARTBEAT, 0, len(run) - 1))
    if scheme:
        start = min(start, scheme.start())
    bearer = _TRAILING_BEARER.search(run, start)
    return match.start() + start, match.end() - (len(bearer.group()) if bearer else 0)


def _heartbeat(match: re.Match[str]) -> str:
    """A space-free run holding a heartbeat URL, redacted from where the box's rule would begin."""
    start, end = _heartbeat_span(match)
    text = match.string
    return text[match.start() : start] + "[redacted]" + text[end : match.end()]


def _redaction(match: re.Match[str]) -> str:
    return match.group() if _is_word(match) else "[redacted]"


def _is_word(match: re.Match[str]) -> bool:
    """Whether a prefixed match is part of a word rather than a credential (issue #794).

    Glued to a letter, digit or combining mark (any script), **and** its run after the prefix holds
    no digit and no uppercase letter — never one without the other, because a serialization can glue
    a key to a letter, and a key's run always holds a digit or a capital with overwhelming
    certainty. A glued word whose run holds one is still redacted: the scrub errs toward redaction.
    Only the prefix rules ask; every other rule redacts unconditionally.

    **A match that took a glued key is never a word** (issue #858): its glued tail can hold a later
    key that was no part of any word (``xhf_…ghp_…-sk-…``, whose ``sk-`` follows a ``-``), and judged
    as one run, all lowercase, it would be kept. No word holds a whole key by a rule's own shape.
    """
    if any(name.startswith("glued") for name, value in match.groupdict().items() if value):
        return False
    if match.start() == 0:
        return False
    if unicodedata.category(match.string[match.start() - 1])[0] not in "LNM":
        return False
    run = match.group()[_PREFIX.match(match.group()).end() :]
    return not any(c.isdigit() or c.isupper() for c in run)


#: The record format every harness line is rendered with — the severity token, a space, the
#: message. It is a *contract*, not a preference: neither emitter on an agent box sets a syslog
#: priority (the wake-runner wraps the harness in ``systemd-cat`` with no ``--priority``), so
#: every line reaches journald at the same priority and **severity survives only as this text
#: token inside the message**. The fleet's ``level`` column extracts it (basecradle-noc#207), so a
#: line rendered any other way has no severity anywhere. Named here rather than inlined at the one
#: `logging.basicConfig` call so a synthetic line (`_log_grammar`) can wear the same envelope the
#: real one does instead of hand-spelling it.
LOG_FORMAT = "%(levelname)s %(message)s"


class Rendered(str):
    """Text the harness composed for a record — a head, a `kv` rendering, a whole structured line.

    The one mark `ScrubbingFormatter` trusts (issue #912): an argument of a harness record that is
    `Rendered` (or a number) is interpolated as written, and **every other argument is foreign** —
    flattened, scrubbed and quoted (`_Foreign`) — because that is the only way a formatter can tell
    the harness's own ``kv(...)`` from a vendor's error text once both are strings. So the safe
    default is the unmarked one: an argument nobody thought about is quoted, never trusted.

    `head`, `kv` and `_full_value` return one, and so does every builder of a whole structured line
    (`_report.billing_onset_line`, `_breaker.breaker_tripped_line`, …). Concatenating one with plain
    text yields plain text, which is the point: ``"x " + kv(...)`` is no longer only the harness's
    words, so a builder that composes a line wraps its result itself.
    """

    __slots__ = ()


#: The bytes one journal record may hold (issue #912). journald splits a stream line longer than its
#: ``LineMax=`` (48K by default) into several records, and every one after the first would begin
#: with whatever text stood at the cut — an unspoken narration is never bounded, so a model could
#: choose it. So a record longer than this is split by the formatter instead (`_records`), into a
#: first record that keeps its own head and continuations that each carry one of the harness's
#: (``continued 2/3: "…"``). Measured in UTF-8 bytes, the unit journald counts in, and set well under
#: the default so a box that lowers ``LineMax`` by less than a third still splits nothing itself.
RECORD_BYTES = 32 * 1024

#: What one continuation record of a split may spend on its envelope — the severity, the
#: ``continued i/n: `` head and the quotes — so its content can be sized before ``n`` is known.
_CONTINUATION_RESERVE = 64


class ScrubbingFormatter(logging.Formatter):
    """The formatter every harness entrypoint's handler writes through: one scrubbed line per record.

    Two floors, each over the whole record as rendered, so no call site has a rule to remember.

    **No credential** (issue #857). `kv` scrubs each value it renders, but a prose line interpolates
    its arguments itself (``_log.warning("…: %s", exc)``), and an exception's text is foreign: a
    vendor's 401 body can echo the key it rejected. So the whole record is scrubbed here — message,
    traceback and stack, a future prose site and another library's line included. A `kv` line is
    scrubbed a second time, which changes nothing a value already lost, and **never across a value's
    closing quote** (`scrub_line`): a quoted value ending in ``/heartbeat/`` or ``?token=`` holds no
    secret, and on the whole line the rule that follows it would take the quote and leave foreign
    text outside it. The traceback and stack are scrubbed apart from the message, by `_full_value`,
    for the same reason. One over-redaction is stated: a bare value ending in ``bearer`` beside a
    next field of eight or more characters goes with it, as it does on the box.

    **One line, and text the harness did not write never at a head's position** (issue #912). The
    fleet recognises a contract line by its head where the head stands — the record's start, after
    the severity (basecradle-noc#1518) — so whatever controls a record's start can forge one. Every
    record is therefore rendered as exactly one line whose start is the harness's:

    - **a harness record's arguments are foreign unless `Rendered`** (`_prose_args`): a vendor's
      error text, a tool's output, a model-chosen name is flattened, scrubbed and quoted when it holds
      a space, an ``=`` or a ``"`` — the value rule of `kv`, unbounded — so a newline in it cannot
      start a record and a ``key=`` in it cannot read as a field. A bare token (a model slug, a uuid,
      a count) reads as it did;
    - **a traceback or stack rides at the line's end** as one quoted string (``… "Traceback (most
      recent call last): …"``), never as the lines Python would print;
    - **another library's record** is the library's logger name and its message, quoted
      (``WARNING httpx: "HTTP Request: …"``), because its text may interpolate anything at all;
    - **a line break that survives** (one in a harness template) becomes a space (`_one_line`);
    - **a record longer than `RECORD_BYTES`** is split by the formatter into continuations that
      each carry a harness head (`_records`), never by journald at a byte nobody chose.

    None of this adds a head or a field to the telemetry contract: every new rendering is prose
    (`docs/telemetry.md` → Prose Lines). A clean line — every argument `Rendered`, a number, or a
    bare token — renders byte for byte as `logging.Formatter` renders it.

    It protects only a process whose handler the harness installed (`log_handler`): a library caller
    that configures its own handler gets `kv`'s scrub and the sites that scrub their own argument
    (`_openai.ModelCatalog.rows`), and can install this formatter itself.
    """

    def format(self, record: logging.LogRecord) -> str:
        return "\n".join(self._records(record, _encodable(self._line(record))))

    def _line(self, record: logging.LogRecord) -> str:
        """The record as one line: its message in the envelope, scrubbed, then its traceback and
        stack, each scrubbed on its own (`_full_value`) so no rule reaches between them."""
        record.message = self._message(record)
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        line = scrub_line(_one_line(self.formatMessage(record)))
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        for text in (record.exc_text, record.stack_info and self.formatStack(record.stack_info)):
            if text:
                line += " " + _full_value(text)
        return line

    @staticmethod
    def _message(record: logging.LogRecord) -> str:
        """The message: a harness record's template over its rendered arguments, or another library's
        logger name and its whole message, quoted."""
        if not _is_harness(record.name):
            return f"{_logger_name(record.name)}: {_full_value(record.getMessage())}"
        message = str(record.msg)
        if not record.args:
            return message
        if isinstance(record.args, Mapping) and "%(" not in message:
            return message % (_Foreign(record.args),)  # ``"%s" % mapping`` renders the mapping
        return message % _prose_args(record.args)

    def _records(self, record: logging.LogRecord, line: str) -> list[str]:
        """`line` as the records journald will keep — itself, or split under `RECORD_BYTES`.

        The first record keeps the line's own head and is closed with a ``"`` when the cut fell
        inside a quoted value, so a reader counting quotes stays in step; each continuation is
        ``continued i/n: "…"`` in the record's own envelope, its piece quoted with its ``"`` turned
        to ``'`` as `kv` turns them. A cut never falls inside a ``[A-Za-z0-9_-]`` run unless one run
        is half the budget long (`_cut`), so a credential the box redacts by length reaches it whole.
        """
        if len(line) * 4 <= RECORD_BYTES or len(line.encode()) <= RECORD_BYTES:
            return [line]
        first, rest = _cut(line, RECORD_BYTES - 1)  # room for the quote that may close it
        pieces = []
        while rest:
            piece, rest = _cut(rest, RECORD_BYTES - _CONTINUATION_RESERVE)
            pieces.append(piece)
        records = [first + '"' if first.count('"') % 2 else first]
        for index, piece in enumerate(pieces, start=2):
            continuation = logging.makeLogRecord(record.__dict__)
            quoted = '"' + piece.replace('"', "'") + '"'
            continuation.message = f"continued {index}/{len(pieces) + 1}: {quoted}"
            records.append(self.formatMessage(continuation))
        return records


def scrub_line(line: str) -> str:
    """`line` scrubbed between its values' closing quotes, never across one.

    A ``"`` closes a value when it is the second, fourth, … in the line: every ``"`` a rendered
    record holds is the harness's own, because `kv` and the foreign renderers turn the text's own
    into ``'``. So a rule's match ends at a closing quote, and an opening quote stays with the text
    after it, where ``api_key="…`` is still read whole."""
    parts, start = [], 0
    for count, match in enumerate(re.finditer('"', line), start=1):
        if count % 2 == 0:
            parts.append(line[start : match.start()])
            start = match.start() + 1
    parts.append(line[start:])
    return '"'.join(scrub(part) for part in parts)


def _encodable(line: str) -> str:
    """`line` with any lone surrogate as the ``\\udXXX`` text the stream would write for it, so the
    record can be measured in UTF-8 bytes and cut (`_records`). A lone surrogate is legal in a
    Python string (``json.loads`` makes one of ``"\\ud83d"``) and has no UTF-8 encoding."""
    try:
        line.encode()
    except UnicodeEncodeError:
        return line.encode("utf-8", "backslashreplace").decode()
    return line


#: A logger name that is one token, which stands at a record's head as it is. Any other name is
#: quoted, because a logger is named by the code that makes it and can be named from data.
_LOGGER_NAME = re.compile(r"[A-Za-z0-9_.:-]+")


def _logger_name(name: str) -> str:
    name = _flatten(name)
    return name if _LOGGER_NAME.fullmatch(name) else '"' + name.replace('"', "'") + '"'


#: The logger hierarchy whose records are the harness's own. Every other logger's record is another
#: library's, rendered whole as foreign text (`ScrubbingFormatter._message`).
_HARNESS_LOGGER = "basecradle_harness"


def _is_harness(name: str) -> bool:
    return name == _HARNESS_LOGGER or name.startswith(_HARNESS_LOGGER + ".")


def _prose_args(args: Any) -> Any:
    """A harness record's arguments with every foreign one wrapped (`_Foreign`)."""
    if isinstance(args, Mapping):
        return {key: _prose_arg(value) for key, value in args.items()}
    return tuple(_prose_arg(value) for value in args)


def _prose_arg(value: Any) -> Any:
    """`value` as written when it is the harness's (`Rendered`) or cannot hold text (a number,
    ``None``), and wrapped as foreign otherwise — so a ``%d`` still gets its number."""
    if value is None or isinstance(value, (Rendered, numbers.Number)):
        return value
    return _Foreign(value)


class _Foreign:
    """A foreign argument, rendered by `kv`'s value rule without its bound when ``%s`` or ``%r``
    asks for it: flattened, scrubbed, and quoted when it holds a space, an ``=`` or a ``"``."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __str__(self) -> str:
        return _prose_value(str(self.value))

    def __repr__(self) -> str:
        return _prose_value(repr(self.value))


def last_word(error: object) -> str:
    """An error's text for a command's direct write to stderr: one line, scrubbed, and quoted when
    it holds a space, an ``=`` or a ``"`` (issue #912).

    A command's last word goes past every handler, straight to the stream a box's journal reads,
    so it gets the formatter's rule for a foreign argument by hand: a newline in it cannot start a
    record and a ``key=`` in it cannot read as a field. A traceback (``traceback.format_exc()``) is
    one line too, read whole."""
    return _prose_value(str(error))


def _prose_value(text: str) -> str:
    text = scrub(_flatten(text))
    if " " in text or "=" in text or '"' in text:
        return '"' + text.replace('"', "'") + '"'
    return text


#: One character that would break a record, or a run of them with the spaces around it: every C0 and
#: C1 control character and DEL, and the Unicode line and paragraph separators — **except an ESC
#: that opens an SGR sequence**, which is the harness's own palette (`head`, `_field`). A foreign
#: value never carries one past `_flatten`, which removes every control character.
_BREAK = re.compile(r"[ ]*(?:(?:\x1b(?!\[[0-9;]*m)|[\x00-\x1a\x1c-\x1f\x7f-\x9f  ])[ ]*)+")


def _one_line(text: str) -> str:
    """`text` with each run of record-breaking characters (`_BREAK`) as one space, trimmed at the
    end — a clean text returned unchanged."""
    return _BREAK.sub(" ", text).rstrip(" ") if _BREAK.search(text) else text


def _cut(text: str, budget: int) -> tuple[str, str]:
    """`text` cut where its head fits `budget` UTF-8 bytes: ``(head, rest)``.

    The cut backs out of a ``[A-Za-z0-9_-]`` run, as `bounded` does, unless that would give up half
    the budget (only a run that long, which no credential is, is ever cut inside), and never splits
    an SGR escape."""
    head = text.encode()[:budget].decode(errors="ignore")
    if len(head) == len(text):
        return text, ""
    if _TOKEN_CHAR.match(text, len(head)):
        backed = _TRAILING_RUN.sub("", head)
        if len(backed.encode()) * 2 >= budget:
            head = backed
    escape = head.rfind("\x1b")
    if escape != -1 and "m" not in head[escape:]:
        head = head[:escape]
    return head, text[len(head) :]


def log_handler(fmt: str = LOG_FORMAT, stream: Any = None) -> logging.Handler:
    """The stderr handler every harness entrypoint logs through — `ScrubbingFormatter` over `fmt`.

    The one way an entrypoint configures logging (``logging.basicConfig(handlers=[log_handler()])``),
    so no entrypoint can write an unscrubbed record; `tests/test_prose_scrub.py` holds every
    ``basicConfig`` call in the package to it.
    """
    handler = JournalHandler(stream)
    handler.setFormatter(ScrubbingFormatter(fmt))
    return handler


class JournalHandler(logging.StreamHandler):
    """A `logging.StreamHandler` whose report of its own failure is one line, on its own stream.

    `logging.Handler.handleError` prints a traceback, the call stack and the record's whole message
    and arguments to ``sys.stderr``, over many lines, **while the handler's lock is held**. In a wake
    ``sys.stderr`` is a pipe the harness reads back through this very handler (`_journal`), so that
    print waits on a reader that waits on the lock: the wake hangs, and a broken stream (``EPIPE``,
    ``ENOSPC``) turns every record into another one (issue #912). So the report is one line,
    ``ERROR A log record could not be written. "…"``, written to the handler's own stream, and when
    that stream is what failed, nothing is written at all.
    """

    def handleError(self, record: logging.LogRecord) -> None:
        if not logging.raiseExceptions:
            return
        error = sys.exc_info()[1]
        # The stream is what failed when this raises, and there is nowhere left to say so.
        with contextlib.suppress(Exception):
            self.stream.write(
                f"ERROR A log record could not be written. {_full_value(repr(error))}"
                + self.terminator
            )
            self.flush()


def install_log_handler() -> None:
    """For an entrypoint that keeps Python's default logging: the same output, scrubbed.

    With no handler configured, a ``WARNING`` reaches stderr through `logging.lastResort` as its
    bare message, unscrubbed. This installs the scrubbing handler in its place, at the same level
    and in the same format, so the command's stderr reads as it did, minus any credential. A no-op
    when logging is already configured, so an embedding application's setup always wins.
    """
    logging.basicConfig(level=logging.WARNING, handlers=[log_handler("%(message)s")])


#: The env var naming this agent's slug when the OS user is not it. The fleet's universal-identity
#: rule makes the OS user, the GitHub App bot, and the platform handle **one** slug, so the login
#: name is the right default and this is the escape hatch for a box that departs from it.
SLUG_ENV = "BASECRADLE_AGENT_SLUG"

#: The env var the router's wake-runner exports so a wake can echo the delivery that spawned it
#: (basecradle-router#170). Optional by contract: absent → the correlation field is simply
#: omitted, so the harness and the router ship independently.
DELIVERY_ID_ENV = "BASECRADLE_DELIVERY_ID"

#: The token-count fields, mapped from every shape a vendor SDK reports them in — OpenAI
#: Responses (``input_tokens``/``output_tokens``), the Chat wire and OpenRouter
#: (``prompt_tokens``/``completion_tokens``), and the xAI protos (both, as attributes). Each
#: candidate is a *path* — one hop for a flat field, two for a nested one (the cached count lives
#: under a details block on the HTTP wires and flat on the xAI proto). First hit wins, so one
#: reader serves every adapter and a provider that reports nothing logs nothing.
#:
#: **Every field means one thing on every vendor** (`docs/telemetry.md` → ``llm``, issue #759): a
#: vendor whose own numbers mean something else — Anthropic's prompt in three disjoint parts,
#: Google's tool-use prompt reported apart, xAI's reasoning reported apart from its completion —
#: is translated into this vocabulary once, at its own adapter's boundary, and never branched on
#: here. ``tokens_total`` is not read at all: it is ``tokens_in`` + ``tokens_out`` by definition
#: (`token_counts`), so no vendor's own total can disagree with the two fields beside it.
_TOKEN_FIELDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "tokens_in": (("input_tokens",), ("prompt_tokens",)),
    "tokens_out": (("output_tokens",), ("completion_tokens",)),
    # Prompt caching is the difference between paying full freight on a 750k-token prefix and
    # paying the cache-read rate for it (~5.4× cheaper, measured live) — and until it is on the
    # line, whether caching is working at all can only be *inferred*. OpenRouter's own
    # ``supports_implicit_caching`` metadata reads ``false`` on endpoints where caching
    # demonstrably works, so the reported count is the only trustworthy witness (issue #274).
    "cached_tokens": (
        ("prompt_tokens_details", "cached_tokens"),  # the Chat wire: OpenAI, OpenRouter
        ("input_tokens_details", "cached_tokens"),  # OpenAI Responses
        ("cached_prompt_text_tokens",),  # the xAI gRPC proto
    ),
    # The part of the prompt written to the cache on this call — billed at its own rate, so a
    # cache that is written on every call and read on none is visible as the cost it is.
    "cache_write_tokens": (
        ("prompt_tokens_details", "cache_write_tokens"),  # the Chat wire: OpenAI, OpenRouter
        ("input_tokens_details", "cache_write_tokens"),  # OpenAI Responses
    ),
    # The part of the output spent reasoning, on every purpose (`_REASONING_FIELDS`).
    "tokens_reasoning": (
        ("completion_tokens_details", "reasoning_tokens"),  # the Chat wire: OpenAI, OpenRouter
        ("output_tokens_details", "reasoning_tokens"),  # OpenAI Responses
        ("reasoning_tokens",),  # flat: the xAI gRPC proto
    ),
}

#: The fields `token_counts` renders, in the `llm` line's order: ``tokens_total`` sits between
#: the two it sums, where the line has always carried it.
_TOKEN_ORDER = (
    "tokens_in",
    "tokens_out",
    "tokens_total",
    "cached_tokens",
    "cache_write_tokens",
    "tokens_reasoning",
)

#: Where a provider states the call's **dollar cost**, when it states one at all. OpenRouter
#: returns it on every response (``usage.cost``, already in USD), as do some OpenAI-compatible
#: vendors reached under the generic profile (Wiro, issue #670); the xAI SDK reports it in
#: ``ticks`` and converts it with its own helper, so that adapter passes the figure to
#: `log_llm_call` directly rather than through this reader. A provider that reports tokens but no
#: dollars logs no ``cost=`` — with **three** stated exceptions, the vendors whose APIs state no price
#: at all: Google's Vertex AI, OpenAI and Anthropic, whose adapters price the call from the vendor's
#: published rates (`basecradle_harness._google_rates`, `basecradle_harness._openai_rates`,
#: `basecradle_harness._anthropic_rates`; the capital's rulings on issues #655, #657 and #725),
#: because a brain with no cost at all would be invisible on the
#: fleet's spend dashboard. Those tables omit rather than guess — an unknown model or tier gets no
#: ``cost=`` — and they hand the figure over as a `ComputedCost`, so the line says which kind of
#: figure it is (`COST_BASIS_COMPUTED`). No other adapter derives one: a stale table is worse than an
#: absent field.
_COST_FIELDS: tuple[tuple[str, ...], ...] = (("cost",),)

#: The ``cost_basis=`` value a line carries when its ``cost=`` is the harness's arithmetic over a
#: vendor's published rates rather than a figure the vendor stated (issues #655, #656, #657). **The
#: fleet's existing literal, never a second one**: the NOC's Steel launcher already tags its computed
#: browser cost ``cost_basis=computed`` (basecradle-noc#797), so a computed dollar reads the same
#: wherever it comes from. Rendered immediately after ``cost=``, as the Steel line renders it, and
#: only when ``cost=`` itself is rendered — a basis with no figure is a claim about nothing. A
#: vendor-stated cost carries no ``cost_basis``, so every such line is byte-identical to what it was.
COST_BASIS_COMPUTED = "computed"


class ComputedCost(float):
    """A dollar figure the harness **computed** from a vendor's published rates.

    A plain ``float`` everywhere it is used — summed, formatted, compared — and the one bit of
    provenance it carries is its type: `log_llm_call` and `log_tool` render
    ``cost_basis=computed`` beside any cost that is one. The provenance travels *with the figure*
    rather than as a second argument because a figure passes through hands that never knew it was
    computed: a non-brain caller writes its own line from what `capture_llm_call` recorded (the
    describer, the reranker), and a basis threaded as a parameter is exactly the field such a
    caller forgets — at which point a computed dollar reads, on the dashboard, as one the vendor
    stated. Arithmetic on it returns a plain ``float`` by design: only the rate tables construct one,
    at the end, from a figure they are prepared to stand behind.
    """

    __slots__ = ()


def cost_basis(cost: Any) -> str | None:
    """``COST_BASIS_COMPUTED`` for a `ComputedCost` that will render, ``None`` otherwise."""
    return COST_BASIS_COMPUTED if isinstance(cost, ComputedCost) and _money(cost) else None


#: Where OpenRouter says a call ran on **the operator's own provider credential** (bring-your-own-
#: key, issue #653) — and where it then puts the dollars. On such a call the provider bills the
#: inference directly, so ``usage.cost`` reads ``0`` (OpenRouter's credits were not spent on it)
#: and the provider's charge moves to ``usage.cost_details.upstream_inference_cost``. A non-BYOK
#: call carries no ``is_byok`` at all (measured 2026-10-06), so absence reads exactly as before.
#: The two figures are never summed: on a BYOK call ``usage.cost`` is not a second charge for the
#: same tokens, and OpenRouter's BYOK fee is taken from credits without being reported per call.
_BYOK_FIELDS: tuple[tuple[str, ...], ...] = (("is_byok",),)
_UPSTREAM_COST_FIELDS: tuple[tuple[str, ...], ...] = (("cost_details", "upstream_inference_cost"),)

#: The ``billing=`` value a BYOK call carries on its `llm` line: the dollar figure in ``cost=`` is
#: on the provider's own invoice, not on OpenRouter's credits (issue #653). **A stable literal** —
#: the NOC's extraction keys on it — and the only value the field takes: a call billed the ordinary
#: way carries no ``billing=``, so its line is byte-identical to what it was before the field
#: existed.
BYOK_BILLING = "byok"

#: Where a provider states how many of the completion's tokens went to **reasoning** — the same
#: paths `_TOKEN_FIELDS` reads for ``tokens_reasoning``, one tuple, so `reasoning_tokens` and the
#: line can never disagree. Until issue #759 the field was read only by the non-brain callers and
#: never reached a ``purpose=main`` line; it is now part of the shared reader, on every purpose.
_REASONING_FIELDS: tuple[tuple[str, ...], ...] = _TOKEN_FIELDS["tokens_reasoning"]


#: Where a provider names the **upstream that actually served the call**. A router is not a server:
#: OpenRouter fronts ~27 distinct endpoints for a single model id, and they differ by up to 10× in
#: context ceiling and 5.4× in prompt price — so ``provider=openrouter`` alone cannot say what a
#: call ran against, or why two identical-looking calls cost different money (issue #274).
#:
#: This is the response's **routing metadata**: the endpoint list OpenRouter routed over, with the
#: one it picked flagged ``selected``. It is the *documented* answer, and it is opt-in — it appears
#: only when the request asks for it (``X-OpenRouter-Metadata: enabled``), which the OpenRouter
#: cells send on every call.
#:
#: **What is deliberately NOT read: the response's top-level ``provider`` field** (issue #280). It is
#: undocumented — absent from OpenRouter's own OpenAPI schema — and it does not mean what its name
#: suggests. It names *the last upstream OpenRouter spoke to*, which is **not** the serving endpoint
#: whenever a server-side tool ran: with the ``openrouter:web_search`` built-in active, a live
#: ``z-ai/glm-5.2`` call returns ``"provider": "OpenAI"`` — a vendor that serves no endpoint in that
#: model's pool — while the routing metadata correctly reports ``StreamLake``. Reading it did not
#: merely lose data, it **fabricated a distribution**: @glm-5.2 logged ``endpoint=OpenAI`` on every
#: search-enabled wake. Better to omit the field than to log a wrong one — the same rule cost obeys.
#:
#: Read as a **capability, not a vendor branch**: every adapter asks this of whatever its SDK
#: returned, and a direct-to-vendor SDK — where the vendor *is* the endpoint — finds nothing here
#: and the field is simply omitted.
_ROUTING_METADATA: tuple[str, ...] = ("openrouter_metadata", "endpoints", "available")


def color_enabled() -> bool:
    """Is the journal colored on this run? Read per line, never cached at import.

    Per-call on purpose: the config home's ``agent.env`` is loaded *after* this module imports, so
    a value frozen at import would answer for the environment the interpreter started in rather
    than the one the agent actually runs under — and it would make the setting untestable without
    reloading the module.
    """
    return not os.environ.get(NO_COLOR_ENV, "")


def agent_slug(explicit: str | None = None) -> str:
    """This agent's slug: an explicit argument, else ``BASECRADLE_AGENT_SLUG``, else the OS user.

    The fleet's universal-identity rule makes an agent's slug **one** string across every system it
    touches — its GitHub App bot, its OS user and home, its platform handle — so the login name is
    the correct default here rather than a guess. The env var is the escape hatch for a box that
    departs from it (a laptop, a shared account), and the argument is for a caller that already
    knows.

    It lives beside `delivery_id` because it is the same kind of value: a correlation identity
    resolved from the environment the process was started in, read by the claims emitter (as a
    manifest ``subject``) and by `_log_grammar` (as the ``agent=`` field the fleet's ``agent``
    label falls back to when the journald identifier is not a wake's).
    """
    if explicit:
        return explicit
    from_env = os.environ.get(SLUG_ENV, "").strip()
    if from_env:
        return from_env
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no passwd entry / no LOGNAME: name it rather than crash
        return "unknown"


def head(text: str, color: str) -> Rendered:
    r"""A line's leading **verdict** token, colored as one contiguous span.

    The token-integrity rule, and it is the whole of the convention: the escape goes *outside* the
    token, never inside it — ``\x1b[32mwake start\x1b[0m timeline=…``, never
    ``wake \x1b[32mstart\x1b[0m``. The searchable bytes stay adjacent, so every substring search
    that worked on the plain line still matches the colored one: ``journalctl | grep 'wake start'``,
    a Better Stack Live Tail filter, a ClickHouse ``match()``. Splitting a token would break every
    one of them silently — nothing errors, the line still *looks* right, and the query that used to
    find it simply returns nothing forever.

    **What this buys, stated precisely, because the gap is where a consumer breaks.** It preserves a
    search for **one whole token**. It does *not* preserve a pattern spanning the gap *between* two
    tokens — a colored head is followed by ``RESET`` and then the separator, so an adjacent-pair
    literal (``wake reported_failure kind=billing``) or a trailing-space anchor (``wake failed ``)
    no longer matches, while the same pattern written with a wildcard across the gap
    (``wake end .*outcome=error``) still does. Anything outside this repo that greps these lines is
    coupled to that distinction; see the color inventory in `README.md` for the lines this applies
    to.

    Which lines get a head, and it is a rule rather than a list: **a head is colored when it states
    a verdict about a unit of work** — did it start, end, fail, get skipped, get blocked, recover —
    and left plain when it names a *fact*: ``llm``, ``tool``, ``unspoken``, ``posted``,
    ``step``, ``memory``, ``context …``. Color everything and the signal is gone; and those
    fact-heads are also the anchors the fleet dashboard extracts on (`` llm provider=``,
    `` tool name=``, `` unspoken timeline=``), which a head span would sit inside.
    """
    return Rendered(f"{color}{text}{RESET}" if color_enabled() else text)


def kv(**fields: Any) -> Rendered:
    """``key=value`` pairs in the order given, dropping the ones with nothing to say.

    ``None`` and ``""`` are omitted — an absent delivery id or trigger leaves no empty
    ``delivery=`` behind — while ``0`` is kept, because "posted nothing" is a fact worth logging.

    Values are rendered by `_value`, never interpolated raw, because **some of them are not the
    harness's text**: a tool's exception message, an SDK refusal, a provider's error body. Left
    raw, such a value could put a newline in the middle of a record (splitting one leveled line
    into unleveled fragments a severity filter shows decapitated), paint the reader's view with an
    ESC of its own, forge a field by containing ``outcome=ok``, run to the length of a whole
    response body, or carry a credential. A parser
    reading this stream must be able to trust that the fields it sees are the fields the harness
    wrote.

    A **verdict** field (`_VERDICT_VALUES`) is colored — the whole ``key=value`` pair, by the same
    token-integrity rule `head` keeps, so ``outcome=ok`` stays greppable as ``outcome=ok`` and never
    becomes ``outcome=`` plus a colored ``ok``. The color is applied to the *rendered* pair, after
    `_value` has flattened, scrubbed, bounded, and quoted it — so the escape can never end up inside
    a value, and a hostile error text still cannot forge one: a tool that returns the literal
    ``outcome=ok`` gets it rendered as a quoted `error` value, which is not a verdict field and is
    not colored.
    """
    colored = color_enabled()
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        rendered = _value(value)
        if rendered == "":
            continue
        parts.append(_field(key, rendered, colored=colored))
    return Rendered(" ".join(parts))


def _field(key: str, rendered: str, *, colored: bool) -> str:
    """One rendered ``key=value`` pair, colored whole when it is a verdict and plain otherwise.

    ``colored`` is resolved once per line by `kv` rather than re-read here, so a line is never half
    colored — and so a wake's hottest formatting path does not read the environment once per field.
    """
    pair = f"{key}={rendered}"
    vocabulary = _VERDICT_VALUES.get(key) if colored else None
    color = vocabulary.get(rendered) if vocabulary else None
    return f"{color}{pair}{RESET}" if color else pair


def _value(value: Any) -> str:
    """One field's value: single-line, redacted, bounded, and quoted when it isn't a bare token.

    Four jobs, in order — flatten (`_flatten`: one line, no control character), scrub (`scrub`),
    truncate (`MAX_VALUE`, by `bounded`, never inside a token), then quote anything holding a space
    or an ``=`` so it reads as *one* value rather than as extra fields. Scrub before the cut, so every
    shape this module knows is redacted whole; the cut's own rule protects the shapes only the box
    knows. A plain uuid, duration, or count passes through untouched, which is what keeps the common
    line greppable.
    """
    text = bounded(scrub(_flatten(value)), MAX_VALUE)
    if text and (" " in text or "=" in text or '"' in text):
        return '"' + text.replace('"', "'") + '"'
    return text


def log_unspoken(text: str | None, *, timeline: str | None = None, kind: str = "narration") -> None:
    """Journal a turn's **unspoken** text — the agent's narration, which no peer will ever see.

    The Unspoken Channel's whole delivery mechanism (issue #293), and its one line:

        unspoken timeline=<uuid> kind=narration chars=412 text="…"

    `kind` names *which* unspoken text this is, because they read very differently after the fact:
    ``narration`` (the ordinary end of a turn), ``reserve`` (the step-cap progress report the model
    writes for its own next turn), and ``stuck`` (the canned note when even the reserve call failed).

    **Rendered in full, and that is the point.** Every other value in this stream is bounded by
    `MAX_VALUE`, because every other value is a breadcrumb to something recoverable elsewhere — a
    tool's error text is still in the transcript, a response body is still the provider's. This one
    is not: an unspoken narration exists *only* here. Bounding it would quietly convert "full
    visibility" into "the first 240 characters of visibility", which is not the trade the founder's
    principle names. So it is flattened to one record (a severity filter must never see a decapitated
    fragment), scrubbed of credential shapes (an agent can paste a key into its own prose), quoted so
    it reads as one value — and never truncated.

    **Nobody is watching this line, and the agent is told so.** It is a flight recorder: written
    always, read rarely (a forensic dig), and fed to the agent's own memory, which is its real
    reader. The harness must never imply otherwise to the model — an agent that thinks its log has an
    audience will escalate *into* it, and an escalation nobody reads is a message to no one.
    """
    body = text or ""
    _log.info(
        "unspoken %s text=%s",
        kv(timeline=timeline, kind=kind, chars=len(body)),
        _full_value(body),
    )


def _full_value(value: str) -> Rendered:
    """`_value`'s rendering minus the cap: flattened, scrubbed, quoted — never truncated.

    The three jobs that make a value *safe* in a log stream are kept (one record per line; no
    credential; one field, not several). The fourth — bounding it — is deliberately dropped for the
    one field that has nowhere else to live (see `log_unspoken`). Quoted unconditionally, because a
    narration is prose: it will contain spaces, and it may well contain an ``=``.
    """
    text = scrub(_flatten(value))
    return Rendered('"' + text.replace('"', "'") + '"')


#: One C0 or C1 control character, or DEL: Unicode's ``Cc`` category exactly, which is what the
#: router's ``logfmt._CONTROL`` and the ``\p{Cc}`` of its ``ai_otel_llm`` remap match, so the fleet's
#: three renderers flatten the same bytes (issue #806).
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _flatten(value: Any) -> str:
    """`value` as one line holding no control character: each becomes a space, then every run of
    whitespace is one space, trimmed at both ends.

    **A control character in a value is never legitimate.** A newline would split one record into
    unleveled fragments; an ESC (a vendor's error body can quote one) would paint the reader's
    terminal, or Better Stack's rendered view, with a color this module did not choose, so the line
    no longer means what its bytes appear to say. The harness's own color (`head`, `_field`) is
    applied after a value is rendered, so it is never removed by this.
    """
    return " ".join(_CONTROL.sub(" ", str(value)).split())


def delivery_id() -> str | None:
    """The router's delivery-correlation id for this wake, or ``None`` when it exported none.

    The router sets `DELIVERY_ID_ENV` on the wake child so one delivery's router-side and
    harness-side lines can be joined in Live Tail. Optional-when-absent by design: a wake run by
    hand, by an older router, or by anything else simply logs without the field.
    """
    return (os.environ.get(DELIVERY_ID_ENV) or "").strip() or None


#: The three things a model can be doing for an agent — the **log category**, a closed set
#: (issue #485, @origin's audit of the fleet dashboard 2026-09-09). Three names in three places
#: and never mixed: the **log** says the category (``memory``); the **UI** says the human name
#: (Memory System — the NOC's business, not this package's); the **software** (MemPalace, Gemini)
#: appears only as a *field value* (``provider=mempalace``, ``model=google/…``). The word
#: "mempalace" therefore never names a category or a line head again.
MAIN = "main"
MEMORY = "memory"
HELPER = "helper"

#: The verdict on one model-call attempt: the ``outcome=`` **every** `llm` line carries (issue
#: #782). Before it, one unit of work (a model call) was described on four heads at four outcomes
#: (`llm`, ``llm retry``, ``llm refused``, and nothing at all for a brain call that raised), so a
#: producer with no wake had nowhere to put a failure. Now the head says *what* (a model call) and
#: this field says *how it ended*, with the fields each outcome requires (`docs/telemetry.md` →
#: ``llm`` → Outcomes):
#:
#: - ``ok``: the vendor answered and the answer was used.
#: - ``refused``: the vendor refused the content. That is its final word, and the call was billed.
#: - ``fallback``: the vendor answered and the caller judged the answer unusable. This is a
#:   non-brain caller only, with an answer slug in `ANSWER_REASONS`.
#: - ``retrying``: the attempt failed and will be re-issued. It returned no usage, so its line
#:   carries no cost.
#: - ``failed``: the attempt failed and nothing will re-issue it, so the call ended with no answer.
OK = "ok"
REFUSED = "refused"
FALLBACK = "fallback"
RETRYING = "retrying"
FAILED = "failed"

#: The `reason` of an answer the vendor sent (and billed) that the harness could not read: a body
#: that did not parse, a tool call's arguments cut off. Its billed line says `fallback` with this
#: reason (`read_answer`), and the harness's decision about the call follows on its own line.
UNREADABLE = "invalid_response"

#: The `reason` of an answer the vendor cut off at the output cap (issue #788): on the billed line
#: of an unreadable answer whose stop reason was the cap (`read_answer`), and on the decision line
#: that follows it. Also an answer slug a caller judges by (`ANSWER_REASONS`): the describer's.
TRUNCATED = "truncated"

#: The `reason` slugs that judge an **answer**, not a fault: the vendor answered (and billed), and
#: the caller could not use what came back. Every other reason is a fault slug, so the call it ends
#: is ``failed``: that includes ``internal`` (the harness itself raised, perhaps before any call)
#: and ``undecodable_video`` (the clip never reached a model), because neither is an answer the
#: vendor billed. `call_outcome` is the one place the split is drawn.
#: ``invalid_response`` is not here: on its own it is a fault (no answer reached the caller), and
#: only the billed line of an answer an adapter could not read says ``fallback`` with it (`answered`).
ANSWER_REASONS = frozenset({"truncated", "empty_response", "missing_parts", "parse"})


def call_outcome(reason: str | None) -> str:
    """The `outcome` of a non-brain call whose caller judged it, by the one `reason` it carries.

    No reason means ``ok``. An answer slug means ``fallback``: the call answered, and the caller
    fell back from what it said. Any other reason (a fault slug, or ``config:<cause>``) means
    ``failed``, because there was no answer to judge.
    """
    if reason is None:
        return OK
    return FALLBACK if reason in ANSWER_REASONS else FAILED


@dataclass(frozen=True)
class WorkContext:
    """Where a model call sits in the harness's own work. These are the optional fields of its line.

    ``timeline`` and ``delivery`` say which wake the call belongs to. They are set once by the wake
    (`in_wake`), and ``delivery`` is the router's id (`delivery_id`), which is the wake's identity
    wherever one exists. ``step`` is the engine step the call was made in, as ``n/max`` (`at_step`),
    so a wake that died at step 5 of 24 says so on the line of the call that killed it. Each is
    absent outside its context: a library caller has no wake, a Turn-0 recall has no step, and a
    Claude Code builder has neither. A producer fills the fields it has and omits the rest.
    """

    timeline: str | None = None
    delivery: str | None = None
    step: str | None = None


#: The ambient `WorkContext`, for `capture_llm_call`'s reason: the `llm` line is written inside the
#: provider adapter, layers below the wake and the engine that know where the call sits, so
#: threading two parameters through every adapter would be the alternative.
#: The default is one shared, frozen instance: nothing can mutate it in place.
_NO_WORK = WorkContext()
_WORK: ContextVar[WorkContext] = ContextVar("llm_work", default=_NO_WORK)


@contextmanager
def in_wake(timeline: str | None) -> Iterator[None]:
    """Stamp every model call made inside the block with its wake's ``timeline`` and ``delivery``."""
    token = _WORK.set(WorkContext(timeline=timeline, delivery=delivery_id(), step=_WORK.get().step))
    try:
        yield
    finally:
        _WORK.reset(token)


@contextmanager
def at_step(step: int, max_steps: int) -> Iterator[None]:
    """Stamp every model call made inside the block with the engine step it was made in."""
    work = _WORK.get()
    token = _WORK.set(WorkContext(work.timeline, work.delivery, f"{step}/{max_steps}"))
    try:
        yield
    finally:
        _WORK.reset(token)


def _work_fields() -> dict[str, str | None]:
    work = _WORK.get()
    return {"timeline": work.timeline, "delivery": work.delivery, "step": work.step}


@dataclass
class LlmCall:
    """What an adapter recorded about a call whose line its **caller** will write.

    Everything the caller needs that only the adapter knows — the endpoint a router picked, the
    usage the vendor reported, the cost it stated, how long the call took, the id the vendor gave
    the call (issue #634), and **why the vendor stopped generating**. Every field stays ``None`` when the call raised before the adapter got
    that far, so the caller's line simply carries less rather than carrying a guess.

    `finish_reason` is the one field here that is **not** rendered on the line, and that is
    deliberate (issue #488). It is not an observation about the call, it is the evidence a caller
    judges the *answer* with: a describer whose clip description stopped at ``length`` did not
    describe the clip, and the line it writes says so as ``reason=truncated outcome=fallback``,
    which is the fact worth grepping. Rendering the raw vendor word as well would add a field to a
    byte-frozen grammar to say the same thing in three vendors' spellings.
    """

    provider: str | None = None
    model: str | None = None
    seconds: float | None = None
    usage: Any = None
    endpoint: str | None = None
    residency: str | None = None
    cost: float | None = None
    #: Why a call the vendor billed carries no computed `cost` (issues #874, #878), for the line's
    #: ``cost_gap``.
    cost_gap: str | None = None
    finish_reason: str | None = None
    generation_id: str | None = None
    #: ``refused`` when the vendor refused the content (issue #782), with its own ``category``, so a
    #: capturing caller's one line can say so. ``None`` for every call the vendor answered.
    outcome: str | None = None
    category: str | None = None
    #: What the caller is using the model for: the ``purpose=`` of its line. Set by
    #: `capture_llm_call`.
    purpose: str = HELPER


#: The capture handle for the model call currently in flight, or ``None`` — which is the ordinary
#: case, the brain's own call, emitted by the adapter as it always was. It is ambient rather than
#: an argument because `log_llm_call` is reached **inside the provider adapter**, layers below the
#: caller: a non-brain caller reaches a model through the brain's own adapter family (which is what
#: keeps them one error taxonomy), so threading a parameter would mean touching every adapter. Set
#: only by `capture_llm_call`, always with try/finally, around exactly one call.
_CAPTURE: ContextVar[LlmCall | None] = ContextVar("llm_capture", default=None)


@contextmanager
def capture_llm_call(purpose: str = HELPER) -> Iterator[LlmCall]:
    """Hold back the `llm` line an adapter would emit, so its caller writes the one line instead.

    The seam a **non-brain** model call wraps its provider call in (issue #485). Inside this block
    `log_llm_call` **records instead of emitting**, and the caller emits once — with a
    ``purpose=``, and with the one field the adapter cannot supply: the **outcome**.

    That split is the whole reason this exists. An adapter's line is written the instant the vendor
    answers, and only the caller can tell an answer that is *usable* from one that is empty or
    malformed. Letting the adapter emit and then adding a second line for the bad answer would put
    **two** `llm` lines on one attempt — double-counting the call, double-counting its dollar, and
    (worse) leaving the good path with no ``outcome=`` at all, since an adapter has no notion of
    one. One attempt, one line, one outcome.

    Yields the `LlmCall` the adapter fills; if the call raises, it comes back empty. `purpose` is
    the caller's (``helper`` for the describer, the only capturing caller), carried on the record so
    a line the adapter itself writes for the call — a refusal — names the same purpose.

    **Wrap exactly one model call, on this thread, and nothing else.** Three properties of an
    ambient capture fail invisibly otherwise, and they are stated rather than left to be
    rediscovered: a *second* call inside the block overwrites the first's record (a lost field); a
    call made for some *other* purpose inside the block is captured and **never emitted at all** (a
    lost line, which is worse); and a `ContextVar` does not cross a thread, so an adapter that
    dispatched its request to a worker would emit its own untagged line while the caller emitted a
    second. No shipped adapter does any of the three — that is the assumption to re-check before
    adding one.
    """
    call = LlmCall(purpose=purpose)
    token = _CAPTURE.set(call)
    try:
        yield call
    finally:
        _CAPTURE.reset(token)


def log_llm_call(
    *,
    provider: str,
    model: str,
    seconds: float | None,
    purpose: str | None = None,
    kind: str | None = None,
    usage: Any = None,
    endpoint: str | None = None,
    residency: str | None = None,
    cost: float | None = None,
    cost_gap: str | None = None,
    finish_reason: str | None = None,
    outcome: str | None = None,
    category: Any = None,
    attempt: str | None = None,
    reason: str | None = None,
    detail: str | None = None,
    extra: Mapping[str, Any] | None = None,
    generation_id: str | None = None,
    level: int = logging.INFO,
) -> None:
    """**One `llm` line per model-call attempt, whatever the outcome** — the whole grammar.

    Every provider adapter calls this around its own SDK call, so the LLM leg of a wake is
    visible on every provider rather than only where someone remembered to instrument it. Since
    issue #485 it is also the *only* emitter of a model call, for every purpose: the brain, the
    MemPalace reranker, and the blind-model describer all land here, in one grammar, with
    ``purpose=`` naming what the model was doing for the agent.

    Before that they were three families built three ways — the brain on this head, the reranker on
    a private ``mempalace rerank`` head, and the describer on a plain ``llm`` line **indistinguishable
    from the brain's**, so a second model's spend read as the first's. One grammar is what lets a
    dashboard split them; the fields are spelled identically across all three so one grep syntax
    reads them all.

    **And one line whatever the outcome** (issue #782): ``outcome`` is on every line, ``ok`` when
    nobody judged the call. A retried attempt (`log_llm_retry`), a refusal (``outcome=refused`` with
    the vendor's ``category``), a caller's ``fallback`` and a final ``failed`` all ride this head,
    each with the fields its outcome requires (`docs/telemetry.md` → ``llm`` → Outcomes), so a
    producer with no wake has somewhere to put a failure. The wake context (`WorkContext`) rides
    every line made inside a wake.

    **The head is byte-frozen at** ``llm provider=``. It is the fleet dashboard's anchor for the
    whole family (and the denominator of its extraction alarm), so a purpose is added *inside* the
    line and never spelled as a new head.

    **Field order is the contract** and matches what the NOC's column regexes were written against:
    ``provider purpose kind endpoint residency model duration tokens_in tokens_out tokens_total
    cached_tokens cache_write_tokens tokens_reasoning cost cost_basis cost_gap outcome reason detail`` then any purpose-specific `extra` (the reranker's ``surface``/``pool``/
    ``picked``), and last of all ``generation_id`` — the vendor's own id for the call (issue #634),
    rendered last so that no column written before it moves — and then, after even that,
    ``billing=byok`` on a call the provider billed directly (issue #653, `BYOK_BILLING`), for the
    same reason. `kv` drops whatever is ``None``, so a call with nothing to report is byte-identical
    to what it always was apart from ``purpose=main``. An ``extra`` that carries a ``generation_id``
    of its own (a fallback line's `diagnostics`) is rendered in that same place, and the argument
    wins when both are given: one key, one value, one position.

    Four of the fields are **capabilities, answered by whoever can**: token counts
    (`token_counts`), the cached-prompt count that says whether caching is doing anything, the
    ``endpoint`` that names the upstream a router actually dispatched to (`serving_endpoint`), and
    the ``cost`` in dollars *as the provider reported it* (`reported_cost`, or the vendor SDK's own
    converter). Each is omitted, cleanly, by a provider with no answer — a direct-to-vendor SDK has
    no serving endpoint distinct from itself, and most vendors report tokens but never dollars. A
    provider that reports none of them still gets its provider/model/duration line, which is the
    part that is always true. ``endpoint`` is the serving upstream **only**; where a vendor runs a
    call is ``residency`` (issue #759, retiring #655's reading of ``endpoint`` on Google): Vertex's
    location (``residency=us``, ``residency=global``) and Anthropic's ``inference_geo``. ``global``
    is the vendor's choice of where to run, not a place, so it says no more than that. A cost
    that is computed (a `ComputedCost`, Google's or OpenAI's) is followed by ``cost_basis=computed``;
    a vendor-stated one carries no basis, so its line is unchanged (`COST_BASIS_COMPUTED`).
    ``cost_gap`` stands where those two would on a billed call an adapter that computes a cost could
    not price, in the reason its once-per-wake ``cost unpriced`` line names (issues #874, #876,
    #878), so each such call is counted rather than inferred from that one line.

    **A usage block of nothing but zeros is not usage** (issue #488), so the token fields and the
    ``cost`` read out of it are omitted rather than printed as zeros. That is the same honest-absence
    rule `endpoint` and `cost` already keep, applied to the one shape that *looks* like an answer:
    a broken stream returned ``tokens_in=0 tokens_out=0 … cost=0`` beside ``outcome=ok``, which
    reads on the dashboard as a free call that worked. A genuinely free call is unaffected — it
    reports real token counts, so its ``cost=0`` is a fact and still prints.

    `finish_reason` is recorded for a `capture_llm_call` caller and **never rendered**: see
    `LlmCall` for why the line says ``reason=truncated`` instead.

    A call with no explicit `purpose` is the brain's (``purpose=main``, no ``kind``) — unless a
    `capture_llm_call` block is active, in which case this **records rather than emits** and its
    caller writes the line. `level` is the caller's, because severity is the *taxonomy* — a
    config-class reranker fault is ERROR and a runtime one WARNING, while a call that worked is
    INFO.
    """
    captured = _CAPTURE.get()
    if captured is not None and purpose is None:
        # Inside `capture_llm_call`: record what only this adapter knows and let the caller write
        # the line, once, when it can also state the outcome. See `capture_llm_call` for why. An
        # explicit `purpose` is a caller writing its own line and is never captured.
        captured.provider, captured.model, captured.seconds = provider, model, seconds
        captured.usage, captured.endpoint, captured.cost = usage, endpoint, cost
        captured.cost_gap = cost_gap
        captured.residency = residency
        captured.finish_reason, captured.generation_id = finish_reason, generation_id
        captured.outcome, captured.category = outcome, slug(category)
        return
    if purpose is None:
        purpose = MAIN
    if outcome is None:
        # Every line states its verdict (issue #782): a call no one judged was answered and used.
        outcome = OK
    counts = token_counts(usage)
    if usage is not None and not usage_reported(usage):
        # A usage block of nothing but zeros is not usage (issue #488) — it is the vendor reporting
        # **nothing**, in a shape that renders as a fact. Printing it is worse than omitting it: a
        # broken stream came back as ``tokens_in=0 … cost=0 outcome=ok``, which reads on the
        # dashboard as a free call that worked.
        #
        # A **zero** dollar figure goes with the counts, because on the endpoint that states one it
        # is read out of that same block (`_COST_FIELDS`) — it is not an independent claim, it is
        # the same absence spelled ``0``. A **non-zero** one is kept: the xAI adapter reads its cost
        # off the *response* rather than off usage, so a real charge beside an unreported usage
        # block is a fact this line must not swallow. Honest absence cuts both ways.
        counts = {}
        cost = cost if _charged(cost) else None
    _log.log(
        level,
        "llm %s",
        kv(
            provider=provider,
            purpose=purpose,
            kind=kind,
            endpoint=endpoint,
            residency=residency,
            model=model,
            duration=None if seconds is None else _secs(seconds),
            **counts,
            cost=_money(cost),
            cost_basis=cost_basis(cost),
            # Only on billed tokens with no cost: a figure that exists is never also reported
            # missing, and a call that reported no usage has nothing a price was missing for.
            cost_gap=(
                cost_gap
                if counts and _money(cost) is None and outcome not in (RETRYING, FAILED)
                else None
            ),
            outcome=outcome,
            category=slug(category),
            attempt=attempt,
            reason=reason,
            detail=detail,
            **_trailing(extra, generation_id, usage),
        ),
    )


def _trailing(
    extra: Mapping[str, Any] | None, generation: str | None, usage: Any
) -> dict[str, Any]:
    """The `llm` line's tail: `extra`, the wake context, ``generation_id``, then ``billing``.

    The wake context (``timeline``, ``delivery``, ``step``: `WorkContext`, issue #782) follows the
    caller's own extras, so every field the NOC already extracted stays ahead of it.

    ``billing`` names who bills the ``cost=`` on the line, and is stated only where the vendor said
    the call ran on the operator's own key (`byok`). It is rendered after everything, so on a BYOK
    line no field the NOC already extracts moves, and every other line is byte-identical to what it
    was before the field existed. Read off `usage`, never passed: one carried in `extra` is
    replaced rather than splatted beside it, for `_generation_last`'s reason.
    """
    # The context fills only what it knows, so a caller's own value for a key it leaves empty stands.
    known = {key: value for key, value in _work_fields().items() if value is not None}
    fields = _generation_last({**(extra or {}), **known}, generation)
    fields.pop("billing", None)
    fields["billing"] = BYOK_BILLING if byok(usage) else None
    return fields


def _generation_last(extra: Mapping[str, Any] | None, generation: str | None) -> dict[str, Any]:
    """`extra` with ``generation_id`` moved to the end — `generation` winning over one it carried.

    Merged as a dict rather than splatted beside a keyword, for `log_llm_retry`'s reason: a
    `diagnostics` dict in `extra` carries a ``generation_id`` too, and a second value under one
    keyword is a ``TypeError`` raised from inside an ``except`` block in a wake.
    """
    fields = dict(extra or {})
    carried = fields.pop("generation_id", None)
    fields["generation_id"] = generation if generation is not None else carried
    return fields


def log_llm_retry(
    *,
    provider: str,
    model: str,
    purpose: str,
    attempt: str,
    reason: str | None,
    next_in: float,
    kind: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """One WARNING `llm` line, ``outcome=retrying``, per failed attempt about to be retried.

    Since issue #782 a retry is an **outcome of the call, not a head of its own**: it was ``llm
    retry``, which kept a refused attempt out of every column keyed on `` llm provider=`` by having
    a word in between. The NOC now counts a call as a line on that head whose ``outcome`` is not
    ``retrying`` or ``failed`` (basecradle-noc#1212), which is the same split stated as a verdict
    instead of as a spelling, and which a builder or any other producer can write too.

    What it records is still the only trace that a wake waited: the final line carries
    ``outcome=ok`` and a ``duration=`` that silently includes the sleeps, so without this line a
    recovered rerank is indistinguishable from a fast one. It carries no tokens and no cost,
    because the attempt returned no usage to price (which is not a claim that the vendor billed
    nothing: `docs/telemetry.md` → Outcomes).

    ``attempt`` arrives pre-rendered as ``"1/3"`` rather than as two numbers, because the *other*
    counter on this line is OpenRouter's own (``routing_attempt=``) and two attempt counters under
    one key is a line nobody can read. ``diagnostics`` is `basecradle_harness._retry.diagnostics`,
    the vendor's own account of the fault, built by one function so that these fields and the ones
    on the final ``outcome=failed`` line can never disagree. ``extra`` is the call site's own
    trailing context (the timeout budgets, ``surface=`` for a rerank, ``subject=`` for a describe).

    **WARNING, always.** A retry is a degradation that recovered: not an error, and not routine.
    """
    # Merged as dicts rather than splatted as keywords: a call site whose `extra` happened to carry
    # one of these names would otherwise raise `TypeError: got multiple values` from inside an
    # `except` block in a wake, which is a logging call taking the wake down over a field name.
    log_llm_call(
        provider=provider,
        model=model,
        seconds=None,
        purpose=purpose,
        kind=kind,
        outcome=RETRYING,
        attempt=attempt,
        reason=reason,
        extra={**(diagnostics or {}), "next_in": _secs(next_in), **(extra or {})},
        level=logging.WARNING,
    )


#: Who ran a tool, as the ``executor=`` field of its `tool` line says (`docs/telemetry.md` → Tools).
#: The harness's own tool classes, a tool an MCP server serves through the harness's proxy, and a
#: tool the **vendor** ran server-side inside a model call — three executors, one head.
HARNESS_EXECUTOR = "harness"
MCP_EXECUTOR = "mcp"
SERVER_EXECUTOR = "server"

#: The ``kind=`` a server-side tool's line carries, by the harness name of the capability — only for
#: the shapes the contract names (`docs/telemetry.md` → ``tool`` → ``kind``). A tool absent here
#: (a fetch, a code run, an X search) carries no ``kind``, never an invented one.
SERVER_TOOL_KINDS: Mapping[str, str] = {"web_search": "search.web"}

_SLUG_CHARS = re.compile(r"[^a-z0-9_:.-]+")


def slug(text: Any) -> str | None:
    """A vendor's own code or status as a ``reason=`` slug: lowercased, every other run an ``_``.

    For the codes a vendor already spells as a slug (``max_uses_exceeded``) this is the identity; for
    an enum name (``OUTCOME_FAILED``, ``URL_RETRIEVAL_STATUS_ERROR``) it is the same word in the
    contract's one spelling. ``None`` for nothing to say.
    """
    if text is None:
        return None
    word = _SLUG_CHARS.sub("_", str(text).strip().lower()).strip("_")
    return word or None


def log_tool(
    *,
    name: str,
    executor: str,
    outcome: str,
    seconds: float | None = None,
    reason: str | None = None,
    error: str | None = None,
    provider: str | None = None,
    kind: str | None = None,
    model: str | None = None,
    count: int | None = None,
    cost: float | None = None,
    cost_gap: str | None = None,
    generation_id: str | None = None,
    status: int | str | None = None,
    target: tuple[str, str] | None = None,
    cut_at: int | None = None,
) -> None:
    """**The** `tool` line — one per tool use, whoever executed it (issue #758).

    The engine writes it for a harness tool and an MCP tool (``executor=harness|mcp``, with the run's
    ``duration``); an adapter writes it for a tool its vendor ran inside a model call
    (``executor=server``, no ``duration`` — the `llm` line already timed that call — the uses in
    ``count``, and the ``generation_id`` of the call, so the tool joins to its `llm` line). A harness
    tool that bought vendor work (an image, a transcription, a grounded search) carries that work's
    ``provider``, ``kind``, ``model`` and fee, recorded on its `ToolMeter` while it ran.

    Before this, a tool was two things: the engine's ``tool`` line (calls and errors) and a separate
    ``media`` line (duration and cost), while a server-side tool wrote no tool line at all and its
    counts sat on the `llm` line — so the fleet's *Tool Calls* and *Tool Cost* read two different
    definitions of "tool", and a vendor's server tool was visible only as a larger ``cost=``.

    ``cost`` is the tool's **own fee** — never model tokens, which stay on the `llm` line — rendered
    through the same `_money` formatter as every other dollar, with ``cost_basis=computed`` beside a
    `ComputedCost`. ``cost_gap`` stands where those two would on a fee the harness computes and could
    not, in the reason its once-per-wake ``cost unpriced`` line names, so every unpriced use is
    counted on its own line (issue #880) — never beside a ``cost``. A failed run (``outcome=error``) names why as a slug in ``reason``; when there is
    foreign text to show, ``error`` writes it on a second, WARNING ``tool`` line carrying only
    ``name`` and ``error``, as before. A server-side failure has a vendor code and no text, so it has
    no companion and is itself written at WARNING, where a severity filter still sees it.

    ``status`` is what the far end answered the tool's request with, in its protocol's own
    vocabulary (an HTTP status code, or ``nxdomain``), recorded by `tool_answered`: a fetch that got
    a 404 did its work, so it is ``outcome=ok`` with the 404 on the line, never an error (issue
    #892). ``target`` is what the call asked the platform for, ``(kind, identifier)``, recorded by
    `tool_target`; it rides the WARNING companion only, as ``target=<kind>:<identifier>``, because
    it is the half of a failure's explanation the error text does not carry. ``cut_at`` is the
    length, in characters, a result was cut at, recorded by `tool_cut`: the work was done and the
    agent got the head, so the line stays ``outcome=ok`` and the cut is still countable.
    """
    failed_server = executor == SERVER_EXECUTOR and outcome == "error"
    _log.log(
        logging.WARNING if failed_server else logging.INFO,
        "tool %s",
        kv(
            name=name,
            executor=executor,
            provider=provider,
            kind=kind,
            model=model,
            count=count,
            duration=None if seconds is None else _secs(seconds),
            outcome=outcome,
            reason=reason,
            status=status,
            cut_at=cut_at,
            cost=_money(cost),
            cost_basis=cost_basis(cost),
            cost_gap=cost_gap if _money(cost) is None else None,
            generation_id=generation_id,
        ),
    )
    if error is not None:
        asked = None if target is None else f"{target[0]}:{target[1]}"
        _log.warning("tool %s", kv(name=name, target=asked, error=error))


def log_server_tool(
    *,
    name: str,
    provider: str,
    model: str,
    count: int,
    generation_id: str | None,
    reason: str | None = None,
    cost: float | None = None,
    cost_gap: str | None = None,
) -> None:
    """One `tool` line for a tool the vendor ran server-side: one per model call per tool.

    ``name`` is the harness's name for the capability (``web_search``, ``web_fetch``,
    ``code_execution``), never the vendor's block type; ``count`` is the uses the vendor reported on
    this call; ``reason`` is the vendor's own code for a use that failed, which makes the line
    ``outcome=error``; ``cost_gap`` is why a fee the adapter computes is missing (issue #880).
    """
    log_tool(
        name=name,
        executor=SERVER_EXECUTOR,
        outcome="ok" if reason is None else "error",
        reason=reason,
        provider=provider,
        kind=SERVER_TOOL_KINDS.get(name),
        model=model,
        count=count,
        cost=cost,
        cost_gap=cost_gap,
        generation_id=generation_id,
    )


@dataclass
class ServerToolTally:
    """One server-side tool's uses on one model call, and the first failure the vendor reported."""

    count: int = 0
    reason: str | None = None


def tally_server_tools(uses: Iterable[tuple[str, str | None]]) -> dict[str, ServerToolTally]:
    """``(name, failure)`` pairs — one per use — tallied per tool, in the order first seen.

    ``failure`` is the vendor's code or status for a use that did not complete (falsy for one that
    did); the first one a tool reports becomes its line's ``reason``.
    """
    found: dict[str, ServerToolTally] = {}
    for name, failure in uses:
        tally = found.setdefault(name, ServerToolTally())
        tally.count += 1
        if failure and tally.reason is None:
            tally.reason = slug(failure)
    return found


def log_server_tools(
    tallies: Mapping[str, ServerToolTally],
    *,
    provider: str,
    model: str,
    generation_id: str | None,
    fees: Mapping[str, float | None] | None = None,
    gaps: Mapping[str, str] | None = None,
) -> None:
    """`log_server_tool` for every tool in `tallies` that ran — the adapters' one call site.

    ``fees`` maps a tool to its computed fee and ``gaps`` to the ``cost_gap`` reason a fee the
    adapter computes could not be (issue #880); a tool in neither carries no price on its line.
    """
    for name, tally in tallies.items():
        if tally.count:
            log_server_tool(
                name=name,
                provider=provider,
                model=model,
                count=tally.count,
                generation_id=generation_id,
                reason=tally.reason,
                cost=(fees or {}).get(name),
                cost_gap=(gaps or {}).get(name),
            )


#: Where a **router** counts the web searches it ran for a call: OpenRouter's
#: ``usage.server_tool_use_details.web_search_requests``, on both its native SDK and the ``openai`` SDK
#: aimed at it. OpenRouter folds the searches' fee into ``usage.cost`` and states it nowhere apart, so
#: the count is the fact; the dollars stay on the `llm` line.
_ROUTED_SEARCH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("server_tool_use_details", "web_search_requests"),
)


def routed_web_searches(usage: Any) -> int:
    """How many web searches a router reports it ran on a call — ``0`` when it reports none."""
    value = _first(usage, _ROUTED_SEARCH_FIELDS) if usage is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def log_cost_unpriced(*, provider: str, charge: str, reason: str, model: str | None = None) -> None:
    """One WARNING naming a real charge the harness cannot price (`docs/telemetry.md`).

    The caller states it **once per wake per gap**: a missing dollar is then a counted fact a
    dashboard can compare across vendors, rather than an absence nobody sees or a sentence in each
    vendor's own words. ``charge`` is what carries no price (``container_time``, …) and ``reason``
    why (``not_reported``, …); ``model`` is omitted where the charge is not per model.
    """
    _log.warning(
        "cost unpriced %s", kv(provider=provider, model=model, charge=charge, reason=reason)
    )


class GapLedger:
    """The ``cost unpriced`` gaps one holder has stated, so each is written once per wake.

    A holder lives as long as the wake that built it (an adapter, a tool), so remembering what it
    has stated is the once-per-wake rule `log_cost_unpriced` asks its callers to keep. A gap is what
    the line says, so a second call unpriced for the same reason writes nothing more here; it is
    counted on its own line, by ``cost_gap`` (issue #880).
    """

    __slots__ = ("_stated",)

    def __init__(self) -> None:
        self._stated: set[tuple[str, str, str | None]] = set()

    def state(self, *, provider: str, charge: str, reason: str, model: str | None = None) -> str:
        """Write the gap's ``cost unpriced`` line unless this holder already has; return `reason`."""
        key = (charge, reason, model)
        if key not in self._stated:
            self._stated.add(key)
            log_cost_unpriced(provider=provider, model=model, charge=charge, reason=reason)
        return reason


#: The ``charge=`` and ``reason=`` of a code sandbox's container time, which Anthropic and OpenAI
#: both bill and no response of either states — the one shared spelling of that gap.
CONTAINER_TIME = "container_time"
NOT_REPORTED = "not_reported"

#: The rest of the ``cost unpriced`` vocabulary (`docs/telemetry.md` → ``cost unpriced``), spelled
#: once so every adapter states the same gap in the same words (issue #759): the call's **tokens**
#: and a search's **grounding** fee are the charges; a model the rate table does not carry, a tier
#: (or long-context band) it does not carry, and a host whose prices are not the table's are the
#: reasons. A price read live from the vendor rather than from a table adds one more (issue #844):
#: the price source itself has not been read (issue #874: no longer held unread for the wake).
TOKENS_CHARGE = "tokens"
GROUNDING_CHARGE = "grounding"
#: A harness tool's or a server-side tool's own fee, named for the family of the work its `tool`
#: line's ``kind`` carries (issue #880): a web search's per-call fee, a made or edited image, a
#: transcribed recording.
SEARCH_CHARGE = "search"
IMAGE_CHARGE = "image"
AUDIO_CHARGE = "audio"
UNKNOWN_MODEL = "unknown_model"
UNPRICED_TIER = "unpriced_tier"
UNPRICED_HOST = "unpriced_host"
RATES_UNAVAILABLE = "rates_unavailable"


class ToolMeter:
    """What a harness tool's **billed vendor work** recorded while it ran, for its `tool` line.

    The engine opens one around every tool run (`tool_meter`); a tool that calls a vendor records
    each unit of work on it through `metered` (or `meter_tool`), and the engine writes what was
    recorded onto the run's one `tool` line — the vendor, the shape of the work, the model, and the
    fee. A tool that buys nothing records nothing and its line is the platform tool's line it always
    was. Outside an engine (a library caller running a tool by hand) there is no meter, no tool line,
    and recording is a no-op.

    More than one unit in one run is summed: ``count`` becomes the units, and the fee the total of
    the units that **finished** — a failed unit bought nothing — tagged computed when any part of it
    is the harness's arithmetic, and absent when a finished unit's fee was, because a partial total
    would be a figure nobody can stand behind. The vendor, kind and model are the first unit's, and
    an unpriced total's ``cost_gap`` the first finished unpriced unit's reason (issue #880).

    **Recording can never raise into a wake.** The fee and the count come off a vendor body, often
    in a ``tools/`` overlay tool someone wrote in an afternoon, and the meter is read *after* the
    tool has already acted — so a figure that is not a real, finite, non-negative number (a string,
    a bool, ``nan``) is dropped as it is recorded, as `_money` drops it from every other line, and a
    count that is not a positive ``int`` likewise.
    """

    __slots__ = ("_units", "cut_at", "failure", "status", "target", "verdict")

    def __init__(self) -> None:
        self._units: list[tuple[str, str, str, int | None, float | None, bool, str | None]] = []
        self.failure: Exception | None = None
        #: ``(reason, error)`` from `tool_failed`: this run failed, with no vendor work to blame.
        self.verdict: tuple[str, str] | None = None
        #: The length, in characters, the run's result was cut at, from `tool_cut` (issue #892).
        self.cut_at: int | None = None
        #: What the far end answered with, from `tool_answered` (issue #892).
        self.status: int | str | None = None
        #: ``(kind, identifier)`` the run last asked the platform for, from `tool_target`.
        self.target: tuple[str, str] | None = None

    def record(
        self,
        *,
        provider: str,
        kind: str,
        model: str,
        count: Any = None,
        cost: Any = None,
        failed: bool = False,
        cost_gap: str | None = None,
    ) -> None:
        units = (
            count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else None
        )
        fee = cost if _money(cost) is not None else None
        gap = cost_gap if isinstance(cost_gap, str) and cost_gap and fee is None else None
        self._units.append((provider, kind, model, units, fee, failed, gap))

    def fields(self) -> dict[str, Any]:
        """The `log_tool` keywords this run's recorded work fills (empty when it recorded none)."""
        if not self._units:
            return {}
        provider, kind, model, count, *_ = self._units[0]
        if len(self._units) > 1:
            count = sum(unit[3] if unit[3] is not None else 1 for unit in self._units)
        finished = [unit for unit in self._units if not unit[5]]
        costs = [unit[4] for unit in finished]
        cost: float | None = None
        if costs and all(c is not None for c in costs):
            total = math.fsum(costs)  # type: ignore[arg-type]
            computed = any(isinstance(c, ComputedCost) for c in costs)
            cost = ComputedCost(total) if computed else total
        gap = None if cost is not None else next((u[6] for u in finished if u[6]), None)
        return {
            "provider": provider,
            "kind": kind,
            "model": model,
            "count": count,
            "cost": cost,
            "cost_gap": gap,
        }


_METER: ContextVar[ToolMeter | None] = ContextVar("basecradle_harness_tool_meter", default=None)


@contextmanager
def tool_meter() -> Iterator[ToolMeter]:
    """Open the `ToolMeter` a tool run records its vendor work on — the engine's half."""
    meter = ToolMeter()
    token = _METER.set(meter)
    try:
        yield meter
    finally:
        _METER.reset(token)


def meter_tool(
    *,
    provider: str,
    kind: str,
    model: str,
    count: Any = None,
    cost: Any = None,
    cost_gap: str | None = None,
) -> None:
    """Record one finished unit of vendor work on the running tool's meter (a no-op with none).

    ``cost_gap`` is why a fee the tool computes is missing, in the reason its ``cost unpriced`` line
    names (issue #880); it is dropped beside a fee.
    """
    meter = _METER.get()
    if meter is not None:
        meter.record(
            provider=provider, kind=kind, model=model, count=count, cost=cost, cost_gap=cost_gap
        )


def meter_failure(exc: Exception, *, provider: str, kind: str, model: str) -> None:
    """Record that the running tool's vendor work failed — what it was buying, and the fault.

    The tool usually turns the error into text for the model, so the engine would otherwise read the
    run as a success; the meter is how its ``tool`` line still says ``outcome=error`` with the
    fault's slug and the vendor it failed at. A no-op outside a tool run.
    """
    meter = _METER.get()
    if meter is not None:
        meter.record(provider=provider, kind=kind, model=model, failed=True)
        meter.failure = exc


def tool_failed(text: str, *, reason: str, error: str | None = None) -> str:
    """Record that the running tool failed, and hand back ``text`` for the model, unchanged.

    **A tool that answers the model with its failure has still failed.** Feeding the failure back
    is how a model recovers, and it also leaves the engine holding a plain string, so without this
    the `tool` line reads ``outcome=ok`` for a call that did nothing — and *Tool Outcomes* shows
    green for a tool failing on every call (issue #822). Read the verdict structurally, never off
    the text: a shell command or a fetched page may well begin with the word "Error".

    ``reason`` is a slug from the `tool` line's vocabulary (``docs/telemetry.md``). ``error`` is what
    the WARNING companion carries, the text itself when not given. Vendor work that recorded its own
    failure on the meter outranks this, since it knows the fault's precise slug. A no-op outside a
    tool run, so a tool author can write ``return tool_failed("Error: …", reason="…")`` freely.
    """
    meter = _METER.get()
    if meter is not None:
        meter.verdict = (reason, text.removeprefix("Error: ") if error is None else error)
    return text


def tool_cut(text: str, *, at: int) -> str:
    """Record that the running tool's result was cut at ``at`` characters, and hand back ``text``.

    The work was done and the agent got the head, marked as cut, so the line stays ``outcome=ok``
    and carries ``cut_at=<at>`` (issue #892): a long shell output is countable, and it is not a
    shortfall a person should be paged for, which is what ``degraded`` means to the fleet's alerts.
    ``text`` is returned unchanged. A no-op outside a tool run.
    """
    meter = _METER.get()
    if meter is not None:
        meter.cut_at = at
    return text


def tool_answered(text: str, *, status: int | str) -> str:
    """Record what the far end answered the running tool's request with, and hand back ``text``.

    ``status`` is in the answering protocol's own vocabulary: an HTTP status code (``404``), or
    ``nxdomain`` when DNS answered that the host name does not exist. An answer is the tool's work
    done, whatever it says, so the line stays ``outcome=ok`` and carries the answer as ``status``
    (issue #892): a run of agents guessing URLs is counted by it, not hidden in a failure rate, and
    it writes no WARNING. ``text`` is returned unchanged. A no-op outside a tool run.
    """
    meter = _METER.get()
    if meter is not None:
        meter.status = status
    return text


def tool_target(kind: str, identifier: str) -> None:
    """Record what the running tool is about to ask the platform for, named by its kind.

    Called just before the platform call, with the kind of thing asked for (``timeline``,
    ``message``, ``asset``, ``task``, ``user``, ``webhook_endpoint``, ``webhook_event``) and the
    identifier the call carries, exactly as the model gave it. It is written only if the run fails,
    on the WARNING companion as ``target=<kind>:<identifier>``, because the platform's own text
    (``No record exists for the given UUID.``) does not say which one it could not find (issue
    #892). An identifier is not a secret; the value is scrubbed like every other. A later call in
    the same run replaces an earlier one, so the line names the request that failed. A no-op
    outside a tool run.
    """
    meter = _METER.get()
    if meter is not None:
        meter.target = (kind, identifier)


class MeteredCall:
    """The handle `metered` yields, so a tool can record the charge its vendor's answer states.

    The charge is knowable only *after* the vendor responds and its body is read — inside the block.
    The tool sets ``cost`` (a vendor-stated USD figure, a `ComputedCost`, or leaves it ``None`` when
    neither is possible), ``count`` (billed units, where there is more than one) and, for a fee it
    computes and could not, ``cost_gap`` (the reason, issue #880).
    """

    __slots__ = ("cost", "cost_gap", "count")

    def __init__(self) -> None:
        self.cost: float | None = None
        self.count: int | None = None
        self.cost_gap: str | None = None


@contextmanager
def metered(*, provider: str, kind: str, model: str) -> Iterator[MeteredCall]:
    """Record one unit of a harness tool's vendor work — the tool's half of the meter.

    Wraps the vendor call. On a clean exit the unit is recorded with whatever charge the block set;
    on a raise the meter keeps the failure (so the run's `tool` line says ``outcome=error`` with the
    fault's slug and the vendor and model it was buying from, even when the tool turns the error into
    text for the model) and the error goes through untouched.
    """
    call = MeteredCall()
    try:
        yield call
    except Exception as exc:
        meter_failure(exc, provider=provider, kind=kind, model=model)
        raise
    meter_tool(
        provider=provider,
        kind=kind,
        model=model,
        count=call.count,
        cost=call.cost,
        cost_gap=call.cost_gap,
    )


def token_counts(usage: Any) -> dict[str, int]:
    """The usage object a vendor SDK returned, normalized to the token fields the line carries.

    Reads a mapping (the ``model_dump()`` of an OpenAI/OpenRouter response) or an object with
    attributes (the xAI gRPC protos) with the same code, tries each vendor's spelling in turn
    (`_TOKEN_FIELDS`), and keeps only the fields that came back as real integers. Anything it
    cannot read is left out rather than guessed — an under-reported line is honest; an invented
    token count is not.
    """
    if usage is None:
        return {}
    read: dict[str, int] = {}
    for field, paths in _TOKEN_FIELDS.items():
        value = _first(usage, paths)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        read[field] = value
    if "tokens_in" in read and "tokens_out" in read:
        # Exactly the two it sums, never the vendor's own total (issue #759): Vertex's counts the
        # tool-use prompt and xAI's counts reasoning its completion excludes, so "total" meant
        # three different things on three vendors.
        read["tokens_total"] = read["tokens_in"] + read["tokens_out"]
    return {field: read[field] for field in _TOKEN_ORDER if field in read}


def reasoning_apart(usage: Any) -> dict[str, Any] | None:
    """`usage` from a vendor that reports its reasoning **apart from** its output, in the shared
    vocabulary — the reasoning folded into the output it was billed as (issue #759).

    On most wires the output count already includes the reasoning, and ``reasoning_tokens`` is a
    breakdown of it. xAI reports it beside the output instead, on every surface: its Chat
    Completions example is 32 prompt + 9 completion + 94 reasoning = 135, and its Responses example
    32 + 9 + 110 = 151 (docs.x.ai, read 2026-10-31). Read as-is, ``tokens_out`` would omit the
    reasoning xAI bills as output. So the adapter that knows its vendor does this hands the line
    this translation — the vendor's convention, applied at its own boundary, never a branch in the
    shared reader. ``None`` for no usage, and the counts unchanged when none is reasoning.
    """
    counts = token_counts(usage)
    if not counts:
        return None
    wire: dict[str, Any] = {}
    if "tokens_in" in counts:
        wire["prompt_tokens"] = counts["tokens_in"]
    if "tokens_out" in counts:
        wire["completion_tokens"] = counts["tokens_out"] + counts.get("tokens_reasoning", 0)
    details = {
        name: counts[field]
        for name, field in (
            ("cached_tokens", "cached_tokens"),
            ("cache_write_tokens", "cache_write_tokens"),
        )
        if field in counts
    }
    if details:
        wire["prompt_tokens_details"] = details
    if "tokens_reasoning" in counts:
        wire["completion_tokens_details"] = {"reasoning_tokens": counts["tokens_reasoning"]}
    return wire


def usage_reported(usage: Any) -> bool:
    """Whether the vendor actually **stated** usage for this call (issue #488).

    A usage block of nothing but zeros is not usage: it is the vendor reporting **nothing** in a
    shape that renders as a fact. The live case: a describe logged ``tokens_in=0 tokens_out=0 …
    cost=0 outcome=ok``, which reads on a dashboard as a free call that worked.

    Read through `token_counts`, so it knows every vendor's spelling by construction and cannot
    drift from what the line prints. ``False`` for ``None`` too, which is the same claim: nothing
    was stated.

    **It is a claim about the accounting and never about the answer** (issue #491). #488 read an
    all-zero block as the signature of a broken stream — a call that returned 1,700 characters
    cannot have consumed zero input tokens — and that inference is wrong: OpenRouter/Google report
    no usage at all for ``google/gemini-3.8-flash``'s *video* calls, while the same model reports
    it for images in the same wake. Complete answers were discarded over a vendor's bookkeeping.
    So both readers use it for what it says: `log_llm_call` omits the token and cost fields rather
    than print zeros, and `_describer` notes once per wake that the vendor stated none — neither
    condemns the text.
    """
    return any(token_counts(usage).values())


#: Every spelling of *"generation stopped because the output budget ran out"* — the one finish
#: reason that means the answer in hand is a **fragment**, not an answer. OpenAI and OpenRouter's
#: chat wire say ``length``; OpenAI's Responses surface says ``max_output_tokens`` (as
#: ``incomplete_details.reason``); the native xAI SDK names its enum ``REASON_MAX_LEN``; Anthropic says
#: ``max_tokens``, and ``model_context_window_exceeded`` when the output ran into the end of the
#: context window rather than into its own cap — a fragment just the same. Compared
#: case-insensitively, because the harness is matching a vendor's vocabulary rather than defining
#: one.
TRUNCATED_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "max_output_tokens", "reason_max_len", "model_context_window_exceeded"}
)


def finish_reason(response: Any) -> str | None:
    """Why the vendor stopped generating, in its own words — ``None`` when it did not say.

    A **capability read, not a vendor branch** (`serving_endpoint`'s shape): one reader over the
    three places a shipped adapter's SDK puts the fact, so an adapter answers it by handing over
    whatever its SDK returned.

    - the chat wire (OpenAI, xAI-over-``openai``, OpenRouter): ``choices[0].finish_reason``;
    - the Responses surface: nothing while ``status`` is ``completed``, and
      ``incomplete_details.reason`` once it is ``incomplete`` — the surface states *why it stopped*
      only when stopping was not the plan;
    - the native ``xai-sdk``: a top-level ``finish_reason`` property naming its proto enum.

    ``None`` is *the vendor said nothing*, and it is also every failure mode here: this is read on
    the logging path, where a surprise in a vendor object must never break a turn. A missing finish
    reason costs a truncation nobody detected; a raise here would cost the wake.
    """
    try:
        return _finish_reason(response)
    except Exception:  # noqa: BLE001 - observability never breaks a turn over a vendor's surprise
        return None


def truncated(reason: str | None) -> bool:
    """Whether `reason` is a vendor's way of saying *the output budget ran out*."""
    return bool(reason) and str(reason).strip().lower() in TRUNCATED_FINISH_REASONS


def _finish_reason(response: Any) -> str | None:
    """`finish_reason`'s read, unguarded — see there for the shapes and why each one is looked at."""
    if response is None:
        return None
    choices = _read(response, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        value = _read(choices[0], "finish_reason")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if _read(response, "status") == "incomplete":
        value = _read(_read(response, "incomplete_details"), "reason")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = _read(response, "finish_reason")
    return value.strip() if isinstance(value, str) and value.strip() else None


def reported_cost(usage: Any) -> float | None:
    """The call's cost in dollars **as the provider reported it**, or ``None`` when it didn't.

    Read off ``usage.cost`` wherever an endpoint states one: OpenRouter does — reachable through
    the native SDK *and* through the ``openai`` SDK pointed at ``openrouter.ai``, which keeps the
    field — and so do OpenAI-compatible vendors reached under the generic profile (Wiro states it,
    issue #670); the ``openai`` SDK keeps an unmodeled field, so it survives ``model_dump``. An
    endpoint that reports tokens and no dollars gets no ``cost=`` from here: see `_COST_FIELDS`
    for the three vendors (Google, OpenAI, Anthropic) whose figure the harness computes instead, and why no
    other is.

    **It is what the operator pays for the call, whichever party bills it** (issue #653). On a
    bring-your-own-key call (`byok`) OpenRouter's ``usage.cost`` is ``0`` and the provider's own
    charge is ``cost_details.upstream_inference_cost``, so that is the figure read — never the sum
    of the two. A BYOK block that states no upstream figure gets ``None``: the ``0`` beside it is
    OpenRouter's share, and printing it would turn a call the provider billed into a free one.
    """
    if usage is not None and byok(usage):
        value = _first(usage, _UPSTREAM_COST_FIELDS)
    else:
        value = _first(usage, _COST_FIELDS) if usage is not None else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def byok(usage: Any) -> bool:
    """Whether the vendor says this call ran on the operator's own provider key (issue #653).

    ``True`` only for a literal ``True`` (`_BYOK_FIELDS`): a truthy string or a ``1`` is a vendor's
    surprise, and reading one as BYOK would move the dollar figure to a field that is not there.
    """
    return usage is not None and _first(usage, _BYOK_FIELDS) is True


def reasoning_tokens(usage: Any) -> int | None:
    """How many completion tokens the provider says went to **reasoning**, or ``None``.

    Read the same way every other usage fact is (`_first` over `_REASONING_FIELDS`), so one reader
    serves a mapping, a typed model, and a proto. ``None`` — the field omitted — whenever the
    provider reports nothing: a non-reasoning model has no number here, and an invented zero would
    be indistinguishable from a reasoning model that happened to spend nothing.
    """
    value = _first(usage, _REASONING_FIELDS) if usage is not None else None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def serving_endpoint(response: Any) -> str | None:
    """The upstream that actually served this call, per the response's routing metadata.

    Given whatever the adapter's SDK handed back — a response mapping, a typed model, a proto — so
    one reader serves every adapter (`_read` handles all three). The router lists every endpoint it
    considered and flags the one it **picked**; that flag is the whole answer, and it stays right
    even when a server-side tool ran, which is exactly where the old read went wrong (`_ROUTING_METADATA`,
    issue #280).

    ``None`` — the field omitted — whenever the response names no selected endpoint: a
    direct-to-vendor SDK (there, the vendor *is* the endpoint), or a router that was not asked for
    its metadata. **Omitting is the correct failure.** The predecessor of this function reached for a
    plausible-looking field instead, and a wrong endpoint is worse than an absent one: it does not
    leave a gap in the routing review, it invents a distribution that never happened.
    """
    if response is None:
        return None
    available = _first(response, (_ROUTING_METADATA,))
    if not isinstance(available, (list, tuple)):
        return None
    for endpoint in available:
        if not _read(endpoint, "selected"):
            continue
        name = _read(endpoint, "provider")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


#: What a vendor's call id looks like — the one shape every id the fleet's vendors issue fits, and
#: nothing else (issue #634). Measured live 2026-10-03: OpenRouter ``gen-1791030923-8abbrs…``,
#: OpenAI ``chatcmpl-EUtF…`` and ``resp_0080…``, xAI a bare uuid. The value is vendor-written text
#: on a line whose columns are partial regex matches, so anything outside this shape is not an id
#: and is omitted rather than quoted: a quoted ``"x kind=y"`` would still feed ``kind=`` to a
#: column that never asked a vendor for it.
_GENERATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")

#: The header OpenRouter sets on **every** response, a refusal included (measured live 2026-10-03
#: on a 400 and a 404 as well as a 200, where it equals the body's ``id``). Matched without regard
#: to case, since an SDK's error may hand its headers back as a plain dict.
GENERATION_ID_HEADER = "x-generation-id"


def generation_id(response: Any) -> str | None:
    """The vendor's own id for this call, read off its response — or ``None`` (issue #634).

    The ``id`` every chat-completions and Responses body carries, and the native xAI response's
    ``id`` attribute: OpenRouter's generation id, OpenAI's completion or response id, xAI's
    response id. It is what a vendor's feedback and refund path asks for, so the line carries it
    and a complaint names the generation instead of a timestamp the vendor has to search by.

    Omitted, never a placeholder, when the response names none or names something that is not an
    id (`_GENERATION_ID`). Not a secret on any vendor: each one looks the id up only for the account
    that made the call, so it identifies a call and grants nothing.
    """
    if response is None:
        return None
    return _as_generation_id(_read(response, "id"))


def generation_id_header(headers: Any) -> str | None:
    """The `GENERATION_ID_HEADER` off a response's headers, or ``None`` (issue #634).

    The only id a **refused** attempt carries — a 429 or a 5xx has no body id — and the one a body
    that failed to parse still carries, so an adapter's error mapper stamps it on the error it
    raises (`ProviderError.generation_id`). Asked of any vendor: one that sets no such header
    yields ``None`` here, which is a capability read and not a vendor branch.
    """
    if headers is None:
        return None
    get = getattr(headers, "get", None)
    if not callable(get):
        return None
    try:
        value = get(GENERATION_ID_HEADER)
        if value is None and isinstance(headers, Mapping):
            value = next(
                (v for k, v in headers.items() if str(k).lower() == GENERATION_ID_HEADER), None
            )
    except Exception:  # noqa: BLE001 - this is the failure path; a header read must never raise
        return None
    return _as_generation_id(value)


def stamp_generation_id(error: BaseException, generation: str | None) -> BaseException:
    """`error` carrying `generation` as its `generation_id`, unless it already names one.

    Returned rather than raised so a mapper reads ``raise stamp_generation_id(mapped, …) from exc``.
    An id an inner layer already stamped is kept: it was read closer to the response.
    """
    if generation is not None and getattr(error, "generation_id", None) is None:
        try:
            error.generation_id = generation  # type: ignore[attr-defined]
        except AttributeError:  # a slotted third-party exception: observability never breaks a turn
            pass
    return error


_T = TypeVar("_T")


def read_answer(
    read: Callable[[], _T],
    response: Any,
    *,
    finish: str | None = None,
    usage: Any = None,
    cap: int | None = None,
) -> tuple[_T | None, Exception | None]:
    """Read the vendor's answer **before** the call's line is written: ``(reply, None)`` or
    ``(None, the error)`` (issue #782).

    An adapter used to write its `llm` line the moment the body arrived and read the turn out of it
    afterwards, so an answer that could not be read (tool-call arguments cut off at the output cap,
    a malformed body) was written ``outcome=ok`` and then retried, which is one billed attempt saying
    two contradictory things. Reading first lets the one billed line say what happened: `answered`
    gives its fields, ``fallback reason=invalid_response`` for an unreadable answer. The harness's
    decision (``retrying`` or ``failed``) follows on its own line, which carries no cost. The error
    is returned, stamped with the response's id (`generation_of`), for the adapter to raise once its
    lines are written.

    **Any** exception a read raises is caught, not only `ProviderResponseError`: a vendor's body
    malformed in a shape the parser did not foresee (a null tool-call ``function``) raises whatever
    the parser trips on, and the call was billed all the same. Catching only the taxonomy's class
    lost the billed line, and with it the spend, on exactly the bodies nobody anticipated. The
    original exception is what the adapter raises; only the line is changed.

    **Except when the answer stopped at the output cap** (issue #788). Then it was not malformed, it
    was cut off: a runaway generation wrote until the cap, and the tool call it was in the middle of
    does not parse. The error becomes `ProviderTruncatedError`, the original chained as its cause,
    so the billed line says ``reason=truncated`` and the decision that follows says the same. Either
    of two facts says the cap fired, and both are needed:

    - **the vendor's stop reason** — `finish` when the adapter already read it, else read off
      `response` (`finish_reason`);
    - **the bill** — the output the vendor billed (`usage`) reached the cap the call carried (`cap`).
      The stop reason alone is not the vendor's word on this everywhere: cut off inside a tool call,
      OpenAI's model behind OpenRouter reports ``finish_reason: "tool_calls"`` with exactly
      ``max_tokens`` billed (measured live, 2026-11-03). A billed output at the cap is the cap
      firing whatever the stop reason says.
    """
    try:
        with generation_of(response):
            return read(), None
    except Exception as exc:  # noqa: BLE001 - the line is written, then the original re-raised
        if isinstance(exc, ProviderTruncatedError):
            return None, exc
        stopped = finish if finish is not None else finish_reason(response)
        written = token_counts(usage).get("tokens_out") if usage is not None else None
        at_cap = cap is not None and written is not None and written >= cap
        if not truncated(stopped) and not at_cap:
            return None, exc
        why = stopped if truncated(stopped) else f"{written} of {cap} tokens"
        cut = ProviderTruncatedError(
            f"The answer stopped at the output cap ({why}) before it could be read: {exc}"
        )
        cut.__cause__ = exc
        return None, stamp_generation_id(cut, generation_id(response))


def answered(error: Exception | None) -> dict[str, Any]:
    """The verdict fields of an answered call's line, given what `read_answer` returned."""
    if error is None:
        return {}
    reason = TRUNCATED if isinstance(error, ProviderTruncatedError) else UNREADABLE
    return {"outcome": FALLBACK, "reason": reason, "level": logging.WARNING}


@contextmanager
def generation_of(response: Any) -> Iterator[None]:
    """Stamp the response's `generation_id` on a provider error raised while reading it.

    The shape of basecradle-noc#963: a body arrives, the `llm` line is written with its id, and
    reading the turn out of it then fails (tool-call arguments cut off at the output cap) — so the
    ``outcome=retrying reason=invalid_response`` line that follows names the same generation.
    """
    try:
        yield
    except ProviderError as exc:
        stamp_generation_id(exc, generation_id(response))
        raise


def _as_generation_id(value: Any) -> str | None:
    """`value` when it has the shape of a vendor's call id (`_GENERATION_ID`), else ``None``."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if _GENERATION_ID.fullmatch(value) else None


def describe_provider(provider: object) -> tuple[str, str]:
    """``(provider, model)`` for the wake's bookend lines, from whatever adapter is wired in.

    Every shipped adapter carries both — ``provider`` is the endpoint vendor (``AI_PROVIDER``:
    the axis that decides *whose* model answered) and ``model`` is the model id. A third-party
    adapter that satisfies the `Provider` protocol without them is not a failure: it reports
    ``unknown`` and the wake runs exactly as before. Observability never breaks a turn.
    """
    return (
        str(getattr(provider, "provider", None) or "unknown"),
        str(getattr(provider, "model", None) or "unknown"),
    )


def _first(payload: Any, paths: tuple[tuple[str, ...], ...]) -> Any:
    """The first of several candidate paths that resolves to something, or ``None``.

    Each vendor spells the same fact differently, so a field is declared as an ordered list of
    where it *might* live and the first hit wins — which is what keeps every adapter reading
    through one code path instead of a branch per vendor.
    """
    for path in paths:
        value: Any = payload
        for name in path:
            if value is None:
                break
            value = _read(value, name)
        if value is not None:
            return value
    return None


def _read(payload: Any, name: str) -> Any:
    """One field, whether the SDK handed back a mapping or an object."""
    if isinstance(payload, Mapping):
        return payload.get(name)
    return getattr(payload, name, None)


def _charged(cost: Any) -> bool:
    """Whether `cost` is a **stated charge** rather than the absence of one (issue #488).

    Asked through `_money`, deliberately: it is already the one place that decides what a readable
    dollar figure is, and a second numeric-shape opinion here would be a second thing to keep in
    step with it. A figure it will not render is not a charge, and neither is a rendered zero.
    """
    return _money(cost) not in (None, "0")


def _money(cost: Any) -> str | None:
    """A dollar cost as plain fixed-point — ``0.0445``, never ``4.45e-02``.

    Scientific notation is what a naive ``str(float)`` produces for a sub-cent call, and nothing
    grepping this stream would read it as money. Trailing zeros are trimmed so the common line
    stays short; a genuine zero renders as ``0`` rather than vanishing, because "this call was
    free" is a fact worth logging.

    **The one cost convention, across every line kind.** ``cost=`` is emitted by the LLM line, the
    tool line, and any future line for a provider-billed call, all through this one formatter — so
    it is always plain decimal USD, ``cost=([0-9.]+)``-matchable, whatever the kind. That uniformity
    is load-bearing: the dashboard splits **LLM cost** from **tool cost** on the line *head*
    (`` llm provider=`` vs everything else), not on the cost field. So the invariant is exactly that
    ``cost=`` keeps this shape on every kind, and the `` llm provider=`` head never appears on a
    non-LLM line (a tool line begins ``tool ``). A call carries ``cost=`` **when the provider
    states the figure** (OpenRouter's ``usage.cost``, xAI's ticks) — and on the three vendors that
    state none — Google's Vertex AI, OpenAI and Anthropic — when the call's model and tier are in the
    harness's transcription of the vendor's published rates (issues #655, #657 and #725,
    `_google_rates`, `_openai_rates`, `_anthropic_rates`), tagged ``cost_basis=computed``. Nowhere else is a figure derived, because a
    stale table is worse than an honest gap.

    Typed loosely on purpose: the figure comes straight off a vendor object (`Response.cost_usd`),
    so anything that is not a real number — ``None``, a string, a bool — is dropped rather than
    formatted, and observability never breaks a turn over a vendor's surprise. That guard extends to
    the two shapes that *are* numbers yet would render **un-``cost=([0-9.]+)``-matchable** and so
    slip a call silently out of the dashboard's rollup: a **negative** figure (``cost=-0.02`` — the
    ``-`` is outside the pattern) and a **non-finite** one (a body carrying ``NaN``/``Infinity``,
    which stdlib ``json`` decodes without complaint → ``cost=nan``/``cost=inf``). A charge is never
    either, so a vendor that reports one is a surprise the field is *omitted* for — honest absence,
    the same as any other unreadable cost — rather than logged in a shape the extraction can't read.
    """
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    if not math.isfinite(cost) or cost < 0:
        return None
    return f"{cost:.8f}".rstrip("0").rstrip(".")


def _secs(seconds: float) -> str:
    """A duration as ``1.23s`` — one unit, two decimals, every line the same."""
    return f"{seconds:.2f}s"
