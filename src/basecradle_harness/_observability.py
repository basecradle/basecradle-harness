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

import getpass
import logging
import math
import os
import re
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
#: ``bc_uat_``/``bc_isk_``), the box's (GitHub ``gh?_`` and fine-grained ``github_pat_``, Google
#: ``AIza``, Hugging Face ``hf_``, Replicate ``r8_``, Steel ``ste-``, a Better Stack heartbeat URL,
#: an unanchored ``api_key=``), and the generic bearer and key-in-query forms.
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
#: ``-`` is kept. The rules are spelled once, in `_PREFIXED`. `_PREFIX` can strip a longer prefix
#: than the alternative that matched (``xsk-or-…`` matched as ``sk-``), and that never changes the
#: verdict: every prefix is lowercase letters, ``_`` and ``-``, which are never evidence of a key.
_PREFIXED = (
    (r"sk-or-|sk-|xai-|bc_[a-z]{3}_", r"[A-Za-z0-9_-]{8,}"),
    (r"bc_(?:uat|isk)_", r"[A-Za-z0-9]+"),
    (r"gh[a-z]_", r"[A-Za-z0-9]{20,}"),
    (r"github_pat_", r"[A-Za-z0-9_]{20,}"),
    (r"AIza", r"[A-Za-z0-9_-]{20,}"),
    (r"hf_", r"[A-Za-z0-9]{20,}"),
    (r"r8_", r"[A-Za-z0-9]{20,}"),
    (r"ste-", r"[A-Za-z0-9]{20,}"),
)
_PREFIX = re.compile("|".join(f"(?:{prefix})" for prefix, _ in _PREFIXED))
_PREFIX_RULES = re.compile("|".join(f"(?:{prefix}){body}" for prefix, body in _PREFIXED))

#: The rules anchored on a literal of their own — a heartbeat URL, a bearer header, a key in a query,
#: an ``api_key=`` assignment — which are redacted wherever they match. **They run in a pass before
#: the prefix rules**, because their literals are made of the characters a prefix rule's run is: in
#: one alternation the leftmost match wins, so ``task-runner-api-key=…`` was consumed as the run
#: ``sk-runner-api-key`` and the value after ``=`` was never matched by anything, and
#: ``task-runner-bearer …`` lost its bearer value the same way. A prefix run cannot reach across the
#: ``[redacted]`` the first pass leaves, so the second pass never undoes it. Inside this pass the
#: same holds between two rules: an assignment whose value is a bearer header
#: (``AUTH_API_KEY=Bearer eyJ…``) takes every ``Bearer`` and the token together, where it once took
#: ``Bearer`` as the whole value and left the token after the space; every, because a token stored
#: with its scheme and sent with another reads ``Bearer Bearer eyJ…``. The heartbeat URL runs first
#: of all, alone (`_HEARTBEAT_RUN`).
_LITERAL_RULES = re.compile(
    r"(?i:bearer)\s+[A-Za-z0-9._~+/-]{8,}"
    r"|(?i:[?&](?:api[-_]?key|access[-_]?token|token|key)=)(?:(?i:bearer)\s+)*[^&\s]+"
    r"|(?i:api[-_]?key=)(?:(?i:bearer)\s+)*[^&\s\"]+"
)

#: A Better Stack heartbeat URL, whose secret is its path, redacted by the box's rule
#: ``(?:https?://\S*)?/heartbeat/\S+`` (basecradle-router#398): a URL with a scheme from the scheme
#: on, so userinfo before the host (``https://john:…@host/…/heartbeat/…``) never ships; one without
#: (``curl uptime.example.com/api/v1/heartbeat/…``) from ``/heartbeat/`` on, keeping its host, which
#: is not the secret. **It runs first, before every other rule**, as the box's does: a rule that
#: matched first and stopped inside the URL (a bearer run takes ``…https`` and stops at the ``:``)
#: would leave the rest of it, userinfo included, to the scheme-less arm alone; and this pattern
#: starts only at a run's start, which a match ending mid-run in one alternation would skip.
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
    is the defect this exists to retire.
    """
    if len(text) <= cap:
        return text
    head = text[: max(cap - len(marker), 0)]
    if _TOKEN_CHAR.match(text, len(head)):  # the next character continues the head's last run
        head = _TRAILING_RUN.sub("", head)
    head = head.rstrip()
    return head + marker if head else ""


def scrub(text: str) -> str:
    """`text` with every credential shape the scrub knows replaced by ``[redacted]``.

    Three passes, in order: the heartbeat URL (`_HEARTBEAT_RUN`), the other literal rules
    (`_LITERAL_RULES`), then the prefix rules (`_PREFIXED`). The last keeps a match `_is_word`
    reads as a word, as written.

    Run on a whole text **before** `bounded` cuts it, so a rule is never shown a credential cut
    short of its own minimum. `kv` does both itself; a site that cuts a text for a log line before
    `kv` sees it calls this first.
    """
    text = _HEARTBEAT_RUN.sub(_heartbeat, text)
    return _PREFIX_RULES.sub(_redaction, _LITERAL_RULES.sub("[redacted]", text))


def _heartbeat(match: re.Match[str]) -> str:
    """A space-free run holding a heartbeat URL, redacted from where the box's rule would begin."""
    run = match.group()
    start = run.index(_HEARTBEAT)
    scheme = _SCHEME.search(run, 0, run.rindex(_HEARTBEAT, 0, len(run) - 1))
    if scheme:
        start = min(start, scheme.start())
    bearer = _TRAILING_BEARER.search(run, start)
    return run[:start] + "[redacted]" + (bearer.group() if bearer else "")


def _redaction(match: re.Match[str]) -> str:
    return match.group() if _is_word(match) else "[redacted]"


def _is_word(match: re.Match[str]) -> bool:
    """Whether a prefixed match is part of a word rather than a credential (issue #794).

    Glued to a letter, digit or combining mark (any script), **and** its run after the prefix holds
    no digit and no uppercase letter — never one without the other, because a serialization can glue
    a key to a letter, and a key's run always holds a digit or a capital with overwhelming
    certainty. A glued word whose run holds one is still redacted: the scrub errs toward redaction.
    Only the prefix pass asks; `_LITERAL_RULES` redacts unconditionally.
    """
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


def head(text: str, color: str) -> str:
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
    return f"{color}{text}{RESET}" if color_enabled() else text


def kv(**fields: Any) -> str:
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
    return " ".join(parts)


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


def _full_value(value: str) -> str:
    """`_value`'s rendering minus the cap: flattened, scrubbed, quoted — never truncated.

    The three jobs that make a value *safe* in a log stream are kept (one record per line; no
    credential; one field, not several). The fourth — bounding it — is deliberately dropped for the
    one field that has nowhere else to live (see `log_unspoken`). Quoted unconditionally, because a
    narration is prose: it will contain spaces, and it may well contain an ``=``.
    """
    text = scrub(_flatten(value))
    return '"' + text.replace('"', "'") + '"'


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
#: - ``retrying``: the attempt failed and will be re-issued. Nothing was generated or billed.
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
    cached_tokens cache_write_tokens tokens_reasoning cost cost_basis outcome reason detail`` then any purpose-specific `extra` (the reranker's ``surface``/``pool``/
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
    because the attempt generated nothing and was billed nothing.

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
    generation_id: str | None = None,
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
    `ComputedCost`. A failed run (``outcome=error``) names why as a slug in ``reason``; when there is
    foreign text to show, ``error`` writes it on a second, WARNING ``tool`` line carrying only
    ``name`` and ``error``, as before. A server-side failure has a vendor code and no text, so it has
    no companion and is itself written at WARNING, where a severity filter still sees it.
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
            cost=_money(cost),
            cost_basis=cost_basis(cost),
            generation_id=generation_id,
        ),
    )
    if error is not None:
        _log.warning("tool %s", kv(name=name, error=error))


def log_server_tool(
    *,
    name: str,
    provider: str,
    model: str,
    count: int,
    generation_id: str | None,
    reason: str | None = None,
    cost: float | None = None,
) -> None:
    """One `tool` line for a tool the vendor ran server-side: one per model call per tool.

    ``name`` is the harness's name for the capability (``web_search``, ``web_fetch``,
    ``code_execution``), never the vendor's block type; ``count`` is the uses the vendor reported on
    this call; ``reason`` is the vendor's own code for a use that failed, which makes the line
    ``outcome=error``.
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
) -> None:
    """`log_server_tool` for every tool in `tallies` that ran — the adapters' one call site."""
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


#: The ``charge=`` and ``reason=`` of a code sandbox's container time, which Anthropic and OpenAI
#: both bill and no response of either states — the one shared spelling of that gap.
CONTAINER_TIME = "container_time"
NOT_REPORTED = "not_reported"

#: The rest of the ``cost unpriced`` vocabulary (`docs/telemetry.md` → ``cost unpriced``), spelled
#: once so every adapter states the same gap in the same words (issue #759): the call's **tokens**
#: and a search's **grounding** fee are the charges; a model the rate table does not carry, a tier
#: (or long-context band) it does not carry, and a host whose prices are not the table's are the
#: reasons. A price read live from the vendor rather than from a table adds one more (issue #844):
#: the price source itself could not be read on this wake.
TOKENS_CHARGE = "tokens"
GROUNDING_CHARGE = "grounding"
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
    would be a figure nobody can stand behind. The vendor, kind and model are the first unit's.

    **Recording can never raise into a wake.** The fee and the count come off a vendor body, often
    in a ``tools/`` overlay tool someone wrote in an afternoon, and the meter is read *after* the
    tool has already acted — so a figure that is not a real, finite, non-negative number (a string,
    a bool, ``nan``) is dropped as it is recorded, as `_money` drops it from every other line, and a
    count that is not a positive ``int`` likewise.
    """

    __slots__ = ("_units", "failure", "verdict")

    def __init__(self) -> None:
        self._units: list[tuple[str, str, str, int | None, float | None, bool]] = []
        self.failure: Exception | None = None
        #: ``(reason, error)`` from `tool_failed`: this run failed, with no vendor work to blame.
        self.verdict: tuple[str, str] | None = None

    def record(
        self,
        *,
        provider: str,
        kind: str,
        model: str,
        count: Any = None,
        cost: Any = None,
        failed: bool = False,
    ) -> None:
        units = (
            count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else None
        )
        fee = cost if _money(cost) is not None else None
        self._units.append((provider, kind, model, units, fee, failed))

    def fields(self) -> dict[str, Any]:
        """The `log_tool` keywords this run's recorded work fills (empty when it recorded none)."""
        if not self._units:
            return {}
        provider, kind, model, count, _, _ = self._units[0]
        if len(self._units) > 1:
            count = sum(unit[3] if unit[3] is not None else 1 for unit in self._units)
        costs = [unit[4] for unit in self._units if not unit[5]]
        cost: float | None = None
        if costs and all(c is not None for c in costs):
            total = math.fsum(costs)  # type: ignore[arg-type]
            computed = any(isinstance(c, ComputedCost) for c in costs)
            cost = ComputedCost(total) if computed else total
        return {"provider": provider, "kind": kind, "model": model, "count": count, "cost": cost}


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
    *, provider: str, kind: str, model: str, count: Any = None, cost: Any = None
) -> None:
    """Record one finished unit of vendor work on the running tool's meter (a no-op with none)."""
    meter = _METER.get()
    if meter is not None:
        meter.record(provider=provider, kind=kind, model=model, count=count, cost=cost)


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


class MeteredCall:
    """The handle `metered` yields, so a tool can record the charge its vendor's answer states.

    The charge is knowable only *after* the vendor responds and its body is read — inside the block.
    The tool sets ``cost`` (a vendor-stated USD figure, a `ComputedCost`, or leaves it ``None`` when
    neither is possible) and ``count`` (billed units, where there is more than one).
    """

    __slots__ = ("cost", "count")

    def __init__(self) -> None:
        self.cost: float | None = None
        self.count: int | None = None


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
    meter_tool(provider=provider, kind=kind, model=model, count=call.count, cost=call.cost)


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
    decision (``retrying`` or ``failed``) follows on its own unbilled line. The error is returned,
    stamped with the response's id (`generation_of`), for the adapter to raise once its lines are
    written.

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
