"""What a secret is, in one place, and the one redaction every text the harness writes goes through.

Issue #999 (`@origin`, 2026-11-23): *"a single place that says what should be redacted, and then
apply that as needed to many systems."* Until it, the credential scrub lived inside the log module
and only log lines went through it; what the harness filed into memory, and the text a browser tool
saved under the agent's home, did not. So the definition moved here, and every place the harness
writes text down asks this module:

- **log lines** (`_observability.kv`, `_observability.ScrubbingFormatter`): the scrub they always
  had, unchanged in what its shapes catch, now with the value half beside it;
- **what is filed into memory**, before it is filed (`_wake.WakeAgent._observe`, the one place the
  harness mines, and `_memory.SqliteMemoryStore.write`, the one place a fact the model writes is
  stored);
- **the text files a tool saved in its capture folder** (`_mcp.McpTool.run`, after the call that
  wrote them; `redact_file`).

**A text held in pieces is redacted as one** (`scrub_parts`, issue #1020): a reply cut off at its
output budget and the resume that finished it are one reply in two messages, so the transcript,
the resume's journal line and the exchange memory is handed redact them together.

**Two halves, and both run everywhere** (`scrub`):

- **By value** (`configured_values`): the agent's own configured secrets, which the harness knows at
  run time — every environment variable whose whole name names a credential by the assignment
  rule's own words (``BASECRADLE_TOKEN``, ``AI_API_KEY``, ``NTFY_DM_TOKEN``), and every credential it
  holds as a `Secret` (an MCP server's credential-named ``env`` entry, a key a library caller passed
  in). This half needs no pattern and catches a vendor nobody has listed.
- **By shape**: the vendor key prefixes, the credential-named assignment, the bearer header, a key
  in a query, an ``api_key=``, an ``x-api-key`` header, a credential-named JSON member and a
  heartbeat URL.

**What a secret looks like is data, and public data** (``data/secret-shapes.json``, documented in
``docs/secret-shapes.md``, the way the price tables are: issue #997). The prefixes and their runs,
the assignment rule's words, tail, stops and floor, the bearer scheme and its token, each family of
label names, the heartbeat path and both markers are read from that file and nowhere else, so the
fleet's own scans and log pipeline read the same definition and no reader imports a harness module.
What stays code is the *reading*: how a value is found once a label is (a shell word, an escaped
quote at any depth, a JSON string's close), and the union every rule's spans are merged into. Those
are this package's, and each reader of the file does its own.

**What neither half can catch, stated plainly**: an unlisted vendor's key that is not one of the
agent's configured secrets. A key the agent was handed in a message, read off a page, or holds in a
file the harness never loaded is not a value it knows, and if it has no listed shape, nothing here
redacts it. That is why the agent is told a secret put anywhere else *may* be redacted, never that
it will be (`_defaults/prompts/initialize.md`).

**Never the places that are meant to hold a secret.** Nothing here walks a directory: each caller
hands it one text, or one file it names, and the capture folder's caller (`_mcp`) names exactly
which files that is. The agent's configuration, its credential files, the vault and the wallets are
not in it, and are never read for redaction or changed.

A file this package cannot read is a broken install, so it fails at import like a missing module
would (`ShapesFileError`), as the prices file does.
"""

from __future__ import annotations

import heapq
import json
import os
import re
import stat
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from basecradle_harness._secret import known_secrets

#: Where the file sits inside the package (``basecradle_harness/data/secret-shapes.json``).
RESOURCE = ("data", "secret-shapes.json")

#: What the file says it is. A reader checks both before trusting a field.
FORMAT = "basecradle-harness-secret-shapes"
FORMAT_VERSION = 1


class ShapesFileError(RuntimeError):
    """The package's secret-shapes file is missing, unparseable, or not a format this version reads."""


@cache
def load() -> Mapping[str, Any]:
    """The whole secret-shapes file, parsed and checked for its format and version."""
    where = "/".join(RESOURCE)
    try:
        node = resources.files("basecradle_harness")
        for part in RESOURCE:  # one segment per call: not every 3.11 Traversable takes several
            node = node.joinpath(part)
        data = json.loads(node.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ShapesFileError(f"the package's {where} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise ShapesFileError(f"the package's {where} is not a JSON object")
    found = (data.get("format"), data.get("format_version"))
    if found != (FORMAT, FORMAT_VERSION):
        raise ShapesFileError(
            f"the package's {where} is {found!r}, not {(FORMAT, FORMAT_VERSION)!r}"
        )
    return data


def _section(name: str) -> Mapping[str, Any]:
    try:
        return load()[name]
    except (KeyError, TypeError) as exc:
        raise ShapesFileError(f"the secret-shapes file carries no {name!r} section") from exc


def _names(section: str) -> str:
    """A family of label names as one regular-expression alternation, longest first, each literal."""
    names = sorted(_section(section)["names"], key=lambda name: (-len(name), name))
    return "|".join(re.escape(name) for name in names)


def _longest(section: str) -> int:
    return max(len(name) for name in _section(section)["names"])


_MARKERS = _section("markers")

#: What stands where a secret stood, but for a credential-named assignment's value.
REDACTED = _MARKERS["redacted"]

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
_PREFIXED = tuple(
    (rule["prefix"], rule["run"], rule["minimum_run"]) for rule in _section("vendor_keys")["rules"]
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
#: glued ``bearer`` and its token, written without lookaround for the box's VRL. **An ``api_key=``
#: value may open with an escaped quote, at any depth, and ends at its escaped close** (issue #934):
#: ``api_key=\"K9…\"`` once matched ``api_key=\`` alone and shipped the key, and in a JSON string
#: (``{"h":"api_key=\"K9…\""}``) the line no longer parsed. A quote is escaped by an odd run of
#: backslashes; after an even run it is a real quote, the backslashes escaping each other
#: (``{"h":"api_key=\\","k":"v"}``), and is never taken as an opener. The value's run still stops at
#: the ``"``. **Whatever the value opened with, an odd run of backslashes before that quote is left
#: whole** (`_literal_end`, issue #937): the run escapes the quote, how much of it is the value's own
#: depends on a depth no reading of the run can tell, and every JSON string the line held parses only
#: while it stays. So the key goes, and no string loses its close to the run. Until
#: #937 a value with no escaped opener took the run, the escape of the line's own closing quote with
#: it (``{"log": "{\"h\":\"api_key=x\\\\\"}"}``), and the line no longer parsed.
#:
#: **An opener at the close of the string it sits in opens nothing** (`_closes`, issue #939). The
#: ``api_key=`` and header values may open with a ``"``, so where the label ends a JSON string, the
#: string's own close was read as the opener and the value ran on into the line's structure:
#: ``{"h":"api_key=","k":"v"}`` scrubbed to ``{"h":"[redacted]"k":"v"}``. A close is followed by
#: structure, and a value that is structure and nothing else is no key, so such a match is no value.
#: **A key in a query still takes the close** (``{"h":"url?api_key=K9…","k":"v"}`` scrubs to
#: ``{"h":"url[redacted]``): its run takes a ``"``, and stopping it at one gave up keys the run
#: took, a value in doubled CSV quotes among them, for a parse no reader in the fleet performs.
#:
#: **A literal rule reads on through a label its value ends in** (`_literal_spans`): a value stops
#: where its run does (at a blank, a ``"`` or a ``&`` for ``api_key=``; a blank or a ``&`` for a
#: query; a blank, a quote, a backslash or a ``&`` for a header), and the label of the next value can stand just before that stop,
#: so the value ends in the next label (``api_key=x,api_key="eyJ…"``, whose value is ``x,api_key=``
#: up to the quote; ``x-api-key: "x-api-key": "K9…"``, the review of #929). Each search starts past
#: the last match, so the value after that label once went to no rule. The span now ends where the
#: last such label's own reading does, as the union of every label's own reading would.
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
#:
#: **A quote is escaped at any depth, in the header's label and its value's opener alike** (issue
#: #935): JSON inside two nested JSON strings writes ``\\\"`` (three backslashes), and a header
#: escaped that deep once matched neither, so the whole header shipped. Both read the piece the
#: ``api_key=`` opener reads (`_ESCAPE`): none, or an odd run of backslashes; after an even run the
#: quote is a real one (``{"x-api-key":"\\","k":"v"}``), never taken. The value's run stops at a
#: backslash, so the escaped close stays as deep as it came and the JSON still parses. **The
#: even-run reading is exact for the outermost string only**: one encoding pass turns a run of
#: ``n`` before a quote into ``2n + 1``, so an inner string whose value ends in an even run reaches
#: the text as an odd one, and its real close is read as an opener. Where structure and the next
#: string's quote follow that close, the opener opens nothing (`_closes`, issue #939) and the inner
#: string keeps it; before a number, a literal, an object or an array the outermost string parses
#: and that inner string loses its close. A quote escaped at every level
#: is a run of exactly ``2**d - 1``, a set no regular expression states for every depth.
_ESCAPE = r"(?:\\(?:\\\\)*)?"
_BEARER_SHAPE = _section("bearer")
_SCHEME_LETTERS = _BEARER_SHAPE["scheme"]
_SCHEME = f"(?i:{_SCHEME_LETTERS})"
_TOKEN = f"{_BEARER_SHAPE['token']}{{{_BEARER_SHAPE['minimum_token']},}}"
_X_API_LABEL = rf"(?i:{_names('headers')})(?:{_ESCAPE}[\"']\s*(?:=>|[:=,])|\s*(?::|=>))"
_X_API_OPENER = rf"\s*(?:b?{_ESCAPE}[\"'])?"
_X_API_RUN = r"[^\s\"'&\\]"
_SCHEME_WORDS = rf"(?:{_SCHEME}\s+)*"
_X_API_KEY = re.compile(
    f"{_X_API_LABEL}(?P<opener>{_X_API_OPENER})(?P<value>{_SCHEME_WORDS}"
    rf"(?:{_X_API_RUN}*?(?P<glued>{_SCHEME})\s+{_TOKEN}|{_X_API_RUN}+))"
)
_QUERY_LABEL = rf"[?&](?i:{_names('query_parameters')})="
_QUERY_RUN = r"[^&\s]"
_QUERY = re.compile(
    f"(?P<label>{_QUERY_LABEL})(?P<value>{_SCHEME_WORDS}"
    rf"(?:{_QUERY_RUN}*?(?P<glued>{_SCHEME})\s+{_TOKEN}|{_QUERY_RUN}+))"
)
_API_KEY_LABEL = rf"(?i:(?:{_names('api_key_assignment')})=)"
_API_KEY_OPENER = f'(?:{_ESCAPE}")?'
_API_KEY_RUN = r"[^&\s\"]"
_API_KEY = re.compile(
    f"(?P<label>{_API_KEY_LABEL})(?P<opener>{_API_KEY_OPENER})(?P<value>{_SCHEME_WORDS}"
    rf"(?:{_API_KEY_RUN}*?(?P<glued>{_SCHEME})\s+{_TOKEN}|{_API_KEY_RUN}+))"
)

#: **A credential-named JSON member's string value** (issue #943, @origin, 2026-11-19): a member
#: whose name is exactly ``api_key``, ``apikey``, ``access_token``, ``refresh_token``,
#: ``client_secret``, ``password``, ``secret`` or ``authorization``, in any case, with ``-``, ``_``
#: or nothing between the words. The list is exact by his decision: never a bare ``key`` or
#: ``token``, names that are everywhere and almost never a secret. The name and the value's quotes
#: stay, and the value goes, read to the string's close (`_string_close`), so the line still parses.
#: The member's quotes are escaped at any depth (`_ESCAPE`), the name's and the value's opener alike
#: (``{"log": "{\\"password\\": \\"…\\"}"}``), and the run before the name's opening quote is taken
#: whole, never from inside a longer one. A value that is not a string (a number, ``null``, an
#: object, an array) is left alone, and only ``"`` quotes a member: a Python ``repr``'s
#: ``{'password': '…'}`` is no JSON member.
_MEMBER_NAME = f"(?i:{_names('json_members')})"
_MEMBER = re.compile(rf'(?<!\\)(?P<escape>{_ESCAPE})"{_MEMBER_NAME}(?P=escape)"\s*:\s*(?P=escape)"')
_STRING_STOP = re.compile(r'["\\\x00-\x1f]')
_BACKSLASHES = re.compile(r"\\*")


def _string_close(text: str, at: int, escape: int, stop: int | None = None) -> tuple[int, int]:
    """Where the JSON string whose value starts at `at` ends, and where the text after it resumes:
    the start of the run of `escape` backslashes before its closing quote and the place past that
    quote, or, when the string never closes, `stop` (the text's end when ``None``) or the first
    control character before it, twice, since no JSON string holds one raw.

    A string ``d`` levels deep opens with a quote escaped by ``escape = 2**d - 1`` backslashes. One
    encoding pass turns a run of ``n`` before a quote into ``2n + 1``, so its close is a quote after
    a run of ``escape + (2 * escape + 2) * m``, the ``m`` being the escaped backslashes that end the
    value, and a quote inside the value after a run ``2**d`` longer than one of those, never one. The
    value's own escaped backslashes before the close go with it, and the close's own run stays, so
    every string the line holds parses as it did. An escape no depth gives (five backslashes) is read
    by the same arithmetic, and a value that never closes runs to the stop: a reading that takes
    more costs a field, and one that takes less a secret."""
    end = len(text) if stop is None else stop
    while (found := _STRING_STOP.search(text, at, end)) is not None:
        i = found.start()
        if text[i] < " ":
            return i, i
        run_end = _BACKSLASHES.match(text, i, end).end()
        run = run_end - i
        if run_end < end and text[run_end] == '"':
            if run >= escape and (run - escape) % (2 * escape + 2) == 0:
                return run_end - escape, run_end + 1
            run_end += 1
        at = run_end
    return end, end


def _members(text: str, stop: int | None = None) -> Iterator[tuple[int, int, int]]:
    """Every credential-named JSON member in ``text[:stop]`` (`_MEMBER`), in order: where it starts,
    where its value starts and where the value's close begins (`_string_close`). Each search goes
    on past the last close, so a name inside a value is that value's text, a closing quote never
    opens the next name (the backslashes the value loses before it would let a second scrub read
    one there), and the text is read once."""
    end = len(text) if stop is None else stop
    pos = 0
    while (match := _MEMBER.search(text, pos, end)) is not None:
        close, pos = _string_close(text, match.end(), len(match.group("escape")), end)
        yield match.start(), match.end(), close


def _member_spans(text: str) -> Iterator[tuple[int, int]]:
    """Every credential-named JSON member's string value in `text`, from where it starts to where
    its close begins, an empty value giving no span."""
    return ((start, end) for _, start, end in _members(text) if end > start)


#: What follows the close of a JSON string, at any depth of nesting: a ``,`` or a ``:`` before the
#: next member, a ``}`` or a ``]`` ending what holds it (`_closes`).
_STRUCTURE = frozenset(",:}]")

#: How far before a place in a literal rule's span a label that would read on past the span can
#: start (`_literal_spans`): the rule's longest label (``x-api-key=>``, ``?access_token=``,
#: ``api_key=``) and the one scheme word its value may open with glued to it, the blank after that
#: being where the span stops. Only these characters are searched, so each rule stays linear.
_LITERAL_REACH = {
    _X_API_KEY: _longest("headers") + len("=>") + len(_SCHEME_LETTERS),
    _QUERY: len("?") + _longest("query_parameters") + len("=") + len(_SCHEME_LETTERS),
    _API_KEY: _longest("api_key_assignment") + len("=") + len(_SCHEME_LETTERS),
}

#: A bearer header: the scheme word in any case and its token, which is eight or more token
#: characters, so a diagnosis like "Bearer token missing" keeps its words, or a token that opens
#: with a quote or a substitution (issue #929), taken whole as the shell word it opens
#: (`_shell_word_end`), as a credential's value is read. Quoted, a token is no run of token
#: characters, and ``Bearer 'eyJ…'`` once matched nothing. **A run of scheme words goes with the
#: token after it** (`_bearer_end`).
_BEARER = re.compile(rf"(?P<scheme>{_SCHEME}\s+)(?:{_TOKEN}|[\"'`]|\$['({{]|\\+[\"'])")
_BEARER_TOKEN = re.compile(_TOKEN)
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
_HEARTBEAT = _section("heartbeat_url")["path"]
_HEARTBEAT_RUN = re.compile(r"(?<!\S)\S*" + re.escape(_HEARTBEAT) + r"\S+")
_URL_SCHEME = re.compile(r"https?://")

#: A bearer scheme word ending a heartbeat run, which `_heartbeat` keeps so the bearer rule still
#: sees it beside its token in the next run. Taken to the run's end, the word went with the URL and
#: left the token with no rule to match it: compact JSON (``{"url":"https://h/heartbeat/K9",
#: "auth":"Bearer eyJ…"}``) and a query value ending ``…&access_token=Bearer eyJ…`` shipped the
#: token, on the box as well. Not glued to a letter or digit, so a heartbeat secret that merely ends
#: in those letters is never shortened into a kept word.
_TRAILING_BEARER = re.compile(rf"(?<![A-Za-z0-9]){_SCHEME}\Z")

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
_ASSIGNMENT = _section("credential_assignment")
CREDENTIAL_WORDS = tuple(_ASSIGNMENT["words"])

#: The one tail a credential's name may carry after its word, as the canary's optional group.
CREDENTIAL_TAIL = _ASSIGNMENT["tail"]

#: The canary reads a value as a run of anything but these, at least `CANARY_VALUE_FLOOR`
#: characters long, from the character after the ``=``.
CANARY_VALUE_STOPS = _ASSIGNMENT["value_stops"]
CANARY_VALUE_FLOOR = _ASSIGNMENT["value_minimum_length"]

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
REDACTED_VALUE = _MARKERS["redacted_value"]

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
# A credential's value is read here exactly as the router reads it (basecradle-router#506 and #510,
# merged as `a8e8f5f`: `logfmt._shell_word_end` and `logfmt._value_end`; issue #934), so both
# writers on a box leave the same bytes where a value stood; `tests/router_reading.py` carries the
# router's functions verbatim and `tests/test_credential_assignment.py` holds the two readings
# equal on fuzzed text. The one addition is `limit`, which only `bounded` passes: the reading goes
# no further than it, and a value that reaches it is reported as ending there.


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


#: The word `_bearer_end` reads after a token run, to see whether it ends in a scheme word: a
#: token's characters and the ``%``, ``:`` and ``=`` a header value glues them with. A word it
#: stops short of is no loss: the bearer rule's search reads on past the token it took.
_GLUE_RUN = re.compile("[A-Za-z0-9._~+/%:=-]*")

#: A bearer header's scheme word, which is never a credential's whole value: the token after it is
#: (issue #926, basecradle-router#497). The fleet's one definition (docs/telemetry.md → Credential
#: Scrubbing): ``bearer`` in any case, glued to anything before it, so a word is a scheme word when
#: it ends in these six letters, whatever comes before them (``Bearer``, ``abc-bearer``,
#: ``x"q"bearer``, ``(a)Bearer``). The router's `logfmt._SCHEME_WORD`, byte for byte in what it
#: matches (basecradle-router#510, merged as `a8e8f5f`; issue #934).
_SCHEME_WORD = re.compile(_SCHEME)
_SCHEME_WORD_LENGTH = len(_SCHEME_LETTERS)

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


def _token_after_a_scheme_word(text: str, floor: int, end: int) -> int | None:
    """Where the token starts, when what was read from `floor` ends at `end` in a bearer scheme word
    (`_SCHEME_WORD`), with a word on its record after its blanks; else ``None``. A value never ends
    between a scheme word and its token, whatever the scheme word is glued to: a word the shell
    reads (``x"q"bearer``), or one a tab glues to the value (``(a)<TAB>Bearer``, the canary's run
    taking the scheme word, basecradle-router#502's review)."""
    if end - floor < _SCHEME_WORD_LENGTH or not _SCHEME_WORD.fullmatch(
        text, end - _SCHEME_WORD_LENGTH, end
    ):
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
    2. **Across blanks** to the next word, while the word ends in a bearer scheme word
       (`_SCHEME_WORD`, glued to anything before it, basecradle-router#510), so a doubled scheme
       word goes with the token after both (issue #926), and wherever the value would end just after
       one, a scheme word a tab glues to it included (`_token_after_a_scheme_word`). A scheme word
       with nothing after it on its record is the value alone.
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
            if (token := _token_after_a_scheme_word(text, at, word_end)) is not None:
                at = token  # a word after the blanks, on the same record
                if at >= run_end:
                    run_end = _run_end(text, at, stop)
                    end = max(end, run_end)
                continue
            lead = _LEAD_WORD.match(text, at)
            if lead is None or word_end > lead.end() or not _reached(text, lead.end()):
                # No lead word, one the shell reads past, or nothing after its tabs: the value ends,
                # unless it ends just after a scheme word.
                if (token := _token_after_a_scheme_word(text, floor, end)) is None:
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
    re.compile(r"(?<!\S)(?P<label>\S*?" + re.escape(_HEARTBEAT) + r")\S*"),
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
    - **A literal rule's label** (issue #931): ``?token=``, ``api_key=``, an ``x-api-key:`` header,
      a heartbeat URL or a credential-named JSON member up to its value's opener (issue #943),
      reached when the head, its trailing blanks gone, ends past the label and inside what its
      value may take (`_OPEN_LABELS`; a member's value to its close, `_members`): ``?token= tail``
      cut to ``?token=…`` scrubs to ``[redacted]``, ten characters for a cap of nine. A blank
      between the label and the marker would keep the label for some rules and not for others (an
      ``x-api-key:`` value and a scheme word's token are read across blanks), so a reached label
      goes whole, as a name does.

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
    items = list(heapq.merge(found, *labels, _members(text, cut)))
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
    """`text` with every secret the definition knows redacted: `scrub_counted`'s text."""
    return scrub_counted(text)[0]


def scrub_counted(text: str) -> tuple[str, int]:
    """`text` with every secret the definition knows redacted, and how many redactions it wrote.

    Six rules, each reading `text` as it came: the agent's own configured secrets by value
    (`_value_spans`, issue #999), the heartbeat URL (`_HEARTBEAT_RUN`), the literal
    rules (`_X_API_KEY`, `_QUERY`, `_API_KEY`, a credential-named JSON member's string value
    (`_MEMBER`), `_BEARER`), the value of every credential-named assignment (`_credential_spans`,
    which `redact_credentials` renders alone), and the prefix rules
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
    values, spans = _spans(text)
    out: list[str] = []
    pos = count = 0
    for start, end, marker in _redactions(text, values, spans):
        out += [text[pos:start], marker]
        count += text[start:end] != marker  # a marker already there is no new redaction
        pos = end
    out.append(text[pos:])
    return "".join(out), count


def scrub_parts(parts: Sequence[str]) -> list[str]:
    """`parts`, one text written in pieces, redacted as that one text: `scrub_parts_counted`'s."""
    return scrub_parts_counted(parts)[0]


def scrub_parts_counted(parts: Sequence[str]) -> tuple[list[str], list[int]]:
    """Each of `parts` as its share of one redaction of them all, and how many redactions each holds.

    **A text the harness holds in pieces is still one text** (issue #1020). A reply cut off at its
    output budget and the resume that finished it are one reply in two messages, and a secret the
    cut fell inside is in neither: its opening may match a shape and be blanked, while the rest opens
    the continuation, where nothing recognizes it, and is written as it stands. So the pieces are
    redacted together: the rules read ``"".join(parts)``, and what each piece keeps is what that
    redaction leaves of it. A redaction is written once, in the **last** piece it took something
    from, and what it took from the pieces before is gone from them. ``"".join`` of the result is
    therefore the joined text redacted, and no piece keeps any part of a secret the joined text holds
    whole. The last piece, never the first: a piece saved with a marker at its end reads, on the next
    save, as a cut a redaction already reached (below), so a marker the joint reading wrote there
    would take the next piece's words again on every save; written after the cut, it is read back
    as itself.

    **Never less than each piece alone.** Joining can make a rule decline what it takes in a piece
    (a key glued to the last word of the piece before reads as a word, `_is_word`), so every piece's
    own spans are taken too, read where they stand in the joined text: the union `scrub` keeps among
    its rules, kept across the pieces. One piece is `scrub_counted` exactly.

    **A piece that ends in a redaction marker is a cut the redaction already reached**
    (`_through_the_cut`): a fragment saved before its resume existed was redacted alone, so a key it
    ended inside is ``[redacted]`` there, and nothing of the key is left to recognize the rest by. So
    the run of key and URL characters that opens the next piece (`_KEY_RUN`) goes with that marker,
    added to the redactions the rules made and never changing one of them. It errs toward
    redaction, and is bounded to that one run: what the model glued onto a marker that ended a key
    exactly at the cut goes too, once (the marker then stands after the cut and reads back as
    itself), and a quote, a bracket, a comma, a blank or a letter outside ASCII is never taken.

    A redaction is counted in every piece it took something from, so a secret across a cut counts
    once in each piece it was split across.
    """
    whole = "".join(parts)
    values, spans = _spans(whole)
    if len(parts) > 1:
        streams = [spans]
        offset = 0
        for part in parts:
            own_values, own = _spans(part)
            values |= {offset + start for start in own_values}
            streams.append(_shifted(own, offset))
            offset += len(part)
        spans = heapq.merge(*streams)
    redactions: Iterable[tuple[int, int, str]] = _redactions(whole, values, spans)
    if len(parts) > 1:
        redactions = _with_runs(list(redactions), _through_the_cut(parts, whole))
    starts = [0]
    for part in parts[:-1]:
        starts.append(starts[-1] + len(part))
    shares: list[list[str]] = [[] for _ in parts]
    counts = [0] * len(parts)

    def touched(begin: int, end: int) -> Iterator[int]:
        """Each piece holding some of ``whole[begin:end]``; an empty piece holds nothing."""
        return (
            k
            for k, first in enumerate(starts)
            if parts[k] and first < end and begin < first + len(parts[k])
        )

    def keep(begin: int, end: int) -> None:
        for k in touched(begin, end):
            shares[k].append(whole[max(begin, starts[k]) : min(end, starts[k] + len(parts[k]))])

    pos = 0
    for start, end, marker in redactions:
        keep(pos, start)
        pieces = list(touched(start, end))
        shares[pieces[-1]].append(marker)  # in the last piece it took from (see above)
        if whole[start:end] != marker:  # a marker already there is no new redaction
            for k in pieces:
                counts[k] += 1
        pos = end
    keep(pos, len(whole))
    return ["".join(share) for share in shares], counts


def _spans(text: str) -> tuple[set[int], Iterator[tuple[int, int]]]:
    """Every rule's spans of `text`, in order, and where each credential name's value starts in it,
    empty values included: what `_redactions` merges (`scrub_counted`)."""
    values: set[int] = set()
    assignments: list[tuple[int, int]] = []
    for start, end in _credential_spans(text):
        values.add(start)
        if end > start:
            assignments.append((start, end))
    spans = heapq.merge(
        _value_spans(text),
        assignments,
        (_heartbeat_span(match) for match in _HEARTBEAT_RUN.finditer(text)),
        _literal_spans(_X_API_KEY, text, values),
        _literal_spans(_QUERY, text, values),
        _literal_spans(_API_KEY, text, values),
        _member_spans(text),
        _bearer_spans(text),
        (match.span() for match in _PREFIX_RULES.finditer(text) if not _is_word(match)),
    )
    return values, spans


#: What `_through_the_cut` takes after a marker that ends a piece: the characters of a key, a token
#: and a URL, ASCII only, and nothing that closes a quoted value, a list or a sentence's clause.
_KEY_RUN = re.compile(r"[A-Za-z0-9._~+/=%&?:@-]+")


def _through_the_cut(parts: Sequence[str], whole: str) -> list[tuple[int, int]]:
    """A span from each `REDACTED` a piece ends in through the `_KEY_RUN` that opens the text after
    it (`scrub_parts_counted`), in order."""
    spans: list[tuple[int, int]] = []
    end = 0
    for part in parts:
        end += len(part)
        if part.endswith(REDACTED) and (run := _KEY_RUN.match(whole, end)) is not None:
            spans.append((end - len(REDACTED), run.end()))
    return spans


def _with_runs(
    redactions: list[tuple[int, int, str]], runs: list[tuple[int, int]]
) -> list[tuple[int, int, str]]:
    """``redactions`` with each of ``runs`` added: a redaction a run overlaps or touches grows to
    cover it, keeping its own marker (`REDACTED_VALUE` if any it joins had one), and a run that
    meets none is written as `REDACTED`. Added after the rules' merge, never fed into it, so a run
    can only add to what the rules took (`_past_marker` is not monotone in a span's end)."""
    merged: list[tuple[int, int, str]] = []
    for start, end, marker in sorted([*redactions, *((s, e, REDACTED) for s, e in runs)]):
        if merged and start <= merged[-1][1]:
            first, last, kept = merged[-1]
            value = REDACTED_VALUE if REDACTED_VALUE in (kept, marker) else REDACTED
            merged[-1] = (first, max(last, end), value)
        else:
            merged.append((start, end, marker))
    return merged


def _shifted(spans: Iterable[tuple[int, int]], offset: int) -> Iterator[tuple[int, int]]:
    """`spans` of a piece, read where the piece stands in the text it is part of."""
    return ((start + offset, end + offset) for start, end in spans)


def _redactions(
    text: str, values: set[int], spans: Iterator[tuple[int, int]]
) -> Iterator[tuple[int, int, str]]:
    """Each redaction `scrub` writes in `text`, in order: where it starts, where it ends, its marker.

    The spans are merged where they overlap or touch, and a span that starts where a credential
    name's value starts (`values`) is written as `REDACTED_VALUE`, taking what a second reading of
    that marker would take (`_past_marker`); every other span is written as `REDACTED`."""
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
        yield start, end, REDACTED_VALUE if marked else REDACTED


def _past_marker(text: str, at: int) -> int:
    """Where a value ends once `scrub` has written its marker for `text` up to `at`: `at` itself when
    a marker ends a value there (`_MARKER_ENDS`), else as far as reading the marker again would go.

    A span another rule ends can end where a value would not (``OPENAI_API_KEY=x-api-key:
    `$'sk-)``: the header's value stops at the ``'``), and a marker written there is read again
    with what follows it as one shell word. So the span takes what that reading takes: the rest of
    the word the marker begins, then the value of a credential name it ends at, or the token after
    the scheme word it ends in, each read as `_value_end` reads one. A value `_value_end` reads
    ends where a marker ends one, so a marker `scrub` writes reads back as itself."""
    if at == len(text) or text[at] in _MARKER_ENDS:
        return at
    word_end, at_a_name = _shell_word_end(text, at)
    if at_a_name:
        return _value_end(text, word_end)
    token = _token_after_a_scheme_word(text, at, word_end)
    return word_end if token is None else max(word_end, _value_end(text, token))


def _literal_spans(rule: re.Pattern[str], text: str, values: set[int]) -> Iterator[tuple[int, int]]:
    """Every match of a literal rule, read on through a label its value ends in, its label left out
    when a credential's value starts after it (`scrub`).

    A value can end in the next value's own label (``api_key=x,api_key="eyJ…"``, whose value is
    ``x,api_key=`` up to the quote; ``x-api-key: "x-api-key": "K9…"``, ``x-api-key: x-api-key:Bearer
    K9…``), and the value after it would then be matched by nothing, since each search starts past
    the last match (the review of #929, issue #934). So while a match whose label stands in the
    value reaches further, it is read too, and the span ends where the last one does. Only two
    places hold such a label, because a label anywhere else is followed by the run the value
    already read, which stops where the value stops, and is left the same odd run of backslashes
    before that stop (`_literal_end`, whatever either opened with: issue #937):

    - **Where the value stops**: the label ends there, before the backslashes that escape the stop
      (``api_key=x,api_key=\\"eyJ…``), or before the one scheme word its value may open with
      (``api_key=x,api_key=Bearer abc``).
    - **Just before the value's glued scheme word**, when the value ended in one and its token
      (``api_key=\\"api_key=bearer<TAB>X-API-KEY=``): that label reads the scheme word as its own
      and then its run, which is wider than a token and goes on past it.

    So only `_LITERAL_REACH` before each is searched, and the rule stays linear. In order, since each
    span starts no earlier than the one before it ends."""
    reach, labelled = _LITERAL_REACH[rule], "label" in rule.groupindex
    pos = 0
    while (match := _literal_search(rule, text, pos)) is not None:
        last, end = match, _literal_end(match)
        while True:
            floor, stop = last.start("value"), end
            while stop > floor and text[stop - 1] == "\\":
                stop -= 1
            places: list[Iterable[int]] = [range(max(floor, stop - reach), end)]
            if (glued := last.start("glued")) >= 0:
                places.append(range(max(floor, glued - reach), glued))
            inner = next(
                (
                    m
                    for place in places
                    for i in place
                    if (m := rule.match(text, i)) and not _closes(m) and _literal_end(m) > end
                ),
                None,
            )
            if inner is None:
                break
            last, end = inner, _literal_end(inner)
        label_end = match.end("label") if labelled else -1
        yield (label_end if label_end in values else match.start()), end
        pos = end


def _literal_search(rule: re.Pattern[str], text: str, pos: int) -> re.Match[str] | None:
    """The first match of a literal rule from `pos` whose opener opens a value (`_closes`)."""
    while (match := rule.search(text, pos)) is not None and _closes(match):
        pos = match.end("opener")
    return match


def _closes(match: re.Match[str]) -> bool:
    """Whether a literal rule's `match` opened its value at the close of the string it sits in.

    So it did when the opener is a ``"``, escaped or not, and the value it opens is JSON's structure
    and nothing else (`_STRUCTURE`) before any backslashes that escape the quote it stops at:
    ``{"h":"api_key=","k":"v"}``, whose ``"`` after the label is the string's own close and whose
    value is the ``,`` after it (issue #939). A close is followed by structure, and no key is a run
    of it, so the quote opens nothing, and the match is no value. A ``'`` closes no JSON string, so
    an opener that is one still opens. A value of backslashes alone follows a real opener
    (``api_key=\\"\\\\"``, a quote and an escaped backslash), since structure, never a backslash,
    follows a close. The query rule's value has no opener (issue #939 left it as it was), so this
    never judges it."""
    opener = match.groupdict().get("opener")
    if not opener or opener[-1] != '"':
        return False
    value = match.group("value").rstrip("\\")
    return bool(value) and set(value) <= _STRUCTURE


def _literal_end(match: re.Match[str]) -> int:
    """Where a literal rule's `match` ends as a span: its end, but before the odd run of backslashes
    that escapes the quote its value stops at, whatever the value opened with (issues #934, #937).

    After an odd run the quote is escaped, and the run is left whole: how much of it is the quote's
    escape and how much the value's own escaped backslashes depends on how deeply the quote is
    nested, which no reading of the run can tell, and every JSON string the line held parses as it
    did only while the run stays. So the key goes and no string it was nested in loses its close to
    the run, the line's own or one inside it (``{"h":"api_key=\\"K9…\\""}`` scrubs to
    ``{"h":"[redacted]\\""}``, and ``{"log": "{\\"h\\":\\"api_key=x\\\\\\\\\\"}"}`` to
    ``{"log": "{\\"h\\":\\"[redacted]\\\\\\\\\\"}"}``). After an even run the quote is a real one, the
    backslashes escaping each other, and they go with the value. A backslash is never a key's. (An
    inner string's real close after an even run, none included, reaches the line as an odd run and
    is read as an escaped opener, which opens nothing where structure and the next string follow it
    (`_closes`), and loses its close before any other value.)"""
    end, text = match.end(), match.string
    if text[end : end + 1] != '"':
        return end
    run = end
    while run > match.start("value") and text[run - 1] == "\\":
        run -= 1
    return run if (end - run) % 2 else end


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
        if (
            word - at < _SCHEME_WORD_LENGTH
            or text[word - _SCHEME_WORD_LENGTH : word].lower() != _SCHEME_LETTERS
        ):
            return end
        if (gap := _BEARER_GAP.match(text, word)) is None:
            return end
        at = gap.end()


def _heartbeat_span(match: re.Match[str]) -> tuple[int, int]:
    """The span of a space-free run holding a heartbeat URL that the box's rule would redact."""
    run = match.group()
    start = run.index(_HEARTBEAT)
    scheme = _URL_SCHEME.search(run, 0, run.rindex(_HEARTBEAT, 0, len(run) - 1))
    if scheme:
        start = min(start, scheme.start())
    bearer = _TRAILING_BEARER.search(run, start)
    return match.start() + start, match.end() - (len(bearer.group()) if bearer else 0)


def _heartbeat(match: re.Match[str]) -> str:
    """A space-free run holding a heartbeat URL, redacted from where the box's rule would begin."""
    start, end = _heartbeat_span(match)
    text = match.string
    return text[match.start() : start] + REDACTED + text[end : match.end()]


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


# --- by value: the agent's own configured secrets (issue #999) -----------------------------------

_CONFIGURED = _section("configured_values")

#: The shortest configured value a bare match blanks. `@briggs`'s rule for memory (issue #999,
#: recorded by the capital 2026-11-23), and the floor the vendor-key rules use: a shorter bare match
#: would take ordinary words out of a memory that does not forget. A shorter value is still blanked
#: where a credential name introduces it, by the assignment rule.
CONFIGURED_VALUE_FLOOR = _CONFIGURED["minimum_length"]

#: A whole name that names a credential, by the assignment rule's own words and tail: the test the
#: environment's names and a held credential's name are put to. The same words as `CREDENTIAL_NAME`,
#: never a second list, and anchored at both ends because here the name is all there is.
_WHOLE_CREDENTIAL_NAME = re.compile("[A-Z0-9_]*" + _WORD_AND_TAIL)


def names_a_credential(name: str) -> bool:
    """Whether `name` (an environment variable's, an MCP server's ``env`` key or header) names a
    credential: upper-cased with ``-`` read as ``_``, it is capitals, digits and ``_`` ending in one
    of the assignment rule's words and, optionally, its tail (``AI_API_KEY``, ``X-API-Key``)."""
    return _WHOLE_CREDENTIAL_NAME.fullmatch(name.upper().replace("-", "_")) is not None


#: Every literal label name the file lists for a value that is a secret by its name alone: the JSON
#: members and the headers (lower-cased, as they are matched in any case).
_SECRET_FIELDS = frozenset(
    name.lower() for section in ("json_members", "headers") for name in _section(section)["names"]
)


def names_a_secret_field(name: str) -> bool:
    """Whether a structured field called `name` (a tool call's argument, a nested map's key) holds
    a secret by its name alone: a credential's name (`names_a_credential`), or one of the file's
    JSON-member or header names, in any case (``password``, ``client_secret``, ``x-api-key``)."""
    return names_a_credential(name) or name.lower() in _SECRET_FIELDS


def stable_cut(text: str, head: int, tail: int) -> tuple[int, int]:
    """Where an excerpt of `text` kept as ``text[:head] + <marker> + text[tail:]`` may end its head
    and start its tail so that `scrub` reads it back as itself (issue #999).

    A persisted excerpt is scrubbed again on every save, so a cut that leaves a value open makes the
    next save read the marker as that value and grow the excerpt past its share: the cap's fixed
    point gone, and a marker naming a size no longer true. So the head backs off before every value
    or label the cut reaches, as `bounded`'s cut does (`_whole_values`), and neither end splits a
    redaction marker. It does not back out of a token run as `bounded` does: that guards a
    length-gated rule shown a cut key, and the text here is already redacted, so the backoff would
    only empty the head of a long word. The result never lengthens the excerpt."""
    head = _whole_values(text, head)[0]
    for marker in (REDACTED_VALUE, REDACTED):
        start = text.rfind(marker, max(0, head - len(marker) + 1), head + len(marker) - 1)
        if start != -1 and start < head < start + len(marker):
            head = start
        start = text.rfind(marker, max(0, tail - len(marker) + 1), tail + len(marker) - 1)
        if start != -1 and start < tail < start + len(marker):
            tail = start + len(marker)
    return head, max(head, tail)


def configured_values() -> list[str]:
    """The agent's own configured secrets, by value, longest first: every environment variable
    whose name names a credential (`names_a_credential`), and every credential the process holds as
    a `Secret` it was told is one, each at least `CONFIGURED_VALUE_FLOOR` characters long.

    Read on every call, never cached: the environment can change under a running process, and a
    cache of these values would be one more place a credential sits in plain text."""
    candidates = [value for name, value in os.environ.items() if names_a_credential(name)]
    candidates.extend(known_secrets())
    # Stripped, as a vendor echoes a key: an ``agent.env`` value can carry a trailing newline.
    values = {value.strip() for value in candidates}
    return sorted((v for v in values if len(v) >= CONFIGURED_VALUE_FLOOR), key=len, reverse=True)


def _value_spans(text: str) -> list[tuple[int, int]]:
    """Every place a configured secret stands in `text`, in order. Each value is found with
    `str.find`, never compiled into a pattern: `re` keeps every pattern it compiles in a module-level
    cache, and that would be a credential held in plain text for the life of the process."""
    spans: list[tuple[int, int]] = []
    for value in configured_values():
        at = text.find(value)
        while at >= 0:
            spans.append((at, at + len(value)))
            at = text.find(value, at + len(value))
    spans.sort()
    return spans


# --- a text file a tool saved (issue #999) --------------------------------------------------------

#: The suffixes of the text files `redact_file` will open, lower-cased: the page snapshots, console
#: and network logs and saved pages a browser tool writes, and the plain text, markup and data
#: formats any tool's capture is written in. **Anything else is never opened**: an image, a video,
#: an archive, a trace, a PDF, a file with no suffix. That is how an image is left byte-identical
#: without its bytes being read to tell.
TEXT_SUFFIXES = frozenset(
    {
        ".csv",
        ".har",
        ".htm",
        ".html",
        ".json",
        ".jsonl",
        ".log",
        ".markdown",
        ".md",
        ".ndjson",
        ".tsv",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)

#: The largest file `redact_file` reads. A capture bigger than this (a long network log) is left as
#: it is and the caller says so, rather than read whole into memory after every call that wrote it.
MAX_FILE_BYTES = 16 * 1024 * 1024


def is_text_file(path: Path) -> bool:
    """Whether `path`'s suffix is one `redact_file` opens (`TEXT_SUFFIXES`)."""
    return path.suffix.lower() in TEXT_SUFFIXES


def redact_file(path: Path) -> int | str:
    """Redact the text file at `path` in place: how many secrets it blanked, or why it left the file.

    Only a regular file with a text suffix (`TEXT_SUFFIXES`), no larger than `MAX_FILE_BYTES`, whose
    bytes are UTF-8, is changed, and only when `scrub` changes its text; any other file is left
    byte for byte as it was, a file that is not text never opened. The open refuses a final symlink
    (``O_NOFOLLOW``) and cannot block or take a terminal (``O_NONBLOCK``, ``O_NOCTTY``), and what it
    opened must still be a regular file.

    **Rewritten in place, never through a temporary file**: the same inode, owner and mode, and
    nothing a crash could leave beside the capture for someone to clean up. The cost is stated: a
    process killed during the write leaves the new text's head over the old text's tail, which is
    no worse than the file before. Every OS error is a reason, never a raise.
    """
    if not is_text_file(path):
        return "it is not a text file"
    flags = os.O_RDWR | os.O_NONBLOCK | os.O_NOCTTY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        return f"it could not be opened ({exc.strerror or exc})"
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return "it is not a regular file"
        if info.st_nlink > 1:  # rewritten in place, it would change every other name it has
            return "it has another hard link"
        if info.st_size > MAX_FILE_BYTES:
            return f"it is larger than {MAX_FILE_BYTES} bytes"
        with open(fd, "r+b", closefd=False) as handle:
            data = handle.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                return f"it is larger than {MAX_FILE_BYTES} bytes"
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                return "it is not UTF-8 text"
            redacted, count = scrub_counted(text)
            if redacted == text:
                return 0
            handle.seek(0)
            handle.write(redacted.encode("utf-8"))
            handle.truncate()
            handle.flush()
            os.fsync(fd)
    except OSError as exc:
        return f"it could not be redacted ({exc.strerror or exc})"
    finally:
        os.close(fd)
    return count
