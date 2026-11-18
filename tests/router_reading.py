"""The router's credential reading, carried verbatim for the parity test (issue #929).

Below the import is lines 114-451 of basecradle-router `src/basecradle_router/logfmt.py` at
`153fb58` (the merge of basecradle-router#506), byte for byte: the whole file's sha256 there is
`1f097b24803d3b25c2d246f58ce760186a0ee8e127fed8beae6500acdc8ad9dd`, and the excerpt's is
`d4e2e522229009a3e01ffe46d1e039cfd4b855a381d65fd34045dffd48bab2ed`, which `tests/test_credential_assignment.py`
checks. It is never edited here: when the router's reading changes, the excerpt is copied again
from the router's commit, and the commit, the line range and both checksums change with it. The
test holds `_observability.redact_credentials` and `_observability._value_end` to these
functions.
"""

# ruff: noqa

import re

# --- credential-named assignments (basecradle-router#491) -------------------------------------
#
# A line this repository renders never carries a credential-named assignment: the name stays,
# the value becomes REDACTED_VALUE, whatever the value looks like and whatever quoting surrounds
# it. "Names a credential" is the NOC's log-store canary's definition, its `credential_assignment`
# rule (basecradle-noc `observability/log-canary.json`), never a list of this repository's own:
#
#     [A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD|PASSPHRASE)(_BASE)?=[^ ",}]{8,}
#
# (widened by @origin on 2026-11-17, basecradle-noc#1547: a bare word is a name, and `_BASE` is the
# one tail after the word, basecradle-router#504).
#
# `tests/conftest.py` holds that pattern byte for byte, and the tests hold the pieces below to
# it: the words, the value's stop characters and its floor must spell it exactly, and the canary's
# own pattern is the oracle every redacted line is checked against.

#: The words that make a name a credential's, in the canary's order.
CREDENTIAL_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSPHRASE")

#: The canary reads a value as a run of anything but these, at least :data:`CANARY_VALUE_FLOOR`
#: characters long, from the character after the ``=``.
CANARY_VALUE_STOPS = ' ",}'
CANARY_VALUE_FLOOR = 8

#: The one tail the canary allows after a credential word, before the ``=`` (``SECRET_KEY_BASE=``).
#: No other: ``GH_TOKEN_FILE=``, ``API_KEY_ID=`` and ``MAX_TOKENS=`` name no credential.
CREDENTIAL_TAIL = "_BASE"

#: A credential name, up to the ``=`` it is assigned with. The canary's own name pattern,
#: ``[A-Z0-9_]*(…)(_BASE)?=``, run unanchored in a backtracking engine (Python's ``re``, jq's
#: Oniguruma) tries every start inside a long run of capitals and is quadratic in it: 50,000 of
#: them took jq 30 seconds, and a translator that stalls stops reading the pipe ``claude`` writes
#: into, which hangs the wake. This spelling matches before exactly the same ``=``s (pinned against
#: the canary's pattern by a test), and is linear: it starts only where a run of name characters
#: starts and extends lazily to a credential word and its tail. ``wake-runner``'s ``jq`` fallback
#: carries the same spelling.
CREDENTIAL_NAME = (
    r"(?<![A-Z0-9_])[A-Z0-9_]*?(?:" + "|".join(CREDENTIAL_WORDS) + f")(?:{CREDENTIAL_TAIL})?="
)
_CREDENTIAL_NAME = re.compile(CREDENTIAL_NAME)

#: What stands where a credential's value stood. It must read as one value to the redactor and as
#: no value to the canary: a single-quoted word, so redacting it again (bounded, then rendered)
#: leaves it as it is, with a space at its fourth character, so the canary's run after the ``=``
#: is ``'<a``, far under its floor, whatever follows, and still under it when a bound cuts the
#: marker and adds its ``…``. ``<redacted>`` alone is ten characters the canary reads as a value,
#: so ``GH_TOKEN=<redacted>`` would be the finding it replaced.
REDACTED_VALUE = "'<a redacted value>'"

#: Every character journald, or Python reading a line, starts a new record or line at; a CRLF is
#: one. The translator cuts model-written text at these (``claude_stream.quoted``), and
#: ``wake-runner``'s ``jq`` fallback splits by this pattern, spelled for jq (pinned by a test).
_BREAK_CHARS = "\n\r\x00\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"
RECORD_BREAKS = "\r\n|[" + _BREAK_CHARS + "]"

#: The canary's value run, where it ends in every record the text can be written into: at one of
#: the canary's own stops, or at a record break, since no record holds a run across one.
_CANARY_RUN = re.compile("[^" + re.escape(CANARY_VALUE_STOPS + _BREAK_CHARS) + "]*")

#: The characters that end an unquoted shell word: the shell's blanks and its newline.
_SHELL_BLANKS = " \t\n"


#: The end of a credential name, its ``=`` included: a credential word, the one tail, the ``=``. The
#: canary's name is unanchored, so whatever comes before the word, this is a name's end.
_NAME_END = re.compile("(?:" + "|".join(CREDENTIAL_WORDS) + f")(?:{CREDENTIAL_TAIL})?=\\Z")
_NAME_END_REACH = max(len(word) for word in CREDENTIAL_WORDS) + len(CREDENTIAL_TAIL) + 1


def _shell_word_end(text: str, start: int) -> tuple[int, bool]:
    """Where the shell word from ``start`` ends: at a blank outside every quote, or the end.

    One word as the shell reads it, so a value is taken whole whatever surrounds it: a single- or
    double-quoted string, an ANSI-C ``$'…'``, a ``$(…)``, ``${…}`` or ``(…)`` (each nesting as the
    shell's does, quotes inside included), a backquoted command, and a backslash escape (but of a
    tab outside every quote: the backslash is then the word's last character, since the canary's
    run crosses the tab and a scheme word after it is the value's, #502's review). Only a
    space, a tab or a newline ends an unquoted word, as in the shell, so no other character can cut
    a value short. A quote or a substitution still open at a record break ends there: it is prose
    (an apostrophe, a markdown backquote) far more often than a value that spans lines, and read
    on it would hide the rest of a reply. Iterative, with its own stack, so no depth of nesting can
    exhaust the interpreter's.

    The word also ends just past a credential name's ``=`` that stands outside every quote and
    substitution in it (``x,DB_PASSWORD=…``), and then says so: the rest of the word is that name's
    value (#504's review, from basecradle-harness#928). A name inside a quote is the quote's text.
    Returns where the word ends, and whether it ends at a name's ``=``.
    """
    stack: list[str] = []  # what closes each open context: ' " ` ) } and $' for ANSI-C
    i = start
    n = len(text)
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
                # A backslash before a tab is the word's last character (#502's review).
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
#: a token's, as a header value spells one. The box's rule builds its glued scheme word from this
#: one spelling (``tests/credential_rule.py``, ``GLUED``), and ``wake-runner``'s ``jq`` fallback
#: carries it verbatim (pinned by a test).
SCHEME_GLUE = "A-Za-z0-9._~+/%:=-"

#: A bearer header's scheme word, which is never a credential's whole value: the token after it is
#: (#497). Alone it is the word in any case; glued to a value only as HTTP spells it, so a value
#: merely ending in the letters (``abc-bearer``) is no header.
_SCHEME_WORD = re.compile(f"(?i:bearer)|[{SCHEME_GLUE}]*Bearer")

#: What stands between a scheme word and its token: blanks, never a newline, as the box reads them.
_SCHEME_BLANKS = re.compile("[ \t]+")

#: The characters, beyond a blank, a canary stop and a record break, that a word may not hold
#: unquoted before the tab ahead of a bearer scheme word (#502): a quote's opener, a backslash's
#: escape and a parenthesis, whose reading a regular expression cannot nest. Each of the three
#: copies builds its lead word from this one spelling: the print site's :data:`_LEAD_WORD`, the
#: box's rule (``tests/credential_rule.py``, ``LEAD_WORD``) and ``wake-runner``'s ``jq`` fallback
#: (pinned by a test).
LEAD_OPENERS = "'`\\("


def _run_char(but: str = "") -> str:
    """A character of the canary's run (a tab is one), and none of ``but``."""
    return "[^" + re.escape(CANARY_VALUE_STOPS + _BREAK_CHARS + but) + "]"


#: A word the canary's run crosses whole, and the tabs after it: what stands before a scheme word
#: that a tab glues to a value (``x<TAB>Bearer``, ``'q'<TAB>KBearer``), so the token after that
#: scheme word goes with the value. Its pieces each close inside the run: a bare character, a
#: backslash and the character after it, a single-quoted piece and a backquoted one (in which a
#: backslash escapes), so each piece has one reading and the word one end in every engine, the
#: shell word's. A backslash before the tab is the word's last character, as the shell reading has
#: it here (``a\<TAB>Bearer``: the canary's run crosses the tab). A word holding a quote that never
#: closes there, a ``"``, a ``(``, a blank or a stop is no lead word, and no scheme word after it is
#: read as one: that is the definition, in all three copies alike.
_LEAD_PIECE = (
    "(?:"
    + "|".join(
        [
            _run_char("\t" + LEAD_OPENERS),  # a bare character: neither a tab nor an opener
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
#: canary's own run ends inside the marker, at its space), as the box's rule keeps one
#: (``tests/credential_rule.py``, ``MARKER_END``), so a value redacted where it is bounded and again
#: where it is rendered stays one marker.
_MARKER_ENDS = frozenset("\t" + CANARY_VALUE_STOPS + _BREAK_CHARS)

#: What a word after a tab may not start with, to be reached across it: a blank, a ``,`` or a ``}``
#: (the canary's stops, which end its run and are structure on a JSON line) and a record break, so
#: no word the value reaches this way starts in the next record. A ``"`` may: it opens the word's
#: quoted piece, as the box's parity reading takes it.
_UNREACHED = frozenset(CANARY_VALUE_STOPS.replace('"', "") + _BREAK_CHARS)


def _reached(text: str, at: int) -> bool:
    return at < len(text) and text[at] not in _UNREACHED


def _token_after_a_glued_scheme_word(text: str, floor: int, end: int) -> int | None:
    """Where the token starts, when a value read from ``floor`` ends at ``end`` just after a bearer
    scheme word a tab glues to it (``(a)<TAB>Bearer tok``: the canary's run takes the scheme word),
    with a word on its record after its blanks; else ``None``. A value never ends between a scheme
    word and its token, whatever stands before the tab (the review of #502). The lead words are
    the shapes the box and the ``jq`` fallback read word by word; this rule is the print site's net
    under every other."""
    tab = text.rfind("\t", floor, end)
    if tab < 0 or not _SCHEME_WORD.fullmatch(text, tab + 1, end):
        return None
    if (blanks := _SCHEME_BLANKS.match(text, end)) is None or not _reached(text, blanks.end()):
        return None
    return blanks.end()


def _value_end(text: str, start: int) -> int:
    """Where a credential's value from ``start`` ends: past every word the value reaches, each read
    whole as a shell word, and past the canary's run.

    A value reaches its first word, after any tabs it opens with, unless it opens with a marker
    where the canary's run would end after it, which is the whole value. From a word it reaches:

    1. **Across a tab** to the next word, while the word is a lead word (:data:`_LEAD_WORD`): the
       canary's run crosses a tab, so a word after one is still the value's (#502). The next word
       must start where a run could go on (:func:`_reached`).
    2. **Across blanks** to the next word, while the word is a bearer scheme word
       (:data:`_SCHEME_WORD`), so a doubled scheme word goes with the token after both (#497), and
       wherever the value would end just after a scheme word a tab glues to it, whatever stands
       before the tab (:func:`_token_after_a_glued_scheme_word`). A scheme word with nothing after
       it on its record is the value alone.
    3. **Into a value of its own** at a credential name's ``=`` outside every quote in the word
       (``x,DB_PASSWORD=<TAB>…``): the redactor's next search starts past that ``=``, so the name's
       value is read here, by this same reading, and goes with this one (#504's review).

    Each word is read once, the canary's run once from each place the value goes on past it, and the
    walk stops where a word's shell reading runs past its lead word's tab (a ``${`` the lead word
    reads as plain text): the value already reaches that far. So the value is linear in its
    length."""
    run_end = _CANARY_RUN.match(text, start).end()
    end = run_end
    at = start
    while True:  # a value from `at`
        floor = at
        if text.startswith(REDACTED_VALUE, at):
            after = at + len(REDACTED_VALUE)
            if after == len(text) or text[after] in _MARKER_ENDS:
                return max(end, after)
        if (tabs := _TABS.match(text, at).end()) > at:
            if not _reached(text, tabs):
                return end  # nothing after the tabs is the value's: a blank, a stop or a break
            at = tabs
        while True:  # the words it reaches
            word_end, at_a_name = _shell_word_end(text, at)
            end = max(end, word_end)
            if at_a_name:
                break  # a credential name's `=`: its value is read as one from here
            if _SCHEME_WORD.fullmatch(text, at, word_end) and (
                blanks := _SCHEME_BLANKS.match(text, word_end)
            ):
                token = blanks.end()
                if _reached(text, token):  # a word after the blanks, on the same record
                    at = token
                    if at >= run_end:
                        run_end = _CANARY_RUN.match(text, at).end()
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
                    run_end = _CANARY_RUN.match(text, at).end()
                    end = max(end, run_end)
                continue
            at = lead.end()
        at = word_end
        if at >= run_end:
            run_end = _CANARY_RUN.match(text, at).end()
            end = max(end, run_end)


def redact_credentials(text: str) -> str:
    """``text`` with every credential-named assignment's value replaced by :data:`REDACTED_VALUE`.

    The name stays. The value (:func:`_value_end`) is every shell word it reaches from the
    character after the ``=`` (:func:`_shell_word_end`), so a quoted value or a ``"$(…)"`` goes
    whole, and the canary's own run, so whatever the canary would read as the value is gone in every
    record the text is written into, a prose line that keeps a tab included. It reaches across a tab
    after a lead word (#502), past a bearer scheme word to its token (#497), and into the value of a
    credential name standing in it outside a quote (#504's review). An empty value (a space, a
    record break or the text's end follows the ``=``) has nothing to hide and is left as it is.
    Linear in ``text``, and idempotent: a marker where the canary's run would end after it is a
    whole value, so a value redacted where it is bounded and again where it is rendered stays one
    marker.
    """
    if "=" not in text:
        return text
    out: list[str] = []
    pos = 0
    while (match := _CREDENTIAL_NAME.search(text, pos)) is not None:
        start = match.end()
        end = _value_end(text, start)
        out.append(text[pos:start])
        if end > start:
            out.append(REDACTED_VALUE)
        pos = end
    out.append(text[pos:])
    return "".join(out)
