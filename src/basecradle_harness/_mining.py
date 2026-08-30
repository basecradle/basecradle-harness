"""The mining boundary: what a mining memory provider may be fed, and what it may never be.

**The invariant** (founder-stated, 2026-08-29 — issue #438):

> A mining memory provider mines exactly two things: **the text sent into the harness** (a
> peer's message) and **the LLM's output** (the agent's own reply). *Nothing the harness
> composes may ever reach the palace* — not the charter, not any part of the Turn-0 brief
> (the now line, the step budget, `initialize.md`, the tool manifest, the defect/safety
> notices, the dashboard, the recalled-memory section itself), not tool results.

The code has *claimed* that boundary since the memory seam shipped — `_wake._observe`'s
docstring promised the hook "is handed the *dialogue* only (user text + the agent's reply —
no briefs, no tool dumps)". The claim was not enforced anywhere, and it was false on four
paths at once (see `_wake._observe`). @briggs read the proof off his own Turn-0 recall: two
of five injected hits were copies of the **pre-0.112.0 recall heading itself**, mined out of
his own brief and served back to him as memory.

**Why a filter is not the fix, and is shipped anyway.** Enforcement lives at the leaking
paths — the wake mines a *dialogue* rendering of an item rather than the model-facing one,
never mines its own canned notes, and no longer mines a compaction summary (`_wake`). This
module's `strip_injected` is a second line only: it removes the recall block's *own* framing
literals, which are the one class of scaffolding that can round-trip through a legitimately
mined turn (the model reads the block and quotes it back in its reply, which is genuinely
LLM output and genuinely minable). It removes nothing else, because a filter that reached
further would start editing dialogue.

**The catalog is the other half, and it faces backwards.** `catalog()` enumerates the
harness-composed text that *could* have crossed the boundary before it was closed, so
`basecradle-harness-scrub-palace` can find and delete it in a palace that is already
polluted. It is deliberately **assembled from the constants the harness composes from**,
never re-typed here: a catalog that spells a marker for itself drifts from the writer the
first time the wording is edited, silently, with nothing failing — the same reason
`_context._SUMMARY_MARKER` and `is_summary` live together. The one exception is
`_LEGACY_RECALL_HEADING`, a constant that no longer exists in the code at all: it is
recorded here with its provenance because a palace mined before 0.112.0 is full of it.

**Matching is exact-literal and never fuzzy** (`classify`). A chunk is scrubbable only when
*every* line in it is scaffolding — so a real memory that merely mentions memory, tools, or
the brief can never match, and a chunk that mixes a quoted scaffolding line with real
content is reported for review and deleted by nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from basecradle_harness._mempalace import _CLOSE_TAG, _INJECTED_HEADING, _OPEN_TAG

#: The recall heading MemPalace's `context` hook injected **before** 0.112.0 (PR #135 through
#: PR #437). It exists in no source file any more, so it is spelled here — the one catalog entry
#: that has to be, and the reason this module exists: every palace mined on a pre-0.112.0 harness
#: has copies of it, and nothing in the code would name it. Recorded with its provenance rather
#: than as a bare string, so a later reader can tell a historical literal from a live one.
_LEGACY_RECALL_HEADING = "Relevant memories from past conversations (across all your timelines):"

#: The literals `strip_injected` removes from text on its way into a mining provider. Deliberately
#: **only** the recall block's own framing: the model reads this block every wake, and a reply that
#: quotes it is real LLM output on a path that is legitimately mined — the one class of scaffolding
#: the closed boundary still cannot exclude structurally. Everything else the harness composes is
#: kept out at its source, which is why nothing else is listed here (see the module docstring).
INJECTED_RECALL_LITERALS = (_INJECTED_HEADING, _LEGACY_RECALL_HEADING, _OPEN_TAG, _CLOSE_TAG)

#: `INJECTED_RECALL_LITERALS` as one case-insensitive alternation, compiled once. Longest first, so
#: an alternation never matches a shorter literal that is a prefix of a longer one and leaves its
#: tail behind — the ordering is the correctness, not the speed.
_INJECTED = re.compile(
    "|".join(
        re.escape(literal) for literal in sorted(INJECTED_RECALL_LITERALS, key=len, reverse=True)
    ),
    re.IGNORECASE,
)

#: Shortest normalized unit that may prove a line is scaffolding. Below it a fragment is not
#: *distinctive* — "1." or "Your tools:" appears inside plenty of real conversation — and a
#: catalog that matched on one would delete memories for the shape of their punctuation. Lines
#: shorter than this are still allowed to *ride along* in a scrubbable chunk, but only when they
#: carry no letters or digits at all (`_is_filler`): a bullet, a quote marker, a blank.
MIN_UNIT_CHARS = 48

#: Leading markers stripped before a line is matched. MemPalace's ``extract_mode="exchange"``
#: files quote the user turn with ``> ``, the recall renderer bullets each hit with ``- ``, and a
#: chunk can carry either (or both, nested) around the very text being classified.
_LEAD = re.compile(r"^[\s>*\-•]+")

#: Runs of whitespace collapse to one space before comparison, so a literal that was re-wrapped
#: on its way through a model still matches the constant it came from. This is the *only*
#: normalization: no stemming, no case-insensitive-except-casefold, no similarity.
_SPACE = re.compile(r"\s+")

#: Any `<mempalace-recall>` tag literal, in any casing — dropped from a line before it is
#: classified, exactly as `_mempalace._TAG_LITERAL` drops it on the way in.
_TAG = re.compile(r"</?mempalace-recall>", re.IGNORECASE)


class Verdict(Enum):
    """What `classify` concluded about one mined chunk.

    Three outcomes, not two, because *"contains scaffolding"* and *"is scaffolding"* are
    different questions and only the second one licenses a delete. `REVIEW` is the discovery
    channel the scrub's dry run reports and never acts on.
    """

    #: Every line is harness scaffolding. Safe to delete: there is no memory in it.
    SCRUB = "scrub"
    #: Some scaffolding, some real content. **Never deleted** — reported for human review, both
    #: because the real content is a memory and because a chunk shaped like this is evidence of a
    #: boundary path nobody has enumerated yet.
    REVIEW = "review"
    #: No catalog match. An ordinary memory.
    KEEP = "keep"


@dataclass(frozen=True)
class Scaffolding:
    """One class of harness-composed text, with the reason it is not a memory.

    Args:
        name: The class's short slug — what the scrub's report groups by.
        why: Where the harness composes it and why it is scaffolding rather than dialogue.
            Printed with the finding, so a reviewer reading a report can judge it without
            reading this file.
        literals: The exact strings, taken from the constants the harness composes from.
        prefixes: Constant *openings* of a line whose tail the harness fills in at runtime (a
            timestamp, a step count, an exception message). Matched with `startswith` on the
            normalized line — still exact-literal, just anchored at the front because the rest
            of the line is not a constant to compare against.
    """

    name: str
    why: str
    literals: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Index:
    """The catalog flattened for matching — see `_build_index` for how each part is used."""

    units: dict[str, str]
    prefixes: tuple[tuple[str, str], ...]
    wholes: tuple[tuple[str, str], ...]
    riders: frozenset[str]


def strip_injected(text: str) -> str:
    """Remove the recall block's own framing literals from `text`, case-insensitively.

    The defense-in-depth half of the boundary, applied at the harness's one mining chokepoint
    (`_wake._observe`). It is **not** the fix and must never become the only line of defense:
    the leaking paths are closed at their source, and this catches the single residue that
    closure cannot reach — the model quoting its own Turn-0 recall back in a reply, which is
    real LLM output arriving on a path that is supposed to be mined.

    Removal, never rejection: a reply that quotes the heading and then says something real
    keeps the real part. An exchange that empties out entirely is dropped by the caller.
    """
    return _INJECTED.sub("", text)


def catalog() -> tuple[Scaffolding, ...]:
    """Every class of harness-composed text a polluted palace could be holding — built once.

    Memoized together with `_index`, and that pairing is the point rather than the speed: the
    catalog reads this box's *live* charter (`_charter_texts`), so two independent builds could
    disagree, and a classifier whose index disagreed with the catalog the report prints would
    delete on evidence nobody could read. One build, both readers. `_reset` exists for tests.

    Built lazily and assembled from the modules that *write* each literal, so the catalog can
    never describe a wording the harness has since changed. Import-time assembly is not an
    option: `_wake` imports this module, so reaching back into it at module scope would be a
    cycle — and the scrub is the only caller, so nothing pays for this on a wake.

    What it deliberately does **not** enumerate: an individual tool's `description` (it reaches
    the model in the function schema, never in the transcript, so it can only ever arrive as a
    model quotation), the operator's own edited `prompts/*.md` (only the *shipped* defaults are
    knowable from here), and the live `dashboard.md` (fetched per wake from the platform). Those
    surface through `Verdict.REVIEW` in a dry run, which is what the discovery pass is for.
    """
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    from basecradle_harness import _brief, _code, _context, _engine, _install, _session, _wake
    from basecradle_harness import _unspoken as unspoken

    charter = _charter_texts(_install)
    _CATALOG = (
        Scaffolding(
            name="recall-block",
            why=(
                "The MemPalace Turn-0 recall's own framing — the heading and the fence tags "
                "the harness wraps around retrieved hits. The observed pollution (issue #438)."
            ),
            literals=INJECTED_RECALL_LITERALS,
        ),
        Scaffolding(
            name="compaction-prompt",
            why=(
                "The harness-composed 'user' side the compaction summary used to be observed "
                "as, and the instruction the summarizer was given. Never dialogue: the harness "
                "wrote both."
            ),
            literals=(_wake._COMPACTION_OBSERVE_NOTE, _context._SUMMARIZE_INSTRUCTION),
            prefixes=(_context._SUMMARY_MARKER,),
        ),
        Scaffolding(
            name="canned-narration",
            why=(
                "The harness's own stand-in for a turn's final text when the engine gave up. "
                "It reads like the agent speaking and is not: no model wrote it."
            ),
            literals=(_wake._STUCK_NOTE,),
        ),
        Scaffolding(
            name="brief-sections",
            why=(
                "Turn-0 brief parts the harness generates: the tool manifest, the safety "
                "opt-out record, the defect notice, the step-budget statement, and the "
                "current-time anchor. Persisted into the transcript on every wake before "
                "issue #275 made the brief ephemeral, which is how they reached a summarizer."
            ),
            literals=(
                _brief.render_manifest([("", None)]).splitlines()[0],
                _brief.render_safety(["x"]).splitlines()[0],
                _brief.render_defects(["x"]).splitlines()[0],
                _wake._NOW_LINE_INSTRUCTION,
            ),
            prefixes=("Step budget: this turn runs for up to ", _engine._STEP_NOTE_HEADER),
        ),
        Scaffolding(
            name="charter",
            why=(
                "`system-prompt.md` and `initialize.md` — the operator's personality charter and "
                "the framework's operating guidance. They compose into the brief, never into the "
                "conversation."
            ),
            literals=charter,
        ),
        Scaffolding(
            name="engine-notes",
            why=(
                "Turns the engine injects into the transcript as a turn's own work: the live "
                "step counter, the out-of-budget reserve nudge, the image caption that stands "
                "in for evicted pixels, and the built-in-called-as-a-function correction."
            ),
            literals=(_engine._RESERVE_NUDGE,),
            prefixes=("(Showing image: ", "(No image input on this model — "),
        ),
        Scaffolding(
            name="turn-hooks",
            why=(
                "The two shipped turn hooks' injected turns: the no-reply informer's nudge "
                "(both armings) and the code-execution bridge's Asset hand-back."
            ),
            literals=(
                unspoken.MENTION_NUDGE,
                unspoken.ONE_ON_ONE_NUDGE,
                _code._artifact_note([]).split("\n")[0],
                _code._artifact_note([]).split("\n\n")[-1],
            ),
        ),
        Scaffolding(
            name="transcript-markers",
            why=(
                "Markers the harness writes into the transcript about the transcript: the "
                "result the loader lays over a tool call whose wake died before it answered, "
                "and the note a turn that raised ends on. Never content."
            ),
            literals=(_session.INTERRUPTED,),
            prefixes=("[turn failed: ",),
        ),
    )
    return _CATALOG


def _charter_texts(install) -> tuple[str, ...]:
    """The charter prompts as the brief composes them — **this box's**, then the shipped defaults.

    Read live (`prompt_text` / `system_prompt_text`) rather than only from the package, because
    on a real agent the charter that reached the brief is the *operator's* edited copy: a scrub
    run as the agent's own user, against its own config home, therefore recognizes the text that
    was actually injectable there. The packaged defaults ride along behind it so an agent whose
    config home has since been edited (or was never installed) still matches what it mined then.

    Comment-stripped and trimmed exactly as `prompt_text` does, because that is the transform
    between the file on disk and the string the model was shown — a catalog holding the raw file
    would carry HTML comments no brief ever contained, and miss the text that one did.
    """
    texts: dict[str, None] = {}
    for live in (install.prompt_text("initialize.md"), install.system_prompt_text()):
        if live:
            texts[live] = None
    for name, raw in sorted(install._packaged_defaults().items()):
        if not (name.startswith("prompts/") and name.endswith(".md")):
            continue
        shipped = install._strip_html_comments(raw).strip()
        if shipped:
            texts[shipped] = None
    return tuple(texts)


def classify(text: str) -> tuple[Verdict, tuple[str, ...]]:
    """Decide what one mined chunk is, and name the catalog classes that matched.

    Line by line, over the chunk's *normalized* lines (leading quote/bullet markers dropped,
    whitespace runs collapsed, recall tags removed, casefolded). A line is scaffolding when it
    equals a catalog unit, opens with a catalog prefix, or — at `MIN_UNIT_CHARS` or longer — is
    a substring of one, which is what catches a chunk boundary that fell inside a literal.

    `SCRUB` needs *every* content-bearing line to be scaffolding, so the failure mode of a
    mistake here is a chunk left alone, never a memory deleted. Two kinds of line ride along
    without voting either way: one that carries no letters or digits at all (`_is_filler` — a
    bullet, a rule), and a **rider** — a line that is genuinely part of a catalog literal but is
    too short to *prove* anything on its own (a markdown heading in the charter, a numbered list
    marker in the summarizer's instruction). A rider can never make a chunk scrubbable, because
    it never adds to `matched`; it only declines to veto one that a distinctive line already
    proved. Without it a mined copy of a whole multi-section charter reads as `REVIEW` forever,
    on the strength of its own subheadings. Anything else that is not scaffolding downgrades the
    verdict to `REVIEW`.
    """
    index = _index()
    matched: list[str] = []
    real = False
    for raw in text.splitlines():
        line = _normalize(raw)
        if not line:
            continue
        owner = _match(line, index)
        if owner is not None:
            if owner not in matched:
                matched.append(owner)
        elif not _is_filler(line) and line not in index.riders:
            real = True
    if not matched:
        return Verdict.KEEP, ()
    return (Verdict.REVIEW if real else Verdict.SCRUB), tuple(matched)


# --- matching internals ------------------------------------------------------

#: The catalog and its match index, memoized as a pair (see `catalog`). Reset with `_reset`.
_CATALOG: tuple[Scaffolding, ...] | None = None

#: The built index, memoized: `catalog()` reaches into half the package, and a scrub walks tens of
#: thousands of chunks past it.
_INDEX: _Index | None = None


def _build_index() -> _Index:
    """The catalog as three lookup structures: exact units, anchored prefixes, whole literals.

    Every structure maps its string back to the *class* that owns it, so a finding can say which
    kind of scaffolding it is rather than only that it matched something.
    """
    units: dict[str, str] = {}
    prefixes: list[tuple[str, str]] = []
    wholes: list[tuple[str, str]] = []
    riders: set[str] = set()
    for entry in catalog():
        for literal in entry.literals:
            whole = _normalize(literal.replace("\n", " "))
            if len(whole) >= MIN_UNIT_CHARS:
                wholes.append((whole, entry.name))
            # A multi-line literal is also matched by each of its own paragraphs and lines: a
            # chunker splits where it likes, and half a charter is no more a memory than all of it.
            for piece in (*literal.split("\n\n"), *literal.splitlines(), literal):
                unit = _normalize(piece)
                if not unit:
                    continue
                if len(unit) >= MIN_UNIT_CHARS:
                    units.setdefault(unit, entry.name)
                else:
                    riders.add(unit)
        for prefix in entry.prefixes:
            normalized = _normalize(prefix)
            if normalized:
                prefixes.append((normalized, entry.name))
    return _Index(
        units=units, prefixes=tuple(prefixes), wholes=tuple(wholes), riders=frozenset(riders)
    )


def _index() -> _Index:
    """`_build_index`, once per process."""
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_index()
    return _INDEX


def _reset() -> None:
    """Drop both memos. For tests, which move the config home the charter is read from."""
    global _CATALOG, _INDEX
    _CATALOG = _INDEX = None


def _match(line: str, index: _Index) -> str | None:
    """The catalog class this line is scaffolding for, or ``None``.

    The substring rule is last and is gated on `MIN_UNIT_CHARS` for the reason the whole module
    is: a short line can be a substring of anything. At 48 normalized characters, being a
    substring of a harness literal means the text *is* that literal, re-wrapped or cut.
    """
    owner = index.units.get(line)
    if owner is not None:
        return owner
    for prefix, name in index.prefixes:
        if line.startswith(prefix):
            return name
    if len(line) >= MIN_UNIT_CHARS:
        for whole, name in index.wholes:
            if line in whole:
                return name
    return None


def _normalize(text: str) -> str:
    """One line as it is compared: markers dropped, whitespace collapsed, casefolded."""
    return _SPACE.sub(" ", _TAG.sub("", _LEAD.sub("", text))).strip().casefold()


def _is_filler(line: str) -> bool:
    """Is this line pure punctuation — a bullet, a rule, a stray bracket?

    Filler rides along in a scrubbable chunk without voting. It is defined by carrying no
    letters and no digits at all, which is narrow on purpose: "ok" is two letters and a real
    thing somebody said.
    """
    return not any(character.isalnum() for character in line)
