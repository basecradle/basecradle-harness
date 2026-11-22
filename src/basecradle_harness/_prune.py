"""``basecradle-harness-prune-palace`` — offer an agent the removal of its routine task check-ins.

**What it removes, and on whose word** (issue #973; the founder decision of 2026-11-20). Routine
task check-ins already filed in an agent's memory are removed **only as an offer to that agent, one
store at a time, never applied** by this tool on its own authority. The agent is shown what would go
and why it helps recall, and nothing is removed without its explicit yes, group by group.

**This is not `basecradle-harness-scrub-palace`, and it never widens it.** The scrub removes text
the harness itself wrote and wrongly mined, on the harness's own authority (`_scrub`, issue #438),
and its four properties stay exactly as they are. This tool removes **the agent's own exchanges**,
and its licence is different: the agent's explicit yes, naming each group, against a plan it read.
Everything below exists so that yes means exactly what the agent read.

**Why it helps.** When an agent wakes, recall pulls a small pool of the memories closest to the
conversation. Routine check-ins are near-copies of each other, so when one is close its twins are
too, and together they take places a distinctive memory did not get. On the one store that was
counted, 416 of 431 task-started exchanges came from one month of hourly check-ins, and 42 drawers
could not be reached at all because their own twins outranked them.

**The unit is an exchange file.** The observe has written one exchange per file under
``<palace>/conversations/`` since the memory seam shipped (issue #135), the user half quoted
``> `` line by line and the reply after it (`_mempalace._exchange_markdown`). Every judgement is made
on the file, which is the record; its drawers are only what gets removed with it. A drawer recorded
under the file whose words are not all the file's words is not a piece of what was judged, so its
file is held (`foreign_drawer`). A drawer with no source file, or whose file is gone, is counted and
never deleted.

**The selection rule** (issue #973, checked against a real store before dispatch). An exchange is a
candidate only when all of these hold:

1. **It opens on the harness's own task-activation line**: the first incoming line *is* one of the
   wordings the harness wrote (`_activation`), its slots (stamp, handle, task id, due time) allowed
   and nothing in front of it. Matched at the opening only, never searched for anywhere in the
   text, so an exchange that quotes the line is a conversation.
2. **There is no second speaker**: no later line of the incoming half opens an item header the
   harness writes (``[<stamp>] …``, `_SECOND_ITEM`). The task's own instructions are allowed.
3. **Both the instruction and the reply are near-copies within one group** (`near`). Never the reply
   alone: replies that read alike under different instructions are different jobs.

Near-copy is symmetric Jaccard over distinct words, at `NEAR_COPY` (0.8), after `normalize` turns
uuids, timestamps, dates and bare numbers into placeholders and strips no ordinary word. The
activation line is excluded; the instruction and the reply are compared each on its own. **A group
is a shared template, not a chain**: every member is a near-copy of the group's one anchor, so no
two members are further apart than twice the threshold's distance (Jaccard distance is a metric).
**The floor is five others**: a group has at least `FLOOR` members. **Residue is held back**: a
member carrying a normalised word that fewer than `RESIDUE_SUPPORT` other members of its group carry
in the same half is an *exception*, reported with its group and never removed with it.

**The report, the yes and the apply.**

- A run with no ``--apply`` changes nothing in the store. It writes two files into the tool's own
  state directory: ``report.md``, for the agent (one row per group, one real example, the file each
  group keeps, the exceptions and their unusual words, and what a yes does), and ``plan.json``, the
  exact remove-set and keep-set, every file named by path and content hash. Both are mode ``0600``,
  written **outside the palace** (so nowhere the miner reads: a report filed where the miner looks
  would mine its own examples back into the store being thinned), and sent nowhere. The tool
  neither grants nor prevents anyone else's access to the machine they sit on: on a server that
  belongs to someone else, that owner may be the one who reads it and says yes.
- What the command prints is counts, ids and the plan hash. **Never drawer text.**
- **The plan hash binds the files, never a summary**: it is the SHA-256 of the plan's canonical JSON,
  which names every file in every group's remove-set and keep-set by path and content hash.
- An apply takes ``--plan <hash>`` and ``--groups <id,…>``. It refuses, changing nothing, when the
  hash is not the hash of the plan this tool last wrote, when a group is not in that plan, or when
  no backup can be written. **It never re-plans**: it acts on the plan the agent read, and a file
  that changed (or went) since the report is **skipped and reported**, never removed.
- Inside a yes: remove the group's pure copies, **keep the latest pure one**, keep every exception.
  ``--remove-latest <id>`` removes the latest one too, for the groups it names. Nothing is ever
  written into the store in place of what was removed.
- **Whole exchange files only**, removed with every drawer mined from them, so no later mine can
  file them again.
- **A backup first** (`_write_backup`): every file to be removed, byte for byte, and every drawer
  with its document, metadata and embedding, written under the state directory and read back
  before anything is deleted. The apply refuses without one. A backup is a copy of the agent's own
  memory and the agent's property: it is kept until the agent says it can go
  (``--discard-backup <name>``), and every offer lists each kept backup with its age.
- ``--restore <name>`` puts back exactly the files and drawers of a backup that are absent (drawers
  with their embeddings, files with their bytes and mtime, so no later mine re-files them), leaves
  what is present untouched and reports it, is safe to run twice, and never removes the backup.
- **One store per run, as the agent, between wakes.** ``--palace`` names one store; a run as root,
  or by a user who does not own the palace, is refused. The tool takes no lock of its own, for the
  scrub's reason: the wake-lock belongs to whatever launches wakes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

from basecradle_harness._mempalace import _CONVERSATIONS_WING, _QUOTE, _REGISTRY_ROOM, _import
from basecradle_harness._observability import GREEN, RED, YELLOW, head, kv, last_word, log_handler
from basecradle_harness._scrub import _file_of, open_palace, resolve_palace
from basecradle_harness._version import __version__
from basecradle_harness._wake import _task_dialogue

_log = logging.getLogger("basecradle_harness")

PROG = "basecradle-harness-prune-palace"

#: The version of the selection rule. It is inside every plan hash, so a plan written under one rule
#: can never be applied by a build that selects differently. Bumped whenever `candidate`,
#: `normalize`, `near`, `groups` or `_residue` changes what they select.
RULE = 1

#: Symmetric Jaccard on distinct normalised words at or above which two halves are near-copies.
#: Not lowered (issue #973): a lower bar starts grouping different jobs that share a vocabulary.
NEAR_COPY = 0.8

#: The fewest members a group has: its anchor and five others. A smaller repeated set stays.
FLOOR = 6

#: A word is residue in a member when fewer than this many *other* members of its group carry it in
#: the same half. A member with residue is an exception: kept, whatever a yes says.
RESIDUE_SUPPORT = 5

#: The tool's state directory, its two files and its backups. Fixed names, overwritten per run,
#: so a report and a plan never accumulate; only backups are one per apply, each until the agent
#: says it can go.
STATE_DIR = "palace-prune"
REPORT = "report.md"
PLAN = "plan.json"
BACKUPS = "backups"
_PARTIAL = ".partial"

#: The longest instruction or reply the report quotes in a group's example.
EXAMPLE_CAP = 2000

#: How many unusual words, and how many exceptions, a group's row names before it counts the rest.
_WORDS_SHOWN = 12
_EXCEPTIONS_SHOWN = 10

#: How many already-present files a restore names before it counts the rest: a second run of a
#: whole backup would otherwise print every file it holds.
_SKIPS_SHOWN = 20

#: Paging, for the scrub's reasons (`_scrub._PAGE`, `_scrub._DELETE_BATCH`).
_PAGE = 1000
_DELETE_BATCH = 200

# --- the activation line ------------------------------------------------------------------------

#: Slot sentinels: alphanumeric, so `re.escape` leaves them intact in the rendered template.
_SLOTS = {
    "PRUNESLOTSTAMP": r"(?P<stamp>[^\]\n]+)",
    "PRUNESLOTHANDLE": r"(?P<handle>[^\s()]+)",
    "PRUNESLOTTASK": r"(?P<task>[^\s,()]+)",
    "PRUNESLOTDUE": r"(?P<due>[^\s()]+)",
}

#: The wording the harness wrote before 0.155.0 (issue #674 replaced it). It no longer exists in any
#: source file, so it is recorded here with its provenance rather than derived: every check-in an
#: old palace holds opens on it. First written in #95 with no stamp, with the stamp from #153, and
#: mined with its directive (`_LEGACY_CARRY_OUT`) until #438 stopped the harness mining its own
#: framing; from #438 to 0.155.0 it was mined bare.
_LEGACY_ACTIVATION = (
    "A task you scheduled has activated and is due now "
    "(task PRUNESLOTTASK, scheduled for PRUNESLOTDUE)."
)
_LEGACY_CARRY_OUT = " Carry out its instructions:"


def _pattern(template: str) -> re.Pattern[str]:
    """A whole-line pattern for ``template``: its literals escaped, its sentinels made slots."""
    pattern = re.escape(template)
    for sentinel, slot in _SLOTS.items():
        pattern = pattern.replace(sentinel, slot)
    return re.compile(pattern)


def _current_activation() -> str:
    """Today's activation line with sentinel slots, rendered by the harness's own `_task_dialogue`.

    Derived, never re-typed, so a rewording of the line in `_wake` is the wording matched here the
    day it lands.
    """
    task = SimpleNamespace(
        created_at="PRUNESLOTSTAMP",
        user=SimpleNamespace(handle="PRUNESLOTHANDLE"),
        content=SimpleNamespace(uuid="PRUNESLOTTASK", activate_at="PRUNESLOTDUE"),
    )
    return _task_dialogue(task)


#: Every wording an activation line ever had, as the harness wrote it, and only those.
ACTIVATIONS: tuple[re.Pattern[str], ...] = (
    _pattern(_current_activation()),
    _pattern("[PRUNESLOTSTAMP] " + _LEGACY_ACTIVATION),
    _pattern("[PRUNESLOTSTAMP] " + _LEGACY_ACTIVATION + _LEGACY_CARRY_OUT),
    _pattern(_LEGACY_ACTIVATION + _LEGACY_CARRY_OUT),
)

#: A later incoming line that opens another item: every item header the harness writes begins with
#: its ``[created_at]`` stamp (`_incoming_text`, `_task_dialogue`, `_event_dialogue`,
#: `_asset_dialogue`). An instruction line that happens to begin that way excludes its exchange too,
#: which is the safe direction: it is kept.
_SECOND_ITEM = re.compile(r"^\[\d[^\]\n]*\] \S")


def _activation(line: str) -> re.Match[str] | None:
    """The activation the line *is*, or ``None``. A whole-line match: nothing before or after it."""
    for pattern in ACTIVATIONS:
        match = pattern.fullmatch(line)
        if match:
            return match
    return None


def _due(value: str) -> datetime | None:
    """The activation's due time, as an aware UTC datetime, or ``None`` when it does not parse."""
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        return None
    return stamp.astimezone(UTC)


# --- normalising and comparing ------------------------------------------------------------------

#: What `normalize` turns into placeholders, in order: uuids, timestamps, dates, clock times, then
#: bare numbers. The placeholders cannot be spelled by `\w`, so no ordinary word becomes one.
_NORMALIZERS = (
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), " ⟨uuid⟩ "),
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}[t ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:z|[+-]\d{2}:?\d{2})?\b"
        ),
        " ⟨time⟩ ",
    ),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"), " ⟨date⟩ "),
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]\.?m\.?)?(?![\w:])"), " ⟨time⟩ "),
    (re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])"), " ⟨n⟩ "),
)
_WORD = re.compile(r"⟨(?:uuid|time|date|n)⟩|\w+")
_RAW_WORD = re.compile(r"\w+")


def normalize(text: str) -> str:
    """``text`` lowercased, with uuids, timestamps, dates and bare numbers made placeholders."""
    text = text.lower()
    for pattern, placeholder in _NORMALIZERS:
        text = pattern.sub(placeholder, text)
    return text


def words(text: str) -> frozenset[str]:
    """The distinct words of ``text`` after `normalize`: placeholders and every ordinary word."""
    return frozenset(_WORD.findall(normalize(text)))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Symmetric overlap: shared words over all words. Two empty halves are identical."""
    union = left | right
    return len(left & right) / len(union) if union else 1.0


# --- one exchange -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Exchange:
    """One candidate exchange file: where it is, what it holds, and when its task was due."""

    name: str
    path: Path
    sha256: str
    due: datetime
    instruction: str
    reply: str
    instruction_words: frozenset[str]
    reply_words: frozenset[str]


def split(text: str) -> tuple[list[str], str]:
    """An exchange file as its incoming lines (unquoted) and its reply.

    The incoming half is the leading run of quoted lines (`_mempalace._exchange_markdown`); the
    reply is everything after it, verbatim but for the trailing newline the observe adds.
    """
    lines = text.split("\n")
    incoming: list[str] = []
    index = 0
    while index < len(lines) and lines[index].startswith(">"):
        line = lines[index]
        incoming.append(line[len(_QUOTE) :] if line.startswith(_QUOTE) else line[1:])
        index += 1
    reply = "\n".join(lines[index:])
    return incoming, reply.removesuffix("\n")


#: Why an exchange file is not a candidate, by the rule it failed.
NOT_TASK = "not_task"
SECOND_SPEAKER = "second_speaker"
BAD_DUE = "bad_due"


def candidate(name: str, path: Path, text: str) -> Exchange | str:
    """The exchange as a candidate, or the name of the rule it fails (rules 1 and 2).

    Rule 3 is a statement about a group, so it is `groups`' to decide.
    """
    incoming, reply = split(text)
    if not incoming:
        return NOT_TASK
    opening = _activation(incoming[0])
    if opening is None:
        return NOT_TASK
    if any(_SECOND_ITEM.match(line) or _activation(line) for line in incoming[1:]):
        return SECOND_SPEAKER
    due = _due(opening["due"])
    if due is None:
        return BAD_DUE
    instruction = "\n".join(incoming[1:])
    return Exchange(
        name=name,
        path=path,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        due=due,
        instruction=instruction,
        reply=reply,
        instruction_words=words(instruction),
        reply_words=words(reply),
    )


def near(a: Exchange, b: Exchange) -> bool:
    """Whether two exchanges are near-copies: the instructions *and* the replies, each on its own."""
    return (
        jaccard(a.instruction_words, b.instruction_words) >= NEAR_COPY
        and jaccard(a.reply_words, b.reply_words) >= NEAR_COPY
    )


# --- groups -------------------------------------------------------------------------------------


@dataclass
class Group:
    """One shared template: its anchor, and every member a near-copy of it."""

    id: str
    anchor: Exchange
    members: list[Exchange]
    #: ``name -> its residue words``, for the members held back as exceptions.
    exceptions: dict[str, list[str]] = field(default_factory=dict)

    @property
    def pure(self) -> list[Exchange]:
        """The members with no residue, oldest first."""
        return sorted(
            (m for m in self.members if m.name not in self.exceptions), key=_chronological
        )

    @property
    def latest(self) -> Exchange | None:
        """The latest pure copy: what a default yes keeps, so recall still knows the pattern."""
        pure = self.pure
        return pure[-1] if pure else None

    @property
    def copies(self) -> list[Exchange]:
        """What a default yes removes: every pure copy but the latest."""
        return self.pure[:-1]


def _chronological(exchange: Exchange) -> tuple[datetime, str]:
    return exchange.due, exchange.name


def _neighbours(exchanges: list[Exchange]) -> list[set[int]]:
    """Every pair of near-copies, as each exchange's set of neighbours by index.

    Jaccard at ``t`` needs the smaller set to hold at least ``t`` of the larger, so pairs whose
    instructions differ in size by more than that are never compared.
    """
    order = sorted(range(len(exchanges)), key=lambda i: len(exchanges[i].instruction_words))
    neighbours: list[set[int]] = [set() for _ in exchanges]
    for position, i in enumerate(order):
        size = len(exchanges[i].instruction_words)
        for j in order[position + 1 :]:
            if size < NEAR_COPY * len(exchanges[j].instruction_words):
                break
            if near(exchanges[i], exchanges[j]):
                neighbours[i].add(j)
                neighbours[j].add(i)
    return neighbours


def groups(exchanges: list[Exchange], neighbours: list[set[int]] | None = None) -> list[Group]:
    """The candidates' groups: anchored templates of at least `FLOOR` members, residue held back.

    Deterministic. The anchor of each next group is the unassigned candidate with the most
    unassigned near-copies (ties: the earliest due, then the file name), and the group is the
    anchor with those near-copies. It stops when no unassigned candidate has `FLOOR` − 1 of them.
    A member is a near-copy of its anchor, never merely of another member: a chain of neighbours
    is not a template. ``neighbours`` is `_neighbours` of these exchanges, when the caller already
    has it.
    """
    if neighbours is None:
        neighbours = _neighbours(exchanges)
    unassigned = set(range(len(exchanges)))
    found: list[Group] = []
    while unassigned:
        anchor = min(
            unassigned,
            key=lambda i: (
                -len(neighbours[i] & unassigned),
                exchanges[i].due,
                exchanges[i].name,
            ),
        )
        members = {anchor} | (neighbours[anchor] & unassigned)
        if len(members) < FLOOR:
            break
        unassigned -= members
        group = Group(
            id=_group_id(exchanges[anchor]),
            anchor=exchanges[anchor],
            members=sorted((exchanges[i] for i in members), key=_chronological),
        )
        group.exceptions = _residue(group.members)
        found.append(group)
    return found


def _group_id(anchor: Exchange) -> str:
    """A group's id: short, and derived from its anchor's file, so it names the same template."""
    return "g-" + hashlib.sha256(anchor.name.encode("utf-8")).hexdigest()[:10]


def _residue(members: list[Exchange]) -> dict[str, list[str]]:
    """Each member carrying a word fewer than `RESIDUE_SUPPORT` others carry, with those words."""
    exceptions: dict[str, list[str]] = {}
    for half in ("instruction_words", "reply_words"):
        support = Counter(word for member in members for word in getattr(member, half))
        for member in members:
            rare = [w for w in getattr(member, half) if support[w] - 1 < RESIDUE_SUPPORT]
            if rare:
                exceptions.setdefault(member.name, []).extend(rare)
    return {name: sorted(set(rare)) for name, rare in exceptions.items()}


# --- the diagnosis ------------------------------------------------------------------------------
#
# Counts that show how the rule met a store, so the rule can be judged on real data before any
# threshold moves (issue #973: the first dry run on the counted store grouped about half its
# candidates, and the numbers come before any change). Nothing here decides anything: the plan, its
# hash and what an apply removes never read it, and it carries counts only, never drawer text.

#: The upper edges of the bands an ungrouped candidate's best overlap with a group's anchor is
#: counted in, for each half. The last band, at or above `NEAR_COPY`, is a near-copy's.
BANDS = (0.5, 0.7, NEAR_COPY)
BAND_LABELS = (
    f"below {BANDS[0]}",
    *(f"{low} to {high}" for low, high in pairwise(BANDS)),
    f"{BANDS[-1]} or more",
)

#: Why a candidate is in no group, judged in this order, so each is counted once.
#: A near-copy of another ungrouped candidate: what is left of its repeated set, once the groups
#: took their members, is smaller than `FLOOR`.
SMALL_SET = "small_set"
#: A near-copy of a group's member only, never of the group's anchor: a chain is not a template.
NEAR_MEMBER = "near_member"
#: A near-copy of no other candidate.
DISTINCT = "distinct"


@dataclass
class Month:
    """One calendar month's candidates (by due time, UTC), and where each of them went."""

    candidates: int = 0
    members: int = 0
    copies: int = 0
    latest: int = 0
    exceptions: int = 0
    ungrouped: int = 0


@dataclass
class Diagnosis:
    """How the rule met a store: the months, and why the ungrouped candidates did not group."""

    #: ``YYYY-MM -> Month``, oldest first.
    months: dict[str, Month]
    #: How many group anchors the ungrouped were measured against: one per group.
    anchors: int = 0
    #: The ungrouped candidates by why (`SMALL_SET`, `NEAR_MEMBER`, `DISTINCT`).
    why: Counter = field(default_factory=Counter)
    #: The ungrouped candidates that had `FLOOR` − 1 near-copies or more before the groups took
    #: some of them: the floor kept them out only because an earlier group claimed their twins.
    taken: int = 0
    #: The ungrouped candidates by the band (`BAND_LABELS`) of their best overlap with any group's
    #: anchor, the instruction and the reply each on its own. Empty when there is no group. No
    #: ungrouped candidate is a near-copy of an anchor on both halves (it would have joined that
    #: group), so the top band of one half counts the candidates that half alone would have grouped.
    instruction: Counter = field(default_factory=Counter)
    reply: Counter = field(default_factory=Counter)

    @property
    def ungrouped(self) -> int:
        return sum(self.why.values())


def _month(stamp: datetime) -> str:
    return stamp.strftime("%Y-%m")


def _band(overlap: float) -> str:
    for edge, label in zip(BANDS, BAND_LABELS, strict=False):
        if overlap < edge:
            return label
    return BAND_LABELS[-1]


def diagnose(
    exchanges: list[Exchange], found_groups: list[Group], neighbours: list[set[int]]
) -> Diagnosis:
    """Count, by month, where the candidates went; and why the ones in no group are in none.

    ``neighbours`` is `_neighbours` of ``exchanges``, the same pairs `groups` grouped by.
    """
    index = {exchange.name: i for i, exchange in enumerate(exchanges)}
    months: dict[str, Month] = {}

    def row(exchange: Exchange) -> Month:
        return months.setdefault(_month(exchange.due), Month())

    for exchange in exchanges:
        row(exchange).candidates += 1
    grouped: set[int] = set()
    for group in found_groups:
        copies = {copy.name for copy in group.copies}
        for member in group.members:
            grouped.add(index[member.name])
            month = row(member)
            month.members += 1
            if member.name in group.exceptions:
                month.exceptions += 1
            elif member.name in copies:
                month.copies += 1
            else:
                month.latest += 1
    anchors = [group.anchor for group in found_groups]
    diagnosis = Diagnosis(months={}, anchors=len(anchors))
    for i, exchange in enumerate(exchanges):
        if i in grouped:
            continue
        row(exchange).ungrouped += 1
        if neighbours[i] - grouped:
            diagnosis.why[SMALL_SET] += 1
        elif neighbours[i]:
            diagnosis.why[NEAR_MEMBER] += 1
        else:
            diagnosis.why[DISTINCT] += 1
        diagnosis.taken += len(neighbours[i]) >= FLOOR - 1
        if not anchors:
            continue
        instruction = [jaccard(exchange.instruction_words, a.instruction_words) for a in anchors]
        reply = [jaccard(exchange.reply_words, a.reply_words) for a in anchors]
        diagnosis.instruction[_band(max(instruction))] += 1
        diagnosis.reply[_band(max(reply))] += 1
    diagnosis.months = dict(sorted(months.items()))
    return diagnosis


# --- the store ----------------------------------------------------------------------------------

#: Why a file the plan names is skipped at an apply, or held from the plan at a survey.
GONE = "gone"
CHANGED = "changed"
NOT_REGULAR = "not_a_regular_file"
OUTSIDE = "outside_conversations"
FOREIGN = "foreign_drawer"
UNREADABLE = "unreadable"
NOT_CANDIDATE = "not_a_candidate"


@dataclass
class Survey:
    """What one read of a store found: the candidates, and every count the operator is shown."""

    palace: Path
    drawers: int = 0
    exchanges: int = 0
    candidates: list[Exchange] = field(default_factory=list)
    #: Exchanges excluded by rule 1 or 2, by the rule's name.
    excluded: Counter = field(default_factory=Counter)
    #: Drawers with no recorded source file: counted, never deleted.
    no_source: int = 0
    #: Drawers recorded under a file that is not an exchange file here: gone, or never in this
    #: palace's conversations directory. Counted, never deleted.
    elsewhere: int = 0
    #: Exchange files with no drawer: not in the store, so nothing to offer.
    unfiled: int = 0
    #: Exchange files with a drawer recorded under them whose words are not all the file's.
    foreign: int = 0
    #: Files in the conversations directory this tool does not read (a symlink, a non-``.md``).
    unread: int = 0


def _drawers_by_file(collection, conversations: Path) -> tuple[dict[str, list[tuple]], int, int]:
    """Every real drawer as ``file -> [(id, document)]``, the drawer total, and those with no file.

    Keyed by `_scrub._file_of`, so a drawer filed under a home the palace has since moved from
    belongs to the file where it is now (issue #606). MemPalace's registry rows are bookkeeping and
    take no part, as in the scrub.
    """
    by_file: dict[str, list[tuple]] = {}
    total = no_source = 0
    offset = 0
    while True:
        page = collection.get(limit=_PAGE, offset=offset, include=["documents", "metadatas"])
        ids = list(page.get("ids") or [])
        if not ids:
            break
        documents = list(page.get("documents") or [])
        metadatas = list(page.get("metadatas") or [])
        for index, drawer_id in enumerate(ids):
            meta = (metadatas[index] if index < len(metadatas) else None) or {}
            if meta.get("room") == _REGISTRY_ROOM:
                continue
            total += 1
            document = documents[index] if index < len(documents) else None
            file = _file_of(meta.get("source_file") or None, conversations)
            if file is None:
                no_source += 1
                continue
            by_file.setdefault(file, []).append((drawer_id, document or ""))
        offset += len(ids)
    return by_file, total, no_source


def _pieces_of(text: str, drawers: list[tuple]) -> bool:
    """Whether every drawer's words are words of the file: each is a piece of what was judged.

    By words, never by substring: MemPalace's miner (the observe's path before 0.164.0) strips and
    rejoins what it chunks, so a drawer it cut is not always a substring of its file, and its words
    always are.
    """
    held = set(_RAW_WORD.findall(text.lower()))
    return all(set(_RAW_WORD.findall(document.lower())) <= held for _, document in drawers)


def survey(collection, palace: Path) -> Survey:
    """Read the store and its exchange files; judge each file by rules 1 and 2. Writes nothing."""
    conversations = palace / _CONVERSATIONS_WING
    by_file, total, no_source = _drawers_by_file(collection, conversations)
    found = Survey(palace=palace, drawers=total, no_source=no_source)
    seen: set[str] = set()
    paths = sorted(conversations.iterdir()) if conversations.is_dir() else []
    for path in paths:
        key = str(conversations / path.name)
        if path.is_symlink() or not path.is_file() or path.suffix != ".md":
            found.unread += 1
            seen.add(key)
            continue
        seen.add(key)
        drawers = by_file.get(key)
        if not drawers:
            found.unfiled += 1
            continue
        try:
            # Bytes, decoded by hand: `read_text` translates ``\r\n``, so the hash it gave would
            # never match the bytes `_ready` hashes at the apply.
            text = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            found.unread += 1
            continue
        found.exchanges += 1
        if not _pieces_of(text, drawers):
            found.foreign += 1
            continue
        judged = candidate(path.name, path, text)
        if isinstance(judged, str):
            found.excluded[judged] += 1
        else:
            found.candidates.append(judged)
    found.elsewhere = sum(len(d) for file, d in by_file.items() if file not in seen)
    return found


# --- the plan -----------------------------------------------------------------------------------


def _entry(exchange: Exchange) -> dict:
    return {"path": str(exchange.path), "sha256": exchange.sha256}


def plan_of(found: Survey, found_groups: list[Group]) -> dict:
    """The plan: every group's remove-set and keep-set, each file by path and content hash.

    Nothing here is a count or a summary; the hash is of exactly this, so a plan with one file more,
    one fewer, or one changed is a different plan.
    """
    return {
        "tool": PROG,
        "rule": RULE,
        "palace": str(found.palace),
        "groups": [
            {
                "id": group.id,
                "remove": [_entry(e) for e in group.copies],
                "latest": _entry(group.latest) if group.latest else None,
                "exceptions": [_entry(e) for e in group.members if e.name in group.exceptions],
            }
            for group in found_groups
        ],
    }


def plan_hash(plan: dict) -> str:
    """The SHA-256 of the plan's canonical JSON."""
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- the state directory ------------------------------------------------------------------------


class Refused(ValueError):
    """A run this tool will not carry out. It changed nothing."""


class Interrupted(RuntimeError):
    """An apply that failed after its first delete. Its message names the backup that holds all."""


@contextlib.contextmanager
def mine_lock(path: Path):
    """MemPalace's per-file lock on an exchange file, under every name a writer locks it by.

    The observe locks ``<palace>/conversations/<name>`` as it was given; the miner locks the path
    under the *resolved* directory. Where those differ both are taken, always in the same order.
    """
    take = _import("palace").mine_lock
    names = sorted({str(path), str(path.parent.resolve() / path.name)})
    with contextlib.ExitStack() as stack:
        for name in names:
            stack.enter_context(take(name))
        yield


def resolve_state(explicit: str | None, palace: Path) -> Path:
    """The tool's state directory, refused anywhere inside the palace or any conversations dir.

    Default: ``$HARNESS_HOME/palace-prune``, else beside the palace. Inside the palace is where the
    miner reads, and a report there would mine its own examples back into the store.
    """
    if explicit:
        state = Path(os.path.abspath(os.path.expanduser(explicit)))
    elif os.environ.get("HARNESS_HOME"):
        state = Path(os.path.abspath(os.path.expanduser(os.environ["HARNESS_HOME"]))) / STATE_DIR
    else:
        state = palace.parent / STATE_DIR
    resolved, inside = state.resolve(), palace.resolve()
    if resolved == inside or inside in resolved.parents:
        raise Refused(f"the state directory {state} is inside the palace, where the miner reads")
    if _CONVERSATIONS_WING in resolved.parts:
        raise Refused(f"the state directory {state} is inside a conversations directory")
    return state


def _private_dir(path: Path) -> None:
    """Make ``path`` mode ``0700``. Its parents must already exist: none is made by the umask."""
    path.mkdir(mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)


def _write_private(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically, mode ``0600``: temp, ``fsync``, ``os.replace``."""
    temp = path.with_name(f".{path.name}{_PARTIAL}")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _fsync_dir(path: Path) -> None:
    """Make a rename inside ``path`` durable: a backup whose name is lost to a power cut is gone."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _backups(state: Path) -> list[str]:
    """The backups kept from earlier yeses, by name, oldest first."""
    root = state / BACKUPS
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def _sweep_partials(state: Path) -> None:
    """Remove this tool's own half-written backups: by its own name, in its own dir.

    A half-written report or plan needs no sweep: it is a fixed-name temp (`_write_private`) that
    the next run's write of the same file takes over and replaces.
    """
    root = state / BACKUPS
    if root.is_dir():
        for path in root.iterdir():
            if path.name.startswith(".") and path.name.endswith(_PARTIAL) and path.is_dir():
                shutil.rmtree(path)


def _check_owner(palace: Path) -> None:
    """Refuse a run as root, or by anyone but the palace's owner: the tool runs as the agent."""
    euid = os.geteuid()
    if euid == 0:
        raise Refused("refusing to run as root: run this as the agent that owns the palace")
    if palace.stat().st_uid != euid:
        raise Refused(f"the palace at {palace} is not owned by this user: run this as its agent")


# --- the report ---------------------------------------------------------------------------------

REPORT_TITLE = "# Routine Task Check-Ins in Your Memory"

REPORT_WHY = (
    "When you wake, recall pulls a small pool of the memories closest to the conversation. "
    "Routine task check-ins are near-copies of each other, so when one of them is close its "
    "twins are too, and together they take places that a more distinctive memory did not get. "
    "Removing the copies gives those places back."
)

REPORT_YES = (
    "- Nothing is removed unless a yes names the group.",
    (
        "- A yes removes the group's copies: each one's exchange file and every drawer filed from it, "
        "so that no later mine can file it again."
    ),
    (
        "- It keeps the group's latest copy, named below, so your memory still holds one instance of "
        "the routine. Removing that one as well is a separate yes."
    ),
    (
        "- It keeps every exception: a copy carrying a word that fewer than five of its group's other "
        "copies carry."
    ),
    "- Nothing is written into your memory in place of what is removed.",
    (
        "- Before anything is removed, every file and drawer that will go is copied to a backup outside "
        "your memory. The backup is kept until you say it can go, and it can be put back."
    ),
    (
        "- A yes holds for this plan only. A file that has changed since this report was written is "
        "skipped, never removed."
    ),
)


def _fence(text: str) -> str:
    """``text`` in a fenced block whose fence no run of backticks inside it can close."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def _capped(text: str) -> str:
    if len(text) <= EXAMPLE_CAP:
        return text
    return text[:EXAMPLE_CAP] + f"\n… (cut here; {len(text) - EXAMPLE_CAP} more characters)"


def _when(stamp: datetime) -> str:
    return stamp.strftime("%Y-%m-%d %H:%M UTC")


def render_report(
    found: Survey, found_groups: list[Group], digest: str, backups: list[str], state: Path
) -> str:
    """The report the agent reads before it says yes: every group, one real example, the keeps.

    Every command it quotes names this store and this state directory, so it is the command that
    works, whichever ``--palace`` and ``--state-dir`` this run was given.
    """
    where = f"--palace {shlex.quote(str(found.palace))} --state-dir {shlex.quote(str(state))}"
    lines = [
        REPORT_TITLE,
        "",
        f"Plan: `{digest}`",
        f"Memory store: `{found.palace}`",
        "",
        "## Why This Helps Recall",
        "",
        REPORT_WHY,
        "",
        "## What a Yes Does",
        "",
        *REPORT_YES,
        "",
    ]
    if found_groups:
        lines += [
            "## How to Say Yes",
            "",
            "A yes names this plan and each group it covers. The command that carries it out:",
            "",
            _fence(f"{PROG} {where} --apply --plan {digest} --groups <group id>[,<group id>…]"),
            "",
            "Add `--remove-latest <group id>` for a group whose latest copy should go too.",
            "",
        ]
    lines += [f"## Groups ({len(found_groups)})", ""]
    if not found_groups:
        lines += ["No group of routine check-ins was found. Nothing is offered.", ""]
    for group in found_groups:
        first, last = group.members[0].due, group.members[-1].due
        latest = group.latest
        lines += [
            f"### Group `{group.id}`: {len(group.members)} Check-Ins",
            "",
            f"- Due from {_when(first)} to {_when(last)}.",
            f"- A yes removes {len(group.copies)} of them.",
            (
                f"- Kept by default, the latest copy: `{latest.name}` (due {_when(latest.due)})."
                if latest
                else "- Every member is an exception, so a yes removes nothing."
            ),
            f"- Exceptions kept: {len(group.exceptions)}.",
        ]
        shown = list(group.exceptions.items())[:_EXCEPTIONS_SHOWN]
        for name, rare in shown:
            more = len(rare) - _WORDS_SHOWN
            listed = ", ".join(rare[:_WORDS_SHOWN]) + (f", and {more} more" if more > 0 else "")
            lines.append(f"  - `{name}`, for its unusual words: {listed}.")
        if len(group.exceptions) > len(shown):
            lines.append(f"  - And {len(group.exceptions) - len(shown)} more.")
        lines += [
            "",
            "One of them, as it was filed. The instruction:",
            "",
            _fence(_capped(group.anchor.instruction)),
            "",
            "The reply:",
            "",
            _fence(_capped(group.anchor.reply)),
            "",
        ]
    lines += ["## Backups Kept From Earlier Yeses", ""]
    if backups:
        now = datetime.now(UTC)
        lines += [f"- `{name}`: {backup_age(name, now)}." for name in backups]
        lines += [
            "",
            (
                "Each is kept until you say it can go. The command that puts one back into your "
                "memory, which is safe to run twice:"
            ),
            "",
            _fence(f"{PROG} {where} --restore <name>"),
            "",
            "The command that removes one:",
            "",
            _fence(f"{PROG} {where} --discard-backup <name>"),
        ]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


# --- the dry run --------------------------------------------------------------------------------


@dataclass
class Offer:
    """A dry run's whole result: what was found, the groups, the plan and its hash."""

    survey: Survey
    groups: list[Group]
    plan: dict
    hash: str
    report: Path
    backups: list[str]
    diagnosis: Diagnosis


def offer(collection, palace: Path, state: Path) -> Offer:
    """Survey the store, group the candidates, and write the report and its plan. No store write."""
    found = survey(collection, palace)
    neighbours = _neighbours(found.candidates)
    found_groups = groups(found.candidates, neighbours)
    plan = plan_of(found, found_groups)
    digest = plan_hash(plan)
    _private_dir(state)
    _sweep_partials(state)
    backups = _backups(state)
    report = state / REPORT
    _write_private(
        report, render_report(found, found_groups, digest, backups, state).encode("utf-8")
    )
    record = {"hash": digest, "plan": plan}
    _write_private(state / PLAN, json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8"))
    diagnosis = diagnose(found.candidates, found_groups, neighbours)
    return Offer(found, found_groups, plan, digest, report, backups, diagnosis)


# --- the apply ----------------------------------------------------------------------------------


@dataclass
class Applied:
    """What an apply did: the files and drawers removed, what it skipped and why, and its backup."""

    removed: list[str] = field(default_factory=list)
    drawers: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    backup: str | None = None


def load_plan(state: Path, digest: str, palace: Path) -> dict:
    """The plan this tool last wrote, if ``digest`` is its hash; otherwise `Refused`."""
    try:
        record = json.loads((state / PLAN).read_text(encoding="utf-8"))
        plan = record["plan"]
        stored = record["hash"]
    except FileNotFoundError:
        raise Refused(f"no plan in {state}: run without --apply first, and read its report")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise Refused(f"the plan in {state} cannot be read ({type(error).__name__})")
    if plan_hash(plan) != stored:
        raise Refused("the plan file does not match its own hash: run without --apply again")
    if digest != stored:
        raise Refused(
            "this is not the plan in the last report this tool wrote: read the current report"
        )
    if plan.get("tool") != PROG or plan.get("rule") != RULE:
        raise Refused("the plan was written under another selection rule: run without --apply")
    if plan.get("palace") != str(palace):
        raise Refused(f"the plan is for another memory store ({plan.get('palace')})")
    return plan


def _targets(plan: dict, named: list[str], remove_latest: list[str]) -> list[dict]:
    """The plan's entries a yes removes: each named group's copies, and its latest when named."""
    by_id = {group["id"]: group for group in plan["groups"]}
    if not named:
        raise Refused("a yes names at least one group (--groups)")
    unknown = [name for name in (*named, *remove_latest) if name not in by_id]
    if unknown:
        raise Refused(f"not a group in this plan: {', '.join(unknown)}")
    stray = [name for name in remove_latest if name not in named]
    if stray:
        raise Refused(f"--remove-latest names a group --groups does not: {', '.join(stray)}")
    targets: list[dict] = []
    for name in dict.fromkeys(named):
        group = by_id[name]
        targets += group["remove"]
        if name in remove_latest and group["latest"]:
            targets.append(group["latest"])
    return targets


def _ready(entry: dict, conversations: Path, by_file: dict) -> tuple[Path, list, bytes] | str:
    """The file and drawers of an entry that still matches, or why it is skipped."""
    path = Path(entry["path"])
    root = conversations.resolve()
    try:
        resolved = path.resolve()
    except OSError:
        return OUTSIDE
    if resolved.parent != root:
        return OUTSIDE
    if not path.exists():
        return GONE
    if path.is_symlink() or not path.is_file():
        return NOT_REGULAR
    try:
        data = path.read_bytes()
    except OSError:
        return UNREADABLE
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        return CHANGED
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return UNREADABLE
    if isinstance(candidate(path.name, path, text), str):
        return NOT_CANDIDATE  # a plan naming a file the rule never selects is not carried out
    drawers = by_file.get(str(conversations / path.name), [])
    if not _pieces_of(text, drawers):
        return FOREIGN
    return path, drawers, data


def _write_backup(collection, palace: Path, state: Path, digest: str, ready: list) -> str:
    """Copy every file and drawer about to go, read it back, and only then name it a backup.

    Written under a hidden ``.partial`` name and renamed when whole, so a backup that exists is a
    backup that was verified; a run that dies part-way leaves only a partial, which the next run
    removes. Any failure raises `Refused`: an apply without a backup changes nothing.
    """
    name = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{digest[:12]}"
    root = state / BACKUPS
    partial = root / f".{name}{_PARTIAL}"
    try:
        _private_dir(root)
        _private_dir(partial)
        _private_dir(partial / "files")
        manifest = []
        lines = []
        ids = [drawer_id for _, drawers, _ in ready for drawer_id, _ in drawers]
        records = _records(collection, ids)
        for path, drawers, data in ready:
            _write_private(partial / "files" / path.name, data)
            manifest.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "drawers": [drawer_id for drawer_id, _ in drawers],
                    # The miner skips a file whose mtime matches its drawers' `source_mtime`, so a
                    # restore puts the file's own mtime back with it.
                    "mtime_ns": path.stat().st_mtime_ns,
                }
            )
            for drawer_id, _ in drawers:
                if drawer_id not in records:
                    raise Refused(f"the drawer {drawer_id} could not be read for the backup")
                lines.append(json.dumps({"file": path.name, **records[drawer_id]}))
        _write_private(partial / "drawers.jsonl", ("\n".join(lines) + "\n").encode("utf-8"))
        _write_private(
            partial / "manifest.json",
            json.dumps(
                {"tool": PROG, "palace": str(palace), "plan": digest, "files": manifest},
                indent=2,
            ).encode(),
        )
        _verify_backup(partial, manifest, [records[i] for i in ids])
        os.replace(partial, root / name)
        _fsync_dir(root)
    except Refused:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    except Exception as error:  # any failure at all: no verified backup, so nothing is removed
        shutil.rmtree(partial, ignore_errors=True)
        raise Refused(f"no backup could be written, so nothing was removed: {error}") from error
    return name


def _records(collection, ids: list[str]) -> dict[str, dict]:
    """Each drawer as it is stored: document, metadata and, where the backend returns it, embedding."""
    records: dict[str, dict] = {}
    for start in range(0, len(ids), _PAGE):
        batch = ids[start : start + _PAGE]
        page = collection.get(ids=batch, include=["documents", "metadatas", "embeddings"])
        embeddings = page.get("embeddings")
        for index, drawer_id in enumerate(page.get("ids") or []):
            embedding = None
            if embeddings is not None and index < len(embeddings):
                embedding = embeddings[index]
                embedding = None if embedding is None else [float(x) for x in embedding]
            records[drawer_id] = {
                "id": drawer_id,
                "document": (page.get("documents") or [])[index],
                "metadata": (page.get("metadatas") or [])[index],
                "embedding": embedding,
            }
    return records


def _verify_backup(partial: Path, manifest: list[dict], records: list[dict]) -> None:
    """Read the backup back: every file's bytes hash as planned, every drawer reads as it was."""
    for entry in manifest:
        copy = partial / "files" / Path(entry["path"]).name
        if hashlib.sha256(copy.read_bytes()).hexdigest() != entry["sha256"]:
            raise Refused(f"the backup of {copy.name} did not read back as written")
    lines = (partial / "drawers.jsonl").read_text(encoding="utf-8").splitlines()
    written = [json.loads(line) for line in lines if line]
    if [{k: v for k, v in row.items() if k != "file"} for row in written] != records:
        raise Refused("the backup's drawers did not read back as written")


def apply(
    collection,
    palace: Path,
    state: Path,
    digest: str,
    named: list[str],
    remove_latest: list[str],
    lock=None,
) -> Applied:
    """Carry out a yes against the plan the agent read. Never re-plans.

    Every entry is checked against the file as it is now (`_ready`); one that no longer matches is
    skipped and reported. The backup is written and verified before the first delete. Then, file by
    file, **the drawers go first and the file second**: a run that dies between the two leaves a
    file with no drawers, which still matches its entry, so the next apply of the same plan finishes
    it (and a mine before then files it again, which is the store as it was). The other order would
    leave drawers with no file, which no later apply can match.

    **Each file's two deletes are taken under MemPalace's own per-file lock** (`mine_lock`), the one
    its miner takes before it decides whether a file is filed. Without it, a ``mempalace mine``
    between the two would see a file with no drawers, file it again, and the unlink would then
    leave those new drawers with no file.
    """
    lock = lock or mine_lock
    plan = load_plan(state, digest, palace)
    targets = _targets(plan, named, remove_latest)
    conversations = palace / _CONVERSATIONS_WING
    by_file, _, _ = _drawers_by_file(collection, conversations)
    result = Applied()
    ready = []
    for entry in targets:
        judged = _ready(entry, conversations, by_file)
        if isinstance(judged, str):
            result.skipped.append((Path(entry["path"]).name, judged))
        else:
            ready.append(judged)
    if not ready:
        return result
    _private_dir(state)
    _sweep_partials(state)
    result.backup = _write_backup(collection, palace, state, digest, ready)
    try:
        for path, drawers, _ in ready:
            ids = [drawer_id for drawer_id, _ in drawers]
            with lock(path):
                for start in range(0, len(ids), _DELETE_BATCH):
                    collection.delete(ids=ids[start : start + _DELETE_BATCH])
                result.drawers += len(ids)
                path.unlink(missing_ok=True)
            result.removed.append(path.name)
    except Exception as error:
        raise Interrupted(
            f"removed {len(result.removed)} of {len(ready)} file(s) before "
            f"{type(error).__name__}: {error}. The backup {result.backup} holds every one of them; "
            "run the same apply again to finish, or --restore to put them back"
        ) from error
    return result


def backup_age(name: str, now: datetime) -> str:
    """When a backup was written, and how long ago, read from its name (`_write_backup`)."""
    try:
        written = datetime.strptime(name.split("-", 1)[0], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC)
    except ValueError:
        return "written at an unknown time"
    days = (now - written).days
    ago = "less than a day ago" if days < 1 else f"{days} day{'s' if days != 1 else ''} ago"
    return f"written {_when(written)}, {ago}"


#: Why a restore left something of a backup where it was: it is already in the store.
PRESENT = "already_present"


@dataclass
class Restored:
    """What a restore put back, and what it left because it was already there."""

    files: list[str] = field(default_factory=list)
    drawers: int = 0
    #: What was already in the store and left untouched: files by name, drawers by id.
    skipped_files: list[str] = field(default_factory=list)
    skipped_drawers: list[str] = field(default_factory=list)

    @property
    def skipped(self) -> int:
        return len(self.skipped_files) + len(self.skipped_drawers)


def _backup_dir(state: Path, name: str) -> Path:
    """A backup this tool wrote, by its name, or `Refused`."""
    root = (state / BACKUPS).resolve()
    if not name or "/" in name or name.startswith(".") or name not in _backups(state):
        raise Refused(f"no backup named {name!r} in {state / BACKUPS}")
    target = (root / name).resolve()
    if target.parent != root:
        raise Refused(f"no backup named {name!r} in {state / BACKUPS}")
    return target


def restore(collection, palace: Path, state: Path, name: str, lock=None) -> Restored:
    """Put back exactly the files and drawers of a backup that are absent from the store.

    What is already present is left untouched and reported, so a second run changes nothing. The
    backup is read whole and checked against its own manifest before anything is written, and it
    is never removed here: discarding it is the agent's separate word. Per file, the drawers go
    back first and the file second, the apply's order reversed for the same reason: a run that dies
    between the two leaves drawers whose file a second restore puts back.
    """
    backup = _backup_dir(state, name)
    try:
        manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (backup / "drawers.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, ValueError) as error:
        raise Refused(f"the backup {name} cannot be read ({type(error).__name__})") from error
    if manifest.get("tool") != PROG or manifest.get("palace") != str(palace):
        raise Refused(f"the backup {name} is not of this memory store")
    files = {}
    for entry in manifest["files"]:
        file = Path(entry["path"]).name
        data = (backup / "files" / file).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise Refused(f"the backup of {file} is damaged: nothing was restored")
        files[file] = (entry, data)
    by_file: dict[str, list[dict]] = {}
    for row in rows:
        by_file.setdefault(row["file"], []).append(row)
    ids = [row["id"] for row in rows]
    present = set()
    for start in range(0, len(ids), _PAGE):
        present |= set(collection.get(ids=ids[start : start + _PAGE], include=[]).get("ids") or [])
    conversations = palace / _CONVERSATIONS_WING
    if not conversations.is_dir():
        raise Refused(f"the palace has no conversations directory ({conversations})")
    lock = lock or mine_lock
    result = Restored()
    for file, (entry, data) in files.items():
        with lock(conversations / file):
            _restore_file(collection, conversations, file, entry, data, by_file, present, result)
    return result


def _restore_file(collection, conversations, file, entry, data, by_file, present, result):
    """One file of a restore: its absent drawers first, then the file if it is absent."""
    absent = []
    for row in by_file.get(file, []):
        if row["id"] in present:
            result.skipped_drawers.append(row["id"])
        else:
            absent.append(row)
    _put_back(collection, absent)
    result.drawers += len(absent)
    target = conversations / file
    if target.exists() or target.is_symlink():
        result.skipped_files.append(file)
        return
    _create(target, data, entry.get("mtime_ns"))
    result.files.append(file)


def _put_back(collection, rows: list[dict]) -> None:
    """Upsert drawers exactly as they were stored; one without an embedding is embedded anew."""
    for embedded in (True, False):
        batch = [row for row in rows if bool(row.get("embedding")) is embedded]
        for start in range(0, len(batch), _DELETE_BATCH):
            part = batch[start : start + _DELETE_BATCH]
            if not part:
                continue
            fields = {
                "ids": [row["id"] for row in part],
                "documents": [row["document"] for row in part],
                "metadatas": [row["metadata"] for row in part],
            }
            if embedded:
                fields["embeddings"] = [row["embedding"] for row in part]
            collection.upsert(**fields)


def _create(target: Path, data: bytes, mtime_ns: int | None) -> None:
    """Write a file that must not exist yet, with its original mtime. Never overwrites."""
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if mtime_ns is not None:
        os.utime(target, ns=(mtime_ns, mtime_ns))


def discard_backup(state: Path, name: str) -> None:
    """Remove one backup the agent has said can go: only a backup this tool wrote, by its name."""
    shutil.rmtree(_backup_dir(state, name))


# --- the CLI ------------------------------------------------------------------------------------


def _ids(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _print_offer(result: Offer) -> None:
    """The operator's view of a dry run: counts, ids and the plan hash. Never drawer text."""
    found = result.survey
    print(f"palace: {found.palace}")
    print(
        f"drawers: {found.drawers}  exchanges: {found.exchanges}  "
        f"candidates: {len(found.candidates)}  "
        f"not a task: {found.excluded[NOT_TASK]}  "
        f"second speaker: {found.excluded[SECOND_SPEAKER]}  "
        f"unreadable due time: {found.excluded[BAD_DUE]}"
    )
    print(
        f"held: no source file {found.no_source} drawer(s), no exchange file here "
        f"{found.elsewhere} drawer(s), a drawer not from its file {found.foreign} file(s), "
        f"unfiled {found.unfiled} file(s), not read {found.unread} file(s)"
    )
    members = sum(len(g.members) for g in result.groups)
    copies = sum(len(g.copies) for g in result.groups)
    exceptions = sum(len(g.exceptions) for g in result.groups)
    print(
        f"groups: {len(result.groups)}  members: {members}  copies a yes removes: {copies}  "
        f"latest kept: {sum(1 for g in result.groups if g.latest)}  exceptions kept: {exceptions}"
    )
    for group in result.groups:
        print(
            f"  group {group.id}: members {len(group.members)}, copies {len(group.copies)}, "
            f"latest {1 if group.latest else 0}, exceptions {len(group.exceptions)}, "
            f"due {_month(group.members[0].due)} to {_month(group.members[-1].due)}"
        )
    _print_diagnosis(result.diagnosis)
    print(f"plan: {result.hash}")
    print(f"report: {result.report}")
    print(f"backups kept: {len(result.backups)}")
    now = datetime.now(UTC)
    for name in result.backups:
        print(f"  backup {name}: {backup_age(name, now)}")
    print("Nothing was removed.")


def _print_diagnosis(diagnosis: Diagnosis) -> None:
    """The candidates by due month, and why the ungrouped ones are in no group. Counts only."""
    print("by due month (UTC):")
    for name, month in diagnosis.months.items():
        print(
            f"  {name}: candidates {month.candidates}, members {month.members}, "
            f"copies {month.copies}, latest {month.latest}, exceptions {month.exceptions}, "
            f"ungrouped {month.ungrouped}"
        )
    why = diagnosis.why
    print(
        f"ungrouped: {diagnosis.ungrouped}  "
        f"in what was left of a repeated set, under the floor of {FLOOR}: {why[SMALL_SET]}  "
        f"near-copy of a group member, not its anchor: {why[NEAR_MEMBER]}  "
        f"near-copy of no candidate: {why[DISTINCT]}"
    )
    print(
        f"  had {FLOOR - 1} near-copies or more before earlier groups took some: {diagnosis.taken}"
    )
    if not diagnosis.anchors:
        print("  best overlap with a group's anchor: no group, so no anchor")
        return
    for half, bands in (("instruction", diagnosis.instruction), ("reply", diagnosis.reply)):
        counts = ", ".join(f"{label} {bands[label]}" for label in BAND_LABELS)
        print(f"  best {half} overlap with a group's anchor: {counts}")


def build_parser() -> argparse.ArgumentParser:
    """The command line: one store per run, so no option takes more than one value."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Offer an agent the removal of routine task check-ins from its own memory store, group "
            "by group. With no --apply it changes nothing in the store: it writes a report for the "
            "agent and the plan its yes would carry out."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROG} {__version__}",
        help="print the installed basecradle-harness version, then exit.",
    )
    parser.add_argument(
        "--palace",
        metavar="DIR",
        help=(
            "the one memory store to read. Defaults to the one this agent would bind: "
            "MEMPALACE_PALACE_PATH, else $HARNESS_HOME/mempalace."
        ),
    )
    parser.add_argument(
        "--state-dir",
        metavar="DIR",
        help=(
            "where the report, the plan and the backups are kept, never inside the palace. "
            f"Defaults to $HARNESS_HOME/{STATE_DIR}, else beside the palace."
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--apply",
        action="store_true",
        help=(
            "carry out the agent's yes: needs --plan and --groups. Removes each named group's "
            "copies (and with --remove-latest its latest copy) after writing a backup."
        ),
    )
    action.add_argument(
        "--restore",
        metavar="NAME",
        help=(
            "put back the files and drawers of one backup that are absent from the store. "
            "Safe to run twice; the backup is kept."
        ),
    )
    action.add_argument(
        "--discard-backup",
        metavar="NAME",
        help="remove one backup the agent has said can go.",
    )
    parser.add_argument(
        "--plan", metavar="HASH", help="the plan hash of the report the agent read."
    )
    parser.add_argument("--groups", metavar="IDS", help="the comma-separated group ids of the yes.")
    parser.add_argument(
        "--remove-latest",
        metavar="IDS",
        help="the groups, among --groups, whose latest copy goes too.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """The ``basecradle-harness-prune-palace`` entrypoint. Exit 0 on success, 1 on a refusal."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.apply and (args.plan or args.groups or args.remove_latest):
        parser.error("--plan, --groups and --remove-latest go with --apply")

    logging.basicConfig(level=logging.INFO, handlers=[log_handler("%(message)s", sys.stderr)])
    palace: Path | None = None
    try:
        palace = resolve_palace(args.palace)
        state = resolve_state(args.state_dir, palace)
        if not palace.is_dir():
            raise Refused(f"no palace at {palace} — pass --palace, or set MEMPALACE_PALACE_PATH")
        _check_owner(palace)
        if args.discard_backup:
            discard_backup(state, args.discard_backup)
            print(f"backup removed: {args.discard_backup}")
            _log.info(
                "%s %s",
                head("palace prune", GREEN),
                kv(palace=str(palace), mode="discard", backup=args.discard_backup),
            )
            return 0
        collection = open_palace(palace)
        if args.restore:
            restored = restore(collection, palace, state, args.restore)
            print(f"palace: {palace}")
            print(f"restored: {len(restored.files)} file(s), {restored.drawers} drawer(s)")
            for item in restored.skipped_files[:_SKIPS_SHOWN]:
                print(f"skipped: {item} ({PRESENT})")
            if len(restored.skipped_files) > _SKIPS_SHOWN:
                more = len(restored.skipped_files) - _SKIPS_SHOWN
                print(f"skipped: {more} more file(s) ({PRESENT})")
            if restored.skipped_drawers:
                print(f"skipped: {len(restored.skipped_drawers)} drawer(s) ({PRESENT})")
            _log.info(
                "%s %s",
                head("palace prune", YELLOW if restored.skipped else GREEN),
                kv(
                    palace=str(palace),
                    mode="restore",
                    backup=args.restore,
                    files_restored=len(restored.files),
                    drawers_restored=restored.drawers,
                    skipped=restored.skipped,
                ),
            )
            return 0
        if args.apply:
            if not args.plan:
                raise Refused("a yes names the plan it was given (--plan)")
            result = apply(
                collection,
                palace,
                state,
                args.plan,
                _ids(args.groups),
                _ids(args.remove_latest),
            )
            print(f"palace: {palace}")
            print(f"plan: {args.plan}")
            print(f"removed: {len(result.removed)} file(s), {result.drawers} drawer(s)")
            for name, why in result.skipped:
                print(f"skipped: {name} ({why})")
            print(f"backup: {result.backup or 'none (nothing was removed)'}")
            _log.info(
                "%s %s",
                head("palace prune", YELLOW if result.skipped else GREEN),
                kv(
                    palace=str(palace),
                    mode="apply",
                    plan=args.plan,
                    accepted=",".join(_ids(args.groups)),
                    files_removed=len(result.removed),
                    drawers_deleted=result.drawers,
                    skipped=len(result.skipped),
                    backup=result.backup,
                ),
            )
            return 0
        result = offer(collection, palace, state)
        _print_offer(result)
        _log.info(
            "%s %s",
            head("palace prune", YELLOW if result.groups else GREEN),
            kv(
                palace=str(palace),
                mode="dry-run",
                drawers=result.survey.drawers,
                exchanges=result.survey.exchanges,
                candidates=len(result.survey.candidates),
                groups=len(result.groups),
                members=sum(len(g.members) for g in result.groups),
                copies=sum(len(g.copies) for g in result.groups),
                exceptions=sum(len(g.exceptions) for g in result.groups),
                plan=result.hash,
                backups=len(result.backups),
            ),
        )
    except (ValueError, ImportError, OSError, Interrupted) as error:
        _log.error(
            "%s %s",
            head("palace prune failed", RED),
            kv(palace=str(palace) if palace else None, error=str(error)),
        )
        print(f"{PROG}: {last_word(error)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
