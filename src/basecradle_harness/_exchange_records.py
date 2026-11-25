"""The MemPalace adapter's own record of each exchange, kept outside the palace (issue #1011).

`basecradle_harness._mempalace_provider.MemPalaceProvider` files every exchange into MemPalace, and
before it does, it writes the exchange here, **in a format that is ours**: one small JSON file per
exchange, beside the palace and never inside it, in a directory no MemPalace command reads. The
record is what the exchange *was* — both halves, redacted, as the wake handed them over — and it
says whether it has been filed. Timelines are deleted fast, so the platform cannot be the copy of
record; MemPalace's drawers cannot be either, because a drawer is the library's and is shaped for
search. So the record comes first and the drawers are filed from it, and a filing that could not
finish (the palace busy past the retry) is finished by the next wake from the record alone.

Layout, under ``<palace parent>/<palace name>-records/`` (`records_dir`):

- ``<timeline>/<id>.json`` — the record. ``id`` is the uuid of the item the exchange answered
  (`MemoryExchange.item`), or a fresh one when it answered none, so the record's path and the
  name every drawer of it is filed under (`exchange_name`) are derived from the same two values.
- ``unfiled/<timeline>.<id>`` — an empty marker that exists exactly while the record has not been
  filed and the door has not set it aside (`Record.skipped`). Written before the record and removed after the record says it is filed, so a process
  killed anywhere in between leaves a marker the next wake reads (`RecordStore.unfiled`) and
  resolves against the record: filed, file it, or (no record was ever written) drop the marker.
- ``tasks/<key>.json`` — for each recurring task, which record holds its last **filed** reply: what
  the door rule compares a new activation's reply against. A task is keyed by its timeline and its
  instructions, normalised as the near-copy measure normalises them (`task_key`), because a
  recurring task on this platform is a new task with the same instructions each time it fires.

**A reply cut off at its output budget is held, never filed as a fragment** (the capital's decision
of 2026-11-25 on issue #1011, reviewed by ``@briggs``, whose memory this is). Its record is written
with ``cut_off`` set and keeps its marker, so it is unfiled; a resume of the same turn completes the
same record (`continued`) and it is filed once, whole. A record whose resume never came is filed as
it stands, with the one marker line `CUT_OFF_MARKER` after its words (`exchange_text`), once nothing
can resume it any more. The record never loses a word: a continuation is added to it, never put in
place of what is there.

**One writer per agent, by the fleet's guarantee, not by a lock here.** The router runs one wake at
a time per agent, which is what the session transcript already relies on (CLAUDE.md, Delivery
Guarantee: "the transcript has exactly one writer, and this repo does not enforce it"). A second
process filing into the same records at once could have its marker resolved by the other's refile.
A process killed after it wrote a marker and before its record replaced the partial leaves both,
which the next wake drops: that exchange is lost, exactly as an observe that fails is, and as the
previous adapter lost one killed between its file and its drawers.

Every write is atomic (a fixed-name ``.<name>.partial`` beside the target, ``fsync``, `os.replace`,
then the directory ``fsync``) at mode ``0600`` in ``0700`` directories. A write killed partway
leaves only its partial, which the next write of the same file takes over; the flush removes the one
a never-completed first write leaves (`RecordStore.resolve_unfiled`). Nothing here ever deletes a
record except `RecordStore.remove`, which the scrub and the prune call for an exchange they delete
with its drawers.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from basecradle_harness._near_copy import normalize

#: The version of the record's own format. A field is added, never renamed or repurposed, without
#: changing it; a reader refuses a version it does not know rather than guess at its fields.
#:
#: 2 (issue #1011, the capital's decision of 2026-11-25): ``kind``, ``cut_off`` and ``skipped``. An
#: unfiled record that is ``cut_off`` must not be filed while its turn can still be resumed, which a
#: format-1 reader would do, so the version moved: a harness that reads only format 1 counts a
#: format-2 record unreadable and keeps its marker, and its scrub and prune hold it. A format-1
#: record reads as one with none of the three fields.
RECORD_FORMAT = 2

#: Every format this harness reads.
READABLE_FORMATS = frozenset({1, 2})

#: The one line a reply cut off at its output budget is filed with when no resume ever finished it
#: (issue #1011, the capital's decision of 2026-11-25), after the reply's own words, where recall
#: shows it. Exactly this text, on its own line, and only on that path: a reply a resume completed
#: carries none. It is the one line the harness writes into a filed exchange of the agent's, by that
#: decision, and it is never scrubbed: it says what happened to the reply, and recall needs it.
CUT_OFF_MARKER = "Cut off. No resume arrived."

#: The first word of every name the adapter files under: ``basecradle:<timeline>:<id>:<piece>``.
NAME_SCHEME = "basecradle"

#: The timeline segment of a name for an exchange that happened on no timeline (a library caller's
#: own exchange). A name always has four segments, so a reader never guesses which one is missing.
NO_TIMELINE = "none"

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

#: Exactly the names `exchange_name` writes, and nothing looser.
NAME = re.compile(
    rf"^{NAME_SCHEME}:(?P<timeline>{_UUID}|{NO_TIMELINE}):(?P<id>{_UUID}):(?P<piece>[1-9][0-9]*)$"
)

#: Where old conversation files are moved to, by configuration: the directory the scrub and the
#: prune look in after the palace's own ``conversations`` directory before they hold an old name.
ARCHIVE_VAR = "HARNESS_MEMPALACE_ARCHIVE"

_PARTIAL = ".partial"
_UNFILED = "unfiled"
_TASKS = "tasks"


def records_dir(palace: Path) -> Path:
    """Where the records of the palace at ``palace`` live: beside it, never inside it.

    ``<palace parent>/<palace name>-records``. Derived from the palace path so the two travel
    together (a copy of an agent's memory is a copy of both directories), and outside it so no
    MemPalace command that is pointed at the palace ever reads a record.
    """
    palace = Path(palace)
    return palace.parent / f"{palace.name}-records"


def archive_dir(explicit: str | None = None) -> Path | None:
    """The archive of old conversation files: ``explicit``, else `ARCHIVE_VAR`, else ``None``."""
    value = explicit or os.environ.get(ARCHIVE_VAR) or ""
    return Path(os.path.abspath(os.path.expanduser(value))) if value.strip() else None


def exchange_name(timeline: str, record_id: str, piece: int) -> str:
    """The name the drawer for ``piece`` (counted from 1) of an exchange is filed under.

    ``basecradle:<timeline uuid>:<item uuid>:<piece number>``. The piece number is always present,
    ``1`` included, so there is one spelling and a retry finds it. Never a path, and never the bare
    timeline: one name per timeline returned 1 of 5 keyword-only memories under union search, and a
    unique name returned 5 of 5 (measured for issue #1011).
    """
    if piece < 1:
        raise ValueError(f"a piece is counted from 1, not {piece}")
    return f"{NAME_SCHEME}:{timeline}:{record_id}:{piece}"


@dataclass(frozen=True)
class Name:
    """A name `exchange_name` wrote, read back into its three parts."""

    timeline: str
    id: str
    piece: int


def parse_name(name: object) -> Name | None:
    """``name`` read as one the adapter files under, or ``None`` when it is not one."""
    match = NAME.match(name) if isinstance(name, str) else None
    if match is None:
        return None
    return Name(timeline=match["timeline"], id=match["id"], piece=int(match["piece"]))


def timeline_segment(timeline: str | None) -> str:
    """The timeline as a name carries it: a lowercased uuid, or `NO_TIMELINE`."""
    candidate = (timeline or "").strip().lower()
    return candidate if re.fullmatch(_UUID, candidate) else NO_TIMELINE


def record_id(item: str | None) -> str:
    """The id a new record takes: the item's uuid when it is one, else a fresh uuid."""
    candidate = (item or "").strip().lower()
    return candidate if re.fullmatch(_UUID, candidate) else str(uuid.uuid4())


def task_key(timeline: str, instructions: str) -> str:
    """The key the door rule files a recurring task's last reply under.

    The timeline and the task's instructions, normalised as `_near_copy.normalize` normalises them
    (uuids, times, dates and numbers made placeholders), hashed so the key is a file name. A task
    rescheduled with a new time or id in its instructions is the same task; the same instructions
    on two timelines are two.
    """
    digest = hashlib.sha256(f"{timeline}\n{normalize(instructions).strip()}".encode()).hexdigest()
    return digest


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Record:
    """One exchange, as the adapter keeps it."""

    id: str
    timeline: str
    agent: str
    user: str
    assistant: str
    recorded_at: str
    filed: bool = False
    seeded: bool = False
    task_instructions: str | None = None
    #: The kind of item the exchange answered, in the wake's words (`MemoryExchange.kind`): what the
    #: adapter hands back to the wake to ask whether a cut-off reply can still be resumed.
    kind: str | None = None
    #: The reply was cut off at its output budget and no resume has finished it. Held unfiled while
    #: a resume can still come; filed with `CUT_OFF_MARKER` once none can.
    cut_off: bool = False
    #: Why the door declined to file this record after it was written, or ``None``. Only a reply that
    #: was whole too late to judge at the door has one (``near_copy``: a cut-off task reply its
    #: resume made a near-copy of the task's last filed reply). Such a record is never filed and
    #: carries no marker; it is kept, because the adapter never deletes.
    skipped: str | None = None
    format: int = RECORD_FORMAT

    @property
    def stem(self) -> str:
        """``basecradle:<timeline>:<id>``: every drawer of this exchange is named under it."""
        return f"{NAME_SCHEME}:{self.timeline}:{self.id}"

    def name(self, piece: int) -> str:
        """The name of piece ``piece`` (from 1) of this exchange (`exchange_name`)."""
        return exchange_name(self.timeline, self.id, piece)

    def to_bytes(self) -> bytes:
        return (json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> Record:
        """A record read back, or `ValueError` for anything that is not one of ours."""
        try:
            raw = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"not a record: {error}") from error
        if not isinstance(raw, dict) or raw.get("format") not in READABLE_FORMATS:
            raise ValueError("not a record of a format this harness reads")
        try:
            record = cls(
                id=str(raw["id"]),
                timeline=str(raw["timeline"]),
                agent=str(raw["agent"]),
                user=str(raw["user"]),
                assistant=str(raw["assistant"]),
                recorded_at=str(raw["recorded_at"]),
                filed=bool(raw["filed"]),
                seeded=bool(raw.get("seeded", False)),
                task_instructions=_optional(raw.get("task_instructions")),
                kind=_optional(raw.get("kind")),
                cut_off=bool(raw.get("cut_off", False)),
                skipped=_optional(raw.get("skipped")),
            )
        except KeyError as error:
            raise ValueError(f"a record is missing {error}") from error
        return record


def _optional(value: object) -> str | None:
    return None if value is None else str(value)


def exchange_text(record: Record) -> str:
    """The exchange as it is filed and judged: MemPalace's exchange format, each incoming line
    quoted ``> `` (`_mempalace._exchange_markdown`), composed from the record's two halves.

    A record still ``cut_off`` ends its reply with `CUT_OFF_MARKER` on a line of its own, so the
    text is a function of the record alone: the filing, a refile of it, and the scrub and the prune
    that judge it all read the same words.
    """
    from basecradle_harness._memory_provider import MemoryExchange, MemoryScope
    from basecradle_harness._mempalace import _exchange_markdown

    assistant = _joined(record.assistant, CUT_OFF_MARKER) if record.cut_off else record.assistant
    exchange = MemoryExchange(user=record.user, assistant=assistant, scope=MemoryScope(""))
    return _exchange_markdown(exchange)


def continued(record: Record, continuation: str, *, cut_off: bool) -> Record:
    """``record`` with a resume's ``continuation`` added after its reply, and ``cut_off`` as given.

    The reply so far, a line break, then the continuation. A cut usually falls at the end of a word
    (a model's tokens mostly carry their own leading space), and a continuation starts its first
    word without one, so the two run together would make one word of two; a line break keeps every
    word as the model wrote it. A word the cut did split stays in its two halves. An empty side adds
    nothing. The record keeps every word it had. (A secret the cut split is the one exception, joined
    without the break so it can be blanked: `_mempalace_provider.MemPalaceProvider._across_the_cut`.)
    """
    return replace(record, assistant=_joined(record.assistant, continuation), cut_off=cut_off)


def _joined(first: str, second: str) -> str:
    return "\n".join(part for part in (first, second) if part.strip())


def new_record(
    *,
    timeline: str | None,
    item: str | None,
    agent: str,
    user: str,
    assistant: str,
    seeded: bool = False,
    task_instructions: str | None = None,
    kind: str | None = None,
    cut_off: bool = False,
) -> Record:
    """A record for an exchange about to be filed (or, ``cut_off``, held): not filed, stamped now."""
    return Record(
        id=record_id(item),
        timeline=timeline_segment(timeline),
        agent=agent,
        user=user,
        assistant=assistant,
        recorded_at=_now(),
        seeded=seeded,
        task_instructions=task_instructions,
        kind=kind,
        cut_off=cut_off,
    )


class RecordStore:
    """The records of one palace (`records_dir`). See the module docstring for the layout."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # --- paths ---------------------------------------------------------------------------------

    def path(self, timeline: str, record_id: str) -> Path:
        """Where the record of ``(timeline, id)`` lives, whether or not it exists."""
        return self.root / timeline / f"{record_id}.json"

    def path_of(self, name: str) -> Path | None:
        """Where the record a drawer named ``name`` was filed from lives, or ``None``."""
        parsed = parse_name(name)
        return None if parsed is None else self.path(parsed.timeline, parsed.id)

    def _marker(self, timeline: str, record_id: str) -> Path:
        return self.root / _UNFILED / f"{timeline}.{record_id}"

    def _task(self, key: str) -> Path:
        return self.root / _TASKS / f"{key}.json"

    # --- reading -------------------------------------------------------------------------------

    def read(self, timeline: str, record_id: str) -> Record | None:
        """The record of ``(timeline, id)``, or ``None`` when there is none.

        A file that is there and is not a record raises `ValueError`: a record this harness cannot
        read is never treated as absent, because absent is what licenses writing a new one.
        """
        try:
            data = self.path(timeline, record_id).read_bytes()
        except FileNotFoundError:
            return None
        return Record.from_bytes(data)

    def records(self) -> Iterator[tuple[Path, bytes]]:
        """Every record file, as ``(path, bytes)``, in a stable order. Unreadable files are skipped.

        For the prune, which judges an exchange from its record. Only ``<timeline>/<id>.json``
        files are yielded: the marker and task directories, partials and anything else are not
        records.
        """
        if not self.root.is_dir():
            return
        for folder in sorted(self.root.iterdir()):
            if folder.name in (_UNFILED, _TASKS) or folder.is_symlink() or not folder.is_dir():
                continue
            if folder.name != NO_TIMELINE and not re.fullmatch(_UUID, folder.name):
                continue
            for path in sorted(folder.iterdir()):
                if path.is_symlink() or not path.is_file() or path.name.startswith("."):
                    continue
                if not re.fullmatch(rf"{_UUID}\.json", path.name):
                    continue
                with contextlib.suppress(OSError):
                    yield path, path.read_bytes()

    def is_unfiled(self, timeline: str, record_id: str) -> bool:
        """Whether the record of ``(timeline, id)`` carries a marker: written, not yet all filed."""
        return self._marker(timeline, record_id).exists()

    def unfiled(self) -> list[tuple[str, str]]:
        """Every ``(timeline, id)`` whose record has not been marked filed, oldest marker first."""
        folder = self.root / _UNFILED
        if not folder.is_dir():
            return []
        found = []
        for path in folder.iterdir():
            timeline, _, record = path.name.partition(".")
            if record and not path.name.endswith(_PARTIAL):
                with contextlib.suppress(OSError):
                    found.append((path.stat().st_mtime_ns, timeline, record))
        return [(timeline, record) for _, timeline, record in sorted(found)]

    def last_task_reply(self, key: str) -> Record | None:
        """The record holding the last filed reply of the task keyed ``key``, or ``None``."""
        try:
            pointer = json.loads(self._task(key).read_text(encoding="utf-8"))
            return self.read(str(pointer["timeline"]), str(pointer["id"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    # --- writing -------------------------------------------------------------------------------

    def write(self, record: Record) -> Path:
        """Write a record. An unfiled one gets its marker **first**, so no kill loses track of it."""
        if not record.filed:
            self._touch(self._marker(record.timeline, record.id))
        target = self.path(record.timeline, record.id)
        self._write(target, record.to_bytes())
        return target

    def mark_filed(self, record: Record) -> Record:
        """Record that every piece of ``record`` is filed; then, and only then, drop its marker."""
        filed = replace(record, filed=True)
        self._write(self.path(filed.timeline, filed.id), filed.to_bytes())
        self._marker(filed.timeline, filed.id).unlink(missing_ok=True)
        return filed

    def set_aside(self, record: Record, reason: str) -> Record:
        """Record that the door declined ``record`` for ``reason`` after it was written; drop its marker.

        For a reply that could be judged only once it was whole (a cut-off task reply, completed by
        its resume, that is a near-copy of the task's last filed reply). The record is kept, saying
        why it was never filed, and the marker goes, so no later wake tries to file it.
        """
        aside = replace(record, skipped=reason)
        self._write(self.path(aside.timeline, aside.id), aside.to_bytes())
        self._marker(aside.timeline, aside.id).unlink(missing_ok=True)
        return aside

    def remember_task(self, key: str, record: Record) -> None:
        """Point the task keyed ``key`` at ``record``, its last filed reply.

        Never backwards: a record filed late (an older activation the next wake refiled) leaves the
        pointer on a newer one, so the door compares against the latest reply, by when each was
        recorded.
        """
        current = self.last_task_reply(key)
        if current is not None and current.recorded_at > record.recorded_at:
            return
        pointer = {"timeline": record.timeline, "id": record.id}
        self._write(self._task(key), (json.dumps(pointer) + "\n").encode("utf-8"))

    def resolve_unfiled(self, timeline: str, record_id: str) -> Record | None:
        """The record behind a marker, or ``None`` after dropping a marker that names none.

        A marker is written before its record, so a process killed between the two leaves a marker
        with no record (and perhaps that record's partial): nothing was filed and nothing is lost,
        so both go, each by its own name in its own place. A record already marked filed has its
        stale marker dropped too, and is returned so the caller can see it is done. A record that
        cannot be read raises `ValueError` and keeps its marker.
        """
        record = self.read(timeline, record_id)
        if record is None:
            target = self.path(timeline, record_id)
            target.with_name(f".{target.name}{_PARTIAL}").unlink(missing_ok=True)
            self._marker(timeline, record_id).unlink(missing_ok=True)
            return None
        if record.filed:
            self._marker(timeline, record_id).unlink(missing_ok=True)
        return record

    def remove(self, timeline: str, record_id: str) -> bool:
        """Delete one record and its marker. ``True`` when a record was there to delete."""
        self._marker(timeline, record_id).unlink(missing_ok=True)
        try:
            self.path(timeline, record_id).unlink()
        except FileNotFoundError:
            return False
        return True

    def _folder(self, folder: Path) -> Path:
        """``folder`` (the root, or one directory directly under it), made ``0700`` if absent."""
        self.root.parent.mkdir(parents=True, exist_ok=True)
        _private_dir(self.root)
        _private_dir(folder)
        return folder

    def _write(self, path: Path, data: bytes) -> None:
        self._folder(path.parent)
        _write_private(path, data)

    def _touch(self, path: Path) -> None:
        self._folder(path.parent)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(descriptor)
        _fsync_dir(path.parent)


def _private_dir(path: Path) -> None:
    """Make ``path`` a directory at mode ``0700`` if it is not one yet. Its parent must exist."""
    if not path.is_dir():
        path.mkdir(mode=0o700, exist_ok=True)
        os.chmod(path, 0o700)


def _write_private(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically at mode ``0600``. Its directory must exist."""
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
    _fsync_dir(path.parent)


def _fsync_dir(path: Path) -> None:
    """Make a rename inside ``path`` durable."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
